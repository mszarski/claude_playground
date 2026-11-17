// The C++ far-field beamformers and array calibration (libs/live beamform, calibrate) for
// Python: the counterparts of gpusonar.flank.farfield_beamform / mvdr_beamform and of
// gpusonar.calibrate. Like the Python stages they condition the record in place first.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/live/beamform.hpp"
#include "sonar/live/calibrate.hpp"
#include "sonar/live/chain.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

// The fields of gpusonar.flank.BeamMap, so that flank.BeamMap(**result) is the Python object.
py::dict beam_map_dict(const live::BeamMap& m) {
  py::array_t<std::complex<float>> data({static_cast<py::ssize_t>(m.n_bearings), static_cast<py::ssize_t>(m.n_samples)});
  std::copy(m.data.begin(), m.data.end(), data.mutable_data());
  py::dict d;
  d["data"] = data;
  d["bearings"] = py::array_t<double>(static_cast<py::ssize_t>(m.bearings.size()), m.bearings.data());
  d["times"] = py::array_t<double>(static_cast<py::ssize_t>(m.times.size()), m.times.data());
  d["ranges"] = py::array_t<double>(static_cast<py::ssize_t>(m.ranges.size()), m.ranges.data());
  d["n_valid"] = m.n_valid;
  return d;
}

live::ChannelErrors channel_errors(const std::vector<double>& gain, const std::vector<double>& phase, const std::vector<Vec3d>& position) {
  live::ChannelErrors e;
  e.gain = gain;
  e.phase = phase;
  e.position = position;
  return e;
}

}  // namespace

void bind_beamform(py::module_& m) {
  m.def(
      "steering_delays_cpp",
      [](const PingRecord& record, const std::vector<double>& bearings) {
        const std::vector<double> tau = live::steering_delays(record, bearings);
        py::array_t<double> a({static_cast<py::ssize_t>(bearings.size()), static_cast<py::ssize_t>(record.geometry.rx.size())});
        std::copy(tau.begin(), tau.end(), a.mutable_data());
        return a;
      },
      py::arg("record"), py::arg("bearings"), "Plane-wave delays p_n . u / c, [n_bearing, n_elements], s.");
  m.def(
      "farfield_beamform_cpp",
      [](PingRecord& record, const std::vector<double>& bearings, const std::string& window) {
        live::BeamMap map;
        {
          py::gil_scoped_release release;
          live::prepare(record);
          map = live::farfield_beamform(record, bearings, window);
        }
        return beam_map_dict(map);
      },
      py::arg("record"), py::arg("bearings"), py::arg("window") = "rect",
      "Conventional far-field beamformer (C++). Conditions the record in place; returns the fields of\n"
      "gpusonar.flank.BeamMap as a dict: data [n_bearing, n_samples], bearings, times, ranges, n_valid.");
  m.def(
      "mvdr_beamform_cpp",
      [](PingRecord& record, const std::vector<double>& bearings, std::int64_t n_subbands, double loading, std::int64_t block,
         std::int64_t guard_blocks) {
        live::MvdrOptions o;
        o.n_subbands = n_subbands;
        o.loading = loading;
        o.block = block;
        o.guard_blocks = guard_blocks;
        live::BeamMap map;
        {
          py::gil_scoped_release release;
          live::prepare(record);
          map = live::mvdr_beamform(record, bearings, o);
        }
        return beam_map_dict(map);
      },
      py::arg("record"), py::arg("bearings"), py::arg("n_subbands") = 32, py::arg("loading") = 3e-2, py::arg("block") = 256,
      py::arg("guard_blocks") = 1,
      "Adaptive (MVDR) beamformer (C++). Conditions the record in place; returns the fields of\n"
      "gpusonar.flank.BeamMap as a dict.");

  m.def(
      "calibrate_channel_responses_cpp",
      [](PingRecord& record, const Vec3d& source_position) {
        live::prepare(record);
        const auto v = live::channel_responses(record, source_position);
        return py::array_t<std::complex<double>>(static_cast<py::ssize_t>(v.size()), v.data());
      },
      py::arg("record"), py::arg("source_position"),
      "Complex response of each channel to a source at a known position (C++). Conditions the record in place.");
  m.def(
      "calibrate_estimate_cpp",
      [](const py::list& records, const std::vector<Vec3d>& source_positions, bool planar) {
        std::vector<const PingRecord*> ptrs;
        for (const auto& item : records) {
          auto& r = item.cast<PingRecord&>();
          live::prepare(r);
          ptrs.push_back(&r);
        }
        live::ChannelErrors e;
        {
          py::gil_scoped_release release;
          e = live::estimate_channel_errors(ptrs, source_positions, planar);
        }
        py::array_t<double> position({static_cast<py::ssize_t>(e.position.size()), py::ssize_t{3}});
        auto p = position.mutable_unchecked<2>();
        for (std::size_t i = 0; i < e.position.size(); ++i) {
          for (int k = 0; k < 3; ++k) p(static_cast<py::ssize_t>(i), k) = e.position[i][k];
        }
        // The fields of gpusonar.calibrate.ChannelErrors.
        py::dict d;
        d["gain"] = py::array_t<double>(static_cast<py::ssize_t>(e.gain.size()), e.gain.data());
        d["phase"] = py::array_t<double>(static_cast<py::ssize_t>(e.phase.size()), e.phase.data());
        d["position"] = position;
        d["residual_rms_deg"] = e.residual_rms_deg;
        return d;
      },
      py::arg("records"), py::arg("source_positions"), py::arg("planar") = true,
      "Gain, phase and element position errors from calibration pings (C++), one known source position\n"
      "per ping. Conditions the records in place; returns the fields of gpusonar.calibrate.ChannelErrors.");
  m.def(
      "calibrate_store_cpp",
      [](PingRecord& record, const std::vector<double>& gain, const std::vector<double>& phase, const std::vector<Vec3d>& position) {
        live::store_calibration(record, channel_errors(gain, phase, position));
      },
      py::arg("record"), py::arg("gain"), py::arg("phase"), py::arg("position"),
      "Write estimated channel errors into a record's calibration tables.");
  m.def(
      "calibrate_apply_cpp",
      [](PingRecord& record, const std::vector<double>& gain, const std::vector<double>& phase, const std::vector<Vec3d>& position) {
        live::prepare(record);
        live::apply_channel_errors(record, channel_errors(gain, phase, position));
      },
      py::arg("record"), py::arg("gain"), py::arg("phase"), py::arg("position"),
      "Condition the record, store the channel errors and remove gain and phase from its samples;\n"
      "the position offsets stay in the calibration table, where the beamformers read them.");
}
