#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>

#include "sonar/sim/synth.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;
using ArrC = py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>;

Vec3d row3(const ArrD& a, py::ssize_t i) {
  auto v = a.unchecked<2>();
  return {v(i, 0), v(i, 1), v(i, 2)};
}

void require_n3(const ArrD& a, py::ssize_t n, const char* name) {
  if (a.ndim() != 2 || a.shape(1) != 3 || a.shape(0) != n) {
    throw std::invalid_argument(std::string(name) + " must have shape [n_paths, 3]");
  }
}

/// Build the path list from arrays. `q`/`p` are [P, 3]; velocities may be None (static).
std::vector<sim::SynthPath> make_paths(const ArrD& q, const ArrD& p, const ArrC& amplitude,
                                       const py::object& q_velocity, const py::object& p_velocity) {
  if (q.ndim() != 2 || q.shape(1) != 3) throw std::invalid_argument("q must have shape [n_paths, 3]");
  const py::ssize_t n = q.shape(0);
  require_n3(p, n, "p");
  if (amplitude.ndim() != 1 || amplitude.shape(0) != n) {
    throw std::invalid_argument("amplitude must have shape [n_paths]");
  }
  const bool has_vq = !q_velocity.is_none();
  const bool has_vp = !p_velocity.is_none();
  ArrD vq, vp;
  if (has_vq) {
    vq = ArrD::ensure(q_velocity);
    require_n3(vq, n, "q_velocity");
  }
  if (has_vp) {
    vp = ArrD::ensure(p_velocity);
    require_n3(vp, n, "p_velocity");
  }
  auto amp = amplitude.unchecked<1>();
  std::vector<sim::SynthPath> paths(static_cast<std::size_t>(n));
  for (py::ssize_t i = 0; i < n; ++i) {
    auto& path = paths[static_cast<std::size_t>(i)];
    path.q = row3(q, i);
    path.p = row3(p, i);
    if (has_vq) path.q_velocity = row3(vq, i);
    if (has_vp) path.p_velocity = row3(vp, i);
    path.amp_re = static_cast<float>(amp(i).real());
    path.amp_im = static_cast<float>(amp(i).imag());
  }
  return paths;
}

}  // namespace

void bind_sim(py::module_& m) {
  m.def(
      "synthesize_into",
      [](PingRecord& record, const ArrD& q, const ArrD& p, const ArrC& amplitude,
         const py::object& q_velocity, const py::object& p_velocity,
         const py::object& absorption, bool spreading, Backend backend,
         const Vec3d& platform_acceleration, const py::object& envelope, const py::object& arrival_passes,
         const py::object& arrival_residual_limit) {
        const auto paths = make_paths(q, p, amplitude, q_velocity, p_velocity);
        sim::SynthParams params = sim::params_from_record(record);
        params.spreading = spreading;
        params.platform_acceleration = platform_acceleration;
        if (!absorption.is_none()) {
          // (f_min, f_max, table of kAbsorptionTableSize values in dB/m)
          const auto t = absorption.cast<py::tuple>();
          params.absorption_f_min = t[0].cast<double>();
          params.absorption_f_max = t[1].cast<double>();
          const auto table = t[2].cast<std::vector<double>>();
          if (table.size() != sim::kAbsorptionTableSize) {
            throw std::invalid_argument("absorption table must have " +
                                        std::to_string(sim::kAbsorptionTableSize) + " entries");
          }
          for (int i = 0; i < sim::kAbsorptionTableSize; ++i) params.absorption_db_per_m[i] = table[i];
        }
        // A sampled pulse: (table [n_fine, n_bands + 1] complex, rate, offset, carrier,
        // knot frequencies [n_bands]); see sim::EnvelopeTable. The array is held to the end.
        ArrC table;
        const bool sampled = record.transmit.waveform.type == WaveformType::Arbitrary;
        if (sampled) {
          if (envelope.is_none()) {
            throw std::invalid_argument("synthesize_into: a sampled (arbitrary) waveform needs `envelope`");
          }
          const auto t = envelope.cast<py::tuple>();
          if (t.size() != 5) throw std::invalid_argument("envelope must be (table, rate, offset, carrier, knot_frequencies)");
          table = ArrC::ensure(t[0]);
          const auto knots = t[4].cast<std::vector<double>>();
          if (!table || table.ndim() != 2 || knots.empty() || knots.size() > sim::kMaxEnvelopeBands ||
              table.shape(1) != static_cast<py::ssize_t>(knots.size()) + 1) {
            throw std::invalid_argument("envelope table must have shape [n_fine, len(knot_frequencies) + 1], with at most " +
                                        std::to_string(sim::kMaxEnvelopeBands) + " knot frequencies");
          }
          params.envelope.data = reinterpret_cast<const double*>(table.data());
          params.envelope.n_fine = table.shape(0);
          params.envelope.n_bands = static_cast<int>(knots.size());
          params.envelope.rate = t[1].cast<double>();
          params.envelope.offset = t[2].cast<double>();
          params.envelope.carrier = t[3].cast<double>();
          for (std::size_t i = 0; i < knots.size(); ++i) params.envelope.knot_frequency[i] = knots[i];
          params.arrival_passes_out = params.arrival_passes_back = sim::kSampledArrivalPasses;
        }
        if (!arrival_passes.is_none() && py::isinstance<py::str>(arrival_passes)) {
          // "auto": the fewest passes that leave under 1e-4 rad for this call's speeds, window and band.
          if (arrival_passes.cast<std::string>() != "auto") {
            throw std::invalid_argument("arrival_passes must be (outbound, return) or \"auto\"");
          }
          sim::set_arrival_passes(params, sim::elements_from_record(record), paths, 1e-4);
        } else if (!arrival_passes.is_none()) {
          const auto n = arrival_passes.cast<std::pair<int, int>>();
          params.arrival_passes_out = n.first;
          params.arrival_passes_back = n.second;
        }
        if (!arrival_residual_limit.is_none()) params.arrival_residual_limit = arrival_residual_limit.cast<double>();
        if (arrival_passes.is_none() && params.arrival_residual_limit > 0.0 && !paths.empty() &&
            params.sound_speed > 0.0 && params.sample_rate > 0.0) {
          // The caller left the passes to us. When the default count would leave more than the
          // limit for this call's speeds, window and band, run as many as it takes (never
          // fewer than the default) and say so, instead of returning a wrong phase or failing
          // a scenario that has no key for this. A caller that states the passes gets the
          // C++ check, which raises.
          const auto rx = sim::elements_from_record(record);
          const sim::ArrivalResidual left = sim::arrival_residual(params, rx, paths);
          if (!(left.out <= params.arrival_residual_limit) || !(left.back <= params.arrival_residual_limit)) {
            sim::SynthParams needed = params;
            sim::set_arrival_passes(needed, rx, paths, 1e-4);
            const int out = std::max(needed.arrival_passes_out, params.arrival_passes_out);
            const int back = std::max(needed.arrival_passes_back, params.arrival_passes_back);
            const std::string text =
                "synthesize_into: (" + std::to_string(params.arrival_passes_out) + ", " +
                std::to_string(params.arrival_passes_back) + ") passes of the arrival-time solve could leave " +
                std::to_string(std::max(left.out, left.back)) + " rad at these speeds, window and frequency; running (" +
                std::to_string(out) + ", " + std::to_string(back) + ") instead";
            params.arrival_passes_out = out;
            params.arrival_passes_back = back;
            if (PyErr_WarnEx(PyExc_RuntimeWarning, text.c_str(), 1) != 0) throw py::error_already_set();
          }
        }
        py::gil_scoped_release release;
        sim::synthesize_into(backend, record, paths, &params);
      },
      py::arg("record"), py::arg("q"), py::arg("p"), py::arg("amplitude"),
      py::arg("q_velocity") = py::none(), py::arg("p_velocity") = py::none(),
      py::arg("absorption") = py::none(), py::arg("spreading") = true,
      py::arg("backend") = Backend::Host,
      py::arg("platform_acceleration") = Vec3d{}, py::arg("envelope") = py::none(),
      py::arg("arrival_passes") = py::none(), py::arg("arrival_residual_limit") = py::none(),
      "Add the echoes of two-way paths into record.samples. The record supplies geometry,\n"
      "waveform, sampling and board clocks; its samples must already be allocated.\n"
      "A sampled (Arbitrary) waveform also needs `envelope` = (table [n_fine, n_bands + 1] complex,\n"
      "fine samples per second, fine index of emission time 0, carrier Hz, knot frequencies [n_bands]):\n"
      "gpusonar.sim.waveform.SampledWaveform.kernel_table() builds it.\n"
      "`arrival_passes` = (outbound, return) passes of the arrival-time solve; default (4, 4) for\n"
      "analytic pulses and (8, 8) for sampled ones; \"auto\" takes the fewest that leave under 1e-4 rad.\n"
      "The call raises when the bound on the phase the passes leave, T_end (v / c)^n 2 pi f_top, exceeds\n"
      "`arrival_residual_limit` (rad; default 0.05, 0 = no check) for passes the caller stated; with\n"
      "`arrival_passes` left out the count is raised to what the bound needs and a RuntimeWarning says so.\n"
      "It also raises when the pulse holds frequencies outside the absorption table by enough to change a\n"
      "level by more than 0.1 dB.");
  m.def("synth_arrival_residual_bound", &sim::arrival_residual_bound, py::arg("window_end"), py::arg("speed"),
        py::arg("sound_speed"), py::arg("frequency"), py::arg("passes"),
        "Upper bound (rad) on the phase one arrival-time solve leaves: window_end (speed / c)^passes 2 pi f.");
  m.def("synth_arrival_passes_for", &sim::arrival_passes_for, py::arg("window_end"), py::arg("speed"),
        py::arg("sound_speed"), py::arg("frequency"), py::arg("tolerance") = 1e-4,
        "Fewest passes of the arrival-time solve that leave under `tolerance` radians.");
  m.attr("SYNTH_ARRIVAL_PASSES") = std::make_pair(sim::SynthParams{}.arrival_passes_out,
                                                  sim::SynthParams{}.arrival_passes_back);
  m.attr("SYNTH_SAMPLED_ARRIVAL_PASSES") = sim::kSampledArrivalPasses;
  m.attr("MAX_ENVELOPE_BANDS") = sim::kMaxEnvelopeBands;
  m.attr("ABSORPTION_TABLE_SIZE") = sim::kAbsorptionTableSize;
}
