#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/dsp/baseband.hpp"
#include "sonar/dsp/pulse_compression.hpp"
#include "sonar/recon/backproject.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

using ArrD = py::array_t<double, py::array::c_style | py::array::forcecast>;

std::vector<const PingRecord*> ping_list(const py::list& pings) {
  std::vector<const PingRecord*> out;
  out.reserve(pings.size());
  for (const auto& item : pings) out.push_back(&item.cast<const PingRecord&>());
  return out;
}

std::vector<Vec3d> point_list(const ArrD& points) {
  if (points.ndim() != 2 || points.shape(1) != 3) {
    throw std::invalid_argument("points must have shape [n, 3]");
  }
  auto v = points.unchecked<2>();
  std::vector<Vec3d> out(static_cast<std::size_t>(points.shape(0)));
  for (py::ssize_t i = 0; i < points.shape(0); ++i) {
    out[static_cast<std::size_t>(i)] = {v(i, 0), v(i, 1), v(i, 2)};
  }
  return out;
}

recon::BpOptions make_options(const std::string& interp, bool stop_and_hop, float fov_factor,
                              bool directivity_weight, int spreading, int cull_block = 0,
                              const py::object& grid_shape = py::none()) {
  recon::BpOptions o;
  if (interp == "nearest") {
    o.interp = recon::Interp::Nearest;
  } else if (interp == "linear") {
    o.interp = recon::Interp::Linear;
  } else if (interp == "sinc") {
    o.interp = recon::Interp::Sinc;
  } else {
    throw std::invalid_argument("interp must be 'nearest', 'linear' or 'sinc'");
  }
  if (spreading < 0 || spreading > 2) throw std::invalid_argument("spreading must be 0, 1 or 2");
  o.stop_and_hop = stop_and_hop;
  o.fov_factor = fov_factor;
  o.directivity_weight = directivity_weight;
  o.spreading = spreading;
  o.cull_block = cull_block;
  if (!grid_shape.is_none()) {
    const auto shape = grid_shape.cast<std::pair<std::int64_t, std::int64_t>>();
    o.grid_rows = shape.first;
    o.grid_cols = shape.second;
  }
  return o;
}

}  // namespace

void bind_recon(py::module_& m) {
  m.def(
      "to_baseband",
      [](PingRecord& record, double center_frequency, double bandwidth, double band_margin,
         double oversample, double edge_taper) {
        dsp::BasebandOptions o;
        o.center_frequency = center_frequency;
        o.bandwidth = bandwidth;
        o.band_margin = band_margin;
        o.oversample = oversample;
        o.edge_taper = edge_taper;
        py::gil_scoped_release release;
        dsp::to_baseband(record, o);
      },
      py::arg("record"), py::arg("center_frequency") = 0.0, py::arg("bandwidth") = 0.0,
      py::arg("band_margin") = 1.5, py::arg("oversample") = 2.0, py::arg("edge_taper") = 0.1,
      "Convert record.samples in place to complex baseband at a reduced rate.");

  m.def(
      "pulse_compress",
      [](PingRecord& record, const std::string& window, double doppler_scale) {
        dsp::PulseCompressionOptions o;
        if (window == "none") {
          o.window = dsp::ReplicaWindow::None;
        } else if (window == "hann") {
          o.window = dsp::ReplicaWindow::Hann;
        } else if (window == "hamming") {
          o.window = dsp::ReplicaWindow::Hamming;
        } else {
          throw std::invalid_argument("window must be 'none', 'hann' or 'hamming'");
        }
        o.doppler_scale = doppler_scale;
        py::gil_scoped_release release;
        dsp::pulse_compress(record, o);
      },
      py::arg("record"), py::arg("window") = "none", py::arg("doppler_scale") = 1.0,
      "Matched-filter record.samples (complex baseband) in place. An echo of delay tau\n"
      "peaks at time tau with its amplitude preserved.");

  m.def(
      "backproject",
      [](const py::list& pings, const ArrD& points, const std::string& interp, bool stop_and_hop,
         float fov_factor, bool directivity_weight, int spreading, const py::object& weights,
         Backend backend, bool return_weight_sum, int cull_block, const py::object& grid_shape,
         const py::object& stats_out) -> py::object {
        const auto ptrs = ping_list(pings);
        const auto pts = point_list(points);
        const auto opt = make_options(interp, stop_and_hop, fov_factor, directivity_weight, spreading,
                                      cull_block, grid_shape);
        recon::BpStats stats;
        std::vector<std::vector<float>> w;
        const bool has_w = !weights.is_none();
        if (has_w) w = weights.cast<std::vector<std::vector<float>>>();
        py::array_t<std::complex<float>> image(static_cast<py::ssize_t>(pts.size()));
        py::array_t<float> wsum(static_cast<py::ssize_t>(return_weight_sum ? pts.size() : 0));
        std::fill_n(image.mutable_data(), pts.size(), std::complex<float>{});
        if (return_weight_sum) std::fill_n(wsum.mutable_data(), pts.size(), 0.0F);
        {
          auto* img = image.mutable_data();
          auto* ws = return_weight_sum ? wsum.mutable_data() : nullptr;
          py::gil_scoped_release release;
          const recon::BpInputs in = recon::make_inputs(ptrs, has_w ? &w : nullptr);
          recon::backproject(backend, in, pts, opt, img, ws, &stats);
        }
        if (!stats_out.is_none()) {
          auto d = stats_out.cast<py::dict>();
          d["ping_blocks_total"] = stats.ping_blocks_total;
          d["ping_blocks_kept"] = stats.ping_blocks_kept;
        }
        if (return_weight_sum) return py::make_tuple(image, wsum);
        return std::move(image);
      },
      py::arg("pings"), py::arg("points"), py::arg("interp") = "sinc",
      py::arg("stop_and_hop") = false, py::arg("fov_factor") = 0.0F,
      py::arg("directivity_weight") = false, py::arg("spreading") = 0,
      py::arg("weights") = py::none(), py::arg("backend") = Backend::Host,
      py::arg("return_weight_sum") = false, py::arg("cull_block") = 0,
      py::arg("grid_shape") = py::none(), py::arg("stats") = py::none(),
      "Time-domain backprojection of conditioned pings onto points [n, 3].\n"
      "Returns the complex coherent sum per point. With cull_block > 0 and grid_shape\n"
      "(rows, cols), blocks of points outside a ping's beams are skipped for that ping;\n"
      "pass a dict as `stats` to receive the block counts.");

  m.def(
      "debug_delays",
      [](const py::list& pings, const ArrD& points, bool stop_and_hop) {
        const auto ptrs = ping_list(pings);
        const auto pts = point_list(points);
        recon::BpOptions opt;
        opt.stop_and_hop = stop_and_hop;
        const recon::BpInputs in = recon::make_inputs(ptrs);
        std::vector<double> pos, phase;
        recon::debug_delays(in, pts, opt, pos, phase);
        const std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(pts.size()),
                                             static_cast<py::ssize_t>(in.records.size())};
        py::array_t<double> a(shape), b(shape);
        std::copy(pos.begin(), pos.end(), a.mutable_data());
        std::copy(phase.begin(), phase.end(), b.mutable_data());
        return py::make_tuple(a, b);
      },
      py::arg("pings"), py::arg("points"), py::arg("stop_and_hop") = false,
      "Test hook: (sample position, carrier phase in cycles) per (point, record), exactly\n"
      "as the kernel computes them.");
}
