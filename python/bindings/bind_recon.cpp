#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/dsp/baseband.hpp"
#include "sonar/dsp/conditioning.hpp"
#include "sonar/dsp/pulse_compression.hpp"
#include "sonar/recon/backproject.hpp"
#include "sonar/recon/backproject_kernel.hpp"
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

recon::BpInputs session_inputs(const py::list& pings, const py::object& weights, int trajectory_upsample) {
  const auto ptrs = ping_list(pings);
  std::vector<std::vector<float>> w;
  const bool has_w = !weights.is_none();
  if (has_w) w = weights.cast<std::vector<std::vector<float>>>();
  py::gil_scoped_release release;
  return recon::make_inputs(ptrs, has_w ? &w : nullptr, trajectory_upsample);
}

py::object session_backproject(recon::BpSession& session, const ArrD& points, const std::string& interp,
                               bool stop_and_hop, float fov_factor, bool directivity_weight,
                               int spreading, Backend backend, bool return_weight_sum, int cull_block,
                               const py::object& grid_shape, const py::object& stats_out,
                               const std::string& position_lookup, const py::object& refraction,
                               int device) {
  const auto pts = point_list(points);
  auto opt = make_options(interp, stop_and_hop, fov_factor, directivity_weight, spreading, cull_block,
                          grid_shape);
  if (position_lookup == "analytic") {
    opt.position_lookup = recon::PositionLookup::Analytic;
  } else if (position_lookup == "software_table") {
    opt.position_lookup = recon::PositionLookup::SoftwareTable;
  } else if (position_lookup == "texture") {
    opt.position_lookup = recon::PositionLookup::Texture;
  } else {
    throw std::invalid_argument("position_lookup must be 'analytic', 'software_table' or 'texture'");
  }
  if (opt.position_lookup != recon::PositionLookup::Analytic && session.inputs().trajectory.empty()) {
    throw std::invalid_argument("this session was built without trajectory tables (trajectory_upsample = 0)");
  }
  opt.device = device;
  if (!refraction.is_none()) {
    // (interface_z, sediment_speed)
    const auto r = refraction.cast<std::pair<double, double>>();
    opt.refraction = true;
    opt.interface_z = r.first;
    opt.sediment_speed = r.second;
  }
  recon::BpStats stats;
  py::array_t<std::complex<float>> image(static_cast<py::ssize_t>(pts.size()));
  py::array_t<float> wsum(static_cast<py::ssize_t>(return_weight_sum ? pts.size() : 0));
  std::fill_n(image.mutable_data(), pts.size(), std::complex<float>{});
  if (return_weight_sum) std::fill_n(wsum.mutable_data(), pts.size(), 0.0F);
  {
    auto* img = image.mutable_data();
    auto* ws = return_weight_sum ? wsum.mutable_data() : nullptr;
    py::gil_scoped_release release;
    session.backproject(backend, pts, opt, img, ws, &stats);
  }
  if (!stats_out.is_none()) {
    auto d = stats_out.cast<py::dict>();
    d["ping_blocks_total"] = stats.ping_blocks_total;
    d["ping_blocks_kept"] = stats.ping_blocks_kept;
  }
  if (return_weight_sum) return py::make_tuple(image, wsum);
  return std::move(image);
}

}  // namespace

void bind_recon(py::module_& m) {
  m.def(
      "to_baseband",
      [](PingRecord& record, double center_frequency, double bandwidth, double band_margin,
         double oversample, double edge_taper, Backend backend, int device) {
        dsp::BasebandOptions o;
        o.center_frequency = center_frequency;
        o.bandwidth = bandwidth;
        o.band_margin = band_margin;
        o.oversample = oversample;
        o.edge_taper = edge_taper;
        py::gil_scoped_release release;
        dsp::to_baseband(record, o, backend, device);
      },
      py::arg("record"), py::arg("center_frequency") = 0.0, py::arg("bandwidth") = 0.0,
      py::arg("band_margin") = 1.5, py::arg("oversample") = 2.0, py::arg("edge_taper") = 0.1,
      py::arg("backend") = Backend::Host, py::arg("device") = 0,
      "Convert record.samples in place to complex baseband at a reduced rate.");

  m.def("remove_dc", [](PingRecord& r) { dsp::remove_dc(r); }, "Remove each channel's mean.");
  m.def("apply_calibration", [](PingRecord& r) { dsp::apply_calibration(r); },
        "Apply per-channel gain, phase and delay calibration to complex baseband samples,\n"
        "then reset those tables.");
  m.def(
      "apply_tvg",
      [](PingRecord& r, double spreading_db_per_decade, double absorption_db_per_m,
         double reference_range, double max_gain_db, bool undo_hardware_ramp) {
        dsp::TvgOptions o;
        o.spreading_db_per_decade = spreading_db_per_decade;
        o.absorption_db_per_m = absorption_db_per_m;
        o.reference_range = reference_range;
        o.max_gain_db = max_gain_db;
        o.undo_hardware_ramp = undo_hardware_ramp;
        dsp::apply_tvg(r, o);
      },
      py::arg("record"), py::arg("spreading_db_per_decade") = 40.0,
      py::arg("absorption_db_per_m") = 0.0, py::arg("reference_range") = 1.0,
      py::arg("max_gain_db") = 120.0, py::arg("undo_hardware_ramp") = true,
      "Deterministic digital time-varying gain; makes up what the hardware ramp did not.");
  m.def(
      "whiten",
      [](const py::list& pings, double gamma, double smooth_hz, bool power_gain) {
        std::vector<PingRecord*> ptrs;
        for (const auto& item : pings) ptrs.push_back(&item.cast<PingRecord&>());
        dsp::WhitenOptions o;
        o.gamma = gamma;
        o.smooth_hz = smooth_hz;
        o.power_gain = power_gain;
        const auto g = dsp::whiten(ptrs, o);
        return py::array_t<float>(static_cast<py::ssize_t>(g.size()), g.data());
      },
      py::arg("pings"), py::arg("gamma") = 0.1, py::arg("smooth_hz") = 0.0,
      py::arg("power_gain") = true,
      "Spectral whitening (ASASIN Eq. 9), in place. Returns G(f) in FFT bin order.");
  m.def(
      "median_tvg",
      [](const py::list& pings, double alpha, bool power_gain, std::int64_t smooth_samples) {
        std::vector<PingRecord*> ptrs;
        for (const auto& item : pings) ptrs.push_back(&item.cast<PingRecord&>());
        dsp::MedianTvgOptions o;
        o.alpha = alpha;
        o.power_gain = power_gain;
        o.smooth_samples = smooth_samples;
        const auto g = dsp::median_tvg(ptrs, o);
        return py::array_t<float>(static_cast<py::ssize_t>(g.size()), g.data());
      },
      py::arg("pings"), py::arg("alpha") = 0.01, py::arg("power_gain") = true,
      py::arg("smooth_samples") = 0,
      "Median time-varying gain (ASASIN Eq. 10), in place. Returns G(t).");
  m.def(
      "drc",
      [](const py::array_t<float, py::array::c_style | py::array::forcecast>& magnitude,
         double target_median, double percentile) {
        std::vector<float> v(magnitude.data(), magnitude.data() + magnitude.size());
        dsp::DrcOptions o;
        o.target_median = target_median;
        o.percentile = percentile;
        const double q = dsp::drc(v, o);
        py::array_t<float> out(std::vector<py::ssize_t>(magnitude.shape(), magnitude.shape() + magnitude.ndim()));
        std::copy(v.begin(), v.end(), out.mutable_data());
        return py::make_tuple(out, q);
      },
      py::arg("magnitude"), py::arg("target_median") = 0.2, py::arg("percentile") = 99.99,
      "Dynamic range compression (ASASIN Eq. 11-12). Returns (image in [0, 1], q).");

  m.def(
      "pulse_compress",
      [](PingRecord& record, const std::string& window, double doppler_scale, Backend backend, int device) {
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
        dsp::pulse_compress(record, o, backend, device);
      },
      py::arg("record"), py::arg("window") = "none", py::arg("doppler_scale") = 1.0,
      py::arg("backend") = Backend::Host, py::arg("device") = 0,
      "Matched-filter record.samples (complex baseband) in place. An echo of delay tau\n"
      "peaks at time tau with its amplitude preserved.");

  const char* bp_doc =
      "Time-domain backprojection of conditioned pings onto points [n, 3].\n"
      "Returns the complex coherent sum per point. With cull_block > 0 and grid_shape\n"
      "(rows, cols), blocks of points outside a ping's beams are skipped for that ping;\n"
      "pass a dict as `stats` to receive the block counts.";
  m.def(
      "backproject",
      [](const py::list& pings, const ArrD& points, const std::string& interp, bool stop_and_hop,
         float fov_factor, bool directivity_weight, int spreading, const py::object& weights,
         Backend backend, bool return_weight_sum, int cull_block, const py::object& grid_shape,
         const py::object& stats_out, const std::string& position_lookup,
         int trajectory_upsample, const py::object& refraction, int device) -> py::object {
        const int upsample = position_lookup == "analytic" ? 0 : trajectory_upsample;
        recon::BpSession session(session_inputs(pings, weights, upsample));
        return session_backproject(session, points, interp, stop_and_hop, fov_factor, directivity_weight,
                                   spreading, backend, return_weight_sum, cull_block, grid_shape,
                                   stats_out, position_lookup, refraction, device);
      },
      py::arg("pings"), py::arg("points"), py::arg("interp") = "sinc",
      py::arg("stop_and_hop") = false, py::arg("fov_factor") = 0.0F,
      py::arg("directivity_weight") = false, py::arg("spreading") = 0,
      py::arg("weights") = py::none(), py::arg("backend") = Backend::Host,
      py::arg("return_weight_sum") = false, py::arg("cull_block") = 0,
      py::arg("grid_shape") = py::none(), py::arg("stats") = py::none(),
      py::arg("position_lookup") = "analytic", py::arg("trajectory_upsample") = 8,
      py::arg("refraction") = py::none(), py::arg("device") = 0, bp_doc);

  py::class_<recon::BpSession>(m, "BackprojectSession",
      "A collection prepared once and imaged many times (tiles, depth layers, several\n"
      "devices). On the CUDA backend the samples stay on the device between calls.")
      .def(py::init([](const py::list& pings, const py::object& weights, int trajectory_upsample) {
             return std::make_unique<recon::BpSession>(session_inputs(pings, weights, trajectory_upsample));
           }),
           py::arg("pings"), py::arg("weights") = py::none(), py::arg("trajectory_upsample") = 0)
      .def_property_readonly("resident_bytes", &recon::BpSession::resident_bytes,
                             "Bytes this session holds on CUDA devices.")
      .def(
          "backproject",
          [](recon::BpSession& session, const ArrD& points, const std::string& interp, bool stop_and_hop,
             float fov_factor, bool directivity_weight, int spreading, Backend backend,
             bool return_weight_sum, int cull_block, const py::object& grid_shape,
             const py::object& stats_out, const std::string& position_lookup,
             const py::object& refraction, int device) -> py::object {
            return session_backproject(session, points, interp, stop_and_hop, fov_factor,
                                       directivity_weight, spreading, backend, return_weight_sum,
                                       cull_block, grid_shape, stats_out, position_lookup, refraction,
                                       device);
          },
          py::arg("points"), py::arg("interp") = "sinc", py::arg("stop_and_hop") = false,
          py::arg("fov_factor") = 0.0F, py::arg("directivity_weight") = false,
          py::arg("spreading") = 0, py::arg("backend") = Backend::Host,
          py::arg("return_weight_sum") = false, py::arg("cull_block") = 0,
          py::arg("grid_shape") = py::none(), py::arg("stats") = py::none(),
          py::arg("position_lookup") = "analytic", py::arg("refraction") = py::none(),
          py::arg("device") = 0, bp_doc);

  m.def("refracted_time", &recon::refracted_time, py::arg("sensor"), py::arg("point"),
        py::arg("c_water"), py::arg("c_sediment"), py::arg("interface_z"),
        "One-way travel time with Snell refraction at a flat interface (the kernel's routine).");

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
