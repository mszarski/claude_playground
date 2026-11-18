// Refraction in operation: depth from multipath timing (libs/live multipath_depth.hpp), for
// gpusonar/multipath_depth.py to be held against and for the Python chain engine, which runs the
// same C++ stage as the C++ engine.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/live/multipath_depth.hpp"
#include "vec_caster.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

live::MultipathOptions multipath_options(const py::dict& d) {
  live::MultipathOptions o;
  o.enabled = true;
  for (const auto& item : d) {
    const std::string key = "multipath_depth_" + item.first.cast<std::string>();
    if (!live::set_multipath_option(o, key, py::str(item.second).cast<std::string>())) throw std::invalid_argument("unknown option " + key);
  }
  return o;
}

live::MultipathGeometry multipath_geometry(const py::dict& d) {
  live::MultipathGeometry g;
  g.range = d["range_m"].cast<double>();
  g.array_depth = d["array_depth"].cast<double>();
  g.projector_depth = d["projector_depth"].cast<double>();
  g.sound_speed = d["sound_speed"].cast<double>();
  g.f_centre = d["f_centre"].cast<double>();
  g.bandwidth = d["bandwidth"].cast<double>();
  if (d.contains("water_depth") && !d["water_depth"].is_none()) g.water_depth = d["water_depth"].cast<double>();
  if (d.contains("slack_m")) g.slack = d["slack_m"].cast<double>();
  return g;
}

py::dict estimate_dict(const live::MultipathEstimate& e) {
  py::dict d;
  d["valid"] = e.valid;
  d["depth"] = e.depth;
  d["sigma"] = e.sigma;
  d["refraction_sigma"] = e.refraction_sigma;
  d["gain"] = e.gain;
  d["arrivals"] = e.arrivals;
  d["method"] = e.method;
  d["ambiguous"] = e.ambiguous;
  d["alternatives"] = e.alternatives;
  d["reason"] = e.reason;
  d["snr_db"] = e.snr_db;
  d["grid"] = py::array_t<double>(static_cast<py::ssize_t>(e.grid.size()), e.grid.data());
  d["envelope"] = py::array_t<double>(static_cast<py::ssize_t>(e.envelope.size()), e.envelope.data());
  return d;
}

live::MultipathEstimate estimate_from(const py::dict& d) {
  live::MultipathEstimate e;
  e.grid = d["grid"].cast<std::vector<double>>();
  e.envelope = d["envelope"].cast<std::vector<double>>();
  return e;
}

}  // namespace

void bind_refraction(py::module_& m) {
  m.def(
      "live_multipath_pattern",
      [](double z, double t0, const py::dict& geometry, const py::dict& options) {
        std::vector<double> delay;
        std::vector<std::complex<double>> amp;
        live::multipath_pattern(z, t0, multipath_geometry(geometry), multipath_options(options), delay, amp);
        return py::make_tuple(delay, amp);
      },
      "Two-way arrivals of a target at depth z whose direct echo has two-way time t0: (delays s, amplitudes re the direct one).");
  m.def("live_multipath_depth_grid",
        [](const py::dict& geometry, const py::dict& options) { return live::multipath_depth_grid(multipath_geometry(geometry), multipath_options(options)); });
  m.def("live_multipath_window_times", [](const py::dict& geometry, const py::dict& options, double spread) {
    double t_first = 0.0, dt = 0.0;
    std::int64_t n = 0, n_noise = 0;
    live::multipath_window_times(multipath_geometry(geometry), multipath_options(options), spread, t_first, dt, n, n_noise);
    return py::make_tuple(t_first, dt, n, n_noise);
  });
  m.def(
      "live_multipath_window",
      [](const PingRecord& record, double bearing, double range, double t_first, double dt, std::int64_t n,
         const std::vector<std::vector<std::int64_t>>& rows, Backend backend) {
        const std::vector<std::complex<double>> y = live::multipath_window(record, bearing, range, t_first, dt, n, rows, backend);
        return py::array_t<std::complex<double>>(static_cast<py::ssize_t>(y.size()), y.data());
      },
      py::arg("record"), py::arg("bearing"), py::arg("range_m"), py::arg("t_first"), py::arg("dt"), py::arg("n"),
      py::arg("rows") = std::vector<std::vector<std::int64_t>>{}, py::arg("backend") = Backend::Host,
      "Compressed baseband output of the array steered at a detection (multipath_depth.hpp).");
  m.def(
      "live_multipath_estimate",
      [](const py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>& y, double t_first, double dt, std::int64_t n_noise,
         const py::dict& geometry, const py::dict& options) {
        return estimate_dict(live::multipath_estimate(y.data(), static_cast<std::int64_t>(y.size()), t_first, dt, n_noise, multipath_geometry(geometry),
                                                      multipath_options(options)));
      },
      "Target depth from one beam window (multipath_depth.hpp).");
  m.def(
      "live_multipath_detection",
      [](const PingRecord& record, const Vec3d& origin, double bearing, double range, const py::dict& options, double first_range, double extent,
         const std::vector<std::vector<std::int64_t>>& rows, Backend backend) {
        return estimate_dict(live::multipath_detection(record, origin, bearing, range, multipath_options(options), first_range, extent, rows, backend));
      },
      py::arg("record"), py::arg("origin"), py::arg("bearing"), py::arg("range_m"), py::arg("options"), py::arg("first_range") = kNaN,
      py::arg("extent") = kNaN, py::arg("rows") = std::vector<std::vector<std::int64_t>>{}, py::arg("backend") = Backend::Host,
      "Target depth for one detection of a conditioned record.");
  m.def("live_multipath_fuse",
        [](double depth_mp, double sigma_mp, double depth_el, double sigma_el, const std::vector<double>& alternatives, double consistency_sigma,
           double refraction_mp, double refraction_el) {
          const live::MultipathFused f =
              live::multipath_fuse(depth_mp, sigma_mp, depth_el, sigma_el, alternatives, consistency_sigma, refraction_mp, refraction_el);
          py::dict d;
          d["depth"] = f.depth;
          d["sigma"] = f.sigma;
          d["consistent"] = f.consistent;
          d["selected"] = f.selected;
          return d;
        },
        py::arg("depth_mp"), py::arg("sigma_mp"), py::arg("depth_el"), py::arg("sigma_el"), py::arg("alternatives") = std::vector<double>{},
        py::arg("consistency_sigma") = 3.0, py::arg("refraction_mp") = 0.0, py::arg("refraction_el") = 0.0);
  m.def("live_multipath_refraction", [](double flat_depth, const py::dict& geometry, const py::dict& options) {
    double depth = 0.0, sigma = 0.0;
    live::multipath_refraction(flat_depth, multipath_geometry(geometry), multipath_options(options), depth, sigma);
    return py::make_tuple(depth, sigma);
  });

  py::class_<live::MultipathDepth>(m, "MultipathDepth")
      .def(py::init([](const py::dict& options) { return live::MultipathDepth(multipath_options(options)); }), py::arg("options") = py::dict(),
           "Options as gpusonar.multipath_depth.cpp_options makes them.")
      .def(
          "process",
          [](live::MultipathDepth& s, const PingRecord& record, const Vec3d& origin, const py::list& hits, const py::list& tracks,
             const std::vector<std::vector<std::int64_t>>& rows, Backend backend) {
            std::vector<live::MultipathHit> mh;
            for (const auto& item : hits) {
              const py::dict h = item.cast<py::dict>();
              live::MultipathHit x;
              x.range = h["range"].cast<double>();
              x.bearing = h["bearing"].cast<double>();
              if (h.contains("first_range")) x.first_range = h["first_range"].cast<double>();
              if (h.contains("extent")) x.extent = h["extent"].cast<double>();
              x.x = h["x"].cast<double>();
              x.y = h["y"].cast<double>();
              x.snr_db = h["snr_db"].cast<double>();
              if (h.contains("depth")) {
                x.has_depth = true;
                x.depth = h["depth"].cast<double>();
                x.depth_sigma = h["depth_sigma"].cast<double>();
                if (h.contains("depth_refraction_sigma")) x.depth_refraction_sigma = h["depth_refraction_sigma"].cast<double>();
                if (h.contains("depth_alternatives")) x.depth_alternatives = h["depth_alternatives"].cast<std::vector<double>>();
              }
              mh.push_back(x);
            }
            std::vector<live::MultipathTrack> mt;
            for (const auto& item : tracks) {
              const py::dict t = item.cast<py::dict>();
              live::MultipathTrack x;
              x.id = t["id"].cast<std::int64_t>();
              x.x = t["x"].cast<double>();
              x.y = t["y"].cast<double>();
              if (t.contains("depth")) {
                x.has_depth = true;
                x.depth = t["depth"].cast<double>();
                x.depth_sigma = t["depth_sigma"].cast<double>();
              }
              mt.push_back(x);
            }
            live::MultipathReport r;
            {
              py::gil_scoped_release release;
              r = s.process(record, origin, mh, mt, rows, backend);
            }
            return py::make_tuple(r.detections, r.tracks, r.seconds);
          },
          py::arg("record"), py::arg("origin"), py::arg("hits"), py::arg("tracks"), py::arg("rows") = std::vector<std::vector<std::int64_t>>{},
          py::arg("backend") = Backend::Host, "One ping: JSON members to add to each detection and each track (strings without braces), and the seconds it took.")
      .def("add", [](live::MultipathDepth& s, std::int64_t track, const py::dict& estimate,
                     const py::dict& geometry) { s.add(track, estimate_from(estimate), multipath_geometry(geometry)); })
      .def("track", [](const live::MultipathDepth& s, std::int64_t id) { return estimate_dict(s.track(id)); });
}
