// Refraction in operation: depth from multipath timing (libs/live multipath_depth.hpp), for
// gpusonar/multipath_depth.py to be held against and for the Python chain engine, which runs the
// same C++ stage as the C++ engine.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/live/coverage.hpp"
#include "sonar/live/multipath_depth.hpp"
#include "sonar/live/raytrace.hpp"
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

live::MultipathProfile multipath_profile(const py::dict& d) {
  live::MultipathProfile p;
  p.depth = d["depth"].cast<std::vector<double>>();
  p.speed = d["speed"].cast<std::vector<double>>();
  if (d.contains("sigma") && !d["sigma"].is_none()) p.sigma = d["sigma"].cast<std::vector<double>>();
  if (d.contains("gradient_sigma")) p.gradient_sigma = d["gradient_sigma"].cast<double>();
  if (d.contains("rung")) p.rung = d["rung"].cast<std::string>();
  if (d.contains("ensemble") && !d["ensemble"].is_none()) p.ensemble = d["ensemble"].cast<std::vector<std::vector<double>>>();
  return p;
}

live::RayBottom ray_bottom(const py::object& bottom) {
  live::RayBottom b;
  if (bottom.is_none()) return b;
  const py::dict d = bottom.cast<py::dict>();
  b.given = true;
  b.speed = d["speed"].cast<double>();
  b.density = d["density"].cast<double>();
  if (d.contains("attenuation")) b.attenuation = d["attenuation"].cast<double>();
  return b;
}

py::dict ray_dict(const live::RayEnd& r) {
  py::dict d;
  d["z"] = r.z;
  d["time"] = r.time;
  d["s"] = r.s;
  d["n_surface"] = r.n_surface;
  d["n_bottom"] = r.n_bottom;
  d["turns"] = r.turns;
  d["ok"] = r.ok;
  return d;
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

// With "profile" in the geometry and `rays` in the options: the rays the fit goes through.
live::MultipathGeometry multipath_geometry(const py::dict& d, const live::MultipathOptions& o) {
  live::MultipathGeometry g = multipath_geometry(d);
  if (o.rays && d.contains("profile") && !d["profile"].is_none()) g.rays = live::multipath_rays(multipath_profile(d["profile"].cast<py::dict>()), g, o);
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
  d["profile_source"] = e.profile_source;
  d["searched"] = py::make_tuple(e.searched_lo, e.searched_hi);
  d["ensemble"] = e.ensemble;
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
        const live::MultipathOptions o = multipath_options(options);
        live::multipath_pattern(z, t0, multipath_geometry(geometry, o), o, delay, amp);
        return py::make_tuple(delay, amp);
      },
      "Two-way arrivals of a target at depth z whose direct echo has two-way time t0: (delays s, amplitudes re the direct one).");
  m.def("live_multipath_depth_grid",
        [](const py::dict& geometry, const py::dict& options) {
          const live::MultipathOptions o = multipath_options(options);
          return live::multipath_depth_grid(multipath_geometry(geometry, o), o);
        });
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
        const live::MultipathOptions o = multipath_options(options);
        return estimate_dict(live::multipath_estimate(y.data(), static_cast<std::int64_t>(y.size()), t_first, dt, n_noise, multipath_geometry(geometry, o), o));
      },
      "Target depth from one beam window (multipath_depth.hpp).");
  m.def(
      "live_multipath_detection",
      [](const PingRecord& record, const Vec3d& origin, double bearing, double range, const py::dict& options, double first_range, double extent,
         const std::vector<std::vector<std::int64_t>>& rows, Backend backend, const py::object& profile) {
        live::MultipathProfile water;
        if (!profile.is_none()) water = multipath_profile(profile.cast<py::dict>());
        return estimate_dict(live::multipath_detection(record, origin, bearing, range, multipath_options(options), first_range, extent, rows, backend, 0,
                                                       profile.is_none() ? nullptr : &water));
      },
      py::arg("record"), py::arg("origin"), py::arg("bearing"), py::arg("range_m"), py::arg("options"), py::arg("first_range") = kNaN,
      py::arg("extent") = kNaN, py::arg("rows") = std::vector<std::vector<std::int64_t>>{}, py::arg("backend") = Backend::Host,
      py::arg("profile") = py::none(),
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

  m.def(
      "live_multipath_rays",
      [](const py::dict& geometry, const py::dict& options, const std::vector<double>& depths) {
        const live::MultipathOptions o = multipath_options(options);
        const live::MultipathGeometry g = multipath_geometry(geometry, o);
        if (!g.rays) throw std::invalid_argument("live_multipath_rays: needs options.rays and geometry.profile");
        py::list out;
        for (double z : depths) {
          py::dict d;
          d["paths"] = live::multipath_ray_paths(*g.rays, z);
          d["surface_delay"] = live::multipath_ray_surface_delay(*g.rays, z);
          if (live::multipath_ray_paths(*g.rays, z) == 3) {
            d["sigma"] = live::multipath_ray_sigma(*g.rays, z);
            d["ensemble"] = live::multipath_ray_ensemble(*g.rays, z);
          }
          out.append(d);
        }
        return out;
      },
      "What the rays of a fit say of each depth: paths (1 direct, 2 surface), the both-ways surface delay, the profile's sigma, the ensemble's depths.");

  // --- the ray code itself (raytrace.hpp), for the tests against gpusonar/raytrace.py and for tools ---
  m.def(
      "ray_trace",
      [](const std::vector<double>& depth, const std::vector<double>& speed, double water_depth, double z0, double launch, double r_end) {
        return ray_dict(live::ray_trace(live::RayProfile(depth, speed, water_depth), z0, launch, r_end));
      },
      "The ray leaving depth z0 at grazing angle launch (rad, positive down), at horizontal range r_end (raytrace.hpp).");
  m.def("ray_depth_arcs", &live::ray_depth_arcs, py::arg("receiver_depth"), py::arg("elevation"), py::arg("slant_range"), py::arg("ssp_depth"),
        py::arg("ssp_speed"), py::arg("water_depth") = 0.0, "Depth an arrival started at, back along the ray in closed-form arcs (raytrace.hpp).");
  m.def(
      "ray_eigenrays",
      [](const std::vector<double>& depth, const std::vector<double>& speed, double water_depth, double z_source, double z_receiver, double range,
         double max_angle_deg, std::int64_t n_fan, int max_bounces, double surface_coefficient, const py::object& bottom, double caustic_clip) {
        live::EigenrayOptions o;
        o.max_angle_deg = max_angle_deg;
        o.n_fan = n_fan;
        o.max_bounces = max_bounces;
        o.surface_coefficient = surface_coefficient;
        o.bottom = ray_bottom(bottom);
        o.caustic_clip = caustic_clip;
        std::vector<live::Eigenray> rays;
        {
          py::gil_scoped_release release;
          rays = live::eigenrays(live::RayProfile(depth, speed, water_depth), z_source, z_receiver, range, o);
        }
        py::list out;
        for (const live::Eigenray& e : rays) {
          py::dict d;
          d["launch"] = e.launch;
          d["arrive"] = e.arrive;
          d["time"] = e.time;
          d["amplitude"] = e.amplitude;
          d["n_surface"] = e.n_surface;
          d["n_bottom"] = e.n_bottom;
          d["turns"] = e.turns;
          d["caustic"] = e.caustic;
          out.append(d);
        }
        return out;
      },
      py::arg("depth"), py::arg("speed"), py::arg("water_depth"), py::arg("z_source"), py::arg("z_receiver"), py::arg("range_m"),
      py::arg("max_angle_deg") = 60.0, py::arg("n_fan") = 2401, py::arg("max_bounces") = 4, py::arg("surface_coefficient") = -1.0,
      py::arg("bottom") = py::none(), py::arg("caustic_clip") = 10.0, "Eigenrays through a layered profile, earliest first (raytrace.hpp).");
  py::class_<live::RayFan>(m, "RayFan")
      .def(py::init([](const std::vector<double>& depth, const std::vector<double>& speed, double water_depth, double z_source,
                       const std::vector<double>& ranges, double max_angle_deg, std::int64_t n, int max_bounces, double caustic_clip) {
             return live::RayFan(live::RayProfile(depth, speed, water_depth), z_source, ranges, max_angle_deg * 3.14159265358979323846 / 180.0, n, max_bounces,
                                 caustic_clip);
           }),
           py::arg("depth"), py::arg("speed"), py::arg("water_depth"), py::arg("z_source"), py::arg("ranges"), py::arg("max_angle_deg"), py::arg("n"),
           py::arg("max_bounces"), py::arg("caustic_clip") = 10.0, "A fan of rays from one depth, read at each of `ranges` (raytrace.hpp).")
      .def(
          "arrivals",
          [](const live::RayFan& fan, std::size_t i_range, double z) {
            std::vector<live::RayArrival> got;
            fan.arrivals(i_range, z, got);
            py::list out;
            for (const live::RayArrival& a : got) {
              py::dict d;
              d["time"] = a.time;
              d["p"] = a.p;
              d["launch"] = a.launch;
              d["s_end"] = a.s_end;
              d["amplitude"] = a.amplitude;
              d["grazing_surface"] = a.grazing_surface;
              d["grazing_bottom"] = a.grazing_bottom;
              d["n_surface"] = a.n_surface;
              d["n_bottom"] = a.n_bottom;
              d["caustic"] = a.caustic;
              out.append(d);
            }
            return out;
          },
          "The arrivals at depth z at ranges[i_range]: geometric amplitude, counts and grazing angles for the caller's coefficients.");

  // --- the coverage picture (coverage.hpp) ---
  auto coverage_options = [](const py::dict& d) {
    live::CoverageOptions o;
    o.enabled = true;
    for (const auto& item : d) {
      const std::string key = "coverage_" + item.first.cast<std::string>();
      if (!live::set_coverage_option(o, key, py::str(item.second).cast<std::string>())) throw std::invalid_argument("unknown option " + key);
    }
    return o;
  };
  auto coverage_sonar = [](const py::dict& d) {
    live::CoverageSonar s;
    s.f_centre = d["f_centre"].cast<double>();
    s.bandwidth = d["bandwidth"].cast<double>();
    s.pulse = d["pulse"].cast<double>();
    return s;
  };
  m.def(
      "coverage_field",
      [coverage_options, coverage_sonar](const py::dict& water, double water_depth, double array_depth, double projector_depth, const std::vector<double>& ranges,
                                         const std::vector<double>& depths, const py::dict& options, const py::dict& sonar) {
        const live::CoverageOptions o = coverage_options(options);
        const live::CoverageSonar s = coverage_sonar(sonar);
        const live::MultipathProfile w = multipath_profile(water);
        live::CoverageField f;
        {
          py::gil_scoped_release release;
          f = live::coverage_field(w, water_depth, array_depth, projector_depth, ranges, depths, o, s);
        }
        const auto n_d = static_cast<py::ssize_t>(depths.size()), n_r = static_cast<py::ssize_t>(ranges.size());
        auto grid = [&](const std::vector<double>& v) {
          py::array_t<double> a({n_d, n_r});
          std::copy(v.begin(), v.end(), a.mutable_data());
          return a;
        };
        py::dict d;
        d["factor_db"] = grid(f.factor_db);
        d["coherent_db"] = grid(f.coherent_db);
        d["incoherent_db"] = grid(f.incoherent_db);
        d["slant"] = grid(f.slant);
        py::array_t<bool> direct({n_d, n_r});
        std::copy(f.direct.begin(), f.direct.end(), direct.mutable_data());
        d["direct"] = direct;
        py::array_t<double> snr({n_d, n_r});
        for (std::size_t i = 0; i < f.factor_db.size(); ++i) snr.mutable_data()[i] = live::coverage_snr_db(o, s, f.slant[i], 0.0, f.factor_db[i]);
        d["snr_0db_target_db"] = snr;
        return d;
      },
      py::arg("water"), py::arg("water_depth"), py::arg("array_depth"), py::arg("projector_depth"), py::arg("ranges"), py::arg("depths"),
      py::arg("options"), py::arg("sonar"),
      "The two-way propagation factor of the coverage picture on target depths x horizontal ranges (coverage.hpp): factor_db, the one-way coherent and "
      "incoherent figures, where a direct path exists, and one row's link-budget SNR of a 0 dB target.");
  py::class_<live::Coverage>(m, "Coverage")
      .def(py::init([coverage_options](const py::dict& options) { return live::Coverage(coverage_options(options)); }), py::arg("options") = py::dict(),
           "Options as gpusonar.coverage.cpp_options makes them.")
      .def("set_pd_table", &live::Coverage::set_pd_table)
      .def(
          "update",
          [](live::Coverage& c, const PingRecord& record, const Vec3d& origin, double water_depth, const py::dict& water) {
            return c.update(record, origin, water_depth, multipath_profile(water));
          },
          "The report member for this ping (a JSON member without braces): recomputed when something changed materially, else repeated.")
      .def(
          "update_numbers",
          [coverage_sonar](live::Coverage& c, const py::dict& sonar, double range_first, double range_last, double array_depth, double projector_depth,
                           double water_depth, const py::dict& water) {
            return c.update(coverage_sonar(sonar), range_first, range_last, array_depth, projector_depth, water_depth, multipath_profile(water));
          })
      .def("refreshed", &live::Coverage::refreshed)
      .def("seconds", &live::Coverage::seconds)
      .def("seen", &live::Coverage::seen, "Detection probability of class i at a horizontal range at the present depth (worst over its depths).");

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
      .def("set_profile", [](live::MultipathDepth& s, const py::dict& profile) { s.set_profile(multipath_profile(profile)); },
           "The water this ping's rays go through (options.rays): {depth, speed, sigma, gradient_sigma, rung, ensemble}.")
      .def("add", [](live::MultipathDepth& s, std::int64_t track, const py::dict& estimate,
                     const py::dict& geometry) { s.add(track, estimate_from(estimate), multipath_geometry(geometry)); })
      .def("track", [](const live::MultipathDepth& s, std::int64_t id) { return estimate_dict(s.track(id)); });
}
