#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sonar/live/chain.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/environment.hpp"
#include "sonar/live/elevation.hpp"
#include "sonar/live/option_value.hpp"
#include "sonar/live/options.hpp"
#include "sonar/live/sbl_kernel.hpp"
#include "sonar/live/udp_source.hpp"

namespace py = pybind11;
using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

py::dict track_dict(const live::Track& t) {
  py::dict d;
  d["id"] = t.id;
  d["state"] = std::vector<double>(t.state, t.state + 4);
  d["hits"] = t.hits;
  d["misses"] = t.misses;
  d["confirmed"] = t.confirmed;
  py::list h;
  for (const auto& p : t.history) h.append(py::make_tuple(p.time, p.x, p.y));
  d["history"] = h;
  py::list m;   // the measured position of each history row, NaN on a miss (detect.py Track.measured; D-157)
  for (const auto& p : t.history) m.append(py::make_tuple(p.mx, p.my));
  d["measured"] = m;
  py::list pings;   // the ping (transmit) time of each history row (detect.py Track.pings; sonar-054)
  for (const auto& p : t.history) pings.append(p.ping);
  d["pings"] = pings;
  d["snr"] = t.snr;
  d["cells"] = t.cells;
  d["time"] = t.time;
  return d;
}

py::dict doa_dict(const live::DoaResult& r) {
  py::dict d;
  d["spectrum"] = py::array_t<float>(static_cast<py::ssize_t>(r.spectrum.size()), r.spectrum.data());
  py::list peaks;
  for (const auto& p : r.peaks) peaks.append(py::make_tuple(p.bearing, p.power, p.snr_db));
  d["peaks"] = peaks;
  d["noise"] = r.noise;
  d["iterations"] = r.iterations;
  d["fallback"] = r.fallback;      // more channels than the SBL kernels take: solved by the host reference
  return d;
}

live::DoaOptions doa_options(int max_iterations, double tolerance, int max_sources, double dynamic_range_db, double min_snr_db,
                             Backend backend = Backend::Host, bool kernels = true) {
  live::DoaOptions o;
  o.backend = backend;
  o.kernels = kernels;
  o.max_iterations = max_iterations;
  o.tolerance = tolerance;
  o.max_sources = max_sources;
  o.dynamic_range_db = dynamic_range_db;
  o.min_snr_db = min_snr_db;
  return o;
}


// --- stage 2: options from dicts (decimation, reference, water depth, sound-speed ladder) --------

// An options dict that remembers every key looked up in it, so that a key nothing looked up (a misspelt one, which
// would otherwise leave its setting at the default without a word) can be refused by name: the keys taken are
// exactly the keys read (sonar-6hf). Made from a py::dict where one is passed, so the helpers below take either.
class Keys {
 public:
  Keys(const py::dict& d) : d_(d) {}  // NOLINT(google-explicit-constructor): every helper is called with a plain dict too
  bool contains(const char* k) const {
    read_.emplace_back(k);
    return d_.contains(k);
  }
  py::object operator[](const char* k) const {
    read_.emplace_back(k);
    return d_[k];
  }
  // Throws ValueError naming the first key of the dict nothing has looked up, and the nearest key that was.
  void refuse_unread(const std::string& where) const {
    for (const auto& item : d_) {
      const std::string key = py::str(item.first).cast<std::string>();
      if (std::find(read_.begin(), read_.end(), key) != read_.end()) continue;
      const std::string nearest = live::nearest_key(key, read_);   // within max(3, len / 3) edits (option_value.hpp)
      throw py::value_error(where + key + ": unknown key" + (nearest.empty() ? std::string(" (no key read here is near it)") : "; did you mean '" + nearest + "'?"));
    }
  }

 private:
  py::dict d_;
  mutable std::vector<std::string> read_;
};

void number(const Keys& d, const char* k, double& v) {
  if (d.contains(k) && !d[k].is_none()) v = d[k].cast<double>();
}

live::DecimateOptions decimate_options(const Keys& d) {
  live::DecimateOptions o;
  o.enabled = true;
  number(d, "f_low", o.f_low);
  number(d, "f_high", o.f_high);
  number(d, "band_margin", o.band_margin);
  number(d, "output_rate", o.output_rate);
  number(d, "oversample", o.oversample);
  number(d, "ripple_db", o.ripple_db);
  number(d, "stopband_db", o.stopband_db);
  if (d.contains("taps") && !d["taps"].is_none()) o.taps = d["taps"].cast<std::int64_t>();
  number(d, "blank_until_s", o.blank_until_s);
  number(d, "blank_taper_s", o.blank_taper_s);
  return o;
}

live::Reference reference_options(const Keys& r) {
  live::Reference o;
  o.f_start = r["f_start"].cast<double>();
  o.f_stop = r["f_stop"].cast<double>();
  o.duration = r["duration"].cast<double>();
  o.lead = r["lead"].cast<double>();
  o.tail = r["tail"].cast<double>();
  if (r.contains("dedicated")) o.dedicated = r["dedicated"].cast<bool>();
  if (r.contains("min_quality") && !r["min_quality"].is_none()) o.min_quality = r["min_quality"].cast<double>();
  number(r, "search_time", o.search_time);
  number(r, "tolerance_us", o.tolerance_us);
  number(r, "clip_level", o.clip_level);
  return o;
}

live::WaterDepthOptions water_options(const Keys& d) {
  live::WaterDepthOptions o;
  if (d.contains("from_altitude")) o.from_altitude = d["from_altitude"].cast<bool>();
  // `source` as a deployment file's nav.water_depth says it (live.py WATER_DEPTH_DEFAULTS): "altitude" is from_altitude.
  if (d.contains("source") && !d["source"].is_none()) {
    const std::string s = d["source"].cast<std::string>();
    if (s != "altitude" && s != "datagram") throw py::value_error("water_depth.source: altitude or datagram, got '" + s + "'");
    if (d.contains("from_altitude") && o.from_altitude != (s == "altitude"))
      throw py::value_error("water_depth: source '" + s + "' and from_altitude disagree");
    o.from_altitude = s == "altitude";
  }
  number(d, "dvl_offset_m", o.dvl_offset_m);
  number(d, "altitude_sigma_m", o.altitude_sigma_m);
  number(d, "depth_sigma_m", o.depth_sigma_m);
  number(d, "hold_s", o.hold_s);
  number(d, "hold_sigma_m_per_s", o.hold_sigma_m_per_s);
  number(d, "hold_sigma_m_per_m", o.hold_sigma_m_per_m);
  number(d, "max_altitude_m", o.max_altitude_m);
  return o;
}

// A binding's options dict read whole, then refused if it holds a key nothing read, as LiveChainCpp does (sonar-1rf).
template <class Read>
auto read_dict(const py::object& d, const std::string& where, Read read) {
  const Keys k(d.cast<py::dict>());
  auto o = read(k);
  k.refuse_unread(where);
  return o;
}

live::DecimateOptions decimate_dict(const py::object& d, const std::string& where) { return read_dict(d, where + "decimate.", decimate_options); }

// The reference as a sonar file describes it: `mode`, `level` and `channels` (the burst as the simulator sends it and
// sync.py finds its channels) are taken without being read here.
live::Reference reference_dict(const py::object& d, const std::string& where) {
  return read_dict(d, where + "reference.", [](const Keys& r) {
    for (const char* k : {"mode", "level", "channels"}) r.contains(k);
    return reference_options(r);
  });
}

live::NavOptions nav_options(const Keys& d, const std::string& where) {
  live::NavOptions o;
  if (d.contains("water_depth") && !d["water_depth"].is_none()) o.water_depth = read_dict(d["water_depth"], where + "water_depth.", water_options);
  number(d, "keep_s", o.keep_s);
  number(d, "surface_z", o.surface_z);
  number(d, "profile_step_m", o.profile_step_m);
  number(d, "history_margin_s", o.history_margin_s);
  // The array from the navigation unit (sonar-3rv): lever_arm_m (m, the unit's body axes), mount_rpy_deg (degrees).
  for (const char* key : {"lever_arm_m", "mount_rpy_deg"}) {
    if (!d.contains(key) || d[key].is_none()) continue;
    const auto q = d[key].cast<std::vector<double>>();
    if (q.size() != 3 || !std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2]))
      throw std::invalid_argument(std::string("nav.") + key + ": three finite numbers");
    const double f = std::string(key) == "lever_arm_m" ? 1.0 : 3.14159265358979323846 / 180.0;
    (std::string(key) == "lever_arm_m" ? o.lever_arm : o.mount_rpy) = Vec3d{q[0] * f, q[1] * f, q[2] * f};
  }
  return o;
}

live::ProfileOptions profile_options(const Keys& d) {
  live::ProfileOptions o;
  o.enabled = true;
  if (d.contains("ladder")) o.ladder = d["ladder"].cast<std::vector<std::string>>();
  number(d, "bin_m", o.bin_m);
  number(d, "sample_sigma", o.sample_sigma);
  number(d, "age_sigma_per_hour", o.age_sigma_per_hour);
  number(d, "distance_sigma_per_km", o.distance_sigma_per_km);
  number(d, "max_age_s", o.max_age_s);
  number(d, "max_distance_m", o.max_distance_m);
  number(d, "min_span_m", o.min_span_m);
  number(d, "extrapolation_sigma_per_s", o.extrapolation_sigma_per_s);
  number(d, "supplied_sigma", o.supplied_sigma);
  number(d, "family_gradient", o.family_gradient);
  number(d, "layer_depth", o.layer_depth);
  number(d, "thermocline_gradient", o.thermocline_gradient);
  number(d, "thermocline_thickness", o.thermocline_thickness);
  number(d, "family_sigma", o.family_sigma);
  if (d.contains("ensemble")) o.ensemble = d["ensemble"].cast<bool>();
  if (d.contains("ensemble_gradients")) o.ensemble_gradients = d["ensemble_gradients"].cast<std::vector<double>>();
  if (d.contains("ensemble_extents")) o.ensemble_extents = d["ensemble_extents"].cast<std::vector<double>>();
  if (d.contains("ensemble_sigma_members")) o.ensemble_sigma_members = d["ensemble_sigma_members"].cast<bool>();
  if (d.contains("file") && !d["file"].is_none()) o.file = d["file"].cast<std::string>();
  if (d.contains("cast") && !d["cast"].is_none()) o.cast = d["cast"].cast<std::string>();
  if (d.contains("family") && !d["family"].is_none()) o.family = d["family"].cast<std::string>();
  return o;
}

py::dict sync_dict(const live::SyncResult& r) {
  py::dict d;
  d["ok"] = r.ok;
  d["offset"] = r.offset;
  d["drift_ppm"] = r.drift_ppm;
  d["residual"] = r.residual;
  d["quality"] = r.quality;
  d["tolerance"] = r.tolerance;
  d["flagged"] = r.flagged;
  return d;
}

py::dict plan_dict(const live::DecimatePlan& p) {
  py::dict d;
  d["factor"] = p.factor;
  d["input_rate"] = p.input_rate;
  d["output_rate"] = p.output_rate;
  d["carrier"] = p.carrier;
  d["passband"] = p.passband;
  d["stopband"] = p.stopband;
  d["taps"] = p.taps;
  d["delay_s"] = p.delay_s();
  return d;
}

py::dict ladder_dict(const live::LadderProfile& p) {
  py::dict d;
  d["rung"] = p.rung;
  d["depth"] = p.depth;
  d["speed"] = p.speed;
  d["sigma"] = p.sigma;
  d["bins"] = p.bins;
  d["span_lo"] = p.span_lo;
  d["span_hi"] = p.span_hi;
  return d;
}

}  // namespace

void bind_live(py::module_& m) {
  m.def("live_decimate_plan",
        [](const PingRecord& record, const py::dict& options) { return plan_dict(live::plan_decimation(record, decimate_dict(options, "live_decimate_plan: "))); },
        py::arg("record"), py::arg("options") = py::dict(),
        "The filter and factor the decimate-first stage uses for a record (decimate.hpp).");
  m.def(
      "live_decimate_record",
      [](PingRecord& record, const py::dict& options, const py::object& reference) {
        const live::DecimateOptions o = decimate_dict(options, "live_decimate_record: ");
        live::SyncResult r;
        if (reference.is_none()) {
          py::gil_scoped_release release;
          r = live::decimate_record(record, o);
        } else {
          const live::Reference ref = reference_dict(reference, "live_decimate_record: ");
          py::gil_scoped_release release;
          r = live::decimate_record(record, o, &ref);
        }
        return sync_dict(r);
      },
      py::arg("record"), py::arg("options") = py::dict(), py::arg("reference") = py::none(),
      "A raw record band-selected and decimated to complex baseband in place; with `reference` its boards are "
      "synchronised from the reference bursts (decimate.hpp). Returns the clock facts.");
  py::class_<live::PingDecimator>(m, "PingDecimator")
      .def(py::init([](const PingRecord& t, const py::dict& options, const py::object& reference) {
             if (reference.is_none()) return live::PingDecimator(t, decimate_dict(options, "PingDecimator: "));
             const live::Reference ref = reference_dict(reference, "PingDecimator: ");
             return live::PingDecimator(t, decimate_dict(options, "PingDecimator: "), &ref);
           }),
           py::arg("template_record"), py::arg("options") = py::dict(), py::arg("reference") = py::none())
      .def("open", &live::PingDecimator::open)
      .def("feed",
           [](live::PingDecimator& d, const py::array_t<float, py::array::c_style | py::array::forcecast>& rows) {
             if (rows.ndim() != 2) throw std::invalid_argument("rows must be [n_samples, n_channels]");
             d.feed(rows.data(), rows.shape(0));
           })
      .def("finish", [](live::PingDecimator& d, PingRecord& like) { return sync_dict(d.finish(like)); })
      .def("plan", [](const live::PingDecimator& d) { return plan_dict(d.plan()); });
  py::class_<live::StreamAssembler>(m, "StreamAssemblerCpp")
      .def(py::init([](const PingRecord& t, std::int64_t chunk, const py::object& decimate) {
             if (decimate.is_none()) return live::StreamAssembler(t, chunk);
             const live::DecimateOptions o = decimate_dict(decimate, "StreamAssemblerCpp: ");
             return live::StreamAssembler(t, chunk, &o);
           }),
           py::arg("template_record"), py::arg("chunk_samples"), py::arg("decimate") = py::none())
      .def("feed",
           [](live::StreamAssembler& a, const py::bytes& block) {
             const std::string data = block;
             std::vector<PingRecord> out;
             a.feed(reinterpret_cast<const unsigned char*>(data.data()), data.size(), out);
             return out;
           })
      .def("set_blanking", &live::StreamAssembler::set_blanking)
      .def_readonly("blanked_samples", &live::StreamAssembler::blanked_samples)
      .def_readonly("dropped_chunks", &live::StreamAssembler::dropped_chunks);
  py::class_<live::Navigation>(m, "NavigationCpp")
      .def(py::init([](const py::dict& options) {
             return live::Navigation(read_dict(options, "NavigationCpp: ", [](const Keys& k) { return nav_options(k, "NavigationCpp: "); }));
           }),
           py::arg("options") = py::dict())
      .def("push", &live::Navigation::push)
      .def("apply", &live::Navigation::apply)
      .def("velocity_given", &live::Navigation::velocity_given,
           "Whether the last apply() had a velocity over ground from navigation (udp_source.hpp; sonar-8is).")
      .def("__len__", &live::Navigation::size);
  py::class_<live::WaterDepthTracker>(m, "WaterDepthTracker")
      .def(py::init([](const py::dict& options) { return live::WaterDepthTracker(read_dict(options, "WaterDepthTracker: ", water_options)); }),
           py::arg("options") = py::dict())
      .def("update",
           [](live::WaterDepthTracker& w, std::int64_t time_ns, double x, double y, double depth, double altitude, bool lock,
              double altitude_sigma, double depth_sigma) {
             const live::WaterDepth r = w.update(time_ns, x, y, depth, altitude, lock, altitude_sigma, depth_sigma);
             return py::make_tuple(r.depth, r.sigma, r.held, r.held_s);
           },
           py::arg("time_ns"), py::arg("x"), py::arg("y"), py::arg("depth"), py::arg("altitude"), py::arg("lock") = true,
           py::arg("altitude_sigma") = kNaN, py::arg("depth_sigma") = kNaN,
           "One fix -> (water depth, sigma, held, seconds held); NaN when unknown (environment.hpp).");
  py::class_<live::SoundSpeedLadder>(m, "SoundSpeedLadder")
      .def(py::init([](const py::dict& options) { return live::SoundSpeedLadder(read_dict(options, "SoundSpeedLadder: ", profile_options)); }),
           py::arg("options") = py::dict())
      .def("add", &live::SoundSpeedLadder::add)
      .def("absorb", &live::SoundSpeedLadder::absorb, py::arg("record"), py::arg("surface_z") = 0.0)
      .def("evaluate",
           [](const live::SoundSpeedLadder& l, std::int64_t time_ns, double x, double y, double own_depth, double sound_speed, bool measured,
              double water_depth, const std::vector<double>& supplied_depth, const std::vector<double>& supplied_speed, double gradient,
              double gradient_sigma) {
             return ladder_dict(l.evaluate(time_ns, x, y, own_depth, sound_speed, measured, water_depth, supplied_depth, supplied_speed,
                                           gradient, gradient_sigma));
           },
           py::arg("time_ns"), py::arg("x"), py::arg("y"), py::arg("own_depth"), py::arg("sound_speed"), py::arg("measured"),
           py::arg("water_depth") = kNaN, py::arg("supplied_depth") = std::vector<double>{},
           py::arg("supplied_speed") = std::vector<double>{}, py::arg("gradient") = 0.0, py::arg("gradient_sigma") = 0.05)
      .def("bins", &live::SoundSpeedLadder::bins)
      .def("save", &live::SoundSpeedLadder::save);
  m.def(
      "live_family_profile",
      [](const py::dict& options, double own_depth, double sound_speed, double bottom) {
        std::vector<double> depth, speed;
        live::family_profile(read_dict(options, "live_family_profile: ", profile_options), own_depth, sound_speed, bottom, depth, speed);
        return py::make_tuple(depth, speed);
      },
      "Knots (depths, speeds) of an assumed profile family through the sound speed at the vehicle (environment.hpp).");
  m.def(
      "live_tilted_profiles",
      [](double own_depth, const std::vector<double>& depth, const std::vector<double>& speed, const std::vector<double>& sigma,
         double gradient_sigma) {
        std::vector<double> knots, fast, slow;
        live::tilted_profiles(own_depth, depth, speed, sigma, gradient_sigma, knots, fast, slow);
        return py::make_tuple(knots, fast, slow);
      },
      "The two profiles a depth sigma is the half difference of: (knots, c + e, c - e) (environment.hpp).");
  m.def("live_profile_depth_sigma", &live::profile_depth_sigma, py::arg("receiver_depth"), py::arg("elevation"), py::arg("slant_range"),
        py::arg("depth"), py::arg("speed"), py::arg("sigma"), py::arg("gradient_sigma"), py::arg("water_depth") = 0.0, py::arg("arcs") = false,
        "One sigma of a depth traced through a profile with the given error at its knots (environment.hpp).");
  m.def(
      "live_water_profile",
      [](const py::object& profile, const py::object& ladder, const std::vector<double>& ssp_depth, const std::vector<double>& ssp_speed,
         bool use_gradient, double gradient, double gradient_sigma, double own_depth, double sound_speed, double water_depth) {
        live::ProfileOptions po;
        if (!profile.is_none()) po = read_dict(profile, "live_water_profile: profile.", profile_options);
        live::LadderProfile lp;
        if (!ladder.is_none()) {
          const py::dict l = ladder.cast<py::dict>();
          lp.rung = l["rung"].cast<std::string>();
          lp.depth = l["depth"].cast<std::vector<double>>();
          lp.speed = l["speed"].cast<std::vector<double>>();
          lp.sigma = l["sigma"].cast<std::vector<double>>();
          lp.bins = l["bins"].cast<std::int64_t>();
          lp.span_lo = l["span_lo"].cast<double>();
          lp.span_hi = l["span_hi"].cast<double>();
        }
        const live::MultipathProfile w = live::water_profile(po, ladder.is_none() ? nullptr : &lp, ssp_depth, ssp_speed, use_gradient, gradient,
                                                             gradient_sigma, own_depth, sound_speed, water_depth);
        py::dict d;
        d["depth"] = w.depth;
        d["speed"] = w.speed;
        d["sigma"] = w.sigma;
        d["gradient_sigma"] = w.gradient_sigma;
        d["rung"] = w.rung;
        d["ensemble"] = w.ensemble;
        return d;
      },
      py::arg("profile"), py::arg("ladder"), py::arg("ssp_depth"), py::arg("ssp_speed"), py::arg("use_gradient"), py::arg("gradient"),
      py::arg("gradient_sigma"), py::arg("own_depth"), py::arg("sound_speed"), py::arg("water_depth") = kNaN,
      "The water a ping's rays go through from what the chain holds (coverage.hpp water_profile): {depth, speed, sigma, gradient_sigma, rung, ensemble}.");
  m.def("live_refraction_sigma", &live::refraction_sigma);
  m.def("live_blank_interval", &live::blank_interval, py::arg("record"), py::arg("t0"), py::arg("t1"), py::arg("taper") = 0.0,
        "Zero every channel between record times t0 and t1 (s after the transmit), raised-cosine edges of `taper` s (sync.hpp).");
  m.def(
      "live_synchronise",
      [](PingRecord& record, const py::dict& reference) { return sync_dict(live::synchronise(record, reference_dict(reference, "live_synchronise: "))); },
      "The C++ board synchronisation on a raw record, in place (sync.hpp): the clock facts, with `flagged` per board.");
  m.def(
      "live_cluster",
      [](const py::array_t<bool, py::array::c_style | py::array::forcecast>& det,
         const py::array_t<float, py::array::c_style | py::array::forcecast>& power,
         const py::array_t<float, py::array::c_style | py::array::forcecast>& ratio, const std::vector<double>& ranges,
         const std::vector<double>& bearings) {
        if (det.ndim() != 2 || power.ndim() != 2 || ratio.ndim() != 2) throw std::invalid_argument("maps must be [n_bearing, n_range]");
        py::list out;
        for (const auto& d : live::cluster(reinterpret_cast<const unsigned char*>(det.data()), power.data(), ratio.data(),
                                           det.shape(0), det.shape(1), ranges, bearings)) {
          out.append(py::make_tuple(d.range, d.bearing, d.snr_db, d.cells));
        }
        return out;
      },
      "C++ clustering of a CFAR map: [(range, bearing, snr_db, cells), ...].");

  py::class_<live::Tracker>(m, "LiveTracker")
      .def(py::init([](double process_noise, double range_sigma, double bearing_sigma, double gate, std::int64_t confirm_hits,
                       std::int64_t max_misses, double initial_speed_sigma, const std::string& association) {
             live::TrackerOptions o{process_noise, range_sigma, bearing_sigma, gate, confirm_hits, max_misses, initial_speed_sigma};
             o.association = live::parse_association(association);
             return live::Tracker(o);
           }),
           py::arg("process_noise") = 0.05, py::arg("range_sigma") = 2.0, py::arg("bearing_sigma") = 0.017453292519943295,
           py::arg("gate") = 4.0, py::arg("confirm_hits") = 3, py::arg("max_misses") = 3, py::arg("initial_speed_sigma") = 10.0,
           py::arg("association") = "likelihood")
      .def("update",
           [](live::Tracker& t, double time, const std::vector<std::tuple<double, double, double, std::int64_t>>& dets,
              const std::optional<std::vector<std::optional<double>>>& times) {
             if (times && times->size() != dets.size()) throw py::value_error("LiveTracker.update: one time per detection");
             std::vector<live::Detection> d;
             for (std::size_t i = 0; i < dets.size(); ++i) {
               const auto& [r, b, s, c] = dets[i];
               d.push_back({r, b, s, c});
               // Each detection's own time (D-156); None: the ping's.
               if (times && (*times)[i]) d.back().time = *(*times)[i];
             }
             py::list out;
             for (const live::Track* tr : t.update(time, d)) out.append(track_dict(*tr));
             return out;
           },
           py::arg("time"), py::arg("detections"), py::arg("times") = py::none(),
           "Advance to the ping at `time` with detections [(range, bearing, snr_db, cells)]; `times`: each one's own time "
           "(s; None: the ping's).")
      .def("classify",
           [](const live::Tracker& t, double min_speed, double static_speed, double significance, double min_seen,
              std::int64_t min_points, double max_fit_rms, bool linear, double linear_length, double linear_width, bool multipath,
              double max_excess_range, double multipath_bearing, double multipath_velocity, double origin_x, double origin_y,
              std::optional<double> min_mover_cells, std::optional<double> row_snr_db, std::optional<double> time, bool fit_measured,
              double outlier_alpha, double max_fit_rms_bearing) {
             // None: the rule is off. Given: zero or more (as classify.py refuses a negative or NaN value).
             for (const auto& [name, val] : {std::pair{"min_mover_cells", min_mover_cells}, std::pair{"row_snr_db", row_snr_db}}) {
               if (val && !(*val >= 0.0)) throw py::value_error(std::string(name) + ": zero or more (None switches it off)");
             }
             if (!(outlier_alpha >= 0.0 && outlier_alpha < 1.0)) throw py::value_error("outlier_alpha: from 0 (off) to below 1");
             if (!(std::isfinite(max_fit_rms_bearing) && max_fit_rms_bearing >= 0.0)) {
               throw py::value_error("max_fit_rms_bearing: a finite angle, 0 (off) or more");
             }
             std::vector<const live::Track*> confirmed;
             // fit_measured false: copies of the tracks without their measured positions, which classify_tracks fits
             // through their filtered history (the rule before D-157; for the tests holding it to its golden).
             std::vector<live::Track> stripped;
             for (const auto& tr : t.tracks()) {
               if (!tr.confirmed) continue;
               if (fit_measured) {
                 confirmed.push_back(&tr);
                 continue;
               }
               stripped.push_back(tr);
               for (auto& p : stripped.back().history) p.mx = p.my = std::numeric_limits<double>::quiet_NaN();
             }
             for (const auto& tr : stripped) confirmed.push_back(&tr);
             const live::ClassifyOptions o{min_speed, static_speed, significance, min_seen, min_points, max_fit_rms, linear,
                                           linear_length, linear_width, multipath, max_excess_range, multipath_bearing,
                                           multipath_velocity, origin_x, origin_y,
                                           min_mover_cells.value_or(-1.0), row_snr_db.value_or(-1.0),
                                           time.value_or(std::numeric_limits<double>::quiet_NaN()), outlier_alpha,
                                           max_fit_rms_bearing};
             py::list out;
             for (const auto& c : live::classify_tracks(confirmed, o)) {
               py::dict d;
               d["id"] = c.id;
               d["kind"] = live::kind_name(c.kind);
               d["position"] = py::make_tuple(c.x, c.y);
               d["velocity"] = py::make_tuple(c.vx, c.vy);
               d["speed"] = c.speed;
               d["speed_sigma"] = c.speed_sigma;
               d["fit_rms"] = c.fit_rms;
               d["seen"] = c.seen;
               d["mean_snr_db"] = c.mean_snr_db;
               d["parent"] = c.parent < 0 ? py::object(py::none()) : py::object(py::int_(c.parent));
               d["extent_cells"] = c.extent_cells;
               out.append(d);
             }
             return out;
           },
           py::arg("min_speed") = 0.5, py::arg("static_speed") = 0.5, py::arg("significance") = 5.0, py::arg("min_seen") = 0.7,
           py::arg("min_points") = 4, py::arg("max_fit_rms") = -1.0, py::arg("linear") = false, py::arg("linear_length") = 40.0,
           py::arg("linear_width") = 2.0, py::arg("multipath") = false, py::arg("max_excess_range") = 800.0,
           py::arg("multipath_bearing") = 0.05235987755982988, py::arg("multipath_velocity") = 2.5, py::arg("origin_x") = 0.0,
           py::arg("origin_y") = 0.0, py::arg("min_mover_cells") = py::none(), py::arg("row_snr_db") = py::none(),
           py::arg("time") = py::none(), py::arg("fit_measured") = true, py::arg("outlier_alpha") = 0.0, py::arg("max_fit_rms_bearing") = 0.0,
           "classify_tracks on the confirmed tracks. fit_measured false: through their filtered history instead of the "
           "measured positions (the rule before D-157), for the tests that hold the classifier to its golden. "
           "outlier_alpha, max_fit_rms_bearing (rad): classify.py's (sonar-ljh, sonar-4jm; 0: off).");

  m.def("element_offsets", &live::element_offsets, "Receive element positions along the array axis, m about the centre.");
  m.def(
      "doa_gate",
      [](const PingRecord& record, double two_way_time, std::int64_t n_fft, const std::vector<double>& bearings,
         const std::string& method, int max_iterations, double tolerance, int max_sources, double dynamic_range_db,
         double min_snr_db, Backend backend, bool kernels) {
        const auto o = doa_options(max_iterations, tolerance, max_sources, dynamic_range_db, min_snr_db, backend, kernels);
        const auto u = live::element_offsets(record);
        live::DoaResult r;
        {
          py::gil_scoped_release release;
          const live::DoaFrame frame = live::gate_frame(record, two_way_time, n_fft);
          r = method == "sbl" ? live::sbl(frame, u, record.environment.sound_speed, bearings, o)
                              : live::conventional(frame, u, record.environment.sound_speed, bearings, o);
        }
        py::dict d = doa_dict(r);
        // Where the spectrum was computed. Only SBL has a device path: the conventional
        // beamformer is a few small matrix products per gate and runs on the host whatever
        // backend is asked for.
        d["backend"] = method == "sbl" && backend == Backend::Cuda ? "cuda" : "host";
        return d;
      },
      py::arg("record"), py::arg("two_way_time"), py::arg("n_fft"), py::arg("bearings"), py::arg("method") = "sbl",
      py::arg("max_iterations") = 60, py::arg("tolerance") = 1e-3, py::arg("max_sources") = 4,
      py::arg("dynamic_range_db") = 20.0, py::arg("min_snr_db") = 0.0, py::arg("backend") = Backend::Host,
      py::arg("kernels") = true,
      "Bearings at one range gate of a conditioned record: SBL or the conventional beamformer on the same snapshots.\n"
      "`backend` chooses where SBL iterates. method=\"conventional\" has no device path and runs on the host\n"
      "whatever the backend; the result's \"backend\" entry (\"host\" or \"cuda\") says where the spectrum was computed.");
  m.attr("DOA_GATE_CUDA_METHODS") = py::make_tuple("sbl");
  m.def(
      "doa_passive",
      [](const PingRecord& record, double f_low, double f_high, std::int64_t n_fft, std::int64_t hop, std::int64_t n_average,
         double first_deg, double last_deg, double step_deg, const std::string& method, int max_iterations, double tolerance,
         int max_sources, double dynamic_range_db, double min_snr_db, Backend backend, bool kernels, bool elevation,
         double elevation_first_deg, double elevation_last_deg, double elevation_step_deg, const std::string& look) {
        live::PassiveOptions p;
        p.elevation.enabled = elevation;
        p.elevation.first_deg = elevation_first_deg;
        p.elevation.last_deg = elevation_last_deg;
        p.elevation.step_deg = elevation_step_deg;
        p.starboard = look != "port";
        p.f_low = f_low;
        p.f_high = f_high;
        p.n_fft = n_fft;
        p.hop = hop;
        p.n_average = n_average;
        p.bearing_first_deg = first_deg;
        p.bearing_last_deg = last_deg;
        p.bearing_step_deg = step_deg;
        p.use_sbl = method == "sbl";
        p.sbl = doa_options(max_iterations, tolerance, max_sources, dynamic_range_db, min_snr_db, backend, kernels);
        std::vector<live::PassiveFrame> frames;
        std::vector<double> bearings;
        {
          py::gil_scoped_release release;
          frames = live::passive_bearings(record, p, &bearings);
        }
        py::list out;
        for (const auto& f : frames) {
          py::dict d;
          d["time"] = f.time;
          d["noise"] = f.noise;
          d["spectrum"] = py::array_t<float>(static_cast<py::ssize_t>(f.spectrum.size()), f.spectrum.data());
          py::list peaks;
          for (const auto& pk : f.peaks) peaks.append(py::make_tuple(pk.bearing, pk.power, pk.snr_db));
          d["peaks"] = peaks;
          if (elevation) {
            d["elevations"] = f.elevations;
            d["elevation_quality"] = f.elevation_quality;
          }
          if (f.rows_as_snapshots > 0) d["rows_as_snapshots"] = f.rows_as_snapshots;
          if (f.sbl_fallback) d["sbl_fallback"] = true;
          out.append(d);
        }
        return py::make_tuple(out, bearings);
      },
      py::arg("record"), py::arg("f_low") = 0.0, py::arg("f_high") = 0.0, py::arg("n_fft") = 0, py::arg("hop") = 0,
      py::arg("n_average") = 8, py::arg("first_deg") = -80.0, py::arg("last_deg") = 80.0, py::arg("step_deg") = 0.5,
      py::arg("method") = "sbl", py::arg("max_iterations") = 60, py::arg("tolerance") = 1e-3, py::arg("max_sources") = 4,
      py::arg("dynamic_range_db") = 20.0, py::arg("min_snr_db") = 0.0, py::arg("backend") = Backend::Host,
      py::arg("kernels") = true, py::arg("elevation") = false, py::arg("elevation_first_deg") = -30.0,
      py::arg("elevation_last_deg") = 30.0, py::arg("elevation_step_deg") = 0.5, py::arg("look") = "starboard",
      "Passive bearing-time record of one stretch of data: (frames, bearing grid in rad). With `elevation` and an array "
      "of stacked rows every frame also has `elevations` (rad, level-referenced, up positive) and `elevation_quality`. "
      "n_fft 0: derived from the array and the sample rate (live_passive_sizes); hop 0: half of n_fft. Stacked rows "
      "without `elevation`: bearings from the rows as snapshots, the frame says `rows_as_snapshots`.");
  m.def(
      "live_passive_sizes",
      [](const PingRecord& record, std::int64_t n_fft, std::int64_t hop, std::int64_t n_average) {
        live::PassiveOptions p;
        p.n_fft = n_fft;
        p.hop = hop;
        p.n_average = n_average;
        live::resolve(p, record);
        return py::make_tuple(p.n_fft, p.hop, p.n_fft + (p.n_average - 1) * p.hop);
      },
      py::arg("record"), py::arg("n_fft") = 0, py::arg("hop") = 0, py::arg("n_average") = 8,
      "(n_fft, hop, samples per frame) of the passive chain for this record, with sizes given as 0 derived (derive.hpp).");
  m.attr("SBL_KERNEL_MAX_CHANNELS") = live::kSblMaxChannels;
  m.def(
      "live_derive",
      [](const PingRecord& record, const py::dict& given_dict) {
        // The chain's derived settings for this record: `given` holds the ones the configuration set (the C++ option names).
        const Keys given(given_dict);
        live::ChainOptions o;
        number(given, "range_cell_m", o.range_cell_m);
        number(given, "cfar_guard_m", o.cfar_guard_m);
        number(given, "cfar_train_m", o.cfar_train_m);
        number(given, "tracker_range_sigma", o.tracker.range_sigma);
        number(given, "tracker_bearing_sigma", o.tracker.bearing_sigma);
        number(given, "multipath_bearing", o.classify.multipath_bearing);
        number(given, "classify_max_fit_rms", o.classify.max_fit_rms);
        number(given, "classify_max_fit_rms_bearing", o.classify.max_fit_rms_bearing);
        number(given, "row_tolerance_m", o.elevation.row_tolerance_m);
        number(given, "bearing_step_deg", o.bearing_step_deg);
        if (given.contains("min_row_elements") && !given["min_row_elements"].is_none()) {
          o.elevation.min_row_elements = given["min_row_elements"].cast<std::int64_t>();
        }
        given.refuse_unread("live_derive: ");
        const live::ChainDefaults d = live::resolve(o, record);
        py::dict out;
        out["range_cell_m"] = d.range_cell_m;
        out["cfar_guard_m"] = d.cfar_guard_m;
        out["cfar_train_m"] = d.cfar_train_m;
        out["tracker_range_sigma"] = d.tracker_range_sigma;
        out["tracker_bearing_sigma"] = d.tracker_bearing_sigma;
        out["multipath_bearing"] = d.multipath_bearing;
        out["classify_max_fit_rms"] = d.classify_max_fit_rms;
        out["classify_max_fit_rms_bearing"] = d.classify_max_fit_rms_bearing;
        out["row_tolerance_m"] = d.row_tolerance_m;
        out["json"] = d.json();
        return out;
      },
      py::arg("record"), py::arg("given") = py::dict(),
      "The chain settings left to be derived, resolved from this record's band, pulse and array (libs/live derive.hpp): "
      "values in the units of the C++ options (angles in rad), and `json`, the line the programs print.");
  m.def(
      "live_detector_plan",
      [](const PingRecord& record, std::int64_t n_cells, std::int64_t samples_per_cell, double step, const std::vector<double>& bearings,
         std::int64_t rows, std::int64_t guard_cells, std::int64_t train_cells, bool hann, const py::dict& options) {
        const Keys kw(options);
        live::DetectorOptions o;
        number(kw, "false_alarms_per_ping", o.false_alarms_per_ping);
        number(kw, "sidelobe_margin_db", o.sidelobe_margin_db);
        number(kw, "sidelobe_floor_db", o.sidelobe_floor_db);
        number(kw, "search_factor", o.search_factor);
        if (kw.contains("range_estimate") && !kw["range_estimate"].is_none()) o.range_estimate = live::parse_range_estimate(kw["range_estimate"].cast<std::string>());
        if (kw.contains("integrate_cells") && !kw["integrate_cells"].is_none()) o.integrate_cells = kw["integrate_cells"].cast<std::vector<std::int64_t>>();
        double tolerance = live::kDerive;
        number(kw, "row_tolerance_m", tolerance);
        std::int64_t min_elements = 3;
        if (kw.contains("min_row_elements") && !kw["min_row_elements"].is_none()) min_elements = kw["min_row_elements"].cast<std::int64_t>();
        const live::SonarFacts facts = live::sonar_facts(record, live::row_tolerance(record, tolerance), min_elements);
        double merge_m = 0.0;
        number(kw, "merge_m", merge_m);
        kw.refuse_unread("live_detector_plan: ");
        // The lines whose maps are added: the array's rows when the chain stacks them (rows > 1), else every channel as one.
        std::vector<std::vector<std::int64_t>> lines;
        if (rows > 1) lines = live::find_rows(record, live::row_tolerance(record, tolerance), min_elements);
        const live::DetectorPlan p = live::plan_detector(record, facts, o, n_cells, samples_per_cell, step, bearings, rows, guard_cells, train_cells, hann, merge_m, lines);
        py::dict out;
        py::list scales;
        for (const live::DetectorScale& s : p.scales) {
          py::dict d;
          d["cells"] = s.cells;
          d["length_m"] = s.length_m;
          d["looks"] = s.looks;
          d["tests"] = s.tests;
          d["search"] = s.search;
          d["pfa"] = s.pfa;
          d["factor"] = s.factor;
          d["weights"] = s.weights;
          scales.append(d);
        }
        out["scales"] = scales;
        out["rows"] = p.rows;
        out["samples_per_cell"] = p.samples_per_cell;
        out["cell_m"] = p.cell_m;
        out["step_m"] = p.step_m;
        out["resolution_m"] = p.resolution_m;
        out["beams"] = p.beams;
        out["train_looks_per_cell"] = p.train_looks_per_cell;
        out["margin"] = p.margin;
        out["aperture_m"] = p.aperture_m;
        out["merge_cells"] = p.merge_cells;
        out["merge_u"] = p.merge_u;
        out["range_envelope"] = py::array_t<double>(static_cast<py::ssize_t>(p.range_envelope.size()), p.range_envelope.data());
        out["bearing_envelope"] = py::array_t<double>(static_cast<py::ssize_t>(p.bearing_envelope.size()), p.bearing_envelope.data());
        out["u_step"] = p.u_step;
        out["source_ratio"] = p.source_ratio;
        out["range_estimate"] = live::range_estimate_name(p.range_estimate);
        out["json"] = p.json();
        return out;
      },
      py::arg("record"), py::arg("n_cells"), py::arg("samples_per_cell"), py::arg("step"), py::arg("bearings"), py::arg("rows") = 1,
      py::arg("guard_cells") = 50, py::arg("train_cells") = 150, py::arg("hann") = true, py::arg("options") = py::dict(),
      "The multiscale detector's plan (libs/live detector.hpp) for a map of n_cells range cells of samples_per_cell samples `step` m apart: "
      "integration lengths, their looks, tests, false-alarm probabilities and thresholds, and the sidelobe envelopes. The C++ chain uses the same function.");
  m.def(
      "live_detector_background",
      [](const py::array_t<float, py::array::c_style | py::array::forcecast>& cell_mean, std::int64_t guard, std::int64_t train) {
        if (cell_mean.ndim() != 2) throw std::invalid_argument("cell_mean must be [n_bearing, n_cells]");
        std::vector<double> level, worth;
        live::detector_background(cell_mean.data(), cell_mean.shape(0), cell_mean.shape(1), guard, train, level, worth);
        py::array_t<double> out({cell_mean.shape(0), cell_mean.shape(1)});
        std::memcpy(out.mutable_data(), level.data(), level.size() * sizeof(double));
        return py::make_tuple(out, py::array_t<double>(static_cast<py::ssize_t>(worth.size()), worth.data()));
      },
      py::arg("cell_mean"), py::arg("guard"), py::arg("train"),
      "The multiscale detector's background of each cell (libs/live detector.hpp) and, per range cell, the number of independent "
      "training cells it is worth: the C++ reference the NumPy one (detect.background) is tested against.");
  m.def(
      "live_multiscale_detect",
      [](const py::array_t<float, py::array::c_style | py::array::forcecast>& fine, const py::array_t<float, py::array::c_style | py::array::forcecast>& cell_mean,
         const std::vector<double>& fine_ranges, const std::vector<double>& cell_ranges, const std::vector<double>& bearings, std::int64_t guard,
         std::int64_t train, const py::dict& plan, Backend backend, std::int64_t fine_offset, int device) {
        // The plan as live_detector_plan gives it, back into the C++ struct.
        if (fine.ndim() != 2 || cell_mean.ndim() != 2 || fine.shape(0) != cell_mean.shape(0)) {
          throw std::invalid_argument("fine [n_bearing, samples] and cell_mean [n_bearing, n_cells]");
        }
        if (guard < 0) throw std::invalid_argument("live_multiscale_detect: guard must be >= 0 cells");
        if (train < 1) throw std::invalid_argument("live_multiscale_detect: train must be >= 1 cell");
        live::DetectorPlan p;
        for (const py::handle h : plan["scales"].cast<py::list>()) {
          const py::dict d = h.cast<py::dict>();
          live::DetectorScale c;
          c.cells = d["cells"].cast<std::int64_t>();
          c.length_m = d["length_m"].cast<double>();
          c.looks = d["looks"].cast<double>();
          c.tests = d["tests"].cast<double>();
          c.search = d["search"].cast<double>();
          c.pfa = d["pfa"].cast<double>();
          c.factor = d["factor"].cast<double>();
          c.weights = d["weights"].cast<std::vector<double>>();
          p.scales.push_back(c);
        }
        p.rows = plan["rows"].cast<std::int64_t>();
        p.samples_per_cell = plan["samples_per_cell"].cast<std::int64_t>();
        p.cell_m = plan["cell_m"].cast<double>();
        p.step_m = plan["step_m"].cast<double>();
        p.resolution_m = plan["resolution_m"].cast<double>();
        p.beams = plan["beams"].cast<double>();
        p.train_looks_per_cell = plan["train_looks_per_cell"].cast<double>();
        p.margin = plan["margin"].cast<double>();
        p.aperture_m = plan["aperture_m"].cast<double>();
        p.merge_cells = plan["merge_cells"].cast<std::int64_t>();
        p.merge_u = plan["merge_u"].cast<double>();
        p.range_envelope = plan["range_envelope"].cast<std::vector<double>>();
        p.bearing_envelope = plan["bearing_envelope"].cast<std::vector<double>>();
        p.u_step = plan["u_step"].cast<double>();
        p.source_ratio = plan["source_ratio"].cast<double>();
        if (plan.contains("range_estimate")) p.range_estimate = live::parse_range_estimate(plan["range_estimate"].cast<std::string>());
        const std::int64_t n_b = cell_mean.shape(0), n_cells = cell_mean.shape(1), stride = fine.shape(1);
        if (fine_offset < 0 || fine_offset + n_cells * p.samples_per_cell > stride) throw std::invalid_argument("fine holds fewer samples than the cells need");
        live::DetectorCells cells;
        live::multiscale_cells(backend, fine.data(), stride, fine_offset, cell_mean.data(), n_b, n_cells, guard, train, p, cells, device);
        py::dict out;
        const auto grid = [&](const auto& v) {
          using T = typename std::decay_t<decltype(v)>::value_type;
          py::array_t<T> a({n_b, n_cells});
          std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
          return a;
        };
        out["level"] = grid(cells.level);
        out["background"] = grid(cells.background);
        out["peak"] = grid(cells.peak);
        out["flag"] = grid(cells.flag);
        std::vector<float> background;
        const std::vector<live::Detection> found =
            live::multiscale_group(fine.data(), stride, fine_offset, n_b, n_cells, fine_ranges, cell_ranges, bearings, p, cells, background);
        py::list dets;
        for (const live::Detection& d : found) {
          py::dict e;
          e["range_m"] = d.range;
          e["bearing"] = d.bearing;
          e["snr_db"] = d.snr_db;
          e["cells"] = d.cells;
          e["first_range_m"] = d.first_range;
          e["extent_m"] = d.extent;
          e["centroid_range_m"] = d.centroid_range;   // the energy-weighted mean range, whichever range_m reports (D-158)
          py::list arr;
          for (const live::Arrival& a : d.arrivals) arr.append(py::make_tuple(a.range, a.snr_db));
          e["arrivals"] = arr;
          dets.append(e);
        }
        out["detections"] = dets;
        return out;
      },
      py::arg("fine"), py::arg("cell_mean"), py::arg("fine_ranges"), py::arg("cell_ranges"), py::arg("bearings"), py::arg("guard"), py::arg("train"),
      py::arg("plan"), py::arg("backend") = Backend::Host, py::arg("fine_offset") = 0, py::arg("device") = 0,
      "The C++ chain's multiscale detector (libs/live detector.hpp) on a map: its statistic on `backend` (multiscale_cells: `level`, "
      "`background`, `peak`, `flag` [n_bearing, n_cells], flags before sidelobes are dropped) and its `detections`. `plan`: live_detector_plan's. "
      "For holding the device's statistic to the host's (tests/python/test_detector_device.py).");
  m.def(
      "live_beam_samples",
      [](const PingRecord& record, const std::vector<double>& ranges, double bearing, bool hann, Backend backend) {
        const std::vector<std::complex<float>> z = live::beam_samples(record, ranges, bearing, hann, backend);
        py::array_t<std::complex<float>> out(static_cast<py::ssize_t>(z.size()));
        std::memcpy(out.mutable_data(), z.data(), z.size() * sizeof(std::complex<float>));
        return out;
      },
      py::arg("record"), py::arg("ranges"), py::arg("bearing"), py::arg("hann") = true, py::arg("backend") = Backend::Host,
      "The compressed, beamformed complex samples of a prepared record at `ranges` (m) along one bearing (rad): what the chain's map is the "
      "squared magnitude of (libs/live chain.hpp, beam_samples). For the arrival structure inside a detection: its first_range_m, extent_m, bearing.");
  m.def("live_sample_correlation", [](const PingRecord& record, double step, std::int64_t n) {
          return live::sample_correlation(record.transmit.waveform, record.environment.sound_speed, step, n);
        }, py::arg("record"), py::arg("step"), py::arg("n"),
        "Amplitude correlation of the compressed pulse between map samples `step` m apart, lags 0..n-1.");
  m.def("live_row_tolerance", &live::row_tolerance, py::arg("record"), py::arg("given") = live::kDerive,
        "Elements within this of one line are one row: `given`, or 2 cm, or a quarter of the shortest wavelength when less.");
  m.def("live_sync_tolerance", &live::sync_tolerance, py::arg("given_us"), py::arg("top_frequency"),
        "What two boards must agree within, s: given_us microseconds if positive, else a twentieth of a period.");
  m.def("live_stream_buffer_bytes", &live::stream_buffer_bytes, py::arg("template_record"), py::arg("sample_bytes") = 3, py::arg("given") = 0,
        "The receive buffer to ask for: `given` if positive, else the larger of 8 MB and half a second of the board stream.");
  m.def("live_stream_bytes_per_second", &live::stream_bytes_per_second, py::arg("template_record"), py::arg("sample_bytes") = 3);

  m.def("live_find_rows", &live::find_rows, py::arg("record"), py::arg("tolerance") = 0.02, py::arg("min_elements") = 3,
        "Channel rows of each line of the receive array (elevation.hpp).");
  m.def(
      "live_passive_elevation",
      [](const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, double bearing, double f_low, double f_high,
         std::int64_t n_fft, std::int64_t hop, std::int64_t first, std::int64_t n_blocks, double first_deg, double last_deg,
         double step_deg, const std::string& look) {
        const live::PassiveElevation e = live::passive_elevation(record, rows, bearing, f_low, f_high, n_fft, hop, first, n_blocks,
                                                                 first_deg, last_deg, step_deg, look != "port");
        return py::make_tuple(e.elevation, e.quality);
      },
      py::arg("record"), py::arg("rows"), py::arg("bearing"), py::arg("f_low"), py::arg("f_high"), py::arg("n_fft") = 1024,
      py::arg("hop") = 512, py::arg("first") = 0, py::arg("n_blocks") = -1, py::arg("first_deg") = -30.0, py::arg("last_deg") = 30.0,
      py::arg("step_deg") = 0.5, py::arg("look") = "starboard",
      "Elevation (rad, array frame, up positive) and quality of a noise source at a cone bearing, from stacked rows.");
  m.def("live_ray_depth", &live::ray_depth, py::arg("receiver_depth"), py::arg("elevation"), py::arg("slant_range"),
        py::arg("ssp_depth"), py::arg("ssp_speed"), py::arg("water_depth") = 0.0,
        "Depth an arrival started at, back along the ray through a sound-speed profile (elevation.hpp).");
  m.def(
      "live_fold_depth",
      [](double z, double water_depth, double surface_z) {
        const auto f = live::fold_depth(z, water_depth, surface_z);
        return py::make_tuple(f.first, f.second);
      },
      py::arg("z"), py::arg("water_depth"), py::arg("surface_z") = 0.0);

  m.def(
      "live_options_from_text",
      [](const std::string& text) {
        live::ChainOptions c;
        live::PassiveOptions p;
        live::program_defaults(c, p);
        std::istringstream in(text);
        live::read_options(c, p, in);
        return live::options_json(c);
      },
      py::arg("text"),
      "key=value lines (one per line, # comments) read as build/bin/sonar_live reads --options (options.hpp), and the active "
      "chain they give as LiveChainCpp.options() writes it.");

  py::class_<live::LiveChain>(m, "LiveChainCpp",
                              "The C++ live chain (libs/live chain.hpp), built from one dict of options: the flat keys\n"
                              "gpusonar.live.cpp_chain_options makes from a deployment file's chain section. Units are SI as\n"
                              "the C++ fields hold them, not as the deployment file or sonar_live's key=value names give them.\n"
                              "In radians although the name has no unit suffix (the deployment file and sonar_live take degrees\n"
                              "under a _deg name):\n"
                              "  tracker_bearing_sigma (rad; file tracker.bearing_sigma_deg),\n"
                              "  multipath_bearing (rad; file classify.multipath.bearing_deg),\n"
                              "  classify_max_fit_rms_bearing (rad; file classify.max_fit_rms_bearing_deg),\n"
                              "  elevation.sector_margin (rad; file elevation.sector_from_depth.margin_deg),\n"
                              "  elevation.sector_rays_margin (rad; file elevation.sector_from_rays.margin_deg),\n"
                              "  elevation.roll_sigma (rad; file elevation.roll_sigma_deg),\n"
                              "  elevation.attitude_rate_sigma (rad/s; file elevation.attitude_rate_sigma_deg_s).\n"
                              "Also radians: the grazing angles of the surface_loss_grazing lists in `coverage` and\n"
                              "`multipath_depth`. Keys ending in _deg (bearing_*_deg, sbl_grid_step_deg, elevation first_deg,\n"
                              "last_deg, step_deg) are degrees.\n"
                              "A key it does not read, at the top or in a section (elevation, elevation.profile, decimate,\n"
                              "blanking, sync, multipath_depth, coverage), is refused: ValueError naming it and the nearest\n"
                              "key it does read (sonar-6hf).")
      .def(py::init([](const py::dict& given) {
        const Keys kw(given);   // every key looked up is recorded; any other is refused below (sonar-6hf)
        live::ChainOptions o;
        auto get = [&](const char* k, double& v) {
          if (kw.contains(k)) v = kw[k].cast<double>();
        };
        get("bearing_first_deg", o.bearing_first_deg);
        get("bearing_last_deg", o.bearing_last_deg);
        get("bearing_step_deg", o.bearing_step_deg);
        get("range_cell_m", o.range_cell_m);
        get("cfar_pfa", o.cfar_pfa);
        get("cfar_guard_m", o.cfar_guard_m);
        get("cfar_train_m", o.cfar_train_m);
        get("min_snr_db", o.min_snr_db);
        get("sbl_grid_step_deg", o.sbl_grid_step_deg);
        get("sbl_min_snr_db", o.sbl.min_snr_db);
        get("sbl_min_contrast_db", o.sbl_min_contrast_db);
        get("sbl_tolerance", o.sbl.tolerance);
        if (kw.contains("sbl_max_iterations")) o.sbl.max_iterations = kw["sbl_max_iterations"].cast<std::int32_t>();
        if (kw.contains("sbl_max_gates")) o.sbl_max_gates = kw["sbl_max_gates"].cast<std::int64_t>();
        get("sbl_dynamic_range_db", o.sbl.dynamic_range_db);
        get("tracker_range_sigma", o.tracker.range_sigma);
        get("tracker_bearing_sigma", o.tracker.bearing_sigma);
        get("tracker_process_noise", o.tracker.process_noise);
        get("tracker_initial_speed_sigma", o.tracker.initial_speed_sigma);
        get("classify_min_speed", o.classify.min_speed);
        get("classify_static_speed", o.classify.static_speed);
        get("classify_max_fit_rms", o.classify.max_fit_rms);
        get("classify_outlier_alpha", o.classify.outlier_alpha);
        if (kw.contains("classify_max_fit_rms_bearing")) {     // left out: derived; given, a finite angle, 0 (off) or more
          const double b = kw["classify_max_fit_rms_bearing"].cast<double>();
          if (!(std::isfinite(b) && b >= 0.0)) throw py::value_error("classify_max_fit_rms_bearing: a finite angle, 0 (off) or more (leave the key out to derive it)");
          o.classify.max_fit_rms_bearing = b;
        }
        if (kw.contains("classify_min_mover_cells")) {        // off unless given; given, zero or more cells
          const double m = kw["classify_min_mover_cells"].cast<double>();
          if (!(m >= 0.0)) throw py::value_error("classify_min_mover_cells: zero or more cells (leave the key out for off)");
          o.classify.min_mover_cells = m;
        }
        if (kw.contains("classify_row_snr_db")) throw py::value_error("classify_row_snr_db: the live chain never runs the row (corridor) test, so this rule would do nothing there; it applies only to classify_tracks with linear (the gallery)");
        get("multipath_max_excess_range", o.classify.multipath_max_excess_range);
        get("multipath_bearing", o.classify.multipath_bearing);
        get("multipath_velocity", o.classify.multipath_velocity);
        if (kw.contains("tracker_confirm_hits")) o.tracker.confirm_hits = kw["tracker_confirm_hits"].cast<std::int64_t>();
        if (kw.contains("tracker_association")) o.tracker.association = live::parse_association(kw["tracker_association"].cast<std::string>());
        if (kw.contains("tracker_max_misses")) o.tracker.max_misses = kw["tracker_max_misses"].cast<std::int64_t>();
        if (kw.contains("sbl_max_sources")) o.sbl.max_sources = kw["sbl_max_sources"].cast<std::int32_t>();
        if (kw.contains("gate_fft")) o.gate_fft = kw["gate_fft"].cast<std::int64_t>();
        if (kw.contains("multipath")) o.classify.multipath = kw["multipath"].cast<bool>();
        if (kw.contains("sbl_kernels")) o.sbl.kernels = kw["sbl_kernels"].cast<bool>();
        if (kw.contains("hann")) o.hann = kw["hann"].cast<bool>();
        if (kw.contains("starboard")) o.starboard = kw["starboard"].cast<bool>();
        if (kw.contains("doa")) o.doa = kw["doa"].cast<std::string>();
        if (kw.contains("cell_peak")) o.cell_peak = kw["cell_peak"].cast<bool>();
        if (kw.contains("cell_multiscale")) o.cell_multiscale = kw["cell_multiscale"].cast<bool>();
        get("cfar_false_alarms_per_ping", o.detector.false_alarms_per_ping);
        get("cfar_sidelobe_margin_db", o.detector.sidelobe_margin_db);
        get("cfar_sidelobe_floor_db", o.detector.sidelobe_floor_db);
        get("cfar_search_factor", o.detector.search_factor);
        get("cfar_merge_m", o.detector.merge_m);
        if (kw.contains("cfar_range_estimate") && !kw["cfar_range_estimate"].is_none()) o.range_estimate = live::parse_range_estimate(kw["cfar_range_estimate"].cast<std::string>());
        if (kw.contains("cfar_integrate_cells")) o.detector.integrate_cells = kw["cfar_integrate_cells"].cast<std::vector<std::int64_t>>();
        if (kw.contains("beamformer")) o.beamformer = kw["beamformer"].cast<std::string>();
        if (kw.contains("backend")) {
          // core.Backend, or the word of the key=value option: auto | host | cuda (options.hpp parse_backend).
          o.backend = py::isinstance<py::str>(kw["backend"]) ? live::parse_backend(kw["backend"].cast<std::string>()) : kw["backend"].cast<Backend>();
        }
        get("sound_speed", o.sound_speed);
        if (kw.contains("dead_channels")) o.dead_channels = kw["dead_channels"].cast<std::vector<std::int64_t>>();
        if (kw.contains("elevation")) {
          const Keys e(kw["elevation"].cast<py::dict>());
          live::ElevationOptions& eo = o.elevation;
          eo.enabled = true;
          auto num = [&](const char* k, double& v) {
            if (e.contains(k)) v = e[k].cast<double>();
          };
          auto flag = [&](const char* k, bool& v) {
            if (e.contains(k)) v = e[k].cast<bool>();
          };
          num("first_deg", eo.first_deg);
          num("last_deg", eo.last_deg);
          num("step_deg", eo.step_deg);
          num("range_window_m", eo.range_window_m);
          if (e.contains("max_detections")) eo.max_detections = e["max_detections"].cast<std::int64_t>();
          flag("first_arrival", eo.first_arrival);
          num("first_arrival_db", eo.first_arrival_db);
          num("first_arrival_snr_db", eo.first_arrival_snr_db);
          num("min_quality", eo.min_quality);
          num("ambiguity_sigma", eo.ambiguity_sigma);
          num("alternatives_db", eo.alternatives_db);
          flag("sector_from_depth", eo.sector_from_depth);
          num("sector_bounces", eo.sector_bounces);
          num("sector_margin", eo.sector_margin);
          num("water_depth_sigma", eo.water_depth_sigma);
          flag("attitude_arrival", eo.attitude_arrival);
          num("attitude_rate_sigma", eo.attitude_rate_sigma);
          flag("use_gradient", eo.use_gradient);
          flag("ray_arcs", eo.ray_arcs);
          flag("sector_from_rays", eo.sector_from_rays);
          num("sector_rays_bounces", eo.sector_rays_bounces);
          num("sector_rays_margin", eo.sector_rays_margin);
          num("sector_rays_max_depth", eo.sector_rays_max_depth);
          flag("multipath_sigma", eo.multipath_sigma);
          num("multipath_sigma_bounces", eo.multipath_sigma_bounces);
          num("multipath_sigma_max_depth", eo.multipath_sigma_max_depth);
          num("gradient", eo.gradient);
          num("gradient_sigma", eo.gradient_sigma);
          num("ambiguity_db", eo.ambiguity_db);
          flag("correct_bearing", eo.correct_bearing);
          flag("depth", eo.depth);
          flag("fold", eo.fold);
          flag("classify", eo.classify);
          num("classify_sigma", eo.classify_sigma);
          num("water_depth", eo.water_depth);
          num("surface_z", eo.surface_z);
          num("roll_sigma", eo.roll_sigma);
          num("depth_process_noise", o.tracker.depth_process_noise);
          num("row_tolerance_m", eo.row_tolerance_m);
          if (e.contains("min_row_elements") && !e["min_row_elements"].is_none()) eo.min_row_elements = e["min_row_elements"].cast<std::int64_t>();
          if (e.contains("ssp_depth")) eo.ssp_depth = e["ssp_depth"].cast<std::vector<double>>();
          if (e.contains("ssp_speed")) eo.ssp_speed = e["ssp_speed"].cast<std::vector<double>>();
          if (eo.ssp_depth.size() != eo.ssp_speed.size()) throw std::invalid_argument("elevation.ssp: depth and speed must be as long as each other");
          if (e.contains("profile") && !e["profile"].is_none()) {
            const Keys p(e["profile"].cast<py::dict>());
            eo.profile = profile_options(p);
            p.refuse_unread("LiveChainCpp: elevation.profile.");
          }
          e.refuse_unread("LiveChainCpp: elevation.");
        }
        if (kw.contains("decimate") && !kw["decimate"].is_none()) {
          const Keys d(kw["decimate"].cast<py::dict>());
          o.decimate = decimate_options(d);
          d.refuse_unread("LiveChainCpp: decimate.");
        }
        if (kw.contains("blanking") && !kw["blanking"].is_none()) {
          const Keys b(kw["blanking"].cast<py::dict>());
          o.blanking.enabled = true;
          number(b, "recovery_s", o.blanking.recovery_s);
          number(b, "taper_s", o.blanking.taper_s);
          if (b.contains("mode")) o.blanking.zero = b["mode"].cast<std::string>() != "ignore";
          b.refuse_unread("LiveChainCpp: blanking.");
        }
        if (kw.contains("multipath_depth") && !kw["multipath_depth"].is_none()) {   // hook: flat {member: number}, as key=value
          o.multipath_depth.enabled = true;
          for (const auto& item : kw["multipath_depth"].cast<py::dict>()) {
            const std::string key = "multipath_depth_" + item.first.cast<std::string>();
            if (!live::set_multipath_option(o.multipath_depth, key, py::str(item.second).cast<std::string>())) throw std::invalid_argument("unknown option " + key);
          }
        }
        if (kw.contains("coverage") && !kw["coverage"].is_none()) {   // hook: flat {member: value}, as key=value
          o.coverage.enabled = true;
          for (const auto& item : kw["coverage"].cast<py::dict>()) {
            const std::string key = "coverage_" + item.first.cast<std::string>();
            if (!live::set_coverage_option(o.coverage, key, py::str(item.second).cast<std::string>())) throw std::invalid_argument("unknown option " + key);
          }
        }
        if (kw.contains("sync")) {
          const Keys r(kw["sync"].cast<py::dict>());
          o.sync = true;
          o.reference = reference_options(r);
          r.refuse_unread("LiveChainCpp: sync.");
        }
        kw.refuse_unread("LiveChainCpp: ");
        return live::LiveChain(o);
      }))
      .def("process",
           [](live::LiveChain& c, PingRecord& record) {
             std::string json;
             {
               py::gil_scoped_release release;
               json = c.process(record).json();
             }
             return json;
           },
           "Process one ping (conditioned in place); returns the report as a JSON line.")
      .def("set_coverage_pd", [](live::LiveChain& c, const std::vector<double>& snr_db, const std::vector<double>& pd) { c.coverage().set_pd_table(snr_db, pd); },
           "The detection probability against one row's link-budget SNR (gpusonar.live.detection_probability) for the coverage picture.")
      .def("coverage_seen", [](live::LiveChain& c, std::size_t i, double range) { return c.coverage().seen(i, range); },
           "Detection probability of target class i at a horizontal range at the present depth (coverage.hpp).")
      .def("coverage_seconds", [](live::LiveChain& c) { return c.coverage().seconds(); })
      .def("detector", &live::LiveChain::detector_json,
           "The multiscale detector's plan for the last map (detector.hpp), as a JSON line; '{}' when it is not in use.")
      .def("options", [](const live::LiveChain& c) { return live::options_json(c.options()); },
           "The chain's settings as a JSON line keyed by the key=value option names (options.hpp options_json); "
           "before the first ping the sizes left to be derived are null.")
      .def("resolved", &live::LiveChain::resolved_json,
           "The settings the first ping resolved (derive.hpp), as a JSON line; '{}' before it.")
      .def("last_map", [](const live::LiveChain& c) {
        py::array_t<float> power({static_cast<py::ssize_t>(c.bearings.size()), static_cast<py::ssize_t>(c.last_ranges.size())});
        std::copy(c.last_power.begin(), c.last_power.end(), power.mutable_data());
        return py::make_tuple(power, c.last_ranges, c.bearings);
      });

  py::class_<live::PingAssembler>(m, "PingAssemblerCpp")
      .def(py::init([](const PingRecord& t, const py::object& decimate, const py::object& reference, int sample_bytes, int reorder_blocks,
                            std::int64_t max_held_bytes) {
             if (reorder_blocks < 0) throw py::value_error("reorder_blocks must be 0 or more");
             auto make = [&]() {
               if (decimate.is_none()) return live::PingAssembler(t, nullptr, nullptr, sample_bytes);
               const live::DecimateOptions o = decimate_dict(decimate, "PingAssemblerCpp: ");
               if (reference.is_none()) return live::PingAssembler(t, &o, nullptr, sample_bytes);
               const live::Reference ref = reference_dict(reference, "PingAssemblerCpp: ");
               return live::PingAssembler(t, &o, &ref, sample_bytes);
             };
             live::PingAssembler a = make();
             a.reorder_blocks = reorder_blocks;
             a.max_held_bytes = max_held_bytes;
             return a;
           }),
           py::arg("template_record"), py::arg("decimate") = py::none(), py::arg("reference") = py::none(), py::arg("sample_bytes") = 3,
           py::arg("reorder_blocks") = live::kReorderBlocks, py::arg("max_held_bytes") = live::kMaxHeldBytes)
      .def("feed",
           [](live::PingAssembler& a, const py::bytes& block) {
             const std::string data = block;
             std::vector<PingRecord> out;
             a.feed(reinterpret_cast<const unsigned char*>(data.data()), data.size(), out);
             return out;
           })
      .def("finish",
           [](live::PingAssembler& a) {
             std::vector<PingRecord> out;
             a.finish(out);
             return out;
           })
      .def("holds", &live::PingAssembler::holds)
      .def_readonly("used_blocks", &live::PingAssembler::used_blocks)
      .def_readonly("held_peak_bytes", &live::PingAssembler::held_peak_bytes)
      .def_readonly("open_peak", &live::PingAssembler::open_peak)
      .def_readonly("max_held_bytes", &live::PingAssembler::max_held_bytes)
      .def_readonly("dropped_pings", &live::PingAssembler::dropped_pings)
      .def_readonly("lost_samples", &live::PingAssembler::lost_samples)
      .def_readonly("overflow_blocks", &live::PingAssembler::overflow_blocks)
      .def_readonly("late_blocks", &live::PingAssembler::late_blocks)
      .def_readonly("too_late_blocks", &live::PingAssembler::too_late_blocks)
      .def_readonly("repeated_blocks", &live::PingAssembler::repeated_blocks)
      .def_readonly("restarts", &live::PingAssembler::restarts)
      .def_readonly("reorder_blocks", &live::PingAssembler::reorder_blocks)
      .def_property_readonly("last_block",
                             [](const live::PingAssembler& a) -> py::object {
                               using B = live::PingAssembler::Block;
                               switch (a.last_block) {
                                 case B::kInOrder: return py::str("in_order");
                                 case B::kGap: return py::str("gap");
                                 case B::kLate: return py::str("late");
                                 case B::kTooLate: return py::str("too_late");
                                 case B::kRepeated: return py::str("repeated");
                                 case B::kHeld: return py::str("held");
                                 case B::kRestart: return py::str("restart");
                                 default: return py::none();
                               }
                             })
      .def_readonly("bad_blocks", &live::PingAssembler::bad_blocks);
}
