#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/environment.hpp"
#include "sonar/live/elevation.hpp"
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
  d["snr"] = t.snr;
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

void number(const py::dict& d, const char* k, double& v) {
  if (d.contains(k) && !d[k].is_none()) v = d[k].cast<double>();
}

live::DecimateOptions decimate_options(const py::dict& d) {
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

live::Reference reference_options(const py::dict& r) {
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

live::WaterDepthOptions water_options(const py::dict& d) {
  live::WaterDepthOptions o;
  if (d.contains("from_altitude")) o.from_altitude = d["from_altitude"].cast<bool>();
  number(d, "dvl_offset_m", o.dvl_offset_m);
  number(d, "altitude_sigma_m", o.altitude_sigma_m);
  number(d, "depth_sigma_m", o.depth_sigma_m);
  number(d, "hold_s", o.hold_s);
  number(d, "hold_sigma_m_per_s", o.hold_sigma_m_per_s);
  number(d, "hold_sigma_m_per_m", o.hold_sigma_m_per_m);
  number(d, "max_altitude_m", o.max_altitude_m);
  return o;
}

live::NavOptions nav_options(const py::dict& d) {
  live::NavOptions o;
  if (d.contains("water_depth") && !d["water_depth"].is_none()) o.water_depth = water_options(d["water_depth"].cast<py::dict>());
  number(d, "keep_s", o.keep_s);
  number(d, "surface_z", o.surface_z);
  number(d, "profile_step_m", o.profile_step_m);
  number(d, "history_margin_s", o.history_margin_s);
  return o;
}

live::ProfileOptions profile_options(const py::dict& d) {
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
        [](const PingRecord& record, const py::dict& options) { return plan_dict(live::plan_decimation(record, decimate_options(options))); },
        py::arg("record"), py::arg("options") = py::dict(),
        "The filter and factor the decimate-first stage uses for a record (decimate.hpp).");
  m.def(
      "live_decimate_record",
      [](PingRecord& record, const py::dict& options, const py::object& reference) {
        const live::DecimateOptions o = decimate_options(options);
        live::SyncResult r;
        if (reference.is_none()) {
          py::gil_scoped_release release;
          r = live::decimate_record(record, o);
        } else {
          const live::Reference ref = reference_options(reference.cast<py::dict>());
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
             if (reference.is_none()) return live::PingDecimator(t, decimate_options(options));
             const live::Reference ref = reference_options(reference.cast<py::dict>());
             return live::PingDecimator(t, decimate_options(options), &ref);
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
             const live::DecimateOptions o = decimate_options(decimate.cast<py::dict>());
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
      .def(py::init([](const py::dict& options) { return live::Navigation(nav_options(options)); }), py::arg("options") = py::dict())
      .def("push", &live::Navigation::push)
      .def("apply", &live::Navigation::apply)
      .def("__len__", &live::Navigation::size);
  py::class_<live::WaterDepthTracker>(m, "WaterDepthTracker")
      .def(py::init([](const py::dict& options) { return live::WaterDepthTracker(water_options(options)); }), py::arg("options") = py::dict())
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
      .def(py::init([](const py::dict& options) { return live::SoundSpeedLadder(profile_options(options)); }), py::arg("options") = py::dict())
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
        live::family_profile(profile_options(options), own_depth, sound_speed, bottom, depth, speed);
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
        if (!profile.is_none()) po = profile_options(profile.cast<py::dict>());
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
      [](PingRecord& record, const py::dict& reference) { return sync_dict(live::synchronise(record, reference_options(reference))); },
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
           [](live::Tracker& t, double time, const std::vector<std::tuple<double, double, double, std::int64_t>>& dets) {
             std::vector<live::Detection> d;
             for (const auto& [r, b, s, c] : dets) d.push_back({r, b, s, c});
             py::list out;
             for (const live::Track* tr : t.update(time, d)) out.append(track_dict(*tr));
             return out;
           })
      .def("classify",
           [](const live::Tracker& t, double min_speed, double static_speed, double significance, double min_seen,
              std::int64_t min_points, double max_fit_rms, bool linear, double linear_length, double linear_width, bool multipath,
              double max_excess_range, double multipath_bearing, double multipath_velocity, double origin_x, double origin_y) {
             std::vector<const live::Track*> confirmed;
             for (const auto& tr : t.tracks()) {
               if (tr.confirmed) confirmed.push_back(&tr);
             }
             const live::ClassifyOptions o{min_speed, static_speed, significance, min_seen, min_points, max_fit_rms, linear,
                                           linear_length, linear_width, multipath, max_excess_range, multipath_bearing,
                                           multipath_velocity, origin_x, origin_y};
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
               out.append(d);
             }
             return out;
           },
           py::arg("min_speed") = 0.5, py::arg("static_speed") = 0.5, py::arg("significance") = 5.0, py::arg("min_seen") = 0.7,
           py::arg("min_points") = 4, py::arg("max_fit_rms") = -1.0, py::arg("linear") = false, py::arg("linear_length") = 40.0,
           py::arg("linear_width") = 2.0, py::arg("multipath") = false, py::arg("max_excess_range") = 800.0,
           py::arg("multipath_bearing") = 0.05235987755982988, py::arg("multipath_velocity") = 2.5, py::arg("origin_x") = 0.0,
           py::arg("origin_y") = 0.0);

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
      [](const PingRecord& record, const py::dict& given) {
        // The chain's derived settings for this record: `given` holds the ones the configuration set (the C++ option names).
        live::ChainOptions o;
        number(given, "range_cell_m", o.range_cell_m);
        number(given, "cfar_guard_m", o.cfar_guard_m);
        number(given, "cfar_train_m", o.cfar_train_m);
        number(given, "tracker_range_sigma", o.tracker.range_sigma);
        number(given, "tracker_bearing_sigma", o.tracker.bearing_sigma);
        number(given, "multipath_bearing", o.classify.multipath_bearing);
        number(given, "classify_max_fit_rms", o.classify.max_fit_rms);
        number(given, "row_tolerance_m", o.elevation.row_tolerance_m);
        number(given, "bearing_step_deg", o.bearing_step_deg);
        if (given.contains("min_row_elements") && !given["min_row_elements"].is_none()) {
          o.elevation.min_row_elements = given["min_row_elements"].cast<std::int64_t>();
        }
        const live::ChainDefaults d = live::resolve(o, record);
        py::dict out;
        out["range_cell_m"] = d.range_cell_m;
        out["cfar_guard_m"] = d.cfar_guard_m;
        out["cfar_train_m"] = d.cfar_train_m;
        out["tracker_range_sigma"] = d.tracker_range_sigma;
        out["tracker_bearing_sigma"] = d.tracker_bearing_sigma;
        out["multipath_bearing"] = d.multipath_bearing;
        out["classify_max_fit_rms"] = d.classify_max_fit_rms;
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
         std::int64_t rows, std::int64_t guard_cells, std::int64_t train_cells, bool hann, const py::dict& kw) {
        live::DetectorOptions o;
        number(kw, "false_alarms_per_ping", o.false_alarms_per_ping);
        number(kw, "sidelobe_margin_db", o.sidelobe_margin_db);
        number(kw, "sidelobe_floor_db", o.sidelobe_floor_db);
        number(kw, "search_factor", o.search_factor);
        if (kw.contains("integrate_cells") && !kw["integrate_cells"].is_none()) o.integrate_cells = kw["integrate_cells"].cast<std::vector<std::int64_t>>();
        double tolerance = live::kDerive;
        number(kw, "row_tolerance_m", tolerance);
        std::int64_t min_elements = 3;
        if (kw.contains("min_row_elements") && !kw["min_row_elements"].is_none()) min_elements = kw["min_row_elements"].cast<std::int64_t>();
        const live::SonarFacts facts = live::sonar_facts(record, live::row_tolerance(record, tolerance), min_elements);
        double merge_m = 0.0;
        number(kw, "merge_m", merge_m);
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

  py::class_<live::LiveChain>(m, "LiveChainCpp")
      .def(py::init([](const py::dict& kw) {
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
        if (kw.contains("cfar_integrate_cells")) o.detector.integrate_cells = kw["cfar_integrate_cells"].cast<std::vector<std::int64_t>>();
        if (kw.contains("beamformer")) o.beamformer = kw["beamformer"].cast<std::string>();
        if (kw.contains("backend")) o.backend = kw["backend"].cast<Backend>();
        get("sound_speed", o.sound_speed);
        if (kw.contains("dead_channels")) o.dead_channels = kw["dead_channels"].cast<std::vector<std::int64_t>>();
        if (kw.contains("elevation")) {
          const py::dict e = kw["elevation"].cast<py::dict>();
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
          if (e.contains("profile") && !e["profile"].is_none()) eo.profile = profile_options(e["profile"].cast<py::dict>());
        }
        if (kw.contains("decimate") && !kw["decimate"].is_none()) o.decimate = decimate_options(kw["decimate"].cast<py::dict>());
        if (kw.contains("blanking") && !kw["blanking"].is_none()) {
          const py::dict b = kw["blanking"].cast<py::dict>();
          o.blanking.enabled = true;
          number(b, "recovery_s", o.blanking.recovery_s);
          number(b, "taper_s", o.blanking.taper_s);
          if (b.contains("mode")) o.blanking.zero = b["mode"].cast<std::string>() != "ignore";
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
          const py::dict r = kw["sync"].cast<py::dict>();
          o.sync = true;
          o.reference = reference_options(r);
        }
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
      .def("resolved", &live::LiveChain::resolved_json,
           "The settings the first ping resolved (derive.hpp), as a JSON line; '{}' before it.")
      .def("last_map", [](const live::LiveChain& c) {
        py::array_t<float> power({static_cast<py::ssize_t>(c.bearings.size()), static_cast<py::ssize_t>(c.last_ranges.size())});
        std::copy(c.last_power.begin(), c.last_power.end(), power.mutable_data());
        return py::make_tuple(power, c.last_ranges, c.bearings);
      });

  py::class_<live::PingAssembler>(m, "PingAssemblerCpp")
      .def(py::init([](const PingRecord& t, const py::object& decimate, const py::object& reference, int sample_bytes) {
             if (decimate.is_none()) return live::PingAssembler(t, nullptr, nullptr, sample_bytes);
             const live::DecimateOptions o = decimate_options(decimate.cast<py::dict>());
             if (reference.is_none()) return live::PingAssembler(t, &o, nullptr, sample_bytes);
             const live::Reference ref = reference_options(reference.cast<py::dict>());
             return live::PingAssembler(t, &o, &ref, sample_bytes);
           }),
           py::arg("template_record"), py::arg("decimate") = py::none(), py::arg("reference") = py::none(), py::arg("sample_bytes") = 3)
      .def("feed",
           [](live::PingAssembler& a, const py::bytes& block) {
             const std::string data = block;
             std::vector<PingRecord> out;
             a.feed(reinterpret_cast<const unsigned char*>(data.data()), data.size(), out);
             return out;
           })
      .def_readonly("dropped_pings", &live::PingAssembler::dropped_pings)
      .def_readonly("lost_samples", &live::PingAssembler::lost_samples)
      .def_readonly("bad_blocks", &live::PingAssembler::bad_blocks);
}
