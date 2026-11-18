#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/elevation.hpp"
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

}  // namespace

void bind_live(py::module_& m) {
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
                       std::int64_t max_misses, double initial_speed_sigma) {
             return live::Tracker({process_noise, range_sigma, bearing_sigma, gate, confirm_hits, max_misses, initial_speed_sigma});
           }),
           py::arg("process_noise") = 0.05, py::arg("range_sigma") = 2.0, py::arg("bearing_sigma") = 0.017453292519943295,
           py::arg("gate") = 4.0, py::arg("confirm_hits") = 3, py::arg("max_misses") = 3, py::arg("initial_speed_sigma") = 10.0)
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
          out.append(d);
        }
        return py::make_tuple(out, bearings);
      },
      py::arg("record"), py::arg("f_low") = 0.0, py::arg("f_high") = 0.0, py::arg("n_fft") = 1024, py::arg("hop") = 512,
      py::arg("n_average") = 8, py::arg("first_deg") = -80.0, py::arg("last_deg") = 80.0, py::arg("step_deg") = 0.5,
      py::arg("method") = "sbl", py::arg("max_iterations") = 60, py::arg("tolerance") = 1e-3, py::arg("max_sources") = 4,
      py::arg("dynamic_range_db") = 20.0, py::arg("min_snr_db") = 0.0, py::arg("backend") = Backend::Host,
      py::arg("kernels") = true, py::arg("elevation") = false, py::arg("elevation_first_deg") = -30.0,
      py::arg("elevation_last_deg") = 30.0, py::arg("elevation_step_deg") = 0.5, py::arg("look") = "starboard",
      "Passive bearing-time record of one stretch of data: (frames, bearing grid in rad). With `elevation` and an array "
      "of stacked rows every frame also has `elevations` (rad, level-referenced, up positive) and `elevation_quality`.");

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
        if (kw.contains("tracker_max_misses")) o.tracker.max_misses = kw["tracker_max_misses"].cast<std::int64_t>();
        if (kw.contains("sbl_max_sources")) o.sbl.max_sources = kw["sbl_max_sources"].cast<std::int32_t>();
        if (kw.contains("gate_fft")) o.gate_fft = kw["gate_fft"].cast<std::int64_t>();
        if (kw.contains("multipath")) o.classify.multipath = kw["multipath"].cast<bool>();
        if (kw.contains("sbl_kernels")) o.sbl.kernels = kw["sbl_kernels"].cast<bool>();
        if (kw.contains("hann")) o.hann = kw["hann"].cast<bool>();
        if (kw.contains("starboard")) o.starboard = kw["starboard"].cast<bool>();
        if (kw.contains("doa")) o.doa = kw["doa"].cast<std::string>();
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
          if (e.contains("ssp_depth")) eo.ssp_depth = e["ssp_depth"].cast<std::vector<double>>();
          if (e.contains("ssp_speed")) eo.ssp_speed = e["ssp_speed"].cast<std::vector<double>>();
          if (eo.ssp_depth.size() != eo.ssp_speed.size()) throw std::invalid_argument("elevation.ssp: depth and speed must be as long as each other");
        }
        if (kw.contains("sync")) {
          const py::dict r = kw["sync"].cast<py::dict>();
          o.sync = true;
          o.reference.f_start = r["f_start"].cast<double>();
          o.reference.f_stop = r["f_stop"].cast<double>();
          o.reference.duration = r["duration"].cast<double>();
          o.reference.lead = r["lead"].cast<double>();
          o.reference.tail = r["tail"].cast<double>();
          if (r.contains("dedicated")) o.reference.dedicated = r["dedicated"].cast<bool>();
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
      .def("last_map", [](const live::LiveChain& c) {
        py::array_t<float> power({static_cast<py::ssize_t>(c.bearings.size()), static_cast<py::ssize_t>(c.last_ranges.size())});
        std::copy(c.last_power.begin(), c.last_power.end(), power.mutable_data());
        return py::make_tuple(power, c.last_ranges, c.bearings);
      });

  py::class_<live::PingAssembler>(m, "PingAssemblerCpp")
      .def(py::init<const PingRecord&>())
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
