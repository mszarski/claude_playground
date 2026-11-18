#include "sonar/live/options.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace sonar::live {
namespace {
constexpr double kDeg = 3.14159265358979323846 / 180.0;
}  // namespace

void set_option(ChainOptions& c, PassiveOptions& p, const std::string& key, const std::string& value) {
  const double v = std::atof(value.c_str());
  if (key == "bearing_first_deg") c.bearing_first_deg = p.bearing_first_deg = v;
  else if (key == "bearing_last_deg") c.bearing_last_deg = p.bearing_last_deg = v;
  else if (key == "bearing_step_deg") c.bearing_step_deg = p.bearing_step_deg = v;
  else if (key == "range_cell_m") c.range_cell_m = v;
  else if (key == "cfar_pfa") c.cfar_pfa = v;
  else if (key == "cfar_guard_m") c.cfar_guard_m = v;
  else if (key == "cfar_train_m") c.cfar_train_m = v;
  else if (key == "min_snr_db") c.min_snr_db = v;
  else if (key == "cell_statistic") {
    if (value != "mean" && value != "peak") throw std::invalid_argument("cell_statistic: mean or peak");
    c.cell_peak = value == "peak";
  }
  else if (key == "beamformer") c.beamformer = value;
  else if (key == "doa") c.doa = value;
  else if (key == "mvdr_subbands") c.mvdr.n_subbands = static_cast<std::int64_t>(v);
  else if (key == "mvdr_loading") c.mvdr.loading = v;
  else if (key == "gate_fft") c.gate_fft = static_cast<std::int64_t>(v);
  else if (key == "sbl_grid_step_deg") c.sbl_grid_step_deg = v;
  else if (key == "sbl_max_iterations") c.sbl.max_iterations = p.sbl.max_iterations = static_cast<std::int32_t>(v);
  else if (key == "sbl_tolerance") c.sbl.tolerance = p.sbl.tolerance = v;
  else if (key == "sbl_max_gates") c.sbl_max_gates = static_cast<std::int64_t>(v);
  else if (key == "sbl_min_contrast_db") c.sbl_min_contrast_db = v;
  else if (key == "sbl_max_sources") c.sbl.max_sources = p.sbl.max_sources = static_cast<std::int32_t>(v);
  else if (key == "sbl_min_snr_db") c.sbl.min_snr_db = p.sbl.min_snr_db = v;
  else if (key == "sbl_dynamic_range_db") c.sbl.dynamic_range_db = p.sbl.dynamic_range_db = v;
  else if (key == "look") c.starboard = p.starboard = value != "port";
  else if (key == "elevation") c.elevation.enabled = p.elevation.enabled = v != 0.0;
  else if (key == "elevation_first_deg") c.elevation.first_deg = p.elevation.first_deg = v;
  else if (key == "elevation_last_deg") c.elevation.last_deg = p.elevation.last_deg = v;
  else if (key == "elevation_step_deg") c.elevation.step_deg = p.elevation.step_deg = v;
  else if (key == "elevation_range_window_m") c.elevation.range_window_m = v;
  else if (key == "elevation_max_detections") c.elevation.max_detections = static_cast<std::int64_t>(v);
  else if (key == "elevation_first_arrival") c.elevation.first_arrival = v != 0.0;
  else if (key == "elevation_first_arrival_db") c.elevation.first_arrival_db = v;
  else if (key == "elevation_first_arrival_snr_db") c.elevation.first_arrival_snr_db = v;
  else if (key == "sound_speed") c.sound_speed = v;
  else if (key == "elevation_ambiguity_sigma") c.elevation.ambiguity_sigma = v;
  else if (key == "elevation_alternatives_db") c.elevation.alternatives_db = v;
  else if (key == "elevation_sector_from_depth") c.elevation.sector_from_depth = v != 0.0;
  else if (key == "elevation_sector_bounces") c.elevation.sector_bounces = v;
  else if (key == "elevation_sector_margin_deg") c.elevation.sector_margin = v * kDeg;
  else if (key == "elevation_water_depth_sigma") c.elevation.water_depth_sigma = v;
  else if (key == "elevation_attitude_arrival") c.elevation.attitude_arrival = v != 0.0;
  else if (key == "elevation_attitude_rate_sigma_deg_s") c.elevation.attitude_rate_sigma = v * kDeg;
  else if (key == "elevation_use_gradient") c.elevation.use_gradient = v != 0.0;
  else if (key == "elevation_gradient") c.elevation.gradient = v;
  else if (key == "elevation_gradient_sigma") c.elevation.gradient_sigma = v;
  else if (key == "elevation_min_quality") c.elevation.min_quality = p.elevation.min_quality = v;
  else if (key == "elevation_ambiguity_db") c.elevation.ambiguity_db = v;
  else if (key == "elevation_correct_bearing") c.elevation.correct_bearing = v != 0.0;
  else if (key == "elevation_depth") c.elevation.depth = v != 0.0;
  else if (key == "elevation_fold") c.elevation.fold = v != 0.0;
  else if (key == "elevation_classify") c.elevation.classify = v != 0.0;
  else if (key == "elevation_classify_sigma") c.elevation.classify_sigma = v;
  else if (key == "elevation_water_depth") c.elevation.water_depth = v;
  else if (key == "elevation_surface_z") c.elevation.surface_z = v;
  else if (key == "elevation_roll_sigma_deg") c.elevation.roll_sigma = v * kDeg;
  else if (key == "elevation_depth_process_noise") c.tracker.depth_process_noise = v;
  else if (key == "elevation_row_tolerance_m") c.elevation.row_tolerance_m = p.elevation.row_tolerance_m = v;
  else if (key == "blanking") c.blanking.enabled = v != 0.0;
  else if (key == "blanking_recovery_s") c.blanking.recovery_s = v;
  else if (key == "blanking_taper_s") c.blanking.taper_s = v;
  else if (key == "blanking_mode") c.blanking.zero = value != "ignore";
  else if (key == "sync_min_quality") c.reference.min_quality = v;
  else if (key == "sync_search_time") c.reference.search_time = v;
  else if (key == "sync_tolerance_us") c.reference.tolerance_us = v;
  else if (key == "sync_clip_level") c.reference.clip_level = v;
  else if (key == "elevation_min_row_elements") c.elevation.min_row_elements = p.elevation.min_row_elements = static_cast<std::int64_t>(v);
  else if (key == "nav_history_margin_s") c.nav.history_margin_s = v;
  // The program around the chain: the board stream, its socket and queue, the report datagrams.
  else if (key == "stream_sample_bytes") c.stream.sample_bytes = static_cast<int>(v);
  else if (key == "stream_buffer_bytes") c.stream.buffer_bytes = static_cast<std::int64_t>(v);
  else if (key == "stream_queue_mb") c.stream.queue_mb = v;
  else if (key == "idle_seconds") c.stream.idle_seconds = v;
  else if (key == "bind") c.stream.bind = value;
  else if (key == "report_udp_max_bytes") c.stream.report_max_bytes = static_cast<std::int64_t>(v);
  else if (key == "decimate") c.decimate.enabled = v != 0.0;
  else if (key == "decimate_streaming") c.decimate_streaming = v != 0.0;
  else if (key == "decimate_f_low") c.decimate.f_low = v;
  else if (key == "decimate_f_high") c.decimate.f_high = v;
  else if (key == "decimate_band_margin") c.decimate.band_margin = v;
  else if (key == "decimate_output_rate") c.decimate.output_rate = v;
  else if (key == "decimate_oversample") c.decimate.oversample = v;
  else if (key == "decimate_ripple_db") c.decimate.ripple_db = v;
  else if (key == "decimate_stopband_db") c.decimate.stopband_db = v;
  else if (key == "decimate_taps") c.decimate.taps = static_cast<std::int64_t>(v);
  else if (key == "nav_water_depth_from_altitude") c.nav.water_depth.from_altitude = v != 0.0;
  else if (key == "nav_dvl_offset_m") c.nav.water_depth.dvl_offset_m = v;
  else if (key == "nav_altitude_sigma_m") c.nav.water_depth.altitude_sigma_m = v;
  else if (key == "nav_depth_sigma_m") c.nav.water_depth.depth_sigma_m = v;
  else if (key == "nav_hold_s") c.nav.water_depth.hold_s = v;
  else if (key == "nav_hold_sigma_m_per_s") c.nav.water_depth.hold_sigma_m_per_s = v;
  else if (key == "nav_hold_sigma_m_per_m") c.nav.water_depth.hold_sigma_m_per_m = v;
  else if (key == "nav_max_altitude_m") c.nav.water_depth.max_altitude_m = v;
  else if (key == "nav_keep_s") c.nav.keep_s = v;
  else if (key == "nav_profile_step_m") c.nav.profile_step_m = v;
  else if (key == "elevation_profile") c.elevation.profile.enabled = v != 0.0;
  else if (key == "elevation_profile_bin_m") c.elevation.profile.bin_m = v;
  else if (key == "elevation_profile_sample_sigma") c.elevation.profile.sample_sigma = v;
  else if (key == "elevation_profile_age_sigma_per_hour") c.elevation.profile.age_sigma_per_hour = v;
  else if (key == "elevation_profile_distance_sigma_per_km") c.elevation.profile.distance_sigma_per_km = v;
  else if (key == "elevation_profile_max_age_s") c.elevation.profile.max_age_s = v;
  else if (key == "elevation_profile_max_distance_m") c.elevation.profile.max_distance_m = v;
  else if (key == "elevation_profile_min_span_m") c.elevation.profile.min_span_m = v;
  else if (key == "elevation_profile_extrapolation_sigma_per_s") c.elevation.profile.extrapolation_sigma_per_s = v;
  else if (key == "elevation_profile_file") c.elevation.profile.file = value;
  else if (key == "elevation_profile_cast") c.elevation.profile.cast = value;
  else if (key == "elevation_profile_supplied_sigma") c.elevation.profile.supplied_sigma = v;
  else if (key == "elevation_profile_family") c.elevation.profile.family = value;
  else if (key == "elevation_profile_family_gradient") c.elevation.profile.family_gradient = v;
  else if (key == "elevation_profile_layer_depth") c.elevation.profile.layer_depth = v;
  else if (key == "elevation_profile_thermocline_gradient") c.elevation.profile.thermocline_gradient = v;
  else if (key == "elevation_profile_thermocline_thickness") c.elevation.profile.thermocline_thickness = v;
  else if (key == "elevation_profile_family_sigma") c.elevation.profile.family_sigma = v;
  else if (key == "elevation_profile_ladder") {
    c.elevation.profile.ladder.clear();
    std::size_t at = 0;
    while (at < value.size()) {
      const std::size_t comma = std::min(value.find(',', at), value.size());
      if (comma > at) c.elevation.profile.ladder.push_back(value.substr(at, comma - at));
      at = comma + 1;
    }
  }
  else if (key == "dead_channels" || key == "elevation_ssp_depth" || key == "elevation_ssp_speed") {
    // Comma-separated lists.
    std::vector<double> list;
    std::size_t at = 0;
    while (at < value.size()) {
      const std::size_t comma = std::min(value.find(',', at), value.size());
      if (comma > at) list.push_back(std::atof(value.substr(at, comma - at).c_str()));
      at = comma + 1;
    }
    if (key == "elevation_ssp_depth") c.elevation.ssp_depth = list;
    else if (key == "elevation_ssp_speed") c.elevation.ssp_speed = list;
    else {
      c.dead_channels.clear();
      for (double x : list) c.dead_channels.push_back(static_cast<std::int64_t>(x));
    }
  }
  else if (key == "backend") c.backend = p.sbl.backend = value == "cuda" ? Backend::Cuda : Backend::Host;
  else if (key == "sbl_kernels") c.sbl.kernels = p.sbl.kernels = v != 0.0;
  else if (key == "tracker_range_sigma") c.tracker.range_sigma = v;
  else if (key == "tracker_bearing_sigma_deg") c.tracker.bearing_sigma = v * kDeg;
  else if (key == "tracker_confirm_hits") c.tracker.confirm_hits = static_cast<std::int64_t>(v);
  else if (key == "tracker_max_misses") c.tracker.max_misses = static_cast<std::int64_t>(v);
  else if (key == "classify_min_speed") c.classify.min_speed = c.classify.static_speed = v;
  else if (key == "classify_static_speed") c.classify.static_speed = v;   // after classify_min_speed, which sets both
  else if (key == "classify_max_fit_rms") c.classify.max_fit_rms = v;
  else if (key == "tracker_initial_speed_sigma") c.tracker.initial_speed_sigma = v;
  else if (key == "tracker_process_noise") c.tracker.process_noise = v;
  else if (key == "multipath_max_excess_range") c.classify.multipath_max_excess_range = v;
  else if (key == "multipath_bearing_deg") c.classify.multipath_bearing = v * kDeg;
  else if (key == "multipath_velocity") c.classify.multipath_velocity = v;
  else if (key == "window") c.hann = value == "hann";
  else if (key == "multipath") c.classify.multipath = v != 0.0;
  else if (key == "sync") c.sync = v != 0.0;
  else if (key == "sync_f_start") c.reference.f_start = v;
  else if (key == "sync_f_stop") c.reference.f_stop = v;
  else if (key == "sync_duration") c.reference.duration = v;
  else if (key == "sync_lead") c.reference.lead = v;
  else if (key == "sync_tail") c.reference.tail = v;
  else if (key == "sync_dedicated") c.reference.dedicated = v != 0.0;
  else if (key == "f_low") p.f_low = v;
  else if (key == "f_high") p.f_high = v;
  else if (key == "n_fft") p.n_fft = static_cast<std::int64_t>(v);
  else if (key == "hop") p.hop = static_cast<std::int64_t>(v);
  else if (key == "n_average") p.n_average = static_cast<std::int64_t>(v);
  else if (key == "passive_method") p.use_sbl = value != "conventional";
  else if (set_tonal_option(p.tonal, key, value)) {}                 // tonal...: Goertzel lines and DEMON (tonal.hpp)
  else if (set_multipath_option(c.multipath_depth, key, value)) {}   // multipath_depth...: depth from multipath timing (multipath_depth.hpp)
  else throw std::invalid_argument("unknown option " + key);
}

void set_pair(ChainOptions& c, PassiveOptions& p, const std::string& pair) {
  const auto eq = pair.find('=');
  if (eq == std::string::npos) throw std::invalid_argument("expected key=value, got " + pair);
  set_option(c, p, pair.substr(0, eq), pair.substr(eq + 1));
}


void load_options(ChainOptions& c, PassiveOptions& p, const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::invalid_argument("cannot read options file " + path);
  std::string line;
  while (std::getline(f, line)) {
    line = line.substr(0, line.find('#'));
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t\r") + 1);
    if (!line.empty()) set_pair(c, p, line);
  }
}

}  // namespace sonar::live
