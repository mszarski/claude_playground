#include "sonar/live/options.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "sonar/core/platform.hpp"
#include "sonar/live/option_value.hpp"

namespace sonar::live {
namespace {
constexpr double kDeg = 3.14159265358979323846 / 180.0;

std::string trimmed(const std::string& text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

[[noreturn]] void refuse(const std::string& key, const std::string& value, const std::string& expected) {
  throw std::invalid_argument(key + ": expected " + expected + ", got '" + value + "'");
}

// The elements of a comma-separated list, blanks trimmed, one pair of enclosing brackets dropped.
std::vector<std::string> list_items(const std::string& value) {
  std::string text = trimmed(value);
  if (text.size() >= 2 && text.front() == '[' && text.back() == ']') text = text.substr(1, text.size() - 2);
  std::vector<std::string> out;
  if (trimmed(text).empty()) return out;
  std::size_t at = 0;
  while (at <= text.size()) {
    const std::size_t comma = std::min(text.find(',', at), text.size());
    out.push_back(trimmed(text.substr(at, comma - at)));
    at = comma + 1;
  }
  return out;
}
}  // namespace

double option_number(const std::string& key, const std::string& value) {
  const std::string text = trimmed(value);
  if (text.empty()) refuse(key, value, "a number");
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(text.c_str(), &end);
  if (end != text.c_str() + text.size() || errno == ERANGE) refuse(key, value, "a number");
  return v;
}

double option_nonnegative(const std::string& key, const std::string& value) {
  const double v = option_number(key, value);
  if (!(v >= 0.0) || !std::isfinite(v)) refuse(key, value, "a finite number >= 0");
  return v;
}

std::int64_t option_integer(const std::string& key, const std::string& value) {
  double v = 0.0;
  try {
    v = option_number(key, value);
  } catch (const std::invalid_argument&) {
    refuse(key, value, "a whole number");
  }
  if (!std::isfinite(v) || v != std::floor(v) || std::abs(v) > 9.0e15) refuse(key, value, "a whole number");
  return static_cast<std::int64_t>(v);
}

bool option_flag(const std::string& key, const std::string& value) {
  const std::string text = trimmed(value);
  if (text == "true" || text == "True") return true;
  if (text == "false" || text == "False") return false;
  double v = 0.0;
  try {
    v = option_number(key, value);
  } catch (const std::invalid_argument&) {
    refuse(key, value, "0 or 1 (or true / false)");
  }
  if (!std::isfinite(v)) refuse(key, value, "0 or 1 (or true / false)");
  return v != 0.0;
}

std::vector<double> option_numbers(const std::string& key, const std::string& value) {
  std::vector<double> out;
  for (const std::string& item : list_items(value)) {
    if (item.empty()) refuse(key, value, "numbers separated by commas");
    try {
      out.push_back(option_number(key, item));
    } catch (const std::invalid_argument&) {
      refuse(key, value, "numbers separated by commas");
    }
  }
  return out;
}

std::vector<std::int64_t> option_integers(const std::string& key, const std::string& value) {
  std::vector<std::int64_t> out;
  for (const std::string& item : list_items(value)) {
    if (item.empty()) refuse(key, value, "whole numbers separated by commas");
    try {
      out.push_back(option_integer(key, item));
    } catch (const std::invalid_argument&) {
      refuse(key, value, "whole numbers separated by commas");
    }
  }
  return out;
}

std::string option_choice(const std::string& key, const std::string& value, const std::vector<std::string>& allowed) {
  for (const std::string& a : allowed) {
    if (value == a) return a;
  }
  std::string words;
  for (std::size_t k = 0; k < allowed.size(); ++k) words += (k ? (k + 1 == allowed.size() ? " or " : ", ") : "") + allowed[k];
  refuse(key, value, words);
}

std::string nearest_key(const std::string& key, const std::vector<std::string>& known) {
  std::string nearest;
  std::size_t best = std::max<std::size_t>(3, key.size() / 3) + 1;   // this many edits or more: no suggestion
  std::vector<std::size_t> row;
  for (const std::string& k : known) {
    row.resize(k.size() + 1);
    for (std::size_t j = 0; j <= k.size(); ++j) row[j] = j;
    for (std::size_t i = 1; i <= key.size(); ++i) {
      std::size_t diagonal = row[0];
      row[0] = i;
      for (std::size_t j = 1; j <= k.size(); ++j) {
        const std::size_t above = row[j];
        row[j] = std::min({row[j] + 1, row[j - 1] + 1, diagonal + (key[i - 1] == k[j - 1] ? 0 : 1)});
        diagonal = above;
      }
    }
    if (row[k.size()] < best) {
      best = row[k.size()];
      nearest = k;
    }
  }
  return nearest;
}

Backend parse_backend(const std::string& value) {
  const std::string b = option_choice("backend", value, {"auto", "host", "cuda"});
  if (b == "cuda") return Backend::Cuda;
  if (b == "host") return Backend::Host;
  // "auto", as gpusonar.live has it: CUDA when the build has it and a device is present, else the host.
  return cuda_available() ? Backend::Cuda : Backend::Host;
}

void program_defaults(ChainOptions& chain, PassiveOptions& passive) {
  chain.backend = passive.sbl.backend = parse_backend("auto");
}

void set_option(ChainOptions& c, PassiveOptions& p, const std::string& key, const std::string& value) {
  const auto v = [&] { return option_number(key, value); };
  const auto i = [&] { return option_integer(key, value); };
  const auto on = [&] { return option_flag(key, value); };
  // Every key compared is noted, so that an unknown one is refused with the nearest of them (sonar-1rf).
  std::vector<std::string> keys;
  const auto is = [&](const char* k) {
    keys.emplace_back(k);
    return key == k;
  };
  if (is("bearing_first_deg")) c.bearing_first_deg = p.bearing_first_deg = v();
  else if (is("bearing_last_deg")) c.bearing_last_deg = p.bearing_last_deg = v();
  else if (is("bearing_step_deg")) c.bearing_step_deg = p.bearing_step_deg = v();
  else if (is("range_cell_m")) c.range_cell_m = v();
  else if (is("cfar_pfa")) c.cfar_pfa = v();
  else if (is("cfar_guard_m")) c.cfar_guard_m = v();
  else if (is("cfar_train_m")) c.cfar_train_m = v();
  else if (is("min_snr_db")) c.min_snr_db = v();
  else if (is("cell_statistic")) {
    if (value != "mean" && value != "peak" && value != "multiscale") throw std::invalid_argument("cell_statistic: mean, peak or multiscale");
    c.cell_peak = value == "peak";
    c.cell_multiscale = value == "multiscale";
  }
  else if (is("cfar_false_alarms_per_ping")) c.detector.false_alarms_per_ping = v();
  else if (is("cfar_sidelobe_margin_db")) c.detector.sidelobe_margin_db = v();
  else if (is("cfar_sidelobe_floor_db")) c.detector.sidelobe_floor_db = v();
  else if (is("cfar_search_factor")) c.detector.search_factor = v();
  else if (is("cfar_merge_m")) c.detector.merge_m = v();
  else if (is("cfar_range_estimate")) c.range_estimate = parse_range_estimate(value);
  else if (is("doppler_mode")) c.doppler.mode = parse_doppler_mode(value);   // off | sweep_bank (doppler.hpp; D4)
  else if (is("doppler_max_range_rate_m_s")) c.doppler.max_range_rate = option_nonnegative(key, value);
  else if (is("doppler_step_m_s")) c.doppler.step = v();          // NaN (nan): c / (2 B T)
  else if (is("doppler_hypotheses")) c.doppler.hypotheses = i();  // odd; 0: from doppler_max_range_rate_m_s
  else if (is("doppler_unknown_ownship_m_s")) c.doppler.unknown_ownship = option_nonnegative(key, value);
  else if (is("doppler_threshold_db")) c.doppler.threshold_db = v();   // NaN (nan): measured (bank_threshold_db)
  else if (is("doppler_report")) c.doppler.report = on();
  else if (is("range_doppler")) c.range_doppler = parse_range_doppler(value);   // off | ownship | full (sonar-fyj)   // centroid | first (D-157); unset: chain_range_estimate (D-158)
  // The two flags as gpusonar.live.cpp_chain_options names them, for a tool that passes its keys on as they are.
  else if (is("cell_multiscale")) {
    c.cell_multiscale = on();
    if (c.cell_multiscale) c.cell_peak = false;
  }
  else if (is("cell_peak")) {
    c.cell_peak = on();
    if (c.cell_peak) c.cell_multiscale = false;
  }
  else if (is("cfar_integrate_cells")) c.detector.integrate_cells = option_integers(key, value);   // range cells: 1,4,16 (or [1, 4, 16])
  else if (is("beamformer")) c.beamformer = option_choice(key, value, {"backprojection", "farfield", "mvdr"});
  else if (is("tracker_association")) c.tracker.association = parse_association(value);
  else if (is("doa")) c.doa = option_choice(key, value, {"map", "sbl"});
  else if (is("mvdr_subbands")) c.mvdr.n_subbands = i();
  else if (is("mvdr_loading")) c.mvdr.loading = v();
  else if (is("gate_fft")) c.gate_fft = i();
  else if (is("sbl_grid_step_deg")) c.sbl_grid_step_deg = v();
  else if (is("sbl_max_iterations")) c.sbl.max_iterations = p.sbl.max_iterations = static_cast<std::int32_t>(i());
  else if (is("sbl_tolerance")) c.sbl.tolerance = p.sbl.tolerance = v();
  else if (is("sbl_max_gates")) c.sbl_max_gates = i();
  else if (is("sbl_min_contrast_db")) c.sbl_min_contrast_db = v();
  else if (is("sbl_max_sources")) c.sbl.max_sources = p.sbl.max_sources = static_cast<std::int32_t>(i());
  else if (is("sbl_min_snr_db")) c.sbl.min_snr_db = p.sbl.min_snr_db = v();
  else if (is("sbl_dynamic_range_db")) c.sbl.dynamic_range_db = p.sbl.dynamic_range_db = v();
  else if (is("look")) c.starboard = p.starboard = option_choice(key, value, {"starboard", "port"}) == "starboard";
  else if (is("elevation")) c.elevation.enabled = p.elevation.enabled = on();
  else if (is("elevation_first_deg")) c.elevation.first_deg = p.elevation.first_deg = v();
  else if (is("elevation_last_deg")) c.elevation.last_deg = p.elevation.last_deg = v();
  else if (is("elevation_step_deg")) c.elevation.step_deg = p.elevation.step_deg = v();
  else if (is("elevation_range_window_m")) c.elevation.range_window_m = v();
  else if (is("elevation_max_detections")) c.elevation.max_detections = i();
  else if (is("elevation_first_arrival")) c.elevation.first_arrival = on();
  else if (is("elevation_first_arrival_db")) c.elevation.first_arrival_db = v();
  else if (is("elevation_first_arrival_snr_db")) c.elevation.first_arrival_snr_db = v();
  else if (is("sound_speed")) c.sound_speed = v();
  else if (is("elevation_ambiguity_sigma")) c.elevation.ambiguity_sigma = v();
  else if (is("elevation_alternatives_db")) c.elevation.alternatives_db = v();
  else if (is("elevation_sector_from_depth")) c.elevation.sector_from_depth = on();
  else if (is("elevation_sector_bounces")) c.elevation.sector_bounces = v();
  else if (is("elevation_sector_margin_deg")) c.elevation.sector_margin = v() * kDeg;
  else if (is("elevation_water_depth_sigma")) c.elevation.water_depth_sigma = v();
  else if (is("elevation_attitude_arrival")) c.elevation.attitude_arrival = on();
  else if (is("elevation_attitude_rate_sigma_deg_s")) c.elevation.attitude_rate_sigma = v() * kDeg;
  else if (is("elevation_use_gradient")) c.elevation.use_gradient = on();
  else if (is("elevation_gradient")) c.elevation.gradient = v();
  else if (is("elevation_gradient_sigma")) c.elevation.gradient_sigma = v();
  else if (is("elevation_min_quality")) c.elevation.min_quality = p.elevation.min_quality = v();
  else if (is("elevation_ambiguity_db")) c.elevation.ambiguity_db = v();
  else if (is("elevation_correct_bearing")) c.elevation.correct_bearing = on();
  else if (is("elevation_depth")) c.elevation.depth = on();
  else if (is("elevation_fold")) c.elevation.fold = on();
  else if (is("elevation_classify")) c.elevation.classify = on();
  else if (is("elevation_classify_sigma")) c.elevation.classify_sigma = v();
  else if (is("elevation_water_depth")) c.elevation.water_depth = v();
  else if (is("elevation_surface_z")) c.elevation.surface_z = v();
  else if (is("elevation_roll_sigma_deg")) c.elevation.roll_sigma = v() * kDeg;
  else if (is("elevation_depth_process_noise")) c.tracker.depth_process_noise = v();
  else if (is("elevation_row_tolerance_m")) c.elevation.row_tolerance_m = p.elevation.row_tolerance_m = v();
  else if (is("blanking")) c.blanking.enabled = on();
  else if (is("blanking_recovery_s")) c.blanking.recovery_s = v();
  else if (is("blanking_taper_s")) c.blanking.taper_s = v();
  else if (is("blanking_mode")) c.blanking.zero = option_choice(key, value, {"zero", "ignore"}) == "zero";
  else if (is("sync_min_quality")) c.reference.min_quality = v();
  else if (is("sync_search_time")) c.reference.search_time = v();
  else if (is("sync_tolerance_us")) c.reference.tolerance_us = v();
  else if (is("sync_clip_level")) c.reference.clip_level = v();
  else if (is("elevation_min_row_elements")) c.elevation.min_row_elements = p.elevation.min_row_elements = i();
  else if (is("nav_surface_z")) c.nav.surface_z = v();   // z of the sea surface in the navigation's frame: depth = z - surface_z
  else if (is("nav_history_margin_s")) c.nav.history_margin_s = v();
  else if (is("nav_lever_arm_m") || is("nav_mount_rpy_deg")) {
    // Three numbers: the array from the navigation unit (sonar-3rv; NavOptions).
    const std::vector<double> q = option_numbers(key, value);
    if (q.size() != 3) refuse(key, value, "three comma-separated numbers");
    for (double x : q) {
      if (!std::isfinite(x)) refuse(key, value, "three finite numbers");
    }
    const double f = is("nav_lever_arm_m") ? 1.0 : kDeg;
    (is("nav_lever_arm_m") ? c.nav.lever_arm : c.nav.mount_rpy) = Vec3d{q[0] * f, q[1] * f, q[2] * f};
  }
  // The program around the chain: the board stream, its socket and queue, the report datagrams.
  else if (is("stream_sample_bytes")) c.stream.sample_bytes = static_cast<int>(i());
  else if (is("stream_buffer_bytes")) c.stream.buffer_bytes = i();
  else if (is("stream_queue_mb")) c.stream.queue_mb = v();
  else if (is("idle_seconds")) c.stream.idle_seconds = v();
  else if (is("bind")) c.stream.bind = value;
  else if (is("report_udp_max_bytes")) c.stream.report_max_bytes = i();
  else if (is("stream_boards")) c.stream.boards = on();
  else if (is("stream_board_channels")) c.stream.board_channels = value;
  else if (is("stream_board_reference")) c.stream.board_reference = static_cast<int>(i());
  else if (is("stream_board_missing")) c.stream.board_zero = option_choice(key, value, {"drop", "zero"}) == "zero";
  else if (is("stream_board_wait_s")) c.stream.board_wait_s = v();
  else if (is("stream_clock_ppm")) c.stream.clock_ppm = v();
  else if (is("stream_board_ppm")) c.stream.board_ppm = value;
  else if (is("stream_clock_estimate")) c.stream.clock_estimate = on();
  else if (is("stream_clock_discipline")) c.stream.clock_discipline = on();
  else if (is("stream_clock_bin_s")) c.stream.clock_bin_s = v();
  else if (is("stream_clock_window_s")) c.stream.clock_window_s = v();
  else if (is("stream_clock_min_span_s")) c.stream.clock_min_span_s = v();
  else if (is("stream_clock_max_sigma_ppm")) c.stream.clock_max_sigma_ppm = v();
  else if (is("decimate")) c.decimate.enabled = on();
  else if (is("decimate_streaming")) c.decimate_streaming = on();
  else if (is("decimate_f_low")) c.decimate.f_low = v();
  else if (is("decimate_f_high")) c.decimate.f_high = v();
  else if (is("decimate_band_margin")) c.decimate.band_margin = v();
  else if (is("decimate_output_rate")) c.decimate.output_rate = v();
  else if (is("decimate_oversample")) c.decimate.oversample = v();
  else if (is("decimate_ripple_db")) c.decimate.ripple_db = v();
  else if (is("decimate_stopband_db")) c.decimate.stopband_db = v();
  else if (is("decimate_taps")) c.decimate.taps = i();
  else if (is("decimate_blank_until_s")) c.decimate.blank_until_s = v();
  else if (is("decimate_blank_taper_s")) c.decimate.blank_taper_s = v();
  else if (is("nav_water_depth_from_altitude")) c.nav.water_depth.from_altitude = on();
  else if (is("nav_dvl_offset_m")) c.nav.water_depth.dvl_offset_m = v();
  else if (is("nav_altitude_sigma_m")) c.nav.water_depth.altitude_sigma_m = v();
  else if (is("nav_depth_sigma_m")) c.nav.water_depth.depth_sigma_m = v();
  else if (is("nav_hold_s")) c.nav.water_depth.hold_s = v();
  else if (is("nav_hold_sigma_m_per_s")) c.nav.water_depth.hold_sigma_m_per_s = v();
  else if (is("nav_hold_sigma_m_per_m")) c.nav.water_depth.hold_sigma_m_per_m = v();
  else if (is("nav_max_altitude_m")) c.nav.water_depth.max_altitude_m = v();
  else if (is("nav_keep_s")) c.nav.keep_s = v();
  else if (is("nav_profile_step_m")) c.nav.profile_step_m = v();
  else if (is("elevation_profile")) c.elevation.profile.enabled = on();
  else if (is("elevation_profile_bin_m")) c.elevation.profile.bin_m = v();
  else if (is("elevation_profile_sample_sigma")) c.elevation.profile.sample_sigma = v();
  else if (is("elevation_profile_age_sigma_per_hour")) c.elevation.profile.age_sigma_per_hour = v();
  else if (is("elevation_profile_distance_sigma_per_km")) c.elevation.profile.distance_sigma_per_km = v();
  else if (is("elevation_profile_max_age_s")) c.elevation.profile.max_age_s = v();
  else if (is("elevation_profile_max_distance_m")) c.elevation.profile.max_distance_m = v();
  else if (is("elevation_profile_min_span_m")) c.elevation.profile.min_span_m = v();
  else if (is("elevation_profile_extrapolation_sigma_per_s")) c.elevation.profile.extrapolation_sigma_per_s = v();
  else if (is("elevation_ray_arcs")) c.elevation.ray_arcs = on();
  else if (is("elevation_sector_from_rays")) c.elevation.sector_from_rays = on();
  else if (is("elevation_sector_rays_bounces")) c.elevation.sector_rays_bounces = v();
  else if (is("elevation_sector_rays_margin_deg")) c.elevation.sector_rays_margin = v() * kDeg;
  else if (is("elevation_sector_rays_max_depth")) c.elevation.sector_rays_max_depth = v();
  else if (is("elevation_multipath_sigma")) c.elevation.multipath_sigma = on();
  else if (is("elevation_multipath_sigma_bounces")) c.elevation.multipath_sigma_bounces = v();
  else if (is("elevation_multipath_sigma_max_depth")) c.elevation.multipath_sigma_max_depth = v();
  else if (is("elevation_profile_ensemble")) c.elevation.profile.ensemble = on();
  else if (is("elevation_profile_ensemble_sigma_members")) c.elevation.profile.ensemble_sigma_members = on();
  else if (is("elevation_profile_file")) c.elevation.profile.file = value;
  else if (is("elevation_profile_cast")) c.elevation.profile.cast = value;
  else if (is("elevation_profile_supplied_sigma")) c.elevation.profile.supplied_sigma = v();
  else if (is("elevation_profile_family")) c.elevation.profile.family = value;
  else if (is("elevation_profile_family_gradient")) c.elevation.profile.family_gradient = v();
  else if (is("elevation_profile_layer_depth")) c.elevation.profile.layer_depth = v();
  else if (is("elevation_profile_thermocline_gradient")) c.elevation.profile.thermocline_gradient = v();
  else if (is("elevation_profile_thermocline_thickness")) c.elevation.profile.thermocline_thickness = v();
  else if (is("elevation_profile_family_sigma")) c.elevation.profile.family_sigma = v();
  else if (is("elevation_profile_ladder")) {
    c.elevation.profile.ladder.clear();
    for (const std::string& rung : list_items(value)) {
      if (rung.empty()) refuse(key, value, "rung names separated by commas");
      c.elevation.profile.ladder.push_back(rung);
    }
  }
  // Comma-separated lists.
  else if (is("elevation_profile_ensemble_gradients")) c.elevation.profile.ensemble_gradients = option_numbers(key, value);
  else if (is("elevation_profile_ensemble_extents")) c.elevation.profile.ensemble_extents = option_numbers(key, value);
  else if (is("elevation_ssp_depth")) c.elevation.ssp_depth = option_numbers(key, value);
  else if (is("elevation_ssp_speed")) c.elevation.ssp_speed = option_numbers(key, value);
  else if (is("dead_channels")) c.dead_channels = option_integers(key, value);
  // auto (as gpusonar.live): CUDA when the build has it and a device is present, else the host. Without the key: the host.
  else if (is("backend")) c.backend = p.sbl.backend = parse_backend(value);
  else if (is("sbl_kernels")) c.sbl.kernels = p.sbl.kernels = on();
  else if (is("tracker_range_sigma")) c.tracker.range_sigma = v();
  else if (is("tracker_bearing_sigma_deg")) c.tracker.bearing_sigma = v() * kDeg;
  else if (is("tracker_confirm_hits")) c.tracker.confirm_hits = i();
  else if (is("tracker_max_misses")) c.tracker.max_misses = i();
  else if (is("tracker_ping_interval_s")) {
    // sonar-4bs: given, seconds above 0 (leave the key out for off).
    const double s = v();
    if (!(std::isfinite(s) && s > 0.0)) throw std::invalid_argument("tracker_ping_interval_s: seconds above 0, got '" + value + "' (leave the key out for off)");
    c.tracker_ping_interval_s = s;
  }
  else if (is("classify_min_speed")) c.classify.min_speed = c.classify.static_speed = v();
  else if (is("classify_static_speed")) c.classify.static_speed = v();   // after classify_min_speed, which sets both
  else if (is("classify_max_fit_rms")) c.classify.max_fit_rms = v();
  else if (is("classify_max_fit_rms_bearing_deg")) {
    const double b = v();
    if (!(std::isfinite(b) && b >= 0.0)) throw std::invalid_argument("classify_max_fit_rms_bearing_deg: a finite angle, 0 (off) or more, got '" + value + "' (leave the key out to derive it)");
    c.classify.max_fit_rms_bearing = b * kDeg;
  }
  else if (is("classify_outlier_alpha")) {
    const double a = v();
    if (!(a >= 0.0 && a < 1.0)) throw std::invalid_argument("classify_outlier_alpha: from 0 (off) to below 1, got '" + value + "'");
    c.classify.outlier_alpha = a;
  }
  else if (is("classify_min_mover_cells")) {
    // Off unless given (the field's default is below 0); given, it must be a number of cells, zero or more.
    const double m = v();
    if (!(m >= 0.0)) throw std::invalid_argument("classify_min_mover_cells: zero or more cells, got '" + value + "' (leave the key out for off)");
    c.classify.min_mover_cells = m;
  } else if (is("classify_row_snr_db")) {
    throw std::invalid_argument("classify_row_snr_db: the live chain never runs the row (corridor) test, so this rule would do nothing there; it applies only to classify_tracks with linear (the gallery)");
  }
  else if (is("tracker_initial_speed_sigma")) c.tracker.initial_speed_sigma = v();
  else if (is("tracker_process_noise")) c.tracker.process_noise = v();
  else if (is("multipath_max_excess_range")) c.classify.multipath_max_excess_range = v();
  else if (is("multipath_bearing_deg")) c.classify.multipath_bearing = v() * kDeg;
  else if (is("multipath_velocity")) c.classify.multipath_velocity = v();
  else if (is("window")) c.hann = option_choice(key, value, {"hann", "rect"}) == "hann";
  else if (is("multipath")) c.classify.multipath = on();
  else if (is("sync")) c.sync = on();
  else if (is("sync_f_start")) c.reference.f_start = v();
  else if (is("sync_f_stop")) c.reference.f_stop = v();
  else if (is("sync_duration")) c.reference.duration = v();
  else if (is("sync_lead")) c.reference.lead = v();
  else if (is("sync_tail")) c.reference.tail = v();
  else if (is("sync_dedicated")) c.reference.dedicated = on();
  else if (is("f_low")) p.f_low = v();
  else if (is("f_high")) p.f_high = v();
  else if (is("n_fft")) p.n_fft = i();
  else if (is("hop")) p.hop = i();
  else if (is("n_average")) p.n_average = i();
  else if (is("passive_method")) p.use_sbl = option_choice(key, value, {"sbl", "conventional"}) == "sbl";
  else if (set_tonal_option(p.tonal, key, value)) {}                 // tonal...: Goertzel lines and DEMON (tonal.hpp)
  else if (set_multipath_option(c.multipath_depth, key, value)) {}   // multipath_depth...: depth from multipath timing (multipath_depth.hpp)
  else if (set_coverage_option(c.coverage, key, value)) {}           // coverage...: the coverage picture (coverage.hpp)
  else {
    // The stages' own keys (tonal_, multipath_depth_, coverage_) are not among those named.
    const std::string nearest = nearest_key(key, keys);
    throw std::invalid_argument("unknown option " + key + (nearest.empty() ? std::string() : "; did you mean " + nearest + "?"));
  }
}

void set_pair(ChainOptions& c, PassiveOptions& p, const std::string& pair) {
  const auto eq = pair.find('=');
  if (eq == std::string::npos) throw std::invalid_argument("expected key=value, got " + pair);
  set_option(c, p, pair.substr(0, eq), pair.substr(eq + 1));
}


void read_options(ChainOptions& c, PassiveOptions& p, std::istream& in) {
  std::string line;
  while (std::getline(in, line)) {
    line = line.substr(0, line.find('#'));
    line.erase(0, line.find_first_not_of(" \t"));
    line.erase(line.find_last_not_of(" \t\r") + 1);
    if (!line.empty()) set_pair(c, p, line);
  }
}

void load_options(ChainOptions& c, PassiveOptions& p, const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::invalid_argument("cannot read options file " + path);
  read_options(c, p, f);
}

namespace {
// A JSON object written member by member, in the units of the key=value options.
class Json {
 public:
  Json() { s_ << std::setprecision(17) << "{"; }
  Json& num(const char* key, double v) {
    head(key);
    if (std::isfinite(v)) s_ << v; else s_ << "null";
    return *this;
  }
  Json& integer(const char* key, std::int64_t v) {
    head(key);
    s_ << v;
    return *this;
  }
  Json& flag(const char* key, bool v) {
    head(key);
    s_ << (v ? "true" : "false");
    return *this;
  }
  Json& text(const char* key, const std::string& v) {
    head(key);
    quoted(v);
    return *this;
  }
  template <class T>
  Json& list(const char* key, const std::vector<T>& v) {
    head(key);
    s_ << "[";
    for (std::size_t k = 0; k < v.size(); ++k) {
      s_ << (k ? ", " : "");
      if constexpr (std::is_same_v<T, std::string>) quoted(v[k]);
      else if (std::isfinite(static_cast<double>(v[k]))) s_ << v[k];
      else s_ << "null";
    }
    s_ << "]";
    return *this;
  }
  std::string str() { return s_.str() + "}"; }

 private:
  void head(const char* key) {
    s_ << (first_ ? "" : ", ") << "\"" << key << "\": ";
    first_ = false;
  }
  void quoted(const std::string& v) {
    s_ << "\"";
    for (char ch : v) {
      if (ch == '"' || ch == '\\') s_ << '\\';
      s_ << ch;
    }
    s_ << "\"";
  }
  std::ostringstream s_;
  bool first_ = true;
};
}  // namespace

std::string options_json(const ChainOptions& c) {
  Json j;
  j.num("bearing_first_deg", c.bearing_first_deg).num("bearing_last_deg", c.bearing_last_deg).num("bearing_step_deg", c.bearing_step_deg);
  j.text("window", c.hann ? "hann" : "rect").text("look", c.starboard ? "starboard" : "port").text("beamformer", c.beamformer);
  j.integer("mvdr_subbands", c.mvdr.n_subbands).num("mvdr_loading", c.mvdr.loading);
  j.num("range_cell_m", c.range_cell_m).num("cfar_pfa", c.cfar_pfa).num("cfar_guard_m", c.cfar_guard_m).num("cfar_train_m", c.cfar_train_m);
  j.flag("cell_peak", c.cell_peak).flag("cell_multiscale", c.cell_multiscale);
  const DetectorOptions& d = c.detector;
  j.num("cfar_false_alarms_per_ping", d.false_alarms_per_ping).list("cfar_integrate_cells", d.integrate_cells).num("cfar_merge_m", d.merge_m)
      .num("cfar_sidelobe_margin_db", d.sidelobe_margin_db).num("cfar_sidelobe_floor_db", d.sidelobe_floor_db).num("cfar_search_factor", d.search_factor)
      .text("cfar_range_estimate", range_estimate_name(chain_range_estimate(c)))   // the value the chain uses (D-158)
      .text("range_doppler", range_doppler_name(c.range_doppler));
  const DopplerOptions& dop = c.doppler;
  j.text("doppler_mode", doppler_mode_name(dop.mode)).num("doppler_max_range_rate_m_s", dop.max_range_rate).num("doppler_step_m_s", dop.step)
      .integer("doppler_hypotheses", dop.hypotheses).num("doppler_unknown_ownship_m_s", dop.unknown_ownship)
      .num("doppler_threshold_db", dop.threshold_db).flag("doppler_report", dop.report);
  j.num("min_snr_db", c.min_snr_db).text("doa", c.doa).integer("gate_fft", c.gate_fft).num("sbl_grid_step_deg", c.sbl_grid_step_deg)
      .integer("sbl_max_gates", c.sbl_max_gates).num("sbl_min_contrast_db", c.sbl_min_contrast_db).integer("sbl_max_iterations", c.sbl.max_iterations)
      .num("sbl_tolerance", c.sbl.tolerance).integer("sbl_max_sources", c.sbl.max_sources).num("sbl_dynamic_range_db", c.sbl.dynamic_range_db)
      .num("sbl_min_snr_db", c.sbl.min_snr_db).flag("sbl_kernels", c.sbl.kernels);
  const TrackerOptions& t = c.tracker;
  j.num("tracker_process_noise", t.process_noise).num("tracker_range_sigma", t.range_sigma).num("tracker_bearing_sigma_deg", t.bearing_sigma / kDeg)
      .integer("tracker_confirm_hits", t.confirm_hits).integer("tracker_max_misses", t.max_misses).num("tracker_initial_speed_sigma", t.initial_speed_sigma)
      .text("tracker_association", t.association == Association::InOrder ? "in_order" : "likelihood")
      .num("tracker_ping_interval_s", c.tracker_ping_interval_s);
  const ClassifyOptions& k = c.classify;
  j.num("classify_min_speed", k.min_speed).num("classify_static_speed", k.static_speed).num("classify_max_fit_rms", k.max_fit_rms)
      .num("classify_min_mover_cells", k.min_mover_cells).num("classify_outlier_alpha", k.outlier_alpha)
      .num("classify_max_fit_rms_bearing_deg", k.max_fit_rms_bearing / kDeg)
      .flag("multipath", k.multipath).num("multipath_max_excess_range", k.multipath_max_excess_range)
      .num("multipath_bearing_deg", k.multipath_bearing / kDeg).num("multipath_velocity", k.multipath_velocity);
  j.flag("sync", c.sync);
  const Reference& r = c.reference;
  j.num("sync_f_start", r.f_start).num("sync_f_stop", r.f_stop).num("sync_duration", r.duration).num("sync_lead", r.lead).num("sync_tail", r.tail)
      .flag("sync_dedicated", r.dedicated).num("sync_search_time", r.search_time).num("sync_min_quality", r.min_quality)
      .num("sync_clip_level", r.clip_level).num("sync_tolerance_us", r.tolerance_us);
  const ElevationOptions& e = c.elevation;
  j.flag("elevation", e.enabled).num("elevation_first_deg", e.first_deg).num("elevation_last_deg", e.last_deg).num("elevation_step_deg", e.step_deg)
      .flag("elevation_sector_from_depth", e.sector_from_depth).num("elevation_sector_bounces", e.sector_bounces)
      .num("elevation_sector_margin_deg", e.sector_margin / kDeg).num("elevation_range_window_m", e.range_window_m)
      .integer("elevation_max_detections", e.max_detections).flag("elevation_first_arrival", e.first_arrival)
      .num("elevation_first_arrival_db", e.first_arrival_db).num("elevation_first_arrival_snr_db", e.first_arrival_snr_db)
      .num("elevation_min_quality", e.min_quality).num("elevation_ambiguity_db", e.ambiguity_db).num("elevation_ambiguity_sigma", e.ambiguity_sigma)
      .num("elevation_alternatives_db", e.alternatives_db).flag("elevation_correct_bearing", e.correct_bearing).flag("elevation_depth", e.depth)
      .flag("elevation_fold", e.fold).flag("elevation_classify", e.classify).num("elevation_classify_sigma", e.classify_sigma)
      .num("elevation_water_depth", e.water_depth).num("elevation_water_depth_sigma", e.water_depth_sigma)
      .flag("elevation_attitude_arrival", e.attitude_arrival).num("elevation_attitude_rate_sigma_deg_s", e.attitude_rate_sigma / kDeg)
      .flag("elevation_use_gradient", e.use_gradient).num("elevation_gradient", e.gradient).num("elevation_gradient_sigma", e.gradient_sigma)
      .num("elevation_surface_z", e.surface_z).list("elevation_ssp_depth", e.ssp_depth).list("elevation_ssp_speed", e.ssp_speed)
      .num("elevation_roll_sigma_deg", e.roll_sigma / kDeg).num("elevation_row_tolerance_m", e.row_tolerance_m)
      .integer("elevation_min_row_elements", e.min_row_elements).flag("elevation_sector_from_rays", e.sector_from_rays)
      .num("elevation_sector_rays_bounces", e.sector_rays_bounces).num("elevation_sector_rays_margin_deg", e.sector_rays_margin / kDeg)
      .num("elevation_sector_rays_max_depth", e.sector_rays_max_depth).flag("elevation_ray_arcs", e.ray_arcs)
      .flag("elevation_multipath_sigma", e.multipath_sigma).num("elevation_multipath_sigma_bounces", e.multipath_sigma_bounces)
      .num("elevation_multipath_sigma_max_depth", e.multipath_sigma_max_depth).num("elevation_depth_process_noise", t.depth_process_noise);
  const ProfileOptions& f = e.profile;
  j.flag("elevation_profile", f.enabled).list("elevation_profile_ladder", f.ladder).num("elevation_profile_bin_m", f.bin_m)
      .num("elevation_profile_sample_sigma", f.sample_sigma).num("elevation_profile_age_sigma_per_hour", f.age_sigma_per_hour)
      .num("elevation_profile_distance_sigma_per_km", f.distance_sigma_per_km).num("elevation_profile_max_age_s", f.max_age_s)
      .num("elevation_profile_max_distance_m", f.max_distance_m).num("elevation_profile_min_span_m", f.min_span_m)
      .num("elevation_profile_extrapolation_sigma_per_s", f.extrapolation_sigma_per_s).text("elevation_profile_file", f.file)
      .text("elevation_profile_cast", f.cast).num("elevation_profile_supplied_sigma", f.supplied_sigma).text("elevation_profile_family", f.family)
      .num("elevation_profile_family_gradient", f.family_gradient).num("elevation_profile_layer_depth", f.layer_depth)
      .num("elevation_profile_thermocline_gradient", f.thermocline_gradient).num("elevation_profile_thermocline_thickness", f.thermocline_thickness)
      .num("elevation_profile_family_sigma", f.family_sigma).flag("elevation_profile_ensemble", f.ensemble)
      .list("elevation_profile_ensemble_gradients", f.ensemble_gradients).list("elevation_profile_ensemble_extents", f.ensemble_extents)
      .flag("elevation_profile_ensemble_sigma_members", f.ensemble_sigma_members);
  j.list("dead_channels", c.dead_channels).num("sound_speed", c.sound_speed);
  j.flag("blanking", c.blanking.enabled).num("blanking_recovery_s", c.blanking.recovery_s).num("blanking_taper_s", c.blanking.taper_s)
      .text("blanking_mode", c.blanking.zero ? "zero" : "ignore");
  const DecimateOptions& m = c.decimate;
  j.flag("decimate", m.enabled).flag("decimate_streaming", c.decimate_streaming).num("decimate_f_low", m.f_low).num("decimate_f_high", m.f_high)
      .num("decimate_band_margin", m.band_margin).num("decimate_output_rate", m.output_rate).num("decimate_oversample", m.oversample)
      .num("decimate_ripple_db", m.ripple_db).num("decimate_stopband_db", m.stopband_db).integer("decimate_taps", m.taps)
      .num("decimate_blank_until_s", m.blank_until_s).num("decimate_blank_taper_s", m.blank_taper_s);
  const NavOptions& n = c.nav;
  j.flag("nav_water_depth_from_altitude", n.water_depth.from_altitude).num("nav_dvl_offset_m", n.water_depth.dvl_offset_m)
      .num("nav_altitude_sigma_m", n.water_depth.altitude_sigma_m).num("nav_depth_sigma_m", n.water_depth.depth_sigma_m)
      .num("nav_hold_s", n.water_depth.hold_s).num("nav_hold_sigma_m_per_s", n.water_depth.hold_sigma_m_per_s)
      .num("nav_hold_sigma_m_per_m", n.water_depth.hold_sigma_m_per_m).num("nav_max_altitude_m", n.water_depth.max_altitude_m)
      .num("nav_keep_s", n.keep_s).num("nav_surface_z", n.surface_z).num("nav_profile_step_m", n.profile_step_m).num("nav_history_margin_s", n.history_margin_s)
      .list("nav_lever_arm_m", std::vector<double>{n.lever_arm.x, n.lever_arm.y, n.lever_arm.z})
      .list("nav_mount_rpy_deg", std::vector<double>{n.mount_rpy.x / kDeg, n.mount_rpy.y / kDeg, n.mount_rpy.z / kDeg});
  const StreamOptions& st = c.stream;
  j.integer("stream_sample_bytes", st.sample_bytes).integer("stream_buffer_bytes", st.buffer_bytes).num("stream_queue_mb", st.queue_mb)
      .num("idle_seconds", st.idle_seconds).text("bind", st.bind).integer("report_udp_max_bytes", st.report_max_bytes)
      .flag("stream_boards", st.boards).text("stream_board_channels", st.board_channels).integer("stream_board_reference", st.board_reference)
      .text("stream_board_missing", st.board_zero ? "zero" : "drop").num("stream_board_wait_s", st.board_wait_s)
      .num("stream_clock_ppm", st.clock_ppm).text("stream_board_ppm", st.board_ppm).flag("stream_clock_estimate", st.clock_estimate)
      .flag("stream_clock_discipline", st.clock_discipline).num("stream_clock_bin_s", st.clock_bin_s).num("stream_clock_window_s", st.clock_window_s)
      .num("stream_clock_min_span_s", st.clock_min_span_s).num("stream_clock_max_sigma_ppm", st.clock_max_sigma_ppm);
  const MultipathOptions& q = c.multipath_depth;
  j.flag("multipath_depth", q.enabled).num("multipath_depth_max_depth", q.max_depth).num("multipath_depth_min_delay_bandwidth", q.min_delay_bandwidth)
      .num("multipath_depth_window_s", q.window_s).num("multipath_depth_range_slack", q.range_slack).integer("multipath_depth_oversample", q.oversample)
      .integer("multipath_depth_fringe_points", q.fringe_points).integer("multipath_depth_noise_cells", q.noise_cells)
      .num("multipath_depth_min_gain", q.min_gain).num("multipath_depth_ambiguity_drop", q.ambiguity_drop)
      .integer("multipath_depth_max_detections", q.max_detections).num("multipath_depth_min_snr_db", q.min_snr_db)
      .num("multipath_depth_surface_coefficient", q.surface_coefficient).num("multipath_depth_surface_rms_height", q.surface_rms_height)
      .num("multipath_depth_surface_loss_db", q.surface_loss_db).num("multipath_depth_surface_min_amplitude", q.surface_min_amplitude)
      .list("multipath_depth_surface_loss_grazing", q.surface_loss_grazing).list("multipath_depth_surface_loss_table_db", q.surface_loss_table_db)
      .flag("multipath_depth_bottom", q.bottom).num("multipath_depth_bottom_speed", q.bottom_speed).num("multipath_depth_bottom_density", q.bottom_density)
      .num("multipath_depth_bottom_attenuation", q.bottom_attenuation)
      .num("multipath_depth_bottom_shear_speed", q.bottom_shear_speed).num("multipath_depth_bottom_shear_attenuation", q.bottom_shear_attenuation).integer("multipath_depth_bottom_order", q.bottom_order)
      .num("multipath_depth_bottom_grazing_limit_deg", q.bottom_grazing_limit_deg).num("multipath_depth_water_depth", q.water_depth)
      .num("multipath_depth_surface_z", q.surface_z).num("multipath_depth_own_depth_sigma", q.own_depth_sigma)
      .num("multipath_depth_model_sigma_fraction", q.model_sigma_fraction).num("multipath_depth_gradient", q.gradient)
      .num("multipath_depth_gradient_sigma", q.gradient_sigma).num("multipath_depth_track_gate", q.track_gate).num("multipath_depth_track_step", q.track_step)
      .num("multipath_depth_track_min_gain", q.track_min_gain).num("multipath_depth_track_memory_pings", q.track_memory_pings)
      .num("multipath_depth_consistency_sigma", q.consistency_sigma).flag("multipath_depth_rays", q.rays)
      .flag("multipath_depth_withhold_flagged", q.withhold_flagged).integer("multipath_depth_ray_fan", q.ray_fan)
      .integer("multipath_depth_ray_surface_order", q.ray_surface_order).num("multipath_depth_ray_min_amplitude", q.ray_min_amplitude)
      .num("multipath_depth_ray_sigma_fan_fraction", q.ray_sigma_fan_fraction);
  const CoverageOptions& v = c.coverage;
  std::string classes;
  {
    std::ostringstream s;
    s << std::setprecision(17);
    for (std::size_t i = 0; i < v.classes.size(); ++i) {
      s << (i ? ";" : "") << v.classes[i].name << ":" << v.classes[i].ts_db << ":" << v.classes[i].depth_lo << ":" << v.classes[i].depth_hi;
    }
    classes = s.str();
  }
  j.flag("coverage", v.enabled).text("coverage_classes", classes).num("coverage_source_level_db", v.source_level_db)
      .num("coverage_noise_spectrum_db", v.noise_spectrum_db).num("coverage_array_gain_db", v.array_gain_db)
      .num("coverage_absorption_db_per_km", v.absorption_db_per_km).num("coverage_pd", v.pd).list("coverage_pd_snr_db", v.pd_snr_db)
      .list("coverage_pd_value", v.pd_value).num("coverage_range_first", v.range_first).num("coverage_range_last", v.range_last)
      .integer("coverage_n_ranges", v.n_ranges).integer("coverage_n_depths", v.n_depths).integer("coverage_n_report", v.n_report)
      .flag("coverage_recommend", v.recommend).num("coverage_own_depth_min", v.own_depth_min).num("coverage_own_depth_max", v.own_depth_max)
      .num("coverage_bottom_clearance", v.bottom_clearance).integer("coverage_n_own", v.n_own).num("coverage_refresh_depth", v.refresh_depth)
      .num("coverage_refresh_water_depth", v.refresh_water_depth).num("coverage_refresh_speed", v.refresh_speed)
      .num("coverage_surface_coefficient", v.surface_coefficient).num("coverage_surface_rms_height", v.surface_rms_height)
      .num("coverage_surface_loss_db", v.surface_loss_db).list("coverage_surface_loss_grazing", v.surface_loss_grazing)
      .list("coverage_surface_loss_table_db", v.surface_loss_table_db).flag("coverage_bottom", v.bottom).num("coverage_bottom_speed", v.bottom_speed)
      .num("coverage_bottom_density", v.bottom_density).num("coverage_bottom_attenuation", v.bottom_attenuation)
      .num("coverage_bottom_shear_speed", v.bottom_shear_speed).num("coverage_bottom_shear_attenuation", v.bottom_shear_attenuation).num("coverage_water_depth", v.water_depth)
      .integer("coverage_ray_fan", v.ray_fan).integer("coverage_max_bounces", v.max_bounces).integer("coverage_max_arrivals", v.max_arrivals)
      .num("coverage_null_db", v.null_db).num("coverage_surface_z", v.surface_z).integer("coverage_reverb", v.reverb)
      .list("coverage_reverb_grazing", v.reverb_grazing).list("coverage_reverb_bottom_strength", v.reverb_bottom_strength)
      .list("coverage_reverb_surface_strength", v.reverb_surface_strength).num("coverage_reverb_beam_integral", v.reverb_beam_integral)
      .integer("coverage_reverb_rings", v.reverb_rings).integer("coverage_reverb_max_bounces", v.reverb_max_bounces)
      .text("coverage_reverb_note", v.reverb_note).flag("coverage_echo_coherent", v.echo_coherent).num("coverage_reverb_steer", v.reverb_steer)
      .num("coverage_reverb_converge_db", v.reverb_converge_db).integer("coverage_reverb_bounce_cap", v.reverb_bounce_cap)
      .num("coverage_reverb_rows_gain_db", v.reverb_rows_gain_db)
      .num("coverage_reverb_rows_gain_against_reverb_db", v.reverb_rows_gain_against_reverb_db);
  j.text("backend", c.backend == Backend::Cuda ? "cuda" : "host");
  return j.str();
}

}  // namespace sonar::live
