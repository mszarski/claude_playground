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
  if (key == "bearing_first_deg") c.bearing_first_deg = p.bearing_first_deg = v();
  else if (key == "bearing_last_deg") c.bearing_last_deg = p.bearing_last_deg = v();
  else if (key == "bearing_step_deg") c.bearing_step_deg = p.bearing_step_deg = v();
  else if (key == "range_cell_m") c.range_cell_m = v();
  else if (key == "cfar_pfa") c.cfar_pfa = v();
  else if (key == "cfar_guard_m") c.cfar_guard_m = v();
  else if (key == "cfar_train_m") c.cfar_train_m = v();
  else if (key == "min_snr_db") c.min_snr_db = v();
  else if (key == "cell_statistic") {
    if (value != "mean" && value != "peak" && value != "multiscale") throw std::invalid_argument("cell_statistic: mean, peak or multiscale");
    c.cell_peak = value == "peak";
    c.cell_multiscale = value == "multiscale";
  }
  else if (key == "cfar_false_alarms_per_ping") c.detector.false_alarms_per_ping = v();
  else if (key == "cfar_sidelobe_margin_db") c.detector.sidelobe_margin_db = v();
  else if (key == "cfar_sidelobe_floor_db") c.detector.sidelobe_floor_db = v();
  else if (key == "cfar_search_factor") c.detector.search_factor = v();
  else if (key == "cfar_merge_m") c.detector.merge_m = v();
  else if (key == "cfar_range_estimate") c.range_estimate = parse_range_estimate(value);   // centroid | first (D-157); unset: chain_range_estimate (D-158)
  // The two flags as gpusonar.live.cpp_chain_options names them, for a tool that passes its keys on as they are.
  else if (key == "cell_multiscale") {
    c.cell_multiscale = on();
    if (c.cell_multiscale) c.cell_peak = false;
  }
  else if (key == "cell_peak") {
    c.cell_peak = on();
    if (c.cell_peak) c.cell_multiscale = false;
  }
  else if (key == "cfar_integrate_cells") c.detector.integrate_cells = option_integers(key, value);   // range cells: 1,4,16 (or [1, 4, 16])
  else if (key == "beamformer") c.beamformer = option_choice(key, value, {"backprojection", "farfield", "mvdr"});
  else if (key == "tracker_association") c.tracker.association = parse_association(value);
  else if (key == "doa") c.doa = option_choice(key, value, {"map", "sbl"});
  else if (key == "mvdr_subbands") c.mvdr.n_subbands = i();
  else if (key == "mvdr_loading") c.mvdr.loading = v();
  else if (key == "gate_fft") c.gate_fft = i();
  else if (key == "sbl_grid_step_deg") c.sbl_grid_step_deg = v();
  else if (key == "sbl_max_iterations") c.sbl.max_iterations = p.sbl.max_iterations = static_cast<std::int32_t>(i());
  else if (key == "sbl_tolerance") c.sbl.tolerance = p.sbl.tolerance = v();
  else if (key == "sbl_max_gates") c.sbl_max_gates = i();
  else if (key == "sbl_min_contrast_db") c.sbl_min_contrast_db = v();
  else if (key == "sbl_max_sources") c.sbl.max_sources = p.sbl.max_sources = static_cast<std::int32_t>(i());
  else if (key == "sbl_min_snr_db") c.sbl.min_snr_db = p.sbl.min_snr_db = v();
  else if (key == "sbl_dynamic_range_db") c.sbl.dynamic_range_db = p.sbl.dynamic_range_db = v();
  else if (key == "look") c.starboard = p.starboard = option_choice(key, value, {"starboard", "port"}) == "starboard";
  else if (key == "elevation") c.elevation.enabled = p.elevation.enabled = on();
  else if (key == "elevation_first_deg") c.elevation.first_deg = p.elevation.first_deg = v();
  else if (key == "elevation_last_deg") c.elevation.last_deg = p.elevation.last_deg = v();
  else if (key == "elevation_step_deg") c.elevation.step_deg = p.elevation.step_deg = v();
  else if (key == "elevation_range_window_m") c.elevation.range_window_m = v();
  else if (key == "elevation_max_detections") c.elevation.max_detections = i();
  else if (key == "elevation_first_arrival") c.elevation.first_arrival = on();
  else if (key == "elevation_first_arrival_db") c.elevation.first_arrival_db = v();
  else if (key == "elevation_first_arrival_snr_db") c.elevation.first_arrival_snr_db = v();
  else if (key == "sound_speed") c.sound_speed = v();
  else if (key == "elevation_ambiguity_sigma") c.elevation.ambiguity_sigma = v();
  else if (key == "elevation_alternatives_db") c.elevation.alternatives_db = v();
  else if (key == "elevation_sector_from_depth") c.elevation.sector_from_depth = on();
  else if (key == "elevation_sector_bounces") c.elevation.sector_bounces = v();
  else if (key == "elevation_sector_margin_deg") c.elevation.sector_margin = v() * kDeg;
  else if (key == "elevation_water_depth_sigma") c.elevation.water_depth_sigma = v();
  else if (key == "elevation_attitude_arrival") c.elevation.attitude_arrival = on();
  else if (key == "elevation_attitude_rate_sigma_deg_s") c.elevation.attitude_rate_sigma = v() * kDeg;
  else if (key == "elevation_use_gradient") c.elevation.use_gradient = on();
  else if (key == "elevation_gradient") c.elevation.gradient = v();
  else if (key == "elevation_gradient_sigma") c.elevation.gradient_sigma = v();
  else if (key == "elevation_min_quality") c.elevation.min_quality = p.elevation.min_quality = v();
  else if (key == "elevation_ambiguity_db") c.elevation.ambiguity_db = v();
  else if (key == "elevation_correct_bearing") c.elevation.correct_bearing = on();
  else if (key == "elevation_depth") c.elevation.depth = on();
  else if (key == "elevation_fold") c.elevation.fold = on();
  else if (key == "elevation_classify") c.elevation.classify = on();
  else if (key == "elevation_classify_sigma") c.elevation.classify_sigma = v();
  else if (key == "elevation_water_depth") c.elevation.water_depth = v();
  else if (key == "elevation_surface_z") c.elevation.surface_z = v();
  else if (key == "elevation_roll_sigma_deg") c.elevation.roll_sigma = v() * kDeg;
  else if (key == "elevation_depth_process_noise") c.tracker.depth_process_noise = v();
  else if (key == "elevation_row_tolerance_m") c.elevation.row_tolerance_m = p.elevation.row_tolerance_m = v();
  else if (key == "blanking") c.blanking.enabled = on();
  else if (key == "blanking_recovery_s") c.blanking.recovery_s = v();
  else if (key == "blanking_taper_s") c.blanking.taper_s = v();
  else if (key == "blanking_mode") c.blanking.zero = option_choice(key, value, {"zero", "ignore"}) == "zero";
  else if (key == "sync_min_quality") c.reference.min_quality = v();
  else if (key == "sync_search_time") c.reference.search_time = v();
  else if (key == "sync_tolerance_us") c.reference.tolerance_us = v();
  else if (key == "sync_clip_level") c.reference.clip_level = v();
  else if (key == "elevation_min_row_elements") c.elevation.min_row_elements = p.elevation.min_row_elements = i();
  else if (key == "nav_surface_z") c.nav.surface_z = v();   // z of the sea surface in the navigation's frame: depth = z - surface_z
  else if (key == "nav_history_margin_s") c.nav.history_margin_s = v();
  // The program around the chain: the board stream, its socket and queue, the report datagrams.
  else if (key == "stream_sample_bytes") c.stream.sample_bytes = static_cast<int>(i());
  else if (key == "stream_buffer_bytes") c.stream.buffer_bytes = i();
  else if (key == "stream_queue_mb") c.stream.queue_mb = v();
  else if (key == "idle_seconds") c.stream.idle_seconds = v();
  else if (key == "bind") c.stream.bind = value;
  else if (key == "report_udp_max_bytes") c.stream.report_max_bytes = i();
  else if (key == "decimate") c.decimate.enabled = on();
  else if (key == "decimate_streaming") c.decimate_streaming = on();
  else if (key == "decimate_f_low") c.decimate.f_low = v();
  else if (key == "decimate_f_high") c.decimate.f_high = v();
  else if (key == "decimate_band_margin") c.decimate.band_margin = v();
  else if (key == "decimate_output_rate") c.decimate.output_rate = v();
  else if (key == "decimate_oversample") c.decimate.oversample = v();
  else if (key == "decimate_ripple_db") c.decimate.ripple_db = v();
  else if (key == "decimate_stopband_db") c.decimate.stopband_db = v();
  else if (key == "decimate_taps") c.decimate.taps = i();
  else if (key == "decimate_blank_until_s") c.decimate.blank_until_s = v();
  else if (key == "decimate_blank_taper_s") c.decimate.blank_taper_s = v();
  else if (key == "nav_water_depth_from_altitude") c.nav.water_depth.from_altitude = on();
  else if (key == "nav_dvl_offset_m") c.nav.water_depth.dvl_offset_m = v();
  else if (key == "nav_altitude_sigma_m") c.nav.water_depth.altitude_sigma_m = v();
  else if (key == "nav_depth_sigma_m") c.nav.water_depth.depth_sigma_m = v();
  else if (key == "nav_hold_s") c.nav.water_depth.hold_s = v();
  else if (key == "nav_hold_sigma_m_per_s") c.nav.water_depth.hold_sigma_m_per_s = v();
  else if (key == "nav_hold_sigma_m_per_m") c.nav.water_depth.hold_sigma_m_per_m = v();
  else if (key == "nav_max_altitude_m") c.nav.water_depth.max_altitude_m = v();
  else if (key == "nav_keep_s") c.nav.keep_s = v();
  else if (key == "nav_profile_step_m") c.nav.profile_step_m = v();
  else if (key == "elevation_profile") c.elevation.profile.enabled = on();
  else if (key == "elevation_profile_bin_m") c.elevation.profile.bin_m = v();
  else if (key == "elevation_profile_sample_sigma") c.elevation.profile.sample_sigma = v();
  else if (key == "elevation_profile_age_sigma_per_hour") c.elevation.profile.age_sigma_per_hour = v();
  else if (key == "elevation_profile_distance_sigma_per_km") c.elevation.profile.distance_sigma_per_km = v();
  else if (key == "elevation_profile_max_age_s") c.elevation.profile.max_age_s = v();
  else if (key == "elevation_profile_max_distance_m") c.elevation.profile.max_distance_m = v();
  else if (key == "elevation_profile_min_span_m") c.elevation.profile.min_span_m = v();
  else if (key == "elevation_profile_extrapolation_sigma_per_s") c.elevation.profile.extrapolation_sigma_per_s = v();
  else if (key == "elevation_ray_arcs") c.elevation.ray_arcs = on();
  else if (key == "elevation_sector_from_rays") c.elevation.sector_from_rays = on();
  else if (key == "elevation_sector_rays_bounces") c.elevation.sector_rays_bounces = v();
  else if (key == "elevation_sector_rays_margin_deg") c.elevation.sector_rays_margin = v() * kDeg;
  else if (key == "elevation_sector_rays_max_depth") c.elevation.sector_rays_max_depth = v();
  else if (key == "elevation_multipath_sigma") c.elevation.multipath_sigma = on();
  else if (key == "elevation_multipath_sigma_bounces") c.elevation.multipath_sigma_bounces = v();
  else if (key == "elevation_multipath_sigma_max_depth") c.elevation.multipath_sigma_max_depth = v();
  else if (key == "elevation_profile_ensemble") c.elevation.profile.ensemble = on();
  else if (key == "elevation_profile_ensemble_sigma_members") c.elevation.profile.ensemble_sigma_members = on();
  else if (key == "elevation_profile_file") c.elevation.profile.file = value;
  else if (key == "elevation_profile_cast") c.elevation.profile.cast = value;
  else if (key == "elevation_profile_supplied_sigma") c.elevation.profile.supplied_sigma = v();
  else if (key == "elevation_profile_family") c.elevation.profile.family = value;
  else if (key == "elevation_profile_family_gradient") c.elevation.profile.family_gradient = v();
  else if (key == "elevation_profile_layer_depth") c.elevation.profile.layer_depth = v();
  else if (key == "elevation_profile_thermocline_gradient") c.elevation.profile.thermocline_gradient = v();
  else if (key == "elevation_profile_thermocline_thickness") c.elevation.profile.thermocline_thickness = v();
  else if (key == "elevation_profile_family_sigma") c.elevation.profile.family_sigma = v();
  else if (key == "elevation_profile_ladder") {
    c.elevation.profile.ladder.clear();
    for (const std::string& rung : list_items(value)) {
      if (rung.empty()) refuse(key, value, "rung names separated by commas");
      c.elevation.profile.ladder.push_back(rung);
    }
  }
  // Comma-separated lists.
  else if (key == "elevation_profile_ensemble_gradients") c.elevation.profile.ensemble_gradients = option_numbers(key, value);
  else if (key == "elevation_profile_ensemble_extents") c.elevation.profile.ensemble_extents = option_numbers(key, value);
  else if (key == "elevation_ssp_depth") c.elevation.ssp_depth = option_numbers(key, value);
  else if (key == "elevation_ssp_speed") c.elevation.ssp_speed = option_numbers(key, value);
  else if (key == "dead_channels") c.dead_channels = option_integers(key, value);
  // auto (as gpusonar.live): CUDA when the build has it and a device is present, else the host. Without the key: the host.
  else if (key == "backend") c.backend = p.sbl.backend = parse_backend(value);
  else if (key == "sbl_kernels") c.sbl.kernels = p.sbl.kernels = on();
  else if (key == "tracker_range_sigma") c.tracker.range_sigma = v();
  else if (key == "tracker_bearing_sigma_deg") c.tracker.bearing_sigma = v() * kDeg;
  else if (key == "tracker_confirm_hits") c.tracker.confirm_hits = i();
  else if (key == "tracker_max_misses") c.tracker.max_misses = i();
  else if (key == "classify_min_speed") c.classify.min_speed = c.classify.static_speed = v();
  else if (key == "classify_static_speed") c.classify.static_speed = v();   // after classify_min_speed, which sets both
  else if (key == "classify_max_fit_rms") c.classify.max_fit_rms = v();
  else if (key == "classify_max_fit_rms_bearing_deg") {
    const double b = v();
    if (!(std::isfinite(b) && b >= 0.0)) throw std::invalid_argument("classify_max_fit_rms_bearing_deg: a finite angle, 0 (off) or more, got '" + value + "' (leave the key out to derive it)");
    c.classify.max_fit_rms_bearing = b * kDeg;
  }
  else if (key == "classify_outlier_alpha") {
    const double a = v();
    if (!(a >= 0.0 && a < 1.0)) throw std::invalid_argument("classify_outlier_alpha: from 0 (off) to below 1, got '" + value + "'");
    c.classify.outlier_alpha = a;
  }
  else if (key == "classify_min_mover_cells") {
    // Off unless given (the field's default is below 0); given, it must be a number of cells, zero or more.
    const double m = v();
    if (!(m >= 0.0)) throw std::invalid_argument("classify_min_mover_cells: zero or more cells, got '" + value + "' (leave the key out for off)");
    c.classify.min_mover_cells = m;
  } else if (key == "classify_row_snr_db") {
    throw std::invalid_argument("classify_row_snr_db: the live chain never runs the row (corridor) test, so this rule would do nothing there; it applies only to classify_tracks with linear (the gallery)");
  }
  else if (key == "tracker_initial_speed_sigma") c.tracker.initial_speed_sigma = v();
  else if (key == "tracker_process_noise") c.tracker.process_noise = v();
  else if (key == "multipath_max_excess_range") c.classify.multipath_max_excess_range = v();
  else if (key == "multipath_bearing_deg") c.classify.multipath_bearing = v() * kDeg;
  else if (key == "multipath_velocity") c.classify.multipath_velocity = v();
  else if (key == "window") c.hann = option_choice(key, value, {"hann", "rect"}) == "hann";
  else if (key == "multipath") c.classify.multipath = on();
  else if (key == "sync") c.sync = on();
  else if (key == "sync_f_start") c.reference.f_start = v();
  else if (key == "sync_f_stop") c.reference.f_stop = v();
  else if (key == "sync_duration") c.reference.duration = v();
  else if (key == "sync_lead") c.reference.lead = v();
  else if (key == "sync_tail") c.reference.tail = v();
  else if (key == "sync_dedicated") c.reference.dedicated = on();
  else if (key == "f_low") p.f_low = v();
  else if (key == "f_high") p.f_high = v();
  else if (key == "n_fft") p.n_fft = i();
  else if (key == "hop") p.hop = i();
  else if (key == "n_average") p.n_average = i();
  else if (key == "passive_method") p.use_sbl = option_choice(key, value, {"sbl", "conventional"}) == "sbl";
  else if (set_tonal_option(p.tonal, key, value)) {}                 // tonal...: Goertzel lines and DEMON (tonal.hpp)
  else if (set_multipath_option(c.multipath_depth, key, value)) {}   // multipath_depth...: depth from multipath timing (multipath_depth.hpp)
  else if (set_coverage_option(c.coverage, key, value)) {}           // coverage...: the coverage picture (coverage.hpp)
  else throw std::invalid_argument("unknown option " + key);
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
      .text("cfar_range_estimate", range_estimate_name(chain_range_estimate(c)));   // the value the chain uses (D-158)
  j.num("min_snr_db", c.min_snr_db).text("doa", c.doa).integer("gate_fft", c.gate_fft).num("sbl_grid_step_deg", c.sbl_grid_step_deg)
      .integer("sbl_max_gates", c.sbl_max_gates).num("sbl_min_contrast_db", c.sbl_min_contrast_db).integer("sbl_max_iterations", c.sbl.max_iterations)
      .num("sbl_tolerance", c.sbl.tolerance).integer("sbl_max_sources", c.sbl.max_sources).num("sbl_dynamic_range_db", c.sbl.dynamic_range_db)
      .num("sbl_min_snr_db", c.sbl.min_snr_db).flag("sbl_kernels", c.sbl.kernels);
  const TrackerOptions& t = c.tracker;
  j.num("tracker_process_noise", t.process_noise).num("tracker_range_sigma", t.range_sigma).num("tracker_bearing_sigma_deg", t.bearing_sigma / kDeg)
      .integer("tracker_confirm_hits", t.confirm_hits).integer("tracker_max_misses", t.max_misses).num("tracker_initial_speed_sigma", t.initial_speed_sigma)
      .text("tracker_association", t.association == Association::InOrder ? "in_order" : "likelihood");
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
      .num("nav_keep_s", n.keep_s).num("nav_surface_z", n.surface_z).num("nav_profile_step_m", n.profile_step_m).num("nav_history_margin_s", n.history_margin_s);
  const StreamOptions& st = c.stream;
  j.integer("stream_sample_bytes", st.sample_bytes).integer("stream_buffer_bytes", st.buffer_bytes).num("stream_queue_mb", st.queue_mb)
      .num("idle_seconds", st.idle_seconds).text("bind", st.bind).integer("report_udp_max_bytes", st.report_max_bytes);
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
      .num("multipath_depth_bottom_attenuation", q.bottom_attenuation).integer("multipath_depth_bottom_order", q.bottom_order)
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
      .num("coverage_bottom_density", v.bottom_density).num("coverage_bottom_attenuation", v.bottom_attenuation).num("coverage_water_depth", v.water_depth)
      .integer("coverage_ray_fan", v.ray_fan).integer("coverage_max_bounces", v.max_bounces).integer("coverage_max_arrivals", v.max_arrivals)
      .num("coverage_null_db", v.null_db).num("coverage_surface_z", v.surface_z);
  j.text("backend", c.backend == Backend::Cuda ? "cuda" : "host");
  return j.str();
}

}  // namespace sonar::live
