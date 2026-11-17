#include "sonar/live/options.hpp"

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
  else if (key == "look") c.starboard = value != "port";
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
