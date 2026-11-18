#include "sonar/live/derive.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "sonar/live/chain.hpp"

namespace sonar::live {
namespace {

// The standing value unless the sonar's floor is above it. Written as a choice, not as std::max of
// two computed numbers, so that a sonar the standing value suits gets that value to the last bit.
double at_least(double standing, double floor, const char* key, std::vector<std::string>& scaled) {
  if (floor > standing) {
    scaled.emplace_back(key);
    return floor;
  }
  return standing;
}

void list(std::ostringstream& s, const char* name, const std::vector<std::string>& items) {
  s << ", \"" << name << "\": [";
  for (std::size_t i = 0; i < items.size(); ++i) s << (i ? ", " : "") << "\"" << items[i] << "\"";
  s << "]";
}

}  // namespace

double row_tolerance(const PingRecord& record, double given) {
  if (is_given(given) && given > 0.0) return given;
  const Waveform& w = record.transmit.waveform;
  const double top = std::max(w.f_start, w.f_stop), c = record.environment.sound_speed;
  if (!(top > 0.0) || !(c > 0.0)) return kStanding.row_tolerance_m;
  return std::min(kStanding.row_tolerance_m, kRowToleranceWavelengths * c / top);
}

SonarFacts sonar_facts(const PingRecord& record, double tolerance, std::int64_t min_row_elements) {
  SonarFacts f;
  const Waveform& w = record.transmit.waveform;
  const Samples& s = record.samples;
  f.sound_speed = record.environment.sound_speed;
  if (!(f.sound_speed > 0.0)) throw std::invalid_argument("derive: the record has no sound speed");
  f.f_low = std::min(w.f_start, w.f_stop);
  f.f_high = std::max(w.f_start, w.f_stop);
  f.pulse = w.duration;
  f.band = f.f_high - f.f_low > 0.0 ? f.f_high - f.f_low : (w.duration > 0.0 ? 1.0 / w.duration : 0.0);
  if (!(f.band > 0.0) || !(f.f_high > 0.0)) throw std::invalid_argument("derive: the record's waveform has neither a band nor a duration");
  f.range_resolution = f.sound_speed / (2.0 * f.band);
  f.wavelength = f.sound_speed / (0.5 * (f.f_low + f.f_high));
  f.wavelength_min = f.sound_speed / f.f_high;
  f.n_channels = static_cast<std::int64_t>(record.geometry.rx.size());
  f.sample_rate = s.sample_rate;
  f.window_m = s.sample_rate > 0.0 ? 0.5 * f.sound_speed * static_cast<double>(s.n_samples) / s.sample_rate : 0.0;
  if (f.n_channels < 2) return f;
  const auto rows = find_rows(record, row_tolerance(record, tolerance), min_row_elements);
  f.n_rows = static_cast<std::int64_t>(rows.size());
  // The first row: its elements along its own line.
  std::vector<std::int64_t> row;
  if (!rows.empty()) row = rows[0];
  else for (std::int64_t i = 0; i < f.n_channels; ++i) row.push_back(i);
  f.row_elements = static_cast<std::int64_t>(row.size());
  const auto& rx = record.geometry.rx;
  const Vec3d first = rx[static_cast<std::size_t>(row.front())].lever_arm, last = rx[static_cast<std::size_t>(row.back())].lever_arm;
  const double length = norm(last - first);
  if (length > 0.0 && row.size() > 1) {
    const Vec3d axis = (last - first) / length;
    double lo = 0.0, hi = 0.0;
    for (std::int64_t i : row) {
      const double u = dot(rx[static_cast<std::size_t>(i)].lever_arm - first, axis);
      lo = std::min(lo, u);
      hi = std::max(hi, u);
    }
    f.aperture = (hi - lo) * static_cast<double>(row.size()) / static_cast<double>(row.size() - 1);
    f.beamwidth = f.wavelength / f.aperture;
  }
  return f;
}

ChainDefaults resolve(ChainOptions& o, const PingRecord& record) {
  ChainDefaults d;
  auto take = [&](double& option, const char* key) {
    if (is_given(option)) {
      d.given.emplace_back(key);
      return true;
    }
    return false;
  };
  const bool tolerance_given = is_given(o.elevation.row_tolerance_m) && o.elevation.row_tolerance_m > 0.0;
  if (tolerance_given) d.given.emplace_back("elevation.row_tolerance_m");
  d.row_tolerance_m = row_tolerance(record, o.elevation.row_tolerance_m);
  if (!tolerance_given && d.row_tolerance_m < kStanding.row_tolerance_m) d.scaled.emplace_back("elevation.row_tolerance_m");
  o.elevation.row_tolerance_m = d.row_tolerance_m;
  d.facts = sonar_facts(record, d.row_tolerance_m, o.elevation.min_row_elements);
  const SonarFacts& f = d.facts;

  if (!take(o.range_cell_m, "range_cell_m")) {
    o.range_cell_m = at_least(kStanding.range_cell_m, kCellsPerResolution * f.range_resolution, "range_cell_m", d.scaled);
  }
  if (!take(o.cfar_guard_m, "cfar.guard_m")) {
    o.cfar_guard_m = at_least(kStanding.cfar_guard_m, kGuardCells * f.range_resolution, "cfar.guard_m", d.scaled);
  }
  if (!take(o.cfar_train_m, "cfar.train_m")) {
    o.cfar_train_m = at_least(kStanding.cfar_train_m, kTrainPerGuard * kGuardCells * f.range_resolution, "cfar.train_m", d.scaled);
  }
  if (!take(o.tracker.range_sigma, "tracker.range_sigma")) {
    o.tracker.range_sigma = at_least(kStanding.tracker_range_sigma, kCellsPerResolution * f.range_resolution, "tracker.range_sigma", d.scaled);
  }
  const double bearing_floor = f.beamwidth / kBeamwidthsPerSigma;      // 0 for an array the facts could not measure
  if (!take(o.tracker.bearing_sigma, "tracker.bearing_sigma_deg")) {
    o.tracker.bearing_sigma = at_least(kStanding.tracker_bearing_sigma, bearing_floor, "tracker.bearing_sigma_deg", d.scaled);
  }
  if (!take(o.classify.multipath_bearing, "classify.multipath.bearing_deg")) {
    o.classify.multipath_bearing =
        at_least(kStanding.multipath_bearing, kMultipathSigmas * bearing_floor, "classify.multipath.bearing_deg", d.scaled);
  }
  if (!take(o.classify.max_fit_rms, "classify.max_fit_rms")) {
    o.classify.max_fit_rms = at_least(kStanding.classify_max_fit_rms, kFitRmsSigmas * kCellsPerResolution * f.range_resolution,
                                      "classify.max_fit_rms", d.scaled);
  }
  d.range_cell_m = o.range_cell_m;
  d.cfar_guard_m = o.cfar_guard_m;
  d.cfar_train_m = o.cfar_train_m;
  d.tracker_range_sigma = o.tracker.range_sigma;
  d.tracker_bearing_sigma = o.tracker.bearing_sigma;
  d.multipath_bearing = o.classify.multipath_bearing;
  d.classify_max_fit_rms = o.classify.max_fit_rms;

  // What a new sonar's owner should look at: said once, never acted on.
  const double usable = f.window_m - 0.5 * f.sound_speed * f.pulse;
  if (usable > 0.0 && usable < 2.0 * (o.cfar_guard_m + o.cfar_train_m)) {
    std::ostringstream s;
    s << "the receive window covers " << usable << " m of whole echoes, less than the detector's guard and training on both sides ("
      << 2.0 * (o.cfar_guard_m + o.cfar_train_m) << " m): its background comes from fewer cells (cfar.guard_m, cfar.train_m)";
    d.warnings.push_back(s.str());
  }
  const double kDeg = 3.14159265358979323846 / 180.0;
  if (f.beamwidth > 0.0 && o.bearing_step_deg * kDeg > 0.5 * f.beamwidth) {
    std::ostringstream s;
    s << "the bearing step (" << o.bearing_step_deg << " deg) is more than half a beamwidth (" << f.beamwidth / kDeg
      << " deg): a target between two bearings loses level (bearings_deg)";
    d.warnings.push_back(s.str());
  }
  if (o.range_cell_m < f.range_resolution) {
    std::ostringstream s;
    s << "range_cell_m (" << o.range_cell_m << " m) is below the range resolution (" << f.range_resolution << " m): the map is only oversampled";
    d.warnings.push_back(s.str());
  }
  return d;
}

void resolve(PassiveOptions& o, const PingRecord& record) {
  if (o.n_fft <= 0) {
    std::int64_t n = kStanding.passive_n_fft;
    const double c = record.environment.sound_speed;
    if (c > 0.0 && record.geometry.rx.size() > 1) {
      const SonarFacts f = sonar_facts(record, o.elevation.row_tolerance_m, o.elevation.min_row_elements);
      const double want = kPassiveTransits * record.samples.sample_rate * f.aperture / c;
      while (static_cast<double>(n) < want) n *= 2;
    }
    o.n_fft = n;
  }
  if (o.hop <= 0) o.hop = std::max<std::int64_t>(o.n_fft / 2, 1);
}

double sync_tolerance(double given_us, double top_frequency) {
  return given_us > 0.0 ? given_us * 1e-6 : 1.0 / (20.0 * top_frequency);
}

double stream_bytes_per_second(const PingRecord& t, int sample_bytes) {
  return static_cast<double>(t.samples.n_channels) * t.samples.sample_rate * static_cast<double>(sample_bytes);
}

std::int64_t stream_buffer_bytes(const PingRecord& t, int sample_bytes, std::int64_t given) {
  if (given > 0) return given;
  return static_cast<std::int64_t>(std::max(kStanding.stream_buffer_bytes, kStreamBufferSeconds * stream_bytes_per_second(t, sample_bytes)));
}

std::string ChainDefaults::json() const {
  const double deg = 180.0 / 3.14159265358979323846;
  std::ostringstream s;
  s << std::setprecision(10);
  s << "{\"range_cell_m\": " << range_cell_m << ", \"cfar_guard_m\": " << cfar_guard_m << ", \"cfar_train_m\": " << cfar_train_m
    << ", \"tracker_range_sigma\": " << tracker_range_sigma << ", \"tracker_bearing_sigma_deg\": " << tracker_bearing_sigma * deg
    << ", \"multipath_bearing_deg\": " << multipath_bearing * deg << ", \"classify_max_fit_rms\": " << classify_max_fit_rms
    << ", \"row_tolerance_m\": " << row_tolerance_m;
  list(s, "given", given);
  list(s, "scaled", scaled);
  list(s, "warnings", warnings);
  s << ", \"sonar\": {\"sound_speed\": " << facts.sound_speed << ", \"f_low\": " << facts.f_low << ", \"f_high\": " << facts.f_high
    << ", \"band\": " << facts.band << ", \"pulse\": " << facts.pulse << ", \"range_resolution_m\": " << facts.range_resolution
    << ", \"wavelength_m\": " << facts.wavelength << ", \"channels\": " << facts.n_channels << ", \"rows\": " << facts.n_rows
    << ", \"row_elements\": " << facts.row_elements << ", \"aperture_m\": " << facts.aperture << ", \"beamwidth_deg\": "
    << facts.beamwidth * deg << ", \"sample_rate\": " << facts.sample_rate << ", \"window_m\": " << facts.window_m << "}}";
  return s.str();
}

}  // namespace sonar::live
