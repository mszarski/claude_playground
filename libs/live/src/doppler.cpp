#include "sonar/live/doppler.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "sonar/dsp/baseband.hpp"

namespace sonar::live {

DopplerMode parse_doppler_mode(const std::string& name) {
  if (name == "off") return DopplerMode::Off;
  if (name == "sweep_bank") return DopplerMode::SweepBank;
  throw std::invalid_argument("doppler mode '" + name + "': one of off, sweep_bank");
}

const char* doppler_mode_name(DopplerMode m) { return m == DopplerMode::SweepBank ? "sweep_bank" : "off"; }

namespace {
// |integral_{-1/2}^{1/2} exp(i 4 phi (u - x0)^2) du| by Simpson's rule fine enough for the phase.
double residual_chirp(double phi, double x0) {
  auto n = static_cast<std::int64_t>(std::max(1000.0, 24.0 * phi * (1.0 + 2.0 * x0)));
  if (n % 2) ++n;
  double re = 0.0, im = 0.0;
  for (std::int64_t i = 0; i <= n; ++i) {
    const double u = -0.5 + static_cast<double>(i) / static_cast<double>(n) - x0;
    const double wgt = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    re += wgt * std::cos(4.0 * phi * u * u);
    im += wgt * std::sin(4.0 * phi * u * u);
  }
  return std::hypot(re, im) / (3.0 * static_cast<double>(n));
}
}  // namespace

double sweep_mismatch(double phi) {
  const double a = std::abs(phi);
  if (a == 0.0) return 1.0;
  // The delay is free: the compressed peak is the largest over where the residual chirp's stationary point lies (x0,
  // in pulse lengths from the centre). A coarse grid, then a golden-section search about its best point.
  double best = residual_chirp(a, 0.0), at = 0.0;
  for (int i = 1; i <= 80; ++i) {
    const double x0 = 0.01 * static_cast<double>(i);
    const double m = residual_chirp(a, x0);
    if (m > best) {
      best = m;
      at = x0;
    }
  }
  double lo = std::max(0.0, at - 0.01), hi = at + 0.01;
  const double g = 0.5 * (std::sqrt(5.0) - 1.0);
  for (int it = 0; it < 30; ++it) {
    const double x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
    if (residual_chirp(a, x1) > residual_chirp(a, x2)) hi = x2; else lo = x1;
  }
  return std::max(best, residual_chirp(a, 0.5 * (lo + hi)));
}

double bank_threshold_db(std::int64_t per_beam) {
  // MEASURED (tools/doppler_bank_threshold.py; docs/tasks/DOPPLER.md D4), not derived. Noise-only maps of the long-range
  // array (800 m, -60 to 60 degrees every 0.5), 40 pings: the multiscale detector with its thresholds unchanged gives on
  // the maximum over N hypotheses FEWER false detections than on one map (budget 3 a ping: 97 on the zero hypothesis,
  // 22, 5, 0, 0, 0 for N = 3, 5, 9, 17, 33; budget 0.3: 11, then 5, 1, 0, 0, 0): its background, a mean of maxima, rises
  // by 2.1, 3.0, 3.9, 4.8, 5.6 dB, more than its tail. The shift that gives the zero hypothesis' count is NEGATIVE
  // (-0.6, -1.1, -1.7, -2.3, -3.0 dB at 3 a ping; -0.3, -0.6, -1.1, -1.7, -2.4 dB at 0.3 a ping) and shrinks by about
  // 0.6 dB a decade of budget towards the default (1e-3 a ping), where it cannot be counted on a CPU (a few hundred
  // thousand pings). Lowering the thresholds by a value measured at a looser budget would raise the false-alarm rate above
  // the one configured at the default, so the chain keeps them (0 dB): its false-alarm rate on the maximum is the
  // configured one or lower (the safe side), and the bank's net gain is its gross gain at an unchanged threshold, a lower
  // bound of the gain at an equal false-alarm rate. `threshold_db` sets a measured value (the GPU job can count it).
  (void)per_beam;
  return 0.0;
}

DopplerPlan plan_doppler(const PingRecord& record, const std::vector<double>& bearings, const DopplerOptions& o) {
  DopplerPlan p;
  if (o.mode == DopplerMode::Off) return p;
  const Waveform& w = record.transmit.waveform;
  const double band = std::abs(w.f_stop - w.f_start);
  if (w.type != WaveformType::Lfm || !(band > 0.0)) {
    throw std::invalid_argument("doppler sweep_bank: the transmit pulse is not a linear sweep (a tone's Doppler bank is another stage, D3)");
  }
  const double c = record.environment.sound_speed;
  if (!(c > 0.0) || !std::isfinite(c)) throw std::invalid_argument("doppler sweep_bank: the record has no sound speed");
  if (!(w.duration > 0.0)) throw std::invalid_argument("doppler sweep_bank: the pulse has no duration");
  p.on = true;
  p.step = is_given(o.step) ? o.step : c / (2.0 * band * w.duration);
  if (!(p.step > 0.0) || !std::isfinite(p.step)) throw std::invalid_argument("doppler step_m_s: a speed above 0");
  if (o.hypotheses < 0 || (o.hypotheses > 0 && o.hypotheses % 2 == 0)) throw std::invalid_argument("doppler hypotheses: an odd count (0: from max_range_rate_m_s)");
  if (!(o.max_range_rate >= 0.0) || !std::isfinite(o.max_range_rate)) throw std::invalid_argument("doppler max_range_rate_m_s: a finite speed, 0 or more");
  if (!(o.unknown_ownship >= 0.0) || !std::isfinite(o.unknown_ownship)) throw std::invalid_argument("doppler unknown_ownship_m_s: a finite speed, 0 or more");

  // The own ship's part along each beam: the mean of the transmit and receive poses' velocities (navigation's).
  const Vec3d v = (record.geometry.tx_pose.velocity + record.geometry.rx_pose.velocity) * 0.5;
  p.velocity_known = std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
  const double span = o.max_range_rate + (p.velocity_known ? 0.0 : o.unknown_ownship);
  const std::int64_t K = o.hypotheses > 0 ? (o.hypotheses - 1) / 2 : static_cast<std::int64_t>(std::ceil(span / p.step - 1e-9));
  p.per_beam = 2 * K + 1;
  // The map's rows look along forward cos b + across sin b (chain.cpp range_bearing_image, the same frame).
  const auto n = record.geometry.rx.size();
  if (n < 2) throw std::invalid_argument("doppler sweep_bank: the record has fewer than two hydrophones");
  std::vector<Vec3d> pos;
  for (const Element& e : record.geometry.rx) pos.push_back(world_position(record.geometry.rx_pose, e.lever_arm));
  const Vec3d forward = world_direction(record.geometry.rx_pose, record.geometry.rx[0].boresight);
  Vec3d across = pos[n - 1] - pos[0];
  across = normalized(across - forward * dot(across, forward));
  std::vector<std::int64_t> centre(bearings.size());
  p.ownship.assign(bearings.size(), 0.0);
  std::int64_t lo = std::numeric_limits<std::int64_t>::max(), hi = std::numeric_limits<std::int64_t>::min();
  for (std::size_t b = 0; b < bearings.size(); ++b) {
    if (p.velocity_known) {
      const Vec3d u = forward * std::cos(bearings[b]) + across * std::sin(bearings[b]);
      p.ownship[b] = -dot(v, u);           // a still scatterer along the beam: opening positive
    }
    centre[b] = static_cast<std::int64_t>(std::llround(p.ownship[b] / p.step));
    lo = std::min(lo, centre[b] - K);
    hi = std::max(hi, centre[b] + K);
  }
  if (bearings.empty()) lo = hi = 0;
  // The LFM coupling of a hypothesis' own rate, f0 T / (f_stop - f_start) (chain.hpp lfm_range_doppler_factor; signed by
  // the sweep): the echo of rate r0 shows on hypothesis r at the zero hypothesis' range minus factor x r, so its samples
  // are read 2 factor r / c later to bring it back there.
  const double factor = 0.5 * (w.f_start + w.f_stop) * w.duration / (w.f_stop - w.f_start);
  for (std::int64_t k = lo; k <= hi; ++k) {
    const double r = static_cast<double>(k) * p.step;
    p.rates.push_back(r);
    p.scales.push_back(k == 0 ? 1.0 : (c - r) / (c + r));
    p.shifts.push_back(k == 0 ? 0.0 : 2.0 * factor * r / c);
  }
  p.pedestal_m = static_cast<double>(p.per_beam) * p.step * w.duration;
  p.zero = (lo <= 0 && 0 <= hi) ? -lo : -1;
  p.first.resize(bearings.size());
  for (std::size_t b = 0; b < bearings.size(); ++b) p.first[b] = centre[b] - K - lo;
  // Baseband: the default band (BasebandOptions) keeps the sweep's band times band_margin, its edges tapered over
  // edge_taper of it; an echo shifted by 2 r f_top / c is untouched while the shift stays inside the untapered part,
  // which is where to_baseband would have widened nothing that matters. Beyond it the allowance is passed (K-6).
  double r_max = 0.0;
  for (double r : p.rates) r_max = std::max(r_max, std::abs(r));
  const dsp::BasebandOptions def;
  const double f_top = std::max(w.f_start, w.f_stop);
  const double shift = 2.0 * r_max * f_top / c;
  const double room = 0.5 * band * def.band_margin * (1.0 - 2.0 * def.edge_taper) - 0.5 * band;
  p.baseband_max_range_rate = shift <= room ? 0.0 : r_max;
  p.threshold_db = is_given(o.threshold_db) ? o.threshold_db : bank_threshold_db(p.per_beam);
  if (!std::isfinite(p.threshold_db)) throw std::invalid_argument("doppler threshold_db: a finite number of dB");
  return p;
}

std::string DopplerPlan::json() const {
  std::ostringstream s;
  s << std::setprecision(10) << "{\"mode\": \"" << (on ? "sweep_bank" : "off") << "\"";
  if (on) {
    s << ", \"step_m_s\": " << step << ", \"hypotheses\": " << rates.size() << ", \"per_beam\": " << per_beam
      << ", \"first_m_s\": " << rates.front() << ", \"last_m_s\": " << rates.back() << ", \"velocity_known\": " << (velocity_known ? "true" : "false")
      << ", \"baseband_max_range_rate_m_s\": " << baseband_max_range_rate << ", \"threshold_db\": " << threshold_db
      << ", \"pedestal_m\": " << pedestal_m;
  }
  s << "}";
  return s.str();
}

void fold_start(std::int64_t n_bearing, std::int64_t n, std::vector<float>& best, std::vector<std::int32_t>& which) {
  best.assign(static_cast<std::size_t>(n_bearing * n), -1.0F);
  which.assign(static_cast<std::size_t>(n_bearing * n), -1);
}

void fold_hypothesis(const DopplerPlan& plan, std::int64_t h, const float* fine, std::int64_t n_bearing, std::int64_t n,
                     std::vector<float>& best, std::vector<std::int32_t>& which) {
  for (std::int64_t b = 0; b < n_bearing; ++b) {
    const std::int64_t first = plan.first[static_cast<std::size_t>(b)];
    if (h < first || h >= first + plan.per_beam) continue;
    const float* row = fine + b * n;
    float* top = best.data() + b * n;
    std::int32_t* at = which.data() + b * n;
    for (std::int64_t i = 0; i < n; ++i) {
      if (row[i] > top[i]) {             // strictly: a tie keeps the earlier hypothesis (the Python engine the same)
        top[i] = row[i];
        at[i] = static_cast<std::int32_t>(h);
      }
    }
  }
}

std::int64_t detection_hypothesis(const float* best, const std::int32_t* which, std::int64_t fine_stride, std::int64_t fine_offset,
                                  std::int64_t n_fine, const std::vector<double>& fine_ranges, const std::vector<double>& bearings,
                                  double bearing, double lo, double hi) {
  if (bearings.empty() || n_fine <= 0) return -1;
  std::size_t ib = 0;
  for (std::size_t q = 1; q < bearings.size(); ++q) {
    if (std::abs(bearings[q] - bearing) < std::abs(bearings[ib] - bearing)) ib = q;
  }
  const float* row = best + static_cast<std::int64_t>(ib) * fine_stride + fine_offset;
  const std::int32_t* at = which + static_cast<std::int64_t>(ib) * fine_stride + fine_offset;
  std::int64_t pick = -1;
  for (std::int64_t i = 0; i < n_fine; ++i) {
    const double r = fine_ranges[static_cast<std::size_t>(fine_offset + i)];
    if (r < lo || r > hi) continue;
    if (pick < 0 || row[i] > row[pick]) pick = i;
  }
  if (pick < 0) {
    for (std::int64_t i = 0; i < n_fine; ++i) {
      if (pick < 0 || row[i] > row[pick]) pick = i;
    }
  }
  return at[pick];
}

}  // namespace sonar::live
