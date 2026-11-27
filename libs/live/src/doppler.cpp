#include "sonar/live/doppler.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "sonar/core/waveform.hpp"
#include "sonar/dsp/baseband.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/live/beamform.hpp"
#include "sonar/live/detector.hpp"
#include "sonar/live/doppler_batched.hpp"
#include "sonar/dsp/exec_host.hpp"

namespace sonar::live {

DopplerMode parse_doppler_mode(const std::string& name) {
  if (name == "off") return DopplerMode::Off;
  if (name == "sweep_bank") return DopplerMode::SweepBank;
  throw std::invalid_argument("doppler mode '" + name + "': one of off, sweep_bank");
}

const char* doppler_mode_name(DopplerMode m) { return m == DopplerMode::SweepBank ? "sweep_bank" : "off"; }

DopplerCombine parse_doppler_combine(const std::string& name) {
  if (name == "max") return DopplerCombine::Max;
  if (name == "cfar") return DopplerCombine::Cfar;
  throw std::invalid_argument("doppler combine '" + name + "': one of max, cfar");
}

const char* doppler_combine_name(DopplerCombine c) { return c == DopplerCombine::Cfar ? "cfar" : "max"; }

DopplerBatched parse_doppler_batched(const std::string& name) {
  if (name == "auto") return DopplerBatched::Auto;
  if (name == "on") return DopplerBatched::On;
  if (name == "off") return DopplerBatched::Off;
  throw std::invalid_argument("doppler batched '" + name + "': one of auto, on, off");
}

const char* doppler_batched_name(DopplerBatched b) { return b == DopplerBatched::On ? "on" : b == DopplerBatched::Off ? "off" : "auto"; }

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
  p.combine = o.combine;
  if (p.combine == DopplerCombine::Cfar) {
    // Each hypothesis normalised by its own background: the statistic is the largest of per_beam normalised maps, each a
    // map the detector's thresholds were planned for. P(max > x) <= per_beam P(one > x) (the union bound; ours): the
    // budget divided by per_beam holds the configured rate, on the safe side where neighbouring hypotheses are correlated
    // (the sweep's are: one step loses 0.97 dB, so their noise is nearly the same). Nothing else moves the thresholds.
    p.budget_divisor = static_cast<double>(p.per_beam);
    p.threshold_db = is_given(o.threshold_db) ? o.threshold_db : 0.0;
  } else {
    p.threshold_db = is_given(o.threshold_db) ? o.threshold_db : bank_threshold_db(p.per_beam);
  }
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
    if (combine == DopplerCombine::Cfar) s << ", \"combine\": \"cfar\", \"budget_divisor\": " << budget_divisor;
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

void normalised_start(std::int64_t n_bearing, std::int64_t n_cells, std::int64_t cell_samples, std::int64_t skip, NormalisedFold& f) {
  f.n_bearing = n_bearing;
  f.n_cells = n_cells;
  f.cell_samples = cell_samples;
  f.stride = n_cells * cell_samples;
  f.skip = std::clamp<std::int64_t>(skip, 0, n_cells);
  f.ratio.assign(static_cast<std::size_t>(n_bearing * f.stride), -1.0F);
  f.cell_ratio.assign(static_cast<std::size_t>(n_bearing * n_cells), -1.0F);
  f.reference.assign(static_cast<std::size_t>(n_bearing * n_cells), 0.0);
  f.which.assign(static_cast<std::size_t>(n_bearing * f.stride), -1);
}

void fold_normalised(const DopplerPlan& plan, std::int64_t h, const float* fine, std::int64_t guard, std::int64_t train, NormalisedFold& f) {
  const std::int64_t n_b = f.n_bearing, n_c = f.n_cells, k = f.cell_samples, skip = f.skip, kept = n_c - skip;
  if (kept <= 0) return;
  // Cell means of the kept cells, in double then float (live.py doppler.fold_normalised the same).
  std::vector<float> cells(static_cast<std::size_t>(n_b * kept));
  for (std::int64_t b = 0; b < n_b; ++b) {
    const float* row = fine + b * f.stride;
    for (std::int64_t j = 0; j < kept; ++j) {
      double sum = 0.0;
      for (std::int64_t q = 0; q < k; ++q) sum += static_cast<double>(row[(skip + j) * k + q]);
      cells[static_cast<std::size_t>(b * kept + j)] = static_cast<float>(sum / static_cast<double>(k));
    }
  }
  std::vector<double> level, worth;
  detector_background(cells.data(), n_b, kept, guard, train, level, worth);
  for (std::int64_t b = 0; b < n_b; ++b) {
    const std::int64_t first = plan.first[static_cast<std::size_t>(b)];
    if (h < first || h >= first + plan.per_beam) continue;
    const bool centre = h == first + (plan.per_beam - 1) / 2;
    const float* row = fine + b * f.stride;
    for (std::int64_t j = 0; j < kept; ++j) {
      const std::size_t at = static_cast<std::size_t>(b * kept + j), cell = static_cast<std::size_t>(b * n_c + skip + j);
      const double floor = std::max(level[at], 1e-300);
      if (centre) f.reference[cell] = level[at];
      const auto c = static_cast<float>(static_cast<double>(cells[at]) / floor);
      if (c > f.cell_ratio[cell]) f.cell_ratio[cell] = c;
      for (std::int64_t q = 0; q < k; ++q) {
        const std::size_t i = static_cast<std::size_t>(b * f.stride + (skip + j) * k + q);
        const auto r = static_cast<float>(static_cast<double>(row[(skip + j) * k + q]) / floor);
        if (r > f.ratio[i]) {               // strictly: a tie keeps the earlier hypothesis
          f.ratio[i] = r;
          f.which[i] = static_cast<std::int32_t>(h);
        }
      }
    }
  }
}

void normalised_map(const NormalisedFold& f, std::vector<float>& fine, std::vector<float>& cells, std::vector<double>& level) {
  fine.assign(static_cast<std::size_t>(f.n_bearing * f.stride), 0.0F);
  cells.assign(static_cast<std::size_t>(f.n_bearing * f.n_cells), 0.0F);
  level = f.reference;
  for (std::int64_t b = 0; b < f.n_bearing; ++b) {
    for (std::int64_t j = f.skip; j < f.n_cells; ++j) {
      const std::size_t cell = static_cast<std::size_t>(b * f.n_cells + j);
      const double ref = f.reference[cell];
      cells[cell] = static_cast<float>(static_cast<double>(std::max(f.cell_ratio[cell], 0.0F)) * ref);
      for (std::int64_t q = 0; q < f.cell_samples; ++q) {
        const std::size_t i = static_cast<std::size_t>(b * f.stride + j * f.cell_samples + q);
        fine[i] = static_cast<float>(static_cast<double>(std::max(f.ratio[i], 0.0F)) * ref);
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

// ------------------------------------------------------------------------------------------------------------------------
// D3 part B: the tone's Doppler filter bank (doppler.hpp says what and why).

namespace sonar::live {

#if SONAR_HAVE_CUDA
// doppler_cuda.cu: the batched tone bank on the device (doppler_batched.hpp).
void tone_bank_cuda(const ToneBankBatchedInput& in, float* power, float* ratio, unsigned char* flag, DopplerProfile& prof, int device);
#endif

namespace {
constexpr double kPi = 3.14159265358979323846;

std::size_t pow2_at_least(std::size_t n) {
  std::size_t m = 1;
  while (m < n) m <<= 1;
  return m;
}
}  // namespace

TonePlan plan_tone(const PingRecord& record, const std::vector<double>& bearings, const DopplerOptions& o, double default_budget) {
  TonePlan p;
  const ToneOptions& t = o.tone;
  if (!t.enabled) return p;
  const Waveform& w = record.transmit.waveform;
  if (w.type == WaveformType::Lfm) return p;                 // a sweep-only ping: no tone this ping
  const double c = record.environment.sound_speed;
  if (!(c > 0.0) || !std::isfinite(c)) throw std::invalid_argument("doppler tone: the record has no sound speed");
  const bool cw = w.type == WaveformType::Cw;
  if (!cw && !is_given(t.frequency)) {
    throw std::invalid_argument("doppler tone: the record's pulse is not a tone; give doppler tone_frequency_hz, tone_start_s, tone_duration_s "
                                "and tone_window_alpha from the train's segment table");
  }
  p.frequency = is_given(t.frequency) ? t.frequency : w.f_start;
  p.start = t.start;
  p.duration = is_given(t.duration) ? t.duration : (cw ? w.duration - t.start : kDerive);
  p.window_alpha = is_given(t.window_alpha) ? t.window_alpha : (cw ? w.window_alpha : kDerive);
  if (!is_given(p.duration) || !is_given(p.window_alpha)) {
    throw std::invalid_argument("doppler tone: a train needs tone_duration_s and tone_window_alpha");
  }
  if (!(p.frequency > 0.0) || !std::isfinite(p.frequency)) throw std::invalid_argument("doppler tone_frequency_hz: a frequency above 0");
  if (!(p.duration > 0.0) || !(p.start >= 0.0) || p.start + p.duration > w.duration * (1.0 + 1e-9) + 1e-9) {
    throw std::invalid_argument("doppler tone: tone_start_s and tone_duration_s must lie inside the pulse");
  }
  if (!(p.window_alpha >= 0.0 && p.window_alpha <= 1.0)) throw std::invalid_argument("doppler tone_window_alpha: from 0 to 1");
  if (!(t.hop > 0.0 && t.hop <= 1.0)) throw std::invalid_argument("doppler tone_hop: a fraction of the tone above 0, at most 1");
  if (t.train < 1 || t.guard < 0) throw std::invalid_argument("doppler tone_train: 1 or more; tone_guard: 0 (derived) or more");
  p.on = true;
  p.tone_only = cw || p.duration >= 0.99 * w.duration;
  p.resolution = c / (2.0 * p.frequency * p.duration);
  p.step = is_given(t.step) ? t.step : 0.5 * p.resolution;
  if (!(p.step > 0.0) || !std::isfinite(p.step)) throw std::invalid_argument("doppler tone_step_m_s: a speed above 0");
  const double r_max = is_given(t.max_range_rate) ? t.max_range_rate : o.max_range_rate;
  if (!(r_max >= 0.0) || !std::isfinite(r_max)) throw std::invalid_argument("doppler tone_max_range_rate_m_s: a finite speed, 0 or more");
  p.hop = t.hop * p.duration;
  p.guard = t.guard > 0 ? t.guard : static_cast<std::int64_t>(std::ceil(2.0 * p.resolution / p.step - 1e-9));
  p.train = t.train;
  p.budget = is_given(t.false_alarms_per_ping) ? t.false_alarms_per_ping : default_budget;
  if (!(p.budget > 0.0) || !std::isfinite(p.budget)) throw std::invalid_argument("doppler tone_false_alarms_per_ping: above 0");
  p.gate_range = is_given(t.gate_range) ? t.gate_range : 0.25 * c * p.duration;
  p.gate_bearing = is_given(t.gate_bearing) ? t.gate_bearing : 2.0 * kPi / 180.0;
  p.range_sigma = is_given(t.range_sigma) ? t.range_sigma : c * p.duration / 8.0;
  p.sidelobe_db = t.sidelobe_db;           // NaN: by the aperture window (tone_stage)

  // The own ship's part along each beam, as the sweep bank (plan_doppler).
  const Vec3d v = (record.geometry.tx_pose.velocity + record.geometry.rx_pose.velocity) * 0.5;
  const bool known = std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
  const double span = r_max + (known ? 0.0 : o.unknown_ownship);
  const auto K = static_cast<std::int64_t>(std::ceil(span / p.step - 1e-9));
  p.per_beam = 2 * K + 1;
  const auto n = record.geometry.rx.size();
  if (n < 2) throw std::invalid_argument("doppler tone: the record has fewer than two hydrophones");
  std::vector<Vec3d> pos;
  for (const Element& e : record.geometry.rx) pos.push_back(world_position(record.geometry.rx_pose, e.lever_arm));
  const Vec3d forward = world_direction(record.geometry.rx_pose, record.geometry.rx[0].boresight);
  Vec3d across = pos[n - 1] - pos[0];
  const double length = norm(across - forward * dot(across, forward));
  across = normalized(across - forward * dot(across, forward));
  const double lambda = c / p.frequency;
  std::vector<std::int64_t> centre(bearings.size());
  p.ownship.assign(bearings.size(), 0.0);
  p.notch.assign(bearings.size(), 0.0);
  std::int64_t lo = std::numeric_limits<std::int64_t>::max(), hi = std::numeric_limits<std::int64_t>::min();
  for (std::size_t b = 0; b < bearings.size(); ++b) {
    double spread = 0.0;
    if (known) {
      const double cb = std::cos(bearings[b]), sb = std::sin(bearings[b]);
      const Vec3d u = forward * cb + across * sb, du = forward * (-sb) + across * cb;
      p.ownship[b] = -dot(v, u);
      const double width = length > 0.0 ? 0.886 * lambda / (length * std::max(std::abs(cb), 0.05)) : kPi;
      spread = std::abs(dot(v, du)) * 0.5 * width;
    }
    p.notch[b] = is_given(t.notch) ? t.notch : spread + t.notch_cells * p.resolution;
    centre[b] = static_cast<std::int64_t>(std::llround(p.ownship[b] / p.step));
    lo = std::min(lo, centre[b] - K);
    hi = std::max(hi, centre[b] + K);
  }
  if (bearings.empty()) lo = hi = 0;
  for (std::int64_t k = lo; k <= hi; ++k) p.rates.push_back(static_cast<double>(k) * p.step);
  p.first.resize(bearings.size());
  for (std::size_t b = 0; b < bearings.size(); ++b) p.first[b] = centre[b] - K - lo;
  // Baseband: the band to_baseband keeps must hold the tone and its shifts in its untapered part.
  double r_top = 0.0;
  for (double r : p.rates) r_top = std::max(r_top, std::abs(r));
  const double need = 2.0 * r_top * p.frequency / c + 8.0 / p.duration;
  const dsp::BasebandOptions def;
  const double f_top = std::max(w.f_start, w.f_stop), centre_f = w.center();
  const double reach = std::max(std::abs(p.frequency + need - centre_f), std::abs(p.frequency - need - centre_f));
  const double want_band = reach / (0.5 - def.edge_taper);
  const double band0 = w.bandwidth() > 0.0 ? w.bandwidth() * def.band_margin : def.cw_band_factor / w.duration;
  p.baseband_max_range_rate = want_band > band0 ? (want_band - band0) * c / (4.0 * f_top) : 0.0;
  return p;
}

std::string TonePlan::json() const {
  std::ostringstream s;
  s << std::setprecision(10) << "{\"tone\": " << (on ? "true" : "false");
  if (on) {
    s << ", \"frequency_hz\": " << frequency << ", \"start_s\": " << start << ", \"duration_s\": " << duration << ", \"window_alpha\": " << window_alpha
      << ", \"tone_only\": " << (tone_only ? "true" : "false") << ", \"step_m_s\": " << step << ", \"resolution_m_s\": " << resolution
      << ", \"hypotheses\": " << rates.size() << ", \"per_beam\": " << per_beam << ", \"hop_s\": " << hop << ", \"guard\": " << guard
      << ", \"train\": " << train << ", \"budget\": " << budget << ", \"baseband_max_range_rate_m_s\": " << baseband_max_range_rate
      << ", \"gate_range_m\": " << gate_range << ", \"gate_bearing_rad\": " << gate_bearing << ", \"range_sigma_m\": " << range_sigma;
  }
  s << "}";
  return s.str();
}

ToneStage tone_stage(const PingRecord& record, const std::vector<double>& bearings, const DopplerOptions& options, bool hann,
                     double default_budget, Backend backend, int device, bool keep_map) {
  ToneStage st;
  st.plan = plan_tone(record, bearings, options, default_budget);
  const TonePlan& p = st.plan;
  if (!p.on || bearings.empty()) return st;
  if (!record.samples.is_complex()) throw std::invalid_argument("doppler tone: the record must be complex baseband (before pulse compression)");
  // D8: the bank, its CFAR and the leakage rule by the batched formulation (doppler_batched.hpp) on the device, or on
  // the host when asked (doppler.batched on); the loop below otherwise (the code before D8). On the device the beams
  // stay where the beamformer makes them; on the host the batched form takes the reference beamformer's beams.
  const bool batched = options.batched == DopplerBatched::On || (options.batched == DopplerBatched::Auto && backend == Backend::Cuda);
  const bool device_beams = batched && backend == Backend::Cuda;
  DopplerProfile& prof = doppler_profile_accumulator();
  const auto t_beams = std::chrono::steady_clock::now();
  BeamMap beams;
  if (!device_beams) beams = farfield_beamform(record, bearings, hann ? "hann" : "rect", backend, device);
  if (!device_beams) prof.tone_beams += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_beams).count();
  ++prof.tone_calls;
  const Samples& s = record.samples;
  const double c = record.environment.sound_speed, fs = s.sample_rate, fc = s.center_frequency, t0 = s.start_time;
  const auto n = static_cast<std::size_t>(device_beams ? s.n_samples : beams.n_samples);
  const auto n_b = static_cast<std::int64_t>(bearings.size());
  // 1. Mix each beam to the tone and cut the band the hypotheses need (an exact FFT decimation, zero-padded so that the
  //    window's length does not wrap).
  const std::size_t N = pow2_at_least(n + static_cast<std::size_t>(std::ceil(p.duration * fs)) + 1);
  const double df = fs / static_cast<double>(N);
  double r_top = 0.0;
  for (double r : p.rates) r_top = std::max(r_top, std::abs(r));
  const double keep = 2.0 * r_top * p.frequency / c + 8.0 / p.duration;
  const auto h = static_cast<std::int64_t>(std::ceil(keep / df));
  const auto M = std::max<std::size_t>(pow2_at_least(static_cast<std::size_t>(2 * h + 1)), 16);
  const auto shift = static_cast<std::int64_t>(std::llround((p.frequency - fc) / df));
  const double f_mix = fc + static_cast<double>(shift) * df;          // the passband frequency the decimated frame sits at
  st.fs = static_cast<double>(M) * df;
  const auto L = static_cast<std::int64_t>(std::llround(p.duration * st.fs));
  const auto H = std::max<std::int64_t>(static_cast<std::int64_t>(std::llround(p.hop * st.fs)), 1);
  const auto n_valid = static_cast<std::int64_t>(std::floor(static_cast<double>(n) * st.fs / fs));
  if (L < 4 || n_valid < L) throw std::invalid_argument("doppler tone: the receive window is shorter than the tone");
  st.n_lags = (n_valid - L) / H + 1;
  for (std::int64_t l = 0; l < st.n_lags; ++l) st.lag_ranges.push_back(0.5 * c * (t0 + static_cast<double>(l * H) / st.fs - p.start));
  // The matched filter's envelope (the tone's taper) and each hypothesis' phasor.
  std::vector<double> win(static_cast<std::size_t>(L));
  double sw = 0.0, sw2 = 0.0;
  for (std::int64_t m = 0; m < L; ++m) {
    win[static_cast<std::size_t>(m)] = tukey(static_cast<double>(m) / static_cast<double>(L), p.window_alpha);
    sw += win[static_cast<std::size_t>(m)];
    sw2 += win[static_cast<std::size_t>(m)] * win[static_cast<std::size_t>(m)];
  }
  const double enbw = static_cast<double>(L) * sw2 / (sw * sw);      // in bins (Hann 1.5, rectangular 1)
  const std::size_t n_r = p.rates.size();
  std::vector<std::complex<double>> phasor(n_r * static_cast<std::size_t>(L));
  for (std::size_t k = 0; k < n_r; ++k) {
    const double nu = p.frequency * (c - p.rates[k]) / (c + p.rates[k]) - f_mix;
    for (std::int64_t m = 0; m < L; ++m) {
      const double ph = -2.0 * kPi * nu * static_cast<double>(m) / st.fs;
      phasor[k * static_cast<std::size_t>(L) + static_cast<std::size_t>(m)] = win[static_cast<std::size_t>(m)] * std::complex<double>(std::cos(ph), std::sin(ph));
    }
  }
  const std::int64_t J = p.per_beam, NL = st.n_lags;
  // The notch, the tests counted and the CFAR's factor (step 3), and the leakage rule's envelope and margin (step 3b),
  // made before the bank: the batched form takes them as inputs.
  std::vector<unsigned char> notched(static_cast<std::size_t>(n_b * J), 0);
  for (std::int64_t b = 0; b < n_b; ++b) {
    for (std::int64_t j = 0; j < J; ++j) {
      const double r = p.rates[static_cast<std::size_t>(p.first[static_cast<std::size_t>(b)] + j)];
      notched[static_cast<std::size_t>(b * J + j)] = std::abs(r - p.ownship[static_cast<std::size_t>(b)]) <= p.notch[static_cast<std::size_t>(b)] ? 1 : 0;
    }
  }
  std::int64_t tested = 0;
  for (std::int64_t b = 0; b < n_b; ++b) {
    for (std::int64_t j = 0; j < J; ++j) tested += notched[static_cast<std::size_t>(b * J + j)] ? 0 : 1;
  }
  st.tests = tested * NL;
  const double pfa = st.tests > 0 ? std::min(p.budget / static_cast<double>(st.tests), 0.5) : 0.5;
  const double independent = std::min(1.0, p.step / (enbw * p.resolution));
  const auto factor_for = [&](double count) {
    const double n_eff = std::max(count * independent, 1.0);
    return n_eff * (std::pow(pfa, -1.0 / n_eff) - 1.0);
  };
  st.factor = factor_for(static_cast<double>(2 * p.train));
  const auto leak_envelope = [&]() {
    const double dnu = 2.0 * p.step * p.frequency / c;
    std::vector<double> env(static_cast<std::size_t>(J + 1), 0.0);
    for (std::int64_t d = 0; d <= J; ++d) {
      std::complex<double> acc(0.0, 0.0);
      for (std::int64_t m = 0; m < L; ++m) {
        const double ph = -2.0 * kPi * static_cast<double>(d) * dnu * static_cast<double>(m) / st.fs;
        acc += win[static_cast<std::size_t>(m)] * std::complex<double>(std::cos(ph), std::sin(ph));
      }
      env[static_cast<std::size_t>(d)] = std::norm(acc) / (sw * sw);
    }
    for (std::int64_t d = J - 1; d >= 0; --d) env[static_cast<std::size_t>(d)] = std::max(env[static_cast<std::size_t>(d)], env[static_cast<std::size_t>(d + 1)]);
    return env;
  };
  const double margin = std::pow(10.0, 0.1 * options.tone.leak_margin_db);
  std::vector<float> power(static_cast<std::size_t>(n_b * J * NL), 0.0F);
  std::vector<float> ratio(power.size(), 0.0F);
  std::vector<std::array<std::int64_t, 3>> flagged;
  if (batched) {
    const std::vector<double> env = leak_envelope();
    std::vector<double> factor(static_cast<std::size_t>(2 * p.train + 1));
    for (std::size_t q = 0; q < factor.size(); ++q) factor[q] = factor_for(static_cast<double>(q));
    std::vector<unsigned char> flag(power.size(), 0);
    ToneBankBatchedInput in;
    std::vector<float> w;
    std::vector<double> tau;
    if (device_beams) {
      w = aperture_weights(record, hann ? "hann" : "rect");
      tau = steering_delays(record, bearings);
      in.samples = s.data.data();
      in.w = w.data();
      in.tau = tau.data();
      in.n_el = s.n_channels;
      in.df = fs / static_cast<double>(n);
      in.fc = fc;
    } else {
      in.beams = reinterpret_cast<const float*>(beams.data.data());
    }
    in.n_b = n_b;
    in.n = static_cast<std::int64_t>(n);
    in.N = static_cast<std::int64_t>(N);
    in.M = static_cast<std::int64_t>(M);
    in.h = h;
    in.shift = shift;
    in.L = L;
    in.H = H;
    in.J = J;
    in.NL = NL;
    in.n_r = static_cast<std::int64_t>(n_r);
    in.scale = static_cast<double>(M) / static_cast<double>(N);
    in.sw2 = sw2;
    in.phasor = reinterpret_cast<const double*>(phasor.data());
    in.first = p.first.data();
    in.notched = notched.data();
    in.factor = factor.data();
    in.guard = p.guard;
    in.train = p.train;
    in.env = env.data();
    in.margin = margin;
    if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
      tone_bank_cuda(in, power.data(), ratio.data(), flag.data(), prof, device);
#else
      throw std::runtime_error("doppler tone: built without the CUDA backend");
#endif
    } else {
      dsp::HostExec ex;
      tone_bank_batched(ex, in, power.data(), ratio.data(), flag.data(), prof);
    }
    // The flagged cells in the loop's order (beam, lag, hypothesis), after its leakage rule.
    for (std::int64_t b = 0; b < n_b; ++b) {
      for (std::int64_t l = 0; l < NL; ++l) {
        for (std::int64_t j = 0; j < J; ++j) {
          if (flag[static_cast<std::size_t>((b * J + j) * NL + l)]) flagged.push_back({b, p.first[static_cast<std::size_t>(b)] + j, l});
        }
      }
    }
  } else {
  const auto t_bank = std::chrono::steady_clock::now();
  // 2. The bank: power [b][j][l], j the beam's own hypotheses.
  std::vector<std::complex<float>> buf(N), dec(M);
  for (std::int64_t b = 0; b < n_b; ++b) {
    std::fill(buf.begin(), buf.end(), std::complex<float>(0.0F, 0.0F));
    std::copy_n(beams.row(b), n, buf.begin());
    dsp::fft(buf.data(), N, false);
    std::fill(dec.begin(), dec.end(), std::complex<float>(0.0F, 0.0F));
    for (std::int64_t q = -h; q <= h; ++q) {
      const auto src = static_cast<std::size_t>(((shift + q) % static_cast<std::int64_t>(N) + static_cast<std::int64_t>(N)) % static_cast<std::int64_t>(N));
      dec[static_cast<std::size_t>((q + static_cast<std::int64_t>(M)) % static_cast<std::int64_t>(M))] = buf[src];
    }
    dsp::fft(dec.data(), M, true);          // x (1 / M): z = the beam at t0 + m / fs_d, mixed by f_mix, x M / N
    const double scale = static_cast<double>(M) / static_cast<double>(N);
    for (std::int64_t j = 0; j < J; ++j) {
      const std::size_t k = static_cast<std::size_t>(p.first[static_cast<std::size_t>(b)] + j);
      const std::complex<double>* ph = phasor.data() + k * static_cast<std::size_t>(L);
      for (std::int64_t l = 0; l < NL; ++l) {
        std::complex<double> acc(0.0, 0.0);
        const std::complex<float>* z = dec.data() + l * H;
        for (std::int64_t m = 0; m < L; ++m) acc += std::complex<double>(z[m]) * ph[m];
        // The lag's own phase (exp(-i 2 pi nu l H / fs)) leaves the power alone.
        power[static_cast<std::size_t>((b * J + j) * NL + l)] = static_cast<float>(std::norm(acc) * scale * scale / sw2);
      }
    }
  }
  prof.tone_bank += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_bank).count();
  const auto t_cfar = std::chrono::steady_clock::now();
  // 3. The CFAR along rate, outside the notch.
  for (std::int64_t b = 0; b < n_b; ++b) {
    for (std::int64_t l = 0; l < NL; ++l) {
      for (std::int64_t j = 0; j < J; ++j) {
        if (notched[static_cast<std::size_t>(b * J + j)]) continue;
        double sum = 0.0;
        std::int64_t count = 0;
        for (std::int64_t d = p.guard + 1; d <= p.guard + p.train; ++d) {
          for (const std::int64_t q : {j - d, j + d}) {
            if (q < 0 || q >= J || notched[static_cast<std::size_t>(b * J + q)]) continue;
            sum += static_cast<double>(power[static_cast<std::size_t>((b * J + q) * NL + l)]);
            ++count;
          }
        }
        if (count == 0 || !(sum > 0.0)) continue;
        const double mean = sum / static_cast<double>(count);
        const double x = static_cast<double>(power[static_cast<std::size_t>((b * J + j) * NL + l)]) / mean;
        ratio[static_cast<std::size_t>((b * J + j) * NL + l)] = static_cast<float>(x);
        if (x > factor_for(static_cast<double>(count))) flagged.push_back({b, p.first[static_cast<std::size_t>(b)] + j, l});
      }
    }
  }
  // 3b. Leakage: the tone's own spectral sidelobes. A cell is not a detection when its power is no more than
  //     `leak_margin_db` above what a stronger cell of the same beam and lag leaks into it through the matched filter's
  //     spectrum: the ridge (the strongest notched hypothesis: the stationary reverberation and every still target) and
  //     every stronger flagged cell. The leak at d hypotheses is the stronger power times the envelope of the window's
  //     normalised power spectrum, |sum w[m] exp(-i 2 pi d dnu m / fs)|^2 / (sum w)^2 (dnu: the hypotheses' spacing in Hz),
  //     its running maximum outward (ours; the detector's sidelobe rule along range and bearing, detector.hpp, along rate).
  if (!flagged.empty()) {
    const std::vector<double> env = leak_envelope();
    const auto pw = [&](std::int64_t b, std::int64_t j, std::int64_t l) { return static_cast<double>(power[static_cast<std::size_t>((b * J + j) * NL + l)]); };
    std::vector<std::array<std::int64_t, 3>> kept;
    for (const auto& f : flagged) {
      const std::int64_t b = f[0], j = f[1] - p.first[static_cast<std::size_t>(b)], l = f[2];
      const double me = pw(b, j, l);
      double leak = 0.0;
      for (std::int64_t q = 0; q < J; ++q) {
        if (q == j) continue;
        const bool source = notched[static_cast<std::size_t>(b * J + q)] != 0 ||
                            static_cast<double>(ratio[static_cast<std::size_t>((b * J + q) * NL + l)]) > 0.0;   // tested cells only: flagged ones are among them
        if (!source) continue;
        const double other = pw(b, q, l);
        if (other <= me) continue;
        leak = std::max(leak, other * env[static_cast<std::size_t>(std::abs(q - j))]);
      }
      if (me > margin * leak) kept.push_back(f);
    }
    flagged = std::move(kept);
  }
  prof.tone_cfar += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cfar).count();
  }   // the loop (not re-indented: as before D8)
  // 4. Groups of touching flagged cells (beam, global hypothesis, lag), one detection each.
  const std::size_t F = flagged.size();
  std::vector<std::size_t> parent(F);
  for (std::size_t i = 0; i < F; ++i) parent[i] = i;
  const auto root = [&](std::size_t i) {
    while (parent[i] != i) i = parent[i] = parent[parent[i]];
    return i;
  };
  for (std::size_t i = 0; i < F; ++i) {           // flagged is ordered by beam, then lag, then hypothesis
    for (std::size_t q = i + 1; q < F; ++q) {
      if (flagged[q][0] > flagged[i][0] + 1) break;
      if (std::abs(flagged[q][1] - flagged[i][1]) <= 1 && std::abs(flagged[q][2] - flagged[i][2]) <= 1) parent[root(q)] = root(i);
    }
  }
  const auto at = [&](std::int64_t b, std::int64_t k, std::int64_t l, double& out) {
    if (b < 0 || b >= n_b || l < 0 || l >= NL) return false;
    const std::int64_t j = k - p.first[static_cast<std::size_t>(b)];
    if (j < 0 || j >= J) return false;
    out = static_cast<double>(power[static_cast<std::size_t>((b * J + j) * NL + l)]);
    return out > 0.0;
  };
  const auto parabola = [](double a, double m, double z) {
    const double la = std::log(a), lm = std::log(m), lz = std::log(z), den = la - 2.0 * lm + lz;
    if (!(den < 0.0)) return 0.0;
    return std::clamp(0.5 * (la - lz) / den, -0.5, 0.5);
  };
  struct Group {
    std::size_t peak;
    double x, pw;
    std::int64_t cells;
  };
  std::vector<Group> groups;
  std::vector<std::int64_t> group_of(F, -1);
  for (std::size_t i = 0; i < F; ++i) {
    const std::size_t r = root(i);
    if (group_of[r] < 0) {
      group_of[r] = static_cast<std::int64_t>(groups.size());
      groups.push_back({i, -1.0, 0.0, 0});
    }
    Group& g = groups[static_cast<std::size_t>(group_of[r])];
    const auto& f = flagged[i];
    const std::int64_t j = f[1] - p.first[static_cast<std::size_t>(f[0])];
    const double x = static_cast<double>(ratio[static_cast<std::size_t>((f[0] * J + j) * NL + f[2])]);
    ++g.cells;
    if (x > g.x) {
      g.x = x;
      g.peak = i;
      g.pw = static_cast<double>(power[static_cast<std::size_t>((f[0] * J + j) * NL + f[2])]);
    }
  }
  // Bearing sidelobes of a stronger group: dropped, strongest first.
  const double side = is_given(p.sidelobe_db) ? p.sidelobe_db : (hann ? -25.0 : -7.0);
  const auto per_pulse = static_cast<std::int64_t>(std::ceil(1.0 / options.tone.hop));
  std::vector<std::size_t> order(groups.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b2) { return groups[a].pw > groups[b2].pw; });
  std::vector<char> alive(groups.size(), 1);
  for (std::size_t a = 0; a < order.size(); ++a) {
    const Group& s0 = groups[order[a]];
    if (!alive[order[a]]) continue;
    const auto& fs0 = flagged[s0.peak];
    for (std::size_t q = a + 1; q < order.size(); ++q) {
      const Group& g = groups[order[q]];
      const auto& fg = flagged[g.peak];
      if (alive[order[q]] && fg[0] != fs0[0] && std::abs(fg[1] - fs0[1]) <= p.guard && std::abs(fg[2] - fs0[2]) <= per_pulse &&
          g.pw <= s0.pw * std::pow(10.0, 0.1 * side)) {
        alive[order[q]] = 0;
      }
    }
  }
  const double bstep = bearings.size() > 1 ? bearings[1] - bearings[0] : 0.0;
  for (std::size_t gi = 0; gi < groups.size(); ++gi) {
    if (!alive[gi]) continue;
    const Group& g = groups[gi];
    const auto& f = flagged[g.peak];
    const std::int64_t b = f[0], k = f[1], l = f[2];
    double m = 0.0, a = 0.0, z = 0.0;
    at(b, k, l, m);
    const double dk = (at(b, k - 1, l, a) && at(b, k + 1, l, z)) ? parabola(a, m, z) : 0.0;
    const double db = (at(b - 1, k, l, a) && at(b + 1, k, l, z)) ? parabola(a, m, z) : 0.0;
    const double dl = (at(b, k, l - 1, a) && at(b, k, l + 1, z)) ? parabola(a, m, z) : 0.0;
    ToneDetection d;
    d.beam = b;
    d.hypothesis = k;
    d.lag = l;
    d.cells = g.cells;
    d.snr_db = 10.0 * std::log10(g.x);
    d.range_rate = p.rates[static_cast<std::size_t>(k)] + dk * p.step;
    d.bearing = bearings[static_cast<std::size_t>(b)] + db * bstep;
    d.range = 0.5 * c * (t0 + (static_cast<double>(l) + dl) * static_cast<double>(H) / st.fs - p.start);
    st.detections.push_back(d);
  }
  std::stable_sort(st.detections.begin(), st.detections.end(), [](const ToneDetection& x, const ToneDetection& y) { return x.snr_db > y.snr_db; });
  if (keep_map) {
    st.power = std::move(power);
    st.ratio = std::move(ratio);
  }
  return st;
}

std::vector<Detection> merge_tone(std::vector<Detection>& sweep, const ToneStage& tone) {
  std::vector<Detection> alone;
  const TonePlan& p = tone.plan;
  std::vector<char> taken(sweep.size(), 0);
  for (const ToneDetection& t : tone.detections) {          // strongest first (tone_stage sorts them)
    std::int64_t pick = -1;
    for (std::size_t i = 0; i < sweep.size(); ++i) {
      if (taken[i] || std::abs(sweep[i].range - t.range) > p.gate_range || std::abs(sweep[i].bearing - t.bearing) > p.gate_bearing) continue;
      if (pick < 0 || sweep[i].snr_db > sweep[static_cast<std::size_t>(pick)].snr_db) pick = static_cast<std::int64_t>(i);
    }
    if (pick >= 0) {
      Detection& d = sweep[static_cast<std::size_t>(pick)];
      taken[static_cast<std::size_t>(pick)] = 1;
      d.range_rate = t.range_rate;
      d.range_rate_step = p.step;
      d.doppler_path = 2;
    } else {
      Detection d{t.range, t.bearing, t.snr_db, t.cells};
      d.range_rate = t.range_rate;
      d.range_rate_step = p.step;
      d.range_sigma = p.range_sigma;
      d.doppler_path = 1;
      alone.push_back(d);
    }
  }
  return alone;
}

std::int64_t sweep_only_replica(Waveform& w, const TonePlan& p) {
  if (!p.on || p.tone_only || w.type != WaveformType::Arbitrary || w.replica.empty() || !(w.replica_sample_rate > 0.0)) return 0;
  std::int64_t zeroed = 0;
  for (std::size_t m = 0; m < w.replica.size(); ++m) {
    const double t = static_cast<double>(m) / w.replica_sample_rate;
    if (t >= p.start && t <= p.start + p.duration) {
      w.replica[m] = 0.0F;
      ++zeroed;
    }
  }
  return zeroed;
}

}  // namespace sonar::live
