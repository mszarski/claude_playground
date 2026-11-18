// Goertzel lines and DEMON on a stream of records. Every step mirrors python/gpusonar/tonal.py,
// where the derivations are; a change here is a change there.
#include "sonar/live/tonal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/live/chain.hpp"
#include "sonar/live/elevation.hpp"

namespace sonar::live {
namespace {

using Cd = std::complex<double>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

// Loops too short to be worth threads (a few hundred bearings): starting a parallel region costs
// more than they do, and far more on a machine whose cores are busy.
template <typename F>
void serial_for(std::int64_t n, const F& f) {
  for (std::int64_t i = 0; i < n; ++i) f(i);
}

std::int64_t rint_up(double x) { return static_cast<std::int64_t>(std::floor(x + 0.5)); }

// Where a line sits between two bins of a Hann-weighted transform, from the ratio of the smaller
// magnitude to the larger; and the bin's response to a line that far away (tonal.py).
double hann_offset(double ratio) { return std::clamp((2.0 * ratio - 1.0) / (ratio + 1.0), 0.0, 1.0); }
double hann_gain(double offset) {
  if (std::abs(1.0 - offset * offset) < 1e-9) return 0.5;
  const double x = kPi * offset;
  return (offset == 0.0 ? 1.0 : std::sin(x) / x) / (1.0 - offset * offset);
}

double gamma_tail(std::int64_t n, double x) {
  double term = std::exp(-x), total = term;
  for (std::int64_t k = 1; k < n; ++k) {
    term *= x / static_cast<double>(k);
    total += term;
  }
  return total;
}

// P(X / Y > t): X the mean of n unit exponentials, Y a Gamma variable of shape m and mean 1 (tonal.py).
double ratio_tail(std::int64_t n, double m, double t) {
  const double a = static_cast<double>(n) * t / m;
  double term = std::pow(1.0 + a, -m), total = term;
  for (std::int64_t k = 1; k < n; ++k) {
    term *= a * (m + static_cast<double>(k) - 1.0) / (static_cast<double>(k) * (1.0 + a));
    total += term;
  }
  return total;
}

// Mean of s[j] over the bins j with guard < |j - k| <= guard + width inside [lo, hi], for every k.
std::vector<double> split_mean(const std::vector<double>& s, std::int64_t guard, std::int64_t width, std::int64_t lo, std::int64_t hi) {
  const auto n = static_cast<std::int64_t>(s.size());
  std::vector<double> cs(s.size() + 1, 0.0), out(s.size(), 0.0);
  for (std::int64_t i = 0; i < n; ++i) cs[static_cast<std::size_t>(i + 1)] = cs[static_cast<std::size_t>(i)] + s[static_cast<std::size_t>(i)];
  for (std::int64_t k = 0; k < n; ++k) {
    const std::int64_t a0 = std::max(k - guard - width, lo), a1 = std::min(k - guard - 1, hi);
    const std::int64_t b0 = std::max(k + guard + 1, lo), b1 = std::min(k + guard + width, hi);
    std::int64_t count = 0;
    double total = 0.0;
    if (a1 >= a0) {
      total += cs[static_cast<std::size_t>(a1 + 1)] - cs[static_cast<std::size_t>(a0)];
      count += a1 - a0 + 1;
    }
    if (b1 >= b0) {
      total += cs[static_cast<std::size_t>(b1 + 1)] - cs[static_cast<std::size_t>(b0)];
      count += b1 - b0 + 1;
    }
    if (count > 0) out[static_cast<std::size_t>(k)] = total / static_cast<double>(count);
  }
  return out;
}

// Goertzel recurrences at several frequencies in one pass over real samples: sums[f] = sum x[n] exp(-j w_f n).
void goertzel_bank(const float* x, std::int64_t n, const std::vector<double>& w, std::vector<Cd>& sums) {
  const std::size_t nf = w.size();
  std::vector<double> coeff(nf), s1(nf, 0.0), s2(nf, 0.0);
  for (std::size_t f = 0; f < nf; ++f) coeff[f] = 2.0 * std::cos(w[f]);
  for (std::int64_t i = 0; i < n; ++i) {
    const auto v = static_cast<double>(x[i]);
    for (std::size_t f = 0; f < nf; ++f) {
      const double s0 = v + coeff[f] * s1[f] - s2[f];
      s2[f] = s1[f];
      s1[f] = s0;
    }
  }
  sums.resize(nf);
  for (std::size_t f = 0; f < nf; ++f) {
    // y = s[N-1] - exp(-jw) s[N-2] = exp(jw(N-1)) X
    sums[f] = (Cd(s1[f], 0.0) - std::polar(1.0, -w[f]) * s2[f]) * std::polar(1.0, -w[f] * static_cast<double>(n - 1));
  }
}

std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream s;
  s << std::setprecision(12) << v;
  return s.str();
}

std::int64_t nearest_peak(const std::vector<double>* peaks_deg, double bearing_deg, double attach_deg) {
  if (!peaks_deg || peaks_deg->empty()) return -1;
  std::size_t best = 0;
  for (std::size_t i = 1; i < peaks_deg->size(); ++i) {
    if (std::abs((*peaks_deg)[i] - bearing_deg) < std::abs((*peaks_deg)[best] - bearing_deg)) best = i;
  }
  return std::abs((*peaks_deg)[best] - bearing_deg) <= attach_deg ? static_cast<std::int64_t>(best) : -1;
}

// One event as a JSON object. `in_frame`: as it goes into a passive frame (no "kind", detected
// lines only, "peak" on every line and source).
std::string event_json(const TonalEvent& e, const TonalAnalyzer& a, bool in_frame, const std::vector<double>* peaks_deg) {
  const double attach = a.options().attach_deg;
  std::ostringstream s;
  s << "{";
  if (!in_frame) s << "\"kind\": \"" << (e.demon ? "demon" : "goertzel") << "\", ";
  s << "\"time_ns\": " << e.time_ns;
  if (!e.demon) {
    s << ", \"block_s\": " << num(e.block_s) << ", \"lines\": [";
    bool first = true;
    for (const TonalLine& q : e.lines) {
      if (in_frame && !q.detected) continue;
      s << (first ? "" : ", ") << "{\"nominal_hz\": " << num(q.nominal) << ", \"detected\": " << (q.detected ? "true" : "false")
        << ", \"frequency_hz\": " << num(q.frequency) << ", \"amplitude\": " << num(q.amplitude) << ", \"level_db\": "
        << num(20.0 * std::log10(std::max(q.amplitude, 1e-300))) << ", \"phase_rad\": " << num(q.phase) << ", \"snr_db\": " << num(q.snr_db)
        << ", \"noise_db\": " << num(q.noise_db) << ", \"bearing_deg\": " << (q.has_bearing ? num(q.bearing / kDeg) : "null")
        << ", \"track_hz\": " << (q.tracking ? num(q.track) : "null") << ", \"rate_hz_s\": " << (q.tracking ? num(q.rate) : "null");
      if (in_frame) s << ", \"peak\": " << (q.has_bearing ? nearest_peak(peaks_deg, q.bearing / kDeg, attach) : -1);
      s << "}";
      first = false;
    }
    s << "]}";
    return s.str();
  }
  s << ", \"segments\": " << e.segments << ", \"independent\": " << num(e.independent) << ", \"threshold_db\": " << num(e.threshold_db)
    << ", \"sources\": [";
  for (std::size_t i = 0; i < e.sources.size(); ++i) {
    const DemonSource& q = e.sources[i];
    s << (i ? ", " : "") << "{\"bearing_deg\": " << num(q.bearing / kDeg) << ", \"shaft_hz\": " << (q.shaft_hz > 0.0 ? num(q.shaft_hz) : "null")
      << ", \"blade_hz\": " << (q.blade_hz > 0.0 ? num(q.blade_hz) : "null") << ", \"blades\": ";
    if (q.blades > 0) s << q.blades;
    else s << "null";
    s << ", \"confidence\": " << num(q.confidence) << ", \"lines\": [";
    for (std::size_t j = 0; j < q.lines.size(); ++j) {
      const DemonLine& l = q.lines[j];
      s << (j ? ", " : "") << "{\"frequency_hz\": " << num(l.frequency) << ", \"snr_db\": " << num(l.snr_db) << ", \"modulation\": "
        << num(l.modulation) << ", \"harmonic\": " << l.harmonic << "}";
    }
    s << "]";
    if (in_frame) s << ", \"peak\": " << nearest_peak(peaks_deg, q.bearing / kDeg, attach);
    s << "}";
  }
  s << "]";
  if (!e.gram_db.empty()) {
    s << ", \"gram\": {\"bearings_deg\": [";
    for (std::size_t i = 0; i < a.bearings.size(); ++i) s << (i ? ", " : "") << num(a.bearings[i] / kDeg);
    s << "], \"frequencies_hz\": [";
    for (std::size_t i = 0; i < a.gram_frequencies.size(); ++i) s << (i ? ", " : "") << num(a.gram_frequencies[i]);
    s << "], \"db\": [";
    const std::size_t nf = a.gram_frequencies.size();
    for (std::size_t b = 0; b < a.bearings.size(); ++b) {
      s << (b ? ", [" : "[");
      for (std::size_t k = 0; k < nf; ++k) {
        s << (k ? ", " : "") << std::round(10.0 * static_cast<double>(e.gram_db[b * nf + k])) / 10.0;
      }
      s << "]";
    }
    s << "]}";
  }
  s << "}";
  return s.str();
}

std::vector<double> number_list(const std::string& value) {
  std::vector<double> out;
  std::size_t at = 0;
  while (at < value.size()) {
    const std::size_t comma = std::min(value.find(',', at), value.size());
    if (comma > at) out.push_back(std::atof(value.substr(at, comma - at).c_str()));
    at = comma + 1;
  }
  return out;
}

}  // namespace

bool set_tonal_option(TonalOptions& o, const std::string& key, const std::string& value) {
  if (key.rfind("tonal", 0) != 0) return false;
  const double v = std::atof(value.c_str());
  const auto i = static_cast<std::int64_t>(v);
  auto& g = o.goertzel;
  auto& d = o.demon;
  if (key == "tonal") o.enabled = v != 0.0;
  else if (key == "tonal_bearing_first_deg") o.bearing_first_deg = v;
  else if (key == "tonal_bearing_last_deg") o.bearing_last_deg = v;
  else if (key == "tonal_bearing_step_deg") o.bearing_step_deg = v;
  else if (key == "tonal_window") o.hann = value == "hann";
  else if (key == "tonal_row_tolerance_m") o.row_tolerance_m = v;
  else if (key == "tonal_attach_deg") o.attach_deg = v;
  else if (key == "tonal_goertzel_lines") {
    g.lines = number_list(value);
    g.enabled = !g.lines.empty();
  } else if (key == "tonal_goertzel_on") g.beams = value != "hydrophones";
  else if (key == "tonal_goertzel_block_s") g.block_s = v;
  else if (key == "tonal_goertzel_hop_s") g.hop_s = v;
  else if (key == "tonal_goertzel_search_bins") g.search_bins = i;
  else if (key == "tonal_goertzel_guard_bins") g.guard_bins = i;
  else if (key == "tonal_goertzel_noise_bins") g.noise_bins = i;
  else if (key == "tonal_goertzel_min_snr_db") g.min_snr_db = v;
  else if (key == "tonal_goertzel_track") g.track = v != 0.0;
  else if (key == "tonal_goertzel_alpha") g.alpha = v;
  else if (key == "tonal_goertzel_beta") g.beta = v;
  else if (key == "tonal_goertzel_max_misses") g.max_misses = i;
  else if (key == "tonal_goertzel_max_shift_hz") g.max_shift_hz = v;
  else if (key == "tonal_demon") d.enabled = v != 0.0;
  else if (key == "tonal_demon_f_low") d.f_low = v;
  else if (key == "tonal_demon_f_high") d.f_high = v;
  else if (key == "tonal_demon_envelope") d.square = value != "hilbert";
  else if (key == "tonal_demon_envelope_rate_hz") d.envelope_rate_hz = v;
  else if (key == "tonal_demon_window_hops") d.window_hops = i;
  else if (key == "tonal_demon_segment_s") d.segment_s = v;
  else if (key == "tonal_demon_segment_overlap") d.segment_overlap = v;
  else if (key == "tonal_demon_n_average") d.n_average = i;
  else if (key == "tonal_demon_min_segments") d.min_segments = i;
  else if (key == "tonal_demon_f_min_hz") d.f_min_hz = v;
  else if (key == "tonal_demon_f_max_hz") d.f_max_hz = v;
  else if (key == "tonal_demon_background_bins") d.background_bins = i;
  else if (key == "tonal_demon_guard_bins") d.guard_bins = i;
  else if (key == "tonal_demon_clip") d.clip = v;
  else if (key == "tonal_demon_pfa") d.pfa = v;
  else if (key == "tonal_demon_max_lines") d.max_lines = i;
  else if (key == "tonal_demon_max_sources") d.max_sources = i;
  else if (key == "tonal_demon_min_separation_deg") d.min_separation_deg = v;
  else if (key == "tonal_demon_shaft_min_hz") d.shaft_min_hz = v;
  else if (key == "tonal_demon_shaft_max_hz") d.shaft_max_hz = v;
  else if (key == "tonal_demon_max_harmonic") d.max_harmonic = i;
  else if (key == "tonal_demon_harmonic_tolerance_bins") d.harmonic_tolerance_bins = v;
  else if (key == "tonal_demon_score_fraction") d.score_fraction = v;
  else if (key == "tonal_demon_blades_min") d.blades_min = i;
  else if (key == "tonal_demon_blades_max") d.blades_max = i;
  else if (key == "tonal_demon_gram") d.gram = v != 0.0;
  else throw std::invalid_argument("unknown option " + key);
  return true;
}

std::complex<double> goertzel(const float* x, std::int64_t n, double frequency, double sample_rate) {
  std::vector<Cd> sums;
  goertzel_bank(x, n, {2.0 * kPi * frequency / sample_rate}, sums);
  return sums[0];
}

std::complex<double> goertzel(const std::complex<float>* x, std::int64_t n, double frequency, double sample_rate) {
  // The coefficient is real: the recurrence runs on the real and imaginary parts separately.
  std::vector<float> part(static_cast<std::size_t>(n));
  std::vector<Cd> re, im;
  for (std::int64_t i = 0; i < n; ++i) part[static_cast<std::size_t>(i)] = x[i].real();
  goertzel_bank(part.data(), n, {2.0 * kPi * frequency / sample_rate}, re);
  for (std::int64_t i = 0; i < n; ++i) part[static_cast<std::size_t>(i)] = x[i].imag();
  goertzel_bank(part.data(), n, {2.0 * kPi * frequency / sample_rate}, im);
  return re[0] + Cd(0.0, 1.0) * im[0];
}

double demon_threshold(double pfa, std::int64_t n_spectra, double correlation, std::int64_t background, double* independent) {
  const auto k = static_cast<double>(std::max<std::int64_t>(n_spectra, 1));
  const double n = std::max(k / (1.0 + 2.0 * correlation * (k - 1.0) / k), 1.0);
  if (independent) *independent = n;
  auto solve = [&](std::int64_t whole) {
    double lo = 1.0, hi = 1e3;
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      // neighbouring Hann-weighted bins are correlated: `background` of them count as 18/35 as many (tonal.py)
      const double tail = background > 0 ? ratio_tail(whole, static_cast<double>(whole * background) * 18.0 / 35.0, mid)
                                         : gamma_tail(whole, static_cast<double>(whole) * mid);
      if (tail > pfa) lo = mid;
      else hi = mid;
    }
    return 0.5 * (lo + hi);
  };
  const auto below = static_cast<std::int64_t>(std::floor(n));
  double t = solve(below);
  if (n > static_cast<double>(below)) t += (n - static_cast<double>(below)) * (solve(below + 1) - t);
  return t;
}

void harmonic_set(DemonSource& src, double df, const TonalOptions::Demon& o) {
  const std::size_t n_lines = src.lines.size();
  src.shaft_hz = src.blade_hz = src.confidence = 0.0;
  src.blades = 0;
  for (DemonLine& l : src.lines) l.harmonic = 0;
  if (n_lines == 0) return;
  const double tol = o.harmonic_tolerance_bins * df;
  auto match = [&](double f0, std::int64_t k, std::vector<std::int64_t>& h) {
    double score = 0.0;
    h.assign(n_lines, 0);
    for (std::size_t j = 0; j < n_lines; ++j) {
      const double f = src.lines[j].frequency;
      const double hn = std::floor(f / f0 + 0.5);
      if (hn >= 1.0 && hn <= static_cast<double>(o.max_harmonic) &&
          std::abs(f - hn * f0) <= std::min(tol * std::max(1.0, hn / static_cast<double>(k)), 0.25 * f0)) {
        h[j] = static_cast<std::int64_t>(hn);
        score += src.lines[j].snr_db;
      }
    }
    return score;
  };
  struct Cand {
    double score, f0;
    std::int64_t k;
  };
  std::vector<Cand> cands;
  std::vector<std::int64_t> h;
  double best = -1.0;
  for (std::size_t i = 0; i < n_lines; ++i) {
    for (std::int64_t k = 1; k <= o.max_harmonic; ++k) {
      const double f0 = src.lines[i].frequency / static_cast<double>(k);
      if (f0 < o.shaft_min_hz || f0 > o.shaft_max_hz) continue;
      const double score = match(f0, k, h);
      cands.push_back({score, f0, k});
      best = std::max(best, score);
    }
  }
  if (cands.empty()) return;
  const Cand* pick = nullptr;
  for (const Cand& c : cands) {
    if (c.score >= o.score_fraction * best && (!pick || c.f0 > pick->f0)) pick = &c;
  }
  match(pick->f0, pick->k, h);
  double num_ = 0.0, den = 0.0, in_weight = 0.0, all_weight = 0.0, strongest_w = -std::numeric_limits<double>::infinity();
  std::int64_t n = 0, strongest = 0;
  for (std::size_t j = 0; j < n_lines; ++j) {
    const double w = src.lines[j].snr_db;
    all_weight += w;
    src.lines[j].harmonic = h[j];
    if (h[j] == 0) continue;
    const auto hd = static_cast<double>(h[j]);
    num_ += w * hd * src.lines[j].frequency;
    den += w * hd * hd;
    in_weight += w;
    ++n;
    if (w > strongest_w) {
      strongest_w = w;
      strongest = h[j];
    }
  }
  if (n < 2) return;  // one line is a line, not a rate
  src.shaft_hz = num_ / den;
  src.confidence = in_weight / all_weight * (1.0 - std::pow(0.5, static_cast<double>(n - 1)));
  if (strongest >= o.blades_min && strongest <= o.blades_max && strongest >= 2) {
    src.blades = strongest;
    src.blade_hz = static_cast<double>(strongest) * src.shaft_hz;
  }
}

TonalAnalyzer::TonalAnalyzer(const TonalOptions& options, double f_low, double f_high)
    : opt_(options), band_low_(f_low), band_high_(f_high) {
  const auto& g = opt_.goertzel;
  if (g.enabled && (g.guard_bins < 1 || g.noise_bins < 1 || g.search_bins < 0)) {
    throw std::invalid_argument("tonal: goertzel guard_bins and noise_bins must be at least 1");
  }
  if (opt_.bearing_step_deg <= 0.0) throw std::invalid_argument("tonal: the bearing step must be positive");
  const auto count = static_cast<std::int64_t>(
      std::ceil((opt_.bearing_last_deg + 0.5 * opt_.bearing_step_deg - opt_.bearing_first_deg) / opt_.bearing_step_deg));
  bearings.assign(static_cast<std::size_t>(std::max<std::int64_t>(count, 0)), 0.0);
  // first + i * step, as numpy.arange makes it
  for (std::size_t i = 0; i < bearings.size(); ++i) {
    bearings[i] = (opt_.bearing_first_deg + static_cast<double>(i) * opt_.bearing_step_deg) * kDeg;
  }
  if (bearings.empty()) throw std::invalid_argument("tonal: empty bearing grid");
  if (g.enabled) {
    for (double f : g.lines) lines_.push_back({f, f, 0.0, false, 0});
  }
}

std::int64_t TonalAnalyzer::time_ns(std::int64_t index) const { return run_ns_ + rint_up(static_cast<double>(index) / fs_ * 1e9); }

void TonalAnalyzer::setup(const PingRecord& record) {
  const Samples& s = record.samples;
  fs_ = s.sample_rate;
  is_complex_ = s.is_complex();
  carrier_ = is_complex_ ? s.center_frequency : 0.0;
  c_ = record.environment.sound_speed;
  n_channels_ = s.n_channels;
  rows_ = find_rows(record, row_tolerance(record, opt_.row_tolerance_m));
  if (rows_.empty()) {
    rows_.emplace_back();
    for (std::int64_t i = 0; i < n_channels_; ++i) rows_[0].push_back(i);
  }
  if (rows_[0].size() < 2) throw std::invalid_argument("tonal: need at least two hydrophones in a row");
  const auto nc = static_cast<std::size_t>(n_channels_);
  std::vector<Vec3d> pos(nc);
  for (std::size_t i = 0; i < nc; ++i) pos[i] = world_position(record.geometry.rx_pose, record.geometry.rx[i].lever_arm);
  Vec3d centre{0.0, 0.0, 0.0};
  std::size_t used = 0;
  for (const auto& r : rows_) used += r.size();
  for (const auto& r : rows_) {
    for (std::int64_t i : r) centre = centre + pos[static_cast<std::size_t>(i)] / static_cast<double>(used);
  }
  const Vec3d axis = normalized(pos[static_cast<std::size_t>(rows_[0].back())] - pos[static_cast<std::size_t>(rows_[0].front())]);
  u_.assign(nc, 0.0);
  weights_.assign(nc, 0.0);
  for (std::size_t i = 0; i < nc; ++i) u_[i] = dot(pos[i] - centre, axis);
  for (const auto& r : rows_) {
    double total = 0.0;
    for (std::size_t j = 0; j < r.size(); ++j) {
      const double w = opt_.hann ? 0.5 - 0.5 * std::cos(2.0 * kPi * (static_cast<double>(j) + 0.5) / static_cast<double>(r.size())) : 1.0;
      weights_[static_cast<std::size_t>(r[j])] = w;
      total += w;
    }
    for (std::int64_t i : r) weights_[static_cast<std::size_t>(i)] /= total;
  }
  re_.assign(nc, {});
  im_.assign(is_complex_ ? nc : 0, {});
  first_ = 0;
  const auto& g = opt_.goertzel;
  if (g.enabled) {
    g_n_ = std::max<std::int64_t>(rint_up(g.block_s * fs_), 8);
    g_hop_ = std::max<std::int64_t>(rint_up((g.hop_s > 0.0 ? g.hop_s : g.block_s) * fs_), 1);
    g_next_ = 0;
    for (const LineState& line : lines_) {
      const double fb = line.nominal - carrier_;
      if (is_complex_ ? std::abs(fb) >= 0.5 * fs_ : (fb <= 0.0 || fb >= 0.5 * fs_)) {
        throw std::invalid_argument("tonal: a Goertzel line is outside the sampled band");
      }
    }
  }
  const auto& d = opt_.demon;
  if (d.enabled) {
    double f_low = d.f_low, f_high = d.f_high;
    if (f_low <= 0.0 && f_high <= 0.0) {
      f_low = band_low_;
      f_high = band_high_;
    }
    if (f_high <= 0.0) {
      const Waveform& w = record.transmit.waveform;
      f_low = std::min(w.f_start, w.f_stop);
      f_high = std::max(w.f_start, w.f_stop);
    }
    d_hop_ = std::max<std::int64_t>(rint_up(fs_ / d.envelope_rate_hz), 1);
    d_n_ = d_hop_ * d.window_hops;
    d_rate_ = fs_ / static_cast<double>(d_hop_);
    d_next_ = 0;
    d_bins_.clear();
    std::vector<double> freqs;
    for (std::int64_t k = 0; k < d_n_; ++k) {
      double f;
      if (is_complex_) {
        f = carrier_ + static_cast<double>(k <= d_n_ / 2 ? k : k - d_n_) * fs_ / static_cast<double>(d_n_);
      } else {
        if (k > d_n_ / 2) break;
        f = static_cast<double>(k) * fs_ / static_cast<double>(d_n_);
      }
      if (f >= f_low && f <= f_high && f > 0.0) {
        d_bins_.push_back(k);
        freqs.push_back(f);
      }
    }
    if (d_bins_.empty()) throw std::invalid_argument("tonal: no bin of the DEMON envelope blocks inside the band asked for");
    d_window_.resize(static_cast<std::size_t>(d_n_));
    for (std::int64_t k = 0; k < d_n_; ++k) {
      d_window_[static_cast<std::size_t>(k)] = 0.5 - 0.5 * std::cos(2.0 * kPi * (static_cast<double>(k) + 0.5) / static_cast<double>(d_n_));
    }
    const std::size_t nb = bearings.size(), nf = d_bins_.size();
    d_steer_re_.resize(nb * nc * nf);
    d_steer_im_.resize(nb * nc * nf);
    for (std::size_t b = 0; b < nb; ++b) {
      const double sb = std::sin(bearings[b]);
      for (std::size_t ch = 0; ch < nc; ++ch) {
        for (std::size_t f = 0; f < nf; ++f) {
          const Cd v = std::polar(weights_[ch], -2.0 * kPi * freqs[f] * u_[ch] * sb / c_);
          d_steer_re_[(b * nc + ch) * nf + f] = v.real();
          d_steer_im_[(b * nc + ch) * nf + f] = v.imag();
        }
      }
    }
    s_n_ = std::max<std::int64_t>(rint_up(d.segment_s * d_rate_), 16);
    s_hop_ = std::max<std::int64_t>(rint_up(static_cast<double>(s_n_) * (1.0 - d.segment_overlap)), 1);
    s_window_.resize(static_cast<std::size_t>(s_n_));
    double energy = 0.0, lag = 0.0;
    for (std::int64_t k = 0; k < s_n_; ++k) {
      s_window_[static_cast<std::size_t>(k)] = 0.5 - 0.5 * std::cos(2.0 * kPi * (static_cast<double>(k) + 0.5) / static_cast<double>(s_n_));
      energy += s_window_[static_cast<std::size_t>(k)] * s_window_[static_cast<std::size_t>(k)];
    }
    for (std::int64_t k = 0; k + s_hop_ < s_n_; ++k) lag += s_window_[static_cast<std::size_t>(k + s_hop_)] * s_window_[static_cast<std::size_t>(k)];
    s_correlation_ = (lag / energy) * (lag / energy);
    s_df_ = d_rate_ / static_cast<double>(s_n_);
    const double f_max = d.f_max_hz > 0.0 ? d.f_max_hz : 0.8 * 0.5 * d_rate_;
    k_lo_ = std::max<std::int64_t>(static_cast<std::int64_t>(std::ceil(d.f_min_hz / s_df_ - 1e-9)), 2);
    k_hi_ = std::min<std::int64_t>(static_cast<std::int64_t>(std::floor(f_max / s_df_ + 1e-9)), s_n_ / 2 - 1);
    env_.clear();
    spectra_.clear();
  }
}

std::vector<TonalEvent> TonalAnalyzer::push(const PingRecord& record) {
  const Samples& s = record.samples;
  const std::int64_t start_ns = record.transmit.time_ns + rint_up(s.start_time * 1e9);
  const bool same = have_key_ && s.sample_rate == fs_ && s.n_channels == n_channels_ && s.is_complex() == is_complex_ &&
                    s.center_frequency == key_center_ && record.environment.sound_speed == c_;
  if (!same) {
    have_key_ = true;
    key_center_ = s.center_frequency;
    setup(record);
    run_ns_ = start_ns;
  } else if (std::abs(static_cast<double>(start_ns - next_ns_)) > 0.5e9 / fs_) {
    // A gap: what is half filled is dropped; averages and tracks carry on.
    for (auto& v : re_) v.clear();
    for (auto& v : im_) v.clear();
    first_ = 0;
    run_ns_ = start_ns;
    g_next_ = 0;
    d_next_ = 0;
    env_.clear();
  }
  for (std::int64_t ch = 0; ch < n_channels_; ++ch) {
    auto& re = re_[static_cast<std::size_t>(ch)];
    if (is_complex_) {
      auto& im = im_[static_cast<std::size_t>(ch)];
      const std::complex<float>* x = s.channel_complex(ch);
      for (std::int64_t i = 0; i < s.n_samples; ++i) {
        re.push_back(x[i].real());
        im.push_back(x[i].imag());
      }
    } else {
      re.insert(re.end(), s.channel(ch), s.channel(ch) + s.n_samples);
    }
  }
  const std::int64_t end = first_ + static_cast<std::int64_t>(re_[0].size());
  next_ns_ = run_ns_ + rint_up(static_cast<double>(end) / fs_ * 1e9);
  std::vector<TonalEvent> events;
  std::int64_t keep = end;
  if (opt_.goertzel.enabled) {
    while (g_next_ + g_n_ <= end) {
      events.push_back(goertzel_block(g_next_ - first_, g_next_ + g_n_));
      g_next_ += g_hop_;
    }
    keep = std::min(keep, g_next_);
  }
  if (opt_.demon.enabled) {
    const std::int64_t n_blocks = end - d_next_ >= d_n_ ? (end - d_next_ - d_n_) / d_hop_ + 1 : 0;
    if (n_blocks > 0) demon_blocks(n_blocks, events);
    keep = std::min(keep, d_next_);
  }
  if (keep > first_) {
    const auto drop = static_cast<std::ptrdiff_t>(keep - first_);
    for (auto& v : re_) v.erase(v.begin(), v.begin() + drop);
    for (auto& v : im_) v.erase(v.begin(), v.begin() + drop);
    first_ = keep;
  }
  std::stable_sort(events.begin(), events.end(), [](const TonalEvent& a, const TonalEvent& b) { return a.time_ns < b.time_ns; });
  return events;
}

TonalEvent TonalAnalyzer::goertzel_block(std::int64_t first, std::int64_t end_index) {
  const auto& g = opt_.goertzel;
  const std::int64_t n = g_n_, m_half = g.search_bins + g.guard_bins + g.noise_bins;
  const std::int64_t n_rect = 2 * m_half + 3, n_h = 2 * m_half + 1;
  const double dt = static_cast<double>(g_hop_) / fs_;
  const double scale = (is_complex_ ? 1.0 : 2.0) / (0.5 * static_cast<double>(n));
  const auto nc = static_cast<std::size_t>(n_channels_);
  TonalEvent ev;
  ev.time_ns = time_ns(end_index);
  ev.block_s = static_cast<double>(n) / fs_;
  std::vector<Cd> h(nc * static_cast<std::size_t>(n_h));
  std::vector<double> f_abs(static_cast<std::size_t>(n_h)), w(static_cast<std::size_t>(n_rect));
  for (LineState& line : lines_) {
    const double centre = (g.track && line.tracking) ? line.frequency + line.rate * dt : line.nominal;
    for (std::int64_t j = 0; j < n_rect; ++j) {
      const double f = centre + static_cast<double>(j - m_half - 1) * fs_ / static_cast<double>(n);
      w[static_cast<std::size_t>(j)] = 2.0 * kPi * (f - carrier_) / fs_;
      if (j >= 1 && j <= n_h) f_abs[static_cast<std::size_t>(j - 1)] = f;
    }
    parallel_for_host(n_channels_, [&](std::int64_t ch) {
      std::vector<Cd> rect, imag;
      goertzel_bank(re_[static_cast<std::size_t>(ch)].data() + first, n, w, rect);
      if (is_complex_) {
        goertzel_bank(im_[static_cast<std::size_t>(ch)].data() + first, n, w, imag);
        for (std::size_t j = 0; j < rect.size(); ++j) rect[j] += Cd(0.0, 1.0) * imag[j];
      }
      // A Hann window is the rectangular one's bin less half its two neighbours.
      for (std::int64_t j = 0; j < n_h; ++j) {
        const auto q = static_cast<std::size_t>(j);
        h[static_cast<std::size_t>(ch) * static_cast<std::size_t>(n_h) + q] = scale * (0.5 * rect[q + 1] - 0.25 * (rect[q] + rect[q + 2]));
      }
    });
    // Mean over the rows of |row steered at a bearing|^2 per bin; `beam0` the first row's beam.
    auto beam_power = [&](double sin_bearing, std::vector<double>& p, std::vector<Cd>* beam0) {
      p.assign(static_cast<std::size_t>(n_h), 0.0);
      for (std::size_t r = 0; r < rows_.size(); ++r) {
        for (std::int64_t j = 0; j < n_h; ++j) {
          Cd beam{};
          for (std::int64_t ch : rows_[r]) {
            const auto c = static_cast<std::size_t>(ch);
            beam += weights_[c] * h[c * static_cast<std::size_t>(n_h) + static_cast<std::size_t>(j)] *
                    std::polar(1.0, -2.0 * kPi * f_abs[static_cast<std::size_t>(j)] * u_[c] * sin_bearing / c_);
          }
          p[static_cast<std::size_t>(j)] += std::norm(beam) / static_cast<double>(rows_.size());
          if (r == 0 && beam0) (*beam0)[static_cast<std::size_t>(j)] = beam;
        }
      }
    };
    const std::int64_t lo = m_half - g.search_bins, hi = m_half + g.search_bins + 1;
    std::vector<double> p;
    std::vector<Cd> first_beam(static_cast<std::size_t>(n_h));
    TonalLine out;
    out.nominal = line.nominal;
    if (g.beams) {
      const std::size_t nb = bearings.size();
      std::vector<std::vector<double>> power(nb);
      serial_for(static_cast<std::int64_t>(nb), [&](std::int64_t b) {
        beam_power(std::sin(bearings[static_cast<std::size_t>(b)]), power[static_cast<std::size_t>(b)], nullptr);
      });
      std::size_t ib = 0;
      std::int64_t im = lo;
      double top = -1.0;
      for (std::size_t b = 0; b < nb; ++b) {
        for (std::int64_t j = lo; j < hi; ++j) {
          if (power[b][static_cast<std::size_t>(j)] > top) {
            top = power[b][static_cast<std::size_t>(j)];
            ib = b;
            im = j;
          }
        }
      }
      double frac = 0.0;
      if (ib > 0 && ib + 1 < nb) {
        const double a = power[ib - 1][static_cast<std::size_t>(im)], b0 = power[ib][static_cast<std::size_t>(im)],
                     c = power[ib + 1][static_cast<std::size_t>(im)];
        const double den = a - 2.0 * b0 + c;
        if (den < 0.0) frac = std::clamp(0.5 * (a - c) / den, -0.5, 0.5);
      }
      out.bearing = nb > 1 ? bearings[ib] + frac * (bearings[1] - bearings[0]) : bearings[ib];
      out.has_bearing = true;
      beam_power(std::sin(out.bearing), p, &first_beam);
    } else {
      p.assign(static_cast<std::size_t>(n_h), 0.0);
      std::size_t used = 0;
      for (const auto& r : rows_) used += r.size();
      for (const auto& r : rows_) {
        for (std::int64_t ch : r) {
          for (std::int64_t j = 0; j < n_h; ++j) {
            p[static_cast<std::size_t>(j)] += std::norm(h[static_cast<std::size_t>(ch) * static_cast<std::size_t>(n_h) + static_cast<std::size_t>(j)]) /
                                              static_cast<double>(used);
          }
        }
      }
      for (std::int64_t j = 0; j < n_h; ++j) {
        first_beam[static_cast<std::size_t>(j)] = h[static_cast<std::size_t>(rows_[0][0]) * static_cast<std::size_t>(n_h) + static_cast<std::size_t>(j)];
      }
    }
    std::int64_t k = lo;
    for (std::int64_t j = lo; j < hi; ++j) {
      if (p[static_cast<std::size_t>(j)] > p[static_cast<std::size_t>(k)]) k = j;
    }
    const auto ku = static_cast<std::size_t>(k);
    const int side = p[ku + 1] >= p[ku - 1] ? 1 : -1;
    const std::size_t kn = side > 0 ? ku + 1 : ku - 1;
    double delta = p[ku] > 0.0 ? hann_offset(std::sqrt(p[kn] / p[ku])) : 0.0;
    delta = std::min(delta, 0.5);
    out.frequency = f_abs[ku] + side * delta * fs_ / static_cast<double>(n);
    const double gain = hann_gain(delta), signal = p[ku] / (gain * gain);
    double floor = 0.0;
    for (std::int64_t j = 0; j < g.noise_bins; ++j) floor += p[static_cast<std::size_t>(j)];
    for (std::int64_t j = n_h - g.noise_bins; j < n_h; ++j) floor += p[static_cast<std::size_t>(j)];
    floor /= static_cast<double>(2 * g.noise_bins);
    const double snr = floor > 0.0 ? std::max(signal / floor - 1.0, 1e-6) : 1e-6;
    out.snr_db = 10.0 * std::log10(snr);
    out.noise_db = 10.0 * std::log10(std::max(floor, 1e-300));
    out.detected = out.snr_db >= g.min_snr_db;
    if (g.track) {
      if (out.detected) {
        if (line.tracking) {
          const double residual = out.frequency - centre;
          line.frequency = centre + g.alpha * residual;
          line.rate += g.beta * residual / dt;
        } else {
          line.frequency = out.frequency;
          line.rate = 0.0;
          line.tracking = true;
        }
        line.misses = 0;
      } else if (line.tracking) {
        line.frequency = centre;
        if (++line.misses > g.max_misses) {
          line.tracking = false;
          line.rate = 0.0;
          line.frequency = line.nominal;
        }
      }
      if (line.tracking && g.max_shift_hz > 0.0 && std::abs(line.frequency - line.nominal) > g.max_shift_hz) {
        line.tracking = false;
        line.rate = 0.0;
        line.frequency = line.nominal;
      }
    }
    out.amplitude = std::sqrt(signal);
    // At the middle of the block: the bin's phase is referred to the first sample, the window is centred n / 2 later.
    out.phase = std::arg(first_beam[ku] * std::polar(1.0, kPi * std::fmod((f_abs[ku] - carrier_) * static_cast<double>(n) / fs_, 2.0)));
    out.tracking = g.track && line.tracking;
    out.track = line.frequency;
    out.rate = line.rate;
    ev.lines.push_back(out);
  }
  return ev;
}

void TonalAnalyzer::demon_blocks(std::int64_t n_blocks, std::vector<TonalEvent>& events) {
  const auto& d = opt_.demon;
  const std::size_t nb = bearings.size(), nf = d_bins_.size(), nc = static_cast<std::size_t>(n_channels_);
  const std::int64_t i0 = d_next_ - first_;
  std::vector<double> env(static_cast<std::size_t>(n_blocks) * nb, 0.0);
  parallel_for_host(n_blocks, [&](std::int64_t k) {
    std::vector<std::complex<float>> work(static_cast<std::size_t>(d_n_));
    // Real and imaginary parts apart, so that the steering products are plain arithmetic the compiler can vectorise.
    std::vector<double> xr(nc * nf), xi(nc * nf), br(nf), bi(nf);
    const std::int64_t at = i0 + k * d_hop_;
    for (std::size_t ch = 0; ch < nc; ++ch) {
      const float* re = re_[ch].data() + at;
      const float* im = is_complex_ ? im_[ch].data() + at : nullptr;
      for (std::int64_t i = 0; i < d_n_; ++i) {
        const auto wv = static_cast<float>(d_window_[static_cast<std::size_t>(i)]);
        work[static_cast<std::size_t>(i)] = {re[i] * wv, im ? im[i] * wv : 0.0F};
      }
      dsp::fft(work.data(), work.size(), false);
      for (std::size_t f = 0; f < nf; ++f) {
        xr[ch * nf + f] = static_cast<double>(work[static_cast<std::size_t>(d_bins_[f])].real());
        xi[ch * nf + f] = static_cast<double>(work[static_cast<std::size_t>(d_bins_[f])].imag());
      }
    }
    for (std::size_t b = 0; b < nb; ++b) {
      double total = 0.0;
      for (const auto& r : rows_) {
        std::fill(br.begin(), br.end(), 0.0);
        std::fill(bi.begin(), bi.end(), 0.0);
        for (std::int64_t ch : r) {
          const double* sr = d_steer_re_.data() + (b * nc + static_cast<std::size_t>(ch)) * nf;
          const double* si = d_steer_im_.data() + (b * nc + static_cast<std::size_t>(ch)) * nf;
          const double* cr = xr.data() + static_cast<std::size_t>(ch) * nf;
          const double* ci = xi.data() + static_cast<std::size_t>(ch) * nf;
          for (std::size_t f = 0; f < nf; ++f) {
            br[f] += sr[f] * cr[f] - si[f] * ci[f];
            bi[f] += sr[f] * ci[f] + si[f] * cr[f];
          }
        }
        for (std::size_t f = 0; f < nf; ++f) total += br[f] * br[f] + bi[f] * bi[f];
      }
      env[static_cast<std::size_t>(k) * nb + b] = d.square ? total : std::sqrt(total);
    }
  });
  const std::int64_t first_block = d_next_;
  d_next_ += n_blocks * d_hop_;
  // Envelope sample j of this call ends at stream index first_block + j * hop + n.
  std::int64_t done = static_cast<std::int64_t>(env_.size() / nb);
  env_.insert(env_.end(), env.begin(), env.end());
  const auto sn = static_cast<std::size_t>(s_n_);
  double window_sum = 0.0;
  for (double v : s_window_) window_sum += v;
  const double scale = (2.0 / window_sum) * (2.0 / window_sum);
  while (static_cast<std::int64_t>(env_.size() / nb) >= s_n_) {
    const std::int64_t last = s_n_ - 1 - done;
    std::vector<double> per((sn / 2 + 1) * nb);
    serial_for(static_cast<std::int64_t>(nb), [&](std::int64_t bi) {
      const auto b = static_cast<std::size_t>(bi);
      double mean = 0.0;
      for (std::size_t i = 0; i < sn; ++i) mean += env_[i * nb + b];
      mean /= static_cast<double>(sn);
      std::vector<std::complex<float>> work(sn);
      for (std::size_t i = 0; i < sn; ++i) {
        work[i] = {mean > 0.0 ? static_cast<float>((env_[i * nb + b] / mean - 1.0) * s_window_[i]) : 0.0F, 0.0F};
      }
      dsp::fft(work.data(), work.size(), false);
      // Scaled so that a modulation 1 + m cos(...) of the envelope reads m^2 on its bin.
      for (std::size_t k = 0; k <= sn / 2; ++k) per[k * nb + b] = static_cast<double>(std::norm(work[k])) * scale;
    });
    spectra_.push_back(std::move(per));
    if (static_cast<std::int64_t>(spectra_.size()) > d.n_average) spectra_.pop_front();
    env_.erase(env_.begin(), env_.begin() + static_cast<std::ptrdiff_t>(static_cast<std::size_t>(s_hop_) * nb));
    done -= s_hop_;
    if (static_cast<std::int64_t>(spectra_.size()) >= d.min_segments) events.push_back(demon_result(first_block + last * d_hop_ + d_n_));
  }
}

TonalEvent TonalAnalyzer::demon_result(std::int64_t end_index) {
  const auto& d = opt_.demon;
  const std::size_t nb = bearings.size(), nf = static_cast<std::size_t>(s_n_) / 2 + 1;
  const auto top = static_cast<std::int64_t>(nf) - 1;
  std::vector<double> s(nf * nb, 0.0), bg(nf * nb), z(nf * nb);
  for (const auto& per : spectra_) {
    for (std::size_t i = 0; i < s.size(); ++i) s[i] += per[i] / static_cast<double>(spectra_.size());
  }
  serial_for(static_cast<std::int64_t>(nb), [&](std::int64_t bi) {
    const auto b = static_cast<std::size_t>(bi);
    std::vector<double> col(nf);
    for (std::size_t k = 0; k < nf; ++k) col[k] = s[k * nb + b];
    const std::vector<double> first = split_mean(col, d.guard_bins, d.background_bins, k_lo_, top);
    std::vector<double> clipped(nf);
    for (std::size_t k = 0; k < nf; ++k) clipped[k] = col[k] > d.clip * first[k] ? first[k] : col[k];
    const std::vector<double> second = split_mean(clipped, d.guard_bins, d.background_bins, k_lo_, top);
    for (std::size_t k = 0; k < nf; ++k) {
      bg[k * nb + b] = second[k];
      z[k * nb + b] = second[k] > 0.0 ? col[k] / second[k] : 0.0;
    }
  });
  TonalEvent ev;
  ev.demon = true;
  ev.time_ns = time_ns(end_index);
  ev.segments = static_cast<std::int64_t>(spectra_.size());
  const double threshold = demon_threshold(d.pfa, ev.segments, s_correlation_, 2 * d.background_bins, &ev.independent);
  ev.threshold_db = 10.0 * std::log10(threshold);
  const double step = nb > 1 ? bearings[1] - bearings[0] : 1.0;
  const std::int64_t sep = nb > 1 ? static_cast<std::int64_t>(std::floor(d.min_separation_deg * kDeg / step + 1e-9)) : 0;
  gram_frequencies.clear();
  for (std::int64_t k = k_lo_; k <= k_hi_; ++k) gram_frequencies.push_back(static_cast<double>(k) * s_df_);
  gram_db.assign(nb * gram_frequencies.size(), 0.0F);
  for (std::size_t b = 0; b < nb; ++b) {
    for (std::int64_t k = k_lo_; k <= k_hi_; ++k) {
      gram_db[b * gram_frequencies.size() + static_cast<std::size_t>(k - k_lo_)] =
          static_cast<float>(10.0 * std::log10(std::max(z[static_cast<std::size_t>(k) * nb + b], 1e-12)));
    }
  }
  if (d.gram) ev.gram_db = gram_db;
  std::vector<DemonLine> found;
  const auto nbi = static_cast<std::int64_t>(nb);
  for (std::int64_t k = k_lo_; k <= k_hi_; ++k) {
    for (std::int64_t b = 0; b < nbi; ++b) {
      const auto at = [&](std::int64_t kk, std::int64_t bb) { return static_cast<std::size_t>(kk) * nb + static_cast<std::size_t>(bb); };
      const double v = z[at(k, b)];
      if (v < threshold) continue;
      // A line is the largest value within a bin in frequency and min_separation_deg in bearing.
      bool is_peak = true;
      for (std::int64_t kk = k - 1; kk <= k + 1 && is_peak; ++kk) {
        for (std::int64_t bb = std::max<std::int64_t>(b - sep, 0); bb <= std::min(b + sep, nbi - 1); ++bb) {
          if (kk == k && bb == b) continue;
          const double other = z[at(kk, bb)];
          if (other > v || (other == v && (kk < k || (kk == k && bb < b)))) {
            is_peak = false;
            break;
          }
        }
      }
      if (!is_peak) continue;
      double excess[3];
      for (int j = 0; j < 3; ++j) excess[j] = std::max(s[at(k - 1 + j, b)] - bg[at(k - 1 + j, b)], 0.0);
      const int side = excess[2] >= excess[0] ? 1 : -1;
      const double delta = excess[1] > 0.0 ? std::min(hann_offset(std::sqrt(excess[1 + side] / excess[1])), 0.5) : 0.0;
      double frac = 0.0;
      if (b > 0 && b < nbi - 1) {
        const double den = z[at(k, b - 1)] - 2.0 * v + z[at(k, b + 1)];
        if (den < 0.0) frac = std::clamp(0.5 * (z[at(k, b - 1)] - z[at(k, b + 1)]) / den, -0.5, 0.5);
      }
      DemonLine line;
      line.frequency = (static_cast<double>(k) + side * delta) * s_df_;
      line.snr_db = 10.0 * std::log10(v);
      line.modulation = std::sqrt(excess[1]) / hann_gain(delta);
      line.bearing = bearings[static_cast<std::size_t>(b)] + frac * step;
      found.push_back(line);
    }
  }
  std::stable_sort(found.begin(), found.end(), [](const DemonLine& a, const DemonLine& b) { return a.snr_db > b.snr_db; });
  for (const DemonLine& q : found) {  // lines by bearing, strongest first
    bool placed = false;
    for (DemonSource& src : ev.sources) {
      if (std::abs(q.bearing - src.lines[0].bearing) < d.min_separation_deg * kDeg) {
        if (static_cast<std::int64_t>(src.lines.size()) < d.max_lines) src.lines.push_back(q);
        placed = true;
        break;
      }
    }
    if (!placed && static_cast<std::int64_t>(ev.sources.size()) < d.max_sources) {
      ev.sources.emplace_back();
      ev.sources.back().lines.push_back(q);
    }
  }
  for (DemonSource& src : ev.sources) {
    double w = 0.0, wb = 0.0;
    for (const DemonLine& q : src.lines) {
      w += q.snr_db;
      wb += q.snr_db * q.bearing;
    }
    src.bearing = wb / w;
    harmonic_set(src, s_df_, d);
  }
  return ev;
}

std::string tonal_events_json(const std::vector<TonalEvent>& events, const TonalAnalyzer& analyzer) {
  std::string out = "[";
  for (std::size_t i = 0; i < events.size(); ++i) out += (i ? ", " : "") + event_json(events[i], analyzer, false, nullptr);
  return out + "]";
}

std::string passive_tonal_json(const PingRecord& record, const std::vector<PassiveFrame>& frames, std::int64_t span,
                               TonalAnalyzer& analyzer) {
  std::vector<TonalEvent> events = analyzer.push(record);
  analyzer.pending.insert(analyzer.pending.end(), events.begin(), events.end());
  if (frames.empty()) return {};      // kept for the next record's frames
  const Samples& s = record.samples;
  std::vector<std::string> goertzel(frames.size()), demon(frames.size());
  for (const TonalEvent& e : analyzer.pending) {
    const double t = static_cast<double>(e.time_ns - record.transmit.time_ns) * 1e-9;
    std::size_t i = frames.size() - 1;
    for (std::size_t j = 0; j < frames.size(); ++j) {
      if (t <= s.start_time + static_cast<double>((static_cast<std::int64_t>(j) + 1) * span) / s.sample_rate + 1e-9) {
        i = j;
        break;
      }
    }
    std::vector<double> peaks;
    for (const DoaPeak& p : frames[i].peaks) peaks.push_back(p.bearing / kDeg);
    std::string& into = e.demon ? demon[i] : goertzel[i];
    into += (into.empty() ? "" : ", ") + event_json(e, analyzer, true, &peaks);
  }
  analyzer.pending.clear();
  std::string out;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    std::string line = passive_json({frames[i]}, record.transmit.time_ns);
    if (!goertzel[i].empty() || !demon[i].empty()) {
      std::string tonal = ", \"tonal\": {";
      if (!goertzel[i].empty()) tonal += "\"goertzel\": [" + goertzel[i] + "]";
      if (!demon[i].empty()) tonal += std::string(goertzel[i].empty() ? "" : ", ") + "\"demon\": [" + demon[i] + "]";
      tonal += "}";
      line.insert(line.rfind('}'), tonal);
    }
    out += line;
  }
  return out;
}

}  // namespace sonar::live
