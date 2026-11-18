#include "sonar/live/sync.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "sonar/dsp/fft.hpp"
#include "sonar/live/derive.hpp"

namespace sonar::live {
namespace {

constexpr double kPi = 3.14159265358979323846;
using Cf = std::complex<float>;

// Analytic signal of a real series (the FFT method, as scipy.signal.hilbert).
std::vector<Cf> analytic(const std::vector<double>& x) {
  const std::size_t n = x.size();
  std::vector<Cf> z(n);
  for (std::size_t i = 0; i < n; ++i) z[i] = Cf(static_cast<float>(x[i]), 0.0F);
  dsp::fft(z.data(), n, false);
  for (std::size_t k = 1; k < n; ++k) {
    if (n % 2 == 0 && k == n / 2) continue;
    z[k] *= k < (n + 1) / 2 ? 2.0F : 0.0F;
  }
  dsp::fft(z.data(), n, true);
  return z;
}

// The reference burst: a Hann-windowed linear chirp.
std::vector<double> burst(const Reference& r, double fs) {
  const auto n = static_cast<std::size_t>(std::ceil(r.duration * fs)) + 1;
  std::vector<double> out(n, 0.0);
  const double k = (r.f_stop - r.f_start) / r.duration;
  for (std::size_t i = 0; i < n; ++i) {
    const double t = static_cast<double>(i) / fs;
    if (t > r.duration) break;
    const double w = std::sin(kPi * t / r.duration);
    out[i] = w * w * std::cos(2.0 * kPi * (r.f_start * t + 0.5 * k * t * t));
  }
  return out;
}

// Fractional sample position where the burst best matches `x` near `expected`: coarse from the
// correlation envelope, refined with its phase. Returns false if the peak is at the edge.
bool burst_position(const float* x, std::int64_t n_x, const std::vector<Cf>& rep, std::int64_t expected, std::int64_t search,
                    double fs, double f_centre, double& position, double& quality) {
  const std::int64_t lo = std::max<std::int64_t>(expected - search, 0);
  const std::int64_t hi = std::min<std::int64_t>(expected + search + static_cast<std::int64_t>(rep.size()), n_x);
  if (hi - lo <= static_cast<std::int64_t>(rep.size()) + 2) return false;
  std::vector<double> seg_real(static_cast<std::size_t>(hi - lo));
  for (std::int64_t i = lo; i < hi; ++i) seg_real[static_cast<std::size_t>(i - lo)] = static_cast<double>(x[i]);
  const std::vector<Cf> seg = analytic(seg_real);
  const std::size_t lags = seg.size() - rep.size() + 1;
  std::vector<std::complex<double>> corr(lags);
  std::size_t best = 0;
  for (std::size_t lag = 0; lag < lags; ++lag) {
    std::complex<double> acc = 0.0;
    for (std::size_t k = 0; k < rep.size(); ++k) acc += std::complex<double>(seg[lag + k]) * std::conj(std::complex<double>(rep[k]));
    corr[lag] = acc;
    if (std::abs(acc) > std::abs(corr[best])) best = lag;
  }
  if (best == 0 || best == lags - 1) return false;
  const double m0 = std::abs(corr[best - 1]), m1 = std::abs(corr[best]), m2 = std::abs(corr[best + 1]);
  const double frac = 0.5 * (m0 - m2) / (m0 - 2.0 * m1 + m2);
  // The correlation at lag i has phase 2 pi f_centre (i - true lag) / fs, to a period.
  double fine = static_cast<double>(best) - std::arg(corr[best]) * fs / (2.0 * kPi * f_centre);
  const double period = fs / f_centre;
  fine += std::round((static_cast<double>(best) + frac - fine) / period) * period;
  double e_seg = 0.0, e_rep = 0.0;
  for (std::size_t k = 0; k < rep.size(); ++k) {
    e_seg += std::norm(std::complex<double>(seg[best + k]));
    e_rep += std::norm(std::complex<double>(rep[k]));
  }
  position = static_cast<double>(lo) + fine;
  quality = m1 / (std::sqrt(e_seg) * std::sqrt(e_rep) + 1e-300);
  return true;
}

// Whether the burst expected at sample `expected` sits in clipped samples.
bool clipped(const float* x, std::int64_t n_x, std::int64_t expected, std::int64_t n_rep, double clip_level) {
  if (!(clip_level > 0.0)) return false;
  const std::int64_t lo = std::max<std::int64_t>(expected, 0), hi = std::min<std::int64_t>(expected + n_rep, n_x);
  std::int64_t railed = 0;
  for (std::int64_t i = lo; i < hi; ++i) railed += std::abs(static_cast<double>(x[i])) >= 0.98 * clip_level ? 1 : 0;
  return hi > lo && railed * 50 > hi - lo;        // more than 2% of the burst's samples at the rail
}

// A board's clock from the positions of its two bursts, or of the one that can be trusted.
// Returns the flag: 0 both, 1 one burst (no drift), 2 none (left as it is).
int board_clock(bool ok_a, bool ok_b, double n_a, double n_b, double t_a, double t_b, double apart, double fs, double& offset, double& drift) {
  offset = drift = 0.0;
  if (ok_a && ok_b) {
    // n = (t - start - offset) fs (1 + drift)
    drift = (n_b - n_a) / (apart * fs) - 1.0;
    offset = t_a - n_a / (fs * (1.0 + drift));
    return 0;
  }
  if (ok_a) offset = t_a - n_a / fs;
  else if (ok_b) offset = t_b - n_b / fs;
  return ok_a || ok_b ? 1 : 2;
}

void estimate(const PingRecord& record, const Reference& r, std::vector<double>& offset, std::vector<double>& drift_ppm,
              std::vector<double>& quality, std::vector<int>* flagged = nullptr) {
  const Samples& s = record.samples;
  const double fs = s.sample_rate;
  const double t_end = s.start_time + static_cast<double>(s.n_samples) / fs;
  const double t_a = s.start_time + r.lead, t_b = t_end - r.tail - r.duration;
  const std::vector<Cf> rep = analytic(burst(r, fs));
  const double f_centre = 0.5 * (r.f_start + r.f_stop);
  const auto search = static_cast<std::int64_t>(r.search_time * fs);
  const bool checked = r.min_quality > 0.0 || r.clip_level > 0.0;
  offset.clear();
  drift_ppm.clear();
  quality.clear();
  if (flagged) flagged->clear();
  std::size_t first_good = 0;
  bool have_good = false;
  for (const Board& board : record.metadata.boards) {
    if (board.reference_channel < 0) throw std::runtime_error("sync: a board has no reference channel");
    const float* x = s.channel(board.reference_channel);
    double n_a = 0.0, n_b = 0.0, q_a = 0.0, q_b = 0.0;
    const auto e_a = static_cast<std::int64_t>((t_a - s.start_time) * fs), e_b = static_cast<std::int64_t>((t_b - s.start_time) * fs);
    bool ok_a = burst_position(x, s.n_samples, rep, e_a, search, fs, f_centre, n_a, q_a);
    bool ok_b = burst_position(x, s.n_samples, rep, e_b, search, fs, f_centre, n_b, q_b);
    if (!checked && (!ok_a || !ok_b)) {
      throw std::runtime_error("sync: reference burst not found on a board; widen search_time or check the wiring");
    }
    if (checked) {
      ok_a = ok_a && q_a >= r.min_quality && !clipped(x, s.n_samples, e_a, static_cast<std::int64_t>(rep.size()), r.clip_level);
      ok_b = ok_b && q_b >= r.min_quality && !clipped(x, s.n_samples, e_b, static_cast<std::int64_t>(rep.size()), r.clip_level);
    }
    double o = 0.0, d = 0.0;
    const int flag = board_clock(ok_a, ok_b, n_a, n_b, t_a - s.start_time, t_b - s.start_time, t_b - t_a, fs, o, d);
    offset.push_back(o);
    drift_ppm.push_back(d * 1e6);
    quality.push_back(flag == 0 ? std::min(q_a, q_b) : flag == 1 ? (ok_a ? q_a : q_b) : 0.0);
    if (flagged) flagged->push_back(flag);
    if (flag != 2 && !have_good) {
      have_good = true;
      first_good = offset.size() - 1;
    }
  }
  // Relative to the first board (the first that could be timed, when the very first could not).
  const double o0 = have_good ? offset[first_good] : 0.0, d0 = have_good ? drift_ppm[first_good] : 0.0;
  for (std::size_t b = 0; b < offset.size(); ++b) {
    if (flagged && (*flagged)[b] == 2) continue;
    offset[b] -= o0;
    drift_ppm[b] -= d0;
  }
}

// Values of a real series at fractional sample positions (Hann-windowed sinc, 16 taps).
void resample(float* x, std::int64_t n, double offset, double drift, double fs) {
  constexpr int taps = 16, half = taps / 2;
  std::vector<float> out(static_cast<std::size_t>(n), 0.0F);
  for (std::int64_t m = 0; m < n; ++m) {
    // Nominal sample m is wanted at time m / fs; this board holds that instant at index `pos`.
    const double pos = (static_cast<double>(m) / fs - offset) * fs * (1.0 + drift);
    const auto base = static_cast<std::int64_t>(std::floor(pos));
    const double frac = pos - static_cast<double>(base);
    double acc = 0.0;
    for (int k = -half + 1; k <= half; ++k) {
      const std::int64_t idx = base + k;
      if (idx < 0 || idx >= n) continue;
      const double d = static_cast<double>(k) - frac;
      const double sinc = std::abs(d) < 1e-12 ? 1.0 : std::sin(kPi * d) / (kPi * d);
      acc += sinc * (0.5 + 0.5 * std::cos(kPi * d / half)) * static_cast<double>(x[idx]);
    }
    out[static_cast<std::size_t>(m)] = static_cast<float>(acc);
  }
  std::copy(out.begin(), out.end(), x);
}

}  // namespace

BurstWindows burst_windows(const Reference& r, double fs, std::int64_t n_samples, std::size_t n_boards) {
  // The search of synchronise() reaches `search` either side of a burst; the check after the
  // correction reads the same stretch shifted by the board's offset and drift, so twice that is kept.
  const auto search = static_cast<std::int64_t>(r.search_time * fs);
  const auto n_rep = static_cast<std::int64_t>(std::ceil(r.duration * fs)) + 1;
  const double t_b = static_cast<double>(n_samples) / fs - r.tail - r.duration;
  BurstWindows w;
  w.length = 4 * search + n_rep + 16;
  w.first_a = static_cast<std::int64_t>(r.lead * fs) - 2 * search - 8;
  w.first_b = static_cast<std::int64_t>(t_b * fs) - 2 * search - 8;
  w.a.assign(n_boards, std::vector<float>(static_cast<std::size_t>(w.length), 0.0F));
  w.b.assign(n_boards, std::vector<float>(static_cast<std::size_t>(w.length), 0.0F));
  return w;
}

SyncResult estimate_from_bursts(const BurstWindows& w, const Reference& r, double fs, std::int64_t n_samples, double top_frequency) {
  const double t_a = r.lead, t_b = static_cast<double>(n_samples) / fs - r.tail - r.duration;   // from the window's opening
  const std::vector<Cf> rep = analytic(burst(r, fs));
  const double f_centre = 0.5 * (r.f_start + r.f_stop);
  const auto search = static_cast<std::int64_t>(r.search_time * fs);
  const auto n_rep = static_cast<std::int64_t>(rep.size());
  // One burst on one board, exactly as estimate() reads it from the whole record: the stretch
  // [expected - search, expected + search + n_rep) clipped to the window, here cut from the kept samples.
  auto measure = [&](const std::vector<float>& kept, std::int64_t first, double t, double offset, double drift, double& position, double& quality) {
    const auto expected = static_cast<std::int64_t>(t * fs);
    const std::int64_t lo = std::max<std::int64_t>(expected - search, 0), hi = std::min<std::int64_t>(expected + search + n_rep, n_samples);
    if (hi <= lo) return false;
    std::vector<float> seg(static_cast<std::size_t>(hi - lo), 0.0F);
    constexpr int taps = 16, half = taps / 2;
    for (std::int64_t m = lo; m < hi; ++m) {
      if (offset == 0.0 && drift == 0.0) {
        const std::int64_t q = m - first;
        if (q >= 0 && q < w.length) seg[static_cast<std::size_t>(m - lo)] = kept[static_cast<std::size_t>(q)];
        continue;
      }
      // After the correction: nominal sample m is the board's sample at `pos` (resample() above).
      const double pos = (static_cast<double>(m) / fs - offset) * fs * (1.0 + drift);
      const auto base = static_cast<std::int64_t>(std::floor(pos));
      const double frac = pos - static_cast<double>(base);
      double acc = 0.0;
      for (int k = -half + 1; k <= half; ++k) {
        const std::int64_t idx = base + k;
        if (idx < 0 || idx >= n_samples || idx - first < 0 || idx - first >= w.length) continue;
        const double d = static_cast<double>(k) - frac;
        const double sinc = std::abs(d) < 1e-12 ? 1.0 : std::sin(kPi * d) / (kPi * d);
        acc += sinc * (0.5 + 0.5 * std::cos(kPi * d / half)) * static_cast<double>(kept[static_cast<std::size_t>(idx - first)]);
      }
      seg[static_cast<std::size_t>(m - lo)] = static_cast<float>(acc);
    }
    if (!burst_position(seg.data(), hi - lo, rep, expected - lo, search, fs, f_centre, position, quality)) return false;
    position += static_cast<double>(lo);
    return true;
  };
  const bool checked = r.min_quality > 0.0 || r.clip_level > 0.0;
  auto clocks = [&](const std::vector<double>* off, const std::vector<double>* ppm, std::vector<double>& offset, std::vector<double>& drift_ppm,
                    std::vector<double>& quality, std::vector<int>& flagged) {
    offset.clear();
    drift_ppm.clear();
    quality.clear();
    flagged.clear();
    std::size_t first_good = 0;
    bool have_good = false;
    for (std::size_t b = 0; b < w.a.size(); ++b) {
      double n_a = 0.0, n_b = 0.0, q_a = 0.0, q_b = 0.0;
      const double o = off ? (*off)[b] : 0.0, d = ppm ? (*ppm)[b] * 1e-6 : 0.0;
      bool ok_a = measure(w.a[b], w.first_a, t_a, o, d, n_a, q_a), ok_b = measure(w.b[b], w.first_b, t_b, o, d, n_b, q_b);
      if (!checked && (!ok_a || !ok_b)) {
        throw std::runtime_error("sync: reference burst not found on a board; widen search_time or check the wiring");
      }
      if (checked) {
        const auto e_a = static_cast<std::int64_t>(t_a * fs) - w.first_a, e_b = static_cast<std::int64_t>(t_b * fs) - w.first_b;
        ok_a = ok_a && q_a >= r.min_quality && !clipped(w.a[b].data(), w.length, e_a, n_rep, r.clip_level);
        ok_b = ok_b && q_b >= r.min_quality && !clipped(w.b[b].data(), w.length, e_b, n_rep, r.clip_level);
      }
      double bo = 0.0, bd = 0.0;
      const int flag = board_clock(ok_a, ok_b, n_a, n_b, t_a, t_b, t_b - t_a, fs, bo, bd);
      offset.push_back(bo);
      drift_ppm.push_back(bd * 1e6);
      quality.push_back(flag == 0 ? std::min(q_a, q_b) : flag == 1 ? (ok_a ? q_a : q_b) : 0.0);
      flagged.push_back(flag);
      if (flag != 2 && !have_good) {
        have_good = true;
        first_good = b;
      }
    }
    const double o0 = have_good ? offset[first_good] : 0.0, d0 = have_good ? drift_ppm[first_good] : 0.0;
    for (std::size_t b = 0; b < offset.size(); ++b) {
      if (flagged[b] == 2) continue;
      offset[b] -= o0;
      drift_ppm[b] -= d0;
    }
  };
  SyncResult out;
  clocks(nullptr, nullptr, out.offset, out.drift_ppm, out.quality, out.flagged);
  std::vector<double> off2, drift2, q2;
  std::vector<int> f2;
  clocks(&out.offset, &out.drift_ppm, off2, drift2, q2, f2);
  const double span = static_cast<double>(n_samples) / fs;
  out.tolerance = sync_tolerance(r.tolerance_us, top_frequency);
  out.ok = true;
  for (std::size_t b = 0; b < off2.size(); ++b) {
    // A board timed from one burst has no drift to check: what is left of its offset is the residual.
    out.residual.push_back(out.flagged[b] == 0 ? std::abs(off2[b]) + std::abs(drift2[b]) * 1e-6 * span : std::abs(off2[b]));
    out.ok = out.ok && out.flagged[b] == 0 && out.residual.back() <= out.tolerance;   // a board timed from one burst has no drift: not ok, but usable
  }
  return out;
}

void blank_interval(PingRecord& record, double t0, double t1, double taper) {
  Samples& s = record.samples;
  const auto stride = static_cast<std::int64_t>(s.stride());
  const double fs = s.sample_rate;
  const auto lo = std::max<std::int64_t>(static_cast<std::int64_t>(std::floor((t0 - taper - s.start_time) * fs)), 0);
  const auto hi = std::min<std::int64_t>(static_cast<std::int64_t>(std::ceil((t1 + taper - s.start_time) * fs)) + 1, s.n_samples);
  for (std::int64_t i = lo; i < hi; ++i) {
    const double t = s.start_time + static_cast<double>(i) / fs;
    const double out = t < t0 ? t0 - t : t > t1 ? t - t1 : 0.0;       // how far outside the blanked interval
    const double gain = out <= 0.0 ? 0.0 : out >= taper ? 1.0 : 0.5 - 0.5 * std::cos(kPi * out / taper);
    if (gain >= 1.0) continue;
    for (std::int64_t c = 0; c < s.n_channels; ++c) {
      float* x = s.channel(c) + i * stride;
      for (std::int64_t q = 0; q < stride; ++q) x[q] = static_cast<float>(gain * static_cast<double>(x[q]));
    }
  }
}

void drop_channels(PingRecord& record, const std::vector<std::int64_t>& rows) {
  std::vector<char> gone(static_cast<std::size_t>(record.samples.n_channels), 0);
  for (std::int64_t r : rows) gone[static_cast<std::size_t>(r)] = 1;
  Samples old = std::move(record.samples);
  Samples& s = record.samples;
  s = old;
  s.data.clear();
  const std::size_t per = static_cast<std::size_t>(old.n_samples) * old.stride();
  auto keep_vec = [&](auto& v) {
    if (v.size() != gone.size()) return;
    std::remove_const_t<std::remove_reference_t<decltype(v)>> out;
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (!gone[i]) out.push_back(v[i]);
    }
    v = std::move(out);
  };
  std::int64_t kept = 0;
  for (std::int64_t c = 0; c < old.n_channels; ++c) {
    if (gone[static_cast<std::size_t>(c)]) continue;
    s.data.insert(s.data.end(), old.channel(c), old.channel(c) + per);
    ++kept;
  }
  s.n_channels = kept;
  keep_vec(record.geometry.rx);
  keep_vec(record.calibration.gain);
  keep_vec(record.calibration.phase);
  keep_vec(record.calibration.delay);
  keep_vec(record.calibration.position_offset);
  keep_vec(record.metadata.channel_element);
  keep_vec(record.metadata.channel_board);
  for (Board& b : record.metadata.boards) b.reference_channel = -1;   // row numbers no longer apply
}

SyncResult synchronise(PingRecord& record, const Reference& reference) {
  Samples& s = record.samples;
  if (s.is_complex()) throw std::invalid_argument("sync: works on raw real samples, before baseband conversion");
  SyncResult out;
  Reference ref = reference;
  if (ref.min_quality > 0.0 && !(ref.clip_level > 0.0) && std::isfinite(record.metadata.adc_full_scale)) ref.clip_level = record.metadata.adc_full_scale;
  estimate(record, ref, out.offset, out.drift_ppm, out.quality, &out.flagged);
  for (std::int64_t c = 0; c < s.n_channels; ++c) {
    const auto b = static_cast<std::size_t>(record.metadata.channel_board[static_cast<std::size_t>(c)]);
    if (out.offset[b] == 0.0 && out.drift_ppm[b] == 0.0) continue;
    resample(s.channel(c), s.n_samples, out.offset[b], out.drift_ppm[b] * 1e-6, s.sample_rate);
  }
  // Measure the bursts again: what is left is the check that the boards now line up.
  std::vector<double> off2, drift2, q2;
  std::vector<int> f2;
  estimate(record, ref, off2, drift2, q2, &f2);
  const double span = static_cast<double>(s.n_samples) / s.sample_rate;
  const Waveform& w = record.transmit.waveform;
  out.tolerance = sync_tolerance(reference.tolerance_us, std::max(w.f_start, w.f_stop));
  out.ok = true;
  for (std::size_t b = 0; b < off2.size(); ++b) {
    // A board timed from one burst has no drift to check: what is left of its offset is the residual.
    out.residual.push_back(out.flagged[b] == 0 ? std::abs(off2[b]) + std::abs(drift2[b]) * 1e-6 * span : std::abs(off2[b]));
    out.ok = out.ok && out.flagged[b] == 0 && out.residual.back() <= out.tolerance;   // a board timed from one burst has no drift: not ok, but usable
    record.metadata.boards[b].clock_offset = out.offset[b];
    record.metadata.boards[b].clock_drift_ppm = out.drift_ppm[b];
    record.metadata.boards[b].synchronised = 1;
  }
  if (reference.dedicated) {
    std::vector<std::int64_t> refs;
    for (const Board& b : record.metadata.boards) {
      if (b.reference_channel >= 0) refs.push_back(b.reference_channel);
    }
    drop_channels(record, refs);
  }
  return out;
}

}  // namespace sonar::live
