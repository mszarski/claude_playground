#include "sonar/live/sync.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#include "sonar/dsp/fft.hpp"

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

void estimate(const PingRecord& record, const Reference& r, std::vector<double>& offset, std::vector<double>& drift_ppm,
              std::vector<double>& quality) {
  const Samples& s = record.samples;
  const double fs = s.sample_rate;
  const double t_end = s.start_time + static_cast<double>(s.n_samples) / fs;
  const double t_a = s.start_time + r.lead, t_b = t_end - r.tail - r.duration;
  const std::vector<Cf> rep = analytic(burst(r, fs));
  const double f_centre = 0.5 * (r.f_start + r.f_stop);
  const auto search = static_cast<std::int64_t>(r.search_time * fs);
  offset.clear();
  drift_ppm.clear();
  quality.clear();
  for (const Board& board : record.metadata.boards) {
    if (board.reference_channel < 0) throw std::runtime_error("sync: a board has no reference channel");
    const float* x = s.channel(board.reference_channel);
    double n_a, n_b, q_a, q_b;
    if (!burst_position(x, s.n_samples, rep, static_cast<std::int64_t>((t_a - s.start_time) * fs), search, fs, f_centre, n_a, q_a) ||
        !burst_position(x, s.n_samples, rep, static_cast<std::int64_t>((t_b - s.start_time) * fs), search, fs, f_centre, n_b, q_b)) {
      throw std::runtime_error("sync: reference burst not found on a board; widen search_time or check the wiring");
    }
    // n = (t - start - offset) fs (1 + drift)
    const double drift = (n_b - n_a) / ((t_b - t_a) * fs) - 1.0;
    offset.push_back((t_a - s.start_time) - n_a / (fs * (1.0 + drift)));
    drift_ppm.push_back(drift * 1e6);
    quality.push_back(std::min(q_a, q_b));
  }
  const double o0 = offset[0], d0 = drift_ppm[0];
  for (double& v : offset) v -= o0;
  for (double& v : drift_ppm) v -= d0;
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
  estimate(record, reference, out.offset, out.drift_ppm, out.quality);
  for (std::int64_t c = 0; c < s.n_channels; ++c) {
    const auto b = static_cast<std::size_t>(record.metadata.channel_board[static_cast<std::size_t>(c)]);
    if (out.offset[b] == 0.0 && out.drift_ppm[b] == 0.0) continue;
    resample(s.channel(c), s.n_samples, out.offset[b], out.drift_ppm[b] * 1e-6, s.sample_rate);
  }
  // Measure the bursts again: what is left is the check that the boards now line up.
  std::vector<double> off2, drift2, q2;
  estimate(record, reference, off2, drift2, q2);
  const double span = static_cast<double>(s.n_samples) / s.sample_rate;
  const Waveform& w = record.transmit.waveform;
  out.tolerance = 1.0 / (20.0 * std::max(w.f_start, w.f_stop));
  out.ok = true;
  for (std::size_t b = 0; b < off2.size(); ++b) {
    out.residual.push_back(std::abs(off2[b]) + std::abs(drift2[b]) * 1e-6 * span);
    out.ok = out.ok && out.residual.back() <= out.tolerance;
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
