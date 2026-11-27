#include "sonar/dsp/pulse_compression.hpp"

#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/core/waveform.hpp"
#include "sonar/dsp/fft.hpp"

namespace sonar::dsp {

namespace {

double replica_window(ReplicaWindow w, double x) {
  const double pi = 3.14159265358979323846;
  switch (w) {
    case ReplicaWindow::Hann: return 0.5 - 0.5 * std::cos(2.0 * pi * x);
    case ReplicaWindow::Hamming: return 0.54 - 0.46 * std::cos(2.0 * pi * x);
    case ReplicaWindow::None:
    default: return 1.0;
  }
}

/// Modified Bessel function I0 by its power series (for the Kaiser window).
double bessel_i0(double x) {
  const double q = 0.25 * x * x;
  double term = 1.0, sum = 1.0;
  for (int k = 1; k < 64; ++k) {
    term *= q / (static_cast<double>(k) * static_cast<double>(k));
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  return sum;
}

/// Complex envelope, about the band centre, of the real replica an arbitrary waveform
/// carries: its analytic signal (negative frequencies removed) mixed down. Sample m is at
/// emission time m / replica_sample_rate. Same steps as SampledWaveform.from_record in
/// python/gpusonar/sim/waveform.py.
std::vector<std::complex<double>> replica_envelope(const Waveform& w) {
  const std::size_t n = w.replica.size();
  if (n < 2 || !(w.replica_sample_rate > 0.0)) {
    throw std::invalid_argument("pulse_compress: an arbitrary waveform needs replica samples and their rate");
  }
  if (!(w.replica_sample_rate > 2.0 * (w.f_start > w.f_stop ? w.f_start : w.f_stop))) {
    throw std::invalid_argument("pulse_compress: replica_sample_rate must exceed twice the top of the band");
  }
  const double pi2 = 6.283185307179586476925;
  const std::size_t pad = n / 8 + 64;
  const std::size_t n_fft = good_fft_size(n + 2 * pad);
  std::vector<std::complex<float>> x(n_fft, {0.0F, 0.0F});
  for (std::size_t m = 0; m < n; ++m) x[pad + m] = {w.replica[m], 0.0F};
  fft(x, false);
  for (std::size_t k = 1; k < n_fft; ++k) {
    if (2 * k < n_fft) {
      x[k] *= 2.0F;
    } else if (2 * k > n_fft) {
      x[k] = {0.0F, 0.0F};
    }
  }
  fft(x, true);
  const double cycles_per_sample = w.center() / w.replica_sample_rate;
  std::vector<std::complex<double>> env(n);
  for (std::size_t m = 0; m < n; ++m) {
    double cyc = cycles_per_sample * static_cast<double>(m);
    cyc -= std::floor(cyc);
    env[m] = std::complex<double>(x[pad + m]) * std::complex<double>(std::cos(pi2 * cyc), -std::sin(pi2 * cyc));
  }
  return env;
}

/// Band-limited interpolation of uniform samples at fractional index `pos`: a sinc with a
/// Kaiser window (kResamplerTaps taps, shape kResamplerKaiserBeta), samples outside the
/// array counting as zero.
std::complex<double> interpolate(const std::vector<std::complex<double>>& x, double pos, double i0_beta) {
  constexpr int kTaps = kResamplerTaps;
  constexpr double kBeta = kResamplerKaiserBeta;
  const double pi = 3.14159265358979323846;
  const double n0 = std::floor(pos);
  const double frac = pos - n0;
  const auto base = static_cast<std::int64_t>(n0);
  const auto size = static_cast<std::int64_t>(x.size());
  std::complex<double> sum{0.0, 0.0};
  for (int k = -kTaps / 2; k < kTaps / 2; ++k) {
    const std::int64_t idx = base - k;
    if (idx < 0 || idx >= size) continue;
    const double z = frac + static_cast<double>(k);
    const double r = 1.0 - (2.0 * z / kTaps) * (2.0 * z / kTaps);
    const double window = r > 0.0 ? bessel_i0(kBeta * std::sqrt(r)) / i0_beta : 1.0 / i0_beta;
    const double sinc = std::abs(z) < 1e-12 ? 1.0 : std::sin(pi * z) / (pi * z);
    sum += x[static_cast<std::size_t>(idx)] * (window * sinc);
  }
  return sum;
}

/// As `build` below for an arbitrary (sampled) waveform: the echo is the replica's envelope
/// b at emission time eta * t on the carrier, e(t) = b(eta t) exp(2 pi i (F eta - fc) t).
void build_arbitrary(const Waveform& waveform, double fs, double fc, const PulseCompressionOptions& opt,
                     std::vector<std::complex<double>>& echo, std::vector<std::complex<double>>& filter) {
  const double pi2 = 6.283185307179586476925;
  const std::vector<std::complex<double>> env = replica_envelope(waveform);
  const double eta = opt.doppler_scale;
  const double i0_beta = bessel_i0(kResamplerKaiserBeta);
  const auto n = static_cast<std::size_t>(std::floor(waveform.duration / eta * fs)) + 1;
  echo.resize(n);
  filter.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double t = static_cast<double>(k) / fs;
    const double u = eta * t;
    if (u < 0.0 || u > waveform.duration) {
      echo[k] = filter[k] = 0.0;
      continue;
    }
    double cyc = waveform.center() * u - fc * t;
    cyc -= std::floor(cyc);
    const std::complex<double> v = interpolate(env, u * waveform.replica_sample_rate, i0_beta) *
                                   std::complex<double>(std::cos(pi2 * cyc), std::sin(pi2 * cyc));
    echo[k] = v;
    filter[k] = v * replica_window(opt.window, n > 1 ? static_cast<double>(k) / static_cast<double>(n - 1) : 0.5);
  }
}

/// Unwindowed and windowed replicas. The windowed one is the filter; the plain one
/// is what an echo looks like, used to normalise the gain.
void build(const Waveform& waveform, double fs, double fc, const PulseCompressionOptions& opt,
           std::vector<std::complex<double>>& echo, std::vector<std::complex<double>>& filter) {
  if (waveform.type == WaveformType::Arbitrary) {
    build_arbitrary(waveform, fs, fc, opt, echo, filter);
    return;
  }
  const double pi2 = 6.283185307179586476925;
  const WaveformParams w = waveform_params(waveform);
  const double eta = opt.doppler_scale;
  // A time-compressed echo of the pulse: e(t) = q(eta * t), lasting duration / eta.
  const double dur = w.duration / eta;
  const auto n = static_cast<std::size_t>(std::floor(dur * fs)) + 1;
  echo.resize(n);
  filter.resize(n);
  for (std::size_t k = 0; k < n; ++k) {
    const double t = static_cast<double>(k) / fs;
    double env = 0.0, phase = 0.0, freq = 0.0;
    if (!waveform_eval(w, eta * t, env, phase, freq)) {
      echo[k] = filter[k] = 0.0;
      continue;
    }
    double cyc = phase - fc * t;
    cyc -= std::floor(cyc);
    const std::complex<double> v = env * std::complex<double>(std::cos(pi2 * cyc), std::sin(pi2 * cyc));
    echo[k] = v;
    filter[k] = v * replica_window(opt.window, n > 1 ? static_cast<double>(k) / static_cast<double>(n - 1) : 0.5);
  }
}

}  // namespace

std::vector<std::complex<float>> make_replica(const Waveform& waveform, double sample_rate,
                                              double center_frequency,
                                              const PulseCompressionOptions& options) {
  std::vector<std::complex<double>> echo, filter;
  build(waveform, sample_rate, center_frequency, options, echo, filter);
  std::vector<std::complex<float>> out(filter.size());
  for (std::size_t i = 0; i < filter.size(); ++i) out[i] = std::complex<float>(filter[i]);
  return out;
}

std::vector<std::complex<float>> compression_filter(const Waveform& waveform, double sample_rate, double center_frequency,
                                                    std::int64_t n_samples, const PulseCompressionOptions& opt, std::int64_t& n_fft_out) {
  std::vector<std::complex<double>> echo, filter;
  build(waveform, sample_rate, center_frequency, opt, echo, filter);
  std::complex<double> norm{0.0, 0.0};
  for (std::size_t k = 0; k < echo.size(); ++k) norm += echo[k] * std::conj(filter[k]);
  if (!(std::abs(norm) > 0.0)) throw std::invalid_argument("pulse_compress: replica has no energy");

  const auto n = static_cast<std::size_t>(n_samples);
  const std::size_t n_fft = good_fft_size(n + filter.size());
  // conj(FFT(filter)), scaled so a unit echo compresses to a unit peak.
  std::vector<std::complex<float>> h(n_fft, {0.0F, 0.0F});
  for (std::size_t k = 0; k < filter.size(); ++k) h[k] = std::complex<float>(filter[k] / std::abs(norm));
  fft(h, false);
  for (auto& v : h) v = std::conj(v);
  n_fft_out = static_cast<std::int64_t>(n_fft);
  return h;
}

#if SONAR_HAVE_CUDA
void pulse_compress_cuda(float* data, std::int64_t n_channels, std::int64_t n, std::int64_t n_fft,
                         const std::complex<float>* h, int device);
#endif

void pulse_compress(PingRecord& record, const PulseCompressionOptions& opt, Backend backend, int device) {
  Samples& s = record.samples;
  if (s.kind != SampleKind::ComplexBaseband) {
    throw std::invalid_argument("pulse_compress: samples must be complex baseband; run to_baseband first");
  }
  if (s.n_samples <= 0 || s.n_channels <= 0) return;
  if (!(opt.doppler_scale > 0.0)) throw std::invalid_argument("pulse_compress: doppler_scale must be positive");

  std::int64_t n_fft_signed = 0;
  const std::vector<std::complex<float>> h =
      compression_filter(record.transmit.waveform, s.sample_rate, s.center_frequency, s.n_samples, opt, n_fft_signed);
  const auto n = static_cast<std::size_t>(s.n_samples);
  const auto n_fft = static_cast<std::size_t>(n_fft_signed);

  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    pulse_compress_cuda(s.data.data(), s.n_channels, s.n_samples, static_cast<std::int64_t>(n_fft), h.data(),
                        device);
    return;
#else
    throw std::runtime_error("pulse_compress: built without the CUDA backend");
#endif
  }
  (void)device;
  parallel_for_host(s.n_channels, [&](std::int64_t c) {
    std::vector<std::complex<float>> x(n_fft, {0.0F, 0.0F});
    std::complex<float>* ch = s.channel_complex(c);
    std::copy(ch, ch + n, x.begin());
    fft(x, false);
    for (std::size_t i = 0; i < n_fft; ++i) x[i] *= h[i];
    fft(x, true);
    // y[m] = sum_k s[m + k] conj(filter[k]): lag m is the replica start, i.e. the delay.
    std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n), ch);
  });
}

}  // namespace sonar::dsp
