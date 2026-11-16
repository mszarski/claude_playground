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

/// Unwindowed and windowed replicas. The windowed one is the filter; the plain one
/// is what an echo looks like, used to normalise the gain.
void build(const Waveform& waveform, double fs, double fc, const PulseCompressionOptions& opt,
           std::vector<std::complex<double>>& echo, std::vector<std::complex<double>>& filter) {
  if (waveform.type == WaveformType::Arbitrary) {
    throw std::invalid_argument("pulse_compress: arbitrary replicas are not supported yet");
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

  std::vector<std::complex<double>> echo, filter;
  build(record.transmit.waveform, s.sample_rate, s.center_frequency, opt, echo, filter);
  std::complex<double> norm{0.0, 0.0};
  for (std::size_t k = 0; k < echo.size(); ++k) norm += echo[k] * std::conj(filter[k]);
  if (!(std::abs(norm) > 0.0)) throw std::invalid_argument("pulse_compress: replica has no energy");

  const auto n = static_cast<std::size_t>(s.n_samples);
  const std::size_t n_fft = good_fft_size(n + filter.size());
  // conj(FFT(filter)), scaled so a unit echo compresses to a unit peak.
  std::vector<std::complex<float>> h(n_fft, {0.0F, 0.0F});
  for (std::size_t k = 0; k < filter.size(); ++k) h[k] = std::complex<float>(filter[k] / std::abs(norm));
  fft(h, false);
  for (auto& v : h) v = std::conj(v);

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
