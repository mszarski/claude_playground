#include "sonar/dsp/baseband.hpp"

#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"

namespace sonar::dsp {

#if SONAR_HAVE_CUDA
void baseband_cuda(const float* in, bool is_real, std::int64_t n_channels, std::int64_t n,
                   std::int64_t n_out, std::int64_t k0, std::int64_t half, float taper_bins,
                   float gain, std::complex<float> rot, float* out, int device);
#endif

double baseband_band(const PingRecord& record, const BasebandOptions& opt) {
  const Waveform& w = record.transmit.waveform;
  if (opt.bandwidth > 0.0) return opt.bandwidth;
  double band = w.bandwidth() * opt.band_margin;
  if (!(band > 0.0)) {
    // CW pulse: the band is set by the pulse length.
    band = opt.cw_band_factor / w.duration;
  }
  if (opt.max_range_rate < 0.0 || opt.doppler_allowance_hz < 0.0) {
    throw std::invalid_argument("to_baseband: max_range_rate and doppler_allowance_hz must not be negative");
  }
  if (opt.max_range_rate > 0.0) {
    const double c = record.environment.sound_speed;
    if (!(c > 0.0) || !std::isfinite(c)) {
      throw std::invalid_argument("to_baseband: max_range_rate needs the record's sound speed");
    }
    const double f_top = w.f_stop > w.f_start ? w.f_stop : w.f_start;
    band += 2.0 * (2.0 * opt.max_range_rate * f_top / c);
  }
  if (opt.doppler_allowance_hz > 0.0) band += 2.0 * opt.doppler_allowance_hz;
  return band;
}

void to_baseband(PingRecord& record, const BasebandOptions& opt, Backend backend, int device) {
  Samples& s = record.samples;
  if (s.n_samples <= 0 || s.n_channels <= 0) return;
  if (s.data.size() != s.expected_size()) throw std::invalid_argument("to_baseband: samples not allocated");
  const Waveform& w = record.transmit.waveform;
  const double pi = 3.14159265358979323846;
  const auto n = static_cast<std::size_t>(s.n_samples);
  const double fs = s.sample_rate;

  const double fc_wanted = opt.center_frequency > 0.0 ? opt.center_frequency : w.center();
  const double band = baseband_band(record, opt);
  if (!(fc_wanted > 0.0)) throw std::invalid_argument("to_baseband: no carrier frequency");

  // Frequency of input bin k, accounting for the carrier already removed from baseband input.
  const double f_offset = s.kind == SampleKind::ComplexBaseband ? s.center_frequency : 0.0;
  const double df = fs / static_cast<double>(n);
  const auto k0 = static_cast<std::int64_t>(std::llround((fc_wanted - f_offset) / df));
  const double fc_exact = f_offset + static_cast<double>(k0) * df;

  auto n_out = static_cast<std::size_t>(std::ceil(opt.oversample * band / df));
  if (n_out < 8) n_out = 8;
  n_out = good_fft_size(n_out);
  if (n_out > n) n_out = n;
  const auto half_keep = static_cast<std::int64_t>(std::floor(0.5 * band / df));
  const auto half_out = static_cast<std::int64_t>(n_out / 2);
  const std::int64_t half = half_keep < half_out ? half_keep : half_out - 1;
  const double taper_bins = opt.edge_taper * static_cast<double>(2 * half + 1);

  const bool is_real = s.kind == SampleKind::RealPassband;
  // One-sided spectrum of a real signal carries half the amplitude.
  const float gain = (is_real ? 2.0F : 1.0F) / static_cast<float>(n);
  // Reference the mixing phase to t = 0 rather than to the first sample.
  const double phase0 = -2.0 * pi * std::fmod((fc_exact - f_offset) * s.start_time, 1.0);
  const std::complex<float> rot(static_cast<float>(std::cos(phase0)), static_cast<float>(std::sin(phase0)));

  std::vector<float> out(static_cast<std::size_t>(s.n_channels) * n_out * 2);
  const auto sn = static_cast<std::int64_t>(n);
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    baseband_cuda(s.data.data(), is_real, s.n_channels, sn, static_cast<std::int64_t>(n_out), k0, half,
                  static_cast<float>(taper_bins), gain, rot, out.data(), device);
#else
    throw std::runtime_error("to_baseband: built without the CUDA backend");
#endif
  } else {
  (void)device;
  parallel_for_host(s.n_channels, [&](std::int64_t c) {
    std::vector<std::complex<float>> x(n);
    if (is_real) {
      const float* src = s.channel(c);
      for (std::size_t i = 0; i < n; ++i) x[i] = {src[i], 0.0F};
    } else {
      const std::complex<float>* src = s.channel_complex(c);
      for (std::size_t i = 0; i < n; ++i) x[i] = src[i];
    }
    fft(x, false);
    std::vector<std::complex<float>> y(n_out, {0.0F, 0.0F});
    for (std::int64_t k = -half; k <= half; ++k) {
      std::int64_t src_bin = (k0 + k) % sn;
      if (src_bin < 0) src_bin += sn;
      const std::int64_t dst_bin = k < 0 ? k + static_cast<std::int64_t>(n_out) : k;
      double wgt = 1.0;
      const double from_edge = static_cast<double>(half - (k < 0 ? -k : k));
      if (taper_bins > 0.0 && from_edge < taper_bins) {
        wgt = 0.5 * (1.0 - std::cos(pi * from_edge / taper_bins));
      }
      y[static_cast<std::size_t>(dst_bin)] = x[static_cast<std::size_t>(src_bin)] * static_cast<float>(wgt);
    }
    // Unnormalised inverse: sum over bins, with the forward 1/n folded into `gain`.
    fft(y.data(), n_out, true);
    auto* dst = reinterpret_cast<std::complex<float>*>(out.data() + static_cast<std::size_t>(c) * n_out * 2);
    const float undo = static_cast<float>(n_out);  // fft(inverse) divided by n_out
    for (std::size_t i = 0; i < n_out; ++i) dst[i] = y[i] * (gain * undo) * rot;
  });
  }

  s.data = std::move(out);
  s.kind = SampleKind::ComplexBaseband;
  s.n_samples = static_cast<std::int64_t>(n_out);
  s.sample_rate = fs * static_cast<double>(n_out) / static_cast<double>(n);
  s.center_frequency = fc_exact;
  s.storage = StorageType::Float32;
  s.storage_scale = 1.0;
}

}  // namespace sonar::dsp
