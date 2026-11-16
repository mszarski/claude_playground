#include "sonar/dsp/conditioning.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"

namespace sonar::dsp {

namespace {

constexpr double kPi = 3.14159265358979323846;

void require_baseband(const PingRecord& r, const char* who) {
  if (r.samples.kind != SampleKind::ComplexBaseband) {
    throw std::invalid_argument(std::string(who) + ": samples must be complex baseband");
  }
}

void require_same_shape(const std::vector<PingRecord*>& records, const char* who) {
  if (records.empty()) throw std::invalid_argument(std::string(who) + ": no records");
  for (const PingRecord* r : records) {
    if (r == nullptr) throw std::invalid_argument(std::string(who) + ": null record");
    require_baseband(*r, who);
    if (r->samples.n_samples != records[0]->samples.n_samples ||
        r->samples.sample_rate != records[0]->samples.sample_rate) {
      throw std::invalid_argument(std::string(who) + ": records must share sample rate and length");
    }
  }
}

/// Centred boxcar smoothing with shrinking windows at the ends (or wrap-around).
std::vector<double> smooth(const std::vector<double>& x, std::int64_t half, bool wrap) {
  if (half <= 0) return x;
  const auto n = static_cast<std::int64_t>(x.size());
  std::vector<double> out(x.size());
  for (std::int64_t i = 0; i < n; ++i) {
    double sum = 0.0;
    std::int64_t count = 0;
    for (std::int64_t k = -half; k <= half; ++k) {
      std::int64_t j = i + k;
      if (wrap) {
        j = ((j % n) + n) % n;
      } else if (j < 0 || j >= n) {
        continue;
      }
      sum += x[static_cast<std::size_t>(j)];
      ++count;
    }
    out[static_cast<std::size_t>(i)] = sum / static_cast<double>(count);
  }
  return out;
}

}  // namespace

void remove_dc(PingRecord& record) {
  Samples& s = record.samples;
  const auto n = static_cast<std::size_t>(s.n_samples) * s.stride();
  for (std::int64_t c = 0; c < s.n_channels; ++c) {
    float* x = s.channel(c);
    for (std::size_t part = 0; part < s.stride(); ++part) {
      double mean = 0.0;
      for (std::size_t i = part; i < n; i += s.stride()) mean += x[i];
      mean /= static_cast<double>(s.n_samples);
      for (std::size_t i = part; i < n; i += s.stride()) x[i] -= static_cast<float>(mean);
    }
  }
}

void apply_calibration(PingRecord& record) {
  require_baseband(record, "apply_calibration");
  Samples& s = record.samples;
  Calibration& cal = record.calibration;
  const auto nc = static_cast<std::size_t>(s.n_channels);
  if (cal.gain.size() != nc || cal.phase.size() != nc || cal.delay.size() != nc) {
    throw std::invalid_argument("apply_calibration: calibration tables must have one entry per channel");
  }
  const auto n = static_cast<std::size_t>(s.n_samples);
  parallel_for_host(s.n_channels, [&](std::int64_t c) {
    const auto ci = static_cast<std::size_t>(c);
    std::complex<float>* x = s.channel_complex(c);
    const double delay = cal.delay[ci];
    if (delay != 0.0) {
      // Advance by `delay`: y(t) = x(t + delay). In baseband that is a phase ramp over
      // frequency plus the carrier's own rotation.
      std::vector<std::complex<float>> buf(x, x + n);
      fft(buf, false);
      for (std::size_t k = 0; k < n; ++k) {
        const double f = (k < (n + 1) / 2 ? static_cast<double>(k) : static_cast<double>(k) - static_cast<double>(n)) *
                         s.sample_rate / static_cast<double>(n);
        const double ph = 2.0 * kPi * (f + s.center_frequency) * delay;
        buf[k] *= std::complex<float>(static_cast<float>(std::cos(ph)), static_cast<float>(std::sin(ph)));
      }
      fft(buf, true);
      std::copy(buf.begin(), buf.end(), x);
    }
    const std::complex<float> corr =
        std::polar(static_cast<float>(1.0 / cal.gain[ci]), static_cast<float>(-cal.phase[ci]));
    for (std::size_t i = 0; i < n; ++i) x[i] *= corr;
  });
  std::fill(cal.gain.begin(), cal.gain.end(), 1.0);
  std::fill(cal.phase.begin(), cal.phase.end(), 0.0);
  std::fill(cal.delay.begin(), cal.delay.end(), 0.0);
}

void apply_tvg(PingRecord& record, const TvgOptions& opt) {
  Samples& s = record.samples;
  Metadata& m = record.metadata;
  const double c = record.environment.sound_speed;
  const auto n = static_cast<std::size_t>(s.n_samples);
  std::vector<float> gain(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double t = s.start_time + static_cast<double>(i) / s.sample_rate;
    const double range = std::max(0.5 * c * t, 1e-6);
    double db = range > opt.reference_range
                    ? opt.spreading_db_per_decade * std::log10(range / opt.reference_range)
                    : 0.0;
    db += 2.0 * opt.absorption_db_per_m * range;
    db = std::min(db, opt.max_gain_db);
    if (opt.undo_hardware_ramp && !m.tvg_time.empty()) {
      // Piecewise-linear analogue ramp, held flat outside its knots.
      const auto& tt = m.tvg_time;
      const auto& gg = m.tvg_gain_db;
      double hw;
      if (t <= tt.front()) {
        hw = gg.front();
      } else if (t >= tt.back()) {
        hw = gg.back();
      } else {
        const auto it = std::upper_bound(tt.begin(), tt.end(), t);
        const auto j = static_cast<std::size_t>(it - tt.begin());
        const double w = (t - tt[j - 1]) / (tt[j] - tt[j - 1]);
        hw = gg[j - 1] + w * (gg[j] - gg[j - 1]);
      }
      db -= hw;
    }
    gain[i] = static_cast<float>(std::pow(10.0, db / 20.0));
  }
  const std::size_t stride = s.stride();
  for (std::int64_t ch = 0; ch < s.n_channels; ++ch) {
    float* x = s.channel(ch);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t p = 0; p < stride; ++p) x[i * stride + p] *= gain[i];
    }
  }
  if (opt.undo_hardware_ramp) {
    m.tvg_time.clear();
    m.tvg_gain_db.clear();
  }
}

std::vector<float> whiten(const std::vector<PingRecord*>& records, const WhitenOptions& opt) {
  require_same_shape(records, "whiten");
  const auto n = static_cast<std::size_t>(records[0]->samples.n_samples);
  const double fs = records[0]->samples.sample_rate;

  // Power spectrum averaged over every channel of every record.
  std::vector<double> power(n, 0.0);
  std::size_t count = 0;
  std::vector<std::complex<float>> buf(n);
  for (PingRecord* r : records) {
    for (std::int64_t c = 0; c < r->samples.n_channels; ++c) {
      const std::complex<float>* x = r->samples.channel_complex(c);
      std::copy(x, x + n, buf.begin());
      fft(buf, false);
      for (std::size_t k = 0; k < n; ++k) power[k] += std::norm(buf[k]);
      ++count;
    }
  }
  for (auto& p : power) p /= static_cast<double>(count);
  const double bw = records[0]->transmit.waveform.bandwidth();
  const double smooth_hz = opt.smooth_hz > 0.0 ? opt.smooth_hz : (bw > 0.0 ? bw / 32.0 : fs / 256.0);
  const auto half = static_cast<std::int64_t>(std::llround(0.5 * smooth_hz / (fs / static_cast<double>(n))));
  power = smooth(power, half, true);

  double mean = 0.0;
  for (double p : power) mean += p;
  mean /= static_cast<double>(n);
  if (!(mean > 0.0)) return std::vector<float>(n, 1.0F);

  // Eq. 9, then h: scale so the largest gain is 1 (minimum attenuation 0 dB).
  std::vector<float> gain(n);
  double g_max = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const double g = 1.0 / (opt.gamma * mean + power[k]);
    g_max = std::max(g_max, g);
    gain[k] = static_cast<float>(g);
  }
  for (auto& g : gain) g = static_cast<float>(static_cast<double>(g) / g_max);

  for (PingRecord* r : records) {
    Samples& s = r->samples;
    parallel_for_host(s.n_channels, [&](std::int64_t c) {
      std::vector<std::complex<float>> x(n);
      std::complex<float>* ch = s.channel_complex(c);
      std::copy(ch, ch + n, x.begin());
      fft(x, false);
      for (std::size_t k = 0; k < n; ++k) x[k] *= opt.power_gain ? std::sqrt(gain[k]) : gain[k];
      fft(x, true);
      std::copy(x.begin(), x.end(), ch);
    });
  }
  return gain;
}

std::vector<float> median_tvg(const std::vector<PingRecord*>& records, const MedianTvgOptions& opt) {
  require_same_shape(records, "median_tvg");
  const auto n = static_cast<std::size_t>(records[0]->samples.n_samples);
  std::size_t population = 0;
  for (const PingRecord* r : records) population += static_cast<std::size_t>(r->samples.n_channels);

  // Population median of the power at each sample time.
  std::vector<double> med(n);
  parallel_for_host(static_cast<std::int64_t>(n), [&](std::int64_t i) {
    std::vector<float> v;
    v.reserve(population);
    for (const PingRecord* r : records) {
      for (std::int64_t c = 0; c < r->samples.n_channels; ++c) {
        v.push_back(std::norm(r->samples.channel_complex(c)[i]));
      }
    }
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    double m = *mid;
    if (v.size() % 2 == 0) {
      m = 0.5 * (m + static_cast<double>(*std::max_element(v.begin(), mid)));
    }
    med[static_cast<std::size_t>(i)] = m;
  });
  med = smooth(med, opt.smooth_samples / 2, false);
  double mean = 0.0;
  for (double m : med) mean += m;
  mean /= static_cast<double>(n);
  std::vector<float> gain(n, 1.0F);
  if (!(mean > 0.0)) return gain;
  for (std::size_t i = 0; i < n; ++i) gain[i] = static_cast<float>(1.0 / (opt.alpha * mean + med[i]));

  for (PingRecord* r : records) {
    for (std::int64_t c = 0; c < r->samples.n_channels; ++c) {
      std::complex<float>* x = r->samples.channel_complex(c);
      for (std::size_t i = 0; i < n; ++i) x[i] *= opt.power_gain ? std::sqrt(gain[i]) : gain[i];
    }
  }
  return gain;
}

double drc(std::vector<float>& magnitude, const DrcOptions& opt) {
  if (magnitude.empty()) return 1.0;
  if (!(opt.target_median > 0.0 && opt.target_median < 1.0)) {
    throw std::invalid_argument("drc: target_median must be in (0, 1)");
  }
  std::vector<float> sorted(magnitude);
  for (auto& v : sorted) v = std::abs(v);
  auto nth = [&](double pct) {
    auto idx = static_cast<std::size_t>(std::llround(pct / 100.0 * static_cast<double>(sorted.size() - 1)));
    idx = std::min(idx, sorted.size() - 1);
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(idx), sorted.end());
    return static_cast<double>(sorted[idx]);
  };
  const double scale = nth(opt.percentile);
  if (!(scale > 0.0)) {
    std::fill(magnitude.begin(), magnitude.end(), 0.0F);
    return 1.0;
  }
  // Median of the normalised, clipped image (clipping the top does not move the median).
  double median = nth(50.0) / scale;
  median = std::min(std::max(median, 1e-12), 1.0 - 1e-12);
  const double b = opt.target_median;
  const double q = (b - b * median) / (median - b * median);  // Eq. 12
  for (auto& v : magnitude) {
    const double i = std::min(static_cast<double>(std::abs(v)) / scale, 1.0);
    v = static_cast<float>(q * i / ((q - 1.0) * i + 1.0));  // Eq. 11
  }
  return q;
}

}  // namespace sonar::dsp
