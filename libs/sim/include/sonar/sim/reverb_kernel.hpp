// The steering kernel of scattering-function reverberation. Compiled by the host
// compiler and by nvcc.
//
// Each azimuth sector contributes one spectrum S_m[f] (its Gaussian sequence convolved
// with the pulse). A hydrophone receives it with the plane-wave lead of that direction:
//
//   total[ch, f] += S_m[f] * exp(i * omega[f] * lead[m, ch])
//
// One element of work is a block of `block` neighbouring frequency bins; it adds every
// sector into every channel for those bins, so no two elements write the same sample.
// The arithmetic is single precision in the order of python/gpusonar/sim/reverb.py: for a
// uniform line array the phase factor of channel ch is the factor of channel 0 times a
// per-sector stride, multiplied up channel by channel.
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/core/platform.hpp"

namespace sonar::sim {

inline constexpr std::int32_t kReverbMaxBlock = 16;

struct ReverbSteerKernel {
  const float* spectra;  ///< [n_sectors, n_low] interleaved complex
  const float* lead;     ///< [n_sectors, n_channels] lead of each channel for each sector, s
  const float* omega;    ///< [n_low] 2 pi (f + fc), rad/s
  float* total;          ///< [n_channels, n_low] interleaved complex; accumulated into
  std::int64_t n_sectors;
  std::int64_t n_channels;
  std::int64_t n_low;
  std::int32_t block;    ///< bins per element, 1..kReverbMaxBlock
  bool uniform;          ///< uniform line array: lead[m, ch] = lead[m, 0] + ch * (lead[m, 1] - lead[m, 0])

  /// Number of elements of work.
  SONAR_HD std::int64_t size() const { return (n_low + block - 1) / block; }

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t f0 = i * block;
    const std::int32_t nb = static_cast<std::int32_t>(n_low - f0 < block ? n_low - f0 : block);
    float row_re[kReverbMaxBlock], row_im[kReverbMaxBlock];
    float step_re[kReverbMaxBlock], step_im[kReverbMaxBlock];
    for (std::int64_t m = 0; m < n_sectors; ++m) {
      const float* s = spectra + 2 * (m * n_low + f0);
      const float* l = lead + m * n_channels;
      if (uniform) {
        const float first = l[0];
        const float stride = l[1] - l[0];
        for (std::int32_t b = 0; b < nb; ++b) {
          const float a0 = omega[f0 + b] * first;
          const float c0 = std::cos(a0), s0 = std::sin(a0);
          row_re[b] = s[2 * b] * c0 - s[2 * b + 1] * s0;
          row_im[b] = s[2 * b] * s0 + s[2 * b + 1] * c0;
          const float a1 = omega[f0 + b] * stride;
          step_re[b] = std::cos(a1);
          step_im[b] = std::sin(a1);
        }
        for (std::int64_t ch = 0; ch < n_channels; ++ch) {
          float* t = total + 2 * (ch * n_low + f0);
          for (std::int32_t b = 0; b < nb; ++b) {
            t[2 * b] += row_re[b];
            t[2 * b + 1] += row_im[b];
            const float re = row_re[b] * step_re[b] - row_im[b] * step_im[b];
            const float im = row_re[b] * step_im[b] + row_im[b] * step_re[b];
            row_re[b] = re;
            row_im[b] = im;
          }
        }
      } else {
        for (std::int64_t ch = 0; ch < n_channels; ++ch) {
          float* t = total + 2 * (ch * n_low + f0);
          const float lc = l[ch];
          for (std::int32_t b = 0; b < nb; ++b) {
            const float a = lc * omega[f0 + b];
            const float c = std::cos(a), sn = std::sin(a);
            t[2 * b] += s[2 * b] * c - s[2 * b + 1] * sn;
            t[2 * b + 1] += s[2 * b] * sn + s[2 * b + 1] * c;
          }
        }
      }
    }
  }
};

// --- The batched formulation (reverb_batched.hpp): everything between the shaped random
// sequences and the record on the device. One element of work is one sample of one
// component, so every kernel below is a plain map.

/// cos and sin of 2 pi `turns`. The turns are reduced in double precision first: a Doppler
/// phase reaches thousands of radians, where a single-precision angle is too coarse.
SONAR_HD inline void reverb_phasor(double turns, float& c, float& s) {
  const double frac = turns - std::floor(turns);
  const float a = static_cast<float>(6.283185307179586476925 * frac);
  c = std::cos(a);
  s = std::sin(a);
}

/// Half the taps of the band-limited interpolation that time-scales a pulse (per-frequency
/// Doppler, `ReverbParams::per_frequency_doppler`): a Blackman-windowed sinc of 2 x 8 taps.
/// Ours: measured error below -75 dB of the peak for tones up to a sixth of the rate (the
/// generator's rate is at least four times the band, so the band lies within an eighth).
inline constexpr std::int32_t kReverbScaleHalf = 8;

/// Sample k of a pulse time-scaled by `eta`: p(eta s) for the signed sample s of k (k, or
/// k - n_low past the middle: the pulse is periodic in n_low, as its transform is), from the
/// pulse's own samples by band-limited interpolation. `pulse` is interleaved complex, [n_taps]
/// (n_taps <= n_low; zero from n_taps to n_low). The echo of a still boundary seen from a
/// moving platform is the pulse time-scaled by Gamma = 1 - dT/dt, so that a line at f comes
/// back at Gamma f (TR 0702 eqs. (63)-(64)). python/gpusonar/sim/reverb.py scaled_pulse is the
/// same in NumPy.
template <class T>
SONAR_HD inline void reverb_scaled_pulse(const T* pulse, std::int64_t n_taps, std::int64_t n_low, std::int64_t k,
                                         double eta, double& re, double& im) {
  constexpr std::int32_t h = kReverbScaleHalf;
  const double pi = 3.14159265358979323846;
  const std::int64_t s = 2 * k <= n_low ? k : k - n_low;
  const double x = eta * static_cast<double>(s);
  const double j0 = std::floor(x);
  const double fr = x - j0;
  const std::int64_t first = static_cast<std::int64_t>(j0) - h + 1;   // the taps are first .. first + 2h - 1
  std::int64_t m = first % n_low;
  if (m < 0) m += n_low;
  re = 0.0;
  im = 0.0;
  if (!(m < n_taps || m + 2 * h - 1 >= n_low)) return;                // no tap on the pulse
  const double sp = std::sin(pi * fr);
  // The window's angle pi d / h at the distance d = fr - i of tap i = t - h + 1, stepped down
  // by pi / h from tap to tap.
  double ca = std::cos(pi * (fr + h - 1) / h), sa = std::sin(pi * (fr + h - 1) / h);
  const double cs = std::cos(pi / h), ss = std::sin(pi / h);
  for (std::int32_t t = 0; t < 2 * h; ++t) {
    const std::int32_t i = t - h + 1;
    std::int64_t idx = m + t;
    while (idx >= n_low) idx -= n_low;
    if (idx < n_taps) {
      const double d = fr - static_cast<double>(i);
      // sin(pi (fr - i)) = (-1)^i sin(pi fr)
      const double sinc = d == 0.0 ? 1.0 : (i % 2 == 0 ? sp : -sp) / (pi * d);
      const double window = 0.42 + 0.5 * ca + 0.08 * (2.0 * ca * ca - 1.0);
      re += sinc * window * static_cast<double>(pulse[2 * idx]);
      im += sinc * window * static_cast<double>(pulse[2 * idx + 1]);
    }
    const double nc = ca * cs + sa * ss;
    sa = sa * cs - ca * ss;
    ca = nc;
  }
}

/// numpy.interp at x with zero outside the grid (the reference code walks the grid; a
/// kernel element finds its interval by bisection).
SONAR_HD inline double reverb_interp(double x, const double* xp, const double* fp, std::int64_t n) {
  if (!(x >= xp[0] && x <= xp[n - 1])) return 0.0;
  std::int64_t lo = 0, hi = n - 1;   // xp[lo] <= x, and x < xp[hi] unless hi is still the last node
  while (hi - lo > 1) {
    const std::int64_t mid = (lo + hi) / 2;
    if (xp[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  if (xp[hi] <= x) lo = hi;
  if (lo == n - 1 || xp[lo] == x) return fp[lo];
  const double slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

/// Each component's sequence from its normal deviates: scaled by the standard deviation of
/// the component at that time, sigma = sqrt(density(t) / fs_low * dphi / 2), and shifted by the
/// component's Doppler: x[j, k] = sigma_j(t_k) g[j, k] exp(2 pi i doppler[j] (t_k - t_0)).
struct ReverbShapeKernel {
  float* x;               ///< [count, n_low] interleaved complex: the deviates in, the sequence out
  const double* coarse;   ///< [n_coarse] times of the density, s
  const double* density;  ///< [count, n_coarse] the rows of these components, or [1, n_coarse] when shared
  const double* doppler;  ///< [count] Hz
  std::int64_t n_low, n_coarse;
  double time0, fs_low, dphi;
  bool shared;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t j = i / n_low, k = i % n_low;
    const double t = time0 + static_cast<double>(k) / fs_low;
    const double* row = density + (shared ? 0 : j) * n_coarse;
    const double sigma = std::sqrt(reverb_interp(t, coarse, row, n_coarse) / fs_low * dphi / 2.0);
    const auto re = static_cast<float>(sigma * static_cast<double>(x[2 * i]));
    const auto im = static_cast<float>(sigma * static_cast<double>(x[2 * i + 1]));
    const double d = doppler[j];
    if (d == 0.0) {
      x[2 * i] = re;
      x[2 * i + 1] = im;
      return;
    }
    float c, s;
    reverb_phasor(d * (t - time0), c, s);
    x[2 * i] = re * c - im * s;
    x[2 * i + 1] = re * s + im * c;
  }
};

/// The pulse of each component, shifted by the component's Doppler and zero padded:
/// p[j, k] = pulse[k] exp(2 pi i doppler[j] k / fs_low) for k < n_taps.
/// With `per_frequency` the pulse is also time-scaled by Gamma = 1 + doppler[j] / fc, so that
/// every frequency f of it is shifted by doppler[j] f / fc:
/// p[j, k] = pulse(Gamma s) exp(2 pi i doppler[j] s / fs_low), s the signed sample of k
/// (reverb_scaled_pulse).
struct ReverbPulseKernel {
  const float* pulse;     ///< [n_taps] interleaved complex
  const double* doppler;  ///< [count] Hz
  float* p;               ///< [count, n_low] interleaved complex
  std::int64_t n_low, n_taps;
  double fs_low;
  bool per_frequency;
  double center_frequency;  ///< fc, Hz: Gamma = 1 + doppler / fc (per_frequency only)

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t j = i / n_low, k = i % n_low;
    if (per_frequency && doppler[j] != 0.0) {
      const double d = doppler[j];
      double re = 0.0, im = 0.0;
      reverb_scaled_pulse(pulse, n_taps, n_low, k, 1.0 + d / center_frequency, re, im);
      const std::int64_t s = 2 * k <= n_low ? k : k - n_low;
      float c, sn;
      reverb_phasor(d * (static_cast<double>(s) * (1.0 / fs_low)), c, sn);
      const auto a = static_cast<float>(re), b = static_cast<float>(im);
      p[2 * i] = a * c - b * sn;
      p[2 * i + 1] = a * sn + b * c;
      return;
    }
    if (k >= n_taps) {
      p[2 * i] = 0.0F;
      p[2 * i + 1] = 0.0F;
      return;
    }
    const float re = pulse[2 * k], im = pulse[2 * k + 1];
    const double d = doppler[j];
    if (d == 0.0) {
      p[2 * i] = re;
      p[2 * i + 1] = im;
      return;
    }
    float c, s;
    reverb_phasor(d * (static_cast<double>(k) * (1.0 / fs_low)), c, s);
    p[2 * i] = re * c - im * s;
    p[2 * i + 1] = re * s + im * c;
  }
};

/// Spectrum of the sequence times spectrum of the pulse: x[j, f] *= p[j, f], or with
/// `shared` the one pulse spectrum p[f] of components without a Doppler shift.
struct ReverbProductKernel {
  float* x;        ///< [count, n_low] interleaved complex
  const float* p;  ///< [count, n_low], or [n_low] when shared
  std::int64_t n_low;
  bool shared;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t at = shared ? i % n_low : i;
    const float a = x[2 * i], b = x[2 * i + 1];
    const float pr = p[2 * at], pi = p[2 * at + 1];
    x[2 * i] = a * pr - b * pi;
    x[2 * i + 1] = a * pi + b * pr;
  }
};

/// The accumulated spectrum of each channel laid out for the transform to the record's
/// rate: bins below the middle at the start, the others at the end, zeros between.
struct ReverbPadKernel {
  const float* total;  ///< [n_channels, n_low] interleaved complex
  float* series;       ///< [n_channels, n_full] interleaved complex
  std::int64_t n_low, n_full;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t ch = i / n_full, k = i % n_full;
    const std::int64_t half = n_low / 2;
    std::int64_t src = -1;
    if (k < half) {
      src = k;
    } else if (k >= n_full - (n_low - half)) {
      src = k - (n_full - n_low);
    }
    if (src < 0) {
      series[2 * i] = 0.0F;
      series[2 * i + 1] = 0.0F;
    } else {
      series[2 * i] = total[2 * (ch * n_low + src)];
      series[2 * i + 1] = total[2 * (ch * n_low + src) + 1];
    }
  }
};

/// The record's window of the transformed series, with the transform's 1 / n_full, the
/// interpolation gain and the carrier.
struct ReverbWindowKernel {
  const float* series;  ///< [n_channels, n_full] interleaved complex, the unscaled inverse transform
  float* out;           ///< [n_channels, n] interleaved complex, or [n_channels, n] real
  std::int64_t n, n_full, start, decimation;
  double sample_rate, start_time, center_frequency;
  bool carrier, real_output;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t ch = i / n, k = i % n;
    const std::int64_t at = start + k;
    float re = 0.0F, im = 0.0F;
    if (at >= 0 && at < n_full) {
      const float scale = 1.0F / static_cast<float>(n_full);
      re = series[2 * (ch * n_full + at)] * scale;
      im = series[2 * (ch * n_full + at) + 1] * scale;
    }
    if (decimation > 1) {
      const float gain = static_cast<float>(decimation);
      re *= gain;
      im *= gain;
    }
    if (carrier) {
      float c, s;
      reverb_phasor(center_frequency * (start_time + static_cast<double>(k) / sample_rate), c, s);
      const float r2 = re * c - im * s, i2 = re * s + im * c;
      re = r2;
      im = i2;
    }
    if (real_output) {
      out[i] = re;
    } else {
      out[2 * i] = re;
      out[2 * i + 1] = im;
    }
  }
};

}  // namespace sonar::sim
