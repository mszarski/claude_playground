// Kernels of delay estimation (delay_estimation.hpp), written once for the host and the
// device (docs/DECISIONS.md A-03). C++17, no std::complex: samples are interleaved floats.
#pragma once

#include <math.h>  // ::hypot, ::atan2, ::sqrt: callable on the host and the device
#include <cstdint>

#include "sonar/core/platform.hpp"

namespace sonar::motion {

/// One thread per (task, lag): the complex correlation sum_j conj(a_j) b_{j+lag} over the
/// window, and the energy of the shifted window of b. task = pair * n_centres + centre.
struct CorrelateKernel {
  const float* a;              // [n_channels, n_samples, 2]
  const float* b;
  const std::int32_t* chan_a;  // [n_pairs]
  const std::int32_t* chan_b;
  const std::int64_t* centres; // [n_centres]
  std::int64_t n_samples, n_centres;
  int half, max_lag;
  double* corr;                // [n_tasks, n_lags, 2]
  double* energy_b;            // [n_tasks, n_lags]

  SONAR_HD void operator()(std::int64_t idx) const {
    const std::int64_t n_lags = 2 * max_lag + 1;
    const std::int64_t task = idx / n_lags;
    const std::int64_t lag = idx % n_lags - max_lag;
    const std::int64_t pair = task / n_centres;
    const std::int64_t start = centres[task % n_centres] - half;
    const float* pa = a + 2 * (static_cast<std::int64_t>(chan_a[pair]) * n_samples + start);
    const float* pb = b + 2 * (static_cast<std::int64_t>(chan_b[pair]) * n_samples + start + lag);
    // Four interleaved partial sums (samples j, j + 1, j + 2, j + 3 go to different ones):
    // independent dependency chains, so the loop pipelines; still all in double.
    double re[4] = {0.0, 0.0, 0.0, 0.0}, im[4] = {0.0, 0.0, 0.0, 0.0}, eb[4] = {0.0, 0.0, 0.0, 0.0};
    const std::int64_t n = 2 * static_cast<std::int64_t>(half);
    for (std::int64_t j = 0; j < n; ++j) {
      const int q = static_cast<int>(j & 3);
      const double ar = pa[2 * j], ai = pa[2 * j + 1];
      const double br = pb[2 * j], bi = pb[2 * j + 1];
      re[q] += ar * br + ai * bi;
      im[q] += ar * bi - ai * br;
      eb[q] += br * br + bi * bi;
    }
    corr[2 * idx] = (re[0] + re[1]) + (re[2] + re[3]);
    corr[2 * idx + 1] = (im[0] + im[1]) + (im[2] + im[3]);
    energy_b[idx] = (eb[0] + eb[1]) + (eb[2] + eb[3]);
  }
};

/// One thread per task: the peak, its quadratic interpolation, the phase and the coherence.
struct PickKernel {
  const float* a;
  const std::int32_t* chan_a;
  const std::int64_t* centres;
  std::int64_t n_samples, n_centres;
  int half, max_lag;
  double sample_rate, center_frequency;
  const double* corr;
  const double* energy_b;
  double* out;                 // [n_tasks, 4]: coarse, fine, coherence, valid (0 or 1)

  SONAR_HD void operator()(std::int64_t task) const {
    const std::int64_t n_lags = 2 * max_lag + 1;
    const double* c = corr + 2 * task * n_lags;
    // The first maximum, as numpy.argmax.
    std::int64_t best = 0;
    double best_mag = -1.0;
    for (std::int64_t l = 0; l < n_lags; ++l) {
      const double m = ::hypot(c[2 * l], c[2 * l + 1]);
      if (m > best_mag) {
        best_mag = m;
        best = l;
      }
    }
    double* o = out + 4 * task;
    if (best == 0 || best == n_lags - 1) {
      o[0] = o[1] = o[2] = o[3] = 0.0;
      return;
    }
    const double y0 = ::hypot(c[2 * (best - 1)], c[2 * (best - 1) + 1]);
    const double y2 = ::hypot(c[2 * (best + 1)], c[2 * (best + 1) + 1]);
    const double frac = 0.5 * (y0 - y2) / (y0 - 2.0 * best_mag + y2);
    const double kTwoPi = 6.283185307179586476925286766559;
    o[0] = (static_cast<double>(best - max_lag) + frac) / sample_rate;
    // Baseband echoes carry exp(-j 2 pi fc tau), so the correlation phase is -2 pi fc delta.
    o[1] = -::atan2(c[2 * best + 1], c[2 * best]) / (kTwoPi * center_frequency);
    const std::int64_t pair = task / n_centres;
    const std::int64_t start = centres[task % n_centres] - half;
    const float* pa = a + 2 * (static_cast<std::int64_t>(chan_a[pair]) * n_samples + start);
    double ea = 0.0;
    for (std::int64_t j = 0; j < 2 * half; ++j) {
      const double ar = pa[2 * j], ai = pa[2 * j + 1];
      ea += ar * ar + ai * ai;
    }
    o[2] = best_mag / ::sqrt(ea * energy_b[task * n_lags + best] + 1e-300);
    o[3] = 1.0;
  }
};

}  // namespace sonar::motion
