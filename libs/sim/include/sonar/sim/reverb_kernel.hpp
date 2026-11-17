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

}  // namespace sonar::sim
