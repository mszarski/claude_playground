// The multiscale detector's statistic (detector.hpp, multiscale_cells) as kernel bodies: compiled by the host compiler
// and by nvcc (docs/DECISIONS.md A-03). The host reference is the loop code in detector.cpp; these are the same
// operations in the same order, one thread per item, so that the device gives the reference's flags:
//   DetectorMedianKernel   one training length of one bearing at either end of the map: its median (the slope's ends)
//   DetectorRowKernel      one bearing: the running sum of its cell means and the background's slope at both ends
//   DetectorCellKernel     one cell: its background, its strongest sample and whether any test passes
// What can differ from the host: exp and log of the device's maths library against the host's (each within an ulp or
// two), so the background differs in the last bits; a flag can differ only where a statistic lies within that of its
// threshold. Nothing here is contracted into a fused multiply-add: no product is added to anything, and
// detector_cuda.cu is compiled with --fmad=false (libs/live/CMakeLists.txt) so that stays so if the code changes.
// A NaN in the map: the host's median sorts with std::sort, whose order with a NaN is unspecified, and the median kernel
// with an insertion sort, so a NaN within the first or last two training lengths of a bearing can give the two a
// different slope there (detector.hpp, multiscale_cells). The chain's maps are powers of finite samples.
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/core/platform.hpp"

namespace sonar::live {

struct DetectorMedianKernel {
  const float* cell_mean;   ///< [n_bearing * n_cells]
  double* scratch;          ///< [n_bearing * 4 * train]
  double* median;           ///< [n_bearing * 4]: the medians of the cells from 0, train, n_cells - 2 train, n_cells - train
  std::int64_t n_cells;
  std::int64_t train;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t b = i / 4, which = i % 4;
    const std::int64_t first = which == 0 ? 0 : (which == 1 ? train : (which == 2 ? n_cells - 2 * train : n_cells - train));
    const float* row = cell_mean + b * n_cells + first;
    double* v = scratch + i * train;
    // Insertion sort: the median of the same values whatever the sort.
    for (std::int64_t q = 0; q < train; ++q) {
      const auto x = static_cast<double>(row[q]);
      std::int64_t p = q;
      while (p > 0 && v[p - 1] > x) {
        v[p] = v[p - 1];
        --p;
      }
      v[p] = x;
    }
    const std::int64_t h = train / 2;
    median[i] = train == 0 ? 0.0 : (train % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]));
  }
};

struct DetectorRowKernel {
  const float* cell_mean;   ///< [n_bearing * n_cells]
  const double* median;     ///< [n_bearing * 4], read when `sloped`
  double* csum;             ///< [n_bearing * (n_cells + 1)]
  double* slope;            ///< [n_bearing * 2]: near end, far end; d ln(power) / d cell, positive when it falls with range
  std::int64_t n_cells;
  std::int64_t train;
  bool sloped;

  SONAR_HD void operator()(std::int64_t b) const {
    const float* row = cell_mean + b * n_cells;
    double* c = csum + b * (n_cells + 1);
    c[0] = 0.0;
    for (std::int64_t j = 0; j < n_cells; ++j) c[j + 1] = c[j] + static_cast<double>(row[j]);
    double near = 0.0, far = 0.0;
    if (sloped) {
      const double t = static_cast<double>(train);
      const double* m = median + 4 * b;
      if (m[0] > 0.0 && m[1] > 0.0) near = std::log(m[0] / m[1]) / t;
      if (m[2] > 0.0 && m[3] > 0.0) far = std::log(m[2] / m[3]) / t;
    }
    slope[2 * b] = near;
    slope[2 * b + 1] = far;
  }
};

struct DetectorCellKernel {
  const float* fine;            ///< [n_bearing * fine_stride]
  std::int64_t fine_stride, fine_offset;
  const double* csum;           ///< DetectorRowKernel's
  const double* slope;          ///< DetectorRowKernel's
  const double* offset;         ///< [n_cells]: of the training cells' centroid from the cell, cells (0 where balanced)
  const double* threshold;      ///< [n_scales * n_cells]: each test's threshold over the background, by range cell
  const std::int64_t* cells;    ///< [n_scales]: 0 for the strongest sample, m >= 1 for the mean of m cells
  std::int64_t n_scales;
  std::int64_t n_cells, k, guard, train;
  double* level;                ///< [n]: the background in double
  float* background;            ///< [n]
  float* peak;                  ///< [n]
  unsigned char* flag;          ///< [n]

  /// std::clamp's result.
  SONAR_HD static std::int64_t clamp(std::int64_t v, std::int64_t lo, std::int64_t hi) { return v < lo ? lo : (hi < v ? hi : v); }

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t b = i / n_cells, j = i % n_cells;
    const double* c = csum + b * (n_cells + 1);
    const std::int64_t lo0 = clamp(j - guard - train, 0, n_cells), lo1 = clamp(j - guard, 0, n_cells);
    const std::int64_t hi0 = clamp(j + guard + 1, 0, n_cells), hi1 = clamp(j + guard + 1 + train, 0, n_cells);
    std::int64_t count = (lo1 - lo0) + (hi1 - hi0);
    if (count < 1) count = 1;
    double mean = (c[lo1] - c[lo0] + c[hi1] - c[hi0]) / static_cast<double>(count);
    const double off = offset[j];
    if (off != 0.0) {
      double e = (off > 0.0 ? slope[2 * b] : slope[2 * b + 1]) * off;
      e = e < -2.3 ? -2.3 : (2.3 < e ? 2.3 : e);
      mean *= std::exp(e);
    }
    level[i] = mean;
    background[i] = static_cast<float>(mean);
    const float* f = fine + b * fine_stride + fine_offset + j * k;
    float top = 0.0F;
    for (std::int64_t s = 0; s < k; ++s) top = top < f[s] ? f[s] : top;
    peak[i] = top;
    const double floor = mean < 1e-300 ? 1e-300 : mean;          // std::max(mean, 1e-300)
    bool hit = static_cast<double>(top) > threshold[j] * floor;
    for (std::int64_t s = 1; s < n_scales && !hit; ++s) {
      const std::int64_t m = cells[s], a = j - (m - 1) / 2, e = a + m;
      if (a < 0 || e > n_cells) continue;
      hit = (c[e] - c[a]) / static_cast<double>(m) > threshold[s * n_cells + j] * floor;
    }
    flag[i] = hit ? 1 : 0;
  }
};

}  // namespace sonar::live
