// The CFAR kernel body. Compiled by the host compiler and by nvcc.
#pragma once

#include <cstdint>

#include "sonar/core/platform.hpp"

namespace sonar::detect {

struct CfarKernel {
  const float* power;        ///< [n_rows * n_cols]
  const double* factor;      ///< threshold multiplier per training-cell count, [2 * train + 1]
  unsigned char* detections; ///< [n_rows * n_cols]
  float* ratio;              ///< [n_rows * n_cols] or null
  std::int64_t n_cols;
  std::int32_t guard;
  std::int32_t train;

  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t col = i % n_cols;
    const float* row = power + (i - col);
    // Training cells: [col - guard - train, col - guard) and (col + guard, col + guard + train].
    std::int64_t lo0 = col - guard - train, lo1 = col - guard;
    std::int64_t hi0 = col + guard + 1, hi1 = col + guard + 1 + train;
    if (lo0 < 0) lo0 = 0;
    if (lo1 < 0) lo1 = 0;
    if (hi0 > n_cols) hi0 = n_cols;
    if (hi1 > n_cols) hi1 = n_cols;
    double sum = 0.0;
    for (std::int64_t k = lo0; k < lo1; ++k) sum += static_cast<double>(row[k]);
    for (std::int64_t k = hi0; k < hi1; ++k) sum += static_cast<double>(row[k]);
    std::int64_t count = (lo1 - lo0) + (hi1 - hi0);
    if (count < 1) count = 1;
    const double mean = sum / static_cast<double>(count);
    const double p = static_cast<double>(row[col]);
    detections[i] = p > factor[count] * mean ? 1 : 0;
    if (ratio != nullptr) ratio[i] = static_cast<float>(p / (mean > 1e-300 ? mean : 1e-300));
  }
};

}  // namespace sonar::detect
