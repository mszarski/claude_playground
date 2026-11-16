// Cell-averaging CFAR and the split-window normaliser on range x bearing power maps.
// One kernel body for the host and CUDA backends (docs/DECISIONS.md A-03).
#pragma once

#include <cstdint>
#include <vector>

#include "sonar/core/platform.hpp"

namespace sonar::detect {

struct CfarOptions {
  double pfa = 1e-4;        ///< false-alarm probability per cell in complex Gaussian noise
  std::int32_t guard = 4;   ///< cells skipped on each side of the cell under test, along range
  std::int32_t train = 32;  ///< training cells on each side, along range
};

/// Threshold multiplier for the mean of `n_train` exponentially distributed cells.
double cfar_threshold_factor(double pfa, std::int32_t n_train);

/// Cell-averaging CFAR along range on `power` [n_rows (bearing), n_cols (range)], row major.
/// `detections` receives 1 where power exceeds the threshold, else 0. `ratio`, when not
/// null, receives power divided by the local background mean (the normaliser's output).
/// At the edges the training window uses the cells that exist.
void cfar(Backend backend, const float* power, std::int64_t n_rows, std::int64_t n_cols,
          const CfarOptions& options, unsigned char* detections, float* ratio = nullptr,
          int device = 0);

}  // namespace sonar::detect
