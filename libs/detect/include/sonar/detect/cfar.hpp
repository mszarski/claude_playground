// Cell-averaging CFAR and the split-window normaliser on range x bearing power maps.
// One kernel body for the host and CUDA backends (docs/DECISIONS.md A-03).
// Sources: docs/REFERENCES.md Sec. 12.1 (cell-averaging factor N (pfa^(-1/N) - 1)).
#pragma once

#include <cstdint>
#include <vector>

#include "sonar/core/platform.hpp"

namespace sonar::detect {

/// The window is counted in CELLS of the map's range axis, and the defaults (4 guard, 32
/// training) assume that one cell is about one range resolution cell. That fails silently
/// on a finely sampled map (docs/HARDCODED_AUDIT.md K-5): at a 6 cm step with a 2 m
/// compressed pulse, 4 guard cells lie inside the target, which then raises its own
/// background and masks itself. `cfar_window` turns a window stated in resolution cells or
/// metres into cells for a given map; with its defaults and a map sampled at the
/// resolution it gives the (4, 32) above.
///
/// The default false-alarm probability here is 1e-4. The Holoscan DetectOp
/// (python/gpusonar/holoscan_ops.py) defaults to 1e-6 and the gallery uses 1e-5 to 1e-7:
/// there is no single right value, and a caller should state its own.
struct CfarOptions {
  double pfa = 1e-4;        ///< false-alarm probability per cell in complex Gaussian noise
  std::int32_t guard = 4;   ///< cells skipped on each side of the cell under test, along range
  std::int32_t train = 32;  ///< training cells on each side, along range
};

/// A CFAR window stated physically.
struct CfarWindow {
  /// Guard and training extent on each side, in range resolution cells (c / 2B, or
  /// c tau / 2 for a tone). The defaults are CfarOptions' cell counts.
  double guard_resolutions = 4.0;
  double train_resolutions = 32.0;
  /// Guard and training extent on each side in metres; > 0 replaces the count above (the
  /// guard is still at least one resolution cell, so a target cannot train on itself).
  double guard_m = 0.0;
  double train_m = 0.0;
};

/// Guard and training cells for a map whose range axis steps by `range_step` metres, for a
/// sonar whose range resolution is `resolution` metres. Each is rounded up and at least 1
/// training cell and 0 guard cells; `pfa` is passed through.
CfarOptions cfar_window(double range_step, double resolution, const CfarWindow& window = {},
                        double pfa = CfarOptions{}.pfa);

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
