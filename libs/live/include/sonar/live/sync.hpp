// Board synchronisation on a shared reference signal, in C++: the counterpart of the
// `reference` method of python/gpusonar/sync.py (docs/DECISIONS.md D-32). Each capture board
// samples on its own clock. Two short chirp bursts per ping, wired to one input of every
// board, give each board's clock offset (one burst) and drift (their spacing); the boards
// are then resampled onto the first board's timeline.
#pragma once

#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

struct Reference {
  double f_start = 38e3, f_stop = 48e3;  ///< Hz
  double duration = 4e-3;                ///< s, each burst
  double lead = 5e-3;                    ///< s from the window's opening to the first burst
  double tail = 10e-3;                   ///< s from the end of the second burst to the window's close
  bool dedicated = false;                ///< the reference inputs carry no hydrophone signal: drop them after
  double search_time = 2e-3;             ///< s either side of where a burst is expected
};

struct SyncResult {
  std::vector<double> offset;     ///< s, per board, relative to the first
  std::vector<double> drift_ppm;  ///< per board, relative to the first
  std::vector<double> quality;    ///< correlation peak quality, 0..1
  std::vector<double> residual;   ///< s, misalignment measured again after resampling
  double tolerance = 0.0;         ///< s: a twentieth of a period at the top of the band
  bool ok = false;
};

/// Estimate every board's clock from the bursts, resample the boards onto the first one's
/// timeline in place, measure what is left, and (for dedicated reference inputs) remove those
/// channels. The record must hold raw real samples.
SyncResult synchronise(PingRecord& record, const Reference& reference);

/// The record without the given channel rows (samples, geometry, calibration, channel maps).
void drop_channels(PingRecord& record, const std::vector<std::int64_t>& rows);

}  // namespace sonar::live
