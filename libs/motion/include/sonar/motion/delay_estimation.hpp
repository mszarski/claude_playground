// Coarse and fine time-delay estimation between overlapping phase-centre channels of two
// consecutive pings. The coarse delay is ASASIN Sec. II-D's (its ref. [11], Saebo 2010, as
// ASASIN describes it): the quadratic-interpolated peak of the correlation magnitude. The
// fine delay is the phase of the complex correlation at the *integer* peak, ambiguous by
// whole carrier periods. That is ours, inherited from correlate_window: ASASIN takes the
// phase at the interpolated point (docs/PAPER_NOTES.md, "Delay measurement").
//
// This is the port of python/gpusonar/motion.py `correlate_window`, which stays the
// reference (tests/python/test_motion_delays.py). Sums are in double. Host and CUDA run
// the same kernels (delay_kernel.hpp).
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "sonar/core/platform.hpp"

namespace sonar::motion {

struct DelayEstimate {
  double coarse = 0.0;     ///< s, delay of the second ping's channel relative to the first's
  double fine = 0.0;       ///< s, within half a carrier period of zero
  double coherence = 0.0;  ///< |correlation| at the peak over the two windows' norms
  bool valid = false;      ///< false: the magnitude peaked at the edge of the lag span
};

/// Delays for every (channel pair, window centre): result[p * centres.size() + w] for
/// pair p (channel_a[p] of `a` against channel_b[p] of `b`) and centre w.
/// `a`, `b`: [n_channels, n_samples] complex baseband, row-major. Each window is
/// [centre - half, centre + half) on `a`, shifted by lags -max_lag..max_lag on `b`; the
/// caller keeps every window inside the record (throws otherwise).
std::vector<DelayEstimate> estimate_delays(const std::complex<float>* a, const std::complex<float>* b,
                                           std::int64_t n_channels, std::int64_t n_samples,
                                           const std::vector<std::int32_t>& channel_a,
                                           const std::vector<std::int32_t>& channel_b,
                                           const std::vector<std::int64_t>& centres, int half, int max_lag,
                                           double sample_rate, double center_frequency,
                                           Backend backend = Backend::Host, int device = 0);

}  // namespace sonar::motion
