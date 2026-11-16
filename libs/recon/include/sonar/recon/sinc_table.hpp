// Kaiser-windowed sinc interpolation table shared by the host and CUDA backends.
#pragma once

#include <vector>

namespace sonar::recon {

/// (kSincPhases + 1) rows of kSincTaps coefficients. Row p interpolates at fraction
/// p / kSincPhases between sample n and n + 1; tap k applies to sample n - 3 + k.
/// Each row sums to one.
std::vector<float> make_sinc_table();

}  // namespace sonar::recon
