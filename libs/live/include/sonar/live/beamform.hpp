// Far-field beamformers for a long-range (flank) array, one ping at a time, in C++: the
// counterparts of farfield_beamform and mvdr_beamform in python/gpusonar/flank.py.
//
//   farfield: conventional delay and sum with plane-wave steering, in the frequency domain.
//             It is what backprojection reduces to at long range, and cheaper per ping.
//   mvdr:     adaptive (minimum-variance distortionless-response) weights per sub-band and
//             range block, from the covariance of the other range blocks. It nulls what
//             extends in range (jammers, directional noise, reverberation) and narrows
//             the beam on what does not.
//
// Both take a conditioned record (complex baseband, pulse compressed: live::prepare) and
// give a bearing x time map whose time axis is the two-way delay to the array centre.
#pragma once

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"

namespace sonar::live {

/// The receive array in the world frame. Forward is the element boresight; across is the
/// direction the elements run along (their principal axis, made perpendicular to forward
/// and pointing the way the element index increases); normal = forward x across.
struct ArrayAxes {
  Vec3d origin, forward, across, normal;
  std::vector<Vec3d> positions;  ///< world, calibration position offsets included
};
ArrayAxes array_axes(const PingRecord& record);

/// Per-channel aperture weights: "rect", "hann" or "hamming". The window is laid across the
/// aperture and reaches its end value half a pitch beyond the outer elements.
std::vector<float> aperture_weights(const PingRecord& record, const std::string& window);

/// Plane-wave delays p_n . u / c, [n_bearing][n_elements] row major: positive where the
/// element is nearer the source and hears it earlier.
std::vector<double> steering_delays(const PingRecord& record, const std::vector<double>& bearings);

struct BeamMap {
  std::int64_t n_bearings = 0, n_samples = 0;
  std::int64_t n_valid = 0;                ///< columns with a fully overlapped replica; later ones are not usable
  std::vector<std::complex<float>> data;   ///< [n_bearings][n_samples] row major
  std::vector<double> bearings;            ///< rad, from the boresight towards increasing element index
  std::vector<double> times;               ///< two-way time of each column, s
  std::vector<double> ranges;              ///< c t / 2, m

  const std::complex<float>* row(std::int64_t bearing) const { return data.data() + bearing * n_samples; }
  /// The map without the tail where the matched filter only partly overlaps the data (the
  /// last pulse length of the record). Detection must not run on that tail.
  BeamMap valid() const;
};

/// Conventional far-field beamformer: B(u, f) = sum_n w_n X_n(f) exp(-j 2 pi (f + fc) p_n . u / c).
/// Each channel is delayed by its plane-wave lead, exactly (a phase ramp over frequency).
/// On the CUDA backend the transforms and the steered sums run on the device
/// (beamform_batched.hpp); `batched` runs that formulation on the host (for tests).
BeamMap farfield_beamform(const PingRecord& record, const std::vector<double>& bearings, const std::string& window = "rect",
                          Backend backend = Backend::Host, int device = 0, bool batched = false);

struct MvdrOptions {
  std::int64_t n_subbands = 32;   ///< narrow against c / aperture, so one steering vector fits each
  double loading = 3e-2;          ///< diagonal loading, as a fraction of the mean element power in the training data
  std::int64_t block = 256;       ///< samples per range block
  std::int64_t guard_blocks = 1;  ///< blocks either side of a block that are kept out of its training data
};

/// Adaptive beamformer. In every sub-band and range block the element covariance R comes
/// from the other range blocks (all but the block and its guards: an echo in its own
/// training data would be cancelled), and w = R^-1 a / (a^H R^-1 a). Sub-band outputs are
/// summed, which restores range resolution; the output is scaled by the number of elements
/// so that a target reads the same as in the conventional sum.
/// On the CUDA backend the sub-band transforms, covariances, Cholesky factors, solves and
/// the application of the weights run on the device (beamform_batched.hpp); `batched` runs
/// that formulation on the host (for tests). A covariance that is not positive definite
/// (possible without loading) sends the whole call to the host code, which then uses LU.
BeamMap mvdr_beamform(const PingRecord& record, const std::vector<double>& bearings, const MvdrOptions& options = {},
                      Backend backend = Backend::Host, int device = 0, bool batched = false);

/// Seconds in each step of the batched beamformers since the last reset (beamform_batched.hpp).
struct BeamformProfile;
BeamformProfile beamform_profile();
void beamform_profile_reset();

}  // namespace sonar::live
