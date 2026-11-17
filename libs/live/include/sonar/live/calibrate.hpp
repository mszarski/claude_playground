// Array calibration from known sources, in C++: the counterpart of
// python/gpusonar/calibrate.py. Per-channel gain, phase and element position errors.
//
// Each calibration ping holds one strong echo (or source) at a known position. Sampling
// every channel at the delay the nominal geometry predicts and restoring the carrier gives
// one complex number per channel and ping. Without errors those are equal across channels.
// With errors,
//
//   v[c, k] = A[k] g[c] exp(j (phi[c] + (2 pi / lambda) u[k] . dp[c]))
//
// for gain g, phase phi and position error dp of channel c, and unit vector u from the
// array to source k. Gains come from magnitudes; phi and dp from a linear least-squares fit
// over sources at different bearings (three or more for a 2-D position error).
//
// What cannot be seen, and is fixed by convention: the mean gain (set to 1), the mean phase
// (0) and the mean position error (0) of the array. They are common to every channel and
// do not affect beamforming.
#pragma once

#include <complex>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

struct ChannelErrors {
  std::vector<double> gain;      ///< [channels], mean 1
  std::vector<double> phase;     ///< [channels] rad, mean 0
  std::vector<Vec3d> position;   ///< [channels] body frame, m, mean 0
  double residual_rms_deg = 0.0; ///< phase left unexplained by the fit
};

/// Complex response of each channel of a conditioned record to a source at a known world
/// position: the channel's compressed samples at the nominal two-way delay (band-limited
/// interpolation, exact), with the carrier restored.
std::vector<std::complex<double>> channel_responses(const PingRecord& record, const Vec3d& source_position);

/// Gain, phase and element position errors from conditioned calibration pings, one known
/// source position per ping. `planar` restricts position errors to the plane of the array
/// axis and the boresight (two unknowns per channel), which is what sources in that plane
/// can observe.
ChannelErrors estimate_channel_errors(const std::vector<const PingRecord*>& records, const std::vector<Vec3d>& source_positions,
                                      bool planar = true);

/// Write the errors into a record's calibration tables. dsp::apply_calibration then removes
/// gain and phase; the beamformers use the position offsets directly.
void store_calibration(PingRecord& record, const ChannelErrors& errors);

/// store_calibration, then remove gain and phase from the (complex baseband) samples.
void apply_channel_errors(PingRecord& record, const ChannelErrors& errors);

}  // namespace sonar::live
