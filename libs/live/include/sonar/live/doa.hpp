// Bearings by sparse Bayesian learning (SBL) on a line array, for active pings and for
// passive listening.
//
// A frame is a set of array snapshots at several frequencies. For an active ping it is one
// range gate of the conditioned record: a short stretch of samples around a two-way time,
// transformed to frequency, one snapshot per bin of the transmit band. For passive
// listening it is a stretch of the record cut into overlapping blocks, each transformed,
// the blocks giving several snapshots per bin of the band asked for.
//
// SBL models the snapshots at every frequency as a sum of plane waves from a grid of
// bearings with unknown powers gamma (shared across frequency) plus noise, and finds the
// gammas that make the data most likely. Most come out zero: the result is a short list
// of bearings with powers. Against a conventional beamformer it separates sources closer
// than a beamwidth and has no sidelobes to mistake for sources; against MVDR and MUSIC it
// works with one snapshot and with coherent arrivals (multipath), because it does not
// invert a sample covariance.
//
// Method: P. Gerstoft, C. F. Mecklenbrauker, A. Xenaki, S. Nannuru, "Multisnapshot sparse
// Bayesian learning for DOA", IEEE Signal Process. Lett. 23 (2016); multi-frequency form
// with a common gamma as in S. Nannuru et al., "Sparse Bayesian learning with
// multiple dictionaries", Signal Processing 159 (2019). Implemented from the papers.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

/// Snapshots of an N-element array at F frequencies, L snapshots each.
struct DoaFrame {
  std::int64_t n_channels = 0;
  std::int64_t n_snapshots = 0;
  std::vector<double> frequencies;                ///< Hz, absolute (carrier included)
  std::vector<std::complex<double>> snapshots;    ///< [F][L][N]
  double time = 0.0;                              ///< s: two-way time of the gate, or the frame's centre in the record
};

/// One range gate of a conditioned (complex baseband, pulse-compressed) record: `n_fft`
/// samples centred on `two_way_time`, Hann weighted, one snapshot per bin of the transmit band.
DoaFrame gate_frame(const PingRecord& record, double two_way_time, std::int64_t n_fft);

/// Passive frames from a record of any kind: blocks of `n_fft` samples stepped by `hop`,
/// `n_average` consecutive blocks per frame, bins from f_low to f_high (Hz, absolute).
std::vector<DoaFrame> passive_frames(const PingRecord& record, std::int64_t n_fft, std::int64_t hop,
                                     std::int64_t n_average, double f_low, double f_high);

struct DoaOptions {
  std::int32_t max_iterations = 60;
  double tolerance = 1e-3;         ///< stop when gamma changes by less than this (relative, L1)
  std::int32_t max_sources = 4;    ///< peaks used for the noise estimate and reported
  double dynamic_range_db = 20.0;  ///< peaks this far below the strongest are not reported
  double min_snr_db = 0.0;         ///< nor peaks whose array output is this little above the noise
  std::int32_t max_frequencies = 24;  ///< bins used, spread evenly over the frame's band; 0 = all of them
  /// Where the iteration runs: the kernels of sbl_kernel.hpp on CPU threads or on a CUDA device.
  /// `kernels = false` (host only) runs the Eigen reference instead: a full solve per frequency,
  /// kept to check the kernels against; it is about half as fast.
  Backend backend = Backend::Host;
  int device = 0;
  bool kernels = true;
};

struct DoaPeak {
  double bearing = 0.0;   ///< rad
  double power = 0.0;     ///< gamma: source power at an element, in the data's units squared
  double snr_db = 0.0;    ///< n_channels * gamma over the noise power
};

struct DoaResult {
  std::vector<float> spectrum;  ///< gamma (SBL) or beam power over the grid
  std::vector<DoaPeak> peaks;   ///< strongest first
  double noise = 0.0;           ///< mean noise power per element and snapshot
  std::int32_t iterations = 0;
  /// The kernel path was asked for with more channels than it takes (kSblMaxChannels): this result is
  /// from the host reference (Eigen), on the host even when the backend asked for was CUDA.
  bool fallback = false;
};

/// SBL over a grid of bearings (rad, from broadside towards increasing element offset).
/// `element_offsets` are the elements' positions along the array axis, m.
DoaResult sbl(const DoaFrame& frame, const std::vector<double>& element_offsets, double sound_speed,
              const std::vector<double>& bearings, const DoaOptions& options = {});

/// Conventional beam power on the same grid from the same frame, for comparison.
DoaResult conventional(const DoaFrame& frame, const std::vector<double>& element_offsets, double sound_speed,
                       const std::vector<double>& bearings, const DoaOptions& options = {});

/// Positions of the receive elements along the array's own axis (first to last), m about the centre.
std::vector<double> element_offsets(const PingRecord& record);

}  // namespace sonar::live
