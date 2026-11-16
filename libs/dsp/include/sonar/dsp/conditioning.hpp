// Conditioning and the ASASIN pre/post-processing steps (Sec. II-E, Eq. 9-12).
// Each function is a stage: plain inputs, no file I/O.
#pragma once

#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::dsp {

/// Remove each channel's mean (real passband data; baseband conversion also removes DC).
void remove_dc(PingRecord& record);

/// Apply the record's per-channel calibration to complex baseband samples: divide by
/// gain * exp(j phase) and advance by delay (a phase ramp in frequency, so sub-sample
/// delays are exact). The gain, phase and delay tables are then reset to neutral so the
/// correction cannot be applied twice. Position offsets are geometry and stay.
void apply_calibration(PingRecord& record);

struct TvgOptions {
  double spreading_db_per_decade = 40.0;  ///< gain = this * log10(R / reference_range); 40 = two-way spherical
  double absorption_db_per_m = 0.0;       ///< two-way absorption made up: 2 * alpha * R
  double reference_range = 1.0;           ///< m; gain is 0 dB here
  double max_gain_db = 120.0;             ///< clamp, so late noise is not amplified without bound
  bool undo_hardware_ramp = true;         ///< divide out the analogue ramp recorded in the metadata first
};

/// Deterministic digital time-varying gain. With undo_hardware_ramp the recorded analogue
/// ramp is removed first, so hardware and digital gain together follow the requested law;
/// the metadata ramp is then cleared.
void apply_tvg(PingRecord& record, const TvgOptions& options = {});

struct WhitenOptions {
  double gamma = 0.1;             ///< Eq. 9 calibration factor
  double smooth_hz = 0.0;         ///< boxcar smoothing of the power estimate; 0 = bandwidth / 32
  bool power_gain = true;         ///< true: multiply amplitude by sqrt(G) (DECISIONS.md D-20)
};

/// Spectral whitening, Eq. 9: G(f) = h(1 / (gamma * mean(P) + P(f))), with P(f) the power
/// spectrum averaged over every channel of every record given, and h scaling so max G = 1.
/// Operates in place on complex baseband records that share a sample rate and length.
/// Returns G(f) in FFT bin order, for the debug products.
std::vector<float> whiten(const std::vector<PingRecord*>& records, const WhitenOptions& options = {});

struct MedianTvgOptions {
  double alpha = 0.01;   ///< Eq. 10 calibration factor
  bool power_gain = true;
  std::int64_t smooth_samples = 0;  ///< boxcar smoothing of the median power; 0 = none
};

/// Time-varying gain from the data, Eq. 10: G(t) = 1 / (alpha * mean_t(Pm) + Pm(t)), with
/// Pm(t) the median over all channels and records of the power at sample t.
/// Returns G(t).
std::vector<float> median_tvg(const std::vector<PingRecord*>& records, const MedianTvgOptions& options = {});

struct DrcOptions {
  double target_median = 0.2;   ///< b in Eq. 12
  double percentile = 99.99;    ///< normalise by this percentile of the magnitude; 100 = max
};

/// Dynamic range compression, Eq. 11 with the closed-form q of Eq. 12. `magnitude` is
/// replaced by values in [0, 1] whose median is `target_median`. Returns q.
double drc(std::vector<float>& magnitude, const DrcOptions& options = {});

}  // namespace sonar::dsp
