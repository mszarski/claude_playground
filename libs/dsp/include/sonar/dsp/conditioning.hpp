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

/// The gain law is  G(R) = spreading_db_per_decade log10(R / reference_range) + 2 alpha R  dB,
/// with R = c t / 2 the two-way range of a sample at time t after transmit (the record's
/// sound speed), clamped at max_gain_db.
///
/// The defaults describe one case and are wrong for others (docs/HARDCODED_AUDIT.md K-4):
///  - spreading 40 dB per decade is an echo's two-way spherical loss. A one-way (passive or
///    transponder) arrival needs 20; a display of a distributed seabed 30.
///  - absorption 0 dB/m is right at a few kHz and 50 dB short at 300 m and 330 kHz
///    (0.083 dB/m at 12 deg C). The value is alpha in dB/m ONE WAY; the law doubles it.
///    It depends on the water, which this library does not model: the caller derives it
///    (python: gpusonar.recon.tvg_parameters, pipeline key `absorption: record`).
///  - the clamp silently flattens the law beyond the range where it is reached; the report
///    returned by apply_tvg says whether and where that happened.
struct TvgOptions {
  double spreading_db_per_decade = 40.0;  ///< gain = this * log10(R / reference_range); 40 = two-way spherical
  double absorption_db_per_m = 0.0;       ///< alpha, dB/m one way, >= 0; made up both ways: 2 * alpha * R
  double reference_range = 1.0;           ///< m; gain is 0 dB here
  double max_gain_db = 120.0;             ///< clamp, so late noise is not amplified without bound
  bool undo_hardware_ramp = true;         ///< divide out the analogue ramp recorded in the metadata first
};

/// What apply_tvg did, so that a clamped law is reported instead of silent.
struct TvgReport {
  double gain_first_db = 0.0;       ///< law at the first sample, dB (before the hardware ramp is removed)
  double gain_last_db = 0.0;        ///< law at the last sample, dB
  std::int64_t clamped_samples = 0; ///< samples whose law was cut at max_gain_db
  double clamp_range = kNaN;        ///< m; range of the first clamped sample, NaN if none
  double shortfall_db = 0.0;        ///< how far below the unclamped law the last sample is, dB
};

/// Deterministic digital time-varying gain. With undo_hardware_ramp the recorded analogue
/// ramp is removed first, so hardware and digital gain together follow the requested law;
/// the metadata ramp is then cleared. Throws std::invalid_argument for a record without a
/// sound speed and for a negative or non-finite absorption or spreading.
TvgReport apply_tvg(PingRecord& record, const TvgOptions& options = {});

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
