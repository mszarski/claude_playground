// Down-conversion to complex baseband with decimation.
#pragma once

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"

namespace sonar::dsp {

struct BasebandOptions {
  double center_frequency = 0.0;  ///< Hz; 0 = centre of the transmit band
  double bandwidth = 0.0;         ///< Hz kept around the centre; 0 = transmit bandwidth * band_margin
  double band_margin = 1.5;       ///< used when bandwidth == 0
  double oversample = 2.0;        ///< output rate / kept bandwidth
  double edge_taper = 0.1;        ///< fraction of the kept band tapered at each edge (raised cosine)
  /// A CW pulse has no swept bandwidth: the band kept is cw_band_factor / duration (4: the
  /// main lobe of the gated tone's spectrum and its first sidelobes).
  double cw_band_factor = 4.0;
  /// Doppler allowance (docs/HARDCODED_AUDIT.md K-6). The band derived from the waveform
  /// assumes echoes at the transmitted frequency. `max_range_rate` is the largest two-way
  /// range rate to be kept, m/s (closing or opening): the band is widened on both sides by
  /// the shift 2 v f_top / c it causes, with c the record's sound speed and f_top the top
  /// of the transmit band. `doppler_allowance_hz` widens it on both sides by that many Hz
  /// directly (a one-way arrival is shifted by v f / c). Both add; 0 = no allowance, which
  /// is right for a stationary scene and silently filters out a mover on a long tone.
  /// Neither applies when `bandwidth` is given: that band is taken as stated.
  double max_range_rate = 0.0;
  double doppler_allowance_hz = 0.0;
};

/// The band (Hz) `to_baseband` keeps for a record with these options, before it is rounded
/// to FFT bins. Throws when the allowance needs a sound speed the record does not have.
double baseband_band(const PingRecord& record, const BasebandOptions& options);

/// Convert record.samples in place to complex baseband at a reduced rate.
///
/// Method: FFT, keep the band around the carrier, shift it to DC, inverse FFT at the
/// shorter length. This is exact band-limited decimation for the block. The carrier is
/// snapped to an FFT bin; the exact value is written to samples.center_frequency.
/// Baseband phase is referenced to the transmit instant (t = 0), the convention the
/// backprojection kernel assumes. Accepts real passband, analytic or baseband input.
/// On the CUDA backend the transforms run on the device (cuFFT); the result differs from
/// the host's only by float32 rounding.
void to_baseband(PingRecord& record, const BasebandOptions& options = {},
                 Backend backend = Backend::Host, int device = 0);

}  // namespace sonar::dsp
