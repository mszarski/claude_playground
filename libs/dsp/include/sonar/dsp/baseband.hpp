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
};

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
