// Pulse compression: matched filtering against the transmitted waveform.
#pragma once

#include "sonar/core/record.hpp"

namespace sonar::dsp {

enum class ReplicaWindow : std::int32_t { None = 0, Hann = 1, Hamming = 2 };

struct PulseCompressionOptions {
  /// Amplitude taper applied to the replica over the pulse, for range sidelobe control
  /// (a mismatched filter: lower sidelobes, wider main lobe, slightly lower gain).
  ReplicaWindow window = ReplicaWindow::None;
  /// Time-scale factor of the echo relative to the transmitted pulse: an echo from a
  /// path closing at range rate v (two-way) has doppler_scale = 1 + 2 v / c. The
  /// replica is compressed accordingly. 1 = no Doppler.
  double doppler_scale = 1.0;
};

/// Matched-filter record.samples in place. Input must be complex baseband.
///
/// The output sample at time t is the correlation of the data with a replica whose
/// start is placed at t, so an echo of delay tau peaks at t = tau, with the phase a
/// baseband impulse at tau would have. The gain is normalised so the peak equals the
/// echo amplitude: a compressed image keeps the units of the input.
void pulse_compress(PingRecord& record, const PulseCompressionOptions& options = {});

/// The baseband replica used above, sampled at `sample_rate` (complex, interleaved).
std::vector<std::complex<float>> make_replica(const Waveform& waveform, double sample_rate,
                                              double center_frequency,
                                              const PulseCompressionOptions& options = {});

}  // namespace sonar::dsp
