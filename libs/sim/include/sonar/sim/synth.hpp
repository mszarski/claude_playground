// Echo synthesis: the simulator's forward model (ASASIN Eq. 1 generalised to
// bistatic, moving, multipath geometry). See DECISIONS.md A-05.
//
// Units and conventions: SI, world frame north-east-down. `WaveformParams::amplitude` is
// the PEAK amplitude at 1 m in the record's sample units (the convention of
// record.hpp; this file never sees a level in dB, so the peak-or-rms reading of a
// source level is settled before it). Absorption is dB/m, one way, applied over the whole
// path length. Signal sign exp(+i w t).
//
// Fixed capacities, and what happens at each (docs/HARDCODED_AUDIT.md S-5 to S-7):
//  - kMaxEnvelopeBands knot frequencies for a sampled pulse: more is an error.
//  - kAbsorptionTableSize points of absorption over [absorption_f_min, absorption_f_max]:
//    a frequency outside is read at the nearest end. `synthesize` throws when that clamp
//    would change a level by more than `absorption_clamp_limit_db` over the longest path
//    in the record's window (absorption_clamp_error_db below is the measure).
//  - 16 amplitude knots along a swept pulse (synth_kernel.hpp, kKnots): beam patterns and
//    loss are interpolated linearly in frequency between them. Not checked: a pattern null
//    that crosses a wide band is under-sampled (small; stated in CONFIGURATION.md).
//  - arrival_passes_out / arrival_passes_back passes of the arrival-time solve: what is
//    left is bounded by arrival_residual below; `synthesize` throws when the bound exceeds
//    `arrival_residual_limit`, and `arrival_passes_for` gives the count a case needs.
#pragma once

#include <cstdint>
#include <vector>

#include "sonar/core/directivity.hpp"
#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/core/vec.hpp"
#include "sonar/core/waveform.hpp"

namespace sonar::sim {

/// A transducer in the world frame with first-order motion.
struct SynthElement {
  Vec3d position;  ///< at the transmit instant, m
  Vec3d velocity;  ///< m/s
  ElementAxes<double> axes;  ///< world frame
  double beamwidth_az = kNaN;
  double beamwidth_el = kNaN;
  /// Sample clock of this channel: true time of sample n is
  /// start_time + clock_offset + n / (sample_rate * (1 + clock_drift)).
  double clock_offset = 0.0;
  double clock_drift = 0.0;  ///< fractional frequency error (ppm * 1e-6)
};

/// One two-way propagation path. Outbound sound travels in a straight line from the
/// transmitter to `q`; returning sound travels in a straight line from `p` to each
/// receiver. For a direct echo q == p == the scatterer. For a boundary bounce they are
/// the scatterer's images. For a traced eigenray they are the apparent positions
/// (travel time times sound speed along the launch or arrival direction).
struct SynthPath {
  Vec3d q, q_velocity;
  Vec3d p, p_velocity;
  float amp_re = 1.0F;  ///< complex amplitude on top of spreading, absorption and
  float amp_im = 0.0F;  ///< directivity: target strength, reflection coefficients
};

inline constexpr int kAbsorptionTableSize = 16;

/// Most knot frequencies of the frequency-dependent gain of a sampled pulse.
inline constexpr int kMaxEnvelopeBands = 33;

/// The complex envelope b(u) of a sampled (arbitrary) pulse, on a grid fine enough to be
/// read with a four-point cubic (python/gpusonar/sim/waveform.py builds it: windowed-sinc
/// interpolation of the envelope's samples onto a grid `table_factor` times finer).
/// The pulse is b(u) exp(2 pi i carrier u) for emission times 0 <= u <= duration.
///
/// For loss that depends on frequency the envelope is split into `n_bands` parts by
/// triangular weights in frequency about `knot_frequency`; a gain g_i on part i is a gain
/// piecewise linear in frequency through (knot_frequency[i], g_i). `data` holds, for each
/// fine sample, the n_bands parts and then their sum, as (re, im) pairs:
/// data[2 * (m * (n_bands + 1) + i)] is the real part of part i at fine sample m.
/// Plain data: `data` is a host pointer for the host backend and is replaced by a device
/// copy inside the CUDA backend.
struct EnvelopeTable {
  const double* data = nullptr;
  std::int64_t n_fine = 0;  ///< fine samples
  int n_bands = 0;          ///< 1 .. kMaxEnvelopeBands
  double rate = 0.0;        ///< fine samples per second of emission time
  double offset = 0.0;      ///< fine index of emission time 0
  double carrier = 0.0;     ///< Hz
  double knot_frequency[kMaxEnvelopeBands] = {};
};

struct SynthParams {
  WaveformParams waveform;
  double sound_speed = kNaN;  ///< m/s; no default: `synthesize` rejects a value that is not stated
  double sample_rate = 0.0;  ///< nominal, Hz
  double start_time = 0.0;   ///< nominal time of sample 0 after transmit, s
  std::int64_t n_samples = 0;
  bool complex_baseband = false;  ///< false: real passband output
  double center_frequency = 0.0;  ///< mixing frequency for baseband output; directivity reference
  bool spreading = true;          ///< apply 1/(R_out * R_back)
  /// Acceleration of the transmitter and every receiver (world, m/s^2): the part of the
  /// platform's motion within a ping that the record's first-order pose cannot express.
  Vec3d platform_acceleration;
  /// Absorption in dB/m, tabulated uniformly over [f_min, f_max].
  double absorption_f_min = 0.0;
  double absorption_f_max = 0.0;
  double absorption_db_per_m[kAbsorptionTableSize] = {};
  SynthElement tx;
  /// Passes of the fixed-point solves for the time sound meets the outbound point and the
  /// receiver. Each pass gains a factor (speed / c); what is left after n passes is about
  /// (R / c) (v / c)^n. Analytic pulses ran (3, 4) until the outbound residual was measured
  /// (0.007 rad at 8 kHz for 12 m/s closing at 400 m); (4, 4) leaves 5.5e-5.
  int arrival_passes_out = 4;
  int arrival_passes_back = 4;
  /// `synthesize` throws when the bound on the phase left by the passes above
  /// (arrival_residual) exceeds this, in radians. 0 or less: no check.
  double arrival_residual_limit = 0.05;
  /// `synthesize` throws when reading the absorption table outside its band would change a
  /// level by more than this (dB) over the longest path in the window. 0 or less: no check.
  double absorption_clamp_limit_db = 0.1;
  /// Waveform type 3 (Arbitrary) only: the pulse's envelope.
  EnvelopeTable envelope;
};

/// Pass count used for sampled pulses (the NumPy reference engine runs the same number).
inline constexpr int kSampledArrivalPasses = 8;

/// Upper bound on the phase error (rad) the arrival-time solves leave: {outbound, return}.
/// Each pass of a solve gains a factor (speed / c) and the first guess is off by at most
/// the leg's travel time, which is at most the end of the record's window, so what is left
/// after n passes is at most  T_end (v / c)^n 2 pi f_top,  with T_end the time of the last
/// sample, v the largest speed that enters the solve (the outbound points' for the first
/// leg; the receivers', with the platform acceleration over T_end, for the second) and
/// f_top the highest frequency of the pulse.
struct ArrivalResidual {
  double out = 0.0;
  double back = 0.0;
};
ArrivalResidual arrival_residual(const SynthParams& params, const std::vector<SynthElement>& rx,
                                 const std::vector<SynthPath>& paths);

/// The same bound from its four numbers, for one leg and `passes` passes.
double arrival_residual_bound(double window_end, double speed, double sound_speed, double frequency, int passes);

/// Fewest passes (1 to 64) that bring the bound above under `tolerance` radians.
int arrival_passes_for(double window_end, double speed, double sound_speed, double frequency, double tolerance);

/// Set both pass counts of `params` to the fewest that leave under `tolerance` radians for
/// these receivers and paths.
void set_arrival_passes(SynthParams& params, const std::vector<SynthElement>& rx,
                        const std::vector<SynthPath>& paths, double tolerance);

/// Largest change of level (dB) that reading the absorption table at its nearest end, for
/// the frequencies of this pulse that lie outside [absorption_f_min, absorption_f_max],
/// causes over the longest path in the window, against the table continued linearly. 0 when
/// every frequency the kernel reads is inside the table, or the table is flat.
double absorption_clamp_error_db(const SynthParams& params);

/// Add the echoes of `paths` into `out` ([channels, n_samples], interleaved complex
/// when complex_baseband). `out` is accumulated into, not cleared. A sampled pulse
/// (waveform type 3) needs `params.envelope`, whose data must outlive the call.
void synthesize(Backend backend, const SynthParams& params, const std::vector<SynthElement>& rx,
                const std::vector<SynthPath>& paths, float* out);

/// Geometry and sampling for `synthesize`, taken from a record whose samples block has
/// its shape, kind, rate and start time set. Board clock errors come from the metadata.
SynthParams params_from_record(const PingRecord& record);
std::vector<SynthElement> elements_from_record(const PingRecord& record);

/// Synthesize straight into record.samples (accumulating).
void synthesize_into(Backend backend, PingRecord& record, const std::vector<SynthPath>& paths,
                     const SynthParams* params_override = nullptr);

}  // namespace sonar::sim
