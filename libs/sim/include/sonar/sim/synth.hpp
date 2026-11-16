// Echo synthesis: the simulator's forward model (ASASIN Eq. 1 generalised to
// bistatic, moving, multipath geometry). See DECISIONS.md A-05.
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

struct SynthParams {
  WaveformParams waveform;
  double sound_speed = 1500.0;
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
};

/// Add the echoes of `paths` into `out` ([channels, n_samples], interleaved complex
/// when complex_baseband). `out` is accumulated into, not cleared.
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
