// Analytic transmit waveforms, evaluated in closed form on host or device.
// Used by the simulator (exact synthesis, DECISIONS.md A-05) and to build
// matched-filter replicas.
#pragma once

#include <cmath>

#include "sonar/core/platform.hpp"

namespace sonar::sim {

/// Plain-data waveform description (a device-friendly copy of sonar::Waveform).
struct WaveformParams {
  int type = 0;           ///< WaveformType as int: 0 CW, 1 LFM, 2 Gaussian
  double f_start = 0.0;   ///< Hz
  double f_stop = 0.0;    ///< Hz
  double duration = 0.0;  ///< s; the pulse occupies emission times [0, duration]
  double window_alpha = 0.0;  ///< Tukey taper fraction (CW, LFM)
  double amplitude = 1.0;     ///< peak amplitude at 1 m
};

/// Tukey (tapered cosine) window on x in [0, 1].
SONAR_INLINE double tukey(double x, double alpha) {
  if (x < 0.0 || x > 1.0) return 0.0;
  if (alpha <= 0.0) return 1.0;
  const double pi = 3.14159265358979323846;
  const double edge = 0.5 * alpha;
  if (x < edge) return 0.5 * (1.0 - std::cos(pi * x / edge));
  if (x > 1.0 - edge) return 0.5 * (1.0 - std::cos(pi * (1.0 - x) / edge));
  return 1.0;
}

/// Standard deviation in time of the Gaussian pulse whose amplitude spectrum is
/// -6 dB at +-bandwidth/2.
SONAR_INLINE double gaussian_sigma(double bandwidth) {
  const double pi = 3.14159265358979323846;
  return std::sqrt(2.0 * 0.6931471805599453) / (pi * bandwidth);
}

/// Evaluate the pulse at emission time u (s). Returns false outside the pulse.
/// The analytic signal is amplitude * envelope * exp(j 2 pi phase_cycles).
/// `freq` is the instantaneous frequency (Hz), for frequency-dependent losses.
SONAR_INLINE bool waveform_eval(const WaveformParams& w, double u, double& envelope,
                                double& phase_cycles, double& freq) {
  if (u < 0.0 || u > w.duration) return false;
  switch (w.type) {
    case 0:  // CW
      envelope = tukey(u / w.duration, w.window_alpha);
      phase_cycles = w.f_start * u;
      freq = w.f_start;
      return true;
    case 1: {  // LFM
      const double k = (w.f_stop - w.f_start) / w.duration;
      envelope = tukey(u / w.duration, w.window_alpha);
      phase_cycles = (w.f_start + 0.5 * k * u) * u;
      freq = w.f_start + k * u;
      return true;
    }
    case 2: {  // Gaussian-enveloped tone, centred in the pulse window
      const double fc = 0.5 * (w.f_start + w.f_stop);
      const double sigma = gaussian_sigma(w.f_stop - w.f_start);
      const double d = u - 0.5 * w.duration;
      envelope = std::exp(-0.5 * (d / sigma) * (d / sigma));
      phase_cycles = fc * d;
      freq = fc;
      return true;
    }
    default:
      return false;
  }
}

}  // namespace sonar::sim
