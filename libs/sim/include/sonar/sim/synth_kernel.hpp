// The synthesis kernel body: one (path, channel) pair. Compiled for host and device.
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/sim/synth.hpp"

namespace sonar::sim {

/// When a pulse emitted at time u reaches a receiver via path (q, p).
/// Also returns the two leg lengths.
SONAR_INLINE double arrival_time(const SynthElement& tx, const SynthElement& rx,
                                 const SynthPath& path, double c, double u, double& r_out,
                                 double& r_back) {
  const Vec3d tx_pos = tx.position + tx.velocity * u;
  // Time the wavefront meets the (slowly moving) outbound point.
  double ts = u;
  for (int i = 0; i < 3; ++i) {
    r_out = norm(path.q + path.q_velocity * ts - tx_pos);
    ts = u + r_out / c;
  }
  // Time the scattered sound meets the moving receiver.
  const Vec3d p_pos = path.p + path.p_velocity * ts;
  double t = ts;
  for (int i = 0; i < 4; ++i) {
    r_back = norm(p_pos - (rx.position + rx.velocity * t));
    t = ts + r_back / c;
  }
  return t;
}

SONAR_INLINE double absorption_db_per_m(const SynthParams& p, double f) {
  const double span = p.absorption_f_max - p.absorption_f_min;
  if (!(span > 0.0)) return p.absorption_db_per_m[0];
  double x = (f - p.absorption_f_min) / span * (kAbsorptionTableSize - 1);
  if (x < 0.0) x = 0.0;
  if (x > kAbsorptionTableSize - 1) x = kAbsorptionTableSize - 1;
  int i = static_cast<int>(x);
  if (i > kAbsorptionTableSize - 2) i = kAbsorptionTableSize - 2;
  const double w = x - i;
  return (1.0 - w) * p.absorption_db_per_m[i] + w * p.absorption_db_per_m[i + 1];
}

/// Adds `value` to `*dst`. Host: plain add (the caller guarantees one thread per
/// channel). Device: atomic add.
struct HostAccumulate {
  SONAR_HD void operator()(float* dst, float value) const { *dst += value; }
};

template <typename Accumulate>
struct SynthKernel {
  SynthParams params;
  const SynthElement* rx;
  const SynthPath* paths;
  float* out;
  std::int64_t n_channels;
  std::int64_t n_paths;
  Accumulate accumulate;

  /// Add path `ip` to channel `ic`.
  SONAR_HD void pair(std::int64_t ic, std::int64_t ip) const {
    const double pi2 = 6.283185307179586476925;
    const SynthElement& e = rx[ic];
    const SynthPath& path = paths[ip];
    const WaveformParams& w = params.waveform;
    const double c = params.sound_speed;
    const double T = w.duration;

    // Exact arrival times of the start, middle and end of the pulse. Emission time
    // as a function of arrival time is then a quadratic through these three points,
    // which captures Doppler and range curvature over the pulse.
    double ro0, rb0, rom, rbm, ro1, rb1;
    const double t0 = arrival_time(params.tx, e, path, c, 0.0, ro0, rb0);
    const double tm = arrival_time(params.tx, e, path, c, 0.5 * T, rom, rbm);
    const double t1 = arrival_time(params.tx, e, path, c, T, ro1, rb1);

    // Channel sample clock.
    const double fs = params.sample_rate * (1.0 + e.clock_drift);
    const double start = params.start_time + e.clock_offset;
    double n_first = std::ceil((t0 - start) * fs);
    double n_last = std::floor((t1 - start) * fs);
    if (n_first < 0.0) n_first = 0.0;
    if (n_last > static_cast<double>(params.n_samples - 1)) {
      n_last = static_cast<double>(params.n_samples - 1);
    }
    if (n_last < n_first) return;

    // Geometry-dependent amplitude at mid-pulse.
    const double ts_mid = 0.5 * T + rom / c;
    const Vec3d tx_pos = params.tx.position + params.tx.velocity * (0.5 * T);
    const Vec3d out_dir = (path.q + path.q_velocity * ts_mid - tx_pos) / rom;
    const Vec3d back_dir =
        (path.p + path.p_velocity * ts_mid - (e.position + e.velocity * tm)) / rbm;
    double gain = w.amplitude;
    if (params.spreading) gain /= (rom * rbm);
    const double path_len = rom + rbm;
    const double f_ref = params.center_frequency;

    const double a_re = static_cast<double>(path.amp_re);
    const double a_im = static_cast<double>(path.amp_im);
    const std::int64_t stride = params.complex_baseband ? 2 : 1;
    float* dst = out + ic * params.n_samples * stride;

    // Lagrange weights for u(t) through (t0, 0), (tm, T/2), (t1, T).
    const double d_m = (tm - t0) * (tm - t1);
    const double d_1 = (t1 - t0) * (t1 - tm);

    for (auto n = static_cast<std::int64_t>(n_first); n <= static_cast<std::int64_t>(n_last); ++n) {
      const double t = start + static_cast<double>(n) / fs;
      const double u =
          0.5 * T * (t - t0) * (t - t1) / d_m + T * (t - t0) * (t - tm) / d_1;
      double env, phase, freq;
      if (!waveform_eval(w, u, env, phase, freq)) continue;
      const double ratio = f_ref > 0.0 ? freq / f_ref : 1.0;
      double amp = gain * env;
      amp *= directivity(params.tx.axes, params.tx.beamwidth_az, params.tx.beamwidth_el, out_dir, ratio);
      amp *= directivity(e.axes, e.beamwidth_az, e.beamwidth_el, back_dir, ratio);
      amp *= std::pow(10.0, -absorption_db_per_m(params, freq) * path_len / 20.0);
      if (amp == 0.0) continue;
      if (params.complex_baseband) {
        // The receiver mixes with its nominal clock, so use nominal sample time.
        const double t_nominal = params.start_time + static_cast<double>(n) / params.sample_rate;
        double cyc = phase - params.center_frequency * t_nominal;
        cyc -= std::floor(cyc);
        const double cs = std::cos(pi2 * cyc), sn = std::sin(pi2 * cyc);
        accumulate(dst + 2 * n, static_cast<float>(amp * (a_re * cs - a_im * sn)));
        accumulate(dst + 2 * n + 1, static_cast<float>(amp * (a_re * sn + a_im * cs)));
      } else {
        double cyc = phase - std::floor(phase);
        const double cs = std::cos(pi2 * cyc), sn = std::sin(pi2 * cyc);
        accumulate(dst + n, static_cast<float>(amp * (a_re * cs - a_im * sn)));
      }
    }
  }
};

}  // namespace sonar::sim
