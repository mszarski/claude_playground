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
                                 double& r_back, const Vec3d& accel = Vec3d{}) {
  const Vec3d tx_pos = tx.position + tx.velocity * u + accel * (0.5 * u * u);
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
    r_back = norm(p_pos - (rx.position + rx.velocity * t + accel * (0.5 * t * t)));
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
    const double t0 = arrival_time(params.tx, e, path, c, 0.0, ro0, rb0, params.platform_acceleration);
    const double tm = arrival_time(params.tx, e, path, c, 0.5 * T, rom, rbm, params.platform_acceleration);
    const double t1 = arrival_time(params.tx, e, path, c, T, ro1, rb1, params.platform_acceleration);

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

    // The frequency-dependent part of the amplitude (both beam patterns and absorption)
    // is smooth in frequency, and the pulse's frequency is constant (CW, Gaussian) or
    // linear in emission time (LFM). It is evaluated at kKnots + 1 emission times and
    // interpolated between them, instead of at every sample: two pattern evaluations and
    // a power per sample were most of the cost of a long pulse.
    constexpr int kKnots = 16;
    double knot[kKnots + 1];
    const bool swept = w.type == 1 && w.f_stop != w.f_start;
    const int n_knots = swept ? kKnots : 1;
    for (int i = 0; i <= n_knots; ++i) {
      const double freq = swept ? w.f_start + (w.f_stop - w.f_start) * i / n_knots
                                : (w.type == 0 ? w.f_start : 0.5 * (w.f_start + w.f_stop));
      const double ratio = f_ref > 0.0 ? freq / f_ref : 1.0;
      knot[i] = gain *
                directivity(params.tx.axes, params.tx.beamwidth_az, params.tx.beamwidth_el, out_dir, ratio) *
                directivity(e.axes, e.beamwidth_az, e.beamwidth_el, back_dir, ratio) *
                std::pow(10.0, -absorption_db_per_m(params, freq) * path_len / 20.0);
    }
    const double knots_per_second = n_knots / T;

    // Emission time as a function of arrival time, u(t) = (t - t0) (ua (t - t1) + ub (t - tm)).
    const double ua = 0.5 * T / d_m, ub = T / d_1;
    const double inv_fs = 1.0 / fs;
    const bool baseband = params.complex_baseband;
    // Phase (cycles) of sample n: the pulse's phase at its emission time, less the mixer's
    // phase for a baseband receiver (which mixes on its nominal clock).
    auto cycles = [&](std::int64_t n) {
      const double t = start + static_cast<double>(n) * inv_fs;
      double cyc = waveform_phase(w, (t - t0) * (ua * (t - t1) + ub * (t - tm)));
      if (baseband) cyc -= params.center_frequency * (params.start_time + static_cast<double>(n) / params.sample_rate);
      return cyc - std::floor(cyc);
    };

    // The phasor is advanced by a recurrence and re-seeded exactly every kBlock samples:
    // z[n+1] = z[n] r[n], r[n+1] = r[n] q. That is exact for a phase quadratic in n; the
    // cubic term left over (range curvature times sweep rate) is below 1e-9 cycles a block.
    constexpr std::int64_t kBlock = 16;
    const auto first = static_cast<std::int64_t>(n_first), last = static_cast<std::int64_t>(n_last);
    for (std::int64_t n0 = first; n0 <= last; n0 += kBlock) {
      const double c0 = cycles(n0), c1 = cycles(n0 + 1), c2 = cycles(n0 + 2);
      double z_re = std::cos(pi2 * c0), z_im = std::sin(pi2 * c0);
      double r_re = std::cos(pi2 * (c1 - c0)), r_im = std::sin(pi2 * (c1 - c0));
      const double q_re = std::cos(pi2 * (c2 - 2.0 * c1 + c0)), q_im = std::sin(pi2 * (c2 - 2.0 * c1 + c0));
      const std::int64_t n_end = n0 + kBlock - 1 < last ? n0 + kBlock - 1 : last;
      for (std::int64_t n = n0; n <= n_end; ++n) {
        const double t = start + static_cast<double>(n) * inv_fs;
        const double u = (t - t0) * (ua * (t - t1) + ub * (t - tm));
        const double env = waveform_envelope(w, u);
        if (env != 0.0) {
          double x = u * knots_per_second;
          int ik = static_cast<int>(x);
          if (ik > n_knots - 1) ik = n_knots - 1;
          x -= ik;
          const double amp = env * ((1.0 - x) * knot[ik] + x * knot[ik + 1]);
          if (amp != 0.0) {
            if (baseband) {
              accumulate(dst + 2 * n, static_cast<float>(amp * (a_re * z_re - a_im * z_im)));
              accumulate(dst + 2 * n + 1, static_cast<float>(amp * (a_re * z_im + a_im * z_re)));
            } else {
              accumulate(dst + n, static_cast<float>(amp * (a_re * z_re - a_im * z_im)));
            }
          }
        }
        const double nz_re = z_re * r_re - z_im * r_im;
        z_im = z_re * r_im + z_im * r_re;
        z_re = nz_re;
        const double nr_re = r_re * q_re - r_im * q_im;
        r_im = r_re * q_im + r_im * q_re;
        r_re = nr_re;
      }
    }
  }
};

}  // namespace sonar::sim
