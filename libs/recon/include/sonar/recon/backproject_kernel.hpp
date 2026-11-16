// Backprojection kernel body: one output point. Compiled for host and device.
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/core/directivity.hpp"
#include "sonar/recon/backproject.hpp"

namespace sonar::recon {

/// Reference delay of one group at one point, solved in double precision.
struct RefDelay {
  double tau;       ///< two-way delay via the reference point, s
  double r_tx;      ///< transmit leg, m
  double r_rx;      ///< receive leg to the reference point at arrival, m
  Vec3f v;          ///< velocity of the reference point at arrival, body frame, m/s
  Vec3f w;          ///< reference-to-point vector at arrival, body frame
  std::int64_t n0;  ///< integer part of the sample position
  float frac0;      ///< fractional part of the sample position
  float phase0;     ///< carrier phase, cycles, in [0, 1)
};

/// Position of the receive reference point at time t after transmit, and the body
/// attitude then.
SONAR_INLINE Vec3d reference_position(const BpGroup& g, double t, Mat3d& rot) {
  rot = rotation_ned(g.rx_rpy.x + g.rx_rates.x * t, g.rx_rpy.y + g.rx_rates.y * t,
                     g.rx_rpy.z + g.rx_rates.z * t);
  return g.rx_pos + g.rx_vel * t + rot * g.ref_lever;
}

SONAR_INLINE RefDelay reference_delay(const BpGroup& g, const Vec3d& x, bool stop_and_hop) {
  RefDelay r;
  const Vec3d from_tx = x - g.tx_pos;
  r.r_tx = g.tx_plane_wave != 0 ? dot(from_tx, g.tx_dir) : norm(from_tx);
  Mat3d rot;
  Vec3d w = x - reference_position(g, 0.0, rot);
  double r_rx = norm(w);
  double tau = (r.r_tx + r_rx) / g.sound_speed;
  Vec3d v_ref{};
  if (!stop_and_hop) {
    // The receiver keeps moving and turning until the echo arrives: tau is implicit.
    // The map contracts by v/c per pass, so three passes reach round-off.
    for (int i = 0; i < 3; ++i) {
      w = x - reference_position(g, tau, rot);
      r_rx = norm(w);
      tau = (r.r_tx + r_rx) / g.sound_speed;
    }
    w = x - reference_position(g, tau, rot);
    r_rx = norm(w);
    // Velocity of the reference point at arrival (translation plus lever-arm rotation),
    // by central difference of the exact position.
    Mat3d rot_h;
    const double h = 1e-3;
    v_ref = (reference_position(g, tau + h, rot_h) - reference_position(g, tau - h, rot_h)) / (2.0 * h);
  }
  r.tau = tau;
  r.r_rx = r_rx;
  // Elements are described in the body frame, so express w there (at arrival attitude).
  r.w = Vec3f(rot.transposed() * w);
  r.v = Vec3f(rot.transposed() * v_ref);
  // Split while still in double: integer sample, fractional sample, fractional cycle.
  const double s = (tau - g.start_time) * g.sample_rate;
  const double s_floor = std::floor(s);
  r.n0 = static_cast<std::int64_t>(s_floor);
  r.frac0 = static_cast<float>(s - s_floor);
  const double cyc = g.carrier * tau;
  r.phase0 = static_cast<float>(cyc - std::floor(cyc));
  return r;
}

/// Extra delay of one element relative to its group's reference, in float32.
/// Only the differential path is computed here; its size is bounded by the aperture,
/// so float32 keeps it accurate to nanometres. `to_point` is in the body frame.
SONAR_INLINE float element_delay(const BpGroup& g, const BpRecord& e, const RefDelay& ref,
                                 Vec3f& to_point, float& r_elem) {
  to_point = ref.w - e.d;
  r_elem = norm(to_point);
  const float r_ref = static_cast<float>(ref.r_rx);
  const float dr = (dot(e.d, e.d) - 2.0F * dot(ref.w, e.d)) / (r_elem + r_ref);
  // The receiver moves on during the differential delay too. Each element has its own
  // range rate, -u_e . v, because it sees the point from a slightly different direction.
  const float range_rate = -dot(to_point, ref.v) / r_elem;
  return dr / (static_cast<float>(g.sound_speed) - range_rate);
}

/// True when `dir` lies inside factor * FWHM / 2 of the boresight in both planes.
SONAR_INLINE bool in_field_of_view(const Vec3f& boresight, const Vec3f& az, const Vec3f& el,
                                   float bw_az, float bw_el, const Vec3f& dir, float factor) {
  const bool omni_az = !(bw_az == bw_az);
  const bool omni_el = !(bw_el == bw_el);
  if (omni_az && omni_el) return true;
  if (dot(dir, boresight) <= 0.0F) return false;
  const float half_pi = 1.5707963F;
  if (!omni_az) {
    const float lim = 0.5F * factor * bw_az;
    if (lim < half_pi && std::abs(dot(dir, az)) > std::sin(lim)) return false;
  }
  if (!omni_el) {
    const float lim = 0.5F * factor * bw_el;
    if (lim < half_pi && std::abs(dot(dir, el)) > std::sin(lim)) return false;
  }
  return true;
}

struct BpKernel {
  const BpGroup* groups;
  const BpRecord* records;
  const float* samples;      ///< interleaved complex
  const Vec3d* points;
  const float* sinc_table;   ///< [(kSincPhases + 1) * kSincTaps]
  float* image;              ///< interleaved complex, accumulated into
  float* weight_sum;         ///< may be null
  std::int64_t n_groups;
  BpOptions opt;

  /// Interpolate record samples `s` (n complex values) at position n + frac.
  SONAR_HD bool interpolate(const float* s, std::int64_t n_samples, std::int64_t n, float frac,
                            float& re, float& im) const {
    switch (opt.interp) {
      case Interp::Nearest: {
        const std::int64_t i = frac >= 0.5F ? n + 1 : n;
        if (i < 0 || i >= n_samples) return false;
        re = s[2 * i];
        im = s[2 * i + 1];
        return true;
      }
      case Interp::Linear: {
        if (n < 0 || n + 1 >= n_samples) return false;
        const float a = 1.0F - frac;
        re = a * s[2 * n] + frac * s[2 * n + 2];
        im = a * s[2 * n + 1] + frac * s[2 * n + 3];
        return true;
      }
      case Interp::Sinc:
      default: {
        const std::int64_t first = n - (kSincTaps / 2 - 1);
        if (first < 0 || first + kSincTaps > n_samples) return false;
        // Blend the two nearest tabulated phases.
        const float pos = frac * static_cast<float>(kSincPhases);
        int p = static_cast<int>(pos);
        if (p >= kSincPhases) p = kSincPhases - 1;
        const float b = pos - static_cast<float>(p);
        const float* h0 = sinc_table + p * kSincTaps;
        const float* h1 = h0 + kSincTaps;
        float ar = 0.0F, ai = 0.0F;
        for (int k = 0; k < kSincTaps; ++k) {
          const float h = h0[k] + b * (h1[k] - h0[k]);
          ar += h * s[2 * (first + k)];
          ai += h * s[2 * (first + k) + 1];
        }
        re = ar;
        im = ai;
        return true;
      }
    }
  }

  /// Contribution of one group to point x. Returns the weight that contributed.
  SONAR_HD float group_sum(const BpGroup& g, const Vec3d& x, float& acc_re, float& acc_im) const {
    const float two_pi = 6.2831853F;
    const RefDelay ref = reference_delay(g, x, opt.stop_and_hop);

    float tx_gain = 1.0F;
    if (opt.fov_factor > 0.0F || opt.directivity_weight) {
      const Vec3f tx_dir = g.tx_plane_wave != 0 ? Vec3f(g.tx_dir)
                                                : Vec3f((x - g.tx_pos) / (r_tx_safe(ref.r_tx)));
      if (opt.fov_factor > 0.0F &&
          !in_field_of_view(g.tx_boresight, g.tx_az, g.tx_el, g.tx_bw_az, g.tx_bw_el, tx_dir,
                            opt.fov_factor)) {
        return 0.0F;
      }
      if (opt.directivity_weight) {
        const ElementAxes<float> axes{g.tx_boresight, g.tx_az, g.tx_el};
        tx_gain = directivity(axes, g.tx_bw_az, g.tx_bw_el, tx_dir);
      }
    }

    const float fs = static_cast<float>(g.sample_rate);
    const float fc = static_cast<float>(g.carrier);
    float wsum = 0.0F;
    for (std::int64_t ir = 0; ir < g.n_records; ++ir) {
      const BpRecord& e = records[g.first_record + ir];
      Vec3f to_point;
      float r_elem;
      const float dtau = element_delay(g, e, ref, to_point, r_elem);
      float wgt = e.weight * tx_gain;
      if (opt.fov_factor > 0.0F || opt.directivity_weight) {
        const Vec3f dir = to_point / r_elem;
        if (opt.fov_factor > 0.0F &&
            !in_field_of_view(e.boresight, e.az, e.el, e.bw_az, e.bw_el, dir, opt.fov_factor)) {
          continue;
        }
        if (opt.directivity_weight) {
          const ElementAxes<float> axes{e.boresight, e.az, e.el};
          wgt *= directivity(axes, e.bw_az, e.bw_el, dir);
        }
      }
      if (opt.spreading == 1) {
        wgt *= std::sqrt(static_cast<float>(ref.r_tx) * r_elem);
      } else if (opt.spreading == 2) {
        wgt *= static_cast<float>(ref.r_tx) * r_elem;
      }

      const float pos = ref.frac0 + dtau * fs;
      const float pos_floor = std::floor(pos);
      const std::int64_t n = ref.n0 + static_cast<std::int64_t>(pos_floor);
      float re, im;
      const float* s = samples + 2 * (g.sample_offset + ir * g.n_samples);
      if (!interpolate(s, g.n_samples, n, pos - pos_floor, re, im)) continue;

      if (fc != 0.0F) {
        float cyc = ref.phase0 + fc * dtau;
        cyc -= std::floor(cyc);
        const float cs = std::cos(two_pi * cyc), sn = std::sin(two_pi * cyc);
        const float rr = re * cs - im * sn;
        im = re * sn + im * cs;
        re = rr;
      }
      acc_re += wgt * re;
      acc_im += wgt * im;
      wsum += wgt;
    }
    return wsum;
  }

  SONAR_HD static double r_tx_safe(double r) { return r > 1e-12 ? r : 1e-12; }

  SONAR_HD void operator()(std::int64_t ip) const {
    const Vec3d x = points[ip];
    float acc_re = 0.0F, acc_im = 0.0F, wsum = 0.0F;
    for (std::int64_t ig = 0; ig < n_groups; ++ig) {
      wsum += group_sum(groups[ig], x, acc_re, acc_im);
    }
    image[2 * ip] += acc_re;
    image[2 * ip + 1] += acc_im;
    if (weight_sum != nullptr) weight_sum[ip] += wsum;
  }
};

}  // namespace sonar::recon
