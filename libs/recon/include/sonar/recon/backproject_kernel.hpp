// Backprojection kernel body: one output point. Compiled for host and device.
#pragma once

#include <cmath>
#include <cstdint>

#include "sonar/core/directivity.hpp"
#include "sonar/recon/backproject.hpp"

#if SONAR_DEVICE_BUILD
#include <cuda_runtime.h>
#endif

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

/// Where the kernel reads trajectory residuals from.
struct TrajectoryAccess {
  const Vec3f* table = nullptr;  ///< concatenated residual tables (host memory or device memory)
  PositionLookup mode = PositionLookup::Analytic;
#if SONAR_DEVICE_BUILD
  cudaTextureObject_t texture = 0;  ///< 2-D float4 texture: row = ping, column = table entry
#endif
};

/// Residual of the platform trajectory about its first-order model at time t.
/// The residual is small (millimetres to centimetres), which is what lets float32
/// values and 8-bit hardware interpolation weights meet the phase budget.
SONAR_INLINE Vec3d trajectory_residual(const BpGroup& g, std::int64_t ig, const TrajectoryAccess& a,
                                       double t) {
  if (a.mode == PositionLookup::Analytic || g.traj_count < 2) return Vec3d{};
  double u = (t - g.traj_t0) / g.traj_dt;
  const double last = static_cast<double>(g.traj_count - 1);
  u = u < 0.0 ? 0.0 : (u > last ? last : u);
#if defined(__CUDA_ARCH__)
  if (a.mode == PositionLookup::Texture) {
    // Hardware linear filtering between texel centres (at integer + 0.5).
    const float4 v = tex2D<float4>(a.texture, static_cast<float>(u) + 0.5F,
                                   static_cast<float>(ig) + 0.5F);
    return Vec3d(static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z));
  }
#else
  (void)ig;
#endif
  auto i = static_cast<std::int64_t>(u);
  if (i > g.traj_count - 2) i = g.traj_count - 2;
  float w = static_cast<float>(u - static_cast<double>(i));
  if (a.mode == PositionLookup::Texture) {
    // Host emulation of the texture unit: the weight is held with 8 fractional bits.
    w = std::floor(w * 256.0F + 0.5F) / 256.0F;
  }
  const Vec3f p0 = a.table[g.traj_offset + i];
  const Vec3f p1 = a.table[g.traj_offset + i + 1];
  return Vec3d(p0 + (p1 - p0) * w);
}

/// Position of the receive reference point at time t after transmit, and the body
/// attitude then.
SONAR_INLINE Vec3d reference_position(const BpGroup& g, double t, Mat3d& rot,
                                      const Vec3d& residual = Vec3d{}) {
  rot = rotation_ned(g.rx_rpy.x + g.rx_rates.x * t, g.rx_rpy.y + g.rx_rates.y * t,
                     g.rx_rpy.z + g.rx_rates.z * t);
  return g.rx_pos + g.rx_vel * t + rot * g.ref_lever + residual;
}

SONAR_INLINE RefDelay reference_delay(const BpGroup& g, const Vec3d& x, bool stop_and_hop,
                                      std::int64_t ig = 0,
                                      const TrajectoryAccess& traj = TrajectoryAccess{}) {
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
      w = x - reference_position(g, tau, rot, trajectory_residual(g, ig, traj, tau));
      r_rx = norm(w);
      tau = (r.r_tx + r_rx) / g.sound_speed;
    }
    w = x - reference_position(g, tau, rot, trajectory_residual(g, ig, traj, tau));
    r_rx = norm(w);
    // Velocity of the reference point at arrival (translation plus lever-arm rotation),
    // by central difference of the exact position.
    Mat3d rot_h;
    const double h = 1e-3;
    v_ref = (reference_position(g, tau + h, rot_h, trajectory_residual(g, ig, traj, tau + h)) -
             reference_position(g, tau - h, rot_h, trajectory_residual(g, ig, traj, tau - h))) /
            (2.0 * h);
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

/// One-way travel time from a sensor in the water to a point, with refraction at a flat
/// interface when the point lies below it. The ray obeys Snell's law, which is the path
/// of least time (Fermat): the crossing point is found by Newton iteration on the
/// derivative of the travel time, which is convex in the crossing offset.
SONAR_INLINE double refracted_time(const Vec3d& sensor, const Vec3d& x, double c_water,
                                   double c_sediment, double interface_z) {
  const double a = interface_z - sensor.z;  // sensor height above the interface
  const double b = x.z - interface_z;       // point depth below it
  if (!(a > 0.0) || !(b > 0.0)) return norm(x - sensor) / c_water;  // no crossing: straight ray
  const double dx = x.x - sensor.x, dy = x.y - sensor.y;
  const double dist = std::sqrt(dx * dx + dy * dy);  // horizontal separation
  // u: horizontal distance from the sensor to the crossing point, in [0, dist].
  double u = dist * a / (a + b);  // straight-line crossing as the starting guess
  for (int i = 0; i < 8; ++i) {
    const double l1 = std::sqrt(u * u + a * a);
    const double l2 = std::sqrt((dist - u) * (dist - u) + b * b);
    const double f = u / (c_water * l1) - (dist - u) / (c_sediment * l2);
    const double df = a * a / (c_water * l1 * l1 * l1) + b * b / (c_sediment * l2 * l2 * l2);
    u -= f / df;
    u = u < 0.0 ? 0.0 : (u > dist ? dist : u);
  }
  return std::sqrt(u * u + a * a) / c_water + std::sqrt((dist - u) * (dist - u) + b * b) / c_sediment;
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

/// Field-of-view test widened by `margin` radians, for whole blocks of points.
SONAR_INLINE bool in_field_of_view_margin(const Vec3f& boresight, const Vec3f& az, const Vec3f& el,
                                          float bw_az, float bw_el, const Vec3f& dir, float factor,
                                          float margin) {
  const bool omni_az = !(bw_az == bw_az);
  const bool omni_el = !(bw_el == bw_el);
  if (omni_az && omni_el) return true;
  const float half_pi = 1.5707963F;
  if (margin >= half_pi) return true;
  if (dot(dir, boresight) <= -std::sin(margin)) return false;
  if (!omni_az) {
    const float lim = 0.5F * factor * bw_az + margin;
    if (lim < half_pi && std::abs(dot(dir, az)) > std::sin(lim)) return false;
  }
  if (!omni_el) {
    const float lim = 0.5F * factor * bw_el + margin;
    if (lim < half_pi && std::abs(dot(dir, el)) > std::sin(lim)) return false;
  }
  return true;
}

/// Number of blocks along a grid dimension of n points.
SONAR_INLINE std::int64_t block_count(std::int64_t n, std::int64_t block) {
  return (n + block - 1) / block;
}

/// Coarse pass of ray culling: decides, per (ping, block), whether any point of the
/// block can lie in both the transmit and the receive beam of that ping.
///
/// A block is kept when any of its four corners passes the field-of-view test. The
/// paper relies on beam convexity for this, but a beam narrower than a block can pass
/// between the corners, so the test is widened by the angle the block subtends from
/// the sonar (plus the array's own extent and its motion during the flight time).
/// Kept blocks are then tested point by point, without the margin, in the main kernel.
struct CullKernel {
  const BpGroup* groups;
  const BpRecord* records;
  const Vec3d* points;
  unsigned char* mask;  ///< [n_groups * n_blocks]
  std::int64_t rows, cols, block;
  std::int64_t blocks_r, blocks_c;
  float fov_factor;
  float aperture_radius;  ///< largest element offset from a group's reference, m

  SONAR_HD void operator()(std::int64_t idx) const {
    const std::int64_t n_blocks = blocks_r * blocks_c;
    const std::int64_t ig = idx / n_blocks;
    const std::int64_t ib = idx % n_blocks;
    const std::int64_t br = ib / blocks_c, bc = ib % blocks_c;
    const std::int64_t r0 = br * block, c0 = bc * block;
    std::int64_t r1 = r0 + block - 1, c1 = c0 + block - 1;
    if (r1 > rows - 1) r1 = rows - 1;
    if (c1 > cols - 1) c1 = cols - 1;
    const Vec3d corner[4] = {points[r0 * cols + c0], points[r0 * cols + c1],
                             points[r1 * cols + c0], points[r1 * cols + c1]};
    const Vec3d centre = (corner[0] + corner[1] + corner[2] + corner[3]) * 0.25;
    double radius = 0.0;
    for (int k = 0; k < 4; ++k) {
      const double d = norm(corner[k] - centre);
      if (d > radius) radius = d;
    }

    const BpGroup& g = groups[ig];
    Mat3d rot;
    const Vec3d rx = reference_position(g, 0.0, rot);
    const BpRecord& e = records[g.first_record];  // elements of a ping share their pattern
    const float half_pi = 1.5707963F;
    bool keep = false;
    for (int k = 0; k < 4 && !keep; ++k) {
      // Receive side.
      const Vec3d to_rx = corner[k] - rx;
      const double dist_rx = norm(to_rx);
      const double reach = radius + static_cast<double>(aperture_radius);
      float margin_rx = half_pi;
      if (dist_rx > reach) {
        const double flight = 2.0 * (dist_rx + radius) / g.sound_speed;
        const double drift = norm(g.rx_vel) * flight;        // receiver motion before arrival
        const double turn = norm(g.rx_rates) * flight;       // attitude change before arrival
        const double a = std::asin((reach + drift) / dist_rx < 1.0 ? (reach + drift) / dist_rx : 1.0) + turn;
        margin_rx = a < static_cast<double>(half_pi) ? static_cast<float>(a) : half_pi;
      }
      const Vec3f dir_rx = Vec3f(rot.transposed() * (to_rx / (dist_rx > 1e-12 ? dist_rx : 1e-12)));
      if (!in_field_of_view_margin(e.boresight, e.az, e.el, e.bw_az, e.bw_el, dir_rx, fov_factor,
                                   margin_rx)) {
        continue;
      }
      // Transmit side (tested separately: the pair may be bistatic).
      if (g.tx_plane_wave != 0) {
        keep = true;
        break;
      }
      const Vec3d to_tx = corner[k] - g.tx_pos;
      const double dist_tx = norm(to_tx);
      float margin_tx = half_pi;
      if (dist_tx > radius) {
        margin_tx = static_cast<float>(std::asin(radius / dist_tx));
      }
      const Vec3f dir_tx = Vec3f(to_tx / (dist_tx > 1e-12 ? dist_tx : 1e-12));
      if (in_field_of_view_margin(g.tx_boresight, g.tx_az, g.tx_el, g.tx_bw_az, g.tx_bw_el, dir_tx,
                                  fov_factor, margin_tx)) {
        keep = true;
      }
    }
    mask[idx] = keep ? 1 : 0;
  }
};

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
  const unsigned char* mask = nullptr;  ///< culling mask [n_groups * n_blocks], or null
  TrajectoryAccess traj;
  std::int64_t blocks_c = 0, n_blocks = 0;

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
  SONAR_HD float group_sum(const BpGroup& g, std::int64_t ig, const Vec3d& x, float& acc_re,
                           float& acc_im) const {
    const float two_pi = 6.2831853F;
    const RefDelay ref = reference_delay(g, x, opt.stop_and_hop, ig, traj);

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

      std::int64_t n;
      float frac, cyc = 0.0F;
      if (opt.refraction) {
        // Each leg is its own Snell path, so the differential shortcut does not apply:
        // the full delay of this element is computed in double. Near-field volumes are
        // small, so the extra cost is acceptable.
        Mat3d rot;
        const Vec3d ref_pos = reference_position(g, opt.stop_and_hop ? 0.0 : ref.tau, rot,
                                                 trajectory_residual(g, ig, traj, ref.tau));
        const Vec3d elem = ref_pos + rot * Vec3d(e.d);
        const double tau_e =
            refracted_time(g.tx_pos, x, g.sound_speed, opt.sediment_speed, opt.interface_z) +
            refracted_time(elem, x, g.sound_speed, opt.sediment_speed, opt.interface_z);
        const double sp = (tau_e - g.start_time) * g.sample_rate;
        const double sp_floor = std::floor(sp);
        n = static_cast<std::int64_t>(sp_floor);
        frac = static_cast<float>(sp - sp_floor);
        const double c2 = g.carrier * tau_e;
        cyc = static_cast<float>(c2 - std::floor(c2));
      } else {
        const float pos = ref.frac0 + dtau * fs;
        const float pos_floor = std::floor(pos);
        n = ref.n0 + static_cast<std::int64_t>(pos_floor);
        frac = pos - pos_floor;
        if (fc != 0.0F) cyc = ref.phase0 + fc * dtau;
      }
      float re, im;
      const float* s = samples + 2 * (g.sample_offset + ir * g.n_samples);
      if (!interpolate(s, g.n_samples, n, frac, re, im)) continue;

      if (fc != 0.0F) {
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
    std::int64_t ib = 0;
    if (mask != nullptr) {
      ib = (ip / opt.grid_cols / opt.cull_block) * blocks_c + (ip % opt.grid_cols) / opt.cull_block;
    }
    for (std::int64_t ig = 0; ig < n_groups; ++ig) {
      if (mask != nullptr && mask[ig * n_blocks + ib] == 0) continue;
      wsum += group_sum(groups[ig], ig, x, acc_re, acc_im);
    }
    image[2 * ip] += acc_re;
    image[2 * ip + 1] += acc_im;
    if (weight_sum != nullptr) weight_sum[ip] += wsum;
  }
};

}  // namespace sonar::recon
