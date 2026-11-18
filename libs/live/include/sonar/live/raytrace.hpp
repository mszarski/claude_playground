// Rays and eigenrays through a layered sound-speed profile: the chain's one ray calculation.
//
// The C++ counterpart of python/gpusonar/raytrace.py (the reference; tests/python/test_raytrace_cpp.py
// holds the two together), written from the same derivation and nothing else: the profile is
// piecewise linear in depth, so in every layer a ray is an arc of a circle and everything is closed
// form. z is depth (down positive), theta the grazing angle, s = sin(theta) signed (positive: going
// down), p = cos(theta) / c the Snell constant of the ray, g = dc/dz of the layer:
//
//     ds/dr = -p g               s is linear in range inside a layer, through a turning point too
//     c = sqrt(1 - s^2) / p      which gives the depth, z = z_1 + (c - c_1) / g
//     t_b - t_a = (atanh(s_a) - atanh(s_b)) / g
//
// A ray is stepped from layer boundary to layer boundary, mirrored at the surface and at the bottom.
// No Bellhop source was read or copied (Bellhop is run as a separate program in the tests, as an oracle).
//
// Three users, one calculation:
//   ray_depth_arcs   the depth an arrival seen at an elevation started at (the stacked rows' depth)
//   RayFan           a fan of rays from one depth, read at one or more ranges: every arrival at any
//                    depth there, by interpolation along the wavefront (below). The multipath-depth
//                    estimator's predicted pattern and the coverage picture's loss come from it.
//   eigenrays        the reference's own search (fan, bracket, bisection), for tests and single cases
//
// Interpolation along the wavefront. At a fixed range the rays of a fan that share a reflection
// history end on one continuous sheet of the wavefront. On it the travel time's gradient is the
// slowness vector: dt/dz = s / c (s at the ray's end), dt/dr = p. So between two neighbouring rays
// that straddle a depth the arrival there has, by a cubic Hermite polynomial in z, a travel time good
// to far below a carrier period, a launch angle (linear), and the spreading of the tube between the
// two rays: I(r) / I(1 m) = cos(theta_0) / (r |dz / dtheta_0| cos(theta_r)). Where two neighbours have
// different histories (one has reflected and the other not) the launch angle between them is halved
// until each part has one history, so a sheet reaches the boundary it ends on. At a caustic
// dz / dtheta_0 -> 0: the amplitude is clipped at `caustic_clip` times the spherical value and the
// arrival is marked; no caustic phase is applied.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

namespace sonar::live {

/// c(z), linear between knots, from the surface (z = 0) to the bottom (`water_depth` <= 0 or not
/// finite: no bottom, the last gradient continues 20 km down). As raytrace.Profile.
struct RayProfile {
  std::vector<double> depth, speed, gradient;
  double bottom = 0.0;
  bool has_bottom = false;
  RayProfile() = default;
  RayProfile(const std::vector<double>& depths, const std::vector<double>& speeds, double water_depth = 0.0);
  double c(double z) const;
  /// Index of the layer a ray at depth z moving down (or up) is in.
  std::size_t layer(double z, bool down) const;
};

struct RayEnd {
  double z = 0.0, time = 0.0, s = 0.0;   ///< depth, travel time, sin of the grazing angle there (positive: going down)
  int n_surface = 0, n_bottom = 0, turns = 0;
  bool ok = false;                       ///< false: it never got there
};

/// One ray, followed in range; `to(r)` may be called with increasing ranges.
class RayStepper {
 public:
  RayStepper(const RayProfile& profile, double z0, double launch);   ///< launch: rad, positive DOWN
  RayEnd to(double r_end, int max_bounces = 1 << 30, std::int64_t max_steps = 100000);
  double p() const { return p_; }

 private:
  const RayProfile* prof_;
  double p_, s_, z_, r_ = 0.0, t_ = 0.0;
  int ns_ = 0, nb_ = 0, turns_ = 0;
  bool dead_ = false;
};

/// The ray leaving depth z0 at grazing angle `launch` (rad, positive DOWN), at horizontal range r_end.
RayEnd ray_trace(const RayProfile& profile, double z0, double launch, double r_end);

/// Depth at which an arrival seen at `elevation` (rad, UP positive) at `receiver_depth` started, back
/// along the ray for the one-way travel time slant_range / c(receiver), in closed form layer by
/// layer (bisection on range for the travel time). As raytrace.ray_depth.
double ray_depth_arcs(double receiver_depth, double elevation, double slant_range, const std::vector<double>& ssp_depth,
                      const std::vector<double>& ssp_speed, double water_depth = 0.0);

/// Whether sound can arrive at `receiver_depth` from elevation `elevation` (rad, UP positive) after a slant range
/// `slant_range` with at most `max_bounces` boundary reflections, from a depth no greater than `max_depth` (m; the water
/// depth when the profile has a bottom above it): the ray traced back from the receiver for the straight path's
/// horizontal distance. What the stacked rows' elevation search may be limited to (elevation.sector_from_rays).
bool ray_arrives(const RayProfile& profile, double receiver_depth, double elevation, double slant_range, int max_bounces, double max_depth);

/// Plane-wave pressure reflection coefficient of a fluid half-space (Rayleigh), as sim/propagation.py:
/// R = (m sin g - sqrt(n^2 - cos^2 g)) / (m sin g + sqrt(n^2 - cos^2 g)), m = rho_b / rho_w, n = c_w / c_b,
/// attenuation (dB per wavelength) as an imaginary part of c_b.
std::complex<double> rayleigh_reflection(double grazing, double c_water, double speed, double density, double attenuation);

struct RayBottom {
  bool given = false;                    ///< false: bottom-reflected rays are left out
  double speed = 0.0, density = 0.0, attenuation = 0.0;   ///< m/s, re water, dB per wavelength
};

struct Eigenray {
  double launch = 0.0, arrive = 0.0;     ///< rad at the source and at the receiver, positive down (the ray's own direction)
  double time = 0.0;
  std::complex<double> amplitude;        ///< pressure re 1 m from the source (spreading and reflection coefficients; no absorption)
  int n_surface = 0, n_bottom = 0, turns = 0;
  bool caustic = false;
};

struct EigenrayOptions {
  double max_angle_deg = 60.0;
  std::int64_t n_fan = 2401;
  int max_bounces = 4;
  double surface_coefficient = -1.0;
  RayBottom bottom;
  double caustic_clip = 10.0;
};

/// Every ray joining a source at `z_source` and a receiver at `z_receiver` a horizontal `range` apart, earliest first.
std::vector<Eigenray> eigenrays(const RayProfile& profile, double z_source, double z_receiver, double range, const EigenrayOptions& options = {});
/// dB re 1 m: incoherent (frequency <= 0) or coherent at `frequency`.
double transmission_loss(const std::vector<Eigenray>& rays, double frequency = 0.0);

/// An arrival read off a fan: geometric only (reflection coefficients are the caller's, from the
/// counts and the grazing angles; no absorption).
struct RayArrival {
  double time = 0.0;
  double p = 0.0;                        ///< horizontal slowness cos(theta) / c, s/m: d time / d range at fixed depth
  double launch = 0.0, s_end = 0.0;
  double amplitude = 0.0;                ///< pressure re 1 m, spreading only (>= 0)
  double grazing_surface = 0.0, grazing_bottom = 0.0;   ///< rad, at the boundaries (Snell)
  int n_surface = 0, n_bottom = 0;
  bool caustic = false;
};

class RayFan {
 public:
  /// `n` rays between -max_angle and +max_angle (rad) from `z_source`, read at each of `ranges` (increasing).
  /// Rays with more than `max_bounces` boundary reflections are dropped.
  RayFan(const RayProfile& profile, double z_source, const std::vector<double>& ranges, double max_angle, std::int64_t n, int max_bounces,
         double caustic_clip = 10.0);
  /// The same with the launch angles given (rad, positive down, increasing, none exactly 0): a fan that is denser where
  /// the far ranges need it.
  RayFan(const RayProfile& profile, double z_source, const std::vector<double>& ranges, const std::vector<double>& launches, int max_bounces,
         double caustic_clip = 10.0);
  /// The arrivals at depth z at ranges[i_range], appended to `out` in fan order.
  void arrivals(std::size_t i_range, double z, std::vector<RayArrival>& out) const;
  const std::vector<double>& ranges() const { return ranges_; }
  double source_depth() const { return z_source_; }
  std::int64_t rays() const { return n_; }

 private:
  RayProfile prof_;
  struct Segment {
    double launch_a, launch_b;
    RayEnd a, b;
  };
  void shoot(int max_bounces);
  void refine(std::size_t i_range, double launch_a, const RayEnd& a, double launch_b, const RayEnd& b, int max_bounces, int level);
  void arrival(const Segment& seg, double r, double z, std::vector<RayArrival>& out) const;
  double z_source_, c_source_, clip_;
  std::vector<std::vector<Segment>> edges_;   ///< [range]: the parts of a sheet between a fan ray and the boundary the sheet ends on
  std::int64_t n_;
  std::vector<double> ranges_, launch_;
  std::vector<RayEnd> ends_;             ///< [range][ray]
};

}  // namespace sonar::live
