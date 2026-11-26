// Eigenray providers without a ray tracer: the direct path, image sources in a flat
// surface and bottom, and refraction into a sediment half-space. The C++ counterpart of
// python/gpusonar/sim/propagation.py, with the same arithmetic in the same order.
// Host only: a provider call is a few operations per scatterer and per path.
//
// Units and conventions: SI, world frame north-east-down (z positive down), angles at a
// boundary are GRAZING angles in radians, sediment attenuation is dB per wavelength,
// sediment density is a ratio to `water_density`. Amplitudes are pressure ratios for unit
// pressure at 1 m (no level convention enters here).
//
// Nothing here has a physical default (docs/HARDCODED_AUDIT.md K-2, S-1, S-2, S-3): the
// sound speeds, the boundary reflection coefficients, the sediment and the number of
// bounces are NaN / unset until the caller states them, and the entry points throw
// std::invalid_argument for a configuration that still holds an unset value it would use.
// Limits: the image series is cut at `max_order` bounces and nothing beyond it is
// computed; `image_series_truncation` gives the size of what the cut leaves out so that
// the caller can report it.
#pragma once

#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

#include "sonar/core/vec.hpp"

namespace sonar::sim {

/// "Not stated": the value of every field that has no physical default.
inline constexpr double kUnset = std::numeric_limits<double>::quiet_NaN();

/// Eigenrays from one source to S field points, K rays each. Arrays are row-major
/// [S, K] (directions [S, K, 3]). `amplitude` is the complex pressure at the field point
/// for unit pressure at 1 m: spreading and boundary reflections, no absorption.
struct EigenraySet {
  std::int64_t n_points = 0;
  std::int64_t n_rays = 0;
  double sound_speed = kUnset;                  ///< m/s; set by the provider
  std::vector<double> delay;                    ///< s
  std::vector<std::complex<double>> amplitude;
  std::vector<double> launch;                   ///< unit vector leaving the source
  std::vector<double> arrival;                  ///< unit vector of travel at the field point
  std::vector<std::int64_t> n_surface;
  std::vector<std::int64_t> n_bottom;
  std::vector<std::uint8_t> valid;

  /// Size every array for [n_points, n_rays], zero-filled, all rays valid.
  void resize(std::int64_t points, std::int64_t rays);
};

/// The fields of an eigenray set that pairing reads, as pointers into arrays owned
/// elsewhere (an EigenraySet, or the arrays of a provider written in Python).
struct EigenrayView {
  std::int64_t n_points = 0;
  std::int64_t n_rays = 0;
  double sound_speed = kUnset;                      ///< m/s
  const double* delay = nullptr;                    ///< [S, K]
  const std::complex<double>* amplitude = nullptr;  ///< [S, K]
  const double* launch = nullptr;                   ///< [S, K, 3]
  const double* arrival = nullptr;                  ///< [S, K, 3]
  const std::uint8_t* valid = nullptr;              ///< [S, K], non-zero for a ray that exists
};

EigenrayView view(const EigenraySet& set);

/// The direct path: delay = distance / c, amplitude = 1 / distance.
EigenraySet free_field_eigenrays(double sound_speed, const Vec3d& source, const double* points,
                                 std::int64_t n_points);

/// A vertical wall: the image in the plane `position` along horizontal axis 0 (x) or 1 (y).
struct Wall {
  int axis = 0;
  double position = 0.0;
  std::complex<double> coefficient{kUnset, 0.0};  ///< pressure reflection coefficient; no default
};

/// A fluid sediment half-space for the Rayleigh reflection coefficient; with `shear_speed` > 0 an
/// elastic solid (rock, A8): Jackson and Richardson (2007) (9.34)-(9.40).
struct FluidBottom {
  double speed = kUnset;          ///< m/s; no default (a sediment is a fact of the site)
  double density = kUnset;        ///< in the units of water_density; no default
  double attenuation = 0.0;       ///< dB per wavelength; 0 = lossless
  double water_density = 1.0;     ///< 1 = `density` is a ratio to water
  double shear_speed = 0.0;       ///< m/s; 0 = a fluid (the coefficient is the fluid expression, unchanged)
  double shear_attenuation = 0.0; ///< dB per wavelength of the shear wave
};

struct ImageSourceConfig {
  double sound_speed = kUnset;    ///< m/s; no default
  bool has_surface = true;
  double surface_z = 0.0;
  bool has_bottom = false;
  double bottom_z = 0.0;
  std::complex<double> surface_coefficient{-1.0, 0.0};  ///< pressure release: physics, not a site
  /// Used for a bottom bounce when `has_fluid_bottom` is false. No default: a bottom
  /// without a stated coefficient or sediment is an error, not a made-up 6 dB a bounce.
  std::complex<double> bottom_coefficient{kUnset, 0.0};
  /// Bounces in total along one path. No default (-1): the caller decides how many carry
  /// the field at its range and depth. Paths with more bounces are NOT computed; see
  /// image_series_truncation for what that leaves out.
  int max_order = -1;
  std::vector<Wall> walls;        ///< first order only, on top of the vertical series
  bool has_fluid_bottom = false;  ///< each bottom bounce uses the Rayleigh coefficient
  FluidBottom fluid_bottom;
};

/// One member of the image series: the mirror planes met in order, and the wall (or -1).
struct ImageRay {
  std::vector<double> mirrors;    ///< z of each reflecting plane, in the order met
  std::int64_t n_surface = 0;
  std::int64_t n_bottom = 0;
  int wall = -1;                  ///< index into ImageSourceConfig::walls
};

/// The rays of a configuration, in the order of the Python provider: the direct path,
/// the series starting at the surface, the series starting at the bottom; each followed
/// by its wall images.
std::vector<ImageRay> image_series(const ImageSourceConfig& config);

EigenraySet image_source_eigenrays(const ImageSourceConfig& config, const Vec3d& source,
                                   const double* points, std::int64_t n_points);

/// Throws std::invalid_argument naming the first field of `config` that the image series
/// would use and that is not stated (sound speed, max_order, a bottom or wall coefficient,
/// the sediment of a fluid bottom).
void check_config(const ImageSourceConfig& config);

/// What cutting the image series at `config.max_order` leaves out, for a source and a
/// receiver at depths `source_z` and `receiver_z` a horizontal distance `range` apart:
/// the largest amplitude among the rays with max_order + 1 bounces (the first order not
/// computed), relative to the direct path's. A caller reports it (or raises the order)
/// instead of dropping the rays silently. 0 when the series has ended anyway (a single
/// boundary). Bottom bounces use the configuration's coefficient or its Rayleigh value.
double image_series_truncation(const ImageSourceConfig& config, double source_z, double receiver_z,
                               double range);

/// The general image construction (sonar-0nsi in NumPy, `propagation._PlaneImages`; here since sonar-k9v7):
/// image sources for any set of planes in water of one sound speed: the surface, a bottom tilted in x and y,
/// and walls as any plane, optionally bounded. The same arithmetic as the NumPy construction in the same order,
/// except the dot products, which NumPy hands to BLAS (summed in another order, with fused multiply-adds on some
/// machines): results agree to rounding, not to the bit.
///
/// A path from the source S to the field point P meeting planes p1, ..., pn in that order has the image
/// I = R1(R2(...Rn(P))) (R: the mirror in a plane) and the length |I - S|. Sequences are grown from P outwards, a
/// mirror applied only where the image so far is on the water side of that plane, with no plane twice in a row,
/// at most `max_order` surface and bottom bounces and at most `wall_order` wall bounces; a sequence is kept as a
/// ray (a column of the result) when it is possible for at least one field point. A ray is valid for a point
/// when, traced back from S, each leg meets the next plane between its ends on the plane's face (within its
/// bounds); the source, the point and every specular point are on the water side of every unbounded plane; and
/// no leg passes through a bounded plane's face. Invalid rays have amplitude 0 and `valid` 0 (their delay and
/// directions are still written: those of the image). Coefficients: a surface's or wall's `coefficient`; a
/// bottom's the Rayleigh coefficient of `fluid_bottom` at that bounce's own grazing angle to its plane, or
/// `bottom_coefficient`. Not modelled: diffraction at a face's edge, refraction, roughness.
enum class PlaneKind { Surface, Bottom, Wall };

/// Distance (m) within which a point is on a plane (scatterers laid on a boundary are): PLANE_TOLERANCE_M.
inline constexpr double kPlaneTolerance = 1e-6;

/// One mirror: the points x with normal . (x - point) = 0.
struct Plane {
  PlaneKind kind = PlaneKind::Wall;
  Vec3d point{};
  /// Unit normal pointing into the water. Taken as given (make_plane normalises), so that a normal
  /// normalised by the caller is not rounded again.
  Vec3d normal{};
  /// Pressure reflection coefficient of a surface or a wall; no default. A bottom's comes from the configuration.
  std::complex<double> coefficient{kUnset, 0.0};
  /// The face, per axis x, y, z: lower[a] <= x[a] <= upper[a] where has_bound[a]. A plane with no bound is
  /// unbounded (a boundary of the water); one with any bound is a face that reflects and blocks only there.
  bool has_bound[3] = {false, false, false};
  double lower[3] = {0.0, 0.0, 0.0};
  double upper[3] = {0.0, 0.0, 0.0};
  bool bounded() const { return has_bound[0] || has_bound[1] || has_bound[2]; }
};

/// A plane of `kind` through `point` with the normal `normal` (any length, pointing into the water) normalised.
Plane make_plane(PlaneKind kind, const Vec3d& point, const Vec3d& normal,
                 std::complex<double> coefficient = {kUnset, 0.0});

struct PlaneImageConfig {
  double sound_speed = kUnset;       ///< m/s; no default
  std::vector<Plane> planes;         ///< in the order rays are listed by (the NumPy order: surface, bottom, walls)
  int max_order = -1;                ///< surface and bottom bounces along one path; no default
  int wall_order = -1;               ///< wall bounces along one path; no default
  /// A bottom bounce's coefficient when `has_fluid_bottom` is false; no default.
  std::complex<double> bottom_coefficient{kUnset, 0.0};
  bool has_fluid_bottom = false;     ///< each bottom bounce uses the Rayleigh coefficient at its own grazing angle
  FluidBottom fluid_bottom;
};

/// Throws std::invalid_argument for an unstated sound speed, order or coefficient, a normal that is not a unit
/// vector, or bounds with lower > upper.
void check_config(const PlaneImageConfig& config);

/// Eigenrays of the general construction from `source` to `points` [S, 3]: rays in the order of
/// `propagation._PlaneImages.sequences` (the direct path first, then the sequences grown level by level).
EigenraySet plane_image_eigenrays(const PlaneImageConfig& config, const Vec3d& source, const double* points,
                                  std::int64_t n_points);

/// Plane-wave pressure reflection coefficient of a fluid half-space at a grazing angle
/// (radians): R = (m sin g - sqrt(n^2 - cos^2 g)) / (m sin g + sqrt(n^2 - cos^2 g)),
/// m = rho_b / rho_w, n = c_w / c_b, attenuation as an imaginary part of c_b.
/// Sources: docs/REFERENCES.md Sec. 12.4 (the dB/wavelength conversion is ours).
std::complex<double> rayleigh_reflection(double grazing, double c_water, const FluidBottom& bottom);

/// Least-time paths from a source in the water to points below a flat interface (Snell).
/// `time` [S] and `crossing` [S, 3] are written. Points above the interface (or a source
/// below it) get the straight path and a crossing point equal to the point itself.
void snell_crossing(const Vec3d& source, const double* points, std::int64_t n_points, double c_water,
                    double c_sediment, double interface_z, double* time, double* crossing);

struct SedimentConfig {
  double sound_speed = kUnset;     ///< m/s in the water; no default
  double sediment_speed = kUnset;  ///< m/s; no default
  double interface_z = 0.0;
  std::complex<double> transmission{1.0, 0.0};  ///< used when `fluid` is false (the default tier)
  /// Opt-in (G-3): the flat-interface fluid field above the critical angle instead of `transmission`.
  bool fluid = false;
  /// The sediment of `fluid`: every field stated (speed equal to `sediment_speed`; the
  /// attenuation in dB per wavelength has no default here: no loss is assumed).
  FluidBottom bottom{kUnset, kUnset, kUnset, 1.0};
  double frequency = kUnset;       ///< Hz, for the attenuation of `fluid`; no default
  /// With `fluid`: a point whose straight line from the source meets the interface below the
  /// critical angle arccos(c_w / c_s) gets the evanescent plane-wave field (15.1) at that line's
  /// grazing angle theta_d, V_wp(theta_d) / sqrt(a_rho) exp(-k_w Im{sqrt(1/a_p^2 - cos^2 theta_d)} d),
  /// d its depth under the interface, instead of the least-time ray's leg. False: the ray's.
  bool evanescent = false;
};

/// One refracted eigenray per point. Amplitude: spreading over the travel-time distance
/// times, for points below the interface, `transmission` (the default) or, with `fluid`,
/// V_wp(theta_w) / sqrt(a_rho) exp(-k_p'' L): V_wp = 1 + V the pressure transmission
/// coefficient at the grazing angle of the water leg (Jackson and Richardson 2007, (8.47),
/// (8.55)), a_rho = density / water_density (the 1 / a_rho of the return leg, (8.68), split
/// evenly between the two legs of an echo), L the length of the sediment leg and
/// k_p'' = k_w delta / nu ((8.18), (8.26)). The same operations in the same order as
/// propagation.SedimentProvider (penetration: fluid), which also says what is still approximate.
EigenraySet sediment_eigenrays(const SedimentConfig& config, const Vec3d& source, const double* points,
                               std::int64_t n_points);

/// Kernel paths for every (outbound eigenray, return eigenray) pair of every scatterer:
/// the apparent positions q (seen from the transmitter) and p (seen from the receive
/// reference point) and the amplitude on top of spherical spreading. Pairs with an
/// invalid ray or zero amplitude are dropped; the order is scatterer, outbound ray,
/// return ray.
struct TwoWayPaths {
  std::vector<double> q;                        ///< [P, 3]
  std::vector<double> p;                        ///< [P, 3]
  std::vector<std::complex<double>> amplitude;  ///< [P]
  std::vector<double> velocity;                 ///< [P, 3], or empty when nothing moves
  std::int64_t size() const { return static_cast<std::int64_t>(amplitude.size()); }
};

/// `amplitudes` [S]. Optional (null to leave out): `velocities` [S, 3]; `lambert` [S],
/// non-zero for boundary patches weighted by sqrt(sin g_in sin g_out); `normals` [S, 3]
/// for patches with their own normal (rows with a NaN have none).
TwoWayPaths two_way_paths(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                          const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                          const double* velocities, const std::uint8_t* lambert, const double* normals);

/// The same into arrays of the caller, each with room for every pair (n_points * rays
/// out * rays back rows): q [.., 3], p [.., 3], amplitude [..] and, when not null,
/// velocity [.., 3] (the scatterer's velocity of each kept pair). Returns the number of
/// pairs kept; rows beyond it are not written.
std::int64_t two_way_paths(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                           const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                           const double* velocities, const std::uint8_t* lambert, const double* normals,
                           double* q, double* p, std::complex<double>* amplitude, double* velocity);

/// Path-length rates of the eigenrays (frequency spreading within a ping: a ray off a moving
/// boundary lengthens at `range_rate` m/s). `out` [S, rays out] and `back` [S, rays back];
/// either may be null (no rate on that leg).
struct RangeRates {
  const double* out = nullptr;
  const double* back = nullptr;
};

/// The pairing with a velocity for each end of each pair, so that a ray can carry its own
/// Doppler: q_velocity = scatterer velocity + out rate * outbound launch direction, and
/// p_velocity = scatterer velocity + back rate * return launch direction (the apparent
/// position recedes along the ray at the rate its path lengthens; the synthesis kernel
/// turns that into the frequency shift). `velocities` may be null (rates only).
/// q_velocity and p_velocity [.., 3] are both required. Otherwise as two_way_paths.
std::int64_t two_way_paths_doppler(const EigenrayView& out, const EigenrayView& back, const Vec3d& tx_position,
                                   const Vec3d& rx_reference, const std::complex<double>* amplitudes,
                                   const double* velocities, const std::uint8_t* lambert, const double* normals,
                                   const RangeRates& rates, double* q, double* p, std::complex<double>* amplitude,
                                   double* q_velocity, double* p_velocity);

/// True when any component of `velocities` [n_points, 3] is not zero (null: false).
bool any_moving(const double* velocities, std::int64_t n_points);

}  // namespace sonar::sim
