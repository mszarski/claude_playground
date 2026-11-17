// Eigenray providers without a ray tracer: the direct path, image sources in a flat
// surface and bottom, and refraction into a sediment half-space. The C++ counterpart of
// python/gpusonar/sim/propagation.py, with the same arithmetic in the same order.
// Host only: a provider call is a few operations per scatterer and per path.
#pragma once

#include <complex>
#include <cstdint>
#include <vector>

#include "sonar/core/vec.hpp"

namespace sonar::sim {

/// Eigenrays from one source to S field points, K rays each. Arrays are row-major
/// [S, K] (directions [S, K, 3]). `amplitude` is the complex pressure at the field point
/// for unit pressure at 1 m: spreading and boundary reflections, no absorption.
struct EigenraySet {
  std::int64_t n_points = 0;
  std::int64_t n_rays = 0;
  double sound_speed = 1500.0;
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
  double sound_speed = 1500.0;
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
  std::complex<double> coefficient{0.8, 0.0};
};

/// A fluid sediment half-space for the Rayleigh reflection coefficient.
struct FluidBottom {
  double speed = 1600.0;          ///< m/s
  double density = 1.8;           ///< in the units of water_density
  double attenuation = 0.0;       ///< dB per wavelength
  double water_density = 1.0;
};

struct ImageSourceConfig {
  double sound_speed = 1500.0;
  bool has_surface = true;
  double surface_z = 0.0;
  bool has_bottom = false;
  double bottom_z = 0.0;
  std::complex<double> surface_coefficient{-1.0, 0.0};
  std::complex<double> bottom_coefficient{0.5, 0.0};
  int max_order = 2;              ///< bounces in total along one path
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

/// Plane-wave pressure reflection coefficient of a fluid half-space at a grazing angle
/// (radians): R = (m sin g - sqrt(n^2 - cos^2 g)) / (m sin g + sqrt(n^2 - cos^2 g)),
/// m = rho_b / rho_w, n = c_w / c_b, attenuation as an imaginary part of c_b.
std::complex<double> rayleigh_reflection(double grazing, double c_water, const FluidBottom& bottom);

/// Least-time paths from a source in the water to points below a flat interface (Snell).
/// `time` [S] and `crossing` [S, 3] are written. Points above the interface (or a source
/// below it) get the straight path and a crossing point equal to the point itself.
void snell_crossing(const Vec3d& source, const double* points, std::int64_t n_points, double c_water,
                    double c_sediment, double interface_z, double* time, double* crossing);

struct SedimentConfig {
  double sound_speed = 1500.0;
  double sediment_speed = 1700.0;
  double interface_z = 0.0;
  std::complex<double> transmission{1.0, 0.0};
};

/// One refracted eigenray per point. Amplitude: spreading over the travel-time distance
/// times `transmission` for points below the interface.
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

/// True when any component of `velocities` [n_points, 3] is not zero (null: false).
bool any_moving(const double* velocities, std::int64_t n_points);

}  // namespace sonar::sim
