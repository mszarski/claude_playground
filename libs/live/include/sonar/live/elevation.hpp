// Elevation, and from it depth, with a flank array of two or more stacked rows: the C++
// counterpart of python/gpusonar/elevation.py (which explains the method); the two are
// compared in tests/python/test_elevation.py.
//
//   rows       the lines of the receive array, found from the element positions in the record
//   map        each row beamformed on its own, the rows' power added (chain.cpp)
//   elevation  at each detection the whole array steered over elevation, by the backprojection
//              kernel (host or CUDA: no kernel of its own), with quality, sigma and an
//              ambiguity flag
//   depth      own depth + range x the vertical part of the direction; folded into the water
//              column for arrivals that came by the surface or the bottom
//   passive    each row steered at a bearing per frequency bin and block, the rows' beams
//              then steered over elevation
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/live/environment.hpp"

namespace sonar::live {

struct ElevationOptions {
  bool enabled = false;            ///< off: the chain as it was; on with a one-row array: the same
  double first_deg = -30.0, last_deg = 30.0, step_deg = 0.25;  ///< elevation searched, vehicle frame, up positive
  /// Search only where water depth and range let an arrival come from: straight rays with this many
  /// boundary bounces, plus a margin (rad). Needs the water depth.
  bool sector_from_depth = false;
  double sector_bounces = 1.0, sector_margin = 0.017453292519943295;
  double range_window_m = 0.0;     ///< half-width of the range search about a detection; 0: 1.5 range cells
  std::int64_t max_detections = 32;  ///< strongest detections given an elevation, per ping
  /// Take the earliest arrival in the range window that is within first_arrival_db of the strongest
  /// and first_arrival_snr_db above the noise (the direct path arrives first); false: the strongest.
  bool first_arrival = true;
  double first_arrival_db = 8.0, first_arrival_snr_db = 13.0;
  double min_quality = 0.6;        ///< below this no elevation is reported
  double ambiguity_db = 1.0;       ///< a second lobe within this of the first is never trusted, and
  double ambiguity_sigma = 3.0;    ///< one further down must lose by this many sigma of the noise: else flagged ambiguous
  double alternatives_db = 6.0;    ///< lobes within this of the first are the candidates a track chooses among
  bool correct_bearing = true;     ///< report and track the horizontal bearing, not the cone angle
  bool depth = true;               ///< report depth, and let tracks carry it
  bool fold = true;                ///< fold a depth outside the water column back in
  bool classify = true;            ///< a track whose depth lies outside the water column is multipath
  double classify_sigma = 2.0;     ///< ... by more than this many sigma
  double water_depth = 0.0;        ///< m; 0: the record's environment.water_depth (NaN or 0 there: unknown)
  double water_depth_sigma = 0.0;  ///< m: into the bottom test and the depth of bottom-bounce arrivals
  /// Roll, pitch and yaw as they were when each echo arrived, from the record's attitude history
  /// (navigation.ins); false: as the record's pose has them.
  bool attitude_arrival = false;
  double attitude_rate_sigma = 0.0;  ///< rad/s: added uncertainty per second of gap to the nearest attitude sample
  /// An assumed mean vertical sound-speed gradient (1/s): depth along the ray through
  /// c(z) = local speed + gradient (z - own depth) instead of the straight line.
  bool use_gradient = false;
  double gradient = 0.0;
  double gradient_sigma = 0.05;    ///< 1/s: how well the gradient is known; depth sigma gains range^2 x this / 2c
  double surface_z = 0.0;          ///< z of the sea surface in the navigation's frame
  /// A sound-speed profile (m, m/s): depth_m then comes from a ray traced through it, not the straight line.
  std::vector<double> ssp_depth, ssp_speed;
  double roll_sigma = 0.0;         ///< rad: the navigation's roll accuracy, added to the elevation sigma
  /// Elements within this of one line are one row. NaN: derived (derive.hpp: 2 cm, or a quarter of the
  /// shortest wavelength when that is less).
  double row_tolerance_m = std::numeric_limits<double>::quiet_NaN();
  std::int64_t min_row_elements = 3;  ///< a line of fewer elements than this is not a row
  /// Opt-in: search only the elevations from which sound can arrive at all, by rays traced back through the water the
  /// chain holds this ping (the ladder's profile, elevation.ssp, the assumed gradient, else isovelocity): at most
  /// `sector_rays_bounces` boundary reflections, from no deeper than `sector_rays_max_depth` (or the bottom), widened by
  /// `sector_rays_margin`. An array of unequally spaced rows has far lobes only a few dB down; where several arrivals
  /// share a range cell one of them can stand above the true ones, and at 30 degrees its bearing correction is a degree.
  bool sector_from_rays = false;
  double sector_rays_bounces = 2.0, sector_rays_margin = 0.017453292519943295, sector_rays_max_depth = 500.0;
  std::vector<double> sector_ray_depth, sector_ray_speed;   ///< the profile, set by the chain each ping
  /// true (the default; `elevation.ray_method: arcs`): depth_m and its refraction sigma by the chain's one ray
  /// calculation (raytrace.hpp: closed-form arcs layer by layer, the same rays the multipath depth and the coverage
  /// picture use). false (`ray_method: steps`, `elevation_ray_arcs=0`): the stepped Snell ray of ray_depth below, the
  /// default before. The two agree to under 1 m out to 12 km in the deployed gradient prior and to a few metres in
  /// strongly refracting water at long range; a ray launched exactly level into water that bends it down is left
  /// level by the stepped ray and bent by the arcs.
  bool ray_arcs = true;
  /// Opt-in: the depth sigma says so when echoes the rows cannot tell apart share the range resolution cell
  /// (multipath_depth_limit below): `depth_sigma_m` gains the term, and the report names it. Paths of at most
  /// `multipath_sigma_bounces` boundary reflections a leg, target depths to `multipath_sigma_max_depth` or the bottom.
  bool multipath_sigma = false;
  double multipath_sigma_bounces = 2.0, multipath_sigma_max_depth = 500.0;
  /// The sound-speed ladder (environment.hpp): which profile the depth is traced through, how its
  /// uncertainty reaches depth_sigma_m, and the rung named with every depth. Off: as before.
  ProfileOptions profile;
};

/// Channel rows of each line of the receive array, in channel order (see find_rows in elevation.py).
std::vector<std::vector<std::int64_t>> find_rows(const PingRecord& record, double tolerance = 0.02,
                                                 std::int64_t min_elements = 3);

/// A copy of the record holding only the given channel rows: an ordinary line array.
PingRecord row_record(const PingRecord& record, const std::vector<std::int64_t>& rows);

/// The stacked array's own frame in the world, from the record's pose.
struct ElevationFrame {
  Vec3d origin;                 ///< mean position of the elements used
  Vec3d along, up, look;        ///< unit: along the rows; the vehicle's up made perpendicular to it; the look side
  std::vector<double> heights;  ///< element positions along `up`, m about the origin (the elements used)
};
ElevationFrame elevation_frame(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, bool starboard);
/// The same with another attitude (roll, pitch, yaw) than the pose's.
ElevationFrame elevation_frame(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, bool starboard, const Vec3d& rpy);

/// Attitude from the record's history (navigation.ins, seconds after the transmit) at time t: interpolated
/// between the samples around it (gap 0), or the nearest sample and its distance in time. False without a history.
bool attitude_at(const PingRecord& record, double t, Vec3d& rpy, double& gap);

/// Depth uncertainty (m) from a mean sound-speed gradient known only to gradient_sigma (1/s): range^2 sigma / 2c.
double refraction_sigma(double slant_range, double sound_speed, double gradient_sigma);

/// World unit vector towards an arrival at a cone angle and an elevation in the array's frame.
Vec3d arrival_direction(const ElevationFrame& frame, double sin_cone, double sin_elevation);

/// Root-mean-square wavenumber of a flat band, rad/m.
double rms_wavenumber(double f_low, double f_high, double sound_speed);

/// Sum of the squared aperture weights of a row of n elements, as the map applies them.
double weight_power(std::int64_t n_elements, bool hann);

struct ElevationEstimate {
  bool valid = false;
  double sin_elevation = 0.0;  ///< array frame, up positive
  double sigma = 0.0;          ///< rad
  double quality = 0.0;        ///< coherent power over the rows' incoherent power, 0..1
  bool ambiguous = false;
  double snr_db = 0.0;         ///< at one hydrophone
  double range = 0.0;          ///< slant range of the peak, m
  Vec3d world;                 ///< unit vector towards the arrival (north, east, down)
  std::vector<Vec3d> alternatives;  ///< the other candidate lobes, when ambiguous
  double attitude_gap = 0.0;   ///< s between the echo's arrival and the attitude used for it
};

struct ElevationHit {
  double range = 0.0, bearing = 0.0;  ///< slant range m, cone bearing rad
  double first_range = 0.0;           ///< range of the cluster's nearest cell, m; NaN: not known
};

/// Elevation of each detection. `backgrounds` is the map's mean noise power at each and
/// `noise_weight` the sum of squared aperture weights that made the map.
std::vector<ElevationEstimate> estimate_elevation(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows,
                                                  const std::vector<ElevationHit>& hits,
                                                  const std::vector<double>& backgrounds, const ElevationFrame& frame,
                                                  const ElevationOptions& options, double noise_weight, double range_cell,
                                                  Backend backend = Backend::Host, int device = 0);

/// A straight-line depth folded back into the water column and the path that put it where it was
/// ("direct", "surface", "bottom", "multiple", "unknown"). water_depth <= 0 or NaN: only the surface is known.
std::pair<double, std::string> fold_depth(double z, double water_depth, double surface_z = 0.0);

/// Depth at which an arrival seen at `elevation` (rad, up positive, level-referenced) at `receiver_depth`
/// started, back along the ray for the one-way travel time slant_range / c through a sound-speed profile
/// (linear between its knots; Snell's law in steps; turning points; mirrored at the surface and, with a
/// water depth > 0, at the bottom). The counterpart of elevation.ray_depth in Python.
double ray_depth(double receiver_depth, double elevation, double slant_range, const std::vector<double>& ssp_depth,
                 const std::vector<double>& ssp_speed, double water_depth = 0.0);

/// Whether a straight-line depth lies outside the water column by more than n_sigma sigma.
bool outside_water(double z, double sigma, double water_depth, double surface_z = 0.0, double n_sigma = 2.0,
                   double water_depth_sigma = 0.0);

/// What unresolved boundary echoes do to a depth from stacked rows (opt-in, elevation.multipath_sigma).
///
/// A target's echo comes back by every pair of paths (out by i, back by k): direct, surface, bottom. Echo (i, k) and its
/// mirror (k, i) have the same length exactly, so they add as pressures and the rows read the MEAN of the two paths'
/// elevations: direct with surface reads the sea surface, direct with bottom reads the sea bed, whatever the target's
/// depth. Echoes closer in range than the range resolution form one clump (one maximum along range). The elevation
/// stage takes the earliest clump within `first_arrival_db` of the strongest (expected power, echoes of different
/// length adding as power). What the rows read at an echo g of it is g's elevation, pulled by every other echo h by
///     s_g^2 = sum_h (a_h / a_g)^2 E2(x_gh) d_gh^2 / 2      (the mean square over their relative phase),
/// E2 the compressed pulse's power x range-resolution cells away (a flat band's sinc^2 with its lobes averaged:
/// 1 / (2 pi^2 x^2), at most 1) and d the difference of the two elevations, at most `aperture_resolution` / (2 pi)
/// (the most a second wave can turn the phase across the rows' height, lambda / (2 pi D) in sine).
///
/// How well a target depth z can give the reading: W(z) = exp(-(reading - elevation_g)^2 / 2 (sigma_angle^2 + s_g^2)),
/// by the echo of the clump that does it best. Which echo of a clump the rows read turns on phases nobody knows, so
/// every echo of it counts fully; and nothing is known of the target, so every depth the water allows is tried (rays
/// through the held profile, RayFan) and counts by W alone. The depth's variance is the mean of (z - reported depth)^2
/// under W. The same mean with the first path alone and resolved is what the angle's own sigma already holds; `extra`
/// is the difference in quadrature: zero where one resolved echo is all there is, range x s_g where the others are
/// weak, and the spread of the water column where the rows read a boundary. No depth with W above
/// exp(-explain_sigma^2 / 2): not `explained`, and nothing is claimed (the reading is not an echo of this picture).
/// Boundaries are perfect mirrors here (no loss is known to this stage); without a water depth there is no bottom.
struct MultipathLimit {
  bool explained = false;
  bool limited = false;          ///< the multipath term is the larger part of the depth sigma
  double depth_sigma = 0.0;      ///< m: the angle's own with the term
  double extra = 0.0;            ///< m: the term alone (in quadrature)
  std::int64_t echoes = 0;       ///< echoes in the clump read, at the depth that gives the reading best
  double depth_lo = 0.0, depth_hi = 0.0;   ///< m below the surface: the shallowest and the deepest depth that can give the reading (within explain_sigma)
  bool bottom = false;           ///< the profile has a bottom (else: surface reflections only, depths to max_depth)
  double deepest = 0.0;          ///< m: the deepest target depth considered
};
MultipathLimit multipath_depth_limit(const std::vector<double>& ssp_depth, const std::vector<double>& ssp_speed, double water_depth,
                                     double own_depth, double slant_range, double elevation, double sigma_angle, double reported_depth,
                                     double range_resolution, double aperture_resolution, bool first_arrival, double first_arrival_db,
                                     double explain_sigma, int max_bounces, double max_depth);

/// The members a detection's report line gains from it (JSON, no braces): `depth_unresolved_sigma_m` (the term, m; null:
/// the reading is no echo of this picture and nothing is claimed), `depth_multipath_limited`, `depth_unresolved_echoes`,
/// `depth_unresolved_span_m` (the shallowest and deepest depth that can give the reading) and, without a water depth,
/// `depth_unresolved_assumed` saying what was assumed in its place.
std::string multipath_limit_members(const MultipathLimit& limit);

struct PassiveElevation {
  double elevation = 0.0;  ///< rad, array frame, up positive
  double quality = 0.0;    ///< 0..1
};

/// Elevation of a noise source at a cone bearing: blocks of n_fft samples stepped by hop, from
/// block `first`, `n_blocks` of them (< 0: to the end), bins from f_low to f_high (Hz, absolute).
PassiveElevation passive_elevation(const PingRecord& record, const std::vector<std::vector<std::int64_t>>& rows, double bearing,
                                   double f_low, double f_high, std::int64_t n_fft, std::int64_t hop, std::int64_t first,
                                   std::int64_t n_blocks, double first_deg, double last_deg, double step_deg, bool starboard);

}  // namespace sonar::live
