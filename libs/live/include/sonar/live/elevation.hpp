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
#include <string>
#include <utility>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"

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
  double row_tolerance_m = 0.02;   ///< elements within this of one line are one row
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
