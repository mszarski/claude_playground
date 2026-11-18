// The coverage picture: what the sonar can see from where it is, in the water it is in.
//
// For each configured target class at the vehicle's present depth: the probability of detecting it
// against range, the greatest range at which that probability holds a stated value, the range and
// depth bands nearer than that where it does not (shadows, surface-image nulls), where the direct
// path ends, and the own depth within the platform's limit and the bottom clearance that would reach
// furthest, with what it would gain. Both chain engines call this one class (opt-in, chain.coverage),
// so their report lines carry the same member; tools/coverage_map.py draws the same numbers.
//
// The numbers, and where each comes from:
//   loss        rays through the profile the sound-speed ladder holds this ping (raytrace.hpp), both
//               legs. At each target position the two-way arrivals are put through the compressed pulse
//               (a sinc of the band): the direct and the once-surface-reflected path of each leg add as
//               pressures (their interference is the Lloyd mirror, deterministic at these grazing angles);
//               everything that touched the bottom or the surface twice, and the part of a surface
//               reflection that roughness scatters out of the coherent path, adds as power. The peak of
//               that over delay, times the direct path's spreading^2, is the two-way factor on the
//               free-field echo (gpusonar/lloyd.py's `peak_db`, which a test holds it to in isovelocity
//               water; Bellhop is the oracle in refracting water).
//   levels      SNR = SL - 2 TL + TS - (N0 + 10 log B) + 10 log (B T) + AG + factor, TL spherical with
//               absorption: the link budget of python/gpusonar/buildspec.py (`budget`), whose numbers the
//               configuration carries (none has a default here: without them the summary says so).
//   detection   the probability that the chain as configured reports a point echo of that SNR: a table
//               made by python/gpusonar/live.py `detection_probability` (the detector's own plan and
//               thresholds) and handed over; not a typed threshold.
//   uncertainty the rung of the ladder the profile came from, and (with elevation.profile.ensemble) the
//               span of the greatest detection range over the ensemble's profiles.
//
// "Could I have seen it": `pd` of each class at the listed ranges is the probability of detecting a
// target of that class at that range at any bearing (worst over the class's depths); Coverage::seen
// interpolates it. An empty sector at a range where it is near one is empty; where it is near zero the
// sonar was blind there.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"
#include "sonar/live/environment.hpp"
#include "sonar/live/multipath_depth.hpp"

namespace sonar::live {

/// The water this ping's rays go through, from what the chain holds: the ladder's rung (`ladder` non-null), else
/// elevation.ssp, else the assumed gradient through the vehicle's sound speed, else isovelocity water; with the sigma
/// the rung claims and, with ProfileOptions::ensemble, the ensemble's members on common knots. One function for both
/// chain engines, the multipath depth, the elevation depth's ensemble and the coverage picture.
MultipathProfile water_profile(const ProfileOptions& options, const LadderProfile* ladder, const std::vector<double>& ssp_depth,
                               const std::vector<double>& ssp_speed, bool use_gradient, double gradient, double gradient_sigma, double own_depth,
                               double sound_speed, double water_depth);

struct CoverageClass {
  std::string name;
  double ts_db = 0.0;                  ///< target strength, dB
  double depth_lo = 0.0, depth_hi = 0.0;   ///< m: the depths the class may be at
};

struct CoverageOptions {
  bool enabled = false;
  std::vector<CoverageClass> classes;
  // The link budget (python/gpusonar/buildspec.py `budget`). No defaults.
  double source_level_db = kNaN;       ///< dB re 1 uPa at 1 m
  double noise_spectrum_db = kNaN;     ///< dB re 1 uPa^2 / Hz at a hydrophone, at the band centre
  double array_gain_db = kNaN;         ///< one row's gain against that noise, dB
  double absorption_db_per_km = kNaN;
  double pd = 0.5;                     ///< the detection probability the ranges are stated for
  std::vector<double> pd_snr_db, pd_value;   ///< detection probability against one row's link-budget SNR (live.detection_probability)
  double range_first = 0.0, range_last = 0.0;   ///< m; 0: the record's receive window
  std::int64_t n_ranges = 48, n_depths = 5, n_report = 12;
  bool recommend = true;               ///< also try other own depths
  double own_depth_min = 5.0, own_depth_max = 500.0, bottom_clearance = 10.0;   ///< m: where the platform may go
  std::int64_t n_own = 10;
  /// Refreshed when the own depth, the water depth or the profile's speed anywhere has changed by this much (m, m, m/s),
  /// or the rung has; otherwise the last summary is repeated.
  double refresh_depth = 5.0, refresh_water_depth = 10.0, refresh_speed = 0.5;
  double surface_coefficient = -1.0, surface_rms_height = 0.0, surface_loss_db = 0.0;
  std::vector<double> surface_loss_grazing, surface_loss_table_db;   ///< a bubble law's loss against grazing angle (rad, dB)
  bool bottom = false;                 ///< a fluid bottom's Rayleigh reflection (a named sediment's, resolved by the configuration)
  double bottom_speed = 0.0, bottom_density = 0.0, bottom_attenuation = 0.0;
  /// m: the site's water depth, used on a ping that carries none (what the ping carries comes first). No default.
  /// A chain with the coverage picture enabled needs this and `bottom`: LiveChain refuses to start without them
  /// (coverage_site_missing). The class itself still takes a water without a bottom (tools: deep water, said by the caller).
  double water_depth = kNaN;
  std::int64_t ray_fan = 720, max_bounces = 3, max_arrivals = 6;
  double null_db = -10.0;              ///< a surface-image null: the factor below this where the direct path exists
  double surface_z = 0.0;
};

/// What a chain's coverage picture must be told about its site and was not: "" when the water depth and the bottom are
/// both given, else the words of the error (LiveChain's constructor throws them; gpusonar.coverage.chain_options says the same).
std::string coverage_site_missing(const CoverageOptions& o);

/// One option by name, as `coverage_<member>=value` (`coverage=1`: enabled; classes as "name:ts:lo:hi;name:ts:lo:hi").
bool set_coverage_option(CoverageOptions& o, const std::string& key, const std::string& value);

struct CoverageSonar {
  double f_centre = 0.0, bandwidth = 0.0, pulse = 0.0;   ///< Hz, Hz, s
};

/// The two-way propagation factor on a grid of target depths x horizontal ranges, row major [depth][range].
struct CoverageField {
  std::vector<double> factor_db;       ///< the coverage statistic (file comment), dB re the free-field echo; -inf: no path
  std::vector<double> coherent_db;     ///< one way, array leg, every arrival as a pressure at the band centre, dB re spherical
  std::vector<double> incoherent_db;   ///< one way, array leg, sum of powers, dB re spherical
  std::vector<char> direct;            ///< a path that touches no boundary joins both the projector and the array to the point
  std::vector<double> slant;           ///< m: sqrt(R_tx R_rx), straight
};
CoverageField coverage_field(const MultipathProfile& water, double water_depth, double array_depth, double projector_depth,
                             const std::vector<double>& ranges, const std::vector<double>& depths, const CoverageOptions& options,
                             const CoverageSonar& sonar);
/// One row's link-budget SNR (dB) of a target at straight slant range `slant` with the two-way factor `factor_db`.
double coverage_snr_db(const CoverageOptions& options, const CoverageSonar& sonar, double slant, double ts_db, double factor_db);
/// The detection probability at that SNR, from the table (NaN without one).
double coverage_pd(const CoverageOptions& options, double snr_db);

class Coverage {
 public:
  explicit Coverage(const CoverageOptions& options = {}) : opt_(options) {}
  void set_pd_table(const std::vector<double>& snr_db, const std::vector<double>& pd) {
    opt_.pd_snr_db = snr_db;
    opt_.pd_value = pd;
    have_ = false;
  }
  /// The report member (`"coverage": {...}`) for this ping: recomputed when something changed materially, else repeated.
  const std::string& update(const PingRecord& record, const Vec3d& origin, double water_depth, const MultipathProfile& water);
  /// The same from the numbers themselves (tools, tests).
  const std::string& update(const CoverageSonar& sonar, double range_first, double range_last, double array_depth, double projector_depth,
                            double water_depth, const MultipathProfile& water);
  bool refreshed() const { return refreshed_; }
  double seconds() const { return seconds_; }   ///< the last refresh
  /// Detection probability of class `i` at horizontal range `range` at the present depth (worst over its depths).
  double seen(std::size_t i, double range) const;
  const CoverageOptions& options() const { return opt_; }

 private:
  CoverageOptions opt_;
  bool have_ = false, refreshed_ = false;
  double seconds_ = 0.0, own_ = 0.0, water_ = 0.0;
  std::string rung_, json_;
  std::vector<double> depth_, speed_, ranges_;
  std::vector<std::vector<double>> pd_;   ///< [class][range]
};

}  // namespace sonar::live
