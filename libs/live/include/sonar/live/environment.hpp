// What the vehicle knows about the water it is in, for the depth estimate of elevation.hpp:
//
//   water depth      from the Doppler log's altitude above the bottom plus the vehicle's own depth,
//                    with an uncertainty, and a rule for when bottom lock is lost (hold the last
//                    value, the uncertainty growing with time and distance, then give up)
//   sound speed      a ladder of what is known about the profile, best rung first:
//                      supplied  a profile given in the configuration (elevation.ssp)
//                      built     a profile the vehicle builds itself from its own (depth, sound
//                                speed) samples as it changes depth; for a static array a cast at
//                                deployment plays the same role
//                      family    an assumed shape (isothermal; mixed layer over a linear
//                                thermocline) hung on the sound speed measured at the vehicle
//                      local     only the sound speed at the vehicle is measured: straight ray
//                                (or the assumed gradient), gradient prior as the uncertainty
//                      prior     nothing measured: the same with the nominal sound speed
//                    The rung used goes into the report with every depth.
//
// Uncertainty of a depth from a profile. The profile's error is taken as fully correlated over
// depth (a profile that is wrong is wrong the same way all along the ray), anchored at the
// vehicle, where the sound speed is measured: c(z) +- sign(z - own depth) e(z), with
//     e(z) = min(sigma(z), gradient_sigma |z - own depth|)
// sigma(z) being what the rung claims and gradient_sigma the prior that holds when nothing is
// known. The depth sigma is the sensitivity of the traced depth to that tilt: half the difference
// of the depths through profiles tilted a twentieth of e(z) each way, times twenty. It is a
// linear measure and is not folded at the surface and bottom, so, like the prior's own rule, it
// can exceed the water depth: the depth is then not known. With sigma(z) infinite this is the
// prior alone, range^2 gradient_sigma / 2c for a near-horizontal ray, which is what the chain
// reports on the lower rungs.
//
// Everything a ping needs travels in the record (no schema change): the samples since the last
// ping and the uncertainties navigation sent go into metadata.extra_json under "nav", so a
// replayed recording builds the same profile as the live run did.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

// --- metadata.extra_json: a few flat values, without a JSON library ------------------------------

/// The first number after "key": in `extra`, or `fallback`.
double extra_number(const std::string& extra, const std::string& key, double fallback);
/// Rows of `width` numbers from "key": [[...], [...]] (empty when the key is not there).
std::vector<std::vector<double>> extra_rows(const std::string& extra, const std::string& key, std::size_t width);
/// Put "key": value (a JSON value without braces inside, or one flat object) at the top level,
/// replacing an earlier one.
void extra_set(std::string& extra, const std::string& key, const std::string& value);

// --- water depth from altitude -------------------------------------------------------------------

struct WaterDepthOptions {
  bool from_altitude = false;        ///< water depth = vehicle depth + altitude above the bottom (+ the log's offset)
  double dvl_offset_m = 0.0;         ///< the log's transducer below the navigation reference point, m (down positive)
  double altitude_sigma_m = 0.0;     ///< used when a fix carries no "altitude_sigma"
  double depth_sigma_m = 0.0;        ///< used when a fix carries no "depth_sigma"
  double hold_s = 60.0;              ///< without bottom lock the last value is held this long, then the depth is unknown
  double hold_sigma_m_per_s = 0.0;   ///< its sigma grows by this per second held
  double hold_sigma_m_per_m = 0.05;  ///< and by this per metre travelled (a bottom slope not ruled out; an assumption)
  double max_altitude_m = 0.0;       ///< an altitude above this is not believed (0: no limit)
};

struct WaterDepth {
  double depth = kNaN, sigma = kNaN;  ///< m; NaN: unknown
  bool held = false;                  ///< no bottom lock at this fix: the last value, held
  double held_s = 0.0;
};

class WaterDepthTracker {
 public:
  explicit WaterDepthTracker(const WaterDepthOptions& options = {}) : opt_(options) {}
  /// One fix: time, horizontal position, the vehicle's depth below the surface, and the log's
  /// altitude (NaN, <= 0 or lock false: no bottom lock). Sigmas NaN: the options' values.
  WaterDepth update(std::int64_t time_ns, double x, double y, double depth, double altitude, bool lock, double altitude_sigma = kNaN,
                    double depth_sigma = kNaN);

 private:
  WaterDepthOptions opt_;
  bool have_ = false;
  std::int64_t time_ns_ = 0;
  double x_ = 0.0, y_ = 0.0, depth_ = 0.0, sigma_ = 0.0;
};

/// What the navigation receiver (udp_source.hpp Navigation, live.py NavListener) is told.
struct NavOptions {
  WaterDepthOptions water_depth;
  double keep_s = 120.0;          ///< fixes are kept this long: a ping is processed after its window has closed,
                                  ///< and its attitude history must still be there (was 600 fixes, whatever their rate)
  double surface_z = 0.0;         ///< z of the sea surface in the navigation's frame: depth = z - surface_z
  double profile_step_m = 0.5;    ///< a (depth, sound speed) sample is kept for the profile every time the depth has changed by this
  double history_margin_s = 2.0;  ///< a ping's attitude history is the fixes from this long before its transmit to this long after its window
};

// --- the sound-speed ladder ----------------------------------------------------------------------

struct ProfileOptions {
  bool enabled = false;
  std::vector<std::string> ladder{"supplied", "built", "family", "local", "prior"};  ///< tried in this order
  // built
  double bin_m = 2.0;                ///< samples are binned by depth
  double sample_sigma = 0.0;         ///< m/s: the sound-speed sensor's accuracy. No default: the built rung needs it set
  double age_sigma_per_hour = 0.5;   ///< m/s per hour since a bin was last sampled (placeholder, unconfirmed)
  double distance_sigma_per_km = 0.5;  ///< m/s per km from where it was sampled (placeholder, unconfirmed)
  double max_age_s = 0.0, max_distance_m = 0.0;  ///< bins older or further than this are not used (0: no limit)
  double min_span_m = 5.0;           ///< the built rung is used once the sampled depths span this much
  /// Outside the sampled depths the profile continues with the assumed gradient (elevation gradient,
  /// else none) and its sigma grows from the edge bin's by this per metre (0: the gradient prior's sigma).
  double extrapolation_sigma_per_s = 0.0;
  std::string file;                  ///< the bins are kept here (read at start, written as they change): a restart keeps the profile
  std::string cast;                  ///< a cast to start from (static array): text lines "depth speed", m and m/s
  // supplied
  double supplied_sigma = 0.0;       ///< m/s: how well the supplied profile holds along the path (0: no claim, the prior)
  // family
  std::string family;                ///< "" none, "isothermal", "mixed_layer"
  double family_gradient = kNaN;     ///< 1/s: isothermal: the gradient; mixed_layer: the gradient in the layer and below the thermocline
  double layer_depth = kNaN;         ///< m: mixed_layer: the bottom of the mixed layer
  double thermocline_gradient = kNaN;  ///< 1/s (negative: slower with depth): mixed_layer
  double thermocline_thickness = kNaN; ///< m: mixed_layer
  double family_sigma = 0.0;         ///< m/s: how well the family holds (0: no claim, the prior)
  /// Ensemble (opt-in, elevation.profile.ensemble): where the vehicle has not sampled the water a single profile with a
  /// linear sigma understates what is not known (a thermocline is not a tilt). A small, deterministic set of profiles
  /// consistent with what is known is traced instead and the spread reported: the sampled part moved by its stated
  /// sigma, and the water outside it continued with each of `ensemble_gradients` (1/s, in place of the assumed
  /// gradient) for `ensemble_extent` metres, parallel to the rung's own profile beyond. No default for either: they
  /// say what water is thought possible here.
  bool ensemble = false;
  std::vector<double> ensemble_gradients;  ///< 1/s: the gradients the unsampled water may have (two or more spanning the range)
  double ensemble_extent = 0.0;            ///< m: how far beyond the sampled depths such a gradient may run
  bool ensemble_sigma_members = true;      ///< two more members: the rung's profile tilted by its sigma each way
};

struct LadderProfile {
  std::string rung = "prior";
  std::vector<double> depth, speed, sigma;  ///< knots, m and m/s; empty on the "local" and "prior" rungs
  double span_lo = kNaN, span_hi = kNaN;    ///< built: the depths sampled
  std::int64_t bins = 0;
};

/// The depths and speeds of an assumed family through (own_depth, sound_speed), surface to `bottom`.
void family_profile(const ProfileOptions& options, double own_depth, double sound_speed, double bottom, std::vector<double>& depth,
                    std::vector<double>& speed);

class SoundSpeedLadder {
 public:
  explicit SoundSpeedLadder(const ProfileOptions& options = {});
  /// One (depth, sound speed) sample of the vehicle's own sensor.
  void add(std::int64_t time_ns, double x, double y, double depth, double sound_speed);
  /// The samples a record carries (metadata.extra_json "nav": the fixes since the last ping, and the
  /// ping's own when its sound speed was measured). Returns whether the local sound speed is a measurement.
  bool absorb(const PingRecord& record, double surface_z = 0.0);
  /// What to trace rays through for a ping at (time, x, y, own_depth) with `sound_speed` at the vehicle.
  /// `supplied_*`: elevation.ssp. `gradient`: the assumed gradient outside what is known (0: none).
  LadderProfile evaluate(std::int64_t time_ns, double x, double y, double own_depth, double sound_speed, bool measured,
                         double water_depth, const std::vector<double>& supplied_depth, const std::vector<double>& supplied_speed,
                         double gradient, double gradient_sigma) const;
  std::int64_t bins() const { return static_cast<std::int64_t>(bins_.size()); }
  void save(const std::string& path) const;
  void load(const std::string& path, bool cast);

 private:
  struct Bin {
    std::int64_t index;      ///< floor(depth / bin_m)
    double depth, speed;     ///< running means
    std::int64_t count, time_ns;
    double x, y;
    bool fixed;              ///< from a cast: no age, no place
  };
  ProfileOptions opt_;
  std::vector<Bin> bins_;    ///< sorted by index
  std::int64_t unsaved_ = 0;
};

/// One sigma of a depth traced through a profile whose error is `sigma` at its knots (see the
/// top of this file): half the difference between the two tilted profiles' depths.
double profile_depth_sigma(double receiver_depth, double elevation, double slant_range, const std::vector<double>& depth,
                           const std::vector<double>& speed, const std::vector<double>& sigma, double gradient_sigma,
                           double water_depth = 0.0, bool arcs = false);
/// The profiles of the ensemble (ProfileOptions) for a rung's profile: common knots and each member's speeds. Member 0
/// is the rung's own profile. A rung that holds nothing (local, prior) is taken as the assumed gradient through
/// (own_depth, sound_speed), sampled nowhere. `gradient`: the assumed gradient the rung continued with.
void profile_ensemble(const ProfileOptions& options, const LadderProfile& ladder, double own_depth, double sound_speed, double gradient,
                      double gradient_sigma, double water_depth, std::vector<double>& knots, std::vector<std::vector<double>>& speeds);
/// The two tilted profiles themselves, on the knots they need (for the Python engine and tests).
void tilted_profiles(double receiver_depth, const std::vector<double>& depth, const std::vector<double>& speed, const std::vector<double>& sigma,
                     double gradient_sigma, std::vector<double>& knots, std::vector<double>& fast, std::vector<double>& slow);

}  // namespace sonar::live
