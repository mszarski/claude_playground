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
//   reverberation (opt-in per configuration: `reverb`; sonar-w2w3, D-198) the boundaries' backscatter at the target's
//               delay, from the SAME scattering strengths the simulator uses (gpusonar/sim/simulator.py
//               `reverberation_item_strength`, handed over as tables of grazing angle in x out) and the SAME rays
//               as the target: the bottom and the surface are cut into rings about the sonar; the paths from the
//               projector and to the array that reach a ring going towards its boundary (spreading, and the
//               coefficients of the boundaries met on the way, as powers) pair up, each pair adding
//               |a_tx|^2 |a_rx|^2 S(g_in, g_out) x ring area x absorption at its delay (sim/reverb.py
//               `reverberation_power`, the scattering-function method; Jackson and Richardson 2007 (G.4)). The
//               sum against delay goes through the compressed pulse as the target's arrivals do (|sinc|^2 of
//               the band, averaged over each delay bin) and is multiplied by the beam's integral over azimuth,
//               Psi = Int |b_rx b_tx|^2 dphi / |b_rx b_tx|^2 on its axis (J&R (G.12)), of an unweighted beam of
//               the array at broadside (`coverage_beam_integral`). Monostatic and direct paths only, this is
//               J&R (G.11)-(G.13) with the pulse length c tau / 2 replaced by the compressed one, c / (2 B): the
//               processing gain is the target's, so it cancels, and the array gain is not applied (the beam's
//               integral is). Signal excess is then over the power sum of noise and reverberation.
//               What is not in it: the reverberation's statistics (non-Rayleigh) in the detection
//               probability, a moving target's Doppler separation from it, beam sidelobes in elevation and the
//               elements' vertical patterns, range-dependent bottoms, volume scattering.
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
  bool bottom = false;                 ///< a bottom's Rayleigh reflection (a named sediment's, resolved by the configuration)
  double bottom_speed = 0.0, bottom_density = 0.0, bottom_attenuation = 0.0;
  /// m/s (0: a fluid), dB per wavelength: an elastic bottom (rock), raytrace.hpp rayleigh_reflection
  double bottom_shear_speed = 0.0, bottom_shear_attenuation = 0.0;
  /// m: the site's water depth, used on a ping that carries none (what the ping carries comes first). No default.
  /// A chain with the coverage picture enabled needs this and `bottom`: LiveChain refuses to start without them
  /// (coverage_site_missing). The class itself still takes a water without a bottom (tools: deep water, said by the caller).
  double water_depth = kNaN;
  std::int64_t ray_fan = 720, max_bounces = 3, max_arrivals = 6;
  double null_db = -10.0;              ///< a surface-image null: the factor below this where the direct path exists
  double surface_z = 0.0;
  /// Reverberation (file comment): 0 off (the noise-only picture, as before the term existed), 1 included, 2 not
  /// included because the configuration gives no scattering (the summary says so). gpusonar.coverage.cpp_options sets it.
  std::int64_t reverb = 0;
  std::vector<double> reverb_grazing;            ///< rad, increasing: the grid of both tables' axes
  std::vector<double> reverb_bottom_strength;    ///< linear per unit area, [g_in][g_out] row major on reverb_grazing; empty: none
  std::vector<double> reverb_surface_strength;   ///< the same for the sea surface
  double reverb_beam_integral = 0.0;   ///< rad: Psi; 0: from the record (Coverage::update), or CoverageSonar's
  std::int64_t reverb_rings = 600;     ///< rings of each boundary out to the farthest delay needed
  /// Boundary reflections a path to a ring may have, each way; 0: as many as a ray at 22 degrees of grazing makes out to
  /// the farthest ring, 0.4 x range / water depth, at least 3 and at most 40 (ours: steeper than about 20 degrees a sand
  /// bottom takes most of each bounce). Not `max_bounces` (the target's): in shallow water the reverberation at
  /// kilometres comes by many bounces (flank_patrol, 100 m deep: 3 give 8 to 14 dB too little at 2.2 to 3.6 km against
  /// the simulated data; tests/python/test_reverb_budget.py).
  std::int64_t reverb_max_bounces = 0;
  /// What the site's scattering is and how it was had, or why it is not included (gpusonar.coverage.site_reverberation:
  /// environment.boundary_scattering, the one rule). With reverb 1 the member carries it as "reverberation_note"; with
  /// reverb 2 it is the "reverberation" text itself (in place of the generic one). Empty: nothing said beyond the default.
  std::string reverb_note;
};

/// What a chain's coverage picture must be told about its site and was not: "" when the water depth and the bottom are
/// both given, else the words of the error (LiveChain's constructor throws them; gpusonar.coverage.chain_options says the same).
std::string coverage_site_missing(const CoverageOptions& o);

/// One option by name, as `coverage_<member>=value` (`coverage=1`: enabled; classes as "name:ts:lo:hi;name:ts:lo:hi").
bool set_coverage_option(CoverageOptions& o, const std::string& key, const std::string& value);

struct CoverageSonar {
  double f_centre = 0.0, bandwidth = 0.0, pulse = 0.0;   ///< Hz, Hz, s
  double beam_integral = 0.0;          ///< rad: Psi of the beam (CoverageOptions::reverb_beam_integral wins when given)
};

/// Psi (rad) of an unweighted beam of hydrophones at horizontal positions (x, y; m) steered to horizontal azimuth
/// `steer` (rad, from x towards y), at `frequency` (Hz) in water of `sound_speed`: Int_0^2pi |AF(phi)|^2 dphi / |AF(steer)|^2
/// with AF(phi) = sum_n exp(i k (x_n cos phi + y_n sin phi)), on `n` azimuths. One hydrophone: 2 pi.
double coverage_beam_integral(const std::vector<double>& x, const std::vector<double>& y, double frequency, double sound_speed, double steer,
                              std::int64_t n = 8192);
/// Psi of a record's receive array (every element's lever arm, projected on the body's horizontal plane) steered to the
/// first element's boresight (broadside of a line), at the band centre.
double coverage_beam_integral(const PingRecord& record);
/// Si(x) = Int_0^x sin(t)/t dt (a series below 4, Abramowitz and Stegun 5.2.38-39 above; absolute error < 1e-6).
double sine_integral(double x);
/// The compressed pulse's power response, |sinc(B u)|^2, averaged over u in [x - width/2, x + width/2] (s); width 0: its value.
double compressed_power_average(double x, double width, double bandwidth);
/// Reverberation level re the source level (dB; -inf: none) at the two-way `delays` (s), by the scattering-function
/// method of the file comment, with the beam's integral `psi` (rad). Without scattering tables: every value -inf.
std::vector<double> coverage_reverberation_db(const MultipathProfile& water, double water_depth, double array_depth, double projector_depth,
                                              const std::vector<double>& delays, const CoverageOptions& options, const CoverageSonar& sonar, double psi);

/// The two-way propagation factor on a grid of target depths x horizontal ranges, row major [depth][range].
struct CoverageField {
  std::vector<double> factor_db;       ///< the coverage statistic (file comment), dB re the free-field echo; -inf: no path
  std::vector<double> coherent_db;     ///< one way, array leg, every arrival as a pressure at the band centre, dB re spherical
  std::vector<double> incoherent_db;   ///< one way, array leg, sum of powers, dB re spherical
  std::vector<char> direct;            ///< a path that touches no boundary joins both the projector and the array to the point
  std::vector<double> slant;           ///< m: sqrt(R_tx R_rx), straight
  std::vector<double> delay;           ///< s: the two-way delay of the factor's peak (NaN: no path)
  std::vector<double> reverb_db;       ///< reverberation level re the source level at that delay, dB (-inf: none or not included)
};
CoverageField coverage_field(const MultipathProfile& water, double water_depth, double array_depth, double projector_depth,
                             const std::vector<double>& ranges, const std::vector<double>& depths, const CoverageOptions& options,
                             const CoverageSonar& sonar);
/// One row's link-budget SNR (dB) of a target at straight slant range `slant` with the two-way factor `factor_db`.
double coverage_snr_db(const CoverageOptions& options, const CoverageSonar& sonar, double slant, double ts_db, double factor_db);
/// The same over the power sum of noise and reverberation at `reverb_db` (re the source level). With the term off
/// (`reverb` != 1) or no reverberation (-inf, NaN) it is the noise-only figure above, exactly.
double coverage_snr_db(const CoverageOptions& options, const CoverageSonar& sonar, double slant, double ts_db, double factor_db, double reverb_db);
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
  double seconds_ = 0.0, own_ = 0.0, water_ = 0.0, psi_ = 0.0;
  std::string rung_, json_;
  std::vector<double> depth_, speed_, ranges_;
  std::vector<std::vector<double>> pd_;   ///< [class][range]
};

}  // namespace sonar::live
