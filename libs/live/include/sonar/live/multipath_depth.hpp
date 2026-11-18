// Depth of a target from the timing of its own multipath: one row of hydrophones is enough.
//
// A submerged target's echo arrives several times: out and back along the direct path, and with a
// reflection at the sea surface on the way out, on the way back, or both (in shallow water also off
// the bottom). For flat boundaries and straight rays, with the projector at depth z_p, the array at
// z_a, the target at z_t a horizontal distance r away:
//
//   one way    tau(z) = (sqrt(r^2 + (z_t + z)^2) - sqrt(r^2 + (z_t - z)^2)) / c  ~  2 z z_t / (r c)
//   two way    delay 0                     amplitude 1                direct out, direct back
//              tau(z_a)                    mu                         direct out, surface back
//              tau(z_p)                    mu                         surface out, direct back
//              tau(z_p) + tau(z_a)         mu^2                       surface both ways
//              mu = -exp(-Gamma^2 / 2) 10^(-L / 20): the surface inverts the echo
//
// so z_t follows from the delays, the range and the sonar's own depth, without the elevation angle
// and (to first order) without the sound-speed profile. For every candidate depth the whole pattern
// (delays, amplitudes, signs, carrier phases) is correlated with the compressed beam output y around
// the detection (baseband, band B about fc, noise power sigma^2):
//
//   T(z, t0) = | sum_k conj(a_k) exp(+i 2 pi fc tau_k) y(t0 + tau_k) |^2
//              / sum_jk a_j conj(a_k) exp(-i 2 pi fc (tau_j - tau_k)) sinc(B (tau_j - tau_k))
//   L(z)     = max_t0 T / sigma^2          L_1 = max_t |y|^2 / sigma^2 (one arrival)
//
// the generalised likelihood ratio for "this pattern, amplitude and time unknown". Resolved arrivals
// ("delay") and overlapping ones (the Lloyd-mirror ripple of the echo's spectrum, period 1 / tau:
// "ripple") are the same statistic. A depth is reported when L(z) - L_1 >= min_gain. Within a ping
// the carrier fringes of L(z) give the fine depth; along a track the fringe envelope is added up over
// pings. Sigma is the spread of exp(L) over depth. python/gpusonar/multipath_depth.py is the NumPy
// reference of everything here and says when the method cannot work; docs/PHYSICS_NOTES.md derives it.
//
// The chain is asked for one thing only: the compressed complex beam samples around a detection
// (`beam_samples`, chain.hpp). A detector that merges a target's arrivals into one detection says where
// they lie (first_range, extent) and the fit is centred there; when the arrivals are so far apart that
// they come as two detections, the window about either holds them all (it reaches the largest spread
// searched either way), and each gets the same depth.
#pragma once

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sonar/core/record.hpp"

namespace sonar::live {

struct MultipathOptions {
  bool enabled = false;
  double max_depth = 500.0;            ///< m: deepest target depth searched (less when the water is shallower)
  double min_delay_bandwidth = 3.0;    ///< shallowest depth searched: where (tau_tx + tau_rx) x bandwidth is this
  double window_s = 0.25;              ///< longest two-way multipath spread searched
  double range_slack = 4.0;            ///< m: the detection's range may be this far from any one of the target's arrivals
  std::int64_t oversample = 8;         ///< samples of the beam window per 1 / bandwidth
  std::int64_t fringe_points = 8;      ///< depths tried per carrier fringe
  std::int64_t noise_cells = 128;      ///< 1 / bandwidth cells ahead of the earliest possible arrival that give the noise power
  double min_gain = 12.0;              ///< a depth needs L(z) - L_1 of at least this
  double ambiguity_drop = 3.0;         ///< a rival depth within this of the best log-likelihood: ambiguous
  std::int64_t max_detections = 16;    ///< strongest detections examined per ping
  double min_snr_db = -1e300;          ///< detections weaker than this are not examined
  double surface_coefficient = -1.0;   ///< flat-surface pressure reflection coefficient
  double surface_rms_height = 0.0;     ///< m: coherent reflection exp(-Gamma^2 / 2), Gamma = 2 k h sin(grazing)
  double surface_loss_db = 0.0;        ///< further loss per surface bounce (bubbles)
  double surface_min_amplitude = 0.25; ///< a surface path weaker than this re the direct one: no_surface_path
  bool bottom = false;                 ///< a fluid bottom's Rayleigh reflection (with the water depth): the next three
  double bottom_speed = 0.0, bottom_density = 0.0, bottom_attenuation = 0.0;  ///< m/s, re water, dB per wavelength
  std::int64_t bottom_order = 1;       ///< boundary reflections per leg with a bottom (1: d, s, b; 2: also sb, bs)
  double water_depth = 0.0;            ///< m; 0: the record's environment.water_depth (navigation's, else unknown)
  double surface_z = 0.0;              ///< z of the sea surface in the navigation's frame
  double own_depth_sigma = 0.0;        ///< m: the navigation's depth accuracy
  double model_sigma_fraction = 0.0;   ///< relative error of the flat, straight-ray delays: sigma gains depth x this
  double track_gate = 60.0;            ///< m: a detection within this of a track's position is that track's
  double track_step = 0.5;             ///< m: depth step of a track's accumulated likelihood
  double track_min_gain = 12.0;        ///< the accumulated gain a track's depth needs
  double track_memory_pings = 0.0;     ///< 0: all pings count alike; n: older ones fade with exp(-1 / n) per ping
  double consistency_sigma = 3.0;      ///< with a stacked-row depth: combined when they agree within this many sigma
};

/// One option by name, as `multipath_depth_<member>=value` (and `multipath_depth=1`: enabled). False: not one of these.
bool set_multipath_option(MultipathOptions& o, const std::string& key, const std::string& value);

struct MultipathGeometry {
  double range = 0.0;                  ///< slant range of the detection: half its two-way path, m
  double array_depth = 0.0, projector_depth = 0.0;   ///< m below the surface
  double sound_speed = 1500.0, f_centre = 0.0, bandwidth = 0.0;
  double water_depth = 0.0;            ///< m; 0: unknown
  double slack = 0.0;                  ///< m: how far `range` may be from any one arrival (0: the options' range_slack)
};

struct MultipathEstimate {
  bool valid = false;
  double depth = kNaN, sigma = kNaN;   ///< m below the surface
  double gain = kNaN;                  ///< L(depth) - L_1
  std::int64_t arrivals = 0;           ///< arrivals of the pattern at that depth (a track: pings added up)
  std::string method = "none";         ///< "delay", "ripple", "track", "none"
  bool ambiguous = false;
  std::vector<double> alternatives;    ///< rival depths, m
  std::string reason;                  ///< why there is no depth: merged, no_surface_path, single_arrival, window
  double snr_db = kNaN;                ///< 10 log10 L_1
  std::vector<double> grid, envelope;  ///< the fringe envelope L(z) - L_1, for a track to add up
};

/// The two-way arrivals of a target at depth z whose direct echo has two-way time t0: delays after
/// the direct one (s) and amplitudes re the direct one. Arrival 0 is direct-direct.
void multipath_pattern(double z, double t0, const MultipathGeometry& g, const MultipathOptions& o, std::vector<double>& delay,
                       std::vector<std::complex<double>>& amp);
/// tau(z_p) + tau(z_a) at the detection's range, s.
double multipath_surface_delay(double z, const MultipathGeometry& g);
/// Depths searched, `fringe_points` per carrier fringe (empty: the arrivals merge at every depth).
std::vector<double> multipath_depth_grid(const MultipathGeometry& g, const MultipathOptions& o);
/// Two-way times re the detection's own of the window's samples: first, step, count, and how many
/// leading samples hold only noise, for a largest spread `spread` (s).
void multipath_window_times(const MultipathGeometry& g, const MultipathOptions& o, double spread, double& t_first, double& dt,
                            std::int64_t& n, std::int64_t& n_noise);
/// Compressed output of the array steered at cone `bearing` (rad from broadside, the map's convention) at two-way times
/// t_first + k dt about the two-way time of slant range `range`, as baseband samples (an arrival delayed by tau carries
/// the phase -2 pi fc tau). These are the chain's own `beam_samples` (chain.hpp), unweighted; with `rows` (stacked
/// rows) the rows' samples are added, which holds for arrivals within a few degrees of the horizontal. The record is conditioned.
std::vector<std::complex<double>> multipath_window(const PingRecord& record, double bearing, double range, double t_first, double dt, std::int64_t n,
                                                   const std::vector<std::vector<std::int64_t>>& rows = {}, Backend backend = Backend::Host,
                                                   int device = 0);
/// Target depth from one beam window.
MultipathEstimate multipath_estimate(const std::complex<double>* y, std::int64_t n, double t_first, double dt, std::int64_t n_noise,
                                     const MultipathGeometry& g, const MultipathOptions& o);
/// The fit's geometry for a detection at slant range `range` from `origin`. `first_range`, `extent` (m; NaN: not known):
/// where the detection's arrivals lie, as a merging detector reports them; the fit is then centred on that stretch and
/// any sample of it may be the strongest arrival.
MultipathGeometry multipath_geometry(const PingRecord& record, const Vec3d& origin, double range, const MultipathOptions& o,
                                     double first_range = kNaN, double extent = kNaN);
/// Target depth for one detection of a conditioned record.
MultipathEstimate multipath_detection(const PingRecord& record, const Vec3d& origin, double bearing, double range, const MultipathOptions& o,
                                      double first_range = kNaN, double extent = kNaN, const std::vector<std::vector<std::int64_t>>& rows = {},
                                      Backend backend = Backend::Host, int device = 0);

struct MultipathFused {
  double depth = kNaN, sigma = kNaN;
  bool consistent = false;
  std::int64_t selected = 0;           ///< index into [the reported depth, the alternatives...]
};
/// The multipath depth with a depth from the rows' elevation and the depths of its other candidates:
/// the candidate nearest the multipath depth is taken; within `consistency_sigma` the two are combined
/// by inverse variance, else the multipath depth stands and `consistent` is false.
MultipathFused multipath_fuse(double depth_mp, double sigma_mp, double depth_el, double sigma_el, const std::vector<double>& alternatives,
                              double consistency_sigma);

/// What the chain hands over for each detection and each track of a ping.
struct MultipathHit {
  double range = 0.0;                  ///< slant range, m
  double bearing = 0.0;                ///< cone bearing, rad from broadside (the map's)
  double first_range = kNaN, extent = kNaN;   ///< m: the stretch of range its arrivals span (a merging detector); NaN: not known
  double x = 0.0, y = 0.0;             ///< where the tracker has it (its own frame)
  double snr_db = 0.0;
  bool has_depth = false;              ///< a depth from the rows' elevation, m below the surface, with its sigma
  double depth = 0.0, depth_sigma = 0.0;
  std::vector<double> depth_alternatives;   ///< depths (below the surface, folded into the water) of the other candidate elevations
};
struct MultipathTrack {
  std::int64_t id = 0;
  double x = 0.0, y = 0.0;             ///< the tracker's frame
  bool has_depth = false;
  double depth = 0.0, depth_sigma = 0.0;
};
struct MultipathReport {
  std::vector<std::string> detections, tracks;   ///< JSON members to add to each (no braces; empty: nothing)
  double seconds = 0.0;
};

/// The stage: every ping's detections, and the tracks' accumulated depths. Both chain engines call
/// this one class, so their reports carry the same fields.
class MultipathDepth {
 public:
  explicit MultipathDepth(const MultipathOptions& options = {}) : opt_(options) {}
  MultipathReport process(const PingRecord& record, const Vec3d& origin, const std::vector<MultipathHit>& hits,
                          const std::vector<MultipathTrack>& tracks, const std::vector<std::vector<std::int64_t>>& rows = {},
                          Backend backend = Backend::Host, int device = 0);
  /// One ping's envelope into a track's sum, and what the sum says now (for tests and the Python reference).
  void add(std::int64_t track, const MultipathEstimate& estimate, const MultipathGeometry& g);
  MultipathEstimate track(std::int64_t id) const;
  const MultipathOptions& options() const { return opt_; }

 private:
  struct Sum {
    std::vector<double> total;
    std::int64_t pings = 0, last_ping = 0;
    double lobe = 0.0, model = 0.0;
  };
  MultipathOptions opt_;
  std::map<std::int64_t, Sum> sums_;
  std::int64_t pings_ = 0;
};

}  // namespace sonar::live
