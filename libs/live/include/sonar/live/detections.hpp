// Detections from a CFAR map, the tracker that follows them and the classes given to its
// tracks. The C++ counterparts of python/gpusonar/detect.py (cluster, Tracker) and
// classify.py; tests/python/test_live_cpp.py checks they give the same answers.
// Sources: docs/REFERENCES.md Sec. 12.3 (constant-velocity Kalman filter; the association is ours, D-141).
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace sonar::live {

/// One local peak of the map along range inside a detection (the multiscale detector): an arrival of its echo.
struct Arrival {
  double range = 0.0;    ///< m, of the map sample that holds the peak
  double snr_db = 0.0;   ///< that sample over the detector's background of its cell
};

struct Detection {
  double range = 0.0;    ///< m
  double bearing = 0.0;  ///< rad
  double snr_db = 0.0;   ///< strongest cell over its local background
  std::int64_t cells = 1;
  double depth = std::numeric_limits<double>::quiet_NaN();        ///< m, straight-line depth from elevation (stacked rows); NaN: none
  double depth_sigma = std::numeric_limits<double>::quiet_NaN();
  double first_range = std::numeric_limits<double>::quiet_NaN();  ///< m, the nearest cell of the cluster (its leading edge)
  /// The elevation was ambiguous: `depth` is one candidate and these are the others. A track that
  /// has a depth takes the one that fits it; a track without one waits.
  bool depth_ambiguous = false;
  std::vector<double> depth_alternatives{};
  double extent = std::numeric_limits<double>::quiet_NaN();       ///< m in range the cluster spans (the multiscale detector); NaN: not reported
  /// m, the energy-weighted mean range of the cluster's samples (the multiscale detector), whichever `range` reports
  /// (RangeEstimate). The elevation search, SBL's range gates, multipath_depth, blanking and the reference-burst filter
  /// go by it (D-158). NaN: `range` is the centroid.
  double centroid_range = std::numeric_limits<double>::quiet_NaN();
  /// The detection's arrivals, nearest first (the multiscale detector): the local peaks along range of the strongest
  /// sample over its bearings, each above the single-sample threshold; at most 8, the strongest. One detection is one
  /// target: its direct echo, its surface and bottom echoes and its highlights within the merge distance are all here,
  /// from `first_range` over `extent`. The complex samples behind them: `beam_samples` (chain.hpp).
  std::vector<Arrival> arrivals{};
  /// s, on the tracker's clock: when the detection shows the target. For a monostatic echo the ping's transmit time +
  /// range / c, with the c that turned the two-way time into range (D-156). NaN (or any value that is not finite): the
  /// ping's time given to Tracker::update.
  double time = std::numeric_limits<double>::quiet_NaN();
  /// m/s, the range rate of the Doppler bank's hypothesis that won at the detection's strongest sample (relative to the
  /// platform, opening positive; doppler.hpp), and the bank's step, its resolution: a COARSE estimate. NaN: no bank.
  double range_rate = std::numeric_limits<double>::quiet_NaN();
  double range_rate_step = std::numeric_limits<double>::quiet_NaN();
  /// m: this detection's own range sigma for the tracker (a tone's coarse range, doppler.hpp ToneOptions::range_sigma);
  /// NaN: the tracker's range_sigma (every detection before D3).
  double range_sigma = std::numeric_limits<double>::quiet_NaN();
  /// Which Doppler path found it (D3): 0 the sweep's map (or no Doppler path), 1 the tone's bank alone, 2 both (a sweep
  /// detection that took a tone detection's range rate).
  int doppler_path = 0;
  /// D5 (opt-in, TrackerOptions::range_rate): the tracker takes `range_rate` as a measurement of the track's velocity with
  /// this one-sigma (m/s), where the platform (the array in the tracker's frame at the ping: across, forward, m, and their
  /// rates over ground, m/s) is finite. NaN: not used (no rate, or one not trusted: the sweep bank's, D-232). detect.py the same.
  double range_rate_sigma = std::numeric_limits<double>::quiet_NaN();
  double platform_x = std::numeric_limits<double>::quiet_NaN();
  double platform_y = std::numeric_limits<double>::quiet_NaN();
  double platform_vx = std::numeric_limits<double>::quiet_NaN();
  double platform_vy = std::numeric_limits<double>::quiet_NaN();
};

/// Group touching detection cells (8-connected) of a map [n_rows (bearing), n_cols (range)]
/// and report one detection per group at its power-weighted centroid. `ratio` is the
/// power over the local background (the CFAR kernel's second output).
std::vector<Detection> cluster(const unsigned char* detections, const float* power, const float* ratio,
                               std::int64_t n_rows, std::int64_t n_cols, const std::vector<double>& ranges,
                               const std::vector<double>& bearings);

struct HistoryPoint {
  double time, x, y;  ///< s; across and forward, m: the filtered position (the track's state) at that time
  /// The position of the detection the track took on this ping, in the same frame (detect.py Track.measured); NaN on a
  /// ping it missed. classify_tracks fits these (D-157). A history whose rows all lack one (built by hand from
  /// positions) is fitted through x, y as before.
  double mx = std::numeric_limits<double>::quiet_NaN(), my = std::numeric_limits<double>::quiet_NaN();
  /// s: the time of the ping this row was added on (the time given to Tracker::update, the transmit time), where `time`
  /// is the detection's own, transmit + R/c (D-156; detect.py Track.pings). classify_tracks refuses a velocity from echo
  /// delays alone with it (sonar-054). NaN: none recorded (a history built by hand), not tested.
  double ping = std::numeric_limits<double>::quiet_NaN();
};

struct Track {
  std::int64_t id = 0;
  double state[4] = {0, 0, 0, 0};  ///< across, forward, and their rates
  double cov[16] = {};             ///< row major
  std::int64_t hits = 1;
  std::int64_t misses = 0;
  bool confirmed = false;
  std::vector<HistoryPoint> history;
  std::vector<double> snr;  ///< SNR (dB) of each detection it took
  std::vector<std::int64_t> cells;  ///< flagged map cells of each detection it took (Detection::cells)
  double depth = std::numeric_limits<double>::quiet_NaN();  ///< m, from the detections that carried one
  double depth_var = std::numeric_limits<double>::infinity();
  /// s: when `state` holds: its last detection's time (Detection::time), or the ping's after a miss.
  double time = std::numeric_limits<double>::quiet_NaN();
  std::int64_t rate_updates = 0;  ///< D5: range-rate measurements the track took (0 with TrackerOptions::range_rate Off)
};

/// How detections are shared out among tracks on each ping (D-141; python/gpusonar/detect.py).
/// Confirmed tracks always choose before tentative ones. Likelihood: the gated (track,
/// detection) pairs most probable first, by d^2 + ln|S|, each track and detection used once;
/// ties go to the earlier track, then the earlier detection. InOrder: the rule before D-141,
/// each track in turn takes its nearest gated detection by d^2.
enum class Association { Likelihood, InOrder };
/// "likelihood" or "in_order"; throws std::invalid_argument for anything else.
Association parse_association(const std::string& name);

/// D5 (opt-in): Off, as before; Update: after its position update a track takes a detection's range rate (when
/// Detection::range_rate_sigma and the platform are finite) as a second, scalar extended Kalman update; UpdateAndGate:
/// also a pair whose rate lies gate sigmas or more from the track's predicted rate is no candidate. detect.py the same.
enum class RangeRate { Off, Update, UpdateAndGate };
/// "off", "update" or "update_and_gate"; throws std::invalid_argument for anything else.
RangeRate parse_range_rate(const std::string& name);
const char* range_rate_name(RangeRate mode);

struct TrackerOptions {
  /// m/s^2: standard deviation of the white acceleration noise on each horizontal axis
  /// (piecewise-constant over a step: Q = G G^T process_noise^2, G = [dt^2/2, dt] per axis).
  double process_noise = 0.05;
  double range_sigma = 2.0;                   ///< m
  double bearing_sigma = 0.017453292519943295;  ///< rad (1 deg)
  /// Association gate in sigmas of the innovation (Mahalanobis distance, 2-D): a detection
  /// is a candidate for a track when d^2 < gate^2.
  double gate = 4.0;
  std::int64_t confirm_hits = 3;  ///< detections (the first included) that confirm a track
  std::int64_t max_misses = 3;    ///< a track is dropped after more pings than this in a row without a detection
  /// m/s: standard deviation of each horizontal velocity component of a new track (it starts at 0).
  double initial_speed_sigma = 10.0;
  double depth_process_noise = 0.5;  ///< m per sqrt(s): how fast a track's depth may wander
  Association association = Association::Likelihood;
  RangeRate range_rate = RangeRate::Off;  ///< D5, opt-in
};

/// The tracker's opt-in tap (Tracker::set_tap; python/gpusonar/detect.py TRACKER_TAP, the same members): what one
/// update did. Read only: the tracks are bit-identical with the tap on or off.
struct TrackerTapTrack {
  std::int64_t id = 0;
  double state[4] = {0, 0, 0, 0};
  double cov[16] = {};
  std::int64_t hits = 0, misses = 0;
  bool confirmed = false;
  double time = std::numeric_limits<double>::quiet_NaN();
};
struct TrackerTapPair {
  std::int64_t track = 0;      ///< id
  std::int64_t detection = 0;  ///< index into the update's detections
  double d2 = 0.0;             ///< squared Mahalanobis distance of the innovation
  double cost = std::numeric_limits<double>::quiet_NaN();  ///< d2 + ln|S| (Likelihood); NaN for InOrder
  bool gated = false;          ///< d2 < gate^2
  bool nearest = false;        ///< the track's smallest d2 (the first on a tie), kept even outside the gate
  double x = 0.0, y = 0.0;     ///< the track predicted to the detection's time (across, forward, m)
  double sxx = 0.0, sxy = 0.0, syy = 0.0;  ///< the innovation covariance there
  double rate_d2 = std::numeric_limits<double>::quiet_NaN();  ///< D5 UpdateAndGate: the rate's y^2 / S; NaN: not used
};
/// D5: one range-rate update (TrackerOptions::range_rate not Off; detect.py TRACKER_TAP `rate`).
struct TrackerTapRate {
  std::int64_t track = 0, detection = 0;  ///< id; index into the update's detections
  double z = 0.0, predicted = 0.0, innovation = 0.0, s = 0.0, sigma = 0.0;  ///< m/s; s in (m/s)^2
};
struct TrackerTap {
  double time = std::numeric_limits<double>::quiet_NaN();
  std::vector<TrackerTapTrack> before;   ///< the tracks entering the update (id, hits, misses, confirmed)
  std::vector<TrackerTapPair> pairs;     ///< gated pairs and each track's nearest, in the order compared
  std::vector<std::pair<std::int64_t, std::int64_t>> took;  ///< (track id, detection index)
  std::vector<std::int64_t> missed;
  std::vector<TrackerTapTrack> dropped;  ///< as they were when dropped (misses > max_misses)
  std::vector<std::pair<std::int64_t, std::int64_t>> born;  ///< (new track id, detection index)
  std::vector<std::int64_t> confirmed_now;
  std::vector<TrackerTapTrack> after;    ///< every track after the update, tentative ones too
  std::vector<TrackerTapRate> rate;      ///< D5: the range-rate updates (written out only with range_rate on)
};

/// Nearest-neighbour Kalman tracker with a constant-velocity model in the horizontal plane.
class Tracker {
 public:
  explicit Tracker(const TrackerOptions& options = {}) : opt_(options) {}
  /// Advance to the ping at `time` (s) with its detections. Each detection is compared with every track predicted to
  /// the detection's own time (Detection::time; the ping's when NaN or otherwise not finite; D-156), a track takes its
  /// detection at that time and a new track is born there; tracks that take none count a miss and are predicted to
  /// `time`, or stay where they are if they already hold at a later time (never moved back in time). A detection timed
  /// before the time a track holds is compared with and taken by it at the track's time (dt = 0; sonar-uc5), so no
  /// history steps back. Returns the confirmed tracks.
  std::vector<const Track*> update(double time, const std::vector<Detection>& detections);
  const std::vector<Track>& tracks() const { return tracks_; }
  const TrackerOptions& options() const { return opt_; }
  /// Opt-in tap (off by default): each update then leaves what it did in last_tap(). Read only.
  void set_tap(bool on) { tap_ = on; }
  bool tap_on() const { return tap_; }
  const TrackerTap& last_tap() const { return last_tap_; }

 private:
  TrackerOptions opt_;
  std::vector<Track> tracks_;
  std::int64_t next_id_ = 1;
  bool tap_ = false;
  TrackerTap last_tap_;
};

enum class Kind { Mover, Static, Line, Multipath, Uncertain };
const char* kind_name(Kind kind);

struct TrackClass {
  std::int64_t id = 0;
  Kind kind = Kind::Uncertain;
  double x = 0.0, y = 0.0;    ///< fitted position (across, forward) at ClassifyOptions::time, else at the track's last history point
  double vx = 0.0, vy = 0.0;  ///< fitted velocity
  double speed = 0.0, speed_sigma = 0.0, fit_rms = 0.0, seen = 0.0, mean_snr_db = 0.0;
  std::int64_t parent = -1;   ///< for multipath: the track it echoes
  double extent_cells = std::numeric_limits<double>::quiet_NaN();  ///< median flagged map cells of its detections; NaN: none
};

struct ClassifyOptions {
  double min_speed = 0.5, static_speed = 0.5, significance = 5.0, min_seen = 0.7;
  std::int64_t min_points = 4;
  double max_fit_rms = -1.0;  ///< < 0: no limit
  bool linear = false;
  double linear_length = 40.0, linear_width = 2.0;
  bool multipath = false;
  double multipath_max_excess_range = 800.0, multipath_bearing = 0.05235987755982988, multipath_velocity = 2.5;
  double origin_x = 0.0, origin_y = 0.0;  ///< where the sonar is, for multipath (across, forward)
  // Rules from the detections themselves (classify.py `detections`; ours), off by default:
  double min_mover_cells = -1.0;  ///< < 0: off. A mover whose median detection extent is below this many cells is uncertain
  double row_snr_db = -1.0;       ///< < 0: off. In the corridor test only tracks at most this many dB weaker make a row
  /// s, the tracker's clock: every position is the fitted line's at this time (the report's: the ping's transmit time;
  /// D-156). NaN (or any value that is not finite): at each track's last history point (classify.py `time`). A track
  /// too short for a fit (fewer than three points) is moved from its last point to this time by its state's velocity.
  /// The line goes through the measured positions (HistoryPoint::mx, my; D-157), the rows of misses left out; fewer
  /// than three measured rows: a short track, from its last filtered point. No row measured: through x, y as before.
  double time = std::numeric_limits<double>::quiet_NaN();
  /// Through measured positions, at most one point the deleted-residual test finds an outlier at this false-alarm
  /// probability per track is left out of the line (classify.py `outlier_alpha`, OUTLIER_ALPHA; sonar-ljh). 0: off, the
  /// default (opt-in, D-159).
  double outlier_alpha = 0.0;
  /// rad, > 0: a track's fit limit is at least this times its range from (origin_x, origin_y) (classify.py
  /// `max_fit_rms_bearing`, sonar-4jm): the scatter of measured positions across the line of sight grows with range.
  /// Applies where max_fit_rms is a limit (not below 0). 0: off.
  double max_fit_rms_bearing = 0.0;
  /// D5 (opt-in; the chain sets it with TrackerOptions::range_rate on): a track that took a range rate whose filter
  /// velocity is distinguishable from standing still (v^T P^-1 v >= significance^2) at min_speed or more is a mover
  /// (rule mover_range_rate) where the fit does not make it one (classify.py `range_rate`).
  bool range_rate = false;
};

/// The classifier's opt-in tap, per track (classify.py CLASS_EVIDENCE, the same members and rule names): the numbers
/// the rules compared and the rule that gave the kind. Read only.
struct ClassEvidence {
  std::int64_t id = 0;
  std::int64_t rows = 0, fitted = 0;
  double speed = 0.0, speed_sigma = 0.0, need = 0.0, seen = 0.0, fit_rms = 0.0;
  double limit = std::numeric_limits<double>::quiet_NaN();   ///< NaN: no limit (max_fit_rms < 0)
  bool steady = true;
  double extent_cells = std::numeric_limits<double>::quiet_NaN();
  std::string rule, kind;
  std::vector<std::string> later;  ///< line, small_extent, bounce, multipath:<parent or none>
  bool has_rate = false;           ///< D5: ClassifyOptions::range_rate on (the members below are written out)
  std::int64_t rate_updates = 0;
  double filter_speed = std::numeric_limits<double>::quiet_NaN(), filter_chi2 = std::numeric_limits<double>::quiet_NaN();
};

/// Mover, static, line, multipath or uncertain, from each track alone (see classify.py). `evidence` (opt-in tap): one
/// ClassEvidence per track appended when given.
std::vector<TrackClass> classify_tracks(const std::vector<const Track*>& tracks, const ClassifyOptions& options = {},
                                        std::vector<ClassEvidence>* evidence = nullptr);

/// The evidence's `later` after echo grouping or the bounces (classify.py evidence_after_grouping): `before` the
/// (kind, parent) of each class before it.
void evidence_after_grouping(std::vector<ClassEvidence>& evidence, const std::vector<std::pair<Kind, std::int64_t>>& before,
                             const std::vector<TrackClass>& classes, const std::vector<std::int64_t>& bounces = {});

/// Echo grouping (D-155, D-172; classify.py group_echoes), in place: classify_tracks' last step when options.multipath
/// (it reads only the multipath_* fields and origin_x, origin_y). `bounces`: ids of tracks the caller knows to be bounces
/// (the live chain's elevation step, sonar-bal): multipath before grouping, so never an anchor nor a group's object; a
/// member of a group like any other track (its parent the group's object), else multipath of nothing (parent -1). With
/// none it is the grouping as before.
void group_echoes(std::vector<TrackClass>& classes, const ClassifyOptions& options, const std::vector<std::int64_t>& bounces = {});

/// The target's part of the LFM range-Doppler correction (sonar-fyj, ChainOptions::range_doppler Full), in place: each
/// class's position moved along the line of sight from (origin_x, origin_y) (across, forward) by `factor`
/// (lfm_range_doppler_factor, s) times the closing speed of its fitted velocity, -v . u. The velocity is not changed;
/// a position at the origin or a velocity that is not finite is left. classify.py shift_for_target_doppler the same.
void shift_for_target_doppler(std::vector<TrackClass>& classes, double factor, double origin_x, double origin_y);

}  // namespace sonar::live
