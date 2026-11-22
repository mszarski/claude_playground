// Detections from a CFAR map, the tracker that follows them and the classes given to its
// tracks. The C++ counterparts of python/gpusonar/detect.py (cluster, Tracker) and
// classify.py; tests/python/test_live_cpp.py checks they give the same answers.
// Sources: docs/REFERENCES.md Sec. 12.3 (constant-velocity Kalman filter; the association is ours, D-141).
#pragma once

#include <cstdint>
#include <limits>
#include <string>
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
  /// The detection's arrivals, nearest first (the multiscale detector): the local peaks along range of the strongest
  /// sample over its bearings, each above the single-sample threshold; at most 8, the strongest. One detection is one
  /// target: its direct echo, its surface and bottom echoes and its highlights within the merge distance are all here,
  /// from `first_range` over `extent`. The complex samples behind them: `beam_samples` (chain.hpp).
  std::vector<Arrival> arrivals{};
  /// s, on the tracker's clock: when the detection shows the target. For a monostatic echo the ping's transmit time +
  /// range / c, with the c that turned the two-way time into range (D-156). NaN (or any value that is not finite): the
  /// ping's time given to Tracker::update.
  double time = std::numeric_limits<double>::quiet_NaN();
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
};

/// How detections are shared out among tracks on each ping (D-141; python/gpusonar/detect.py).
/// Confirmed tracks always choose before tentative ones. Likelihood: the gated (track,
/// detection) pairs most probable first, by d^2 + ln|S|, each track and detection used once;
/// ties go to the earlier track, then the earlier detection. InOrder: the rule before D-141,
/// each track in turn takes its nearest gated detection by d^2.
enum class Association { Likelihood, InOrder };
/// "likelihood" or "in_order"; throws std::invalid_argument for anything else.
Association parse_association(const std::string& name);

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
};

/// Nearest-neighbour Kalman tracker with a constant-velocity model in the horizontal plane.
class Tracker {
 public:
  explicit Tracker(const TrackerOptions& options = {}) : opt_(options) {}
  /// Advance to the ping at `time` (s) with its detections. Each detection is compared with every track predicted to
  /// the detection's own time (Detection::time; the ping's when NaN or otherwise not finite; D-156), a track takes its
  /// detection at that time and a new track is born there; tracks that take none count a miss and are predicted to
  /// `time`, or stay where they are if they already hold at a later time (never moved back in time). Returns the
  /// confirmed tracks.
  std::vector<const Track*> update(double time, const std::vector<Detection>& detections);
  const std::vector<Track>& tracks() const { return tracks_; }

 private:
  TrackerOptions opt_;
  std::vector<Track> tracks_;
  std::int64_t next_id_ = 1;
};

enum class Kind { Mover, Static, Line, Multipath, Uncertain };
const char* kind_name(Kind kind);

struct TrackClass {
  std::int64_t id = 0;
  Kind kind = Kind::Uncertain;
  double x = 0.0, y = 0.0;    ///< fitted position (across, forward) at ClassifyOptions::time, else at the track's last point
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
};

/// Mover, static, line, multipath or uncertain, from each track alone (see classify.py).
std::vector<TrackClass> classify_tracks(const std::vector<const Track*>& tracks, const ClassifyOptions& options = {});

}  // namespace sonar::live
