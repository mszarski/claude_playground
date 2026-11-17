// Detections from a CFAR map, the tracker that follows them and the classes given to its
// tracks. The C++ counterparts of python/gpusonar/detect.py (cluster, Tracker) and
// classify.py; tests/python/test_live_cpp.py checks they give the same answers.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sonar::live {

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
};

/// Group touching detection cells (8-connected) of a map [n_rows (bearing), n_cols (range)]
/// and report one detection per group at its power-weighted centroid. `ratio` is the
/// power over the local background (the CFAR kernel's second output).
std::vector<Detection> cluster(const unsigned char* detections, const float* power, const float* ratio,
                               std::int64_t n_rows, std::int64_t n_cols, const std::vector<double>& ranges,
                               const std::vector<double>& bearings);

struct HistoryPoint {
  double time, x, y;  ///< s; across and forward, m
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
  double depth = std::numeric_limits<double>::quiet_NaN();  ///< m, from the detections that carried one
  double depth_var = std::numeric_limits<double>::infinity();
};

struct TrackerOptions {
  double process_noise = 0.05;
  double range_sigma = 2.0;                   ///< m
  double bearing_sigma = 0.017453292519943295;  ///< rad
  double gate = 4.0;
  std::int64_t confirm_hits = 3;
  std::int64_t max_misses = 3;
  double initial_speed_sigma = 10.0;
  double depth_process_noise = 0.5;  ///< m per sqrt(s): how fast a track's depth may wander
};

/// Nearest-neighbour Kalman tracker with a constant-velocity model in the horizontal plane.
class Tracker {
 public:
  explicit Tracker(const TrackerOptions& options = {}) : opt_(options) {}
  /// Advance to `time` (s) with this ping's detections. Returns the confirmed tracks.
  std::vector<const Track*> update(double time, const std::vector<Detection>& detections);
  const std::vector<Track>& tracks() const { return tracks_; }

 private:
  TrackerOptions opt_;
  std::vector<Track> tracks_;
  std::int64_t next_id_ = 1;
  bool started_ = false;
  double time_ = 0.0;
};

enum class Kind { Mover, Static, Line, Multipath, Uncertain };
const char* kind_name(Kind kind);

struct TrackClass {
  std::int64_t id = 0;
  Kind kind = Kind::Uncertain;
  double x = 0.0, y = 0.0;    ///< fitted position at the last ping (across, forward)
  double vx = 0.0, vy = 0.0;  ///< fitted velocity
  double speed = 0.0, speed_sigma = 0.0, fit_rms = 0.0, seen = 0.0, mean_snr_db = 0.0;
  std::int64_t parent = -1;   ///< for multipath: the track it echoes
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
};

/// Mover, static, line, multipath or uncertain, from each track alone (see classify.py).
std::vector<TrackClass> classify_tracks(const std::vector<const Track*>& tracks, const ClassifyOptions& options = {});

}  // namespace sonar::live
