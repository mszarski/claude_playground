// The long-range array chain, one ping at a time, in C++: the counterpart of
// python/gpusonar/live.py and what the standalone executable apps/sonar_live runs.
//
//   condition -> range x bearing map (backprojection kernel, far-field or MVDR beamformer) -> CFAR -> cluster
//     -> [SBL bearings at each detected range gate] -> track in a world-fixed frame
//     -> classify -> report (JSON)
//
// With `elevation` and an array of stacked rows (elevation.hpp): each row is mapped on its own and
// the rows' power added, every detection gets an elevation and a depth, its bearing is corrected
// for the elevation, and tracks carry depth. One row, or the option off: the chain as it was.
//
// and, for listening without transmitting, PassiveChain: frames -> SBL -> a bearing-time
// record for a downstream tracker.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/live/beamform.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/detections.hpp"
#include "sonar/live/doa.hpp"
#include "sonar/live/elevation.hpp"
#include "sonar/live/sync.hpp"
#include "sonar/live/tonal.hpp"

namespace sonar::live {

/// The projector's own sound reaches the hydrophones far above their full scale: for the length of
/// the pulse, and for as long after as the inputs take to recover, the samples mean nothing.
struct BlankingOptions {
  bool enabled = false;
  double recovery_s = 0.0;   ///< the inputs' recovery after the pulse ends (hardware; unconfirmed until measured)
  double taper_s = 0.0;      ///< raised-cosine return to full gain after the blanked stretch
  bool zero = true;          ///< zero the samples before conditioning (mode zero); false (ignore): leave them, only do not detect there
  /// Samples are blanked from the transmit to this long after it; echoes from nearer than min_range are not reported.
  double until(const PingRecord& record) const { return record.transmit.waveform.duration + recovery_s; }
};

struct ChainOptions {
  double bearing_first_deg = -40.0, bearing_last_deg = 40.0, bearing_step_deg = 0.5;  ///< from broadside
  bool hann = true;            ///< Hann aperture weighting for the map
  /// "backprojection": the kernel, exact in the near field too, host or CUDA.
  /// "farfield": plane-wave delay and sum in the frequency domain (host). "mvdr": adaptive (host).
  std::string beamformer = "backprojection";
  MvdrOptions mvdr;
  bool starboard = true;       ///< which side of a line array the bearings are on
  double range_cell_m = 2.0;   ///< power is averaged into cells this long before detection
  double cfar_pfa = 1e-5, cfar_guard_m = 100.0, cfar_train_m = 300.0;
  double min_snr_db = 9.0;
  /// "map": bearings are the centroids of the detection clusters on the beamformed map.
  /// "sbl": the map finds the ranges; SBL then resolves the bearings at each range gate.
  std::string doa = "map";
  std::int64_t gate_fft = 0;   ///< samples per SBL range gate; 0 = from the array length and bandwidth
  double sbl_grid_step_deg = 0.25;
  std::int64_t sbl_max_gates = 24;           // range gates SBL is run on per ping, strongest detections first
  double sbl_min_contrast_db = 8.0;          // peak over median of a gate's conventional spectrum below which SBL is skipped
  // Bearings on a gate move by a few hundredths of a degree after 15 iterations (RESULTS.md),
  // so the active chain stops at 20: iterations, tolerance, sources, dynamic range, SNR, frequencies.
  DoaOptions sbl{20, 1e-2, 6, 20.0, 6.0, 24};
  TrackerOptions tracker{0.05, 15.0, 0.017453292519943295, 4.0, 3, 2, 8.0};
  ClassifyOptions classify{1.0, 1.0, 5.0, 0.7, 4, 60.0, false, 40.0, 2.0, true, 800.0, 0.05235987755982988, 2.5, 0.0, 0.0};
  bool sync = false;           ///< several capture boards: synchronise on the shared reference first
  Reference reference;
  ElevationOptions elevation;  ///< stacked rows: elevation and depth (off by default)
  std::vector<std::int64_t> dead_channels;  ///< channel rows (after synchronisation) left out of every stage
  double sound_speed = 0.0;    ///< m/s at the vehicle; > 0 replaces the record's (navigation's) value
  BlankingOptions blanking;    ///< off: nothing is blanked, as before
  /// Decimate first (decimate.hpp): a raw record is band-selected, decimated to complex baseband and
  /// (with `sync`) its boards synchronised from the reference bursts, before anything else.
  DecimateOptions decimate;
  // For the program around the chain (apps/sonar_live), not the chain itself:
  bool decimate_streaming = false;  ///< the receiver decimates as blocks arrive and the recording holds the decimated ping
  NavOptions nav;                   ///< the navigation receiver's settings
  Backend backend = Backend::Host;
  int device = 0;
};

struct ReportDetection {
  double range_m, bearing_deg, snr_db;
  // With stacked rows (Report::rows > 0):
  bool elevation_valid = false;                      ///< an elevation was found with enough quality
  double elevation_deg = 0.0, elevation_sigma_deg = 0.0, elevation_quality = 0.0;  ///< level-referenced, up positive
  bool elevation_ambiguous = false;
  std::vector<double> elevation_alternatives_deg{};  ///< the other candidate lobes, when ambiguous
  double depth_refraction_sigma_m = 0.0;             ///< the part of depth_sigma_m that is the unknown gradient
  bool bearing_corrected = false;
  double cone_bearing_deg = 0.0;                     ///< what the line array alone reported
  bool has_depth = false;
  double depth_m = 0.0, depth_sigma_m = 0.0, image_depth_m = 0.0;
  std::string path{};
  std::string profile_source{};                      ///< the rung of the sound-speed ladder this depth came by (ladder on)
};
struct ReportTrack {
  std::int64_t id, parent;
  std::string kind;
  double x, y, vx, vy, speed, range_m, mean_snr_db, seen;  ///< world x north, y east
  bool depth_reported = false, has_depth = false;    ///< stacked rows with depth on; the track has taken a depth
  double depth_m = 0.0, depth_sigma_m = 0.0;
  std::string path{};
  std::string profile_source{};
};
struct Report {
  std::int64_t ping = 0, sequence = 0, time_ns = 0;
  Vec3d ownship;
  std::vector<ReportDetection> detections;
  std::vector<ReportTrack> tracks;
  double blanking_min_range_m = -1.0;   ///< >= 0: blanking is on; nothing nearer than this is reported
  bool blanking_zero = true;
  std::vector<std::int64_t> sync_flagged;  ///< boards timed from one burst (their number) or not at all (-number - 1)
  bool synced = false, sync_ok = false;
  double sync_worst_residual_us = 0.0;
  std::vector<double> sync_offsets_us, sync_drift_ppm;
  double processing_seconds = 0.0;
  std::int64_t rows = 0;     ///< > 0: the array's rows, and elevation fields are in the report
  std::string profile_rung{};  ///< the sound-speed ladder's rung this ping (ladder on), with what a built profile holds
  std::int64_t profile_bins = 0;
  double profile_lo = 0.0, profile_hi = 0.0;
  std::string json() const;  ///< one line, the same fields as the Python chain's report
};

/// Complex baseband and pulse compression, in place, once (the record remembers).
void prepare(PingRecord& record, Backend backend = Backend::Host, int device = 0);

/// Range x bearing power [n_bearing][n_range] by the backprojection kernel, row major.
std::vector<float> range_bearing_power(const PingRecord& record, const std::vector<double>& ranges,
                                       const std::vector<double>& bearings, bool hann, Backend backend, int device = 0);

class LiveChain {
 public:
  explicit LiveChain(const ChainOptions& options = {});
  /// Process one ping (conditioned in place). The record's pose is taken as the vehicle's.
  Report process(PingRecord& record);
  const ChainOptions& options() const { return opt_; }
  SoundSpeedLadder& ladder() { return ladder_; }
  // The last ping's map, for display and taps.
  std::vector<float> last_power;
  std::vector<double> last_ranges, bearings;

 private:
  ChainOptions opt_;
  Tracker tracker_;
  SoundSpeedLadder ladder_;
  std::int64_t pings_ = 0;
  bool have_origin_ = false;
  Vec3d origin_;
  std::int64_t t0_ns_ = 0;
};

struct PassiveOptions {
  double f_low = 0.0, f_high = 0.0;   ///< Hz, absolute; both 0 = the record's transmit band
  std::int64_t n_fft = 1024, hop = 512, n_average = 8;
  double bearing_first_deg = -80.0, bearing_last_deg = 80.0, bearing_step_deg = 0.5;
  bool use_sbl = true;                ///< false: conventional beam power
  DoaOptions sbl;
  ElevationOptions elevation;         ///< stacked rows: an elevation with every bearing
  TonalOptions tonal;                 ///< Goertzel lines and DEMON (tonal.hpp); off unless tonal.enabled
  bool starboard = true;
};

struct PassiveFrame {
  double time = 0.0;                  ///< s into the record
  std::vector<DoaPeak> peaks;
  std::vector<float> spectrum;
  double noise = 0.0;
  std::vector<double> elevations;     ///< rad, level-referenced, up positive: one per peak (stacked rows), else empty
  std::vector<double> elevation_quality;
};

/// Bearing-time record of one stretch of data: a frame every (n_fft + (n_average - 1) hop) samples.
std::vector<PassiveFrame> passive_bearings(const PingRecord& record, const PassiveOptions& options,
                                           std::vector<double>* bearings = nullptr);

/// The frames as JSON lines: {"time_ns", "time_s", "bearings_deg": [...], "power_db": [...], "snr_db": [...]}
/// and, with stacked rows, "elevations_deg": [...], "elevation_quality": [...].
std::string passive_json(const std::vector<PassiveFrame>& frames, std::int64_t record_time_ns);

}  // namespace sonar::live
