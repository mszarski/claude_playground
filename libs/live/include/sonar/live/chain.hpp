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

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "sonar/core/platform.hpp"
#include "sonar/core/record.hpp"
#include "sonar/live/beamform.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/derive.hpp"
#include "sonar/live/detections.hpp"
#include "sonar/live/detector.hpp"
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

/// The program around the chain (apps/sonar_live, the Holoscan source): the board stream and the report datagrams.
struct StreamOptions {
  /// Bytes of one sample in a block: a signed little-endian integer, full scale 2^(8 bytes - 1) counts.
  /// The block header does not carry it, so sender and receiver are configured alike (2, 3 or 4).
  int sample_bytes = 3;
  std::int64_t buffer_bytes = 0;      ///< receive buffer asked of the system; 0: the larger of 8 MB and half a second of the stream
  double queue_mb = 1024.0;           ///< blocks waiting for the chain may hold this much; beyond it arriving blocks are dropped and counted (0: no bound)
  double idle_seconds = 30.0;         ///< the program ends when no block has arrived for this long (0: never)
  std::string bind = "0.0.0.0";       ///< the address the block and navigation sockets listen on
  std::int64_t report_max_bytes = 65000;  ///< a report line longer than this goes out as several datagrams (Report::datagrams)
};

struct ChainOptions {
  double bearing_first_deg = -40.0, bearing_last_deg = 40.0, bearing_step_deg = 0.5;  ///< from broadside
  bool hann = true;            ///< Hann aperture weighting for the map
  /// "backprojection": the kernel, exact in the near field too, host or CUDA.
  /// "farfield": plane-wave delay and sum in the frequency domain (host). "mvdr": adaptive (host).
  std::string beamformer = "backprojection";
  MvdrOptions mvdr;
  bool starboard = true;       ///< which side of a line array the bearings are on
  // Sizes that follow from the sonar are left to be derived from the first record (derive.hpp); a number here wins.
  double range_cell_m = kDerive;   ///< power is averaged into cells this long before detection
  double cfar_pfa = 1e-5, cfar_guard_m = kDerive, cfar_train_m = kDerive;
  /// What a range cell holds of the finer map it is made from (samples no more than 0.75 of a range
  /// resolution apart). false ("mean", as always): their mean power; an echo one resolution long in a
  /// cell of k samples then reads about 10 log10(k x 0.75) dB below its peak. true ("peak"): the largest
  /// of them; the background is then the mean of such maxima, about the k-th harmonic number above the
  /// mean noise power. docs/DEPLOYMENT.md, "What snr_db is".
  bool cell_peak = false;
  /// "multiscale", the default (detector.hpp): each cell is tested on its strongest sample and on its mean (and on
  /// the means of several cells when asked), each against the threshold of its own statistic at
  /// `detector.false_alarms_per_ping`; `cfar_pfa` is then not used. false with cell_peak false: "mean", the detector
  /// until then, line for line. cell_peak with this set is refused.
  bool cell_multiscale = true;
  DetectorOptions detector;
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
  TrackerOptions tracker{0.05, kDerive, kDerive, 4.0, 3, 2, 8.0};      // range and bearing sigma: derived
  ClassifyOptions classify{1.0, 1.0, 5.0, 0.7, 4, kDerive, false, 40.0, 2.0, true, 800.0, kDerive, 2.5, 0.0, 0.0};   // fit limit, multipath bearing: derived
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
  StreamOptions stream;             ///< the board stream, its socket and queue, and the report datagrams
  Backend backend = Backend::Host;
  int device = 0;
};

struct ReportDetection {
  double range_m, bearing_deg, snr_db;
  double extent_m = -1.0;                            ///< >= 0: the cluster's extent in range (multiscale detector)
  double first_range_m = 0.0;                        ///< where that extent begins: the detection covers [first_range_m - cell / 2, + extent_m]
  std::vector<Arrival> arrivals{};                   ///< its arrivals along range (detections.hpp); reported when there are two or more
  // With stacked rows (Report::rows > 0):
  bool elevation_valid = false;                      ///< an elevation was found with enough quality
  double elevation_deg = 0.0, elevation_sigma_deg = 0.0, elevation_quality = 0.0;  ///< level-referenced, up positive
  bool elevation_ambiguous = false;
  std::vector<double> elevation_alternatives_deg{};  ///< the other candidate lobes, when ambiguous
  double depth_refraction_sigma_m = 0.0;             ///< the part of depth_sigma_m that is the unknown gradient
  bool bearing_corrected = false;
  bool bearing_uncorrected = false;                  ///< correction asked for, elevation ambiguous: the row's bearing and the slant range kept
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
  bool sbl_fallback = false;  ///< SBL was asked for on more channels than its kernels take and ran on the host reference instead
  std::int64_t rows = 0;     ///< > 0: the array's rows, and elevation fields are in the report
  std::string profile_rung{};  ///< the sound-speed ladder's rung this ping (ladder on), with what a built profile holds
  std::int64_t profile_bins = 0;
  double profile_lo = 0.0, profile_hi = 0.0;
  std::string json() const;  ///< one line, the same fields as the Python chain's report
  /// The line as UDP datagrams of at most `max_bytes`: the line itself when it fits. Otherwise several
  /// lines, each a complete report of this ping holding a share of the detections and tracks and a
  /// field "part": [k, n] (k from 1): a reader that knows nothing of parts parses each as a report with
  /// fewer detections; one that does joins the lists of the n parts with the same "ping".
  std::vector<std::string> datagrams(std::size_t max_bytes) const;

 private:
  std::string json_part(std::size_t det_first, std::size_t det_last, std::size_t track_first, std::size_t track_last,
                        std::size_t part, std::size_t parts) const;
};

/// Text of several lines as datagrams of at most `max_bytes`, whole lines in each (a single line longer
/// than that is put alone in one, which UdpSocket::send_to then refuses and reports).
std::vector<std::string> line_datagrams(const std::string& lines, std::size_t max_bytes);

/// Complex baseband and pulse compression, in place, once (the record remembers).
void prepare(PingRecord& record, Backend backend = Backend::Host, int device = 0);

/// Range x bearing power [n_bearing][n_range] by the backprojection kernel, row major.
std::vector<float> range_bearing_power(const PingRecord& record, const std::vector<double>& ranges,
                                       const std::vector<double>& bearings, bool hann, Backend backend, int device = 0);

/// The compressed, beamformed COMPLEX samples of one line of hydrophones at `ranges` (m, any spacing; the map's own
/// is a quarter of a metre for 4-6 kHz) along one bearing (rad from broadside, the map's convention): what the map's
/// power is the squared magnitude of. For a module that needs the arrival structure inside a detection (the delay and
/// phase between a direct and a surface echo): take the detection's `first_range` and `extent` (or its `arrivals`) and
/// its bearing, and ask for the samples over that window. `record` must be prepared (`prepare`; LiveChain::process
/// leaves it so); with stacked rows pass one row's record (`row_record`) per call: the chain adds the rows in POWER,
/// and their complex sum is not the map. The phase is that of the backprojection kernel (range from the line's centre).
std::vector<std::complex<float>> beam_samples(const PingRecord& record, const std::vector<double>& ranges, double bearing, bool hann,
                                              Backend backend = Backend::Host, int device = 0);

class LiveChain {
 public:
  explicit LiveChain(const ChainOptions& options = {});
  /// Process one ping (conditioned in place). The record's pose is taken as the vehicle's.
  Report process(PingRecord& record);
  /// The options; after the first ping with every derived setting filled in.
  const ChainOptions& options() const { return opt_; }
  /// What the first ping's record resolved (derive.hpp); an empty JSON object before it.
  std::string resolved_json() const { return resolved_ ? defaults_.json() : "{}"; }
  /// The multiscale detector's plan for the last map (detector.hpp): lengths, looks, thresholds; "{}" when it is not in use.
  std::string detector_json() const { return planned_ ? plan_.json() : "{}"; }
  const DetectorPlan& detector_plan() const { return plan_; }
  bool resolved() const { return resolved_; }
  SoundSpeedLadder& ladder() { return ladder_; }
  // The last ping's map, for display and taps.
  std::vector<float> last_power;
  std::vector<double> last_ranges, bearings;

 private:
  ChainOptions opt_;
  ChainDefaults defaults_;
  bool resolved_ = false;
  Tracker tracker_;
  SoundSpeedLadder ladder_;
  std::int64_t pings_ = 0;
  std::int64_t cell_samples_ = 1;   ///< fine samples per range cell in the last map
  std::vector<float> fine_;         ///< the map before it is averaged into cells (multiscale detector), rows added
  std::vector<double> fine_ranges_;
  std::vector<float> background_;   ///< the detector's background of each cell (multiscale detector)
  DetectorPlan plan_;
  bool planned_ = false;
  std::int64_t plan_cells_ = 0, plan_train_ = 0, plan_rows_ = 0;
  bool have_origin_ = false;
  Vec3d origin_;
  std::int64_t t0_ns_ = 0;
};

struct PassiveOptions {
  double f_low = 0.0, f_high = 0.0;   ///< Hz, absolute; both 0 = the record's transmit band
  std::int64_t n_fft = 0, hop = 0, n_average = 8;   ///< n_fft 0: derived (1024, or two array transits); hop 0: half of n_fft
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
  bool sbl_fallback = false;          ///< more channels than the SBL kernels take: solved on the host reference
  std::int64_t rows_as_snapshots = 0; ///< > 0: stacked rows without the elevation option: bearings from this many rows as snapshots
};

/// The length of a passive frame in samples: n_fft + (n_average - 1) hop, with the derived sizes.
std::int64_t passive_span(const PassiveOptions& options, const PingRecord& record);

/// Bearing-time record of one stretch of data: a frame every (n_fft + (n_average - 1) hop) samples.
std::vector<PassiveFrame> passive_bearings(const PingRecord& record, const PassiveOptions& options,
                                           std::vector<double>* bearings = nullptr);

/// The frames as JSON lines: {"time_ns", "time_s", "bearings_deg": [...], "power_db": [...], "snr_db": [...]}
/// and, with stacked rows, "elevations_deg": [...], "elevation_quality": [...].
std::string passive_json(const std::vector<PassiveFrame>& frames, std::int64_t record_time_ns);

}  // namespace sonar::live
