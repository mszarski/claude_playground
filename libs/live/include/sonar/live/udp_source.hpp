// Capture-board blocks in, ping records out, in C++ (the counterpart of
// python/gpusonar/sources/flank_stream.py; the block layout is described there), a UDP
// receiver for them, and the vehicle's navigation as JSON datagrams.
//
// The assembler needs to know the array: it clones a template record (geometry, waveform,
// receive window, ADC scale) for every ping. `tools/sonar_live.py --export-template`
// writes one from a sonar's YAML file, so the executable needs no YAML parser.
#pragma once

#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <utility>
#include <string>
#include <vector>

#include <memory>

#include "sonar/core/record.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/environment.hpp"

namespace sonar::live {

/// Blocks the stream may move past a hole before it is given up: 2 lets the last block of a window and the one
/// after it swap places (ours; flank_stream.py REORDER_BLOCKS is the same).
constexpr int kReorderBlocks = 2;
/// Raw bytes a ping may hold behind holes on the decimating path before it is given up (ours; flank_stream.py
/// MAX_HELD_BYTES is the same).
constexpr std::int64_t kMaxHeldBytes = std::int64_t{64} << 20;
/// Pings open at once (the reorder grace keeps an earlier one open while a later one fills); flank_stream.py the same.
constexpr std::size_t kMaxOpenPings = 2;
/// Blocks held as a possible restart; also how many in a row, consistent, confirm one. flank_stream.py the same.
constexpr std::int64_t kMaxHeldBlocks = 8;
/// Sample indices and transmit samples at or above this are refused (bad_blocks), as in flank_stream.py.
constexpr std::int64_t kMaxSampleIndex = std::int64_t{1} << 62;

/// The rules are those of python/gpusonar/sources/flank_stream.py PingAssembler; its docstring states them and
/// the engine-parity tests (tests/python/test_hsb_receive.py) hold both to the same counters and decisions. In short:
/// blocks are placed by their sample counter; an earlier ping stays open beside a later one only for the reorder
/// grace (at most kMaxOpenPings, and never when the later one was transmitted inside its window: then it is given up,
/// as before 2026-11); records come out in transmit order; a hole is given up (lost_samples) `reorder_blocks` blocks
/// after it unless it reaches into an open window; an open ping is given up (dropped_pings) `reorder_blocks` blocks
/// after its window's end; a repeat is ignored (repeated_blocks), a block in a hole already given up is too late
/// (too_late_blocks, in late_blocks too); a jump of d in the ping index counts d - 1 dropped pings when the transmits
/// are d ping intervals apart (median of the last 5, within half an interval, and beyond d - 1 of the longest by
/// half an interval), nothing before an interval is known; a block that does not fit is held, and a restart
/// (restarts) needs two contiguous held blocks of another capture start, or of this one before the first sample
/// heard, or kMaxHeldBlocks held in a row consistent with each other; a capture left is never returned to.
/// finish() gives up what is still awaited at the end of a stream. Raw memory: at most kMaxOpenPings windows;
/// decimating: rows held behind holes at most max_held_bytes in all.
class PingAssembler {
 public:
  /// With `decimate` (enabled) every ping is band-selected and decimated to complex baseband as
  /// its blocks arrive (decimate.hpp) and, with `reference`, its boards synchronised: the records
  /// that come out are complex, and the full-rate ping is never held (but for rows waiting behind a hole).
  /// `sample_bytes`: the size of one sample in a block (2, 3 or 4: a signed little-endian integer whose
  /// full scale is 2^(8 bytes - 1) counts). The block header does not carry it; the sender is configured alike.
  explicit PingAssembler(const PingRecord& template_record, const DecimateOptions* decimate = nullptr, const Reference* reference = nullptr,
                         int sample_bytes = 3);
  /// Take one block; append the ping records it releases (usually none or one).
  /// Returns false, and counts it, if the block is malformed.
  bool feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out);
  /// The stream has ended: give up what is still awaited (counted) and append the pings that were waiting.
  void finish(std::vector<PingRecord>& out);
  /// Whether samples [a, b) may still arrive in time (not yet reached, or in a hole not given up).
  bool holds(std::int64_t a, std::int64_t b) const;
  std::int64_t dropped_pings = 0, lost_samples = 0, overflow_blocks = 0, bad_blocks = 0;
  std::int64_t late_blocks = 0, too_late_blocks = 0, repeated_blocks = 0, restarts = 0, used_blocks = 0, held_peak_bytes = 0, open_peak = 0;
  /// Set before the first block (0 or more).
  int reorder_blocks = kReorderBlocks;
  std::int64_t max_held_bytes = kMaxHeldBytes;
  /// The board clock's rate error in ppm (positive: it counts fast; sonar-ubg). 0: transmit.time_ns = capture start +
  /// tx / fs, as always; else capture start + tx / (fs (1 + ppm 1e-6)) (stamp_offset_ns). May change between blocks.
  double clock_ppm = 0.0;
  enum class Block { kNone, kInOrder, kGap, kLate, kTooLate, kRepeated, kHeld, kRestart };
  Block last_block = Block::kNone;   ///< what the last well-formed block was

 private:
  struct Hole {
    std::int64_t a, b;
  };
  struct Header {
    std::int64_t first = 0, n = 0, tx = -1, index = 0, start_ns = 0;
    std::uint16_t flags = 0;
    std::vector<unsigned char> bytes;   ///< the whole block
  };
  enum class State { kOpen, kComplete, kDropped };
  struct Ping {
    std::int64_t tx = 0, index = 0, w0 = 0, covered = 0, fed = 0, held = 0, start_ns = 0;
    State state = State::kOpen;
    std::vector<std::int32_t> buffer;                      ///< [window_length][n_channels]
    std::map<std::int64_t, std::vector<float>> pending;    ///< decimated path: rows waiting, by offset in the window
    PingRecord record;
  };
  void reset();
  bool fits(const Header& h) const;
  bool known(std::int64_t tx, std::int64_t index) const;
  Block take(const Header& h, std::vector<PingRecord>& out);
  void name(const Header& h);
  void place(Ping& p, const Header& h, std::int64_t a, std::int64_t b);
  void drain(Ping& p);
  void settle_pings();
  bool gap_open(const Ping& p) const;
  std::vector<const Header*> restart_run(const Header& h) const;
  void release(std::vector<PingRecord>& out, bool wait);
  void close(std::vector<PingRecord>& out, bool drop_open);
  void drop(Ping& p);
  void lose(std::int64_t a, std::int64_t b);
  void give_up(std::int64_t limit);
  void give_up_to(std::int64_t end, std::int64_t n);
  bool in_holes(std::int64_t a, std::int64_t b) const;
  bool take_expired(std::int64_t a, std::int64_t b);
  PingRecord make_record(Ping& p);
  std::int64_t span(std::int64_t n) const { return window_start_ + window_length_ + (reorder_blocks + 1) * n; }

  PingRecord template_;
  std::int64_t window_start_ = 0, window_length_ = 0, n_channels_ = 0, sample_bytes_ = 3;
  double volts_per_count_ = 1.0;
  bool started_ = false, have_first_tx_ = false, have_max_ = false;
  std::int64_t frontier_ = 0, origin_ = 0, stream_ns_ = 0, first_tx_ = 0, fitting_ = 0, held_run_ = 0;
  std::vector<std::int64_t> left_;                            ///< capture starts left by a restart: never returned to
  std::pair<std::int64_t, std::int64_t> max_named_{0, 0};
  std::vector<Hole> holes_, expired_;
  std::vector<Hole> lost_;   ///< given up, ending after the last ping released: an index jump counts only with one between (sonar-2ht)
  std::list<Ping> pings_;                                     ///< open or undecided, in transmit order
  Ping* owner_ = nullptr;                                     ///< the ping the decimator is open for
  std::vector<std::pair<std::int64_t, std::int64_t>> released_;   ///< (tx, index) emitted or dropped, the latest 8
  std::vector<std::int64_t> deltas_;                          ///< transmit gaps between consecutive indices, the latest 5
  std::deque<std::pair<std::int64_t, Header>> suspects_;      ///< held blocks that do not fit: maybe a restart
  std::shared_ptr<PingDecimator> decimator_;
  std::vector<float> rows_;
};

/// The same blocks as one continuous stream, for listening without transmitting: every
/// `chunk_samples` contiguous samples become one record (real samples, start_time 0,
/// transmit.time_ns = the time of its first sample). Transmit marks in the blocks are ignored.
/// A gap discards the chunk it falls in; the next chunk starts at the block after the gap.
class StreamAssembler {
 public:
  /// With `decimate` the stream is decimated to complex baseband as it arrives and `chunk_samples`
  /// counts decimated samples (phase continuous from chunk to chunk; no board synchronisation,
  /// as without it).
  StreamAssembler(const PingRecord& template_record, std::int64_t chunk_samples, const DecimateOptions* decimate = nullptr,
                  int sample_bytes = 3);
  bool feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out);
  std::int64_t dropped_chunks = 0, lost_samples = 0, bad_blocks = 0, chunks = 0, blanked_samples = 0;
  /// A platform that also pings: samples from each transmit the blocks mark until `until_s` after it are
  /// zeroed, with a raised-cosine return over `taper_s` (the projector's own sound, and the inputs' recovery).
  void set_blanking(double until_s, double taper_s) {
    blank_until_ = until_s;
    blank_taper_ = taper_s;
  }

 private:
  PingRecord template_;
  std::int64_t chunk_ = 0, n_channels_ = 0, filled_ = 0, first_ = 0, start_ns_ = 0, next_sample_ = 0, sample_bytes_ = 3;
  double volts_per_count_ = 1.0;
  bool have_next_ = false;
  std::vector<float> buffer_;  ///< [n_channels][chunk]
  std::shared_ptr<StreamDecimator> decimator_;
  DecimatePlan plan_;
  std::vector<std::vector<std::complex<float>>> pending_;
  std::vector<float> rows_;
  std::int64_t stretch_first_ = 0, emitted_ = 0;
  double blank_until_ = 0.0, blank_taper_ = 0.0;
};

/// One sample of a block, sign-extended: `bytes` little-endian bytes (2, 3 or 4).
inline std::int32_t block_sample(const unsigned char* q, std::int64_t bytes) {
  if (bytes == 3) {
    std::int32_t v = static_cast<std::int32_t>(q[0]) | (static_cast<std::int32_t>(q[1]) << 8) | (static_cast<std::int32_t>(q[2]) << 16);
    if (v & 0x800000) v -= 0x1000000;        // sign-extend 24 bits
    return v;
  }
  if (bytes == 2) return static_cast<std::int16_t>(static_cast<std::uint16_t>(q[0]) | (static_cast<std::uint16_t>(q[1]) << 8));
  return static_cast<std::int32_t>(static_cast<std::uint32_t>(q[0]) | (static_cast<std::uint32_t>(q[1]) << 8) |
                                   (static_cast<std::uint32_t>(q[2]) << 16) | (static_cast<std::uint32_t>(q[3]) << 24));
}
/// Counts at full scale for samples of `bytes` bytes: 2^(8 bytes - 1). Throws for a size other than 2, 3 or 4.
double block_full_scale(std::int64_t bytes);

/// The largest UDP payload: a datagram longer than this cannot be sent.
constexpr std::size_t kMaxDatagram = 65507;

/// A bound UDP socket with a receive timeout. receive() returns the datagram's size, 0 on timeout.
class UdpSocket {
 public:
  /// `buffer_bytes`: the receive buffer asked of the system (0: 8 MB). What the system granted is
  /// buffer_granted(); it can be less (net.core.rmem_max), and the caller should say so.
  UdpSocket(const std::string& host, int port, double timeout_s = 0.2, std::int64_t buffer_bytes = 0);
  ~UdpSocket();
  UdpSocket(const UdpSocket&) = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;
  std::size_t receive(std::vector<unsigned char>& buffer);
  /// False, and nothing is sent, when the text is longer than a datagram can be (it is never cut) or the
  /// address is bad; both are counted in `refused`.
  bool send_to(const std::string& host, int port, const std::string& text);
  int port() const { return port_; }
  std::int64_t buffer_asked() const { return asked_; }
  std::int64_t buffer_granted() const { return granted_; }   ///< usable bytes (Linux reports twice this; halved here)
  std::int64_t refused = 0;

 private:
  int fd_ = -1, port_ = 0;
  std::int64_t asked_ = 0, granted_ = 0;
};

/// Navigation as JSON datagrams, one fix each:
///   {"time_ns", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"}      north, east, down (m); rad; m/s over ground
/// and, where the vehicle has them (an inertial unit with a Doppler log does):
///   "sound_speed" (m/s at the vehicle), "altitude" (m above the bottom), "bottom_lock" (0 | 1; absent:
///   locked when the altitude is a positive number), "water_depth" (m, when the vehicle works it out itself),
///   and one-sigma uncertainties "roll_sigma", "pitch_sigma", "yaw_sigma" (rad), "position_sigma" (m),
///   "velocity_sigma" (m/s), "depth_sigma", "altitude_sigma", "water_depth_sigma" (m), "sound_speed_sigma" (m/s).
/// apply() gives a record the pose at its transmit time, interpolated between the fixes around it,
/// and everything else the chain reads from a record (so that a recording replays the same).
/// With NavOptions::lever_arm or mount_rpy set, push() moves each fix's position and attitude from the
/// navigation unit to the array (nav_to_array) before it is kept: the pose, the attitude history and the
/// depth samples are then the array's. The water depth from altitude and the (depth, sound speed) samples
/// are worked from the unit's own position, as before (nav_dvl_offset_m is measured from the unit).
class Navigation {
 public:
  Navigation() = default;
  explicit Navigation(const NavOptions& options) : opt_(options), water_(options.water_depth) {}
  /// Parse one datagram; false if it has no time_ns.
  bool push(const std::string& json);
  /// Into the record: the pose at `record.transmit.time_ns`; the fixes around its receive window as
  /// attitude history (navigation.ins) and, where sent, as depth and Doppler-log samples
  /// (navigation.depth, navigation.dvl); the vehicle's sound speed and the water depth (sent, or
  /// from altitude + depth: environment); and in metadata.extra_json under "nav" the uncertainties
  /// sent, whether the sound speed is a measurement, and the (depth, sound speed) samples taken since
  /// the last ping (environment.hpp). False before any fix has arrived.
  bool apply(PingRecord& record);
  std::size_t size() const { return fixes_.size(); }
  /// Whether the last apply() had a velocity over ground from navigation: every fix its pose came from (the two
  /// around the transmit time, or the one it was held from) sent "vx", "vy" and "vz" as finite numbers. Without it
  /// the pose's velocity is zero there (a component not sent or not finite counts as 0) and the backprojection does
  /// not follow the array's own motion over the round trip (-24 m at 12 km; sonar-2hz, sonar-8is). False before
  /// any apply(); apps/sonar_live counts it in Report::pings_without_velocity.
  bool velocity_given() const { return velocity_given_; }

  enum Field { kSoundSpeed = 9, kWaterDepth, kAltitude, kLock, kRollSigma, kPitchSigma, kYawSigma, kPositionSigma, kVelocitySigma,
               kDepthSigma, kAltitudeSigma, kWaterDepthSigma, kSoundSpeedSigma, kFields };

 private:
  struct Fix {
    std::int64_t time_ns;
    double v[kFields];  ///< position, rpy, velocity, then the optional fields (NaN: not sent)
    WaterDepth water;   ///< from altitude + depth, when configured
    bool velocity;      ///< vx, vy and vz were all sent as finite numbers (else the missing ones are 0)
  };
  struct Sample {
    std::int64_t time_ns;
    double x, y, z, speed;
  };
  NavOptions opt_;
  WaterDepthTracker water_;
  std::deque<Fix> fixes_;
  std::deque<Sample> samples_;   ///< for the profile, since the last ping
  bool velocity_given_ = false;  ///< velocity_given()
};

/// The navigation unit's pose -> the array's, in place (sonar-3rv; live.py nav_to_array is the same):
///   position <- position + R(rpy) lever_arm,   rpy <- the Eq. 2 angles of R(rpy) R(mount_rpy),
/// R the body-to-NED rotation of ASASIN Eq. 2 (rotation_ned). v: north, east, down (m), roll, pitch, yaw (rad),
/// then velocity (m/s), which is left as it is. Zero lever arm and mount: nothing is touched.
void nav_to_array(const NavOptions& options, double* v);

/// What a program that applies navigation to its pings counts (apps/sonar_live, holoscan_ops UdpSourceOp; the Python
/// LiveChain counts the same): pings with no fix (Report::pings_without_nav) and pings whose fix had no velocity over
/// ground (Navigation::velocity_given; Report::pings_without_velocity). Cumulative.
struct NavCounts {
  std::int64_t without_nav = 0, without_velocity = 0;
  /// Apply `nav` to `record` and count. Call it under whatever guards `nav` (the apply and the velocity flag are read
  /// together). True the first time a ping has no velocity, for the caller to warn once (sonar-8is).
  bool apply(Navigation& nav, PingRecord& record);
};

}  // namespace sonar::live
