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

/// The rules are those of python/gpusonar/sources/flank_stream.py PingAssembler (its docstring states them; the
/// engine-parity tests hold both to the same counters and decisions). In short: blocks are placed by their sample
/// counter, so a block that arrives late but while its ping is still open fills its place (late_blocks) and the
/// ping is kept; a hole is given up whole (lost_samples) once the stream is `reorder_blocks` blocks past its end, or,
/// if it reaches into the open ping's window, when that ping is given up (dropped_pings: a later ping named, the stream `reorder_blocks` blocks
/// past the window's end, or a restart); a ping is emitted as soon as its window is complete, so nothing waits;
/// a block of samples already received is a repeat (repeated_blocks), one inside a hole already given up is too
/// late (too_late_blocks, also in late_blocks); a jump of d in the header's ping index (modulo 2^32) counts d - 1
/// pings none of whose blocks arrived, when the transmits are d windows apart or more; another capture start, a
/// block far behind or one naming a transmit not named recently is a restart (restarts).
class PingAssembler {
 public:
  /// With `decimate` (enabled) every ping is band-selected and decimated to complex baseband as
  /// its blocks arrive (decimate.hpp) and, with `reference`, its boards synchronised: the records
  /// that come out are complex, and the full-rate ping is never held (but for blocks waiting behind a hole).
  /// `sample_bytes`: the size of one sample in a block (2, 3 or 4: a signed little-endian integer whose
  /// full scale is 2^(8 bytes - 1) counts). The block header does not carry it; the sender is configured alike.
  explicit PingAssembler(const PingRecord& template_record, const DecimateOptions* decimate = nullptr, const Reference* reference = nullptr,
                         int sample_bytes = 3);
  /// Take one block; append the ping records it completes (none or one).
  /// Returns false, and counts it, if the block is malformed.
  bool feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out);
  std::int64_t dropped_pings = 0, lost_samples = 0, overflow_blocks = 0, bad_blocks = 0;
  std::int64_t late_blocks = 0, too_late_blocks = 0, repeated_blocks = 0, restarts = 0;
  /// Set before the first block (0 or more).
  int reorder_blocks = kReorderBlocks;
  enum class Block { kNone, kInOrder, kGap, kLate, kTooLate, kRepeated, kRestart };
  Block last_block = Block::kNone;   ///< what the last well-formed block was

 private:
  struct Hole {
    std::int64_t a, b;
  };
  void reset();
  void give_up(std::int64_t limit, const Hole* window = nullptr);
  void abandon(std::int64_t n);
  void name(std::int64_t first, std::int64_t n, std::int64_t tx, std::int64_t index, std::int64_t start_ns);
  bool in_holes(std::int64_t a, std::int64_t b) const;
  std::int64_t span(std::int64_t n) const { return window_start_ + window_length_ + (reorder_blocks + 1) * n; }

  PingRecord template_;
  std::int64_t window_start_ = 0, window_length_ = 0, n_channels_ = 0, sample_bytes_ = 3;
  double volts_per_count_ = 1.0;
  bool started_ = false, open_ = false, have_first_tx_ = false;
  std::int64_t frontier_ = 0, origin_ = 0, stream_ns_ = 0, first_tx_ = 0;
  std::vector<Hole> holes_, expired_;
  std::deque<std::pair<std::int64_t, std::int64_t>> named_;   ///< (tx_sample, ping_index), the latest 4
  std::int64_t tx_ = 0, index_ = 0, w0_ = 0, covered_ = 0, fed_ = 0, start_ns_ = 0;
  std::vector<std::int32_t> buffer_;  ///< [window_length][n_channels]
  std::map<std::int64_t, std::vector<float>> pending_;   ///< decimated path: rows behind a hole, by offset in the window
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
