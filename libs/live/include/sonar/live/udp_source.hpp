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
#include <string>
#include <vector>

#include <memory>

#include "sonar/core/record.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/environment.hpp"

namespace sonar::live {

class PingAssembler {
 public:
  /// With `decimate` (enabled) every ping is band-selected and decimated to complex baseband as
  /// its blocks arrive (decimate.hpp) and, with `reference`, its boards synchronised: the records
  /// that come out are complex, and the full-rate ping is never held.
  explicit PingAssembler(const PingRecord& template_record, const DecimateOptions* decimate = nullptr, const Reference* reference = nullptr);
  /// Take one block; append the ping records it completes (usually none or one).
  /// Returns false, and counts it, if the block is malformed.
  bool feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out);
  std::int64_t dropped_pings = 0, lost_samples = 0, overflow_blocks = 0, bad_blocks = 0;

 private:
  PingRecord template_;
  std::int64_t window_start_ = 0, window_length_ = 0, n_channels_ = 0;
  double volts_per_count_ = 1.0;
  bool have_next_ = false, open_ = false, have_done_ = false;
  std::int64_t next_sample_ = 0, done_ping_ = 0;
  std::int64_t tx_ = 0, index_ = 0, filled_ = 0, start_ns_ = 0;
  std::vector<std::int32_t> buffer_;  ///< [window_length][n_channels]
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
  StreamAssembler(const PingRecord& template_record, std::int64_t chunk_samples, const DecimateOptions* decimate = nullptr);
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
  std::int64_t chunk_ = 0, n_channels_ = 0, filled_ = 0, first_ = 0, start_ns_ = 0, next_sample_ = 0;
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

/// A bound UDP socket with a receive timeout. receive() returns the datagram's size, 0 on timeout.
class UdpSocket {
 public:
  UdpSocket(const std::string& host, int port, double timeout_s = 0.2);
  ~UdpSocket();
  UdpSocket(const UdpSocket&) = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;
  std::size_t receive(std::vector<unsigned char>& buffer);
  void send_to(const std::string& host, int port, const std::string& text);
  int port() const { return port_; }

 private:
  int fd_ = -1, port_ = 0;
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

  enum Field { kSoundSpeed = 9, kWaterDepth, kAltitude, kLock, kRollSigma, kPitchSigma, kYawSigma, kPositionSigma, kVelocitySigma,
               kDepthSigma, kAltitudeSigma, kWaterDepthSigma, kSoundSpeedSigma, kFields };

 private:
  struct Fix {
    std::int64_t time_ns;
    double v[kFields];  ///< position, rpy, velocity, then the optional fields (NaN: not sent)
    WaterDepth water;   ///< from altitude + depth, when configured
  };
  struct Sample {
    std::int64_t time_ns;
    double x, y, z, speed;
  };
  NavOptions opt_;
  WaterDepthTracker water_;
  std::deque<Fix> fixes_;
  std::deque<Sample> samples_;   ///< for the profile, since the last ping
};

}  // namespace sonar::live
