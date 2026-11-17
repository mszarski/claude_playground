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

#include "sonar/core/record.hpp"

namespace sonar::live {

class PingAssembler {
 public:
  explicit PingAssembler(const PingRecord& template_record);
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
};

/// The same blocks as one continuous stream, for listening without transmitting: every
/// `chunk_samples` contiguous samples become one record (real samples, start_time 0,
/// transmit.time_ns = the time of its first sample). Transmit marks in the blocks are ignored.
/// A gap discards the chunk it falls in; the next chunk starts at the block after the gap.
class StreamAssembler {
 public:
  StreamAssembler(const PingRecord& template_record, std::int64_t chunk_samples);
  bool feed(const unsigned char* data, std::size_t size, std::vector<PingRecord>& out);
  std::int64_t dropped_chunks = 0, lost_samples = 0, bad_blocks = 0, chunks = 0;

 private:
  PingRecord template_;
  std::int64_t chunk_ = 0, n_channels_ = 0, filled_ = 0, first_ = 0, start_ns_ = 0, next_sample_ = 0;
  double volts_per_count_ = 1.0;
  bool have_next_ = false;
  std::vector<float> buffer_;  ///< [n_channels][chunk]
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

/// Navigation fixes {"time_ns", "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz"} and the
/// pose at a ping's transmit time, interpolated between the fixes around it.
class Navigation {
 public:
  /// Parse one datagram; false if it has no time_ns.
  bool push(const std::string& json);
  /// Put the pose at `record.transmit.time_ns` into the record, with the fixes around its receive
  /// window as its attitude history (navigation.ins) and, where the datagrams carry them, the
  /// vehicle's sound speed and the water depth. False before any fix has arrived.
  bool apply(PingRecord& record) const;

 private:
  struct Fix {
    std::int64_t time_ns;
    double v[11];  ///< position, rpy, velocity, then sound speed and water depth (NaN: not sent)
  };
  std::deque<Fix> fixes_;
};

}  // namespace sonar::live
