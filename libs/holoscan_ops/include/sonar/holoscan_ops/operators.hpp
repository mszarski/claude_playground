// The long-range array chain as C++ Holoscan operators (Holoscan SDK 4.x).
//
// Messages are shared pointers, passed between operators in one process:
//   "ping"    std::shared_ptr<sonar::PingRecord>   one ping (or one stretch of an untriggered stream)
//   "report"  std::shared_ptr<std::string>         JSON lines, as build/bin/sonar_live writes them
//
//   ReplaySourceOp ─┐
//   UdpSourceOp    ─┴─ ping ─ RecordSinkOp ─ ping ─ LiveChainOp ───── report ─ ReportSinkOp
//                                                 └ PassiveBearingsOp ┘
//
// Each operator is a thin shell around the library the standalone program uses (libs/live),
// so a pipeline gives the same reports as build/bin/sonar_live on the same input.
// Settings are the chain's key=value options (sonar/live/options.hpp), passed as one
// string of space-separated pairs in the `options` parameter.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "sonar/io/record_file.hpp"
#include "sonar/live/chain.hpp"
#include "sonar/live/udp_source.hpp"

namespace sonar::holoscan_ops {

using PingMessage = std::shared_ptr<PingRecord>;
using ReportMessage = std::shared_ptr<std::string>;

/// Pings from a container file, one per tick; stops the pipeline after the last.
class ReplaySourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ReplaySourceOp)
  ReplaySourceOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> path_;
  holoscan::Parameter<int64_t> max_pings_;
  std::unique_ptr<io::RecordReader> reader_;
  std::int64_t next_ = 0;
};

/// Capture-board blocks from a UDP port, assembled into pings on the transmit marks, or with
/// `listen` into fixed-length stretches of the untriggered stream. Navigation datagrams on
/// `nav_port` (0: none) set each ping's pose. A thread of its own drains the socket.
/// Stops the pipeline after `max_pings`, or when no block has arrived for `idle_seconds`.
class UdpSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UdpSourceOp)
  UdpSourceOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> template_path_;
  holoscan::Parameter<int> port_, nav_port_;
  holoscan::Parameter<bool> listen_;
  holoscan::Parameter<int64_t> chunk_samples_, max_pings_;
  holoscan::Parameter<double> idle_seconds_;
  holoscan::Parameter<std::string> options_;
  std::size_t queue_limit_ = 0, queued_bytes_ = 0;   ///< bytes the waiting blocks may hold (0: no bound) and do hold
  std::int64_t queue_dropped_ = 0;
  std::unique_ptr<live::PingAssembler> pings_;
  std::unique_ptr<live::StreamAssembler> stream_;
  std::unique_ptr<live::UdpSocket> socket_, nav_socket_;
  live::Navigation nav_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::vector<unsigned char>> blocks_;
  std::deque<PingRecord> done_;
  std::atomic<bool> stop_{false};
  std::thread receiver_, nav_thread_;
  std::int64_t emitted_ = 0;
  live::NavCounts nav_counts_;   ///< pings without a fix, and without a velocity from navigation (stop() reports both)
};

/// Writes every ping to a container file as it arrives and passes it on unchanged.
class RecordSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RecordSinkOp)
  RecordSinkOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> path_;
  std::unique_ptr<io::RecordWriter> writer_;
};

/// sonar::live::LiveChain: synchronise, condition, beamform, detect, bearings, track, classify.
class LiveChainOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LiveChainOp)
  LiveChainOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> options_;
  std::unique_ptr<live::LiveChain> chain_;
};

/// Passive bearings of each stretch of samples: one JSON line per frame.
class PassiveBearingsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PassiveBearingsOp)
  PassiveBearingsOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> options_;
  live::PassiveOptions passive_;
};

/// Report lines to a file (empty path: standard output) and, when `udp_host` is set, by UDP.
class ReportSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ReportSinkOp)
  ReportSinkOp() = default;
  void setup(holoscan::OperatorSpec& spec) override;
  void start() override;
  void stop() override;
  void compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<std::string> path_, udp_host_;
  holoscan::Parameter<int> udp_port_;
  holoscan::Parameter<int64_t> max_bytes_;
  std::ofstream file_;
  std::unique_ptr<live::UdpSocket> socket_;
};

/// Apply "key=value key=value ..." to chain and passive options.
void apply_options(const std::string& text, live::ChainOptions& chain, live::PassiveOptions& passive);

}  // namespace sonar::holoscan_ops
