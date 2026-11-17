#include "sonar/holoscan_ops/operators.hpp"

#include <chrono>
#include <iostream>
#include <sstream>

#include "sonar/live/options.hpp"

namespace sonar::holoscan_ops {

void apply_options(const std::string& text, live::ChainOptions& chain, live::PassiveOptions& passive) {
  std::istringstream words(text);
  std::string pair;
  while (words >> pair) live::set_pair(chain, passive, pair);
}

// --- ReplaySourceOp ----------------------------------------------------------------------

void ReplaySourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output<PingMessage>("ping");
  spec.param(path_, "path", "Recording", "Container file of ping records", std::string());
  spec.param(max_pings_, "max_pings", "Pings", "Stop after this many (negative: the whole file)", static_cast<int64_t>(-1));
}

void ReplaySourceOp::start() {
  reader_ = std::make_unique<io::RecordReader>(path_.get());
  next_ = 0;
}

void ReplaySourceOp::compute(holoscan::InputContext&, holoscan::OutputContext& out, holoscan::ExecutionContext&) {
  const std::int64_t limit = max_pings_.get() < 0 ? reader_->size() : std::min<std::int64_t>(max_pings_.get(), reader_->size());
  if (next_ >= limit) {
    stop_execution();
    return;
  }
  out.emit(std::make_shared<PingRecord>(reader_->read(next_++)), "ping");
  if (next_ >= limit) stop_execution();    // nothing more to send: let the pipeline drain and end
}

// --- UdpSourceOp -------------------------------------------------------------------------

void UdpSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output<PingMessage>("ping");
  spec.param(template_path_, "template_path", "Template", "Container file with one record of the sonar", std::string());
  spec.param(port_, "port", "Port", "UDP port the boards send blocks to", 5700);
  spec.param(nav_port_, "nav_port", "Navigation port", "UDP port of navigation datagrams; 0 for none", 0);
  spec.param(listen_, "listen", "Listen", "Untriggered stream: cut into stretches of chunk_samples", false);
  spec.param(chunk_samples_, "chunk_samples", "Chunk", "Samples per stretch when listening", static_cast<int64_t>(8192));
  spec.param(max_pings_, "max_pings", "Pings", "Stop after this many (negative: never)", static_cast<int64_t>(-1));
  spec.param(idle_seconds_, "idle_seconds", "Idle time", "Stop when no block has arrived for this long", 30.0);
}

void UdpSourceOp::start() {
  const PingRecord template_record = io::RecordReader(template_path_.get()).read(0);
  if (listen_.get()) {
    stream_ = std::make_unique<live::StreamAssembler>(template_record, chunk_samples_.get());
  } else {
    pings_ = std::make_unique<live::PingAssembler>(template_record);
  }
  socket_ = std::make_unique<live::UdpSocket>("0.0.0.0", port_.get());
  if (nav_port_.get() > 0) nav_socket_ = std::make_unique<live::UdpSocket>("0.0.0.0", nav_port_.get());
  stop_ = false;
  receiver_ = std::thread([this] {
    std::vector<unsigned char> buffer;
    while (!stop_) {
      if (socket_->receive(buffer) == 0) continue;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        blocks_.push_back(buffer);
      }
      ready_.notify_one();
    }
  });
  nav_thread_ = std::thread([this] {
    std::vector<unsigned char> buffer;
    while (nav_socket_ && !stop_) {
      if (nav_socket_->receive(buffer) == 0) continue;
      const std::lock_guard<std::mutex> lock(mutex_);
      nav_.push(std::string(buffer.begin(), buffer.end()));
    }
  });
}

void UdpSourceOp::stop() {
  stop_ = true;
  if (receiver_.joinable()) receiver_.join();
  if (nav_thread_.joinable()) nav_thread_.join();
  std::cerr << "UdpSourceOp: " << emitted_ << " emitted, "
            << (pings_ ? pings_->dropped_pings : stream_ ? stream_->dropped_chunks : 0) << " dropped, "
            << (pings_ ? pings_->lost_samples : stream_ ? stream_->lost_samples : 0) << " samples lost, " << without_nav_
            << " without navigation\n";
}

void UdpSourceOp::compute(holoscan::InputContext&, holoscan::OutputContext& out, holoscan::ExecutionContext&) {
  // Wait for the next complete ping; one is emitted per call.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(idle_seconds_.get());
  std::vector<PingRecord> ready;
  std::vector<unsigned char> block;
  while (done_.empty()) {
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!ready_.wait_until(lock, deadline, [this] { return !blocks_.empty(); })) {
        stop_execution();                  // the stream has ended
        return;
      }
      block = std::move(blocks_.front());
      blocks_.pop_front();
    }
    ready.clear();
    if (stream_) {
      stream_->feed(block.data(), block.size(), ready);
    } else {
      pings_->feed(block.data(), block.size(), ready);
    }
    for (PingRecord& rec : ready) {
      if (nav_socket_ && !stream_) {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (!nav_.apply(rec)) ++without_nav_;
      }
      done_.push_back(std::move(rec));
    }
  }
  out.emit(std::make_shared<PingRecord>(std::move(done_.front())), "ping");
  done_.pop_front();
  ++emitted_;
  if (max_pings_.get() >= 0 && emitted_ >= max_pings_.get()) stop_execution();
}

// --- RecordSinkOp ------------------------------------------------------------------------

void RecordSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<PingMessage>("ping");
  spec.output<PingMessage>("ping_out");
  spec.param(path_, "path", "Recording", "Container file to write; empty to write nothing", std::string());
}

void RecordSinkOp::start() {
  if (!path_.get().empty()) writer_ = std::make_unique<io::RecordWriter>(path_.get());
}

void RecordSinkOp::stop() { writer_.reset(); }

void RecordSinkOp::compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext&) {
  auto message = in.receive<PingMessage>("ping");
  if (!message) return;
  const PingMessage ping = message.value();
  if (writer_) {
    writer_->write(*ping);                 // raw samples, before anything downstream touches them
    writer_->flush();
  }
  out.emit(ping, "ping_out");
}

// --- LiveChainOp -------------------------------------------------------------------------

void LiveChainOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<PingMessage>("ping");
  spec.output<ReportMessage>("report");
  spec.param(options_, "options", "Options", "Chain options as space-separated key=value pairs", std::string());
}

void LiveChainOp::start() {
  live::ChainOptions chain;
  live::PassiveOptions unused;
  apply_options(options_.get(), chain, unused);
  chain_ = std::make_unique<live::LiveChain>(chain);
}

void LiveChainOp::compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext&) {
  auto message = in.receive<PingMessage>("ping");
  if (!message) return;
  // The chain conditions the record in place; work on a copy so that a recorder or another
  // branch holding the same message keeps the raw samples.
  PingRecord record = *message.value();
  out.emit(std::make_shared<std::string>(chain_->process(record).json() + "\n"), "report");
}

// --- PassiveBearingsOp -------------------------------------------------------------------

void PassiveBearingsOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<PingMessage>("ping");
  spec.output<ReportMessage>("report");
  spec.param(options_, "options", "Options", "Passive options as space-separated key=value pairs", std::string());
}

void PassiveBearingsOp::start() {
  live::ChainOptions unused;
  passive_ = live::PassiveOptions{};
  apply_options(options_.get(), unused, passive_);
}

void PassiveBearingsOp::compute(holoscan::InputContext& in, holoscan::OutputContext& out, holoscan::ExecutionContext&) {
  auto message = in.receive<PingMessage>("ping");
  if (!message) return;
  const PingRecord& record = *message.value();
  out.emit(std::make_shared<std::string>(live::passive_json(live::passive_bearings(record, passive_), record.transmit.time_ns)), "report");
}

// --- ReportSinkOp ------------------------------------------------------------------------

void ReportSinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input<ReportMessage>("report");
  spec.param(path_, "path", "Report file", "JSON lines; empty for standard output", std::string());
  spec.param(udp_host_, "udp_host", "UDP host", "Also send each report here; empty for none", std::string());
  spec.param(udp_port_, "udp_port", "UDP port", "Port for udp_host", 0);
}

void ReportSinkOp::start() {
  if (!path_.get().empty()) file_.open(path_.get());
  if (!udp_host_.get().empty()) socket_ = std::make_unique<live::UdpSocket>("0.0.0.0", 0);
}

void ReportSinkOp::stop() {
  if (file_.is_open()) file_.close();
}

void ReportSinkOp::compute(holoscan::InputContext& in, holoscan::OutputContext&, holoscan::ExecutionContext&) {
  auto message = in.receive<ReportMessage>("report");
  if (!message) return;
  const std::string& text = *message.value();
  std::ostream& sink = file_.is_open() ? static_cast<std::ostream&>(file_) : std::cout;
  sink << text << std::flush;
  if (socket_) socket_->send_to(udp_host_.get(), udp_port_.get(), text);
}

}  // namespace sonar::holoscan_ops
