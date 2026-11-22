#include "sonar/holoscan_ops/operators.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>

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
  // No default port and no default stretch length: the application that hosts the operator names them
  // (a default would be one network plan's, or a frame length unrelated to the passive options).
  spec.param(port_, "port", "Port", "UDP port the boards send blocks to (required; 0: any free port)", -1);
  spec.param(nav_port_, "nav_port", "Navigation port", "UDP port of navigation datagrams; 0 for none", 0);
  spec.param(listen_, "listen", "Listen", "Untriggered stream: cut into stretches of chunk_samples", false);
  spec.param(chunk_samples_, "chunk_samples", "Chunk", "Samples per stretch when listening (required then: live::passive_span)",
             static_cast<int64_t>(0));
  spec.param(max_pings_, "max_pings", "Pings", "Stop after this many (negative: never)", static_cast<int64_t>(-1));
  spec.param(idle_seconds_, "idle_seconds", "Idle time", "Stop when no block has arrived for this long (0: never)", 30.0);
  spec.param(options_, "options", "Options",
             "Stream options as space-separated key=value pairs: stream_sample_bytes, stream_buffer_bytes, stream_queue_mb, bind, "
             "and the nav_... keys",
             std::string());
}

void UdpSourceOp::start() {
  if (port_.get() < 0) throw std::invalid_argument("UdpSourceOp: port is required");
  if (listen_.get() && chunk_samples_.get() < 1) throw std::invalid_argument("UdpSourceOp: chunk_samples is required when listening");
  live::ChainOptions chain;
  live::PassiveOptions unused;
  apply_options(options_.get(), chain, unused);
  const live::StreamOptions& so = chain.stream;
  nav_ = live::Navigation(chain.nav);
  const PingRecord template_record = io::RecordReader(template_path_.get()).read(0);
  if (listen_.get()) {
    stream_ = std::make_unique<live::StreamAssembler>(template_record, chunk_samples_.get(), nullptr, so.sample_bytes);
  } else {
    pings_ = std::make_unique<live::PingAssembler>(template_record, nullptr, nullptr, so.sample_bytes);
  }
  // The socket buffer follows the stream's data rate; a shortfall in what the system grants is said.
  socket_ = std::make_unique<live::UdpSocket>(so.bind, port_.get(), 0.2,
                                              live::stream_buffer_bytes(template_record, so.sample_bytes, so.buffer_bytes));
  std::cerr << "UdpSourceOp: socket buffer asked " << socket_->buffer_asked() << " bytes, granted " << socket_->buffer_granted()
            << (socket_->buffer_granted() < socket_->buffer_asked() ? " (WARNING: less than asked; raise net.core.rmem_max)" : "") << "\n";
  if (nav_port_.get() > 0) nav_socket_ = std::make_unique<live::UdpSocket>(so.bind, nav_port_.get());
  queue_limit_ = static_cast<std::size_t>(so.queue_mb * 1024.0 * 1024.0);
  stop_ = false;
  receiver_ = std::thread([this] {
    std::vector<unsigned char> buffer;
    while (!stop_) {
      if (socket_->receive(buffer) == 0) continue;
      {
        const std::lock_guard<std::mutex> lock(mutex_);
        if (queue_limit_ > 0 && queued_bytes_ + buffer.size() > queue_limit_) {
          ++queue_dropped_;                 // bounded: the assembler sees the gap and drops the ping it falls in
          continue;
        }
        queued_bytes_ += buffer.size();
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
            << (pings_ ? pings_->lost_samples : stream_ ? stream_->lost_samples : 0) << " samples lost, " << nav_counts_.without_nav
            << " without navigation, " << nav_counts_.without_velocity << " without velocity, " << queue_dropped_ << " blocks dropped at the full queue\n";
}

void UdpSourceOp::compute(holoscan::InputContext&, holoscan::OutputContext& out, holoscan::ExecutionContext&) {
  // Wait for the next complete ping; one is emitted per call.
  // idle_seconds 0: wait for ever (a board that sends only inside its receive windows, with any ping interval).
  const double idle = idle_seconds_.get() > 0.0 ? idle_seconds_.get() : 1e9;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(idle));
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
      queued_bytes_ -= block.size();
    }
    ready.clear();
    if (stream_) {
      stream_->feed(block.data(), block.size(), ready);
    } else {
      pings_->feed(block.data(), block.size(), ready);
    }
    for (PingRecord& rec : ready) {
      if (nav_socket_ && !stream_) {
        bool first_without_velocity = false;
        {
          const std::lock_guard<std::mutex> lock(mutex_);
          first_without_velocity = nav_counts_.apply(nav_, rec);   // the fix and its velocity flag under one lock (sonar-8is)
        }
        if (first_without_velocity) std::cerr << "UdpSourceOp: navigation sends no velocity (vx, vy, vz); counted as pings without velocity\n";
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
  live::program_defaults(chain, unused);         // backend auto, as build/bin/sonar_live starts (D-147)
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
  live::program_defaults(unused, passive_);      // backend auto, as build/bin/sonar_live starts (D-147)
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
  spec.param(max_bytes_, "max_bytes", "Datagram size", "Lines are grouped into datagrams of at most this many bytes", static_cast<int64_t>(65000));
}

void ReportSinkOp::start() {
  if (!path_.get().empty()) file_.open(path_.get());
  if (!udp_host_.get().empty()) socket_ = std::make_unique<live::UdpSocket>("0.0.0.0", 0);
}

void ReportSinkOp::stop() {
  if (file_.is_open()) file_.close();
  if (socket_ && socket_->refused > 0) {
    std::cerr << "ReportSinkOp: " << socket_->refused << " report lines were too long for a datagram and were not sent by UDP "
              << "(they are in the report file; build/bin/sonar_live splits such lines)\n";
  }
}

void ReportSinkOp::compute(holoscan::InputContext& in, holoscan::OutputContext&, holoscan::ExecutionContext&) {
  auto message = in.receive<ReportMessage>("report");
  if (!message) return;
  const std::string& text = *message.value();
  std::ostream& sink = file_.is_open() ? static_cast<std::ostream&>(file_) : std::cout;
  sink << text << std::flush;
  // Whole lines in each datagram. One line longer than a datagram is refused and counted, never cut
  // (this sink sees text, not the report's lists: it cannot split a line as live::Report::datagrams does).
  if (socket_) {
    for (const std::string& d : live::line_datagrams(text, static_cast<std::size_t>(std::max<int64_t>(max_bytes_.get(), 1)))) {
      socket_->send_to(udp_host_.get(), udp_port_.get(), d);
    }
  }
}

}  // namespace sonar::holoscan_ops
