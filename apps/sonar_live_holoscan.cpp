// sonar_live_holoscan: the long-range array chain as a Holoscan application of C++ operators.
// Same inputs, options and reports as build/bin/sonar_live (see the top of apps/sonar_live.cpp):
//
//   sonar_live_holoscan --replay recording.h5 [--report out.jsonl] [--recording copy.h5] [options]
//   sonar_live_holoscan --udp PORT --template template.h5 [--nav-port PORT] [--listen --band LOW HIGH]
//                       [--recording out.h5] [--report out.jsonl] [--report-udp HOST:PORT] [--pings N] [options]
//
// Operators: libs/holoscan_ops. Needs the Holoscan SDK (cmake -DSONAR_BUILD_HOLOSCAN=ON).
#include <cstdio>
#include <cstdlib>
#include <string>

#include <holoscan/holoscan.hpp>

#include "sonar/holoscan_ops/operators.hpp"
#include "sonar/io/record_file.hpp"
#include "sonar/live/chain.hpp"
#include "sonar/live/options.hpp"

namespace ops = sonar::holoscan_ops;

struct Settings {
  std::string replay, template_path, recording, report, report_host, options, stream_options;
  int udp_port = -1, nav_port = 0, report_port = 0;
  bool listen = false;
  int64_t max_pings = -1, chunk = 0, report_max_bytes = 65000;
  double idle_seconds = 30.0;
};

class LiveApp : public holoscan::Application {
 public:
  explicit LiveApp(Settings settings) : s_(std::move(settings)) {}

  void compose() override {
    using holoscan::Arg;
    std::shared_ptr<holoscan::Operator> source;
    if (!s_.replay.empty()) {
      source = make_operator<ops::ReplaySourceOp>("replay", Arg("path", s_.replay), Arg("max_pings", s_.max_pings));
    } else {
      source = make_operator<ops::UdpSourceOp>("boards", Arg("template_path", s_.template_path), Arg("port", s_.udp_port),
                                               Arg("nav_port", s_.nav_port), Arg("listen", s_.listen),
                                               Arg("chunk_samples", s_.chunk), Arg("max_pings", s_.max_pings),
                                               Arg("idle_seconds", s_.idle_seconds), Arg("options", s_.stream_options));
    }
    auto recorder = make_operator<ops::RecordSinkOp>("recorder", Arg("path", s_.recording));
    std::shared_ptr<holoscan::Operator> process;
    if (s_.listen) {
      process = make_operator<ops::PassiveBearingsOp>("passive", Arg("options", s_.options));
    } else {
      process = make_operator<ops::LiveChainOp>("chain", Arg("options", s_.options));
    }
    auto sink = make_operator<ops::ReportSinkOp>("report", Arg("path", s_.report), Arg("udp_host", s_.report_host),
                                                 Arg("udp_port", s_.report_port), Arg("max_bytes", s_.report_max_bytes));
    add_flow(source, recorder, {{"ping", "ping"}});
    add_flow(recorder, process, {{"ping_out", "ping"}});
    add_flow(process, sink, {{"report", "report"}});
  }

 private:
  Settings s_;
};

int main(int argc, char** argv) {
  try {
    Settings s;
    sonar::live::ChainOptions chain;
    sonar::live::PassiveOptions passive;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--replay") s.replay = next();
      else if (a == "--udp") s.udp_port = std::atoi(next().c_str());
      else if (a == "--template") s.template_path = next();
      else if (a == "--nav-port") s.nav_port = std::atoi(next().c_str());
      else if (a == "--listen") s.listen = true;
      else if (a == "--band") {
        s.options += " f_low=" + next();
        s.options += " f_high=" + next();
      } else if (a == "--recording") s.recording = next();
      else if (a == "--report") s.report = next();
      else if (a == "--report-udp") {
        const std::string hp = next();
        s.report_host = hp.substr(0, hp.find(':'));
        s.report_port = std::atoi(hp.substr(hp.find(':') + 1).c_str());
      } else if (a == "--pings") s.max_pings = std::atoll(next().c_str());
      else s.options += " " + a;
    }
    if (s.replay.empty() && (s.udp_port < 0 || s.template_path.empty())) {
      throw std::invalid_argument("give --replay FILE or --udp PORT --template FILE");
    }
    sonar::live::program_defaults(chain, passive);        // backend auto, as build/bin/sonar_live (D-147)
    ops::apply_options(s.options, chain, passive);        // fail on a bad option before the pipeline starts
    // The source operator takes one whole-array stream; per-board streams are not merged here (sonar-6f9).
    if (chain.stream.boards) {
      throw std::invalid_argument("stream_boards: the Holoscan source operator takes one whole-array stream; merge the boards' streams "
                                  "with tools/board_merge.py in front of it, or run build/bin/sonar_live");
    }
    // The stream's own settings go to the source operator; the frame length of a listening run follows from the
    // passive options and, for block sizes left to be derived, from the sonar in the template.
    s.idle_seconds = chain.stream.idle_seconds;
    s.report_max_bytes = chain.stream.report_max_bytes;
    s.stream_options = "stream_sample_bytes=" + std::to_string(chain.stream.sample_bytes) + " stream_buffer_bytes=" +
                       std::to_string(chain.stream.buffer_bytes) + " stream_queue_mb=" + std::to_string(chain.stream.queue_mb) +
                       " bind=" + chain.stream.bind + " nav_keep_s=" + std::to_string(chain.nav.keep_s) + " nav_history_margin_s=" +
                       std::to_string(chain.nav.history_margin_s);
    if (s.listen) s.chunk = sonar::live::passive_span(passive, sonar::io::RecordReader(s.template_path).read(0));
    auto app = holoscan::make_application<LiveApp>(s);
    app->run();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sonar_live_holoscan: %s\n", e.what());
    return 1;
  }
}
