// sonar_live: the long-range array chain as a standalone C++ program (no Python at run time).
//
//   sonar_live --replay recording.h5 [--report out.jsonl] [options]
//   sonar_live --udp PORT --template template.h5 [--nav-port PORT] [--recording out.h5]
//              [--report out.jsonl] [--report-udp HOST:PORT] [--pings N] [options]
//   sonar_live --passive recording.h5 --band F_LOW F_HIGH [--report out.jsonl] [options]
//
// Options are key=value pairs after the flags, or in a file given with --options, one per
// line (# starts a comment):
//   bearing_first_deg bearing_last_deg bearing_step_deg range_cell_m cfar_pfa cfar_guard_m
//   cfar_train_m min_snr_db doa (map | sbl) gate_fft sbl_grid_step_deg sbl_max_sources
//   sbl_min_snr_db sbl_dynamic_range_db look (starboard | port) backend (host | cuda)
//   tracker_range_sigma tracker_bearing_sigma_deg tracker_confirm_hits tracker_max_misses
//   classify_min_speed classify_max_fit_rms multipath (0 | 1)
//   sync (0 | 1) sync_f_start sync_f_stop sync_duration sync_lead sync_tail sync_dedicated (0 | 1)
//   passive: n_fft hop n_average passive_method (sbl | conventional)
//
// --template: a container file holding one ping record of the sonar (geometry, waveform,
// receive window); write it with `tools/sonar_live.py CONFIG --export-template template.h5`.
// Reports are JSON lines with the same fields as the Python chain's.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sonar/io/record_file.hpp"
#include "sonar/live/chain.hpp"
#include "sonar/live/udp_source.hpp"

using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kDeg = 3.14159265358979323846 / 180.0;

void set_option(live::ChainOptions& c, live::PassiveOptions& p, const std::string& key, const std::string& value) {
  const double v = std::atof(value.c_str());
  if (key == "bearing_first_deg") c.bearing_first_deg = p.bearing_first_deg = v;
  else if (key == "bearing_last_deg") c.bearing_last_deg = p.bearing_last_deg = v;
  else if (key == "bearing_step_deg") c.bearing_step_deg = p.bearing_step_deg = v;
  else if (key == "range_cell_m") c.range_cell_m = v;
  else if (key == "cfar_pfa") c.cfar_pfa = v;
  else if (key == "cfar_guard_m") c.cfar_guard_m = v;
  else if (key == "cfar_train_m") c.cfar_train_m = v;
  else if (key == "min_snr_db") c.min_snr_db = v;
  else if (key == "doa") c.doa = value;
  else if (key == "gate_fft") c.gate_fft = static_cast<std::int64_t>(v);
  else if (key == "sbl_grid_step_deg") c.sbl_grid_step_deg = v;
  else if (key == "sbl_max_iterations") c.sbl.max_iterations = p.sbl.max_iterations = static_cast<std::int32_t>(v);
  else if (key == "sbl_tolerance") c.sbl.tolerance = p.sbl.tolerance = v;
  else if (key == "sbl_max_gates") c.sbl_max_gates = static_cast<std::int64_t>(v);
  else if (key == "sbl_min_contrast_db") c.sbl_min_contrast_db = v;
  else if (key == "sbl_max_sources") c.sbl.max_sources = p.sbl.max_sources = static_cast<std::int32_t>(v);
  else if (key == "sbl_min_snr_db") c.sbl.min_snr_db = p.sbl.min_snr_db = v;
  else if (key == "sbl_dynamic_range_db") c.sbl.dynamic_range_db = p.sbl.dynamic_range_db = v;
  else if (key == "look") c.starboard = value != "port";
  else if (key == "backend") c.backend = value == "cuda" ? Backend::Cuda : Backend::Host;
  else if (key == "tracker_range_sigma") c.tracker.range_sigma = v;
  else if (key == "tracker_bearing_sigma_deg") c.tracker.bearing_sigma = v * kDeg;
  else if (key == "tracker_confirm_hits") c.tracker.confirm_hits = static_cast<std::int64_t>(v);
  else if (key == "tracker_max_misses") c.tracker.max_misses = static_cast<std::int64_t>(v);
  else if (key == "classify_min_speed") c.classify.min_speed = c.classify.static_speed = v;
  else if (key == "classify_static_speed") c.classify.static_speed = v;   // after classify_min_speed, which sets both
  else if (key == "classify_max_fit_rms") c.classify.max_fit_rms = v;
  else if (key == "tracker_initial_speed_sigma") c.tracker.initial_speed_sigma = v;
  else if (key == "tracker_process_noise") c.tracker.process_noise = v;
  else if (key == "multipath_max_excess_range") c.classify.multipath_max_excess_range = v;
  else if (key == "multipath_bearing_deg") c.classify.multipath_bearing = v * kDeg;
  else if (key == "multipath_velocity") c.classify.multipath_velocity = v;
  else if (key == "window") c.hann = value == "hann";
  else if (key == "multipath") c.classify.multipath = v != 0.0;
  else if (key == "sync") c.sync = v != 0.0;
  else if (key == "sync_f_start") c.reference.f_start = v;
  else if (key == "sync_f_stop") c.reference.f_stop = v;
  else if (key == "sync_duration") c.reference.duration = v;
  else if (key == "sync_lead") c.reference.lead = v;
  else if (key == "sync_tail") c.reference.tail = v;
  else if (key == "sync_dedicated") c.reference.dedicated = v != 0.0;
  else if (key == "n_fft") p.n_fft = static_cast<std::int64_t>(v);
  else if (key == "hop") p.hop = static_cast<std::int64_t>(v);
  else if (key == "n_average") p.n_average = static_cast<std::int64_t>(v);
  else if (key == "passive_method") p.use_sbl = value != "conventional";
  else throw std::invalid_argument("unknown option " + key);
}

void set_pair(live::ChainOptions& c, live::PassiveOptions& p, const std::string& pair) {
  const auto eq = pair.find('=');
  if (eq == std::string::npos) throw std::invalid_argument("expected key=value, got " + pair);
  set_option(c, p, pair.substr(0, eq), pair.substr(eq + 1));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    live::ChainOptions chain_options;
    live::PassiveOptions passive_options;
    std::string replay, passive, template_path, report_path, recording_path, report_host;
    int udp_port = -1, nav_port = -1, report_port = 0;
    std::int64_t max_pings = -1;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--replay") replay = next();
      else if (a == "--passive") passive = next();
      else if (a == "--band") {
        passive_options.f_low = std::atof(next().c_str());
        passive_options.f_high = std::atof(next().c_str());
      } else if (a == "--udp") udp_port = std::atoi(next().c_str());
      else if (a == "--template") template_path = next();
      else if (a == "--nav-port") nav_port = std::atoi(next().c_str());
      else if (a == "--report") report_path = next();
      else if (a == "--recording") recording_path = next();
      else if (a == "--report-udp") {
        const std::string hp = next();
        report_host = hp.substr(0, hp.find(':'));
        report_port = std::atoi(hp.substr(hp.find(':') + 1).c_str());
      } else if (a == "--pings") max_pings = std::atoll(next().c_str());
      else if (a == "--options") {
        std::ifstream f(next());
        std::string line;
        while (std::getline(f, line)) {
          line = line.substr(0, line.find('#'));
          line.erase(0, line.find_first_not_of(" \t"));
          line.erase(line.find_last_not_of(" \t\r") + 1);
          if (!line.empty()) set_pair(chain_options, passive_options, line);
        }
      } else if (a == "--help" || a == "-h") {
        std::cout << "see the comment at the top of apps/sonar_live.cpp and docs/DEPLOYMENT.md\n";
        return 0;
      } else set_pair(chain_options, passive_options, a);
    }
    std::ofstream report_file;
    if (!report_path.empty()) report_file.open(report_path);
    std::ostream& report = report_path.empty() ? std::cout : report_file;

    if (!passive.empty()) {
      io::RecordReader reader(passive);
      for (std::int64_t i = 0; i < reader.size() && (max_pings < 0 || i < max_pings); ++i) {
        const PingRecord rec = reader.read(i);
        report << live::passive_json(live::passive_bearings(rec, passive_options), rec.transmit.time_ns) << std::flush;
      }
      return 0;
    }

    live::LiveChain chain(chain_options);
    std::unique_ptr<io::RecordWriter> recorder;
    if (!recording_path.empty()) recorder = std::make_unique<io::RecordWriter>(recording_path);
    std::unique_ptr<live::UdpSocket> out_socket;
    if (!report_host.empty()) out_socket = std::make_unique<live::UdpSocket>("0.0.0.0", 0);
    live::Navigation nav;
    std::int64_t done = 0, without_nav = 0;
    auto handle = [&](PingRecord& rec, bool use_nav) {
      if (use_nav && !nav.apply(rec)) ++without_nav;
      if (recorder) {
        recorder->write(rec);        // raw samples, with the pose, before anything else touches them
        recorder->flush();
      }
      const std::string line = chain.process(rec).json();
      report << line << "\n" << std::flush;
      if (out_socket) out_socket->send_to(report_host, report_port, line);
      ++done;
    };

    if (!replay.empty()) {
      io::RecordReader reader(replay);
      for (std::int64_t i = 0; i < reader.size() && (max_pings < 0 || done < max_pings); ++i) {
        PingRecord rec = reader.read(i);
        handle(rec, false);
      }
    } else if (udp_port >= 0) {
      if (template_path.empty()) throw std::invalid_argument("--udp needs --template");
      live::PingAssembler assembler(io::RecordReader(template_path).read(0));
      live::UdpSocket blocks("0.0.0.0", udp_port);
      std::unique_ptr<live::UdpSocket> nav_socket;
      if (nav_port >= 0) nav_socket = std::make_unique<live::UdpSocket>("0.0.0.0", nav_port);
      std::fprintf(stderr, "sonar_live: listening for blocks on UDP port %d\n", blocks.port());
      // One thread does nothing but drain the block socket, so that processing a ping (tens to
      // hundreds of milliseconds) cannot overflow the socket buffer; another takes navigation.
      std::mutex mutex;
      std::condition_variable ready_cv;
      std::deque<std::vector<unsigned char>> queue;
      std::atomic<bool> stop{false};
      std::thread receiver([&] {
        std::vector<unsigned char> buffer;
        while (!stop) {
          if (blocks.receive(buffer) == 0) continue;
          {
            const std::lock_guard<std::mutex> lock(mutex);
            queue.push_back(buffer);
          }
          ready_cv.notify_one();
        }
      });
      std::thread nav_thread([&] {
        std::vector<unsigned char> buffer;
        while (nav_socket && !stop) {
          if (nav_socket->receive(buffer) == 0) continue;
          const std::lock_guard<std::mutex> lock(mutex);
          nav.push(std::string(buffer.begin(), buffer.end()));
        }
      });
      std::vector<PingRecord> ready;
      std::vector<unsigned char> block;
      auto last = std::chrono::steady_clock::now();
      while (max_pings < 0 || done < max_pings) {
        {
          std::unique_lock<std::mutex> lock(mutex);
          if (!ready_cv.wait_for(lock, std::chrono::milliseconds(200), [&] { return !queue.empty(); })) {
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(30)) break;   // the stream has ended
            continue;
          }
          block = std::move(queue.front());
          queue.pop_front();
        }
        last = std::chrono::steady_clock::now();
        ready.clear();
        assembler.feed(block.data(), block.size(), ready);
        for (PingRecord& rec : ready) {
          bool have_nav = true;
          if (nav_socket) {
            const std::lock_guard<std::mutex> lock(mutex);
            have_nav = nav.apply(rec);
          }
          if (!have_nav) ++without_nav;
          handle(rec, false);
        }
      }
      stop = true;
      receiver.join();
      nav_thread.join();
      std::fprintf(stderr, "sonar_live: %lld pings, %lld dropped, %lld samples lost, %lld bad blocks, %lld pings without navigation\n",
                   static_cast<long long>(done), static_cast<long long>(assembler.dropped_pings),
                   static_cast<long long>(assembler.lost_samples), static_cast<long long>(assembler.bad_blocks),
                   static_cast<long long>(without_nav));
    } else {
      throw std::invalid_argument("give --replay FILE, --udp PORT --template FILE, or --passive FILE");
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sonar_live: %s\n", e.what());
    return 1;
  }
}
