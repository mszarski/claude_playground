// sonar_live: the long-range array chain as a standalone C++ program (no Python at run time).
//
//   sonar_live --replay recording.h5 [--report out.jsonl] [options]
//   sonar_live --udp PORT --template template.h5 --listen --band F_LOW F_HIGH [--recording out.h5]
//                                                  passive bearings from the untriggered board stream
//   sonar_live --udp PORT --template template.h5 [--nav-port PORT] [--recording out.h5]
//              [--report out.jsonl] [--report-udp HOST:PORT] [--pings N] [options]
//   sonar_live --passive recording.h5 --band F_LOW F_HIGH [--report out.jsonl] [options]
//
// Options are key=value pairs after the flags, or in a file given with --options, one per
// line (# starts a comment):
//   bearing_first_deg bearing_last_deg bearing_step_deg range_cell_m cfar_pfa cfar_guard_m
//   cfar_train_m min_snr_db beamformer (backprojection | farfield | mvdr) doa (map | sbl)
//   gate_fft sbl_grid_step_deg sbl_max_sources
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
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sonar/io/record_file.hpp"
#include "sonar/live/chain.hpp"
#include "sonar/live/options.hpp"
#include "sonar/live/udp_source.hpp"

using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {
// strftime fields in an output path (%Y%m%d_%H%M%S), filled with the start time (UTC), so that
// a restart never writes over an earlier run's recording.
std::string stamped(const std::string& path) {
  if (path.find('%') == std::string::npos) return path;
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char out[1024];
  return std::strftime(out, sizeof(out), path.c_str(), &utc) > 0 ? std::string(out) : path;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    live::ChainOptions chain_options;
    live::PassiveOptions passive_options;
    bool listen = false;
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
      else if (a == "--listen") listen = true;
      else if (a == "--nav-port") nav_port = std::atoi(next().c_str());
      else if (a == "--report") report_path = stamped(next());
      else if (a == "--recording") recording_path = stamped(next());
      else if (a == "--report-udp") {
        const std::string hp = next();
        report_host = hp.substr(0, hp.find(':'));
        report_port = std::atoi(hp.substr(hp.find(':') + 1).c_str());
      } else if (a == "--pings") max_pings = std::atoll(next().c_str());
      else if (a == "--options") {
        live::load_options(chain_options, passive_options, next());
      } else if (a == "--help" || a == "-h") {
        std::cout << "see the comment at the top of apps/sonar_live.cpp and docs/DEPLOYMENT.md\n";
        return 0;
      } else live::set_pair(chain_options, passive_options, a);
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
      const PingRecord template_record = io::RecordReader(template_path).read(0);
      live::PingAssembler assembler(template_record);
      // --listen: the stream is not cut on transmits; every stretch long enough for one passive
      // frame is recorded and solved for bearings as it completes.
      const std::int64_t chunk = passive_options.n_fft + (passive_options.n_average - 1) * passive_options.hop;
      live::StreamAssembler stream(template_record, listen ? chunk : 1);
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
        if (listen) {
          stream.feed(block.data(), block.size(), ready);
          for (PingRecord& rec : ready) {
            if (recorder) {
              recorder->write(rec);
              recorder->flush();
            }
            const std::string lines = live::passive_json(live::passive_bearings(rec, passive_options), rec.transmit.time_ns);
            report << lines << std::flush;
            if (out_socket) out_socket->send_to(report_host, report_port, lines);
            ++done;
          }
          continue;
        }
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
      if (listen) {
        std::fprintf(stderr, "sonar_live: %lld frames, %lld dropped, %lld samples lost, %lld bad blocks\n", static_cast<long long>(done),
                     static_cast<long long>(stream.dropped_chunks), static_cast<long long>(stream.lost_samples),
                     static_cast<long long>(stream.bad_blocks));
        return 0;
      }
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
