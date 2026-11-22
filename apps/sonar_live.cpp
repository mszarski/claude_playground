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
//   sbl_min_snr_db sbl_dynamic_range_db look (starboard | port) window (hann | rect)
//   backend (auto | host | cuda; auto, also without the key: CUDA when built with it and a device is present, else
//   the host; cuda without a usable device is an error)
//   tracker_range_sigma tracker_bearing_sigma_deg tracker_confirm_hits tracker_max_misses
//   tracker_association (likelihood | in_order)
//   classify_min_speed classify_static_speed classify_max_fit_rms classify_max_fit_rms_bearing_deg (the fit limit's
//   growth with range, degrees: a track's limit is at least this times its range; 0: off) classify_outlier_alpha
//   (opt-in: at most one measured point an outlier test finds at this probability is left out of a track's line; 0, the
//   default: off)
//   classify_min_mover_cells multipath (0 | 1)
//   sync (0 | 1) sync_f_start sync_f_stop sync_duration sync_lead sync_tail sync_dedicated (0 | 1)
//   passive: n_fft hop n_average passive_method (sbl | conventional)
//   passive tonal analysis (Goertzel lines, DEMON): tonal=1 and tonal_... (docs/CONFIGURATION.md, gpusonar/tonal.py)
//   stacked rows (elevation and depth; active and passive): elevation (0 | 1) elevation_first_deg
//   elevation_last_deg elevation_step_deg elevation_first_arrival (0 | 1) elevation_first_arrival_db
//   elevation_first_arrival_snr_db elevation_range_window_m elevation_max_detections
//   elevation_min_quality elevation_ambiguity_db elevation_correct_bearing elevation_depth
//   elevation_fold elevation_classify elevation_classify_sigma (0 | 1 each, or a number)
//   elevation_water_depth elevation_surface_z elevation_roll_sigma_deg elevation_depth_process_noise
//   elevation_ssp_depth elevation_ssp_speed (comma-separated lists) elevation_row_tolerance_m
//   dead_channels (comma-separated channel rows)
//   blanking of the projector's own sound: blanking (0 | 1) blanking_recovery_s blanking_taper_s
//   blanking_mode (zero | ignore); sync_min_quality (a reference burst below it is not used, the board flagged)
//   decimate first (docs/DEPLOYMENT.md): decimate (0 | 1) decimate_streaming (0 | 1: the receiver decimates as
//   blocks arrive and the recording holds the decimated pings; 0: the raw ping is recorded, the chain decimates)
//   decimate_f_low decimate_f_high decimate_band_margin decimate_output_rate decimate_oversample
//   decimate_ripple_db decimate_stopband_db decimate_taps decimate_blank_until_s decimate_blank_taper_s
//   navigation: nav_water_depth_from_altitude (0 | 1) nav_dvl_offset_m nav_altitude_sigma_m nav_depth_sigma_m
//   nav_hold_s nav_hold_sigma_m_per_s nav_hold_sigma_m_per_m nav_max_altitude_m nav_keep_s nav_profile_step_m
//   nav_surface_z (z of the sea surface in the navigation's frame: depth = z - nav_surface_z)
//   sound-speed ladder: elevation_profile (0 | 1) elevation_profile_ladder (comma-separated rungs)
//   elevation_profile_bin_m elevation_profile_sample_sigma elevation_profile_age_sigma_per_hour
//   elevation_profile_distance_sigma_per_km elevation_profile_max_age_s elevation_profile_max_distance_m
//   elevation_profile_min_span_m elevation_profile_extrapolation_sigma_per_s elevation_profile_file
//   elevation_profile_cast elevation_profile_supplied_sigma elevation_profile_family (isothermal | mixed_layer)
//   elevation_profile_family_gradient elevation_profile_layer_depth elevation_profile_thermocline_gradient
//   elevation_profile_thermocline_thickness elevation_profile_family_sigma
//   the program itself (docs/DEPLOYMENT.md, "Keys a new sonar must review"): bind (address the sockets listen on)
//   idle_seconds (end after this long without a block; 0: never) stream_sample_bytes (2 | 3 | 4: the boards' word)
//   stream_buffer_bytes (socket buffer; 0: from the data rate) stream_queue_mb (blocks waiting for the chain; 0: no bound)
//   report_udp_max_bytes (a longer report line goes out as several datagrams, each a complete line with "part": [k, n])
//   sync_search_time sync_tolerance_us elevation_min_row_elements nav_history_margin_s
// range_cell_m, cfar_guard_m, cfar_train_m, tracker_range_sigma, tracker_bearing_sigma_deg, multipath_bearing_deg,
// classify_max_fit_rms, classify_max_fit_rms_bearing_deg, elevation_row_tolerance_m, n_fft and hop are derived from the sonar when not given
// (libs/live derive.hpp); what was resolved is printed once on stderr.
//
// --template: a container file holding one ping record of the sonar (geometry, waveform,
// receive window); write it with `tools/sonar_live.py CONFIG --export-template template.h5`.
// Reports are JSON lines with the same fields as the Python chain's (tests/python/test_live_cpp.py
// test_report_fields_match compares them), pings_without_nav included.
// A value that is not what its key expects (a number, 0 | 1 | true | false, a comma-separated list, one of
// the key's words) is refused with the key named: libs/live option_value.hpp.
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
  // Room for every field to grow; a path that still does not fit is an error, never the unstamped path
  // (which a restart would write over).
  std::vector<char> out(2 * path.size() + 256);
  if (std::strftime(out.data(), out.size(), path.c_str(), &utc) == 0) {
    throw std::invalid_argument("output path too long to stamp with the start time: " + path);
  }
  return std::string(out.data());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    live::ChainOptions chain_options;
    live::PassiveOptions passive_options;
    live::program_defaults(chain_options, passive_options);
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

    // Passive report lines of one stretch of data; with tonal=1 the Goertzel lines and DEMON results too (tonal.hpp).
    std::unique_ptr<live::TonalAnalyzer> tonal;
    if (passive_options.tonal.enabled) {
      tonal = std::make_unique<live::TonalAnalyzer>(passive_options.tonal, passive_options.f_low, passive_options.f_high);
    }
    auto passive_lines = [&](const PingRecord& rec) {
      const auto frames = live::passive_bearings(rec, passive_options);
      if (!tonal) return live::passive_json(frames, rec.transmit.time_ns);
      return live::passive_tonal_json(rec, frames, live::passive_span(passive_options, rec), *tonal);
    };
    if (!passive.empty()) {
      io::RecordReader reader(passive);
      for (std::int64_t i = 0; i < reader.size() && (max_pings < 0 || i < max_pings); ++i) {
        PingRecord rec = reader.read(i);
        if (chain_options.decimate.enabled) {
          // Listening on the decimated stream: the band asked for, unless the options name another.
          live::DecimateOptions d = chain_options.decimate;
          if (d.f_low <= 0.0 && d.f_high <= 0.0) {
            d.f_low = passive_options.f_low;
            d.f_high = passive_options.f_high;
          }
          live::decimate_record(rec, d);
        }
        report << passive_lines(rec) << std::flush;
      }
      return 0;
    }

    live::LiveChain chain(chain_options);
    std::unique_ptr<io::RecordWriter> recorder;
    if (!recording_path.empty()) recorder = std::make_unique<io::RecordWriter>(recording_path);
    std::unique_ptr<live::UdpSocket> out_socket;
    const live::StreamOptions& stream_options = chain_options.stream;
    if (!report_host.empty()) out_socket = std::make_unique<live::UdpSocket>("0.0.0.0", 0);
    live::Navigation nav(chain_options.nav);
    live::NavCounts nav_counts;                 // pings without a fix, and without a velocity from navigation
    std::int64_t done = 0, split_reports = 0;
    // A report too long for one datagram goes out as several complete lines ("part": [k, n]); never cut.
    auto send_report = [&](const std::vector<std::string>& datagrams) {
      if (!out_socket) return;
      if (datagrams.size() > 1) ++split_reports;
      for (const std::string& d : datagrams) out_socket->send_to(report_host, report_port, d);
    };
    const auto report_bytes = static_cast<std::size_t>(std::max<std::int64_t>(stream_options.report_max_bytes, 1));
    // Navigation is applied by the caller (the UDP path, under its lock) before a ping is handled.
    auto handle = [&](PingRecord& rec) {
      if (recorder) {
        recorder->write(rec);        // raw samples, with the pose, before anything else touches them
        recorder->flush();
      }
      live::Report rep = chain.process(rec);
      rep.pings_without_nav = nav_counts.without_nav;     // as the Python chain reports it: pings so far without a fix, this one included
      rep.pings_without_velocity = nav_counts.without_velocity;   // pings so far whose fix had no velocity (sonar-8is)
      report << rep.json() << "\n" << std::flush;
      if (out_socket) send_report(rep.datagrams(report_bytes));
      if (done == 0) {
        std::fprintf(stderr, "sonar_live: chain settings %s\n", chain.resolved_json().c_str());
        // The detector's tests and thresholds for this map (cell_statistic multiscale; "{}" for mean and peak).
        std::fprintf(stderr, "sonar_live: detector %s\n", chain.detector_json().c_str());
      }
      ++done;
    };

    if (!replay.empty()) {
      io::RecordReader reader(replay);
      for (std::int64_t i = 0; i < reader.size() && (max_pings < 0 || done < max_pings); ++i) {
        PingRecord rec = reader.read(i);
        handle(rec);
      }
    } else if (udp_port >= 0) {
      if (template_path.empty()) throw std::invalid_argument("--udp needs --template");
      const PingRecord template_record = io::RecordReader(template_path).read(0);
      // decimate=1 decimate_streaming=1: every ping is decimated as its blocks arrive, and synchronised at
      // its end from the reference stretches kept aside; the recording then holds the decimated pings.
      const bool streaming = chain_options.decimate.enabled && chain_options.decimate_streaming;
      const live::BlankingOptions& blanking = chain_options.blanking;
      live::DecimateOptions stream_decimate = chain_options.decimate;
      if (blanking.enabled && blanking.zero) {       // blanked on the way into the filter
        stream_decimate.blank_until_s = blanking.until(template_record);
        stream_decimate.blank_taper_s = blanking.taper_s;
      }
      live::PingAssembler assembler(template_record, streaming ? &stream_decimate : nullptr,
                                    chain_options.sync ? &chain_options.reference : nullptr, stream_options.sample_bytes);
      live::DecimateOptions listen_decimate = chain_options.decimate;
      if (listen_decimate.f_low <= 0.0 && listen_decimate.f_high <= 0.0) {
        listen_decimate.f_low = passive_options.f_low;
        listen_decimate.f_high = passive_options.f_high;
      }
      // --listen: the stream is not cut on transmits; every stretch long enough for one passive
      // frame is recorded and solved for bearings as it completes.
      // (With decimate=1 the frame lengths n_fft and hop count decimated samples.)
      std::int64_t chunk = 1;
      if (listen) {
        PingRecord as_listened = template_record;       // the stretches' sample rate decides a derived block length
        if (listen_decimate.enabled) as_listened.samples.sample_rate = live::plan_decimation(template_record, listen_decimate).output_rate;
        live::resolve(passive_options, as_listened);    // fixed for the run: every stretch is cut to it
        chunk = passive_options.n_fft + (passive_options.n_average - 1) * passive_options.hop;
      }
      live::StreamAssembler stream(template_record, chunk, listen && listen_decimate.enabled ? &listen_decimate : nullptr,
                                   stream_options.sample_bytes);
      // The socket buffer follows the stream's data rate; what the system granted is said, and a shortfall warned of.
      const double rate = live::stream_bytes_per_second(template_record, stream_options.sample_bytes);
      const std::int64_t want = live::stream_buffer_bytes(template_record, stream_options.sample_bytes, stream_options.buffer_bytes);
      live::UdpSocket blocks(stream_options.bind, udp_port, 0.2, want);
      std::fprintf(stderr, "sonar_live: stream %.1f MB/s; socket buffer asked %lld bytes, granted %lld (%.2f s of stream)\n", rate / 1e6,
                   static_cast<long long>(blocks.buffer_asked()), static_cast<long long>(blocks.buffer_granted()),
                   rate > 0.0 ? static_cast<double>(blocks.buffer_granted()) / rate : 0.0);
      if (blocks.buffer_granted() < blocks.buffer_asked()) {
        std::fprintf(stderr, "sonar_live: WARNING the system granted a smaller socket buffer than asked: raise net.core.rmem_max to %lld "
                             "or blocks may be lost while a ping is processed\n", static_cast<long long>(blocks.buffer_asked()));
      }
      std::unique_ptr<live::UdpSocket> nav_socket;
      if (nav_port >= 0) nav_socket = std::make_unique<live::UdpSocket>(stream_options.bind, nav_port);
      // Listening on a platform that also pings: the same window, about every transmit the blocks mark.
      if (blanking.enabled && blanking.zero) stream.set_blanking(blanking.until(template_record), blanking.taper_s);
      std::fprintf(stderr, "sonar_live: listening for blocks on UDP port %d\n", blocks.port());
      // One thread does nothing but drain the block socket, so that processing a ping (tens to
      // hundreds of milliseconds) cannot overflow the socket buffer; another takes navigation.
      std::mutex mutex;
      std::condition_variable ready_cv;
      std::deque<std::vector<unsigned char>> queue;
      // The queue is bounded: when the chain falls behind for longer than it holds, arriving blocks are
      // dropped and counted (the assembler then sees a gap and drops the ping it falls in) instead of
      // memory growing until the process is killed.
      const auto queue_limit = static_cast<std::size_t>(stream_options.queue_mb * 1024.0 * 1024.0);
      std::size_t queued_bytes = 0;
      std::int64_t queue_dropped = 0;
      std::atomic<bool> stop{false};
      std::thread receiver([&] {
        std::vector<unsigned char> buffer;
        while (!stop) {
          if (blocks.receive(buffer) == 0) continue;
          {
            const std::lock_guard<std::mutex> lock(mutex);
            if (queue_limit > 0 && queued_bytes + buffer.size() > queue_limit) {
              ++queue_dropped;
              continue;
            }
            queued_bytes += buffer.size();
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
            // idle_seconds without a block: the stream has ended (0: wait for ever).
            if (stream_options.idle_seconds > 0.0 &&
                std::chrono::steady_clock::now() - last > std::chrono::duration<double>(stream_options.idle_seconds)) break;
            continue;
          }
          block = std::move(queue.front());
          queue.pop_front();
          queued_bytes -= block.size();
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
            const std::string lines = passive_lines(rec);
            report << lines << std::flush;
            send_report(live::line_datagrams(lines, report_bytes));
            ++done;
          }
          continue;
        }
        assembler.feed(block.data(), block.size(), ready);
        for (PingRecord& rec : ready) {
          if (nav_socket) {
            bool first_without_velocity = false;
            {
              const std::lock_guard<std::mutex> lock(mutex);
              first_without_velocity = nav_counts.apply(nav, rec);     // the fix and its velocity flag under one lock
            }
            // Navigation sent no velocity over ground: the backprojection takes the array at rest over the round trip.
            if (first_without_velocity) std::fprintf(stderr, "sonar_live: navigation sends no velocity (vx, vy, vz); counted in pings_without_velocity\n");
          }
          handle(rec);
        }
      }
      stop = true;
      receiver.join();
      nav_thread.join();
      if (listen) {
        std::fprintf(stderr, "sonar_live: %lld frames, %lld dropped, %lld samples lost, %lld bad blocks, %lld blocks dropped at the full queue, "
                             "%lld reports split, %lld report datagrams refused\n", static_cast<long long>(done),
                     static_cast<long long>(stream.dropped_chunks), static_cast<long long>(stream.lost_samples),
                     static_cast<long long>(stream.bad_blocks), static_cast<long long>(queue_dropped), static_cast<long long>(split_reports),
                     static_cast<long long>(out_socket ? out_socket->refused : 0));
        return 0;
      }
      std::fprintf(stderr, "sonar_live: %lld pings, %lld dropped, %lld samples lost, %lld bad blocks, %lld pings without navigation, "
                           "%lld pings without velocity, %lld blocks dropped at the full queue, %lld reports split, %lld report datagrams refused\n",
                   static_cast<long long>(done), static_cast<long long>(assembler.dropped_pings),
                   static_cast<long long>(assembler.lost_samples), static_cast<long long>(assembler.bad_blocks),
                   static_cast<long long>(nav_counts.without_nav), static_cast<long long>(nav_counts.without_velocity), static_cast<long long>(queue_dropped), static_cast<long long>(split_reports),
                   static_cast<long long>(out_socket ? out_socket->refused : 0));
    } else {
      throw std::invalid_argument("give --replay FILE, --udp PORT --template FILE, or --passive FILE");
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "sonar_live: %s\n", e.what());
    return 1;
  }
}
