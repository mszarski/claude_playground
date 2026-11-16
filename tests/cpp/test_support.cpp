#include "test_support.hpp"

#include <unistd.h>

#include <filesystem>
#include <random>

namespace sonar::test {

PingRecord make_full_record(std::int64_t channels, std::int64_t samples, SampleKind kind,
                            std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  PingRecord r;
  r.samples.resize(channels, samples, kind);
  for (auto& v : r.samples.data) v = static_cast<float>(u(rng));
  r.samples.sample_rate = 64000.0;
  r.samples.center_frequency = 8000.0;
  r.samples.start_time = 0.0125;

  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 6000.0;
  r.transmit.waveform.f_stop = 10000.0;
  r.transmit.waveform.duration = 0.25;
  r.transmit.waveform.window_alpha = 0.1;
  r.transmit.waveform.source_level_db = 190.0;
  r.transmit.waveform.amplitude = 3.5;
  r.transmit.time_ns = 1'700'000'000'123'456'789LL;
  r.transmit.kind = TransmitKind::Point;
  r.transmit.direction = {0.0, 0.6, 0.8};

  r.geometry.tx_pose = {{1.0, 2.0, -3.0}, {0.01, -0.02, 0.5}, {1.5, 0.1, -0.05}, {0.001, 0.002, -0.003}};
  r.geometry.rx_pose = {{1.1, 2.1, -3.1}, {0.02, -0.01, 0.4}, {1.4, 0.2, -0.06}, {0.004, 0.005, -0.006}};
  r.geometry.tx = {{0.5, 0.0, 0.1}, {0.0, 1.0, 0.0}, 1.0, 0.5};
  for (std::int64_t i = 0; i < channels; ++i) {
    Element e;
    e.lever_arm = {0.094 * static_cast<double>(i), 0.3, 0.05};
    e.boresight = {0.0, 0.8, 0.6};
    e.beamwidth_az = 0.7 + 0.01 * static_cast<double>(i);
    e.beamwidth_el = kNaN;
    r.geometry.rx.push_back(e);
  }
  r.geometry.frame_origin_lla = {-31.95, 115.86, 0.0};

  r.navigation.ins.push_back({0.0, {1, 2, 3}, {0.1, 0.2, 0.3}, {4, 5, 6}, {0.01, 0.02, 0.03}});
  r.navigation.ins.push_back({0.1, {1.1, 2.1, 3.1}, {0.1, 0.2, 0.3}, {4, 5, 6}, {0.01, 0.02, 0.03}});
  r.navigation.dvl.push_back({0.05, {1.5, 0.0, 0.01}, 12.5});
  r.navigation.gps.push_back({0.0, -31.95, 115.86, 1.0, 2.5});
  r.navigation.depth.push_back({0.02, 7.25});

  set_default_channel_tables(r);
  for (std::size_t i = 0; i < r.calibration.gain.size(); ++i) {
    r.calibration.gain[i] = 1.0 + 0.01 * static_cast<double>(i);
    r.calibration.phase[i] = 0.02 * static_cast<double>(i);
    r.calibration.delay[i] = 1e-7 * static_cast<double>(i);
    r.calibration.position_offset[i] = {1e-4 * static_cast<double>(i), 0.0, -1e-4};
  }

  r.environment.sound_speed = 1512.5;
  r.environment.temperature = 18.0;
  r.environment.salinity = 35.0;
  r.environment.water_depth = 40.0;
  r.environment.profile_depth = {0.0, 10.0, 40.0};
  r.environment.profile_speed = {1515.0, 1512.0, 1508.0};

  r.metadata.array_id = "flank_32";
  r.metadata.mode = Mode::Flank;
  r.metadata.sequence = 42;
  r.metadata.sub_sequence = 1;
  r.metadata.boards = {Board{1e-5, 3.5, 100.25, 0, 0}, Board{-2e-5, -1.5, 99.75, static_cast<std::int32_t>(channels - 1), 1}};
  for (std::size_t i = 0; i < r.metadata.channel_board.size(); ++i) {
    r.metadata.channel_board[i] = static_cast<std::int32_t>(i % 2);
    r.metadata.channel_element[i] = static_cast<std::int32_t>(2 * i + 1);
  }
  r.metadata.hardware_gain_db = 24.0;
  r.metadata.tvg_time = {0.0, 1.2e-4};
  r.metadata.tvg_gain_db = {0.0, 12.0};
  r.metadata.adc_bits = 24;
  r.metadata.adc_full_scale = 8.0;
  r.metadata.extra_json = R"({"note": "test"})";
  return r;
}

std::string temp_path(const std::string& stem) {
  static int counter = 0;
  const auto dir = std::filesystem::temp_directory_path();
  return (dir / (stem + "_" + std::to_string(::getpid()) + "_" + std::to_string(counter++) + ".h5"))
      .string();
}

}  // namespace sonar::test
