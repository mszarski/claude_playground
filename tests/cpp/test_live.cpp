// The C++ live library on its own: clustering, tracking, classification, SBL bearings and
// the board-stream assembler. tests/python/test_live_cpp.py and test_doa.py hold the same
// code against the Python stages and against simulated pings.
#include <cmath>
#include <complex>
#include <cstring>
#include <random>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/udp_source.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

TEST(Cluster, TouchingCellsBecomeOneDetectionAtTheirCentroid) {
  const std::int64_t rows = 5, cols = 8;
  std::vector<unsigned char> det(rows * cols, 0);
  std::vector<float> power(rows * cols, 1.0F), ratio(rows * cols, 1.0F);
  auto set = [&](std::int64_t r, std::int64_t c, float p) {
    det[r * cols + c] = 1;
    power[r * cols + c] = p;
    ratio[r * cols + c] = p;
  };
  set(1, 1, 10.0F);
  set(2, 2, 30.0F);   // touches (1, 1) diagonally
  set(4, 7, 100.0F);  // alone
  std::vector<double> ranges(cols), bearings(rows);
  for (std::int64_t c = 0; c < cols; ++c) ranges[c] = 100.0 + 2.0 * static_cast<double>(c);
  for (std::int64_t r = 0; r < rows; ++r) bearings[r] = 0.01 * static_cast<double>(r);
  const auto d = cluster(det.data(), power.data(), ratio.data(), rows, cols, ranges, bearings);
  ASSERT_EQ(d.size(), 2U);
  EXPECT_EQ(d[0].cells, 2);
  EXPECT_NEAR(d[0].range, (10.0 * 102.0 + 30.0 * 104.0) / 40.0, 1e-9);
  EXPECT_NEAR(d[0].bearing, (10.0 * 0.01 + 30.0 * 0.02) / 40.0, 1e-12);
  EXPECT_NEAR(d[0].snr_db, 10.0 * std::log10(30.0), 1e-6);
  EXPECT_EQ(d[1].cells, 1);
  EXPECT_NEAR(d[1].range, 114.0, 1e-9);
}

TEST(Tracker, FollowsAMoverAndAStandingObjectAndClassifiesThem) {
  TrackerOptions o;
  o.range_sigma = 1.5;
  o.bearing_sigma = 0.4 * kDeg;
  o.max_misses = 2;
  Tracker tracker(o);
  std::mt19937 gen(3);
  std::normal_distribution<double> noise(0.0, 0.2);
  std::vector<const Track*> confirmed;
  for (int i = 0; i < 10; ++i) {
    const double t = 0.5 * i;
    std::vector<Detection> dets;
    for (const auto& [x, y] : {std::pair<double, double>{200.0 + 4.0 * t, 30.0}, {150.0, -40.0}}) {
      const double px = x + noise(gen), py = y + noise(gen);
      dets.push_back({std::hypot(px, py), std::atan2(py, px), 20.0, 2});   // bearing from forward towards across
    }
    confirmed = tracker.update(t, dets);
  }
  ASSERT_EQ(confirmed.size(), 2U);
  ClassifyOptions c;
  const auto classes = classify_tracks(confirmed, c);
  ASSERT_EQ(classes.size(), 2U);
  // State is (across, forward): the mover advances along forward at 4 m/s.
  EXPECT_EQ(classes[0].kind, Kind::Mover);
  EXPECT_NEAR(classes[0].vy, 4.0, 0.4);
  EXPECT_NEAR(classes[0].vx, 0.0, 0.4);
  EXPECT_EQ(classes[1].kind, Kind::Static);
  EXPECT_LT(classes[1].speed, 0.5);
  EXPECT_NEAR(classes[0].seen, 1.0, 1e-12);
  EXPECT_STREQ(kind_name(classes[0].kind), "mover");
}

TEST(Tracker, DropsATrackThatIsNoLongerSeen) {
  TrackerOptions o;
  o.max_misses = 2;
  Tracker tracker(o);
  for (int i = 0; i < 3; ++i) tracker.update(1.0 * i, {{100.0, 0.1, 15.0, 1}});
  EXPECT_EQ(tracker.tracks().size(), 1U);
  for (int i = 3; i < 6; ++i) tracker.update(1.0 * i, {});
  EXPECT_TRUE(tracker.tracks().empty());
}

// Plane waves on a 32-element half-wavelength array at 16 frequencies, one snapshot each.
DoaFrame plane_waves(const std::vector<std::pair<double, double>>& sources, const std::vector<double>& u, double noise, unsigned seed) {
  DoaFrame frame;
  frame.n_channels = static_cast<std::int64_t>(u.size());
  frame.n_snapshots = 1;
  std::mt19937 gen(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int f = 0; f < 16; ++f) frame.frequencies.push_back(4000.0 + 125.0 * f);
  for (double freq : frame.frequencies) {
    std::vector<std::complex<double>> y(u.size(), 0.0);
    for (const auto& [bearing, amplitude] : sources) {
      const double k = 2.0 * kPi * freq * std::sin(bearing) / 1500.0;
      for (std::size_t n = 0; n < u.size(); ++n) y[n] += std::polar(amplitude, k * u[n] + 0.3 * freq);   // one waveform from every bearing: coherent
    }
    for (auto& v : y) v += std::complex<double>(g(gen), g(gen)) * (noise / std::sqrt(2.0));
    frame.snapshots.insert(frame.snapshots.end(), y.begin(), y.end());
  }
  return frame;
}

TEST(Sbl, SeparatesTwoCoherentArrivalsInsideOneBeam) {
  std::vector<double> u(32), grid;
  for (int n = 0; n < 32; ++n) u[n] = 0.15 * (n - 15.5);
  for (double b = -40.0; b <= 40.0 + 1e-9; b += 0.25) grid.push_back(b * kDeg);
  const DoaFrame frame = plane_waves({{2.0 * kDeg, 1.0}, {4.0 * kDeg, 0.8}}, u, 0.05, 1);
  DoaOptions o;
  o.dynamic_range_db = 15.0;
  const DoaResult r = sbl(frame, u, 1500.0, grid, o);
  ASSERT_EQ(r.peaks.size(), 2U);
  const double a = std::min(r.peaks[0].bearing, r.peaks[1].bearing) / kDeg, b = std::max(r.peaks[0].bearing, r.peaks[1].bearing) / kDeg;
  EXPECT_NEAR(a, 2.0, 0.4);
  EXPECT_NEAR(b, 4.0, 0.4);
  EXPECT_NEAR(r.peaks[0].power, 1.0, 0.35);       // the stronger one's power at an element
  EXPECT_NEAR(r.noise, 0.05 * 0.05, 0.004);
  EXPECT_LT(r.iterations, o.max_iterations);
  // The conventional spectrum of the same frame has one maximum between them.
  const DoaResult c = conventional(frame, u, 1500.0, grid, o);
  int maxima = 0;
  for (std::size_t m = 1; m + 1 < grid.size(); ++m) {
    if (grid[m] > 1.0 * kDeg && grid[m] < 5.0 * kDeg && c.spectrum[m] > c.spectrum[m - 1] && c.spectrum[m] >= c.spectrum[m + 1]) ++maxima;
  }
  EXPECT_EQ(maxima, 1);
}

TEST(Sbl, ReportsNothingForNoiseAboveItsFloor) {
  std::vector<double> u(32), grid;
  for (int n = 0; n < 32; ++n) u[n] = 0.15 * (n - 15.5);
  for (double b = -40.0; b <= 40.0 + 1e-9; b += 0.5) grid.push_back(b * kDeg);
  DoaOptions o;
  o.min_snr_db = 12.0;
  EXPECT_TRUE(sbl(plane_waves({}, u, 1.0, 2), u, 1500.0, grid, o).peaks.empty());
}

// One block of the board stream, as python/gpusonar/sources/flank_stream.py lays it out.
std::vector<unsigned char> block(std::int64_t first, std::int64_t n, std::int64_t channels, std::int64_t tx, std::uint32_t ping) {
  std::vector<unsigned char> b(128 + static_cast<std::size_t>(n * channels * 3), 0);
  std::memcpy(b.data(), "GSFB", 4);
  const std::uint16_t version = 1, ch = static_cast<std::uint16_t>(channels);
  const auto ns = static_cast<std::uint32_t>(n);
  const auto fi = static_cast<std::uint64_t>(first);
  const std::uint64_t start_ns = 1000;
  std::memcpy(b.data() + 4, &version, 2);
  std::memcpy(b.data() + 6, &ch, 2);
  std::memcpy(b.data() + 8, &ns, 4);
  std::memcpy(b.data() + 12, &fi, 8);
  std::memcpy(b.data() + 20, &tx, 8);
  std::memcpy(b.data() + 28, &ping, 4);
  std::memcpy(b.data() + 36, &start_ns, 8);
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t c = 0; c < channels; ++c) {
      const std::int32_t v = static_cast<std::int32_t>((first + i) * 10 + c) - 300;   // negative values too
      unsigned char* q = b.data() + 128 + (i * channels + c) * 3;
      q[0] = static_cast<unsigned char>(v & 0xFF);
      q[1] = static_cast<unsigned char>((v >> 8) & 0xFF);
      q[2] = static_cast<unsigned char>((v >> 16) & 0xFF);
    }
  }
  return b;
}

PingRecord stream_template() {
  PingRecord t;
  t.samples.resize(2, 40, SampleKind::RealPassband);
  t.samples.sample_rate = 1000.0;
  t.samples.start_time = 0.020;                   // the window opens 20 samples after the transmit
  t.metadata.adc_full_scale = 8388608.0;          // one volt per count, to read the samples back exactly
  t.geometry.rx.resize(2);
  return t;
}

TEST(PingAssembler, CutsTheStreamIntoPingsAndDropsOneThatLosesABlock) {
  PingAssembler assembler(stream_template());
  std::vector<PingRecord> out;
  // Transmits at samples 10 and 110; windows are samples 30..69 and 130..169; blocks of 16.
  for (std::int64_t first = 0; first < 200; first += 16) {
    if (first == 144) continue;                   // lost: it falls inside the second ping's window
    const std::int64_t tx = first + 16 > 110 ? 110 : (first + 16 > 10 ? 10 : -1);
    const auto b = block(first, 16, 2, tx, tx == 110 ? 1U : 0U);
    EXPECT_TRUE(assembler.feed(b.data(), b.size(), out));
  }
  ASSERT_EQ(out.size(), 1U);
  EXPECT_EQ(out[0].metadata.sequence, 0);
  EXPECT_EQ(out[0].samples.n_samples, 40);
  EXPECT_EQ(out[0].transmit.time_ns, 1000 + 10 * 1000000);
  EXPECT_FLOAT_EQ(out[0].samples.channel(0)[0], 30.0F * 10.0F - 300.0F);
  EXPECT_FLOAT_EQ(out[0].samples.channel(1)[39], 69.0F * 10.0F + 1.0F - 300.0F);
  EXPECT_EQ(assembler.dropped_pings, 1);
  EXPECT_EQ(assembler.lost_samples, 16);
  const unsigned char junk[4] = {1, 2, 3, 4};
  EXPECT_FALSE(assembler.feed(junk, 4, out));
  EXPECT_EQ(assembler.bad_blocks, 1);
}

TEST(StreamAssembler, CutsAnUntriggeredStreamIntoChunksAndRestartsAfterAGap) {
  StreamAssembler assembler(stream_template(), 40);
  std::vector<PingRecord> out;
  for (std::int64_t first = 0; first < 208; first += 16) {
    if (first == 96) continue;                    // lost: inside the third chunk (samples 80..119)
    const auto b = block(first, 16, 2, -1, 0U);
    EXPECT_TRUE(assembler.feed(b.data(), b.size(), out));
  }
  // Chunks 0..39 and 40..79 complete; 80..95 is discarded at the gap; then 112..151 and 152..191.
  ASSERT_EQ(out.size(), 4U);
  EXPECT_EQ(assembler.dropped_chunks, 1);
  EXPECT_EQ(assembler.lost_samples, 16);
  EXPECT_FLOAT_EQ(out[1].samples.channel(0)[0], 40.0F * 10.0F - 300.0F);
  EXPECT_FLOAT_EQ(out[2].samples.channel(1)[0], 112.0F * 10.0F + 1.0F - 300.0F);
  EXPECT_EQ(out[2].transmit.time_ns, 1000 + 112 * 1000000);
  EXPECT_EQ(out[3].samples.n_samples, 40);
  EXPECT_DOUBLE_EQ(out[3].samples.start_time, 0.0);
}

TEST(Navigation, InterpolatesThePoseToTheTransmitTime) {
  live::Navigation nav;
  PingRecord rec;
  EXPECT_FALSE(nav.apply(rec));
  EXPECT_FALSE(nav.push("{\"x\": 1.0}"));
  EXPECT_TRUE(nav.push("{\"time_ns\": 0, \"x\": 0.0, \"y\": 5.0, \"z\": 10.0, \"yaw\": 3.1, \"vx\": 1.5}"));
  EXPECT_TRUE(nav.push("{\"time_ns\": 2000000000, \"x\": 3.0, \"y\": 5.0, \"z\": 10.0, \"yaw\": -3.1, \"vx\": 1.5}"));
  rec.transmit.time_ns = 1000000000;
  EXPECT_TRUE(nav.apply(rec));
  EXPECT_NEAR(rec.geometry.rx_pose.position.x, 1.5, 1e-9);
  EXPECT_NEAR(rec.geometry.rx_pose.position.z, 10.0, 1e-9);
  EXPECT_NEAR(rec.geometry.tx_pose.velocity.x, 1.5, 1e-9);
  EXPECT_NEAR(std::abs(rec.geometry.rx_pose.rpy.z), kPi, 0.05);   // 3.1 to -3.1 goes through pi, not through zero
}

TEST(Report, IsOneJsonLineWithTheFieldsTheVehicleReads) {
  Report r;
  r.ping = 3;
  r.detections.push_back({1200.0, -4.5, 17.0});
  r.tracks.push_back({7, -1, "mover", 10.0, 20.0, -6.0, 0.1, 6.0, 1200.0, 18.0, 1.0});
  const std::string j = r.json();
  EXPECT_EQ(j.find('\n'), std::string::npos);
  for (const char* key : {"\"ping\": 3", "\"detections\"", "\"range_m\": 1200", "\"kind\": \"mover\"", "\"parent\": null", "\"sync\": null",
                          "\"processing_seconds\""}) {
    EXPECT_NE(j.find(key), std::string::npos) << key;
  }
}

}  // namespace
