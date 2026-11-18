// The chain's derived settings and its capacity limits, in C++ on records built here:
//   - the standing values come out to the last bit for the arrays shipped today (4-6 kHz and 6-10 kHz
//     rows of 32, a 20-35 kHz row of 32), and a narrower band, a wider beam, shorter wavelengths or a
//     longer array move the settings the way the formulas in derive.hpp say;
//   - a value given in the options wins;
//   - SBL on more channels than its kernels take runs on the host reference and says so, with the
//     reference's answer;
//   - passive bearings on stacked rows without the elevation option come from the rows as snapshots;
//   - a report line too long for a datagram becomes several complete lines, never a cut one;
//   - blocks of 2-, 3- and 4-byte samples; the socket buffer from the data rate; the sync tolerance.
// tests/python/test_hardcoded.py holds the same against simulated pings, both engines and the program.
#include <cmath>
#include <complex>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/derive.hpp"
#include "sonar/live/doa.hpp"
#include "sonar/live/sbl_kernel.hpp"
#include "sonar/live/udp_source.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

// Rows of `per_row` hydrophones along body x at `pitch`, stacked at the given heights (m up), real samples.
PingRecord array_record(std::int64_t per_row, double pitch, const std::vector<double>& up, double f_low, double f_high, double pulse,
                        double fs, std::int64_t n_samples, double c = 1500.0) {
  PingRecord r;
  r.samples.resize(per_row * static_cast<std::int64_t>(up.size()), n_samples, SampleKind::RealPassband);
  r.samples.sample_rate = fs;
  r.samples.start_time = 0.2;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = f_low;
  r.transmit.waveform.f_stop = f_high;
  r.transmit.waveform.duration = pulse;
  r.environment.sound_speed = c;
  for (double z : up) {
    for (std::int64_t e = 0; e < per_row; ++e) {
      Element el;
      el.lever_arm = Vec3d{(static_cast<double>(e) - 0.5 * static_cast<double>(per_row - 1)) * pitch, 0.0, -z};
      el.boresight = Vec3d{0.0, 1.0, 0.0};       // stated: the record has no default
      r.geometry.rx.push_back(el);
    }
  }
  r.geometry.tx_pose = r.geometry.rx_pose;
  set_default_channel_tables(r);
  return r;
}

// A far source of noise in a band, as a sum of tones with random phases: element i hears it earlier by (p_i . u) / c.
void add_far_source(PingRecord& r, const Vec3d& towards, double f_low, double f_high, unsigned seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi), freq(f_low, f_high);
  const double c = r.environment.sound_speed, fs = r.samples.sample_rate;
  for (int k = 0; k < 60; ++k) {
    const double f = freq(gen), ph = phase(gen);
    for (std::int64_t e = 0; e < r.samples.n_channels; ++e) {
      const double lead = dot(r.geometry.rx[static_cast<std::size_t>(e)].lever_arm, towards) / c;
      float* x = r.samples.channel(e);
      for (std::int64_t i = 0; i < r.samples.n_samples; ++i) {
        x[i] += static_cast<float>(std::cos(2.0 * kPi * f * (static_cast<double>(i) / fs + lead) + ph));
      }
    }
  }
  std::normal_distribution<float> noise(0.0F, 0.5F);
  for (float& v : r.samples.data) v += noise(gen);
}

}  // namespace

TEST(Derive, StandingValuesToTheLastBitForTheArraysShippedToday) {
  struct Case {
    std::int64_t n;
    double pitch, f_low, f_high, pulse, c;
  };
  // The long-range row, the test vehicle's row, the flank array, a 20-35 kHz row, each at three sound speeds.
  for (const Case& k : {Case{32, 0.15, 4e3, 6e3, 1.0, 1500.0}, Case{32, 0.15, 4e3, 6e3, 0.1, 1440.0}, Case{32, 0.09, 4e3, 6e3, 0.1, 1540.0},
                        Case{32, 0.094, 6e3, 10e3, 1.0, 1500.0}, Case{32, 0.094, 6e3, 10e3, 1.0, 1460.0}, Case{32, 0.025, 20e3, 35e3, 0.5, 1500.0}}) {
    const PingRecord rec = array_record(k.n, k.pitch, {0.0}, k.f_low, k.f_high, k.pulse, 64000.0, 16, k.c);
    ChainOptions o;
    const ChainDefaults d = resolve(o, rec);
    EXPECT_EQ(o.range_cell_m, 2.0);
    EXPECT_EQ(o.cfar_guard_m, 100.0);
    EXPECT_EQ(o.cfar_train_m, 300.0);
    EXPECT_EQ(o.tracker.range_sigma, 15.0);
    EXPECT_EQ(o.tracker.bearing_sigma, 0.017453292519943295);
    EXPECT_EQ(o.classify.multipath_bearing, 0.05235987755982988);
    EXPECT_EQ(o.classify.max_fit_rms, 60.0);
    EXPECT_EQ(o.elevation.row_tolerance_m, 0.02 < 0.25 * k.c / k.f_high ? 0.02 : 0.25 * k.c / k.f_high);
    EXPECT_TRUE(d.given.empty());
    EXPECT_EQ(d.facts.row_elements, k.n);
    EXPECT_NEAR(d.facts.aperture, static_cast<double>(k.n) * k.pitch, 1e-9);
  }
}

TEST(Derive, ANarrowBandAndAWideBeamScaleTheSettings) {
  // Eight hydrophones half a metre apart, a 50 ms pulse at 1 kHz that does not sweep, in fresh water:
  // 37 m of range resolution and a 21 degree beam.
  const double c = 1480.0, pulse = 0.05;
  PingRecord rec = array_record(8, 0.5, {0.0}, 1000.0, 1000.0, pulse, 8000.0, 16, c);
  ChainOptions o;
  const ChainDefaults d = resolve(o, rec);
  const double resolution = c / (2.0 * (1.0 / pulse)), beam = (c / 1000.0) / (8 * 0.5);
  EXPECT_DOUBLE_EQ(d.facts.range_resolution, resolution);
  EXPECT_DOUBLE_EQ(o.range_cell_m, resolution);                 // one resolution, not 2 m
  EXPECT_DOUBLE_EQ(o.cfar_guard_m, 4.0 * resolution);            // 148 m, not 100
  EXPECT_DOUBLE_EQ(o.cfar_train_m, 12.0 * resolution);           // three guards
  EXPECT_DOUBLE_EQ(o.tracker.range_sigma, resolution);           // 37 m, not 15
  EXPECT_DOUBLE_EQ(o.classify.max_fit_rms, 4.0 * resolution);
  EXPECT_DOUBLE_EQ(o.tracker.bearing_sigma, beam / 16.0);        // 1.32 deg, not 1
  EXPECT_DOUBLE_EQ(o.classify.multipath_bearing, 3.0 * beam / 16.0);
  EXPECT_NEAR(o.tracker.bearing_sigma / kDeg, 1.325, 0.01);
  EXPECT_EQ(d.scaled.size(), 7U);
  // Twice the band: half the floors, down to the standing values.
  rec.transmit.waveform.duration = 0.5 * pulse;
  ChainOptions half;
  resolve(half, rec);
  EXPECT_DOUBLE_EQ(half.range_cell_m, 0.5 * resolution);
  EXPECT_DOUBLE_EQ(half.cfar_guard_m, 100.0);                    // 74 m is below the standing 100
  EXPECT_DOUBLE_EQ(half.cfar_train_m, 300.0);
  // A setting in between: the same pulse swept over 300 Hz has 2.47 m of resolution, so only the cell moves.
  rec.transmit.waveform.f_stop = 1300.0;
  ChainOptions swept;
  const ChainDefaults ds = resolve(swept, rec);
  EXPECT_NEAR(swept.range_cell_m, c / 600.0, 1e-12);
  EXPECT_EQ(swept.cfar_guard_m, 100.0);
  EXPECT_EQ(swept.tracker.range_sigma, 15.0);
  EXPECT_EQ(ds.scaled.size(), 3U);                               // the cell and the two bearing settings
}

TEST(Derive, AGivenValueWinsAndIsListed) {
  const PingRecord rec = array_record(8, 0.5, {0.0}, 1000.0, 1000.0, 0.05, 8000.0, 16, 1480.0);
  ChainOptions o;
  o.range_cell_m = 5.0;
  o.cfar_guard_m = 40.0;
  o.tracker.bearing_sigma = 0.5 * kDeg;
  const ChainDefaults d = resolve(o, rec);
  EXPECT_EQ(o.range_cell_m, 5.0);
  EXPECT_EQ(o.cfar_guard_m, 40.0);
  EXPECT_EQ(o.tracker.bearing_sigma, 0.5 * kDeg);
  EXPECT_GT(o.cfar_train_m, 300.0);                              // not given: still derived
  ASSERT_EQ(d.given.size(), 3U);
  EXPECT_EQ(d.given[0], "range_cell_m");
  EXPECT_FALSE(d.warnings.empty());                              // a 5 m cell under 37 m of resolution is said
  EXPECT_NE(d.json().find("\"range_cell_m\": 5"), std::string::npos);
}

TEST(Derive, TheChainResolvesAtItsFirstPingAndNotBefore) {
  LiveChain chain;
  EXPECT_FALSE(chain.resolved());
  EXPECT_EQ(chain.resolved_json(), "{}");
  EXPECT_FALSE(is_given(chain.options().range_cell_m));
}

TEST(Derive, RowToleranceFollowsTheWavelength) {
  // Two rows 8 mm apart at 100 kHz (half a wavelength is 7.5 mm): one row to a 2 cm tolerance, two to the derived one.
  const PingRecord close = array_record(16, 0.0075, {0.0, 0.008}, 90e3, 100e3, 0.001, 400e3, 16);
  EXPECT_DOUBLE_EQ(row_tolerance(close), 0.25 * 1500.0 / 100e3);
  EXPECT_EQ(find_rows(close, 0.02).size(), 1U);
  EXPECT_EQ(find_rows(close, row_tolerance(close)).size(), 2U);
  EXPECT_EQ(sonar_facts(close).n_rows, 2);
  EXPECT_EQ(sonar_facts(close).row_elements, 16);
  EXPECT_DOUBLE_EQ(row_tolerance(close, 0.001), 0.001);
  // At 4-6 kHz a quarter wavelength is 6 cm: the standing 2 cm.
  EXPECT_EQ(row_tolerance(array_record(32, 0.15, {0.0, 0.2}, 4e3, 6e3, 0.1, 102400.0, 16)), 0.02);
  // A line of fewer hydrophones than min_row_elements is not a row.
  const PingRecord pair = array_record(4, 0.15, {0.0, 0.2}, 4e3, 6e3, 0.1, 102400.0, 16);
  EXPECT_EQ(find_rows(pair, 0.02, 3).size(), 2U);
  EXPECT_EQ(find_rows(pair, 0.02, 5).size(), 0U);
}

TEST(Derive, PassiveBlockFollowsTheArraysTransitTime) {
  PassiveOptions p;
  resolve(p, array_record(32, 0.15, {0.0}, 4e3, 6e3, 0.1, 102400.0, 16));
  EXPECT_EQ(p.n_fft, 1024);                    // two transits of 4.8 m are 655 samples: the standing block
  EXPECT_EQ(p.hop, 512);
  PassiveOptions longer;
  const PingRecord row96 = array_record(96, 0.15, {0.0}, 4e3, 6e3, 0.1, 102400.0, 16);
  resolve(longer, row96);
  EXPECT_EQ(longer.n_fft, 2048);               // 14.4 m: 1966 samples
  EXPECT_EQ(longer.hop, 1024);
  EXPECT_EQ(passive_span(PassiveOptions{}, row96), 2048 + 7 * 1024);
  PassiveOptions given;
  given.n_fft = 256;
  resolve(given, row96);
  EXPECT_EQ(given.n_fft, 256);
  EXPECT_EQ(given.hop, 128);
}

TEST(Derive, SyncToleranceAndStreamBuffer) {
  EXPECT_DOUBLE_EQ(sync_tolerance(0.0, 6000.0), 1.0 / 120000.0);
  EXPECT_DOUBLE_EQ(sync_tolerance(0.87, 6000.0), 0.87e-6);
  const PingRecord one = array_record(32, 0.15, {0.0}, 4e3, 6e3, 0.1, 102400.0, 16);
  const PingRecord three = array_record(32, 0.15, {0.0, 0.2, 0.5}, 4e3, 6e3, 0.1, 102400.0, 16);
  EXPECT_EQ(stream_buffer_bytes(one, 3), 8 << 20);                       // half a second is 4.9 MB: the standing 8 MB
  EXPECT_EQ(stream_buffer_bytes(three, 3), 96 * 102400 * 3 / 2);          // 14.7 MB
  EXPECT_EQ(stream_buffer_bytes(three, 3, 1 << 20), 1 << 20);
  EXPECT_DOUBLE_EQ(stream_bytes_per_second(three, 2), 96.0 * 102400.0 * 2.0);
}

TEST(Capacity, SblAboveTheKernelLimitFallsBackAndSaysSo) {
  // A single row of 72 hydrophones: more than the kernels' 64.
  PingRecord rec = array_record(72, 0.15, {0.0}, 4e3, 6e3, 0.1, 32000.0, 1024 + 7 * 512);
  add_far_source(rec, Vec3d{std::sin(20.0 * kDeg), std::cos(20.0 * kDeg), 0.0}, 4000.0, 6000.0, 5);
  const std::vector<DoaFrame> frames = passive_frames(rec, 1024, 512, 8, 4000.0, 6000.0);
  ASSERT_EQ(frames.size(), 1U);
  std::vector<double> bearings;
  for (double b = -60.0; b <= 60.0; b += 0.5) bearings.push_back(b * kDeg);
  const std::vector<double> u = element_offsets(rec);
  DoaOptions kernels, reference;
  reference.kernels = false;
  const DoaResult asked = sbl(frames[0], u, 1500.0, bearings, kernels);
  const DoaResult eigen = sbl(frames[0], u, 1500.0, bearings, reference);
  EXPECT_TRUE(asked.fallback);
  EXPECT_FALSE(eigen.fallback);                 // the reference was what was asked for: nothing fell back
  ASSERT_FALSE(asked.peaks.empty());
  ASSERT_EQ(asked.peaks.size(), eigen.peaks.size());
  EXPECT_EQ(asked.peaks[0].bearing, eigen.peaks[0].bearing);
  EXPECT_NEAR(asked.peaks[0].bearing / kDeg, 20.0, 0.3);
  // At the limit nothing falls back.
  PingRecord at_limit = array_record(kSblMaxChannels, 0.15, {0.0}, 4e3, 6e3, 0.1, 32000.0, 1024 + 7 * 512);
  add_far_source(at_limit, Vec3d{std::sin(20.0 * kDeg), std::cos(20.0 * kDeg), 0.0}, 4000.0, 6000.0, 5);
  EXPECT_FALSE(sbl(passive_frames(at_limit, 1024, 512, 8, 4000.0, 6000.0)[0], element_offsets(at_limit), 1500.0, bearings, kernels).fallback);
  // Through the passive chain the frame and its report line carry the flag.
  PassiveOptions p;
  p.bearing_first_deg = -60.0;
  p.bearing_last_deg = 60.0;
  const auto out = passive_bearings(rec, p);
  ASSERT_EQ(out.size(), 1U);
  EXPECT_TRUE(out[0].sbl_fallback);
  EXPECT_NE(passive_json(out, 0).find("\"sbl_fallback\": true"), std::string::npos);
  EXPECT_EQ(passive_json(passive_bearings(at_limit, p), 0).find("sbl_fallback"), std::string::npos);
}

TEST(Capacity, PassiveOnStackedRowsWithoutElevationUsesTheRowsAsSnapshots) {
  // Three rows of 24; a source 20 degrees along the rows and 25 degrees above the horizontal. All 72
  // hydrophones taken as one line (what happened before) have phases between the rows that a line's
  // model does not hold; the rows as snapshots of one line give the cone angle.
  PingRecord rec = array_record(24, 0.15, {0.0, 0.2, 0.5}, 4e3, 6e3, 0.1, 32000.0, 1024 + 7 * 512);
  const double cone = 20.0 * kDeg, up = 25.0 * kDeg;
  const Vec3d towards{std::sin(cone), std::sqrt(1.0 - std::sin(cone) * std::sin(cone) - std::sin(up) * std::sin(up)), -std::sin(up)};
  add_far_source(rec, towards, 4000.0, 6000.0, 9);
  PassiveOptions p;
  p.bearing_first_deg = -60.0;
  p.bearing_last_deg = 60.0;
  const auto frames = passive_bearings(rec, p);
  ASSERT_EQ(frames.size(), 1U);
  ASSERT_FALSE(frames[0].peaks.empty());
  EXPECT_NEAR(frames[0].peaks[0].bearing / kDeg, 20.0, 0.5);
  EXPECT_EQ(frames[0].rows_as_snapshots, 3);
  EXPECT_FALSE(frames[0].sbl_fallback);         // 24 channels a row: the kernels take it
  EXPECT_TRUE(frames[0].elevations.empty());
  EXPECT_NE(passive_json(frames, 0).find("\"rows_as_snapshots\": 3"), std::string::npos);
  // With the option the same bearing, and the elevation with it.
  p.elevation.enabled = true;
  const auto with = passive_bearings(rec, p);
  ASSERT_FALSE(with[0].peaks.empty());
  EXPECT_EQ(with[0].peaks[0].bearing, frames[0].peaks[0].bearing);
  EXPECT_EQ(with[0].rows_as_snapshots, 0);
  ASSERT_EQ(with[0].elevations.size(), with[0].peaks.size());
  EXPECT_NEAR(with[0].elevations[0] / kDeg, 25.0, 1.5);
  // One row: exactly as before, no flag.
  PingRecord line = array_record(24, 0.15, {0.0}, 4e3, 6e3, 0.1, 32000.0, 1024 + 7 * 512);
  add_far_source(line, towards, 4000.0, 6000.0, 9);
  EXPECT_EQ(passive_json(passive_bearings(line, p), 0).find("rows_as_snapshots"), std::string::npos);
}

TEST(Capacity, ALongReportBecomesSeveralCompleteLines) {
  Report rep;
  rep.ping = 7;
  rep.sequence = 42;
  rep.time_ns = 1700000000123456789LL;
  rep.rows = 3;
  rep.synced = true;
  rep.sync_ok = true;
  rep.sync_offsets_us = {0.0, 1.5, -2.25};
  rep.sync_drift_ppm = {0.0, 3.0, -1.0};
  for (int i = 0; i < 400; ++i) {
    ReportDetection d{1000.0 + i, -30.0 + 0.1 * i, 12.5};
    d.elevation_valid = true;
    d.elevation_deg = 0.01 * i;
    d.has_depth = true;
    d.depth_m = 50.0 + i;
    d.path = "direct";
    if (i % 7 == 0) d.elevation_alternatives_deg = {1.0, 2.0, 3.0};
    rep.detections.push_back(d);
  }
  for (int i = 0; i < 50; ++i) rep.tracks.push_back({i, -1, "mover", 1.0 * i, 2.0 * i, 0.5, -0.5, 0.7, 1500.0, 14.0, 0.9});
  const std::string whole = rep.json();
  // It fits: the line itself, without a part field.
  ASSERT_EQ(rep.datagrams(whole.size()).size(), 1U);
  EXPECT_EQ(rep.datagrams(whole.size())[0], whole);
  EXPECT_EQ(whole.find("\"part\""), std::string::npos);
  // It does not: complete lines, each within the limit, the lists shared out in order.
  const std::size_t limit = 4000;
  const std::vector<std::string> parts = rep.datagrams(limit);
  ASSERT_GT(parts.size(), 10U);
  std::size_t detections = 0, tracks = 0;
  for (std::size_t k = 0; k < parts.size(); ++k) {
    const std::string& p = parts[k];
    EXPECT_LE(p.size(), limit);
    EXPECT_EQ(p.front(), '{');
    EXPECT_EQ(p.back(), '}');
    EXPECT_EQ(p.rfind("{\"ping\": 7, \"sequence\": 42, \"time_ns\": 1700000000123456789", 0), 0U);
    EXPECT_NE(p.find("\"sync\": {\"ok\": true"), std::string::npos);     // every part repeats what is not a list
    EXPECT_NE(p.find("\"rows\": 3"), std::string::npos);
    const std::string tail = "\"part\": [" + std::to_string(k + 1) + ", " + std::to_string(parts.size()) + "]}";
    EXPECT_EQ(p.compare(p.size() - tail.size(), tail.size(), tail), 0);
    for (std::size_t at = p.find("{\"range_m\""); at != std::string::npos; at = p.find("{\"range_m\"", at + 1)) ++detections;
    for (std::size_t at = p.find("{\"id\""); at != std::string::npos; at = p.find("{\"id\"", at + 1)) ++tracks;
    // Brackets balance: nothing is cut inside an object or a list.
    int depth = 0;
    for (char ch : p) {
      depth += (ch == '{' || ch == '[') ? 1 : (ch == '}' || ch == ']') ? -1 : 0;
      ASSERT_GE(depth, 0);
    }
    EXPECT_EQ(depth, 0);
  }
  EXPECT_EQ(detections, 400U);
  EXPECT_EQ(tracks, 50U);
  // The first detection of the second part follows the last of the first.
  EXPECT_NE(parts[0].find("\"range_m\": 1000,"), std::string::npos);
  // Lines of a passive record: whole lines in each datagram.
  const std::string lines = "{\"a\": 1}\n{\"b\": 22}\n{\"c\": 333}\n";
  const std::vector<std::string> grouped = line_datagrams(lines, 20);
  ASSERT_EQ(grouped.size(), 2U);
  EXPECT_EQ(grouped[0], "{\"a\": 1}\n{\"b\": 22}\n");
  EXPECT_EQ(grouped[1], "{\"c\": 333}\n");
  EXPECT_EQ(line_datagrams(lines, 1000).size(), 1U);
}

TEST(Capacity, ADatagramIsNeverCut) {
  UdpSocket in("127.0.0.1", 0, 0.5), out("127.0.0.1", 0);
  EXPECT_GT(in.buffer_granted(), 0);
  EXPECT_EQ(in.buffer_asked(), 8 << 20);
  EXPECT_FALSE(out.send_to("127.0.0.1", in.port(), std::string(kMaxDatagram + 1, 'x')));
  EXPECT_EQ(out.refused, 1);
  EXPECT_TRUE(out.send_to("127.0.0.1", in.port(), std::string(60000, 'y')));
  std::vector<unsigned char> buffer;
  EXPECT_EQ(in.receive(buffer), 60000U);        // the next thing to arrive is the whole second text, not a piece of the first
  UdpSocket asked("127.0.0.1", 0, 0.2, 1 << 16);
  EXPECT_EQ(asked.buffer_asked(), 1 << 16);
  EXPECT_GE(asked.buffer_granted(), 1 << 16);   // a buffer this small is always granted
}

TEST(Stream, BlocksOfTwoThreeAndFourByteSamples) {
  const unsigned char three[3] = {0xFF, 0xFF, 0xFF}, two[2] = {0x00, 0x80}, four[4] = {0x00, 0x00, 0x00, 0x80};
  EXPECT_EQ(block_sample(three, 3), -1);
  EXPECT_EQ(block_sample(two, 2), -32768);
  EXPECT_EQ(block_sample(four, 4), -2147483647 - 1);
  EXPECT_DOUBLE_EQ(block_full_scale(2), 32768.0);
  EXPECT_DOUBLE_EQ(block_full_scale(3), 8388608.0);
  EXPECT_THROW(block_full_scale(5), std::invalid_argument);
  // One ping of two channels in 16-bit words through the assembler: the volts that were sent.
  for (int bytes : {2, 3, 4}) {
    PingRecord t = array_record(2, 0.15, {0.0}, 4e3, 6e3, 0.001, 1000.0, 8);
    t.samples.start_time = 0.0;
    t.metadata.adc_full_scale = 2.0;
    PingAssembler assembler(t, nullptr, nullptr, bytes);
    std::vector<unsigned char> block(128 + 8 * 2 * static_cast<std::size_t>(bytes), 0);
    std::memcpy(block.data(), "GSFB", 4);
    const std::uint16_t version = 1, channels = 2;
    const std::uint32_t n = 8;
    const std::uint64_t first = 0;
    const std::int64_t tx = 0;
    std::memcpy(block.data() + 4, &version, 2);
    std::memcpy(block.data() + 6, &channels, 2);
    std::memcpy(block.data() + 8, &n, 4);
    std::memcpy(block.data() + 12, &first, 8);
    std::memcpy(block.data() + 20, &tx, 8);
    const double full = block_full_scale(bytes);
    for (int i = 0; i < 8; ++i) {
      for (int c = 0; c < 2; ++c) {
        const auto v = static_cast<std::int32_t>(std::lround((c == 0 ? 0.25 : -0.5) * full * (i + 1) / 8.0));
        std::memcpy(block.data() + 128 + static_cast<std::size_t>((i * 2 + c) * bytes), &v, static_cast<std::size_t>(bytes));   // little-endian
      }
    }
    std::vector<PingRecord> out;
    ASSERT_TRUE(assembler.feed(block.data(), block.size(), out)) << bytes;
    ASSERT_EQ(out.size(), 1U) << bytes;
    EXPECT_NEAR(out[0].samples.channel(0)[7], 0.5, 1e-4) << bytes;      // a quarter of full scale of 2 V
    EXPECT_NEAR(out[0].samples.channel(1)[3], -0.5, 1e-4) << bytes;     // minus half of full scale, half way up the ramp
    // The same bytes read with another word length are not a block of this sonar.
    PingAssembler other(t, nullptr, nullptr, bytes == 2 ? 3 : 2);
    EXPECT_FALSE(other.feed(block.data(), block.size(), out));
    EXPECT_EQ(other.bad_blocks, 1);
  }
}
