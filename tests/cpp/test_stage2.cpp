// Stage 2 of the long-range array's reference design, in C++ on records built here where the
// answer is known in closed form: the decimate-first stage (a tone through it, streaming against
// whole-ping, two boards on their own clocks), blanking, the navigation receiver (water depth from
// altitude and the loss of bottom lock, pings outside the fixes' span, what it leaves in a record),
// the values carried in metadata.extra_json, and the sound-speed ladder (rungs, the built profile,
// the depth sigma's limit). tests/python/test_decimate.py, test_blanking.py, test_nav_adapter.py and
// test_profile_ladder.py hold the same code against simulated pings and the Python stages.
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/environment.hpp"
#include "sonar/live/udp_source.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFs = 102400.0;

// A raw record: `boards` boards of four inputs each, a 4-6 kHz pulse, a window of `n` samples from 0.1 s.
PingRecord raw_record(std::int64_t boards, std::int64_t n) {
  PingRecord r;
  r.samples.resize(4 * boards, n, SampleKind::RealPassband);
  r.samples.sample_rate = kFs;
  r.samples.start_time = 0.1;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 4000.0;
  r.transmit.waveform.f_stop = 6000.0;
  r.transmit.waveform.duration = 0.05;
  r.environment.sound_speed = 1500.0;
  r.metadata.adc_full_scale = 2.5;
  for (std::int64_t c = 0; c < 4 * boards; ++c) {
    Element el;
    el.lever_arm = Vec3d{0.15 * static_cast<double>(c), 0.0, 0.0};
    r.geometry.rx.push_back(el);
  }
  set_default_channel_tables(r);
  r.metadata.boards.clear();
  r.metadata.channel_board.clear();
  for (std::int64_t b = 0; b < boards; ++b) {
    Board board;
    board.reference_channel = static_cast<std::int32_t>(4 * b);
    r.metadata.boards.push_back(board);
    for (int k = 0; k < 4; ++k) r.metadata.channel_board.push_back(static_cast<std::int32_t>(b));
  }
  return r;
}

// The reference burst (a Hann-windowed chirp, as sync.cpp builds it) at time t after its start.
double burst_at(const Reference& ref, double t) {
  if (t < 0.0 || t > ref.duration) return 0.0;
  const double k = (ref.f_stop - ref.f_start) / ref.duration, w = std::sin(kPi * t / ref.duration);
  return w * w * std::cos(2.0 * kPi * (ref.f_start * t + 0.5 * k * t * t));
}

}  // namespace

TEST(Decimate, PlanFollowsTheBandAndATonePassesWithItsPhase) {
  PingRecord r = raw_record(1, 40000);
  const DecimatePlan plan = plan_decimation(r, {});
  EXPECT_EQ(plan.factor, 17);                       // floor(102400 / (2 x 1.5 x 2000))
  EXPECT_DOUBLE_EQ(plan.carrier, 5000.0);
  EXPECT_DOUBLE_EQ(plan.passband, 1500.0);
  EXPECT_NEAR(plan.stopband, kFs / 17.0 - 1500.0, 1e-9);
  EXPECT_EQ(plan.taps.size() % 2, 1U);
  EXPECT_NEAR(plan.delay_s(), 0.5 * static_cast<double>(plan.taps.size() - 1) / kFs, 1e-15);
  // cos(2 pi f t + phi) in the band comes out as exp(j (2 pi (f - fc) t + phi)), t from the transmit; a tone at the
  // synchronisation reference's frequency does not come out at all.
  const double f = 5600.0, phi = 0.7;
  for (std::int64_t i = 0; i < r.samples.n_samples; ++i) {
    const double t = r.samples.start_time + static_cast<double>(i) / kFs;
    r.samples.channel(0)[i] = static_cast<float>(std::cos(2.0 * kPi * f * t + phi));
    r.samples.channel(1)[i] = static_cast<float>(std::cos(2.0 * kPi * 43000.0 * t));
  }
  PingRecord whole = r;
  DecimateOptions on;
  on.enabled = true;
  decimate_record(whole, on);
  ASSERT_TRUE(whole.samples.is_complex());
  EXPECT_EQ(whole.samples.n_samples, (r.samples.n_samples - 1) / 17 + 1);
  EXPECT_DOUBLE_EQ(whole.samples.center_frequency, 5000.0);
  EXPECT_DOUBLE_EQ(whole.samples.start_time, 0.1);
  for (std::int64_t m = 40; m < whole.samples.n_samples - 40; ++m) {
    const double t = 0.1 + static_cast<double>(m) * 17.0 / kFs;
    const std::complex<double> want = std::polar(1.0, 2.0 * kPi * (f - 5000.0) * t + phi);
    EXPECT_LT(std::abs(std::complex<double>(whole.samples.channel_complex(0)[m]) - want), 0.01) << m;
    EXPECT_LT(std::abs(whole.samples.channel_complex(1)[m]), 2e-4F) << m;
  }
  // Streaming, row by row in uneven pieces: the same samples exactly.
  PingDecimator dec(r, on);
  std::vector<float> rows(static_cast<std::size_t>(r.samples.n_samples * 4));
  for (std::int64_t i = 0; i < r.samples.n_samples; ++i) {
    for (std::int64_t c = 0; c < 4; ++c) rows[static_cast<std::size_t>(i * 4 + c)] = r.samples.channel(c)[i];
  }
  std::int64_t at = 0;
  for (std::int64_t n : {1, 5, 128, 1000, 7, 100000}) {
    const std::int64_t take = std::min(n, r.samples.n_samples - at);
    dec.feed(rows.data() + at * 4, take);
    at += take;
  }
  PingRecord like = r;
  dec.finish(like);
  ASSERT_EQ(like.samples.data.size(), whole.samples.data.size());
  for (std::size_t i = 0; i < like.samples.data.size(); ++i) ASSERT_EQ(like.samples.data[i], whole.samples.data[i]) << i;
  // A band that does not fit is refused, not bent.
  DecimateOptions bad = on;
  bad.f_low = 60000.0;
  bad.f_high = 70000.0;
  EXPECT_THROW(plan_decimation(r, bad), std::invalid_argument);
}

TEST(Decimate, ClocksComeFromTheKeptStretchesAndTheBasebandIsCorrected) {
  // Two boards: the second samples 180 us late and 20 ppm fast. A 5.3 kHz tone on every input, the reference bursts
  // on each board's first input, each as that board's clock sees it.
  const std::int64_t n = 60000;
  const double offset = 180e-6, drift = 20e-6;
  PingRecord r = raw_record(2, n);
  Reference ref;
  const double t_a = ref.lead, t_b = static_cast<double>(n) / kFs - ref.tail - ref.duration;      // from the window's opening
  for (std::int64_t c = 0; c < 8; ++c) {
    const bool late = c >= 4;
    for (std::int64_t i = 0; i < n; ++i) {
      // Sample i of a board was taken at true time start + offset + i / (fs (1 + drift)).
      const double since = (late ? offset : 0.0) + static_cast<double>(i) / (kFs * (late ? 1.0 + drift : 1.0));
      double v = 1e-3 * std::cos(2.0 * kPi * 5300.0 * (r.samples.start_time + since));
      if (c % 4 == 0) v += 2e-3 * (burst_at(ref, since - t_a) + burst_at(ref, since - t_b));
      r.samples.channel(c)[i] = static_cast<float>(v);
    }
  }
  PingRecord full = r;
  const SyncResult a = synchronise(full, ref);
  DecimateOptions on;
  on.enabled = true;
  PingRecord staged = r;
  const SyncResult b = decimate_record(staged, on, &ref);
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  EXPECT_NEAR(b.offset[1], offset, 1e-6);
  EXPECT_NEAR(b.drift_ppm[1], 20.0, 0.5);
  EXPECT_NEAR(b.offset[1], a.offset[1], 1e-9);        // the same stretches, the same estimate
  EXPECT_NEAR(b.drift_ppm[1], a.drift_ppm[1], 1e-3);
  EXPECT_EQ(staged.metadata.boards[1].synchronised, 1);
  SyncResult facts;
  ASSERT_TRUE(decimated_sync_facts(staged, facts));
  EXPECT_TRUE(facts.ok);
  EXPECT_NEAR(facts.offset[1], b.offset[1], 1e-15);
  // After the correction the late board's tone has the first board's phase; without it, it is 0.95 of a cycle out.
  PingRecord uncorrected = r;
  decimate_record(uncorrected, on);
  const std::int64_t m = staged.samples.n_samples / 2;
  const auto phase_between = [&](const PingRecord& rec) {
    return std::arg(std::complex<double>(rec.samples.channel_complex(5)[m]) * std::conj(std::complex<double>(rec.samples.channel_complex(1)[m])));
  };
  EXPECT_NEAR(phase_between(staged), 0.0, 0.02);
  EXPECT_GT(std::abs(phase_between(uncorrected)), 0.2);
  // A record decimated without a reference says nothing about clocks.
  SyncResult none;
  PingRecord plain = uncorrected;
  EXPECT_FALSE(decimated_sync_facts(plain, none));
}

TEST(Blanking, ZeroesFromTheTransmitWithARaisedCosineReturn) {
  PingRecord r = raw_record(1, 20000);
  for (float& v : r.samples.data) v = 1.0F;
  BlankingOptions b;
  b.enabled = true;
  b.recovery_s = 0.07;
  b.taper_s = 0.01;
  EXPECT_DOUBLE_EQ(b.until(r), 0.12);                 // 0.05 s of pulse and the recovery
  blank_interval(r, -1e300, b.until(r), b.taper_s);
  for (std::int64_t i = 0; i < r.samples.n_samples; i += 97) {
    const double t = 0.1 + static_cast<double>(i) / kFs, out = t - 0.12;
    const double want = out <= 0.0 ? 0.0 : out >= 0.01 ? 1.0 : 0.5 - 0.5 * std::cos(kPi * out / 0.01);
    EXPECT_NEAR(r.samples.channel(2)[i], want, 1e-6) << i;
  }
}

TEST(Navigation, WaterDepthFromAltitudeHoldsThroughALossOfBottomLock) {
  WaterDepthOptions o;
  o.from_altitude = true;
  o.altitude_sigma_m = 0.3;
  o.depth_sigma_m = 0.4;
  o.hold_s = 30.0;
  o.hold_sigma_m_per_s = 0.1;
  WaterDepthTracker w(o);
  const std::int64_t s = 1000000000LL;
  WaterDepth d = w.update(0, 0.0, 0.0, 20.0, 80.0, true);
  EXPECT_DOUBLE_EQ(d.depth, 100.0);
  EXPECT_DOUBLE_EQ(d.sigma, 0.5);
  EXPECT_FALSE(d.held);
  d = w.update(10 * s, 20.0, 0.0, 22.0, std::nan(""), true);      // no altitude: the last value, less certain
  EXPECT_TRUE(d.held);
  EXPECT_DOUBLE_EQ(d.depth, 100.0);
  EXPECT_NEAR(d.sigma, std::sqrt(0.25 + 1.0 + 1.0), 1e-12);        // 0.1 m/s x 10 s and 0.05 m/m x 20 m, in quadrature
  d = w.update(20 * s, 40.0, 0.0, 22.0, 75.0, false);              // an altitude the log does not vouch for
  EXPECT_TRUE(d.held);
  EXPECT_TRUE(std::isnan(w.update(45 * s, 90.0, 0.0, 22.0, -1.0, true).depth));    // held too long: unknown
  EXPECT_DOUBLE_EQ(w.update(46 * s, 92.0, 0.0, 22.0, 70.0, true).depth, 92.0);
  EXPECT_TRUE(std::isnan(WaterDepthTracker().update(0, 0.0, 0.0, 20.0, 80.0, true).depth));   // not asked for: nothing
}

TEST(Navigation, PingsOutsideTheFixesTakeTheNearestAndTheRecordCarriesWhatWasSaid) {
  NavOptions o;
  o.water_depth.from_altitude = true;
  live::Navigation nav(o);
  PingRecord rec = raw_record(1, 2048);
  EXPECT_FALSE(nav.apply(rec));                                    // no fix yet
  for (int i = 0; i < 40; ++i) {
    char json[512];
    std::snprintf(json, sizeof(json),
                  "{\"time_ns\": %lld, \"x\": %.3f, \"y\": 0.0, \"z\": %.3f, \"roll\": %.4f, \"pitch\": 0.0, \"yaw\": 0.0, \"vx\": 1.5, \"vy\": 0.0, "
                  "\"vz\": 0.5, \"sound_speed\": %.2f, \"altitude\": %.3f, \"bottom_lock\": %s, \"roll_sigma\": 0.0005, \"altitude_sigma\": 0.2}",
                  1000000000LL + i * 100000000LL, 0.15 * i, 20.0 + 0.05 * i, 0.001 * i, 1500.0 - 0.1 * i, 80.0 - 0.05 * i, i < 30 ? "true" : "false");
    ASSERT_TRUE(nav.push(json));
  }
  EXPECT_FALSE(nav.push("{\"x\": 1.0}"));                          // no time: not a fix
  EXPECT_EQ(nav.size(), 40U);
  // Between fixes: interpolated; the water depth from the Doppler log; history, depth and log samples in the record.
  PingRecord r = raw_record(1, 2048);
  r.transmit.time_ns = 1000000000LL + 1050000000LL;                // halfway between fixes 10 and 11
  ASSERT_TRUE(nav.apply(r));
  EXPECT_NEAR(r.geometry.rx_pose.position.z, 20.525, 1e-9);
  EXPECT_NEAR(r.environment.water_depth, 100.0, 1e-6);
  EXPECT_GT(r.navigation.ins.size(), 20U);
  EXPECT_EQ(r.navigation.depth.size(), r.navigation.ins.size());
  EXPECT_EQ(r.navigation.dvl.size(), r.navigation.ins.size());
  EXPECT_NEAR(r.navigation.dvl.front().velocity_body.x, 1.5, 1e-9);
  const std::string& extra = r.metadata.extra_json;
  EXPECT_DOUBLE_EQ(extra_number(extra, "sound_speed_measured", 0.0), 1.0);
  EXPECT_DOUBLE_EQ(extra_number(extra, "roll_sigma", 0.0), 0.0005);
  EXPECT_DOUBLE_EQ(extra_number(extra, "water_depth_sigma_m", 0.0), 0.2);
  EXPECT_DOUBLE_EQ(extra_number(extra, "water_depth_held_s", -1.0), 0.0);
  const auto samples = extra_rows(extra, "sound_speed_samples", 5);
  ASSERT_GE(samples.size(), 3U);                                   // one every half metre of depth: 2 m of dive
  EXPECT_NEAR(samples[1][3] - samples[0][3], 0.5, 1e-6);
  // The next ping gets the samples taken since, not the same ones again.
  PingRecord next = raw_record(1, 2048);
  next.transmit.time_ns = r.transmit.time_ns + 1;
  ASSERT_TRUE(nav.apply(next));
  EXPECT_TRUE(extra_rows(next.metadata.extra_json, "sound_speed_samples", 5).empty());
  // Before the first fix and after the last: the nearest one, all of it and nothing past it.
  const std::vector<std::pair<std::int64_t, int>> outside{{0, 0}, {900000000000LL, 39}};
  for (const auto& [t_ns, index] : outside) {
    PingRecord q = raw_record(1, 2048);
    q.transmit.time_ns = t_ns;
    ASSERT_TRUE(nav.apply(q));
    EXPECT_NEAR(q.geometry.rx_pose.position.z, 20.0 + 0.05 * index, 1e-9);
    EXPECT_NEAR(q.geometry.rx_pose.rpy.x, 0.001 * index, 1e-12);
    EXPECT_NEAR(q.geometry.rx_pose.velocity.z, 0.5, 1e-12);
    EXPECT_NEAR(q.environment.sound_speed, 1500.0 - 0.1 * index, 1e-9);
    EXPECT_DOUBLE_EQ(q.geometry.tx_pose.position.z, q.geometry.rx_pose.position.z);
  }
}

TEST(ExtraJson, ValuesAreSetReplacedAndReadBesideWhatIsThere) {
  std::string extra;
  extra_set(extra, "nav", "{\"roll_sigma\": 0.001}");
  EXPECT_EQ(extra, "{\"nav\": {\"roll_sigma\": 0.001}}");
  extra = "{\"gpusonar\": {\"compressed\": true}, \"note\": \"a, b] }\"}";
  extra_set(extra, "decimated", "{\"factor\": 17, \"rows\": [[1, 2.5, -3e-2], [4, null, 6], [7, 8, 9]]}");
  extra_set(extra, "nav", "{\"roll_sigma\": 0.001}");
  extra_set(extra, "nav", "{\"roll_sigma\": 0.002, \"water_depth_sigma_m\": 1.5}");     // replaced, not added again
  EXPECT_EQ(extra.find("\"nav\""), extra.rfind("\"nav\""));
  EXPECT_DOUBLE_EQ(extra_number(extra, "roll_sigma", 0.0), 0.002);
  EXPECT_DOUBLE_EQ(extra_number(extra, "factor", 0.0), 17.0);
  EXPECT_DOUBLE_EQ(extra_number(extra, "absent", -4.0), -4.0);
  EXPECT_DOUBLE_EQ(extra_number(extra, "compressed", -4.0), -4.0);  // true is not a number
  const auto rows = extra_rows(extra, "rows", 3);
  ASSERT_EQ(rows.size(), 2U);                                       // the row with a null in it is not three numbers
  EXPECT_DOUBLE_EQ(rows[0][2], -0.03);
  EXPECT_DOUBLE_EQ(rows[1][0], 7.0);
  EXPECT_NE(extra.find("\"note\": \"a, b] }\""), std::string::npos);  // what was there is still there
}

TEST(Ladder, RungsInOrderABuiltProfileAndTheSigmasLimit) {
  ProfileOptions o;
  o.enabled = true;
  o.sample_sigma = 0.1;
  o.age_sigma_per_hour = 0.5;
  SoundSpeedLadder ladder(o);
  const std::vector<double> none;
  EXPECT_EQ(ladder.evaluate(0, 0.0, 0.0, 50.0, 1490.0, false, 200.0, none, none, 0.0, 0.05).rung, "prior");
  EXPECT_EQ(ladder.evaluate(0, 0.0, 0.0, 50.0, 1490.0, true, 200.0, none, none, 0.0, 0.05).rung, "local");
  for (double z = 40.0; z <= 60.0; z += 1.0) ladder.add(0, 0.0, 0.0, z, 1490.0 - 0.1 * (z - 50.0));
  const std::int64_t hour = 3600LL * 1000000000LL;
  const LadderProfile built = ladder.evaluate(hour, 0.0, 0.0, 50.0, 1490.0, true, 200.0, none, none, 0.017, 0.05);
  ASSERT_EQ(built.rung, "built");
  EXPECT_EQ(built.bins, 11);
  EXPECT_NEAR(built.span_lo, 40.5, 0.51);
  EXPECT_DOUBLE_EQ(built.depth.front(), 0.0);
  for (std::size_t i = 1; i + 1 < built.depth.size(); ++i) EXPECT_NEAR(built.sigma[i], std::hypot(0.1, 0.5), 1e-12);   // an hour old
  // Outside the sampled depths: the assumed gradient, and a sigma wider by the prior's gradient per metre.
  EXPECT_NEAR(built.speed.front(), built.speed[1] - 0.017 * built.depth[1], 1e-9);
  EXPECT_NEAR(built.sigma.front(), built.sigma[1] + 0.05 * built.depth[1], 1e-9);
  EXPECT_EQ(ladder.evaluate(0, 0.0, 0.0, 50.0, 1490.0, true, 200.0, {0.0, 200.0}, {1500.0, 1503.4}, 0.0, 0.05).rung, "supplied");
  // A profile nobody vouches for: the depth sigma is the prior's range^2 sigma_g / 2c.
  const double sigma = profile_depth_sigma(60.0, 1.0 * kPi / 180.0, 2000.0, {0.0, 5000.0}, {1500.0, 1500.0}, {1e9, 1e9}, 0.05);
  EXPECT_NEAR(sigma, refraction_sigma(2000.0, 1500.0, 0.05), 0.03 * refraction_sigma(2000.0, 1500.0, 0.05));
  // And a profile that is vouched for gives less.
  EXPECT_LT(profile_depth_sigma(60.0, 1.0 * kPi / 180.0, 2000.0, {0.0, 5000.0}, {1500.0, 1500.0}, {0.1, 0.1}, 0.05), 0.5 * sigma);
  ProfileOptions bad = o;
  bad.ladder = {"guess"};
  EXPECT_THROW(SoundSpeedLadder{bad}, std::invalid_argument);
  ProfileOptions family = o;
  family.ladder = {"family"};
  family.family = "mixed_layer";
  EXPECT_THROW(SoundSpeedLadder(family).evaluate(0, 0.0, 0.0, 50.0, 1490.0, true, 200.0, none, none, 0.0, 0.05), std::invalid_argument);
}
