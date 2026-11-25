// Stage 2 of the long-range array's reference design, in C++ on records built here where the
// answer is known in closed form: the decimate-first stage (a tone through it, streaming against
// whole-ping, two boards on their own clocks), blanking, the navigation receiver (water depth from
// altitude and the loss of bottom lock, pings outside the fixes' span, what it leaves in a record),
// the values carried in metadata.extra_json, and the sound-speed ladder (rungs, the built profile,
// the depth sigma's limit). tests/python/test_decimate.py, test_blanking.py, test_nav_adapter.py and
// test_profile_ladder.py hold the same code against simulated pings and the Python stages.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/decimate.hpp"
#include "sonar/live/environment.hpp"
#include "sonar/live/sync.hpp"
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

// The board resampler's 16 weights at fractional position `frac` from the formula, a sine and a cosine a tap (sync.cpp
// until sonar-fy6).
void direct_weights(double frac, double* w) {
  constexpr int half = 8;
  for (int k = -half + 1; k <= half; ++k) {
    const double d = static_cast<double>(k) - frac;
    const double sinc = std::abs(d) < 1e-12 ? 1.0 : std::sin(kPi * d) / (kPi * d);
    w[k + half - 1] = sinc * (0.5 + 0.5 * std::cos(kPi * d / half));
  }
}

// The board resampler one channel row at a time (sync.cpp until sonar-drc; the reference for the tests below): every
// sample's weights computed again for every row, from the formula (`direct`, sync.cpp until sonar-fy6) or from
// resample_weights (sync.cpp since).
void resample_row_by_row(float* x, std::int64_t n, double offset, double drift, double fs, bool direct) {
  constexpr int taps = 16, half = taps / 2;
  std::vector<float> out(static_cast<std::size_t>(n), 0.0F);
  for (std::int64_t m = 0; m < n; ++m) {
    const double pos = (static_cast<double>(m) / fs - offset) * fs * (1.0 + drift);
    const auto base = static_cast<std::int64_t>(std::floor(pos));
    const double frac = pos - static_cast<double>(base);
    double w[taps];
    if (direct) {
      direct_weights(frac, w);
    } else {
      resample_weights(frac, w);
    }
    double acc = 0.0;
    for (int k = -half + 1; k <= half; ++k) {
      const std::int64_t idx = base + k;
      if (idx < 0 || idx >= n) continue;
      acc += w[k + half - 1] * static_cast<double>(x[idx]);
    }
    out[static_cast<std::size_t>(m)] = static_cast<float>(acc);
  }
  std::copy(out.begin(), out.end(), x);
}

// The largest difference between a weight from three trig calls and the formula's, over every tap and position below
// (measured 2.8e-16 with this C++ library's sine and cosine, 2.5e-16 with numpy's; the weights are at most 1): a weight's
// last bit or two.
constexpr double kWeightBound = 4.5e-16;

}  // namespace

TEST(Sync, ResampleWeightsFromThreeTrigCallsAreTheFormulasTapByTap) {
  // sonar-fy6: resample_weights makes a position's 16 weights from sin(pi frac), cos(pi frac / 8) and sin(pi frac / 8).
  // Tap by tap against the formula over a dense set of fractions, the ones at the ends (0, the smallest, next to 1/2 and
  // to 1) and the fractions the chain makes (positions (m / fs - offset) fs (1 + drift) of a board late or early,
  // with or without drift).
  std::vector<double> fracs;
  for (int i = 0; i < 200000; ++i) fracs.push_back(static_cast<double>(i) / 200000.0);
  for (double f : {0.0, std::numeric_limits<double>::denorm_min(), 1e-300, 1e-13, 1e-12, 1.1e-12, 1e-9, 0.5, std::nextafter(0.5, 0.0),
                   std::nextafter(0.5, 1.0), std::nextafter(1.0, 0.0), 1.0 - 1e-12, 1.0 - 1.1e-12, 1.0 - 1e-9, 1.0 - 1e-15}) {
    fracs.push_back(f);
  }
  for (double offset : {3.1e-4, -1.7e-4, 2.1e-3, 1.234567e-5}) {
    for (double drift : {0.0, 12e-6, -55.5e-6}) {
      for (std::int64_t m = 0; m < 20000; m += 7) {
        const double pos = (static_cast<double>(m) / kFs - offset) * kFs * (1.0 + drift);
        fracs.push_back(pos - std::floor(pos));
      }
    }
  }
  double worst = 0.0;
  for (double f : fracs) {
    ASSERT_GE(f, 0.0);
    ASSERT_LT(f, 1.0);
    double got[16], want[16];
    resample_weights(f, got);
    direct_weights(f, want);
    for (int k = 0; k < 16; ++k) {
      ASSERT_TRUE(std::isfinite(got[k])) << f << " " << k;
      worst = std::max(worst, std::abs(got[k] - want[k]));
    }
  }
  std::printf("resample_weights against the formula: largest difference %.3g over %zu positions\n", worst, fracs.size());
  EXPECT_LE(worst, kWeightBound);
  // On a sample: the tap at k = 0 is 1 and every other tap 0, exactly, so the sample is passed on as it is.
  double w[16];
  resample_weights(0.0, w);
  for (int k = -7; k <= 8; ++k) EXPECT_EQ(w[k + 7], k == 0 ? 1.0 : 0.0) << k;
  // Next to the following sample (frac = 1 - 2^-53, 1 - 1e-9): its tap (k = 1) is 1 to a bit or two, as the formula's;
  // sin(pi frac) taken as is there (not from frac - 1) is off by up to 3e-16 / (pi (1 - frac)) relatively: all of it at 1 - 2^-53.
  for (double f : {std::nextafter(1.0, 0.0), 1.0 - 1e-9}) {
    resample_weights(f, w);
    EXPECT_NEAR(w[8], 1.0, kWeightBound) << f;
  }
}

TEST(Sync, BoardsResampledTogetherGiveTheRowByRowSamplesToTheBit) {
  // sonar-drc: synchronise() computes a board's sinc weights once a block of samples for all its rows, in groups of
  // rows. Its samples must be the row-by-row resampler's with the same weights, to the bit, with the clocks it
  // estimated: three boards of 4, 11 (more than one group) and 1 inputs, a window that is not a whole number of blocks,
  // a board 2.1 ms late (taps past the start) and one early (taps past the end), drifts both ways. Each board's first
  // input carries the reference bursts as its clock sees them, every input noise and a tone.
  // sonar-fy6: the weights come from three trig calls a sample, not a sine and a cosine a tap; against the formula's
  // weights every sample is within one float spacing of its value plus 16 x kWeightBound x the largest input.
  const std::int64_t n = 3 * 4096 + 1237;
  const std::vector<std::int64_t> counts = {4, 11, 1};
  const std::vector<double> late = {0.0, 2.1e-3, -1.3e-3}, fast = {0.0, 35e-6, -20e-6};
  Reference ref;
  ref.lead = 5e-3;
  ref.tail = 10e-3;
  ref.search_time = 3e-3;
  PingRecord r = raw_record(1, n);
  r.samples.resize(16, n, SampleKind::RealPassband);
  r.samples.sample_rate = kFs;
  r.geometry.rx.resize(16);
  r.metadata.boards.assign(3, Board{});
  r.metadata.channel_board.clear();
  r.metadata.channel_element.clear();
  std::uint64_t state = 12345U;
  auto noise = [&state] {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(state >> 11) / 9007199254740992.0 - 0.5;
  };
  const double t_a = ref.lead, t_b = static_cast<double>(n) / kFs - ref.tail - ref.duration;
  std::int64_t c = 0;
  for (std::size_t b = 0; b < counts.size(); ++b) {
    r.metadata.boards[b].reference_channel = static_cast<std::int32_t>(c);
    for (std::int64_t k = 0; k < counts[b]; ++k, ++c) {
      r.metadata.channel_board.push_back(static_cast<std::int32_t>(b));
      r.metadata.channel_element.push_back(static_cast<std::int32_t>(c));
      for (std::int64_t i = 0; i < n; ++i) {
        const double since = late[b] + static_cast<double>(i) / (kFs * (1.0 + fast[b]));
        double v = 1e-3 * std::cos(2.0 * kPi * (4700.0 + 37.0 * static_cast<double>(c)) * since) + 1e-4 * noise();
        if (k == 0) v += 2e-3 * (burst_at(ref, since - t_a) + burst_at(ref, since - t_b));
        r.samples.channel(c)[i] = static_cast<float>(v);
      }
    }
  }
  PingRecord got = r;
  const SyncResult s = synchronise(got, ref);
  ASSERT_EQ(s.offset.size(), 3U);
  EXPECT_NEAR(s.offset[1], late[1], 1e-6);
  EXPECT_NEAR(s.offset[2], late[2], 1e-6);
  EXPECT_NEAR(s.drift_ppm[1], 35.0, 1.0);
  EXPECT_NEAR(s.drift_ppm[2], -20.0, 1.0);
  PingRecord want = r, formula = r;
  for (std::int64_t ch = 0; ch < 16; ++ch) {
    const auto b = static_cast<std::size_t>(want.metadata.channel_board[static_cast<std::size_t>(ch)]);
    if (s.offset[b] == 0.0 && s.drift_ppm[b] == 0.0) continue;
    resample_row_by_row(want.samples.channel(ch), n, s.offset[b], s.drift_ppm[b] * 1e-6, kFs, false);
    resample_row_by_row(formula.samples.channel(ch), n, s.offset[b], s.drift_ppm[b] * 1e-6, kFs, true);
  }
  ASSERT_EQ(got.samples.data.size(), want.samples.data.size());
  float largest = 0.0F;
  for (float v : r.samples.data) largest = std::max(largest, std::abs(v));
  std::size_t moved = 0, ulps[4] = {0, 0, 0, 0};
  for (std::size_t i = 0; i < want.samples.data.size(); ++i) {
    ASSERT_EQ(got.samples.data[i], want.samples.data[i]) << i;   // exact: the same float, not a near one
    moved += want.samples.data[i] != r.samples.data[i] ? 1 : 0;
    const float g = got.samples.data[i], f = formula.samples.data[i];
    const float spacing = std::nextafter(std::max(std::abs(g), std::abs(f)), std::numeric_limits<float>::infinity()) -
                          std::max(std::abs(g), std::abs(f));
    ASSERT_LE(std::abs(static_cast<double>(g) - f), spacing + 16.0 * kWeightBound * largest) << i;
    float step = f;
    std::size_t apart = 0;
    while (step != g && apart < 3) {
      step = std::nextafter(step, g);
      ++apart;
    }
    ++ulps[apart];
  }
  std::printf("against the formula's weights: %zu samples equal, %zu one float apart, %zu two, %zu more\n", ulps[0], ulps[1], ulps[2],
              ulps[3]);
  EXPECT_GT(moved, want.samples.data.size() / 2);   // the boards that run late or early were resampled
}

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
