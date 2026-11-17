// Stacked rows in C++ (libs/live elevation.hpp) on records built here, where the answer is
// known in closed form: rows from positions, elevation and depth with the vehicle rolled, the
// second lobe of two rows, depth folding, the passive estimate, and the chain with one row
// unchanged. tests/python/test_elevation.py holds the same code against the Python stages,
// the Cramer-Rao bound and simulated pings.
#include <cmath>
#include <complex>
#include <random>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/elevation.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kC = 1500.0, kFs = 8000.0, kFc = 5000.0, kPitch = 0.15, kSigma = 4e-4;
constexpr std::int64_t kPerRow = 16, kSamples = 2048;
using Cf = std::complex<float>;
using Cd = std::complex<double>;

// A conditioned (complex baseband, compressed) record of rows of 16 hydrophones along body x at
// the given heights (m up) and offsets out (m along body y, the look direction), with nothing in it.
PingRecord empty_record(const std::vector<double>& up, const std::vector<double>& out, double roll = 0.0, double depth = 50.0) {
  PingRecord r;
  r.samples.resize(kPerRow * static_cast<std::int64_t>(up.size()), kSamples, SampleKind::ComplexBaseband);
  r.samples.sample_rate = kFs;
  r.samples.center_frequency = kFc;
  r.samples.start_time = 0.6;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 4000.0;
  r.transmit.waveform.f_stop = 6000.0;
  r.transmit.waveform.duration = 0.01;
  r.environment.sound_speed = kC;
  for (std::size_t k = 0; k < up.size(); ++k) {
    for (std::int64_t e = 0; e < kPerRow; ++e) {
      Element el;
      el.lever_arm = Vec3d{(static_cast<double>(e) - 0.5 * (kPerRow - 1)) * kPitch, out[k], -up[k]};
      el.boresight = Vec3d{0.0, 1.0, 0.0};
      r.geometry.rx.push_back(el);
    }
  }
  r.geometry.rx_pose.position = Vec3d{0.0, 0.0, depth};
  r.geometry.rx_pose.rpy = Vec3d{roll, 0.0, 0.0};
  r.geometry.tx_pose = r.geometry.rx_pose;
  r.geometry.tx.lever_arm = Vec3d{0.0, 0.0, 1.0};          // the projector hangs a metre below the array
  set_default_channel_tables(r);
  r.metadata.extra_json = "{\"gpusonar\": {\"compressed\": true, \"replica_window\": \"none\"}}";
  return r;
}

// The compressed echo of a point at a world position: a Gaussian pulse at each element's exact
// two-way delay (projector -> point -> element), with the carrier phase of that delay.
void add_echo(PingRecord& r, const Vec3d& source, double amplitude) {
  const Vec3d tx = world_position(r.geometry.tx_pose, r.geometry.tx.lever_arm);
  for (std::int64_t e = 0; e < r.samples.n_channels; ++e) {
    const Vec3d p = world_position(r.geometry.rx_pose, r.geometry.rx[static_cast<std::size_t>(e)].lever_arm);
    const double tau = (norm(source - tx) + norm(source - p)) / kC;
    const Cd scale = std::polar(amplitude, -2.0 * kPi * kFc * tau);
    Cf* x = r.samples.channel_complex(e);
    for (std::int64_t i = 0; i < kSamples; ++i) {
      const double t = r.samples.start_time + static_cast<double>(i) / kFs - tau;
      x[i] += Cf(scale * std::exp(-0.5 * t * t / (kSigma * kSigma)));
    }
  }
}

void add_noise(PingRecord& r, double rms, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> g(0.0F, static_cast<float>(rms / std::sqrt(2.0)));
  for (float& v : r.samples.data) v += g(gen);
}

struct Truth {
  double range, cone, elevation;
};

Truth truth(const ElevationFrame& f, const Vec3d& target) {
  const Vec3d d = target - f.origin;
  const double r = norm(d);
  return {r, std::asin(dot(d, f.along) / r), std::asin(dot(d, f.up) / r)};
}

ElevationEstimate estimate_one(const PingRecord& rec, const Vec3d& target, double background = 1e-30, ElevationOptions o = {}) {
  const auto rows = find_rows(rec);
  const ElevationFrame f = elevation_frame(rec, rows, true);
  const Truth t = truth(f, target);
  return estimate_elevation(rec, rows, {{t.range + 0.5, t.cone, std::nan("")}}, {background}, f, o,
                            static_cast<double>(rec.samples.n_channels), 2.0)[0];
}

TEST(Rows, AreFoundFromThePositionsAndSurviveMissingChannels) {
  const PingRecord rec = empty_record({-0.19, 0.0, 0.19}, {0.0, 0.079, 0.0});
  const auto rows = find_rows(rec);
  ASSERT_EQ(rows.size(), 3U);
  for (std::size_t k = 0; k < 3; ++k) {
    ASSERT_EQ(rows[k].size(), static_cast<std::size_t>(kPerRow));
    EXPECT_EQ(rows[k].front(), static_cast<std::int64_t>(k) * kPerRow);
  }
  // Take out a hydrophone of the first row and all but two of the last: two rows are left.
  std::vector<std::int64_t> keep;
  for (std::int64_t c = 0; c < 3 * kPerRow; ++c) {
    if (c != 3 && c < 2 * kPerRow + 2) keep.push_back(c);
  }
  const PingRecord less = row_record(rec, keep);
  EXPECT_EQ(less.samples.n_channels, static_cast<std::int64_t>(keep.size()));
  EXPECT_EQ(less.geometry.rx.size(), keep.size());
  const auto left = find_rows(less);
  ASSERT_EQ(left.size(), 2U);
  EXPECT_EQ(left[0].size(), static_cast<std::size_t>(kPerRow - 1));
  const ElevationFrame f = elevation_frame(less, left, true);
  EXPECT_NEAR(f.along.x, 1.0, 1e-12);
  EXPECT_NEAR(f.up.z, -1.0, 1e-12);
  EXPECT_NEAR(f.look.y, 1.0, 1e-12);
  EXPECT_NEAR(elevation_frame(less, left, false).look.y, -1.0, 1e-12);
  // One line is one row.
  EXPECT_EQ(find_rows(empty_record({0.0}, {0.0})).size(), 1U);
}

TEST(Elevation, IsExactWithoutNoiseWithTheVehicleRolledAndGivesTheDepth) {
  for (const auto& [roll_deg, target] : {std::pair<double, Vec3d>{0.0, {100.0, 500.0, 10.0}}, {5.0, {-150.0, 480.0, 120.0}},
                                         {-8.0, {40.0, 520.0, 50.0}}}) {
    PingRecord rec = empty_record({-0.19, 0.19}, {0.0, 0.0}, roll_deg * kDeg);
    add_echo(rec, target, 1.0);
    const ElevationFrame f = elevation_frame(rec, find_rows(rec), true);
    const Truth t = truth(f, target);
    const ElevationEstimate e = estimate_one(rec, target);
    EXPECT_TRUE(e.valid);
    EXPECT_NEAR(std::asin(e.sin_elevation) / kDeg, t.elevation / kDeg, 0.02) << roll_deg;
    EXPECT_GT(e.quality, 0.999);
    EXPECT_NEAR(f.origin.z + e.range * e.world.z, target.z, 0.3) << roll_deg;     // depth, the roll taken out
    EXPECT_NEAR(e.range, t.range, 0.2);
  }
}

TEST(Elevation, SigmaIsTheCramerRaoBoundAtTheMeasuredSnrAndErrorsStayWithinIt) {
  const Vec3d target{0.0, 500.0, 20.0};
  const double amplitude = 1.0, noise = 0.5;            // 6 dB at a hydrophone
  std::vector<double> errors;
  double sigma = 0.0;
  for (unsigned seed = 0; seed < 40; ++seed) {
    PingRecord rec = empty_record({-0.19, 0.19}, {0.0, 0.0});
    add_echo(rec, target, amplitude);
    add_noise(rec, noise, seed);
    const ElevationFrame f = elevation_frame(rec, find_rows(rec), true);
    const ElevationEstimate e = estimate_one(rec, target, 2.0 * kPerRow * noise * noise);
    errors.push_back(std::asin(e.sin_elevation) - truth(f, target).elevation);
    sigma += e.sigma / 40.0;
    EXPECT_NEAR(e.snr_db, 6.0, 1.5);
  }
  // 32 hydrophones at +-0.19 m: sum (h - mean)^2 = 32 x 0.19^2.
  const double bound = 1.0 / (rms_wavenumber(4000.0, 6000.0, kC) * std::sqrt(2.0 * 4.0 * 32.0 * 0.19 * 0.19));
  EXPECT_NEAR(rms_wavenumber(4000.0, 6000.0, kC), 2.0 * kPi / kC * std::sqrt(5000.0 * 5000.0 + 2000.0 * 2000.0 / 12.0), 1e-9);
  EXPECT_NEAR(sigma, bound, 0.15 * bound);
  double mean = 0.0, var = 0.0;
  for (double v : errors) mean += v / 40.0;
  for (double v : errors) var += (v - mean) * (v - mean) / 40.0;
  EXPECT_LT(std::abs(mean), 3.0 * bound / std::sqrt(40.0));
  EXPECT_GT(std::sqrt(var), bound / 1.4);
  EXPECT_LT(std::sqrt(var), bound * 1.4);
}

TEST(Elevation, TwoRowsFlagASteepArrivalAndThreeRowsResolveIt) {
  const Vec3d steep{0.0, 470.0, 50.0 - 171.0};          // 20 degrees up
  PingRecord two = empty_record({-0.19, 0.19}, {0.0, 0.0});
  add_echo(two, steep, 1.0);
  EXPECT_TRUE(estimate_one(two, steep).ambiguous);       // rows 0.38 m apart: a second lobe inside +-30 degrees
  PingRecord three = empty_record({-0.19, 0.0, 0.19}, {0.0, 0.079, 0.0});
  add_echo(three, steep, 1.0);
  const ElevationEstimate e = estimate_one(three, steep);
  EXPECT_FALSE(e.ambiguous);
  EXPECT_NEAR(std::asin(e.sin_elevation) / kDeg, truth(elevation_frame(three, find_rows(three), true), steep).elevation / kDeg, 0.05);
  PingRecord level = empty_record({-0.19, 0.19}, {0.0, 0.0});
  add_echo(level, {0.0, 500.0, 50.0}, 1.0);
  EXPECT_FALSE(estimate_one(level, {0.0, 500.0, 50.0}).ambiguous);
}

TEST(Elevation, TakesTheFirstArrivalOfSeveralInTheRangeWindow) {
  // A direct echo and, 3 m later and 3 dB stronger, its surface bounce from 6 degrees above.
  PingRecord rec = empty_record({-0.19, 0.19}, {0.0, 0.0});
  const Vec3d direct{0.0, 500.0, 60.0};
  const double up = 503.0 * std::sin(6.0 * kDeg);
  add_echo(rec, direct, 1.0);
  add_echo(rec, {0.0, std::sqrt(503.0 * 503.0 - up * up), 50.0 - up}, 1.4);
  const auto rows = find_rows(rec);
  const ElevationFrame f = elevation_frame(rec, rows, true);
  const Truth t = truth(f, direct);
  ElevationOptions o;
  const double noise = 1e-4;
  // The detection's centroid sits between the two; its cluster starts at the direct echo.
  const std::vector<ElevationHit> hit{{t.range + 2.0, t.cone, t.range - 1.0}};
  ElevationEstimate e = estimate_elevation(rec, rows, hit, {32.0 * noise}, f, o, 32.0, 2.0)[0];
  EXPECT_NEAR(std::asin(e.sin_elevation) / kDeg, t.elevation / kDeg, 0.3);
  o.first_arrival = false;
  e = estimate_elevation(rec, rows, hit, {32.0 * noise}, f, o, 32.0, 2.0)[0];
  EXPECT_NEAR(std::asin(e.sin_elevation) / kDeg, 6.0, 0.3);
}

TEST(Elevation, RowsManyWavelengthsApartOfferCandidatesAndAThirdRowOrABoundedSectorDecides) {
  // The 2 m x 9 m face: rows 1.8 m apart repeat every 9.6 degrees at 5 kHz.
  const Vec3d target{80.0, 500.0, 50.0 - 44.0};          // 5 degrees up
  PingRecord two = empty_record({-0.9, 0.9}, {0.0, 0.0});
  add_echo(two, target, 1.0);
  const ElevationFrame f2 = elevation_frame(two, find_rows(two), true);
  const double want = truth(f2, target).elevation / kDeg;
  ElevationEstimate e = estimate_one(two, target);
  EXPECT_TRUE(e.ambiguous);
  ASSERT_GE(e.alternatives.size(), 2U);
  // The candidates are the repeats, and the truth is the estimate or one of them.
  double nearest = std::abs(std::asin(e.sin_elevation) / kDeg - want);
  for (const Vec3d& w : e.alternatives) nearest = std::min(nearest, std::abs(std::asin(-w.z) / kDeg - want));
  EXPECT_LT(nearest, 0.1);
  // Searching only half a repeat either side of the truth: one answer.
  ElevationOptions narrow;
  narrow.first_deg = want - 4.5;
  narrow.last_deg = want + 4.5;
  e = estimate_one(two, target, 1e-30, narrow);
  EXPECT_FALSE(e.ambiguous);
  EXPECT_NEAR(std::asin(e.sin_elevation) / kDeg, want, 0.05);
  // The same from the water depth: 12 m of water, the array at 6 m, one bounce, 500 m away: nothing arrives
  // from more than 2 degrees off level (plus the margin), well inside half a repeat.
  PingRecord shallow = empty_record({-0.9, 0.9}, {0.0, 0.0}, 0.0, 6.0);
  const Vec3d near_level{80.0, 500.0, 9.0};
  add_echo(shallow, near_level, 1.0);
  ElevationOptions prior;
  prior.sector_from_depth = true;
  prior.sector_bounces = 1.0;
  prior.water_depth = 12.0;
  e = estimate_one(shallow, near_level, 1e-30, prior);
  EXPECT_FALSE(e.ambiguous);
  EXPECT_NEAR(elevation_frame(shallow, find_rows(shallow), true).origin.z + e.range * e.world.z, 9.0, 0.3);
  EXPECT_TRUE(estimate_one(shallow, near_level).ambiguous);
  // Three rows, the middle one 0.55 m above the lowest: the highest repeat is 4 dB down, one answer.
  PingRecord three = empty_record({-0.9, -0.35, 0.9}, {0.0, 0.0, 0.0});
  add_echo(three, target, 1.0);
  e = estimate_one(three, target);
  EXPECT_FALSE(e.ambiguous);
  EXPECT_TRUE(e.alternatives.empty());
  EXPECT_NEAR(std::asin(e.sin_elevation) / kDeg, truth(elevation_frame(three, find_rows(three), true), target).elevation / kDeg, 0.05);
  // ... unless the noise could swap them: 48 hydrophones at -9 dB each.
  e = estimate_one(three, target, 48.0 * 8.0);
  EXPECT_TRUE(e.ambiguous);
}

TEST(Elevation, UsesTheAttitudeAtTheEchosArrivalWhenTheRecordHasItsHistory) {
  // The vehicle was level at the ping and rolled 3 degrees by the time the echo came back (0.67 s later; the
  // pulse here is 10 ms). The echo is made with the rolled array; the record's pose says level.
  const Vec3d target{0.0, 500.0, 80.0};
  PingRecord rolled = empty_record({-0.9, -0.35, 0.9}, {0.0, 0.0, 0.0}, 3.0 * kDeg);
  add_echo(rolled, target, 1.0);
  PingRecord rec = empty_record({-0.9, -0.35, 0.9}, {0.0, 0.0, 0.0}, 0.0);
  rec.samples.data = rolled.samples.data;
  for (int i = 0; i <= 8; ++i) {
    InsSample a{};
    a.t = 0.1 * i;
    a.rpy = Vec3d{4.5 * kDeg * a.t, 0.0, 0.0};          // 3 degrees at 0.667 s
    rec.navigation.ins.push_back(a);
  }
  Vec3d rpy;
  double gap = 1.0;
  ASSERT_TRUE(attitude_at(rec, 0.25, rpy, gap));
  EXPECT_NEAR(rpy.x / kDeg, 1.125, 1e-9);
  EXPECT_EQ(gap, 0.0);
  ASSERT_TRUE(attitude_at(rec, 1.0, rpy, gap));
  EXPECT_NEAR(gap, 0.2, 1e-12);
  const ElevationFrame f = elevation_frame(rec, find_rows(rec), true);
  ElevationOptions o;
  ElevationEstimate e = estimate_one(rec, target, 1e-30, o);
  EXPECT_NEAR(std::abs(f.origin.z + e.range * e.world.z - target.z), 500.0 * std::sin(3.0 * kDeg), 2.0);   // 26 m with the pose's attitude
  o.attitude_arrival = true;
  o.attitude_rate_sigma = 1.0 * kDeg;
  e = estimate_one(rec, target, 1e-30, o);
  EXPECT_NEAR(f.origin.z + e.range * e.world.z, target.z, 0.6);
  EXPECT_EQ(e.attitude_gap, 0.0);
  // No history: the pose's attitude, and the sigma owns up to the time of flight at the rate the vehicle may roll.
  rec.navigation.ins.clear();
  e = estimate_one(rec, target, 1e-30, o);
  EXPECT_NEAR(e.attitude_gap, 2.0 * 500.9 / kC + 0.005, 0.01);
  EXPECT_NEAR(e.sigma / kDeg, e.attitude_gap, 0.01);
}

TEST(Depth, SigmaTermsThatDoNotComeFromNoise) {
  EXPECT_NEAR(refraction_sigma(3000.0, 1500.0, 0.017), 51.0, 1e-9);     // isothermal water, 3 km
  EXPECT_NEAR(refraction_sigma(12000.0, 1500.0, 0.05), 2400.0, 1e-9);
  EXPECT_TRUE(outside_water(125.0, 10.0, 100.0));
  EXPECT_FALSE(outside_water(125.0, 10.0, 100.0, 0.0, 2.0, 10.0));      // the bottom itself known to 10 m
  // The assumed-gradient ray: range^2 g / 2c above the straight line.
  const double g = 0.05, own = 100.0;
  const double z = ray_depth(own, -2.0 * kDeg, 3000.0, {0.0, 10100.0}, {1500.0 - g * own, 1500.0 + g * 10000.0});
  EXPECT_NEAR(z - (own + 3000.0 * std::sin(2.0 * kDeg)), -3000.0 * 3000.0 * g / 3000.0, 6.0);
}

TEST(Tracker, ChoosesAmongAnAmbiguousDetectionsCandidates) {
  TrackerOptions o;
  o.confirm_hits = 1;
  o.depth_process_noise = 0.0;
  Tracker tracker(o);
  auto det = [](double depth, std::vector<double> others, bool ambiguous) {
    Detection d{500.0, 0.1, 20.0, 1};
    d.depth = depth;
    d.depth_sigma = 4.0;
    d.depth_ambiguous = ambiguous;
    d.depth_alternatives = std::move(others);
    return d;
  };
  tracker.update(0.0, {det(136.0, {52.0}, true)});
  EXPECT_TRUE(std::isnan(tracker.tracks()[0].depth));                  // a track does not start on a guess
  tracker.update(1.0, {det(50.0, {}, false)});
  EXPECT_NEAR(tracker.tracks()[0].depth, 50.0, 1e-9);
  tracker.update(2.0, {det(136.0, {52.0, -32.0}, true)});              // the candidate that fits
  EXPECT_NEAR(tracker.tracks()[0].depth, 51.0, 1e-9);
  tracker.update(3.0, {det(136.0, {220.0}, true)});                    // none fits: left alone
  EXPECT_NEAR(tracker.tracks()[0].depth, 51.0, 1e-9);
}

TEST(Depth, FoldsBounceArrivalsBackIntoTheWaterColumn) {
  EXPECT_EQ(fold_depth(50.0, 100.0).second, "direct");
  EXPECT_NEAR(fold_depth(-50.0, 100.0).first, 50.0, 1e-12);
  EXPECT_EQ(fold_depth(-50.0, 100.0).second, "surface");
  EXPECT_NEAR(fold_depth(150.0, 100.0).first, 50.0, 1e-12);
  EXPECT_EQ(fold_depth(150.0, 100.0).second, "bottom");
  EXPECT_EQ(fold_depth(250.0, 100.0).second, "multiple");
  EXPECT_NEAR(fold_depth(-150.0, 100.0).first, 50.0, 1e-12);
  EXPECT_NEAR(fold_depth(-20.0, std::nan("")).first, 20.0, 1e-12);     // no water depth: the surface only
  EXPECT_EQ(fold_depth(300.0, 0.0).second, "direct");
  EXPECT_NEAR(fold_depth(5.0, 100.0, 10.0).first, 15.0, 1e-12);        // the surface at z = 10 in the navigation's frame
  EXPECT_TRUE(outside_water(-30.0, 10.0, 100.0));
  EXPECT_FALSE(outside_water(-15.0, 10.0, 100.0));
  EXPECT_TRUE(outside_water(125.0, 10.0, 100.0));
  EXPECT_FALSE(outside_water(125.0, 10.0, std::nan("")));
}

TEST(Tracker, CarriesDepthFromTheDetectionsThatHaveOne) {
  TrackerOptions o;
  o.confirm_hits = 2;
  o.depth_process_noise = 0.0;
  Tracker tracker(o);
  std::vector<const Track*> confirmed;
  for (int i = 0; i < 3; ++i) {
    Detection d{500.0, 0.1, 20.0, 1};
    if (i != 1) {
      d.depth = i == 0 ? 40.0 : 60.0;
      d.depth_sigma = 4.0;
    }
    confirmed = tracker.update(static_cast<double>(i), {d});
  }
  ASSERT_EQ(confirmed.size(), 1U);
  EXPECT_NEAR(confirmed[0]->depth, 50.0, 1e-9);          // two equal measurements
  EXPECT_NEAR(confirmed[0]->depth_var, 8.0, 1e-9);
}

TEST(PassiveElevation, FindsABroadbandSourceAboveTheArray) {
  // White noise from 12 degrees up at 20 degrees off broadside, as a plane wave on two rolled rows.
  PingRecord rec = empty_record({-0.19, 0.19}, {0.0, 0.0}, 4.0 * kDeg);
  const auto rows = find_rows(rec);
  const ElevationFrame f = elevation_frame(rec, rows, true);
  const double cone = 20.0 * kDeg, el = 12.0 * kDeg;
  const Vec3d u = arrival_direction(f, std::sin(cone), std::sin(el));
  std::mt19937 gen(5);
  std::normal_distribution<double> g(0.0, 1.0);
  // A sum of tones on the FFT grid of the record: exact delays per element.
  const std::int64_t n_tones = 200;
  for (std::int64_t q = 0; q < n_tones; ++q) {
    const double f_bb = -1000.0 + 2000.0 * (static_cast<double>(q) + 0.37) / static_cast<double>(n_tones);
    const Cd a(g(gen), g(gen));
    for (std::int64_t e = 0; e < rec.samples.n_channels; ++e) {
      const Vec3d p = world_position(rec.geometry.rx_pose, rec.geometry.rx[static_cast<std::size_t>(e)].lever_arm) - f.origin;
      const double lead = dot(p, u) / kC;
      Cf* x = rec.samples.channel_complex(e);
      for (std::int64_t i = 0; i < kSamples; ++i) {
        x[i] += Cf(a * std::polar(0.1, 2.0 * kPi * (f_bb * static_cast<double>(i) / kFs + (kFc + f_bb) * lead)));
      }
    }
  }
  add_noise(rec, 1.0, 9);
  const PassiveElevation e = passive_elevation(rec, rows, cone, 4000.0, 6000.0, 256, 128, 0, -1, -30.0, 30.0, 0.5, true);
  EXPECT_NEAR(e.elevation / kDeg, 12.0, 0.5);
  EXPECT_GT(e.quality, 0.5);
}

ChainOptions small_chain() {
  ChainOptions o;
  o.bearing_first_deg = -40.0;
  o.bearing_last_deg = 40.0;
  o.bearing_step_deg = 1.0;
  o.range_cell_m = 1.0;
  o.cfar_guard_m = 6.0;
  o.cfar_train_m = 30.0;
  o.cfar_pfa = 1e-6;
  o.min_snr_db = 12.0;
  o.classify.multipath = false;
  return o;
}

std::string without_time(const std::string& json) { return json.substr(0, json.find("\"processing_seconds\"")); }

TEST(Chain, ReportsElevationAndDepthWithRowsAndIsUnchangedWithOne) {
  const Vec3d target{80.0, 520.0, 95.0};
  // Two rows: every detection has an elevation; the strongest is the target, at its depth.
  PingRecord two = empty_record({-0.19, 0.19}, {0.0, 0.0}, 3.0 * kDeg);
  add_echo(two, target, 1.0);
  add_noise(two, 0.02, 1);
  ChainOptions o = small_chain();
  o.elevation.enabled = true;
  LiveChain chain(o);
  const Report rep = chain.process(two);
  EXPECT_EQ(rep.rows, 2);
  ASSERT_FALSE(rep.detections.empty());
  const ReportDetection* best = &rep.detections[0];
  for (const auto& d : rep.detections) {
    if (d.snr_db > best->snr_db) best = &d;
  }
  EXPECT_TRUE(best->elevation_valid);
  EXPECT_NEAR(best->image_depth_m, target.z, 2.0);
  EXPECT_NEAR(best->depth_m, target.z, 2.0);
  EXPECT_EQ(best->path, "direct");
  EXPECT_NEAR(best->elevation_deg, -std::asin(45.0 / norm(target - Vec3d{0.0, 0.0, 50.0})) / kDeg, 0.2);
  EXPECT_NE(rep.json().find("\"depth_m\""), std::string::npos);
  EXPECT_NE(rep.json().find("\"rows\": 2"), std::string::npos);
  // One row: the option changes nothing, to the last digit of the report line.
  std::string lines[2];
  for (int on = 0; on < 2; ++on) {
    PingRecord one = empty_record({0.0}, {0.0}, 3.0 * kDeg);
    add_echo(one, target, 1.0);
    add_noise(one, 0.02, 1);
    ChainOptions c = small_chain();
    c.elevation.enabled = on == 1;
    LiveChain single(c);
    lines[on] = without_time(single.process(one).json());
  }
  EXPECT_EQ(lines[0], lines[1]);
  EXPECT_EQ(lines[0].find("elevation"), std::string::npos);
  EXPECT_NE(lines[0].find("\"range_m\""), std::string::npos);
}

TEST(Chain, LeavesOutDeadChannels) {
  const Vec3d target{80.0, 520.0, 95.0};
  PingRecord rec = empty_record({-0.19, 0.19}, {0.0, 0.0});
  add_echo(rec, target, 1.0);
  add_noise(rec, 0.02, 2);
  // A dead hydrophone that only makes noise, and half a board of the other row.
  std::mt19937 gen(3);
  std::normal_distribution<float> g(0.0F, 5.0F);
  for (std::int64_t i = 0; i < 2 * kSamples; ++i) rec.samples.channel(5)[i] = g(gen);
  ChainOptions o = small_chain();
  o.elevation.enabled = true;
  o.dead_channels = {5, 24, 25, 26, 27};
  LiveChain chain(o);
  const Report rep = chain.process(rec);
  ASSERT_FALSE(rep.detections.empty());
  const ReportDetection* best = &rep.detections[0];
  for (const auto& d : rep.detections) {
    if (d.snr_db > best->snr_db) best = &d;
  }
  EXPECT_TRUE(best->elevation_valid);
  EXPECT_NEAR(best->image_depth_m, target.z, 3.0);
  ChainOptions bad = small_chain();
  bad.dead_channels = {99};
  LiveChain refuses(bad);
  PingRecord again = empty_record({-0.19, 0.19}, {0.0, 0.0});
  EXPECT_THROW(refuses.process(again), std::invalid_argument);
}

}  // namespace
