// The C++ live library on its own: clustering, tracking, classification, SBL bearings and
// the board-stream assembler. tests/python/test_live_cpp.py and test_doa.py hold the same
// code against the Python stages and against simulated pings.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/coverage.hpp"
#include "sonar/live/multipath_depth.hpp"
#include "sonar/live/raytrace.hpp"
#include "sonar/sim/paths.hpp"
#include "sonar/core/platform.hpp"
#include "sonar/live/option_value.hpp"
#include "sonar/live/options.hpp"
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

TEST(Tracker, TakesEachDetectionAtItsOwnTimeAndClassifiesAtThePing) {
  // D-156: a ship at 12 km crossing at 6 m/s, pinged every 20 s by a monostatic sonar at the origin; each detection
  // shows it where it was when the sound reached it, transmit + R/c. Stamped so, the track holds at its detections'
  // times and its fitted position at the ping time is the ship's then; stamped at the transmit time (the rule before,
  // and what a detection without a time gets) it is v R / c = 48 m ahead along its course.
  const double c = 1500.0, vx = 6.0, x0 = -60.0, y0 = 12000.0;
  TrackerOptions o;
  o.range_sigma = 2.0;
  o.bearing_sigma = 0.01 * kDeg;
  Tracker timed(o), plain(o), at_ping(o);
  std::vector<const Track*> a, b, p;
  double tp = 0.0, t_last = 0.0;
  for (int k = 0; k < 10; ++k) {
    tp = 20.0 * k;
    double tr = tp;
    for (int i = 0; i < 8; ++i) tr = tp + std::hypot(x0 + vx * tr, y0) / c;
    const double px = x0 + vx * tr, py = y0;
    Detection d{std::hypot(px, py), std::atan2(px, py), 20.0, 4};   // (across, forward)
    p = plain.update(tp, {d});
    Detection same = d;
    same.time = tp;
    EXPECT_TRUE(std::isnan(d.time));
    const auto q = at_ping.update(tp, {same});
    d.time = tp + d.range / c;
    t_last = d.time;
    a = timed.update(tp, {d});
    ASSERT_EQ(q.size(), p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {   // no time is the ping's time, bit for bit
      for (int j = 0; j < 4; ++j) EXPECT_EQ(q[i]->state[j], p[i]->state[j]);
    }
  }
  ASSERT_EQ(a.size(), 1U);
  ASSERT_EQ(p.size(), 1U);
  EXPECT_NEAR(a[0]->time, t_last, 1e-9);
  EXPECT_EQ(a[0]->history.back().time, a[0]->time);
  EXPECT_EQ(p[0]->time, tp);
  ClassifyOptions co;
  co.time = tp;
  const auto ca = classify_tracks(a, co), cp = classify_tracks(p, co);
  const double truth_x = x0 + vx * tp;
  EXPECT_NEAR(ca[0].x, truth_x, 0.5);
  EXPECT_NEAR(ca[0].y, y0, 0.5);
  EXPECT_NEAR(cp[0].x - truth_x, vx * 8.0, 3.0);
  EXPECT_NEAR(ca[0].speed, vx, 0.1);
  // Without a time the fit is read at the last point: the last detection's time for the stamped track.
  const auto at_last = classify_tracks(a, ClassifyOptions{});
  EXPECT_NEAR(at_last[0].x - ca[0].x, vx * (t_last - tp), 0.05);
}

TEST(Tracker, NeverMovesAMissedTrackBackAndKeepsItsDepthVariancePositive) {
  // D-156 with pings every 5 s, less than R/c = 8 s at 12 km: a track that took a detection at t_k + R/c and misses ping
  // k+1 already holds later than t_{k+1}. It keeps its state, time and depth variance (predicted back to the ping, the
  // depth variance lost q_d^2 x 3 s and went below zero, and the history went out of time order).
  const double c = 1500.0, vx = 6.0, x0 = -60.0, y0 = 12000.0;
  TrackerOptions o;
  o.bearing_sigma = 0.01 * kDeg;
  o.initial_speed_sigma = 8.0;
  Tracker tracker(o);
  int held = 0;
  for (int k = 0; k < 14; ++k) {
    const double tp = 5.0 * k;
    double tr = tp;
    for (int i = 0; i < 8; ++i) tr = tp + std::hypot(x0 + vx * tr, y0) / c;
    const double px = x0 + vx * tr;
    std::vector<Detection> dets;
    const bool miss = k == 4 || k == 7 || k == 8;
    if (!miss) {
      Detection d{std::hypot(px, y0), std::atan2(px, y0), 20.0, 4};
      d.depth = 30.0;
      d.depth_sigma = 0.3;
      d.time = tp + d.range / c;
      dets.push_back(d);
    }
    std::vector<double> before_time, before_var;
    for (const Track& t : tracker.tracks()) {
      before_time.push_back(t.time);
      before_var.push_back(t.depth_var);
    }
    tracker.update(tp, dets);
    ASSERT_EQ(tracker.tracks().size(), 1U);
    const Track& t = tracker.tracks()[0];
    for (std::size_t i = 1; i < t.history.size(); ++i) EXPECT_LE(t.history[i - 1].time, t.history[i].time);
    EXPECT_GT(t.depth_var, 0.0);
    if (miss && !before_time.empty() && before_time[0] > tp) {
      EXPECT_EQ(t.time, before_time[0]);
      EXPECT_EQ(t.depth_var, before_var[0]);
      ++held;
    }
  }
  EXPECT_EQ(held, 2);
}

TEST(Tracker, AnEarlierDetectionIsTakenAtTheTracksTimeWithItsDepthVariance) {
  // A detection earlier than the time the track holds (D-156: a nearer echo is earlier). Before sonar-uc5 the track was
  // predicted back to it and gained q_d^2 |dt| of depth variance; now it takes it at its own time (dt = 0, no history
  // step back), so the update weighs the track's own depth variance against the detection's and the time stays 10 s.
  TrackerOptions o;
  o.depth_process_noise = 0.5;
  Tracker tracker(o);
  Detection first{1000.0, 0.0, 20.0, 4};
  first.depth = 20.0;
  first.depth_sigma = 0.4;
  first.time = 10.0;
  tracker.update(0.0, {first});
  Detection second = first;
  second.depth = 21.0;
  second.time = 9.0;
  tracker.update(1.0, {second});
  ASSERT_EQ(tracker.tracks().size(), 1U);
  const Track& t = tracker.tracks()[0];
  const double prior = 0.16, r = 0.16, k = prior / (prior + r);
  EXPECT_NEAR(t.depth_var, prior * (1.0 - k), 1e-12);
  EXPECT_NEAR(t.depth, 20.0 + k * 1.0, 1e-12);
  EXPECT_EQ(t.time, 10.0);
  EXPECT_EQ(t.history.back().time, 10.0);
}

TEST(Tracker, ADetectionTimeThatIsNotFiniteIsThePings) {
  TrackerOptions o;
  Tracker plain(o), inf(o);
  for (int k = 0; k < 5; ++k) {
    Detection d{200.0 + 3.0 * k, 0.2, 20.0, 4};
    plain.update(0.5 * k, {d});
    d.time = std::numeric_limits<double>::infinity();
    inf.update(0.5 * k, {d});
  }
  ASSERT_EQ(plain.tracks().size(), inf.tracks().size());
  for (int j = 0; j < 4; ++j) EXPECT_EQ(plain.tracks()[0].state[j], inf.tracks()[0].state[j]);
  EXPECT_EQ(plain.tracks()[0].time, inf.tracks()[0].time);
}

TEST(Classify, FitsTimeBetweenAllPointsAndMovesAShortTrackByItsState) {
  // The fit's guard is max - min of the times (numpy.ptp in classify.py), not last - first.
  Track t;
  t.id = 1;
  t.hits = 3;
  t.confirmed = true;
  t.history = {{0.0, 0.0, 0.0}, {10.0, 10.0, 0.0}, {0.0, 0.0, 0.0}};
  ClassifyOptions co;
  auto out = classify_tracks({&t}, co);
  EXPECT_NEAR(out[0].vx, 1.0, 1e-12);
  EXPECT_NEAR(out[0].vy, 0.0, 1e-12);
  // Fewer than three points: no fit; at a time given, the last point moved there by the state's velocity.
  Track s;
  s.id = 2;
  s.hits = 2;
  s.confirmed = true;
  s.history = {{0.0, 46.0, 50.0}, {10.0, 106.0, 50.0}};
  s.state[0] = 106.0;
  s.state[1] = 50.0;
  s.state[2] = 6.0;
  co.time = 2.0;
  out = classify_tracks({&s}, co);
  EXPECT_NEAR(out[0].x, 58.0, 1e-12);
  EXPECT_NEAR(out[0].y, 50.0, 1e-12);
  EXPECT_EQ(out[0].vx, 0.0);
  for (double at : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    co.time = at;
    out = classify_tracks({&s}, co);
    EXPECT_EQ(out[0].x, 106.0);
    EXPECT_EQ(out[0].y, 50.0);
  }
}

TEST(Classify, LeavesOneGlintingMeasurementOutOfTheLine) {
  // sonar-ljh (classify.py _outlier): six measured points of a 6 m/s crossing with a small scatter, the last 50 m
  // ahead (80 m: six points give the per-axis test 3 degrees of freedom, so it needs |t| > 13.7, and 50 m over this
  // scatter is |t| 11.9). At 0.01 (opt-in, D-159) the deleted-residual test leaves it out and the line holds 6 m/s; at outlier_alpha 0 it is kept
  // and pulls the line. A track whose rows record no measurement (the filtered history) is never tested.
  Track t;
  t.id = 1;
  t.hits = 6;
  t.confirmed = true;
  const double scatter[6] = {-3.0, 3.0, 2.0, -2.0, 1.0, 80.0};
  for (int k = 0; k < 6; ++k) {
    const double tk = 20.0 * k, x = 6.0 * tk;
    t.history.push_back({tk, x, 12000.0, x + scatter[k], 12000.0 + (k % 2 ? 1.0 : -1.0)});
  }
  ClassifyOptions co;
  EXPECT_EQ(co.outlier_alpha, 0.0);   // off by default (D-159); the test itself at 0.01
  co.outlier_alpha = 0.01;
  co.time = 100.0;
  auto out = classify_tracks({&t}, co);
  EXPECT_NEAR(out[0].vx, 6.0, 0.1);
  EXPECT_NEAR(out[0].x, 600.0, 3.0);
  co.outlier_alpha = 0.0;
  out = classify_tracks({&t}, co);
  EXPECT_GT(out[0].vx, 6.3);
  EXPECT_GT(out[0].x - 600.0, 15.0);
  Track h = t;
  for (auto& p : h.history) {
    p.x = p.mx;
    p.mx = p.my = std::numeric_limits<double>::quiet_NaN();
  }
  co.outlier_alpha = 0.01;
  EXPECT_GT(classify_tracks({&h}, co)[0].vx, 6.3);
}

TEST(Options, ClassifyOutlierAlphaIsReadEchoedAndChecked) {
  ChainOptions a;
  PassiveOptions pa;
  EXPECT_EQ(a.classify.outlier_alpha, 0.0);   // off by default (D-159)
  std::istringstream in("classify_outlier_alpha=0.01\n");
  read_options(a, pa, in);
  EXPECT_EQ(a.classify.outlier_alpha, 0.01);
  EXPECT_NE(options_json(a).find("\"classify_outlier_alpha\": 0.01"), std::string::npos) << options_json(a);
  for (const char* bad : {"classify_outlier_alpha=-0.1\n", "classify_outlier_alpha=1\n", "classify_outlier_alpha=nan\n"}) {
    ChainOptions b;
    std::istringstream t(bad);
    EXPECT_THROW(read_options(b, pa, t), std::invalid_argument) << bad;
  }
}

TEST(Options, ClassifyFitRmsBearingIsReadEchoedAndChecked) {
  // sonar-4jm review: left out it is derived (NaN until the first record); given, a finite angle, 0 (off) or more, in the
  // reader and in the chain itself.
  ChainOptions a;
  PassiveOptions pa;
  EXPECT_TRUE(std::isnan(a.classify.max_fit_rms_bearing));
  std::istringstream in("classify_max_fit_rms_bearing_deg=0.5\n");
  read_options(a, pa, in);
  EXPECT_DOUBLE_EQ(a.classify.max_fit_rms_bearing, 0.5 * kDeg);
  EXPECT_NE(options_json(a).find("\"classify_max_fit_rms_bearing_deg\": 0.5"), std::string::npos) << options_json(a);
  for (const char* bad : {"classify_max_fit_rms_bearing_deg=-1\n", "classify_max_fit_rms_bearing_deg=inf\n",
                          "classify_max_fit_rms_bearing_deg=nan\n"}) {
    ChainOptions b;
    std::istringstream t(bad);
    EXPECT_THROW(read_options(b, pa, t), std::invalid_argument) << bad;
  }
  for (double bad : {-1.0, std::numeric_limits<double>::infinity()}) {
    ChainOptions c;
    c.classify.max_fit_rms_bearing = bad;
    EXPECT_THROW(LiveChain{c}, std::invalid_argument) << bad;
  }
}

TEST(Tracker, SharesDetectionsMostProbableFirstByDefaultAndKeepsTheOldRuleByName) {
  // D-141: the default and the names; the two rules themselves are held to the Python tracker's in
  // tests/python/test_live_cpp.py, and the case that separates them in tests/python/test_detect.py.
  EXPECT_EQ(live::TrackerOptions{}.association, live::Association::Likelihood);
  EXPECT_EQ(live::ChainOptions{}.tracker.association, live::Association::Likelihood);
  EXPECT_EQ(live::parse_association("likelihood"), live::Association::Likelihood);
  EXPECT_EQ(live::parse_association("in_order"), live::Association::InOrder);
  EXPECT_THROW(live::parse_association("nearest"), std::invalid_argument);
  live::ChainOptions c;
  live::PassiveOptions p;
  live::set_option(c, p, "tracker_association", "in_order");
  EXPECT_EQ(c.tracker.association, live::Association::InOrder);
  EXPECT_THROW(live::set_option(c, p, "tracker_association", "best"), std::invalid_argument);
}

TEST(Options, AnUnknownKeyIsRefusedWithTheNearestKnownOne) {
  // sonar-1rf: set_option refused an unknown key without saying which was meant.
  live::ChainOptions c;
  live::PassiveOptions p;
  const auto message = [&](const std::string& key) {
    try {
      live::set_option(c, p, key, "1");
    } catch (const std::invalid_argument& e) {
      return std::string(e.what());
    }
    return std::string("accepted");
  };
  EXPECT_EQ(message("classify_min_mover_cell"), "unknown option classify_min_mover_cell; did you mean classify_min_mover_cells?");
  EXPECT_EQ(message("elevation_profile_bin"), "unknown option elevation_profile_bin; did you mean elevation_profile_bin_m?");
  EXPECT_EQ(message("passive_methods"), "unknown option passive_methods; did you mean passive_method?");   // the last key compared
  EXPECT_EQ(message("zzzzzzzzzzzz"), "unknown option zzzzzzzzzzzz");                                     // nothing near
  EXPECT_EQ(live::nearest_key("keep", {"keep_s", "hold_s"}), "keep_s");
  EXPECT_EQ(live::nearest_key("abcdefgh", {"keep_s"}), "");
}

TEST(Options, RefuseTextThatIsNotWhatTheKeyExpects) {
  // A tool that passed Python text on (`decimate={}`, `True`, `None`) used to switch a stage off without a word:
  // atof read it as 0. Every reader now refuses it and names the key.
  ChainOptions c;
  PassiveOptions p;
  for (const char* pair : {"decimate={}", "elevation_profile_ensemble=None", "cfar_pfa=1e-5x", "range_cell_m=", "tracker_confirm_hits=2.5",
                           "elevation_ssp_depth=0,abc", "dead_channels=1,,2", "look=Starboard", "window=hamming", "backend=gpu", "blanking_mode=zeros",
                           "doa=SBL", "multipath_depth_max_depth=deep", "coverage_n_ranges=1.5", "tonal_window=hamming", "nav_surface_z=sea"}) {
    const std::string text = pair;
    const std::string key = text.substr(0, text.find('='));
    try {
      set_pair(c, p, text);
      ADD_FAILURE() << "accepted " << text;
    } catch (const std::invalid_argument& e) {
      EXPECT_NE(std::string(e.what()).find(key), std::string::npos) << e.what();
    }
  }
}

TEST(Options, ReadNumbersSwitchesListsAndWords) {
  ChainOptions c;
  PassiveOptions p;
  for (const char* on : {"1", "true", "True", "2"}) {
    c.elevation.profile.ensemble = false;
    set_option(c, p, "elevation_profile_ensemble", on);
    EXPECT_TRUE(c.elevation.profile.ensemble) << on;
  }
  for (const char* off : {"0", "false", "False", " 0 "}) {
    c.decimate.enabled = true;
    set_option(c, p, "decimate", off);
    EXPECT_FALSE(c.decimate.enabled) << off;
  }
  set_option(c, p, "elevation_profile_ensemble_gradients", "[0.0, 0.017]");
  ASSERT_EQ(c.elevation.profile.ensemble_gradients.size(), 2U);
  EXPECT_DOUBLE_EQ(c.elevation.profile.ensemble_gradients[1], 0.017);
  set_option(c, p, "cfar_integrate_cells", "1,4,16");
  EXPECT_EQ(c.detector.integrate_cells, (std::vector<std::int64_t>{1, 4, 16}));
  set_option(c, p, "decimate_taps", "1e2");
  EXPECT_EQ(c.decimate.taps, 100);
  set_option(c, p, "nav_surface_z", "-1.25");
  EXPECT_DOUBLE_EQ(c.nav.surface_z, -1.25);
  set_option(c, p, "look", "port");
  set_option(c, p, "window", "rect");
  EXPECT_FALSE(c.starboard);
  EXPECT_FALSE(p.starboard);
  EXPECT_FALSE(c.hann);
  set_option(c, p, "backend", "host");
  EXPECT_EQ(c.backend, Backend::Host);
  set_option(c, p, "backend", "auto");                 // as gpusonar.live: CUDA when built with it and a device is present
  EXPECT_EQ(c.backend, cuda_available() ? Backend::Cuda : Backend::Host);
  EXPECT_EQ(p.sbl.backend, c.backend);
  EXPECT_EQ(ChainOptions{}.backend, Backend::Host);    // without the key: the host
  EXPECT_THROW(option_integer("k", "nan"), std::invalid_argument);
  EXPECT_TRUE(option_numbers("k", "").empty());
}

TEST(Options, TheProgramStartsFromAutoAndAChainRefusesCudaWithoutADevice) {
  // Owner's rulings 2026-11-21: sonar_live defaults to auto, as gpusonar.live; cuda never falls back silently.
  ChainOptions c;
  PassiveOptions p;
  program_defaults(c, p);
  EXPECT_EQ(c.backend, cuda_available() ? Backend::Cuda : Backend::Host);
  EXPECT_EQ(p.sbl.backend, c.backend);
  ChainOptions want_cuda;
  want_cuda.backend = Backend::Cuda;
  if (cuda_available()) {
    EXPECT_NO_THROW(LiveChain{want_cuda});
  } else {
    EXPECT_THROW(LiveChain{want_cuda}, std::runtime_error);
  }
  EXPECT_NO_THROW(LiveChain{ChainOptions{}});           // host is always there
}

TEST(Options, FileTextAndDumpAgree) {
  // load_options (a file) and read_options (text) are one reader; options_json says what they set, by key name.
  const std::string text = "# a comment\nlook=port  \nelevation=1\nelevation_profile=1\nelevation_profile_ensemble=1\n"
                           "elevation_profile_ensemble_gradients=0,0.017\nnav_surface_z=0.5 # trailing\ndecimate=1\n";
  ChainOptions a, b;
  PassiveOptions pa, pb;
  std::istringstream in(text);
  read_options(a, pa, in);
  const std::string path = ::testing::TempDir() + "live_options_test.options";
  {
    std::ofstream f(path);
    f << text;
  }
  load_options(b, pb, path);
  std::remove(path.c_str());
  EXPECT_EQ(options_json(a), options_json(b));
  const std::string j = options_json(a);
  for (const char* member : {"\"look\": \"port\"", "\"elevation_profile_ensemble\": true", "\"elevation_profile_ensemble_gradients\": [0, 0.017000000000000001]",
                             "\"nav_surface_z\": 0.5", "\"decimate\": true", "\"range_cell_m\": null"}) {
    EXPECT_NE(j.find(member), std::string::npos) << member << " in " << j;
  }
  EXPECT_NE(options_json(ChainOptions{}), j);
}

TEST(Options, ClassifyDetectionRulesAreOffUnlessGiven) {
  // classify.py `detections` (ours): off by default (below 0), set by sonar_live's key, echoed by options_json.
  ChainOptions a;
  PassiveOptions pa;
  EXPECT_LT(a.classify.min_mover_cells, 0.0);
  EXPECT_LT(a.classify.row_snr_db, 0.0);
  std::istringstream in("classify_min_mover_cells=6\n");
  read_options(a, pa, in);
  EXPECT_DOUBLE_EQ(a.classify.min_mover_cells, 6.0);
  EXPECT_NE(options_json(a).find("\"classify_min_mover_cells\": 6"), std::string::npos) << options_json(a);
  // Given explicitly, a negative or NaN count is refused (as classify.py does), not taken as off.
  for (const char* bad : {"classify_min_mover_cells=-1\n", "classify_min_mover_cells=nan\n"}) {
    ChainOptions b;
    std::istringstream t(bad);
    EXPECT_THROW(read_options(b, pa, t), std::invalid_argument) << bad;
  }
  // The row rule acts only in the corridor test, which the chain never runs: refused with a reason, in the reader and
  // in the chain itself.
  ChainOptions c;
  std::istringstream r("classify_row_snr_db=6\n");
  try {
    read_options(c, pa, r);
    ADD_FAILURE() << "classify_row_snr_db accepted";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("never runs the row"), std::string::npos) << e.what();
  }
  ChainOptions d;
  d.classify.row_snr_db = 6.0;
  EXPECT_THROW(LiveChain{d}, std::invalid_argument);
}

TEST(Report, CarriesPingsWithoutNavigation) {
  Report r;
  EXPECT_NE(r.json().find("\"pings_without_nav\": 0"), std::string::npos);
  r.pings_without_nav = 3;
  EXPECT_NE(r.json().find("\"pings_without_nav\": 3"), std::string::npos);
}

TEST(Report, CarriesPingsWithoutVelocityOnlyWhenThereWereAny) {
  // sonar-8is: a line is the one it was until a ping's navigation had no velocity over ground.
  Report r;
  EXPECT_EQ(r.json().find("pings_without_velocity"), std::string::npos);
  r.pings_without_velocity = 2;
  EXPECT_NE(r.json().find("\"pings_without_nav\": 0, \"pings_without_velocity\": 2, \"detections\""), std::string::npos) << r.json();
  for (const std::string& d : r.datagrams(10)) EXPECT_NE(d.find("\"pings_without_velocity\": 2"), std::string::npos);
}

TEST(Report, CarriesPingsWithoutTimeOnlyWhenThereWereAny) {
  // sonar-4bs: pings whose transmit time did not increase; after pings_without_velocity, on a line only once above 0.
  Report r;
  EXPECT_EQ(r.json().find("pings_without_time"), std::string::npos);
  r.pings_without_time = 4;
  EXPECT_NE(r.json().find("\"pings_without_nav\": 0, \"pings_without_time\": 4, \"detections\""), std::string::npos) << r.json();
  r.pings_without_velocity = 2;
  EXPECT_NE(r.json().find("\"pings_without_velocity\": 2, \"pings_without_time\": 4, \"detections\""), std::string::npos) << r.json();
  for (const std::string& d : r.datagrams(10)) EXPECT_NE(d.find("\"pings_without_time\": 4"), std::string::npos);
}

TEST(Tracker, TakesAnEarlierDetectionAtItsOwnTime) {
  // sonar-uc5: pings 10 ms apart, one standing target whose detection alternates between arrivals at 1500 and 1480 m,
  // each at transmit + R/c: the nearer one is earlier than the time the track holds. It is taken there (dt = 0), so the
  // history stays in time order (before: predicted back, 13 ms each time).
  TrackerOptions o;
  o.range_sigma = 5.0;
  o.bearing_sigma = 0.5 * kDeg;
  o.initial_speed_sigma = 2.0;
  Tracker tracker(o);
  int earlier = 0;
  for (int k = 0; k < 10; ++k) {
    const double tp = 0.01 * k, r = k % 2 ? 1480.0 : 1500.0;
    Detection d{r, 0.05, 20.0, 4};
    d.time = tp + r / 1500.0;
    const double held = tracker.tracks().empty() ? -1.0 : tracker.tracks()[0].time;
    tracker.update(tp, {d});
    ASSERT_EQ(tracker.tracks().size(), 1u);
    const Track& t = tracker.tracks()[0];
    if (d.time < held) {
      ++earlier;
      EXPECT_EQ(t.time, held);
      EXPECT_DOUBLE_EQ(t.history.back().mx, r * std::sin(0.05));
    }
    for (std::size_t i = 1; i < t.history.size(); ++i) EXPECT_LE(t.history[i - 1].time, t.history[i].time);
  }
  EXPECT_GE(earlier, 3);
  EXPECT_EQ(tracker.tracks()[0].hits, 10);
}

TEST(Classify, ABounceIsNeverAnEchoGroupsObject) {
  // sonar-bal (classify.py group_echoes, test_classify.py the same case): three arrivals at one bearing and velocity, the
  // nearest a bounce; a fourth track, a bounce at another bearing. The group forms about the second; the bounce within
  // its gates is its multipath, the other bounce multipath of nothing; no parent is multipath.
  auto tc = [](std::int64_t id, double range, double bearing_deg, double snr) {
    TrackClass c;
    c.id = id;
    c.kind = Kind::Mover;
    c.x = range * std::sin(bearing_deg * kDeg);
    c.y = range * std::cos(bearing_deg * kDeg);
    c.vx = 0.0;
    c.vy = -6.0;
    c.speed = 6.0;
    c.mean_snr_db = snr;
    return c;
  };
  ClassifyOptions o;
  o.multipath = true;
  o.multipath_bearing = 3.0 * kDeg;
  const std::vector<TrackClass> start{tc(1, 6000.0, 4.0, 14.0), tc(2, 6070.0, 4.0, 22.0), tc(3, 6130.0, 4.0, 12.0), tc(4, 6050.0, 20.0, 15.0)};
  std::vector<TrackClass> out = start;
  group_echoes(out, o, {1, 4});
  EXPECT_EQ(out[0].kind, Kind::Multipath);
  EXPECT_EQ(out[0].parent, 2);
  EXPECT_EQ(out[1].kind, Kind::Mover);
  EXPECT_EQ(out[1].parent, -1);
  EXPECT_EQ(out[2].kind, Kind::Multipath);
  EXPECT_EQ(out[2].parent, 2);
  EXPECT_EQ(out[3].kind, Kind::Multipath);
  EXPECT_EQ(out[3].parent, -1);
  // No bounce: the nearest established arrival is the object, as before.
  out = start;
  group_echoes(out, o);
  EXPECT_EQ(out[0].kind, Kind::Mover);
  EXPECT_EQ(out[1].parent, 1);
  EXPECT_EQ(out[2].parent, 1);
  EXPECT_EQ(out[3].kind, Kind::Mover);
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

// Transmits at samples 10, 110, 210 (and 310) (pings 0, 1, 2, 3), windows 30..69, 130..169, 230..269 (330..369); blocks
// of 16 from 0, each naming the latest transmit it reaches. `order` lists the blocks (by first sample) in arrival order.
std::vector<PingRecord> feed_stream(PingAssembler& assembler, const std::vector<std::int64_t>& order, std::uint32_t pings = 3) {
  std::vector<PingRecord> out;
  for (const std::int64_t first : order) {
    std::int64_t tx = -1;
    std::uint32_t ping = 0;
    for (std::uint32_t k = 0; k < pings; ++k) {
      if (first + 16 > 10 + 100 * static_cast<std::int64_t>(k)) {
        tx = 10 + 100 * static_cast<std::int64_t>(k);
        ping = k;
      }
    }
    const auto b = block(first, 16, 2, tx, ping);
    EXPECT_TRUE(assembler.feed(b.data(), b.size(), out));
  }
  return out;
}

std::vector<std::int64_t> stream_order(std::int64_t end = 320) {
  std::vector<std::int64_t> order;
  for (std::int64_t first = 0; first < end; first += 16) order.push_back(first);
  return order;
}

std::vector<std::int64_t> sequences(const std::vector<PingRecord>& out) {
  std::vector<std::int64_t> s;
  for (const PingRecord& r : out) s.push_back(r.metadata.sequence);
  return s;
}

TEST(PingAssembler, CountsAPingNoneOfWhoseBlocksArrived) {
  // Every block naming ping 2 lost (208..303): no header names it; the jump of the ping index from 1 to 3, two ping
  // intervals (100 samples, from pings 0 and 1) apart, counts it.
  PingAssembler assembler(stream_template());
  std::vector<std::int64_t> order;
  for (const std::int64_t first : stream_order(400)) {
    if (first < 208 || first >= 304) order.push_back(first);
  }
  const auto out = feed_stream(assembler, order, 4);
  EXPECT_EQ(sequences(out), (std::vector<std::int64_t>{0, 1, 3}));
  EXPECT_EQ(assembler.dropped_pings, 1);
  EXPECT_EQ(assembler.lost_samples, 6 * 16);
  EXPECT_EQ(assembler.late_blocks + assembler.repeated_blocks + assembler.restarts, 0);
}

TEST(PingAssembler, PlacesBlocksThatArriveOutOfOrderAndIgnoresRepeats) {
  {  // 144 and 160 (both in ping 1's window) swapped: ping 1 kept whole.
    PingAssembler assembler(stream_template());
    auto order = stream_order();
    std::swap(order[9], order[10]);
    const auto out = feed_stream(assembler, order);
    ASSERT_EQ(sequences(out), (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_FLOAT_EQ(out[1].samples.channel(0)[0], 130.0F * 10.0F - 300.0F);
    EXPECT_FLOAT_EQ(out[1].samples.channel(1)[39], 169.0F * 10.0F + 1.0F - 300.0F);
    EXPECT_EQ(assembler.late_blocks, 1);
    EXPECT_EQ(assembler.lost_samples + assembler.dropped_pings + assembler.repeated_blocks + assembler.too_late_blocks, 0);
  }
  {  // 144 twice: a repeat (same capture start, same transmit, samples already in), not a restart.
    PingAssembler assembler(stream_template());
    auto order = stream_order();
    order.insert(order.begin() + 10, 144);
    const auto out = feed_stream(assembler, order);
    EXPECT_EQ(sequences(out), (std::vector<std::int64_t>{0, 1, 2}));
    EXPECT_EQ(assembler.repeated_blocks, 1);
    EXPECT_EQ(assembler.last_block, PingAssembler::Block::kInOrder);
    EXPECT_EQ(assembler.lost_samples + assembler.dropped_pings + assembler.late_blocks + assembler.restarts, 0);
  }
  {  // 144 after 208: ping 1 was given up when the stream reached 2 blocks past its window (block 192); 144 is too late.
    PingAssembler assembler(stream_template());
    auto order = stream_order();
    order.erase(order.begin() + 9);
    order.insert(order.begin() + 13, 144);
    const auto out = feed_stream(assembler, order);
    EXPECT_EQ(sequences(out), (std::vector<std::int64_t>{0, 2}));
    EXPECT_EQ(assembler.dropped_pings, 1);
    EXPECT_EQ(assembler.lost_samples, 16);
    EXPECT_EQ(assembler.late_blocks, 1);
    EXPECT_EQ(assembler.too_late_blocks, 1);
    EXPECT_EQ(assembler.repeated_blocks, 0);
  }
}

TEST(PingAssembler, AnotherCaptureStartIsARestart) {
  PingAssembler assembler(stream_template());
  std::vector<std::int64_t> order;
  for (std::int64_t first = 0; first < 160; first += 16) order.push_back(first);    // into ping 1's window
  auto out = feed_stream(assembler, order);
  ASSERT_EQ(out.size(), 1U);
  // One block of another capture is held, not yet a restart; the next one, contiguous with it, confirms it.
  auto b = block(0, 16, 2, 10, 0U);
  const std::uint64_t start_ns = 2000;
  std::memcpy(b.data() + 36, &start_ns, 8);
  EXPECT_TRUE(assembler.feed(b.data(), b.size(), out));
  EXPECT_EQ(assembler.last_block, PingAssembler::Block::kHeld);
  EXPECT_EQ(assembler.restarts, 0);
  auto c = block(16, 16, 2, 10, 0U);
  std::memcpy(c.data() + 36, &start_ns, 8);
  EXPECT_TRUE(assembler.feed(c.data(), c.size(), out));
  EXPECT_EQ(assembler.restarts, 1);
  EXPECT_EQ(assembler.dropped_pings, 1);                 // ping 1, open at the restart
  EXPECT_EQ(assembler.last_block, PingAssembler::Block::kRestart);
}

TEST(PingAssembler, FinishCountsWhatIsStillAwaited) {
  // Block 144 (in ping 1's window) lost and the stream ends at 176: not yet given up, until finish().
  PingAssembler assembler(stream_template());
  std::vector<std::int64_t> order;
  for (std::int64_t first = 0; first < 176; first += 16) {
    if (first != 144) order.push_back(first);
  }
  auto out = feed_stream(assembler, order);
  EXPECT_EQ(sequences(out), (std::vector<std::int64_t>{0}));
  EXPECT_EQ(assembler.dropped_pings, 0);
  EXPECT_EQ(assembler.lost_samples, 0);
  std::vector<PingRecord> rest;
  assembler.finish(rest);
  EXPECT_TRUE(rest.empty());
  EXPECT_EQ(assembler.dropped_pings, 1);
  EXPECT_EQ(assembler.lost_samples, 16);
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

TEST(Navigation, MarksAPoseWithoutVelocityAndTakesAMissingOrNonFiniteComponentAsZero) {
  // sonar-8is: the backprojection follows the array over the round trip by the pose's velocity; a fix that does not
  // send vx, vy and vz as finite numbers is marked, and what it lacks is 0 (never NaN in the record).
  live::Navigation nav;
  PingRecord rec;
  EXPECT_FALSE(nav.velocity_given());
  EXPECT_TRUE(nav.push("{\"time_ns\": 0, \"x\": 0.0, \"vx\": 1.5, \"vy\": -0.5, \"vz\": 0.1}"));
  EXPECT_TRUE(nav.push("{\"time_ns\": 1000, \"x\": 1.0, \"vx\": 1.5, \"vy\": -0.5, \"vz\": 0.1}"));
  EXPECT_TRUE(nav.push("{\"time_ns\": 2000, \"x\": 2.0, \"vx\": 1.5, \"vy\": -0.5}"));                     // no vz
  EXPECT_TRUE(nav.push("{\"time_ns\": 3000, \"x\": 3.0, \"vx\": NaN, \"vy\": -0.5, \"vz\": 0.1}"));       // not finite
  EXPECT_TRUE(nav.push("{\"time_ns\": 4000, \"x\": 4.0, \"vx\": null, \"vy\": -0.5, \"vz\": 0.1}"));      // null
  const auto at = [&](std::int64_t t) {
    rec.transmit.time_ns = t;
    EXPECT_TRUE(nav.apply(rec));
    EXPECT_TRUE(std::isfinite(rec.geometry.rx_pose.velocity.x) && std::isfinite(rec.geometry.rx_pose.velocity.z));
    return nav.velocity_given();
  };
  EXPECT_TRUE(at(-5));                     // held from the first fix
  EXPECT_TRUE(at(500));                    // between two full fixes
  EXPECT_NEAR(rec.geometry.rx_pose.velocity.z, 0.1, 1e-12);
  EXPECT_FALSE(at(1500));                  // one of the two has no vz
  EXPECT_NEAR(rec.geometry.rx_pose.velocity.z, 0.05, 1e-12);
  EXPECT_FALSE(at(3000));
  EXPECT_NEAR(rec.geometry.rx_pose.velocity.x, 0.0, 1e-12);       // interpolated to the NaN fix's time: its 0
  EXPECT_FALSE(at(9000));                  // held from the null one
  EXPECT_TRUE(at(1000));                   // at a fix's own time: it and the one before
}

TEST(Navigation, CountsWhatTheProgramReportsAndSaysOnceWhenVelocityFirstGoesMissing) {
  // sonar-8is: apps/sonar_live and holoscan_ops UdpSourceOp count with NavCounts after each apply.
  live::Navigation nav;
  live::NavCounts counts;
  PingRecord rec;
  rec.transmit.time_ns = 500;
  EXPECT_FALSE(counts.apply(nav, rec));                                     // no fix yet
  EXPECT_EQ(counts.without_nav, 1);
  EXPECT_EQ(counts.without_velocity, 0);
  ASSERT_TRUE(nav.push("{\"time_ns\": 0, \"x\": 0.0}"));
  ASSERT_TRUE(nav.push("{\"time_ns\": 1000, \"x\": 1.0, \"vx\": 1.0, \"vy\": 0.0, \"vz\": 0.0}"));
  ASSERT_TRUE(nav.push("{\"time_ns\": 2000, \"x\": 2.0, \"vx\": 1.0, \"vy\": 0.0, \"vz\": 0.0}"));
  EXPECT_TRUE(counts.apply(nav, rec));                                      // first without velocity: warn
  EXPECT_FALSE(counts.apply(nav, rec));                                     // second: counted, no warning
  rec.transmit.time_ns = 1500;
  EXPECT_FALSE(counts.apply(nav, rec));                                     // with velocity: not counted
  EXPECT_EQ(counts.without_nav, 1);
  EXPECT_EQ(counts.without_velocity, 2);
}

TEST(Report, WritesNonFiniteNumbersAsNull) {
  // The Python engine writes the same (live.finite_or_null; tests/python/test_hardcoded.py).
  Report r;
  r.detections.push_back({1200.0, -4.5, std::numeric_limits<double>::quiet_NaN()});
  r.tracks.push_back({7, -1, "mover", 10.0, 20.0, -6.0, 0.1, 6.0, 1200.0, std::numeric_limits<double>::infinity(), 1.0});
  const std::string j = r.json();
  EXPECT_NE(j.find("\"snr_db\": null"), std::string::npos) << j;
  EXPECT_NE(j.find("\"mean_snr_db\": null"), std::string::npos) << j;
  EXPECT_EQ(j.find("nan"), std::string::npos) << j;
  EXPECT_EQ(j.find("inf"), std::string::npos) << j;
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

TEST(Classify, FitsTheMeasuredPositionsOfANoiseFree12kmCrossing) {
  // D-157 (tests/python/test_classify.py, popoto_12km_noise_free): exact detections of a ship crossing at 6 m/s 12 km
  // away, six pings 19.9 s apart from an ownship making 1.5 m/s, each at transmit + R/c, through the tracker the live
  // chain derives for that array (15 m, 1 deg). Classified at the last ping's transmit time the position is within 2 m
  // of the ship. The fit through the filtered history (before D-157) was 38.7 m behind along track.
  const double c = 1496.82;
  TrackerOptions o;
  o.range_sigma = 15.0;
  o.bearing_sigma = 1.0 * kDeg;
  o.confirm_hits = 3;
  o.max_misses = 2;
  o.initial_speed_sigma = 8.0;
  o.process_noise = 0.05;
  for (const int drop : {-1, 3}) {
    Tracker tracker(o);
    std::vector<const Track*> got;
    for (int i = 0; i < 6; ++i) {
      const double t = 19.9 * i, own = 1.5 * t;
      double tau = std::hypot(800.0 - 6.0 * t - own, 12000.0) / c;
      for (int k = 0; k < 4; ++k) tau = std::hypot(800.0 - 6.0 * (t + tau) - own, 12000.0) / c;
      const double px = 800.0 - 6.0 * (t + tau), py = 12000.0;    // world (x forward, y across)
      Detection d{std::hypot(px, py), std::atan2(py, px), 20.0, 1};
      d.time = t + std::hypot(px - own, py) / c;
      got = tracker.update(t, i == drop ? std::vector<Detection>{} : std::vector<Detection>{d});
    }
    ASSERT_EQ(got.size(), 1U);
    const Track& tr = *got[0];
    ASSERT_EQ(tr.history.size(), 6U);
    for (int i = 0; i < 6; ++i) EXPECT_EQ(std::isnan(tr.history[static_cast<std::size_t>(i)].mx), i == drop);
    ClassifyOptions co;
    co.time = 99.5;
    const auto out = classify_tracks(got, co);
    EXPECT_NEAR(out[0].x, 12000.0, 2.0);
    EXPECT_NEAR(out[0].y, 800.0 - 6.0 * 99.5, 2.0);
    EXPECT_NEAR(out[0].speed, 6.0, 0.01);
    EXPECT_LT(out[0].fit_rms, 1e-6);
    // The same rows without their measurements: the fit through the filtered positions lags.
    Track plain = tr;
    for (auto& p : plain.history) p.mx = p.my = std::numeric_limits<double>::quiet_NaN();
    const auto lag = classify_tracks({&plain}, co);
    EXPECT_GT(lag[0].y - (800.0 - 6.0 * 99.5), 30.0);
  }
}

TEST(Options, RangeEstimateIsFirstWithMultiscaleAndCentroidOtherwiseUnlessGiven) {
  // D-157, D-158: cfar_range_estimate, centroid or first; any other word refused (D-145). Not given: first with the
  // multiscale detector (the default), centroid with mean or peak; first given outside multiscale refused by the chain,
  // centroid always accepted; options_json prints the value used. The detector's own default stays centroid.
  ChainOptions a;
  PassiveOptions pa;
  EXPECT_EQ(DetectorOptions{}.range_estimate, RangeEstimate::Centroid);
  EXPECT_FALSE(a.range_estimate.has_value());
  EXPECT_EQ(chain_range_estimate(a), RangeEstimate::First);
  EXPECT_NE(options_json(a).find("\"cfar_range_estimate\": \"first\""), std::string::npos) << options_json(a);
  for (const char* statistic : {"mean", "peak"}) {
    ChainOptions m;
    std::istringstream t(std::string("cell_statistic=") + statistic + "\n");
    read_options(m, pa, t);
    EXPECT_EQ(chain_range_estimate(m), RangeEstimate::Centroid) << statistic;
    EXPECT_NE(options_json(m).find("\"cfar_range_estimate\": \"centroid\""), std::string::npos) << options_json(m);
    EXPECT_NO_THROW(LiveChain{m}) << statistic;
    // Given explicitly, in either order with the statistic: first refused, centroid accepted.
    for (const std::string text : {std::string("cfar_range_estimate=first\ncell_statistic=") + statistic + "\n",
                                   std::string("cell_statistic=") + statistic + "\ncfar_range_estimate=first\n"}) {
      ChainOptions f;
      std::istringstream ft(text);
      read_options(f, pa, ft);
      EXPECT_EQ(chain_range_estimate(f), RangeEstimate::First);
      EXPECT_THROW(LiveChain{f}, std::invalid_argument) << text;
    }
    ChainOptions c;
    std::istringstream ct(std::string("cfar_range_estimate=centroid\ncell_statistic=") + statistic + "\n");
    read_options(c, pa, ct);
    EXPECT_NO_THROW(LiveChain{c}) << statistic;
  }
  std::istringstream in("cfar_range_estimate=centroid\n");
  read_options(a, pa, in);
  ASSERT_TRUE(a.range_estimate.has_value());
  EXPECT_EQ(chain_range_estimate(a), RangeEstimate::Centroid);
  EXPECT_NE(options_json(a).find("\"cfar_range_estimate\": \"centroid\""), std::string::npos) << options_json(a);
  for (const char* bad : {"cfar_range_estimate=leading\n", "cfar_range_estimate=First\n", "cfar_range_estimate=\n"}) {
    ChainOptions b;
    std::istringstream t(bad);
    EXPECT_THROW(read_options(b, pa, t), std::invalid_argument) << bad;
  }
  a.range_estimate = RangeEstimate::First;
  a.cell_multiscale = false;
  EXPECT_THROW(LiveChain{a}, std::invalid_argument);
  a.cell_multiscale = true;
  EXPECT_NO_THROW(LiveChain{a});
  // The chain's setting is ChainOptions::range_estimate: one set on the detector's options is refused, not ignored.
  ChainOptions d;
  d.detector.range_estimate = RangeEstimate::First;
  EXPECT_THROW(LiveChain{d}, std::invalid_argument);
}

}  // namespace

TEST(Options, RangeDopplerIsOffUnlessGivenAndRefusesOtherWords) {
  // sonar-fyj: range_doppler off (default) | ownship | full; options_json prints it; any other word refused.
  ChainOptions a;
  PassiveOptions pa;
  EXPECT_EQ(a.range_doppler, RangeDoppler::Off);
  EXPECT_NE(options_json(a).find("\"range_doppler\": \"off\""), std::string::npos) << options_json(a);
  for (const char* word : {"ownship", "full"}) {
    ChainOptions m;
    std::istringstream t(std::string("range_doppler=") + word + "\n");
    read_options(m, pa, t);
    EXPECT_EQ(range_doppler_name(m.range_doppler), std::string(word));
  }
  ChainOptions bad;
  std::istringstream t("range_doppler=on\n");
  EXPECT_THROW(read_options(bad, pa, t), std::invalid_argument);
}

TEST(RangeDoppler, FactorIsTheCentreFrequencyTimesThePulseOverTheSignedSweep) {
  // f0 T / (f_stop - f_start) for an LFM (fls_170k: 0.17 s), negative for a down-chirp, 0 for anything else.
  Waveform w;
  w.type = WaveformType::Lfm;
  w.f_start = 160e3;
  w.f_stop = 180e3;
  w.duration = 20e-3;
  EXPECT_NEAR(lfm_range_doppler_factor(w), 0.17, 1e-12);
  std::swap(w.f_start, w.f_stop);
  EXPECT_NEAR(lfm_range_doppler_factor(w), -0.17, 1e-12);
  w.type = WaveformType::Cw;
  EXPECT_EQ(lfm_range_doppler_factor(w), 0.0);
}

TEST(RangeDoppler, TargetPartMovesThePositionAlongTheLineOfSightByItsClosingSpeed) {
  // A track 100 m ahead (forward, y) closing at 8 m/s moves 8 x factor farther; a crossing one does not move; one at the
  // origin is left; the velocity is kept.
  std::vector<TrackClass> c(3);
  c[0].x = 0.0, c[0].y = 100.0, c[0].vx = 0.0, c[0].vy = -8.0;
  c[1].x = 0.0, c[1].y = 100.0, c[1].vx = 5.0, c[1].vy = 0.0;
  c[2].x = 0.0, c[2].y = 0.0, c[2].vx = 1.0, c[2].vy = 1.0;
  shift_for_target_doppler(c, 0.17, 0.0, 0.0);
  EXPECT_NEAR(c[0].y, 100.0 + 8.0 * 0.17, 1e-12);
  EXPECT_NEAR(c[0].x, 0.0, 1e-12);
  EXPECT_EQ(c[0].vy, -8.0);
  EXPECT_NEAR(c[1].y, 100.0, 1e-12);
  EXPECT_EQ(c[2].y, 0.0);
}

// --- an elastic bottom (rock) in the live chain's depth and multipath models -------------------------------------------

TEST(Rock, LiveCoefficientIsTheSimulators) {
  // The `rock` row on 1500 m/s water (configs/sediments.yaml: speed ratio 3820 / 1528, density ratio 2557.5 / 1023, delta
  // 0.01374; shear 1900 / 1528 of the water, shear delta 0.085, unconfirmed): live::rayleigh_reflection with the shear is
  // sim::rayleigh_reflection bit for bit (one function, sonar/core/reflection.hpp), and its per-bounce losses are A8's
  // table (tests/python/test_elastic_bottom.py: 1.56 / 2.90 / 5.45 / 14.57 / 8.78 / 6.25 / 4.54 / 2.80 dB).
  const double c = 1500.0, per_wl = 40.0 * kPi / std::log(10.0);
  const double speed = 3820.0 / 1528.0 * c, density = 2557.5 / 1023.0, att = 0.01374 * per_wl;
  const double shear = 1900.0 / 1528.0 * c, shear_att = 0.085 * per_wl;
  sim::FluidBottom rock{speed, density, att, 1.0, shear, shear_att};
  sim::FluidBottom fluid{speed, density, att, 1.0};
  for (double g = 0.001; g <= 0.5 * kPi; g += 0.003) {
    EXPECT_EQ(rayleigh_reflection(g, c, speed, density, att, shear, shear_att), sim::rayleigh_reflection(g, c, rock)) << g;
    // zero shear: the live chain's own fluid expression, unchanged (the five-argument call); it agrees with the
    // simulator's fluid one to rounding (they divide differently)
    EXPECT_EQ(rayleigh_reflection(g, c, speed, density, att, 0.0, shear_att), rayleigh_reflection(g, c, speed, density, att)) << g;
    EXPECT_LT(std::abs(rayleigh_reflection(g, c, speed, density, att) - sim::rayleigh_reflection(g, c, fluid)), 1e-13) << g;
  }
  const double angles[] = {5.0, 10.0, 20.0, 30.0, 40.0, 45.0, 60.0, 90.0};
  const double table[] = {1.56, 2.90, 5.45, 14.57, 8.78, 6.25, 4.54, 2.80};
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(-20.0 * std::log10(std::abs(rayleigh_reflection(angles[i] * kDeg, c, speed, density, att, shear, shear_att))), table[i], 0.005) << angles[i];
  }
}

TEST(Rock, MultipathPatternAndCoverageTakeTheShear) {
  // The pattern's bottom-reflected one-leg arrival (out by the bottom image, back direct; the surface's coefficient 0
  // removes the surface images) over rock is the elastic coefficient at its grazing angle; the options carry the shear
  // by name and refuse a negative one; the coverage options the same.
  const double c = 1500.0, per_wl = 40.0 * kPi / std::log(10.0);
  MultipathOptions o;
  for (const auto& [k, v] : std::vector<std::pair<std::string, std::string>>{
           {"bottom", "1"}, {"bottom_speed", std::to_string(3820.0 / 1528.0 * c)}, {"bottom_density", std::to_string(2557.5 / 1023.0)},
           {"bottom_attenuation", std::to_string(0.01374 * per_wl)}, {"bottom_shear_speed", std::to_string(1900.0 / 1528.0 * c)},
           {"bottom_shear_attenuation", std::to_string(0.085 * per_wl)}, {"bottom_order", "1"}, {"surface_coefficient", "0"}}) {
    ASSERT_TRUE(set_multipath_option(o, "multipath_depth_" + k, v)) << k;
  }
  EXPECT_GT(o.bottom_shear_speed, 1800.0);
  EXPECT_THROW(set_multipath_option(o, "multipath_depth_bottom_shear_speed", "-1"), std::invalid_argument);
  const double own = 30.0, zt = 50.0, depth = 100.0, lift = 2.0 * depth - zt - own;
  for (const double deg : {10.0, 30.0, 45.0}) {
    MultipathGeometry g;
    g.range = lift / std::tan(deg * kDeg);
    g.array_depth = g.projector_depth = own;
    g.sound_speed = c;
    g.f_centre = 8000.0;
    g.bandwidth = 2000.0;
    g.water_depth = depth;
    std::vector<double> delay;
    std::vector<std::complex<double>> amp;
    const double r_h = g.range;
    multipath_pattern(zt, 2.0 * std::hypot(r_h, zt - own) / c, g, o, delay, amp);
    ASSERT_EQ(amp.size(), 9u);
    const double spread = std::hypot(r_h, zt - own) / std::hypot(r_h, lift);
    const auto want = rayleigh_reflection(std::asin(lift / std::hypot(r_h, lift)), c, o.bottom_speed, o.bottom_density, o.bottom_attenuation,
                                          o.bottom_shear_speed, o.bottom_shear_attenuation);
    EXPECT_LT(std::abs(amp[6] / spread - want), 1e-9) << deg;
    const auto fluid = rayleigh_reflection(std::asin(lift / std::hypot(r_h, lift)), c, o.bottom_speed, o.bottom_density, o.bottom_attenuation);
    EXPECT_GT(std::abs(fluid) - std::abs(want), 0.1) << deg;     // the elastic rock returns less than the fluid one
  }
  CoverageOptions v;
  ASSERT_TRUE(set_coverage_option(v, "coverage_bottom_shear_speed", "1865.5"));
  ASSERT_TRUE(set_coverage_option(v, "coverage_bottom_shear_attenuation", "4.6"));
  EXPECT_EQ(v.bottom_shear_speed, 1865.5);
  EXPECT_EQ(v.bottom_shear_attenuation, 4.6);
  EXPECT_THROW(set_coverage_option(v, "coverage_bottom_shear_attenuation", "nan"), std::invalid_argument);
}

TEST(Rock, EigenraysTakeTheShear) {
  // One bottom bounce in isovelocity water: the eigenray over elastic rock is the fluid one times V_elastic / V_fluid at
  // its grazing angle (the launch angle); a shear speed of 0 is the fluid bit for bit.
  const double c = 1500.0, per_wl = 40.0 * kPi / std::log(10.0);
  EigenrayOptions o;
  o.n_fan = 801;
  o.max_bounces = 2;
  o.bottom = {true, 3820.0 / 1528.0 * c, 2557.5 / 1023.0, 0.01374 * per_wl, 1900.0 / 1528.0 * c, 0.085 * per_wl};
  EigenrayOptions f = o;
  f.bottom.shear_speed = 0.0;
  const RayProfile prof({0.0, 100.0}, {c, c}, 100.0);
  const auto rock = eigenrays(prof, 30.0, 50.0, 400.0, o), fluid = eigenrays(prof, 30.0, 50.0, 400.0, f);
  ASSERT_EQ(rock.size(), fluid.size());
  int checked = 0;
  for (std::size_t i = 0; i < rock.size(); ++i) {
    if (rock[i].n_bottom != 1 || rock[i].n_surface != 0) continue;
    const double g = std::abs(rock[i].launch);
    const auto ratio = rayleigh_reflection(g, c, o.bottom.speed, o.bottom.density, o.bottom.attenuation, o.bottom.shear_speed, o.bottom.shear_attenuation) /
                       rayleigh_reflection(g, c, o.bottom.speed, o.bottom.density, o.bottom.attenuation);
    EXPECT_LT(std::abs(rock[i].amplitude / fluid[i].amplitude - ratio), 1e-9);
    ++checked;
  }
  EXPECT_EQ(checked, 1);
}

// --- the coverage picture's reverberation term (coverage.hpp; sonar-w2w3) ---

// A Lambert bottom (mu, linear) on `n` grazing angles from 0 to 90 degrees, [g_in][g_out] as the options take it.
CoverageOptions lambert_reverb(double mu, std::size_t n = 46) {
  CoverageOptions o;
  o.reverb = 1;
  for (std::size_t i = 0; i < n; ++i) o.reverb_grazing.push_back(0.5 * kPi * static_cast<double>(i) / static_cast<double>(n - 1));
  for (double a : o.reverb_grazing) {
    for (double b : o.reverb_grazing) o.reverb_bottom_strength.push_back(mu * std::sin(a) * std::sin(b));
  }
  o.surface_coefficient = 0.0;           // no surface-reflected paths: one path each way, as (G.11) has it
  o.max_bounces = 1;
  o.reverb_max_bounces = 1;
  o.ray_fan = 1440;
  return o;
}

MultipathProfile isovelocity(double c) {
  MultipathProfile w;
  w.depth = {0.0, 1.0};
  w.speed = {c, c};
  return w;
}

TEST(CoverageReverb, SineIntegralAndTheCompressedPulsesAverage) {
  // Si against tabulated values (Abramowitz and Stegun Table 5.1 / the series to 16 digits); the bin average of
  // sinc^2 integrates to 1 / B over delay whatever the bin, and is the sinc^2 itself for a narrow bin.
  const std::pair<double, double> si[] = {{0.5, 0.493107418043067}, {1.0, 0.946083070367183}, {5.0, 1.549931244944674},
                                          {10.0, 1.658347594218874}, {20.0, 1.548241701043440}};
  for (const auto& [x, v] : si) EXPECT_NEAR(sine_integral(x), v, 1e-6) << x;
  EXPECT_NEAR(sine_integral(-1.0), -sine_integral(1.0), 1e-15);
  const double band = 4000.0;
  for (double width : {1e-6, 1e-4, 1e-3, 1e-2}) {
    double sum = 0.0;
    const double step = width;
    for (double x = -0.5; x < 0.5; x += step) sum += compressed_power_average(x, width, band) * step;
    EXPECT_NEAR(sum * band, 1.0, 0.01) << width;
  }
  const double s = std::sin(kPi * band * 1e-4) / (kPi * band * 1e-4);
  EXPECT_NEAR(compressed_power_average(1e-4, 1e-9, band), s * s, 1e-12);
}

TEST(CoverageReverb, LambertBottomIsJacksonAndRichardsonG13) {
  // One direct path each way to a flat Lambert bottom: RL - SL = 10 log10(mu sin^2 g Psi c / (2 B) / r^3) - 2 alpha r,
  // (G.11)-(G.13) with the compressed pulse 1 / B in place of tau; held within 0.2 dB from 1 to 4 km.
  const double c = 1500.0, depth = 100.0, own = 30.0, mu = std::pow(10.0, -2.7), psi = 0.1, alpha = 0.5;   // dB/km
  CoverageOptions o = lambert_reverb(mu);
  o.absorption_db_per_km = alpha;
  CoverageSonar sonar;
  sonar.f_centre = 8000.0;
  sonar.bandwidth = 4000.0;
  sonar.pulse = 1.0;
  std::vector<double> ranges = {1000.0, 1500.0, 2200.0, 3000.0, 4000.0}, delays;
  for (double r : ranges) delays.push_back(2.0 * r / c);
  const std::vector<double> got = coverage_reverberation_db(isovelocity(c), depth, own, own, delays, o, sonar, psi);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const double r = ranges[i], g = std::asin((depth - own) / r);
    const double want = 10.0 * std::log10(mu * std::sin(g) * std::sin(g) * psi * c / (2.0 * sonar.bandwidth) / (r * r * r)) - 2.0 * alpha * r / 1000.0;
    EXPECT_NEAR(got[i], want, 0.2) << r;
  }
}

TEST(CoverageReverb, LimitsOfTheTerm) {
  const double c = 1500.0, depth = 100.0, own = 30.0;
  CoverageOptions o = lambert_reverb(std::pow(10.0, -2.7));
  o.source_level_db = 200.0;
  o.noise_spectrum_db = 45.0;
  o.array_gain_db = 15.0;
  o.absorption_db_per_km = 0.5;
  CoverageSonar sonar;
  sonar.f_centre = 8000.0;
  sonar.bandwidth = 4000.0;
  sonar.pulse = 1.0;
  sonar.beam_integral = 0.1;
  const std::vector<double> ranges = {1000.0, 2000.0, 3000.0}, depths = {40.0};
  // Zero scattering strength: no reverberation at all, and the SNR is the noise-only figure bit for bit.
  CoverageOptions zero = o;
  std::fill(zero.reverb_bottom_strength.begin(), zero.reverb_bottom_strength.end(), 0.0);
  const CoverageField fz = coverage_field(isovelocity(c), depth, own, own, ranges, depths, zero, sonar);
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_TRUE(std::isinf(fz.reverb_db[i]) && fz.reverb_db[i] < 0.0);
    EXPECT_EQ(coverage_snr_db(zero, sonar, fz.slant[i], 0.0, fz.factor_db[i], fz.reverb_db[i]), coverage_snr_db(zero, sonar, fz.slant[i], 0.0, fz.factor_db[i]));
  }
  // The term off: the noise-only figure, whatever the tables hold.
  CoverageOptions off = o;
  off.reverb = 0;
  const CoverageField f0 = coverage_field(isovelocity(c), depth, own, own, ranges, depths, off, sonar);
  for (std::size_t i = 0; i < ranges.size(); ++i) EXPECT_EQ(coverage_snr_db(off, sonar, f0.slant[i], 0.0, f0.factor_db[i], -20.0), coverage_snr_db(off, sonar, f0.slant[i], 0.0, f0.factor_db[i]));
  // Reverberation-limited: with the noise 80 dB down, the SNR does not move with the source level.
  const CoverageField f = coverage_field(isovelocity(c), depth, own, own, ranges, depths, o, sonar);
  CoverageOptions quiet = o, loud = o;
  quiet.noise_spectrum_db = loud.noise_spectrum_db = -60.0;
  loud.source_level_db = 220.0;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    EXPECT_NEAR(coverage_snr_db(quiet, sonar, f.slant[i], 0.0, f.factor_db[i], f.reverb_db[i]), coverage_snr_db(loud, sonar, f.slant[i], 0.0, f.factor_db[i], f.reverb_db[i]), 1e-6);
    EXPECT_LT(coverage_snr_db(o, sonar, f.slant[i], 0.0, f.factor_db[i], f.reverb_db[i]), coverage_snr_db(o, sonar, f.slant[i], 0.0, f.factor_db[i]));
  }
  // A CW pulse twice as long (its band 1 / T): 3 dB more reverberation, far from the bottom's first return.
  CoverageSonar cw = sonar, cw2 = sonar;
  cw.bandwidth = 1.0 / 0.01;
  cw2.bandwidth = 1.0 / 0.02;
  const std::vector<double> delays = {2.0 * 2000.0 / c, 2.0 * 3000.0 / c};
  const std::vector<double> a = coverage_reverberation_db(isovelocity(c), depth, own, own, delays, o, cw, 0.1);
  const std::vector<double> b = coverage_reverberation_db(isovelocity(c), depth, own, own, delays, o, cw2, 0.1);
  for (std::size_t i = 0; i < delays.size(); ++i) EXPECT_NEAR(b[i] - a[i], 10.0 * std::log10(2.0), 0.1);
  // The options by name, and an out-of-range switch refused.
  CoverageOptions v;
  ASSERT_TRUE(set_coverage_option(v, "coverage_reverb", "1"));
  ASSERT_TRUE(set_coverage_option(v, "coverage_reverb_grazing", "0,0.5,1.5707963"));
  ASSERT_TRUE(set_coverage_option(v, "coverage_reverb_beam_integral", "0.2"));
  EXPECT_EQ(v.reverb, 1);
  EXPECT_EQ(v.reverb_grazing.size(), 3u);
  EXPECT_THROW(set_coverage_option(v, "coverage_reverb", "3"), std::invalid_argument);
}

TEST(CoverageReverb, CwTriangleConvergedBouncesNoteAndCandidates) {
  // C(e): a CW pulse's matched-filter power response (1 - |u| / T)^2: 1 at 0, 0 beyond T, integral 2 T / 3 whatever the bin.
  const double T = 0.01;
  EXPECT_DOUBLE_EQ(compressed_power_average(0.0, 0.0, 1.0 / T, T), 1.0);
  EXPECT_DOUBLE_EQ(compressed_power_average(1.01 * T, 0.0, 1.0 / T, T), 0.0);
  for (double width : {1e-5, 1e-4, 1e-3}) {
    double sum = 0.0;
    for (double x = -2.0 * T; x < 2.0 * T; x += width) sum += compressed_power_average(x + 0.5 * width, width, 1.0 / T, T) * width;
    EXPECT_NEAR(sum, 2.0 * T / 3.0, 1e-3 * T) << width;
  }
  // C(c): the converged count is a 160-reflection run's within 0.1 dB in 30 m of water at 0.8 to 12.7 km.
  const double c = 1500.0;
  CoverageOptions o = lambert_reverb(std::pow(10.0, -2.7));
  o.absorption_db_per_km = 0.32;
  CoverageSonar sonar;
  sonar.f_centre = 5000.0;
  sonar.bandwidth = 2000.0;
  sonar.pulse = 1.0;
  std::vector<double> delays;
  for (double r = 800.0; r <= 12700.0; r += 1700.0) delays.push_back(2.0 * r / c);
  CoverageOptions fixed = o;
  fixed.reverb_max_bounces = 160;
  const std::vector<double> got = coverage_reverberation_db(isovelocity(c), 30.0, 15.0, 15.0, delays, o, sonar, 0.05);
  const std::vector<double> ref = coverage_reverberation_db(isovelocity(c), 30.0, 15.0, 15.0, delays, fixed, sonar, 0.05);
  for (std::size_t i = 0; i < delays.size(); ++i) EXPECT_NEAR(got[i], ref[i], 0.1) << delays[i];
  // The note, the steering and the echo's rule by name; the echo's rule is the old one unless asked for.
  CoverageOptions v;
  EXPECT_FALSE(v.echo_coherent);
  ASSERT_TRUE(set_coverage_option(v, "coverage_reverb_note", "not included: no wind"));
  ASSERT_TRUE(set_coverage_option(v, "coverage_echo_coherent", "1"));
  ASSERT_TRUE(set_coverage_option(v, "coverage_reverb_steer", "1.0471975512"));
  EXPECT_EQ(v.reverb_note, "not included: no wind");
  EXPECT_TRUE(v.echo_coherent);
  EXPECT_NEAR(v.reverb_steer, kPi / 3.0, 1e-9);
}

TEST(CoverageReverb, BeamIntegralOfALineIsTwiceLambdaOverItsLength) {
  // An unweighted line of N hydrophones at half a wavelength, broadside: Int |AF|^2 dphi / N^2 = 2 pi / N exactly
  // (the cross terms integrate to J0(k d) = J0(m pi), small; both sides of the line count). One hydrophone: 2 pi.
  const double c = 1500.0, f = 8000.0, d = 0.5 * c / f;
  std::vector<double> x, y;
  for (int i = 0; i < 32; ++i) {
    x.push_back(d * i);
    y.push_back(0.0);
  }
  const double psi = coverage_beam_integral(x, y, f, c, 0.5 * kPi);
  double cross = 0.0;
  for (int m = 1; m < 32; ++m) cross += 2.0 * (32 - m) * std::cyl_bessel_j(0.0, m * kPi);
  EXPECT_NEAR(psi, 2.0 * kPi * (32.0 + cross) / (32.0 * 32.0), 1e-6);
  EXPECT_NEAR(coverage_beam_integral({0.0}, {0.0}, f, c, 0.0), 2.0 * kPi, 1e-9);
}
