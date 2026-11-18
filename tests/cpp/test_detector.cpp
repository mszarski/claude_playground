// The detector at the range resolution (libs/live detector.hpp) and the statistics it is built on
// (libs/detect statistic.hpp), in C++ on maps built here:
//   - the tail of a weighted sum of exponentials against the Erlang and hypoexponential closed forms,
//     its inverse, and the cell-averaging factor as its one-sample special case;
//   - the weights of a mean of correlated samples in the two limits where they are known;
//   - the plan: more rows or a longer integration lower the threshold, a smaller budget raises it;
//   - a point echo is one detection at its place with its sample's ratio, not its cell's mean;
//   - a strong echo's range and bearing sidelobes are not detections; a second echo above them is;
//   - two echoes further apart than the merge distance are two, nearer are one;
//   - a background that falls with range is followed at the ends of the map, where the plain mean
//     of a one-sided window is low.
// tests/python/test_detector.py holds the same against simulated pings, both engines and Monte Carlo.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sonar/detect/cfar.hpp"
#include "sonar/detect/statistic.hpp"
#include "sonar/live/derive.hpp"
#include "sonar/live/detector.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;

TEST(Statistic, TailOfOneSampleIsErlangInTheRows) {
  for (double x : {2.0, 9.0, 18.0}) {
    EXPECT_NEAR(detect::weighted_exponential_tail({1.0}, 1, x), std::exp(-x), 1e-15);
    const double y = 3.0 * x;
    EXPECT_NEAR(detect::weighted_exponential_tail({1.0}, 3, x) / (std::exp(-y) * (1.0 + y + 0.5 * y * y)), 1.0, 1e-12);
  }
}

TEST(Statistic, SaddlepointTailAgainstTheHypoexponentialClosedForm) {
  // X = sum w_i E_i with distinct weights: P(X > x) = sum_i exp(-x / w_i) prod_{j != i} w_i / (w_i - w_j).
  const std::vector<double> w{0.5, 0.3, 0.2};
  for (double x : {2.0, 4.0, 6.0, 9.0}) {
    double exact = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
      double c = 1.0;
      for (std::size_t j = 0; j < w.size(); ++j) {
        if (j != i) c *= w[i] / (w[i] - w[j]);
      }
      exact += c * std::exp(-x / w[i]);
    }
    EXPECT_NEAR(detect::weighted_exponential_tail(w, 1, x) / exact, 1.0, 0.05) << x;   // 1e-1 down to 4e-8
  }
  // Equal weights are an Erlang: the saddlepoint formula against the exact one.
  const std::vector<double> equal(6, 1.0 / 6.0);
  for (double x : {2.0, 3.0, 4.5}) {
    EXPECT_NEAR(detect::weighted_exponential_tail(equal, 1, x) / detect::weighted_exponential_tail({1.0}, 6, x), 1.0, 0.03) << x;
  }
}

TEST(Statistic, ThresholdIsTheInverseAndTheCellAveragingFactorIsItsOneSampleCase) {
  const std::vector<double> w{0.4, 0.3, 0.2, 0.1};
  for (double pfa : {1e-3, 1e-6, 1e-9}) {
    for (int rows : {1, 3}) {
      const double t = detect::weighted_exponential_threshold(w, rows, pfa);
      EXPECT_NEAR(detect::weighted_exponential_tail(w, rows, t) / pfa, 1.0, 1e-6);
    }
    EXPECT_NEAR(detect::weighted_exponential_threshold({1.0}, 1, pfa), -std::log(pfa), 1e-9);
    for (int n : {16, 300, 2400}) {
      EXPECT_NEAR(detect::weighted_exponential_threshold({1.0}, 1, pfa) * detect::training_correction(pfa, n),
                  detect::cfar_threshold_factor(pfa, n), 1e-9 * detect::cfar_threshold_factor(pfa, n));
    }
  }
  // More looks, in the mean or in the rows: a lower threshold over the mean.
  EXPECT_LT(detect::weighted_exponential_threshold({1.0}, 3, 1e-6), detect::weighted_exponential_threshold({1.0}, 1, 1e-6));
  EXPECT_LT(detect::weighted_exponential_threshold(std::vector<double>(8, 0.125), 1, 1e-6), detect::weighted_exponential_threshold(w, 1, 1e-6));
}

TEST(Statistic, WeightsOfAMeanInTheLimitsWhereTheyAreKnown) {
  // Uncorrelated samples: n equal weights. Fully correlated: one. A flat band sampled at its resolution: uncorrelated.
  const auto independent = detect::mean_power_weights({1.0, 0.0, 0.0, 0.0}, 4);
  ASSERT_EQ(independent.size(), 4U);
  for (double v : independent) EXPECT_NEAR(v, 0.25, 1e-12);
  const auto same = detect::mean_power_weights({1.0, 1.0, 1.0, 1.0}, 4);
  EXPECT_NEAR(same[0], 1.0, 1e-9);
  EXPECT_NEAR(detect::equivalent_looks(same), 1.0, 1e-9);
  // The correlation is the replica's own autocorrelation (white noise through the matched filter). An untapered sweep
  // of 200 cycles of band has a band flat to a per cent: sampled at its resolution the samples are uncorrelated.
  PingRecord r;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 4000.0;
  r.transmit.waveform.f_stop = 6000.0;
  r.transmit.waveform.duration = 0.1;
  const auto at_resolution = sample_correlation(r.transmit.waveform, 1500.0, 0.375, 8);
  EXPECT_NEAR(detect::equivalent_looks(detect::mean_power_weights(at_resolution, 8)), 8.0, 0.05);
  // The chain's map: 8 samples a quarter of a metre apart are 2 m, 5.33 resolutions: between 5.33 and 8 looks.
  const double looks = detect::equivalent_looks(detect::mean_power_weights(sample_correlation(r.transmit.waveform, 1500.0, 0.25, 8), 8));
  EXPECT_GT(looks, 5.33);
  EXPECT_LT(looks, 6.5);
  // Past the exact size the variance-equivalent number is used: the same looks within a few per cent.
  const auto corr = sample_correlation(r.transmit.waveform, 1500.0, 0.25, 64);
  EXPECT_NEAR(detect::equivalent_looks(detect::mean_power_weights(corr, 64, 16)) / detect::equivalent_looks(detect::mean_power_weights(corr, 64)), 1.0, 0.02);
  // A taper narrows the band: fewer looks in the same cell (a Tukey taper of 10 %: 5.5 against 5.8).
  Waveform tapered = r.transmit.waveform;
  tapered.window_alpha = 0.1;
  const double fewer = detect::equivalent_looks(detect::mean_power_weights(sample_correlation(tapered, 1500.0, 0.25, 8), 8));
  EXPECT_LT(fewer, looks - 0.15);
  EXPECT_GT(fewer, looks - 0.6);
  // A pulse that does not sweep: the triangle of its own length (the replica is sampled 260 times over the pulse).
  r.transmit.waveform.type = WaveformType::Cw;
  r.transmit.waveform.f_stop = 4000.0;
  const auto tone = sample_correlation(r.transmit.waveform, 1500.0, 7.5, 12);
  EXPECT_NEAR(tone[5], 0.5, 0.005);
  EXPECT_NEAR(tone[10], 0.0, 0.005);
}

// One row of 32 hydrophones at 0.15 m, 4-6 kHz: the long-range array's.
PingRecord row_record() {
  PingRecord r;
  r.samples.resize(32, 16, SampleKind::RealPassband);
  r.samples.sample_rate = 32000.0;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 4000.0;
  r.transmit.waveform.f_stop = 6000.0;
  r.transmit.waveform.duration = 0.1;
  r.environment.sound_speed = 1500.0;
  for (int e = 0; e < 32; ++e) {
    Element el;
    el.lever_arm = Vec3d{(e - 15.5) * 0.15, 0.0, 0.0};
    el.boresight = Vec3d{0.0, 1.0, 0.0};
    r.geometry.rx.push_back(el);
  }
  r.geometry.tx_pose = r.geometry.rx_pose;
  set_default_channel_tables(r);
  return r;
}

struct Map {
  std::int64_t n_b = 161, n_cells = 700, k = 8;
  std::vector<double> bearings, fine_ranges, cell_ranges;
  std::vector<float> fine, mean;
  DetectorPlan plan;
  std::int64_t guard = 50, train = 150;

  explicit Map(unsigned seed, std::int64_t rows = 1, const DetectorOptions& options = {}, double merge_m = 15.0) {
    for (std::int64_t b = 0; b < n_b; ++b) bearings.push_back((-40.0 + 0.5 * static_cast<double>(b)) * kPi / 180.0);
    for (std::int64_t i = 0; i < n_cells * k; ++i) fine_ranges.push_back(1000.0 + 0.25 * static_cast<double>(i));
    for (std::int64_t j = 0; j < n_cells; ++j) cell_ranges.push_back(1000.0 + 2.0 * static_cast<double>(j) + 0.875);
    std::mt19937 gen(seed);
    std::exponential_distribution<float> noise(1.0F);
    fine.resize(static_cast<std::size_t>(n_b * n_cells * k));
    for (float& v : fine) {
      v = 0.0F;
      for (std::int64_t q = 0; q < rows; ++q) v += noise(gen);          // rows of unit noise added in power
    }
    const PingRecord rec = row_record();
    plan = plan_detector(rec, sonar_facts(rec), options, n_cells, k, 0.25, bearings, rows, guard, train, true, merge_m);
  }
  void add(std::int64_t b, std::int64_t sample, float power) { fine[static_cast<std::size_t>(b * n_cells * k + sample)] += power; }
  std::vector<Detection> detect() {
    mean.assign(static_cast<std::size_t>(n_b * n_cells), 0.0F);
    for (std::int64_t i = 0; i < n_b * n_cells; ++i) {
      double s = 0.0;
      for (std::int64_t q = 0; q < k; ++q) s += static_cast<double>(fine[static_cast<std::size_t>(i * k + q)]);
      mean[static_cast<std::size_t>(i)] = static_cast<float>(s / static_cast<double>(k));
    }
    std::vector<float> background;
    return multiscale_detect(fine.data(), n_cells * k, 0, mean.data(), n_b, n_cells, fine_ranges, cell_ranges, bearings, guard, train, plan, background);
  }
};

TEST(Detector, PlanFollowsTheBudgetTheRowsAndTheLength) {
  const Map one(1), three(1, 3);
  DetectorOptions strict;
  strict.false_alarms_per_ping = 1e-5;
  const Map tight(1, 1, strict);
  ASSERT_EQ(one.plan.scales.size(), 2U);              // the sample and the cell
  EXPECT_EQ(one.plan.scales[0].cells, 0);
  EXPECT_EQ(one.plan.scales[1].cells, 1);
  // One sample of one row: the threshold is -ln(pfa), and the tests times the pfa of the two lengths are the budget.
  EXPECT_NEAR(one.plan.scales[0].factor, -std::log(one.plan.scales[0].pfa), 1e-9);
  EXPECT_NEAR(one.plan.scales[0].tests * one.plan.scales[0].pfa + one.plan.scales[1].tests * one.plan.scales[1].pfa, 1e-3, 1e-9);
  EXPECT_NEAR(10.0 * std::log10(one.plan.scales[0].factor), 12.9, 0.3);       // 13 dB for a sample, 7 dB for the mean of a cell
  EXPECT_NEAR(10.0 * std::log10(one.plan.scales[1].factor), 7.2, 0.3);
  EXPECT_LT(three.plan.scales[0].factor, 0.5 * one.plan.scales[0].factor);    // three rows: 9.4 dB
  EXPECT_GT(tight.plan.scales[0].factor, one.plan.scales[0].factor + 4.0);    // a hundredth of the budget: ln(100) more
  EXPECT_EQ(one.plan.merge_cells, 7);                                         // 15 m in 2 m cells
  // The envelope: 1 in the beam's core and the next cell, the compressed pulse's sidelobes beyond, the window's in bearing.
  EXPECT_EQ(one.plan.range_envelope[1], 1.0);
  EXPECT_NEAR(10.0 * std::log10(one.plan.range_envelope[2]), -24.5, 0.2);     // 1 / (pi 5.33)^2
  EXPECT_EQ(one.plan.bearing_envelope[0], 1.0);
  EXPECT_LT(one.plan.bearing_envelope[static_cast<std::size_t>(0.3 / one.plan.u_step)], 2e-3);
  DetectorOptions longer;
  longer.integrate_cells = {1, 4, 16, 400};                                   // 400 cells: longer than the guard, dropped
  const Map with(1, 1, longer);
  ASSERT_EQ(with.plan.scales.size(), 4U);
  EXPECT_LT(with.plan.scales[3].factor, with.plan.scales[2].factor);
  EXPECT_LT(with.plan.scales[2].factor, with.plan.scales[1].factor);
  // The envelope is that of the hydrophones the record has: a dead board (8 of 32 gone from the middle) leaves sidelobes
  // near -12 dB where the whole row's are below -31 dB, and the plan knows it.
  PingRecord gap = row_record();
  gap.geometry.rx.erase(gap.geometry.rx.begin() + 12, gap.geometry.rx.begin() + 20);
  const std::vector<double> bearings(one.bearings);
  const DetectorPlan holed = plan_detector(gap, sonar_facts(row_record()), {}, one.n_cells, one.k, 0.25, bearings, 1, one.guard, one.train, true, 15.0);
  double whole = 0.0, with_gap = 0.0;
  for (std::size_t i = static_cast<std::size_t>(0.15 / one.plan.u_step); i < static_cast<std::size_t>(0.5 / one.plan.u_step); ++i) {
    whole = std::max(whole, one.plan.bearing_envelope[i]);
    with_gap = std::max(with_gap, holed.bearing_envelope[i]);
  }
  EXPECT_LT(whole, 1e-3);
  EXPECT_GT(with_gap, 2e-2);
}

TEST(Detector, NoiseAloneGivesNothingAndAPointEchoIsOneDetectionAtItsSampleRatio) {
  Map m(7);
  EXPECT_TRUE(m.detect().empty());
  // 16 dB in one sample: its 2 m cell's mean is up by 16 dB - 9 dB = 7 dB only; the detector until now needed 10.7 dB there.
  m.add(60, 300 * 8 + 3, 40.0F);
  const auto hits = m.detect();
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits[0].snr_db, 16.0, 0.6);
  EXPECT_NEAR(hits[0].range, 1000.0 + 0.25 * (300 * 8 + 3), 1.0);
  EXPECT_NEAR(hits[0].bearing, m.bearings[60], 1e-9);
  EXPECT_EQ(hits[0].extent, 2.0);
  // An echo spread evenly over a cell at 8 dB a sample: no sample reaches 13 dB, the cell's mean passes its own 7 dB.
  Map spread(8);
  for (int q = 0; q < 8; ++q) spread.fine[static_cast<std::size_t>(40 * spread.n_cells * 8 + 200 * 8 + q)] = 7.3F;
  const auto found = spread.detect();
  ASSERT_EQ(found.size(), 1U);
  EXPECT_LT(found[0].snr_db, 10.0 * std::log10(spread.plan.scales[0].factor));
}

TEST(Detector, SidelobesOfAStrongEchoAreNotDetectionsAndASecondEchoAboveThemIs) {
  Map m(11);
  const std::int64_t b0 = 80, s0 = 350 * 8 + 4;
  const double p0 = 1e6;                                             // 60 dB
  // The echo with its range sidelobes (sinc^2 of the compressed pulse) and its beam (Hann: cos^4-like core, -32 dB sidelobes).
  for (std::int64_t b = 0; b < m.n_b; ++b) {
    const double du = std::abs(std::sin(m.bearings[static_cast<std::size_t>(b)]) - std::sin(m.bearings[static_cast<std::size_t>(b0)]));
    const double x = du / 0.0625;                                    // in beamwidths (wavelength / aperture)
    const double beam = x < 2.0 ? std::pow(std::cos(0.25 * kPi * x), 4.0) : 6e-4 / (x * x);
    for (std::int64_t s = s0 - 320; s <= s0 + 320; ++s) {       // 80 m either side: inside the guard
      const double r = 0.25 * static_cast<double>(s - s0) / 0.375;
      const double pulse = std::abs(r) < 1e-9 ? 1.0 : std::pow(std::sin(kPi * r) / (kPi * r), 2.0);
      m.add(b, s, static_cast<float>(p0 * beam * pulse));
    }
  }
  auto hits = m.detect();
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits[0].snr_db, 60.0, 0.5);
  EXPECT_NEAR(hits[0].range, m.fine_ranges[static_cast<std::size_t>(s0)], 0.5);
  EXPECT_LE(hits[0].extent, 6.0);                                    // the cell and its neighbours, not the sidelobes
  // Its bearing: the whole half-power beam is kept, both sides (no cell of it is put down to another's sidelobes), and
  // the weights fall to nothing at its ends: within a twentieth of a degree on a half-degree grid.
  EXPECT_NEAR(hits[0].bearing, m.bearings[static_cast<std::size_t>(b0)], 0.05 * kPi / 180.0);
  EXPECT_GE(hits[0].cells, 9);
  // A second echo 60 m further and 20 degrees off, 35 dB down (25 dB): well above the first one's sidelobes there. Found.
  m.add(b0 + 40, s0 + 240, static_cast<float>(p0 * 10.0 / 3.16e4));
  hits = m.detect();
  ASSERT_EQ(hits.size(), 2U);
  // A third 20 m behind the first on its own bearing at 23 dB: within the margin of the first one's range sidelobes
  // there (-43.6 dB + 10 dB under 60 dB is 26.4 dB): not told apart from them, not reported.
  m.add(b0, s0 + 80, 200.0F);
  EXPECT_EQ(m.detect().size(), 2U);
}

TEST(Detector, TwoEchoesBeyondTheMergeDistanceAreTwoAndNearerAreOne) {
  // In range: merge_cells + 1 = 8 cells (16 m) or less apart are one detection.
  for (const auto& [cells_apart, want] : std::vector<std::pair<std::int64_t, std::size_t>>{{2, 1}, {8, 1}, {10, 2}, {40, 2}}) {
    Map m(21);
    m.add(70, 300 * 8, 300.0F);
    m.add(70, (300 + cells_apart) * 8, 300.0F);
    const auto hits = m.detect();
    ASSERT_EQ(hits.size(), want) << cells_apart;
    if (want == 1) EXPECT_EQ(hits[0].extent, 2.0 * static_cast<double>(cells_apart + 1));
  }
  // In bearing at one range: within half the half-power beamwidth (2.6 degrees here) they are one detection; on the
  // skirt of the other's beam (3.5 degrees: within the margin of its own response there) the weaker is put down to it;
  // beyond one and a half half-power beamwidths (7.9 degrees) they are two.
  for (const auto& [steps, want] : std::vector<std::pair<std::int64_t, std::size_t>>{{3, 1}, {5, 1}, {7, 1}, {18, 2}, {40, 2}}) {
    Map m(22);
    m.add(80, 300 * 8, 300.0F);
    m.add(80 + steps, 300 * 8, 300.0F);
    EXPECT_EQ(m.detect().size(), want) << steps;
  }
  // No merging asked for: touching cells only.
  Map plain(21, 1, {}, 0.0);
  plain.add(70, 300 * 8, 300.0F);
  plain.add(70, 302 * 8, 300.0F);
  EXPECT_EQ(plain.detect().size(), 2U);
}

TEST(Detector, BackgroundFollowsASlopeAtTheEndsOfTheMap) {
  // Power falling 3 dB per 300 cells: the plain mean of a one-sided window is low by the slope times the centroid's
  // offset (125 cells at the very end: 1.25 dB); the detector's background is within 0.15 dB everywhere.
  const std::int64_t n_b = 4, n_cells = 1200, guard = 50, train = 150;
  std::vector<float> power(static_cast<std::size_t>(n_b * n_cells));
  const double slope = std::log(2.0) / 300.0;
  std::mt19937 gen(3);
  std::gamma_distribution<double> looks(40.0, 1.0 / 40.0);            // a cell mean of 40 looks
  for (std::int64_t b = 0; b < n_b; ++b) {
    for (std::int64_t j = 0; j < n_cells; ++j) power[static_cast<std::size_t>(b * n_cells + j)] = static_cast<float>(std::exp(-slope * static_cast<double>(j)) * looks(gen));
  }
  std::vector<double> level, worth;
  detector_background(power.data(), n_b, n_cells, guard, train, level, worth);
  double worst = 0.0, plain_first = 0.0;
  for (std::int64_t b = 0; b < n_b; ++b) {
    for (std::int64_t j = 0; j < n_cells; ++j) {
      worst = std::max(worst, std::abs(10.0 * std::log10(level[static_cast<std::size_t>(b * n_cells + j)] / std::exp(-slope * static_cast<double>(j)))));
    }
    double s = 0.0;
    for (std::int64_t j = guard + 1; j <= guard + train; ++j) s += static_cast<double>(power[static_cast<std::size_t>(b * n_cells + j)]);
    plain_first += 10.0 * std::log10(s / static_cast<double>(train)) / static_cast<double>(n_b);
  }
  EXPECT_LT(worst, 0.35);
  EXPECT_NEAR(plain_first, -1.25, 0.15);
  // In the middle the window is balanced and worth all its cells; at the end it is worth a fifth of one side.
  EXPECT_NEAR(worth[600], 300.0, 1e-9);
  EXPECT_NEAR(worth[0], 150.0 / (1.0 + kPi * 0.7 + 2.0 * 0.8367), 3.0);
  // A map too short for a slope: the plain mean.
  detector_background(power.data(), n_b, 200, guard, train, level, worth);
  double s = 0.0;
  for (std::int64_t j = guard + 1; j <= guard + train && j < 200; ++j) s += static_cast<double>(power[static_cast<std::size_t>(j)]);
  EXPECT_NEAR(level[0], s / 149.0, 1e-9);
}

}  // namespace
