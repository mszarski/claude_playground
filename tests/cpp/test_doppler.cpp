// The live chain's sweep Doppler bank (D4; libs/live doppler.hpp) on records built here: the mismatch closed form, the
// plan's hypotheses and their use per beam, the fold over hypotheses, the detector's offset and pedestal, the options,
// and the chain with the bank against the chain without it. tests/python/test_doppler_bank.py and test_live_cpp.py hold
// the same code against the Python engine and simulated pings.
#include <cmath>
#include <complex>
#include <random>
#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "sonar/live/chain.hpp"
#include "sonar/live/detector.hpp"
#include "sonar/live/doppler.hpp"
#include "sonar/live/options.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kC = 1500.0, kFs = 12000.0, kFc = 8000.0, kPitch = 0.075;
constexpr std::int64_t kElements = 16;
using Cf = std::complex<float>;

// Complex baseband (not compressed) of a 16-element line along body x looking along body y, a 6-10 kHz, 1 s LFM
// without taper, 190 to 310 m of window, nothing in it.
PingRecord empty_record() {
  PingRecord r;
  const double start = 2.0 * 190.0 / kC;
  const auto n = static_cast<std::int64_t>(std::ceil((2.0 * 310.0 / kC + 1.0 - start) * kFs));
  r.samples.resize(kElements, n, SampleKind::ComplexBaseband);
  r.samples.sample_rate = kFs;
  r.samples.center_frequency = kFc;
  r.samples.start_time = start;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 6000.0;
  r.transmit.waveform.f_stop = 10000.0;
  r.transmit.waveform.duration = 1.0;
  r.environment.sound_speed = kC;
  for (std::int64_t e = 0; e < kElements; ++e) {
    Element el;
    el.lever_arm = Vec3d{(static_cast<double>(e) - 0.5 * (kElements - 1)) * kPitch, 0.0, 0.0};
    el.boresight = Vec3d{0.0, 1.0, 0.0};
    r.geometry.rx.push_back(el);
  }
  set_default_channel_tables(r);
  return r;
}

// The echo of a point at `range` along body y, opening at `rate`: the pulse time-scaled by (c - rate) / (c + rate) at
// each element's two-way delay, mixed down with the record's carrier (phase referenced to the transmit).
void add_echo(PingRecord& r, double range, double rate, double amplitude) {
  const Waveform& w = r.transmit.waveform;
  const double eta = (kC - rate) / (kC + rate), k = (w.f_stop - w.f_start) / w.duration;
  const Vec3d target{0.0, range, 0.0};
  for (std::int64_t e = 0; e < kElements; ++e) {
    const Vec3d p = r.geometry.rx[static_cast<std::size_t>(e)].lever_arm;
    const double tau = (norm(target) + norm(target - p)) / kC;
    Cf* x = r.samples.channel_complex(e);
    for (std::int64_t i = 0; i < r.samples.n_samples; ++i) {
      const double t = r.samples.start_time + static_cast<double>(i) / kFs;
      const double u = eta * (t - tau);
      if (u < 0.0 || u > w.duration) continue;
      const double cycles = w.f_start * u + 0.5 * k * u * u - kFc * t;
      x[i] += Cf(static_cast<float>(amplitude * std::cos(2.0 * kPi * cycles)), static_cast<float>(amplitude * std::sin(2.0 * kPi * cycles)));
    }
  }
}

void add_noise(PingRecord& r, double rms, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> g(0.0F, static_cast<float>(rms / std::sqrt(2.0)));
  for (float& v : r.samples.data) v += g(gen);
}

ChainOptions options(bool bank, double max_rate = 2.5) {
  ChainOptions o;
  o.bearing_first_deg = -30.0;
  o.bearing_last_deg = 30.0;
  o.bearing_step_deg = 2.0;
  o.range_cell_m = 1.0;
  o.cfar_guard_m = 10.0;
  o.cfar_train_m = 30.0;
  o.tracker.range_sigma = 3.0;
  o.tracker.bearing_sigma = 2.0 * kDeg;
  if (bank) {
    o.doppler.mode = DopplerMode::SweepBank;
    o.doppler.max_range_rate = max_rate;
    o.doppler.report = true;
  }
  return o;
}

std::vector<double> bearings_of(const ChainOptions& o) {
  std::vector<double> b;
  for (double d = o.bearing_first_deg; d <= o.bearing_last_deg + 1e-9; d += o.bearing_step_deg) b.push_back(d * kDeg);
  return b;
}

TEST(Doppler, MismatchIsTheFresnelIntegralWithTheDelayFree) {
  EXPECT_EQ(sweep_mismatch(0.0), 1.0);
  EXPECT_NEAR(-20.0 * std::log10(sweep_mismatch(kPi / 2.0)), 0.967, 0.002);   // the design note's tolerance
  EXPECT_NEAR(-20.0 * std::log10(sweep_mismatch(kPi / 4.0)), 0.239, 0.002);   // half of it: the bank's worst
  // 2 m/s on a 1 s, 4 kHz sweep: phi = pi B T r / c = 16.76; centred 13.5 dB, the delay free 11.7 dB.
  EXPECT_NEAR(-20.0 * std::log10(sweep_mismatch(kPi * 4000.0 * 2.0 / kC)), 11.70, 0.02);
  EXPECT_EQ(sweep_mismatch(-1.0), sweep_mismatch(1.0));
}

TEST(Doppler, PlanSpacesHypothesesByTheSweepsToleranceAndKeepsTheZeroOneExact) {
  const PingRecord r = empty_record();
  const ChainOptions o = options(true);
  const DopplerPlan p = plan_doppler(r, bearings_of(o), o.doppler);
  ASSERT_TRUE(p.on);
  EXPECT_NEAR(p.step, kC / (2.0 * 4000.0 * 1.0), 1e-15);
  EXPECT_EQ(p.per_beam, 29);                      // K = ceil(2.5 / 0.1875) = 14
  ASSERT_EQ(p.rates.size(), 29U);
  ASSERT_EQ(p.zero, 14);
  EXPECT_EQ(p.rates[14], 0.0);
  EXPECT_EQ(p.scales[14], 1.0);                   // the compression without the bank, exactly
  EXPECT_EQ(p.shifts[14], 0.0);
  for (std::size_t h = 0; h < p.rates.size(); ++h) {
    const double r_h = p.rates[h];
    EXPECT_NEAR(r_h, (static_cast<double>(h) - 14.0) * p.step, 1e-12);
    EXPECT_NEAR(p.scales[h], (kC - r_h) / (kC + r_h), 1e-15);
    EXPECT_NEAR(p.shifts[h], 2.0 * lfm_range_doppler_factor(r.transmit.waveform) * r_h / kC, 1e-15);
  }
  for (std::int64_t f : p.first) EXPECT_EQ(f, 0);
  EXPECT_EQ(p.baseband_max_range_rate, 0.0);      // 2 x 2.6 x 10 kHz / c = 35 Hz: inside the band's 400 Hz of room
  EXPECT_EQ(p.threshold_db, bank_threshold_db(29));
  EXPECT_NEAR(p.pedestal_m, 29.0 * p.step * 1.0, 1e-12);
  EXPECT_EQ(bank_threshold_db(1), 0.0);
  DopplerOptions count = o.doppler;
  count.hypotheses = 4;
  EXPECT_THROW(plan_doppler(r, bearings_of(o), count), std::invalid_argument);
  count.hypotheses = 5;
  EXPECT_EQ(plan_doppler(r, bearings_of(o), count).per_beam, 5);
  PingRecord tone = r;
  tone.transmit.waveform.type = WaveformType::Cw;
  EXPECT_THROW(plan_doppler(tone, bearings_of(o), o.doppler), std::invalid_argument);
  DopplerOptions off;
  EXPECT_FALSE(plan_doppler(r, bearings_of(o), off).on);
}

TEST(Doppler, EachBeamIsCentredOnItsOwnShipRateAndAnUnknownVelocityWidensTheSpan) {
  PingRecord r = empty_record();
  r.geometry.tx_pose.velocity = r.geometry.rx_pose.velocity = Vec3d{1.5, 0.0, 0.0};   // along the array
  const ChainOptions o = options(true);
  const std::vector<double> b = bearings_of(o);
  const DopplerPlan p = plan_doppler(r, b, o.doppler);
  EXPECT_TRUE(p.velocity_known);
  for (std::size_t i = 0; i < b.size(); ++i) {
    EXPECT_NEAR(p.ownship[i], -1.5 * std::sin(b[i]), 1e-12);
    const double centre = p.rates[static_cast<std::size_t>(p.first[i] + (p.per_beam - 1) / 2)];
    EXPECT_LE(std::abs(centre - p.ownship[i]), 0.5 * p.step + 1e-12);
  }
  EXPECT_EQ(p.per_beam, 29);
  EXPECT_EQ(p.rates.size(), 29U + 8U);           // 1.5 sin 30 = 0.75 m/s either side: 4 steps more each way
  r.geometry.tx_pose.velocity.x = std::numeric_limits<double>::quiet_NaN();
  const DopplerPlan q = plan_doppler(r, b, o.doppler);
  EXPECT_FALSE(q.velocity_known);
  EXPECT_EQ(q.per_beam, 2 * static_cast<std::int64_t>(std::ceil(4.5 / q.step - 1e-9)) + 1);
  for (std::size_t i = 0; i < b.size(); ++i) EXPECT_EQ(q.ownship[i], 0.0);
}

TEST(Doppler, FoldKeepsTheLargestOfEachBeamsOwnHypothesesAndTheEarlierOnATie) {
  DopplerPlan p;
  p.on = true;
  p.per_beam = 2;
  p.first = {0, 1};                                // beam 0 takes hypotheses 0 and 1, beam 1 takes 1 and 2
  std::vector<float> best;
  std::vector<std::int32_t> which;
  fold_start(2, 3, best, which);
  const std::vector<float> h0{1, 5, 2, 9, 9, 9}, h1{3, 5, 1, 1, 1, 1}, h2{0, 0, 7, 2, 0, 4};
  fold_hypothesis(p, 0, h0.data(), 2, 3, best, which);
  fold_hypothesis(p, 1, h1.data(), 2, 3, best, which);
  fold_hypothesis(p, 2, h2.data(), 2, 3, best, which);
  EXPECT_EQ(best, (std::vector<float>{3, 5, 2, 2, 1, 4}));
  EXPECT_EQ(which, (std::vector<std::int32_t>{1, 0, 0, 2, 1, 2}));   // beam 1 never sees hypothesis 0's 9s
  const std::vector<double> ranges{10.0, 11.0, 12.0};
  EXPECT_EQ(detection_hypothesis(best.data(), which.data(), 3, 0, 3, ranges, {0.0, 0.1}, 0.09, 10.5, 12.5), 2);   // beam 1, 11 or 12 m: 4 at 12 m
  EXPECT_EQ(detection_hypothesis(best.data(), which.data(), 3, 0, 3, ranges, {0.0, 0.1}, 0.0, 9.5, 10.5), 1);
  EXPECT_EQ(detection_hypothesis(best.data(), which.data(), 3, 0, 3, ranges, {0.0, 0.1}, 0.0, 50.0, 60.0), 0);   // nothing there: the row's best
}

TEST(Detector, AnOffsetAndAPedestalOfZeroLeaveThePlanAsItWas) {
  const PingRecord r = empty_record();
  const SonarFacts facts = sonar_facts(r);
  const std::vector<double> b = bearings_of(options(false));
  DetectorOptions o;
  const DetectorPlan a = plan_detector(r, facts, o, 120, 8, 0.125, b, 1, 10, 30, true, 3.0);
  o.threshold_offset_db = 0.0;
  o.pedestal_m = 0.0;
  const DetectorPlan same = plan_detector(r, facts, o, 120, 8, 0.125, b, 1, 10, 30, true, 3.0);
  EXPECT_EQ(a.json(), same.json());
  EXPECT_EQ(a.range_envelope, same.range_envelope);
  o.threshold_offset_db = -1.5;
  o.pedestal_m = 6.0;
  const DetectorPlan c = plan_detector(r, facts, o, 120, 8, 0.125, b, 1, 10, 30, true, 3.0);
  for (std::size_t s = 0; s < a.scales.size(); ++s) {
    EXPECT_NEAR(c.scales[s].factor, a.scales[s].factor * std::pow(10.0, -0.15), 1e-12 * a.scales[s].factor);
    EXPECT_EQ(c.scales[s].pfa, a.scales[s].pfa);
  }
  bool raised = false;
  for (std::size_t d = 0; d < a.range_envelope.size(); ++d) {
    EXPECT_GE(c.range_envelope[d], a.range_envelope[d]);
    raised = raised || c.range_envelope[d] > a.range_envelope[d];
    if (d >= 8) EXPECT_EQ(c.range_envelope[d], a.range_envelope[d]);   // 6 m of 1 m cells: within 7 cells only
  }
  EXPECT_TRUE(raised);
}

TEST(Options, DopplerKeysAreReadEchoedAndChecked) {
  ChainOptions a;
  PassiveOptions pa;
  EXPECT_EQ(a.doppler.mode, DopplerMode::Off);
  EXPECT_NE(options_json(a).find("\"doppler_mode\": \"off\""), std::string::npos);
  std::istringstream t("doppler_mode=sweep_bank\ndoppler_max_range_rate_m_s=1.5\ndoppler_step_m_s=0.25\ndoppler_hypotheses=9\n"
                       "doppler_unknown_ownship_m_s=3\ndoppler_threshold_db=-1.25\ndoppler_report=1\n");
  read_options(a, pa, t);
  EXPECT_EQ(a.doppler.mode, DopplerMode::SweepBank);
  EXPECT_EQ(a.doppler.max_range_rate, 1.5);
  EXPECT_EQ(a.doppler.step, 0.25);
  EXPECT_EQ(a.doppler.hypotheses, 9);
  EXPECT_EQ(a.doppler.unknown_ownship, 3.0);
  EXPECT_EQ(a.doppler.threshold_db, -1.25);
  EXPECT_TRUE(a.doppler.report);
  EXPECT_NE(options_json(a).find("\"doppler_mode\": \"sweep_bank\""), std::string::npos);
  for (const char* bad : {"doppler_mode=bank\n", "doppler_max_range_rate_m_s=-1\n", "doppler_mod=sweep_bank\n"}) {
    ChainOptions m;
    std::istringstream b(bad);
    EXPECT_THROW(read_options(m, pa, b), std::invalid_argument) << bad;
  }
  ChainOptions mean = options(true);
  mean.cell_multiscale = false;
  EXPECT_THROW(LiveChain{mean}, std::invalid_argument);
  ChainOptions far = options(true);
  far.beamformer = "farfield";
  EXPECT_THROW(LiveChain{far}, std::invalid_argument);
}

TEST(Doppler, OneHypothesisIsTheChainWithoutTheBankLineForLine) {
  PingRecord a = empty_record();
  add_echo(a, 250.0, 1.0, 1.0);
  add_noise(a, 0.05, 3);
  PingRecord b = a;
  ChainOptions zero = options(true, 0.0);
  zero.doppler.report = false;
  LiveChain off(options(false)), one(zero);
  Report ra = off.process(a), rb = one.process(b);
  ra.processing_seconds = rb.processing_seconds = 0.0;
  EXPECT_EQ(ra.json(), rb.json());
  ASSERT_GE(ra.detections.size(), 1U);
  EXPECT_EQ(one.doppler_plan().rates.size(), 1U);
  EXPECT_TRUE(one.last_hypotheses().empty());
}

TEST(Doppler, TheBankGetsAMoversPeakBackAndItsRangeRateAndLeavesItsRange) {
  // Opening at 2 m/s on a 1 s, 4 kHz sweep: the zero hypothesis loses 11.7 dB of peak (the closed form); the bank finds
  // the hypothesis of 2 m/s and the detection keeps its range (every hypothesis aligned to the zero one).
  PingRecord a = empty_record();
  add_echo(a, 250.0, 2.0, 0.2);
  add_noise(a, 0.05, 4);
  PingRecord b = a;
  LiveChain off(options(false)), bank(options(true));
  const Report ra = off.process(a), rb = bank.process(b);
  ASSERT_EQ(ra.detections.size(), 1U);
  ASSERT_EQ(rb.detections.size(), 1U);
  const DopplerPlan& p = bank.doppler_plan();
  EXPECT_NEAR(rb.detections[0].range_rate_m_s, 2.0, 0.5 * p.step + 1e-12);
  EXPECT_EQ(rb.detections[0].range_rate_step_m_s, p.step);
  // The leading edge of the zero hypothesis' echo is smeared by the mismatch over +-|r| T (2 m) about where the matched
  // hypothesis shows it, a sharp peak: the ranges agree to within that smear.
  EXPECT_NEAR(rb.detections[0].range_m, ra.detections[0].range_m, 2.0);
  EXPECT_NEAR(rb.detections[0].bearing_deg, ra.detections[0].bearing_deg, 0.5);
  // snr_db is the strongest sample over the background: the peak's 11.7 dB less the background's rise (the maximum of
  // noise over 29 hypotheses) and the map's sampling of a narrower peak.
  EXPECT_GT(rb.detections[0].snr_db - ra.detections[0].snr_db, 5.0) << ra.detections[0].snr_db << " -> " << rb.detections[0].snr_db;
  EXPECT_NE(rb.json().find("\"range_rate_m_s\": 2"), std::string::npos) << rb.json();
  EXPECT_EQ(ra.json().find("range_rate"), std::string::npos);
  EXPECT_EQ(bank.last_hypotheses().size(), bank.last_ranges.size() * bank.bearings.size() * static_cast<std::size_t>(bank.detector_plan().samples_per_cell));
}

}  // namespace
