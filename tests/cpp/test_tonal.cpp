// Passive tonal analysis on its own: the Goertzel sum against the Fourier sum, the DEMON
// threshold and harmonic grouping, and the analyzer on a line array hearing a tone and a
// modulated noise source, whole and in pieces. tests/python/test_tonal.py holds the same code
// against the NumPy reference and against closed forms.
#include <cmath>
#include <complex>
#include <random>

#include <gtest/gtest.h>

#include "sonar/live/tonal.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kC = 1500.0;

// A line of `n` hydrophones `pitch` apart, real samples at `fs`.
PingRecord line_record(std::int64_t n, double pitch, std::int64_t samples, double fs, std::int64_t time_ns = 0) {
  PingRecord rec;
  rec.geometry.rx.resize(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) rec.geometry.rx[static_cast<std::size_t>(i)].lever_arm = {pitch * static_cast<double>(i), 0.0, 0.0};
  rec.environment.sound_speed = kC;
  rec.samples.resize(n, samples, SampleKind::RealPassband);
  rec.samples.sample_rate = fs;
  rec.samples.start_time = 0.0;
  rec.transmit.time_ns = time_ns;
  rec.transmit.waveform.f_start = 800.0;
  rec.transmit.waveform.f_stop = 1600.0;
  return rec;
}

double offset(std::int64_t i, std::int64_t n, double pitch) { return pitch * (static_cast<double>(i) - 0.5 * static_cast<double>(n - 1)); }

// Unit noise on every hydrophone; a tone from one bearing; noise modulated at a shaft rate and
// its fifth harmonic from another. The noise source reaches hydrophone i earlier by a whole
// number of samples per element (the bearing is chosen for that), so no interpolation is needed.
PingRecord scene(double seconds, unsigned seed) {
  const std::int64_t n = 12;
  const double fs = 4000.0, pitch = 0.6;
  const auto samples = static_cast<std::int64_t>(seconds * fs);
  PingRecord rec = line_record(n, pitch, samples, fs);
  std::mt19937 gen(seed);
  std::normal_distribution<double> g(0.0, 1.0);
  const std::int64_t lag = 1;  // samples per element: sin(b) = lag c / (fs pitch) = 0.625, b = 38.68 degrees
  std::vector<double> source(static_cast<std::size_t>(samples + n * lag));
  for (std::size_t i = 0; i < source.size(); ++i) {
    const double t = static_cast<double>(i) / fs;
    source[i] = 1.0 * (1.0 + 0.2 * std::cos(2.0 * kPi * 2.5 * t) + 0.5 * std::cos(2.0 * kPi * 12.5 * t + 1.0)) * g(gen);
  }
  const double tone_bearing = -20.0 * kDeg;
  for (std::int64_t ch = 0; ch < n; ++ch) {
    float* x = rec.samples.channel(ch);
    const double lead = offset(ch, n, pitch) * std::sin(tone_bearing) / kC;
    for (std::int64_t i = 0; i < samples; ++i) {
      const double t = static_cast<double>(i) / fs;
      x[i] = static_cast<float>(g(gen) + 0.5 * std::cos(2.0 * kPi * 1000.4 * (t + lead) + 0.5) +
                                source[static_cast<std::size_t>(i + ch * lag)]);
    }
  }
  return rec;
}

TonalOptions options() {
  TonalOptions o;
  o.enabled = true;
  o.bearing_first_deg = -60.0;
  o.bearing_last_deg = 60.0;
  o.bearing_step_deg = 2.0;
  o.goertzel.enabled = true;
  o.goertzel.lines = {1000.0};
  o.goertzel.block_s = 0.5;
  o.demon.enabled = true;
  o.demon.f_low = 800.0;
  o.demon.f_high = 1600.0;
  o.demon.segment_s = 4.0;
  return o;
}

TEST(Goertzel, IsTheFourierSumAtAnyFrequency) {
  std::mt19937 gen(1);
  std::normal_distribution<float> g(0.0F, 1.0F);
  const std::int64_t n = 500;
  const double fs = 2000.0;
  std::vector<float> x(n);
  std::vector<std::complex<float>> z(n);
  for (std::int64_t i = 0; i < n; ++i) {
    x[static_cast<std::size_t>(i)] = g(gen);
    z[static_cast<std::size_t>(i)] = {g(gen), g(gen)};
  }
  for (double f : {64.0, 301.37, 999.0}) {
    std::complex<double> real_sum{}, complex_sum{};
    for (std::int64_t i = 0; i < n; ++i) {
      const std::complex<double> e = std::polar(1.0, -2.0 * kPi * f * static_cast<double>(i) / fs);
      real_sum += static_cast<double>(x[static_cast<std::size_t>(i)]) * e;
      complex_sum += std::complex<double>(z[static_cast<std::size_t>(i)]) * e;
    }
    EXPECT_LT(std::abs(goertzel(x.data(), n, f, fs) - real_sum), 1e-9 * static_cast<double>(n));
    EXPECT_LT(std::abs(goertzel(z.data(), n, f, fs) - complex_sum), 1e-9 * static_cast<double>(n));
  }
  // a cosine of amplitude A on the frequency: |X| = A N / 2
  for (std::int64_t i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = static_cast<float>(0.7 * std::cos(2.0 * kPi * 250.0 * static_cast<double>(i) / fs + 0.4));
  const std::complex<double> v = goertzel(x.data(), n, 250.0, fs);
  EXPECT_NEAR(std::abs(v), 0.7 * static_cast<double>(n) / 2.0, 1e-4);
  EXPECT_NEAR(std::arg(v), 0.4, 1e-6);
}

TEST(Demon, ThresholdIsTheGammaTail) {
  double n = 0.0;
  EXPECT_NEAR(demon_threshold(1e-3, 1, 0.0, 0, &n), -std::log(1e-3), 1e-9);  // one spectrum, known background: exponential
  EXPECT_EQ(n, 1.0);
  // two: P(mean > t) = exp(-2t) (1 + 2t)
  const double t2 = demon_threshold(1e-2, 2, 0.0);
  EXPECT_NEAR(std::exp(-2.0 * t2) * (1.0 + 2.0 * t2), 1e-2, 1e-9);
  EXPECT_LT(demon_threshold(1e-4, 16, 0.0), demon_threshold(1e-4, 4, 0.0));  // more averaging, lower threshold
  // one spectrum over the mean of m independent others: P(x / y > t) = (1 + t / m)^-m; 35 correlated bins count as 18
  const double t1 = demon_threshold(1e-3, 1, 0.0, 35);
  EXPECT_NEAR(std::pow(1.0 + t1 / 18.0, -18.0), 1e-3, 1e-9);
  EXPECT_GT(t1, -std::log(1e-3));  // an estimated background needs a higher threshold
  demon_threshold(1e-2, 8, 0.25, 0, &n);
  EXPECT_NEAR(n, 8.0 / (1.0 + 2.0 * 0.25 * 7.0 / 8.0), 1e-12);  // overlapped segments count for fewer: 5.57
}

TEST(Demon, HarmonicSetGivesShaftRateAndBladeCount) {
  TonalOptions::Demon o;
  auto source = [](std::initializer_list<std::pair<double, double>> lines) {
    DemonSource s;
    for (const auto& [f, w] : lines) {
      DemonLine l;
      l.frequency = f;
      l.snr_db = w;
      s.lines.push_back(l);
    }
    return s;
  };
  DemonSource s = source({{15.01, 30.0}, {3.0, 15.0}, {29.98, 22.0}, {22.4, 6.0}});
  harmonic_set(s, 0.125, o);
  EXPECT_NEAR(s.shaft_hz, 3.0, 0.005);
  EXPECT_EQ(s.blades, 5);
  EXPECT_NEAR(s.blade_hz, 15.0, 0.02);
  EXPECT_EQ(s.lines[0].harmonic, 5);
  EXPECT_EQ(s.lines[2].harmonic, 10);
  EXPECT_EQ(s.lines[3].harmonic, 0);
  EXPECT_NEAR(s.confidence, 67.0 / 73.0 * 0.75, 1e-12);
  s = source({{17.6, 20.0}, {35.2, 12.0}});  // the blade rate and its second harmonic alone: a rate, no blade count
  harmonic_set(s, 0.125, o);
  EXPECT_NEAR(s.shaft_hz, 17.6, 0.01);
  EXPECT_EQ(s.blades, 0);
  s = source({{9.0, 20.0}});  // one line is not a rate
  harmonic_set(s, 0.125, o);
  EXPECT_EQ(s.shaft_hz, 0.0);
  EXPECT_EQ(s.confidence, 0.0);
}

TEST(TonalAnalyzer, FindsATonesFrequencyBearingAndAmplitudeAndAPropellersRates) {
  const PingRecord rec = scene(24.0, 3);
  TonalAnalyzer analyzer(options());
  const std::vector<TonalEvent> events = analyzer.push(rec);
  std::int64_t blocks = 0;
  const TonalEvent* last_demon = nullptr;
  for (const TonalEvent& e : events) {
    if (e.demon) {
      last_demon = &e;
      continue;
    }
    ++blocks;
    ASSERT_EQ(e.lines.size(), 1U);
    const TonalLine& q = e.lines[0];
    EXPECT_TRUE(q.detected);
    EXPECT_NEAR(q.frequency, 1000.4, 0.2);
    EXPECT_NEAR(q.bearing / kDeg, -20.0, 2.0);
    // the bin holds noise of 0.019 rms beside the tone
    EXPECT_NEAR(q.amplitude, 0.5, 0.07);
    EXPECT_EQ(e.time_ns, blocks * 500000000);
  }
  EXPECT_EQ(blocks, 48);
  ASSERT_NE(last_demon, nullptr);
  ASSERT_GE(last_demon->sources.size(), 1U);
  const DemonSource& src = last_demon->sources[0];
  EXPECT_NEAR(src.bearing / kDeg, std::asin(0.625) / kDeg, 1.0);
  EXPECT_NEAR(src.shaft_hz, 2.5, 0.01);
  EXPECT_EQ(src.blades, 5);
  EXPECT_NEAR(src.blade_hz, 12.5, 0.05);
  EXPECT_GE(src.confidence, 0.5);  // two lines of the set: 0.5 by the definition
  ASSERT_FALSE(analyzer.gram_frequencies.empty());
  EXPECT_EQ(analyzer.gram_db.size(), analyzer.bearings.size() * analyzer.gram_frequencies.size());
}

TEST(TonalAnalyzer, AStreamInPiecesGivesTheSameAsTheWhole) {
  const PingRecord rec = scene(20.0, 5);
  TonalAnalyzer whole(options()), stream(options());
  const std::string want = tonal_events_json(whole.push(rec), whole);
  std::vector<TonalEvent> got;
  const std::int64_t lengths[] = {1000, 7777, 3, 20000};
  std::int64_t at = 0, i = 0;
  while (at < rec.samples.n_samples) {
    const std::int64_t n = std::min(lengths[i++ % 4], rec.samples.n_samples - at);
    PingRecord part = line_record(rec.samples.n_channels, 0.6, n, rec.samples.sample_rate,
                                  static_cast<std::int64_t>(std::llround(static_cast<double>(at) / rec.samples.sample_rate * 1e9)));
    for (std::int64_t ch = 0; ch < rec.samples.n_channels; ++ch) std::copy_n(rec.samples.channel(ch) + at, n, part.samples.channel(ch));
    const auto events = stream.push(part);
    got.insert(got.end(), events.begin(), events.end());
    at += n;
  }
  EXPECT_GT(got.size(), 40U);
  EXPECT_EQ(tonal_events_json(got, stream), want);
}

TEST(TonalOptions, KeysAreSetFromTextAndOthersAreLeftAlone) {
  TonalOptions o;
  EXPECT_FALSE(set_tonal_option(o, "n_fft", "256"));
  EXPECT_TRUE(set_tonal_option(o, "tonal", "1"));
  EXPECT_TRUE(set_tonal_option(o, "tonal_goertzel_lines", "300,450.5"));
  EXPECT_TRUE(set_tonal_option(o, "tonal_demon", "1"));
  EXPECT_TRUE(set_tonal_option(o, "tonal_demon_envelope", "hilbert"));
  EXPECT_TRUE(o.enabled && o.goertzel.enabled && o.demon.enabled && !o.demon.square);
  ASSERT_EQ(o.goertzel.lines.size(), 2U);
  EXPECT_EQ(o.goertzel.lines[1], 450.5);
  EXPECT_THROW(set_tonal_option(o, "tonal_nothing", "1"), std::invalid_argument);
}

}  // namespace
