// The C++ far-field beamformers and array calibration on records built here, where the
// answer is known in closed form. tests/python/test_beamform_cpp.py holds the same code
// against the Python stages on simulated pings.
#include <cmath>
#include <complex>
#include <random>

#include <gtest/gtest.h>

#include "sonar/dsp/fft.hpp"
#include "sonar/live/beamform.hpp"
#include "sonar/live/calibrate.hpp"

using namespace sonar;        // NOLINT(google-build-using-namespace)
using namespace sonar::live;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;
constexpr double kC = 1500.0, kFs = 8000.0, kFc = 8000.0, kPitch = 0.09, kSigma = 4e-4;
constexpr std::int64_t kElements = 16, kSamples = 2048;
using Cf = std::complex<float>;
using Cd = std::complex<double>;

// A conditioned (complex baseband, compressed) record of a 16-element line array along body x,
// looking along body y, with nothing in it.
PingRecord empty_record() {
  PingRecord r;
  r.samples.resize(kElements, kSamples, SampleKind::ComplexBaseband);
  r.samples.sample_rate = kFs;
  r.samples.center_frequency = kFc;
  r.samples.start_time = 0.6;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 4000.0;           // the band is the whole of the baseband sample rate
  r.transmit.waveform.f_stop = 12000.0;
  r.transmit.waveform.duration = 0.01;
  r.environment.sound_speed = kC;
  for (std::int64_t e = 0; e < kElements; ++e) {
    Element el;
    el.lever_arm = Vec3d{(static_cast<double>(e) - 0.5 * (kElements - 1)) * kPitch, 0.0, 0.0};
    el.boresight = Vec3d{0.0, 1.0, 0.0};  // stated: a record has no default look direction
    r.geometry.rx.push_back(el);
  }
  set_default_channel_tables(r);
  return r;
}

Vec3d at(double range, double bearing_deg) {
  return {range * std::sin(bearing_deg * kDeg), range * std::cos(bearing_deg * kDeg), 0.0};
}

// The compressed echo of a point at a world position: a Gaussian pulse (well inside the band)
// at each element's exact two-way delay, with the carrier phase of that delay. Each channel
// can be given a gain, a phase and a displacement the record does not know about.
void add_echo(PingRecord& r, const Vec3d& source, double amplitude, const std::vector<double>* gain = nullptr,
              const std::vector<double>* phase = nullptr, const std::vector<Vec3d>* displacement = nullptr) {
  for (std::int64_t e = 0; e < kElements; ++e) {
    const auto ei = static_cast<std::size_t>(e);
    Vec3d p = r.geometry.rx[ei].lever_arm;
    if (displacement) p += (*displacement)[ei];
    const double tau = (norm(source) + norm(source - p)) / kC;
    const Cd scale = std::polar(amplitude * (gain ? (*gain)[ei] : 1.0), (phase ? (*phase)[ei] : 0.0) - 2.0 * kPi * kFc * tau);
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

// A far-field broadband noise source: the same noise on every channel, each with its
// plane-wave lead (a phase ramp over frequency, carrier included).
void add_jammer(PingRecord& r, double bearing_deg, double rms, unsigned seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> g(0.0F, static_cast<float>(rms / std::sqrt(2.0)));
  std::vector<Cf> noise(static_cast<std::size_t>(kSamples));
  for (Cf& v : noise) v = Cf(g(gen), g(gen));
  dsp::fft(noise, false);
  for (std::int64_t e = 0; e < kElements; ++e) {
    const double lead = r.geometry.rx[static_cast<std::size_t>(e)].lever_arm.x * std::sin(bearing_deg * kDeg) / kC;
    std::vector<Cf> ch = noise;
    for (std::int64_t k = 0; k < kSamples; ++k) {
      const double f = static_cast<double>(k < kSamples / 2 ? k : k - kSamples) * kFs / kSamples + kFc;
      ch[static_cast<std::size_t>(k)] *= Cf(std::polar(1.0, 2.0 * kPi * f * lead));
    }
    dsp::fft(ch, true);
    Cf* x = r.samples.channel_complex(e);
    for (std::int64_t i = 0; i < kSamples; ++i) x[i] += ch[static_cast<std::size_t>(i)];
  }
}

std::vector<double> bearing_grid(double first_deg, double last_deg, double step_deg) {
  std::vector<double> b;
  for (double v = first_deg; v <= last_deg + 0.5 * step_deg; v += step_deg) b.push_back(v * kDeg);
  return b;
}

// Peak at a sample over the RMS of the same beam more than 100 samples away, dB, and the peak.
double beam_snr_db(const BeamMap& m, std::int64_t bearing, std::int64_t sample, double* peak = nullptr) {
  const Cf* row = m.row(bearing);
  double top = 0.0, floor = 0.0;
  std::int64_t count = 0;
  for (std::int64_t i = 0; i < m.n_valid; ++i) {
    const double a = std::abs(row[i]);
    if (std::abs(i - sample) <= 3) top = std::max(top, a);
    if (std::abs(i - sample) > 100) {
      floor += a * a;
      ++count;
    }
  }
  if (peak) *peak = top;
  return 10.0 * std::log10(top * top / (floor / static_cast<double>(count)));
}

TEST(Beamform, SteeringDelaysArePositiveWhereTheElementIsNearerTheSource) {
  const PingRecord r = empty_record();
  const ArrayAxes f = array_axes(r);
  EXPECT_NEAR(f.forward.y, 1.0, 1e-12);
  EXPECT_NEAR(f.across.x, 1.0, 1e-12);          // the way the element index increases
  EXPECT_NEAR(f.normal.z, -1.0, 1e-12);
  const std::vector<double> tau = steering_delays(r, {30.0 * kDeg});
  ASSERT_EQ(tau.size(), static_cast<std::size_t>(kElements));
  for (std::int64_t e = 0; e < kElements; ++e) {
    const double x = (static_cast<double>(e) - 7.5) * kPitch;
    EXPECT_NEAR(tau[static_cast<std::size_t>(e)], x * 0.5 / kC, 1e-15);
  }
  // Hann across the aperture, reaching zero half a pitch beyond the outer elements.
  const std::vector<float> w = aperture_weights(r, "hann");
  EXPECT_NEAR(w[0], std::pow(std::cos(kPi * 7.5 / 16.0), 2), 1e-6);
  EXPECT_NEAR(w[7], std::pow(std::cos(kPi * 0.5 / 16.0), 2), 1e-6);
  EXPECT_THROW(aperture_weights(r, "kaiser"), std::invalid_argument);
}

TEST(Beamform, FarfieldGivesTheCoherentSumAtTheSourceBearing) {
  const double range = 600.0, bearing = 12.0, amplitude = 2.0;
  PingRecord r = empty_record();
  add_echo(r, at(range, bearing), amplitude);
  const std::vector<double> bearings = bearing_grid(-30.0, 30.0, 0.5);
  const BeamMap m = farfield_beamform(r, bearings);
  ASSERT_EQ(m.n_bearings, 121);
  ASSERT_EQ(m.n_samples, kSamples);
  EXPECT_EQ(m.n_valid, kSamples - 80);            // the last pulse length (10 ms at 8 kHz) is not usable
  EXPECT_NEAR(m.ranges[100], 0.5 * kC * (0.6 + 100.0 / kFs), 1e-9);
  // The largest value of the map is at the source: its bearing, its two-way time, N times its amplitude.
  std::size_t best = 0;
  for (std::size_t i = 0; i < m.data.size(); ++i) {
    if (std::abs(m.data[i]) > std::abs(m.data[best])) best = i;
  }
  const auto ib = static_cast<std::int64_t>(best) / kSamples, it = static_cast<std::int64_t>(best) % kSamples;
  EXPECT_NEAR(bearings[static_cast<std::size_t>(ib)] / kDeg, bearing, 0.25);
  EXPECT_NEAR(m.times[static_cast<std::size_t>(it)], 2.0 * range / kC, 0.5 / kFs);
  EXPECT_NEAR(std::abs(m.data[best]), kElements * amplitude, 0.01 * kElements * amplitude);
  // Along the source's beam the output is the pulse itself, times N, at the two-way delay to
  // the array centre and with that delay's carrier phase: 600 m is 27 Fraunhofer distances
  // (2 L^2 / lambda = 22 m), so the plane-wave model holds to a fraction of a percent.
  const std::int64_t i12 = 84;                    // (12 + 30) / 0.5
  double worst = 0.0;
  for (std::int64_t i = 0; i < kSamples; ++i) {
    const double t = m.times[static_cast<std::size_t>(i)] - 2.0 * range / kC;
    const Cd want = std::polar(kElements * amplitude * std::exp(-0.5 * t * t / (kSigma * kSigma)), -2.0 * kPi * kFc * 2.0 * range / kC);
    worst = std::max(worst, std::abs(Cd(m.row(i12)[i]) - want));
  }
  EXPECT_LT(worst, 0.005 * kElements * amplitude);
  // valid() drops the tail and nothing else.
  const BeamMap v = m.valid();
  EXPECT_EQ(v.n_samples, kSamples - 80);
  EXPECT_EQ(v.data.size(), static_cast<std::size_t>(121 * (kSamples - 80)));
  EXPECT_EQ(v.row(i12)[it], m.row(i12)[it]);
}

TEST(Beamform, HannWindowLowersTheSidelobes) {
  PingRecord r = empty_record();
  add_echo(r, at(600.0, 0.0), 1.0);
  const std::vector<double> bearings = bearing_grid(-60.0, 60.0, 0.25);
  const auto it = static_cast<std::int64_t>(std::llround((2.0 * 600.0 / kC - 0.6) * kFs));
  auto pattern = [&](const std::string& window) {
    const BeamMap m = farfield_beamform(r, bearings, window);
    std::vector<double> p(bearings.size());
    for (std::size_t b = 0; b < p.size(); ++b) p[b] = std::abs(m.row(static_cast<std::int64_t>(b))[it]);
    return p;
  };
  auto far_lobe_db = [&](const std::vector<double>& p) {       // highest value beyond 20 degrees, re the peak
    double side = 0.0;
    for (std::size_t b = 0; b < p.size(); ++b) {
      if (std::abs(bearings[b]) > 20.0 * kDeg) side = std::max(side, p[b]);
    }
    return 20.0 * std::log10(side / p[p.size() / 2]);
  };
  const std::vector<double> rect = pattern("rect"), hann = pattern("hann");
  EXPECT_NEAR(rect[rect.size() / 2], 16.0, 0.05);
  EXPECT_NEAR(hann[hann.size() / 2], 8.0, 0.05);               // sum of cos^2 over a full period: N / 2
  EXPECT_GT(far_lobe_db(rect), -25.0);                         // rectangular aperture: sidelobes fall off slowly
  EXPECT_LT(far_lobe_db(hann), -30.0);
  EXPECT_LT(far_lobe_db(hann), far_lobe_db(rect) - 12.0);
}

TEST(Beamform, RejectsRecordsThatAreNotConditioned) {
  PingRecord r = empty_record();
  r.samples.resize(kElements, kSamples, SampleKind::RealPassband);
  EXPECT_THROW(farfield_beamform(r, {0.0}), std::invalid_argument);
  EXPECT_THROW(mvdr_beamform(r, {0.0}), std::invalid_argument);
}

TEST(Mvdr, IsDistortionlessInWhiteNoise) {
  const double amplitude = 1.0;
  PingRecord r = empty_record();
  add_echo(r, at(600.0, 6.0), amplitude);
  add_noise(r, 0.2, 4);
  const std::vector<double> bearings = bearing_grid(-40.0, 40.0, 1.0);
  const auto it = static_cast<std::int64_t>(std::llround((2.0 * 600.0 / kC - 0.6) * kFs));
  MvdrOptions o;
  o.block = 128;
  double peak_c = 0.0, peak_m = 0.0;
  const double snr_c = beam_snr_db(farfield_beamform(r, bearings), 46, it, &peak_c);
  const double snr_m = beam_snr_db(mvdr_beamform(r, bearings, o), 46, it, &peak_m);
  EXPECT_NEAR(peak_c, kElements * amplitude, 0.05 * kElements);
  EXPECT_NEAR(peak_m, peak_c, 0.1 * peak_c);
  // With nothing directional to reject, estimating the covariance only costs noise gain.
  EXPECT_LT(snr_m, snr_c + 1.0);
  EXPECT_GT(snr_m, snr_c - 4.0);
}

TEST(Mvdr, FindsATargetUnderAJammerThatHidesItFromTheConventionalBeamformer) {
  // A broadband noise source at -20 degrees, 40 dB above the ambient noise and 20 dB above
  // the echo on each element. Its sidelobe leakage buries the target in the conventional beam.
  const double amplitude = 1.0;
  PingRecord r = empty_record();
  add_echo(r, at(600.0, 6.0), amplitude);
  add_noise(r, 0.1, 4);
  add_jammer(r, -20.0, 0.1 * std::pow(10.0, 40.0 / 20.0), 9);
  const std::vector<double> bearings = bearing_grid(-40.0, 40.0, 1.0);
  const auto it = static_cast<std::int64_t>(std::llround((2.0 * 600.0 / kC - 0.6) * kFs));
  MvdrOptions o;
  o.block = 128;
  const BeamMap conv = farfield_beamform(r, bearings), mvdr = mvdr_beamform(r, bearings, o);
  double peak_m = 0.0;
  const double snr_c = beam_snr_db(conv, 46, it), snr_m = beam_snr_db(mvdr, 46, it, &peak_m);
  EXPECT_LT(snr_c, 12.0);
  EXPECT_GT(snr_m, snr_c + 10.0);
  EXPECT_NEAR(peak_m, kElements * amplitude, 0.25 * kElements * amplitude);   // the target itself is untouched
  // Looking straight at the jammer nothing is rejected: that is the look direction.
  auto mean_power = [&](const BeamMap& m) {
    double p = 0.0;
    for (std::int64_t i = 0; i < m.n_valid; ++i) p += std::norm(m.row(20)[i]);
    return p / static_cast<double>(m.n_valid);
  };
  EXPECT_NEAR(10.0 * std::log10(mean_power(mvdr) / mean_power(conv)), 0.0, 1.5);
}

TEST(Mvdr, RefusesARecordWithNoTrainingDataLeft) {
  PingRecord r = empty_record();
  add_noise(r, 0.1, 1);
  MvdrOptions o;
  o.block = 1024;                                 // two blocks: a block and its guards are all there is
  EXPECT_THROW(mvdr_beamform(r, {0.0}, o), std::invalid_argument);
  o.block = 256;
  o.n_subbands = 0;
  EXPECT_THROW(mvdr_beamform(r, {0.0}, o), std::invalid_argument);
  // All zeros and no loading: the covariance cannot be inverted, and that is said.
  PingRecord z = empty_record();
  o.n_subbands = 4;
  o.loading = 0.0;
  EXPECT_THROW(mvdr_beamform(z, {0.0}, o), std::runtime_error);
}

struct Injected {
  std::vector<double> gain, phase;
  std::vector<Vec3d> position;
};

// 1 dB RMS of gain, 20 degrees RMS of phase, 5 mm RMS of position in the array plane.
Injected injected_errors() {
  std::mt19937 gen(11);
  std::normal_distribution<double> g(0.0, 1.0);
  Injected e;
  for (std::int64_t c = 0; c < kElements; ++c) {
    e.gain.push_back(std::pow(10.0, g(gen) / 20.0));
    e.phase.push_back(20.0 * kDeg * g(gen));
    e.position.push_back(Vec3d{5e-3 * g(gen), 5e-3 * g(gen), 0.0});
  }
  return e;
}

ChannelErrors estimate(const Injected& e, std::vector<PingRecord>& pings) {
  std::vector<Vec3d> sources;
  for (double b : {-40.0, -20.0, 0.0, 20.0, 40.0}) {
    sources.push_back(at(600.0, b));
    pings.push_back(empty_record());
    add_echo(pings.back(), sources.back(), 1.0, &e.gain, &e.phase, &e.position);
  }
  std::vector<const PingRecord*> ptrs;
  for (const PingRecord& p : pings) ptrs.push_back(&p);
  return estimate_channel_errors(ptrs, sources);
}

TEST(Calib, RecoversInjectedGainPhaseAndPosition) {
  const Injected e = injected_errors();
  std::vector<PingRecord> pings;
  const ChannelErrors est = estimate(e, pings);
  // Compare in the gauge the estimator reports: mean gain 1, mean phase 0, mean offset 0.
  double g_mean = 0.0, p_mean = 0.0;
  Vec3d d_mean;
  for (std::size_t c = 0; c < e.gain.size(); ++c) {
    g_mean += e.gain[c] / kElements;
    p_mean += e.phase[c] / kElements;
    d_mean += e.position[c] / static_cast<double>(kElements);
  }
  for (std::size_t c = 0; c < e.gain.size(); ++c) {
    EXPECT_NEAR(20.0 * std::log10(est.gain[c] / (e.gain[c] / g_mean)), 0.0, 0.02) << c;   // dB, against 1 dB RMS injected
    EXPECT_NEAR((est.phase[c] - (e.phase[c] - p_mean)) / kDeg, 0.0, 1.0) << c;
    EXPECT_NEAR(est.position[c].x, e.position[c].x - d_mean.x, 0.3e-3) << c;
    EXPECT_NEAR(est.position[c].y, e.position[c].y - d_mean.y, 0.3e-3) << c;
    EXPECT_EQ(est.position[c].z, 0.0);
  }
  EXPECT_LT(est.residual_rms_deg, 0.5);
  // The response of an error-free array is the same on every channel: the echo's amplitude, no phase.
  PingRecord clean = empty_record();
  add_echo(clean, at(600.0, 20.0), 3.0);
  for (const Cd& v : channel_responses(clean, at(600.0, 20.0))) {
    EXPECT_NEAR(v.real(), 3.0, 1e-4);
    EXPECT_NEAR(v.imag(), 0.0, 1e-4);
  }
  // Two unknowns per channel plus the phase need three sources.
  std::vector<const PingRecord*> two{&pings[0], &pings[1]};
  EXPECT_THROW(estimate_channel_errors(two, {at(600.0, -40.0), at(600.0, -20.0)}), std::invalid_argument);
  EXPECT_THROW(estimate_channel_errors(two, {at(600.0, -40.0)}), std::invalid_argument);
}

TEST(Calib, ApplyingTheEstimateRestoresTheBeam) {
  const Injected e = injected_errors();
  std::vector<PingRecord> pings;
  const ChannelErrors est = estimate(e, pings);
  // A fresh ping on a target that was not one of the calibration sources.
  const Vec3d target = at(600.0, 11.0);
  const std::vector<double> bearings = bearing_grid(-20.0, 40.0, 0.25);
  auto peak = [&](const PingRecord& r) {
    double top = 0.0;
    for (const Cf& v : farfield_beamform(r, bearings).data) top = std::max(top, static_cast<double>(std::abs(v)));
    return top;
  };
  PingRecord ideal = empty_record(), raw = empty_record();
  add_echo(ideal, target, 1.0);
  add_echo(raw, target, 1.0, &e.gain, &e.phase, &e.position);
  PingRecord fixed = raw;
  apply_channel_errors(fixed, est);
  double g_mean = 0.0;
  for (double g : e.gain) g_mean += g / kElements;
  EXPECT_NEAR(peak(ideal), 16.0, 0.05);
  EXPECT_LT(peak(raw), 0.97 * peak(ideal) * g_mean);            // errors cost coherent gain
  // Calibration gives it back, up to the one thing it cannot see: the array's mean gain.
  EXPECT_NEAR(peak(fixed), peak(ideal) * g_mean, 0.01 * peak(ideal) * g_mean);
  // Gain and phase are spent (reset to neutral); the position offsets stay for the beamformer.
  EXPECT_EQ(fixed.calibration.gain[3], 1.0);
  EXPECT_EQ(fixed.calibration.phase[3], 0.0);
  EXPECT_EQ(fixed.calibration.position_offset[3].x, est.position[3].x);
  std::vector<double> short_gain(3, 1.0);
  EXPECT_THROW(store_calibration(raw, ChannelErrors{short_gain, short_gain, {}, 0.0}), std::invalid_argument);
}

}  // namespace
