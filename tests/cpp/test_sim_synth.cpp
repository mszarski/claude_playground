// The synthesis kernel's sampled-pulse branch and arrival-time solve (sonar/sim/synth.hpp,
// synth_kernel.hpp), and the path pairing with per-ray range rates (sonar/sim/paths.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

#include "sonar/sim/paths.hpp"
#include "sonar/sim/synth.hpp"
#include "sonar/sim/synth_kernel.hpp"

using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kC = 1500.0;
constexpr double kF0 = 8000.0;
constexpr double kDuration = 0.02;

sim::SynthElement element(const Vec3d& position, const Vec3d& velocity) {
  sim::SynthElement e;
  e.position = position;
  e.velocity = velocity;
  e.axes = element_axes(Vec3d{0.0, 1.0, 0.0});
  return e;  // beamwidths NaN: omnidirectional
}

/// A rectangular CW ping (type 0), or the same as a sampled pulse (type 3).
sim::SynthParams params(int type, bool baseband) {
  sim::SynthParams p;
  p.waveform.type = type;
  p.waveform.f_start = p.waveform.f_stop = kF0;
  p.waveform.duration = kDuration;
  p.sound_speed = kC;
  p.sample_rate = baseband ? 16000.0 : 48000.0;
  p.start_time = 0.5;
  p.n_samples = static_cast<std::int64_t>(0.06 * p.sample_rate);
  p.complex_baseband = baseband;
  p.center_frequency = kF0;
  p.tx = element(Vec3d{0.0, 0.0, 0.0}, Vec3d{1.0, 2.0, 0.0});
  return p;
}

/// An envelope table of `bands` equal parts of 1 and their sum.
std::vector<double> ones_table(std::int64_t n_fine, int bands) {
  std::vector<double> t(static_cast<std::size_t>(2 * n_fine * (bands + 1)), 0.0);
  for (std::int64_t m = 0; m < n_fine; ++m) {
    for (int i = 0; i < bands; ++i) t[static_cast<std::size_t>(2 * (m * (bands + 1) + i))] = 1.0 / bands;
    t[static_cast<std::size_t>(2 * (m * (bands + 1) + bands))] = 1.0;
  }
  return t;
}

void attach(sim::SynthParams& p, const std::vector<double>& table, std::int64_t n_fine, int bands) {
  p.envelope.data = table.data();
  p.envelope.n_fine = n_fine;
  p.envelope.n_bands = bands;
  p.envelope.rate = static_cast<double>(n_fine - 6) / kDuration;
  p.envelope.offset = 2.0;
  p.envelope.carrier = kF0;
  for (int i = 0; i < bands; ++i) p.envelope.knot_frequency[i] = kF0;
}

std::vector<sim::SynthPath> two_paths() {
  sim::SynthPath a;
  a.q = a.p = Vec3d{0.0, 400.0, 5.0};
  a.q_velocity = a.p_velocity = Vec3d{0.0, -12.0, 0.0};
  sim::SynthPath b;
  b.q = Vec3d{20.0, 395.0, -30.0};
  b.p = Vec3d{20.0, 395.0, 40.0};
  b.q_velocity = Vec3d{0.3, 0.0, 0.0};
  b.amp_re = 0.2F;
  b.amp_im = -0.7F;
  return {a, b};
}

}  // namespace

TEST(Synth, ArrivalTimePassesConvergeToTheClosedForm) {
  // A point closing at v on a still, co-located transmitter and receiver: sound emitted at
  // 0 returns at 2 r0 / (c + v). What n passes leave outbound is about (r0 / c)(v / c)^n.
  const double r0 = 400.0, v = 12.0;
  const sim::SynthElement at_origin = element(Vec3d{}, Vec3d{});
  sim::SynthPath path;
  path.q = path.p = Vec3d{0.0, r0, 0.0};
  path.q_velocity = path.p_velocity = Vec3d{0.0, -v, 0.0};
  const double exact = 2.0 * r0 / (kC + v);
  double r_out = 0.0, r_back = 0.0;
  const double t_default = sim::arrival_time(at_origin, at_origin, path, kC, 0.0, r_out, r_back);
  const double t34 = sim::arrival_time(at_origin, at_origin, path, kC, 0.0, r_out, r_back, Vec3d{}, 3, 4);
  EXPECT_EQ(t_default, t34);  // the defaults are (3, 4)
  const double left = (r0 / kC) * std::pow(v / kC, 3);  // 0.137 us
  EXPECT_NEAR(std::abs(t34 - exact), left, 0.1 * left);
  const double t44 = sim::arrival_time(at_origin, at_origin, path, kC, 0.0, r_out, r_back, Vec3d{}, 4, 4);
  EXPECT_NEAR(std::abs(t44 - exact), left * v / kC, 0.2 * left * v / kC);
  const double t88 = sim::arrival_time(at_origin, at_origin, path, kC, 0.0, r_out, r_back, Vec3d{}, 8, 8);
  EXPECT_NEAR(t88, exact, 1e-15);
}

TEST(Synth, SampledPulseWithAConstantEnvelopeIsTheAnalyticTone) {
  // b(u) = 1: the sampled branch gives exp(2 pi i F u(t)), which is the analytic kernel's
  // rectangular CW pulse. Same pass count on both.
  const std::vector<sim::SynthElement> rx{element(Vec3d{0.1, 0.0, 0.0}, Vec3d{1.0, 2.0, 0.0}),
                                         element(Vec3d{-0.4, 0.0, 0.2}, Vec3d{1.0, 2.0, 0.1})};
  const auto paths = two_paths();
  for (const bool baseband : {true, false}) {
    for (const int bands : {1, 4}) {
      sim::SynthParams cw = params(0, baseband);
      cw.arrival_passes_out = cw.arrival_passes_back = sim::kSampledArrivalPasses;
      cw.platform_acceleration = Vec3d{0.2, 0.0, 0.1};
      const std::size_t n = rx.size() * static_cast<std::size_t>(cw.n_samples) * (baseband ? 2U : 1U);
      std::vector<float> ref(n, 0.0F), out(n, 0.0F);
      sim::synthesize(Backend::Host, cw, rx, paths, ref.data());

      sim::SynthParams sampled = cw;
      sampled.waveform.type = 3;
      const std::int64_t n_fine = 1000;
      const auto table = ones_table(n_fine, bands);
      attach(sampled, table, n_fine, bands);
      sim::synthesize(Backend::Host, sampled, rx, paths, out.data());

      double peak = 0.0, worst = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        peak = std::max(peak, static_cast<double>(std::abs(ref[i])));
        worst = std::max(worst, static_cast<double>(std::abs(ref[i] - out[i])));
      }
      EXPECT_GT(peak, 1e-6);
      EXPECT_LT(worst, 5e-7 * peak) << "baseband " << baseband << ", bands " << bands;
    }
  }
}

TEST(Synth, SampledPulseReadsTheEnvelopeAtTheEmissionTime) {
  // A still geometry and an envelope b(u) = u / T (a ramp, which the cubic reads exactly):
  // sample n of the baseband record is (u / T) exp(2 pi i F (u - t)) / (r_out r_back),
  // u = t - (r_out + r_back) / c.
  sim::SynthParams p = params(3, true);
  p.tx = element(Vec3d{}, Vec3d{});
  const std::vector<sim::SynthElement> rx{element(Vec3d{0.5, 0.0, 0.0}, Vec3d{})};
  sim::SynthPath path;
  path.q = path.p = Vec3d{30.0, 400.0, -20.0};
  const std::int64_t n_fine = 806;
  std::vector<double> table(static_cast<std::size_t>(2 * n_fine * 2), 0.0);
  for (std::int64_t m = 0; m < n_fine; ++m) {
    const double u = (static_cast<double>(m) - 2.0) / 800.0;  // u / T at fine sample m
    table[static_cast<std::size_t>(4 * m)] = table[static_cast<std::size_t>(4 * m + 2)] = u;
  }
  attach(p, table, n_fine, 1);
  std::vector<float> out(static_cast<std::size_t>(2 * p.n_samples), 0.0F);
  sim::synthesize(Backend::Host, p, rx, {path}, out.data());
  const double r_out = norm(path.q), r_back = norm(path.p - rx[0].position);
  int inside = 0;
  for (std::int64_t n = 0; n < p.n_samples; ++n) {
    const double t = p.start_time + static_cast<double>(n) / p.sample_rate;
    const double u = t - (r_out + r_back) / kC;
    std::complex<double> expected{0.0, 0.0};
    if (u >= 0.0 && u <= kDuration) {
      expected = std::polar(u / kDuration / (r_out * r_back), 2.0 * M_PI * kF0 * (u - t));
      ++inside;
    } else if (std::abs(u) < 1e-9 || std::abs(u - kDuration) < 1e-9) {
      continue;  // a sample on the pulse's edge may fall either side
    }
    EXPECT_NEAR(out[static_cast<std::size_t>(2 * n)], expected.real(), 2e-12) << n;
    EXPECT_NEAR(out[static_cast<std::size_t>(2 * n + 1)], expected.imag(), 2e-12) << n;
  }
  EXPECT_GT(inside, 300);
}

TEST(Synth, SampledPulseNeedsItsTable) {
  sim::SynthParams p = params(3, true);
  const std::vector<sim::SynthElement> rx{element(Vec3d{}, Vec3d{})};
  std::vector<float> out(static_cast<std::size_t>(2 * p.n_samples), 0.0F);
  EXPECT_THROW(sim::synthesize(Backend::Host, p, rx, two_paths(), out.data()), std::invalid_argument);
  const auto table = ones_table(100, 1);
  attach(p, table, 100, 1);
  EXPECT_NO_THROW(sim::synthesize(Backend::Host, p, rx, two_paths(), out.data()));
  p.envelope.n_bands = sim::kMaxEnvelopeBands + 1;
  EXPECT_THROW(sim::synthesize(Backend::Host, p, rx, two_paths(), out.data()), std::invalid_argument);
  p.envelope.n_bands = 1;
  p.arrival_passes_out = 0;
  EXPECT_THROW(sim::synthesize(Backend::Host, p, rx, two_paths(), out.data()), std::invalid_argument);
}

TEST(Synth, CudaAndHostAgreeOnASampledPulse) {
  if (!cuda_available()) GTEST_SKIP() << "no CUDA device";
  const std::vector<sim::SynthElement> rx{element(Vec3d{0.1, 0.0, 0.0}, Vec3d{1.0, 2.0, 0.0}),
                                         element(Vec3d{-0.4, 0.0, 0.2}, Vec3d{1.0, 2.0, 0.1})};
  const auto paths = two_paths();
  for (const bool baseband : {true, false}) {
    sim::SynthParams p = params(3, baseband);
    p.arrival_passes_out = p.arrival_passes_back = sim::kSampledArrivalPasses;
    const std::int64_t n_fine = 1000;
    const int bands = 4;
    auto table = ones_table(n_fine, bands);
    // An envelope with structure, and absorption that makes the gain vary across the band.
    for (std::int64_t m = 0; m < n_fine; ++m) {
      for (int i = 0; i <= bands; ++i) {
        const double x = static_cast<double>(m) / static_cast<double>(n_fine);
        const double scale = i == bands ? 1.0 : 1.0 / bands;
        table[static_cast<std::size_t>(2 * (m * (bands + 1) + i))] = scale * std::cos(9.0 * x);
        table[static_cast<std::size_t>(2 * (m * (bands + 1) + i) + 1)] = scale * std::sin(5.0 * x);
      }
    }
    attach(p, table, n_fine, bands);
    for (int i = 0; i < bands; ++i) p.envelope.knot_frequency[i] = 6000.0 + 1000.0 * i;
    p.absorption_f_min = 6000.0;
    p.absorption_f_max = 10000.0;
    for (int i = 0; i < sim::kAbsorptionTableSize; ++i) p.absorption_db_per_m[i] = 0.002 + 0.0005 * i;
    const std::size_t n = rx.size() * static_cast<std::size_t>(p.n_samples) * (baseband ? 2U : 1U);
    std::vector<float> host(n, 0.0F), device(n, 0.0F);
    sim::synthesize(Backend::Host, p, rx, paths, host.data());
    sim::synthesize(Backend::Cuda, p, rx, paths, device.data());
    double peak = 0.0, worst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      peak = std::max(peak, static_cast<double>(std::abs(host[i])));
      worst = std::max(worst, static_cast<double>(std::abs(host[i] - device[i])));
    }
    EXPECT_GT(peak, 1e-7);
    EXPECT_LT(worst, 1e-6 * peak) << "baseband " << baseband;
  }
}

TEST(Paths, TwoWayPathsCarryARangeRatePerRay) {
  sim::ImageSourceConfig cfg;  // direct and surface image
  cfg.max_order = 1;
  const Vec3d tx{0.0, 0.0, 10.0}, rx{0.0, 1.0, 10.0};
  const std::vector<double> points{30.0, 0.0, 20.0, 50.0, 5.0, 15.0};
  const auto out = sim::image_source_eigenrays(cfg, tx, points.data(), 2);
  const auto back = sim::image_source_eigenrays(cfg, rx, points.data(), 2);
  ASSERT_EQ(out.n_rays, 2);
  const std::vector<std::complex<double>> amplitude{{2.0, 0.0}, {1.0, 0.0}};
  // The surface-bounce ray of each leg lengthens: 0.5 m/s out, -0.25 m/s back.
  const std::vector<double> rate_out{0.0, 0.5, 0.0, 0.5}, rate_back{0.0, -0.25, 0.0, -0.25};
  const std::vector<double> velocity{1.0, 0.0, 0.0, 0.0, 2.0, 0.0};
  const std::size_t room = 8;
  std::vector<double> q(3 * room), p(3 * room), vq(3 * room, 99.0), vp(3 * room, 99.0);
  std::vector<std::complex<double>> amp(room);

  const std::int64_t kept = sim::two_way_paths_doppler(
      sim::view(out), sim::view(back), tx, rx, amplitude.data(), velocity.data(), nullptr, nullptr,
      sim::RangeRates{rate_out.data(), rate_back.data()}, q.data(), p.data(), amp.data(), vq.data(), vp.data());
  ASSERT_EQ(kept, 8);
  // The positions and amplitudes are those of the plain pairing.
  const auto plain = sim::two_way_paths(sim::view(out), sim::view(back), tx, rx, amplitude.data(), velocity.data(),
                                        nullptr, nullptr);
  ASSERT_EQ(plain.size(), 8);
  for (std::size_t i = 0; i < 3 * room; ++i) {
    EXPECT_EQ(q[i], plain.q[i]);
    EXPECT_EQ(p[i], plain.p[i]);
  }
  // Order: scatterer, outbound ray, return ray.
  for (std::int64_t i = 0; i < 2; ++i) {
    for (std::int64_t a = 0; a < 2; ++a) {
      for (std::int64_t b = 0; b < 2; ++b) {
        const auto at = static_cast<std::size_t>(i * 4 + a * 2 + b);
        EXPECT_EQ(amp[at], plain.amplitude[at]);
        for (std::size_t x = 0; x < 3; ++x) {
          const double v = velocity[static_cast<std::size_t>(3 * i) + x];
          const double launch_out = out.launch[static_cast<std::size_t>(3 * (i * 2 + a)) + x];
          const double launch_back = back.launch[static_cast<std::size_t>(3 * (i * 2 + b)) + x];
          EXPECT_DOUBLE_EQ(vq[3 * at + x], v + rate_out[static_cast<std::size_t>(i * 2 + a)] * launch_out);
          EXPECT_DOUBLE_EQ(vp[3 * at + x], v + rate_back[static_cast<std::size_t>(i * 2 + b)] * launch_back);
        }
      }
    }
  }
  // Direct-direct pair of the first scatterer: no rate, the scatterer's own velocity.
  EXPECT_EQ(vq[0], 1.0);
  EXPECT_EQ(vp[0], 1.0);

  // One leg only, and no scatterer velocities: the other end stays still.
  const std::int64_t kept_out = sim::two_way_paths_doppler(
      sim::view(out), sim::view(back), tx, rx, amplitude.data(), nullptr, nullptr, nullptr,
      sim::RangeRates{rate_out.data(), nullptr}, q.data(), p.data(), amp.data(), vq.data(), vp.data());
  ASSERT_EQ(kept_out, 8);
  for (std::size_t i = 0; i < 3 * room; ++i) EXPECT_EQ(vp[i], 0.0);
  EXPECT_DOUBLE_EQ(vq[3 * 2 + 2], 0.5 * out.launch[3 * 1 + 2]);  // scatterer 0, bounce ray out
  // The speed of the apparent position along the ray is the range rate.
  EXPECT_NEAR(std::sqrt(vq[6] * vq[6] + vq[7] * vq[7] + vq[8] * vq[8]), 0.5, 1e-15);

  EXPECT_THROW(sim::two_way_paths_doppler(sim::view(out), sim::view(back), tx, rx, amplitude.data(), nullptr, nullptr,
                                          nullptr, sim::RangeRates{}, q.data(), p.data(), amp.data(), vq.data(), nullptr),
               std::invalid_argument);
}
