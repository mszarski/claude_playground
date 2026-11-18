// The batched (device) formulations run on the host through HostExec, against the reference
// host code: far-field and MVDR beamformers (sonar/live/beamform_batched.hpp) and the
// reverberation generator (sonar/sim/reverb_batched.hpp). The CUDA backend runs the same
// sequences and kernels; tests/python/test_batched.py holds the same comparisons on simulated
// pings and, where a device is present, against the device.
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sonar/dsp/exec_host.hpp"
#include "sonar/dsp/fft.hpp"
#include "sonar/live/beamform.hpp"
#include "sonar/live/beamform_kernel.hpp"
#include "sonar/sim/reverb.hpp"
#include "sonar/sim/reverb_batched.hpp"

using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kPi = 3.14159265358979323846;
using Cf = std::complex<float>;

// A conditioned record of a 12-element line array with noise in it and a far-field noise
// source at 20 degrees (the same noise on every channel, each with its plane-wave lead).
PingRecord noisy_record(std::int64_t n_samples = 1500) {
  const std::int64_t n_el = 12;
  const double c = 1500.0, fs = 8000.0, fc = 8000.0, pitch = 0.09;
  PingRecord r;
  r.samples.resize(n_el, n_samples, SampleKind::ComplexBaseband);
  r.samples.sample_rate = fs;
  r.samples.center_frequency = fc;
  r.samples.start_time = 0.6;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 5000.0;
  r.transmit.waveform.f_stop = 11000.0;
  r.transmit.waveform.duration = 0.01;
  r.environment.sound_speed = c;
  for (std::int64_t e = 0; e < n_el; ++e) {
    Element el;
    el.lever_arm = Vec3d{(static_cast<double>(e) - 0.5 * static_cast<double>(n_el - 1)) * pitch, 0.0, 0.0};
    r.geometry.rx.push_back(el);
  }
  set_default_channel_tables(r);
  std::mt19937 gen(7);
  std::normal_distribution<float> g(0.0F, 1.0F);
  for (float& v : r.samples.data) v = g(gen);
  std::vector<Cf> jam(static_cast<std::size_t>(n_samples));
  for (Cf& v : jam) v = Cf(5.0F * g(gen), 5.0F * g(gen));
  dsp::fft(jam, false);
  const double sine = std::sin(20.0 * kPi / 180.0);
  for (std::int64_t e = 0; e < n_el; ++e) {
    const double lead = r.geometry.rx[static_cast<std::size_t>(e)].lever_arm.x * sine / c;
    std::vector<Cf> x = jam;
    for (std::int64_t k = 0; k < n_samples; ++k) {
      const double f = fc + static_cast<double>(live::beam_signed_bin(k, n_samples)) * fs / static_cast<double>(n_samples);
      x[static_cast<std::size_t>(k)] *= Cf(std::polar(1.0, 2.0 * kPi * f * lead));
    }
    dsp::fft(x, true);
    Cf* row = r.samples.channel_complex(e);
    for (std::int64_t k = 0; k < n_samples; ++k) row[k] += x[static_cast<std::size_t>(k)];
  }
  return r;
}

std::vector<double> bearings() {
  std::vector<double> b;
  for (int d = -40; d <= 40; d += 3) b.push_back(d * kPi / 180.0);   // 27: not a multiple of the apply group
  return b;
}

// Largest difference over the largest reference value.
template <class T>
double worst(const std::vector<T>& got, const std::vector<T>& want) {
  double diff = 0.0, peak = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    diff = std::max(diff, static_cast<double>(std::abs(got[i] - want[i])));
    peak = std::max(peak, static_cast<double>(std::abs(want[i])));
  }
  return diff / peak;
}

TEST(Batched, FarfieldMatchesTheReference) {
  const PingRecord r = noisy_record();
  for (const char* window : {"rect", "hann"}) {
    const live::BeamMap want = live::farfield_beamform(r, bearings(), window);
    const live::BeamMap got = live::farfield_beamform(r, bearings(), window, Backend::Host, 0, true);
    ASSERT_EQ(got.data.size(), want.data.size());
    EXPECT_LT(worst(got.data, want.data), 1e-5);
    EXPECT_EQ(got.n_valid, want.n_valid);
  }
}

TEST(Batched, MvdrMatchesTheReference) {
  const PingRecord r = noisy_record();
  for (const std::int64_t block : {128, 200}) {          // 200: the last block is short
    live::MvdrOptions o;
    o.n_subbands = 6;
    o.block = block;
    o.guard_blocks = block == 200 ? 2 : 1;
    const live::BeamMap want = live::mvdr_beamform(r, bearings(), o);
    const live::BeamMap got = live::mvdr_beamform(r, bearings(), o, Backend::Host, 0, true);
    ASSERT_EQ(got.data.size(), want.data.size());
    EXPECT_LT(worst(got.data, want.data), 1e-4);
  }
}

TEST(Batched, MvdrKernelsSolveTheLoadedCovarianceAndPassTheLookDirection) {
  // The factor and solve kernels by themselves, against the definition: with
  // R = (sum of the other blocks' x x^H) / count + loading tr(R) / N, the solved vector z
  // satisfies R z = a, and the weights w = z / (a^H z) have unit response to a: sum_e conj(w_e) a_e = 1.
  using Cd = std::complex<double>;
  const std::int64_t n_el = 9, n_blocks = 4, block = 20, n = 73, n_b = 3;   // the last block has 13 samples
  const double loading = 0.02, f_abs = 9000.0;
  std::mt19937 gen(11);
  std::normal_distribution<double> g(0.0, 1.0);
  std::vector<float> xt(static_cast<std::size_t>(2 * n * n_el));
  for (float& v : xt) v = static_cast<float>(g(gen));
  std::vector<double> tau(static_cast<std::size_t>(n_b * n_el));
  for (double& v : tau) v = 1e-4 * g(gen);
  const auto nn = static_cast<std::size_t>(n_el * n_el);
  std::vector<double> outer(2 * n_blocks * nn), total(2 * nn), r(2 * n_blocks * nn), work(2 * n_blocks * n_b * n_el), f{f_abs};
  std::vector<float> coeff(2 * n_blocks * n_b * n_el);
  std::vector<std::int32_t> status(n_blocks), singular(n_blocks * n_b);
  const live::MvdrOuterKernel k_outer{xt.data(), outer.data(), n, n_el, n_blocks, block};
  for (std::int64_t i = 0; i < n_blocks * n_el * n_el; ++i) k_outer(i);
  const live::MvdrTotalKernel k_total{outer.data(), total.data(), n_el, n_blocks};
  for (std::int64_t i = 0; i < n_el * n_el; ++i) k_total(i);
  const live::MvdrFactorKernel k_factor{outer.data(), total.data(), r.data(), status.data(), n, n_el, n_blocks, block, 0, loading};
  for (std::int64_t i = 0; i < n_blocks; ++i) k_factor(i);
  const live::MvdrSolveKernel k_solve{r.data(), status.data(), tau.data(), f.data(), work.data(), coeff.data(), singular.data(),
                                      n_el, n_blocks, n_b};
  for (std::int64_t i = 0; i < n_blocks * n_b; ++i) k_solve(i);
  const auto x = [&](std::int64_t t, std::int64_t e) {
    return Cd(xt[static_cast<std::size_t>(2 * (t * n_el + e))], xt[static_cast<std::size_t>(2 * (t * n_el + e) + 1)]);
  };
  for (std::int64_t i = 0; i < n_blocks; ++i) {
    ASSERT_EQ(status[static_cast<std::size_t>(i)], 0);
    // R of block i, from the definition.
    std::vector<Cd> cov(nn);
    std::int64_t count = 0;
    for (std::int64_t t = 0; t < n; ++t) {
      if (t / block == i) continue;
      ++count;
      for (std::int64_t p = 0; p < n_el; ++p) {
        for (std::int64_t q = 0; q < n_el; ++q) cov[static_cast<std::size_t>(p * n_el + q)] += x(t, p) * std::conj(x(t, q));
      }
    }
    double trace = 0.0;
    for (Cd& v : cov) v /= static_cast<double>(count);
    for (std::int64_t p = 0; p < n_el; ++p) trace += cov[static_cast<std::size_t>(p * n_el + p)].real();
    for (std::int64_t p = 0; p < n_el; ++p) cov[static_cast<std::size_t>(p * n_el + p)] += loading * trace / static_cast<double>(n_el);
    for (std::int64_t b = 0; b < n_b; ++b) {
      ASSERT_EQ(singular[static_cast<std::size_t>(i * n_b + b)], 0);
      const double* z = work.data() + 2 * (i * n_b + b) * n_el;
      const float* cf = coeff.data() + 2 * (i * n_b + b) * n_el;
      Cd response = 0.0;
      for (std::int64_t p = 0; p < n_el; ++p) {
        const Cd a = std::polar(1.0, 2.0 * kPi * f_abs * tau[static_cast<std::size_t>(b * n_el + p)]);
        Cd rz = 0.0;
        for (std::int64_t q = 0; q < n_el; ++q) rz += cov[static_cast<std::size_t>(p * n_el + q)] * Cd(z[2 * q], z[2 * q + 1]);
        EXPECT_LT(std::abs(rz - a), 1e-10) << "block " << i << " bearing " << b << " element " << p;
        response += Cd(cf[2 * p], cf[2 * p + 1]) * a;                   // n_el conj(w_p) a_p
      }
      EXPECT_LT(std::abs(response - static_cast<double>(n_el)), 1e-5);
    }
  }
}

TEST(Batched, MvdrRefusesSilenceWithoutLoading) {
  PingRecord r = noisy_record();
  std::fill(r.samples.data.begin(), r.samples.data.end(), 0.0F);
  live::MvdrOptions o;
  o.loading = 0.0;
  EXPECT_THROW(live::mvdr_beamform(r, bearings(), o, Backend::Host, 0, true), std::runtime_error);
}

struct ReverbCase {
  sim::ReverbParams params;
  std::vector<double> coarse, density, direction, doppler, normals, offsets;
  std::vector<std::complex<double>> pulse;
  std::int64_t n_rows = 0, n_channels = 0;
  sim::ReverbPower power(bool components) const {
    sim::ReverbPower p{coarse.data(), static_cast<std::int64_t>(coarse.size()), density.data(), n_rows};
    if (components) {
      p.direction = direction.data();
      p.doppler = doppler.data();
    }
    return p;
  }
};

ReverbCase reverb_case(std::int64_t n_channels, bool uniform) {
  ReverbCase c;
  c.n_rows = 24;
  c.n_channels = n_channels;
  c.params.sample_rate = 8000.0;
  c.params.n_samples = 900;
  c.params.time0 = 0.02;
  c.params.start_time = 0.025;
  c.params.center_frequency = 20e3;
  c.params.n_azimuth = c.n_rows;
  c.params.decimation = 3;
  c.params.n_low = 331;                       // a prime
  c.params.velocity = Vec3d{1.5, -0.5, 0.0};
  std::mt19937 gen(3);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::normal_distribution<double> g(0.0, 1.0);
  for (int k = 0; k < 20; ++k) c.coarse.push_back(0.01 * k);
  for (std::int64_t r = 0; r < c.n_rows; ++r) {
    for (int k = 0; k < 20; ++k) c.density.push_back(0.5 + u(gen));
    const double az = 2.0 * kPi * u(gen), el = 0.5 * (u(gen) - 0.5);
    c.direction.insert(c.direction.end(), {std::cos(az) * std::cos(el), std::sin(az) * std::cos(el), std::sin(el)});
    c.doppler.push_back(r == 2 ? 0.0 : 60.0 * (u(gen) - 0.5));
  }
  for (std::int64_t i = 0; i < c.n_rows * 2 * c.params.n_low; ++i) c.normals.push_back(g(gen));
  for (std::int64_t ch = 0; ch < n_channels; ++ch) {
    if (uniform) {
      c.offsets.insert(c.offsets.end(), {0.05 * static_cast<double>(ch) - 0.1, 0.0, 0.0});
    } else {
      c.offsets.insert(c.offsets.end(), {0.3 * g(gen), 0.3 * g(gen), 0.1 * g(gen)});
    }
  }
  for (int k = 0; k < 25; ++k) c.pulse.push_back(std::polar(1.0, 2.0 * kPi * u(gen)));
  return c;
}

void expect_reverb_agrees(const ReverbCase& c, bool components, bool carrier, bool real_output) {
  sim::ReverbParams p = c.params;
  p.carrier = carrier;
  p.real_output = real_output;
  const sim::ReverbPower power = c.power(components);
  const std::size_t n_out = static_cast<std::size_t>(c.n_channels * p.n_samples) * (real_output ? 1 : 2);
  std::vector<float> want(n_out), got(n_out);
  sim::reverb_series(Backend::Host, p, power, c.normals.data(), c.n_rows, c.pulse.data(),
                     static_cast<std::int64_t>(c.pulse.size()), c.offsets.data(), c.n_channels, want.data());
  p.batched = true;
  sim::reverb_series(Backend::Host, p, power, c.normals.data(), c.n_rows, c.pulse.data(),
                     static_cast<std::int64_t>(c.pulse.size()), c.offsets.data(), c.n_channels, got.data());
  EXPECT_LT(worst(got, want), 5e-6) << "components " << components << " carrier " << carrier << " real " << real_output;
}

TEST(Batched, ReverberationMatchesTheReference) {
  for (const bool uniform : {false, true}) {
    const ReverbCase c = reverb_case(uniform ? 6 : 5, uniform);
    ASSERT_EQ(sim::reverb_uniform_line(c.offsets.data(), c.n_channels), uniform);
    expect_reverb_agrees(c, true, false, false);    // components: a direction and a Doppler shift per row
    expect_reverb_agrees(c, false, false, false);   // azimuth sectors, the sonar's own speed
    expect_reverb_agrees(c, true, true, false);
    expect_reverb_agrees(c, false, true, true);
  }
}

TEST(Batched, NamedRowsGiveTheSameSpectrumInAnyRuns) {
  const ReverbCase c = reverb_case(5, false);
  const sim::ReverbPower power = c.power(true);
  const std::int64_t n_low = c.params.n_low;
  std::vector<float> all(static_cast<std::size_t>(2 * c.n_channels * n_low), 0.0F), runs = all;
  sim::reverb_accumulate(Backend::Host, c.params, power, 0, c.normals.data(), c.n_rows, c.pulse.data(),
                         static_cast<std::int64_t>(c.pulse.size()), c.offsets.data(), c.n_channels, all.data());
  for (std::int64_t first = 0; first < c.n_rows; first += 7) {
    std::vector<std::int64_t> rows;
    for (std::int64_t r = first; r < std::min(first + 7, c.n_rows); ++r) rows.push_back(r);
    sim::reverb_accumulate(Backend::Host, c.params, power, -1, c.normals.data() + first * 2 * n_low,
                           static_cast<std::int64_t>(rows.size()), c.pulse.data(), static_cast<std::int64_t>(c.pulse.size()),
                           c.offsets.data(), c.n_channels, runs.data(), 0, rows.data());
  }
  EXPECT_EQ(all, runs);                      // the reference code: bit for bit
  std::vector<std::int64_t> bad{0, c.n_rows};
  EXPECT_THROW(sim::reverb_accumulate(Backend::Host, c.params, power, 0, c.normals.data(), 2, c.pulse.data(),
                                      static_cast<std::int64_t>(c.pulse.size()), c.offsets.data(), c.n_channels, runs.data(),
                                      0, bad.data()),
               std::invalid_argument);
}

TEST(Batched, PhasorKeepsItsAccuracyAtLargePhases) {
  // 1e6 turns and a fraction: a single-precision angle would be off by a radian.
  for (const double turns : {0.125, 1234567.125, -98765.875}) {
    float c = 0.0F, s = 0.0F;
    sim::reverb_phasor(turns, c, s);
    EXPECT_NEAR(c, std::cos(2.0 * kPi * 0.125), 1e-6);
    EXPECT_NEAR(s, std::sin(2.0 * kPi * 0.125), 1e-6);
    live::beam_phasor(turns, c, s);
    EXPECT_NEAR(c, std::cos(2.0 * kPi * 0.125), 1e-6);
  }
}

TEST(Batched, UnscaledTransformIsTheScaledOneTimesN) {
  std::vector<Cf> a(37), b;
  std::mt19937 gen(1);
  std::normal_distribution<float> g(0.0F, 1.0F);
  for (Cf& v : a) v = Cf(g(gen), g(gen));
  b = a;
  dsp::fft(a, true);
  dsp::fft_unscaled(b.data(), b.size(), true);
  for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(std::abs(b[i] - 37.0F * a[i]), 0.0, 1e-4);
}

}  // namespace
