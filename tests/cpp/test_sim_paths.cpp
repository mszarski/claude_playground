// Analytic cases for the eigenray providers (sonar/sim/paths.hpp) and the reverberation
// generator (sonar/sim/reverb.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "sonar/sim/paths.hpp"
#include "sonar/sim/reverb.hpp"

using sonar::Backend;
using sonar::Vec3d;
namespace sim = sonar::sim;

namespace {
constexpr double kC = 1500.0;
constexpr double kPi = 3.14159265358979323846;
}  // namespace

TEST(Paths, FreeFieldDelayIsDistanceOverSoundSpeed) {
  const Vec3d source{1.0, -2.0, 3.0};
  const std::vector<double> points{4.0, 2.0, 3.0,     // 5 m away, level
                                   1.0, -2.0, 103.0,  // 100 m straight down
                                   -11.0, 7.0, 11.0}; // 17 m
  const auto e = sim::free_field_eigenrays(kC, source, points.data(), 3);
  ASSERT_EQ(e.n_points, 3);
  ASSERT_EQ(e.n_rays, 1);
  const double range[3] = {5.0, 100.0, 17.0};
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(e.delay[i], range[i] / kC, 1e-15);
    EXPECT_NEAR(e.amplitude[i].real(), 1.0 / range[i], 1e-15);
    EXPECT_EQ(e.amplitude[i].imag(), 0.0);
    EXPECT_EQ(e.valid[i], 1);
    double norm2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      norm2 += e.launch[3 * i + a] * e.launch[3 * i + a];
      EXPECT_EQ(e.launch[3 * i + a], e.arrival[3 * i + a]);
    }
    EXPECT_NEAR(norm2, 1.0, 1e-14);
  }
  EXPECT_NEAR(e.launch[3 * 1 + 2], 1.0, 1e-15);  // straight down
}

TEST(Paths, ImagePathLengthsInASurfaceAndBottomWaveguide) {
  // Source at 20 m, point at 60 m depth and 300 m range, in 100 m of water.
  sim::ImageSourceConfig cfg;
  cfg.sound_speed = kC;
  cfg.has_bottom = true;
  cfg.bottom_z = 100.0;
  cfg.bottom_coefficient = {0.5, 0.0};
  cfg.max_order = 2;
  const Vec3d source{0.0, 0.0, 20.0};
  const std::vector<double> point{300.0, 0.0, 60.0};
  const auto e = sim::image_source_eigenrays(cfg, source, point.data(), 1);
  // Direct, surface, surface-bottom, bottom, bottom-surface.
  ASSERT_EQ(e.n_rays, 5);
  // Image depths: 60, -60, 2 * 100 + 60 = 260, 2 * 100 - 60 = 140, -140.
  const double image_z[5] = {60.0, -60.0, 260.0, 140.0, -140.0};
  const double coefficient[5] = {1.0, -1.0, -0.5, 0.5, -0.5};
  const int n_surface[5] = {0, 1, 1, 0, 1};
  const int n_bottom[5] = {0, 0, 1, 1, 1};
  for (int j = 0; j < 5; ++j) {
    const double length = std::hypot(300.0, image_z[j] - 20.0);
    EXPECT_NEAR(e.delay[j], length / kC, 1e-15) << j;
    EXPECT_NEAR(e.amplitude[j].real(), coefficient[j] / length, 1e-15) << j;
    EXPECT_EQ(e.n_surface[j], n_surface[j]);
    EXPECT_EQ(e.n_bottom[j], n_bottom[j]);
    // An odd number of bounces turns the vertical direction of travel around.
    const double sign = ((n_surface[j] + n_bottom[j]) % 2 == 1) ? -1.0 : 1.0;
    EXPECT_DOUBLE_EQ(e.arrival[3 * j + 2], sign * e.launch[3 * j + 2]);
    EXPECT_DOUBLE_EQ(e.arrival[3 * j], e.launch[3 * j]);
  }
}

TEST(Paths, ImageSeriesWithoutABottomStopsAtTheSurface) {
  sim::ImageSourceConfig cfg;  // surface only
  cfg.sound_speed = kC;
  cfg.max_order = 3;
  EXPECT_EQ(sim::image_series(cfg).size(), 2U);
  cfg.walls.push_back({1, 50.0, {0.8, 0.0}});
  const auto series = sim::image_series(cfg);
  ASSERT_EQ(series.size(), 4U);  // each vertical image, then its wall image
  EXPECT_EQ(series[1].wall, 0);
  const Vec3d source{0.0, 0.0, 10.0};
  const std::vector<double> point{40.0, 20.0, 10.0};
  const auto e = sim::image_source_eigenrays(cfg, source, point.data(), 1);
  // The wall image of the direct path is at y = 2 * 50 - 20 = 80.
  EXPECT_NEAR(e.delay[1], std::hypot(40.0, 80.0) / kC, 1e-15);
  EXPECT_NEAR(e.amplitude[1].real(), 0.8 / std::hypot(40.0, 80.0), 1e-15);
}

TEST(Paths, RayleighCoefficientLimits) {
  const sim::FluidBottom sand{1700.0, 1.9, 0.0, 1.0};
  // Normal incidence: (Z2 - Z1) / (Z2 + Z1).
  const double z1 = kC, z2 = 1.9 * 1700.0;
  const auto normal = sim::rayleigh_reflection(0.5 * kPi, kC, sand);
  EXPECT_NEAR(normal.real(), (z2 - z1) / (z2 + z1), 1e-12);
  EXPECT_NEAR(normal.imag(), 0.0, 1e-12);
  // Below the critical angle acos(c_w / c_b) a lossless bottom reflects everything.
  const double critical = std::acos(kC / 1700.0);
  EXPECT_NEAR(std::abs(sim::rayleigh_reflection(0.5 * critical, kC, sand)), 1.0, 1e-12);
  EXPECT_LT(std::abs(sim::rayleigh_reflection(1.5 * critical, kC, sand)), 0.9);
}

TEST(Paths, RayleighCoefficientLossyBottom) {
  // Audit B3: the lossy branch. A loss of alpha dB per wavelength is a plane wave whose
  // amplitude falls by exp(-2 pi delta) over one wavelength, so delta = -ln(10^(-alpha/20)) / (2 pi)
  // and the sediment's complex speed is c_b / (1 + i delta) (k = omega (1 + i delta) / c_b).
  // At normal incidence the coefficient is the impedance ratio (Z2 - Z1) / (Z2 + Z1).
  const double alpha = 0.8, rho = 1.9, cb = 1650.0;
  const sim::FluidBottom lossy{cb, rho, alpha, 1.0};
  const double delta = -std::log(std::pow(10.0, -alpha / 20.0)) / (2.0 * kPi);
  const std::complex<double> z2 = rho * cb / std::complex<double>(1.0, delta);
  const std::complex<double> expected = (z2 - kC) / (z2 + kC);
  const auto normal = sim::rayleigh_reflection(0.5 * kPi, kC, lossy);
  EXPECT_NEAR(normal.real(), expected.real(), 1e-12);
  EXPECT_NEAR(normal.imag(), expected.imag(), 1e-12);
  EXPECT_LT(normal.imag(), 0.0);  // a lossy sediment's impedance has a phase: not a real coefficient
  // Below the critical angle the lossless bottom reflects everything; a lossy one does not,
  // and at grazing incidence every fluid bottom reflects -1.
  const sim::FluidBottom lossless{cb, rho, 0.0, 1.0};
  const double critical = std::acos(kC / cb);
  for (const double g : {0.2 * critical, 0.5 * critical, 0.9 * critical}) {
    EXPECT_NEAR(std::abs(sim::rayleigh_reflection(g, kC, lossless)), 1.0, 1e-12);
    const double r = std::abs(sim::rayleigh_reflection(g, kC, lossy));
    EXPECT_LT(r, 1.0 - 1e-4) << g;
    EXPECT_GT(r, 0.5) << g;
  }
  const auto flat = sim::rayleigh_reflection(1e-9, kC, lossy);
  EXPECT_NEAR(flat.real(), -1.0, 1e-6);
  EXPECT_NEAR(flat.imag(), 0.0, 1e-6);
}

TEST(Paths, SnellCrossingObeysSnellsLaw) {
  const double c1 = 1500.0, c2 = 1700.0;
  const Vec3d source{0.0, 0.0, -10.0};
  const std::vector<double> points{12.0, 5.0, 2.0,    // buried 2 m
                                   8.0, 0.0, -3.0};   // in the water
  double time[2];
  double cross[6];
  sim::snell_crossing(source, points.data(), 2, c1, c2, 0.0, time, cross);
  const double dist = std::hypot(12.0, 5.0);
  const double u = std::hypot(cross[0], cross[1]);
  EXPECT_DOUBLE_EQ(cross[2], 0.0);
  const double sin1 = u / std::hypot(u, 10.0);
  const double sin2 = (dist - u) / std::hypot(dist - u, 2.0);
  EXPECT_NEAR(sin1 / c1, sin2 / c2, 1e-12);
  EXPECT_NEAR(time[0], std::hypot(u, 10.0) / c1 + std::hypot(dist - u, 2.0) / c2, 1e-15);
  EXPECT_LT(time[0], std::hypot(dist, 12.0) / c1);  // faster than the straight line at the water speed
  // A point in the water: the straight path, crossing point equal to the point.
  EXPECT_NEAR(time[1], std::hypot(8.0, 7.0) / c1, 1e-15);
  EXPECT_EQ(cross[3], 8.0);
  EXPECT_EQ(cross[5], -3.0);

  const auto e = sim::sediment_eigenrays({c1, c2, 0.0, {0.7, 0.0}}, source, points.data(), 2);
  EXPECT_NEAR(e.amplitude[0].real(), 0.7 / (c1 * time[0]), 1e-15);
  EXPECT_NEAR(e.amplitude[1].real(), 1.0 / std::hypot(8.0, 7.0), 1e-15);
}

TEST(Paths, TwoWayPathsPairEveryOutboundAndReturnRay) {
  sim::ImageSourceConfig cfg;  // direct and surface image
  cfg.sound_speed = kC;
  cfg.max_order = 1;
  const Vec3d tx{0.0, 0.0, 10.0}, rx{0.0, 1.0, 10.0};
  const std::vector<double> points{30.0, 0.0, 20.0, 50.0, 5.0, 15.0};
  const auto out = sim::image_source_eigenrays(cfg, tx, points.data(), 2);
  const auto back = sim::image_source_eigenrays(cfg, rx, points.data(), 2);
  const std::vector<std::complex<double>> amplitude{{2.0, 0.0}, {0.0, 0.0}};  // the second returns nothing
  const auto paths = sim::two_way_paths(sim::view(out), sim::view(back), tx, rx, amplitude.data(), nullptr, nullptr, nullptr);
  ASSERT_EQ(paths.size(), 4);  // 2 x 2 pairs of the first scatterer only
  EXPECT_TRUE(paths.velocity.empty());
  // Direct-direct: q = p = the scatterer, amplitude as given.
  for (int a = 0; a < 3; ++a) {
    EXPECT_NEAR(paths.q[a], points[a], 1e-12);
    EXPECT_NEAR(paths.p[a], points[a], 1e-12);
  }
  EXPECT_NEAR(paths.amplitude[0].real(), 2.0, 1e-12);
  // Direct out, surface back: p is the image above the surface; one bounce, coefficient -1.
  EXPECT_NEAR(paths.p[3 + 2], -20.0, 1e-12);
  EXPECT_NEAR(paths.amplitude[1].real(), -2.0, 1e-12);
  EXPECT_NEAR(paths.amplitude[3].real(), 2.0, 1e-12);  // surface both ways

  // Lambert weighting with sqrt(sin g_in sin g_out), and velocities repeated per pair.
  const std::vector<std::uint8_t> lambert{1, 0};
  const std::vector<double> velocity{1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto weighted =
      sim::two_way_paths(sim::view(out), sim::view(back), tx, rx, amplitude.data(), velocity.data(), lambert.data(), nullptr);
  ASSERT_EQ(weighted.size(), 4);
  const double sin_in = 10.0 / std::hypot(30.0, 10.0);
  const double sin_out = 10.0 / std::sqrt(30.0 * 30.0 + 1.0 + 100.0);
  EXPECT_NEAR(weighted.amplitude[0].real(), 2.0 * std::sqrt(sin_in * sin_out), 1e-12);
  ASSERT_EQ(weighted.velocity.size(), 12U);
  EXPECT_EQ(weighted.velocity[9], 1.0);
}

TEST(Reverb, SigmaInterpolatesThePowerGrid) {
  sim::ReverbParams p;
  p.sample_rate = 1000.0;
  p.n_azimuth = 4;
  p.decimation = 2;
  p.n_low = 8;
  p.time0 = 0.0;
  p.sound_speed = kC;
  const std::vector<double> coarse{0.004, 0.008};
  const std::vector<double> density{2.0, 6.0};
  const sim::ReverbPower power{coarse.data(), 2, density.data(), 1};
  const auto sigma = sim::reverb_sigma(p, power);
  ASSERT_EQ(sigma.size(), 8U);
  EXPECT_EQ(sim::reverb_active_sectors(p, power), 4);
  const std::vector<double> none{0.0, 0.0};
  EXPECT_EQ(sim::reverb_active_sectors(p, {coarse.data(), 2, none.data(), 1}), 0);
  const double scale = (2.0 * kPi / 4.0) / 2.0 / 500.0;
  EXPECT_EQ(sigma[1], 0.0);                              // t = 2 ms: before the grid
  EXPECT_NEAR(sigma[2], std::sqrt(2.0 * scale), 1e-15);  // t = 4 ms
  EXPECT_NEAR(sigma[3], std::sqrt(4.0 * scale), 1e-12);  // t = 6 ms: halfway
  EXPECT_NEAR(sigma[4], std::sqrt(6.0 * scale), 1e-15);
  EXPECT_EQ(sigma[5], 0.0);                              // after the grid
}

TEST(Reverb, SeriesHasThePowerOfItsSectors) {
  // Constant sigma, a one-sample pulse, no decimation: every channel is a sum of
  // n_azimuth independent sequences of variance 2 sigma^2, whatever the element layout.
  sim::ReverbParams p;
  p.sample_rate = 8000.0;
  p.n_samples = 4096;
  p.start_time = 0.0;
  p.time0 = 0.0;
  p.center_frequency = 20e3;
  p.n_azimuth = 12;
  p.n_low = 4096;
  p.sound_speed = kC;
  // sigma = sqrt(density / fs * dphi / 2) = 0.5 throughout.
  const std::vector<double> coarse{-1.0, 1.0};
  const double level = 0.25 * 8000.0 * 2.0 / (2.0 * kPi / 12.0);
  const std::vector<double> density{level, level};
  const sim::ReverbPower power{coarse.data(), 2, density.data(), 1};
  std::mt19937 gen(4);
  std::normal_distribution<double> normal;
  std::vector<double> normals(static_cast<std::size_t>(12 * 2 * 4096));
  for (double& v : normals) v = normal(gen);
  const std::vector<std::complex<double>> pulse{{1.0, 0.0}};
  for (const bool line : {true, false}) {
    std::vector<double> offsets;
    for (int ch = 0; ch < 4; ++ch) {
      offsets.insert(offsets.end(), {0.03 * ch, line ? 0.0 : 0.01 * ch * ch, 0.0});
    }
    EXPECT_EQ(sim::reverb_uniform_line(offsets.data(), 4), line);
    std::vector<float> out(static_cast<std::size_t>(2 * 4 * 4096));
    sim::reverb_series(Backend::Host, p, power, normals.data(), 12, pulse.data(), 1, offsets.data(), 4, out.data());
    for (int ch = 0; ch < 4; ++ch) {
      double power = 0.0;
      for (int k = 0; k < 4096; ++k) {
        const float re = out[2 * (ch * 4096 + k)], im = out[2 * (ch * 4096 + k) + 1];
        power += re * re + im * im;
      }
      EXPECT_NEAR(power / 4096.0, 12 * 2 * 0.25, 0.6) << ch;  // 6 +- 10 %
    }
    // Channel 0 at zero offset is the plain sum of the sectors' sequences.
    double re0 = 0.0;
    for (int m = 0; m < 12; ++m) re0 += 0.5 * normals[static_cast<std::size_t>(m * 2 * 4096 + 7)];
    EXPECT_NEAR(out[2 * 7], re0, 1e-4);
  }
}

TEST(Reverb, RejectsNormalsThatDoNotMatchTheActiveSectors) {
  sim::ReverbParams p;
  p.sample_rate = 1000.0;
  p.n_samples = 16;
  p.n_azimuth = 3;
  p.n_low = 16;
  p.sound_speed = kC;
  const std::vector<double> coarse{-1.0, 1.0}, density{1.0, 1.0}, normals(3 * 2 * 16, 0.0), offsets(3, 0.0);
  const sim::ReverbPower power{coarse.data(), 2, density.data(), 1};
  const std::vector<std::complex<double>> pulse{{1.0, 0.0}};
  std::vector<float> out(2 * 16);
  EXPECT_THROW(sim::reverb_series(Backend::Host, p, power, normals.data(), 2, pulse.data(), 1, offsets.data(), 1,
                                  out.data()),
               std::invalid_argument);
  EXPECT_NO_THROW(sim::reverb_series(Backend::Host, p, power, normals.data(), 3, pulse.data(), 1, offsets.data(), 1,
                                     out.data()));
  // No sound speed is assumed (audit A9): left unset it is refused.
  EXPECT_TRUE(std::isnan(sim::ReverbParams{}.sound_speed));
  sim::ReverbParams unset = p;
  unset.sound_speed = sim::ReverbParams{}.sound_speed;
  EXPECT_THROW(sim::reverb_series(Backend::Host, unset, power, normals.data(), 3, pulse.data(), 1, offsets.data(), 1,
                                  out.data()),
               std::invalid_argument);
}

TEST(Reverb, ComponentsHaveTheirOwnDirectionAndDoppler) {
  // Component mode: a row of the density is a component with an arrival direction (here
  // straight down, which no azimuth sector can be) and a frequency shift. With a
  // one-sample pulse the series of the element at the origin is the component's sequence
  // times exp(2 pi i f_d t); an element three samples of travel further along the
  // direction receives it three samples earlier; one displaced across it, unchanged.
  sim::ReverbParams p;
  p.sample_rate = 8000.0;
  p.n_samples = 256;
  p.center_frequency = 16000.0;  // a whole number of cycles per sample: no carrier phase from a lead in samples
  p.sound_speed = kC;
  p.n_azimuth = 4;               // only scales the density here
  p.n_low = 256;
  p.velocity = Vec3d{50.0, 50.0, 0.0};  // not used in component mode
  const std::vector<double> coarse{-1.0, 1.0};
  const double level = 0.25 * 8000.0 * 2.0 / (2.0 * kPi / 4.0);  // sigma = 0.5
  const std::vector<double> density{level, level, 0.0, 0.0};     // the second component has no power
  const std::vector<double> direction{0.0, 0.0, 1.0, 1.0, 0.0, 0.0};
  const std::vector<double> doppler{1000.0, -300.0};
  sim::ReverbPower power{coarse.data(), 2, density.data(), 2};
  power.direction = direction.data();
  power.doppler = doppler.data();
  EXPECT_EQ(sim::reverb_active_sectors(p, power), 1);
  std::mt19937 gen(5);
  std::normal_distribution<double> normal;
  std::vector<double> normals(2 * 256);
  for (double& v : normals) v = normal(gen);
  const std::vector<std::complex<double>> pulse{{1.0, 0.0}};
  const std::vector<double> offsets{0.0, 0.0, 0.0, 0.0, 0.0, 3.0 * kC / 8000.0, 0.3, 0.2, 0.0};
  std::vector<float> out(2 * 3 * 256);
  sim::reverb_series(Backend::Host, p, power, normals.data(), 1, pulse.data(), 1, offsets.data(), 3, out.data());
  const auto expected = [&](int k) {
    const double a = 2.0 * kPi * 1000.0 * k / 8000.0;
    return 0.5 * std::complex<double>(normals[k], normals[256 + k]) * std::complex<double>(std::cos(a), std::sin(a));
  };
  for (int k = 0; k < 256; ++k) {
    const std::complex<double> origin(out[2 * k], out[2 * k + 1]);
    const std::complex<double> deeper(out[2 * (256 + k)], out[2 * (256 + k) + 1]);
    const std::complex<double> aside(out[2 * (512 + k)], out[2 * (512 + k) + 1]);
    EXPECT_NEAR(std::abs(origin - expected(k)), 0.0, 2e-4) << k;
    EXPECT_NEAR(std::abs(deeper - expected((k + 3) % 256)), 0.0, 2e-4) << k;
    EXPECT_NEAR(std::abs(aside - origin), 0.0, 2e-4) << k;
  }
  // Both arrays or neither; and the normals must cover the components with power.
  sim::ReverbPower half = power;
  half.doppler = nullptr;
  EXPECT_THROW(sim::reverb_active_sectors(p, half), std::invalid_argument);
  EXPECT_THROW(sim::reverb_series(Backend::Host, p, power, normals.data(), 2, pulse.data(), 1, offsets.data(), 3,
                                  out.data()),
               std::invalid_argument);
}
