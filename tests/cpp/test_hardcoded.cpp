// The fixes of docs/HARDCODED_AUDIT.md in the C++ outside the live chain: values that had
// a typed default are stated by the caller or the call fails, and every fixed capacity
// either raises or reports what it cut.
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include "sonar/core/record.hpp"
#include "sonar/detect/cfar.hpp"
#include "sonar/dsp/baseband.hpp"
#include "sonar/dsp/conditioning.hpp"
#include "sonar/dsp/pulse_compression.hpp"
#include "sonar/recon/backproject.hpp"
#include "sonar/sim/paths.hpp"
#include "sonar/sim/synth.hpp"

using namespace sonar;  // NOLINT(google-build-using-namespace)

namespace {

constexpr double kC = 1500.0;
constexpr double kPi = 3.14159265358979323846;

/// A complex baseband record of ones: `channels` omnidirectional elements along x.
PingRecord ones_record(std::int64_t channels, std::int64_t samples, double fs, double fc, double start) {
  PingRecord r;
  r.samples.resize(channels, samples, SampleKind::ComplexBaseband);
  for (std::size_t i = 0; i < r.samples.data.size(); i += 2) r.samples.data[i] = 1.0F;
  r.samples.sample_rate = fs;
  r.samples.center_frequency = fc;
  r.samples.start_time = start;
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = fc - 0.25 * fs;
  r.transmit.waveform.f_stop = fc + 0.25 * fs;
  r.transmit.waveform.duration = 1e-3;
  r.geometry.rx.resize(static_cast<std::size_t>(channels));
  for (std::int64_t i = 0; i < channels; ++i) {
    r.geometry.rx[static_cast<std::size_t>(i)].lever_arm = {0.01 * static_cast<double>(i), 0.0, 0.0};
    r.geometry.rx[static_cast<std::size_t>(i)].boresight = {0.0, 1.0, 0.0};
  }
  r.geometry.tx.boresight = {0.0, 1.0, 0.0};
  r.environment.sound_speed = kC;
  set_default_channel_tables(r);
  return r;
}

// --- K-2, K-3: no physical default in the record ---------------------------------------------

TEST(Defaults, RecordHasNoSoundSpeedOrBoresight) {
  const PingRecord r;
  EXPECT_TRUE(std::isnan(r.environment.sound_speed));
  EXPECT_TRUE(std::isnan(Element{}.boresight.x));
  EXPECT_TRUE(std::isnan(r.transmit.direction.y));
  EXPECT_TRUE(std::isnan(sim::SynthParams{}.sound_speed));
  EXPECT_TRUE(std::isnan(sim::ImageSourceConfig{}.sound_speed));
  EXPECT_TRUE(std::isnan(sim::SedimentConfig{}.sediment_speed));
  EXPECT_TRUE(std::isnan(recon::BpOptions{}.sediment_speed));
  EXPECT_TRUE(std::isnan(recon::BpGroup{}.sound_speed));
}

TEST(Defaults, KernelEntryPointsRejectARecordWithoutSoundSpeed) {
  PingRecord r = ones_record(2, 64, 1e5, 2e5, 0.01);
  r.environment.sound_speed = kNaN;
  EXPECT_THROW(recon::make_inputs({&r}), std::invalid_argument);
  EXPECT_THROW(dsp::apply_tvg(r), std::invalid_argument);
  sim::SynthParams p;  // sound speed never stated
  p.sample_rate = 1e5;
  p.n_samples = 64;
  p.waveform.duration = 1e-3;
  p.waveform.f_start = p.waveform.f_stop = 2e5;
  std::vector<float> out(64);
  sim::SynthElement e;
  e.axes = element_axes(Vec3d{0.0, 1.0, 0.0});
  sim::SynthPath path;
  path.q = path.p = Vec3d{0.0, 10.0, 0.0};
  EXPECT_THROW(sim::synthesize(Backend::Host, p, {e}, {path}, out.data()), std::invalid_argument);
  const double pt[3] = {0.0, 10.0, 5.0};
  EXPECT_THROW(sim::free_field_eigenrays(kNaN, Vec3d{}, pt, 1), std::invalid_argument);
}

// --- S-1, S-2, S-3: the image series has no made-up boundary and no default order ------------

TEST(ImageSeries, NothingIsAssumedAboutTheBottomOrTheOrder) {
  const double pt[3] = {100.0, 0.0, 10.0};
  sim::ImageSourceConfig cfg;
  cfg.sound_speed = kC;
  EXPECT_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1), std::invalid_argument);  // no max_order
  cfg.max_order = 2;
  EXPECT_NO_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1));  // surface only: nothing missing
  cfg.has_bottom = true;
  cfg.bottom_z = 30.0;
  EXPECT_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1), std::invalid_argument);  // no coefficient
  cfg.has_fluid_bottom = true;
  EXPECT_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1), std::invalid_argument);  // no sediment
  cfg.fluid_bottom.speed = 1650.0;
  cfg.fluid_bottom.density = 1.9;
  EXPECT_NO_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1));
  cfg.has_fluid_bottom = false;
  cfg.bottom_coefficient = {0.5, 0.0};
  EXPECT_NO_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1));
  cfg.walls.push_back(sim::Wall{});  // a wall without a coefficient
  EXPECT_THROW(sim::image_source_eigenrays(cfg, Vec3d{0, 0, 5}, pt, 1), std::invalid_argument);
}

TEST(ImageSeries, TruncationIsTheFirstOrderLeftOut) {
  // Closed form: with coefficients Rs, Rb and max_order n, the strongest ray of order n + 1
  // is |Rs|^ns |Rb|^nb / r_image, relative to 1 / r_direct.
  sim::ImageSourceConfig cfg;
  cfg.sound_speed = kC;
  cfg.has_bottom = true;
  cfg.bottom_z = 100.0;
  cfg.bottom_coefficient = {0.5, 0.0};
  cfg.max_order = 0;
  const double zs = 20.0, zr = 60.0, range = 3000.0;
  const double direct = std::hypot(range, zr - zs);
  // Order 1: the surface image of the receiver is at -zr, the bottom image at 2 * 100 - zr.
  const double surface = 1.0 / std::hypot(range, -zr - zs), bottom = 0.5 / std::hypot(range, 200.0 - zr - zs);
  EXPECT_NEAR(sim::image_series_truncation(cfg, zs, zr, range), std::max(surface, bottom) * direct, 1e-12);
  // What is left out falls as the order rises, and the eigenray set agrees with it: the
  // strongest ray of order n + 1 in a series computed to n + 1 is the truncation at n.
  double previous = 2.0;
  for (int order = 0; order < 6; ++order) {
    cfg.max_order = order;
    const double left = sim::image_series_truncation(cfg, zs, zr, range);
    EXPECT_LT(left, previous);
    previous = left;
    sim::ImageSourceConfig more = cfg;
    more.max_order = order + 1;
    const double pt[3] = {range, 0.0, zr};
    const sim::EigenraySet e = sim::image_source_eigenrays(more, Vec3d{0.0, 0.0, zs}, pt, 1);
    double strongest = 0.0;
    for (std::int64_t k = 0; k < e.n_rays; ++k) {
      if (e.n_surface[static_cast<std::size_t>(k)] + e.n_bottom[static_cast<std::size_t>(k)] == order + 1) {
        strongest = std::max(strongest, std::abs(e.amplitude[static_cast<std::size_t>(k)]));
      }
    }
    EXPECT_NEAR(left, strongest * direct, 1e-12 * left);
  }
  // At 3 km in 100 m of water the first order that two bounces leave out is still more than
  // a fifth of the direct path: the cut is not small, which is why it is reported.
  cfg.max_order = 2;
  EXPECT_GT(sim::image_series_truncation(cfg, zs, zr, range), 0.2);
  // A single boundary: the series ends at one bounce, nothing is cut.
  cfg.has_bottom = false;
  cfg.max_order = 1;
  EXPECT_EQ(sim::image_series_truncation(cfg, zs, zr, range), 0.0);
}

// --- S-6, S-7: synthesis limits are checked ---------------------------------------------------

sim::SynthParams lfm_params() {
  sim::SynthParams p;
  p.waveform.type = 1;
  p.waveform.f_start = 6000.0;
  p.waveform.f_stop = 10000.0;
  p.waveform.duration = 0.01;
  p.sound_speed = kC;
  p.sample_rate = 32000.0;
  p.start_time = 0.5;
  p.n_samples = 3200;  // window ends at 0.6 s
  p.center_frequency = 8000.0;
  p.tx.axes = element_axes(Vec3d{0.0, 1.0, 0.0});
  return p;
}

TEST(SynthLimits, ArrivalResidualIsTheStatedBoundAndMatchesTheMeasuredCase) {
  // The header's measured case: 12 m/s closing at 400 m and 8 kHz left 0.007 rad after three
  // passes and 5.5e-5 after four. The bound with the leg's own travel time reproduces both.
  const double leg = 400.0 / kC;
  EXPECT_NEAR(sim::arrival_residual_bound(leg, 12.0, kC, 8000.0, 3), 0.007, 0.001);
  EXPECT_NEAR(sim::arrival_residual_bound(leg, 12.0, kC, 8000.0, 4), 5.5e-5, 0.5e-5);
  EXPECT_EQ(sim::arrival_residual_bound(leg, 0.0, kC, 8000.0, 1), 0.0);
  // Fewest passes for a tolerance: one more pass gains a factor v / c.
  const int n = sim::arrival_passes_for(0.6, 12.0, kC, 10000.0, 1e-4);
  EXPECT_LE(sim::arrival_residual_bound(0.6, 12.0, kC, 10000.0, n), 1e-4);
  EXPECT_GT(sim::arrival_residual_bound(0.6, 12.0, kC, 10000.0, n - 1), 1e-4);
  EXPECT_THROW(sim::arrival_passes_for(0.6, 2000.0, kC, 1e4, 1e-4), std::invalid_argument);

  // From a call's own numbers: window end 0.6 s, top frequency 10 kHz.
  sim::SynthParams p = lfm_params();
  sim::SynthElement rx;
  rx.axes = element_axes(Vec3d{0.0, 1.0, 0.0});
  rx.velocity = Vec3d{3.0, 0.0, 0.0};
  sim::SynthPath path;
  path.q = path.p = Vec3d{0.0, 400.0, 0.0};
  path.q_velocity = path.p_velocity = Vec3d{0.0, -12.0, 0.0};
  const sim::ArrivalResidual left = sim::arrival_residual(p, {rx}, {path});
  EXPECT_DOUBLE_EQ(left.out, 0.6 * std::pow(12.0 / kC, 4) * 2.0 * kPi * 10000.0);
  EXPECT_DOUBLE_EQ(left.back, 0.6 * std::pow(3.0 / kC, 4) * 2.0 * kPi * 10000.0);
  // The platform's acceleration over the window adds to the receivers' speed.
  p.platform_acceleration = Vec3d{0.0, 5.0, 0.0};
  EXPECT_DOUBLE_EQ(sim::arrival_residual(p, {rx}, {path}).back,
                   0.6 * std::pow((3.0 + 5.0 * 0.6) / kC, 4) * 2.0 * kPi * 10000.0);
}

TEST(SynthLimits, TooFewPassesIsAnErrorNotAWrongPhase) {
  sim::SynthParams p = lfm_params();
  sim::SynthElement rx;
  rx.axes = element_axes(Vec3d{0.0, 1.0, 0.0});
  sim::SynthPath path;
  path.q = path.p = Vec3d{0.0, 400.0, 0.0};
  path.q_velocity = path.p_velocity = Vec3d{0.0, -12.0, 0.0};
  std::vector<float> out(static_cast<std::size_t>(p.n_samples), 0.0F);
  EXPECT_NO_THROW(sim::synthesize(Backend::Host, p, {rx}, {path}, out.data()));  // (4, 4): 1.5e-4 rad
  p.arrival_passes_out = 2;  // 0.6 s * (0.008)^2 * 2 pi 1e4 = 2.4 rad
  EXPECT_THROW(sim::synthesize(Backend::Host, p, {rx}, {path}, out.data()), std::invalid_argument);
  sim::set_arrival_passes(p, {rx}, {path}, 1e-4);
  EXPECT_GE(p.arrival_passes_out, 5);
  EXPECT_EQ(p.arrival_passes_back, 1);  // the receiver does not move
  EXPECT_NO_THROW(sim::synthesize(Backend::Host, p, {rx}, {path}, out.data()));
  p.arrival_passes_out = 2;
  p.arrival_residual_limit = 0.0;  // the check can be turned off
  EXPECT_NO_THROW(sim::synthesize(Backend::Host, p, {rx}, {path}, out.data()));
}

TEST(SynthLimits, AbsorptionTableThatDoesNotCoverThePulseIsAnError) {
  sim::SynthParams p = lfm_params();
  // A table over 7 to 8 kHz rising 1e-4 dB/m per kHz; the pulse runs 6 to 10 kHz.
  p.absorption_f_min = 7000.0;
  p.absorption_f_max = 8000.0;
  for (int i = 0; i < sim::kAbsorptionTableSize; ++i) {
    p.absorption_db_per_m[i] = 1e-4 * (7.0 + static_cast<double>(i) / (sim::kAbsorptionTableSize - 1));
  }
  // Clamped at 10 kHz the table reads 8e-4 instead of 1e-3: 2e-4 dB/m over c * 0.6 s = 900 m.
  EXPECT_NEAR(sim::absorption_clamp_error_db(p), 2e-4 * kC * 0.6, 1e-9);
  sim::SynthElement rx;
  rx.axes = element_axes(Vec3d{0.0, 1.0, 0.0});
  sim::SynthPath path;
  path.q = path.p = Vec3d{0.0, 400.0, 0.0};
  std::vector<float> out(static_cast<std::size_t>(p.n_samples), 0.0F);
  EXPECT_THROW(sim::synthesize(Backend::Host, p, {rx}, {path}, out.data()), std::invalid_argument);
  // The table over the pulse's own band: nothing is clamped.
  p.absorption_f_min = 6000.0;
  p.absorption_f_max = 10000.0;
  EXPECT_EQ(sim::absorption_clamp_error_db(p), 0.0);
  EXPECT_NO_THROW(sim::synthesize(Backend::Host, p, {rx}, {path}, out.data()));
  // One value for every frequency (no span) is not a clamp.
  p.absorption_f_min = p.absorption_f_max = 0.0;
  EXPECT_EQ(sim::absorption_clamp_error_db(p), 0.0);
}

// --- K-4: the gain law -------------------------------------------------------------------------

TEST(Tvg, LawIsSpreadingPlusTwoWayAbsorptionAndTheClampIsReported) {
  // 330 kHz class: samples from 30 m to 330 m of two-way range.
  const double fs = 1e4, start = 2.0 * 30.0 / kC;
  const auto n = static_cast<std::int64_t>((2.0 * 300.0 / kC) * fs);
  PingRecord r = ones_record(1, n, fs, 3.3e5, start);
  dsp::TvgOptions o;
  o.absorption_db_per_m = 0.083;
  o.max_gain_db = 400.0;
  const dsp::TvgReport rep = dsp::apply_tvg(r, o);
  EXPECT_EQ(rep.clamped_samples, 0);
  EXPECT_TRUE(std::isnan(rep.clamp_range));
  const std::int64_t i = n - 1;
  const double range = 0.5 * kC * (start + static_cast<double>(i) / fs);
  const double db = 40.0 * std::log10(range) + 2.0 * 0.083 * range;
  EXPECT_NEAR(20.0 * std::log10(r.samples.data[static_cast<std::size_t>(2 * i)]), db, 1e-3);
  EXPECT_NEAR(rep.gain_last_db, db, 1e-9);
  // The absorption part alone is what the default law leaves out: about 50 dB at 300 m.
  EXPECT_NEAR(2.0 * 0.083 * 300.0, 49.8, 1e-9);

  // With the default clamp the same law is cut, and the report says where and by how much.
  PingRecord c = ones_record(1, n, fs, 3.3e5, start);
  o.max_gain_db = 120.0;
  const dsp::TvgReport cut = dsp::apply_tvg(c, o);
  EXPECT_GT(cut.clamped_samples, 0);
  EXPECT_NEAR(40.0 * std::log10(cut.clamp_range) + 2.0 * 0.083 * cut.clamp_range, 120.0, 0.1);
  EXPECT_NEAR(cut.shortfall_db, db - 120.0, 1e-9);
  EXPECT_NEAR(cut.gain_last_db, 120.0, 1e-12);

  // One-way spreading is 20 dB per decade.
  PingRecord p = ones_record(1, n, fs, 3.3e5, start);
  dsp::TvgOptions one_way;
  one_way.spreading_db_per_decade = 20.0;
  EXPECT_NEAR(dsp::apply_tvg(p, one_way).gain_last_db, 20.0 * std::log10(range), 1e-9);

  PingRecord bad = ones_record(1, 16, fs, 3.3e5, start);
  o.absorption_db_per_m = -1.0;
  EXPECT_THROW(dsp::apply_tvg(bad, o), std::invalid_argument);
}

// --- K-5: the CFAR window in metres ------------------------------------------------------------

TEST(CfarWindow, CellsFollowTheMapAndTheResolution) {
  // A map sampled at the resolution: the cell counts CFAR has always defaulted to.
  const detect::CfarOptions same = detect::cfar_window(0.0375, 0.0375);
  EXPECT_EQ(same.guard, detect::CfarOptions{}.guard);
  EXPECT_EQ(same.train, detect::CfarOptions{}.train);
  EXPECT_EQ(same.pfa, detect::CfarOptions{}.pfa);
  // The audit's case: a 6 cm step and a 2 m compressed pulse. Four cells would be 24 cm,
  // inside the target; four resolution cells are 8 m, 134 cells.
  const detect::CfarOptions fine = detect::cfar_window(0.06, 2.0);
  EXPECT_EQ(fine.guard, 134);
  EXPECT_EQ(fine.train, 1067);
  // In metres: a 100 m guard and 300 m of training at a 2 m cell.
  const detect::CfarOptions metres = detect::cfar_window(2.0, 2.0, {4.0, 32.0, 100.0, 300.0});
  EXPECT_EQ(metres.guard, 50);
  EXPECT_EQ(metres.train, 150);
  // A guard in metres is never less than one resolution cell.
  EXPECT_EQ(detect::cfar_window(0.06, 2.0, {4.0, 32.0, 0.1, 30.0}).guard, 34);
  EXPECT_THROW(detect::cfar_window(0.0, 1.0), std::invalid_argument);
}

// --- K-6: Doppler allowance of the baseband filter ---------------------------------------------

TEST(Baseband, CwBandAllowsForTheStatedRangeRate) {
  PingRecord r = ones_record(1, 64, 1e5, 5000.0, 0.0);
  r.transmit.waveform.type = WaveformType::Cw;
  r.transmit.waveform.f_start = r.transmit.waveform.f_stop = 5000.0;
  r.transmit.waveform.duration = 1.0;
  dsp::BasebandOptions o;
  EXPECT_DOUBLE_EQ(dsp::baseband_band(r, o), 4.0);  // the audit's 4 Hz for a 1 s tone
  // 1 m/s of closing speed shifts the echo 2 v f / c = 6.67 Hz: outside +-2 Hz.
  const double shift = 2.0 * 1.0 * 5000.0 / kC;
  EXPECT_GT(shift, 0.5 * dsp::baseband_band(r, o));
  o.max_range_rate = 1.0;
  EXPECT_DOUBLE_EQ(dsp::baseband_band(r, o), 4.0 + 2.0 * shift);
  EXPECT_LT(shift, 0.5 * dsp::baseband_band(r, o));
  o.max_range_rate = 0.0;
  o.doppler_allowance_hz = 3.0;
  EXPECT_DOUBLE_EQ(dsp::baseband_band(r, o), 10.0);
  o.bandwidth = 50.0;  // a stated band is taken as stated
  EXPECT_DOUBLE_EQ(dsp::baseband_band(r, o), 50.0);
  // A swept pulse: the margin, then the allowance at the top of the band.
  r.transmit.waveform.type = WaveformType::Lfm;
  r.transmit.waveform.f_start = 4000.0;
  r.transmit.waveform.f_stop = 6000.0;
  dsp::BasebandOptions lfm;
  EXPECT_DOUBLE_EQ(dsp::baseband_band(r, lfm), 3000.0);
  lfm.max_range_rate = 15.0;
  EXPECT_DOUBLE_EQ(dsp::baseband_band(r, lfm), 3000.0 + 4.0 * 15.0 * 6000.0 / kC);
  r.environment.sound_speed = kNaN;
  EXPECT_THROW(dsp::baseband_band(r, lfm), std::invalid_argument);
}

TEST(Baseband, AMovingEchoOfALongToneSurvivesOnlyWithTheAllowance) {
  // A tone at 5 kHz + 6.67 Hz (an echo closing at 1 m/s), 1 s long, in a real record.
  const double fs = 16000.0, f0 = 5000.0, shift = 2.0 * 1.0 * f0 / kC;
  const auto n = static_cast<std::int64_t>(2.0 * fs);
  const auto energy = [&](double max_range_rate) {
    PingRecord r;
    r.samples.resize(1, n, SampleKind::RealPassband);
    r.samples.sample_rate = fs;
    for (std::int64_t i = static_cast<std::int64_t>(0.5 * fs); i < static_cast<std::int64_t>(1.5 * fs); ++i) {
      r.samples.data[static_cast<std::size_t>(i)] =
          static_cast<float>(std::cos(2.0 * kPi * (f0 + shift) * static_cast<double>(i) / fs));
    }
    r.transmit.waveform.type = WaveformType::Cw;
    r.transmit.waveform.f_start = r.transmit.waveform.f_stop = f0;
    r.transmit.waveform.duration = 1.0;
    r.environment.sound_speed = kC;
    dsp::BasebandOptions o;
    o.max_range_rate = max_range_rate;
    dsp::to_baseband(r, o);
    double e = 0.0;
    for (std::size_t i = 0; i < r.samples.data.size(); ++i) e += static_cast<double>(r.samples.data[i]) * r.samples.data[i];
    return e / r.samples.sample_rate;  // amplitude^2 seconds
  };
  const double without = energy(0.0), with = energy(1.5);
  EXPECT_NEAR(with, 1.0, 0.1);          // a unit tone for one second
  EXPECT_LT(without, 0.1 * with);       // today's default filters the mover out: more than 10 dB down
}

// --- K-7: the resampler's two numbers ----------------------------------------------------------

TEST(Resampler, ConstantsAreTheOnesTheReplicaUses) {
  EXPECT_EQ(dsp::kResamplerTaps, 16);
  EXPECT_DOUBLE_EQ(dsp::kResamplerKaiserBeta, 12.0);
}

// --- R-1: the trajectory allowance of the field-of-view pre-test -------------------------------

TEST(Backproject, TrajectoryAllowanceComesFromThePingsOwnTable) {
  PingRecord r = ones_record(4, 256, 1e5, 2e5, 0.01);
  // INS samples that stray 0.5 m sideways from the first-order pose at mid-ping.
  for (int k = 0; k < 5; ++k) {
    InsSample s{};
    s.t = 0.01 * k;
    s.position = Vec3d{0.0, k == 2 ? 0.5 : 0.0, 0.0};
    r.navigation.ins.push_back(s);
  }
  // The behaviour before D-92, by its name: 2 cm whatever the table holds.
  const recon::BpInputs fixed = recon::make_inputs({&r}, nullptr, 4, recon::kTrajectoryAllowance);
  EXPECT_FLOAT_EQ(fixed.groups[0].traj_allowance, 0.02F);
  EXPECT_NEAR(fixed.groups[0].traj_residual_max, 0.5F, 1e-6F);
  // The default takes it from the table.
  const recon::BpInputs table = recon::make_inputs({&r}, nullptr, 4);
  EXPECT_NEAR(table.groups[0].traj_allowance, 0.5F, 1e-6F);
  EXPECT_NEAR(recon::make_inputs({&r}, nullptr, 4, recon::kTrajectoryAllowanceFromTable).groups[0].traj_allowance,
              0.5F, 1e-6F);
  const recon::BpInputs stated = recon::make_inputs({&r}, nullptr, 4, 0.1F);
  EXPECT_FLOAT_EQ(stated.groups[0].traj_allowance, 0.1F);
  // Without a table the derived allowance is the 2 cm, never less.
  const recon::BpInputs none = recon::make_inputs({&r}, nullptr, 0);
  EXPECT_FLOAT_EQ(none.groups[0].traj_allowance, recon::kTrajectoryAllowance);
  EXPECT_FLOAT_EQ(none.groups[0].traj_residual_max, 0.0F);
}

TEST(Backproject, RefractionNeedsItsSedimentSpeed) {
  PingRecord r = ones_record(2, 256, 1e5, 2e5, 0.0);
  const recon::BpInputs in = recon::make_inputs({&r});
  recon::BpOptions o;
  o.refraction = true;  // sediment speed never stated
  std::vector<std::complex<float>> image(1);
  EXPECT_THROW(recon::backproject(Backend::Host, in, {Vec3d{0.0, 0.3, 0.2}}, o, image.data()), std::invalid_argument);
  o.sediment_speed = 1700.0;
  EXPECT_NO_THROW(recon::backproject(Backend::Host, in, {Vec3d{0.0, 0.3, 0.2}}, o, image.data()));
}

}  // namespace
