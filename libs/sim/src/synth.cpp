#include "sonar/sim/synth.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "sonar/core/parallel.hpp"
#include "sonar/sim/synth_kernel.hpp"

namespace sonar::sim {

#if SONAR_HAVE_CUDA
void synthesize_cuda(const SynthParams& params, const std::vector<SynthElement>& rx,
                     const std::vector<SynthPath>& paths, float* out);
#endif

namespace {

SynthElement make_element(const Pose& pose, const Element& e, const Vec3d& extra_offset) {
  SynthElement s;
  const Vec3d lever = e.lever_arm + extra_offset;
  s.position = world_position(pose, lever);
  s.velocity = world_velocity(pose, lever);
  const ElementAxes<double> body = element_axes(e.boresight);
  s.axes.boresight = world_direction(pose, body.boresight);
  s.axes.az = world_direction(pose, body.az);
  s.axes.el = world_direction(pose, body.el);
  s.beamwidth_az = e.beamwidth_az;
  s.beamwidth_el = e.beamwidth_el;
  return s;
}

/// Highest frequency the pulse holds, Hz.
double top_frequency(const SynthParams& p) {
  double f = std::max(p.waveform.f_start, p.waveform.f_stop);
  if (p.waveform.type == 3) {
    for (int i = 0; i < p.envelope.n_bands; ++i) f = std::max(f, p.envelope.knot_frequency[i]);
    f = std::max(f, p.envelope.carrier);
  }
  return std::max(f, p.center_frequency);
}

double window_end(const SynthParams& p) {
  return std::max(p.start_time + static_cast<double>(p.n_samples) / p.sample_rate, 0.0);
}

}  // namespace

double arrival_residual_bound(double window_end, double speed, double sound_speed, double frequency, int passes) {
  if (!(speed > 0.0)) return 0.0;
  const double pi2 = 6.283185307179586476925;
  return window_end * std::pow(speed / sound_speed, passes) * pi2 * frequency;
}

int arrival_passes_for(double window_end, double speed, double sound_speed, double frequency, double tolerance) {
  if (!(tolerance > 0.0)) throw std::invalid_argument("arrival_passes_for: tolerance must be positive");
  if (!(speed < sound_speed)) throw std::invalid_argument("arrival_passes_for: the speed must be below the sound speed");
  for (int n = 1; n < 64; ++n) {
    if (arrival_residual_bound(window_end, speed, sound_speed, frequency, n) <= tolerance) return n;
  }
  return 64;
}

namespace {

/// Largest speeds that enter the two solves: {outbound points, receivers}.
ArrivalResidual solve_speeds(const SynthParams& params, const std::vector<SynthElement>& rx,
                             const std::vector<SynthPath>& paths) {
  double v_out = 0.0, v_back = 0.0;
  for (const SynthPath& p : paths) v_out = std::max(v_out, norm(p.q_velocity));
  const double v_acc = norm(params.platform_acceleration) * window_end(params);
  for (const SynthElement& e : rx) v_back = std::max(v_back, norm(e.velocity) + v_acc);
  return {v_out, v_back};
}

}  // namespace

ArrivalResidual arrival_residual(const SynthParams& params, const std::vector<SynthElement>& rx,
                                 const std::vector<SynthPath>& paths) {
  const ArrivalResidual v = solve_speeds(params, rx, paths);
  const double t_end = window_end(params), f = top_frequency(params);
  return {arrival_residual_bound(t_end, v.out, params.sound_speed, f, params.arrival_passes_out),
          arrival_residual_bound(t_end, v.back, params.sound_speed, f, params.arrival_passes_back)};
}

void set_arrival_passes(SynthParams& params, const std::vector<SynthElement>& rx,
                        const std::vector<SynthPath>& paths, double tolerance) {
  const ArrivalResidual v = solve_speeds(params, rx, paths);
  const double t_end = window_end(params), f = top_frequency(params);
  params.arrival_passes_out = arrival_passes_for(t_end, v.out, params.sound_speed, f, tolerance);
  params.arrival_passes_back = arrival_passes_for(t_end, v.back, params.sound_speed, f, tolerance);
}

double absorption_clamp_error_db(const SynthParams& p) {
  const double span = p.absorption_f_max - p.absorption_f_min;
  if (!(span > 0.0)) return 0.0;  // one value for every frequency: nothing is clamped
  const double step = span / (kAbsorptionTableSize - 1);
  const double slope_lo = (p.absorption_db_per_m[1] - p.absorption_db_per_m[0]) / step;
  const double slope_hi =
      (p.absorption_db_per_m[kAbsorptionTableSize - 1] - p.absorption_db_per_m[kAbsorptionTableSize - 2]) / step;
  double worst = 0.0;  // dB/m
  const auto read = [&](double f) {
    if (f < p.absorption_f_min) worst = std::max(worst, std::abs(slope_lo) * (p.absorption_f_min - f));
    if (f > p.absorption_f_max) worst = std::max(worst, std::abs(slope_hi) * (f - p.absorption_f_max));
  };
  const WaveformParams& w = p.waveform;
  if (w.type == 3) {
    for (int i = 0; i < p.envelope.n_bands; ++i) read(p.envelope.knot_frequency[i]);
  } else if (w.type == 0) {
    read(w.f_start);
  } else if (w.type == 1) {
    read(w.f_start);
    read(w.f_stop);
  } else {
    read(0.5 * (w.f_start + w.f_stop));
  }
  return worst * p.sound_speed * window_end(p);  // the longest two-way path the window holds
}

void synthesize(Backend backend, const SynthParams& params, const std::vector<SynthElement>& rx,
                const std::vector<SynthPath>& paths, float* out) {
  if (params.n_samples <= 0 || rx.empty() || paths.empty()) return;
  if (!(params.sample_rate > 0.0)) throw std::invalid_argument("synthesize: sample_rate must be positive");
  if (!(params.sound_speed > 0.0) || !std::isfinite(params.sound_speed)) {
    throw std::invalid_argument("synthesize: sound_speed must be stated (positive and finite); there is no default");
  }
  if (!(params.waveform.duration > 0.0)) throw std::invalid_argument("synthesize: pulse duration must be positive");
  if (params.waveform.type < 0 || params.waveform.type > 3) {
    throw std::invalid_argument("synthesize: unknown waveform type");
  }
  if (params.arrival_passes_out < 1 || params.arrival_passes_back < 1) {
    throw std::invalid_argument("synthesize: arrival passes must be at least 1");
  }
  if (params.waveform.type == 3) {
    const EnvelopeTable& t = params.envelope;
    if (t.data == nullptr || t.n_fine < 4 || t.n_bands < 1 || t.n_bands > kMaxEnvelopeBands || !(t.rate > 0.0) ||
        !(t.carrier > 0.0)) {
      throw std::invalid_argument(
          "synthesize: a sampled (arbitrary) waveform needs its envelope table (SynthParams::envelope)");
    }
  }
  // Both limits are checked here, on the host, for either backend: the kernel is unchanged.
  if (params.absorption_clamp_limit_db > 0.0) {
    const double err = absorption_clamp_error_db(params);
    if (err > params.absorption_clamp_limit_db) {
      throw std::invalid_argument(
          "synthesize: the pulse holds frequencies outside the absorption table [" +
          std::to_string(params.absorption_f_min) + ", " + std::to_string(params.absorption_f_max) +
          "] Hz; reading the table at its end would change a level by up to " + std::to_string(err) +
          " dB over the longest path in the window. Tabulate absorption over the whole band of the pulse "
          "(or raise absorption_clamp_limit_db)");
    }
  }
  if (params.arrival_residual_limit > 0.0) {
    const ArrivalResidual left = arrival_residual(params, rx, paths);
    if (!(left.out <= params.arrival_residual_limit) || !(left.back <= params.arrival_residual_limit)) {
      throw std::invalid_argument(
          "synthesize: the arrival-time solve may leave " + std::to_string(std::max(left.out, left.back)) +
          " rad of phase with (" + std::to_string(params.arrival_passes_out) + ", " +
          std::to_string(params.arrival_passes_back) + ") passes at these speeds, window and frequency (limit " +
          std::to_string(params.arrival_residual_limit) +
          " rad). Raise arrival_passes (arrival_passes_for gives the count; \"auto\" in the Python binding)");
    }
  }
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    synthesize_cuda(params, rx, paths, out);
    return;
#else
    throw std::runtime_error("synthesize: built without the CUDA backend");
#endif
  }
  SynthKernel<HostAccumulate> k{params, rx.data(), paths.data(), out,
                               static_cast<std::int64_t>(rx.size()),
                               static_cast<std::int64_t>(paths.size()), HostAccumulate{}};
  // One thread per channel, so plain adds into a channel's samples never race.
  parallel_for_host(k.n_channels, [&k](std::int64_t ic) {
    for (std::int64_t ip = 0; ip < k.n_paths; ++ip) k.pair(ic, ip);
  });
}

SynthParams params_from_record(const PingRecord& record) {
  SynthParams p;
  p.waveform = waveform_params(record.transmit.waveform);
  p.sound_speed = record.environment.sound_speed;
  p.sample_rate = record.samples.sample_rate;
  p.start_time = record.samples.start_time;
  p.n_samples = record.samples.n_samples;
  p.complex_baseband = record.samples.kind == SampleKind::ComplexBaseband;
  if (record.samples.kind == SampleKind::ComplexAnalytic) {
    throw std::invalid_argument("synthesize: analytic output is not supported; use real or baseband");
  }
  p.center_frequency = record.samples.center_frequency;
  p.tx = make_element(record.geometry.tx_pose, record.geometry.tx, Vec3d{});
  return p;
}

std::vector<SynthElement> elements_from_record(const PingRecord& record) {
  const auto n = record.geometry.rx.size();
  if (static_cast<std::int64_t>(n) != record.samples.n_channels) {
    throw std::invalid_argument("synthesize: one receive element per channel is required");
  }
  std::vector<SynthElement> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    // The simulated truth includes each element's true position offset.
    const Vec3d offset =
        i < record.calibration.position_offset.size() ? record.calibration.position_offset[i] : Vec3d{};
    SynthElement s = make_element(record.geometry.rx_pose, record.geometry.rx[i], offset);
    if (i < record.metadata.channel_board.size()) {
      const auto b = static_cast<std::size_t>(record.metadata.channel_board[i]);
      if (b < record.metadata.boards.size()) {
        s.clock_offset = record.metadata.boards[b].clock_offset;
        s.clock_drift = record.metadata.boards[b].clock_drift_ppm * 1e-6;
      }
    }
    out.push_back(s);
  }
  return out;
}

void synthesize_into(Backend backend, PingRecord& record, const std::vector<SynthPath>& paths,
                     const SynthParams* params_override) {
  if (record.samples.data.size() != record.samples.expected_size()) {
    throw std::invalid_argument("synthesize: record samples are not allocated");
  }
  const SynthParams params = params_override != nullptr ? *params_override : params_from_record(record);
  synthesize(backend, params, elements_from_record(record), paths, record.samples.data.data());
}

}  // namespace sonar::sim
