#include "sonar/sim/synth.hpp"

#include <stdexcept>

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

}  // namespace

void synthesize(Backend backend, const SynthParams& params, const std::vector<SynthElement>& rx,
                const std::vector<SynthPath>& paths, float* out) {
  if (params.n_samples <= 0 || rx.empty() || paths.empty()) return;
  if (!(params.sample_rate > 0.0)) throw std::invalid_argument("synthesize: sample_rate must be positive");
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
