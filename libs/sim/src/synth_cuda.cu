// CUDA backend of the synthesis kernel. Same kernel body as the host backend.
#include <vector>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/sim/synth_kernel.hpp"

namespace sonar::sim {

namespace {

struct AtomicAccumulate {
  __device__ void operator()(float* dst, float value) const { atomicAdd(dst, value); }
};

struct PairFunctor {
  SynthKernel<AtomicAccumulate> k;
  __device__ void operator()(std::int64_t i) const { k.pair(i / k.n_paths, i % k.n_paths); }
};

}  // namespace

void synthesize_cuda(const SynthParams& params, const std::vector<SynthElement>& rx,
                     const std::vector<SynthPath>& paths, float* out) {
  const std::size_t stride = params.complex_baseband ? 2 : 1;
  const std::size_t n_out = rx.size() * static_cast<std::size_t>(params.n_samples) * stride;
  DeviceBuffer<SynthElement> d_rx(rx);
  DeviceBuffer<SynthPath> d_paths(paths);
  DeviceBuffer<float> d_out(out, n_out);  // upload: the kernel accumulates into existing data
  // A sampled pulse: the envelope table goes to the device and the kernel's copy of the
  // parameters points at it. Nothing else differs from the host backend.
  SynthParams device_params = params;
  DeviceBuffer<double> d_envelope;
  if (params.waveform.type == 3) {
    const std::size_t n_table = 2 * static_cast<std::size_t>(params.envelope.n_fine) *
                                static_cast<std::size_t>(params.envelope.n_bands + 1);
    d_envelope = DeviceBuffer<double>(params.envelope.data, n_table);
    device_params.envelope.data = d_envelope.data();
  }
  PairFunctor f{{device_params, d_rx.data(), d_paths.data(), d_out.data(),
                 static_cast<std::int64_t>(rx.size()), static_cast<std::int64_t>(paths.size()),
                 AtomicAccumulate{}}};
  parallel_for_cuda(f.k.n_channels * f.k.n_paths, f);
  d_out.download(out);
}

}  // namespace sonar::sim
