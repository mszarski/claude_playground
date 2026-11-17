// CUDA backend of scattering-function reverberation: the batched formulation of
// reverb_batched.hpp on the device. The sequence of steps and the kernels are the ones the
// host runs with HostExec (ReverbParams::batched), so there is nothing here but the executor.
#include <cstdint>

#include "sonar/dsp/exec_cuda.hpp"
#include "sonar/sim/reverb_batched.hpp"

namespace sonar::sim {

/// Add the components of one call into `total` ([n_channels, n_low] interleaved complex,
/// host memory, accumulated into). `shape` fills the host arrays of each run of components.
void reverb_accumulate_cuda(const ReverbBatchSetup& setup, std::int64_t n_components, ReverbShapeFn shape, void* context,
                            float* total, ReverbProfile& profile, int device) {
  dsp::CudaExec ex(device);
  reverb_accumulate_batched(ex, setup, n_components, shape, context, total, profile);
}

/// The interpolation to the record's rate, the window and the carrier.
void reverb_finish_cuda(const float* total, std::int64_t n_channels, std::int64_t n_low, std::int64_t n,
                        std::int64_t decimation, std::int64_t start, double sample_rate, double start_time,
                        double center_frequency, bool carrier, bool real_output, float* out, int device) {
  dsp::CudaExec ex(device);
  reverb_finish_batched(ex, total, n_channels, n_low, n, decimation, start, sample_rate, start_time, center_frequency,
                        carrier, real_output, out);
}

}  // namespace sonar::sim
