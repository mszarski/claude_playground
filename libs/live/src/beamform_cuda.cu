// CUDA backend of the far-field beamformers: the batched formulation of beamform_batched.hpp
// on the device. The sequence of steps and the kernels are the ones the host runs with
// HostExec (the `batched` argument of farfield_beamform and mvdr_beamform), so there is
// nothing here but the executor.
#include <cstdint>

#include "sonar/dsp/exec_cuda.hpp"
#include "sonar/live/beamform_batched.hpp"

namespace sonar::live {

void farfield_cuda(const float* samples, const float* w, const double* tau, std::int64_t n, std::int64_t n_el,
                   std::int64_t n_b, double df, double fc, float* out, BeamformProfile& profile, int device) {
  dsp::CudaExec ex(device);
  farfield_batched(ex, samples, w, tau, n, n_el, n_b, df, fc, out, profile);
}

MvdrBatchedStatus mvdr_cuda(const MvdrBatchedInput& in, float* out, BeamformProfile& profile, int device) {
  dsp::CudaExec ex(device);
  return mvdr_batched(ex, in, out, profile);
}

}  // namespace sonar::live
