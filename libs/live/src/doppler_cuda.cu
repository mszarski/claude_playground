// CUDA backend of the batched Doppler stages (doppler_batched.hpp): the templates there on the device. Nothing here but
// the executor and the device's free memory; the steps and the kernels are the ones doppler_batched.cpp runs with
// HostExec. Compiled with --fmad=false (libs/live/CMakeLists.txt): the folds compare powers and ratios, and the
// background is the detector's, so no product may be contracted into a fused multiply-add (detector_kernel.hpp).
#include <cstdint>

#include "sonar/dsp/exec_cuda.hpp"
#include "sonar/live/doppler_batched.hpp"

namespace sonar::live {

std::int64_t doppler_cuda_free_bytes(int device) {
  cuda_check(cudaSetDevice(device), "select device");
  std::size_t free_bytes = 0, total_bytes = 0;
  cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory");
  return static_cast<std::int64_t>(free_bytes);
}

void sweep_bank_cuda(const SweepBankBatchedInput& in, const SweepBankBatchedOutput& out, DopplerProfile& prof, int device) {
  dsp::CudaExec ex(device);
  sweep_bank_batched(ex, in, out, prof);
}

void tone_bank_cuda(const ToneBankBatchedInput& in, float* power, float* ratio, unsigned char* flag, DopplerProfile& prof, int device) {
  dsp::CudaExec ex(device);
  BeamformProfile beams;
  tone_bank_batched(ex, in, power, ratio, flag, prof, beams);
}

}  // namespace sonar::live
