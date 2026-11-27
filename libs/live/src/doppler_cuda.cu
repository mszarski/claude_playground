// CUDA backend of the batched Doppler stages (doppler_batched.hpp): the templates there on the device. Nothing here but
// the executor and the device's free memory; the steps and the kernels are the ones doppler_batched.cpp runs with
// HostExec. Compiled with --fmad=false (libs/live/CMakeLists.txt): the folds compare powers and ratios, so no product
// may be contracted into a fused multiply-add. The kernels shared with the default chain are not instantiated here
// (doppler_batched.hpp run_shared): the backprojection kernel runs as backproject_cuda.cu compiles it, the detector's as
// detector_cuda.cu does, the far-field beams as beamform_cuda.cu does.
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

// beamform_cuda.cu: the far-field beams left on the device.
DeviceBuffer<float> farfield_cuda_device(const float* samples, const float* w, const double* tau, std::int64_t n, std::int64_t n_el,
                                         std::int64_t n_b, double df, double fc, BeamformProfile& profile, int device);

void tone_bank_cuda(const ToneBankBatchedInput& in, float* power, float* ratio, unsigned char* flag, DopplerProfile& prof, int device) {
  ToneBankBatchedInput run = in;
  DeviceBuffer<float> beams;
  if (in.beams == nullptr) {
    // Made before this function's executor exists: each executor holds the staging lock while it lives.
    BeamformStopwatch watch;
    BeamformProfile unused;
    beams = farfield_cuda_device(in.samples, in.w, in.tau, in.n, in.n_el, in.n_b, in.df, in.fc, unused, device);
    watch.lap(prof.tone_beams);
    run.device_beams = beams.data();
  }
  dsp::CudaExec ex(device);
  tone_bank_batched(ex, run, power, ratio, flag, prof);
}

}  // namespace sonar::live
