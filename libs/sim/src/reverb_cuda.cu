// CUDA backend of the reverberation steering kernel. Same kernel body as the host backend.
#include <cstdint>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/sim/reverb_kernel.hpp"

namespace sonar::sim {

/// Add `n_sectors` sector spectra into `total` ([n_channels, n_low] interleaved complex,
/// host memory, accumulated into). One element of work per frequency bin.
void reverb_steer_cuda(const float* spectra, const float* lead, const float* omega, float* total,
                       std::int64_t n_sectors, std::int64_t n_channels, std::int64_t n_low, bool uniform,
                       int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const DeviceBuffer<float> d_spectra(spectra, static_cast<std::size_t>(2 * n_sectors * n_low));
  const DeviceBuffer<float> d_lead(lead, static_cast<std::size_t>(n_sectors * n_channels));
  const DeviceBuffer<float> d_omega(omega, static_cast<std::size_t>(n_low));
  const DeviceBuffer<float> d_total(total, static_cast<std::size_t>(2 * n_channels * n_low));
  const ReverbSteerKernel k{d_spectra.data(), d_lead.data(), d_omega.data(), d_total.data(), n_sectors,
                            n_channels,       n_low,         1,              uniform};
  parallel_for_cuda(k.size(), k);
  d_total.download(total);
}

}  // namespace sonar::sim
