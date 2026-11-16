// CUDA backend of the CFAR kernel. Same kernel body as the host backend.
#include <vector>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/detect/cfar.hpp"
#include "sonar/detect/cfar_kernel.hpp"

namespace sonar::detect {

void cfar_cuda(const float* power, std::int64_t n, std::int64_t n_cols, const CfarOptions& options,
               const std::vector<double>& factor, unsigned char* detections, float* ratio, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const auto count = static_cast<std::size_t>(n);
  const DeviceBuffer<float> d_power(power, count);
  const DeviceBuffer<double> d_factor(factor);
  const DeviceBuffer<unsigned char> d_det(count);
  DeviceBuffer<float> d_ratio;
  if (ratio != nullptr) d_ratio = DeviceBuffer<float>(count);
  const CfarKernel k{d_power.data(), d_factor.data(), d_det.data(),
                     ratio != nullptr ? d_ratio.data() : nullptr, n_cols, options.guard, options.train};
  parallel_for_cuda(n, k);
  d_det.download(detections);
  if (ratio != nullptr) d_ratio.download(ratio);
}

}  // namespace sonar::detect
