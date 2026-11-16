// CUDA backend of the backprojection kernel. Same kernel body as the host backend.
#include <complex>
#include <vector>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/recon/backproject_kernel.hpp"

namespace sonar::recon {

void backproject_cuda(const BpInputs& inputs, const std::vector<Vec3d>& points,
                      const BpOptions& options, const std::vector<float>& sinc_table,
                      std::complex<float>* image, float* weight_sum) {
  DeviceBuffer<BpGroup> d_groups(inputs.groups);
  DeviceBuffer<BpRecord> d_records(inputs.records);
  DeviceBuffer<float> d_samples(reinterpret_cast<const float*>(inputs.samples.data()),
                                2 * inputs.samples.size());
  DeviceBuffer<Vec3d> d_points(points);
  DeviceBuffer<float> d_table(sinc_table);
  // Upload the image: the kernel accumulates into what is already there.
  DeviceBuffer<float> d_image(reinterpret_cast<const float*>(image), 2 * points.size());
  DeviceBuffer<float> d_wsum;
  if (weight_sum != nullptr) d_wsum = DeviceBuffer<float>(weight_sum, points.size());

  BpKernel k{d_groups.data(), d_records.data(), d_samples.data(), d_points.data(), d_table.data(),
             d_image.data(), weight_sum != nullptr ? d_wsum.data() : nullptr,
             static_cast<std::int64_t>(inputs.groups.size()), options};
  parallel_for_cuda(static_cast<std::int64_t>(points.size()), k);
  d_image.download(reinterpret_cast<float*>(image));
  if (weight_sum != nullptr) d_wsum.download(weight_sum);
}

}  // namespace sonar::recon
