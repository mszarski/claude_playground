// CUDA backend of the backprojection kernel. Same kernel body as the host backend.
#include <complex>
#include <vector>

#include "sonar/core/allocator.hpp"
#include "sonar/core/parallel_cuda.hpp"
#include "sonar/recon/backproject_kernel.hpp"

namespace sonar::recon {

float aperture_radius(const BpInputs& inputs);

namespace {

/// Upload `count` items into a block taken from the slab.
template <typename T>
T* slab_upload(SlabAllocator& slab, const T* host, std::size_t count) {
  T* dev = slab.allocate_array<T>(count);
  if (count > 0) {
    cuda_check(cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  return dev;
}

}  // namespace

/// Slab bytes one backprojection call needs. This is the stage's memory report.
std::size_t backproject_memory_required(const BpInputs& inputs, std::size_t n_points,
                                        const BpOptions& options, bool with_weight_sum,
                                        std::size_t alignment) {
  std::size_t n = 0;
  n += align_up(inputs.groups.size() * sizeof(BpGroup), alignment);
  n += align_up(inputs.records.size() * sizeof(BpRecord), alignment);
  n += align_up(inputs.samples.size() * 2 * sizeof(float), alignment);
  n += align_up(n_points * sizeof(Vec3d), alignment);
  n += align_up(static_cast<std::size_t>((kSincPhases + 1) * kSincTaps) * sizeof(float), alignment);
  n += align_up(n_points * 2 * sizeof(float), alignment);
  if (with_weight_sum) n += align_up(n_points * sizeof(float), alignment);
  if (options.cull_block > 0) {
    const std::int64_t blocks = block_count(options.grid_rows, options.cull_block) *
                                block_count(options.grid_cols, options.cull_block);
    n += align_up(inputs.groups.size() * static_cast<std::size_t>(blocks), alignment);
  }
  return n;
}

void backproject_cuda(const BpInputs& inputs, const std::vector<Vec3d>& points,
                      const BpOptions& options, const std::vector<float>& sinc_table,
                      std::complex<float>* image, float* weight_sum, BpStats* stats) {
  // One device allocation for the whole call (ASASIN Sec. II-C).
  SlabAllocator slab(Backend::Cuda);
  slab.reserve(backproject_memory_required(inputs, points.size(), options, weight_sum != nullptr,
                                           slab.alignment()));
  slab.commit();
  const std::size_t n_points = points.size();
  const BpGroup* d_groups = slab_upload(slab, inputs.groups.data(), inputs.groups.size());
  const BpRecord* d_records = slab_upload(slab, inputs.records.data(), inputs.records.size());
  const float* d_samples = slab_upload(slab, reinterpret_cast<const float*>(inputs.samples.data()),
                                       2 * inputs.samples.size());
  const Vec3d* d_points = slab_upload(slab, points.data(), n_points);
  const float* d_table = slab_upload(slab, sinc_table.data(), sinc_table.size());
  // Upload the image: the kernel accumulates into what is already there.
  float* d_image = slab_upload(slab, reinterpret_cast<const float*>(image), 2 * n_points);
  float* d_wsum = weight_sum != nullptr ? slab_upload(slab, weight_sum, n_points) : nullptr;

  BpKernel k{d_groups, d_records, d_samples, d_points, d_table, d_image, d_wsum,
             static_cast<std::int64_t>(inputs.groups.size()), options};
  // Coarse culling pass on the device, then the per-point kernel reads the mask.
  if (options.cull_block > 0) {
    const std::int64_t br = block_count(options.grid_rows, options.cull_block);
    const std::int64_t bc = block_count(options.grid_cols, options.cull_block);
    const auto n_groups = static_cast<std::int64_t>(inputs.groups.size());
    const auto n_mask = static_cast<std::size_t>(n_groups * br * bc);
    auto* d_mask = slab.allocate_array<unsigned char>(n_mask);
    const CullKernel ck{d_groups, d_records, d_points, d_mask, options.grid_rows, options.grid_cols,
                        options.cull_block, br, bc, options.fov_factor, aperture_radius(inputs)};
    parallel_for_cuda(n_groups * br * bc, ck);
    k.mask = d_mask;
    k.blocks_c = bc;
    k.n_blocks = br * bc;
    if (stats != nullptr) {
      std::vector<unsigned char> host_mask(n_mask);
      cuda_check(cudaMemcpy(host_mask.data(), d_mask, n_mask, cudaMemcpyDeviceToHost), "download mask");
      stats->ping_blocks_total = static_cast<std::int64_t>(n_mask);
      stats->ping_blocks_kept = 0;
      for (unsigned char m : host_mask) stats->ping_blocks_kept += m;
    }
  } else if (stats != nullptr) {
    *stats = BpStats{};
  }
  parallel_for_cuda(static_cast<std::int64_t>(n_points), k);
  cuda_check(cudaMemcpy(image, d_image, 2 * n_points * sizeof(float), cudaMemcpyDeviceToHost),
             "download image");
  if (weight_sum != nullptr) {
    cuda_check(cudaMemcpy(weight_sum, d_wsum, n_points * sizeof(float), cudaMemcpyDeviceToHost),
               "download weights");
  }
}

}  // namespace sonar::recon
