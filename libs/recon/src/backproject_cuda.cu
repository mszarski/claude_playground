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

/// Slab bytes of the per-collection inputs.
std::size_t collection_bytes(const BpInputs& inputs, std::size_t table_size, std::size_t alignment) {
  std::size_t n = 0;
  n += align_up(inputs.groups.size() * sizeof(BpGroup), alignment);
  n += align_up(inputs.records.size() * sizeof(BpRecord), alignment);
  n += align_up(inputs.samples.size() * 2 * sizeof(float), alignment);
  n += align_up(table_size * sizeof(float), alignment);
  return n;
}

}  // namespace

/// The per-collection inputs on one device, in a slab of their own.
struct BpResident {
  explicit BpResident(int dev) : device(dev), slab(Backend::Cuda) {}
  int device;
  SlabAllocator slab;
  const BpGroup* groups = nullptr;
  const BpRecord* records = nullptr;
  const float* samples = nullptr;
  const float* table = nullptr;
};

BpResident* bp_resident_create(const BpInputs& inputs, const std::vector<float>& sinc_table, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  auto* r = new BpResident(device);
  try {
    r->slab.reserve(collection_bytes(inputs, sinc_table.size(), r->slab.alignment()));
    r->slab.commit();
    r->groups = slab_upload(r->slab, inputs.groups.data(), inputs.groups.size());
    r->records = slab_upload(r->slab, inputs.records.data(), inputs.records.size());
    r->samples = slab_upload(r->slab, reinterpret_cast<const float*>(inputs.samples.data()),
                             2 * inputs.samples.size());
    r->table = slab_upload(r->slab, sinc_table.data(), sinc_table.size());
  } catch (...) {
    delete r;
    throw;
  }
  return r;
}

void bp_resident_destroy(BpResident* resident) {
  if (resident == nullptr) return;
  cudaSetDevice(resident->device);  // the slab must be freed on the device that holds it
  delete resident;
}

int bp_resident_device(const BpResident* resident) { return resident->device; }
std::size_t bp_resident_bytes(const BpResident* resident) { return resident->slab.capacity(); }

/// Slab bytes one backprojection call needs. This is the stage's memory report. With the
/// collection already resident on the device, only the per-call arrays count.
std::size_t backproject_memory_required(const BpInputs& inputs, std::size_t n_points,
                                        const BpOptions& options, bool with_weight_sum,
                                        std::size_t alignment, bool resident = false) {
  std::size_t n = 0;
  if (!resident) {
    n += collection_bytes(inputs, static_cast<std::size_t>((kSincPhases + 1) * kSincTaps), alignment);
  }
  n += align_up(n_points * sizeof(Vec3d), alignment);
  n += align_up(n_points * 2 * sizeof(float), alignment);
  if (options.position_lookup == PositionLookup::SoftwareTable) {
    n += align_up(inputs.trajectory.size() * sizeof(Vec3f), alignment);
  }
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
                      std::complex<float>* image, float* weight_sum, BpStats* stats,
                      const BpResident* resident) {
  cuda_check(cudaSetDevice(options.device), "select device");
  // One device allocation for the whole call (ASASIN Sec. II-C).
  SlabAllocator slab(Backend::Cuda);
  slab.reserve(backproject_memory_required(inputs, points.size(), options, weight_sum != nullptr,
                                           slab.alignment(), resident != nullptr));
  slab.commit();
  const std::size_t n_points = points.size();
  const BpGroup* d_groups;
  const BpRecord* d_records;
  const float* d_samples;
  const float* d_table;
  if (resident != nullptr) {
    d_groups = resident->groups;
    d_records = resident->records;
    d_samples = resident->samples;
    d_table = resident->table;
  } else {
    d_groups = slab_upload(slab, inputs.groups.data(), inputs.groups.size());
    d_records = slab_upload(slab, inputs.records.data(), inputs.records.size());
    d_samples = slab_upload(slab, reinterpret_cast<const float*>(inputs.samples.data()),
                            2 * inputs.samples.size());
    d_table = slab_upload(slab, sinc_table.data(), sinc_table.size());
  }
  const Vec3d* d_points = slab_upload(slab, points.data(), n_points);
  // Upload the image: the kernel accumulates into what is already there.
  float* d_image = slab_upload(slab, reinterpret_cast<const float*>(image), 2 * n_points);
  float* d_wsum = weight_sum != nullptr ? slab_upload(slab, weight_sum, n_points) : nullptr;

  BpKernel k{d_groups, d_records, d_samples, d_points, d_table, d_image, d_wsum,
             static_cast<std::int64_t>(inputs.groups.size()), options, nullptr, TrajectoryAccess{},
             0, 0};
  // Trajectory residuals: a plain device table, or a texture with hardware filtering.
  cudaArray_t traj_array = nullptr;
  cudaTextureObject_t traj_texture = 0;
  if (!inputs.trajectory.empty() && options.position_lookup != PositionLookup::Analytic) {
    k.traj.mode = options.position_lookup;
    if (options.position_lookup == PositionLookup::SoftwareTable) {
      k.traj.table = slab_upload(slab, inputs.trajectory.data(), inputs.trajectory.size());
    } else {
      // One row per ping, one column per table entry. Texture memory is managed by the
      // driver, outside the slab.
      std::size_t width = 1;
      for (const BpGroup& g : inputs.groups) {
        if (static_cast<std::size_t>(g.traj_count) > width) width = static_cast<std::size_t>(g.traj_count);
      }
      const std::size_t height = inputs.groups.size();
      std::vector<float> texels(width * height * 4, 0.0F);
      for (std::size_t ig = 0; ig < height; ++ig) {
        const BpGroup& g = inputs.groups[ig];
        for (std::int32_t c = 0; c < g.traj_count; ++c) {
          const Vec3f& v = inputs.trajectory[static_cast<std::size_t>(g.traj_offset + c)];
          float* t = texels.data() + (ig * width + static_cast<std::size_t>(c)) * 4;
          t[0] = v.x;
          t[1] = v.y;
          t[2] = v.z;
        }
      }
      const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
      cuda_check(cudaMallocArray(&traj_array, &desc, width, height), "allocate trajectory texture");
      cuda_check(cudaMemcpy2DToArray(traj_array, 0, 0, texels.data(), width * 4 * sizeof(float),
                                     width * 4 * sizeof(float), height, cudaMemcpyHostToDevice),
                 "upload trajectory texture");
      cudaResourceDesc res{};
      res.resType = cudaResourceTypeArray;
      res.res.array.array = traj_array;
      cudaTextureDesc tex{};
      tex.addressMode[0] = cudaAddressModeClamp;
      tex.addressMode[1] = cudaAddressModeClamp;
      tex.filterMode = cudaFilterModeLinear;   // the hardware interpolation the paper uses
      tex.readMode = cudaReadModeElementType;
      tex.normalizedCoords = 0;
      cuda_check(cudaCreateTextureObject(&traj_texture, &res, &tex, nullptr), "create trajectory texture");
      k.traj.texture = traj_texture;
    }
  }

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
  if (traj_texture != 0) cudaDestroyTextureObject(traj_texture);
  if (traj_array != nullptr) cudaFreeArray(traj_array);
  cuda_check(cudaMemcpy(image, d_image, 2 * n_points * sizeof(float), cudaMemcpyDeviceToHost),
             "download image");
  if (weight_sum != nullptr) {
    cuda_check(cudaMemcpy(weight_sum, d_wsum, n_points * sizeof(float), cudaMemcpyDeviceToHost),
               "download weights");
  }
}

void launch_on_device(const BpKernel& kernel, std::int64_t n_points) { parallel_for_cuda(n_points, kernel); }

}  // namespace sonar::recon
