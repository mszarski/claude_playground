// CUDA backend of the multiscale detector's statistic (detector.hpp, multiscale_cells): the kernels of
// detector_kernel.hpp on the device. The map comes from host memory (the chain's map is on the host by then) and the
// statistic goes back to it (`level`, `background`, `peak`, `flag`: [n_b * n_cells] host arrays; `level` may be null);
// dropping sidelobes and grouping stay on the host (detector.cpp, multiscale_group).
#include <cstdint>
#include <type_traits>
#include <vector>

#include "sonar/core/allocator.hpp"
#include "sonar/core/parallel_cuda.hpp"
#include "sonar/live/detector_kernel.hpp"

namespace sonar::live {

void multiscale_cells_cuda(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean, std::int64_t n_b,
                           std::int64_t n_cells, std::int64_t guard, std::int64_t train, bool sloped, const std::vector<double>& offset,
                           const std::vector<double>& threshold, const std::vector<std::int64_t>& cells, std::int64_t k, double* level,
                           float* background, float* peak, unsigned char* flag, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const auto n = static_cast<std::size_t>(n_b * n_cells);
  const auto n_fine = static_cast<std::size_t>(n_b * fine_stride);
  const std::size_t n_scratch = sloped ? static_cast<std::size_t>(n_b * 4 * train) : 0U;
  const auto n_median = static_cast<std::size_t>(4 * n_b), n_csum = static_cast<std::size_t>(n_b * (n_cells + 1));
  const auto n_slope = static_cast<std::size_t>(2 * n_b);
  // One device allocation for the call (as the backprojection kernel does), carved into the arrays.
  SlabAllocator slab(Backend::Cuda);
  for (const std::size_t bytes : {n_fine * sizeof(float), n * sizeof(float), offset.size() * sizeof(double), threshold.size() * sizeof(double),
                                  cells.size() * sizeof(std::int64_t), n_scratch * sizeof(double), n_median * sizeof(double),
                                  n_csum * sizeof(double), n_slope * sizeof(double), n * sizeof(double), n * sizeof(float),
                                  n * sizeof(float), n * sizeof(unsigned char)}) {
    slab.reserve(bytes);
  }
  slab.commit();
  const auto upload = [&](const auto* host, std::size_t count) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(host)>>;
    T* dev = slab.allocate_array<T>(count);
    if (count > 0) cuda_check(cudaMemcpy(dev, host, count * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return dev;
  };
  const float* d_fine = upload(fine, n_fine);
  const float* d_mean = upload(cell_mean, n);
  const double* d_offset = upload(offset.data(), offset.size());
  const double* d_threshold = upload(threshold.data(), threshold.size());
  const std::int64_t* d_cells = upload(cells.data(), cells.size());
  double* d_scratch = slab.allocate_array<double>(n_scratch);
  double* d_median = slab.allocate_array<double>(n_median);
  cuda_check(cudaMemset(d_median, 0, n_median * sizeof(double)), "memset");
  double* d_csum = slab.allocate_array<double>(n_csum);
  double* d_slope = slab.allocate_array<double>(n_slope);
  double* d_level = slab.allocate_array<double>(n);
  float* d_background = slab.allocate_array<float>(n);
  float* d_peak = slab.allocate_array<float>(n);
  unsigned char* d_flag = slab.allocate_array<unsigned char>(n);
  if (sloped) parallel_for_cuda(4 * n_b, DetectorMedianKernel{d_mean, d_scratch, d_median, n_cells, train});
  parallel_for_cuda(n_b, DetectorRowKernel{d_mean, d_median, d_csum, d_slope, n_cells, train, sloped});
  parallel_for_cuda(n_b * n_cells, DetectorCellKernel{d_fine, fine_stride, fine_offset, d_csum, d_slope, d_offset, d_threshold, d_cells,
                                                      static_cast<std::int64_t>(cells.size()), n_cells, k, guard, train, d_level,
                                                      d_background, d_peak, d_flag});
  const auto download = [&](auto* host, const auto* dev, std::size_t count) {
    if (host != nullptr && count > 0) cuda_check(cudaMemcpy(host, dev, count * sizeof(*dev), cudaMemcpyDeviceToHost), "download");
  };
  download(level, d_level, n);           // null: not wanted (the chain)
  download(background, d_background, n);
  download(peak, d_peak, n);
  download(flag, d_flag, n);
}

void launch_on_device(const DetectorMedianKernel& kernel, std::int64_t n) { parallel_for_cuda(n, kernel); }
void launch_on_device(const DetectorRowKernel& kernel, std::int64_t n) { parallel_for_cuda(n, kernel); }
void launch_on_device(const DetectorCellKernel& kernel, std::int64_t n) { parallel_for_cuda(n, kernel); }

}  // namespace sonar::live
