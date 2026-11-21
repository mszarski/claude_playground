// CUDA backend of the multiscale detector's statistic (detector.hpp, multiscale_cells): the kernels of
// detector_kernel.hpp on the device. The map comes from host memory (the chain's map is on the host by then) and the
// statistic goes back to it (`level`, `background`, `peak`, `flag`: [n_b * n_cells] host arrays); dropping sidelobes
// and grouping stay on the host (detector.cpp, multiscale_group).
#include <cstdint>
#include <vector>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/live/detector_kernel.hpp"

namespace sonar::live {

void multiscale_cells_cuda(const float* fine, std::int64_t fine_stride, std::int64_t fine_offset, const float* cell_mean, std::int64_t n_b,
                           std::int64_t n_cells, std::int64_t guard, std::int64_t train, bool sloped, const std::vector<double>& offset,
                           const std::vector<double>& threshold, const std::vector<std::int64_t>& cells, std::int64_t k, double* level,
                           float* background, float* peak, unsigned char* flag, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const auto n = static_cast<std::size_t>(n_b * n_cells);
  const DeviceBuffer<float> d_fine(fine, static_cast<std::size_t>(n_b * fine_stride));
  const DeviceBuffer<float> d_mean(cell_mean, n);
  const DeviceBuffer<double> d_offset(offset);
  const DeviceBuffer<double> d_threshold(threshold);
  const DeviceBuffer<std::int64_t> d_cells(cells);
  const DeviceBuffer<double> d_scratch(sloped ? static_cast<std::size_t>(n_b * 4 * train) : 0U);
  DeviceBuffer<double> d_median(static_cast<std::size_t>(4 * n_b));
  d_median.zero();
  const DeviceBuffer<double> d_csum(static_cast<std::size_t>(n_b * (n_cells + 1)));
  const DeviceBuffer<double> d_slope(static_cast<std::size_t>(2 * n_b));
  const DeviceBuffer<double> d_level(n);
  const DeviceBuffer<float> d_background(n), d_peak(n);
  const DeviceBuffer<unsigned char> d_flag(n);
  if (sloped) parallel_for_cuda(4 * n_b, DetectorMedianKernel{d_mean.data(), d_scratch.data(), d_median.data(), n_cells, train});
  parallel_for_cuda(n_b, DetectorRowKernel{d_mean.data(), d_median.data(), d_csum.data(), d_slope.data(), n_cells, train, sloped});
  parallel_for_cuda(n_b * n_cells, DetectorCellKernel{d_fine.data(), fine_stride, fine_offset, d_csum.data(), d_slope.data(), d_offset.data(),
                                                      d_threshold.data(), d_cells.data(), static_cast<std::int64_t>(cells.size()), n_cells, k,
                                                      guard, train, d_level.data(), d_background.data(), d_peak.data(), d_flag.data()});
  d_level.download(level);
  d_background.download(background);
  d_peak.download(peak);
  d_flag.download(flag);
}

}  // namespace sonar::live
