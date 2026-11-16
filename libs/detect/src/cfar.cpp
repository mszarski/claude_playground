#include "sonar/detect/cfar.hpp"

#include <cmath>
#include <stdexcept>

#include "sonar/core/parallel.hpp"
#include "sonar/detect/cfar_kernel.hpp"

namespace sonar::detect {

#if SONAR_HAVE_CUDA
void cfar_cuda(const float* power, std::int64_t n, std::int64_t n_cols, const CfarOptions& options,
               const std::vector<double>& factor, unsigned char* detections, float* ratio, int device);
#endif

double cfar_threshold_factor(double pfa, std::int32_t n_train) {
  const auto n = static_cast<double>(n_train);
  return n * (std::pow(pfa, -1.0 / n) - 1.0);
}

void cfar(Backend backend, const float* power, std::int64_t n_rows, std::int64_t n_cols,
          const CfarOptions& options, unsigned char* detections, float* ratio, int device) {
  if (options.guard < 0 || options.train < 1) throw std::invalid_argument("cfar: guard >= 0 and train >= 1");
  if (!(options.pfa > 0.0 && options.pfa < 1.0)) throw std::invalid_argument("cfar: pfa must be in (0, 1)");
  (void)device;  // used by the CUDA backend only
  if (n_rows <= 0 || n_cols <= 0) return;
  // The window shrinks at the edges, so the multiplier depends on the cell count.
  std::vector<double> factor(static_cast<std::size_t>(2 * options.train + 1), 0.0);
  for (std::int32_t c = 1; c <= 2 * options.train; ++c) {
    factor[static_cast<std::size_t>(c)] = cfar_threshold_factor(options.pfa, c);
  }
  const std::int64_t n = n_rows * n_cols;
  if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
    cfar_cuda(power, n, n_cols, options, factor, detections, ratio, device);
    return;
#else
    throw std::runtime_error("cfar: built without the CUDA backend");
#endif
  }
  const CfarKernel k{power, factor.data(), detections, ratio, n_cols, options.guard, options.train};
  parallel_for_host(n, k);
}

}  // namespace sonar::detect
