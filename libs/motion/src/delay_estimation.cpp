#include "sonar/motion/delay_estimation.hpp"

#include <stdexcept>
#include <string>

#include "sonar/core/parallel.hpp"
#include "sonar/motion/delay_kernel.hpp"

namespace sonar::motion {

#if SONAR_HAVE_CUDA
// delay_cuda.cu: the same two kernels on the device; `out` is [n_tasks, 4].
void estimate_delays_cuda(const float* a, const float* b, std::int64_t n_channels, std::int64_t n_samples,
                          const std::vector<std::int32_t>& channel_a, const std::vector<std::int32_t>& channel_b,
                          const std::vector<std::int64_t>& centres, int half, int max_lag, double sample_rate,
                          double center_frequency, double* out, int device);
#endif

std::vector<DelayEstimate> estimate_delays(const std::complex<float>* a, const std::complex<float>* b,
                                           std::int64_t n_channels, std::int64_t n_samples,
                                           const std::vector<std::int32_t>& channel_a,
                                           const std::vector<std::int32_t>& channel_b,
                                           const std::vector<std::int64_t>& centres, int half, int max_lag,
                                           double sample_rate, double center_frequency, Backend backend,
                                           int device) {
  if (channel_a.size() != channel_b.size()) {
    throw std::invalid_argument("estimate_delays: channel_a and channel_b differ in length");
  }
  if (half < 2 || max_lag < 1) throw std::invalid_argument("estimate_delays: half must be >= 2 and max_lag >= 1");
  if (!(sample_rate > 0.0) || !(center_frequency > 0.0)) {
    throw std::invalid_argument("estimate_delays: sample_rate and center_frequency must be positive");
  }
  for (std::size_t p = 0; p < channel_a.size(); ++p) {
    if (channel_a[p] < 0 || channel_a[p] >= n_channels || channel_b[p] < 0 || channel_b[p] >= n_channels) {
      throw std::invalid_argument("estimate_delays: channel index out of range");
    }
  }
  for (const std::int64_t c : centres) {
    if (c - half - max_lag < 0 || c + half + max_lag > n_samples) {
      throw std::invalid_argument("estimate_delays: window centre " + std::to_string(c) +
                                  " with its lag span does not fit the record");
    }
  }
  const auto n_tasks = static_cast<std::int64_t>(channel_a.size() * centres.size());
  const std::int64_t n_lags = 2 * static_cast<std::int64_t>(max_lag) + 1;
  std::vector<double> out(static_cast<std::size_t>(4 * n_tasks));
  const auto* fa = reinterpret_cast<const float*>(a);
  const auto* fb = reinterpret_cast<const float*>(b);
  if (n_tasks > 0) {
    if (backend == Backend::Cuda) {
#if SONAR_HAVE_CUDA
      estimate_delays_cuda(fa, fb, n_channels, n_samples, channel_a, channel_b, centres, half, max_lag, sample_rate,
                           center_frequency, out.data(), device);
#else
      (void)device;
      throw std::runtime_error("estimate_delays: built without the CUDA backend");
#endif
    } else {
      std::vector<double> corr(static_cast<std::size_t>(2 * n_tasks * n_lags));
      std::vector<double> energy(static_cast<std::size_t>(n_tasks * n_lags));
      const auto n_centres = static_cast<std::int64_t>(centres.size());
      const CorrelateKernel ck{fa, fb, channel_a.data(), channel_b.data(), centres.data(), n_samples, n_centres,
                               half, max_lag, corr.data(), energy.data()};
      parallel_for_host(n_tasks * n_lags, ck);
      const PickKernel pk{fa, channel_a.data(), centres.data(), n_samples, n_centres, half, max_lag,
                          sample_rate, center_frequency, corr.data(), energy.data(), out.data()};
      parallel_for_host(n_tasks, pk);
    }
  }
  std::vector<DelayEstimate> result(static_cast<std::size_t>(n_tasks));
  for (std::int64_t t = 0; t < n_tasks; ++t) {
    const double* o = &out[static_cast<std::size_t>(4 * t)];
    result[static_cast<std::size_t>(t)] = DelayEstimate{o[0], o[1], o[2], o[3] != 0.0};
  }
  return result;
}

}  // namespace sonar::motion
