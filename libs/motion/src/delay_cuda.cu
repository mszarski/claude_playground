// CUDA backend of delay estimation: the host's two kernels (delay_kernel.hpp) on the device.
#include <cstdint>
#include <vector>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/motion/delay_kernel.hpp"

namespace sonar::motion {

void estimate_delays_cuda(const float* a, const float* b, std::int64_t n_channels, std::int64_t n_samples,
                          const std::vector<std::int32_t>& channel_a, const std::vector<std::int32_t>& channel_b,
                          const std::vector<std::int64_t>& centres, int half, int max_lag, double sample_rate,
                          double center_frequency, double* out, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const auto n_tasks = static_cast<std::int64_t>(channel_a.size() * centres.size());
  const std::int64_t n_lags = 2 * static_cast<std::int64_t>(max_lag) + 1;
  const auto n_values = static_cast<std::size_t>(2 * n_channels * n_samples);
  DeviceBuffer<float> d_a(a, n_values), d_b(b, n_values);
  DeviceBuffer<std::int32_t> d_ca(channel_a), d_cb(channel_b);
  DeviceBuffer<std::int64_t> d_centres(centres);
  DeviceBuffer<double> d_corr(static_cast<std::size_t>(2 * n_tasks * n_lags));
  DeviceBuffer<double> d_energy(static_cast<std::size_t>(n_tasks * n_lags));
  DeviceBuffer<double> d_out(static_cast<std::size_t>(4 * n_tasks));
  const auto n_centres = static_cast<std::int64_t>(centres.size());
  parallel_for_cuda(n_tasks * n_lags,
                    CorrelateKernel{d_a.data(), d_b.data(), d_ca.data(), d_cb.data(), d_centres.data(), n_samples,
                                    n_centres, half, max_lag, d_corr.data(), d_energy.data()});
  parallel_for_cuda(n_tasks, PickKernel{d_a.data(), d_ca.data(), d_centres.data(), n_samples, n_centres, half,
                                        max_lag, sample_rate, center_frequency, d_corr.data(), d_energy.data(),
                                        d_out.data()});
  d_out.download(out);
}

}  // namespace sonar::motion
