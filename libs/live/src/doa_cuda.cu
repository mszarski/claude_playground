// CUDA backend of the SBL iteration. Same kernel bodies as the host backend (sbl_kernel.hpp).
// The steering vectors and the snapshots are uploaded once per frame; each iteration uploads
// gamma, the list of bearings in play and the noise, and downloads num and den.
#include <cstdint>

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/live/sbl_kernel.hpp"

namespace sonar::live {

struct SblCudaState {
  std::int64_t n = 0, m = 0, n_f = 0, n_l = 0;
  DeviceBuffer<double> a, y, cov, z, gamma, sigma2, num, den;
  DeviceBuffer<std::int32_t> active;
};

SblCudaState* sbl_cuda_open(std::int64_t n, std::int64_t m, std::int64_t n_f, std::int64_t n_l, const double* a, const double* y,
                            int device) {
  cuda_check(cudaSetDevice(device), "select device");
  auto* s = new SblCudaState;
  try {
    s->n = n;
    s->m = m;
    s->n_f = n_f;
    s->n_l = n_l;
    s->a = DeviceBuffer<double>(a, static_cast<std::size_t>(2 * n_f * m * n));
    s->y = DeviceBuffer<double>(y, static_cast<std::size_t>(2 * n_f * n_l * n));
    s->cov = DeviceBuffer<double>(static_cast<std::size_t>(2 * n_f * n * n));
    s->z = DeviceBuffer<double>(static_cast<std::size_t>(2 * n_f * n_l * n));
    s->gamma = DeviceBuffer<double>(static_cast<std::size_t>(m));
    s->sigma2 = DeviceBuffer<double>(static_cast<std::size_t>(n_f));
    s->num = DeviceBuffer<double>(static_cast<std::size_t>(n_f * m));
    s->den = DeviceBuffer<double>(static_cast<std::size_t>(n_f * m));
    s->active = DeviceBuffer<std::int32_t>(static_cast<std::size_t>(m));
  } catch (...) {
    delete s;
    throw;
  }
  return s;
}

void sbl_cuda_step(SblCudaState* s, const double* gamma, const std::int32_t* active, std::int64_t n_active, const double* sigma2,
                   double* num, double* den) {
  const auto bytes = [](std::int64_t count, std::size_t size) { return static_cast<std::size_t>(count) * size; };
  cuda_check(cudaMemcpy(s->gamma.data(), gamma, bytes(s->m, sizeof(double)), cudaMemcpyHostToDevice), "upload gamma");
  cuda_check(cudaMemcpy(s->active.data(), active, bytes(n_active, sizeof(std::int32_t)), cudaMemcpyHostToDevice), "upload active");
  cuda_check(cudaMemcpy(s->sigma2.data(), sigma2, bytes(s->n_f, sizeof(double)), cudaMemcpyHostToDevice), "upload noise");
  parallel_for_cuda(s->n_f * s->n * s->n,
                    SblCovKernel{s->a.data(), s->gamma.data(), s->active.data(), s->sigma2.data(), s->cov.data(), s->n, s->m, n_active});
  parallel_for_cuda(s->n_f, SblFactorKernel{s->cov.data(), s->y.data(), s->z.data(), s->n, s->n_l});
  parallel_for_cuda(s->n_f * n_active,
                    SblUpdateKernel{s->a.data(), s->cov.data(), s->z.data(), s->active.data(), s->num.data(), s->den.data(), s->n,
                                    s->m, n_active, s->n_l});
  cuda_check(cudaMemcpy(num, s->num.data(), bytes(s->n_f * n_active, sizeof(double)), cudaMemcpyDeviceToHost), "download num");
  cuda_check(cudaMemcpy(den, s->den.data(), bytes(s->n_f * n_active, sizeof(double)), cudaMemcpyDeviceToHost), "download den");
}

void sbl_cuda_close(SblCudaState* s) { delete s; }

}  // namespace sonar::live
