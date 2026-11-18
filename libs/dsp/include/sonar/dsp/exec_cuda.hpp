// Where a batched (device-style) algorithm runs: the CUDA version of the executor described
// in exec_host.hpp. Include only from .cu translation units.
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>

#include "sonar/core/parallel_cuda.hpp"

namespace sonar::dsp {

/// In-place complex transforms of `batch` rows of length n on the device (interleaved floats,
/// device memory), neither direction scaled. Plans are kept for the most recent sizes only.
void fft_cuda_device(float* data, std::int64_t n, std::int64_t batch, bool inverse, int device);

/// Page-locked host memory of at least `bytes`, kept between calls (slots 0..3). Use it with
/// the lock of cuda_staging_mutex() held.
void* cuda_staging(int slot, std::size_t bytes);
std::mutex& cuda_staging_mutex();

class CudaExec {
 public:
  template <typename T>
  using Buffer = DeviceBuffer<T>;

  static constexpr bool on_device = true;

  /// Selects the device; holds the staging memory until it is destroyed.
  explicit CudaExec(int device) : device_(device), lock_(cuda_staging_mutex()) {
    cuda_check(cudaSetDevice(device), "select device");
  }

  template <typename T>
  Buffer<T> alloc(std::size_t n) {
    return Buffer<T>(n);
  }
  template <typename T>
  Buffer<T> upload(const T* host, std::size_t n) {
    return Buffer<T>(host, n);
  }
  template <typename T>
  void download(const Buffer<T>& b, T* host) {
    b.download(host);
  }
  template <typename T>
  void zero(Buffer<T>& b) {
    b.zero();
  }
  template <typename T>
  T* staging(int slot, std::size_t n) {
    return static_cast<T*>(cuda_staging(slot, n * sizeof(T)));
  }
  template <typename F>
  void run(std::int64_t n, const F& f) {
    parallel_for_cuda(n, f);
  }
  void fft(float* data, std::int64_t n, std::int64_t batch, bool inverse) {
    fft_cuda_device(data, n, batch, inverse, device_);
  }

 private:
  int device_;
  std::unique_lock<std::mutex> lock_;
};

}  // namespace sonar::dsp
