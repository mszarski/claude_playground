// parallel_for on the CUDA backend, plus a minimal owning device buffer.
// Include only from .cu translation units.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sonar {

inline void cuda_check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(status));
  }
}

template <typename F>
__global__ void parallel_for_kernel(std::int64_t n, F f) {
  const std::int64_t i =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}

/// Call f(i) for i in [0, n) on the device. f is copied by value to the device, so it
/// may hold device pointers and plain data only.
template <typename F>
void parallel_for_cuda(std::int64_t n, const F& f) {
  if (n <= 0) return;
  const int block = 256;
  // One launch covers at most 2^31-1 blocks; split larger ranges.
  const std::int64_t max_per_launch = static_cast<std::int64_t>(0x7fffffff) * block;
  if (n > max_per_launch) throw std::runtime_error("CUDA: parallel_for range too large");
  const auto grid = static_cast<unsigned>((n + block - 1) / block);
  parallel_for_kernel<F><<<grid, block>>>(n, f);
  cuda_check(cudaGetLastError(), "kernel launch");
  cuda_check(cudaDeviceSynchronize(), "kernel execution");
}

/// Owning device array of trivially copyable T.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) { allocate(n); }
  DeviceBuffer(const T* host, std::size_t n) {
    allocate(n);
    if (n > 0) cuda_check(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.data(), host.size()) {}
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }
  void zero() {
    if (size_ > 0) cuda_check(cudaMemset(ptr_, 0, size_ * sizeof(T)), "memset");
  }
  void download(T* host) const {
    if (size_ > 0) cuda_check(cudaMemcpy(host, ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost), "download");
  }
  T* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  void allocate(std::size_t n) {
    size_ = n;
    if (n > 0) cuda_check(cudaMalloc(&ptr_, n * sizeof(T)), "allocate");
  }
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace sonar
