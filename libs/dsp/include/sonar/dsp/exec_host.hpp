// Where a batched (device-style) algorithm runs: the host version.
//
// Some stages are written for the device as a short sequence of steps on whole arrays:
// upload, a kernel over every element, a batched transform, another kernel, download
// (sonar/sim/reverb_batched.hpp, sonar/live/beamform_batched.hpp). Such a sequence is a
// template over an executor. CudaExec (exec_cuda.hpp, .cu files only) runs it on the device;
// HostExec runs the very same sequence and the very same kernels on CPU threads, so the
// device formulation can be tested, and compared with the reference host code, on a
// machine without a GPU.
//
// An executor provides:
//   Buffer<T>                    owning array where the kernels run; data(), size()
//   alloc<T>(n)                  a buffer, contents undefined
//   upload(const T* host, n)     a buffer holding a copy of host memory
//   download(buffer, T* host)    copy back
//   zero(buffer)
//   staging<T>(slot, n)          host memory to fill before an upload; page-locked on the
//                                device, kept between calls (slots 0..3)
//   run(n, kernel)               kernel(i) for i in [0, n)
//   fft(data, n, batch, inverse) in-place complex transforms of `batch` rows of length n,
//                                interleaved (re, im) floats, neither direction scaled
//
// This header is C++17 so that the algorithms that include it also compile under nvcc.
#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "sonar/core/parallel.hpp"
#include "sonar/dsp/fft.hpp"

namespace sonar::dsp {

template <typename T>
class HostBuffer {
 public:
  HostBuffer() = default;
  explicit HostBuffer(std::size_t n) : ptr_(n > 0 ? new T[n] : nullptr), size_(n) {}
  T* data() const { return ptr_.get(); }
  std::size_t size() const { return size_; }

 private:
  std::unique_ptr<T[]> ptr_;
  std::size_t size_ = 0;
};

class HostExec {
 public:
  template <typename T>
  using Buffer = HostBuffer<T>;

  static constexpr bool on_device = false;

  template <typename T>
  Buffer<T> alloc(std::size_t n) {
    return Buffer<T>(n);
  }
  template <typename T>
  Buffer<T> upload(const T* host, std::size_t n) {
    Buffer<T> b(n);
    if (n > 0) std::memcpy(b.data(), host, n * sizeof(T));
    return b;
  }
  template <typename T>
  void download(const Buffer<T>& b, T* host) {
    if (b.size() > 0) std::memcpy(host, b.data(), b.size() * sizeof(T));
  }
  template <typename T>
  void zero(const Buffer<T>& b) {
    if (b.size() > 0) std::memset(b.data(), 0, b.size() * sizeof(T));
  }
  template <typename T>
  T* staging(int slot, std::size_t n) {
    std::vector<unsigned char>& s = staging_[slot];
    if (s.size() < n * sizeof(T)) s.resize(n * sizeof(T));
    return reinterpret_cast<T*>(s.data());
  }
  template <typename F>
  void run(std::int64_t n, const F& f) {
    parallel_for_host(n, f);
  }
  void fft(float* data, std::int64_t n, std::int64_t batch, bool inverse) {
    parallel_for_host(batch, [&](std::int64_t r) {
      fft_unscaled(reinterpret_cast<std::complex<float>*>(data) + r * n, static_cast<std::size_t>(n), inverse);
    });
  }

 private:
  std::vector<unsigned char> staging_[4];
};

}  // namespace sonar::dsp
