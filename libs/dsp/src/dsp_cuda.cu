// CUDA backend of baseband conversion and pulse compression: cuFFT for the transforms,
// small kernels for everything between them. Same arithmetic as the host versions.
#include <cufft.h>

#include <complex>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "sonar/dsp/exec_cuda.hpp"

#include "sonar/core/parallel_cuda.hpp"
#include "sonar/core/platform.hpp"

namespace sonar::dsp {

namespace {

void cufft_check(cufftResult status, const char* what) {
  if (status != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string("cuFFT: ") + what + ": error " + std::to_string(static_cast<int>(status)));
  }
}

// Plans are costly to make and are reused for every ping of a collection. A plan belongs
// to the device it was made on, and one plan must not execute from two threads at once,
// so the cache lock is held while a transform runs.
std::mutex g_plan_mutex;
std::map<std::tuple<int, int, int>, cufftHandle> g_plans;

cufftHandle plan_for(int device, int n, int batch) {
  const auto key = std::make_tuple(device, n, batch);
  const auto it = g_plans.find(key);
  if (it != g_plans.end()) return it->second;
  cufftHandle plan = 0;
  int dims[1] = {n};
  cufft_check(cufftPlanMany(&plan, 1, dims, nullptr, 1, n, nullptr, 1, n, CUFFT_C2C, batch), "plan");
  g_plans[key] = plan;
  return plan;
}

void transform(int device, float* data, std::int64_t n, std::int64_t batch, int direction) {
  const std::lock_guard<std::mutex> lock(g_plan_mutex);
  const cufftHandle plan = plan_for(device, static_cast<int>(n), static_cast<int>(batch));
  auto* x = reinterpret_cast<cufftComplex*>(data);
  cufft_check(cufftExecC2C(plan, x, x, direction), "execute");
}

struct LoadKernel {  // real or complex input -> complex work array
  const float* in;
  float* x;
  bool is_real;
  SONAR_HD void operator()(std::int64_t i) const {
    if (is_real) {
      x[2 * i] = in[i];
      x[2 * i + 1] = 0.0F;
    } else {
      x[2 * i] = in[2 * i];
      x[2 * i + 1] = in[2 * i + 1];
    }
  }
};

struct BandKernel {  // keep the band around bin k0, shifted to DC, tapered at its edges
  const float* x;    // [channels, n] spectra
  float* y;          // [channels, n_out]
  std::int64_t n, n_out, k0, half;
  float taper_bins;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t d = i % n_out;
    const std::int64_t c = i / n_out;
    std::int64_t k;
    if (d <= half) {
      k = d;
    } else if (d >= n_out - half) {
      k = d - n_out;
    } else {
      y[2 * i] = 0.0F;
      y[2 * i + 1] = 0.0F;
      return;
    }
    std::int64_t src = (k0 + k) % n;
    if (src < 0) src += n;
    float w = 1.0F;
    const float from_edge = static_cast<float>(half - (k < 0 ? -k : k));
    if (taper_bins > 0.0F && from_edge < taper_bins) {
      w = 0.5F * (1.0F - cosf(3.14159265358979F * from_edge / taper_bins));
    }
    y[2 * i] = x[2 * (c * n + src)] * w;
    y[2 * i + 1] = x[2 * (c * n + src) + 1] * w;
  }
};

struct ScaleKernel {  // multiply by a complex constant
  float* y;
  float re, im;
  SONAR_HD void operator()(std::int64_t i) const {
    const float a = y[2 * i], b = y[2 * i + 1];
    y[2 * i] = a * re - b * im;
    y[2 * i + 1] = a * im + b * re;
  }
};

struct PadKernel {  // [channels, n] -> zero-padded [channels, n_fft]
  const float* in;
  float* x;
  std::int64_t n, n_fft;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n_fft;
    const std::int64_t c = i / n_fft;
    if (k < n) {
      x[2 * i] = in[2 * (c * n + k)];
      x[2 * i + 1] = in[2 * (c * n + k) + 1];
    } else {
      x[2 * i] = 0.0F;
      x[2 * i + 1] = 0.0F;
    }
  }
};

struct MultiplyKernel {  // spectrum times the filter's
  float* x;
  const float* h;  // [n_fft]
  std::int64_t n_fft;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n_fft;
    const float a = x[2 * i], b = x[2 * i + 1];
    const float hr = h[2 * k], hi = h[2 * k + 1];
    x[2 * i] = a * hr - b * hi;
    x[2 * i + 1] = a * hi + b * hr;
  }
};

struct CropKernel {  // first n samples of each padded row, with the inverse's 1/n_fft
  const float* x;
  float* out;
  std::int64_t n, n_fft;
  float scale;
  SONAR_HD void operator()(std::int64_t i) const {
    const std::int64_t k = i % n;
    const std::int64_t c = i / n;
    out[2 * i] = x[2 * (c * n_fft + k)] * scale;
    out[2 * i + 1] = x[2 * (c * n_fft + k) + 1] * scale;
  }
};

// Plans of fft_cuda_device: the most recently used few, the oldest destroyed when a new size
// arrives. The batched stages (reverberation, beamformers) come with sizes that follow the
// scenario, and a plan for a length with large prime factors holds work memory in
// proportion to its batch, so these are not kept for ever.
struct RecentPlan {
  int device, n, batch;
  cufftHandle plan;
  std::uint64_t used;
};
constexpr std::size_t kRecentPlans = 6;
std::vector<RecentPlan> g_recent;
std::uint64_t g_recent_clock = 0;

cufftHandle recent_plan(int device, int n, int batch) {
  for (RecentPlan& p : g_recent) {
    if (p.device == device && p.n == n && p.batch == batch) {
      p.used = ++g_recent_clock;
      return p.plan;
    }
  }
  if (g_recent.size() >= kRecentPlans) {
    std::size_t oldest = 0;
    for (std::size_t i = 1; i < g_recent.size(); ++i) {
      if (g_recent[i].used < g_recent[oldest].used) oldest = i;
    }
    cufftDestroy(g_recent[oldest].plan);
    g_recent.erase(g_recent.begin() + static_cast<std::ptrdiff_t>(oldest));
  }
  cufftHandle plan = 0;
  int dims[1] = {n};
  cufft_check(cufftPlanMany(&plan, 1, dims, nullptr, 1, n, nullptr, 1, n, CUFFT_C2C, batch), "plan");
  g_recent.push_back(RecentPlan{device, n, batch, plan, ++g_recent_clock});
  return plan;
}

struct Staging {
  void* ptr = nullptr;
  std::size_t bytes = 0;
};
Staging g_staging[4];
std::mutex g_staging_mutex;

}  // namespace

void fft_cuda_device(float* data, std::int64_t n, std::int64_t batch, bool inverse, int device) {
  if (n <= 0 || batch <= 0) return;
  if (n > 0x7fffffff || batch > 0x7fffffff) throw std::runtime_error("cuFFT: transform too large");
  const std::lock_guard<std::mutex> lock(g_plan_mutex);
  const cufftHandle plan = recent_plan(device, static_cast<int>(n), static_cast<int>(batch));
  auto* x = reinterpret_cast<cufftComplex*>(data);
  cufft_check(cufftExecC2C(plan, x, x, inverse ? CUFFT_INVERSE : CUFFT_FORWARD), "execute");
  cuda_check(cudaDeviceSynchronize(), "transform");
}

std::mutex& cuda_staging_mutex() { return g_staging_mutex; }

void* cuda_staging(int slot, std::size_t bytes) {
  if (slot < 0 || slot >= 4) throw std::runtime_error("CUDA: no such staging slot");
  Staging& s = g_staging[slot];
  if (s.bytes < bytes) {
    if (s.ptr != nullptr) cudaFreeHost(s.ptr);
    s.ptr = nullptr;
    s.bytes = 0;
    cuda_check(cudaMallocHost(&s.ptr, bytes), "allocate page-locked memory");
    s.bytes = bytes;
  }
  return s.ptr;
}

void baseband_cuda(const float* in, bool is_real, std::int64_t n_channels, std::int64_t n,
                   std::int64_t n_out, std::int64_t k0, std::int64_t half, float taper_bins,
                   float gain, std::complex<float> rot, float* out, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const auto total = static_cast<std::size_t>(n_channels * n);
  const auto total_out = static_cast<std::size_t>(n_channels * n_out);
  const DeviceBuffer<float> d_in(in, total * (is_real ? 1 : 2));
  const DeviceBuffer<float> d_x(total * 2);
  const DeviceBuffer<float> d_y(total_out * 2);
  parallel_for_cuda(n_channels * n, LoadKernel{d_in.data(), d_x.data(), is_real});
  transform(device, d_x.data(), n, n_channels, CUFFT_FORWARD);
  parallel_for_cuda(n_channels * n_out, BandKernel{d_x.data(), d_y.data(), n, n_out, k0, half, taper_bins});
  transform(device, d_y.data(), n_out, n_channels, CUFFT_INVERSE);   // unnormalised: a sum over bins
  const std::complex<float> scale = gain * rot;
  parallel_for_cuda(n_channels * n_out, ScaleKernel{d_y.data(), scale.real(), scale.imag()});
  d_y.download(out);
}

void pulse_compress_cuda(float* data, std::int64_t n_channels, std::int64_t n, std::int64_t n_fft,
                         const std::complex<float>* h, int device) {
  cuda_check(cudaSetDevice(device), "select device");
  const auto total = static_cast<std::size_t>(n_channels * n);
  const DeviceBuffer<float> d_data(data, total * 2);
  const DeviceBuffer<float> d_h(reinterpret_cast<const float*>(h), static_cast<std::size_t>(n_fft) * 2);
  const DeviceBuffer<float> d_x(static_cast<std::size_t>(n_channels * n_fft) * 2);
  parallel_for_cuda(n_channels * n_fft, PadKernel{d_data.data(), d_x.data(), n, n_fft});
  transform(device, d_x.data(), n_fft, n_channels, CUFFT_FORWARD);
  parallel_for_cuda(n_channels * n_fft, MultiplyKernel{d_x.data(), d_h.data(), n_fft});
  transform(device, d_x.data(), n_fft, n_channels, CUFFT_INVERSE);
  parallel_for_cuda(n_channels * n, CropKernel{d_x.data(), d_data.data(), n, n_fft,
                                               1.0F / static_cast<float>(n_fft)});
  d_data.download(data);
}

}  // namespace sonar::dsp
