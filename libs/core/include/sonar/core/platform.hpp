// Host/device portability macros. Kernel code is written once and compiled both
// by the host compiler and by nvcc (docs/DECISIONS.md A-03).
#pragma once

#if defined(__CUDACC__)
#define SONAR_HD __host__ __device__
#define SONAR_DEVICE_BUILD 1
#else
#define SONAR_HD
#define SONAR_DEVICE_BUILD 0
#endif

#define SONAR_INLINE SONAR_HD inline

namespace sonar {

/// Where a kernel runs.
enum class Backend {
  Host,  ///< CPU threads. Always available.
  Cuda,  ///< CUDA device. Available when built with SONAR_HAVE_CUDA and a device is present.
};

/// True when the library was built with the CUDA backend.
bool cuda_compiled();
/// True when the CUDA backend was built and at least one device is usable.
bool cuda_available();
/// Number of usable CUDA devices (0 without the CUDA backend).
int cuda_device_count();
/// Name of a CUDA device, or an empty string.
const char* cuda_device_name(int device);

}  // namespace sonar
