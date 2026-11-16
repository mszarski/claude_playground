#include "sonar/core/platform.hpp"

#if SONAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace sonar {

bool cuda_compiled() {
#if SONAR_HAVE_CUDA
  return true;
#else
  return false;
#endif
}

bool cuda_available() {
#if SONAR_HAVE_CUDA
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
#else
  return false;
#endif
}

}  // namespace sonar
