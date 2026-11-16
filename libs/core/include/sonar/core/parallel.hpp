// parallel_for on the host backend. The CUDA counterpart is in parallel_cuda.hpp
// and is only included from .cu files.
#pragma once

#include <cstdint>

namespace sonar {

/// Call f(i) for i in [0, n) on CPU threads. f must be safe to call concurrently
/// for different i.
template <typename F>
void parallel_for_host(std::int64_t n, const F& f) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 16)
#endif
  for (std::int64_t i = 0; i < n; ++i) f(i);
}

}  // namespace sonar
