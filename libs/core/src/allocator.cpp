#include "sonar/core/allocator.hpp"

#include <cstdlib>
#include <new>
#include <stdexcept>

#if SONAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

namespace sonar {

SlabAllocator::SlabAllocator(Backend backend, std::size_t alignment)
    : backend_(backend), alignment_(alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("SlabAllocator: alignment must be a power of two");
  }
#if !SONAR_HAVE_CUDA
  if (backend == Backend::Cuda) {
    throw std::runtime_error("SlabAllocator: built without the CUDA backend");
  }
#endif
}

SlabAllocator::~SlabAllocator() { release(); }

std::size_t SlabAllocator::aligned(std::size_t bytes) const { return align_up(bytes, alignment_); }

void SlabAllocator::reserve(std::size_t bytes) {
  if (committed_) {
    throw std::logic_error("SlabAllocator: reserve() after commit(); stages must report their "
                           "memory before the slab is allocated");
  }
  reserved_ += aligned(bytes);
}

void SlabAllocator::commit() {
  if (committed_) throw std::logic_error("SlabAllocator: commit() called twice");
  capacity_ = reserved_;
  committed_ = true;
  if (capacity_ == 0) return;
  if (backend_ == Backend::Host) {
    slab_ = std::aligned_alloc(alignment_, capacity_);
    if (slab_ == nullptr) throw std::bad_alloc();
  } else {
#if SONAR_HAVE_CUDA
    // cudaMalloc returns memory aligned to at least 256 bytes.
    if (cudaMalloc(&slab_, capacity_) != cudaSuccess) {
      slab_ = nullptr;
      throw std::runtime_error("SlabAllocator: device allocation of " + std::to_string(capacity_) +
                               " bytes failed");
    }
#endif
  }
}

void* SlabAllocator::allocate(std::size_t bytes) {
  if (!committed_) throw std::logic_error("SlabAllocator: allocate() before commit()");
  const std::size_t need = aligned(bytes);
  if (used_ + need > capacity_) {
    throw std::runtime_error("SlabAllocator: a stage asked for " + std::to_string(bytes) +
                             " bytes but only " + std::to_string(capacity_ - used_) +
                             " of the reserved slab remain; it under-reported its memory need");
  }
  void* p = static_cast<char*>(slab_) + used_;
  used_ += need;
  if (used_ > high_water_) high_water_ = used_;
  return p;
}

void SlabAllocator::reset() { used_ = 0; }

void SlabAllocator::release() {
  if (slab_ != nullptr) {
    if (backend_ == Backend::Host) {
      std::free(slab_);
    } else {
#if SONAR_HAVE_CUDA
      cudaFree(slab_);
#endif
    }
  }
  slab_ = nullptr;
  committed_ = false;
  reserved_ = capacity_ = used_ = 0;
}

}  // namespace sonar
