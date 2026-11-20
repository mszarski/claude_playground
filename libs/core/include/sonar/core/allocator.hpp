// Slab memory allocator (ASASIN Sec. II-C; DECISIONS.md D-08).
//
// A user states every block it will need before anything runs. One slab is then
// allocated to cover all requests, and the blocks are taken from it, aligned, so the
// user makes a single device allocation and does not fragment GPU memory that other
// processes (a display, a GIS tool) are also using.
//
// Only the CUDA backprojection (libs/recon/src/backproject_cuda.cu) uses it: one slab
// per call, plus one per device for the inputs kept resident between calls (its
// trajectory texture is a separate cudaMallocArray). The other CUDA code (detect, dsp,
// live, sim) allocates per buffer through DeviceBuffer (parallel_cuda.hpp) and does not
// go through this allocator.
//
// Protocol:
//   1. reserve(bytes) once per future allocation, while the pipeline is built;
//   2. commit() makes the one allocation;
//   3. allocate(bytes) hands out blocks; asking for more than was reserved throws;
//   4. reset() rewinds so the next run reuses the slab.
#pragma once

#include <cstddef>
#include <string>

#include "sonar/core/platform.hpp"

namespace sonar {

class SlabAllocator {
 public:
  explicit SlabAllocator(Backend backend = Backend::Host, std::size_t alignment = 256);
  ~SlabAllocator();
  SlabAllocator(const SlabAllocator&) = delete;
  SlabAllocator& operator=(const SlabAllocator&) = delete;

  /// Phase 1: announce a future allocation of `bytes`.
  void reserve(std::size_t bytes);
  /// Make the single slab allocation covering everything reserved so far.
  void commit();
  /// Phase 2: an aligned block of `bytes` from the slab.
  void* allocate(std::size_t bytes);
  template <typename T>
  T* allocate_array(std::size_t count) {
    return static_cast<T*>(allocate(count * sizeof(T)));
  }
  /// Hand every block back. The slab itself stays allocated.
  void reset();
  /// Free the slab and forget all reservations.
  void release();

  Backend backend() const { return backend_; }
  std::size_t alignment() const { return alignment_; }
  bool committed() const { return slab_ != nullptr || (committed_ && capacity_ == 0); }
  std::size_t reserved() const { return reserved_; }    ///< bytes announced (aligned)
  std::size_t capacity() const { return capacity_; }    ///< bytes in the slab
  std::size_t used() const { return used_; }            ///< bytes handed out since reset
  std::size_t high_water() const { return high_water_; }

 private:
  std::size_t aligned(std::size_t bytes) const;

  Backend backend_;
  std::size_t alignment_;
  void* slab_ = nullptr;
  bool committed_ = false;
  std::size_t reserved_ = 0;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
  std::size_t high_water_ = 0;
};

/// Base class of processing stages that use slab memory.
class Stage {
 public:
  virtual ~Stage() = default;
  virtual std::string name() const = 0;
  /// Bytes of slab memory this stage will allocate, including alignment padding
  /// (round each block up to the allocator's alignment).
  virtual std::size_t memory_required(std::size_t alignment) const = 0;
};

/// Round `bytes` up to a multiple of `alignment`.
inline std::size_t align_up(std::size_t bytes, std::size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

}  // namespace sonar
