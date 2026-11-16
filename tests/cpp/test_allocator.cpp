// Slab allocator (ASASIN Sec. II-C).
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "sonar/core/allocator.hpp"

using namespace sonar;

namespace {
struct FakeStage : Stage {
  std::size_t a, b;
  FakeStage(std::size_t a_, std::size_t b_) : a(a_), b(b_) {}
  std::string name() const override { return "fake"; }
  std::size_t memory_required(std::size_t alignment) const override {
    return align_up(a, alignment) + align_up(b, alignment);
  }
};
}  // namespace

TEST(SlabAllocator, StagesReportThenOneSlabServesThemAll) {
  SlabAllocator slab(Backend::Host, 256);
  const FakeStage s1(1000, 10), s2(4096, 1);
  slab.reserve(s1.memory_required(slab.alignment()));
  slab.reserve(s2.memory_required(slab.alignment()));
  EXPECT_FALSE(slab.committed());
  slab.commit();
  EXPECT_TRUE(slab.committed());
  EXPECT_EQ(slab.capacity(), 1024U + 256U + 4096U + 256U);

  auto* p1 = slab.allocate_array<float>(250);   // 1000 bytes
  auto* p2 = slab.allocate_array<char>(10);
  auto* p3 = slab.allocate_array<std::uint8_t>(4096);
  auto* p4 = slab.allocate_array<char>(1);
  // Blocks are aligned, distinct and inside one contiguous slab.
  for (void* p : {static_cast<void*>(p1), static_cast<void*>(p2), static_cast<void*>(p3), static_cast<void*>(p4)}) {
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 256U, 0U);
  }
  EXPECT_EQ(reinterpret_cast<char*>(p2) - reinterpret_cast<char*>(p1), 1024);
  EXPECT_EQ(reinterpret_cast<char*>(p4) - reinterpret_cast<char*>(p1), 1024 + 256 + 4096);
  EXPECT_EQ(slab.used(), slab.capacity());
  std::memset(p1, 0, 1000);  // writable
  std::memset(p3, 0xff, 4096);
}

TEST(SlabAllocator, UnderReportingIsCaughtLoudly) {
  SlabAllocator slab;
  slab.reserve(1000);
  slab.commit();
  slab.allocate(1000);
  EXPECT_THROW(slab.allocate(1), std::runtime_error);
}

TEST(SlabAllocator, ProtocolOrderIsEnforced) {
  SlabAllocator slab;
  EXPECT_THROW(slab.allocate(8), std::logic_error);   // before commit
  slab.reserve(64);
  slab.commit();
  EXPECT_THROW(slab.reserve(64), std::logic_error);   // after commit
  EXPECT_THROW(slab.commit(), std::logic_error);      // twice
}

TEST(SlabAllocator, ResetReusesTheSameMemory) {
  SlabAllocator slab;
  slab.reserve(512);
  slab.reserve(512);
  slab.commit();
  void* first = slab.allocate(512);
  slab.allocate(512);
  slab.reset();
  EXPECT_EQ(slab.used(), 0U);
  EXPECT_EQ(slab.allocate(512), first);     // no new allocation: same address
  EXPECT_EQ(slab.high_water(), 1024U);
}

TEST(SlabAllocator, ReleaseStartsOver) {
  SlabAllocator slab;
  slab.reserve(100);
  slab.commit();
  slab.release();
  EXPECT_FALSE(slab.committed());
  slab.reserve(5000);
  slab.commit();
  EXPECT_EQ(slab.capacity(), align_up(5000, 256));
}

TEST(SlabAllocator, RejectsBadAlignmentAndEmptySlabWorks) {
  EXPECT_THROW(SlabAllocator(Backend::Host, 100), std::invalid_argument);
  SlabAllocator slab;
  slab.commit();
  EXPECT_TRUE(slab.committed());
  EXPECT_THROW(slab.allocate(1), std::runtime_error);
}

TEST(SlabAllocator, DeviceBackendNeedsCudaBuild) {
  if (cuda_compiled()) {
    GTEST_SKIP() << "CUDA build: device slab is exercised by the GPU tests";
  }
  EXPECT_THROW(SlabAllocator(Backend::Cuda), std::runtime_error);
}
