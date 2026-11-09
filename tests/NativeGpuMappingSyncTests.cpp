// CPU-only integration regression: exercise the real GemBacking mapping methods
// with DRM ioctls replaced at link time. No render node, GPU submission or
// kernel settings are used.
#include "graphics/native_gpu/gemBacking.h"

#include <chrono>
#include <cstdarg>
#include <future>

using namespace std::chrono_literals;
using Libs::Graphics::NativeGpu::GemBacking;

static std::promise<void> fence_entered, release_fence;
static auto fence_release = release_fence.get_future();
static std::atomic<bool> block_fence{false};
static std::atomic<unsigned> fence_calls{0};

extern "C" int __wrap_ioctl(int, unsigned long request, ...) {
  va_list args;
  va_start(args, request);
  void *argument = va_arg(args, void *);
  va_end(args);
  if (request == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) {
    ++fence_calls;
    if (block_fence.exchange(false)) {
      fence_entered.set_value();
      fence_release.wait();
    }
  } else if (request == DRM_IOCTL_AMDGPU_GEM_USERPTR) {
    static uint32_t handle = 100;
    static_cast<drm_amdgpu_gem_userptr *>(argument)->handle = handle++;
  }
  return 0;
}

static void Check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "NativeGpuMappingSyncTests: %s\n", message);
    std::abort();
  }
}

namespace Libs::Graphics::NativeGpu {
struct GemBackingMappingTest {
  static std::unique_ptr<GemBacking> Make() {
    std::unique_ptr<GemBacking> g(new GemBacking);
    g->m_size = GemBacking::BlockSize();
    g->m_mem_fd = ::memfd_create("mapping-sync-test", MFD_CLOEXEC);
    Check(g->m_mem_fd >= 0 && ::ftruncate(g->m_mem_fd, g->m_size) == 0,
          "memfd backing");
    g->m_blocks.resize(1);
    g->m_blocks[0].size = g->m_size;
    g->m_va_syncobj = 1; // all DRM calls are intercepted by __wrap_ioctl
    return g;
  }
};
} // namespace Libs::Graphics::NativeGpu

int main() {
  auto gem = Libs::Graphics::NativeGpu::GemBackingMappingTest::Make();
  void *reservation = ::mmap(nullptr, 0x8000, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  Check(reservation != MAP_FAILED, "reserve host views");
  const auto base = reinterpret_cast<uint64_t>(reservation);
  Check(gem->MapView(base, 0x2000, 0, PROT_READ | PROT_WRITE), "MapView");
  Check(gem->IsBacked(base) && gem->IsGpuMapped(base) &&
            !gem->IsGpuWritable(base),
        "CPU view publication");
  Check(gem->BindView(base, 0x2000, 0), "first BindView");
  Check(gem->IsGpuWritable(base) && fence_calls == 1,
        "binding retains its fence wait");
  Check(gem->MapView(base + 0x4000, 0x2000, 0x4000, PROT_READ | PROT_WRITE),
        "second MapView");
  block_fence = true;
  auto writer = std::async(std::launch::async, [&] {
    return gem->BindView(base + 0x4000, 0x2000, 0x4000);
  });
  Check(fence_entered.get_future().wait_for(2s) == std::future_status::ready,
        "writer reached fence");
  auto reader = std::async(std::launch::async, [&] {
    uint64_t lo = 0, hi = 0;
    Check(gem->IsGpuMapped(base) && gem->IsBacked(base),
          "mapping queries while backing lock is held");
    Check(gem->GpuWritableRange(base + 4, &lo, &hi) && lo == base &&
              hi == base + 0x2000,
          "writable range while backing lock is held");
    Check(!gem->IsGpuMapped(base + 0x4000), "pending bind not published early");
  });
  Check(reader.wait_for(2s) == std::future_status::ready,
        "queries must finish before GPU fence is released");
  reader.get();
  release_fence.set_value();
  Check(writer.get() && fence_calls == 2,
        "one fence wait per bind, including explicit Finish");
  Check(gem->IsGpuWritable(base + 0x4000), "binding published after wait");
  const auto generation = gem->ViewsGeneration();
  Check(gem->UnbindView(base + 0x1000, 0x4000), "partial unbind");
  Check(gem->ViewsGeneration() > generation, "unbind invalidates query cache");
  Check(gem->IsGpuWritable(base) && !gem->IsGpuMapped(base + 0x1000) &&
            !gem->IsGpuMapped(base + 0x4000) &&
            gem->IsGpuWritable(base + 0x5000),
        "unbound hole and surviving tails");

  // External registration and a partial removal recreate both surviving
  // buffers. The index must publish those tails and their protection, with no
  // stale coverage of the removed hole.
  Check(gem->BindExternal(base + 0x2000, 0x3000, true), "external bind");
  Check(gem->IsGpuWritable(base + 0x3000) && !gem->IsBacked(base + 0x3000),
        "external query");
  Check(gem->UnbindExternal(base + 0x3000, 0x1000), "external partial unbind");
  Check(gem->IsGpuWritable(base + 0x2000) && !gem->IsGpuMapped(base + 0x3000) &&
            gem->IsGpuWritable(base + 0x4000),
        "external surviving tails");
  Check(gem->BindExternal(base + 0x2000, 0x1000, false),
        "external protection replacement");
  Check(gem->IsGpuMapped(base + 0x2000) && !gem->IsGpuWritable(base + 0x2000),
        "external read-only protection");
  gem.reset();
  ::munmap(reservation, 0x8000);
  std::puts("NativeGpuMappingSyncTests: passed (mock DRM; no GPU used)");
}
