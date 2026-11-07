#include "graphics/native_gpu/gpuMappingIndex.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <random>
#include <thread>

using Libs::Graphics::NativeGpu::GpuMappingIndex;
using Kind = GpuMappingIndex::Kind;
using namespace std::chrono_literals;

static void Check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "NativeGpuMappingIndexTests: %s\n", message);
    std::abort();
  }
}

int main() {
  GpuMappingIndex index;
  index.Replace(Kind::External, {100, 300, true});
  index.Replace(Kind::Direct, {150, 250, false});
  Check(index.Lookup(149).writable && !index.Lookup(149).backed,
        "external range");
  Check(index.Lookup(150).mapped && index.Lookup(150).backed &&
            !index.Lookup(150).writable,
        "CPU view precedence");
  const auto generation = index.Generation();
  index.Replace(Kind::Direct, {150, 250, true});
  Check(index.Generation() > generation && index.Lookup(150).writable,
        "bind invalidates cached protection");
  index.Erase(Kind::Direct, 180, 220);
  Check(index.Lookup(179).end == 180 && index.Lookup(220).begin == 220,
        "split preserves both tails");
  Check(!index.Lookup(180).backed && index.Lookup(180).writable,
        "hole exposes external range");
  Check(!index.Lookup(300).mapped && !index.Lookup(99).mapped,
        "half-open endpoints");
  index.Erase(Kind::External, 0, 1000);
  Check(!index.Lookup(180).mapped && index.Lookup(179).backed,
        "independent mapping kinds");

  // Differential coverage against byte-addressed state exercises overlapping
  // replacements, holes, boundary queries and split tails without duplicating
  // the interval implementation.
  GpuMappingIndex random_index;
  std::array<std::array<int, 256>, 2> expected{};
  std::mt19937 rng(73);
  for (int step = 0; step < 4000; ++step) {
    const auto kind = rng() % 2;
    const auto begin = rng() % 256;
    const auto end = begin + 1 + rng() % (256 - begin);
    const int value =
        static_cast<int>(rng() % 3); // absent, read-only, writable
    if (value == 0) {
      random_index.Erase(kind == 0 ? Kind::Direct : Kind::External, begin, end);
    } else {
      random_index.Replace(kind == 0 ? Kind::Direct : Kind::External,
                           {begin, end, value == 2});
    }
    for (auto a = begin; a < end; ++a) {
      expected[kind][a] = value;
    }
    for (uint64_t a = 0; a < 256; ++a) {
      const auto result = random_index.Lookup(a);
      const int wanted = expected[0][a] ? expected[0][a] : expected[1][a];
      Check(result.mapped == (wanted != 0) &&
                result.writable == (wanted == 2) &&
                result.backed == (expected[0][a] != 0),
            "randomized range coverage");
      if (result.mapped) {
        Check(result.begin <= a && a < result.end,
              "answer contains queried address");
      }
    }
  }

  // Hold a writer's simulated GPU fence until the reader completes. This tests
  // progress during the wait (not merely eventual success after a timeout),
  // plus publication ordering.
  GpuMappingIndex concurrent;
  concurrent.Replace(Kind::Direct, {0x1000, 0x2000, true});
  concurrent.Replace(Kind::External, {0x3000, 0x4000, true});
  std::promise<void> pending, release_fence;
  auto released = release_fence.get_future();
  std::thread writer([&] {
    concurrent.Erase(Kind::External, 0x3000, 0x4000);
    pending.set_value();
    released.wait();
    concurrent.Replace(Kind::External, {0x3000, 0x4000, false});
  });
  pending.get_future().wait();
  auto reader = std::async(std::launch::async, [&] {
    for (int i = 0; i < 10000; ++i) {
      Check(concurrent.Lookup(0x1100).writable,
            "unrelated mapping remains usable during fence wait");
      Check(!concurrent.Lookup(0x3100).mapped,
            "replacement remains unpublished until fence wait completes");
    }
  });
  Check(reader.wait_for(2s) == std::future_status::ready,
        "reader must complete before fence is released");
  reader.get();
  release_fence.set_value();
  writer.join();
  Check(concurrent.Lookup(0x3100).mapped && !concurrent.Lookup(0x3100).writable,
        "new protection published after wait");
  std::puts("NativeGpuMappingIndexTests: passed");
}
