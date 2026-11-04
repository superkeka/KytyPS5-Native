#include "kernel/syncOnAddress.h"

#include "libs/errno.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Libs::LibKernel::SyncOnAddress::Wait32;
using Libs::LibKernel::SyncOnAddress::Wait64;
using Libs::LibKernel::SyncOnAddress::Wake;

std::atomic<int> g_signal_poll_count{0};

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "SyncOnAddressTests: failed: %s\n", text);
    std::abort();
  }
}

template <typename T> void Store(T *address, T value) {
  std::atomic_ref<T>(*address).store(value, std::memory_order_release);
}

void CountSignalPoll() {
  g_signal_poll_count.fetch_add(1, std::memory_order_relaxed);
}

void TestInvalidAddress() {
  uint32_t timeout = 1;
  Check(Wait32(nullptr, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait32 rejects a null address");
  Check(Wait64(nullptr, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait64 rejects a null address");
  Check(Wake(nullptr, 1) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wake rejects a null address");

  alignas(uint64_t) uint8_t bytes[16] = {};
  auto *misaligned = reinterpret_cast<uint32_t *>(bytes + 1);
  Check(Wait32(misaligned, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait32 rejects a misaligned address");
  Check(Wait64(reinterpret_cast<uint64_t *>(bytes + 4), 0, &timeout) ==
            Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wait64 rejects a misaligned address");
  Check(Wake(misaligned, 1) == 0, "wake takes any address: a byte wait's may be odd");
  uint64_t aligned = 0;
  Check(Wake(&aligned, -1) == Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "wake rejects a negative count");
}

void TestByteAndHalfwordWaits() {
  alignas(uint64_t) uint8_t bytes[8] = {};
  uint32_t timeout = 20000;
  Check(Libs::LibKernel::SyncOnAddress::Wait8(bytes + 1, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "a byte wait on an odd address times out");
  timeout = 1000;
  Check(Libs::LibKernel::SyncOnAddress::Wait8(bytes + 1, 1, &timeout) == OK, "a mismatched byte returns at once");
  auto* half = reinterpret_cast<uint16_t*>(bytes + 2);
  timeout    = 1000;
  Check(Libs::LibKernel::SyncOnAddress::Wait16(half + 1, 1, &timeout) == OK, "a mismatched halfword returns at once");
  Check(Libs::LibKernel::SyncOnAddress::Wait16(reinterpret_cast<uint16_t*>(bytes + 1), 0, &timeout) ==
            Libs::LibKernel::KERNEL_ERROR_EINVAL,
        "a misaligned halfword is rejected");

  // A store to a neighbour in the futex word does not end the wait; the wake of the waited byte does.
  std::atomic<int> result {1};
  std::atomic_bool finished {false};
  std::thread      waiter([&] {
    uint32_t long_timeout = 5000000;
    result.store(Libs::LibKernel::SyncOnAddress::Wait8(bytes + 1, 0, &long_timeout));
    finished.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  __atomic_store_n(bytes, 9, __ATOMIC_RELEASE);
  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  Check(!finished.load(), "a neighbour's store does not end a byte wait");
  __atomic_store_n(bytes + 1, 1, __ATOMIC_RELEASE);
  (void)Wake(bytes + 1, INT_MAX);
  waiter.join();
  Check(result.load() == OK, "the byte wait ends with OK");
}

void TestWakeCountsWaiters() {
  uint32_t         word = 0;
  std::atomic<int> sleeping {0};
  std::thread      waiters[3];
  for (auto& t: waiters) {
    t = std::thread([&] {
      sleeping.fetch_add(1);
      uint32_t timeout = 5000000;
      (void)Wait32(&word, 0, &timeout);
    });
  }
  while (sleeping.load() != 3) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(55));
  // The value stays, so a waiter ends only by a wake: the counts add up to the waiters. One between
  // two slices of its sleep is counted by the next wake.
  int        woken    = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
  while (woken < 3 && std::chrono::steady_clock::now() < deadline) {
    woken += Wake(&word, INT_MAX);
  }
  Check(woken == 3, "wake says how many it woke");
  for (auto& t: waiters) {
    t.join();
  }
  Check(Wake(&word, INT_MAX) == 0, "nobody to wake");
}

void TestMismatchReturnsImmediately() {
  uint32_t word = 7;
  uint64_t word64 = UINT64_C(0x100000000);
  uint32_t timeout = 500000;
  const auto start = std::chrono::steady_clock::now();
  Check(Wait32(&word, 6, &timeout, CountSignalPoll) == OK,
        "mismatch succeeds without parking");
  Check(Wait64(&word64, 0, &timeout, CountSignalPoll) == OK,
        "wait64 compares all 64 bits");
  Check(std::chrono::steady_clock::now() - start <
            std::chrono::milliseconds(100),
        "mismatched value is a fast path");
  Check(g_signal_poll_count.load(std::memory_order_relaxed) >= 2,
        "mismatch remains a guest signal safe-point");
}

void TestTimeout() {
  uint32_t word = 0;
  uint32_t timeout = 20000;
  const auto start = std::chrono::steady_clock::now();
  Check(Wait32(&word, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "matching value times out");
  const auto elapsed = std::chrono::steady_clock::now() - start;
  Check(elapsed >= std::chrono::milliseconds(10),
        "timeout does not return too early");
  Check(elapsed < std::chrono::milliseconds(500), "timeout remains bounded");

  timeout = 0;
  Check(Wait32(&word, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "zero timeout polls a matching value");
  word = 1;
  Check(Wait32(&word, 0, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "zero timeout does not look at the value");

  timeout = 20000;
  Check(Wait32(&word, 1, &timeout) == Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT && timeout == 0,
        "a timeout leaves nothing of it");
  timeout = 5000000;
  Check(Wait32(&word, 0, &timeout) == OK && timeout > 4000000 && timeout <= 5000000,
        "a mismatched value leaves what is left of the timeout");
}

void TestValueChangeAndWake() {
  uint64_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<bool> ready{false};
  int result = Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;

  std::thread waiter([&] {
    ready.store(true, std::memory_order_release);
    result = Wait64(&word, 0, &timeout);
  });
  while (!ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  Store(&word, UINT64_C(0x100000000));
  Check(Wake(&word, 1) >= 0, "wake-one succeeds");
  waiter.join();
  Check(result == OK, "a value change plus wake releases the waiter");
}

void TestNanosecondTimeout() {
  uint64_t word = 0;
  const auto start = std::chrono::steady_clock::now();
  Check(Wait64(&word, 0, std::chrono::milliseconds(1)) ==
            Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "64-bit timespec wait expires");
  Check(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(1),
        "timespec wait preserves its deadline");
  Check(Wait64(&word, 0, std::chrono::nanoseconds(0)) ==
            Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT,
        "zero timespec polls a matching value");
  Check(Wait64(&word, 1, std::chrono::nanoseconds(0)) == OK,
        "zero timespec succeeds on mismatch");

  g_signal_poll_count.store(0, std::memory_order_relaxed);
  std::thread waker([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(35));
    Store(&word, UINT64_C(0x100000000));
    Check(Wake(&word, 1) >= 0, "wake releases a long timespec wait");
  });
  const int result = Wait64(&word, 0, std::chrono::nanoseconds::max(), CountSignalPoll);
  waker.join();
  Check(result == OK, "long timespec wait does not overflow its deadline");
  Check(g_signal_poll_count.load(std::memory_order_relaxed) >= 1,
        "a wait ends at a guest signal safe-point");
}

void TestWakeOneThenAll() {
  constexpr int WAITER_COUNT = 4;
  uint32_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<int> ready{0};
  std::atomic<int> returned{0};
  int results[WAITER_COUNT] = {};
  std::vector<std::thread> waiters;

  for (int i = 0; i < WAITER_COUNT; i++) {
    waiters.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      results[i] = Wait32(&word, 0, &timeout);
      returned.fetch_add(1, std::memory_order_release);
    });
  }
  while (ready.load(std::memory_order_acquire) != WAITER_COUNT) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(55));
  Check(Wake(&word, 1) >= 0, "wake-one succeeds with multiple waiters");
  const auto one_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (returned.load(std::memory_order_acquire) == 0 &&
         std::chrono::steady_clock::now() < one_deadline) {
    std::this_thread::yield();
  }
  Check(returned.load(std::memory_order_acquire) == 1,
        "wake-one releases exactly one waiter");

  Check(Wake(&word, 2) >= 0, "wake-two succeeds");
  const auto two_deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (returned.load(std::memory_order_acquire) < 3 &&
         std::chrono::steady_clock::now() < two_deadline) {
    std::this_thread::yield();
  }
  Check(returned.load(std::memory_order_acquire) == 3,
        "wake-two releases exactly two more waiters");

  Check(Wake(&word, INT_MAX) >= 0, "wake-all succeeds");
  for (auto &waiter : waiters) {
    waiter.join();
  }
  Check(returned.load(std::memory_order_acquire) == WAITER_COUNT,
        "wake-all releases the remaining waiters");
  for (int result : results) {
    Check(result == OK, "explicitly woken waiters return success");
  }
}

void TestAddressesAreIsolated() {
  uint32_t first = 0;
  uint32_t second = 0;
  uint32_t first_timeout = 1000000;
  uint32_t second_timeout = 1000000;
  std::atomic<int> ready{0};
  std::atomic<bool> first_returned{false};
  std::atomic<bool> second_returned{false};
  int first_result = 0;
  int second_result = 0;

  std::thread first_waiter([&] {
    ready.fetch_add(1, std::memory_order_release);
    first_result = Wait32(&first, 0, &first_timeout);
    first_returned.store(true, std::memory_order_release);
  });
  std::thread second_waiter([&] {
    ready.fetch_add(1, std::memory_order_release);
    second_result = Wait32(&second, 0, &second_timeout);
    second_returned.store(true, std::memory_order_release);
  });
  while (ready.load(std::memory_order_acquire) != 2) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(55));
  Check(Wake(&first, 1) >= 0, "first address wakes");
  std::this_thread::sleep_for(std::chrono::milliseconds(55));
  Check(first_returned.load(std::memory_order_acquire),
        "first address waiter returned");
  Check(!second_returned.load(std::memory_order_acquire),
        "waking one address does not release another address");
  Check(Wake(&second, 1) >= 0, "second address wakes");

  first_waiter.join();
  second_waiter.join();
  Check(first_result == OK && second_result == OK,
        "isolated waiters return success");
}

void TestCompareRegisterWakeRace() {
  for (int i = 0; i < 100; i++) {
    uint32_t word = 0;
    uint32_t timeout = 500000;
    std::atomic<bool> ready{false};
    int result = Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;
    std::thread waiter([&] {
      ready.store(true, std::memory_order_release);
      result = Wait32(&word, 0, &timeout);
    });
    while (!ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    Store(&word, uint32_t{1});
    (void)Wake(&word, 1);
    waiter.join();
    Check(result == OK, "compare/register/wake race never loses progress");
  }
}

void TestWakeZeroIsNoOp() {
  constexpr int WAITER_COUNT = 2;
  uint32_t word = 0;
  uint32_t timeout = 1000000;
  std::atomic<int> ready{0};
  std::atomic<int> returned{0};
  int results[WAITER_COUNT] = {};
  std::vector<std::thread> waiters;

  for (int i = 0; i < WAITER_COUNT; i++) {
    waiters.emplace_back([&, i] {
      ready.fetch_add(1, std::memory_order_release);
      results[i] = Wait32(&word, 0, &timeout);
      returned.fetch_add(1, std::memory_order_release);
    });
  }
  while (ready.load(std::memory_order_acquire) != WAITER_COUNT) {
    std::this_thread::yield();
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(55));
  Check(Wake(&word, 0) >= 0, "zero-count wake succeeds");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  Check(returned.load(std::memory_order_acquire) == 0,
        "zero-count wake releases no waiters");
  Check(Wake(&word, INT_MAX) >= 0, "wake-all succeeds after zero-count wake");
  for (auto &waiter : waiters) {
    waiter.join();
  }
  for (int result : results) {
    Check(result == OK, "wake-all releases waiters after zero-count no-op");
  }
}

} // namespace

int main() {
  TestInvalidAddress();
  TestMismatchReturnsImmediately();
  TestTimeout();
  TestByteAndHalfwordWaits();
  TestWakeCountsWaiters();
  TestValueChangeAndWake();
  TestNanosecondTimeout();
  TestWakeOneThenAll();
  TestAddressesAreIsolated();
  TestCompareRegisterWakeRace();
  TestWakeZeroIsNoOp();
  std::printf("SyncOnAddressTests: all passed\n");
  return 0;
}
