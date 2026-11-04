#include "kernel/syncOnAddress.h"

#include "common/threads.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
#include <cerrno>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::LibKernel::SyncOnAddress {

namespace {

constexpr uint32_t SIGNAL_POLL_MICROS = 10000;

using Clock = std::chrono::steady_clock;

template <typename T>
[[nodiscard]] bool IsValidWaitAddress(const volatile T* address) {
	return address != nullptr && (reinterpret_cast<uintptr_t>(address) & (alignof(T) - 1u)) == 0;
}

// The top four address bits carry the wait's width in the console's system call.
[[nodiscard]] bool IsWaitAddressInRange(const volatile void* address) {
	return (reinterpret_cast<uintptr_t>(address) >> 60u) == 0;
}

template <typename T>
[[nodiscard]] T ReadWord(const volatile T* address) {
	return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}

struct WaitDeadline {
	bool              finite = false;
	Clock::time_point end {};
};

[[nodiscard]] WaitDeadline MakeDeadline(std::chrono::nanoseconds timeout) {
	const auto now = Clock::now();
	const auto remaining = Clock::time_point::max() - now;
	return {true, timeout >= remaining ? Clock::time_point::max() : now + timeout};
}

[[nodiscard]] uint32_t GetWaitSliceMicros(const WaitDeadline& deadline, bool first_wait) {
	if (!deadline.finite) {
		return SIGNAL_POLL_MICROS;
	}

	const auto now = Clock::now();
	if (now >= deadline.end) {
		return first_wait ? 0u : UINT32_MAX;
	}

	const auto remaining =
	    std::chrono::ceil<std::chrono::microseconds>(deadline.end - now).count();
	return static_cast<uint32_t>(std::min<int64_t>(remaining, SIGNAL_POLL_MICROS));
}

void PollSignals(signal_poll_func_t signal_poll) {
	if (signal_poll != nullptr) {
		signal_poll();
	}
}

// Sleepers per address bucket, for HasWaiters. 8-byte granules so a 64-bit wait and the store to
// either of its words share a bucket.
std::atomic<uint32_t> g_waiters[1024];

std::atomic<uint32_t>& WaiterCount(volatile void* address) {
	const auto granule = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address) >> 3u);
	return g_waiters[(granule * 0x9e3779b1u) >> 22u];
}

struct CountedWaiter {
	std::atomic<uint32_t>& count;
	explicit CountedWaiter(volatile void* address): count(WaiterCount(address)) { count.fetch_add(1, std::memory_order_seq_cst); }
	~CountedWaiter() { count.fetch_sub(1, std::memory_order_release); }
	CountedWaiter(const CountedWaiter&)            = delete;
	CountedWaiter& operator=(const CountedWaiter&) = delete;
};

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)

// The compare and the sleep are one step in the futex, so no wake is lost, and a raised exception comes
// as a host signal, which ends the sleep: it sleeps the whole wait, not in slices that would leave gaps a
// wake of an unchanged value could fall into.
template <typename T>
int WaitLinux(volatile T* address, T expected, const WaitDeadline& deadline,
              signal_poll_func_t signal_poll) {
	const CountedWaiter counted(address);

	// A futex is a 32-bit word: a 64-bit wait sleeps on its low half, a byte or halfword wait on the word
	// holding it.
	auto* word = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(address) & ~uintptr_t {3});

	for (;;) {
		const uint32_t word_value = sizeof(T) >= sizeof(uint32_t) ? static_cast<uint32_t>(expected)
		                                                          : __atomic_load_n(word, __ATOMIC_ACQUIRE);
		if (ReadWord(address) != expected) {
			return OK;
		}

		timespec  remaining {};
		timespec* timeout = nullptr;
		if (deadline.finite) {
			const auto now = Clock::now();
			if (now >= deadline.end) {
				return KERNEL_ERROR_ETIMEDOUT;
			}
			const auto nanos  = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.end - now).count();
			remaining.tv_sec  = static_cast<time_t>(nanos / 1000000000);
			remaining.tv_nsec = static_cast<long>(nanos % 1000000000);
			timeout           = &remaining;
		}

		const long result = syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, word_value, timeout, nullptr, 0);
		if (result == 0) {
			return OK;
		}
		const int wait_error = errno;
		if (wait_error == EAGAIN) {
			if constexpr (sizeof(T) >= sizeof(uint32_t)) {
				return OK;
			}
			// The word changed: it may be a neighbour of the waited byte or halfword.
			continue;
		}
		if (wait_error != ETIMEDOUT && wait_error != EINTR) {
			return KERNEL_ERROR_EINVAL;
		}
		PollSignals(signal_poll);
	}
}

int WakeLinux(volatile void* address, int32_t count) {
	// The legacy FUTEX_WAKE path may wake one waiter when count is zero.
	if (count == 0) {
		return 0;
	}

	const auto result = syscall(SYS_futex, reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(address) & ~uintptr_t {3}),
	                            FUTEX_WAKE_PRIVATE, count, nullptr, nullptr, 0);
	return result < 0 ? KERNEL_ERROR_EINVAL : static_cast<int>(result);
}

#endif

struct PortableWaiter {
	Common::CondVar condition;
	bool            wake_requested = false;
};

struct PortableAddressEntry {
	Common::Mutex              mutex;
	std::list<PortableWaiter*> waiters;
};

struct PortableAddressRegistry {
	std::mutex                                                           mutex;
	std::unordered_map<uintptr_t, std::shared_ptr<PortableAddressEntry>> entries;
};

PortableAddressRegistry& GetPortableRegistry() {
	static PortableAddressRegistry registry;
	return registry;
}

std::shared_ptr<PortableAddressEntry> RegisterPortableWaiter(volatile void*  address,
                                                             PortableWaiter* waiter) {
	auto&           registry = GetPortableRegistry();
	std::lock_guard registry_lock(registry.mutex);
	auto&           entry = registry.entries[reinterpret_cast<uintptr_t>(address)];
	if (!entry) {
		entry = std::make_shared<PortableAddressEntry>();
	}
	entry->mutex.Lock();
	entry->waiters.push_back(waiter);
	return entry;
}

void UnregisterPortableWaiter(volatile void*                               address,
                              const std::shared_ptr<PortableAddressEntry>& entry,
                              PortableWaiter*                              waiter) {
	entry->waiters.remove(waiter);
	const bool empty = entry->waiters.empty();
	entry->mutex.Unlock();
	if (!empty) {
		return;
	}

	auto&           registry = GetPortableRegistry();
	std::lock_guard registry_lock(registry.mutex);
	entry->mutex.Lock();
	const auto it = registry.entries.find(reinterpret_cast<uintptr_t>(address));
	if (it != registry.entries.end() && it->second == entry && entry->waiters.empty()) {
		registry.entries.erase(it);
	}
	entry->mutex.Unlock();
}

template <typename T>
int WaitPortable(volatile T* address, T expected, const WaitDeadline& deadline,
                 signal_poll_func_t signal_poll) {
	PortableWaiter waiter;
	auto           entry      = RegisterPortableWaiter(address, &waiter);
	bool           first_wait = true;
	int            result     = OK;

	while (ReadWord(address) == expected && !waiter.wake_requested) {
		const auto slice_micros = GetWaitSliceMicros(deadline, first_wait);
		if (slice_micros == UINT32_MAX) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		if (slice_micros == 0) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}

		(void)waiter.condition.WaitFor(&entry->mutex, slice_micros);
		entry->mutex.Unlock();
		PollSignals(signal_poll);
		entry->mutex.Lock();

		if (deadline.finite && Clock::now() >= deadline.end && ReadWord(address) == expected &&
		    !waiter.wake_requested) {
			result = KERNEL_ERROR_ETIMEDOUT;
			break;
		}
		first_wait = false;
	}

	UnregisterPortableWaiter(address, entry, &waiter);
	return result;
}

int WakePortable(volatile void* address, int32_t count) {
	if (count == 0) {
		return 0;
	}
	auto&            registry = GetPortableRegistry();
	std::unique_lock registry_lock(registry.mutex);
	const auto       it = registry.entries.find(reinterpret_cast<uintptr_t>(address));
	if (it == registry.entries.end()) {
		return 0;
	}
	auto entry = it->second;
	entry->mutex.Lock();
	registry_lock.unlock();

	int32_t woken = 0;
	for (auto* waiter: entry->waiters) {
		if (!waiter->wake_requested) {
			waiter->wake_requested = true;
			waiter->condition.Signal();
			if (++woken == count) {
				break;
			}
		}
	}
	entry->mutex.Unlock();
	return woken;
}

template <typename T>
int WaitImpl(volatile T* address, T expected, const WaitDeadline& deadline,
             signal_poll_func_t signal_poll) {
	if (!IsValidWaitAddress(address)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (!IsWaitAddressInRange(address)) {
		return KERNEL_ERROR_EFAULT;
	}

	int result = OK;
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	result = WaitLinux(address, expected, deadline, signal_poll);
#else
	result = WaitPortable(address, expected, deadline, signal_poll);
#endif
	PollSignals(signal_poll);
	return result;
}

// The timeout goes in and out: what is left of it after a wake, 0 after a timeout. A timeout of 0 has
// already passed: the wait times out without looking at the value.
template <typename T>
int WaitMicros(volatile T* address, T expected, uint32_t* timeout_micros, signal_poll_func_t signal_poll) {
	if (timeout_micros == nullptr) {
		return WaitImpl(address, expected, WaitDeadline {}, signal_poll);
	}
	if (!IsValidWaitAddress(address)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (!IsWaitAddressInRange(address)) {
		return KERNEL_ERROR_EFAULT;
	}
	if (*timeout_micros == 0) {
		return KERNEL_ERROR_ETIMEDOUT;
	}
	const auto start  = Clock::now();
	const int  result = WaitImpl(address, expected, MakeDeadline(std::chrono::microseconds(*timeout_micros)), signal_poll);
	if (result == KERNEL_ERROR_ETIMEDOUT) {
		*timeout_micros = 0;
	} else if (result == OK) {
		const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
		*timeout_micros    = elapsed >= *timeout_micros ? 0u : *timeout_micros - static_cast<uint32_t>(elapsed);
	}
	return result;
}

} // namespace

int Wait8(volatile uint8_t* address, uint8_t expected, uint32_t* timeout_micros, signal_poll_func_t signal_poll) {
	return WaitMicros(address, expected, timeout_micros, signal_poll);
}

int Wait16(volatile uint16_t* address, uint16_t expected, uint32_t* timeout_micros, signal_poll_func_t signal_poll) {
	return WaitMicros(address, expected, timeout_micros, signal_poll);
}

int Wait32(volatile uint32_t* address, uint32_t expected, uint32_t* timeout_micros, signal_poll_func_t signal_poll) {
	return WaitMicros(address, expected, timeout_micros, signal_poll);
}

int Wait64(volatile uint64_t* address, uint64_t expected, uint32_t* timeout_micros, signal_poll_func_t signal_poll) {
	return WaitMicros(address, expected, timeout_micros, signal_poll);
}

int Wait64(volatile uint64_t* address, uint64_t expected, std::chrono::nanoseconds timeout,
           signal_poll_func_t signal_poll) {
	return WaitImpl(address, expected, MakeDeadline(timeout), signal_poll);
}

bool HasWaiters(volatile void* address) {
	std::atomic_thread_fence(std::memory_order_seq_cst);
	return WaiterCount(address).load(std::memory_order_relaxed) != 0;
}

int Wake(volatile void* address, int32_t count) {
	if (address == nullptr || count < 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if ((reinterpret_cast<uintptr_t>(address) >> 63u) != 0) {
		return 0;
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	return WakeLinux(address, count);
#endif
	return WakePortable(address, count);
}

} // namespace Libs::LibKernel::SyncOnAddress
