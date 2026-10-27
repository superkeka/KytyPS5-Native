#ifndef KERNEL_LONGWAIT_H_
#define KERNEL_LONGWAIT_H_

#include <pthread.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace Libs::LibKernel {

// KYTY_LONG_WAITS=<ms>: every guest wait (event queue, condition variable, mutex, semaphore, event
// flag, sync-on-address, sleep) that blocks at least that long goes to stderr with its CLOCK_MONOTONIC
// span (the native flip trace's time base), the thread, the waited object and the guest caller, to
// line a stall up with the frames it delays. Off (no clock read) when unset.
// KYTY_LONG_WAITS_TRACE=<hex address>[,...]: every wait on those objects is logged whatever its length,
// and so is every signal to them (NoteSignal), to see who wakes a waiter and when.
// KYTY_LONG_WAITS_THREAD=<name>: that thread's waits are logged from 0.5 ms on.
// A logged wait on an object that was signaled while it waited names the last signaler.
class LongWait {
public:
	using Describe = int (*)(const void* arg, char* text, size_t size);

	LongWait(const char* kind, const void* object, const void* caller, uint64_t detail = 0)
	    : m_kind(kind), m_object(object), m_caller(caller), m_detail(detail) {
		if (Threshold() > 0.0) {
			m_t0 = Now();
		}
	}
	LongWait(const LongWait&)            = delete;
	LongWait& operator=(const LongWait&) = delete;
	// What the wait returned, described when it is logged (an event queue's events).
	void SetDescribe(Describe describe, const void* arg) {
		m_describe = describe;
		m_arg      = arg;
	}
	// A signal, broadcast, unlock or trigger on a traced object.
	static void NoteSignal(const char* kind, const void* object, const void* caller, uint64_t detail = 0) {
		if (Threshold() <= 0.0) {
			return;
		}
		Signal&     slot = Slot(object);
		const char* self = ThreadName();
		slot.time.store(0.0, std::memory_order_relaxed);
		slot.object = object;
		slot.caller = caller;
		std::memcpy(slot.thread, self, sizeof(slot.thread));
		slot.time.store(Now(), std::memory_order_release);
		if (!Traced(object)) {
			return;
		}
		char name[32] = "?";
		pthread_getname_np(pthread_self(), name, sizeof(name));
		std::fprintf(stderr, "wait signal: %s %.4f thread %s object %p caller %p detail 0x%llx\n", kind, Now(), name, object, caller,
		             static_cast<unsigned long long>(detail));
	}

	~LongWait() {
		if (m_t0 == 0.0) {
			return;
		}
		const double t1 = Now();
		char         name[32] = "?";
		pthread_getname_np(pthread_self(), name, sizeof(name));
		if (t1 - m_t0 < Threshold() && !Traced(m_object) && !(t1 - m_t0 >= 0.0005 && WatchedThread(name))) {
			return;
		}
		char what[200] = "";
		if (m_describe != nullptr) {
			m_describe(m_arg, what, sizeof(what));
		} else if (const Signal& slot = Slot(m_object); slot.object == m_object) {
			const double t = slot.time.load(std::memory_order_acquire);
			if (t >= m_t0 && t <= t1) {
				std::snprintf(what, sizeof(what), "signaled by %s caller %p at %.4f", slot.thread, slot.caller, t);
			}
		}
		std::fprintf(stderr, "long wait: %s %.1f ms %.4f-%.4f thread %s object %p caller %p detail 0x%llx%s%s\n", m_kind,
		             (t1 - m_t0) * 1e3, m_t0, t1, name, m_object, m_caller, static_cast<unsigned long long>(m_detail),
		             what[0] != '\0' ? " " : "", what);
	}

private:
	struct Signal {
		std::atomic<double> time {0.0};
		const void*         object = nullptr;
		const void*         caller = nullptr;
		char                thread[32] {};
	};
	// The calling thread's name, read again once a second (guest threads are named after they start).
	static const char* ThreadName() {
		thread_local char   name[32] = "?";
		thread_local double read_at  = -1.0;
		if (const double now = Now(); now - read_at > 1.0) {
			pthread_getname_np(pthread_self(), name, sizeof(name));
			read_at = now;
		}
		return name;
	}
	static Signal& Slot(const void* object) {
		static Signal table[4096];
		return table[(reinterpret_cast<uintptr_t>(object) >> 3u) % 4096u];
	}
	static bool WatchedThread(const char* name) {
		static const char* watched = std::getenv("KYTY_LONG_WAITS_THREAD");
		return watched != nullptr && std::strcmp(watched, name) == 0;
	}
	static bool Traced(const void* object) {
		static const auto traced = [] {
			struct List {
				uint64_t addr[8] {};
				int      n = 0;
			} list;
			const char* v = std::getenv("KYTY_LONG_WAITS_TRACE");
			while (v != nullptr && *v != '\0' && list.n < 8) {
				char* end = nullptr;
				list.addr[list.n++] = std::strtoull(v, &end, 16);
				v = (end != nullptr && *end == ',') ? end + 1 : nullptr;
			}
			return list;
		}();
		for (int i = 0; i < traced.n; i++) {
			if (traced.addr[i] == reinterpret_cast<uint64_t>(object)) {
				return true;
			}
		}
		return false;
	}
	static double Threshold() {
		static const double threshold = [] {
			const char* v = std::getenv("KYTY_LONG_WAITS");
			return v == nullptr ? 0.0 : std::atof(v) * 1e-3;
		}();
		return threshold;
	}
	static double Now() {
		struct timespec ts {};
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
	}

	const char* m_kind;
	const void* m_object;
	const void* m_caller;
	uint64_t    m_detail;
	double      m_t0       = 0.0;
	Describe    m_describe = nullptr;
	const void* m_arg      = nullptr;
};

} // namespace Libs::LibKernel

#endif /* KERNEL_LONGWAIT_H_ */
