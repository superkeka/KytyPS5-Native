#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLED g_print_name

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLE(flag) PRINT_NAME_ENABLED = flag;

namespace Libs {
// KYTY_TRACE_HLE=all, or a comma-separated list of library names (libkernel, Graphics5Driver,
// VideoOut, ...): every HLE call of those libraries is logged on every thread.
// KYTY_TRACE_HLE_SKIP: comma-separated substrings of function names left out (MutexLock, ...).
inline bool ListHas(const char* list, const char* name, bool substring) {
	if (list == nullptr || name == nullptr) {
		return false;
	}
	const size_t n = std::strlen(name);
	for (const char* p = list; *p != '\0';) {
		const char*  end = std::strchr(p, ',');
		const size_t len = end != nullptr ? static_cast<size_t>(end - p) : std::strlen(p);
		if (len != 0 && (substring ? std::string_view(name).find(std::string_view(p, len)) != std::string_view::npos
		                           : (len == n && std::strncmp(p, name, len) == 0))) {
			return true;
		}
		p += len + (end != nullptr ? 1 : 0);
	}
	return false;
}
// KYTY_COUNT_HLE=all or a list of library names, like KYTY_TRACE_HLE: those calls are counted instead
// of logged (DumpHleCounts), to see which HLE functions a frame runs and how often.
inline const char* CountHleList() {
	static const char* list = std::getenv("KYTY_COUNT_HLE");
	return list;
}
inline bool PrintNameWanted(const char* library) {
	static const char* list  = std::getenv("KYTY_TRACE_HLE");
	const char*        count = CountHleList();
	return (list != nullptr && (std::strcmp(list, "all") == 0 || ListHas(list, library, false))) ||
	       (count != nullptr && (std::strcmp(count, "all") == 0 || ListHas(count, library, false)));
}
// Call counts by function name (the address of its __func__), open addressing without locks.
struct HleCount {
	std::atomic<const char*> function {nullptr};
	std::atomic<uint64_t>    calls {0};
};
inline HleCount* HleCounts() {
	static HleCount table[2048];
	return table;
}
inline void CountHleCall(const char* function) {
	HleCount*    table = HleCounts();
	const size_t start = (reinterpret_cast<uintptr_t>(function) >> 3u) % 2048u;
	for (size_t i = 0; i < 2048u; i++) {
		HleCount&   slot = table[(start + i) % 2048u];
		const char* cur  = slot.function.load(std::memory_order_acquire);
		if (cur == nullptr && slot.function.compare_exchange_strong(cur, function, std::memory_order_acq_rel)) {
			cur = function;
		}
		if (cur == function) {
			slot.calls.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
}
// The counted calls per `frames`, most frequent first, then the counts start again.
inline void DumpHleCounts(std::FILE* f, double frames, size_t top) {
	HleCount*                                    table = HleCounts();
	std::vector<std::pair<uint64_t, const char*>> calls;
	for (size_t i = 0; i < 2048u; i++) {
		if (const char* fn = table[i].function.load(std::memory_order_acquire); fn != nullptr) {
			if (const uint64_t n = table[i].calls.exchange(0, std::memory_order_relaxed); n != 0) {
				calls.emplace_back(n, fn);
			}
		}
	}
	std::sort(calls.rbegin(), calls.rend());
	std::fprintf(f, "hle calls per frame:");
	for (size_t i = 0; i < calls.size() && i < top; i++) {
		std::fprintf(f, " %s:%.1f", calls[i].second, static_cast<double>(calls[i].first) / frames);
	}
	std::fprintf(f, "\n");
}
inline bool PrintNameSkipped(const char* function) {
	static const char* list = std::getenv("KYTY_TRACE_HLE_SKIP");
	return ListHas(list, function, true);
}
} // namespace Libs

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_DEFINE(name) void name(Loader::SymbolDatabase* s)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_NAME(l, m)                                                                             \
	[[maybe_unused]] static thread_local bool PRINT_NAME_ENABLED = ::Libs::PrintNameWanted(l);     \
	static constexpr char                     g_library[]        = l;                              \
	static constexpr char                     g_module[]         = m;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_VERSION(l, lv, m, mv1, mv2)                                                            \
	LIB_NAME(l, m);                                                                                \
	static constexpr int g_library_version      = lv;                                              \
	static constexpr int g_module_version_major = mv1;                                             \
	static constexpr int g_module_version_minor = mv2;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_ADD(n, f, t)                                                                           \
	{                                                                                              \
		Loader::SymbolResolve sr {};                                                               \
		sr.name                 = n;                                                               \
		sr.library              = g_library;                                                       \
		sr.library_version      = g_library_version;                                               \
		sr.module               = g_module;                                                        \
		sr.module_version_major = g_module_version_major;                                          \
		sr.module_version_minor = g_module_version_minor;                                          \
		sr.type                 = t;                                                               \
		auto        func        = reinterpret_cast<uint64_t>(f);                                   \
		const char* dbg_name    = "" #f;                                                           \
		s->Add(sr, func, dbg_name);                                                                \
	}
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_OBJECT(n, f) LIB_ADD(n, f, Loader::SymbolType::Object)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f) LIB_ADD(n, f, Loader::SymbolType::Func)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	if (PRINT_NAME_ENABLED) {                                                                      \
		Libs::PrintName(g_library, g_module, __func__);                                              \
	}

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
[[gnu::noinline]] inline void PrintName(const char* library, const char* module, const char* function) {
	if (CountHleList() != nullptr) {
		CountHleCall(function);
		return;
	}
	if (Log::GetDirection() != Log::Direction::Silent && !PrintNameSkipped(function)) {
		const auto elapsed_ms      = static_cast<uint64_t>(Loader::Timer::GetTimeMs());
		const auto print_name_time = fmt::format(
		    "{:02}:{:02}:{:02}.{:03}", elapsed_ms / 3600000, (elapsed_ms / 60000) % 60,
		    (elapsed_ms / 1000) % 60, elapsed_ms % 1000);
		LOGF_COLOR(Log::Color::Cyan, "[%d][%s] %s::%s::%s()\n",
		           Common::Thread::GetThreadIdUnique(), print_name_time.c_str(), library, module,
		           function);
	}
}

void InitAll(Loader::SymbolDatabase* s);

} // namespace Libs
#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_ */
