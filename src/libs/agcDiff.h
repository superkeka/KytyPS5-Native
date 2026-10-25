#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_AGC_DIFF_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_AGC_DIFF_H_

#include "common/abi.h"

#include <csetjmp>
#include <cstdint>
#include <tuple>
#include <type_traits>

// Differential check of the HLE libSceAgc against the real one. With KYTY_AGC_REFERENCE naming a
// decrypted libSceAgc ELF, the emulator loads it as a reference module (see
// Loader::Program::reference_only) and every wrapped export runs twice: the HLE function on the
// title's data, then the reference on a private copy, and the results are compared.
//  - first argument a CommandBuffer*: the dwords each one writes (from both cursors);
//  - first argument a uint32_t* packet: the packet after each one patched its own copy;
//  - only arithmetic arguments: the return value.
// Everything else runs the HLE function only. Mismatches go to stderr (KYTY_AGC_DIFF_LOG=N: the
// first N per export, default 4) with a per-export summary at exit.

namespace Libs::Graphics::Gen5 {

struct CommandBuffer;

namespace AgcDiff {

// KYTY_AGC_REFERENCE is set (decided when the exports are registered).
bool  Requested();
// key: the wrapper registered for the export (Wrap<F, Tag>::Call).
void  Register(const void* key, const char* nid, const char* name);
// The reference function for an export, nullptr if the reference is not loaded or lacks it.
void* Reference(const void* key);

struct CbCall {
	const void*    hle     = nullptr;
	CommandBuffer* buf     = nullptr;
	CommandBuffer* shadow  = nullptr;
	uint32_t*      up0     = nullptr;
	uint32_t*      down0   = nullptr;
	uint32_t*      bottom0 = nullptr;
	uint64_t       stubs0  = 0;
};
// Snapshots the title's buffer and returns a private buffer for the reference (nullptr: skip).
CommandBuffer* CbBegin(CbCall* c, const void* hle, CommandBuffer* buf);
// After the HLE function: fills as much of the private buffer as it wrote (plus a margin) with a
// sentinel, so dwords the reference leaves for the title to fill are not compared.
void           CbBeforeReference(CbCall* c);
void           CbEnd(CbCall* c, bool has_ret, bool ret_is_ptr, uint64_t hle_ret, uint64_t ref_ret);

struct PatchCall {
	const void* hle    = nullptr;
	uint32_t*   cmd    = nullptr;
	uint64_t    stubs0 = 0;
};
// Copies the packet before the HLE function patches it; returns the reference's copy.
uint32_t* PatchBegin(PatchCall* c, const void* hle, uint32_t* cmd);
void      PatchEnd(PatchCall* c, bool has_ret, uint64_t hle_ret, uint64_t ref_ret);

void Value(const void* hle, uint64_t stubs0, uint64_t hle_ret, uint64_t ref_ret);
uint64_t StubCalls();
// AgcInit runs in the reference too (once, on a state of its own): its start state is what the
// reference's queue resets and context state ops load.
uint32_t* InitState(const void* key);

// A reference call that faults comes back here (the host fault handler jumps to t_reference_jmp)
// and its export is not called again.
extern thread_local sigjmp_buf* t_reference_jmp;
extern thread_local uint64_t    t_fault_pc;
extern thread_local uint64_t    t_fault_addr;
void Faulted(const void* key);

template <typename Fn>
bool Guarded(const void* key, Fn&& fn) {
	sigjmp_buf jb;
	if (sigsetjmp(jb, 1) != 0) {
		t_reference_jmp = nullptr;
		Faulted(key);
		return false;
	}
	t_reference_jmp = &jb;
	fn();
	t_reference_jmp = nullptr;
	return true;
}

template <typename T>
uint64_t ToU64(T v) {
	if constexpr (std::is_pointer_v<T>) {
		return reinterpret_cast<uint64_t>(v);
	} else if constexpr (std::is_enum_v<T>) {
		return static_cast<uint64_t>(v);
	} else {
		return static_cast<uint64_t>(v);
	}
}

// One instance per registration (Tag): the same HLE function can stand for two exports (DCB and
// asynchronous variants) that the real library implements separately.
template <auto F, int Tag>
struct Wrap;

template <typename R, typename... A, R (*F)(A...), int Tag>
struct Wrap<F, Tag> {
	static KYTY_SYSV_ABI R Call(A... a) {
		using Fn          = R (*)(A...);
		const void* key   = reinterpret_cast<const void*>(&Call);
		auto*       ref   = reinterpret_cast<Fn>(Reference(key));
		using First       = std::tuple_element_t<0, std::tuple<A..., void>>;
		constexpr bool cb = sizeof...(A) > 0 && std::is_same_v<First, CommandBuffer*>;
		constexpr bool pk = sizeof...(A) > 0 && std::is_same_v<First, uint32_t*>;
		constexpr bool pure = ((std::is_arithmetic_v<A> || std::is_enum_v<A>) && ...) && !std::is_void_v<R> &&
		                      !std::is_pointer_v<R>;
		if (ref == nullptr) {
			return F(a...);
		}
		if constexpr (cb || pk) {
			auto        args  = std::make_tuple(a...);
			First       first = std::get<0>(args);
			auto        swap  = [&](First alt) {
                return std::apply([&](First, auto... rest) { return ref(alt, rest...); }, args);
			};
			if constexpr (cb) {
				CbCall c;
				CommandBuffer* shadow = CbBegin(&c, key, first);
				if (shadow == nullptr) {
					return F(a...);
				}
				if constexpr (std::is_void_v<R>) {
					F(a...);
					CbBeforeReference(&c);
					if (Guarded(key, [&] { swap(shadow); })) {
						CbEnd(&c, false, false, 0, 0);
					}
				} else {
					R r = F(a...);
					R l {};
					CbBeforeReference(&c);
					if (Guarded(key, [&] { l = swap(shadow); })) {
						CbEnd(&c, true, std::is_pointer_v<R>, ToU64(r), ToU64(l));
					}
					return r;
				}
			} else {
				PatchCall c;
				uint32_t* copy = PatchBegin(&c, key, first);
				if (copy == nullptr) {
					if constexpr (std::is_void_v<R>) {
						F(a...);
						if (uint32_t* state = InitState(key); state != nullptr) {
							Guarded(key, [&] { swap(state); });
						}
						return;
					} else {
						R r = F(a...);
						if (uint32_t* state = InitState(key); state != nullptr) {
							Guarded(key, [&] { swap(state); });
						}
						return r;
					}
				}
				if constexpr (std::is_void_v<R>) {
					F(a...);
					if (Guarded(key, [&] { swap(copy); })) {
						PatchEnd(&c, false, 0, 0);
					}
				} else {
					R r = F(a...);
					R l {};
					if (Guarded(key, [&] { l = swap(copy); })) {
						PatchEnd(&c, !std::is_pointer_v<R>, ToU64(r), ToU64(l));
					}
					return r;
				}
			}
		} else if constexpr (pure) {
			const uint64_t stubs0 = StubCalls();
			R              r      = F(a...);
			R              l {};
			if (Guarded(key, [&] { l = ref(a...); })) {
				Value(key, stubs0, ToU64(r), ToU64(l));
			}
			return r;
		} else {
			return F(a...);
		}
	}
};

} // namespace AgcDiff

} // namespace Libs::Graphics::Gen5

#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_AGC_DIFF_H_ */
