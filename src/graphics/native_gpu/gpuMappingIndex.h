#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GPUMAPPINGINDEX_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GPUMAPPINGINDEX_H_

#include <atomic>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>

namespace Libs::Graphics::NativeGpu {

// Address/protection metadata only. Never hold this lock across a GPU operation, a fence wait,
// or acquisition of the backing lock. Mapping writers are serialized by GemBacking's lock;
// command preparation can keep querying unrelated ranges while a writer waits for the GPU.
// Invalidate before replacing/freeing a mapping, and publish additions after mapping sync.
class GpuMappingIndex {
public:
	enum class Kind { Direct, External };
	struct Range {
		uint64_t begin;
		uint64_t end;
		bool     writable;
	};
	struct Answer {
		uint64_t begin    = 0;
		uint64_t end      = 0;
		bool     backed   = false;
		bool     mapped   = false;
		bool     writable = false;
	};

	void Replace(Kind kind, Range range) {
		if (range.begin >= range.end) {
			return;
		}
		std::lock_guard lock(m_mutex);
		auto&           ranges = kind == Kind::Direct ? m_direct : m_external;
		EraseUnlocked(ranges, range.begin, range.end);
		ranges.emplace(range.begin, range);
		m_generation.fetch_add(1, std::memory_order_release);
	}

	void Erase(Kind kind, uint64_t begin, uint64_t end) {
		if (begin >= end) {
			return;
		}
		std::lock_guard lock(m_mutex);
		if (EraseUnlocked(kind == Kind::Direct ? m_direct : m_external, begin, end)) {
			m_generation.fetch_add(1, std::memory_order_release);
		}
	}

	[[nodiscard]] Answer Lookup(uint64_t address) const {
		std::lock_guard lock(m_mutex);
		if (const auto* r = Find(m_direct, address)) {
			// Match GemBacking: a CPU view counts as mapped, but becomes GPU-writable only at
			// BindView.
			return {r->begin, r->end, true, true, r->writable};
		}
		if (const auto* r = Find(m_external, address)) {
			return {r->begin, r->end, false, true, r->writable};
		}
		return {};
	}

	[[nodiscard]] uint64_t Generation() const {
		return m_generation.load(std::memory_order_acquire);
	}

private:
	using Ranges = std::map<uint64_t, Range>;
	static const Range* Find(const Ranges& ranges, uint64_t address) {
		auto it = ranges.upper_bound(address);
		if (it == ranges.begin()) {
			return nullptr;
		}
		--it;
		return address < it->second.end ? &it->second : nullptr;
	}

	static bool EraseUnlocked(Ranges& ranges, uint64_t begin, uint64_t end) {
		auto it = ranges.upper_bound(begin);
		if (it != ranges.begin()) {
			--it;
		}
		bool changed = false;
		while (it != ranges.end() && it->first < end) {
			const Range old = it->second;
			if (old.end <= begin) {
				++it;
				continue;
			}
			it      = ranges.erase(it);
			changed = true;
			if (old.begin < begin) {
				ranges.emplace(old.begin, Range {old.begin, begin, old.writable});
			}
			if (old.end > end) {
				ranges.emplace(end, Range {end, old.end, old.writable});
			}
		}
		return changed;
	}

	mutable std::mutex    m_mutex;
	Ranges                m_direct;
	Ranges                m_external;
	std::atomic<uint64_t> m_generation {0};
};

} // namespace Libs::Graphics::NativeGpu

#endif
