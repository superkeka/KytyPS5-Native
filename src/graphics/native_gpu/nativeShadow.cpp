#include "graphics/native_gpu/nativeShadow.h"

#include "graphics/native_gpu/nativePm4.h"
#include "graphics/native_gpu/nativeSubmit.h" // KYTY_GEM_BACKING_SUPPORTED + NativeSubmitter
#include "graphics/native_gpu/shaderPatcher.h"

#include "kernel/memory.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <span>
#include <vector>

namespace Libs::Graphics::NativeGpu {

namespace {

bool EnvOn(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// Shadow-mode state, built lazily on the first DCB so it sees the final env and the created GEM
// backing. One instance for the process; submit_dcb is driven from the guest graphics thread, but
// a mutex keeps it safe if that ever changes.
class Shadow {
public:
	static Shadow& Instance() {
		static Shadow s;
		return s;
	}

	// True once shadow submission is set up and has not disabled itself.
	bool Active() {
		std::lock_guard<std::mutex> lock(m_mutex);
		return m_initialized && m_enabled && !m_disabled;
	}

	void Submit(std::span<const uint32_t> dcb) {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_initialized) {
			Init();
			m_initialized = true;
		}
		if (!m_enabled || m_disabled) {
			return;
		}

		const uint64_t frame = m_frames++;
		Pm4TranslateResult result = m_translator.Translate(dcb);

		DetectShaders(result.shader_addrs);

		if (!result.ok) {
			if (m_refused < 8) {
				std::fprintf(stderr,
				             "native shadow: frame %llu refused, unknown packet op 0x%02x at dw %u "
				             "(%u packets translated)\n",
				             static_cast<unsigned long long>(frame), result.unknown_op,
				             result.unknown_dw, result.stats.packets);
			}
			m_refused++;
			return;
		}

#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter == nullptr) {
			return;
		}
		// Every GEM buffer the GPU may touch must be resident in the submission: the kernel only
		// revalidates and fences userptr buffers that are listed, so one left out could have its pages
		// migrated under a live GPU access. CreatedHandles() is the direct-memory chunks plus the
		// readable external slices (CPU stacks are never bound, so they are not here).
		std::vector<uint32_t> handles = m_gem->CreatedHandles();

		// KYTY_NATIVE_GPU_STATS=1: bracket the IB with pipeline-statistics samples written into the
		// submitter's scratch area, to see how far each frame gets (vertices, primitives, pixel
		// shader invocations) when the output stays empty.
		std::span<const uint32_t> ib = result.ib;
		std::vector<uint32_t>     wrapped;
		if (m_stats) {
			const uint64_t sva = m_submitter->ScratchVa();
			for (uint32_t i = 0; i < 2 * kStatSlots * 8; i++) {
				m_submitter->ScratchCpu()[i] = 0;
			}
			wrapped.reserve(result.ib.size() + 16);
			wrapped.push_back(StatPkt3(kItEventWrite, 1));
			wrapped.push_back(kEventPipelineStatStart);
			EmitSample(wrapped, sva);
			wrapped.insert(wrapped.end(), result.ib.begin(), result.ib.end());
			EmitSample(wrapped, sva + kStatSlots * 8);
			while ((wrapped.size() & 7u) != 0) {
				wrapped.push_back(0xffff1000u);
			}
			ib = wrapped;
		}
		NativeSubmitter::Result r = m_submitter->Submit(ib, handles, m_timeout_sec);

		const bool detail = frame < 3 || (frame % 120 == 0);
		if (!r.submitted) {
			std::fprintf(stderr,
			             "native shadow: frame %llu submit failed: %s (after %d retries, %zu BOs)\n",
			             static_cast<unsigned long long>(frame), std::strerror(-r.err), r.retries,
			             handles.size());
			Disable("submit error");
			return;
		}
		if (r.timed_out || r.reset || !r.completed) {
			std::fprintf(stderr,
			             "native shadow: frame %llu %s after %.1f ms (ib %zu dw, %u draws); "
			             "disabling native path\n",
			             static_cast<unsigned long long>(frame),
			             r.timed_out ? "TIMED OUT" : (r.reset ? "context RESET" : "did not complete"),
			             r.seconds * 1e3, result.ib.size(), result.stats.draws);
			Disable("timeout or reset");
			return;
		}

		m_completed++;
		m_gpu_ms_total += r.seconds * 1e3;
		if (m_stats && result.stats.draws > 0 && m_stats_logged < 12) {
			LogStats(frame, result.stats.draws);
			m_stats_logged++;
		}
		if (m_dump_regs_frame >= 0 && frame == static_cast<uint64_t>(m_dump_regs_frame)) {
			DumpRegs(frame);
		}
		if (detail) {
			std::fprintf(stderr,
			             "native shadow: frame %llu ok, gpu(submit->fence) %.2f ms, %d cs-retries, "
			             "ib %zu dw (preamble cx/sh/uc %u/%u/%u), %u draws, %u/%u/%u loads, %u dispatch, "
			             "%u acquire, %u uc-skipped, %u/%u waits unmet/dropped\n",
			             static_cast<unsigned long long>(frame), r.seconds * 1e3, r.retries,
			             result.ib.size(), result.stats.preamble_cx, result.stats.preamble_sh,
			             result.stats.preamble_uc, result.stats.draws, result.stats.loaded_cx_regs,
			             result.stats.loaded_sh_regs, result.stats.loaded_uc_regs,
			             result.stats.dispatches, result.stats.acquire_mem, result.stats.uconfig_skipped,
			             result.stats.waits_unmet, result.stats.waits_dropped);
		}
		if (m_completed > 0 && m_completed % 120 == 0) {
			std::fprintf(stderr,
			             "native shadow: %llu frames completed, avg gpu(submit->fence) %.2f ms, "
			             "%u shaders scanned, %u v_dot2c_f32_f16 sites\n",
			             static_cast<unsigned long long>(m_completed),
			             m_gpu_ms_total / static_cast<double>(m_completed), m_shaders_scanned,
			             m_dot2c_sites);
		}
#else
		(void)result;
#endif
	}

private:
	// Pipeline statistics: 11 64-bit counters per sample on GFX10, in hardware slot order
	// (RADV pipeline_statistics_indices): PS invocations, clipper primitives, clipper invocations,
	// VS invocations, GS invocations, GS primitives, IA primitives, IA vertices, HS, DS, CS.
	static constexpr uint32_t kStatSlots              = 11;
	static constexpr uint32_t kItEventWrite           = 0x46;
	static constexpr uint32_t kEventPipelineStatStart = 0x19;
	static constexpr uint32_t kEventSamplePipelineStat = 0x1e | (2u << 8u); // EVENT_INDEX 2

	static uint32_t StatPkt3(uint32_t op, uint32_t body_dw) {
		return (3u << 30u) | (((body_dw - 1u) & 0x3fffu) << 16u) | (op << 8u);
	}

	static void EmitSample(std::vector<uint32_t>& ib, uint64_t va) {
		ib.push_back(StatPkt3(kItEventWrite, 3));
		ib.push_back(kEventSamplePipelineStat);
		ib.push_back(static_cast<uint32_t>(va));
		ib.push_back(static_cast<uint32_t>(va >> 32u));
	}

	void LogStats(uint64_t frame, uint32_t draws) const {
#if KYTY_GEM_BACKING_SUPPORTED
		static const char* const kNames[kStatSlots] = {"ps", "c_prim", "c_inv", "vs", "gs", "gs_prim",
		                                               "ia_prim", "ia_vert", "hs", "ds", "cs"};
		const volatile uint8_t*  base               = m_submitter->ScratchCpu();
		std::fprintf(stderr, "native shadow: frame %llu pipeline stats (%u draws):",
		             static_cast<unsigned long long>(frame), draws);
		for (uint32_t i = 0; i < kStatSlots; i++) {
			uint64_t begin = 0;
			uint64_t end   = 0;
			for (uint32_t b = 0; b < 8; b++) {
				begin |= static_cast<uint64_t>(base[i * 8 + b]) << (8u * b);
				end |= static_cast<uint64_t>(base[kStatSlots * 8 + i * 8 + b]) << (8u * b);
			}
			std::fprintf(stderr, " %s=%llu", kNames[i], static_cast<unsigned long long>(end - begin));
		}
		std::fprintf(stderr, "\n");
#else
		(void)frame;
		(void)draws;
#endif
	}

	void DumpRegs(uint64_t frame) const {
		const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
		char        path[512];
		std::snprintf(path, sizeof(path), "%s/shadow_regs_%llu.txt",
		              (dir != nullptr && dir[0] != '\0') ? dir : ".", static_cast<unsigned long long>(frame));
		if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
			m_translator.DumpShadow(f);
			std::fclose(f);
			std::fprintf(stderr, "native shadow: dumped shadow registers of frame %llu to %s\n",
			             static_cast<unsigned long long>(frame), path);
		}
	}

	// The first native CS pays a large one-off cost while amdgpu validates the whole userptr
	// backing (the shader-range binding does not remove that), so the wait must outlast it or a
	// successful-but-slow submit is misread as a hang. Overridable with KYTY_NATIVE_GPU_TIMEOUT_SEC.
	static constexpr double kDefaultTimeoutSec = 12.0;

	void Init() {
		if (!EnvOn("KYTY_NATIVE_GPU_SHADOW")) {
			return;
		}
		m_translator.SeedPs5Defaults();
		m_stats = EnvOn("KYTY_NATIVE_GPU_STATS");
		if (const char* d = std::getenv("KYTY_NATIVE_GPU_DUMP_REGS"); d != nullptr && d[0] != '\0') {
			m_dump_regs_frame = std::atoll(d);
		}
		if (const char* t = std::getenv("KYTY_NATIVE_GPU_TIMEOUT_SEC"); t != nullptr && t[0] != '\0') {
			const double v = std::atof(t);
			if (v > 0.0) {
				m_timeout_sec = v;
			}
		}
		if (!EnvOn("KYTY_NATIVE_GPU_GEM")) {
			std::fprintf(stderr,
			             "native shadow: KYTY_NATIVE_GPU_SHADOW set but KYTY_NATIVE_GPU_GEM is not; "
			             "native path needs GEM-backed guest memory. Staying off.\n");
			return;
		}
#if KYTY_GEM_BACKING_SUPPORTED
		m_gem = LibKernel::Memory::GetGemBacking();
		if (m_gem == nullptr) {
			std::fprintf(stderr, "native shadow: GEM backing unavailable; staying off.\n");
			return;
		}
		m_submitter = NativeSubmitter::Create(m_gem->DrmFd());
		if (m_submitter == nullptr) {
			std::fprintf(stderr, "native shadow: submitter init failed; staying off.\n");
			return;
		}
		m_enabled = true;
		std::fprintf(stderr, "native shadow: enabled (gfx-ring shadow submission, %.1f s timeout)\n",
		             m_timeout_sec);
#else
		std::fprintf(stderr, "native shadow: not supported on this platform; staying off.\n");
#endif
	}

	void Disable(const char* reason) {
		m_disabled = true;
#if KYTY_GEM_BACKING_SUPPORTED
		delete m_submitter;
		m_submitter = nullptr;
#endif
		std::fprintf(stderr,
		             "native shadow: disabled (%s) after %llu completed frames, avg %.2f ms, "
		             "%u v_dot2c_f32_f16 sites in %u shaders\n",
		             reason, static_cast<unsigned long long>(m_completed),
		             m_completed > 0 ? m_gpu_ms_total / static_cast<double>(m_completed) : 0.0,
		             m_dot2c_sites, m_shaders_scanned);
	}

	void DetectShaders(const std::vector<uint64_t>& addrs) {
		for (uint64_t va: addrs) {
			if (va == 0 || !m_seen_shaders.insert(va).second) {
				continue;
			}
			// Guest VA == host VA in Kyty, so the ISA is readable through the pointer.
			const auto* code = reinterpret_cast<const uint32_t*>(va);
			Dot2cScan   scan = CountDot2c(std::span<const uint32_t>(code, 1u << 16u));
			m_shaders_scanned++;
			m_dot2c_sites += scan.sites;
			if (scan.sites > 0) {
				std::fprintf(stderr,
				             "native shadow: shader 0x%llx has %u v_dot2c_f32_f16 site(s) "
				             "(%u instrs, %s); patching is future work\n",
				             static_cast<unsigned long long>(va), scan.sites, scan.instructions,
				             scan.ended ? "s_endpgm" : (scan.decode_error ? "decode stop" : "capped"));
			}
		}
	}

	std::mutex    m_mutex;
	bool          m_initialized = false;
	bool          m_enabled     = false;
	bool          m_disabled    = false;
	Pm4Translator m_translator;
	bool          m_stats          = false;
	uint32_t      m_stats_logged   = 0;
	int64_t       m_dump_regs_frame = -1;
	uint64_t      m_frames         = 0;
	uint64_t      m_completed      = 0;
	uint32_t      m_refused        = 0;
	double        m_gpu_ms_total   = 0.0;
	double        m_timeout_sec    = kDefaultTimeoutSec;
	uint32_t      m_shaders_scanned = 0;
	uint32_t      m_dot2c_sites    = 0;
	std::set<uint64_t> m_seen_shaders;
#if KYTY_GEM_BACKING_SUPPORTED
	GemBacking*      m_gem       = nullptr;
	NativeSubmitter* m_submitter = nullptr;
#endif
};

} // namespace

void ShadowSubmitDcb(const uint32_t* dcb, uint32_t size_in_dwords) {
	if (dcb == nullptr || size_in_dwords == 0) {
		return;
	}
	Shadow::Instance().Submit(std::span<const uint32_t>(dcb, size_in_dwords));
}

namespace {

std::atomic<bool>& ShowNative() {
	static std::atomic<bool> show {EnvOn("KYTY_NATIVE_GPU_SHOW")};
	return show;
}

} // namespace

bool ShadowShowsNativeFrames() {
	return ShowNative().load(std::memory_order_relaxed) && Shadow::Instance().Active();
}

void ShadowToggleNativeFrames() {
	const bool now = !ShowNative().load(std::memory_order_relaxed);
	ShowNative().store(now, std::memory_order_relaxed);
	std::fprintf(stderr, "native shadow: showing %s frames\n", now ? "NATIVE" : "Vulkan");
}

void ShadowOnFlip(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                  uint32_t bytes_per_pixel, uint32_t pixel_format, uint32_t tile_mode) {
	static std::atomic<uint64_t> flips {0};
	const uint64_t               n    = flips.fetch_add(1, std::memory_order_relaxed);
	const char*                  list = std::getenv("KYTY_NATIVE_GPU_DUMP_FLIPS");
	if (list == nullptr || address == 0 || size == 0 || !Shadow::Instance().Active()) {
		return;
	}
	bool wanted = false;
	for (const char* p = list; *p != '\0';) {
		char*                    end = nullptr;
		const unsigned long long v   = std::strtoull(p, &end, 10);
		if (end == p) {
			break;
		}
		wanted = wanted || v == n;
		p      = (*end == ',') ? end + 1 : end;
	}
	if (!wanted) {
		return;
	}
	const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
	if (dir == nullptr || dir[0] == '\0') {
		dir = ".";
	}
	char path[512];
	std::snprintf(path, sizeof(path), "%s/flip_%llu.raw", dir, static_cast<unsigned long long>(n));
	if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
		std::fwrite(reinterpret_cast<const void*>(address), 1, size, f);
		std::fclose(f);
	}
	std::snprintf(path, sizeof(path), "%s/flip_%llu.txt", dir, static_cast<unsigned long long>(n));
	if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
		std::fprintf(f,
		             "address 0x%llx\nsize %llu\nwidth %u\nheight %u\nbytes_per_pixel %u\n"
		             "pixel_format %u\ntile_mode 0x%x\nshowing_native %d\n",
		             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
		             width, height, bytes_per_pixel, pixel_format, tile_mode,
		             ShowNative().load() ? 1 : 0);
		std::fclose(f);
	}
	std::fprintf(stderr, "native shadow: dumped flip %llu (%ux%u, %llu bytes) to %s\n",
	             static_cast<unsigned long long>(n), width, height,
	             static_cast<unsigned long long>(size), dir);
}

} // namespace Libs::Graphics::NativeGpu
