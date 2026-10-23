#include "graphics/native_gpu/nativeShadow.h"

#include "graphics/native_gpu/nativePm4.h"
#include "graphics/native_gpu/nativeSubmit.h" // KYTY_GEM_BACKING_SUPPORTED + NativeSubmitter
#include "graphics/native_gpu/shaderPatcher.h"

#include "kernel/memory.h"
#include "libs/agc.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <ctime>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace Libs::Graphics::NativeGpu {

namespace {

double NowSecShadow() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

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

	// The color targets the DCBs bound since the last call, for the flip dump.
	std::vector<RenderTargetInfo> TakeFrameTargets() {
		std::lock_guard<std::mutex>   lock(m_mutex);
		std::vector<RenderTargetInfo> out;
		for (const auto& [addr, t]: m_frame_targets) {
			out.push_back(t);
		}
		m_frame_targets.clear();
		return out;
	}

	// Reads guest memory through the backing (clipped to the view holding `addr`); 0 when unavailable.
	uint64_t ReadGuest(uint64_t addr, void* dst, uint64_t size) {
#if KYTY_GEM_BACKING_SUPPORTED
		return m_gem != nullptr ? m_gem->ReadGuest(addr, dst, size) : 0;
#else
		(void)addr;
		(void)dst;
		(void)size;
		return 0;
#endif
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
		SetStage(Stage::Translate, frame);
		const double       t_translate = NowSecShadow();
		Pm4TranslateResult result      = m_translator.Translate(dcb);
		const double       translate_ms = (NowSecShadow() - t_translate) * 1e3;

		DetectShaders(result.shader_addrs);
		for (const auto& snap: result.inspected) {
			InspectDraw(frame, snap);
		}
		// A shader that uses GDS hung the board in ASTRO BOT even with a GDS partition; until that is
		// understood such DCBs stay on Kyty unless KYTY_NATIVE_GPU_GDS_SHADERS=1.
		// Skip mode learns GDS shaders as they appear: the DCB that introduces one was translated
		// before the scan, so it goes to Kyty whole and the translator drops that shader's draws from
		// the next DCB on.
		if (result.ok && m_skip_gds_shaders && m_gds_shaders.size() != m_skipped_shader_count) {
			m_skipped_shader_count = m_gds_shaders.size();
			std::set<uint64_t> skip = m_gds_shaders;
			skip.insert(m_skip_ps.begin(), m_skip_ps.end());
			m_translator.SetSkippedPixelShaders(skip);
			m_refused++;
			return;
		}
		if (result.ok && !m_allow_gds_shaders && !m_skip_gds_shaders) {
			for (uint64_t va: result.shader_addrs) {
				if (m_gds_shaders.count(va) != 0) {
					if (m_gds_refused++ < 4) {
						std::fprintf(stderr, "native shadow: frame %llu left to Kyty: shader 0x%llx uses GDS\n",
						             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(va));
					}
					m_refused++;
					return;
				}
			}
		}
		if (m_collect_targets) {
			for (const auto& t: result.render_targets) {
				auto& acc = m_frame_targets[t.addr];
				const uint32_t draws = acc.draws + t.draws;
				acc       = t;
				acc.draws = draws;
			}
		}
		for (uint32_t off: result.skipped_uconfig) {
			if (m_skipped_uconfig.insert(off).second) {
				std::fprintf(stderr, "native shadow: dropping non-whitelisted UCONFIG 0x%05x (offset 0x%03x)\n",
				             0x30000u + off * 4u, off);
			}
		}

		if (!result.ok) {
			for (uint32_t op: result.unknown_ops) {
				if (m_unknown_ops.insert(op).second) {
					std::fprintf(stderr, "native shadow: untranslatable packet op 0x%02x (first in frame %llu)\n", op,
					             static_cast<unsigned long long>(frame));
				}
			}
			if (m_refused < 8) {
				std::fprintf(stderr,
				             "native shadow: frame %llu refused, unknown packet op 0x%02x at dw %u "
				             "(%u packets translated)%s",
				             static_cast<unsigned long long>(frame), result.unknown_op,
				             result.unknown_dw, result.stats.packets, result.unknown_detail.empty() ? "\n" : ", body");
				for (uint32_t v: result.unknown_detail) {
					std::fprintf(stderr, " 0x%08x", v);
				}
				if (!result.unknown_detail.empty()) {
					std::fprintf(stderr, "\n");
				}
			}
			m_refused++;
			return;
		}

#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter == nullptr) {
			return;
		}
		// Guest direct memory is resident on the GPU only where needed (gemBacking.h): before the IB
		// runs, promote what the CPU has written and what this DCB references.
		const double t_residency = NowSecShadow();
		SetStage(Stage::Residency, frame);
		MakeResident(result);
		m_residency_ms_total += (NowSecShadow() - t_residency) * 1e3;

		// Every GEM buffer the GPU may touch must be resident in the submission: the kernel only
		// validates and fences buffers that are listed, so a userptr one left out could have its pages
		// migrated under a live GPU access. CreatedHandles() is the resident blocks, the dummy and the
		// readable external slices (CPU stacks are never bound, so they are not here).
		std::vector<uint32_t> handles = m_gem->CreatedHandles();

		// KYTY_NATIVE_GPU_STATS=1: bracket the IB with pipeline-statistics samples written into the
		// submitter's scratch area, to see how far each frame gets (vertices, primitives, pixel
		// shader invocations) when the output stays empty.
		// Every IB is bracketed with two GPU timestamps (COPY_DATA from the GPU clock) in the
		// scratch area, so the log can split submit->fence time into the CS ioctl, the GPU work and
		// the rest. With KYTY_NATIVE_GPU_STATS=1 pipeline-statistics samples are added as well.
		const uint64_t        sva = m_submitter->ScratchVa();
		std::vector<uint32_t> wrapped;
		wrapped.reserve(result.ib.size() + 32);
		EmitTimestamp(wrapped, sva + kTimestampOffset);
		if (m_stats) {
			for (uint32_t i = 0; i < 2 * kStatSlots * 8; i++) {
				m_submitter->ScratchCpu()[i] = 0;
			}
			wrapped.push_back(StatPkt3(kItEventWrite, 1));
			wrapped.push_back(kEventPipelineStatStart);
			EmitSample(wrapped, sva);
		}
		wrapped.insert(wrapped.end(), result.ib.begin(), result.ib.end());
		if (m_stats) {
			EmitSample(wrapped, sva + kStatSlots * 8);
		}
		EmitTimestamp(wrapped, sva + kTimestampOffset + 8);
		while ((wrapped.size() & 7u) != 0) {
			wrapped.push_back(0xffff1000u);
		}
		std::span<const uint32_t> ib = wrapped;
		if (!m_breadcrumbs.empty()) {
			WriteBreadcrumb(frame, dcb, result, ib);
		}
		SetStage(Stage::Gpu, frame);
		NativeSubmitter::Result r = m_submitter->Submit(ib, handles, m_timeout_sec);
		// ENOENT: a buffer in the list was closed meanwhile (another guest thread unmapped an external
		// range between CreatedHandles and the ioctl). Take a fresh list and try again.
		for (int retry = 0; !r.submitted && r.err == -ENOENT && retry < 3; retry++) {
			handles = m_gem->CreatedHandles();
			r       = m_submitter->Submit(ib, handles, m_timeout_sec);
		}
		SetStage(Stage::Idle, frame);

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
		if (m_completed % 30 == 0 && m_gem->DummyWritten()) {
			m_dummy_writes++;
			if (m_dummy_writes <= 8) {
				std::fprintf(stderr,
				             "native shadow: frame %llu: the GPU wrote to guest memory that was not "
				             "resident (a reference the translator missed)\n",
				             static_cast<unsigned long long>(frame));
			}
		}
		m_gpu_ms_total += r.seconds * 1e3;
		const double gpu_ms = GpuMs();
		m_cs_ms_total += r.cs_seconds * 1e3;
		m_gpu_only_ms_total += gpu_ms;
		m_translate_ms_total += translate_ms;
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
			const double n = static_cast<double>(m_completed);
			std::fprintf(stderr,
			             "native shadow: avg per frame: translate %.2f ms, residency %.2f ms, CS ioctl %.2f ms, "
			             "GPU %.2f ms, submit->fence %.2f ms; %llu MiB of guest memory resident, %u dummy "
			             "writes seen\n",
			             m_translate_ms_total / n, m_residency_ms_total / n, m_cs_ms_total / n,
			             m_gpu_only_ms_total / n, m_gpu_ms_total / n,
			             static_cast<unsigned long long>(m_gem->ResidentBytes() >> 20u), m_dummy_writes);
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
	// Watchdog (KYTY_NATIVE_GPU_WATCHDOG=1): every 2 s it checks whether the shadow path is stuck inside a
	// submission (which stage, for how long) or whether the game has stopped submitting DCBs at all, so an
	// intermittent freeze shows where it sits without a debugger.
	enum class Stage : int { Idle = 0, Translate = 1, Residency = 2, Gpu = 3 };
	void SetStage(Stage stage, uint64_t frame) {
		m_stage.store(static_cast<int>(stage), std::memory_order_relaxed);
		m_stage_frame.store(frame, std::memory_order_relaxed);
		m_stage_since.store(NowSecShadow(), std::memory_order_relaxed);
	}
	void StartWatchdog() {
		std::thread([this] {
			static const char* const kNames[] = {"idle (waiting for the next DCB)", "translate", "residency", "GPU submit/wait"};
			double reported = -1.0;
			for (;;) {
				struct timespec nap = {2, 0};
				nanosleep(&nap, nullptr);
				const double since = m_stage_since.load(std::memory_order_relaxed);
				const double stuck = NowSecShadow() - since;
				if (since > 0.0 && stuck >= 10.0 && since != reported) {
					reported = since;
					const int stage = m_stage.load(std::memory_order_relaxed);
					std::fprintf(stderr, "native shadow: WATCHDOG: %.0f s in stage %s at frame %llu (promotion in progress: %s)\n",
					             stuck, kNames[stage], static_cast<unsigned long long>(m_stage_frame.load()),
					             PromotionInProgress() ? "yes" : "no");
				}
			}
		}).detach();
	}
	static bool PromotionInProgress() {
#if KYTY_GEM_BACKING_SUPPORTED
		return GemBacking::WaitIfPromotingWouldWait();
#else
		return false;
#endif
	}
	std::atomic<int>      m_stage {0};
	std::atomic<uint64_t> m_stage_frame {0};
	std::atomic<double>   m_stage_since {0.0};

#if KYTY_GEM_BACKING_SUPPORTED
	// KYTY_NATIVE_GPU_BREADCRUMBS: a GPU hang on this board can take the whole machine down before
	// any log reaches the disk, so before each submission the guest DCB, the IB about to run and a
	// summary are written and synced: after a power cycle, last.txt names the submission that hung
	// and dcb_<n>.bin / ib_<n>.bin (the last four, n = frame % 4) hold it.
	void WriteBreadcrumb(uint64_t frame, std::span<const uint32_t> dcb, const Pm4TranslateResult& result,
	                     std::span<const uint32_t> ib) const {
		auto write_file = [&](const std::string& name, const void* data, size_t bytes) {
			const std::string path = m_breadcrumbs + "/" + name;
			const int         fd   = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
			if (fd < 0) {
				return;
			}
			const auto* p = static_cast<const char*>(data);
			while (bytes > 0) {
				const ssize_t n = ::write(fd, p, bytes);
				if (n <= 0) {
					break;
				}
				p += n;
				bytes -= static_cast<size_t>(n);
			}
			::fdatasync(fd);
			::close(fd);
		};
		const unsigned slot = static_cast<unsigned>(frame % 4u);
		write_file("dcb_" + std::to_string(slot) + ".bin", dcb.data(), dcb.size_bytes());
		write_file("ib_" + std::to_string(slot) + ".bin", ib.data(), ib.size_bytes());
		char summary[512];
		const int n = std::snprintf(summary, sizeof(summary),
		                            "frame %llu slot %u: dcb %zu dw, ib %zu dw, %u draws, %u dispatches, %u dma, "
		                            "%u set_base, %llu MiB resident\n",
		                            static_cast<unsigned long long>(frame), slot, dcb.size(), ib.size(),
		                            result.stats.draws, result.stats.dispatches, result.stats.dma_data,
		                            result.stats.set_base, static_cast<unsigned long long>(m_gem->ResidentBytes() >> 20u));
		write_file("last.txt", summary, static_cast<size_t>(n));
	}

	// Makes resident, before `result`'s IB runs: every block the CPU has written, every shader, and
	// every reference the translator reported that lies in guest direct memory. Pointer candidates
	// from shader user data are followed two levels deep (resource tables, and tables of tables),
	// scanning each table for 48-bit buffer pointers (V#, whose dwords 1-2 give stride and record
	// count, so the whole buffer is promoted) and 40-bit image bases (T#).
	// KYTY_NATIVE_GPU_LOG_UNMAPPED=1: report addresses found in shader user data and resource tables
	// that fall in the guest address range but are mapped for the GPU nowhere (a GPU access there is a
	// page fault, which hangs this board). Candidates are heuristic, so some reports are just data.
	void NoteUnmapped(uint64_t addr, const char* kind, uint64_t table) {
		if (!m_log_unmapped || addr < 0x100000000ull || addr >= 0x10000000000ull || m_unmapped_seen.size() >= 4096 ||
		    !m_unmapped_seen.insert(addr >> 16u).second || m_gem->IsGpuMapped(addr)) {
			return;
		}
		if (m_unmapped_logged++ < 40) {
			std::fprintf(stderr, "native shadow: %s 0x%llx (from table 0x%llx) is not mapped for the GPU\n", kind,
			             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(table));
		}
	}

	// KYTY_NATIVE_GPU_INSPECT_PS: for a few draws with that pixel shader, log both stages' user data and
	// decode the descriptors around them: inline V#s in the user data, and V#/T# in the 0x200 bytes
	// behind every user-data pointer, each with whether its base is mapped for the GPU.
	void InspectDraw(uint64_t frame, const DrawSnapshot& snap) {
		if (m_inspected++ >= 3) {
			return;
		}
		auto mapped = [&](uint64_t a) { return a == 0 ? "null" : (m_gem->IsGpuMapped(a) ? "mapped" : "NOT MAPPED"); };
		std::fprintf(stderr, "native shadow: INSPECT frame %llu ps 0x%llx es 0x%llx\n", static_cast<unsigned long long>(frame),
		             static_cast<unsigned long long>(snap.ps), static_cast<unsigned long long>(snap.es));
		auto dump_stage = [&](const char* stage, const uint32_t* ud) {
			for (uint32_t i = 0; i < 32; i++) {
				std::fprintf(stderr, "%s%s ud[%2u]=0x%08x", i % 8 == 0 ? "\n  " : " ", stage, i, ud[i]);
			}
			std::fprintf(stderr, "\n");
			for (uint32_t i = 0; i + 1 < 32; i++) {
				const uint64_t ptr = ud[i] | (static_cast<uint64_t>(ud[i + 1] & 0xffffu) << 32u);
				if (ptr < 0x100000000ull || ptr >= 0x10000000000ull) {
					continue;
				}
				std::fprintf(stderr, "  %s ud[%u:%u] -> 0x%llx (%s)\n", stage, i, i + 1, static_cast<unsigned long long>(ptr), mapped(ptr));
				uint32_t     t[128];
				const size_t n = m_gem->ReadGuest(ptr, t, sizeof(t)) / 4;
				for (size_t k = 0; k + 3 < n; k += 4) {
					const uint64_t vbase = t[k] | (static_cast<uint64_t>(t[k + 1] & 0xffffu) << 32u);
					const uint64_t tbase = (t[k] | (static_cast<uint64_t>(t[k + 1] & 0xffu) << 32u)) << 8u;
					if ((vbase >= 0x100000000ull && vbase < 0x10000000000ull) || (tbase >= 0x100000000ull && tbase < 0x10000000000ull)) {
						std::fprintf(stderr, "    +0x%03zx: %08x %08x %08x %08x  as V# 0x%llx (%s) stride %u records %u | as T# 0x%llx (%s)\n",
						             k * 4, t[k], t[k + 1], t[k + 2], t[k + 3], static_cast<unsigned long long>(vbase), mapped(vbase),
						             (t[k + 1] >> 16u) & 0x3fffu, t[k + 2], static_cast<unsigned long long>(tbase), mapped(tbase));
					}
				}
			}
		};
		dump_stage("ps", snap.ps_user_data);
		dump_stage("gs", snap.gs_user_data);
	}

	void MakeResident(const Pm4TranslateResult& result) {
		constexpr uint64_t kTableSpan = 4096;
		constexpr uint64_t kImageSpan = 32ull << 20u;
		constexpr uint64_t kBufferCap = 64ull << 20u;
		constexpr size_t   kMaxTables = 2048;
		constexpr uint32_t kBucketShift = 24; // 16 MiB guest-address buckets, the block size
		m_gem->PromoteTouched();

		// Membership tests run thousands of times per frame: binary search over a cached copy of the
		// view ranges, refreshed when the views change (which also forgets which buckets are known
		// to be resident).
		if (const uint64_t gen = m_gem->ViewsGeneration(); gen != m_views_generation) {
			m_view_ranges      = m_gem->ViewRanges();
			m_views_generation = gen;
			m_resident_buckets.clear();
		}
		auto backed = [&](uint64_t addr) {
			auto it = std::upper_bound(m_view_ranges.begin(), m_view_ranges.end(),
			                           std::make_pair(addr, ~0ull));
			return it != m_view_ranges.begin() && addr < std::prev(it)->second;
		};
		auto promote = [&](uint64_t addr, uint64_t size) {
			for (uint64_t b = addr >> kBucketShift; b <= (addr + size - 1) >> kBucketShift; b++) {
				if (m_resident_buckets.insert(b).second) {
					m_gem->PromoteGuestRange(b << kBucketShift, 1ull << kBucketShift);
				}
			}
		};

		for (uint64_t addr: result.shader_addrs) {
			if (backed(addr)) {
				promote(addr, 64u * 1024u);
			}
		}
		std::vector<uint64_t> tables;
		for (const auto& ref: result.refs) {
			if (ref.pointer) {
				NoteUnmapped(ref.addr, "user-data pointer", 0);
			}
			// An index buffer the GPU cannot reach at all is a page fault, and a fault hangs this board.
			if (ref.index && !backed(ref.addr) && !m_gem->IsGpuMapped(ref.addr) && m_unmapped_logged++ < 16) {
				std::fprintf(stderr, "native shadow: index buffer 0x%llx (0x%llx bytes) is not mapped for the GPU\n",
				             static_cast<unsigned long long>(ref.addr), static_cast<unsigned long long>(ref.size));
			}
			if (backed(ref.addr)) {
				promote(ref.addr, ref.size != 0 ? ref.size : kTableSpan);
				if (ref.pointer) {
					tables.push_back(ref.addr);
				}
			}
		}
		std::set<uint64_t> seen;
		for (int depth = 0; depth < 2 && !tables.empty(); depth++) {
			std::vector<uint64_t> next;
			for (uint64_t table: tables) {
				if (seen.size() >= kMaxTables || !seen.insert(table).second) {
					continue;
				}
				uint32_t     d[64];
				const size_t n = m_gem->ReadGuest(table, d, sizeof(d)) / 4;
				for (size_t i = 0; i + 1 < n; i++) {
					const uint64_t ptr = d[i] | (static_cast<uint64_t>(d[i + 1] & 0xffffu) << 32u);
					NoteUnmapped(ptr, "buffer", table);
					if (ptr != 0 && backed(ptr)) {
						const uint64_t stride  = (d[i + 1] >> 16u) & 0x3fffu;
						const uint64_t records = i + 2 < n ? d[i + 2] : 0;
						const uint64_t bytes   = stride != 0 ? stride * records : records;
						promote(ptr, std::clamp<uint64_t>(bytes, kTableSpan, kBufferCap));
						next.push_back(ptr);
					}
					const uint64_t image = (d[i] | (static_cast<uint64_t>(d[i + 1] & 0xffu) << 32u)) << 8u;
					NoteUnmapped(image, "image", table);
					if (image != 0 && backed(image)) {
						promote(image, kImageSpan);
					}
				}
			}
			tables = std::move(next);
		}
	}
#endif

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

	// COPY_DATA SRC_SEL 9 (GPU clock, gpu_counter_freq = 100 MHz on BC-250) -> memory, 64 bit.
	static constexpr uint32_t kItCopyData       = 0x40;
	static constexpr uint32_t kTimestampOffset  = 512;
	static constexpr double   kGpuClockHz       = 100e6;

	static void EmitTimestamp(std::vector<uint32_t>& ib, uint64_t va) {
		ib.push_back(StatPkt3(kItCopyData, 5));
		ib.push_back(9u | (5u << 8u) | (1u << 16u) | (1u << 20u)); // SRC TIMESTAMP, DST MEM, 64-bit, confirm
		ib.push_back(0);
		ib.push_back(0);
		ib.push_back(static_cast<uint32_t>(va));
		ib.push_back(static_cast<uint32_t>(va >> 32u));
	}

	double GpuMs() const {
#if KYTY_GEM_BACKING_SUPPORTED
		const volatile uint8_t* base  = m_submitter->ScratchCpu() + kTimestampOffset;
		uint64_t                begin = 0;
		uint64_t                end   = 0;
		for (uint32_t b = 0; b < 8; b++) {
			begin |= static_cast<uint64_t>(base[b]) << (8u * b);
			end |= static_cast<uint64_t>(base[8 + b]) << (8u * b);
		}
		return end > begin ? static_cast<double>(end - begin) * 1e3 / kGpuClockHz : 0.0;
#else
		return 0.0;
#endif
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

	// The first native CS pays a large one-off cost while amdgpu validates the userptr ranges and
	// places the freshly created GTT chunks, so the wait must outlast it or a
	// successful-but-slow submit is misread as a hang. Overridable with KYTY_NATIVE_GPU_TIMEOUT_SEC.
	static constexpr double kDefaultTimeoutSec = 12.0;

	void Init() {
		if (!EnvOn("KYTY_NATIVE_GPU_SHADOW") && !EnvOn("KYTY_NATIVE_GPU_PRIMARY")) {
			return;
		}
		m_translator.SeedPs5Defaults();
		// The rest of the PS5 context state the system sets up before a title runs: the graphics
		// library's register defaults. amdgpu's CLEAR_STATE leaves many of these at other values
		// (rasterizer and depth controls, scissors, guard band, blend and DCC defaults).
		if (!EnvOn("KYTY_NATIVE_GPU_NO_PS5_DEFAULTS")) {
			const auto defaults = Gen5::AgcContextRegisterDefaults();
			m_translator.SeedContextDefaults(defaults);
			std::fprintf(stderr, "native shadow: seeded %zu PS5 context register defaults\n", defaults.size());
		}
		m_stats = EnvOn("KYTY_NATIVE_GPU_STATS");
		m_dump_shaders = EnvOn("KYTY_NATIVE_GPU_DUMP_SHADERS");
		m_log_unmapped = EnvOn("KYTY_NATIVE_GPU_LOG_UNMAPPED");
		if (EnvOn("KYTY_NATIVE_GPU_NO_DCC")) {
			m_translator.SetNoDcc(true);
			std::fprintf(stderr, "native shadow: render targets without DCC (experiment)\n");
		}
		if (EnvOn("KYTY_NATIVE_GPU_NO_CMASK_CLEAR")) {
			m_translator.SetNoCmaskClear(true);
			std::fprintf(stderr, "native shadow: no CMASK fast clear on targets without DCC (experiment)\n");
		}
		if (const char* mask = std::getenv("KYTY_NATIVE_GPU_TARGET_MASK"); mask != nullptr) {
			m_translator.SetTargetMask(static_cast<uint32_t>(std::strtoul(mask, nullptr, 0)));
			std::fprintf(stderr, "native shadow: CB_TARGET_MASK limited to %s (experiment)\n", mask);
		}
		if (const char* ps = std::getenv("KYTY_NATIVE_GPU_INSPECT_PS"); ps != nullptr) {
			m_translator.SetInspectedPixelShader(std::strtoull(ps, nullptr, 0));
		}
		// KYTY_NATIVE_GPU_SKIP_PS=0xADDR[,0xADDR...]: leave draws with these pixel shaders out of the IB.
		if (const char* list = std::getenv("KYTY_NATIVE_GPU_SKIP_PS"); list != nullptr) {
			for (const char* p = list; *p != '\0';) {
				char* end = nullptr;
				m_skip_ps.insert(std::strtoull(p, &end, 0));
				if (end == p) {
					break;
				}
				p = (*end == ',') ? end + 1 : end;
			}
			m_translator.SetSkippedPixelShaders(m_skip_ps);
			std::fprintf(stderr, "native shadow: leaving draws with %zu pixel shader(s) out: %s\n", m_skip_ps.size(), list);
		}
		if (const char* ops = std::getenv("KYTY_NATIVE_GPU_SKIP_OPS"); ops != nullptr) {
			std::set<uint32_t> skip;
			for (const char* p = ops; *p != '\0';) {
				char* end = nullptr;
				skip.insert(static_cast<uint32_t>(std::strtoul(p, &end, 0)));
				p = (*end == ',') ? end + 1 : end;
				if (end == p && *p != '\0') {
					break;
				}
			}
			m_translator.SetSkippedOpcodes(skip);
			std::fprintf(stderr, "native shadow: leaving %zu packet type(s) out of the IB: %s\n", skip.size(), ops);
		}
		// KYTY_NATIVE_GPU_GDS_SHADERS=1 runs draws with GDS shaders natively, =skip leaves just those
		// draws out of the IB (the rest of their DCB runs natively); unset leaves whole DCBs to Kyty.
		if (const char* g = std::getenv("KYTY_NATIVE_GPU_GDS_SHADERS"); g != nullptr) {
			m_skip_gds_shaders  = std::strcmp(g, "skip") == 0;
			m_allow_gds_shaders = !m_skip_gds_shaders && g[0] == '1';
		}
		m_collect_targets = std::getenv("KYTY_NATIVE_GPU_DUMP_FLIPS") != nullptr;
		if (EnvOn("KYTY_NATIVE_GPU_WATCHDOG")) {
			StartWatchdog();
		}
		if (const char* dir = std::getenv("KYTY_NATIVE_GPU_BREADCRUMBS"); dir != nullptr && dir[0] != '\0') {
			m_breadcrumbs = dir;
			std::fprintf(stderr, "native shadow: breadcrumbs (every submission synced to disk) in %s\n", dir);
		}

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
		m_translator.SetGdsAvailable(m_submitter->HasGds());
		if (const char* depth = std::getenv("KYTY_NATIVE_GPU_DEPTH"); depth != nullptr) {
			const bool hiz_off  = std::strcmp(depth, "hiz_off") == 0;
			const bool no_htile = std::strcmp(depth, "no_htile") == 0;
			m_translator.SetDepthDebugMode(hiz_off ? Pm4Translator::DepthDebugMode::HizOff
			                                       : (no_htile ? Pm4Translator::DepthDebugMode::NoHtile
			                                                   : Pm4Translator::DepthDebugMode::None));
			std::fprintf(stderr, "native shadow: depth experiment %s\n", depth);
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
			if (scan.gds_ops + scan.ordered_ops + scan.gws_ops > 0) {
				m_gds_shaders.insert(va);
				std::fprintf(stderr,
				             "native shadow: shader 0x%llx uses GDS: %u gds, %u ds_ordered_count, %u ds_gws (%u instrs, %s)\n",
				             static_cast<unsigned long long>(va), scan.gds_ops, scan.ordered_ops, scan.gws_ops,
				             scan.instructions, scan.ended ? "s_endpgm" : (scan.decode_error ? "decode stop" : "capped"));
			}
			// Keep the code for disassembly (llvm-mc -arch=amdgcn -mcpu=gfx1013 --disassemble): GDS
			// shaders always, every shader with KYTY_NATIVE_GPU_DUMP_SHADERS=1.
			if (scan.gds_ops + scan.ordered_ops + scan.gws_ops > 0 || m_dump_shaders) {
				if (const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR"); dir != nullptr && scan.dwords != 0) {
					char path[512];
					std::snprintf(path, sizeof(path), "%s/shader_%llx.bin", dir, static_cast<unsigned long long>(va));
					// KYTY_NATIVE_GPU_DUMP_TAIL=<dwords> also keeps code past the first s_endpgm, where
					// compilers put out-of-line blocks such as the all-lanes-killed epilogue.
					const char*  tail  = std::getenv("KYTY_NATIVE_GPU_DUMP_TAIL");
					const size_t extra = tail != nullptr ? std::strtoul(tail, nullptr, 0) : 0;
					if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
						std::fwrite(code, 4, scan.dwords + extra, f);
						std::fclose(f);
					}
				}
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
	std::string   m_breadcrumbs; // KYTY_NATIVE_GPU_BREADCRUMBS directory, empty when off
	bool          m_collect_targets = false;           // flip dumps compare render targets too
	std::map<uint64_t, RenderTargetInfo> m_frame_targets; // color targets bound since the last flip dump
	uint64_t      m_frames         = 0;
	uint64_t      m_completed      = 0;
	uint32_t      m_refused        = 0;
	double        m_gpu_ms_total   = 0.0;
	double        m_cs_ms_total    = 0.0;
	double        m_gpu_only_ms_total = 0.0;
	double        m_translate_ms_total = 0.0;
	double        m_residency_ms_total = 0.0;
#if KYTY_GEM_BACKING_SUPPORTED
	uint64_t                                   m_views_generation = ~0ull;
	std::vector<std::pair<uint64_t, uint64_t>> m_view_ranges;
	std::set<uint64_t>                         m_resident_buckets; // 16 MiB guest buckets already promoted
#endif
	uint32_t      m_dummy_writes       = 0;
	uint32_t      m_unmapped_logged    = 0;
	bool          m_dump_shaders       = false;
	bool          m_log_unmapped       = false;
	uint32_t      m_inspected          = 0;
	std::set<uint64_t> m_unmapped_seen;     // 64 KiB pages already checked by NoteUnmapped
	std::set<uint64_t> m_skip_ps;           // KYTY_NATIVE_GPU_SKIP_PS
	std::set<uint32_t> m_unknown_ops;
	std::set<uint64_t> m_gds_shaders;         // shaders seen using GDS instructions
	bool               m_allow_gds_shaders = false;
	bool               m_skip_gds_shaders  = false;
	size_t             m_skipped_shader_count = 0;
	uint32_t           m_gds_refused       = 0;
	double        m_timeout_sec    = kDefaultTimeoutSec;
	uint32_t      m_shaders_scanned = 0;
	uint32_t      m_dot2c_sites    = 0;
	std::set<uint64_t> m_seen_shaders;
	std::set<uint32_t> m_skipped_uconfig;
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

bool PrimaryRendering() {
	static const bool primary = EnvOn("KYTY_NATIVE_GPU_PRIMARY");
	// KYTY_NATIVE_GPU_NULL=1: nothing renders at all, natively or through Vulkan, while the game and
	// Kyty's command processor run on. For measuring what a game does on the CPU side (e.g. how much
	// guest memory it really touches) when rendering is what stops it.
	static const bool null_rendering = EnvOn("KYTY_NATIVE_GPU_NULL");
	return null_rendering || (primary && Shadow::Instance().Active());
}

bool ShadowShowsNativeFrames() {
	return (ShowNative().load(std::memory_order_relaxed) || PrimaryRendering()) &&
	       Shadow::Instance().Active();
}

void ShadowToggleNativeFrames() {
	const bool now = !ShowNative().load(std::memory_order_relaxed);
	ShowNative().store(now, std::memory_order_relaxed);
	std::fprintf(stderr, "native shadow: showing %s frames\n", now ? "NATIVE" : "Vulkan");
}

namespace {

// KYTY_NATIVE_GPU_DUMP_FLIPS: comma-separated flip numbers or ranges "first-last" with an optional
// "/step" ("300,600" or "300-2000/10").
bool FlipWanted(const char* list, uint64_t n) {
	for (const char* p = list; *p != '\0';) {
		char*                    end   = nullptr;
		const unsigned long long first = std::strtoull(p, &end, 10);
		if (end == p) {
			return false;
		}
		unsigned long long last = first;
		unsigned long long step = 1;
		p                       = end;
		if (*p == '-') {
			last = std::strtoull(p + 1, &end, 10);
			p    = end;
		}
		if (*p == '/') {
			step = std::strtoull(p + 1, &end, 10);
			p    = end;
		}
		if (n >= first && n <= last && step != 0 && (n - first) % step == 0) {
			return true;
		}
		if (*p == ',') {
			p++;
		} else if (*p != '\0') {
			return false;
		}
	}
	return false;
}

void WriteFile(const char* path, const void* data, uint64_t size) {
	if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
		std::fwrite(data, 1, size, f);
		std::fclose(f);
	}
}

} // namespace

namespace {

// Bytes per pixel of a CB_COLORn_INFO.FORMAT, 0 when unknown.
uint32_t ColorFormatBytes(uint32_t info) {
	switch ((info >> 2u) & 0x1fu) {
		case 1: return 1;                                            // 8
		case 2: case 3: case 16: case 17: case 18: case 19: return 2; // 16, 8_8, 5_6_5, 1_5_5_5, 5_5_5_1, 4_4_4_4
		case 4: case 5: case 6: case 7: case 8: case 9: case 10: case 20: case 21: return 4;
		case 11: case 12: case 22: return 8;                         // 32_32, 16_16_16_16, X24_8_32
		case 14: return 16;                                          // 32_32_32_32
		default: return 0;
	}
}

// For each color target bound since the last dump: guest memory as the native path left it, then
// after Kyty's Vulkan copy is written back over it. Differing targets are written out; a summary of
// all of them goes to flip_<n>.rts.txt, so the first target whose contents diverge can be found.
void DumpRenderTargets(const char* dir, unsigned long long flip,
                       const std::function<uint32_t(uint64_t, uint64_t)>& download_region) {
	auto       targets = Shadow::Instance().TakeFrameTargets();
	char       path[512];
	std::snprintf(path, sizeof(path), "%s/flip_%llu.rts.txt", dir, flip);
	std::FILE* summary = std::fopen(path, "w");
	if (summary == nullptr) {
		return;
	}
	std::fprintf(summary, "# addr width height format swizzle draws bytes vulkan_images differing_bytes\n");
	for (const auto& t: targets) {
		const uint32_t bpp = ColorFormatBytes(t.info);
		// The surface itself, padded to whole 64 KiB swizzle blocks.
		const uint64_t want = bpp != 0 ? (static_cast<uint64_t>(t.width) * t.height * bpp + 0xffffu) & ~0xffffull
		                               : std::min<uint64_t>(t.size, 64ull << 20u);
		std::vector<uint8_t> native(want), vulkan(want);
		const uint64_t       got = Shadow::Instance().ReadGuest(t.addr, native.data(), want);
		if (got == 0) {
			continue;
		}
		const uint32_t images = download_region ? download_region(t.addr, got) : 0;
		Shadow::Instance().ReadGuest(t.addr, vulkan.data(), got);
		uint64_t differing = 0;
		for (uint64_t i = 0; i < got; i++) {
			differing += native[i] != vulkan[i] ? 1u : 0u;
		}
		std::fprintf(summary, "0x%llx %u %u %u 0x%x %u %llu %u %llu\n", static_cast<unsigned long long>(t.addr),
		             t.width, t.height, (t.info >> 2u) & 0x1fu, t.attrib3 & 0x1fu, t.draws,
		             static_cast<unsigned long long>(got), images, static_cast<unsigned long long>(differing));
		if (differing != 0) {
			std::snprintf(path, sizeof(path), "%s/flip_%llu.rt_%llx.native.raw", dir, flip,
			              static_cast<unsigned long long>(t.addr));
			WriteFile(path, native.data(), got);
			std::snprintf(path, sizeof(path), "%s/flip_%llu.rt_%llx.vulkan.raw", dir, flip,
			              static_cast<unsigned long long>(t.addr));
			WriteFile(path, vulkan.data(), got);
		}
	}
	std::fclose(summary);
}

} // namespace

void ShadowOnFlip(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                  uint32_t bytes_per_pixel, uint32_t pixel_format, uint32_t tile_mode,
                  const std::function<uint32_t(uint64_t, uint64_t)>& download_region) {
	static std::atomic<uint64_t> flips {0};
	const uint64_t               n    = flips.fetch_add(1, std::memory_order_relaxed);
	const char*                  list = std::getenv("KYTY_NATIVE_GPU_DUMP_FLIPS");
	if (list == nullptr || address == 0 || size == 0 || !Shadow::Instance().Active()) {
		return;
	}
	if (!FlipWanted(list, n)) {
		(void)Shadow::Instance().TakeFrameTargets(); // keep only the dumped frame's targets
		return;
	}
	const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
	if (dir == nullptr || dir[0] == '\0') {
		dir = ".";
	}
	const auto flip = static_cast<unsigned long long>(n);
	char       path[512];
	// The native result first: it is what guest memory holds right now.
	std::snprintf(path, sizeof(path), "%s/flip_%llu.native.raw", dir, flip);
	WriteFile(path, reinterpret_cast<const void*>(address), size);
	// Then Kyty's Vulkan rendering of the same DCBs, written back over the same guest bytes.
	uint32_t vulkan_images = 0;
	if (!PrimaryRendering() && download_region) {
		vulkan_images = download_region(address, size);
		if (vulkan_images > 0) {
			std::snprintf(path, sizeof(path), "%s/flip_%llu.vulkan.raw", dir, flip);
			WriteFile(path, reinterpret_cast<const void*>(address), size);
		}
		DumpRenderTargets(dir, flip, download_region);
	}
	std::snprintf(path, sizeof(path), "%s/flip_%llu.txt", dir, flip);
	if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
		std::fprintf(f,
		             "address 0x%llx\nsize %llu\nwidth %u\nheight %u\nbytes_per_pixel %u\n"
		             "pixel_format %u\ntile_mode 0x%x\nvulkan_images %u\n",
		             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
		             width, height, bytes_per_pixel, pixel_format, tile_mode, vulkan_images);
		std::fclose(f);
	}
	std::fprintf(stderr, "native shadow: dumped flip %llu (%ux%u) native%s to %s\n", flip, width,
	             height, vulkan_images > 0 ? " + vulkan" : "", dir);
}

} // namespace Libs::Graphics::NativeGpu
