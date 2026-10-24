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
#include <condition_variable>
#include <deque>
#include <functional>
#include <ctime>
#include <map>
#include <mutex>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <set>
#include <tuple>
#include <span>
#include <string>
#include <thread>
#include <utility>
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

// What native primary mode needs to run games correctly is on there by default; setting the switch
// to 0 turns a feature off (and setting it turns it on outside primary mode).
bool PrimaryFeature(const char* name) {
	const char* v = std::getenv(name);
	if (v != nullptr && v[0] != '\0') {
		return v[0] != '0';
	}
	return EnvOn("KYTY_NATIVE_GPU_PRIMARY");
}

uint64_t ToNs(double seconds) {
	return static_cast<uint64_t>(seconds * 1e9);
}

// Capture files (KYTY_NATIVE_GPU_CAPTURE_DIR), read by tools/bc250-pm4dbg. Both are a sequence of
// records: u32 type, u32 payload bytes, payload. Times are CLOCK_MONOTONIC nanoseconds, GPU values are
// GPU clock ticks (TraceHeader::counter_khz); a GPU value of 0 was never written.
//   trace.bin (KYTY_NATIVE_GPU_PROFILE_FLIPS): timing of every buffer, batch and flip in the window.
//   pm4.bin   (KYTY_NATIVE_GPU_PM4_DUMP_FLIPS): every guest buffer and its IB, with guest memory that
//             shader user data points at.
enum TraceType : uint32_t {
	kTraceHeader = 1, // TraceHeader
	kTraceCalib  = 2, // TraceCalib: CPU time <-> GPU clock
	kTraceFlip   = 3, // TraceFlip: a video-out flip reached the presenter
	kTraceBatch  = 4, // TraceBatch: one CS; its TraceBuffer records follow it
	kTraceBuffer = 5, // TraceBuffer: one guest DCB or ACB
	kTraceItem   = 6, // TraceItem: one draw or dispatch (KYTY_NATIVE_GPU_PROFILE_DRAWS)
	kTraceThen   = 7, // TraceThen: Kyty's command processor ran the buffer after the GPU finished it
	kDumpBuffer  = 16, // DumpBuffer, then raw_dw guest dwords, then ib_dw IB dwords
	kDumpMemory  = 17, // DumpMemory, then `bytes` bytes of guest memory
};
struct TraceHeader {
	uint32_t version;
	uint32_t counter_khz;
	uint64_t cpu_ns;
};
struct TraceCalib {
	uint64_t cpu_before_ns;
	uint64_t gpu;
	uint64_t cpu_after_ns;
};
struct TraceFlip {
	uint64_t flip;
	uint64_t cpu_ns;
};
struct TraceBatch {
	uint64_t id;
	uint64_t seq;
	uint64_t cs_begin_ns; // the CS ioctl
	uint64_t cs_end_ns;
	uint64_t fence_ns; // the completion thread saw the fence
	uint32_t buffers;
	uint32_t handles;  // BOs in the list
};
struct TraceBuffer {
	uint64_t batch;
	uint64_t number; // frame / acb counter of the shadow log
	uint64_t enqueue_ns; // the guest submitted it
	uint64_t translated_ns; // translated and appended to the batch
	uint64_t gpu_issued; // the CP started it
	uint64_t gpu_done; // it had finished
	uint32_t acb;
	uint32_t draws;
	uint32_t dispatches;
	uint32_t flips; // flip markers in it
	uint32_t ib_dw;
	uint32_t wait_barriers;
};
struct TraceItem {
	uint64_t number;
	uint64_t shader;
	uint64_t es;
	uint64_t gpu_issued;
	uint64_t gpu_done;
	uint32_t acb;
	uint32_t index;
	uint32_t dw;
	uint32_t op;
	uint32_t args[3];
	uint32_t pad;
};
struct TraceThen {
	uint64_t number;
	uint64_t start_ns;
	uint64_t end_ns;
	uint32_t acb;
	uint32_t pad;
};
struct DumpBuffer {
	uint64_t number;
	uint64_t flip;
	uint32_t acb;
	uint32_t raw_dw;
	uint32_t ib_dw;
	uint32_t pad;
};
struct DumpMemory {
	uint64_t addr;
	uint32_t bytes;
	uint32_t pad;
};
static_assert(sizeof(TraceBuffer) == 72 && sizeof(TraceItem) == 72 && sizeof(TraceBatch) == 48);

void WriteRecord(std::FILE* f, uint32_t type, const void* payload, uint32_t bytes, const void* extra = nullptr,
                 uint32_t extra_bytes = 0, const void* extra2 = nullptr, uint32_t extra2_bytes = 0) {
	const uint32_t head[2] = {type, bytes + extra_bytes + extra2_bytes};
	std::fwrite(head, sizeof(head), 1, f);
	std::fwrite(payload, bytes, 1, f);
	if (extra_bytes != 0) {
		std::fwrite(extra, extra_bytes, 1, f);
	}
	if (extra2_bytes != 0) {
		std::fwrite(extra2, extra2_bytes, 1, f);
	}
}

// What the profiler keeps of a buffer until its batch has completed.
struct BufferProfile {
	uint64_t                  number = 0;
	bool                      acb    = false;
	uint32_t                  slot   = 0;
	Pm4TranslateStats         stats;
	uint32_t                  ib_dw      = 0;
	double                    enqueued   = 0.0;
	double                    translated = 0.0;
	std::vector<ProfiledItem> items;
};
struct BatchProfile {
	uint64_t                   id       = 0;
	uint64_t                   seq      = 0;
	double                     cs_begin = 0.0;
	double                     cs_end   = 0.0;
	uint32_t                   handles  = 0;
	std::vector<BufferProfile> buffers;
};

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
	// Lock-free: Kyty's command processor asks for every draw and dispatch, while the pipeline's
	// worker may hold m_mutex for a whole submission.
	bool Active() const { return m_active.load(std::memory_order_acquire); }

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

	// `acb`: an asynchronous compute buffer. ACBs go through their own translator (their own shadow
	// of the compute registers) and run on the same gfx ring as the DCBs, in guest submission order.
	// `enqueued`: when the guest submitted it (pipeline mode), for the profile. `number` receives the
	// buffer's frame / acb number.
	void Submit(std::span<const uint32_t> dcb, bool acb = false, double enqueued = 0.0, uint64_t* number = nullptr) {
		const double                t_enter = NowSecShadow();
		std::lock_guard<std::mutex> lock(m_mutex);
		const double                t_locked = NowSecShadow();
		m_wait_lock_s += t_locked - t_enter;
		struct BusyScope {
			double& total;
			double  start;
			~BusyScope() { total += NowSecShadow() - start; }
		} busy {m_busy_s, t_locked};
		if (!m_initialized) {
			Init();
			m_initialized = true;
		}
		if (!m_enabled || m_disabled || (acb && m_acb_mode == AcbMode::Off)) {
			return;
		}

		const uint64_t     frame = acb ? m_acbs++ : m_frames++;
		const char* const  kind  = acb ? "acb" : "frame";
		Pm4Translator&     tr    = acb ? m_acb_translator : m_translator;
		if (number != nullptr) {
			*number = frame;
		}
		const bool profiling = m_profiling.load(std::memory_order_relaxed);
		tr.SetProfile(profiling && m_profile_items ? &m_ring : nullptr);
		if (m_markers) {
			tr.SetDrawMarkers(m_ring.va - kProfileRingOffset + kMarkerOffset,
			                  (acb ? 0x80000000u : 0u) | static_cast<uint32_t>(frame & 0x7fffffffu));
		}
		SetStage(Stage::Translate, frame);
		const double       t_translate = NowSecShadow();
		Pm4TranslateResult result      = tr.Translate(dcb);
		const double       translate_ms = (NowSecShadow() - t_translate) * 1e3;
		for (const auto& h: result.find_hits) {
			const auto key = std::make_tuple(h.shader, h.user_reg, h.depth, h.word);
			if (m_find_seen.insert(key).second && m_find_seen.size() <= 64) {
				std::fprintf(stderr,
				             "native shadow: FIND %s %llu %s %u: shader 0x%llx (es 0x%llx) user reg 0x%03x depth %u table 0x%llx "
				             "word %u -> %s 0x%llx\n",
				             kind, static_cast<unsigned long long>(frame), h.dispatch ? "dispatch" : "draw", h.index,
				             static_cast<unsigned long long>(h.shader), static_cast<unsigned long long>(h.es), h.user_reg,
				             h.depth, static_cast<unsigned long long>(h.table), h.word, h.image ? "T#" : "ptr",
				             static_cast<unsigned long long>(h.target));
			}
		}
		// KYTY_NATIVE_GPU_LOG_WAITS=N: report the waits of the first N buffers that were not met when
		// translated, i.e. work the native path runs before what the guest made it wait for.
		static const unsigned long log_waits = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_LOG_WAITS");
			return v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
		}();
		if (log_waits != 0 && m_waits_logged < log_waits && !result.unmet_waits.empty()) {
			m_waits_logged++;
			for (const auto& w: result.unmet_waits) {
				std::fprintf(stderr,
				             "native shadow: WAIT %s %llu dw %u (after %u draws, %u dispatches): [0x%llx] = 0x%llx, "
				             "needs func %u ref 0x%llx\n",
				             kind, static_cast<unsigned long long>(frame), w.dw, w.draws, w.dispatches,
				             static_cast<unsigned long long>(w.addr), static_cast<unsigned long long>(w.value), w.function,
				             static_cast<unsigned long long>(w.ref));
			}
		}
		if (acb && SkipAcb(frame, result)) {
			return;
		}

		// A wait the GPU runs past (guest waits become barriers) is only safe when the label's writer
		// was submitted before it. Report writers that come later: their work runs after the waiter here.
		if (m_check_order) {
			auto satisfies = [](const Pm4TranslateResult::UnmetWait& w, uint64_t v) {
				switch (w.function) {
					case 1: return v < w.ref;
					case 2: return v <= w.ref;
					case 4: return v != w.ref;
					case 5: return v >= w.ref;
					case 6: return v > w.ref;
					default: return v == w.ref;
				}
			};
			for (const auto& [a, v]: result.label_writes) {
				if (auto it = m_unmet_by_addr.find(a); it != m_unmet_by_addr.end() && satisfies(it->second.wait, v)) {
					m_late_writers++;
					if (m_late_writers <= 24) {
						std::fprintf(stderr,
						             "native shadow: ORDER label 0x%llx (needs 0x%llx) waited for by %s %llu is written by %s %llu, "
						             "submitted later\n",
						             static_cast<unsigned long long>(a), static_cast<unsigned long long>(it->second.wait.ref),
						             it->second.acb ? "acb" : "frame", static_cast<unsigned long long>(it->second.buffer), kind,
						             static_cast<unsigned long long>(frame));
					}
					m_unmet_by_addr.erase(it);
				}
				m_last_label[a] = v;
			}
			for (const auto& w: result.unmet_waits) {
				// Already satisfied by a write in an earlier buffer, which runs first: no hazard.
				if (auto it = m_last_label.find(w.addr); it != m_last_label.end() && satisfies(w, it->second)) {
					continue;
				}
				if (m_unmet_by_addr.size() < 4096) {
					m_unmet_by_addr[w.addr] = {w, acb, frame};
				}
			}
		}
		for (const auto& [off, bytes]: result.gds_dma) {
			if (m_gds_seen.insert((static_cast<uint64_t>(acb) << 63u) | (static_cast<uint64_t>(off) << 20u) | bytes).second &&
			    m_gds_seen.size() <= 64) {
				std::fprintf(stderr, "native shadow: GDS DMA %s %s offset 0x%x, %u bytes\n", acb ? "acb" : "dcb",
				             (off & 0x80000000u) != 0 ? "write" : "read", off & 0x7fffffffu, bytes);
			}
		}
		(acb ? m_gds_dma_acb : m_gds_dma_dcb)[0] += result.stats.dma_gds_dst;
		(acb ? m_gds_dma_acb : m_gds_dma_dcb)[1] += result.stats.dma_gds_src;
		DetectShaders(result.shader_addrs);
		// KYTY_NATIVE_GPU_INSPECT_AFTER_S: only inspect draws this many seconds after start (past the title).
		static const double inspect_after = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_INSPECT_AFTER_S");
			return v != nullptr ? std::strtod(v, nullptr) : 0.0;
		}();
		static const double start_s = NowSecShadow();
		for (const auto& snap: result.inspected) {
			if (NowSecShadow() - start_s < inspect_after) {
				break;
			}
			InspectDraw(frame, snap);
		}
		// A shader that uses GDS hung the board in ASTRO BOT even with a GDS partition; until that is
		// understood such DCBs stay on Kyty unless KYTY_NATIVE_GPU_GDS_SHADERS=1.
		// Skip mode learns GDS shaders as they appear: the DCB that introduces one was translated
		// before the scan, so it goes to Kyty whole and the translator drops that shader's draws from
		// the next DCB on.
		if (result.ok && !acb && m_skip_gds_shaders && m_gds_shaders.size() != m_skipped_shader_count) {
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
					std::fprintf(stderr, "native shadow: untranslatable packet op 0x%02x (first in %s %llu)\n", op, kind,
					             static_cast<unsigned long long>(frame));
				}
			}
			if (m_refused < 8) {
				std::fprintf(stderr,
				             "native shadow: %s %llu refused, unknown packet op 0x%02x at dw %u "
				             "(%u packets translated)%s",
				             kind, static_cast<unsigned long long>(frame), result.unknown_op,
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
		wrapped.reserve(result.ib.size() + 48);
		// Profile window: the buffer's own slot brackets it (issued when the CP starts it, done when it
		// has finished); its draws and dispatches have theirs inside the IB.
		const bool     profiled_buffer = profiling && m_pipelined;
		const uint32_t buffer_slot     = profiled_buffer ? m_ring.Take() : 0;
		if (profiled_buffer) {
			ClearSlot(buffer_slot);
			for (const auto& it: result.profiled) {
				ClearSlot(it.slot);
			}
			EmitGpuClockCopy(wrapped, m_ring.SlotVa(buffer_slot));
		}
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
		if (profiled_buffer) {
			EmitGpuClockEop(wrapped, m_ring.SlotVa(buffer_slot) + 8);
		}
		while ((wrapped.size() & 7u) != 0) {
			wrapped.push_back(0xffff1000u);
		}
		if (!m_breadcrumbs.empty()) {
			WriteBreadcrumb(acb ? "acb" : "", frame, dcb, result, wrapped);
		}
		if (m_dump != nullptr) {
			DumpPm4(acb, frame, dcb, wrapped);
		}
		// Pipelined: the IB joins the batch the pipeline submits with one CS once it has taken every
		// queued buffer (FlushBatch). One BO-list validation then covers them all.
		if (m_pipelined) {
			if ((m_batch.size() + wrapped.size()) * 4 > NativeSubmitter::kSlotSize) {
				SubmitBatchLocked();
			}
			m_batch.insert(m_batch.end(), wrapped.begin(), wrapped.end());
			if (profiled_buffer) {
				BufferProfile p;
				p.number     = frame;
				p.acb        = acb;
				p.slot       = buffer_slot;
				p.stats      = result.stats;
				p.ib_dw      = static_cast<uint32_t>(wrapped.size());
				p.enqueued   = enqueued;
				p.translated = NowSecShadow();
				p.items      = std::move(result.profiled);
				m_pending_buffers.push_back(std::move(p));
			}
			return;
		}
		std::span<const uint32_t> ib = wrapped;
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
			             "native shadow: %s %llu submit failed: %s (after %d retries, %zu BOs)\n", kind,
			             static_cast<unsigned long long>(frame), std::strerror(-r.err), r.retries,
			             handles.size());
			Disable("submit error");
			return;
		}
		m_last_seq = r.seq;
		if (r.timed_out || r.reset || !r.completed) {
			std::fprintf(stderr,
			             "native shadow: %s %llu %s after %.1f ms (ib %zu dw, %u draws); "
			             "disabling native path\n",
			             kind, static_cast<unsigned long long>(frame),
			             r.timed_out ? "TIMED OUT" : (r.reset ? "context RESET" : "did not complete"),
			             r.seconds * 1e3, result.ib.size(), result.stats.draws);
			Disable("timeout or reset");
			return;
		}

		if (acb) {
			m_acbs_completed++;
			if (m_acbs_completed % 30 == 0 && WaitIdle() && m_gem->DummyWritten() && m_acb_dummy_writes++ < 8) {
				std::fprintf(stderr,
				             "native shadow: acb %llu: the GPU wrote to guest memory that was not resident "
				             "(a reference the translator missed)\n",
				             static_cast<unsigned long long>(frame));
			}
			if (m_acbs_completed <= 3 || m_acbs_completed % 600 == 0) {
				std::fprintf(stderr, "native shadow: acb %llu ok (%llu done), %.2f ms, ib %zu dw, %u dispatches\n",
				             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(m_acbs_completed),
				             r.seconds * 1e3, result.ib.size(), result.stats.dispatches);
			}
			return;
		}
		m_completed++;
		if (m_completed % 30 == 0 && WaitIdle() && m_gem->DummyWritten()) {
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
			             "native shadow: blocks made resident by: cpu writes %llu, shaders %llu, user-data pointers %llu, "
			             "direct refs %llu, table buffers %llu, table images %llu\n",
			             static_cast<unsigned long long>(m_promoted_by[0]), static_cast<unsigned long long>(m_promoted_by[1]),
			             static_cast<unsigned long long>(m_promoted_by[2]), static_cast<unsigned long long>(m_promoted_by[3]),
			             static_cast<unsigned long long>(m_promoted_by[4]), static_cast<unsigned long long>(m_promoted_by[5]));
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

	// Display buffers are written by the GPU (often by compute through a descriptor the translator
	// cannot see), so whatever is flipped is made resident; until then such writes land in the dummy.
	void KeepResident(uint64_t addr, uint64_t size) {
#if KYTY_GEM_BACKING_SUPPORTED
		std::lock_guard<std::mutex> lock(m_mutex);
		if (m_gem == nullptr || !m_flipped.insert(addr).second) {
			return;
		}
		const uint32_t promoted = m_gem->PromoteGuestRange(addr, size);
		std::fprintf(stderr, "native shadow: display buffer 0x%llx (%llu MiB) kept resident, %u block(s) promoted\n",
		             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size >> 20u), promoted);
#else
		(void)addr;
		(void)size;
#endif
	}

	// The dummy check needs an idle GPU; in async mode that means waiting for what is in flight.
	bool WaitIdle() {
#if KYTY_GEM_BACKING_SUPPORTED
		return !m_submitter->Async() || m_submitter->WaitIdle(m_timeout_sec);
#else
		return true;
#endif
	}

	// Per-flip timing (KYTY_NATIVE_GPU_FLIP_STATS=1): native time spent in Submit and waiting for its
	// lock between flips, with the presenter's share passed in.
	uint64_t ResidentMiB() const {
#if KYTY_GEM_BACKING_SUPPORTED
		return m_gem != nullptr ? m_gem->ResidentBytes() >> 20u : 0;
#else
		return 0;
#endif
	}

	void AddGpuBusy(double seconds) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_gpu_busy_s += seconds;
	}

	// One pipelined buffer finished: `queued` waiting for the worker, `latency` from the guest's
	// submit to its continuation being done, `then` the continuation (Kyty's command processor).
	void AddChain(double queued, double latency, double then) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_chain_n++;
		m_chain_queued_s += queued;
		m_chain_latency_s += latency;
		m_chain_then_s += then;
		m_chain_max_s = std::max(m_chain_max_s, latency);
	}

	void FlipStats(double present_s) {
		std::lock_guard<std::mutex> lock(m_mutex);
		const double now = NowSecShadow();
		m_flips++;
		m_present_s += present_s;
		if (m_flip_t0 == 0.0) {
			m_flip_t0 = now;
			m_busy_s = m_wait_lock_s = m_present_s = 0.0;
			m_flips  = 0;
			return;
		}
		if (m_flips == 120) {
			const double per = 1e3 / static_cast<double>(m_flips);
			std::fprintf(stderr,
			             "native shadow: per flip: interval %.1f ms, native submit %.1f ms, waiting for the native "
			             "lock %.1f ms, presenter %.1f ms, GPU busy %.1f ms; %llu MiB guest memory resident\n",
			             (now - m_flip_t0) * per, m_busy_s * per, m_wait_lock_s * per, m_present_s * per,
			             m_gpu_busy_s * per, static_cast<unsigned long long>(ResidentMiB()));
#if defined(__GLIBC__)
			if (EnvOn("KYTY_NATIVE_GPU_MALLOC_STATS")) {
				const struct mallinfo2 mi = mallinfo2();
				std::fprintf(stderr, "native shadow: host malloc: %zu MiB in arenas, %zu MiB mmapped, %zu MiB in use\n",
				             mi.arena >> 20u, mi.hblkhd >> 20u, (mi.uordblks + mi.hblkhd) >> 20u);
			}
#endif
			// The pipelined path never reaches the per-submission check: look at the dummy here. The GPU
			// may be writing it right now; a write that lands during the clear is caught next time.
			if (m_gem != nullptr && m_gem->DummyWritten()) {
				m_dummy_writes++;
			}
			if (m_dummy_writes > 0) {
				std::fprintf(stderr, "native shadow: %u checks found GPU writes to non-resident guest memory\n",
				             m_dummy_writes);
			}
			if (m_gds_dma_dcb[0] + m_gds_dma_dcb[1] + m_gds_dma_acb[0] + m_gds_dma_acb[1] > 0) {
				std::fprintf(stderr, "native shadow: DMA_DATA with GDS: dcb %llu to / %llu from, acb %llu to / %llu from\n",
				             static_cast<unsigned long long>(m_gds_dma_dcb[0]), static_cast<unsigned long long>(m_gds_dma_dcb[1]),
				             static_cast<unsigned long long>(m_gds_dma_acb[0]), static_cast<unsigned long long>(m_gds_dma_acb[1]));
			}
			if (m_chain_n > 0) {
				const double per_buf = 1e3 / static_cast<double>(m_chain_n);
				std::fprintf(stderr,
				             "native shadow: per buffer (%.1f per flip): queued %.2f ms, submit to done %.2f ms "
				             "(max %.1f), continuation %.2f ms\n",
				             static_cast<double>(m_chain_n) / static_cast<double>(m_flips), m_chain_queued_s * per_buf,
				             m_chain_latency_s * per_buf, m_chain_max_s * 1e3, m_chain_then_s * per_buf);
			}
			m_chain_n = 0;
			m_chain_queued_s = m_chain_latency_s = m_chain_then_s = m_chain_max_s = 0.0;
			m_flip_t0 = now;
			m_busy_s = m_wait_lock_s = m_present_s = m_gpu_busy_s = 0.0;
			m_flips  = 0;
		}
	}

	bool NativeAcb() const { return Active() && m_acb_mode == AcbMode::On; }

	bool DmabufPieces(uint64_t vaddr, uint64_t size, std::vector<NativeDmabufPiece>& out) {
#if KYTY_GEM_BACKING_SUPPORTED
		std::vector<GemBacking::DmabufPiece> pieces;
		if (m_gem == nullptr || !m_gem->DmabufPieces(vaddr, size, pieces)) {
			return false;
		}
		out.clear();
		for (const auto& p: pieces) {
			out.push_back(NativeDmabufPiece {p.fd, p.bo_size, p.bo_offset, p.len});
		}
		return true;
#else
		(void)vaddr;
		(void)size;
		(void)out;
		return false;
#endif
	}

	// Pipeline mode: Submit batches IBs; FlushBatch submits them with one CS.
	void SetPipelined() {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_pipelined = true;
	}
	void FlushBatch() {
		std::lock_guard<std::mutex> lock(m_mutex);
		SubmitBatchLocked();
	}

	// The submission the last Submit queued in async mode (0: none), for the pipeline to wait on.
	uint64_t TakeLastSeq() {
		std::lock_guard<std::mutex> lock(m_mutex);
		return std::exchange(m_last_seq, 0);
	}

	// Waits for an async submission; false on a timeout (the native path then disables itself).
	bool WaitSubmission(uint64_t seq) {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter != nullptr && m_submitter->WaitSeq(seq, m_timeout_sec)) {
			return true;
		}
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_disabled) {
			std::fprintf(stderr, "native shadow: submission %llu TIMED OUT; disabling native path\n",
			             static_cast<unsigned long long>(seq));
			Disable("timeout");
		}
#else
		(void)seq;
#endif
		return false;
	}

private:
	void SubmitBatchLocked() {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_batch.empty() || m_disabled || m_submitter == nullptr) {
			m_batch.clear();
			return;
		}
		std::vector<uint32_t>   handles  = m_gem->CreatedHandles();
		const double            cs_begin = NowSecShadow();
		NativeSubmitter::Result r        = m_submitter->Submit(m_batch, handles, m_timeout_sec);
		for (int retry = 0; !r.submitted && r.err == -ENOENT && retry < 3; retry++) {
			handles = m_gem->CreatedHandles();
			r       = m_submitter->Submit(m_batch, handles, m_timeout_sec);
		}
		m_batch.clear();
		if (!r.submitted) {
			std::fprintf(stderr, "native shadow: batch submit failed: %s (%zu BOs)\n", std::strerror(-r.err),
			             handles.size());
			m_pending_buffers.clear();
			Disable("submit error");
			return;
		}
		m_cs_ms_total += r.cs_seconds * 1e3;
		m_last_seq = r.seq;
		if (!m_pending_buffers.empty()) {
			BatchProfile b;
			b.id       = m_batch_ids++;
			b.seq      = r.seq;
			b.cs_begin = cs_begin;
			b.cs_end   = NowSecShadow();
			b.handles  = static_cast<uint32_t>(handles.size());
			b.buffers  = std::move(m_pending_buffers);
			m_pending_buffers.clear();
			m_pending_batches.push_back(std::move(b));
		}
#endif
	}

public:
	// The profiled batches submitted since the last call, for the completion thread to report once
	// they have finished.
	std::vector<BatchProfile> TakeBatchProfiles() {
		std::lock_guard<std::mutex> lock(m_mutex);
		return std::exchange(m_pending_batches, {});
	}

	// Called by the completion thread once `batches` have finished (`fence`: when it saw that).
	void ProfileBatches(const std::vector<BatchProfile>& batches, double fence) {
#if KYTY_GEM_BACKING_SUPPORTED
		std::lock_guard<std::mutex> lock(m_trace_mutex);
		if (m_trace == nullptr) {
			return;
		}
		const volatile uint64_t* slots = m_submitter->ProfileCpu() + kProfileRingOffset / 8;
		for (const auto& b: batches) {
			const TraceBatch tb {b.id, b.seq, ToNs(b.cs_begin), ToNs(b.cs_end), ToNs(fence),
			                     static_cast<uint32_t>(b.buffers.size()), b.handles};
			WriteRecord(m_trace, kTraceBatch, &tb, sizeof(tb));
			for (const auto& p: b.buffers) {
				TraceBuffer t {};
				t.batch         = b.id;
				t.number        = p.number;
				t.enqueue_ns    = ToNs(p.enqueued);
				t.translated_ns = ToNs(p.translated);
				t.gpu_issued    = slots[2 * p.slot];
				t.gpu_done      = slots[2 * p.slot + 1];
				t.acb           = p.acb ? 1u : 0u;
				t.draws         = p.stats.draws;
				t.dispatches    = p.stats.dispatches;
				t.flips         = p.stats.marker_flip;
				t.ib_dw         = p.ib_dw;
				t.wait_barriers = p.stats.wait_barriers;
				WriteRecord(m_trace, kTraceBuffer, &t, sizeof(t));
				// GPU busy time, buffers taken in ring order: the part of [issued, done] not already covered.
				if (t.gpu_issued != 0 && t.gpu_done >= t.gpu_issued) {
					const uint64_t from = std::max(t.gpu_issued, m_busy_until);
					if (t.gpu_done > from) {
						m_busy_ticks[t.acb] += t.gpu_done - from;
					}
					m_busy_until = std::max(m_busy_until, t.gpu_done);
					m_span_first = m_span_first == 0 ? t.gpu_issued : m_span_first;
					m_span_last  = std::max(m_span_last, t.gpu_done);
				}
				for (const auto& it: p.items) {
					TraceItem ti {};
					ti.number     = p.number;
					ti.shader     = it.shader;
					ti.es         = it.es;
					ti.gpu_issued = slots[2 * it.slot];
					ti.gpu_done   = slots[2 * it.slot + 1];
					ti.acb        = t.acb;
					ti.index      = it.index;
					ti.dw         = it.dw;
					ti.op         = it.op;
					ti.args[0]    = it.args[0];
					ti.args[1]    = it.args[1];
					ti.args[2]    = it.args[2];
					WriteRecord(m_trace, kTraceItem, &ti, sizeof(ti));
				}
			}
		}
#else
		(void)batches;
		(void)fence;
#endif
	}

	// Kyty's command processor ran buffer `number` from `start` to `end` (pipeline mode).
	void ProfileThen(bool acb, uint64_t number, double start, double end) {
		std::lock_guard<std::mutex> lock(m_trace_mutex);
		if (m_trace != nullptr) {
			const TraceThen t {number, ToNs(start), ToNs(end), acb ? 1u : 0u, 0};
			WriteRecord(m_trace, kTraceThen, &t, sizeof(t));
		}
	}

	// Every video-out flip: opens the capture window KYTY_NATIVE_GPU_CAPTURE_AT seconds after the first
	// submission, records the flip and closes the window again.
	void NoteFlip(uint64_t flip) {
		if (!m_capture_ready.load(std::memory_order_acquire)) {
			return;
		}
		m_flip_no.store(flip, std::memory_order_relaxed);
		std::lock_guard<std::mutex> lock(m_trace_mutex);
		if (!m_capture_started) {
			if (NowSecShadow() - m_capture_t0 < m_capture_at) {
				return;
			}
			m_capture_started = true;
			m_capture_flip0   = flip;
			StartCapture();
		}
		const uint64_t k = flip - m_capture_flip0;
		if (m_dumping.load(std::memory_order_relaxed) && k >= m_dump_flips) {
			m_dumping.store(false, std::memory_order_relaxed); // Submit closes the file
		}
		if (m_trace == nullptr) {
			return;
		}
		const TraceFlip f {flip, ToNs(NowSecShadow())};
		WriteRecord(m_trace, kTraceFlip, &f, sizeof(f));
		WriteCalib();
		std::fflush(m_trace);
		if (m_profiling.load(std::memory_order_relaxed) && k >= m_profile_flips) {
			m_profiling.store(false, std::memory_order_relaxed);
		}
		// Batches still in flight when the window closed report a little later.
		if (k >= m_profile_flips + 30) {
			const double span = m_span_last > m_span_first ? static_cast<double>(m_span_last - m_span_first) : 0.0;
			const double per  = 1e3 / (m_counter_khz * 1e3) / static_cast<double>(m_profile_flips);
			std::fprintf(stderr,
			             "native shadow: profile of %u flips: GPU span %.2f ms/flip, busy %.2f ms/flip (dcb %.2f, acb %.2f), "
			             "idle %.2f ms/flip; trace in %s/trace.bin\n",
			             m_profile_flips, span * per, static_cast<double>(m_busy_ticks[0] + m_busy_ticks[1]) * per,
			             static_cast<double>(m_busy_ticks[0]) * per, static_cast<double>(m_busy_ticks[1]) * per,
			             (span - static_cast<double>(m_busy_ticks[0] + m_busy_ticks[1])) * per, m_capture_dir.c_str());
			std::fclose(m_trace);
			m_trace = nullptr;
		}
	}

	// KYTY_NATIVE_GPU_DRAW_MARKERS: the last draw/dispatch the CP started and the last that finished.
	void ReportMarkers(const char* why) const {
#if KYTY_GEM_BACKING_SUPPORTED
		if (!m_markers || m_submitter == nullptr) {
			return;
		}
		const volatile uint64_t* m = m_submitter->ProfileCpu() + kMarkerOffset / 8;
		auto describe = [](uint64_t v, char* out, size_t n) {
			const auto tag = static_cast<uint32_t>(v >> 32u);
			std::snprintf(out, n, "%s %u item %u", (tag & 0x80000000u) != 0 ? "acb" : "frame", tag & 0x7fffffffu,
			              static_cast<uint32_t>(v));
		};
		char issued[64];
		char done[64];
		describe(m[0], issued, sizeof(issued));
		describe(m[1], done, sizeof(done));
		std::fprintf(stderr, "native shadow: MARKERS (%s): last draw/dispatch started: %s; last finished: %s\n", why, issued,
		             done);
#else
		(void)why;
#endif
	}

private:
	// The profile buffer: markers first, then the timestamp ring.
	static constexpr uint64_t kMarkerOffset      = 0;
	static constexpr uint64_t kProfileRingOffset = 4096;
	static constexpr uint64_t kProfileBytes      = 8ull << 20u;

	void ClearSlot(uint32_t slot) {
#if KYTY_GEM_BACKING_SUPPORTED
		auto* s = const_cast<volatile uint64_t*>(m_submitter->ProfileCpu() + kProfileRingOffset / 8);
		s[2 * slot]     = 0;
		s[2 * slot + 1] = 0;
#else
		(void)slot;
#endif
	}

	// Under m_trace_mutex.
	void WriteCalib() {
#if KYTY_GEM_BACKING_SUPPORTED
		TraceCalib c {};
		c.cpu_before_ns = ToNs(NowSecShadow());
		c.gpu           = m_submitter->GpuClock();
		c.cpu_after_ns  = ToNs(NowSecShadow());
		WriteRecord(m_trace, kTraceCalib, &c, sizeof(c));
#endif
	}

	// Under m_trace_mutex.
	void StartCapture() {
		if (m_profile_flips != 0) {
			if (!m_pipelined) {
				std::fprintf(stderr, "native shadow: the profile needs the pipeline (KYTY_NATIVE_GPU_THREAD); no trace\n");
			} else if (std::FILE* f = std::fopen((m_capture_dir + "/trace.bin").c_str(), "wb"); f != nullptr) {
				m_trace = f;
				const TraceHeader h {1, m_counter_khz, ToNs(NowSecShadow())};
				WriteRecord(m_trace, kTraceHeader, &h, sizeof(h));
				WriteCalib();
				m_profiling.store(true, std::memory_order_relaxed);
			}
		}
		if (m_dump_flips != 0) {
			std::lock_guard<std::mutex> lock(m_mutex);
			m_dump = std::fopen((m_capture_dir + "/pm4.bin").c_str(), "wb");
			m_dumping.store(m_dump != nullptr, std::memory_order_relaxed);
		}
		std::fprintf(stderr, "native shadow: capture started at flip %llu: profile %u flips%s, PM4 dump %u flips, in %s\n",
		             static_cast<unsigned long long>(m_capture_flip0), m_trace != nullptr ? m_profile_flips : 0,
		             m_profile_items ? " with every draw/dispatch" : "", m_dump != nullptr ? m_dump_flips : 0,
		             m_capture_dir.c_str());
	}

	// KYTY_NATIVE_GPU_PM4_DUMP_FLIPS: the guest buffer, its IB and, for the decoder's V#/T#/S# tables,
	// guest memory that the IB's shader user data points at (two levels deep), taken now, i.e. shortly
	// before the GPU runs it. Under m_mutex.
	void DumpPm4(bool acb, uint64_t number, std::span<const uint32_t> guest, std::span<const uint32_t> ib) {
		if (!m_dumping.load(std::memory_order_relaxed)) {
			std::fclose(m_dump);
			m_dump = nullptr;
			std::fprintf(stderr, "native shadow: PM4 dump done (%llu buffers, %llu MiB of guest memory) in %s/pm4.bin\n",
			             static_cast<unsigned long long>(m_dumped_buffers),
			             static_cast<unsigned long long>(m_dumped_bytes >> 20u), m_capture_dir.c_str());
			return;
		}
		m_dumped_buffers++;
		const DumpBuffer h {number, m_flip_no.load(std::memory_order_relaxed), acb ? 1u : 0u,
		                    static_cast<uint32_t>(guest.size()), static_cast<uint32_t>(ib.size()), 0};
		WriteRecord(m_dump, kDumpBuffer, &h, sizeof(h), guest.data(), static_cast<uint32_t>(guest.size_bytes()), ib.data(),
		            static_cast<uint32_t>(ib.size_bytes()));
#if KYTY_GEM_BACKING_SUPPORTED
		// Shader user data (SPI_SHADER_USER_DATA_*, COMPUTE_USER_DATA_*) and SET_BASE addresses.
		auto user_data = [](uint32_t off) {
			return (off >= 0x240u && off < 0x250u) || (off < 0x180u && (off & 0x3fu) >= 0x0Cu && (off & 0x3fu) <= 0x2Bu);
		};
		auto plausible = [](uint64_t a) {
			return a >= 0x100000000ull && a < 0x100000000000ull && (a & 3u) == 0;
		};
		std::vector<uint64_t> level;
		std::vector<uint32_t> sh(0x400, 0);
		for (size_t dw = 0; dw < ib.size();) {
			const uint32_t header = ib[dw];
			const uint32_t size   = (header & 0x3fffff00u) == 0x3fff1000u ? 1u : ((header >> 16u) & 0x3fffu) + 2u;
			const uint32_t op     = (header >> 8u) & 0xffu;
			if ((header >> 30u) != 3u || dw + size > ib.size()) {
				break;
			}
			if (op == 0x76 && size >= 3) { // SET_SH_REG
				const uint32_t base = ib[dw + 1] & 0xffffu;
				// Every pair of user-data registers around a write (a pair whose other half is written
				// next is taken again then, with both halves current).
				for (uint32_t i = 0; i + 2 < size; i++) {
					const uint32_t off = base + i;
					if (off + 1 >= sh.size() || off == 0) {
						continue;
					}
					sh[off] = ib[dw + 2 + i];
					if (user_data(off) && user_data(off - 1)) {
						level.push_back(sh[off - 1] | (static_cast<uint64_t>(sh[off] & 0xffffu) << 32u));
					}
					if (user_data(off) && user_data(off + 1)) {
						level.push_back(sh[off] | (static_cast<uint64_t>(sh[off + 1] & 0xffffu) << 32u));
					}
				}
			} else if (op == 0x11 && size >= 4) { // SET_BASE
				level.push_back(ib[dw + 2] | (static_cast<uint64_t>(ib[dw + 3] & 0xffffu) << 32u));
			}
			dw += size;
		}
		// Each buffer gets its own copies (tables are rewritten every frame), up to 1 GiB in all.
		constexpr uint32_t    kTableBytes = 512;
		std::vector<uint32_t> data(kTableBytes / 4);
		std::set<uint64_t>    taken;
		for (int depth = 0; depth < 2 && !level.empty(); depth++) {
			std::vector<uint64_t> next;
			for (uint64_t a: level) {
				if (!plausible(a) || m_dumped_bytes > (1ull << 30u) || !taken.insert(a).second) {
					continue;
				}
				const auto got = static_cast<uint32_t>(m_gem->ReadGuest(a, data.data(), kTableBytes));
				if (got == 0) {
					continue;
				}
				const DumpMemory mh {a, got, 0};
				WriteRecord(m_dump, kDumpMemory, &mh, sizeof(mh), data.data(), got);
				m_dumped_bytes += got;
				for (uint32_t i = 0; depth == 0 && i + 1 < got / 4; i++) {
					next.push_back(data[i] | (static_cast<uint64_t>(data[i + 1] & 0xffffu) << 32u));
				}
			}
			level = std::move(next);
		}
#endif
	}


	// KYTY_NATIVE_GPU_ACB: "1" runs async compute buffers natively (and Kyty skips their dispatches),
	// "dry" only translates them and reports what the translator lacks.
	enum class AcbMode { Off, Dry, On };

	// Dry mode: count and report, never submit. Returns true when the ACB is done with.
	bool SkipAcb(uint64_t n, const Pm4TranslateResult& result) {
		if (m_acb_mode != AcbMode::Dry) {
			return false;
		}
		if (result.ok) {
			m_acbs_completed++;
		} else {
			m_acbs_refused++;
			for (uint32_t op: result.unknown_ops) {
				if (m_unknown_acb_ops.insert(op).second) {
					std::fprintf(stderr, "native shadow: acb dry run: untranslatable packet op 0x%02x (first in acb %llu)\n",
					             op, static_cast<unsigned long long>(n));
				}
			}
		}
		if (n < 3 || n % 600 == 0) {
			std::fprintf(stderr, "native shadow: acb dry run: acb %llu %s, %u dispatches; %llu ok, %llu refused so far\n",
			             static_cast<unsigned long long>(n), result.ok ? "ok" : "refused", result.stats.dispatches,
			             static_cast<unsigned long long>(m_acbs_completed), static_cast<unsigned long long>(m_acbs_refused));
		}
		return true;
	}

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
	void WriteBreadcrumb(const std::string& prefix, uint64_t frame, std::span<const uint32_t> dcb,
	                     const Pm4TranslateResult& result, std::span<const uint32_t> ib) const {
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
		// KYTY_NATIVE_GPU_BREADCRUMB_SLOTS: how many recent submissions to keep (default 4).
		static const unsigned slots = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_BREADCRUMB_SLOTS");
			const unsigned long n = v != nullptr ? std::strtoul(v, nullptr, 0) : 4ul;
			return static_cast<unsigned>(n == 0 ? 4ul : n);
		}();
		const unsigned slot = static_cast<unsigned>(frame % slots);
		write_file(prefix + "dcb_" + std::to_string(slot) + ".bin", dcb.data(), dcb.size_bytes());
		write_file(prefix + "ib_" + std::to_string(slot) + ".bin", ib.data(), ib.size_bytes());
		char summary[512];
		const int n = std::snprintf(summary, sizeof(summary),
		                            "%s %llu slot %u: dcb %zu dw, ib %zu dw, %u draws, %u dispatches, %u dma, "
		                            "%u set_base, %llu MiB resident\n",
		                            prefix.empty() ? "frame" : prefix.c_str(), static_cast<unsigned long long>(frame), slot, dcb.size(), ib.size(),
		                            result.stats.draws, result.stats.dispatches, result.stats.dma_data,
		                            result.stats.set_base, static_cast<unsigned long long>(m_gem->ResidentBytes() >> 20u));
		write_file(prefix + "last.txt", summary, static_cast<size_t>(n));
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
		if (m_inspected++ >= 6) {
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
						// Small buffers (constants, exposure and other per-frame scalars): their first values.
						const uint32_t stride = (t[k + 1] >> 16u) & 0x3fffu;
						const uint64_t bytes  = static_cast<uint64_t>(t[k + 2]) * (stride != 0 ? stride : 1u);
						if (vbase >= 0x100000000ull && vbase < 0x10000000000ull && bytes != 0 && bytes <= 4096) {
							float f[8] = {};
							const size_t got = m_gem->ReadGuest(vbase, f, std::min<uint64_t>(sizeof(f), bytes)) / 4;
							std::fprintf(stderr, "      V# 0x%llx data:", static_cast<unsigned long long>(vbase));
							for (size_t q = 0; q < got; q++) {
								std::fprintf(stderr, " %g", static_cast<double>(f[q]));
							}
							std::fprintf(stderr, "\n");
						}
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
		m_promoted_by[0] += m_gem->PromoteTouched();

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
		// `kind` counts which reference made blocks resident (m_promoted_by, logged with the averages).
		auto promote = [&](uint64_t addr, uint64_t size, int kind) {
			for (uint64_t b = addr >> kBucketShift; b <= (addr + size - 1) >> kBucketShift; b++) {
				if (m_resident_buckets.insert(b).second) {
					m_promoted_by[kind] += m_gem->PromoteGuestRange(b << kBucketShift, 1ull << kBucketShift);
				}
			}
		};

		for (uint64_t addr: result.shader_addrs) {
			if (backed(addr)) {
				promote(addr, 64u * 1024u, 1);
			}
		}
		for (uint32_t b: result.ref_buckets) {
			promote(static_cast<uint64_t>(b) << kBucketShift, 1ull << kBucketShift, 3);
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
				promote(ref.addr, ref.size != 0 ? ref.size : kTableSpan, ref.pointer ? 2 : 3);
				if (ref.pointer) {
					tables.push_back(ref.addr);
				}
			}
		}
		// Following descriptor tables guesses: any dword pair that looks like a guest address makes its
		// whole block resident. In ASTRO BOT that was 336 of 712 blocks (5.4 GB) and nothing needed them
		// (what the GPU reads was already resident through CPU writes and direct references), so tables
		// are only followed with KYTY_NATIVE_GPU_TABLE_PROMOTE=1. A GPU write into memory left out still
		// shows as a dummy write in the log.
		static const bool follow_tables = EnvOn("KYTY_NATIVE_GPU_TABLE_PROMOTE");
		if (!follow_tables) {
			tables.clear();
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
						promote(ptr, std::clamp<uint64_t>(bytes, kTableSpan, kBufferCap), 4);
						next.push_back(ptr);
					}
					const uint64_t image = (d[i] | (static_cast<uint64_t>(d[i + 1] & 0xffu) << 32u)) << 8u;
					NoteUnmapped(image, "image", table);
					if (image != 0 && backed(image)) {
						promote(image, kImageSpan, 5);
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
		if (PrimaryFeature("KYTY_NATIVE_GPU_NO_CMASK_CLEAR")) {
			// Only MRT4 by default: ASTRO BOT's CMASK-only fast clear there hangs the CB. Turning it off
			// everywhere leaves other fast-cleared targets stale (8x8 squares around moving objects).
			const char*    v     = std::getenv("KYTY_NATIVE_GPU_NO_CMASK_CLEAR_SLOTS");
			const uint32_t slots = v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0x10u;
			m_translator.SetNoCmaskClear(true, slots);
			std::fprintf(stderr, "native shadow: no CMASK fast clear on targets without DCC in MRT slots 0x%02x\n", slots);
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
			m_acb_translator.SetSkippedOpcodes(skip);
			std::fprintf(stderr, "native shadow: leaving %zu packet type(s) out of the IB: %s\n", skip.size(), ops);
		}
		// KYTY_NATIVE_GPU_GDS_SHADERS=1 runs draws with GDS shaders natively, =skip leaves just those
		// draws out of the IB (the rest of their DCB runs natively); unset leaves whole DCBs to Kyty.
		if (const char* g = std::getenv("KYTY_NATIVE_GPU_GDS_SHADERS"); g != nullptr) {
			m_skip_gds_shaders  = std::strcmp(g, "skip") == 0;
			m_allow_gds_shaders = !m_skip_gds_shaders && g[0] == '1';
		} else {
			m_allow_gds_shaders = EnvOn("KYTY_NATIVE_GPU_PRIMARY"); // primary mode has the GDS partition
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
		InitCapture();
		m_translator.SetGdsAvailable(m_submitter->HasGds());
		m_acb_translator.SetGdsAvailable(m_submitter->HasGds());
		m_acb_translator.SetComputeQueue(true);
		// Residency only needs the 16 MiB buckets the GPU may touch, unless descriptor tables are
		// followed or unmapped references logged: those need every reference.
		if (!EnvOn("KYTY_NATIVE_GPU_TABLE_PROMOTE") && !m_log_unmapped) {
			m_translator.SetCompactRefs(true);
			m_acb_translator.SetCompactRefs(true);
		}
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_PFP_SYNC_INDIRECT"); v != nullptr && v[0] == '0') {
			m_translator.SetPfpSyncIndirect(false);
			m_acb_translator.SetPfpSyncIndirect(false);
		}
		if (NativeLodStats()) {
			m_translator.SetLodStats(true);
			m_acb_translator.SetLodStats(true);
			std::fprintf(stderr, "native shadow: GET_LOD_STATS runs on the GPU (experiment)\n");
		}
		// KYTY_NATIVE_GPU_GUESS_SPAN_KIB: residency for each user-data value that may be an image base.
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_GUESS_SPAN_KIB"); v != nullptr) {
			const uint64_t bytes = std::strtoull(v, nullptr, 0) << 10u;
			m_translator.SetGuessSpan(bytes);
			m_acb_translator.SetGuessSpan(bytes);
			std::fprintf(stderr, "native shadow: image-base guesses make %llu KiB resident\n",
			             static_cast<unsigned long long>(bytes >> 10u));
		}
		// KYTY_NATIVE_GPU_ASYNC=1: submissions return without waiting for the GPU (experiment; the
		// guest's labels are still written by Kyty's CPU command processor, possibly too early).
		if (EnvOn("KYTY_NATIVE_GPU_ASYNC") || ShadowThreaded()) {
			m_submitter->SetAsync(true);
			std::fprintf(stderr, "native shadow: asynchronous submission (experiment)\n");
		}
		if (PrimaryFeature("KYTY_NATIVE_GPU_WAIT_BARRIERS")) {
			m_translator.SetFenceAddress(m_submitter->ScratchVa() + 1024);
			m_acb_translator.SetFenceAddress(m_submitter->ScratchVa() + 1032);
			std::fprintf(stderr, "native shadow: guest waits become GPU barriers\n");
		}
		// KYTY_NATIVE_GPU_FIND=0xADDR:0xSIZE: log the draws/dispatches whose resources reach that range.
		if (const char* find = std::getenv("KYTY_NATIVE_GPU_FIND"); find != nullptr) {
			char*          end  = nullptr;
			const uint64_t addr = std::strtoull(find, &end, 0);
			const uint64_t size = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 0) : 1;
			m_translator.SetFindRange(addr, size);
			m_acb_translator.SetFindRange(addr, size);
			std::fprintf(stderr, "native shadow: looking for references to [0x%llx, +0x%llx)\n",
			             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size));
		}
		if (const char* acb = std::getenv("KYTY_NATIVE_GPU_ACB"); acb == nullptr ? EnvOn("KYTY_NATIVE_GPU_PRIMARY")
		                                                                       : (acb[0] != '\0' && acb[0] != '0')) {
			m_acb_mode = acb != nullptr && std::strcmp(acb, "dry") == 0 ? AcbMode::Dry : AcbMode::On;
			std::fprintf(stderr, "native shadow: async compute buffers %s\n",
			             m_acb_mode == AcbMode::Dry ? "translated only (dry run)" : "run natively on the gfx ring");
		}
		if (const char* depth = std::getenv("KYTY_NATIVE_GPU_DEPTH"); depth != nullptr) {
			const bool hiz_off  = std::strcmp(depth, "hiz_off") == 0;
			const bool no_htile = std::strcmp(depth, "no_htile") == 0;
			m_translator.SetDepthDebugMode(hiz_off ? Pm4Translator::DepthDebugMode::HizOff
			                                       : (no_htile ? Pm4Translator::DepthDebugMode::NoHtile
			                                                   : Pm4Translator::DepthDebugMode::None));
			std::fprintf(stderr, "native shadow: depth experiment %s\n", depth);
		}
		m_enabled = true;
		m_active.store(true, std::memory_order_release);
		std::fprintf(stderr, "native shadow: enabled (gfx-ring shadow submission, %.1f s timeout)\n",
		             m_timeout_sec);
#else
		std::fprintf(stderr, "native shadow: not supported on this platform; staying off.\n");
#endif
	}

#if KYTY_GEM_BACKING_SUPPORTED
	// Diagnostics for the BC-250 native path, all off by default:
	//   KYTY_NATIVE_GPU_CAPTURE_DIR=<dir>      where the capture files go (needed by the next three)
	//   KYTY_NATIVE_GPU_CAPTURE_AT=<seconds>   start the window this long after the first submission
	//   KYTY_NATIVE_GPU_PROFILE_FLIPS=<n>      trace.bin: GPU timestamps of every buffer for n flips
	//   KYTY_NATIVE_GPU_PROFILE_DRAWS=1        ... and of every draw/dispatch in them
	//   KYTY_NATIVE_GPU_PM4_DUMP_FLIPS=<n>     pm4.bin: guest buffers, IBs and tables for n flips
	//   KYTY_NATIVE_GPU_DRAW_MARKERS=1         progress markers, logged when a submission times out
	void InitCapture() {
		m_markers = EnvOn("KYTY_NATIVE_GPU_DRAW_MARKERS");
		auto count = [](const char* name) {
			const char* v = std::getenv(name);
			return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0u;
		};
		if (const char* dir = std::getenv("KYTY_NATIVE_GPU_CAPTURE_DIR"); dir != nullptr && dir[0] != '\0') {
			m_capture_dir   = dir;
			m_profile_flips = count("KYTY_NATIVE_GPU_PROFILE_FLIPS");
			m_dump_flips    = count("KYTY_NATIVE_GPU_PM4_DUMP_FLIPS");
			m_profile_items = EnvOn("KYTY_NATIVE_GPU_PROFILE_DRAWS");
			const char* at  = std::getenv("KYTY_NATIVE_GPU_CAPTURE_AT");
			m_capture_at    = at != nullptr ? std::strtod(at, nullptr) : 0.0;
		}
		const bool capture = !m_capture_dir.empty() && (m_profile_flips != 0 || m_dump_flips != 0);
		if (!capture && !m_markers) {
			return;
		}
		if (!m_submitter->CreateProfileBuffer(kProfileBytes)) {
			m_markers = false;
			return;
		}
		m_ring.va       = m_submitter->ProfileVa() + kProfileRingOffset;
		m_ring.capacity = static_cast<uint32_t>((kProfileBytes - kProfileRingOffset) / 16);
		m_counter_khz   = m_submitter->CounterKhz();
		if (m_counter_khz == 0) {
			m_counter_khz = static_cast<uint32_t>(kGpuClockHz / 1e3);
		}
		m_capture_t0 = NowSecShadow();
		std::fprintf(stderr,
		             "native shadow: diagnostics: markers %s, capture %s (at %.0f s: profile %u flips%s, PM4 dump %u flips), "
		             "GPU clock %u kHz\n",
		             m_markers ? "on" : "off", capture ? m_capture_dir.c_str() : "off", m_capture_at, m_profile_flips,
		             m_profile_items ? " + draws" : "", m_dump_flips, m_counter_khz);
		m_capture_ready.store(capture, std::memory_order_release);
	}
#else
	void InitCapture() {}
#endif

	void Disable(const char* reason) {
		ReportMarkers(reason);
		m_disabled = true;
		m_active.store(false, std::memory_order_release);
#if KYTY_GEM_BACKING_SUPPORTED
		// The pipeline's completion thread may still wait on submissions of this submitter.
		if (m_submitter != nullptr && !m_submitter->Async()) {
			delete m_submitter;
			m_submitter = nullptr;
		}
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

	// Capture and markers (InitCapture). m_trace and the busy counters are guarded by m_trace_mutex,
	// m_dump and the dump counters by m_mutex.
	std::string           m_capture_dir;
	double                m_capture_at      = 0.0;
	double                m_capture_t0      = 0.0;
	uint32_t              m_profile_flips   = 0;
	uint32_t              m_dump_flips      = 0;
	bool                  m_profile_items   = false;
	bool                  m_markers         = false;
	uint32_t              m_counter_khz     = 0;
	std::atomic<bool>     m_capture_ready {false};
	std::atomic<bool>     m_profiling {false};
	std::atomic<bool>     m_dumping {false};
	std::atomic<uint64_t> m_flip_no {0};
	bool                  m_capture_started = false;
	uint64_t              m_capture_flip0   = 0;
	ProfileRing           m_ring;
	std::vector<BufferProfile> m_pending_buffers; // in the batch being built
	std::vector<BatchProfile>  m_pending_batches; // submitted, not yet taken by the pipeline
	uint64_t              m_batch_ids       = 0;
	std::mutex            m_trace_mutex;
	std::FILE*            m_trace           = nullptr;
	uint64_t              m_busy_ticks[2]   = {}; // dcb, acb
	uint64_t              m_busy_until      = 0;
	uint64_t              m_span_first      = 0;
	uint64_t              m_span_last       = 0;
	std::FILE*            m_dump            = nullptr;
	uint64_t              m_dumped_buffers  = 0;
	uint64_t              m_dumped_bytes    = 0;

	std::mutex    m_mutex;
	bool          m_initialized = false;
	bool          m_enabled     = false;
	bool          m_disabled    = false;
	Pm4Translator m_translator;
	Pm4Translator m_acb_translator; // async compute buffers (KYTY_NATIVE_GPU_ACB)
	AcbMode       m_acb_mode = AcbMode::Off;
	uint64_t      m_acbs           = 0;
	uint64_t      m_acbs_completed = 0;
	uint64_t      m_acbs_refused   = 0;
	std::set<uint32_t> m_unknown_acb_ops;
	uint32_t      m_acb_dummy_writes = 0;
	uint64_t      m_promoted_by[6]   = {};
	uint64_t      m_last_seq         = 0;
	bool          m_pipelined        = false;
	std::vector<uint32_t> m_batch;
	std::atomic<bool> m_active {false};
	unsigned long m_waits_logged     = 0;
	std::set<uint64_t> m_flipped; // display buffers already made resident
	double        m_busy_s      = 0.0;
	double        m_wait_lock_s = 0.0;
	double        m_present_s   = 0.0;
	double        m_gpu_busy_s  = 0.0;
	bool          m_check_order = EnvOn("KYTY_NATIVE_GPU_CHECK_ORDER");
	struct PendingWait {
		Pm4TranslateResult::UnmetWait wait;
		bool                          acb;
		uint64_t                      buffer;
	};
	std::map<uint64_t, PendingWait> m_unmet_by_addr; // label -> the wait no earlier buffer satisfies
	std::map<uint64_t, uint64_t>    m_last_label;    // label -> last value written by a translated buffer
	uint64_t      m_late_writers = 0;
	std::set<uint64_t> m_gds_seen; // distinct (queue, direction, offset, size) of GDS DMA, for the log
	uint64_t      m_gds_dma_dcb[2]  = {}; // DMA_DATA to / from GDS in graphics buffers
	uint64_t      m_gds_dma_acb[2]  = {}; // and in async compute buffers
	uint64_t      m_chain_n         = 0; // pipelined buffers finished since the last flip report
	double        m_chain_queued_s  = 0.0;
	double        m_chain_latency_s = 0.0;
	double        m_chain_then_s    = 0.0;
	double        m_chain_max_s     = 0.0;
	double        m_flip_t0     = 0.0;
	uint32_t      m_flips       = 0;
	std::set<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>> m_find_seen; // KYTY_NATIVE_GPU_FIND hits logged
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

// KYTY_NATIVE_GPU_THREAD=1: guest submissions leave the game thread at once. A worker translates and
// submits them in order without waiting for the GPU; a completion thread waits for each one and only
// then hands the buffer to Kyty's command processor, so the labels, EOP events and flips Kyty
// produces still follow the GPU work they stand for.
thread_local bool t_pipeline_thread = false;

class Pipeline {
public:
	static Pipeline& Instance() {
		static Pipeline p;
		return p;
	}

	void Enqueue(const uint32_t* buffer, uint32_t size_dw, bool acb, std::function<void()> then) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_jobs.push_back(Job {buffer, size_dw, acb, std::move(then), NowSecShadow()});
		m_cv.notify_one();
	}

private:
	struct Job {
		const uint32_t*       buffer;
		uint32_t              size_dw;
		bool                  acb;
		std::function<void()> then;
		double                enqueued; // when the guest submitted it
	};
	struct Done {
		uint64_t              seq;
		std::function<void()> then;
		double                submitted; // when the batch went to the GPU
		double                enqueued;
		double                picked;    // when the worker took it
		bool                  acb    = false;
		uint64_t              number = ~0ull; // the shadow's frame / acb number, ~0 when not translated
		std::vector<BatchProfile> profile;    // the profiled batches of this round (first job only)
	};

	Pipeline() {
		Shadow::Instance().SetPipelined();
		std::thread([this] { Work(); }).detach();
		std::thread([this] { Complete(); }).detach();
	}

	// Takes every queued buffer, translates them into one batch and submits it with a single CS. Their
	// continuations all wait for that last submission: the gfx ring runs submissions in order.
	void Work() {
		for (;;) {
			std::deque<Job> jobs;
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				m_cv.wait(lock, [this] { return !m_jobs.empty(); });
				jobs.swap(m_jobs);
			}
			const double          picked = NowSecShadow();
			std::vector<uint64_t> numbers(jobs.size(), ~0ull);
			for (size_t i = 0; i < jobs.size(); i++) {
				if (jobs[i].buffer != nullptr) { // nullptr: only keeps `then` in order
					Shadow::Instance().Submit(std::span<const uint32_t>(jobs[i].buffer, jobs[i].size_dw), jobs[i].acb,
					                          jobs[i].enqueued, &numbers[i]);
				}
			}
			Shadow::Instance().FlushBatch();
			const uint64_t seq     = Shadow::Instance().TakeLastSeq();
			auto           profile = Shadow::Instance().TakeBatchProfiles();
			const double   now     = NowSecShadow();
			std::lock_guard<std::mutex> lock(m_done_mutex);
			for (size_t i = 0; i < jobs.size(); i++) {
				Done d {seq, std::move(jobs[i].then), now, jobs[i].enqueued, picked, jobs[i].acb, numbers[i], {}};
				if (i == 0) {
					d.profile = std::move(profile);
				}
				m_done.push_back(std::move(d));
			}
			m_done_cv.notify_one();
		}
	}

	void Complete() {
		t_pipeline_thread = true;
		for (;;) {
			Done done;
			{
				std::unique_lock<std::mutex> lock(m_done_mutex);
				m_done_cv.wait(lock, [this] { return !m_done.empty(); });
				done = std::move(m_done.front());
				m_done.pop_front();
			}
			if (done.seq != 0 && done.seq != m_waited_seq) {
				(void)Shadow::Instance().WaitSubmission(done.seq);
				// GPU busy time: from the batch's submission (or the previous batch's end) to its fence.
				const double end = NowSecShadow();
				Shadow::Instance().AddGpuBusy(end - std::max(done.submitted, m_last_end));
				m_last_end   = end;
				m_waited_seq = done.seq;
			}
			if (!done.profile.empty()) {
				Shadow::Instance().ProfileBatches(done.profile, m_last_end);
			}
			const double then_start = NowSecShadow();
			done.then();
			const double then_end = NowSecShadow();
			if (done.number != ~0ull) {
				Shadow::Instance().ProfileThen(done.acb, done.number, then_start, then_end);
			}
			Shadow::Instance().AddChain(done.picked - done.enqueued, then_end - done.enqueued, then_end - then_start);
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::deque<Job>         m_jobs;
	std::mutex              m_done_mutex;
	std::condition_variable m_done_cv;
	std::deque<Done>        m_done;
	uint64_t                m_waited_seq = 0;
	double                  m_last_end   = 0.0;
};

} // namespace

bool ShadowOnPipelineThread() {
	return t_pipeline_thread;
}

bool ShadowThreaded() {
	static const bool on = PrimaryFeature("KYTY_NATIVE_GPU_THREAD");
	return on;
}

void ShadowEnqueue(const uint32_t* buffer, uint32_t size_in_dwords, bool acb, std::function<void()> then) {
	Pipeline::Instance().Enqueue(buffer, size_in_dwords, acb, std::move(then));
}

bool ShadowDmabufPieces(uint64_t vaddr, uint64_t size, std::vector<NativeDmabufPiece>& out) {
	static const bool on = !EnvOn("KYTY_NATIVE_GPU_NO_ZERO_COPY");
	return on && PrimaryRendering() && Shadow::Instance().DmabufPieces(vaddr, size, out);
}

void ShadowFlipStats(double present_seconds) {
	static const bool on = EnvOn("KYTY_NATIVE_GPU_FLIP_STATS");
	if (on && Shadow::Instance().Active()) {
		Shadow::Instance().FlipStats(present_seconds);
	}
}

void ShadowSubmitAcb(const uint32_t* acb, uint32_t size_in_dwords) {
	if (acb == nullptr || size_in_dwords == 0) {
		return;
	}
	Shadow::Instance().Submit(std::span<const uint32_t>(acb, size_in_dwords), true);
}

bool NativeAsyncCompute() {
	return PrimaryRendering() && Shadow::Instance().NativeAcb();
}

namespace {

std::atomic<bool>& ShowNative() {
	static std::atomic<bool> show {EnvOn("KYTY_NATIVE_GPU_SHOW")};
	return show;
}

} // namespace

bool NativeLodStats() {
	static const bool on = EnvOn("KYTY_NATIVE_GPU_LOD_STATS");
	return on;
}

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
		if (!download_region) { // primary mode: nothing to compare with, keep every native target
			std::snprintf(path, sizeof(path), "%s/flip_%llu.rt_%llx.native.raw", dir, flip,
			              static_cast<unsigned long long>(t.addr));
			WriteFile(path, native.data(), got);
		} else if (differing != 0) {
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
	Shadow::Instance().NoteFlip(n);
	if (address != 0 && size != 0 && Shadow::Instance().Active()) {
		Shadow::Instance().KeepResident(address, size);
	}
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
	} else if (PrimaryRendering() && EnvOn("KYTY_NATIVE_GPU_DUMP_RTS")) {
		DumpRenderTargets(dir, flip, nullptr);
	}
	// KYTY_NATIVE_GPU_DUMP_RANGES=0xADDR:0xSIZE[,...]: guest ranges saved next to the flip, e.g. a
	// game's intermediate targets when the presented surface stays empty.
	if (const char* ranges = std::getenv("KYTY_NATIVE_GPU_DUMP_RANGES"); ranges != nullptr) {
		for (const char* p = ranges; *p != '\0';) {
			char*          end   = nullptr;
			const uint64_t raddr = std::strtoull(p, &end, 0);
			if (end == p || *end != ':') {
				break;
			}
			const uint64_t rsize = std::strtoull(end + 1, &end, 0);
			std::vector<uint8_t> bytes(rsize);
			const uint64_t       got = Shadow::Instance().ReadGuest(raddr, bytes.data(), rsize);
			std::snprintf(path, sizeof(path), "%s/flip_%llu.range_%llx.raw", dir, flip, static_cast<unsigned long long>(raddr));
			WriteFile(path, bytes.data(), got);
			p = (*end == ',') ? end + 1 : end;
		}
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
