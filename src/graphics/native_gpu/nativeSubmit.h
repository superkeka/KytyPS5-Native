#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESUBMIT_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESUBMIT_H_

#include "graphics/native_gpu/gemBacking.h" // KYTY_GEM_BACKING_SUPPORTED + amdgpu uAPI headers
#include "graphics/native_gpu/nativeKfd.h"
#include "graphics/native_gpu/nativePm4.h"

// Submits a translated IB to the amdgpu gfx ring for the BC-250 native backend, on the same DRM
// file whose GPU VM holds the guest memory bindings (GemBacking::DrmFd()). Only real Linux with the
// amdgpu uAPI has it; elsewhere this header is empty and the shadow hook compiles to a no-op.

#if KYTY_GEM_BACKING_SUPPORTED

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics::NativeGpu {

// Which amdgpu ring a submission runs on. The default is the gfx ring; compute rings are
// AMDGPU_HW_IP_COMPUTE, ip_instance 0, ring index a bit set in NativeSubmitter::ComputeRingMask().
// Outside the class: a default argument of the class's own member functions cannot use a nested
// type's default member initializers.
struct NativeRingTarget {
	uint32_t ip_type = AMDGPU_HW_IP_GFX;
	uint32_t ring    = 0;
};

class NativeSubmitter {
public:
	// The IB lives at a GPU VA above every guest range (guest memory tops out well below this and
	// the extended-memory aperture is at 0x80000000000; the device's low VA range reaches
	// 0x800000000000, so this sits clear of both).
	static constexpr uint64_t kIbVa      = 0x400000000000ull;
	// 128 slots of ~512 KiB in async mode: a KFD queue waiting on a label keeps the slots of its IBs, and
	// a title with many compute queues (My First Gran Turismo has a dozen) must not leave the next gfx
	// IB, the one that writes that label, without a slot. Longer IBs take several slots.
	static constexpr uint64_t kIbSize    = 64ull * 1024 * 1024;
	static constexpr uint32_t kSlots     = 128;
	// The last bytes of the IB buffer are a scratch area the IB itself may write into (pipeline
	// statistics samples); IBs must stay below it.
	static constexpr uint64_t kScratchSize = 4096;
	static constexpr uint64_t kSlotSize    = ((kIbSize - kScratchSize) / kSlots) & ~0xfffull;
	// Extra DRM_IOCTL_AMDGPU_CS attempts spent waiting out a userptr revalidation (-EFAULT).
	static constexpr int      kCsRetries = 64;

	using RingTarget = NativeRingTarget;

	struct Result {
		bool   submitted = false; // the CS ioctl was accepted
		bool   completed = false; // the fence signalled before the timeout
		bool   timed_out = false; // the wait timed out
		bool   reset     = false; // the context reported a reset/guilty state afterwards
		int    err       = 0;     // negative errno from a failed ioctl
		double seconds   = 0.0;   // submit -> fence wall time
		double cs_seconds = 0.0;  // time spent inside the CS ioctl(s), retries included
		int    retries   = 0;     // extra CS attempts spent on userptr revalidation (EFAULT)
		uint64_t seq       = 0;     // async mode: the submission to wait for
	};

	// Allocates a context and the IB buffer on `drm_fd`. Returns nullptr on failure. With `gem` and
	// KYTY_NATIVE_GPU_KFD_QUEUES=1 the IB buffer is a KFD buffer, so KFD queues can run IBs from it too.
	static NativeSubmitter* Create(int drm_fd, GemBacking* gem = nullptr) {
		union drm_amdgpu_ctx ctx;
		std::memset(&ctx, 0, sizeof(ctx));
		ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		if (Ioctl(drm_fd, DRM_IOCTL_AMDGPU_CTX, &ctx) != 0) {
			std::fprintf(stderr, "NativeSubmitter: context alloc failed: %s\n", std::strerror(errno));
			return nullptr;
		}

		auto* self     = new NativeSubmitter();
		self->m_drm_fd = drm_fd;
		self->m_ctx    = ctx.out.alloc.ctx_id;
		self->m_gem    = gem;

		// KYTY_NATIVE_GPU_KFD_IB=0 keeps the render-node IB buffer (and with it no KFD queue).
		const char* kfd_ib  = std::getenv("KYTY_NATIVE_GPU_KFD_IB");
		const bool  use_kfd = gem != nullptr && KfdQueuesWanted() && gem->KfdFd() >= 0 && (kfd_ib == nullptr || kfd_ib[0] != '0');
		if (!(use_kfd ? self->CreateKfdIb() : self->CreateIb())) {
			delete self;
			return nullptr;
		}
		self->QueryComputeRings();
		// Opt-in: with the whole GDS, GWS and OA partitions in every submission (so amdgpu reprograms
		// them for each job) the BC-250 hung within ~30 s in Dead Cells and ASTRO BOT alike, even when
		// no packet or shader used GDS. KYTY_NATIVE_GPU_GDS=1 asks for the whole partitions,
		// "gds=<bytes>,gws=<n>,oa=<n>" for chosen sizes (0 or absent: none of that resource).
		// Native primary mode defaults to the GDS the PS5 kernel gives a title's VMID, 48 KiB from base
		// 0, with no GWS or OA (those partitions were what hung).
		const char* opt     = std::getenv("KYTY_NATIVE_GPU_GDS");
		const char* primary = std::getenv("KYTY_NATIVE_GPU_PRIMARY");
		if (opt == nullptr && primary != nullptr && primary[0] != '\0' && primary[0] != '0') {
			opt = "gds=49152,gws=0,oa=0";
		}
		if (opt != nullptr && opt[0] != '\0' && opt[0] != '0') {
			self->CreateGds(opt);
		}
		return self;
	}

	~NativeSubmitter() {
		for (KfdQueue*& q: m_kfd_queue) {
			delete q;
			q = nullptr;
		}
		for (uint32_t h: m_gds_handles) {
			struct drm_gem_close c;
			std::memset(&c, 0, sizeof(c));
			c.handle = h;
			Ioctl(m_drm_fd, DRM_IOCTL_GEM_CLOSE, &c);
		}
		if (m_ib_va != 0) {
			VaOp(AMDGPU_VA_OP_UNMAP, m_ib_handle, m_ib_va);
		}
		if (m_ib_cpu != nullptr) {
			::munmap(m_ib_cpu, kIbSize);
		}
		if (m_ib_kfd_handle != 0) {
			m_gem->FreeKfdBuffer(m_ib_handle, m_ib_dmabuf, m_ib_kfd_handle);
		} else if (m_ib_handle != 0) {
			struct drm_gem_close c;
			std::memset(&c, 0, sizeof(c));
			c.handle = m_ib_handle;
			Ioctl(m_drm_fd, DRM_IOCTL_GEM_CLOSE, &c);
		}
		if (m_ctx != 0) {
			union drm_amdgpu_ctx ctx;
			std::memset(&ctx, 0, sizeof(ctx));
			ctx.in.op     = AMDGPU_CTX_OP_FREE_CTX;
			ctx.in.ctx_id = m_ctx;
			Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
		}
	}

	[[nodiscard]] uint64_t ScratchVa() const { return m_ib_va + kIbSize - kScratchSize; }

	// True when every submission owns the GFX partition of GDS (and its GWS and OA), so guest
	// packets and shaders that use GDS can run natively.
	[[nodiscard]] bool HasGds() const { return m_gds_size != 0; }
	// AMDGPU_INFO_HW_IP_INFO available_rings for compute, ip_instance 0. 0 when the kernel has none.
	[[nodiscard]] uint32_t ComputeRingMask() const { return m_compute_rings; }
	[[nodiscard]] uint32_t GdsSize() const { return m_gds_size; }
	// How many KFD queues EnableKfdQueues can give (ring indexes 0 .. this - 1).
	static constexpr uint32_t MaxKfdQueues() { return kMaxKfdQueues; }
	// How many KFD queues EnableKfdQueues created (indexes 0 .. this - 1).
	[[nodiscard]] uint32_t KfdQueueCount() const { return m_kfd_count; }
	// KFD's queue priority (0-15) for ring index `ring`'s queue, set before its first submission.
	void SetKfdQueuePriority(uint32_t ring, uint32_t priority) {
		if (ring < kMaxKfdQueues) {
			m_kfd_priority[ring] = std::min(priority, 15u);
		}
	}
	// Called with the ring index after each KFD queue is created (it is mapped to a hardware queue by
	// then), on the submitting thread.
	void SetKfdQueueHook(std::function<void(uint32_t)> hook) { m_kfd_queue_hook = std::move(hook); }

	// KYTY_NATIVE_GPU_KFD_QUEUES: compute submissions to the ring indexes in `rings` (bits below
	// kMaxKfdQueues) go to KFD user-mode queues, one per index, rung by doorbell instead of a CS
	// ioctl: a hardware queue each, as the PS5 gives each of a title's compute queues. A queue is
	// created on its index's first submission, with the priority SetKfdQueuePriority gave it. Needs
	// the KFD IB buffer; false without it.
	bool EnableKfdQueues(uint32_t rings) {
		if (m_ib_kfd_handle == 0) {
			return false;
		}
		m_kfd_rings = rings & ((1u << kMaxKfdQueues) - 1u);
		// Every queue is created now, before any work. Creating one later makes the firmware scheduler
		// preempt the running queues to rebuild its runlist, and one busy with a title's work did not
		// preempt on the BC-250 (My First Gran Turismo's eleventh queue: "Queue preemption failed", and
		// every queue stopped). The PS5 kernel too sets a title's queues up once.
		// KYTY_NATIVE_GPU_KFD_QUEUE_COUNT (default 16) of them; more guest queues than that share.
		uint32_t count = 16;
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_KFD_QUEUE_COUNT"); v != nullptr && v[0] != '\0') {
			count = static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
		}
		count       = std::clamp<uint32_t>(count, 1u, kMaxKfdQueues);
		uint32_t made = 0;
		for (uint32_t ring = 0; ring < count && (m_kfd_rings & (1u << ring)) != 0; ring++) {
			EnsureKfdQueue({AMDGPU_HW_IP_COMPUTE, ring});
			if (m_kfd_queue[ring] == nullptr) {
				break;
			}
			made++;
		}
		m_kfd_rings = made == 0 ? 0u : (made >= 32 ? ~0u : (1u << made) - 1u);
		m_kfd_count = made;
		if (made == 0) {
			return false;
		}
		// KYTY_NATIVE_GPU_KFD_TRACE=1: every 0.5 s, each KFD queue's pointers and each CS slot's latest
		// submission, so a hang that takes the board down leaves its last picture in the log.
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_KFD_TRACE"); v != nullptr && v[0] == '1') {
			std::thread([this] {
				for (uint32_t tick = 0;; tick++) {
					struct timespec nap = {0, 500000000};
					nanosleep(&nap, nullptr);
					std::string line = "kfd trace " + std::to_string(tick) + ":" + QueueState({});
					for (uint32_t i = 0; i < kSlots; i++) {
						const SlotUse s = m_slot[i];
						if (s.seq != 0 && Queue(s.target) == nullptr) {
							line += "\n  cs slot " + std::to_string(i) + " ip " + std::to_string(s.target.ip_type) + " ring " +
							        std::to_string(s.target.ring) + " seq " + std::to_string(s.seq) +
							        (WaitSeq(s.seq, 0.0, s.target) ? " done" : " pending");
						}
					}
					std::fprintf(stderr, "%s\n", line.c_str());
				}
			}).detach();
		}
		return true;
	}
	[[nodiscard]] volatile uint8_t* ScratchCpu() const {
		return static_cast<volatile uint8_t*>(m_ib_cpu) + kIbSize - kScratchSize;
	}

	NativeSubmitter(const NativeSubmitter&)            = delete;
	NativeSubmitter& operator=(const NativeSubmitter&) = delete;

	// Copies `ib` into the IB buffer, submits it to the gfx ring with `bo_handles` (plus the IB)
	// resident, waits up to `timeout_sec` and queries the context state.
	// Async mode (SetAsync): the IB buffer is a ring of kSlots slots and Submit returns right after
	// the CS ioctl; it only waits for the submission that last used the slot it is about to reuse.
	void SetAsync(bool async) { m_async = async; }
	[[nodiscard]] bool Async() const { return m_async; }

	// Waits until every async submission so far has completed. Returns false on a timeout.
	bool WaitIdle(double timeout_sec) {
		for (SlotUse& slot: m_slot) {
			if (slot.seq != 0 && !WaitSeq(slot.seq, timeout_sec, slot.target)) {
				return false;
			}
			slot.seq = 0;
		}
		return true;
	}

	Result Submit(std::span<const uint32_t> ib, const std::vector<uint32_t>& bo_handles,
	              double timeout_sec, RingTarget target = {}) {
		Result res;
		// An IB longer than a slot (a level load's DCB carries megabytes of inline data) is cut at
		// packet boundaries into several IBs of one CS, each in a slot of its own and padded to 8
		// dwords: the kernel runs a job's IBs back to back with the state carried over, which is what
		// the CP does with the one buffer (an INDIRECT_BUFFER packet holds at most 2^20 dwords, so a
		// long IB could not be one anyway).
		EnsureKfdQueue(target);
		const uint64_t                         room = m_async ? kSlotSize : kIbSize - kScratchSize;
		std::vector<std::pair<size_t, size_t>> parts;
		if (ib.size() * 4 <= room) {
			parts.emplace_back(0, ib.size());
		} else if (m_async) {
			parts = SplitIbAtPackets(ib, room / 4 - 8);
		}
		if (parts.empty() || parts.size() > kSlots || (parts.size() > 1 && Queue(target) != nullptr)) {
			res.err = -E2BIG;
			return res;
		}
		std::vector<uint32_t> slots;
		std::bitset<kSlots>   taken;
		for (const auto& [at, len]: parts) {
			const uint32_t slot = m_async ? PickSlot(target, taken) : 0;
			taken.set(slot);
			if (m_async && m_slot[slot].seq != 0) {
				if (!WaitSeq(m_slot[slot].seq, timeout_sec, m_slot[slot].target)) {
					res.timed_out = true;
					return res;
				}
				m_slot[slot].seq = 0;
			}
			auto* dst = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(m_ib_cpu) + slot * kSlotSize);
			std::memcpy(dst, ib.data() + at, len * 4);
			size_t padded = len;
			while (parts.size() > 1 && (padded & 7u) != 0) {
				dst[padded++] = 0xffff1000u;
			}
			slots.push_back(slot);
			parts[slots.size() - 1].second = padded;
		}
		const uint32_t slot = slots[0];
		__sync_synchronize();

		if (std::getenv("KYTY_NATIVE_GPU_KFD_TRACE") != nullptr && m_trace_submits < 24) {
			m_trace_submits++;
			std::fprintf(stderr, "submit: ip %u ring %u, %zu dwords, slot %u, %s\n", target.ip_type, target.ring, ib.size(), slot,
			             Queue(target) != nullptr && m_async ? "KFD queue" : "CS");
		}
		if (KfdQueue* q = Queue(target); q != nullptr && m_async) {
			// The VM changed since the last KFD submission: its queues could still translate the old
			// mappings (GemBacking::FlushKfdTlb).
			// Only with KYTY_NATIVE_GPU_KFD_TLB_FLUSH=1: the flush rebuilds the firmware scheduler's
			// runlist, which preempts every KFD queue, and a busy one did not preempt on the BC-250
			// ("Queue preemption failed", every queue stopped). The PS5 kernel never touches a title's
			// queues while it runs either; without it My First Gran Turismo runs at 45 FPS.
			static const bool flush = [] {
				const char* v = std::getenv("KYTY_NATIVE_GPU_KFD_TLB_FLUSH");
				return v != nullptr && v[0] == '1';
			}();
			if (const uint64_t changes = m_gem->VaChanges(); flush && changes != m_kfd_va_changes) {
				const double tf = NowSec();
				const bool   ok = m_gem->FlushKfdTlb();
				m_kfd_va_changes = changes;
				if (m_kfd_tlb_flushes++ < 8 || !ok) {
					std::fprintf(stderr, "NativeSubmitter: KFD TLB flush after GEM_VA changes %s, %.1f ms\n", ok ? "done" : "FAILED",
					             (NowSec() - tf) * 1e3);
				}
			}
			const double t0 = NowSec();
			uint64_t     seq = 0;
			if (!q->Submit(m_ib_kfd_va + slot * kSlotSize, static_cast<uint32_t>(ib.size()), timeout_sec, &seq)) {
				res.timed_out = true;
				return res;
			}
			m_slot[slot].seq    = seq;
			m_slot[slot].target = target;
			m_slot[slot].order  = ++m_slot_order;
			res.submitted       = true;
			res.completed       = true;
			res.seq             = seq;
			res.cs_seconds      = NowSec() - t0;
			res.seconds         = res.cs_seconds;
			return res;
		}

		std::vector<drm_amdgpu_bo_list_entry> entries;
		entries.reserve(bo_handles.size() + 1);
		drm_amdgpu_bo_list_entry e;
		std::memset(&e, 0, sizeof(e));
		e.bo_handle = m_ib_handle;
		entries.push_back(e);
		// A GDS, GWS or OA buffer in the list is what gives the job its GDS partition. Gfx and compute
		// jobs list the same buffers, so they share one partition at base 0 (the PS5 VMID partition).
		for (uint32_t h: m_gds_handles) {
			e.bo_handle = h;
			entries.push_back(e);
		}
		for (uint32_t h: bo_handles) {
			e.bo_handle = h;
			entries.push_back(e);
		}

		struct drm_amdgpu_bo_list_in list;
		std::memset(&list, 0, sizeof(list));
		list.operation    = ~0u;
		list.list_handle  = ~0u;
		list.bo_number    = static_cast<uint32_t>(entries.size());
		list.bo_info_size = sizeof(entries[0]);
		list.bo_info_ptr  = reinterpret_cast<uintptr_t>(entries.data());

		std::vector<drm_amdgpu_cs_chunk_ib> ib_infos(parts.size());
		std::vector<drm_amdgpu_cs_chunk>    chunks(parts.size() + 1);
		std::vector<uint64_t>               chunk_ptrs(parts.size() + 1);
		for (size_t k = 0; k < parts.size(); k++) {
			auto& info = ib_infos[k];
			std::memset(&info, 0, sizeof(info));
			info.va_start    = m_ib_va + slots[k] * kSlotSize;
			info.ib_bytes    = static_cast<uint32_t>(parts[k].second * 4);
			info.ip_type     = target.ip_type;
			info.ip_instance = 0;
			info.ring        = target.ring;
			std::memset(&chunks[k], 0, sizeof(chunks[k]));
			chunks[k].chunk_id   = AMDGPU_CHUNK_ID_IB;
			chunks[k].length_dw  = sizeof(info) / 4;
			chunks[k].chunk_data = reinterpret_cast<uintptr_t>(&info);
		}
		auto& bo_chunk = chunks.back();
		std::memset(&bo_chunk, 0, sizeof(bo_chunk));
		bo_chunk.chunk_id   = AMDGPU_CHUNK_ID_BO_HANDLES;
		bo_chunk.length_dw  = sizeof(list) / 4;
		bo_chunk.chunk_data = reinterpret_cast<uintptr_t>(&list);
		for (size_t k = 0; k < chunks.size(); k++) {
			chunk_ptrs[k] = reinterpret_cast<uintptr_t>(&chunks[k]);
		}

		union drm_amdgpu_cs cs;
		std::memset(&cs, 0, sizeof(cs));
		cs.in.ctx_id     = m_ctx;
		cs.in.num_chunks = static_cast<uint32_t>(chunks.size());
		cs.in.chunks     = reinterpret_cast<uintptr_t>(chunk_ptrs.data());

		const double t0 = NowSec();
		// userptr BOs (the external ranges outside direct memory) are revalidated against
		// the CPU page tables at submit; a guest write to one between validation passes makes the CS
		// return -EFAULT, and the caller is expected to retry, exactly as Mesa does. The pages settle
		// within a few attempts; cap it so a persistently bad range still disables the path cleanly.
		int r = 0;
		for (int attempt = 0; attempt <= kCsRetries; attempt++) {
			r = Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_CS, &cs);
			if (r != -EFAULT) {
				break;
			}
			res.retries++;
			struct timespec nap = {0, 500000}; // 0.5 ms
			nanosleep(&nap, nullptr);
		}
		res.cs_seconds = NowSec() - t0;
		if (r != 0) {
			res.err = r;
			return res;
		}
		res.submitted      = true;
		const uint64_t seq = cs.out.handle;
		if (m_async) {
			for (const uint32_t used: slots) {
				m_slot[used].seq    = seq;
				m_slot[used].target = target;
				m_slot[used].order  = ++m_slot_order;
			}
			res.completed    = true;
			res.seq          = seq;
			res.seconds      = NowSec() - t0;
			return res;
		}

		union drm_amdgpu_wait_cs w;
		std::memset(&w, 0, sizeof(w));
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		w.in.handle  = seq;
		w.in.timeout = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
		               static_cast<uint64_t>(ts.tv_nsec) + static_cast<uint64_t>(timeout_sec * 1e9);
		w.in.ip_type = target.ip_type;
		w.in.ring    = target.ring;
		w.in.ctx_id  = m_ctx;
		r            = Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_WAIT_CS, &w);
		res.seconds  = NowSec() - t0;
		if (r != 0) {
			res.err = r;
			return res;
		}
		if (w.out.status != 0) {
			res.timed_out = true;
		} else {
			res.completed = true;
		}
		res.reset = ContextReset();
		return res;
	}

	[[nodiscard]] int DrmFd() const { return m_drm_fd; }

	// COMPUTE_STATIC_THREAD_MGMT_SE0..3 masks of the CUs amdgpu found at init (AMDGPU_INFO_DEV_INFO
	// cu_bitmap: SH0 in the low 16 bits, SH1 in the high ones); zeros when the query fails.
	[[nodiscard]] std::array<uint32_t, 4> InitCuMasks() const {
		std::array<uint32_t, 4>     masks {};
		struct drm_amdgpu_info_device dev;
		std::memset(&dev, 0, sizeof(dev));
		struct drm_amdgpu_info request;
		std::memset(&request, 0, sizeof(request));
		request.return_pointer = reinterpret_cast<uintptr_t>(&dev);
		request.return_size    = sizeof(dev);
		request.query          = AMDGPU_INFO_DEV_INFO;
		if (Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_INFO, &request) != 0) {
			return masks;
		}
		for (uint32_t se = 0; se < 4; se++) {
			masks[se] = (dev.cu_bitmap[se][0] & 0xffffu) | ((dev.cu_bitmap[se][1] & 0xffffu) << 16u);
		}
		return masks;
	}

	// The KFD queues' state for a timeout report (every queue: a gfx wait may be stuck on one); empty
	// without KFD queues.
	[[nodiscard]] std::string QueueState(RingTarget /*target*/) const {
		std::string out;
		for (uint32_t ring = 0; ring < kMaxKfdQueues; ring++) {
			if (m_kfd_queue[ring] != nullptr) {
				char text[192];
				m_kfd_queue[ring]->Describe(text, sizeof(text));
				out += "\n  ring " + std::to_string(ring) + " " + text;
			}
		}
		return out;
	}

	bool WaitSeq(uint64_t seq, double timeout_sec, RingTarget target = {}) {
		if (const KfdQueue* q = Queue(target); q != nullptr) {
			return q->Wait(seq, timeout_sec);
		}
		union drm_amdgpu_wait_cs w;
		std::memset(&w, 0, sizeof(w));
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		w.in.handle  = seq;
		w.in.timeout = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec) +
		               static_cast<uint64_t>(timeout_sec * 1e9);
		w.in.ip_type = target.ip_type;
		w.in.ring    = target.ring;
		w.in.ctx_id  = m_ctx;
		return Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_WAIT_CS, &w) == 0 && w.out.status == 0;
	}

private:
	// A KFD queue per guest compute queue, up to the tracker's 32 slots less the gfx one (PS5 titles
	// address 56; more than this share).
	static constexpr uint32_t kMaxKfdQueues   = 31;
	static constexpr uint64_t kKfdQueueVa     = 0x6e0000000000ull; // above GemBacking's KFD allocations
	static constexpr uint64_t kKfdQueueStride = 0x10000000ull;

	NativeSubmitter() = default;

	// Creates `target`'s KFD queue on its first submission (EnableKfdQueues). A queue that cannot be
	// created leaves the index to the CS ring of that number.
	void EnsureKfdQueue(RingTarget target) {
		if (target.ip_type != AMDGPU_HW_IP_COMPUTE || target.ring >= kMaxKfdQueues || (m_kfd_rings & (1u << target.ring)) == 0 ||
		    m_kfd_queue[target.ring] != nullptr) {
			return;
		}
		m_kfd_queue[target.ring] = KfdQueue::Create(m_gem->KfdFd(), m_gem->KfdGpuId(), m_drm_fd,
		                                            kKfdQueueVa + target.ring * kKfdQueueStride, m_kfd_priority[target.ring]);
		std::fprintf(stderr, "NativeSubmitter: KFD queue for compute index %u (priority %u)%s\n", target.ring,
		             m_kfd_priority[target.ring], m_kfd_queue[target.ring] != nullptr ? "" : " failed");
		if (m_kfd_queue[target.ring] == nullptr) {
			m_kfd_rings &= ~(1u << target.ring);
		} else if (m_kfd_queue_hook) {
			m_kfd_queue_hook(target.ring);
		}
	}

	[[nodiscard]] KfdQueue* Queue(RingTarget target) const {
		return target.ip_type == AMDGPU_HW_IP_COMPUTE && target.ring < kMaxKfdQueues ? m_kfd_queue[target.ring] : nullptr;
	}

	// The IB buffer as a KFD buffer (uncached on the GPU like CreateIb's), imported into m_drm_fd and
	// mapped at kIbVa as well: CS jobs and KFD queues share the slots and the scratch area.
	bool CreateKfdIb() {
		if (!m_gem->CreateKfdBuffer(kIbSize, true, &m_ib_handle, &m_ib_dmabuf, &m_ib_kfd_va, &m_ib_kfd_handle)) {
			std::fprintf(stderr, "NativeSubmitter: KFD IB buffer failed\n");
			return false;
		}
		void* p = ::mmap(nullptr, kIbSize, PROT_READ | PROT_WRITE, MAP_SHARED, m_ib_dmabuf, 0);
		if (p == MAP_FAILED) {
			std::fprintf(stderr, "NativeSubmitter: KFD IB mmap failed: %s\n", std::strerror(errno));
			return false;
		}
		m_ib_cpu = p;
		if (VaOp(AMDGPU_VA_OP_MAP, m_ib_handle, kIbVa) != 0) {
			std::fprintf(stderr, "NativeSubmitter: KFD IB GEM_VA at 0x%llx failed: %s\n", static_cast<unsigned long long>(kIbVa),
			             std::strerror(errno));
			return false;
		}
		m_ib_va = kIbVa;
		std::fprintf(stderr, "NativeSubmitter: IB buffer is a KFD buffer (KFD VA 0x%llx)\n", static_cast<unsigned long long>(m_ib_kfd_va));
		return true;
	}

	static int Ioctl(int fd, unsigned long request, void* arg) {
		int r = -1;
		do {
			r = ::ioctl(fd, request, arg);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		return r == -1 ? -errno : 0;
	}

	static double NowSec() {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
	}

	int VaOp(uint32_t op, uint32_t handle, uint64_t va) const {
		struct drm_amdgpu_gem_va v;
		std::memset(&v, 0, sizeof(v));
		v.handle       = handle;
		v.operation    = op;
		v.flags        = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
		v.va_address   = va;
		v.offset_in_bo = 0;
		v.map_size     = kIbSize;
		return Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_GEM_VA, &v);
	}

	void QueryComputeRings() {
		struct drm_amdgpu_info_hw_ip ip;
		std::memset(&ip, 0, sizeof(ip));
		struct drm_amdgpu_info request;
		std::memset(&request, 0, sizeof(request));
		request.return_pointer          = reinterpret_cast<uintptr_t>(&ip);
		request.return_size             = sizeof(ip);
		request.query                   = AMDGPU_INFO_HW_IP_INFO;
		request.query_hw_ip.type        = AMDGPU_HW_IP_COMPUTE;
		request.query_hw_ip.ip_instance = 0;
		if (Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_INFO, &request) != 0) {
			std::fprintf(stderr, "NativeSubmitter: AMDGPU_INFO_HW_IP_INFO(compute) failed: %s\n", std::strerror(errno));
			return;
		}
		m_compute_rings = ip.available_rings;
		std::fprintf(stderr, "NativeSubmitter: compute rings mask 0x%x\n", m_compute_rings);
	}

	bool CreateIb() {
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size      = kIbSize;
		c.in.alignment    = 64 * 1024;
		c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
		// Write-combined, like Mesa's IBs, and uncached on the GPU: the slots are rewritten in place, and
		// the CP fetching a reused slot through GL2 can execute stale lines of the previous IB there
		// (mem_probe on the BC-250: the third job from one IB address ran the second one's commands).
		c.in.domain_flags = AMDGPU_GEM_CREATE_CPU_GTT_USWC | AMDGPU_GEM_CREATE_UNCACHED;
		if (Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c) != 0) {
			std::fprintf(stderr, "NativeSubmitter: IB GEM_CREATE failed: %s\n", std::strerror(errno));
			return false;
		}
		m_ib_handle = c.out.handle;

		union drm_amdgpu_gem_mmap m;
		std::memset(&m, 0, sizeof(m));
		m.in.handle = m_ib_handle;
		if (Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &m) != 0) {
			std::fprintf(stderr, "NativeSubmitter: IB GEM_MMAP failed: %s\n", std::strerror(errno));
			return false;
		}
		void* p = ::mmap(nullptr, kIbSize, PROT_READ | PROT_WRITE, MAP_SHARED, m_drm_fd,
		                 static_cast<off_t>(m.out.addr_ptr));
		if (p == MAP_FAILED) {
			std::fprintf(stderr, "NativeSubmitter: IB mmap failed: %s\n", std::strerror(errno));
			return false;
		}
		m_ib_cpu = p;

		if (VaOp(AMDGPU_VA_OP_MAP, m_ib_handle, kIbVa) != 0) {
			std::fprintf(stderr, "NativeSubmitter: IB GEM_VA at 0x%llx failed: %s\n",
			             static_cast<unsigned long long>(kIbVa), std::strerror(errno));
			return false;
		}
		m_ib_va = kIbVa;
		return true;
	}

	// Allocates the whole GFX partition of GDS and its GWS and OA resources once. PS5 titles address
	// GDS directly (counters, ordered append), so the native jobs need the same partition from offset
	// 0. Optional: without it, packets that use GDS stay untranslated.
	void CreateGds(const char* spec) {
		struct drm_amdgpu_info_gds gds;
		std::memset(&gds, 0, sizeof(gds));
		struct drm_amdgpu_info request;
		std::memset(&request, 0, sizeof(request));
		request.return_pointer = reinterpret_cast<uintptr_t>(&gds);
		request.return_size    = sizeof(gds);
		request.query          = AMDGPU_INFO_GDS_CONFIG;
		const bool queried = Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_INFO, &request) == 0;
		// Newer kernels report GDS only through the compute-partition and total fields (the GFX
		// partition fields read 0), so take the largest of each.
		uint32_t gds_size = std::max({gds.gds_gfx_partition_size, gds.compute_partition_size, gds.gds_total_size});
		uint32_t gws_size = std::max(gds.gws_per_gfx_partition, gds.gws_per_compute_partition);
		uint32_t oa_size  = std::max(gds.oa_per_gfx_partition, gds.oa_per_compute_partition);
		if (std::strcmp(spec, "1") != 0) {
			auto field = [&](const char* key, uint32_t limit) -> uint32_t {
				const char* at = std::strstr(spec, key);
				return at == nullptr ? 0u : std::min(limit, static_cast<uint32_t>(std::strtoul(at + std::strlen(key), nullptr, 0)));
			};
			gds_size = field("gds=", gds_size);
			gws_size = field("gws=", gws_size);
			oa_size  = field("oa=", oa_size);
		}
		if (!queried || gds_size == 0) {
			std::fprintf(stderr, "NativeSubmitter: no GDS config, GDS packets stay untranslated\n");
			return;
		}
		const struct {
			uint32_t domain;
			uint64_t size;
		} parts[] = {{AMDGPU_GEM_DOMAIN_GDS, gds_size}, {AMDGPU_GEM_DOMAIN_GWS, gws_size}, {AMDGPU_GEM_DOMAIN_OA, oa_size}};
		for (const auto& part: parts) {
			if (part.size == 0) {
				continue;
			}
			union drm_amdgpu_gem_create c;
			std::memset(&c, 0, sizeof(c));
			c.in.bo_size   = part.size;
			c.in.alignment = 1;
			c.in.domains   = part.domain;
			if (Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c) != 0) {
				std::fprintf(stderr, "NativeSubmitter: GEM_CREATE in domain 0x%x (%llu) failed: %s\n", part.domain,
				             static_cast<unsigned long long>(part.size), std::strerror(errno));
				continue;
			}
			m_gds_handles.push_back(c.out.handle);
			if (part.domain == AMDGPU_GEM_DOMAIN_GDS) {
				m_gds_size = static_cast<uint32_t>(part.size);
			}
		}
		std::fprintf(stderr, "NativeSubmitter: GDS %u bytes, GWS %u, OA %u allocated for every job\n", m_gds_size,
		             gws_size, oa_size);
	}

	bool ContextReset() const {
		union drm_amdgpu_ctx ctx;
		std::memset(&ctx, 0, sizeof(ctx));
		ctx.in.op     = AMDGPU_CTX_OP_QUERY_STATE2;
		ctx.in.ctx_id = m_ctx;
		if (Ioctl(m_drm_fd, DRM_IOCTL_AMDGPU_CTX, &ctx) != 0) {
			return false;
		}
		return ctx.out.state.flags != 0;
	}

	int      m_drm_fd    = -1;
	uint32_t m_ctx       = 0;
	uint32_t m_ib_handle = 0;
	void*    m_ib_cpu    = nullptr;
	uint64_t m_ib_va     = 0;
	GemBacking* m_gem           = nullptr;
	int         m_ib_dmabuf     = -1; // CreateKfdIb
	uint64_t    m_ib_kfd_va     = 0;
	uint64_t    m_ib_kfd_handle = 0;
	KfdQueue*   m_kfd_queue[kMaxKfdQueues] {};
	uint32_t    m_kfd_rings = 0;                     // ring indexes whose submissions go to KFD queues
	uint32_t    m_kfd_count = 0;                     // KFD queues created (EnableKfdQueues)
	uint64_t    m_kfd_va_changes = 0;                // GemBacking::VaChanges at the last KFD TLB flush
	uint32_t    m_kfd_tlb_flushes = 0;
	std::array<uint32_t, kMaxKfdQueues> m_kfd_priority = [] {
		std::array<uint32_t, kMaxKfdQueues> a {};
		a.fill(7);
		return a;
	}();
	std::function<void(uint32_t)> m_kfd_queue_hook; // SetKfdQueueHook
	uint32_t    m_trace_submits = 0; // KYTY_NATIVE_GPU_KFD_TRACE
	struct SlotUse {
		uint64_t   seq   = 0;
		uint64_t   order = 0; // when it was last submitted
		RingTarget target;
	};

	// A free slot, or one whose submission has finished; failing that the oldest slot of `target`'s
	// own ring, so a compute ring parked on a WAIT_REG_MEM never holds up the submission that would
	// release it (with shared round-robin slots, that is a deadlock). Only then the oldest overall.
	// Slots in the `taken` mask are skipped (the other IBs of the same CS).
	uint32_t PickSlot(RingTarget target, const std::bitset<kSlots>& taken = {}) {
		for (uint32_t i = 0; i < kSlots; i++) {
			if (!taken[i] && m_slot[i].seq == 0) {
				return i;
			}
		}
		for (uint32_t i = 0; i < kSlots; i++) {
			if (!taken[i] && WaitSeq(m_slot[i].seq, 0.0, m_slot[i].target)) {
				m_slot[i].seq = 0;
				return i;
			}
		}
		uint32_t own = kSlots, any = kSlots;
		for (uint32_t i = 0; i < kSlots; i++) {
			if (taken[i]) {
				continue;
			}
			const bool same = m_slot[i].target.ip_type == target.ip_type && m_slot[i].target.ring == target.ring;
			if (same && (own == kSlots || m_slot[i].order < m_slot[own].order)) {
				own = i;
			}
			if (any == kSlots || m_slot[i].order < m_slot[any].order) {
				any = i;
			}
		}
		return own != kSlots ? own : any;
	}
	bool     m_async     = false;
	uint64_t m_slot_order = 0;
	SlotUse  m_slot[kSlots];
	uint32_t m_compute_rings = 0; // bitmask from AMDGPU_INFO_HW_IP_INFO
	std::vector<uint32_t> m_gds_handles; // GDS, GWS and OA buffers listed in every submission
	uint32_t              m_gds_size = 0;
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESUBMIT_H_
