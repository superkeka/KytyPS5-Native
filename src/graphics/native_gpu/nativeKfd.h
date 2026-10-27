#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEKFD_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEKFD_H_

#include "graphics/native_gpu/gemBacking.h"

#if KYTY_GEM_BACKING_SUPPORTED

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace Libs::Graphics::NativeGpu {

// A KFD user-mode compute queue: a PM4 ring of our own in the GPU VM of the GemBacking's render-node
// file, submitted by writing its write pointer to a doorbell, with no CS ioctl. This is how the PS5
// kernel gives a title its compute queues (a hardware queue per queue, rung directly), and what the
// translator's asynchronous compute buffers run on with KYTY_NATIVE_GPU_KFD_QUEUES=1.
//
// Every IB the queue runs is followed by an end-of-pipe RELEASE_MEM of its sequence number to the
// queue's fence dword, so a submission is done once Fence() has reached it. The IB itself, and all
// memory it reaches, must be KFD buffers (GemBacking with KYTY_NATIVE_GPU_KFD_MEMORY): KFD keeps those
// resident, while a render-node buffer is only valid during a CS that lists it.
//
// Queue creation follows the kernel's checks (kfd_queue_acquire_buffers): a 4 KiB EOP buffer in VRAM,
// a context save area of exactly cwsr_size plus 32 bytes per wave, a write/read pointer page, and the
// doorbell mapped through the process's whole 8 KiB doorbell slice.
class KfdQueue {
public:
	// `va` is a private GPU VA range of at least 64 MiB for the queue's own buffers.
	// `priority` is KFD's queue priority, 0-15 (the CP_HQD_QUEUE_PRIORITY the scheduler gives it).
	static KfdQueue* Create(int kfd_fd, uint32_t gpu_id, int drm_fd, uint64_t va, uint32_t priority = 7) {
		auto* q    = new KfdQueue();
		q->m_kfd   = kfd_fd;
		q->m_gpu   = gpu_id;
		q->m_drm   = drm_fd;
		uint32_t cu = 0, cwsr = 0, ctl = 0;
		ReadTopology(gpu_id, &cu, &cwsr, &ctl);
		const uint64_t debug = (static_cast<uint64_t>(cu) * 32 * 32 + 63) & ~63ull; // 32 waves per CU, 32 bytes each
		const uint64_t save  = (cwsr + debug + 4095) & ~4095ull;
		// The flags kfd_probe's queue runs with (cached, coherent GTT).
		constexpr uint32_t gtt = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
		                         KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
		                         KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
		if (cu == 0 || cwsr == 0 || !q->Alloc(&q->m_ring, va, kRingBytes, gtt) || !q->Alloc(&q->m_ptrs, va + 0x100000, 4096, gtt) ||
		    // The CP's fetcher reaches the EOP buffer with execute permission (an end-of-pipe RELEASE_MEM
		    // without it is a permission fault on the queue's VMID).
		    !q->Alloc(&q->m_eop, va + 0x200000, 4096,
		              KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
		                  KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE) ||
		    !q->Alloc(&q->m_save, va + 0x1000000, save, gtt)) {
			std::fprintf(stderr, "KfdQueue: buffers for a queue failed (%u CUs, cwsr %u)\n", cu, cwsr);
			delete q;
			return nullptr;
		}
		std::memset(q->m_ring.cpu, 0, kRingBytes);
		std::memset(q->m_ptrs.cpu, 0, 4096);
		struct kfd_ioctl_create_queue_args a {};
		a.ring_base_address        = q->m_ring.va;
		a.write_pointer_address    = q->m_ptrs.va;
		a.read_pointer_address     = q->m_ptrs.va + kRptrOffset;
		a.ring_size                = kRingBytes;
		a.gpu_id                   = gpu_id;
		a.queue_type               = KFD_IOC_QUEUE_TYPE_COMPUTE;
		a.queue_percentage         = 100;
		a.queue_priority           = priority;
		a.eop_buffer_address       = q->m_eop.va;
		a.eop_buffer_size          = 4096;
		a.ctx_save_restore_address = q->m_save.va;
		a.ctx_save_restore_size    = cwsr;
		a.ctl_stack_size           = ctl;
		int r = q->KfdIoctl(AMDKFD_IOC_CREATE_QUEUE, &a);
		if (r != 0) {
			std::fprintf(stderr, "KfdQueue: CREATE_QUEUE failed: %s\n", std::strerror(-r));
			delete q;
			return nullptr;
		}
		q->m_queue_id    = a.queue_id;
		q->m_has_queue   = true;
		constexpr uint64_t kSlice = 8192;
		void* db = ::mmap(nullptr, kSlice, PROT_READ | PROT_WRITE, MAP_SHARED, kfd_fd, static_cast<off_t>(a.doorbell_offset & ~(kSlice - 1)));
		if (db == MAP_FAILED) {
			std::fprintf(stderr, "KfdQueue: doorbell mmap failed: %s\n", std::strerror(errno));
			delete q;
			return nullptr;
		}
		q->m_db_page  = db;
		q->m_doorbell = reinterpret_cast<volatile uint64_t*>(static_cast<uint8_t*>(db) + (a.doorbell_offset & (kSlice - 1)));
		// A first packet right away (a fence write), as kfd_probe does: the queue is known to run before
		// any work depends on it.
		const uint64_t s = ++q->m_seq;
		q->Push(Pkt3(0x37, 4)); // WRITE_DATA: memory, write confirm
		q->Push((5u << 8u) | (1u << 20u));
		q->Push(static_cast<uint32_t>(q->FenceVa()));
		q->Push(static_cast<uint32_t>(q->FenceVa() >> 32u));
		q->Push(static_cast<uint32_t>(s));
		__sync_synchronize();
		*reinterpret_cast<volatile uint64_t*>(q->m_ptrs.cpu) = q->m_wptr;
		__sync_synchronize();
		*q->m_doorbell = q->m_wptr;
		if (!q->Wait(s, 1.0)) {
			std::fprintf(stderr, "KfdQueue: queue %u does not run its first packet\n", q->m_queue_id);
			delete q;
			return nullptr;
		}
		return q;
	}

	~KfdQueue() {
		if (m_has_queue) {
			struct kfd_ioctl_destroy_queue_args d {};
			d.queue_id = m_queue_id;
			KfdIoctl(AMDKFD_IOC_DESTROY_QUEUE, &d);
		}
		if (m_db_page != nullptr) {
			::munmap(m_db_page, 8192);
		}
		Free(&m_save);
		Free(&m_eop);
		Free(&m_ptrs);
		Free(&m_ring);
	}

	KfdQueue(const KfdQueue&)            = delete;
	KfdQueue& operator=(const KfdQueue&) = delete;

	// Runs the `dwords` of IB at GPU VA `ib_va`, then writes the returned sequence number to the fence.
	// False when the ring stayed full for `timeout_sec`.
	bool Submit(uint64_t ib_va, uint32_t dwords, double timeout_sec, uint64_t* seq) {
		constexpr uint32_t kPacket = 8 + 4 + 8; // ACQUIRE_MEM + INDIRECT_BUFFER + RELEASE_MEM
		const double       end     = Now() + timeout_sec;
		while (m_wptr + kPacket - Rptr() > kRingDwords - 8) {
			if (Now() > end) {
				return false;
			}
			::sched_yield();
		}
		const uint64_t s = ++m_seq;
		// Before every IB, what amdgpu puts before every IB of a CS (gfx_v10_0_emit_mem_sync): an ACQUIRE_MEM
		// that writes back and invalidates GL2, GLM, GL1, GLV, GLK and GLI. GL2 caches guest memory (GTT,
		// MTYPE NC) and does not see the title's CPU stores, so without it a dispatch read descriptors the
		// title had rewritten since (My First Gran Turismo: SQC faults at garbage addresses once tracker
		// stores stopped invalidating GL2); on the PS5 memory shared with the GPU is coherent. Once per
		// submitted ACB, not at every label. KYTY_NATIVE_GPU_KFD_IB_SYNC=0 leaves it out, =1 also ends the
		// IB with amdgpu's fence form.
		static const char* const ib_sync_env = std::getenv("KYTY_NATIVE_GPU_KFD_IB_SYNC");
		static const bool        ib_acquire  = ib_sync_env == nullptr || ib_sync_env[0] != '0';
		static const bool        ib_sync     = ib_sync_env != nullptr && ib_sync_env[0] == '1';
		if (ib_acquire) {
			Push(Pkt3(0x58, 7));
			Push(0u);
			Push(0xffffffffu);
			Push(0x00ffffffu);
			Push(0u);
			Push(0u);
			Push(0x0000000au);
			Push(1u | (1u << 4u) | (1u << 5u) | (1u << 7u) | (1u << 8u) | (1u << 9u) | (1u << 14u) | (1u << 15u));
		}
		// INDIRECT_BUFFER: address (dword aligned), size and VALID; the queue's own VMID runs it.
		Push(Pkt3(0x3f, 3));
		Push(static_cast<uint32_t>(ib_va) & ~3u);
		Push(static_cast<uint32_t>(ib_va >> 32u));
		Push(dwords | (1u << 23u));
		// RELEASE_MEM end of pipe, the 32-bit data written past the caches: the queue's own sequence, which
		// only says the IB finished (its slot can be reused). With KYTY_NATIVE_GPU_KFD_IB_SYNC=1 amdgpu's
		// fence form (CACHE_FLUSH_AND_INV_TS_EVENT, GLM and GL2 written back in sequence); else bottom of
		// pipe without cache actions, the title's releases make its results visible.
		Push(Pkt3(0x49, 7));
		Push(ib_sync ? 0x14u | (5u << 8u) | (1u << 12u) | (1u << 13u) | (1u << 21u) | (1u << 22u) | (3u << 25u)
		             : 0x28u | (5u << 8u) | (3u << 25u));
		Push(1u << 29u);
		Push(static_cast<uint32_t>(FenceVa()));
		Push(static_cast<uint32_t>(FenceVa() >> 32u));
		Push(static_cast<uint32_t>(s));
		Push(0u);
		Push(0u);
		__sync_synchronize();
		*reinterpret_cast<volatile uint64_t*>(m_ptrs.cpu) = m_wptr;
		__sync_synchronize();
		*m_doorbell = m_wptr;
		*seq        = s;
		return true;
	}

	// The last sequence number the queue has finished (32-bit wrap-around compare with Reached).
	[[nodiscard]] uint32_t Fence() const { return *reinterpret_cast<volatile const uint32_t*>(static_cast<uint8_t*>(m_ptrs.cpu) + kFenceOffset); }
	[[nodiscard]] bool     Reached(uint64_t seq) const { return static_cast<int32_t>(Fence() - static_cast<uint32_t>(seq)) >= 0; }

	// Where the queue stands, for a timeout report.
	void Describe(char* out, size_t size) const {
		const uint64_t rptr = *reinterpret_cast<volatile const uint64_t*>(static_cast<uint8_t*>(m_ptrs.cpu) + kRptrOffset);
		std::snprintf(out, size, "queue %u: wptr %llu, rptr %llu (ring %u dwords), fence %u of %llu submitted", m_queue_id,
		              static_cast<unsigned long long>(m_wptr), static_cast<unsigned long long>(rptr), kRingDwords, Fence(),
		              static_cast<unsigned long long>(m_seq));
	}

	// Waits until `seq` is done; false on a timeout.
	bool Wait(uint64_t seq, double timeout_sec) const {
		const double end = Now() + timeout_sec;
		while (!Reached(seq)) {
			if (Now() > end) {
				return false;
			}
			struct timespec nap = {0, 50000};
			nanosleep(&nap, nullptr);
		}
		return true;
	}

	// True when KFD sized its queues' wave save areas for more CUs than amdgpu counted at init (the
	// BC-250's patched amdgpu: every CU the chip has), so waves may run on CUs enabled after init.
	static bool SaveAreaCoversLateCus(uint32_t gpu_id) {
		uint32_t cu = 0, cwsr = 0, ctl = 0, counted = 0;
		ReadTopology(gpu_id, &cu, &cwsr, &ctl, &counted);
		return cu > counted;
	}

private:
	static constexpr uint32_t kRingBytes   = 64 * 1024;
	static constexpr uint32_t kRingDwords  = kRingBytes / 4;
	static constexpr uint32_t kRptrOffset  = 64;
	static constexpr uint32_t kFenceOffset = 128;

	struct Mem {
		uint64_t va     = 0;
		uint64_t size   = 0;
		uint64_t handle = 0;
		void*    cpu    = nullptr;
	};

	KfdQueue() = default;

	static double Now() {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
	}
	// PM4 type 3 for a compute queue (shader type bit set).
	static uint32_t Pkt3(uint32_t op, uint32_t body) { return 0xc0000000u | ((body - 1u) << 16u) | (op << 8u) | 2u; }

	[[nodiscard]] uint64_t FenceVa() const { return m_ptrs.va + kFenceOffset; }
	[[nodiscard]] uint64_t Rptr() const {
		// The CP reports the read pointer in dwords, modulo the ring size.
		const uint64_t r = *reinterpret_cast<volatile const uint64_t*>(static_cast<uint8_t*>(m_ptrs.cpu) + kRptrOffset) % kRingDwords;
		const uint64_t w = m_wptr % kRingDwords;
		return m_wptr - ((w + kRingDwords - r) % kRingDwords);
	}
	void Push(uint32_t v) { static_cast<uint32_t*>(m_ring.cpu)[(m_wptr++) % kRingDwords] = v; }

	int KfdIoctl(unsigned long request, void* arg) const {
		int r = -1;
		do {
			r = ::ioctl(m_kfd, request, arg);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		return r == -1 ? -errno : 0;
	}

	bool Alloc(Mem* m, uint64_t va, uint64_t size, uint32_t flags) {
		struct kfd_ioctl_alloc_memory_of_gpu_args a {};
		a.va_addr = va;
		a.size    = size;
		a.gpu_id  = m_gpu;
		a.flags   = flags;
		if (KfdIoctl(AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a) != 0) {
			return false;
		}
		m->va     = va;
		m->size   = size;
		m->handle = a.handle;
		struct kfd_ioctl_map_memory_to_gpu_args mp {};
		mp.handle               = a.handle;
		mp.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_gpu);
		mp.n_devices            = 1;
		if (KfdIoctl(AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mp) != 0) {
			return false;
		}
		if ((flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) != 0) {
			void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_drm, static_cast<off_t>(a.mmap_offset));
			if (p == MAP_FAILED) {
				return false;
			}
			m->cpu = p;
		}
		return true;
	}

	void Free(Mem* m) {
		if (m->cpu != nullptr) {
			::munmap(m->cpu, m->size);
		}
		if (m->handle != 0) {
			struct kfd_ioctl_unmap_memory_from_gpu_args um {};
			um.handle               = m->handle;
			um.device_ids_array_ptr = reinterpret_cast<uintptr_t>(&m_gpu);
			um.n_devices            = 1;
			KfdIoctl(AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &um);
			struct kfd_ioctl_free_memory_of_gpu_args fr {};
			fr.handle = m->handle;
			KfdIoctl(AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
		}
		*m = Mem {};
	}

	static void ReadTopology(uint32_t gpu_id, uint32_t* cu, uint32_t* cwsr, uint32_t* ctl, uint32_t* counted_out = nullptr) {
		for (int n = 0; n < 16; n++) {
			char path[128];
			std::snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", n);
			FILE*    f  = std::fopen(path, "re");
			unsigned id = 0;
			if (f == nullptr) {
				continue;
			}
			const bool ok = std::fscanf(f, "%u", &id) == 1 && id == gpu_id;
			std::fclose(f);
			if (!ok) {
				continue;
			}
			std::snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/properties", n);
			if ((f = std::fopen(path, "re")) == nullptr) {
				return;
			}
			char     line[128];
			unsigned simd = 0, per_cu = 1, arrays = 0, cu_per_array = 0;
			while (std::fgets(line, sizeof(line), f) != nullptr) {
				std::sscanf(line, "simd_count %u", &simd);
				std::sscanf(line, "simd_per_cu %u", &per_cu);
				std::sscanf(line, "array_count %u", &arrays);
				std::sscanf(line, "cu_per_simd_array %u", &cu_per_array);
				std::sscanf(line, "cwsr_size %u", cwsr);
				std::sscanf(line, "ctl_stack_size %u", ctl);
			}
			std::fclose(f);
			// The CUs KFD sized the save area for: the ones amdgpu counted at init, or every CU the chip
			// has (the BC-250's patched amdgpu: 40, of which bc250-cu-live-manager enables 16 after
			// init). The control stack, 12 bytes for each of 32 waves per CU plus a 48-byte header,
			// page aligned, tells which; the debug area the queue's buffer also holds follows from it.
			const uint32_t counted  = per_cu != 0 ? simd / per_cu : 0;
			const uint32_t physical = arrays * cu_per_array;
			const auto     ctl_for  = [](uint32_t cus) { return std::min<uint32_t>((cus * 32 * 12 + 48 + 4095) & ~4095u, 0x7000); };
			*cu                     = physical > counted && *ctl == ctl_for(physical) ? physical : counted;
			if (counted_out != nullptr) {
				*counted_out = counted;
			}
			return;
		}
	}

	int               m_kfd = -1;
	int               m_drm = -1;
	uint32_t          m_gpu = 0;
	Mem               m_ring, m_ptrs, m_eop, m_save;
	uint32_t          m_queue_id  = 0;
	bool              m_has_queue = false;
	void*             m_db_page   = nullptr;
	volatile uint64_t* m_doorbell = nullptr;
	uint64_t          m_wptr      = 0;
	uint64_t          m_seq       = 0;
};

// The end-of-pipe interrupt of the tracker's releases: a KFD signal event whose id they carry as
// their context id (INT_SEL 2), so the tracker sleeps until the GPU passes a point, woken by the
// interrupt the way the PS5 kernel takes a title's EOP interrupts. Stock KFD drops EOP interrupts of
// render-node submissions (their VMIDs are not KFD's); the patched amdgpu (tools/bc250-amdgpu,
// bc250_gfx_eop_events=1) signals the event its context id names, from gfx rings, amdgpu compute
// rings and KFD queues alike. The event resets as a wait takes it, so interrupts that land while
// nobody waits leave it set for the next wait and none is lost.
class KfdEopEvent {
public:
	// The event's id, 0 when the kernel would not deliver it (stock amdgpu, or no KFD process).
	static uint32_t Create(int kfd_fd) {
		int on = 0;
		if (FILE* f = std::fopen("/sys/module/amdgpu/parameters/bc250_gfx_eop_events", "re"); f != nullptr) {
			if (std::fscanf(f, "%d", &on) != 1) {
				on = 0;
			}
			std::fclose(f);
		}
		if (on != 1 || kfd_fd < 0) {
			return 0;
		}
		struct kfd_ioctl_create_event_args a {};
		a.event_type = KFD_IOC_EVENT_SIGNAL;
		a.auto_reset = 1;
		if (::ioctl(kfd_fd, AMDKFD_IOC_CREATE_EVENT, &a) != 0) {
			std::fprintf(stderr, "KfdEopEvent: CREATE_EVENT failed: %s\n", std::strerror(errno));
			return 0;
		}
		return a.event_id;
	}

	// True when the event fired within `timeout_ms` (or had fired since the last wait).
	static bool Wait(int kfd_fd, uint32_t id, uint32_t timeout_ms) {
		struct kfd_event_data             ev {};
		struct kfd_ioctl_wait_events_args w {};
		ev.event_id  = id;
		w.events_ptr = reinterpret_cast<uintptr_t>(&ev);
		w.num_events = 1;
		w.timeout    = timeout_ms;
		int r        = 0;
		do {
			r = ::ioctl(kfd_fd, AMDKFD_IOC_WAIT_EVENTS, &w);
		} while (r == -1 && errno == EINTR);
		return r == 0 && w.wait_result == KFD_IOC_WAIT_RESULT_COMPLETE;
	}
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEKFD_H_
