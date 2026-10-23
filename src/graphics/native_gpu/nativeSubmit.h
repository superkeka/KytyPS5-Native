#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESUBMIT_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESUBMIT_H_

#include "graphics/native_gpu/gemBacking.h" // KYTY_GEM_BACKING_SUPPORTED + amdgpu uAPI headers

// Submits a translated IB to the amdgpu gfx ring for the BC-250 native backend, on the same DRM
// file whose GPU VM holds the guest memory bindings (GemBacking::DrmFd()). Only real Linux with the
// amdgpu uAPI has it; elsewhere this header is empty and the shadow hook compiles to a no-op.

#if KYTY_GEM_BACKING_SUPPORTED

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <span>
#include <vector>

namespace Libs::Graphics::NativeGpu {

class NativeSubmitter {
public:
	// The IB lives at a GPU VA above every guest range (guest memory tops out well below this and
	// the extended-memory aperture is at 0x80000000000; the device's low VA range reaches
	// 0x800000000000, so this sits clear of both).
	static constexpr uint64_t kIbVa      = 0x400000000000ull;
	static constexpr uint64_t kIbSize    = 2ull * 1024 * 1024;
	// The last bytes of the IB buffer are a scratch area the IB itself may write into (pipeline
	// statistics samples); IBs must stay below it.
	static constexpr uint64_t kScratchSize = 4096;
	// Extra DRM_IOCTL_AMDGPU_CS attempts spent waiting out a userptr revalidation (-EFAULT).
	static constexpr int      kCsRetries = 64;

	struct Result {
		bool   submitted = false; // the CS ioctl was accepted
		bool   completed = false; // the fence signalled before the timeout
		bool   timed_out = false; // the wait timed out
		bool   reset     = false; // the context reported a reset/guilty state afterwards
		int    err       = 0;     // negative errno from a failed ioctl
		double seconds   = 0.0;   // submit -> fence wall time
		double cs_seconds = 0.0;  // time spent inside the CS ioctl(s), retries included
		int    retries   = 0;     // extra CS attempts spent on userptr revalidation (EFAULT)
	};

	// Allocates a context and the IB buffer on `drm_fd`. Returns nullptr on failure.
	static NativeSubmitter* Create(int drm_fd) {
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

		if (!self->CreateIb()) {
			delete self;
			return nullptr;
		}
		// Opt-in: with GDS, GWS and OA buffers in every submission (so amdgpu reprograms the GDS
		// partition for each job) the BC-250 hung within ~30 s in Dead Cells and ASTRO BOT alike, even
		// when no packet or shader used GDS.
		if (const char* opt = std::getenv("KYTY_NATIVE_GPU_GDS"); opt != nullptr && opt[0] == '1') {
			self->CreateGds();
		}
		return self;
	}

	~NativeSubmitter() {
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
		if (m_ib_handle != 0) {
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
	[[nodiscard]] uint32_t GdsSize() const { return m_gds_size; }
	[[nodiscard]] volatile uint8_t* ScratchCpu() const {
		return static_cast<volatile uint8_t*>(m_ib_cpu) + kIbSize - kScratchSize;
	}

	NativeSubmitter(const NativeSubmitter&)            = delete;
	NativeSubmitter& operator=(const NativeSubmitter&) = delete;

	// Copies `ib` into the IB buffer, submits it to the gfx ring with `bo_handles` (plus the IB)
	// resident, waits up to `timeout_sec` and queries the context state.
	Result Submit(std::span<const uint32_t> ib, const std::vector<uint32_t>& bo_handles,
	              double timeout_sec) {
		Result res;
		if (ib.size() * 4 > kIbSize - kScratchSize) {
			res.err = -E2BIG;
			return res;
		}
		std::memcpy(m_ib_cpu, ib.data(), ib.size() * 4);
		__sync_synchronize();

		std::vector<drm_amdgpu_bo_list_entry> entries;
		entries.reserve(bo_handles.size() + 1);
		drm_amdgpu_bo_list_entry e;
		std::memset(&e, 0, sizeof(e));
		e.bo_handle = m_ib_handle;
		entries.push_back(e);
		// A GDS, GWS or OA buffer in the list is what gives the job its GDS partition.
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

		struct drm_amdgpu_cs_chunk_ib ib_info;
		std::memset(&ib_info, 0, sizeof(ib_info));
		ib_info.va_start = m_ib_va;
		ib_info.ib_bytes = static_cast<uint32_t>(ib.size() * 4);
		ib_info.ip_type  = AMDGPU_HW_IP_GFX;
		ib_info.ring     = 0;

		struct drm_amdgpu_cs_chunk chunks[2];
		std::memset(chunks, 0, sizeof(chunks));
		chunks[0].chunk_id   = AMDGPU_CHUNK_ID_IB;
		chunks[0].length_dw  = sizeof(ib_info) / 4;
		chunks[0].chunk_data = reinterpret_cast<uintptr_t>(&ib_info);
		chunks[1].chunk_id   = AMDGPU_CHUNK_ID_BO_HANDLES;
		chunks[1].length_dw  = sizeof(list) / 4;
		chunks[1].chunk_data = reinterpret_cast<uintptr_t>(&list);
		uint64_t chunk_ptrs[2] = {reinterpret_cast<uintptr_t>(&chunks[0]),
		                          reinterpret_cast<uintptr_t>(&chunks[1])};

		union drm_amdgpu_cs cs;
		std::memset(&cs, 0, sizeof(cs));
		cs.in.ctx_id     = m_ctx;
		cs.in.num_chunks = 2;
		cs.in.chunks     = reinterpret_cast<uintptr_t>(chunk_ptrs);

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

		union drm_amdgpu_wait_cs w;
		std::memset(&w, 0, sizeof(w));
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		w.in.handle  = seq;
		w.in.timeout = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull +
		               static_cast<uint64_t>(ts.tv_nsec) + static_cast<uint64_t>(timeout_sec * 1e9);
		w.in.ip_type = AMDGPU_HW_IP_GFX;
		w.in.ring    = 0;
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

private:
	NativeSubmitter() = default;

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

	bool CreateIb() {
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size      = kIbSize;
		c.in.alignment    = 64 * 1024;
		c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
		c.in.domain_flags = AMDGPU_GEM_CREATE_CPU_GTT_USWC; // write-combined, like Mesa's IBs
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
	void CreateGds() {
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
		const uint32_t gds_size = std::max({gds.gds_gfx_partition_size, gds.compute_partition_size, gds.gds_total_size});
		const uint32_t gws_size = std::max(gds.gws_per_gfx_partition, gds.gws_per_compute_partition);
		const uint32_t oa_size  = std::max(gds.oa_per_gfx_partition, gds.oa_per_compute_partition);
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
	std::vector<uint32_t> m_gds_handles; // GDS, GWS and OA buffers listed in every submission
	uint32_t              m_gds_size = 0;
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVESUBMIT_H_
