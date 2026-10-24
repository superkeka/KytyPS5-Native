// BC-250 guest-memory model probe: what an exact memory model for the native backend can rely on.
//
//   alloc    when a VM_ALWAYS_VALID GTT buffer takes RAM: at create, GEM_VA, CPU touch
//   views    CPU views of an always-valid buffer at any page offset (DRM mmap + mremap), aliasing,
//            PROT_EXEC, mprotect, and that such a buffer refuses dma-buf export
//   cs       a gfx-ring job writing an always-valid buffer with no BO list at all
//   csbench  CS ioctl cost: no BO list, a persistent BO list handle, a per-CS list of N buffers
//   gds      (GPU, gfx ring) does DMA_DATA address GDS relative to the job's partition or absolutely
//   gdsfull  (GPU, gfx ring) the whole GDS + OA in every job: repeated GDS jobs, one partition
//   gdsfull-shader-first  same, but jobs after the first touch GDS from the shader first
//
// Raw ioctls, no libdrm. Build in the bc250-mesa-dev distrobox: make mem_probe.

#define _GNU_SOURCE
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <setjmp.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "shaders.h"

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define KiB           ((uint64_t)1024)
#define MiB           ((uint64_t)1024 * KiB)
#define GiB           ((uint64_t)1024 * MiB)

static int g_fd;
static uint32_t g_ctx;

static double NowSec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int Ioctl(unsigned long req, void* arg) {
	int r;
	do {
		r = ioctl(g_fd, req, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

static const char* Err(int r) { return r == 0 ? "ok" : strerror(-r); }

static uint64_t MemAvailMiB(void) {
	FILE* f = fopen("/proc/meminfo", "r");
	char  line[256];
	unsigned long long kb = 0;
	while (f != NULL && fgets(line, sizeof(line), f) != NULL) {
		if (sscanf(line, "MemAvailable: %llu kB", &kb) == 1) {
			break;
		}
	}
	if (f != NULL) {
		fclose(f);
	}
	return kb / 1024;
}

static uint64_t GttUsedMiB(void) {
	FILE* f = fopen("/sys/class/drm/renderD128/device/mem_info_gtt_used", "r");
	unsigned long long b = 0;
	if (f != NULL) {
		if (fscanf(f, "%llu", &b) != 1) {
			b = 0;
		}
		fclose(f);
	}
	return b / MiB;
}

static void Mem(const char* what) {
	printf("    %-44s MemAvailable %6" PRIu64 " MiB, GTT used %6" PRIu64 " MiB\n", what, MemAvailMiB(), GttUsedMiB());
}

static int Create(uint64_t size, uint32_t domain, uint64_t flags, uint32_t* handle) {
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = size >= 2 * MiB ? 2 * MiB : 4 * KiB;
	c.in.domains      = domain;
	c.in.domain_flags = flags;
	int r             = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
	*handle           = r == 0 ? c.out.handle : 0;
	return r;
}

static void Close(uint32_t handle) {
	struct drm_gem_close c = {.handle = handle};
	Ioctl(DRM_IOCTL_GEM_CLOSE, &c);
}

static uint64_t MmapOffset(uint32_t handle) {
	union drm_amdgpu_gem_mmap m;
	memset(&m, 0, sizeof(m));
	m.in.handle = handle;
	return Ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &m) == 0 ? m.out.addr_ptr : 0;
}

#define VA_RWX (AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE)

static int Va(uint32_t op, uint32_t handle, uint64_t va, uint64_t off, uint64_t size) {
	struct drm_amdgpu_gem_va v;
	memset(&v, 0, sizeof(v));
	v.handle       = handle;
	v.operation    = op;
	v.flags        = op == AMDGPU_VA_OP_CLEAR ? 0 : VA_RWX;
	v.va_address   = va;
	v.offset_in_bo = off;
	v.map_size     = size;
	return Ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &v);
}

// Maps buffer bytes [inner, inner + len) at exactly `dst`: TTM only maps a buffer from its start
// (a DRM mmap at offset + inner is EINVAL), so map [0, inner + len) anywhere and move the tail.
static int MapSlice(uint64_t mmap_off, uint64_t inner, uint64_t len, int prot, void* dst) {
	uint8_t* t = mmap(NULL, inner + len, prot, MAP_SHARED, g_fd, (off_t)mmap_off);
	if (t == MAP_FAILED) {
		return -errno;
	}
	void* r = mremap(t + inner, len, len, MREMAP_MAYMOVE | MREMAP_FIXED, dst);
	int   e = r == MAP_FAILED ? -errno : 0;
	if (inner > 0) {
		munmap(t, inner);
	}
	if (e != 0) {
		munmap(t + inner, len);
	}
	return e;
}

static sigjmp_buf g_jmp;
static void       OnSegv(int sig) {
	(void)sig;
	siglongjmp(g_jmp, 1);
}

// True when a write of one byte at p faults.
static bool WriteFaults(volatile uint8_t* p) {
	struct sigaction sa, old;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = OnSegv;
	sigaction(SIGSEGV, &sa, &old);
	bool faulted = false;
	if (sigsetjmp(g_jmp, 1) == 0) {
		*p = *p;
	} else {
		faulted = true;
	}
	sigaction(SIGSEGV, &old, NULL);
	return faulted;
}

// ---------------------------------------------------------------------------------------------

static void ProbeAlloc(uint64_t size) {
	printf("== alloc: %" PRIu64 " MiB GTT buffers, plain and VM_ALWAYS_VALID\n", size / MiB);
	const uint64_t flags[] = {0, AMDGPU_GEM_CREATE_VM_ALWAYS_VALID};
	for (int k = 0; k < 2; k++) {
		printf("  %s\n", k == 0 ? "plain" : "always-valid");
		Mem("before");
		uint32_t h = 0;
		int      r = Create(size, AMDGPU_GEM_DOMAIN_GTT, flags[k], &h);
		if (r != 0) {
			printf("    create: %s\n", Err(r));
			continue;
		}
		Mem("after create");
		const uint64_t va = 0x1000000000ull;
		r                 = Va(AMDGPU_VA_OP_MAP, h, va, 0, size);
		Mem(r == 0 ? "after GEM_VA" : "GEM_VA FAILED");
		uint8_t* p = mmap((void*)va, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, g_fd,
		                  (off_t)MmapOffset(h));
		if (p == MAP_FAILED) {
			printf("    identity mmap: %s\n", strerror(errno));
		} else {
			printf("    identity CPU map at the GPU VA: %s\n", (uint64_t)p == va ? "ok" : "moved");
			p[0] = 1;
			Mem("after touching one page");
			double t0 = NowSec();
			for (uint64_t i = 0; i < size; i += 4096) {
				p[i] = 1;
			}
			printf("    touching every page: %.0f ms\n", (NowSec() - t0) * 1e3);
			Mem("after touching every page");
			munmap(p, size);
		}
		Va(AMDGPU_VA_OP_CLEAR, 0, va, 0, size);
		Close(h);
		Mem("after close");
	}
}

static void ProbeViews(void) {
	const uint64_t size = 256 * MiB;
	printf("== views: CPU views of a %" PRIu64 " MiB always-valid buffer at arbitrary page offsets\n", size / MiB);
	uint32_t h = 0;
	int      r = Create(size, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_GEM_CREATE_VM_ALWAYS_VALID, &h);
	if (r != 0) {
		printf("  create: %s\n", Err(r));
		return;
	}
	const uint64_t off = MmapOffset(h);
	uint8_t*       a   = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, (off_t)off);
	printf("  whole-buffer map: %s\n", a == MAP_FAILED ? strerror(errno) : "ok");
	if (a == MAP_FAILED) {
		Close(h);
		return;
	}
	for (uint64_t i = 0; i < size; i += 4096) {
		*(uint64_t*)(a + i) = i;
	}
	uint8_t* bad = mmap(NULL, 64 * KiB, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, (off_t)(off + 0x40000));
	printf("  direct mmap at buffer offset 0x40000: %s (expected EINVAL)\n",
	       bad == MAP_FAILED ? strerror(errno) : "ok");
	if (bad != MAP_FAILED) {
		munmap(bad, 64 * KiB);
	}

	// A guest-like reservation: PROT_NONE anonymous placeholder the views replace.
	const uint64_t resv_size = 64 * MiB;
	uint8_t* resv = mmap((void*)0x2000000000ull, resv_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
	                     -1, 0);
	if (resv == MAP_FAILED) {
		printf("  reservation: %s\n", strerror(errno));
		goto out;
	}
	{
		const uint64_t inner = 0x3454000; // 16 KiB-aligned, not 2 MiB-aligned
		const uint64_t len   = 0x2c000;
		uint8_t*       v1    = resv + 0x100000;
		uint8_t*       v2    = resv + 0x800000;
		double         t0    = NowSec();
		r                    = MapSlice(off, inner, len, PROT_READ | PROT_WRITE, v1);
		printf("  view 1 (buffer +0x%" PRIx64 ", 0x%" PRIx64 " bytes) inside the reservation: %s (%.1f us)\n", inner,
		       len, Err(r), (NowSec() - t0) * 1e6);
		if (r == 0) {
			bool same = true;
			for (uint64_t i = 0; i < len; i += 4096) {
				same = same && *(uint64_t*)(v1 + i) == inner + i;
			}
			printf("    reads the right buffer bytes: %s\n", same ? "yes" : "NO");
			*(uint64_t*)(v1 + 0x1000) = 0x1234567890abcdefull;
			printf("    write through the view lands in the buffer: %s\n",
			       *(uint64_t*)(a + inner + 0x1000) == 0x1234567890abcdefull ? "yes" : "NO");
		}
		r = MapSlice(off, inner + 0x1000, 0x4000, PROT_READ | PROT_WRITE, v2);
		printf("  view 2 aliasing view 1 (+0x1000): %s, sees view 1's write: %s\n", Err(r),
		       r == 0 && *(uint64_t*)v2 == 0x1234567890abcdefull ? "yes" : "NO");
		r = mprotect(v1, len, PROT_READ);
		printf("  mprotect view 1 read-only: %s, write faults: %s\n", r == 0 ? "ok" : strerror(errno),
		       WriteFaults(v1) ? "yes" : "NO");
		mprotect(v1, len, PROT_READ | PROT_WRITE);
		printf("  write after mprotect back: %s\n", WriteFaults(v1) ? "FAULTS" : "ok");
		r = MapSlice(off, inner, 0x4000, PROT_READ | PROT_EXEC, resv + 0x1000000);
		printf("  PROT_EXEC view: %s\n", Err(r));
		// Remapping over an existing view, as a guest MAP_FIXED over a mapping does.
		r = MapSlice(off, 0x10000000 - 0x8000, 0x8000, PROT_READ | PROT_WRITE, v1);
		printf("  view over view 1 (buffer tail): %s, reads: %s\n", Err(r),
		       r == 0 && *(uint64_t*)v1 == 0x10000000 - 0x8000 ? "right bytes" : "WRONG");
		munmap(a, size);
		a = NULL;
		printf("  views after the whole-buffer map is gone: view 2 still reads: %s\n",
		       *(uint64_t*)v2 == 0x1234567890abcdefull ? "yes" : "NO");
		Close(h);
		h = 0;
		printf("  views after GEM_CLOSE of the handle: view 2 still reads: %s\n",
		       *(uint64_t*)v2 == 0x1234567890abcdefull ? "yes" : "NO");
		// Counting VMAs: a view per guest mapping must stay within vm.max_map_count.
		int   n    = 0;
		char  line[512];
		FILE* maps = fopen("/proc/self/maps", "r");
		while (maps != NULL && fgets(line, sizeof(line), maps) != NULL) {
			n++;
		}
		if (maps != NULL) {
			fclose(maps);
		}
		printf("  VMAs in this process: %d\n", n);
		munmap(resv, resv_size);
	}
	{
		uint32_t h2 = 0;
		if (Create(64 * KiB, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_GEM_CREATE_VM_ALWAYS_VALID, &h2) == 0) {
			struct drm_prime_handle p = {.handle = h2, .flags = DRM_CLOEXEC | DRM_RDWR};
			r                         = Ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &p);
			printf("  dma-buf export of an always-valid buffer: %s (expected EPERM)\n", Err(r));
			if (r == 0) {
				close(p.fd);
			}
			Close(h2);
		}
	}
out:
	if (a != NULL) {
		munmap(a, size);
	}
	if (h != 0) {
		Close(h);
	}
}

// ---------------------------------------------------------------------------------------------
// gfx-ring jobs

#define PKT3_CS(op, n)  ((3u << 30) | ((((n)-1u) & 0x3fffu) << 16) | ((op) << 8) | (1u << 1))
#define PKT3_GFX(op, n) ((3u << 30) | ((((n)-1u) & 0x3fffu) << 16) | ((op) << 8))
#define IT_WRITE_DATA      0x37u
#define IT_DMA_DATA        0x50u
#define IT_EVENT_WRITE     0x46u
#define IT_SET_SH_REG      0x76u
#define IT_DISPATCH_DIRECT 0x15u

typedef struct {
	uint32_t dw[1024];
	uint32_t n;
} Ib;

static void Push(Ib* ib, uint32_t v) {
	if (ib->n < ARRAY_SIZE(ib->dw)) {
		ib->dw[ib->n++] = v;
	}
}

static void PushWriteData(Ib* ib, uint64_t va, uint32_t value) {
	Push(ib, PKT3_GFX(IT_WRITE_DATA, 4));
	Push(ib, (5u << 8) | (1u << 20)); // DST_SEL memory, WR_CONFIRM, ENGINE ME
	Push(ib, (uint32_t)va);
	Push(ib, (uint32_t)(va >> 32));
	Push(ib, value);
}

// DMA_DATA: sel 0 = memory, 1 = GDS, 2 = immediate data (source only), 3 = memory via L2.
static void PushDma(Ib* ib, uint32_t src_sel, uint64_t src, uint32_t dst_sel, uint64_t dst, uint32_t bytes) {
	Push(ib, PKT3_GFX(IT_DMA_DATA, 6));
	Push(ib, (1u << 31) | (src_sel << 29) | (dst_sel << 20)); // CP_SYNC, ENGINE ME
	Push(ib, (uint32_t)src);
	Push(ib, (uint32_t)(src >> 32));
	Push(ib, (uint32_t)dst);
	Push(ib, (uint32_t)(dst >> 32));
	Push(ib, bytes | (1u << 30)); // RAW_WAIT
}

static void PushCsPartialFlush(Ib* ib) {
	Push(ib, PKT3_GFX(IT_EVENT_WRITE, 1));
	Push(ib, 7u | (4u << 8)); // CS_PARTIAL_FLUSH, EVENT_INDEX 4
}

static void PushSh(Ib* ib, uint32_t reg, const uint32_t* v, uint32_t n) {
	Push(ib, PKT3_CS(IT_SET_SH_REG, n + 1));
	Push(ib, (reg - 0xb000u) / 4u);
	for (uint32_t i = 0; i < n; i++) {
		Push(ib, v[i]);
	}
}

// One wave64 group of 64 lanes running code at code_va with user SGPRs ud[0..n).
static void PushDispatch(Ib* ib, uint64_t code_va, const uint32_t* ud, uint32_t n, uint32_t vgprs) {
	uint32_t pgm[2]  = {(uint32_t)(code_va >> 8), (uint32_t)(code_va >> 40)};
	uint32_t rsrc[2] = {((vgprs + 3) / 4 - 1) | (0xc0u << 12) | (1u << 21) | (1u << 30), (n & 0x1fu) << 1};
	uint32_t zero    = 0;
	uint32_t all[2]  = {0xffffffffu, 0xffffffffu};
	uint32_t start[3] = {0, 0, 0}, threads[3] = {64, 1, 1};
	PushSh(ib, 0xb830u, pgm, 2);
	PushSh(ib, 0xb848u, rsrc, 2);
	PushSh(ib, 0xb8a0u, &zero, 1);
	PushSh(ib, 0xb854u, &zero, 1);
	PushSh(ib, 0xb860u, &zero, 1);
	PushSh(ib, 0xb858u, all, 2);
	PushSh(ib, 0xb864u, all, 2);
	PushSh(ib, 0xb810u, start, 3);
	PushSh(ib, 0xb81cu, threads, 3);
	PushSh(ib, 0xb900u, ud, n);
	Push(ib, PKT3_CS(IT_DISPATCH_DIRECT, 4));
	Push(ib, 1);
	Push(ib, 1);
	Push(ib, 1);
	Push(ib, 1u | (1u << 2) | (1u << 6)); // COMPUTE_SHADER_EN, FORCE_START_AT_0, ORDER_MODE
}

static void Pad(Ib* ib) {
	while ((ib->n & 7u) != 0) {
		Push(ib, 0xffff1000u);
	}
}

typedef struct {
	uint32_t handle;
	uint8_t* cpu;
	uint64_t va;
	uint64_t size;
} Buf;

// An always-valid GTT buffer mapped at va for the GPU and anywhere for the CPU.
static bool MakeBuf(Buf* b, uint64_t size, uint64_t va, uint64_t extra_flags) {
	memset(b, 0, sizeof(*b));
	int r = Create(size, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_GEM_CREATE_VM_ALWAYS_VALID | extra_flags, &b->handle);
	if (r == 0) {
		r = Va(AMDGPU_VA_OP_MAP, b->handle, va, 0, size);
	}
	if (r == 0) {
		b->cpu = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, (off_t)MmapOffset(b->handle));
		r      = b->cpu == MAP_FAILED ? -errno : 0;
	}
	if (r != 0) {
		printf("  buffer at 0x%" PRIx64 ": %s\n", va, Err(r));
		return false;
	}
	b->va   = va;
	b->size = size;
	return true;
}

static void FreeBuf(Buf* b) {
	if (b->cpu != NULL && b->cpu != MAP_FAILED) {
		munmap(b->cpu, b->size);
	}
	if (b->handle != 0) {
		Va(AMDGPU_VA_OP_CLEAR, 0, b->va, 0, b->size);
		Close(b->handle);
	}
	memset(b, 0, sizeof(*b));
}

static Buf      g_ib;               // always-valid IB buffer
static bool     g_ib_rotate = false; // --ib-rotate: every job's IB at a new 4 KiB slot
static uint32_t g_ib_slot   = 0;

// Submits ib on the gfx ring. bo_list_handle != 0 uses that list; otherwise `handles` (may be
// empty: then no BO_HANDLES chunk at all). Returns the seq or 0 on error.
static uint64_t Submit(const Ib* ib, const uint32_t* handles, uint32_t n, uint32_t bo_list_handle, double* cs_sec) {
	const uint64_t slot = g_ib_rotate ? (uint64_t)(g_ib_slot++ % (g_ib.size / (4 * KiB))) * 4 * KiB : 0;
	memcpy(g_ib.cpu + slot, ib->dw, ib->n * 4u);
	__sync_synchronize();
	struct drm_amdgpu_cs_chunk_ib ib_info;
	memset(&ib_info, 0, sizeof(ib_info));
	ib_info.va_start = g_ib.va + slot;
	ib_info.ib_bytes = ib->n * 4u;
	ib_info.ip_type  = AMDGPU_HW_IP_GFX;
	struct drm_amdgpu_bo_list_entry* entries = NULL;
	struct drm_amdgpu_bo_list_in     list;
	struct drm_amdgpu_cs_chunk       chunks[2];
	uint32_t                         nchunks = 1;
	chunks[0].chunk_id                       = AMDGPU_CHUNK_ID_IB;
	chunks[0].length_dw                      = sizeof(ib_info) / 4;
	chunks[0].chunk_data                     = (uintptr_t)&ib_info;
	if (n > 0) {
		entries = calloc(n, sizeof(*entries));
		for (uint32_t i = 0; i < n; i++) {
			entries[i].bo_handle = handles[i];
		}
		memset(&list, 0, sizeof(list));
		list.operation       = ~0u;
		list.list_handle     = ~0u;
		list.bo_number       = n;
		list.bo_info_size    = sizeof(entries[0]);
		list.bo_info_ptr     = (uintptr_t)entries;
		chunks[1].chunk_id   = AMDGPU_CHUNK_ID_BO_HANDLES;
		chunks[1].length_dw  = sizeof(list) / 4;
		chunks[1].chunk_data = (uintptr_t)&list;
		nchunks              = 2;
	}
	uint64_t ptrs[2] = {(uintptr_t)&chunks[0], (uintptr_t)&chunks[1]};
	union drm_amdgpu_cs cs;
	memset(&cs, 0, sizeof(cs));
	cs.in.ctx_id         = g_ctx;
	cs.in.bo_list_handle = bo_list_handle;
	cs.in.num_chunks     = nchunks;
	cs.in.chunks         = (uintptr_t)ptrs;
	double t0            = NowSec();
	int    r             = Ioctl(DRM_IOCTL_AMDGPU_CS, &cs);
	if (cs_sec != NULL) {
		*cs_sec = NowSec() - t0;
	}
	free(entries);
	if (r != 0) {
		printf("  CS: %s\n", Err(r));
		return 0;
	}
	return cs.out.handle;
}

static bool Wait(uint64_t seq, double timeout_sec) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	union drm_amdgpu_wait_cs w;
	memset(&w, 0, sizeof(w));
	w.in.handle  = seq;
	w.in.timeout = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec + (uint64_t)(timeout_sec * 1e9);
	w.in.ip_type = AMDGPU_HW_IP_GFX;
	w.in.ctx_id  = g_ctx;
	int r        = Ioctl(DRM_IOCTL_AMDGPU_WAIT_CS, &w);
	if (r != 0 || w.out.status != 0) {
		printf("  wait: %s\n", r != 0 ? Err(r) : "TIMEOUT");
		return false;
	}
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op     = AMDGPU_CTX_OP_QUERY_STATE2;
	ctx.in.ctx_id = g_ctx;
	if (Ioctl(DRM_IOCTL_AMDGPU_CTX, &ctx) == 0 && ctx.out.state.flags != 0) {
		printf("  context state 0x%" PRIx64 " after the job\n", (uint64_t)ctx.out.state.flags);
		return false;
	}
	return true;
}

static int BoListCreate(const uint32_t* handles, uint32_t n, uint32_t* list_handle) {
	struct drm_amdgpu_bo_list_entry* e = calloc(n, sizeof(*e));
	for (uint32_t i = 0; i < n; i++) {
		e[i].bo_handle = handles[i];
	}
	union drm_amdgpu_bo_list l;
	memset(&l, 0, sizeof(l));
	l.in.operation    = AMDGPU_BO_LIST_OP_CREATE;
	l.in.bo_number    = n;
	l.in.bo_info_size = sizeof(e[0]);
	l.in.bo_info_ptr  = (uintptr_t)e;
	int r             = Ioctl(DRM_IOCTL_AMDGPU_BO_LIST, &l);
	free(e);
	*list_handle = r == 0 ? l.out.list_handle : 0;
	return r;
}

static void ProbeCs(void) {
	printf("== cs: gfx-ring WRITE_DATA into an always-valid buffer, no BO list\n");
	Buf data;
	if (!MakeBuf(&data, 2 * MiB, 0x0300000000ull, 0)) {
		return;
	}
	memset(data.cpu, 0, 4096);
	Ib ib = {.n = 0};
	for (uint32_t i = 0; i < 4; i++) {
		PushWriteData(&ib, data.va + i * 4, 0xc0de0000u + i);
	}
	Pad(&ib);
	double   cs  = 0;
	uint64_t seq = Submit(&ib, NULL, 0, 0, &cs);
	bool     ok  = seq != 0 && Wait(seq, 3.0);
	uint32_t* p  = (uint32_t*)data.cpu;
	printf("  job %s (CS ioctl %.1f us), CPU reads 0x%08x 0x%08x 0x%08x 0x%08x: %s\n", ok ? "done" : "FAILED", cs * 1e6,
	       p[0], p[1], p[2], p[3], p[0] == 0xc0de0000u && p[3] == 0xc0de0003u ? "correct" : "WRONG");
	FreeBuf(&data);
}

static void ProbeCsBench(uint32_t list_size, uint64_t resident_gib) {
	printf("== csbench: CS ioctl cost with %" PRIu64 " GiB of always-valid memory mapped\n", resident_gib);
	Buf      data;
	if (!MakeBuf(&data, 2 * MiB, 0x0300000000ull, 0)) {
		return;
	}
	// Big always-valid buffers: does their mere presence cost per CS?
	uint32_t big[64];
	uint32_t nbig = 0;
	for (uint64_t i = 0; i < resident_gib && nbig < 64; i++) {
		if (Create(GiB, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_GEM_CREATE_VM_ALWAYS_VALID, &big[nbig]) != 0) {
			break;
		}
		Va(AMDGPU_VA_OP_MAP, big[nbig], 0x4000000000ull + i * GiB, 0, GiB);
		nbig++;
	}
	Mem("with the big buffers");
	// Small plain buffers for the per-CS list case, like the 16 MiB residency blocks today.
	uint32_t* small = calloc(list_size, sizeof(uint32_t));
	for (uint32_t i = 0; i < list_size; i++) {
		Create(64 * KiB, AMDGPU_GEM_DOMAIN_GTT, 0, &small[i]);
		Va(AMDGPU_VA_OP_MAP, small[i], 0x6000000000ull + (uint64_t)i * 64 * KiB, 0, 64 * KiB);
	}
	uint32_t list_handle = 0;
	BoListCreate(small, 3, &list_handle);
	Ib ib = {.n = 0};
	PushWriteData(&ib, data.va, 1);
	Pad(&ib);
	const int iters = 2000;
	const struct {
		const char* name;
		uint32_t    n;
		uint32_t    list;
	} cases[] = {{"no BO list", 0, 0}, {"persistent 3-entry BO list handle", 0, 1}, {"per-CS list of 3", 3, 0},
	             {"per-CS list of N (N = --list)", list_size, 0}};
	for (size_t k = 0; k < ARRAY_SIZE(cases); k++) {
		double   total = 0, worst = 0;
		uint64_t seq   = 0;
		for (int i = 0; i < iters; i++) {
			double cs = 0;
			seq       = Submit(&ib, small, cases[k].n, cases[k].list ? list_handle : 0, &cs);
			if (seq == 0) {
				break;
			}
			total += cs;
			worst = cs > worst ? cs : worst;
			if ((i & 63) == 63) {
				Wait(seq, 3.0);
			}
		}
		bool ok = seq != 0 && Wait(seq, 3.0);
		printf("  %-36s N=%-4u: %6.1f us per CS ioctl (worst %6.1f us)%s\n", cases[k].name,
		       cases[k].n ? cases[k].n : (cases[k].list ? 3 : 0), total / iters * 1e6, worst * 1e6, ok ? "" : " FAILED");
	}
	for (uint32_t i = 0; i < list_size; i++) {
		Close(small[i]);
	}
	free(small);
	for (uint32_t i = 0; i < nbig; i++) {
		Close(big[i]);
	}
	FreeBuf(&data);
}

// ---------------------------------------------------------------------------------------------
// GDS

static void GdsInfo(void) {
	struct drm_amdgpu_info_gds gds;
	memset(&gds, 0, sizeof(gds));
	struct drm_amdgpu_info req;
	memset(&req, 0, sizeof(req));
	req.return_pointer = (uintptr_t)&gds;
	req.return_size    = sizeof(gds);
	req.query          = AMDGPU_INFO_GDS_CONFIG;
	int r              = Ioctl(DRM_IOCTL_AMDGPU_INFO, &req);
	printf("  GDS config (%s): gfx partition %u, compute partition %u, total %u bytes; GWS %u/%u; OA %u/%u\n", Err(r),
	       gds.gds_gfx_partition_size, gds.compute_partition_size, gds.gds_total_size, gds.gws_per_gfx_partition,
	       gds.gws_per_compute_partition, gds.oa_per_gfx_partition, gds.oa_per_compute_partition);
}

static uint32_t g_gds_total = 0, g_oa_total = 0;

static void GdsSizes(void) {
	struct drm_amdgpu_info_gds gds;
	memset(&gds, 0, sizeof(gds));
	struct drm_amdgpu_info req;
	memset(&req, 0, sizeof(req));
	req.return_pointer = (uintptr_t)&gds;
	req.return_size    = sizeof(gds);
	req.query          = AMDGPU_INFO_GDS_CONFIG;
	Ioctl(DRM_IOCTL_AMDGPU_INFO, &req);
	g_gds_total = gds.gds_total_size > gds.compute_partition_size ? gds.gds_total_size : gds.compute_partition_size;
	if (gds.gds_gfx_partition_size > g_gds_total) {
		g_gds_total = gds.gds_gfx_partition_size;
	}
	g_oa_total = gds.oa_per_compute_partition > gds.oa_per_gfx_partition ? gds.oa_per_compute_partition
	                                                                       : gds.oa_per_gfx_partition;
}

// Job: DMA X, X+1 into GDS dwords 0/1, then the shader reads them (relative to the job's partition)
// and writes Y to GDS dword 2, then DMA copies GDS dwords 2 and 0 out. The job's partition is that
// of the GDS buffer in `gds_handles`.
static bool GdsJob(Buf* out, Buf* code, uint32_t out_off, const uint32_t* gds_handles, uint32_t ngds, uint32_t x,
                   uint32_t y, bool write) {
	Ib ib = {.n = 0};
	if (write) {
		PushDma(&ib, 2, x, 1, 0, 4);
		PushDma(&ib, 2, x + 1, 1, 4, 4);
	}
	uint32_t ud[3] = {(uint32_t)(out->va + out_off), (uint32_t)((out->va + out_off) >> 32), y};
	PushDispatch(&ib, code->va, ud, 3, 8);
	PushCsPartialFlush(&ib);
	PushDma(&ib, 1, 8, 3, out->va + out_off + 0x200, 4);
	PushDma(&ib, 1, 0, 3, out->va + out_off + 0x204, 4);
	PushWriteData(&ib, out->va + out_off + 0x208, y ^ 0xffffffffu); // proves this IB ran
	Pad(&ib);
	uint64_t seq = Submit(&ib, gds_handles, ngds, 0, NULL);
	return seq != 0 && Wait(seq, 5.0);
}

static bool MakeCode(Buf* code) {
	if (!MakeBuf(code, 64 * KiB, 0x0310000000ull, 0)) {
		return false;
	}
	memset(code->cpu, 0, 64 * KiB);
	memcpy(code->cpu, k_gdsaddr_w64, sizeof(k_gdsaddr_w64));
	__sync_synchronize();
	return true;
}

static void ProbeGds(void) {
	printf("== gds: DMA_DATA GDS addressing on the gfx ring, relative to the job's partition or absolute\n");
	GdsInfo();
	uint32_t a = 0, b = 0;
	int      ra = Create(4 * KiB, AMDGPU_GEM_DOMAIN_GDS, 0, &a);
	int      rb = Create(4 * KiB, AMDGPU_GEM_DOMAIN_GDS, 0, &b);
	printf("  GDS buffers A, B of 4 KiB (B is placed above A): %s, %s\n", Err(ra), Err(rb));
	Buf out, code;
	if (ra != 0 || rb != 0 || !MakeBuf(&out, 64 * KiB, 0x0300000000ull, 0) || !MakeCode(&code)) {
		return;
	}
	memset(out.cpu, 0xee, 64 * KiB);
	const uint32_t x = 0x5a5a0000u ^ (uint32_t)getpid(), y = 0xb0b00000u ^ (uint32_t)getpid();
	// Clear A's first dwords through A's own partition first, so a stale value cannot fake a match.
	bool ok = GdsJob(&out, &code, 0x2000, &a, 1, 0x11110000u, 0x22220000u, true);
	printf("  job 0 (A: init A)                     : %s\n", ok ? "done" : "FAILED");
	ok = ok && GdsJob(&out, &code, 0x0000, &b, 1, x, y, true);
	uint32_t* o = (uint32_t*)out.cpu;
	printf("  job 1 (B: DMA X to 0/4, shader reads) : %s\n", ok ? "done" : "FAILED");
	if (!ok) {
		return;
	}
	printf("    shader read GDS[0] = 0x%08x, GDS[4] = 0x%08x   (X = 0x%08x)\n", o[0], o[1], x);
	printf("    DMA read  GDS[8] = 0x%08x (shader wrote Y = 0x%08x), GDS[0] = 0x%08x\n", o[0x200 / 4], y, o[0x204 / 4]);
	ok = GdsJob(&out, &code, 0x1000, &a, 1, 0, 0x33330000u, false);
	printf("  job 2 (A: no DMA writes, shader reads): %s\n", ok ? "done" : "FAILED");
	printf("    shader read A[0] = 0x%08x; DMA read GDS[0] in A's job = 0x%08x\n", o[0x1000 / 4], o[0x1204 / 4]);
	const bool shader_sees_dma = o[0] == x && o[1] == x + 1;
	const bool dma_sees_shader = o[0x200 / 4] == y;
	const bool a_got_x         = o[0x1000 / 4] == x;
	printf("  => DMA_DATA GDS addressing is %s\n",
	       shader_sees_dma && dma_sees_shader && !a_got_x
	           ? "RELATIVE to the job's GDS partition (like shader GDS ops)"
	           : (!shader_sees_dma && a_got_x ? "ABSOLUTE (ignores the partition base)" : "UNCLEAR, see values"));
	FreeBuf(&out);
	FreeBuf(&code);
	Close(a);
	Close(b);
}

// shader_first: only job 0 starts with a DMA write; every later job starts with the GDS shader.
static void ProbeGdsFull(int jobs, bool shader_first) {
	printf("== gdsfull: the whole GDS and OA in every job, %d GDS jobs, %s\n", jobs,
	       shader_first ? "later jobs start with the GDS shader" : "every job starts with a GDS DMA write");
	GdsSizes();
	uint32_t gds = 0, oa = 0;
	int      rg = Create(g_gds_total, AMDGPU_GEM_DOMAIN_GDS, 0, &gds);
	int      ro = Create(g_oa_total, AMDGPU_GEM_DOMAIN_OA, 0, &oa);
	printf("  GDS %u bytes: %s, OA %u: %s\n", g_gds_total, Err(rg), g_oa_total, Err(ro));
	Buf out, code;
	if (rg != 0 || !MakeBuf(&out, 64 * KiB, 0x0300000000ull, 0) || !MakeCode(&code)) {
		return;
	}
	uint32_t handles[2] = {gds, oa};
	uint32_t n          = ro == 0 ? 2 : 1;
	double   t0         = NowSec();
	int      good       = 0;
	for (int i = 0; i < jobs; i++) {
		const bool     write = !shader_first || i == 0;
		const uint32_t x = write ? 0x1000u * (uint32_t)i : 0, y = 0x1000u * (uint32_t)i + 7;
		if (!GdsJob(&out, &code, 0, handles, n, x, y, write)) {
			printf("  job %d failed\n", i);
			break;
		}
		uint32_t* o = (uint32_t*)out.cpu;
		if (o[0] != x || o[0x200 / 4] != y || o[0x208 / 4] != (y ^ 0xffffffffu)) {
			printf("  job %d: shader read 0x%08x (want 0x%08x), DMA read 0x%08x (want 0x%08x), IB marker %s\n", i,
			       o[0], x, o[0x200 / 4], y, o[0x208 / 4] == (y ^ 0xffffffffu) ? "ok" : "STALE");
			break;
		}
		good++;
	}
	printf("  %d/%d jobs correct, %.2f ms per job\n", good, jobs, (NowSec() - t0) * 1e3 / (jobs > 0 ? jobs : 1));
	FreeBuf(&out);
	FreeBuf(&code);
	if (ro == 0) {
		Close(oa);
	}
	Close(gds);
}

int main(int argc, char** argv) {
	uint64_t alloc_mib = 1024, resident_gib = 4;
	uint32_t list      = 600;
	int      jobs      = 1000;
	setvbuf(stdout, NULL, _IOLBF, 0);
	g_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	if (g_fd < 0) {
		perror("open");
		return 1;
	}
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	if (Ioctl(DRM_IOCTL_AMDGPU_CTX, &ctx) != 0) {
		perror("ctx");
		return 1;
	}
	g_ctx = ctx.out.alloc.ctx_id;
	uint64_t ib_flags = AMDGPU_GEM_CREATE_CPU_GTT_USWC;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--ib-uncached") == 0) {
			ib_flags |= AMDGPU_GEM_CREATE_UNCACHED;
		} else if (strcmp(argv[i], "--ib-rotate") == 0) {
			g_ib_rotate = true;
		}
	}
	if (!MakeBuf(&g_ib, 256 * KiB, 0x400000000000ull, ib_flags)) {
		return 1;
	}
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--ib-uncached") == 0 || strcmp(argv[i], "--ib-rotate") == 0) {
			continue;
		} else if (strcmp(argv[i], "--alloc-mib") == 0 && i + 1 < argc) {
			alloc_mib = strtoull(argv[++i], NULL, 0);
		} else if (strcmp(argv[i], "--list") == 0 && i + 1 < argc) {
			list = (uint32_t)strtoul(argv[++i], NULL, 0);
		} else if (strcmp(argv[i], "--resident-gib") == 0 && i + 1 < argc) {
			resident_gib = strtoull(argv[++i], NULL, 0);
		} else if (strcmp(argv[i], "--jobs") == 0 && i + 1 < argc) {
			jobs = atoi(argv[++i]);
		} else if (strcmp(argv[i], "alloc") == 0) {
			ProbeAlloc(alloc_mib * MiB);
		} else if (strcmp(argv[i], "views") == 0) {
			ProbeViews();
		} else if (strcmp(argv[i], "cs") == 0) {
			ProbeCs();
		} else if (strcmp(argv[i], "csbench") == 0) {
			ProbeCsBench(list, resident_gib);
		} else if (strcmp(argv[i], "gds") == 0) {
			ProbeGds();
		} else if (strcmp(argv[i], "gdsfull") == 0) {
			ProbeGdsFull(jobs, false);
		} else if (strcmp(argv[i], "gdsfull-shader-first") == 0) {
			ProbeGdsFull(jobs, true);
		} else {
			fprintf(stderr, "unknown probe %s\n", argv[i]);
			return 2;
		}
	}
	FreeBuf(&g_ib);
	return 0;
}
