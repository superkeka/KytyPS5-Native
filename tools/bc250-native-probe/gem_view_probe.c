// Can guest direct memory be real amdgpu GEM buffers instead of userptr slices of a memfd?
//
// A guest maps arbitrary (offset, size) views of direct memory at arbitrary addresses, the same
// offset possibly several times. mmap of a GEM buffer through the DRM file only maps a buffer whole
// and from its start, but a dma-buf export of it takes an inner page offset. This probe checks, on
// one 256 MiB GTT buffer:
//   1. two CPU views of the same inner range (dma-buf mmap at offset 64 MiB) see each other's writes,
//      and so does a whole-buffer view;
//   2. mprotect on a view works (guests reprotect their memory);
//   3. the GPU, with the buffer mapped into its VM at a view address and listed in the CS, writes
//      through WRITE_DATA and the CPU views see it;
//   4. how long a CS listing the buffer takes, against the userptr chunks' ~31 ms per frame.
// Build: cc -O2 -o gem_view_probe gem_view_probe.c (needs the amdgpu uAPI headers).

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

#define MIB (1024ull * 1024ull)

static int g_fd = -1;

static int Ioctl(unsigned long req, void* arg) {
	int r;
	do {
		r = ioctl(g_fd, req, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

static double Now(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int Fail(const char* what, int r) {
	printf("FAIL %s: %s\n", what, strerror(-r));
	return 1;
}

static int GemCreate(uint64_t size, uint32_t domain, uint32_t* handle) {
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = 2 * MIB;
	c.in.domains      = domain;
	int r             = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
	*handle           = c.out.handle;
	return r;
}

static int VaMap(uint32_t handle, uint64_t va, uint64_t offset, uint64_t size) {
	struct drm_amdgpu_gem_va v;
	memset(&v, 0, sizeof(v));
	v.handle       = handle;
	v.operation    = AMDGPU_VA_OP_MAP;
	v.flags        = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
	v.va_address   = va;
	v.offset_in_bo = offset;
	v.map_size     = size;
	return Ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &v);
}

int main(void) {
	g_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	if (g_fd < 0) {
		printf("FAIL open: %s\n", strerror(errno));
		return 1;
	}
	const uint64_t kBo = 256 * MIB, kInner = 64 * MIB, kView = 32 * MIB;

	uint32_t bo = 0;
	int      r  = GemCreate(kBo, AMDGPU_GEM_DOMAIN_GTT, &bo);
	if (r != 0) {
		return Fail("GEM_CREATE 256 MiB GTT", r);
	}
	struct drm_prime_handle p;
	memset(&p, 0, sizeof(p));
	p.handle = bo;
	p.flags  = DRM_CLOEXEC | DRM_RDWR;
	if ((r = Ioctl(DRM_IOCTL_PRIME_HANDLE_TO_FD, &p)) != 0) {
		return Fail("PRIME_HANDLE_TO_FD", r);
	}
	const int dmabuf = p.fd;

	// A guest-like address range: reserve, then place views with MAP_FIXED.
	uint8_t* base = mmap(NULL, 1024 * MIB, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	if (base == MAP_FAILED) {
		return Fail("reserve", -errno);
	}
	uint8_t* a     = base;               // view of [64 MiB, 96 MiB)
	uint8_t* b     = base + 256 * MIB;   // second view of the same range
	uint8_t* whole = base + 512 * MIB;   // the whole buffer
	if (mmap(a, kView, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, dmabuf, (off_t)kInner) != a) {
		return Fail("dma-buf mmap at inner offset (view a)", -errno);
	}
	if (mmap(b, kView, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, dmabuf, (off_t)kInner) != b) {
		return Fail("dma-buf mmap at inner offset (view b)", -errno);
	}
	if (mmap(whole, kBo, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, dmabuf, 0) != whole) {
		return Fail("dma-buf mmap whole", -errno);
	}
	printf("ok   dma-buf views mapped at inner offset 64 MiB (twice) and whole\n");

	a[123] = 0x5a;
	b[4 * MIB + 7] = 0xa5;
	const int alias_ok = b[123] == 0x5a && a[4 * MIB + 7] == 0xa5 && whole[kInner + 123] == 0x5a &&
	                     whole[kInner + 4 * MIB + 7] == 0xa5;
	printf("%s CPU aliasing between the views\n", alias_ok ? "ok  " : "FAIL");

	if (mprotect(b, kView, PROT_READ) != 0) {
		return Fail("mprotect view b read-only", -errno);
	}
	if (mprotect(b, kView, PROT_READ | PROT_WRITE) != 0) {
		return Fail("mprotect view b back to rw", -errno);
	}
	printf("ok   mprotect on a view\n");

	// GPU: map view a's backing at view a's address, as GemBacking::BindView would.
	if ((r = VaMap(bo, (uint64_t)(uintptr_t)a, kInner, kView)) != 0) {
		return Fail("GEM_VA map at view address", r);
	}
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	if ((r = Ioctl(DRM_IOCTL_AMDGPU_CTX, &ctx)) != 0) {
		return Fail("CTX alloc", r);
	}
	uint32_t ib_bo = 0;
	if ((r = GemCreate(2 * MIB, AMDGPU_GEM_DOMAIN_GTT, &ib_bo)) != 0) {
		return Fail("IB GEM_CREATE", r);
	}
	const uint64_t ib_va = 0x400000000000ull;
	if ((r = VaMap(ib_bo, ib_va, 0, 2 * MIB)) != 0) {
		return Fail("IB GEM_VA", r);
	}
	union drm_amdgpu_gem_mmap gm;
	memset(&gm, 0, sizeof(gm));
	gm.in.handle = ib_bo;
	if ((r = Ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &gm)) != 0) {
		return Fail("IB GEM_MMAP", r);
	}
	uint32_t* ib = mmap(NULL, 2 * MIB, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, (off_t)gm.out.addr_ptr);
	if (ib == MAP_FAILED) {
		return Fail("IB mmap", -errno);
	}

	double best = 1e9, worst = 0;
	int    gpu_ok = 1;
	for (int it = 0; it < 20; it++) {
		const uint64_t dst   = (uint64_t)(uintptr_t)(a + 1 * MIB) + (uint64_t)it * 4;
		const uint32_t value = 0xC0DE0000u + (uint32_t)it;
		uint32_t       n     = 0;
		ib[n++] = 0xC0033700u; // PKT3 WRITE_DATA, 3 + 1 dwords
		ib[n++] = (5u << 8) | (1u << 20); // DST_SEL memory, WR_CONFIRM
		ib[n++] = (uint32_t)dst;
		ib[n++] = (uint32_t)(dst >> 32);
		ib[n++] = value;
		while (n & 7u) {
			ib[n++] = 0xffff1000u;
		}
		__sync_synchronize();

		struct drm_amdgpu_bo_list_entry entries[2];
		memset(entries, 0, sizeof(entries));
		entries[0].bo_handle = ib_bo;
		entries[1].bo_handle = bo;
		struct drm_amdgpu_bo_list_in list;
		memset(&list, 0, sizeof(list));
		list.operation    = ~0u;
		list.list_handle  = ~0u;
		list.bo_number    = 2;
		list.bo_info_size = sizeof(entries[0]);
		list.bo_info_ptr  = (uintptr_t)entries;
		struct drm_amdgpu_cs_chunk_ib ibi;
		memset(&ibi, 0, sizeof(ibi));
		ibi.va_start = ib_va;
		ibi.ib_bytes = n * 4;
		ibi.ip_type  = AMDGPU_HW_IP_GFX;
		struct drm_amdgpu_cs_chunk chunks[2];
		memset(chunks, 0, sizeof(chunks));
		chunks[0].chunk_id   = AMDGPU_CHUNK_ID_IB;
		chunks[0].length_dw  = sizeof(ibi) / 4;
		chunks[0].chunk_data = (uintptr_t)&ibi;
		chunks[1].chunk_id   = AMDGPU_CHUNK_ID_BO_HANDLES;
		chunks[1].length_dw  = sizeof(list) / 4;
		chunks[1].chunk_data = (uintptr_t)&list;
		uint64_t        ptrs[2] = {(uintptr_t)&chunks[0], (uintptr_t)&chunks[1]};
		union drm_amdgpu_cs cs;
		memset(&cs, 0, sizeof(cs));
		cs.in.ctx_id     = ctx.out.alloc.ctx_id;
		cs.in.num_chunks = 2;
		cs.in.chunks     = (uintptr_t)ptrs;
		const double t0  = Now();
		if ((r = Ioctl(DRM_IOCTL_AMDGPU_CS, &cs)) != 0) {
			return Fail("CS", r);
		}
		const double dt = Now() - t0;
		best            = dt < best ? dt : best;
		worst           = dt > worst ? dt : worst;

		union drm_amdgpu_wait_cs w;
		memset(&w, 0, sizeof(w));
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		w.in.handle  = cs.out.handle;
		w.in.timeout = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec + 2000000000ull;
		w.in.ip_type = AMDGPU_HW_IP_GFX;
		w.in.ctx_id  = ctx.out.alloc.ctx_id;
		if ((r = Ioctl(DRM_IOCTL_AMDGPU_WAIT_CS, &w)) != 0 || w.out.status != 0) {
			return Fail("WAIT_CS (timeout or error)", r != 0 ? r : -ETIMEDOUT);
		}
		const uint32_t* via_a = (const uint32_t*)(uintptr_t)dst;
		const uint32_t* via_b = (const uint32_t*)(b + 1 * MIB + (uint64_t)it * 4);
		const uint32_t* via_w = (const uint32_t*)(whole + kInner + 1 * MIB + (uint64_t)it * 4);
		if (*via_a != value || *via_b != value || *via_w != value) {
			printf("FAIL GPU write %d: a 0x%08x b 0x%08x whole 0x%08x, want 0x%08x\n", it, *via_a, *via_b,
			       *via_w, value);
			gpu_ok = 0;
		}
	}
	printf("%s GPU WRITE_DATA through the VM mapping visible in all CPU views (20 submits)\n",
	       gpu_ok ? "ok  " : "FAIL");
	printf("     CS ioctl with the 256 MiB GEM buffer listed: best %.3f ms, worst %.3f ms\n", best * 1e3,
	       worst * 1e3);

	union drm_amdgpu_ctx fr;
	memset(&fr, 0, sizeof(fr));
	fr.in.op     = AMDGPU_CTX_OP_FREE_CTX;
	fr.in.ctx_id = ctx.out.alloc.ctx_id;
	Ioctl(DRM_IOCTL_AMDGPU_CTX, &fr);
	return alias_ok && gpu_ok ? 0 : 1;
}
