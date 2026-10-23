// Could guest direct memory stay an ordinary memfd while the GPU gets only the blocks a game uses,
// without copying? Each block would be a udmabuf over the memfd pages, imported into amdgpu and
// mapped into the GPU VM at the guest address.
//
// Result on the BC-250 (2026-10-23): import and GPU->CPU work, and a CS with 64 imported blocks
// listed costs 0.056 ms, but the GPU does not snoop the CPU cache on these pages: a CPU write still
// in a dirty cache line is invisible to the GPU (correct only after clflush). Imported foreign
// dma-bufs are not usable as coherent guest memory; amdgpu's own GTT buffers are.
// An earlier version also mapped unused blocks AMDGPU_VM_PAGE_PRT: a GPU access there hung the job
// and took the board down. Do not use PRT on this board.
// Build: cc -O2 -o udmabuf_probe udmabuf_probe.c (amdgpu uAPI headers). /dev/udmabuf must be openable.

#define _GNU_SOURCE
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
#include <linux/udmabuf.h>

#define MIB (1024ull * 1024ull)

static int g_fd = -1;
static uint32_t g_ctx, g_ib_bo;
static uint32_t* g_ib;
static const uint64_t kIbVa = 0x400000000000ull;

static int Ioctl(int fd, unsigned long req, void* arg) {
	int r;
	do {
		r = ioctl(fd, req, arg);
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

static int VaOp(uint32_t op, uint32_t handle, uint64_t va, uint64_t offset, uint64_t size, uint32_t flags) {
	struct drm_amdgpu_gem_va v;
	memset(&v, 0, sizeof(v));
	v.handle       = handle;
	v.operation    = op;
	v.flags        = flags;
	v.va_address   = va;
	v.offset_in_bo = offset;
	v.map_size     = size;
	return Ioctl(g_fd, DRM_IOCTL_AMDGPU_GEM_VA, &v);
}

#define RWX (AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE)

// Submits `n` dwords already in g_ib with `handles` listed, waits, returns 0 or -errno; *cs_ms gets the
// CS ioctl time.
static int Submit(uint32_t n, const uint32_t* handles, uint32_t count, double* cs_ms) {
	while (n & 7u) {
		g_ib[n++] = 0xffff1000u;
	}
	__sync_synchronize();
	struct drm_amdgpu_bo_list_entry entries[80];
	memset(entries, 0, sizeof(entries));
	entries[0].bo_handle = g_ib_bo;
	for (uint32_t i = 0; i < count; i++) {
		entries[1 + i].bo_handle = handles[i];
	}
	struct drm_amdgpu_bo_list_in list;
	memset(&list, 0, sizeof(list));
	list.operation    = ~0u;
	list.list_handle  = ~0u;
	list.bo_number    = count + 1;
	list.bo_info_size = sizeof(entries[0]);
	list.bo_info_ptr  = (uintptr_t)entries;
	struct drm_amdgpu_cs_chunk_ib ibi;
	memset(&ibi, 0, sizeof(ibi));
	ibi.va_start = kIbVa;
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
	cs.in.ctx_id     = g_ctx;
	cs.in.num_chunks = 2;
	cs.in.chunks     = (uintptr_t)ptrs;
	const double t0  = Now();
	int          r   = Ioctl(g_fd, DRM_IOCTL_AMDGPU_CS, &cs);
	if (cs_ms != NULL) {
		*cs_ms = (Now() - t0) * 1e3;
	}
	if (r != 0) {
		return r;
	}
	union drm_amdgpu_wait_cs w;
	memset(&w, 0, sizeof(w));
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	w.in.handle  = cs.out.handle;
	w.in.timeout = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec + 3000000000ull;
	w.in.ip_type = AMDGPU_HW_IP_GFX;
	w.in.ctx_id  = g_ctx;
	r            = Ioctl(g_fd, DRM_IOCTL_AMDGPU_WAIT_CS, &w);
	if (r != 0) {
		return r;
	}
	return w.out.status != 0 ? -ETIMEDOUT : 0;
}

static uint32_t WriteData(uint32_t n, uint64_t dst, uint32_t value) {
	g_ib[n++] = 0xC0033700u; // WRITE_DATA, 4 body dwords
	g_ib[n++] = (5u << 8) | (1u << 20);
	g_ib[n++] = (uint32_t)dst;
	g_ib[n++] = (uint32_t)(dst >> 32);
	g_ib[n++] = value;
	return n;
}

static uint32_t CopyData(uint32_t n, uint64_t src, uint64_t dst) {
	g_ib[n++] = 0xC0044000u; // COPY_DATA, 5 body dwords
	g_ib[n++] = 2u | (5u << 8) | (1u << 20); // src TC/L2, dst memory, wr_confirm
	g_ib[n++] = (uint32_t)src;
	g_ib[n++] = (uint32_t)(src >> 32);
	g_ib[n++] = (uint32_t)dst;
	g_ib[n++] = (uint32_t)(dst >> 32);
	return n;
}

int main(void) {
	const uint64_t kSize = 256 * MIB, kBlock = 16 * MIB;
	setvbuf(stdout, NULL, _IOLBF, 0);
	g_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	int ufd = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
	if (g_fd < 0 || ufd < 0) {
		printf("FAIL open renderD128/udmabuf: %s\n", strerror(errno));
		return 1;
	}
	int mem = memfd_create("probe-direct", MFD_ALLOW_SEALING | MFD_CLOEXEC);
	if (mem < 0 || ftruncate(mem, (off_t)kSize) != 0 || fcntl(mem, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
		return Fail("memfd + F_SEAL_SHRINK", -errno);
	}
	uint8_t* alias = mmap(NULL, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, mem, 0);
	// A guest-like view at a fixed address, like a direct-memory mapping.
	uint8_t* view = mmap((void*)0x900000000ull, kSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, mem, 0);
	if (alias == MAP_FAILED || view == MAP_FAILED) {
		return Fail("alias/view mmap", -errno);
	}

	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	int r     = Ioctl(g_fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
	if (r != 0) {
		return Fail("CTX", r);
	}
	g_ctx = ctx.out.alloc.ctx_id;
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size   = 2 * MIB;
	c.in.alignment = 4096;
	c.in.domains   = AMDGPU_GEM_DOMAIN_GTT;
	if ((r = Ioctl(g_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c)) != 0) {
		return Fail("IB create", r);
	}
	g_ib_bo = c.out.handle;
	if ((r = VaOp(AMDGPU_VA_OP_MAP, g_ib_bo, kIbVa, 0, 2 * MIB, RWX)) != 0) {
		return Fail("IB VA", r);
	}
	union drm_amdgpu_gem_mmap gm;
	memset(&gm, 0, sizeof(gm));
	gm.in.handle = g_ib_bo;
	if ((r = Ioctl(g_fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &gm)) != 0) {
		return Fail("IB GEM_MMAP", r);
	}
	g_ib = mmap(NULL, 2 * MIB, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, (off_t)gm.out.addr_ptr);
	if (g_ib == MAP_FAILED) {
		return Fail("IB mmap", -errno);
	}
	printf("ok   setup\n");

	// Blocks 0..11 of the view mapped to imported udmabufs; 12..15 stay unmapped on the GPU (never touched).
	uint32_t handles[12];
	double   t_import = Now();
	for (int b = 0; b < 12; b++) {
		struct udmabuf_create u;
		memset(&u, 0, sizeof(u));
		u.memfd  = (uint32_t)mem;
		u.flags  = UDMABUF_FLAGS_CLOEXEC;
		u.offset = (uint64_t)b * kBlock;
		u.size   = kBlock;
		int dbuf = ioctl(ufd, UDMABUF_CREATE, &u);
		if (dbuf < 0) {
			return Fail("UDMABUF_CREATE 16 MiB", -errno);
		}
		struct drm_prime_handle p;
		memset(&p, 0, sizeof(p));
		p.fd = dbuf;
		if ((r = Ioctl(g_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &p)) != 0) {
			return Fail("PRIME_FD_TO_HANDLE (amdgpu import of udmabuf)", r);
		}
		close(dbuf);
		handles[b] = p.handle;
		if ((r = VaOp(AMDGPU_VA_OP_REPLACE, p.handle, (uint64_t)(uintptr_t)view + (uint64_t)b * kBlock, 0, kBlock, RWX)) != 0) {
			return Fail("GEM_VA REPLACE of an imported block", r);
		}
	}
	printf("ok   12 x 16 MiB udmabuf blocks imported and mapped (%.1f ms)\n", (Now() - t_import) * 1e3);

	// 1. GPU write into block 3 -> CPU view and alias.
	const uint64_t dst = (uint64_t)(uintptr_t)view + 3 * kBlock + 4096;
	double         cs  = 0;
	uint32_t       n   = WriteData(0, dst, 0xC0DE1234u);
	if ((r = Submit(n, handles, 12, &cs)) != 0) {
		return Fail("submit WRITE_DATA", r);
	}
	const int gpu_to_cpu = *(volatile uint32_t*)(view + 3 * kBlock + 4096) == 0xC0DE1234u &&
	                       *(volatile uint32_t*)(alias + 3 * kBlock + 4096) == 0xC0DE1234u;
	printf("%s GPU write visible through the memfd view and alias\n", gpu_to_cpu ? "ok  " : "FAIL");

	// 2. CPU write in block 5 -> GPU copies it to block 6. The destination is pre-set so the result
	//    tells "copied a stale 0" (GPU read memory past the CPU cache) from "copy did not land".
	*(volatile uint32_t*)(view + 6 * kBlock + 64) = 0x11111111u;
	*(volatile uint32_t*)(view + 5 * kBlock + 64) = 0x5EEDF00Du;
	__sync_synchronize();
	n = CopyData(0, (uint64_t)(uintptr_t)view + 5 * kBlock + 64, (uint64_t)(uintptr_t)view + 6 * kBlock + 64);
	if ((r = Submit(n, handles, 12, &cs)) != 0) {
		return Fail("submit COPY_DATA", r);
	}
	const uint32_t got_plain = *(volatile uint32_t*)(view + 6 * kBlock + 64);
	//    Same after flushing the source line out of the CPU cache.
	*(volatile uint32_t*)(view + 6 * kBlock + 128) = 0x11111111u;
	*(volatile uint32_t*)(view + 5 * kBlock + 128) = 0x5EEDF00Eu;
	__builtin_ia32_clflush((const void*)(view + 5 * kBlock + 128));
	__builtin_ia32_clflush((const void*)(view + 6 * kBlock + 128));
	__builtin_ia32_mfence();
	n = CopyData(0, (uint64_t)(uintptr_t)view + 5 * kBlock + 128, (uint64_t)(uintptr_t)view + 6 * kBlock + 128);
	if ((r = Submit(n, handles, 12, &cs)) != 0) {
		return Fail("submit COPY_DATA after clflush", r);
	}
	const uint32_t got_flushed = *(volatile uint32_t*)(view + 6 * kBlock + 128);
	//    And a GPU write to a line the CPU has cached: is the CPU's view updated?
	(void)*(volatile uint32_t*)(view + 8 * kBlock);
	n = WriteData(0, (uint64_t)(uintptr_t)view + 8 * kBlock, 0xCAFED00Du);
	if ((r = Submit(n, handles, 12, &cs)) != 0) {
		return Fail("submit WRITE_DATA to a CPU-cached line", r);
	}
	const uint32_t got_cached = *(volatile uint32_t*)(view + 8 * kBlock);
	const int cpu_to_gpu = got_plain == 0x5EEDF00Du;
	printf("%s CPU write read by the GPU: plain 0x%08x, after clflush 0x%08x (want 5eedf00d / 5eedf00e); "
	       "GPU write over a CPU-cached line seen as 0x%08x (want cafed00d)\n",
	       cpu_to_gpu ? "ok  " : "FAIL", got_plain, got_flushed, got_cached);

	// 4. CS cost: 16 blocks listed, then 64 blocks (48 more 16 MiB udmabufs from a second memfd range
	//    would exceed this memfd; reuse: import the same 16 MiB ranges again as separate buffers).
	double best12 = 1e9;
	for (int i = 0; i < 20; i++) {
		n = WriteData(0, dst, (uint32_t)i);
		if ((r = Submit(n, handles, 12, &cs)) != 0) {
			return Fail("timing submit", r);
		}
		best12 = cs < best12 ? cs : best12;
	}
	uint32_t many[64];
	for (int i = 0; i < 64; i++) {
		struct udmabuf_create u;
		memset(&u, 0, sizeof(u));
		u.memfd  = (uint32_t)mem;
		u.flags  = UDMABUF_FLAGS_CLOEXEC;
		u.offset = (uint64_t)(i % 16) * kBlock;
		u.size   = kBlock;
		int dbuf = ioctl(ufd, UDMABUF_CREATE, &u);
		struct drm_prime_handle p;
		memset(&p, 0, sizeof(p));
		p.fd = dbuf;
		if (dbuf < 0 || Ioctl(g_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &p) != 0) {
			return Fail("extra udmabuf import", -errno);
		}
		close(dbuf);
		many[i] = p.handle;
	}
	double best64 = 1e9;
	for (int i = 0; i < 20; i++) {
		n = WriteData(0, dst, (uint32_t)i);
		if ((r = Submit(n, many, 64, &cs)) != 0) {
			return Fail("timing submit 64", r);
		}
		best64 = cs < best64 ? cs : best64;
	}
	printf("     CS ioctl: %.3f ms with 12 imported blocks listed, %.3f ms with 64\n", best12, best64);
	return gpu_to_cpu && cpu_to_gpu ? 0 : 1;
}
