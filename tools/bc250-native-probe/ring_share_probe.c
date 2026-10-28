// Gfx ring sharing probe for the BC-250's patched amdgpu (bc250_gfx_queues=2: two gfx queues on pipe 0,
// as the PS5's main and AGR rings). Does a WAIT_REG_MEM that ring 1 sits on hold up ring 0? If not,
// a title's command buffers can run on ring 1 while Kyty's own Vulkan work (RADV, ring 0) carries on
// past the title's GPU waits, such as WaitUntilSafeForRendering's wait for a flip that work makes.
//   ring 1: WAIT_REG_MEM on label A == 1 (the CPU sets it after 2 s), then label C = 3;
//   ring 0: label B = 2, submitted 100 ms later; how long until B lands.
// If ring 0 waits for ring 1, B lands only after A is set (~2 s): the rings share the pipe in order.
// Build: cc -O2 -o ring_share_probe ring_share_probe.c. Needs bc250_gfx_queues=2.
#define _GNU_SOURCE
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static uint32_t ctx_id;
static int      drm = -1;

static uint32_t ctx_id;

static int Ioctl(int fd, unsigned long req, void* arg) {
	int r;
	do {
		r = ioctl(fd, req, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}
static double Now(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

struct Bo {
	uint32_t handle;
	uint64_t va, size;
	void*    cpu;
};

static int BoAt(struct Bo* b, uint64_t va, uint64_t size) {
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = 4096;
	c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
	c.in.domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED | AMDGPU_GEM_CREATE_CPU_GTT_USWC;
	int r             = Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
	if (r) {
		return r;
	}
	b->handle = c.out.handle;
	b->size   = size;
	b->va     = va;
	union drm_amdgpu_gem_mmap m;
	memset(&m, 0, sizeof(m));
	m.in.handle = b->handle;
	if ((r = Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_MMAP, &m))) {
		return r;
	}
	b->cpu = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm, (off_t)m.out.addr_ptr);
	if (b->cpu == MAP_FAILED) {
		return -errno;
	}
	memset(b->cpu, 0, size);
	struct drm_amdgpu_gem_va v;
	memset(&v, 0, sizeof(v));
	v.handle     = b->handle;
	v.operation  = AMDGPU_VA_OP_MAP;
	v.flags      = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE | AMDGPU_VM_MTYPE_UC;
	v.va_address = va;
	v.map_size   = size;
	return Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_VA, &v);
}

static struct Bo ib, label;

// Submits `n` dwords of `ib` to (ip, ring) with the IB and the label resident; 0 or a negative errno.
static int SubmitRing(uint32_t ip, uint32_t ring, uint32_t ib_offset, uint32_t n, uint64_t* seq) {
	struct drm_amdgpu_bo_list_entry entries[2] = {{ib.handle, 0}, {label.handle, 0}};
	struct drm_amdgpu_bo_list_in    list;
	memset(&list, 0, sizeof(list));
	list.operation    = ~0u;
	list.list_handle  = ~0u;
	list.bo_number    = 2;
	list.bo_info_size = sizeof(entries[0]);
	list.bo_info_ptr  = (uintptr_t)entries;
	struct drm_amdgpu_cs_chunk_ib info;
	memset(&info, 0, sizeof(info));
	info.va_start = ib.va + ib_offset;
	info.ib_bytes = n * 4;
	info.ip_type  = ip;
	info.ring     = ring;
	struct drm_amdgpu_cs_chunk chunks[2] = {{AMDGPU_CHUNK_ID_IB, sizeof(info) / 4, (uintptr_t)&info},
	                                        {AMDGPU_CHUNK_ID_BO_HANDLES, sizeof(list) / 4, (uintptr_t)&list}};
	uint64_t                   ptrs[2]   = {(uintptr_t)&chunks[0], (uintptr_t)&chunks[1]};
	union drm_amdgpu_cs        cs;
	memset(&cs, 0, sizeof(cs));
	cs.in.ctx_id     = ctx_id;
	cs.in.num_chunks = 2;
	cs.in.chunks     = (uintptr_t)ptrs;
	int r            = Ioctl(drm, DRM_IOCTL_AMDGPU_CS, &cs);
	*seq             = cs.out.handle;
	return r;
}

static int WaitCsRing(uint32_t ip, uint32_t ring, uint64_t seq, double sec) {
	union drm_amdgpu_wait_cs w;
	memset(&w, 0, sizeof(w));
	w.in.handle  = seq;
	w.in.timeout = (uint64_t)((Now() + sec) * 1e9); // absolute CLOCK_MONOTONIC
	w.in.ip_type = ip;
	w.in.ring    = ring;
	w.in.ctx_id  = ctx_id;
	int r        = Ioctl(drm, DRM_IOCTL_AMDGPU_WAIT_CS, &w);
	return r ? r : (w.out.status ? 1 : 0);
}



int main(void) {
	setvbuf(stdout, NULL, _IOLBF, 0);
	FILE* q = fopen("/sys/module/amdgpu/parameters/bc250_gfx_queues", "r");
	int   queues = 0;
	if (q) {
		if (fscanf(q, "%d", &queues) != 1) {
			queues = 0;
		}
		fclose(q);
	}
	printf("bc250_gfx_queues: %d\n", queues);
	drm = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	union drm_amdgpu_ctx c;
	memset(&c, 0, sizeof(c));
	c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	if (drm < 0 || Ioctl(drm, DRM_IOCTL_AMDGPU_CTX, &c)) {
		fprintf(stderr, "no render node or context\n");
		return 1;
	}
	ctx_id = c.out.alloc.ctx_id;
	if (BoAt(&ib, 0x510000000000ull, 65536) || BoAt(&label, 0x510000100000ull, 4096)) {
		fprintf(stderr, "buffers\n");
		return 2;
	}
	volatile uint32_t* l = (volatile uint32_t*)label.cpu; // A at 0, B at 64, C at 128 (UC mapping)
	uint32_t*          d = (uint32_t*)ib.cpu;
	// ring 1 IB at 0: WAIT_REG_MEM(A == 1), then WRITE_DATA C = 3 with confirm
	uint32_t n = 0;
	d[n++] = 0xc0053c00u; // WAIT_REG_MEM, 7 dwords
	d[n++] = 0x13u;       // EQUAL, memory, PFP
	d[n++] = (uint32_t)label.va;
	d[n++] = (uint32_t)(label.va >> 32);
	d[n++] = 1;
	d[n++] = 0xffffffffu;
	d[n++] = 4;
	d[n++] = 0xc0033700u; // WRITE_DATA, 5 dwords
	d[n++] = (5u << 8) | (1u << 20); // DST memory, write confirm
	d[n++] = (uint32_t)(label.va + 128);
	d[n++] = (uint32_t)((label.va + 128) >> 32);
	d[n++] = 3;
	while (n & 7u) {
		d[n++] = 0xffff1000u;
	}
	const uint32_t n1 = n;
	// ring 0 IB at 4096: WRITE_DATA B = 2 with confirm
	uint32_t* e = d + 1024;
	uint32_t  m = 0;
	e[m++] = 0xc0033700u;
	e[m++] = (5u << 8) | (1u << 20);
	e[m++] = (uint32_t)(label.va + 64);
	e[m++] = (uint32_t)((label.va + 64) >> 32);
	e[m++] = 2;
	while (m & 7u) {
		e[m++] = 0xffff1000u;
	}
	__sync_synchronize();
	uint64_t s1 = 0, s0 = 0;
	const double t0 = Now();
	if (SubmitRing(AMDGPU_HW_IP_GFX, 1, 0, n1, &s1)) {
		fprintf(stderr, "ring 1 CS failed (bc250_gfx_queues=2?)\n");
		return 3;
	}
	struct timespec nap = {0, 100000000};
	nanosleep(&nap, NULL);
	if (SubmitRing(AMDGPU_HW_IP_GFX, 0, 4096, m, &s0)) {
		fprintf(stderr, "ring 0 CS failed\n");
		l[0] = 1;
		return 3;
	}
	const double t_sub0 = Now();
	double       t_b    = 0;
	while (Now() - t0 < 2.1) {
		if (t_b == 0 && l[16] == 2) {
			t_b = Now();
		}
	}
	l[0] = 1; // release ring 1
	__sync_synchronize();
	const double t_release = Now();
	while (l[32] != 3 && Now() - t_release < 3.0) {
		if (t_b == 0 && l[16] == 2) {
			t_b = Now();
		}
	}
	if (t_b == 0 && l[16] == 2) {
		t_b = Now();
	}
	const int done1 = WaitCsRing(AMDGPU_HW_IP_GFX, 1, s1, 3.0);
	const int done0 = WaitCsRing(AMDGPU_HW_IP_GFX, 0, s0, 3.0);
	printf("ring 0 label B: %s %.1f ms after its submission (ring 1 released %.1f ms after it)\n", t_b ? "landed" : "NEVER landed",
	       t_b ? (t_b - t_sub0) * 1e3 : 0.0, (t_release - t_sub0) * 1e3);
	printf("ring 1 label C: %u, fences: ring 1 %s, ring 0 %s\n", l[32], done1 == 0 ? "signaled" : "NOT", done0 == 0 ? "signaled" : "NOT");
	printf("%s\n", t_b && t_b < t_release ? "ring 0 runs past ring 1's wait" : "ring 0 waited for ring 1");
	return 0;
}
