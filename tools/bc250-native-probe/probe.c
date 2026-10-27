// BC-250 native backend probe: checks what the stock amdgpu kernel driver and firmware
// allow from user mode, using raw ioctls (no libdrm). Only safe cases are implemented:
// nothing here is expected to fault, hang or reset the GPU.

#define _GNU_SOURCE
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
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

// gfx10 compute registers (byte offsets, see Mesa src/amd/registers/gfx10.json).
#define SH_REG_BASE                    0xb000u
#define COMPUTE_START_X                0xb810u
#define COMPUTE_NUM_THREAD_X           0xb81cu
#define COMPUTE_PGM_LO                 0xb830u
#define COMPUTE_PGM_RSRC1              0xb848u
#define COMPUTE_RESOURCE_LIMITS        0xb854u
#define COMPUTE_STATIC_THREAD_MGMT_SE0 0xb858u
#define COMPUTE_TMPRING_SIZE           0xb860u
#define COMPUTE_STATIC_THREAD_MGMT_SE2 0xb864u
#define COMPUTE_PGM_RSRC3              0xb8a0u
#define COMPUTE_USER_DATA_0            0xb900u

#define PKT3_NOP             0x10u
#define PKT3_DISPATCH_DIRECT 0x15u
#define PKT3_SET_SH_REG      0x76u

#define INITIATOR_COMPUTE_SHADER_EN (1u << 0)
#define INITIATOR_FORCE_START_AT_0  (1u << 2)
#define INITIATOR_ORDER_MODE        (1u << 6)
#define INITIATOR_CS_W32_EN         (1u << 15)

static uint32_t Pkt3(uint32_t op, uint32_t body_dw) {
	return (3u << 30) | (((body_dw - 1u) & 0x3fffu) << 16) | (op << 8) | (1u << 1);
}

static double NowSec(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int Ioctl(int fd, unsigned long req, void* arg) {
	int r = -EINVAL;
	do {
		r = ioctl(fd, req, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

// ---------------------------------------------------------------------------------------------
// Device, context, buffers

typedef struct {
	int                         fd;
	uint32_t                    ctx;
	struct drm_amdgpu_info_device info;
} Device;

typedef struct {
	uint32_t handle;
	uint64_t size;
	void*    cpu;
	uint64_t va;
	bool     userptr;
	void*    userptr_mem;
} Bo;

static int QueryInfo(Device* d, uint32_t query, void* out, uint32_t size, uint32_t fw_type) {
	struct drm_amdgpu_info req;
	memset(&req, 0, sizeof(req));
	req.return_pointer   = (uintptr_t)out;
	req.return_size      = size;
	req.query            = query;
	req.query_fw.fw_type = fw_type;
	return Ioctl(d->fd, DRM_IOCTL_AMDGPU_INFO, &req);
}

static int QueryHwIp(Device* d, uint32_t ip, struct drm_amdgpu_info_hw_ip* out) {
	struct drm_amdgpu_info req;
	memset(&req, 0, sizeof(req));
	req.return_pointer       = (uintptr_t)out;
	req.return_size          = sizeof(*out);
	req.query                = AMDGPU_INFO_HW_IP_INFO;
	req.query_hw_ip.type     = ip;
	req.query_hw_ip.ip_instance = 0;
	return Ioctl(d->fd, DRM_IOCTL_AMDGPU_INFO, &req);
}

static int DeviceOpen(Device* d, const char* path) {
	memset(d, 0, sizeof(*d));
	d->fd = open(path, O_RDWR | O_CLOEXEC);
	if (d->fd < 0) {
		fprintf(stderr, "open %s: %s\n", path, strerror(errno));
		return -1;
	}
	int r = QueryInfo(d, AMDGPU_INFO_DEV_INFO, &d->info, sizeof(d->info), 0);
	if (r != 0) {
		fprintf(stderr, "AMDGPU_INFO_DEV_INFO: %s\n", strerror(-r));
		return -1;
	}
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	r         = Ioctl(d->fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
	if (r != 0) {
		fprintf(stderr, "CTX alloc: %s\n", strerror(-r));
		return -1;
	}
	d->ctx = ctx.out.alloc.ctx_id;
	return 0;
}

// Returns the context reset flags (AMDGPU_CTX_QUERY2_FLAGS_*), or a negative errno.
static int64_t ContextState(Device* d) {
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op     = AMDGPU_CTX_OP_QUERY_STATE2;
	ctx.in.ctx_id = d->ctx;
	int r         = Ioctl(d->fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
	return r != 0 ? r : (int64_t)ctx.out.state.flags;
}

static int BoCreate(Device* d, Bo* bo, uint64_t size, uint32_t domain, uint64_t flags) {
	memset(bo, 0, sizeof(*bo));
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = 64 * KiB;
	c.in.domains      = domain;
	c.in.domain_flags = flags;
	int r             = Ioctl(d->fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
	if (r != 0) {
		return r;
	}
	bo->handle = c.out.handle;
	bo->size   = size;
	return 0;
}

static int BoCreateUserptr(Device* d, Bo* bo, uint64_t size, void* fixed_cpu) {
	memset(bo, 0, sizeof(*bo));
	int   mflags = MAP_PRIVATE | MAP_ANONYMOUS | (fixed_cpu != NULL ? MAP_FIXED_NOREPLACE : 0);
	void* mem    = mmap(fixed_cpu, size, PROT_READ | PROT_WRITE, mflags, -1, 0);
	if (mem == MAP_FAILED) {
		return -errno;
	}
	if (fixed_cpu != NULL && mem != fixed_cpu) {
		munmap(mem, size);
		return -EEXIST;
	}
	memset(mem, 0, size);
	struct drm_amdgpu_gem_userptr u;
	memset(&u, 0, sizeof(u));
	u.addr  = (uintptr_t)mem;
	u.size  = size;
	u.flags = AMDGPU_GEM_USERPTR_ANONONLY | AMDGPU_GEM_USERPTR_VALIDATE | AMDGPU_GEM_USERPTR_REGISTER;
	int r   = Ioctl(d->fd, DRM_IOCTL_AMDGPU_GEM_USERPTR, &u);
	if (r != 0) {
		munmap(mem, size);
		return r;
	}
	bo->handle      = u.handle;
	bo->size        = size;
	bo->cpu         = mem;
	bo->userptr     = true;
	bo->userptr_mem = mem;
	return 0;
}

static int BoMap(Device* d, Bo* bo, void* fixed_cpu) {
	if (bo->userptr) {
		return 0;
	}
	union drm_amdgpu_gem_mmap m;
	memset(&m, 0, sizeof(m));
	m.in.handle = bo->handle;
	int r       = Ioctl(d->fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &m);
	if (r != 0) {
		return r;
	}
	int   mflags = MAP_SHARED | (fixed_cpu != NULL ? MAP_FIXED_NOREPLACE : 0);
	void* p      = mmap(fixed_cpu, bo->size, PROT_READ | PROT_WRITE, mflags, d->fd, (off_t)m.out.addr_ptr);
	if (p == MAP_FAILED) {
		return -errno;
	}
	if (fixed_cpu != NULL && p != fixed_cpu) {
		munmap(p, bo->size);
		return -EEXIST;
	}
	bo->cpu = p;
	return 0;
}

static int VaOp(Device* d, Bo* bo, uint32_t op, uint64_t va, uint32_t flags) {
	struct drm_amdgpu_gem_va v;
	memset(&v, 0, sizeof(v));
	v.handle       = bo->handle;
	v.operation    = op;
	v.flags        = flags;
	v.va_address   = va;
	v.offset_in_bo = 0;
	v.map_size     = bo->size;
	return Ioctl(d->fd, DRM_IOCTL_AMDGPU_GEM_VA, &v);
}

#define VA_RWX (AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE)

static int BoBind(Device* d, Bo* bo, uint64_t va, uint32_t mtype) {
	int r = VaOp(d, bo, AMDGPU_VA_OP_MAP, va, VA_RWX | mtype);
	if (r == 0) {
		bo->va = va;
	}
	return r;
}

static void BoDestroy(Device* d, Bo* bo) {
	if (bo->handle == 0) {
		return;
	}
	if (bo->va != 0) {
		VaOp(d, bo, AMDGPU_VA_OP_UNMAP, bo->va, VA_RWX);
	}
	if (bo->cpu != NULL && !bo->userptr) {
		munmap(bo->cpu, bo->size);
	}
	struct drm_gem_close c;
	memset(&c, 0, sizeof(c));
	c.handle = bo->handle;
	Ioctl(d->fd, DRM_IOCTL_GEM_CLOSE, &c);
	if (bo->userptr_mem != NULL) {
		munmap(bo->userptr_mem, bo->size);
	}
	memset(bo, 0, sizeof(*bo));
}

// ---------------------------------------------------------------------------------------------
// Command submission

typedef struct {
	uint32_t dw[512];
	uint32_t n;
} Ib;

static void IbPush(Ib* ib, uint32_t v) {
	if (ib->n < ARRAY_SIZE(ib->dw)) {
		ib->dw[ib->n++] = v;
	}
}

static void IbSetSh(Ib* ib, uint32_t reg, const uint32_t* vals, uint32_t count) {
	IbPush(ib, Pkt3(PKT3_SET_SH_REG, count + 1));
	IbPush(ib, (reg - SH_REG_BASE) / 4);
	for (uint32_t i = 0; i < count; i++) {
		IbPush(ib, vals[i]);
	}
}

static void IbSetSh1(Ib* ib, uint32_t reg, uint32_t val) {
	IbSetSh(ib, reg, &val, 1);
}

typedef struct {
	uint64_t        code_va;
	uint32_t        wave_size;
	uint32_t        vgprs;
	uint32_t        user_sgprs;
	bool            tgid_x;
	uint32_t        group_size;
	uint32_t        groups;
	const uint32_t* user_data;
} Dispatch;

static void IbDispatch(Ib* ib, const Dispatch* p) {
	uint32_t granule = p->wave_size == 32 ? 8u : 4u;
	uint32_t rsrc1   = ((p->vgprs + granule - 1) / granule - 1) & 0x3fu; // VGPRS
	rsrc1 |= 0xc0u << 12;                                                 // FLOAT_MODE: denorms on
	rsrc1 |= 1u << 21;                                                    // DX10_CLAMP
	rsrc1 |= 1u << 30;                                                    // MEM_ORDERED
	uint32_t rsrc2 = (p->user_sgprs & 0x1fu) << 1;
	rsrc2 |= p->tgid_x ? 1u << 7 : 0u;

	uint32_t pgm[2] = {(uint32_t)(p->code_va >> 8), (uint32_t)(p->code_va >> 40)};
	IbSetSh(ib, COMPUTE_PGM_LO, pgm, 2);
	uint32_t rsrc12[2] = {rsrc1, rsrc2};
	IbSetSh(ib, COMPUTE_PGM_RSRC1, rsrc12, 2);
	IbSetSh1(ib, COMPUTE_PGM_RSRC3, 0);
	IbSetSh1(ib, COMPUTE_RESOURCE_LIMITS, 0);
	IbSetSh1(ib, COMPUTE_TMPRING_SIZE, 0);
	uint32_t all[2] = {0xffffffffu, 0xffffffffu};
	IbSetSh(ib, COMPUTE_STATIC_THREAD_MGMT_SE0, all, 2);
	IbSetSh(ib, COMPUTE_STATIC_THREAD_MGMT_SE2, all, 2);
	uint32_t start[3] = {0, 0, 0};
	IbSetSh(ib, COMPUTE_START_X, start, 3);
	uint32_t threads[3] = {p->group_size, 1, 1};
	IbSetSh(ib, COMPUTE_NUM_THREAD_X, threads, 3);
	if (p->user_sgprs > 0) {
		IbSetSh(ib, COMPUTE_USER_DATA_0, p->user_data, p->user_sgprs);
	}
	uint32_t initiator = INITIATOR_COMPUTE_SHADER_EN | INITIATOR_FORCE_START_AT_0 | INITIATOR_ORDER_MODE;
	if (p->wave_size == 32) {
		initiator |= INITIATOR_CS_W32_EN;
	}
	IbPush(ib, Pkt3(PKT3_DISPATCH_DIRECT, 4));
	IbPush(ib, p->groups);
	IbPush(ib, 1);
	IbPush(ib, 1);
	IbPush(ib, initiator);
}

// Pads to 8 dwords; the CP fetches IBs in 32-byte units.
static void IbPad(Ib* ib) {
	while ((ib->n & 7u) != 0) {
		IbPush(ib, 0xffff1000u); // single-dword NOP (type 3, count 0x3fff)
	}
}

// Submits `bytes` of commands at `va` (inside ib_bo) with the given buffers resident. Returns the fence seq.
static int SubmitVa(Device* d, uint32_t ip, uint32_t ring, Bo* ib_bo, uint64_t va, uint32_t bytes, Bo* const* bos,
                    uint32_t num_bos, uint64_t* seq) {
	struct drm_amdgpu_bo_list_entry entries[16];
	uint32_t                        n = 0;
	entries[n].bo_handle              = ib_bo->handle;
	entries[n++].bo_priority          = 0;
	for (uint32_t i = 0; i < num_bos && n < ARRAY_SIZE(entries); i++) {
		entries[n].bo_handle     = bos[i]->handle;
		entries[n++].bo_priority = 0;
	}
	struct drm_amdgpu_bo_list_in list;
	memset(&list, 0, sizeof(list));
	list.operation    = ~0u;
	list.list_handle  = ~0u;
	list.bo_number    = n;
	list.bo_info_size = sizeof(entries[0]);
	list.bo_info_ptr  = (uintptr_t)entries;

	struct drm_amdgpu_cs_chunk_ib ib_info;
	memset(&ib_info, 0, sizeof(ib_info));
	ib_info.va_start = va;
	ib_info.ib_bytes = bytes;
	ib_info.ip_type  = ip;
	ib_info.ring     = ring;

	struct drm_amdgpu_cs_chunk chunks[2];
	chunks[0].chunk_id   = AMDGPU_CHUNK_ID_IB;
	chunks[0].length_dw  = sizeof(ib_info) / 4;
	chunks[0].chunk_data = (uintptr_t)&ib_info;
	chunks[1].chunk_id   = AMDGPU_CHUNK_ID_BO_HANDLES;
	chunks[1].length_dw  = sizeof(list) / 4;
	chunks[1].chunk_data = (uintptr_t)&list;
	uint64_t chunk_ptrs[2] = {(uintptr_t)&chunks[0], (uintptr_t)&chunks[1]};

	union drm_amdgpu_cs cs;
	memset(&cs, 0, sizeof(cs));
	cs.in.ctx_id     = d->ctx;
	cs.in.num_chunks = 2;
	cs.in.chunks     = (uintptr_t)chunk_ptrs;
	int r            = Ioctl(d->fd, DRM_IOCTL_AMDGPU_CS, &cs);
	if (r == 0) {
		*seq = cs.out.handle;
	}
	return r;
}

// Copies ib into ib_bo and submits it with the given buffers resident. Returns the fence seq.
static int SubmitOn(Device* d, uint32_t ip, uint32_t ring, Bo* ib_bo, const Ib* ib, Bo* const* bos,
                    uint32_t num_bos, uint64_t* seq) {
	memcpy(ib_bo->cpu, ib->dw, ib->n * 4u);
	__sync_synchronize();
	return SubmitVa(d, ip, ring, ib_bo, ib_bo->va, ib->n * 4u, bos, num_bos, seq);
}

static int Submit(Device* d, Bo* ib_bo, const Ib* ib, Bo* const* bos, uint32_t num_bos, uint64_t* seq) {
	return SubmitOn(d, AMDGPU_HW_IP_COMPUTE, 0, ib_bo, ib, bos, num_bos, seq);
}

// Returns 0 when the fence signaled, 1 on timeout, negative errno on error.
static int WaitOn(Device* d, uint32_t ip, uint32_t ring, uint64_t seq, double timeout_sec) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	union drm_amdgpu_wait_cs w;
	memset(&w, 0, sizeof(w));
	w.in.handle  = seq;
	w.in.timeout = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec +
	               (uint64_t)(timeout_sec * 1e9);
	w.in.ip_type = ip;
	w.in.ring    = ring;
	w.in.ctx_id  = d->ctx;
	int r        = Ioctl(d->fd, DRM_IOCTL_AMDGPU_WAIT_CS, &w);
	if (r != 0) {
		return r;
	}
	return w.out.status != 0 ? 1 : 0;
}

static int Wait(Device* d, uint64_t seq, double timeout_sec) {
	return WaitOn(d, AMDGPU_HW_IP_COMPUTE, 0, seq, timeout_sec);
}

// Submits and waits; prints a failure line and returns false on error, timeout or context reset.
static bool RunIb(Device* d, Bo* ib_bo, const Ib* ib, Bo* const* bos, uint32_t num_bos, double* elapsed) {
	uint64_t seq = 0;
	double   t0  = NowSec();
	int      r   = Submit(d, ib_bo, ib, bos, num_bos, &seq);
	if (r != 0) {
		printf("    submit failed: %s\n", strerror(-r));
		return false;
	}
	r = Wait(d, seq, 3.0);
	if (elapsed != NULL) {
		*elapsed = NowSec() - t0;
	}
	if (r != 0) {
		printf("    wait failed: %s\n", r == 1 ? "timeout" : strerror(-r));
		return false;
	}
	int64_t state = ContextState(d);
	if (state != 0) {
		printf("    context state after job: 0x%" PRIx64 " (reset or guilty)\n", (uint64_t)state);
		return false;
	}
	return true;
}

// ---------------------------------------------------------------------------------------------
// Probes

static const char* ErrStr(int r) {
	return r == 0 ? "ok" : strerror(-r);
}

static void ProbeInfo(Device* d) {
	const struct drm_amdgpu_info_device* i = &d->info;
	printf("== info\n");
	printf("  device_id 0x%x family %u external_rev 0x%x\n", i->device_id, i->family, i->external_rev);
	printf("  se %u sa/se %u cu_active %u wave_size %u num_rb_pipes %u\n", i->num_shader_engines,
	       i->num_shader_arrays_per_engine, i->cu_active_number, i->wave_front_size, i->num_rb_pipes);
	printf("  cu_bitmap:");
	for (int se = 0; se < 4; se++) {
		for (int sa = 0; sa < 4; sa++) {
			if (i->cu_bitmap[se][sa] != 0) {
				printf(" se%d.sa%d=0x%x", se, sa, i->cu_bitmap[se][sa]);
			}
		}
	}
	printf("\n");
	printf("  max_engine_clock %" PRIu64 " kHz max_memory_clock %" PRIu64 " kHz\n",
	       (uint64_t)i->max_engine_clock, (uint64_t)i->max_memory_clock);
	printf("  gpu_counter_freq %u kHz\n", i->gpu_counter_freq);
	printf("  va low  [0x%" PRIx64 ", 0x%" PRIx64 ")\n", (uint64_t)i->virtual_address_offset,
	       (uint64_t)i->virtual_address_max);
	printf("  va high [0x%" PRIx64 ", 0x%" PRIx64 ")\n", (uint64_t)i->high_va_offset, (uint64_t)i->high_va_max);
	printf("  va alignment 0x%x pte_fragment 0x%x gart_page 0x%x\n", i->virtual_address_alignment,
	       i->pte_fragment_size, i->gart_page_size);

	struct drm_amdgpu_memory_info mem;
	memset(&mem, 0, sizeof(mem));
	if (QueryInfo(d, AMDGPU_INFO_MEMORY, &mem, sizeof(mem), 0) == 0) {
		printf("  vram total %" PRIu64 " MiB usable %" PRIu64 " MiB, gtt total %" PRIu64 " MiB usable %" PRIu64
		       " MiB\n",
		       (uint64_t)mem.vram.total_heap_size / MiB, (uint64_t)mem.vram.usable_heap_size / MiB,
		       (uint64_t)mem.gtt.total_heap_size / MiB, (uint64_t)mem.gtt.usable_heap_size / MiB);
	}

	static const struct {
		uint32_t    ip;
		const char* name;
	} ips[] = {{AMDGPU_HW_IP_GFX, "gfx"}, {AMDGPU_HW_IP_COMPUTE, "compute"}, {AMDGPU_HW_IP_DMA, "sdma"}};
	for (size_t k = 0; k < ARRAY_SIZE(ips); k++) {
		struct drm_amdgpu_info_hw_ip hw;
		memset(&hw, 0, sizeof(hw));
		if (QueryHwIp(d, ips[k].ip, &hw) == 0) {
			printf("  ip %-7s v%u.%u rings_mask 0x%x (%d rings)\n", ips[k].name, hw.hw_ip_version_major,
			       hw.hw_ip_version_minor, hw.available_rings, __builtin_popcount(hw.available_rings));
		}
	}

	static const struct {
		uint32_t    type;
		const char* name;
	} fws[] = {{AMDGPU_INFO_FW_GFX_ME, "me"},   {AMDGPU_INFO_FW_GFX_PFP, "pfp"}, {AMDGPU_INFO_FW_GFX_MEC, "mec"},
	           {AMDGPU_INFO_FW_GFX_RLC, "rlc"}, {AMDGPU_INFO_FW_SMC, "smc"},     {AMDGPU_INFO_FW_SDMA, "sdma"}};
	printf("  fw:");
	for (size_t k = 0; k < ARRAY_SIZE(fws); k++) {
		struct drm_amdgpu_info_firmware fw;
		memset(&fw, 0, sizeof(fw));
		if (QueryInfo(d, AMDGPU_INFO_FW_VERSION, &fw, sizeof(fw), fws[k].type) == 0) {
			printf(" %s=%u/%u", fws[k].name, fw.ver, fw.feature);
		}
	}
	printf("\n");
}

// Guest addresses taken from KytyPS5's address space layout (src/kernel/memory*.{cpp,inc,h}).
static const struct {
	uint64_t    va;
	const char* what;
} k_guest_vas[] = {
    {0x0000000000ull, "zero page"},
    {0x0000010000ull, "64 KiB"},
    {0x0000040000ull, "HOST_SYSTEM_MANAGED_MIN"},
    {0x0000200000ull, "2 MiB"},
    {0x0200000000ull, "GUEST_DEFAULT_MAP_BASE / DEFAULT_PS5_BASE"},
    {0x07ffe00000ull, "top of system managed"},
    {0x0f00000000ull, "PRT_APERTURE_START"},
    {0x1000000000ull, "HOST_USER_MIN (Linux)"},
    {0x7000000000ull, "HOST_USER_MIN (other)"},
    {0xfbffe00000ull, "HOST_USER_MAX"},
    {0x080000000000ull, "kExtendedMemoryBase"},
    {0x7fffffe00000ull, "top of 47-bit"},
};

static void ProbeVa(Device* d) {
	printf("== va: map a 2 MiB GTT buffer at guest addresses\n");
	for (size_t k = 0; k < ARRAY_SIZE(k_guest_vas); k++) {
		Bo  bo;
		int r = BoCreate(d, &bo, 2 * MiB, AMDGPU_GEM_DOMAIN_GTT, 0);
		if (r != 0) {
			printf("  create failed: %s\n", ErrStr(r));
			return;
		}
		r = BoBind(d, &bo, k_guest_vas[k].va, 0);
		printf("  0x%012" PRIx64 " %-42s %s\n", k_guest_vas[k].va, k_guest_vas[k].what, ErrStr(r));
		BoDestroy(d, &bo);
	}

	printf("== va: MTYPE flags on a GTT mapping\n");
	static const struct {
		uint32_t    flag;
		const char* name;
	} mtypes[] = {{AMDGPU_VM_MTYPE_DEFAULT, "DEFAULT"}, {AMDGPU_VM_MTYPE_NC, "NC"}, {AMDGPU_VM_MTYPE_WC, "WC"},
	              {AMDGPU_VM_MTYPE_CC, "CC"},           {AMDGPU_VM_MTYPE_UC, "UC"}, {AMDGPU_VM_MTYPE_RW, "RW"}};
	for (size_t k = 0; k < ARRAY_SIZE(mtypes); k++) {
		Bo  bo;
		int r = BoCreate(d, &bo, 2 * MiB, AMDGPU_GEM_DOMAIN_GTT, 0);
		if (r == 0) {
			r = BoBind(d, &bo, 0x0200000000ull, mtypes[k].flag);
		}
		printf("  %-8s %s\n", mtypes[k].name, ErrStr(r));
		BoDestroy(d, &bo);
	}
}

static void ProbeAlloc(Device* d, uint64_t max_bytes) {
	printf("== alloc: largest single buffers (up to %" PRIu64 " MiB)\n", max_bytes / MiB);
	static const uint64_t sizes[] = {256 * MiB, 1 * GiB, 2 * GiB, 4 * GiB, 6 * GiB, 8 * GiB};
	for (size_t k = 0; k < ARRAY_SIZE(sizes) && sizes[k] <= max_bytes; k++) {
		Bo     bo;
		double t0 = NowSec();
		int    r  = BoCreate(d, &bo, sizes[k], AMDGPU_GEM_DOMAIN_GTT, 0);
		if (r == 0) {
			r = BoBind(d, &bo, 0x1000000000ull, 0);
		}
		printf("  gtt     %5" PRIu64 " MiB: %s (%.0f ms)\n", sizes[k] / MiB, ErrStr(r), (NowSec() - t0) * 1e3);
		BoDestroy(d, &bo);

		t0 = NowSec();
		r  = BoCreateUserptr(d, &bo, sizes[k], NULL);
		if (r == 0) {
			r = BoBind(d, &bo, 0x1000000000ull, 0);
		}
		printf("  userptr %5" PRIu64 " MiB: %s (%.0f ms)\n", sizes[k] / MiB, ErrStr(r), (NowSec() - t0) * 1e3);
		BoDestroy(d, &bo);
	}
}

typedef enum { MEM_VRAM, MEM_GTT, MEM_GTT_USWC, MEM_USERPTR } MemKind;

static const char* MemName(MemKind k) {
	switch (k) {
		case MEM_VRAM: return "vram";
		case MEM_GTT: return "gtt";
		case MEM_GTT_USWC: return "gtt-uswc";
		case MEM_USERPTR: return "userptr";
	}
	return "?";
}

static int MemCreate(Device* d, Bo* bo, MemKind k, uint64_t size, void* fixed_cpu) {
	int r = -EINVAL;
	switch (k) {
		case MEM_VRAM:
			r = BoCreate(d, bo, size, AMDGPU_GEM_DOMAIN_VRAM, AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED);
			break;
		case MEM_GTT: r = BoCreate(d, bo, size, AMDGPU_GEM_DOMAIN_GTT, 0); break;
		case MEM_GTT_USWC: r = BoCreate(d, bo, size, AMDGPU_GEM_DOMAIN_GTT, AMDGPU_GEM_CREATE_CPU_GTT_USWC); break;
		case MEM_USERPTR: return BoCreateUserptr(d, bo, size, fixed_cpu);
	}
	if (r == 0) {
		r = BoMap(d, bo, fixed_cpu);
	}
	return r;
}

typedef struct {
	Bo ib;
	Bo code;
} Common;

// Commands and code live in write-combined GTT, like Mesa's IBs, so the CP never sees stale lines.
static int CommonInit(Device* d, Common* c) {
	int r = MemCreate(d, &c->ib, MEM_GTT_USWC, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &c->ib, 0x0080000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &c->code, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &c->code, 0x0081000000ull, 0);
	}
	return r;
}

// Pushes CPU writes out to memory before the GPU fetches commands or code from it.
static void FlushCpu(const void* p, size_t size) {
	for (uintptr_t a = (uintptr_t)p & ~(uintptr_t)63; a < (uintptr_t)p + size; a += 64) {
		__builtin_ia32_clflush((const void*)a);
	}
	__sync_synchronize();
}

static void LoadCode(Common* c, const uint8_t* code, size_t size) {
	memset(c->code.cpu, 0, c->code.size);
	memcpy(c->code.cpu, code, size);
	FlushCpu(c->code.cpu, c->code.size);
}

// Runs the store shader writing (base + global thread id) to dst and checks every value.
static bool StoreAndCheck(Device* d, Common* c, Bo* dst, Bo* code, uint32_t wave, uint32_t groups) {
	uint32_t base   = 0x5000000u + wave;
	uint32_t ud[3]  = {(uint32_t)dst->va, (uint32_t)(dst->va >> 32), base};
	Dispatch p      = {code->va, wave, 8, 3, true, 64, groups, ud};
	Ib       ib     = {.n = 0};
	IbDispatch(&ib, &p);
	IbPad(&ib);
	Bo* bos[] = {dst, code};
	memset(dst->cpu, 0, groups * 64u * 4u);
	if (!RunIb(d, &c->ib, &ib, bos, 2, NULL)) {
		return false;
	}
	const uint32_t* out = (const uint32_t*)dst->cpu;
	for (uint32_t i = 0; i < groups * 64u; i++) {
		if (out[i] != base + i) {
			printf("    mismatch at %u: 0x%08x != 0x%08x\n", i, out[i], base + i);
			return false;
		}
	}
	return true;
}

static void ProbeDispatch(Device* d, Common* c) {
	printf("== dispatch: store shader into each memory kind at a guest VA\n");
	static const MemKind kinds[] = {MEM_VRAM, MEM_GTT, MEM_GTT_USWC, MEM_USERPTR};
	for (size_t k = 0; k < ARRAY_SIZE(kinds); k++) {
		for (uint32_t wave = 32; wave <= 64; wave *= 2) {
			Bo  dst;
			int r = MemCreate(d, &dst, kinds[k], 2 * MiB, NULL);
			if (r == 0) {
				r = BoBind(d, &dst, 0x0200000000ull, 0);
			}
			if (r != 0) {
				printf("  %-9s w%u: setup failed: %s\n", MemName(kinds[k]), wave, ErrStr(r));
				BoDestroy(d, &dst);
				continue;
			}
			LoadCode(c, wave == 32 ? k_store_w32 : k_store_w64, wave == 32 ? sizeof(k_store_w32) : sizeof(k_store_w64));
			bool ok = StoreAndCheck(d, c, &dst, &c->code, wave, 64);
			printf("  %-9s w%u: %s\n", MemName(kinds[k]), wave, ok ? "ok" : "FAILED");
			BoDestroy(d, &dst);
		}
	}

	printf("== dispatch: shader code executed from userptr (guest-owned) memory\n");
	Bo code;
	Bo dst;
	int r = MemCreate(d, &code, MEM_USERPTR, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &code, 0x0300000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &dst, MEM_GTT, 2 * MiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &dst, 0x0200000000ull, 0);
	}
	if (r == 0) {
		memcpy(code.cpu, k_store_w64, sizeof(k_store_w64));
		FlushCpu(code.cpu, code.size);
		bool ok = StoreAndCheck(d, c, &dst, &code, 64, 64);
		printf("  userptr code: %s\n", ok ? "ok" : "FAILED");
	} else {
		printf("  setup failed: %s\n", ErrStr(r));
	}
	BoDestroy(d, &dst);
	BoDestroy(d, &code);
}

// The same address for CPU and GPU: what a native backend needs for guest pointers.
static void ProbeIdentity(Device* d, Common* c) {
	printf("== identity: CPU pointer == GPU VA\n");
	static const MemKind kinds[] = {MEM_GTT, MEM_USERPTR};
	uint64_t             guest   = 0x0200000000ull;
	for (size_t k = 0; k < ARRAY_SIZE(kinds); k++, guest += 0x10000000ull) {
		Bo  bo;
		int r = MemCreate(d, &bo, kinds[k], 2 * MiB, (void*)(uintptr_t)guest);
		if (r == 0) {
			r = BoBind(d, &bo, guest, 0);
		}
		if (r != 0) {
			printf("  %-8s at 0x%" PRIx64 ": setup failed: %s\n", MemName(kinds[k]), guest, ErrStr(r));
			BoDestroy(d, &bo);
			continue;
		}
		LoadCode(c, k_store_w64, sizeof(k_store_w64));
		bool ok = StoreAndCheck(d, c, &bo, &c->code, 64, 64);
		bool same = (uintptr_t)bo.cpu == guest && bo.va == guest;
		printf("  %-8s at 0x%" PRIx64 ": %s%s\n", MemName(kinds[k]), guest, ok ? "ok" : "FAILED",
		       same ? "" : " (addresses differ)");
		BoDestroy(d, &bo);
	}
}

static void ProbeBandwidth(Device* d, Common* c) {
	printf("== bandwidth: GPU streaming read, wave64, 256 threads/group, best of 5\n");
	static const struct {
		MemKind     kind;
		uint32_t    mtype;
		const char* mname;
		uint64_t    size;
	} cases[] = {
	    {MEM_VRAM, AMDGPU_VM_MTYPE_DEFAULT, "default", 256 * MiB},
	    {MEM_GTT, AMDGPU_VM_MTYPE_DEFAULT, "default", 1 * GiB},
	    {MEM_GTT, AMDGPU_VM_MTYPE_UC, "UC", 1 * GiB},
	    {MEM_GTT, AMDGPU_VM_MTYPE_CC, "CC", 1 * GiB},
	    {MEM_GTT_USWC, AMDGPU_VM_MTYPE_DEFAULT, "default", 1 * GiB},
	    {MEM_USERPTR, AMDGPU_VM_MTYPE_DEFAULT, "default", 1 * GiB},
	    {MEM_USERPTR, AMDGPU_VM_MTYPE_UC, "UC", 1 * GiB},
	};
	const uint32_t steps = 64; // 64 x 16 KiB = 1 MiB per group
	for (size_t k = 0; k < ARRAY_SIZE(cases); k++) {
		Bo  src;
		Bo  dst;
		int r = MemCreate(d, &src, cases[k].kind, cases[k].size, NULL);
		if (r == 0) {
			r = BoBind(d, &src, 0x1000000000ull, cases[k].mtype);
		}
		uint32_t groups = (uint32_t)(cases[k].size / (steps * 16 * KiB));
		if (r == 0) {
			r = MemCreate(d, &dst, MEM_GTT, (uint64_t)groups * 256u * 4u + 64 * KiB, NULL);
		}
		if (r == 0) {
			r = BoBind(d, &dst, 0x0400000000ull, 0);
		}
		if (r != 0) {
			printf("  %-9s %-8s: setup failed: %s\n", MemName(cases[k].kind), cases[k].mname, ErrStr(r));
			BoDestroy(d, &src);
			BoDestroy(d, &dst);
			continue;
		}
		LoadCode(c, k_stream_w64, sizeof(k_stream_w64));
		uint32_t ud[5] = {(uint32_t)src.va, (uint32_t)(src.va >> 32), (uint32_t)dst.va, (uint32_t)(dst.va >> 32), steps};
		Dispatch p     = {c->code.va, 64, 32, 5, true, 256, groups, ud};
		Ib       ib    = {.n = 0};
		IbDispatch(&ib, &p);
		IbPad(&ib);
		Bo*    bos[] = {&src, &dst, &c->code};
		double best  = 1e9;
		bool   ok    = true;
		for (int it = 0; it < 5 && ok; it++) {
			double t = 0;
			ok       = RunIb(d, &c->ib, &ib, bos, 3, &t);
			if (t < best) {
				best = t;
			}
		}
		if (ok) {
			printf("  %-9s %-8s %5" PRIu64 " MiB: %6.1f GB/s (%.2f ms incl. submit)\n", MemName(cases[k].kind),
			       cases[k].mname, cases[k].size / MiB, (double)cases[k].size / best / 1e9, best * 1e3);
		} else {
			printf("  %-9s %-8s: FAILED\n", MemName(cases[k].kind), cases[k].mname);
		}
		BoDestroy(d, &src);
		BoDestroy(d, &dst);
	}
}

// A running wave polls a flag that the CPU writes 20 ms after submission: does the GPU see
// CPU writes without an explicit cache invalidation, per memory kind and MTYPE?
static void ProbeCoherency(Device* d, Common* c) {
	printf("== coherency: GPU polls a flag written by the CPU mid-job\n");
	static const struct {
		MemKind     kind;
		uint32_t    mtype;
		const char* mname;
	} cases[] = {
	    {MEM_GTT, AMDGPU_VM_MTYPE_DEFAULT, "default"}, {MEM_GTT, AMDGPU_VM_MTYPE_UC, "UC"},
	    {MEM_GTT_USWC, AMDGPU_VM_MTYPE_DEFAULT, "default"}, {MEM_USERPTR, AMDGPU_VM_MTYPE_DEFAULT, "default"},
	    {MEM_USERPTR, AMDGPU_VM_MTYPE_UC, "UC"},
	};
	const uint32_t max_iter = 500000u;
	const uint32_t magic    = 0xc0ffee11u;
	for (size_t k = 0; k < ARRAY_SIZE(cases); k++) {
		for (int cached = 0; cached <= 1; cached++) {
			Bo  flag;
			Bo  res;
			int r = MemCreate(d, &flag, cases[k].kind, 64 * KiB, NULL);
			if (r == 0) {
				r = BoBind(d, &flag, 0x0500000000ull, cases[k].mtype);
			}
			if (r == 0) {
				r = MemCreate(d, &res, MEM_GTT, 64 * KiB, NULL);
			}
			if (r == 0) {
				r = BoBind(d, &res, 0x0600000000ull, 0);
			}
			if (r != 0) {
				printf("  %-9s %-8s: setup failed: %s\n", MemName(cases[k].kind), cases[k].mname, ErrStr(r));
				BoDestroy(d, &flag);
				BoDestroy(d, &res);
				continue;
			}
			volatile uint32_t* f = (volatile uint32_t*)flag.cpu;
			volatile uint32_t* o = (volatile uint32_t*)res.cpu;
			*f                   = 0;
			*o                   = 0xdeadbeefu;
			if (cached) {
				LoadCode(c, k_poll_cached_w64, sizeof(k_poll_cached_w64));
			} else {
				LoadCode(c, k_poll_w64, sizeof(k_poll_w64));
			}
			uint32_t ud[6] = {(uint32_t)flag.va, (uint32_t)(flag.va >> 32), magic, max_iter, (uint32_t)res.va,
			                  (uint32_t)(res.va >> 32)};
			Dispatch p     = {c->code.va, 64, 8, 6, false, 64, 1, ud};
			Ib       ib    = {.n = 0};
			IbDispatch(&ib, &p);
			IbPad(&ib);
			Bo*      bos[] = {&flag, &res, &c->code};
			uint64_t seq   = 0;
			double   t0    = NowSec();
			r              = Submit(d, &c->ib, &ib, bos, 3, &seq);
			if (r == 0) {
				usleep(20000);
				*f = magic;
				__sync_synchronize();
				r = Wait(d, seq, 3.0);
			}
			double  t     = NowSec() - t0;
			int64_t state = ContextState(d);
			const char* label = cached ? "cached load" : "glc+dlc load";
			if (r != 0 || state != 0) {
				printf("  %-9s %-8s %-13s: job error (%s, ctx 0x%" PRIx64 ")\n", MemName(cases[k].kind),
				       cases[k].mname, label, r == 1 ? "timeout" : ErrStr(r), (uint64_t)state);
			} else if (*o == 0xffffffffu) {
				printf("  %-9s %-8s %-13s: NOT seen after %u polls (%.0f ms)\n", MemName(cases[k].kind),
				       cases[k].mname, label, max_iter, t * 1e3);
			} else {
				printf("  %-9s %-8s %-13s: seen after %u polls (%.0f ms)\n", MemName(cases[k].kind),
				       cases[k].mname, label, *o, t * 1e3);
			}
			BoDestroy(d, &flag);
			BoDestroy(d, &res);
		}
	}
}

static void ProbeDmabuf(Device* d) {
	printf("== dmabuf: export for presentation (KMS/Vulkan import)\n");
	static const MemKind kinds[] = {MEM_VRAM, MEM_GTT, MEM_USERPTR};
	for (size_t k = 0; k < ARRAY_SIZE(kinds); k++) {
		Bo  bo;
		int r = MemCreate(d, &bo, kinds[k], 2 * MiB, NULL);
		if (r == 0) {
			struct drm_prime_handle p;
			memset(&p, 0, sizeof(p));
			p.handle = bo.handle;
			p.flags  = DRM_CLOEXEC | DRM_RDWR;
			r        = Ioctl(d->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &p);
			if (r == 0) {
				close(p.fd);
			}
		}
		printf("  %-8s %s\n", MemName(kinds[k]), ErrStr(r));
		BoDestroy(d, &bo);
	}
}

static void ProbeSubmitLatency(Device* d, Common* c) {
	printf("== submit: cost of one CS ioctl and of a full round trip (NOP IB)\n");
	Ib ib = {.n = 0};
	IbPush(&ib, 0xffff1000u);
	IbPad(&ib);
	const int n        = 2000;
	double    t_submit = 0;
	uint64_t  seq      = 0;
	double    t0       = NowSec();
	for (int i = 0; i < n; i++) {
		double s = NowSec();
		int    r = Submit(d, &c->ib, &ib, NULL, 0, &seq);
		t_submit += NowSec() - s;
		if (r != 0) {
			printf("  submit failed: %s\n", ErrStr(r));
			return;
		}
	}
	Wait(d, seq, 3.0);
	double t_batch = NowSec() - t0;
	double t_rt    = 0;
	for (int i = 0; i < 200; i++) {
		double s = NowSec();
		Submit(d, &c->ib, &ib, NULL, 0, &seq);
		Wait(d, seq, 3.0);
		t_rt += NowSec() - s;
	}
	printf("  ioctl only %.1f us, pipelined %.1f us/IB, round trip %.1f us\n", t_submit / n * 1e6,
	       t_batch / n * 1e6, t_rt / 200 * 1e6);
}

// The gfx-ring form of a PM4 header (no compute shader-type bit).
static uint32_t Pkt3Gfx(uint32_t op, uint32_t body_dw) {
	return Pkt3(op, body_dw) & ~(1u << 1);
}

// INDIRECT_BUFFER from a user IB: the CP runs `dwords` at va as IB2 and comes back.
static uint32_t* PutIb2(uint32_t* p, uint64_t va, uint32_t dwords) {
	*p++ = Pkt3Gfx(0x3fu, 3);
	*p++ = (uint32_t)va & ~3u;
	*p++ = (uint32_t)(va >> 32);
	*p++ = dwords | (1u << 23); // IB_SIZE, VALID
	return p;
}

// P0 of the direct-execution plan: what an IB2 call costs the gfx CP. The same NOP stream (count
// NOP packets of `size` dwords) runs inline in the IB1, as one IB2, and as `count` IB2 calls of one
// packet each, timed by GPU clocks; the bodies sit in cached GTT (like guest command
// buffers) or in write-combined GTT. A first job checks that an IB2 call runs at all.
static void ProbeIb2(Device* d, Common* c) {
	(void)c;
	printf("== ib2: cost of INDIRECT_BUFFER calls from a user IB on the gfx ring (us per job, 20 jobs in flight)\n");
	const uint32_t max_count = 4096;
	Bo             cmd, body[2], ts;
	memset(body, 0, sizeof(body));
	int r = MemCreate(d, &cmd, MEM_GTT_USWC, 8u * MiB, NULL);
	if (r == 0) {
		r = BoBind(d, &cmd, 0x0090000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &body[0], MEM_GTT, 4u * MiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &body[0], 0x00a0000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &body[1], MEM_GTT_USWC, 4u * MiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &body[1], 0x00a1000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &ts, MEM_GTT, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &ts, 0x00a2000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		return;
	}
	Bo*                bos[] = {&body[0], &body[1], &ts};

	// Sanity: one IB2 holding WRITE_DATA of a magic value.
	uint32_t* b = (uint32_t*)body[0].cpu;
	b[0]        = Pkt3Gfx(0x37u, 4); // WRITE_DATA, memory, write confirm
	b[1]        = (5u << 8) | (1u << 20);
	b[2]        = (uint32_t)(ts.va + 0x100);
	b[3]        = (uint32_t)((ts.va + 0x100) >> 32);
	b[4]        = 0x1b2c0de5u;
	for (uint32_t i = 5; i < 8; i++) {
		b[i] = 0xffff1000u;
	}
	((volatile uint32_t*)ts.cpu)[0x40] = 0;
	uint32_t* p0 = (uint32_t*)cmd.cpu;
	uint32_t* p  = PutIb2(p0, body[0].va, 8);
	while (((p - p0) & 7) != 0) {
		*p++ = 0xffff1000u;
	}
	__sync_synchronize();
	uint64_t seq = 0;
	r            = SubmitVa(d, AMDGPU_HW_IP_GFX, 0, &cmd, cmd.va, (uint32_t)(p - p0) * 4u, bos, 3, &seq);
	int     w    = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 2.0) : r;
	int64_t st   = ContextState(d);
	uint32_t got = ((volatile uint32_t*)ts.cpu)[0x40];
	printf("  one IB2 with WRITE_DATA: %s, ctx 0x%" PRIx64 ", wrote 0x%08x\n", w == 1 ? "TIMEOUT" : ErrStr(w),
	       (uint64_t)st, got);
	if (w != 0 || st != 0) {
		_exit(3);
	}
	if (got != 0x1b2c0de5u) {
		goto out;
	}

	// Multi-dword NOPs (Kyty never sends any) inline and in an IB2, stopping at the first stop of the ring.
	static const struct {
		const char* name;
		uint32_t    nops;     // NOP packets
		uint32_t    nop_dw;   // dwords per NOP packet (1: the single-dword 0xffff1000 form)
		bool        in_ib2;   // in one IB2 instead of inline
	} steps[] = {
	    {"1 x 1 dw NOP inline", 1, 1, false},
	    {"1 x 2 dw NOP inline", 1, 2, false}, {"1 x 16 dw NOP inline", 1, 16, false},
	    {"1 x 16 dw NOP in IB2", 1, 16, true}, {"64 x 16 dw NOP in IB2", 64, 16, true},
	};
	for (size_t k = 0; k < ARRAY_SIZE(steps); k++) {
		uint32_t* bb = (uint32_t*)body[0].cpu;
		uint32_t  n  = 0;
		for (uint32_t i = 0; i < steps[k].nops; i++) {
			if (steps[k].nop_dw == 1) {
				bb[n++] = 0xffff1000u;
			} else {
				bb[n++] = Pkt3Gfx(0x10u, steps[k].nop_dw - 1);
				for (uint32_t j = 1; j < steps[k].nop_dw; j++) {
					bb[n++] = 0;
				}
			}
		}
		while ((n & 7) != 0) {
			bb[n++] = 0xffff1000u;
		}
		uint32_t* q = p0;
		if (steps[k].in_ib2) {
			q = PutIb2(q, body[0].va, n);
		} else {
			memcpy(q, bb, n * 4u);
			q += n;
		}
		while (((q - p0) & 7) != 0) {
			*q++ = 0xffff1000u;
		}
		__sync_synchronize();
		r = SubmitVa(d, AMDGPU_HW_IP_GFX, 0, &cmd, cmd.va, (uint32_t)(q - p0) * 4u, bos, 3, &seq);
		w = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 1.0) : r;
		printf("  step %s: %s", steps[k].name, w == 1 ? "TIMEOUT" : ErrStr(w));
		printf("\n");
		if (w != 0 || ContextState(d) != 0) {
			_exit(3);
		}
	}

	static const uint32_t sizes[] = {16, 64, 256};
	for (size_t si = 0; si < ARRAY_SIZE(sizes); si++) {
		const uint32_t size  = sizes[si];
		const uint32_t count = size * max_count > (1u << 19) ? (1u << 19) / size : max_count; // IB_SIZE is 20 bits
		for (int bk = 0; bk < 2; bk++) {
			uint32_t* bb = (uint32_t*)body[bk].cpu;
			for (uint32_t i = 0; i < count; i++) {
				bb[i * size] = Pkt3Gfx(0x10u, size - 1);
				memset(&bb[i * size + 1], 0, (size - 1) * 4u);
			}
		}
		// mode 0: inline in the IB1, 1: one IB2 (cached), 2: count IB2 (cached), 3: count IB2 (WC)
		double best[4];
		static const int order[] = {1, 2, 3, 0};
		for (int oi = 0; oi < 4; oi++) {
			const int mode = order[oi];
			best[mode]     = 1e9;
			Bo*      src = mode == 3 ? &body[1] : &body[0];
			uint32_t* q  = p0;
			if (mode == 0) {
				for (uint32_t i = 0; i < count; i++) {
					*q++ = Pkt3Gfx(0x10u, size - 1);
					memset(q, 0, (size - 1) * 4u);
					q += size - 1;
				}
			} else if (mode == 1) {
				q = PutIb2(q, src->va, count * size);
			} else {
				for (uint32_t i = 0; i < count; i++) {
					q = PutIb2(q, src->va + (uint64_t)i * size * 4u, size);
				}
			}
			while (((q - p0) & 7) != 0) {
				*q++ = 0xffff1000u;
			}
			__sync_synchronize();
			for (int rep = 0; rep < 4; rep++) {
				const int jobs = 20;
				double    t0   = NowSec();
				for (int j = 0; j < jobs && r == 0; j++) {
					r = SubmitVa(d, AMDGPU_HW_IP_GFX, 0, &cmd, cmd.va, (uint32_t)(q - p0) * 4u, bos, 3, &seq);
				}
				w = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 1.0) : r;
				if (w != 0 || ContextState(d) != 0) {
					printf("  size %u mode %d (%u dw): %s\n", size, mode, (uint32_t)(q - p0), w == 1 ? "TIMEOUT" : ErrStr(w));
					_exit(3); // freeing buffers would wait on the stopped ring
				}
				double us = (NowSec() - t0) / jobs * 1e6;
				if (rep > 0 && us < best[mode]) {
					best[mode] = us;
				}
			}
		}
		printf("  %4u x %3u dw NOP: inline %7.1f, one IB2 %7.1f, %u IB2 cached %7.1f (%.3f us each), WC %7.1f\n",
		       count, size, best[0], best[1], count, best[2], (best[2] - best[1]) / count, best[3]);
	}
out:
	BoDestroy(d, &ts);
	BoDestroy(d, &body[1]);
	BoDestroy(d, &body[0]);
	BoDestroy(d, &cmd);
}

// ---------------------------------------------------------------------------------------------
// Dangerous cases: each one may fault, hang or reset the GPU. Never part of the default run;
// each needs its own --danger NAME and the board owner's explicit approval.

static void PrintGpuvmFault(Device* d, const char* when) {
	struct drm_amdgpu_info_gpuvm_fault f;
	memset(&f, 0, sizeof(f));
	int r = QueryInfo(d, AMDGPU_INFO_GPUVM_FAULT, &f, sizeof(f), 0);
	if (r != 0) {
		printf("  last gpuvm fault (%s): query failed: %s\n", when, ErrStr(r));
		return;
	}
	printf("  last gpuvm fault (%s): addr 0x%" PRIx64 " status 0x%x vmhub %u\n", when, (uint64_t)f.addr, f.status,
	       f.vmhub);
}

// One wave64 stores 64 dwords (one 256-byte run, a single page) to a GPU VA that has no
// mapping, then a normal store checks that the GPU still works.
static bool DangerVmFault(Device* d, Common* c) {
	const uint64_t unmapped = 0x0900000000ull;
	printf("== danger vmfault: wave64 store of 256 bytes to unmapped VA 0x%" PRIx64 "\n", unmapped);
	PrintGpuvmFault(d, "before");

	LoadCode(c, k_store_w64, sizeof(k_store_w64));
	uint32_t ud[3] = {(uint32_t)unmapped, (uint32_t)(unmapped >> 32), 0x5000000u};
	Dispatch p     = {c->code.va, 64, 8, 3, true, 64, 1, ud};
	Ib       ib    = {.n = 0};
	IbDispatch(&ib, &p);
	IbPad(&ib);
	Bo*      bos[] = {&c->code};
	uint64_t seq   = 0;
	double   t0    = NowSec();
	int      r     = Submit(d, &c->ib, &ib, bos, 1, &seq);
	if (r != 0) {
		printf("  submit failed: %s\n", ErrStr(r));
		return false;
	}
	// Longer than amdgpu's 5 s lockup timeout, so a hang shows up as a reset, not as our timeout.
	r             = Wait(d, seq, 10.0);
	double  t     = NowSec() - t0;
	int64_t state = ContextState(d);
	printf("  faulting job: %s after %.1f ms, context state 0x%" PRIx64 "\n",
	       r == 0 ? "signaled" : (r == 1 ? "TIMEOUT" : ErrStr(r)), t * 1e3, (uint64_t)state);
	PrintGpuvmFault(d, "after");
	if (state != 0) {
		return false;
	}

	Bo dst;
	r = MemCreate(d, &dst, MEM_GTT, 2 * MiB, NULL);
	if (r == 0) {
		r = BoBind(d, &dst, 0x0200000000ull, 0);
	}
	bool ok = r == 0 && StoreAndCheck(d, c, &dst, &c->code, 64, 64);
	printf("  follow-up store on the same context: %s\n", ok ? "ok" : "FAILED");
	BoDestroy(d, &dst);
	return ok;
}

// One valid wave runs for 12 s (no fault, nothing wrong but its length) while the CPU watches
// the elapsed time it publishes. Shows whether amdgpu kills long compute jobs, and when.
static bool DangerLong(Device* d, Common* c) {
	const double   run_sec   = 12.0;
	const uint32_t tick_khz  = d->info.gpu_counter_freq;
	const uint32_t run_ticks = (uint32_t)(run_sec * tick_khz * 1000.0);
	printf("== danger long: one wave64 spins for %.0f s (%u ticks at %u kHz)\n", run_sec, run_ticks, tick_khz);

	Bo  prog;
	int r = MemCreate(d, &prog, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &prog, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &prog);
		return false;
	}
	volatile uint32_t* pv = (volatile uint32_t*)prog.cpu;
	pv[0] = pv[1] = pv[2] = 0;
	LoadCode(c, k_long_w64, sizeof(k_long_w64));
	uint32_t ud[3] = {(uint32_t)prog.va, (uint32_t)(prog.va >> 32), run_ticks};
	Dispatch p     = {c->code.va, 64, 8, 3, false, 64, 1, ud};
	Ib       ib    = {.n = 0};
	IbDispatch(&ib, &p);
	IbPad(&ib);
	Bo*      bos[] = {&prog, &c->code};
	uint64_t seq   = 0;
	double   t0    = NowSec();
	r              = Submit(d, &c->ib, &ib, bos, 2, &seq);
	if (r != 0) {
		printf("  submit failed: %s\n", ErrStr(r));
		BoDestroy(d, &prog);
		return false;
	}
	int wr = 1;
	for (int s = 1; s <= 20 && wr == 1; s++) {
		wr                = Wait(d, seq, 1.0);
		uint64_t elapsed  = (uint64_t)pv[0] | ((uint64_t)pv[1] << 32);
		printf("  t=%5.1f s  gpu elapsed %6.2f s  finished %u\n", NowSec() - t0,
		       (double)elapsed / (tick_khz * 1000.0), pv[2]);
		fflush(stdout);
	}
	double   t       = NowSec() - t0;
	int64_t  state   = ContextState(d);
	uint64_t elapsed = (uint64_t)pv[0] | ((uint64_t)pv[1] << 32);
	printf("  job: %s after %.1f s, gpu ran %.2f s, finished on its own: %s, context state 0x%" PRIx64 "\n",
	       wr == 0 ? "signaled" : (wr == 1 ? "TIMEOUT" : ErrStr(wr)), t, (double)elapsed / (tick_khz * 1000.0),
	       pv[2] == 1 ? "yes" : "no", (uint64_t)state);
	BoDestroy(d, &prog);

	Bo dst;
	r = MemCreate(d, &dst, MEM_GTT, 2 * MiB, NULL);
	if (r == 0) {
		r = BoBind(d, &dst, 0x0200000000ull, 0);
	}
	bool ok = r == 0 && StoreAndCheck(d, c, &dst, &c->code, 64, 64);
	printf("  follow-up store on the same context: %s\n", ok ? "ok" : "FAILED");
	BoDestroy(d, &dst);
	return ok;
}

// Runs a 64-lane shader that takes only a destination address and returns the lane values.
static bool RunLanes(Device* d, Common* c, const uint8_t* code, size_t size, Bo* extra, uint32_t out[64],
                     double timeout_sec, uint32_t vgprs) {
	Bo  dst;
	int r = MemCreate(d, &dst, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &dst, 0x0200000000ull, 0);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &dst);
		return false;
	}
	memset(dst.cpu, 0xee, 64 * 4);
	LoadCode(c, code, size);
	uint32_t ud[2] = {(uint32_t)dst.va, (uint32_t)(dst.va >> 32)};
	Dispatch p     = {c->code.va, 64, vgprs, 2, false, 64, 1, ud};
	Ib       ib    = {.n = 0};
	IbDispatch(&ib, &p);
	IbPad(&ib);
	Bo*      bos[] = {&dst, &c->code, extra};
	uint64_t seq   = 0;
	double   t0    = NowSec();
	r              = Submit(d, &c->ib, &ib, bos, extra != NULL ? 3 : 2, &seq);
	if (r == 0) {
		r = Wait(d, seq, timeout_sec);
	}
	int64_t state = ContextState(d);
	printf("  job: %s after %.1f ms, context state 0x%" PRIx64 "\n",
	       r == 0 ? "signaled" : (r == 1 ? "TIMEOUT" : ErrStr(r)), (NowSec() - t0) * 1e3, (uint64_t)state);
	memcpy(out, dst.cpu, 64 * 4);
	BoDestroy(d, &dst);
	return r == 0 && state == 0;
}

static bool DangerDot2c(Device* d, Common* c) {
	printf("== danger dot2c: v_dot2c_f32_f16 (gfx10.3 encoding, VOP2 0x02), expect 22.5 in all lanes\n");
	uint32_t out[64];
	if (!RunLanes(d, c, k_dot2c_w64, sizeof(k_dot2c_w64), NULL, out, 15.0, 16)) {
		return false;
	}
	uint32_t good = 0;
	for (int i = 0; i < 64; i++) {
		good += out[i] == 0x41b40000u;
	}
	float f;
	memcpy(&f, &out[0], 4);
	printf("  lane 0 = 0x%08x (%g), %u/64 lanes correct\n", out[0], f, good);
	return good == 64;
}

static bool DangerDot2x(Device* d, Common* c) {
	printf("== danger dot2x: what VOP2 opcode 0x02 does on this GPU (see dot2x.s)\n");
	uint32_t out[64];
	if (!RunLanes(d, c, k_dot2x_w64, sizeof(k_dot2x_w64), NULL, out, 15.0, 16)) {
		return false;
	}
	static const char* const what[4] = {"acc 10.0, (1.5,2).(3,4)  expect 0x41b40000",
	                                    "acc 0x12345678, same     expect (dot) ",
	                                    "acc 10.0, zero inputs    expect 0x41200000",
	                                    "control                  expect 0xabcd0123"};
	for (int lane = 0; lane < 2; lane++) {
		for (int k = 0; k < 4; k++) {
			printf("  lane %d +%2d %s: 0x%08x\n", lane, k * 4, what[k], out[lane * 4 + k]);
		}
	}
	return true;
}

static bool DangerGds(Device* d, Common* c) {
	printf("== danger gds: allocate GDS/GWS/OA, then 64 lanes ds_add_rtn_u32 on GDS dword 0\n");
	static const struct {
		uint32_t    domain;
		uint64_t    size;
		const char* name;
	} doms[] = {{AMDGPU_GEM_DOMAIN_GDS, 4 * KiB, "gds 4 KiB"}, {AMDGPU_GEM_DOMAIN_GWS, 1, "gws 1"},
	            {AMDGPU_GEM_DOMAIN_OA, 1, "oa 1"}};
	for (size_t k = 0; k < ARRAY_SIZE(doms); k++) {
		Bo  bo;
		int r = BoCreate(d, &bo, doms[k].size, doms[k].domain, 0);
		printf("  alloc %-9s: %s\n", doms[k].name, ErrStr(r));
		BoDestroy(d, &bo);
	}
	Bo  gds;
	int r = BoCreate(d, &gds, 4 * KiB, AMDGPU_GEM_DOMAIN_GDS, 0);
	if (r != 0) {
		printf("  no GDS buffer, skipping the atomic test\n");
		return true;
	}
	uint32_t out[64];
	bool     ok = RunLanes(d, c, k_gds_w64, sizeof(k_gds_w64), &gds, out, 15.0, 16);
	BoDestroy(d, &gds);
	if (!ok) {
		return false;
	}
	uint64_t seen = 0;
	for (int i = 0; i < 64; i++) {
		if (out[i] < 64) {
			seen |= 1ull << out[i];
		}
	}
	printf("  lane values 0x%x 0x%x 0x%x ... unique 0..63: %s\n", out[0], out[1], out[2],
	       seen == ~0ull ? "yes" : "no");
	return true;
}

// GDS through DMA_DATA, as My First Gran Turismo's async compute does it: 400 bytes from memory into
// GDS at 0x190, then back out of GDS into another buffer, with the title's exact packet controls.
// Runs on compute ring 0, then on the gfx ring.
static bool DangerGdsDma(Device* d, Common* c) {
	printf("== danger gdsdma: DMA_DATA memory -> GDS 0x190 (400 bytes) -> memory, compute ring 0 then gfx\n");
	Bo  gds;
	Bo  buf;
	int r = BoCreate(d, &gds, 4 * KiB, AMDGPU_GEM_DOMAIN_GDS, 0);
	if (r == 0) {
		r = MemCreate(d, &buf, MEM_GTT, 64 * KiB, NULL);
		if (r == 0) {
			r = BoBind(d, &buf, 0x0300000000ull, AMDGPU_VM_MTYPE_UC);
		}
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		return false;
	}
	static const struct {
		uint32_t    ip;
		const char* name;
	} rings[] = {{AMDGPU_HW_IP_COMPUTE, "compute 0"}, {AMDGPU_HW_IP_GFX, "gfx"}};
	bool ok = true;
	for (size_t k = 0; k < ARRAY_SIZE(rings) && ok; k++) {
		uint32_t* src = (uint32_t*)buf.cpu;
		uint32_t* dst = src + 1024;
		for (int i = 0; i < 100; i++) {
			src[i] = 0x5a000000u + (uint32_t)k * 0x10000u + (uint32_t)i;
			dst[i] = 0xeeeeeeeeu;
		}
		__sync_synchronize();
		const uint64_t sva = buf.va;
		const uint64_t dva = buf.va + 4096;
		Ib ib = {.n = 0};
		IbPush(&ib, Pkt3(0x50u, 6)); // DMA_DATA memory (L2) -> GDS
		IbPush(&ib, 0x60104000u);
		IbPush(&ib, (uint32_t)sva);
		IbPush(&ib, (uint32_t)(sva >> 32));
		IbPush(&ib, 0x190u);
		IbPush(&ib, 0u);
		IbPush(&ib, 0x40000190u);
		IbPush(&ib, Pkt3(0x50u, 6)); // DMA_DATA GDS -> memory (L2)
		IbPush(&ib, 0x26300000u);
		IbPush(&ib, 0x190u);
		IbPush(&ib, 0u);
		IbPush(&ib, (uint32_t)dva);
		IbPush(&ib, (uint32_t)(dva >> 32));
		IbPush(&ib, 0x40000190u | (1u << 31)); // RAW_WAIT: the read waits for the write
		IbPad(&ib);
		Bo*      bos[] = {&buf, &gds};
		uint64_t seq   = 0;
		double   t0    = NowSec();
		r              = SubmitOn(d, rings[k].ip, 0, &c->ib, &ib, bos, 2, &seq);
		if (r == 0) {
			r = WaitOn(d, rings[k].ip, 0, seq, 10.0);
		}
		int good = 0;
		for (int i = 0; i < 100; i++) {
			good += dst[i] == src[i];
		}
		printf("  %-9s: %s after %.1f ms, %d/100 dwords back, dst[0] 0x%08x, context state 0x%" PRIx64 "\n",
		       rings[k].name, r == 0 ? "signaled" : (r == 1 ? "TIMEOUT" : ErrStr(r)), (NowSec() - t0) * 1e3, good,
		       dst[0], (uint64_t)ContextState(d));
		ok = r == 0;
	}
	BoDestroy(d, &buf);
	BoDestroy(d, &gds);
	return ok;
}

// Job A on compute ring 1 waits (WAIT_REG_MEM) for a memory flag; 100 ms later job B on
// compute ring 0 writes it (WRITE_DATA). Checks that queues can wait on each other via memory.
static bool DangerCrossQueue(Device* d, Common* c) {
	printf("== danger crossqueue: ring 1 waits on a flag, ring 0 writes it 100 ms later\n");
	Bo  flag;
	Bo  ib2;
	int r = MemCreate(d, &flag, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &flag, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r == 0) {
		r = MemCreate(d, &ib2, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &ib2, 0x0082000000ull, 0);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &flag);
		BoDestroy(d, &ib2);
		return false;
	}
	*(volatile uint32_t*)flag.cpu = 0;

	Ib wait = {.n = 0};
	IbPush(&wait, Pkt3(0x3cu, 6));        // WAIT_REG_MEM
	IbPush(&wait, 3u | (1u << 4));        // function == , memory space
	IbPush(&wait, (uint32_t)flag.va);
	IbPush(&wait, (uint32_t)(flag.va >> 32));
	IbPush(&wait, 1u);                    // reference
	IbPush(&wait, 0xffffffffu);           // mask
	IbPush(&wait, 4u);                    // poll interval
	IbPad(&wait);

	Ib write = {.n = 0};
	IbPush(&write, Pkt3(0x37u, 4));       // WRITE_DATA
	IbPush(&write, (5u << 8) | (1u << 20)); // dst_sel memory, write confirm
	IbPush(&write, (uint32_t)flag.va);
	IbPush(&write, (uint32_t)(flag.va >> 32));
	IbPush(&write, 1u);
	IbPad(&write);

	Bo*      bos[]  = {&flag};
	uint64_t seq_a  = 0;
	uint64_t seq_b  = 0;
	double   t0     = NowSec();
	r               = SubmitOn(d, AMDGPU_HW_IP_COMPUTE, 1, &ib2, &wait, bos, 1, &seq_a);
	int      a_mid  = r == 0 ? WaitOn(d, AMDGPU_HW_IP_COMPUTE, 1, seq_a, 0.1) : r;
	int      rb     = SubmitOn(d, AMDGPU_HW_IP_COMPUTE, 0, &c->ib, &write, bos, 1, &seq_b);
	int      wb     = rb == 0 ? WaitOn(d, AMDGPU_HW_IP_COMPUTE, 0, seq_b, 15.0) : rb;
	int      wa     = r == 0 ? WaitOn(d, AMDGPU_HW_IP_COMPUTE, 1, seq_a, 15.0) : r;
	double   t      = NowSec() - t0;
	int64_t  state  = ContextState(d);
	printf("  A pending before B: %s; B: %s; A: %s after %.1f ms; flag %u; context state 0x%" PRIx64 "\n",
	       a_mid == 1 ? "yes" : (a_mid == 0 ? "NO (finished early)" : ErrStr(a_mid)),
	       wb == 0 ? "signaled" : (wb == 1 ? "TIMEOUT" : ErrStr(wb)),
	       wa == 0 ? "signaled" : (wa == 1 ? "TIMEOUT" : ErrStr(wa)), t * 1e3, *(volatile uint32_t*)flag.cpu,
	       (uint64_t)state);
	BoDestroy(d, &flag);
	BoDestroy(d, &ib2);
	return wa == 0 && wb == 0 && state == 0;
}

#define UCONFIG_BASE 0x30000u

static void IbCopyRegToMem(Ib* ib, uint32_t reg, uint64_t va) {
	IbPush(ib, Pkt3(0x40u, 5));                  // COPY_DATA
	IbPush(ib, 0u | (5u << 8) | (1u << 20));     // src register, dst memory, write confirm
	IbPush(ib, reg >> 2);
	IbPush(ib, 0);
	IbPush(ib, (uint32_t)va);
	IbPush(ib, (uint32_t)(va >> 32));
}

static void IbSetUconfig(Ib* ib, uint32_t reg, uint32_t value) {
	IbPush(ib, Pkt3(0x79u, 2));                  // SET_UCONFIG_REG
	IbPush(ib, (reg - UCONFIG_BASE) / 4);
	IbPush(ib, value);
}

// The register-table packets libSceAgc writes (0x63 SH, 0x9F context, 0x64 UCONFIG), bit for bit AMD's
// LOAD_SH_REG_INDEX / LOAD_CONTEXT_REG_INDEX / LOAD_UCONFIG_REG_INDEX: does the BC-250's microcode run
// them? One packet on the gfx ring with a one-pair table {offset, value} into a harmless register,
// read back with COPY_DATA, and the register put back with a plain SET. `which`: "sh", "ctx", "uc".
// An opcode the microcode lacks stops the ring (with gpu_recovery=0 until a reboot).
static bool DangerRegIndex(Device* d, Common* c, const char* which) {
	struct Case {
		const char* name;
		uint32_t    op;
		uint32_t    set_op;
		uint32_t    base; // byte address of the register space
		uint32_t    reg;  // byte address of the register
		const char* reg_name;
	};
	static const struct Case cases[] = {
	    {"sh", 0x63u, 0x76u, 0xb000u, 0xb030u, "SPI_SHADER_USER_DATA_PS_0"},
	    {"ctx", 0x9fu, 0x69u, 0x28000u, 0x28030u, "PA_SC_SCREEN_SCISSOR_TL"},
	    {"uc", 0x64u, 0x79u, 0x30000u, 0x30e0cu, "TEXTURE_GRADIENT_FACTORS"}, // a COPY_DATA read of VGT_OBJECT_ID stopped the ring
	};
	const struct Case* k = NULL;
	for (size_t i = 0; i < ARRAY_SIZE(cases); i++) {
		if (strcmp(cases[i].name, which) == 0) {
			k = &cases[i];
		}
	}
	if (k == NULL) {
		fprintf(stderr, "regindex case: sh, ctx or uc\n");
		return false;
	}
	printf("== danger regindex-%s: packet 0x%02x (one pair into %s) on the gfx ring\n", k->name, k->op, k->reg_name);
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		return false;
	}
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem};
	m[0] = m[1] = m[2] = 0xeeeeeeeeu;

	// Pass 1: the register's value now (a plain read).
	Ib ib = {.n = 0};
	IbCopyRegToMem(&ib, k->reg, mem.va);
	IbPad(&ib);
	uint64_t seq = 0;
	r            = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
	int     w    = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 10.0) : r;
	int64_t st   = ContextState(d);
	if (w != 0 || st != 0) {
		printf("  read: %s ctx 0x%" PRIx64 "\n", w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st);
		BoDestroy(d, &mem);
		return false;
	}
	const uint32_t orig  = m[0];
	const uint32_t value = orig ^ 0x00a5005au;

	// The table at +0x100: {offset from the register space, value}.
	m[0x40] = (k->reg - k->base) / 4;
	m[0x41] = value;
	__sync_synchronize();
	const uint64_t table = mem.va + 0x100;

	// Pass 2: the table packet, read back, put the old value back, read again.
	ib.n = 0;
	IbPush(&ib, Pkt3(k->op, 4));
	IbPush(&ib, (uint32_t)table & ~3u); // INDEX 0: direct address
	IbPush(&ib, (uint32_t)(table >> 32));
	IbPush(&ib, 0x80000000u);           // DATA_FORMAT offset_and_data, REG_OFFSET 0
	IbPush(&ib, 1u);                    // one pair
	IbCopyRegToMem(&ib, k->reg, mem.va + 4);
	IbPush(&ib, Pkt3(k->set_op, 2));
	IbPush(&ib, (k->reg - k->base) / 4);
	IbPush(&ib, orig);
	IbCopyRegToMem(&ib, k->reg, mem.va + 8);
	IbPad(&ib);
	r  = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
	w  = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 10.0) : r;
	st = ContextState(d);
	if (w != 0 || st != 0) {
		printf("  packet 0x%02x: %s ctx 0x%" PRIx64 " (read 0x%08x before)\n", k->op, w == 1 ? "TIMEOUT" : ErrStr(w),
		       (uint64_t)st, orig);
		BoDestroy(d, &mem);
		return false;
	}
	const char* verdict = m[1] == value ? "the table was loaded" : m[1] == orig ? "register unchanged" : "readback differs";
	printf("  %s: before 0x%08x, after packet 0x%08x (want 0x%08x: %s), restored 0x%08x\n", k->reg_name, orig, m[1],
	       value, verdict, m[2]);
	const bool loaded = m[1] == value;
	BoDestroy(d, &mem);
	return loaded;
}

// P2 of the direct-execution plan: libSceAgc's internal-data tags, SET_UCONFIG_REG of
// SQ_THREAD_TRACE_USERDATA_2 (0x342) around waits and multi draws, as the CP would get them verbatim.
// DCB form on the gfx ring (header 0xc0017904: bit 2 set), ACB form on a compute ring (0x342 and the
// profiler's 0x343), each followed by a WRITE_DATA that shows the ring went on. No readback.
// Result on BC-250: the gfx ring runs them; the compute ring (amdgpu's MEC queue) stops.
static bool DangerTag342(Device* d, Common* c) {
	printf("== danger tag342: SET_UCONFIG_REG of SQ_THREAD_TRACE_USERDATA_2/3 from user IBs\n");
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		return false;
	}
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem};
	bool               ok    = true;
	for (int k = 0; k < 2 && ok; k++) {
		const bool     gfx    = k == 0;
		const uint32_t ip     = gfx ? AMDGPU_HW_IP_GFX : AMDGPU_HW_IP_COMPUTE;
		const uint32_t magic  = 0x7a690000u | (uint32_t)k;
		const uint32_t header = gfx ? 0xc0017904u : Pkt3(0x79u, 2);
		m[k]                  = 0;
		__sync_synchronize();
		Ib ib = {.n = 0};
		IbPush(&ib, header);
		IbPush(&ib, 0x342u);
		IbPush(&ib, 0xc8010000u); // AGC_WAIT_USER_DATA_BEGIN_32_TAG
		IbPush(&ib, header);
		IbPush(&ib, gfx ? 0x342u : 0x343u);
		IbPush(&ib, 0xc8000000u); // AGC_WAIT_USER_DATA_END_TAG
		IbPush(&ib, gfx ? Pkt3Gfx(0x37u, 4) : Pkt3(0x37u, 4));
		IbPush(&ib, (5u << 8) | (1u << 20));
		IbPush(&ib, (uint32_t)(mem.va + 4u * k));
		IbPush(&ib, (uint32_t)((mem.va + 4u * k) >> 32));
		IbPush(&ib, magic);
		IbPad(&ib);
		uint64_t seq = 0;
		r            = SubmitOn(d, ip, 0, &c->ib, &ib, bos, 1, &seq);
		int     w    = r == 0 ? WaitOn(d, ip, 0, seq, 1.0) : r;
		int64_t st   = ContextState(d);
		printf("  %s ring: %s, ctx 0x%" PRIx64 ", WRITE_DATA after the tags %s\n", gfx ? "gfx" : "compute",
		       w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st, m[k] == magic ? "landed" : "missing");
		if (w != 0 || st != 0) {
			_exit(3); // freeing buffers would wait on the stopped ring
		}
		ok = m[k] == magic;
	}
	BoDestroy(d, &mem);
	return ok;
}

// The guest's own GPU waits (libSceAgc's AgcDcbWaitRegMem: WAIT_REG_MEM and WAIT_REG_MEM_64, ENGINE_SEL ME
// or PFP, any compare function) as written, on the gfx ring: whether the BC-250's CP runs them as the PS5's
// does, so the translator can pass them through. Each step is one job, the wait then a WRITE_DATA marker;
// the first one that does not finish stops the run. Memory is UC so a value the CPU writes later is seen.
static bool DangerWaitForms(Device* d, Common* c) {
	printf("== danger waitforms: guest WAIT_REG_MEM / WAIT_REG_MEM_64 forms on the gfx ring\n");
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0710000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		return false;
	}
	// CONTROL as AgcDcbWaitRegMem builds it: function [2:0], MEM_SPACE memory (bit 4), ENGINE_SEL [9:8].
	static const struct {
		const char* name;
		bool        wide;
		uint32_t    control;
		uint64_t    value, ref, mask;
		int         release_ms; // the CPU writes `ref` into the label this long after submitting (0: already there)
	} steps[] = {
	    {"32-bit EQUAL, ME, met", false, 0x13u, 5, 5, 0xffffffffu, 0},
	    {"32-bit EQUAL, PFP, met", false, 0x113u, 6, 6, 0xffffffffu, 0},
	    {"32-bit GREATER_EQUAL, ME, met", false, 0x15u, 9, 7, 0xffffffffu, 0},
	    {"64-bit EQUAL, ME, met", true, 0x13u, 0x100000005ull, 0x100000005ull, ~0ull, 0},
	    {"64-bit GREATER_EQUAL, ME, met only in 64 bits", true, 0x15u, 0x100000000ull, 0xffffffffull, ~0ull, 0},
	    {"64-bit EQUAL, PFP, met", true, 0x113u, 0x200000007ull, 0x200000007ull, ~0ull, 0},
	    {"64-bit EQUAL, ME, released by the CPU after 100 ms", true, 0x13u, 0x300000000ull, 0x300000008ull, ~0ull, 100},
	    {"32-bit EQUAL, PFP, released by the CPU after 100 ms", false, 0x113u, 0, 9, 0xffffffffu, 100},
	};
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem};
	bool               ok    = true;
	const char*        one   = getenv("WAITFORMS_STEP"); // one step per run: a wait the CP does not finish stops the ring
	for (size_t k = 0; k < ARRAY_SIZE(steps) && ok; k++) {
		if (one != NULL && (size_t)atoi(one) != k) {
			continue;
		}
		const uint64_t label  = mem.va + 0x100 + 0x40 * k;
		const uint64_t marker = mem.va + 0x800 + 4 * k;
		volatile uint64_t* lv = (volatile uint64_t*)((uint8_t*)mem.cpu + 0x100 + 0x40 * k);
		*lv                   = steps[k].value;
		m[(0x800 + 4 * k) / 4] = 0;
		__sync_synchronize();
		Ib ib = {.n = 0};
		if (steps[k].wide) {
			IbPush(&ib, Pkt3Gfx(0x93u, 8));
			IbPush(&ib, steps[k].control);
			IbPush(&ib, (uint32_t)label & ~7u);
			IbPush(&ib, (uint32_t)(label >> 32) & 0x3ffffu);
			IbPush(&ib, (uint32_t)steps[k].ref);
			IbPush(&ib, (uint32_t)(steps[k].ref >> 32));
			IbPush(&ib, (uint32_t)steps[k].mask);
			IbPush(&ib, (uint32_t)(steps[k].mask >> 32));
			IbPush(&ib, 4);
		} else {
			IbPush(&ib, Pkt3Gfx(0x3cu, 6));
			IbPush(&ib, steps[k].control);
			IbPush(&ib, (uint32_t)label & ~3u);
			IbPush(&ib, (uint32_t)(label >> 32) & 0x3ffffu);
			IbPush(&ib, (uint32_t)steps[k].ref);
			IbPush(&ib, (uint32_t)steps[k].mask);
			IbPush(&ib, 4);
		}
		IbPush(&ib, Pkt3Gfx(0x37u, 4)); // WRITE_DATA, memory, write confirm
		IbPush(&ib, (5u << 8) | (1u << 20));
		IbPush(&ib, (uint32_t)marker);
		IbPush(&ib, (uint32_t)(marker >> 32));
		IbPush(&ib, 0x3a170000u | (uint32_t)k);
		IbPad(&ib);
		uint64_t seq = 0;
		r            = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
		if (r == 0 && steps[k].release_ms != 0) {
			usleep((useconds_t)steps[k].release_ms * 1000u);
			const uint32_t early = m[(0x800 + 4 * k) / 4];
			*lv                  = steps[k].ref;
			__sync_synchronize();
			if (early != 0) {
				printf("  %s: the marker landed before the CPU released the label\n", steps[k].name);
				ok = false;
			}
		}
		int     w  = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 1.0) : r;
		int64_t st = ContextState(d);
		const bool landed = m[(0x800 + 4 * k) / 4] == (0x3a170000u | (uint32_t)k);
		printf("  %s: %s, ctx 0x%" PRIx64 ", marker %s\n", steps[k].name, w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st,
		       landed ? "landed" : "missing");
		fflush(stdout);
		if (w != 0 || st != 0) {
			_exit(3); // freeing buffers would wait on the stopped ring
		}
		ok = ok && landed;
	}
	BoDestroy(d, &mem);
	return ok;
}

// P5 of the direct-execution plan: does the CPU see a label that a RELEASE_MEM wrote with only the
// guest's own event (no GLM/GL2 write-back, which the translator adds today)? Per GPU mapping type of
// cached GTT: the gfx ring writes the label (BOTTOM_OF_PIPE_TS, DATA_SEL 1, no cache actions), then
// waits on a dword the CPU writes once it saw the label or gave up after 50 ms; the job's own fence
// (which writes back GL2) comes only after that.
static bool DangerLabelVis(Device* d, Common* c) {
	printf("== danger labelvis: CPU visibility of a RELEASE_MEM label without cache write-back\n");
	static const struct {
		const char* name;
		uint32_t    mtype;
	} kinds[] = {{"default", 0}, {"nc", AMDGPU_VM_MTYPE_NC}, {"uc", AMDGPU_VM_MTYPE_UC}, {"wc", AMDGPU_VM_MTYPE_WC}, {"cc", AMDGPU_VM_MTYPE_CC}};
	const char* only = getenv("LABELVIS_KIND"); // one mapping type per run: a wait that never sees the CPU stops the ring
	for (size_t k = 0; k < ARRAY_SIZE(kinds); k++) {
		if (only != NULL && strcmp(only, kinds[k].name) != 0) {
			continue;
		}
		Bo  mem;
		int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
		if (r == 0) {
			r = BoBind(d, &mem, 0x0700000000ull + k * 0x100000ull, kinds[k].mtype);
		}
		if (r != 0) {
			printf("  %-7s: setup failed: %s\n", kinds[k].name, ErrStr(r));
			BoDestroy(d, &mem);
			continue;
		}
		volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
		Bo*                bos[] = {&mem};
		m[0] = 0;          // the label
		m[16] = 0;         // the CPU's go
		__sync_synchronize();
		Ib ib = {.n = 0};
		// LABELVIS_EVENT: the event dword (default BOTTOM_OF_PIPE_TS, no GCR actions); LABELVIS_DST=1:
		// the label to L2 (DST_SEL 1), as libSceAgc writes them, instead of memory.
		const char* ev  = getenv("LABELVIS_EVENT");
		const char* dst = getenv("LABELVIS_DST");
		IbPush(&ib, Pkt3Gfx(0x49u, 7)); // RELEASE_MEM
		IbPush(&ib, ev != NULL ? (uint32_t)strtoul(ev, NULL, 0) : 0x28u | (5u << 8));
		// LABELVIS_WIDE=1: a 64-bit label (DATA_SEL 2), its high dword 0x5eed0000 + kind.
		const bool wide = getenv("LABELVIS_WIDE") != NULL;
		IbPush(&ib, ((wide ? 2u : 1u) << 29) | ((dst != NULL ? (uint32_t)strtoul(dst, NULL, 0) & 3u : 0u) << 16)); // no interrupt
		IbPush(&ib, (uint32_t)mem.va);
		IbPush(&ib, (uint32_t)(mem.va >> 32));
		IbPush(&ib, 0x1abe1000u + (uint32_t)k);
		IbPush(&ib, wide ? 0x5eed0000u + (uint32_t)k : 0u);
		IbPush(&ib, 0);
		IbPush(&ib, Pkt3Gfx(0x3cu, 6)); // WAIT_REG_MEM: memory, equal, ME
		IbPush(&ib, 3u | (1u << 4));
		IbPush(&ib, (uint32_t)(mem.va + 64));
		IbPush(&ib, (uint32_t)((mem.va + 64) >> 32));
		IbPush(&ib, 1u);
		IbPush(&ib, 0xffffffffu);
		IbPush(&ib, 4u);
		IbPad(&ib);
		uint64_t seq = 0;
		r            = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
		if (r != 0) {
			printf("  %-7s: submit failed: %s\n", kinds[k].name, ErrStr(r));
			BoDestroy(d, &mem);
			continue;
		}
		const double t0   = NowSec();
		double       seen = -1;
		while (NowSec() - t0 < 0.05) {
			if (m[0] == 0x1abe1000u + (uint32_t)k) {
				seen = NowSec() - t0;
				break;
			}
		}
		m[16] = 1;
		__sync_synchronize();
		int     w  = WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 1.0);
		int64_t st = ContextState(d);
		if (seen >= 0) {
			printf("  %-7s: label seen by the CPU after %.1f us (job %s), dwords 0x%08x 0x%08x\n", kinds[k].name, seen * 1e6,
			       w == 1 ? "TIMEOUT" : ErrStr(w), m[0], m[1]);
		} else {
			printf("  %-7s: label NOT seen in 50 ms (job %s, after the fence 0x%08x)\n", kinds[k].name,
			       w == 1 ? "TIMEOUT" : ErrStr(w), m[0]);
		}
		if (w != 0 || st != 0) {
			_exit(3); // freeing buffers would wait on the stopped ring
		}
		BoDestroy(d, &mem);
	}
	return true;
}

// --awake SEC: one wave spins on the compute ring for SEC seconds (the `long` shader, in buffers of its
// own) while a danger case runs, so the graphics block never idles into GFXOFF. Probe jobs that read
// the GPU clock stopped the idle gfx ring, while Kyty's frames, with the GPU always busy, read it
// thousands of times a second.
typedef struct {
	Bo       prog, code, ib;
	uint64_t seq;
	bool     on;
} Awake;

static bool AwakeStart(Device* d, Awake* a, double sec) {
	memset(a, 0, sizeof(*a));
	int r = MemCreate(d, &a->prog, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &a->prog, 0x0780000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r == 0) {
		r = MemCreate(d, &a->code, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &a->code, 0x0781000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &a->ib, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &a->ib, 0x0782000000ull, 0);
	}
	if (r != 0) {
		printf("awake: setup failed: %s\n", ErrStr(r));
		return false;
	}
	memcpy(a->code.cpu, k_long_w64, sizeof(k_long_w64));
	FlushCpu(a->code.cpu, sizeof(k_long_w64));
	const uint32_t ticks = (uint32_t)(sec * d->info.gpu_counter_freq * 1000.0);
	uint32_t       ud[3] = {(uint32_t)a->prog.va, (uint32_t)(a->prog.va >> 32), ticks};
	Dispatch       p     = {a->code.va, 64, 8, 3, false, 64, 1, ud};
	Ib             ib    = {.n = 0};
	IbDispatch(&ib, &p);
	IbPad(&ib);
	Bo* bos[] = {&a->prog, &a->code};
	r         = SubmitOn(d, AMDGPU_HW_IP_COMPUTE, 0, &a->ib, &ib, bos, 2, &a->seq);
	if (r != 0) {
		printf("awake: submit failed: %s\n", ErrStr(r));
		return false;
	}
	volatile uint32_t* pv = (volatile uint32_t*)a->prog.cpu;
	const double       t0 = NowSec();
	while (pv[0] == 0 && pv[1] == 0 && NowSec() - t0 < 1.0) {
	}
	printf("awake: a compute wave spins for %.1f s (%s)\n", sec, pv[0] || pv[1] ? "running" : "not seen running");
	a->on = true;
	return true;
}

static void AwakeStop(Device* d, Awake* a) {
	if (!a->on) {
		return;
	}
	Wait(d, a->seq, 20.0);
	BoDestroy(d, &a->ib);
	BoDestroy(d, &a->code);
	BoDestroy(d, &a->prog);
}

// Reads the GPU clock on the gfx ring the two ways Kyty's frames do (COPY_DATA SRC_SEL 9; RELEASE_MEM
// DATA_SEL 3 with CACHE_FLUSH_AND_INV_TS) and the probe's first try (RELEASE_MEM DATA_SEL 3 with
// BOTTOM_OF_PIPE_TS). Each stopped the idle ring from the probe; run with and without --awake.
static bool DangerClock(Device* d, Common* c) {
	printf("== danger clock: GPU clock reads on the gfx ring\n");
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		return false;
	}
	volatile uint64_t* t     = (volatile uint64_t*)mem.cpu;
	Bo*                bos[] = {&mem};
	for (int k = 0; k < 3; k++) {
		t[0] = t[1] = 0;
		__sync_synchronize();
		Ib ib = {.n = 0};
		for (int i = 0; i < 2; i++) {
			const uint64_t va = mem.va + 8u * i;
			if (k == 0) {
				IbPush(&ib, Pkt3Gfx(0x40u, 5));
				IbPush(&ib, 9u | (5u << 8) | (1u << 16) | (1u << 20));
				IbPush(&ib, 0);
				IbPush(&ib, 0);
			} else {
				IbPush(&ib, Pkt3Gfx(0x49u, 7));
				IbPush(&ib, k == 1 ? 0x14u | (5u << 8) | (1u << 12) | (1u << 13) | (1u << 14) | (1u << 15) | (1u << 21) |
				                         (1u << 22) | (3u << 25)
				                   : 0x28u | (5u << 8));
				IbPush(&ib, 3u << 29);
			}
			IbPush(&ib, (uint32_t)va);
			IbPush(&ib, (uint32_t)(va >> 32));
			if (k != 0) {
				IbPush(&ib, 0);
				IbPush(&ib, 0);
				IbPush(&ib, 0);
			}
		}
		IbPad(&ib);
		uint64_t seq = 0;
		r            = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
		int w        = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 1.0) : r;
		static const char* names[] = {"COPY_DATA SRC_SEL 9", "RELEASE_MEM CACHE_FLUSH_AND_INV_TS DATA_SEL 3",
		                              "RELEASE_MEM BOTTOM_OF_PIPE_TS DATA_SEL 3"};
		printf("  %-46s: %s, clocks %" PRIu64 " %" PRIu64 "\n", names[k], w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)t[0],
		       (uint64_t)t[1]);
		if (w != 0 || ContextState(d) != 0) {
			_exit(3); // freeing buffers would wait on the stopped ring
		}
	}
	BoDestroy(d, &mem);
	return true;
}

// The register-table packets without register reads: a COPY_DATA read of a UCONFIG register stops
// the gfx ring from a user IB (VGT_OBJECT_ID, TEXTURE_GRADIENT_FACTORS; GFX kept awake or not), so a
// one-pair table loads a register a user IB may write and a WRITE_DATA after the packet shows the CP
// ran it and went on. "uc342": 0x64 into SQ_THREAD_TRACE_USERDATA_2 (written fine: --danger tag342);
// "sh-nr": 0x63 into SPI_SHADER_USER_DATA_PS_0; "ctx-nr": 0x9F into PA_SC_SCREEN_SCISSOR_TL.
// Result on BC-250: 0x64 runs, with or without --awake, and the ring stays healthy.
static bool DangerRegIndexNoRead(Device* d, Common* c, uint32_t op, uint32_t offset, const char* reg_name) {
	printf("== danger regindex no read: packet 0x%02x loading %s, then WRITE_DATA\n", op, reg_name);
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		return false;
	}
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem};
	m[0]                     = 0;
	m[0x40]                  = offset;      // {offset from the register space, value}
	m[0x41]                  = 0xc8010000u; // a libSceAgc tag, or any user data
	__sync_synchronize();
	const uint64_t table = mem.va + 0x100;
	Ib             ib    = {.n = 0};
	IbPush(&ib, Pkt3Gfx(op, 4));
	IbPush(&ib, (uint32_t)table & ~3u); // INDEX 0: direct address
	IbPush(&ib, (uint32_t)(table >> 32));
	IbPush(&ib, 0x80000000u); // DATA_FORMAT offset_and_data, REG_OFFSET 0
	IbPush(&ib, 1u);          // one pair
	IbPush(&ib, Pkt3Gfx(0x37u, 4));
	IbPush(&ib, (5u << 8) | (1u << 20));
	IbPush(&ib, (uint32_t)mem.va);
	IbPush(&ib, (uint32_t)(mem.va >> 32));
	IbPush(&ib, 0x64646464u);
	IbPad(&ib);
	uint64_t seq = 0;
	r            = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
	int     w    = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 1.0) : r;
	int64_t st   = ContextState(d);
	printf("  packet 0x%02x: %s, ctx 0x%" PRIx64 ", WRITE_DATA after it %s\n", op, w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st,
	       m[0] == 0x64646464u ? "landed" : "missing");
	if (w != 0 || st != 0) {
		_exit(3); // freeing buffers would wait on the stopped ring
	}
	const bool ok = m[0] == 0x64646464u;
	BoDestroy(d, &mem);
	return ok;
}

// UCONFIG registers that PS5 command buffers write (per KytyPS5's PM4 handlers) but RADV never
// does, plus one RADV-written control per ring. Each register is read, written with its low bit
// flipped, read back and restored within one IB, so other processes never see the change.
static bool DangerUconfig(Device* d, Common* c) {
	printf("== danger uconfig: user-mode reads/writes of PS5 UCONFIG registers RADV never writes\n");
	static const struct {
		uint32_t    ip;
		uint32_t    reg;
		const char* name;
	} regs[] = {
	    {AMDGPU_HW_IP_COMPUTE, 0x30e00u, "TA_CS_BC_BASE_ADDR (control)"},
	    {AMDGPU_HW_IP_COMPUTE, 0x31074u, "GDS_OA_CNTL"},
	    {AMDGPU_HW_IP_COMPUTE, 0x31078u, "GDS_OA_COUNTER"},
	    {AMDGPU_HW_IP_COMPUTE, 0x3107cu, "GDS_OA_ADDRESS"},
	    {AMDGPU_HW_IP_GFX, 0x30908u, "VGT_PRIMITIVE_TYPE (control)"},
	    {AMDGPU_HW_IP_GFX, 0x30920u, "VGT_OBJECT_ID"},
	    {AMDGPU_HW_IP_GFX, 0x30e0cu, "TEXTURE_GRADIENT_FACTORS"},
	    {AMDGPU_HW_IP_GFX, 0x30e10u, "TEXTURE_GRADIENT_CONTROL"},
	};
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		return false;
	}
	volatile uint32_t* m  = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem};
	bool               all_ok = true;
	for (size_t k = 0; k < ARRAY_SIZE(regs) && all_ok; k++) {
		const char* ring = regs[k].ip == AMDGPU_HW_IP_GFX ? "gfx" : "comp";
		m[0] = m[1] = m[2] = 0xeeeeeeeeu;

		// Pass 1: read only.
		Ib ib = {.n = 0};
		IbCopyRegToMem(&ib, regs[k].reg, mem.va);
		IbPad(&ib);
		uint64_t seq = 0;
		r            = SubmitOn(d, regs[k].ip, 0, &c->ib, &ib, bos, 1, &seq);
		int w        = r == 0 ? WaitOn(d, regs[k].ip, 0, seq, 15.0) : r;
		int64_t st   = ContextState(d);
		if (w != 0 || st != 0) {
			printf("  %-4s 0x%05x %-30s read: %s ctx 0x%" PRIx64 "\n", ring, regs[k].reg, regs[k].name,
			       w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st);
			all_ok = false;
			break;
		}
		uint32_t orig = m[0];

		// Pass 2: write flipped value, read back, restore, read again.
		ib.n = 0;
		IbSetUconfig(&ib, regs[k].reg, orig ^ 1u);
		IbCopyRegToMem(&ib, regs[k].reg, mem.va + 4);
		IbSetUconfig(&ib, regs[k].reg, orig);
		IbCopyRegToMem(&ib, regs[k].reg, mem.va + 8);
		IbPad(&ib);
		r  = SubmitOn(d, regs[k].ip, 0, &c->ib, &ib, bos, 1, &seq);
		w  = r == 0 ? WaitOn(d, regs[k].ip, 0, seq, 15.0) : r;
		st = ContextState(d);
		if (w != 0 || st != 0) {
			printf("  %-4s 0x%05x %-30s read 0x%08x, write: %s ctx 0x%" PRIx64 "\n", ring, regs[k].reg,
			       regs[k].name, orig, w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st);
			all_ok = false;
			break;
		}
		const char* verdict = m[1] == (orig ^ 1u) ? "write took effect"
		                      : m[1] == orig      ? "write IGNORED"
		                                          : "readback differs";
		printf("  %-4s 0x%05x %-30s orig 0x%08x, after write 0x%08x (%s), restored 0x%08x\n", ring, regs[k].reg,
		       regs[k].name, orig, m[1], verdict, m[2]);
		fflush(stdout);
	}
	BoDestroy(d, &mem);
	return all_ok;
}

static float HalfToFloat(uint16_t h) {
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
	uint32_t exp  = (h >> 10) & 0x1fu;
	uint32_t man  = h & 0x3ffu;
	uint32_t bits;
	if (exp == 0) {
		float f = (float)man * (1.0f / 16777216.0f); // subnormal: man * 2^-24
		memcpy(&bits, &f, 4);
		bits |= sign;
	} else if (exp == 31) {
		bits = sign | 0x7f800000u | (man << 13);
	} else {
		bits = sign | ((exp + 112u) << 23) | (man << 13);
	}
	float f;
	memcpy(&f, &bits, 4);
	return f;
}

// Normal halves only: sign, exponent in [11, 18] (|x| in [2^-4, 2^4)), random mantissa.
static uint16_t RandomHalf(uint32_t* state) {
	*state       = *state * 1664525u + 1013904223u;
	uint32_t r   = *state >> 8;
	uint32_t exp = 11u + (r & 7u);
	return (uint16_t)(((r >> 3) & 1u) << 15 | exp << 10 | ((r >> 4) & 0x3ffu));
}

// Checks the v_dot2c_f32_f16 replacement (two v_fma_mix_f32) bit-exactly against
// fma(a.hi, b.hi, fma(a.lo, b.lo, acc)) and reports its distance from a single-rounded dot.
static void ProbeDot2Emu(Device* d, Common* c) {
	printf("== dot2emu: v_dot2c_f32_f16 replacement with two v_fma_mix_f32, 64 random lanes\n");
	Bo  in;
	Bo  out;
	int r = MemCreate(d, &in, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &in, 0x0300000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &out, MEM_GTT, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &out, 0x0400000000ull, 0);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &in);
		BoDestroy(d, &out);
		return;
	}
	uint32_t* iv    = (uint32_t*)in.cpu;
	uint32_t  state = 12345u;
	float     ref_fma[64];
	double    ref_exact[64];
	for (int i = 0; i < 64; i++) {
		uint16_t a0 = RandomHalf(&state), a1 = RandomHalf(&state);
		uint16_t b0 = RandomHalf(&state), b1 = RandomHalf(&state);
		state       = state * 1664525u + 1013904223u;
		float acc   = (float)((int32_t)(state >> 8) % 20000) / 100.0f;
		iv[i * 3 + 0] = (uint32_t)a1 << 16 | a0;
		iv[i * 3 + 1] = (uint32_t)b1 << 16 | b0;
		memcpy(&iv[i * 3 + 2], &acc, 4);
		float fa0 = HalfToFloat(a0), fa1 = HalfToFloat(a1), fb0 = HalfToFloat(b0), fb1 = HalfToFloat(b1);
		ref_fma[i]   = __builtin_fmaf(fa1, fb1, __builtin_fmaf(fa0, fb0, acc));
		ref_exact[i] = (double)acc + (double)fa0 * fb0 + (double)fa1 * fb1;
	}
	LoadCode(c, k_dot2emu_w64, sizeof(k_dot2emu_w64));
	uint32_t ud[4] = {(uint32_t)in.va, (uint32_t)(in.va >> 32), (uint32_t)out.va, (uint32_t)(out.va >> 32)};
	Dispatch p     = {c->code.va, 64, 8, 4, false, 64, 1, ud};
	Ib       ib    = {.n = 0};
	IbDispatch(&ib, &p);
	IbPad(&ib);
	Bo* bos[] = {&in, &out, &c->code};
	if (RunIb(d, &c->ib, &ib, bos, 3, NULL)) {
		const float* ov        = (const float*)out.cpu;
		int          exact_fma = 0;
		int          max_ulp   = 0;
		for (int i = 0; i < 64; i++) {
			exact_fma += memcmp(&ov[i], &ref_fma[i], 4) == 0;
			float    single = (float)ref_exact[i];
			int32_t  a, b;
			memcpy(&a, &ov[i], 4);
			memcpy(&b, &single, 4);
			int ulp = a > b ? a - b : b - a;
			max_ulp = ulp > max_ulp ? ulp : max_ulp;
		}
		printf("  bit-exact vs fma(fma()) reference: %d/64; max distance from single-rounded dot: %d ulp\n",
		       exact_fma, max_ulp);
		printf("  lane 0: got %.9g, reference %.9g\n", ov[0], ref_fma[0]);
	}
	BoDestroy(d, &in);
	BoDestroy(d, &out);
}

// Runs a raw shader binary (e.g. one produced by patch_blob) as one 64-lane wave64 group whose
// only input is the destination address in s[0:1], and prints the first two lanes' 4 dwords.
static bool RunBlob(Device* d, Common* c, const char* path, uint32_t vgprs) {
	printf("== blob: %s (vgprs %u)\n", path, vgprs);
	FILE* f = fopen(path, "rb");
	if (f == NULL) {
		printf("  open failed: %s\n", strerror(errno));
		return false;
	}
	static uint8_t code[48 * 1024];
	size_t         size = fread(code, 1, sizeof(code), f);
	fclose(f);
	if (size == 0 || (size & 3u) != 0) {
		printf("  bad size %zu\n", size);
		return false;
	}
	uint32_t out[64];
	if (!RunLanes(d, c, code, size, NULL, out, 15.0, vgprs)) {
		return false;
	}
	for (int lane = 0; lane < 2; lane++) {
		printf("  lane %d: 0x%08x 0x%08x 0x%08x 0x%08x\n", lane, out[lane * 4], out[lane * 4 + 1], out[lane * 4 + 2],
		       out[lane * 4 + 3]);
	}
	return true;
}

static void Usage(const char* argv0) {
	printf("usage: %s [--device PATH] [--max-alloc-gib N] [--danger CASE [--awake SEC]] [--blob FILE [--vgprs N]] [probe...]\n",
	       argv0);
	printf("probes: info va alloc dispatch identity bandwidth coherency dmabuf submit ib2 dot2emu (default: all)\n");
	printf("danger cases (owner approval required, run alone): vmfault long dot2c dot2x gds gdsdma crossqueue uconfig tag342 labelvis clock regindex-{sh,ctx,uc,uc342,sh-nr,ctx-nr}\n");
}

int main(int argc, char** argv) {
	setvbuf(stdout, NULL, _IONBF, 0); // a danger case can stop the ring: what it printed must be out by then
	const char* path        = "/dev/dri/renderD128";
	uint64_t    max_alloc   = 4 * GiB;
	const char* danger      = NULL;
	double      awake_sec   = 0;
	const char* blob        = NULL;
	uint32_t    blob_vgprs  = 16;
	const char* selected[16];
	int         num_selected = 0;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
			path = argv[++i];
		} else if (strcmp(argv[i], "--max-alloc-gib") == 0 && i + 1 < argc) {
			max_alloc = strtoull(argv[++i], NULL, 10) * GiB;
		} else if (strcmp(argv[i], "--danger") == 0 && i + 1 < argc) {
			danger = argv[++i];
		} else if (strcmp(argv[i], "--awake") == 0 && i + 1 < argc) {
			awake_sec = strtod(argv[++i], NULL);
		} else if (strcmp(argv[i], "--blob") == 0 && i + 1 < argc) {
			blob = argv[++i];
		} else if (strcmp(argv[i], "--vgprs") == 0 && i + 1 < argc) {
			blob_vgprs = (uint32_t)strtoul(argv[++i], NULL, 10);
		} else if (strcmp(argv[i], "--help") == 0) {
			Usage(argv[0]);
			return 0;
		} else if (num_selected < 16) {
			selected[num_selected++] = argv[i];
		}
	}
	Device d;
	if (DeviceOpen(&d, path) != 0) {
		return 1;
	}
	Common c;
	memset(&c, 0, sizeof(c));
	int r = CommonInit(&d, &c);
	if (r != 0) {
		fprintf(stderr, "common buffers: %s\n", ErrStr(r));
		return 1;
	}
	if (blob != NULL) {
		bool ok = RunBlob(&d, &c, blob, blob_vgprs);
		BoDestroy(&d, &c.code);
		BoDestroy(&d, &c.ib);
		close(d.fd);
		return ok ? 0 : 2;
	}
	if (danger != NULL) {
		bool  ok;
		Awake awake = {0};
		if (awake_sec > 0 && !AwakeStart(&d, &awake, awake_sec)) {
			return 1;
		}
		if (strcmp(danger, "vmfault") == 0) {
			ok = DangerVmFault(&d, &c);
		} else if (strcmp(danger, "long") == 0) {
			ok = DangerLong(&d, &c);
		} else if (strcmp(danger, "dot2c") == 0) {
			ok = DangerDot2c(&d, &c);
		} else if (strcmp(danger, "dot2x") == 0) {
			ok = DangerDot2x(&d, &c);
		} else if (strcmp(danger, "gds") == 0) {
			ok = DangerGds(&d, &c);
		} else if (strcmp(danger, "gdsdma") == 0) {
			ok = DangerGdsDma(&d, &c);
		} else if (strcmp(danger, "crossqueue") == 0) {
			ok = DangerCrossQueue(&d, &c);
		} else if (strcmp(danger, "uconfig") == 0) {
			ok = DangerUconfig(&d, &c);
		} else if (strcmp(danger, "clock") == 0) {
			ok = DangerClock(&d, &c);
		} else if (strcmp(danger, "labelvis") == 0) {
			ok = DangerLabelVis(&d, &c);
		} else if (strcmp(danger, "tag342") == 0) {
			ok = DangerTag342(&d, &c);
		} else if (strcmp(danger, "waitforms") == 0) {
			ok = DangerWaitForms(&d, &c);
		} else if (strcmp(danger, "regindex-uc342") == 0) {
			ok = DangerRegIndexNoRead(&d, &c, 0x64u, 0x342u, "SQ_THREAD_TRACE_USERDATA_2");
		} else if (strcmp(danger, "regindex-sh-nr") == 0) {
			ok = DangerRegIndexNoRead(&d, &c, 0x63u, (0xb030u - 0xb000u) / 4, "SPI_SHADER_USER_DATA_PS_0");
		} else if (strcmp(danger, "regindex-ctx-nr") == 0) {
			ok = DangerRegIndexNoRead(&d, &c, 0x9fu, (0x28030u - 0x28000u) / 4, "PA_SC_SCREEN_SCISSOR_TL");
		} else if (strncmp(danger, "regindex-", 9) == 0) {
			ok = DangerRegIndex(&d, &c, danger + 9);
		} else {
			fprintf(stderr, "unknown danger case '%s'\n", danger);
			return 1;
		}
		AwakeStop(&d, &awake);
		BoDestroy(&d, &c.code);
		BoDestroy(&d, &c.ib);
		close(d.fd);
		return ok ? 0 : 2;
	}
	static const char* const order[] = {"info",      "va",        "alloc",  "dispatch", "identity",
	                                    "bandwidth", "coherency", "dmabuf", "submit",   "ib2", "dot2emu"};
	for (size_t k = 0; k < ARRAY_SIZE(order); k++) {
		const char* n    = order[k];
		bool        want = num_selected == 0;
		for (int s = 0; s < num_selected; s++) {
			want |= strcmp(selected[s], n) == 0;
		}
		if (!want) {
			continue;
		}
		if (strcmp(n, "info") == 0) {
			ProbeInfo(&d);
		} else if (strcmp(n, "va") == 0) {
			ProbeVa(&d);
		} else if (strcmp(n, "alloc") == 0) {
			ProbeAlloc(&d, max_alloc);
		} else if (strcmp(n, "dispatch") == 0) {
			ProbeDispatch(&d, &c);
		} else if (strcmp(n, "identity") == 0) {
			ProbeIdentity(&d, &c);
		} else if (strcmp(n, "bandwidth") == 0) {
			ProbeBandwidth(&d, &c);
		} else if (strcmp(n, "coherency") == 0) {
			ProbeCoherency(&d, &c);
		} else if (strcmp(n, "dmabuf") == 0) {
			ProbeDmabuf(&d);
		} else if (strcmp(n, "submit") == 0) {
			ProbeSubmitLatency(&d, &c);
		} else if (strcmp(n, "ib2") == 0) {
			ProbeIb2(&d, &c);
		} else if (strcmp(n, "dot2emu") == 0) {
			ProbeDot2Emu(&d, &c);
		}
		fflush(stdout);
		int64_t state = ContextState(&d);
		if (state != 0) {
			printf("!! context state 0x%" PRIx64 " after '%s', stopping\n", (uint64_t)state, n);
			return 2;
		}
	}
	BoDestroy(&d, &c.code);
	BoDestroy(&d, &c.ib);
	close(d.fd);
	return 0;
}
