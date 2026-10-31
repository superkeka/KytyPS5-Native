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

// One wave64 loads the dword at VA 0, which has no mapping (a read through a null descriptor's
// base), and stores what it read, then a normal store checks that the GPU still works.
static bool DangerVmFaultRead(Device* d, Common* c) {
	printf("== danger vmfault-read: wave64 load of a dword at unmapped VA 0\n");
	PrintGpuvmFault(d, "before");
	uint32_t out[64];
	// RunLanes waits longer than amdgpu's 5 s lockup timeout, so a hang shows up as a reset.
	bool done = RunLanes(d, c, k_readfault_w64, sizeof(k_readfault_w64), NULL, out, 15.0, 16);
	PrintGpuvmFault(d, "after");
	if (!done) {
		return false;
	}
	printf("  lane 0 read 0x%08x (markers 0x%08x 0x%08x 0x%08x)\n", out[0], out[1], out[2], out[3]);

	LoadCode(c, k_store_w64, sizeof(k_store_w64));
	Bo  dst;
	int r = MemCreate(d, &dst, MEM_GTT, 2 * MiB, NULL);
	if (r == 0) {
		r = BoBind(d, &dst, 0x0200000000ull, 0);
	}
	bool ok = r == 0 && StoreAndCheck(d, c, &dst, &c->code, 64, 64);
	printf("  follow-up store on the same context: %s\n", ok ? "ok" : "FAILED");
	BoDestroy(d, &dst);
	return ok;
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
	    // A compute register from a graphics-type header (shader type 0), as My First GT's DCB dispatches load theirs.
	    {"shc", 0x63u, 0x76u, 0xb000u, 0xb900u, "COMPUTE_USER_DATA_0"},
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
		fprintf(stderr, "regindex case: sh, shc, ctx or uc\n");
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
// Result on BC-250: the gfx ring runs them, the 64-bit wait's two-register begin tag too; the compute
// ring (amdgpu's MEC queue) stops. The gfx step also writes a debug marker's string to 0xc343 with
// WRITE_DATA to a register, as libSceAgcDriver does.
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
	// PROBE_TAG342_GFX=1: the gfx ring only (the compute ring is known to stop).
	const int          rings = getenv("PROBE_TAG342_GFX") != NULL ? 1 : 2;
	for (int k = 0; k < rings && ok; k++) {
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
		if (gfx) {
			// The begin tag of a 64-bit wait: 0x342 and 0x343 in one packet (the tag with address bits
			// 47:32, then the low dword), header 0xc0027904.
			IbPush(&ib, 0xc0027904u);
			IbPush(&ib, 0x342u);
			IbPush(&ib, 0xc8020000u | (uint32_t)((mem.va >> 32) & 0xffffu));
			IbPush(&ib, (uint32_t)mem.va);
			// A push marker's string past its first 4 characters: WRITE_DATA to the register
			// SQ_THREAD_TRACE_USERDATA_3 (0xc343, DST_SEL 0, one address), as libSceAgcDriver writes it.
			IbPush(&ib, 0xc0043704u);
			IbPush(&ib, 0x06010000u);
			IbPush(&ib, 0xc343u);
			IbPush(&ib, 0u);
			IbPush(&ib, 0x6c6c6568u); // "hell"
			IbPush(&ib, 0x0000006fu); // "o"
		}
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

// libSceAgc's conditional branch (AgcCbBranch): INDIRECT_BUFFER with 14 dwords, AMD's COND_INDIRECT_BUFFER.
// MODE 1 if-then, 2 if-then-else; a 64-bit compare (FUNCTION as WAIT_REG_MEM's) of memory against a
// reference under a mask; two IB targets (address, size with the cache policy in 29:28). Each branch
// writes its own marker; a WRITE_DATA after the packet shows the ring went on. PROBE_CONDIB_VALID=0
// leaves bit 23 (VALID, which amdgpu sets on gfx10 IBs) out of the sizes, as libSceAgc writes them.
// Result on BC-250: the CP picks the right branch (with or without VALID) but does not come back from it
// (a chain, not a call): the rest of the IB runs only when no branch is taken. With each branch ending in
// a chain back to the rest of the IB (PROBE_CONDIB_CHAINBACK=1) then and else both come back. MODE 1
// with a false condition took the else target in that run: a translator gives MODE 2 with an else of its own.
static bool DangerCondIb(Device* d, Common* c) {
	const bool valid = getenv("PROBE_CONDIB_VALID") == NULL || getenv("PROBE_CONDIB_VALID")[0] != '0';
	printf("== danger condib: COND_INDIRECT_BUFFER (libSceAgc's branch) on the gfx ring, sizes %s VALID\n",
	       valid ? "with" : "without");
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
	volatile uint32_t* m = (volatile uint32_t*)mem.cpu;
	// The branches: each writes its marker (then: dword 16, else: dword 17), padded to 8 dwords.
	for (int b = 0; b < 2; b++) {
		volatile uint32_t* t = m + (b == 0 ? 0x400 : 0x800);
		const uint64_t     dst = mem.va + 4u * (16u + (uint32_t)b);
		t[0] = Pkt3Gfx(0x37u, 4);
		t[1] = (5u << 8) | (1u << 20);
		t[2] = (uint32_t)dst;
		t[3] = (uint32_t)(dst >> 32);
		t[4] = b == 0 ? 0x7e4e0000u : 0xe15e0000u;
		t[5] = t[6] = t[7] = 0xffff1000u;
	}
	struct {
		uint32_t    mode;
		uint64_t    value;
		const char* want;
	} cases[] = {{2, 5, "then"}, {2, 6, "else"}, {1, 6, "none"}};
	Bo*  bos[] = {&mem};
	bool ok    = true;
	for (int k = 0; k < 3; k++) {
		m[0] = (uint32_t)cases[k].value;
		m[1] = 0;
		m[16] = m[17] = m[18] = 0;
		__sync_synchronize();
		const uint32_t v  = valid ? (1u << 23) : 0u;
		Ib             ib = {.n = 0};
		IbPush(&ib, Pkt3Gfx(0x3fu, 13));
		IbPush(&ib, cases[k].mode | (3u << 8)); // EQUAL
		IbPush(&ib, (uint32_t)mem.va);
		IbPush(&ib, (uint32_t)(mem.va >> 32));
		IbPush(&ib, 0xffffffffu);
		IbPush(&ib, 0xffffffffu);
		IbPush(&ib, 5u);
		IbPush(&ib, 0u);
		IbPush(&ib, (uint32_t)(mem.va + 0x1000));
		IbPush(&ib, (uint32_t)((mem.va + 0x1000) >> 32));
		IbPush(&ib, 8u | v);
		IbPush(&ib, (uint32_t)(mem.va + 0x2000));
		IbPush(&ib, (uint32_t)((mem.va + 0x2000) >> 32));
		IbPush(&ib, 8u | v);
		IbPush(&ib, Pkt3Gfx(0x37u, 4));
		IbPush(&ib, (5u << 8) | (1u << 20));
		IbPush(&ib, (uint32_t)(mem.va + 72));
		IbPush(&ib, (uint32_t)((mem.va + 72) >> 32));
		IbPush(&ib, 0xaf7e0000u + (uint32_t)k);
		IbPad(&ib);
		// PROBE_CONDIB_CHAINBACK=1: each branch ends with a chain back to the rest of this IB, right after
		// the 14-dword packet, which is how a call would return.
		if (getenv("PROBE_CONDIB_CHAINBACK") != NULL) {
			for (int b = 0; b < 2; b++) {
				volatile uint32_t* t    = m + (b == 0 ? 0x400 : 0x800);
				const uint64_t     cont = c->ib.va + 14u * 4u;
				t[5]                    = Pkt3Gfx(0x3fu, 3);
				t[6]                    = (uint32_t)cont;
				t[7]                    = (uint32_t)(cont >> 32);
				t[8]                    = (ib.n - 14u) | (1u << 20) | (1u << 23);
			}
			__sync_synchronize();
			ib.dw[10] = 9u | v; // the branches' sizes cover the chain
			ib.dw[13] = 9u | v;
		}
		uint64_t seq = 0;
		r            = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
		int     w    = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 2.0) : r;
		int64_t st   = ContextState(d);
		const char* got = m[16] == 0x7e4e0000u ? (m[17] ? "then+else" : "then") : (m[17] == 0xe15e0000u ? "else" : "none");
		printf("  mode %u, value %llu (ref 5): %s, ctx 0x%" PRIx64 ", branch taken %s (want %s), WRITE_DATA after it %s\n",
		       cases[k].mode, (unsigned long long)cases[k].value, w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st, got,
		       cases[k].want, m[18] == 0xaf7e0000u + (uint32_t)k ? "landed" : "missing");
		fflush(stdout);
		if (w != 0 || st != 0) {
			_exit(3);
		}
		ok = ok && strcmp(got, cases[k].want) == 0 && m[18] == 0xaf7e0000u + (uint32_t)k;
	}
	BoDestroy(d, &mem);
	return ok;
}

// libSceAgc's ATOMIC_MEM around a context push or pop (AgcDcbContextStateOp) as it writes it, on the gfx
// ring: engine bit 30, COMMAND 2, atomic 0x67, 64-bit source {1, 0} over flags that hold {0, 1}. The flags
// should read {1, 0} after it, and a WRITE_DATA after it lands.
static bool DangerAtomicMem(Device* d, Common* c) {
	printf("== danger atomicmem: libSceAgc's ATOMIC_MEM (atomic 0x67, command 2) on the gfx ring\n");
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
	volatile uint32_t* m = (volatile uint32_t*)mem.cpu;
	m[0]                 = 0;
	m[1]                 = 1;
	m[4]                 = 0;
	__sync_synchronize();
	Ib ib = {.n = 0};
	IbPush(&ib, 0xc0071e00u);
	IbPush(&ib, (1u << 30) | (2u << 8) | 0x67u);
	IbPush(&ib, (uint32_t)mem.va);
	IbPush(&ib, (uint32_t)(mem.va >> 32));
	IbPush(&ib, 1u);
	IbPush(&ib, 0u);
	IbPush(&ib, 0u);
	IbPush(&ib, 0u);
	IbPush(&ib, 6u);
	IbPush(&ib, Pkt3Gfx(0x37u, 4));
	IbPush(&ib, (5u << 8) | (1u << 20));
	IbPush(&ib, (uint32_t)(mem.va + 16));
	IbPush(&ib, (uint32_t)((mem.va + 16) >> 32));
	IbPush(&ib, 0x61746f6du);
	IbPad(&ib);
	Bo*      bos[] = {&mem};
	uint64_t seq   = 0;
	r              = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, &ib, bos, 1, &seq);
	int     w      = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 2.0) : r;
	int64_t st     = ContextState(d);
	printf("  gfx ring: %s, ctx 0x%" PRIx64 ", flags {%u, %u} (want {1, 0}), WRITE_DATA after it %s\n", w == 1 ? "TIMEOUT" : ErrStr(w),
	       (uint64_t)st, m[0], m[1], m[4] == 0x61746f6du ? "landed" : "missing");
	if (w != 0 || st != 0) {
		_exit(3);
	}
	const bool ok = m[0] == 1 && m[1] == 0 && m[4] == 0x61746f6du;
	BoDestroy(d, &mem);
	return ok;
}

// libSceAgc's ACB packets as it writes them, without the shader-type bit (header bit 1) the translator
// sets for MEC: does the BC-250's MEC run them as the PS5's does? One job per step on the compute ring,
// each checked by what it writes; the first that does not finish stops the run (a stopped MEC needs a
// reboot with gpu_recovery=0). Steps: a store dispatch (SET_SH_REG, DISPATCH_DIRECT), EVENT_WRITE,
// ACQUIRE_MEM, DMA_DATA, WAIT_REG_MEM_64, RELEASE_MEM.
// Result on BC-250: the store dispatch runs without the bit and writes all 64 values.
static bool DangerMecForms(Device* d, Common* c) {
	printf("== danger mecforms: ACB packets without the shader-type bit on the compute ring\n");
	Bo  mem;
	int r = MemCreate(d, &mem, MEM_GTT, 2 * MiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0700000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		return false;
	}
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem, &c->code};
	const uint32_t     magic = 0x6d656300u;
	// Clears the shader-type bit in every packet header of `ib` from dword `from` on.
	// PROBE_MEC_STEPS: the steps to run, e.g. "1,2a,3,4,5" (the default; 2e is EVENT_WRITE CS_PARTIAL_FLUSH,
	// which stopped amdgpu's compute ring even with the bit, 2a ACQUIRE_MEM). PROBE_MEC_BIT=1 keeps the
	// shader-type bit (the control run).
	const char* steps   = getenv("PROBE_MEC_STEPS") != NULL ? getenv("PROBE_MEC_STEPS") : "1,2a,3,4,5";
	const bool  keepbit = getenv("PROBE_MEC_BIT") != NULL;
	#define MEC_STEP(name) (strstr(steps, (name)) != NULL)
	#define MEC_PLAIN(ib, from)                                                                     \
		for (uint32_t k_ = (from); k_ < (ib).n;) {                                              \
			const uint32_t h_ = (ib).dw[k_];                                                \
			(ib).dw[k_]       = keepbit ? h_ : h_ & ~2u;                                    \
			k_ += (h_ & 0x3fffff00u) == 0x3fff1000u ? 1u : ((h_ >> 16) & 0x3fffu) + 2u;     \
		}
	#define MEC_MARK(ib, off, value)                                                                \
		do {                                                                                    \
			IbPush(&(ib), Pkt3(0x37u, 4));                                                  \
			IbPush(&(ib), (5u << 8) | (1u << 20));                                          \
			IbPush(&(ib), (uint32_t)(mem.va + (off)));                                      \
			IbPush(&(ib), (uint32_t)((mem.va + (off)) >> 32));                              \
			IbPush(&(ib), (value));                                                         \
		} while (0)
	#define MEC_RUN(ib, what)                                                                       \
		do {                                                                                    \
			IbPad(&(ib));                                                                   \
			uint64_t seq_ = 0;                                                              \
			r             = Submit(d, &c->ib, &(ib), bos, 2, &seq_);                         \
			int     w_    = r == 0 ? Wait(d, seq_, 3.0) : r;                                \
			int64_t st_   = ContextState(d);                                                \
			printf("  %-34s %s, ctx 0x%" PRIx64 "\n", (what), w_ == 1 ? "TIMEOUT" : ErrStr(w_), (uint64_t)st_); \
			fflush(stdout);                                                                 \
			if (w_ != 0 || st_ != 0) {                                                      \
				_exit(3); /* freeing buffers would wait on the stopped ring */              \
			}                                                                               \
		} while (0)

	printf("  shader-type bit %s; steps %s\n", keepbit ? "kept (control)" : "cleared", steps);
	memset(mem.cpu, 0, 64 * KiB);
	Ib   ib = {.n = 0};
	bool ok = true;
	// 1. A store dispatch: SET_SH_REG and DISPATCH_DIRECT.
	if (MEC_STEP("1")) {
		LoadCode(c, k_store_w64, sizeof(k_store_w64));
		const uint32_t base  = 0x5100000u;
		uint32_t       ud[3] = {(uint32_t)mem.va, (uint32_t)(mem.va >> 32), base};
		Dispatch       p     = {c->code.va, 64, 8, 3, true, 64, 1, ud};
		IbDispatch(&ib, &p);
		MEC_PLAIN(ib, 0);
		MEC_RUN(ib, "SET_SH_REG + DISPATCH_DIRECT:");
		uint32_t good = 0;
		for (uint32_t i = 0; i < 64; i++) {
			good += m[i] == base + i ? 1u : 0u;
		}
		printf("    store dispatch wrote %u of 64 values\n", good);
		ok = good == 64;
	}

	// 2e. EVENT_WRITE CS_PARTIAL_FLUSH.
	if (MEC_STEP("2e")) {
		ib.n = 0;
		IbPush(&ib, Pkt3(0x46u, 1));
		IbPush(&ib, 7u | (4u << 8));
		MEC_PLAIN(ib, 0);
		MEC_MARK(ib, 0x1000, magic | 2u);
		MEC_RUN(ib, "EVENT_WRITE:");
		ok = ok && m[0x400] == (magic | 2u);
	}
	// 2a. ACQUIRE_MEM (gfx10 form, as amdgpu's emit_mem_sync: GL2/GL1/GLV invalidate + GL2 writeback).
	if (MEC_STEP("2a")) {
		ib.n = 0;
		IbPush(&ib, Pkt3(0x58u, 7));
		IbPush(&ib, 0u);
		IbPush(&ib, 0xffffffffu);
		IbPush(&ib, 0xffffffu);
		IbPush(&ib, 0u);
		IbPush(&ib, 0u);
		IbPush(&ib, 0xau);
		IbPush(&ib, 0x8000u | 0x4000u | 0x200u | 0x100u);
		MEC_PLAIN(ib, 0);
		MEC_MARK(ib, 0x1040, magic | 3u);
		MEC_RUN(ib, "ACQUIRE_MEM:");
		ok = ok && m[0x410] == (magic | 3u);
	}

	// 3. DMA_DATA memory to memory, CP_SYNC.
	if (MEC_STEP("3")) {
	for (uint32_t i = 0; i < 4; i++) {
		m[0x800 + i] = magic | (0x30u + i);
	}
	__sync_synchronize();
	ib.n = 0;
	IbPush(&ib, Pkt3(0x50u, 6));
	IbPush(&ib, 1u << 31);
	IbPush(&ib, (uint32_t)(mem.va + 0x2000));
	IbPush(&ib, (uint32_t)((mem.va + 0x2000) >> 32));
	IbPush(&ib, (uint32_t)(mem.va + 0x3000));
	IbPush(&ib, (uint32_t)((mem.va + 0x3000) >> 32));
	IbPush(&ib, 16u);
	MEC_PLAIN(ib, 0);
	MEC_RUN(ib, "DMA_DATA:");
	ok = ok && m[0xc00] == (magic | 0x30u) && m[0xc03] == (magic | 0x33u);
	printf("    DMA_DATA copy %s\n", m[0xc00] == (magic | 0x30u) && m[0xc03] == (magic | 0x33u) ? "landed" : "missing");
	}

	// 4. WAIT_REG_MEM_64 on a 64-bit value already there (EQUAL, mask ~0), then a marker.
	if (MEC_STEP("4")) {
	m[0x1000] = 1u;
	m[0x1001] = 0u;
	__sync_synchronize();
	ib.n = 0;
	IbPush(&ib, Pkt3(0x93u, 8));
	IbPush(&ib, 0x13u);
	IbPush(&ib, (uint32_t)(mem.va + 0x4000));
	IbPush(&ib, (uint32_t)((mem.va + 0x4000) >> 32));
	IbPush(&ib, 1u);
	IbPush(&ib, 0u);
	IbPush(&ib, 0xffffffffu);
	IbPush(&ib, 0xffffffffu);
	IbPush(&ib, 0x19u);
	MEC_PLAIN(ib, 0);
	MEC_MARK(ib, 0x1100, magic | 4u);
	MEC_RUN(ib, "WAIT_REG_MEM_64:");
	ok = ok && m[0x440] == (magic | 4u);
	}

	// 5. RELEASE_MEM: CACHE_FLUSH_AND_INV_TS_EVENT, 32-bit data to memory, no interrupt.
	if (MEC_STEP("5")) {
	ib.n = 0;
	IbPush(&ib, Pkt3(0x49u, 7));
	IbPush(&ib, 0x14u | (5u << 8));
	IbPush(&ib, 1u << 29);
	IbPush(&ib, (uint32_t)(mem.va + 0x5000));
	IbPush(&ib, (uint32_t)((mem.va + 0x5000) >> 32));
	IbPush(&ib, magic | 5u);
	IbPush(&ib, 0u);
	IbPush(&ib, 0u);
	MEC_PLAIN(ib, 0);
	MEC_RUN(ib, "RELEASE_MEM:");
	for (int i = 0; i < 100 && m[0x1400] != (magic | 5u); i++) {
		usleep(1000);
	}
	printf("    RELEASE_MEM data %s\n", m[0x1400] == (magic | 5u) ? "landed" : "missing");
	ok = ok && m[0x1400] == (magic | 5u);
	}
	#undef MEC_STEP
	#undef MEC_PLAIN
	#undef MEC_MARK
	#undef MEC_RUN
	BoDestroy(d, &mem);
	printf("  %s\n", ok ? "every step ran and wrote what it should" : "a step did not write what it should");
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

// COND_EXEC (0x22: address, address high, 0, EXEC_COUNT): when the dword at the address is 0 the CP
// discards the next EXEC_COUNT dwords. Kyty's exit guards what follows each GPU wait with such an
// abort word. Each step (CONDEXEC_STEP=k runs one) builds: [wait on a label the CPU releases] [PFP_SYNC_ME]
// COND_EXEC on the word over [filler NOPs] [IB2 call writing C] [WRITE_DATA A], then WRITE_DATA B. The
// CPU clears the word 50 ms in and releases the label 50 ms later: if A lands, the CP had already
// decided the COND_EXEC (the PFP parses ahead of an ME wait). B shows the CP went on past the guard.
static bool DangerCondExec(Device* d, Common* c) {
	(void)c;
	printf("== danger condexec: COND_EXEC on an abort word, gfx ring and compute ring 1\n");
	Bo  mem, big, ib2;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0720000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r == 0) {
		r = MemCreate(d, &big, MEM_GTT_USWC, 512 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &big, 0x0084000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &ib2, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &ib2, 0x0083000000ull, 0);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		BoDestroy(d, &big);
		BoDestroy(d, &ib2);
		return false;
	}
	static const struct {
		const char* name;
		bool        compute;
		uint32_t    word;   // the abort word at submission (0: skip)
		int         wait;   // 0 none, 1 ME wait, 2 ME wait + PFP_SYNC_ME, 3 PFP wait
		uint32_t    filler; // single-dword NOPs guarded before A
		bool        ib2;    // the guarded part calls an IB2 that writes C
		bool        lands;  // A should land
	} steps[] = {
	    {"gfx, word 1: the guarded write runs", false, 1, 0, 0, true, true},
	    {"gfx, word 0: the guarded write and IB2 call are skipped", false, 0, 0, 0, true, false},
	    {"gfx, word 0, EXEC_COUNT 20000 over filler NOPs", false, 0, 0, 20000, false, false},
	    {"gfx, word 0, EXEC_COUNT 70000 over filler NOPs", false, 0, 0, 70000, false, false},
	    {"gfx, ME wait, word cleared while it waits", false, 1, 1, 0, false, false},
	    {"gfx, ME wait + PFP_SYNC_ME, word cleared while it waits", false, 1, 2, 0, true, false},
	    {"gfx, PFP wait, word cleared while it waits", false, 1, 3, 0, false, false},
	    {"compute ring 1, word 0", true, 0, 0, 0, false, false},
	    {"compute ring 1, wait, word cleared while it waits", true, 1, 1, 0, false, false},
	};
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem, &ib2};
	bool               ok    = true;
	const char*        one   = getenv("CONDEXEC_STEP"); // a CP that misparses stops the ring: one step per run
	for (size_t k = 0; k < ARRAY_SIZE(steps) && ok; k++) {
		if (one != NULL && (size_t)atoi(one) != k) {
			continue;
		}
		const bool     cs   = steps[k].compute;
		const uint64_t word = mem.va, label = mem.va + 0x40, a = mem.va + 0x100, b = mem.va + 0x104, cc = mem.va + 0x108;
		m[0]               = steps[k].word;
		m[0x40 / 4]        = 0;
		m[0x100 / 4]       = 0;
		m[0x104 / 4]       = 0;
		m[0x108 / 4]       = 0;
		uint32_t* q        = (uint32_t*)ib2.cpu;
		*q++               = Pkt3Gfx(0x37u, 4); // WRITE_DATA C
		*q++               = (5u << 8) | (1u << 20);
		*q++               = (uint32_t)cc;
		*q++               = (uint32_t)(cc >> 32);
		*q++               = 0xc0de0000u | (uint32_t)k;
		while (((q - (uint32_t*)ib2.cpu) & 7) != 0) {
			*q++ = 0xffff1000u;
		}
		__sync_synchronize();
		uint32_t* const ib0 = (uint32_t*)big.cpu;
		uint32_t*       p   = ib0;
#define CE_PKT(op, body) (cs ? Pkt3((op), (body)) : Pkt3Gfx((op), (body)))
		if (steps[k].wait != 0) {
			*p++ = CE_PKT(0x3cu, 6); // WAIT_REG_MEM == 1
			*p++ = steps[k].wait == 3 ? 0x113u : 0x13u;
			*p++ = (uint32_t)label;
			*p++ = (uint32_t)(label >> 32);
			*p++ = 1u;
			*p++ = 0xffffffffu;
			*p++ = 4u;
		}
		if (steps[k].wait == 2) {
			*p++ = Pkt3Gfx(0x42u, 1); // PFP_SYNC_ME
			*p++ = 0;
		}
		*p++                 = CE_PKT(0x22u, 4); // COND_EXEC
		*p++                 = (uint32_t)word;
		*p++                 = (uint32_t)(word >> 32);
		*p++                 = 0;
		uint32_t* const cnt  = p++;
		uint32_t* const from = p;
		for (uint32_t i = 0; i < steps[k].filler; i++) {
			*p++ = 0xffff1000u;
		}
		if (steps[k].ib2) {
			p = PutIb2(p, ib2.va, 8);
		}
		*p++ = CE_PKT(0x37u, 4); // WRITE_DATA A
		*p++ = (5u << 8) | (1u << 20);
		*p++ = (uint32_t)a;
		*p++ = (uint32_t)(a >> 32);
		*p++ = 0xa0000000u | (uint32_t)k;
		*cnt = (uint32_t)(p - from);
		*p++ = CE_PKT(0x37u, 4); // WRITE_DATA B
		*p++ = (5u << 8) | (1u << 20);
		*p++ = (uint32_t)b;
		*p++ = (uint32_t)(b >> 32);
		*p++ = 0xb0000000u | (uint32_t)k;
		while (((p - ib0) & 7) != 0) {
			*p++ = 0xffff1000u;
		}
#undef CE_PKT
		__sync_synchronize();
		const uint32_t ip   = cs ? AMDGPU_HW_IP_COMPUTE : AMDGPU_HW_IP_GFX;
		const uint32_t ring = cs ? 1u : 0u;
		uint64_t       seq  = 0;
		r                   = SubmitVa(d, ip, ring, &big, big.va, (uint32_t)(p - ib0) * 4u, bos, 2, &seq);
		bool early          = false;
		if (r == 0 && steps[k].wait != 0) {
			usleep(50000);
			early = m[0x100 / 4] != 0 || m[0x104 / 4] != 0;
			m[0]  = 0; // abort
			__sync_synchronize();
			usleep(50000);
			m[0x40 / 4] = 1; // release the wait
			__sync_synchronize();
		}
		const int     w      = r == 0 ? WaitOn(d, ip, ring, seq, 1.0) : r;
		const int64_t st     = ContextState(d);
		const bool    landed = m[0x100 / 4] == (0xa0000000u | (uint32_t)k);
		const bool    after  = m[0x104 / 4] == (0xb0000000u | (uint32_t)k);
		const bool    callee = m[0x108 / 4] == (0xc0de0000u | (uint32_t)k);
		printf("  %s: %s, ctx 0x%" PRIx64 ", EXEC_COUNT %u: A %s (%s), B %s%s%s\n", steps[k].name, w == 1 ? "TIMEOUT" : ErrStr(w),
		       (uint64_t)st, *cnt, landed ? "landed" : "skipped", landed == steps[k].lands ? "as expected" : "NOT as expected",
		       after ? "landed" : "MISSING", steps[k].ib2 ? (callee ? ", IB2 C landed" : ", IB2 C skipped") : "",
		       early ? ", a marker landed before the release" : "");
		fflush(stdout);
		if (w != 0 || st != 0) {
			_exit(3); // freeing buffers would wait on the stopped ring
		}
		ok = ok && after && !early && landed == steps[k].lands;
	}
	BoDestroy(d, &mem);
	BoDestroy(d, &big);
	BoDestroy(d, &ib2);
	return ok;
}

// The exit guard as Kyty emits it: COND_EXEC over a 4-dword INDIRECT_BUFFER with CHAIN to an empty IB, on an
// abort word that is 0 while the title runs. The CP skips the chain then; once the word is 1 it chains to
// the empty IB, whose end returns to the ring: the rest of the IB (A, B) never runs and the job ends. Each
// step (GUARD_STEP=k runs one): [wait on a label the CPU releases] [PFP_SYNC_ME] guard, A, B; the CPU sets
// the word 50 ms in (or before submitting) and releases the label 50 ms later.
static bool DangerExitGuard(Device* d, Common* c) {
	(void)c;
	printf("== danger exitguard: COND_EXEC + INDIRECT_BUFFER chain to an empty IB on an abort word\n");
	Bo  mem, ib, stub;
	int r = MemCreate(d, &mem, MEM_GTT, 64 * KiB, NULL);
	if (r == 0) {
		r = BoBind(d, &mem, 0x0720000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r == 0) {
		r = MemCreate(d, &ib, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &ib, 0x0084000000ull, 0);
	}
	if (r == 0) {
		r = MemCreate(d, &stub, MEM_GTT_USWC, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &stub, 0x0083000000ull, 0);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &mem);
		BoDestroy(d, &ib);
		BoDestroy(d, &stub);
		return false;
	}
	for (uint32_t i = 0; i < 8; i++) {
		((uint32_t*)stub.cpu)[i] = 0xffff1000u;
	}
	static const struct {
		const char* name;
		bool        compute;
		int         wait;  // 0 none (the word is set before submitting), 1 ME wait + PFP_SYNC_ME, 2 PFP wait
		uint32_t    abort; // the word once set
	} steps[] = {
	    {"gfx, word 0: the guard passes", false, 0, 0},
	    {"gfx, word 1: the guard leaves the IB", false, 0, 1},
	    {"gfx, ME wait + PFP_SYNC_ME, word set while it waits", false, 1, 1},
	    {"gfx, PFP wait, word set while it waits", false, 2, 1},
	    {"compute ring 1, word 1", true, 0, 1},
	    {"compute ring 1, wait, word set while it waits", true, 1, 1},
	};
	volatile uint32_t* m     = (volatile uint32_t*)mem.cpu;
	Bo*                bos[] = {&mem, &stub};
	bool               ok    = true;
	const char*        one   = getenv("GUARD_STEP");
	for (size_t k = 0; k < ARRAY_SIZE(steps) && ok; k++) {
		if (one != NULL && (size_t)atoi(one) != k) {
			continue;
		}
		const bool     cs   = steps[k].compute;
		const uint64_t word = mem.va, label = mem.va + 0x40, a = mem.va + 0x100, b = mem.va + 0x104;
		m[0]               = steps[k].wait != 0 ? 0 : steps[k].abort;
		m[0x40 / 4]        = 0;
		m[0x100 / 4]       = 0;
		m[0x104 / 4]       = 0;
		__sync_synchronize();
		uint32_t* const ib0 = (uint32_t*)ib.cpu;
		uint32_t*       p   = ib0;
#define EG_PKT(op, body) (cs ? Pkt3((op), (body)) : Pkt3Gfx((op), (body)))
		if (steps[k].wait != 0) {
			*p++ = EG_PKT(0x3cu, 6); // WAIT_REG_MEM == 1
			*p++ = steps[k].wait == 2 && !cs ? 0x113u : 0x13u;
			*p++ = (uint32_t)label;
			*p++ = (uint32_t)(label >> 32);
			*p++ = 1u;
			*p++ = 0xffffffffu;
			*p++ = 4u;
			if (steps[k].wait == 1 && !cs) {
				*p++ = Pkt3Gfx(0x42u, 1); // PFP_SYNC_ME
				*p++ = 0;
			}
		}
		*p++ = EG_PKT(0x22u, 4); // COND_EXEC over the chain
		*p++ = (uint32_t)word;
		*p++ = (uint32_t)(word >> 32);
		*p++ = 0;
		*p++ = 4;
		*p++ = EG_PKT(0x3fu, 3); // INDIRECT_BUFFER, CHAIN, to the empty IB
		*p++ = (uint32_t)stub.va;
		*p++ = (uint32_t)(stub.va >> 32) & 0xffffu;
		*p++ = 8u | (1u << 20) | (1u << 23);
		for (int i = 0; i < 2; i++) {
			const uint64_t at = i == 0 ? a : b;
			*p++              = EG_PKT(0x37u, 4); // WRITE_DATA A, B
			*p++              = (5u << 8) | (1u << 20);
			*p++              = (uint32_t)at;
			*p++              = (uint32_t)(at >> 32);
			*p++              = (i == 0 ? 0xa0000000u : 0xb0000000u) | (uint32_t)k;
		}
		while (((p - ib0) & 7) != 0) {
			*p++ = 0xffff1000u;
		}
#undef EG_PKT
		__sync_synchronize();
		const uint32_t ip   = cs ? AMDGPU_HW_IP_COMPUTE : AMDGPU_HW_IP_GFX;
		const uint32_t ring = cs ? 1u : 0u;
		uint64_t       seq  = 0;
		r                   = SubmitVa(d, ip, ring, &ib, ib.va, (uint32_t)(p - ib0) * 4u, bos, 2, &seq);
		bool early          = false;
		if (r == 0 && steps[k].wait != 0) {
			usleep(50000);
			early = m[0x100 / 4] != 0 || m[0x104 / 4] != 0;
			m[0]  = steps[k].abort;
			__sync_synchronize();
			usleep(50000);
			m[0x40 / 4] = 1;
			__sync_synchronize();
		}
		const int     w     = r == 0 ? WaitOn(d, ip, ring, seq, 1.0) : r;
		const int64_t st    = ContextState(d);
		const bool    ran_a = m[0x100 / 4] == (0xa0000000u | (uint32_t)k);
		const bool    ran_b = m[0x104 / 4] == (0xb0000000u | (uint32_t)k);
		const bool    want  = steps[k].abort == 0;
		printf("  %s: job %s, ctx 0x%" PRIx64 ": A %s, B %s (%s)%s\n", steps[k].name, w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st,
		       ran_a ? "ran" : "skipped", ran_b ? "ran" : "skipped", ran_a == want && ran_b == want ? "as expected" : "NOT as expected",
		       early ? ", a marker landed before the release" : "");
		fflush(stdout);
		if (w != 0 || st != 0) {
			_exit(3);
		}
		ok = ok && !early && ran_a == want && ran_b == want;
	}
	BoDestroy(d, &mem);
	BoDestroy(d, &ib);
	BoDestroy(d, &stub);
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

// CP register shadowing, as PAL and Mesa set it up on gfx9-gfx10.3 for mid command buffer preemption (PAL
// gfx9QueueContexts.cpp WritePerSubmitPreamble, Mesa ac_shadowed_regs.c ac_create_shadowing_ib_preamble):
// CONTEXT_CONTROL's second dword turns on writing SET_*_REG values to memory per register class, the LOAD_*_REG
// packets tell the CP where each class's shadow is (and load from it), and a register at R of a space starting at B
// is kept at that address + (R - B) (PAL: once "the LOADs are done and HW knows the shadow memory", a SET initializes
// "the register and also the shadow memory"). Does the BC-250's microcode do it, for the register tables
// (0x9F/0x63/0x64) too? On the PS5 the CP does libSceAgc's context push and pop with it.

#define CC_UPDATE    (1u << 31) // CONTEXT_CONTROL: update the load enables (dword 1) or the shadow enables (dword 2)
#define CC_CONTEXT   (1u << 1)  // the same bit per register class in both dwords
#define CC_UCONFIG   (1u << 15)
#define CC_GFX_SH    (1u << 16)
#define CC_CS_SH     (1u << 24)
#define CC_ALL       (CC_CONTEXT | CC_UCONFIG | CC_GFX_SH | CC_CS_SH)
#define CONTEXT_BASE 0x28000u

// The shadow buffer S, laid out as Mesa's (sid.h SI_SHADOWED_*): SH space, context space, UCONFIG space. SHADOW_LOAD=0
// points the CP at a second set of regions 1 MiB in.
#define SHADOW_SH_AT  0x0000u
#define SHADOW_CTX_AT 0x1000u
#define SHADOW_UC_AT  0x9000u
#define SHADOW_SET_B  0x100000u
#define SHADOW_SIZE   (2 * MiB)

typedef struct {
	const char* name;
	uint32_t    reg;  // byte address
	char        cls;  // 'x' context, 's' graphics SH, 'c' compute SH (SET and LOAD with the compute-type header),
	                  // 'u' UCONFIG
	char        how;  // 's' SET, 't' a one-pair register table (0x9F/0x63/0x64 with libSceAgc's graphics-type
	                  // header), 'i' SET in an IB2, 'j' a one-pair table in an IB2 (as direct runs execute titles' tables)
	bool        load; // in the LOAD ranges (as on PAL's Navi10 restore lists); the SH ones are also read back
	uint32_t    rest; // UCONFIG: left in the register afterwards (Mesa's gfx10 preamble values)
} ShadowReg;

static const ShadowReg k_shadow_regs[] = {
    {"PA_SC_SCREEN_SCISSOR_TL", 0x28030u, 'x', 's', true, 0},
    {"PA_SC_SCREEN_SCISSOR_BR", 0x28034u, 'x', 's', true, 0},
    {"PA_SC_WINDOW_OFFSET", 0x28200u, 'x', 't', true, 0},
    {"PA_SC_GENERIC_SCISSOR_TL", 0x28240u, 'x', 'i', true, 0},
    {"PA_SC_GENERIC_SCISSOR_BR", 0x28244u, 'x', 'j', true, 0},
    {"SPI_SHADER_USER_DATA_PS_0", 0xb030u, 's', 's', true, 0},
    {"SPI_SHADER_USER_DATA_PS_1", 0xb034u, 's', 's', true, 0},
    {"SPI_SHADER_USER_DATA_PS_2", 0xb038u, 's', 't', true, 0},
    {"SPI_SHADER_USER_DATA_PS_3", 0xb03cu, 's', 'j', true, 0},
    {"COMPUTE_USER_DATA_0", 0xb900u, 'c', 's', true, 0},
    {"COMPUTE_USER_DATA_1", 0xb904u, 'c', 's', true, 0},
    {"COMPUTE_USER_DATA_2", 0xb908u, 'c', 't', true, 0}, // a graphics-type 0x63, as My First GT's DCB dispatches
    {"GE_MAX_VTX_INDX", 0x30964u, 'u', 's', true, 0xffffffffu},
    {"GE_INDX_OFFSET", 0x30928u, 'u', 't', true, 0},
    {"GE_MIN_VTX_INDX", 0x30924u, 'u', 'j', true, 0},
    {"SQ_THREAD_TRACE_USERDATA_2", 0x30d08u, 'u', 's', false, 0}, // on PAL's never-shadowed list
};

// S, and T (64 KiB): +0x000 the SH registers after the LOADs, +0x080 after the writes, +0x100 markers, +0x200 the
// register tables (8 bytes per register), +0x400 the IB2s (32 bytes per register), +0x800 shadow-restore's reads
// (0x80 per job), +0xa00 shadow-pushpop's copies of S.
typedef struct {
	Bo s;
	Bo t;
} ShadowBufs;

static const ShadowReg* ShadowRegNamed(const char* name) {
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (strcmp(k_shadow_regs[k].name, name) == 0) {
			return &k_shadow_regs[k];
		}
	}
	abort();
}

static uint32_t ShadowMagic(size_t k) { // what the test writes
	return 0x5ad00000u | ((uint32_t)k << 8) | (uint32_t)k;
}

static uint32_t ShadowSentinel(size_t k, int set) { // what the CPU leaves at a loaded register's place in S
	return (set != 0 ? 0x20ad0000u : 0x10ad0000u) | (uint32_t)k;
}

static uint32_t ShadowPre(size_t k) { // the SH registers before the LOADs
	return 0x0e7e0000u | (uint32_t)k;
}

static uint32_t ShadowSpace(char cls) {
	return cls == 'x' ? CONTEXT_BASE : cls == 'u' ? UCONFIG_BASE : SH_REG_BASE;
}

static uint32_t ShadowRegion(char cls) {
	return cls == 'x' ? SHADOW_CTX_AT : cls == 'u' ? SHADOW_UC_AT : SHADOW_SH_AT;
}

static uint32_t ShadowBit(char cls) {
	return cls == 'x' ? CC_CONTEXT : cls == 'u' ? CC_UCONFIG : cls == 's' ? CC_GFX_SH : CC_CS_SH;
}

static bool ShadowIsSh(const ShadowReg* r) {
	return r->cls == 's' || r->cls == 'c';
}

// Byte offset in S of a register's shadow, in region set 0 or SHADOW_SET_B.
static uint32_t ShadowAt(const ShadowReg* r, uint32_t set) {
	return set + ShadowRegion(r->cls) + (r->reg - ShadowSpace(r->cls));
}

static const char* ShadowClassName(char cls) {
	return cls == 'x' ? "ctx" : cls == 's' ? "sh" : cls == 'c' ? "cs" : "uc";
}

static const char* ShadowHowName(const ShadowReg* r) {
	if (r->how == 't') {
		return r->cls == 'x' ? "0x9F" : r->cls == 'u' ? "0x64" : "0x63";
	}
	if (r->how == 'j') {
		return r->cls == 'x' ? "IB2 0x9F" : r->cls == 'u' ? "IB2 0x64" : "IB2 0x63";
	}
	return r->how == 'i' ? "IB2 SET" : "SET";
}

// The one-pair register table of a test register: {offset in its space, value} at T + 0x200 + 8 * k, and the packet
// that loads it (libSceAgc's graphics-type header, INDEX 0, offset_and_data).
static void IbShadowTable(Ib* ib, const ShadowBufs* b, size_t k) {
	const ShadowReg* r  = &k_shadow_regs[k];
	const uint32_t   at = 0x200u + 8u * (uint32_t)k;
	uint32_t*        tm = (uint32_t*)b->t.cpu;
	tm[at / 4]          = (r->reg - ShadowSpace(r->cls)) / 4;
	tm[at / 4 + 1]      = ShadowMagic(k);
	IbPush(ib, Pkt3Gfx(r->cls == 'x' ? 0x9fu : r->cls == 'u' ? 0x64u : 0x63u, 4));
	IbPush(ib, (uint32_t)(b->t.va + at) & ~3u); // INDEX 0: direct address
	IbPush(ib, (uint32_t)((b->t.va + at) >> 32));
	IbPush(ib, 0x80000000u); // DATA_FORMAT offset_and_data, REG_OFFSET 0
	IbPush(ib, 1u);          // one pair
}

static void IbContextControl(Ib* ib, uint32_t load, uint32_t shadow) {
	IbPush(ib, Pkt3Gfx(0x28u, 2));
	IbPush(ib, load);
	IbPush(ib, shadow);
}

static void IbWriteData(Ib* ib, uint64_t va, uint32_t value) {
	IbPush(ib, Pkt3Gfx(0x37u, 4)); // WRITE_DATA, memory, write confirm
	IbPush(ib, (5u << 8) | (1u << 20));
	IbPush(ib, (uint32_t)va);
	IbPush(ib, (uint32_t)(va >> 32));
	IbPush(ib, value);
}

// COPY_DATA of a dword from memory (SRC_SEL 2) to memory (DST_SEL 5), write confirm.
static void IbCopyMemToMem(Ib* ib, uint64_t src, uint64_t dst) {
	IbPush(ib, Pkt3Gfx(0x40u, 5));
	IbPush(ib, 2u | (5u << 8) | (1u << 20));
	IbPush(ib, (uint32_t)src);
	IbPush(ib, (uint32_t)(src >> 32));
	IbPush(ib, (uint32_t)dst);
	IbPush(ib, (uint32_t)(dst >> 32));
}

// SET_CONTEXT_REG / SET_SH_REG / SET_UCONFIG_REG of one test register.
static void IbShadowSet(Ib* ib, const ShadowReg* r, uint32_t value) {
	const uint32_t op = r->cls == 'x' ? 0x69u : r->cls == 'u' ? 0x79u : 0x76u;
	IbPush(ib, r->cls == 'c' ? Pkt3(op, 2) : Pkt3Gfx(op, 2));
	IbPush(ib, (r->reg - ShadowSpace(r->cls)) / 4);
	IbPush(ib, value);
}

// LOAD_UCONFIG_REG, LOAD_CONTEXT_REG, LOAD_SH_REG with the graphics-type then the compute-type header (both into the
// SH region, as PAL), from S + set: one range per test register of the LOAD lists, so nothing else is loaded.
static void IbShadowLoads(Ib* ib, uint64_t s_va, uint32_t set) {
	static const char order[] = "uxsc";
	for (int o = 0; o < 4; o++) {
		const char cls = order[o];
		uint32_t   n   = 0;
		for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
			n += k_shadow_regs[k].cls == cls && k_shadow_regs[k].load;
		}
		if (n == 0) {
			continue;
		}
		const uint32_t op = cls == 'x' ? 0x61u : cls == 'u' ? 0x5eu : 0x5fu;
		const uint64_t va = s_va + set + ShadowRegion(cls);
		IbPush(ib, cls == 'c' ? Pkt3(op, 2 + 2 * n) : Pkt3Gfx(op, 2 + 2 * n));
		IbPush(ib, (uint32_t)va); // BASE_ADDRESS, dword aligned
		IbPush(ib, (uint32_t)(va >> 32));
		for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
			const ShadowReg* r = &k_shadow_regs[k];
			if (r->cls == cls && r->load) {
				IbPush(ib, (r->reg - ShadowSpace(cls)) / 4); // REG_OFFSET
				IbPush(ib, 1u);                              // NUM_DWORDS
			}
		}
	}
}

// COPY_DATA of every SH test register to T + at. SH reads are safe; context and UCONFIG reads stop the ring.
static void IbShadowReadSh(Ib* ib, uint64_t t_va, uint32_t at) {
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (ShadowIsSh(&k_shadow_regs[k])) {
			IbCopyRegToMem(ib, k_shadow_regs[k].reg, t_va + at + 4u * k);
		}
	}
}

// Shadowing and loading off (Kyty's preamble does the same), then the UCONFIG test registers back to Mesa's values.
static void IbShadowOff(Ib* ib) {
	IbContextControl(ib, CC_UPDATE, CC_UPDATE);
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (k_shadow_regs[k].cls == 'u') {
			IbShadowSet(ib, &k_shadow_regs[k], k_shadow_regs[k].rest);
		}
	}
}

static bool ShadowSetup(Device* d, ShadowBufs* b) {
	memset(b, 0, sizeof(*b));
	int r = MemCreate(d, &b->s, MEM_GTT, SHADOW_SIZE, NULL);
	if (r == 0) {
		r = BoBind(d, &b->s, 0x0720000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r == 0) {
		r = MemCreate(d, &b->t, MEM_GTT, 64 * KiB, NULL);
	}
	if (r == 0) {
		r = BoBind(d, &b->t, 0x0721000000ull, AMDGPU_VM_MTYPE_UC);
	}
	if (r != 0) {
		printf("  setup failed: %s\n", ErrStr(r));
		BoDestroy(d, &b->t);
		BoDestroy(d, &b->s);
		return false;
	}
	memset(b->s.cpu, 0, SHADOW_SIZE);
	memset(b->t.cpu, 0, 64 * KiB);
	uint32_t* s = (uint32_t*)b->s.cpu;
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (k_shadow_regs[k].load) {
			s[ShadowAt(&k_shadow_regs[k], 0) / 4]            = ShadowSentinel(k, 0);
			s[ShadowAt(&k_shadow_regs[k], SHADOW_SET_B) / 4] = ShadowSentinel(k, 1);
		}
	}
	__sync_synchronize();
	return true;
}

// Runs one job on the gfx ring and waits 2 s. One that does not finish ends the process: freeing buffers would wait
// on the stopped ring.
static void ShadowRun(Device* d, Common* c, ShadowBufs* b, const Ib* ib, const char* what) {
	Bo*      bos[] = {&b->s, &b->t};
	uint64_t seq   = 0;
	int      r     = SubmitOn(d, AMDGPU_HW_IP_GFX, 0, &c->ib, ib, bos, 2, &seq);
	int      w     = r == 0 ? WaitOn(d, AMDGPU_HW_IP_GFX, 0, seq, 2.0) : r;
	int64_t  st    = ContextState(d);
	printf("  %s (%u dw): %s, ctx 0x%" PRIx64 "\n", what, ib->n, w == 1 ? "TIMEOUT" : ErrStr(w), (uint64_t)st);
	if (w != 0 || st != 0) {
		_exit(3);
	}
}

// Names a byte offset of S: the register whose shadow place it is.
static void ShadowWhere(uint32_t off, char* buf, size_t size) {
	const uint32_t set = off >= SHADOW_SET_B ? SHADOW_SET_B : 0;
	const uint32_t o   = off - set;
	const char*    b   = set != 0 ? ", set B" : "";
	if (o < SHADOW_CTX_AT) {
		snprintf(buf, size, " (SH 0x%05x%s)", SH_REG_BASE + o, b);
	} else if (o < SHADOW_UC_AT) {
		snprintf(buf, size, " (context 0x%05x%s)", CONTEXT_BASE + o - SHADOW_CTX_AT, b);
	} else if (o < SHADOW_UC_AT + 0x10000u) {
		snprintf(buf, size, " (UCONFIG 0x%05x%s)", UCONFIG_BASE + o - SHADOW_UC_AT, b);
	} else {
		buf[0] = 0;
	}
}

// Non-zero dwords of S away from every test register's place: whatever else the CP wrote. Prints the first 8.
static uint32_t ShadowOthers(const ShadowBufs* b) {
	const uint32_t* s     = (const uint32_t*)b->s.cpu;
	uint32_t        count = 0;
	for (uint32_t i = 0; i < SHADOW_SIZE / 4; i++) {
		bool place = s[i] == 0;
		for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs) && !place; k++) {
			place = i * 4 == ShadowAt(&k_shadow_regs[k], 0) || i * 4 == ShadowAt(&k_shadow_regs[k], SHADOW_SET_B);
		}
		if (!place) {
			if (count < 8) {
				char where[48];
				ShadowWhere(i * 4, where, sizeof(where));
				printf("  other dword in S: +0x%06x = 0x%08x%s\n", i * 4, s[i], where);
			}
			count++;
		}
	}
	printf("  other non-zero dwords in S: %u\n", count);
	return count;
}

// The first job: the SH test registers set with shadowing off, then shadowing on for `classes` with every load on and
// the LOADs pointing the CP at S (a PAL / Mesa preamble); with load_off, LOADs from the second region set follow with
// loading off (do they still move the address?). The SH registers are read, the test values written by SET, by
// register table and by SET in an IB2, the SH registers read again, a marker. Shadowing goes off before the job ends:
// the next job on the ring may be another process's, and amdgpu's CONTEXT_CONTROL before an IB leaves the shadow
// enables alone (RADV's, radeonsi's and Kyty's preambles turn them off themselves).
static void ShadowIb1(Ib* ib, const ShadowBufs* b, uint32_t classes, bool load_off) {
	const uint64_t t = b->t.va;
	IbContextControl(ib, CC_UPDATE, CC_UPDATE);
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (ShadowIsSh(&k_shadow_regs[k])) {
			IbShadowSet(ib, &k_shadow_regs[k], ShadowPre(k));
		}
	}
	IbContextControl(ib, CC_UPDATE | CC_ALL, CC_UPDATE | classes);
	IbShadowLoads(ib, b->s.va, 0);
	if (load_off) {
		IbContextControl(ib, CC_UPDATE, CC_UPDATE | classes);
		IbShadowLoads(ib, b->s.va, SHADOW_SET_B);
	}
	IbShadowReadSh(ib, t, 0x000);
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		const ShadowReg* r = &k_shadow_regs[k];
		if (r->how == 's') {
			IbShadowSet(ib, r, ShadowMagic(k));
		} else if (r->how == 't') {
			IbShadowTable(ib, b, k);
		} else {
			const uint32_t at   = 0x400u + 0x20u * (uint32_t)k;
			Ib             body = {.n = 0};
			if (r->how == 'j') {
				IbShadowTable(&body, b, k);
			} else {
				IbShadowSet(&body, r, ShadowMagic(k));
			}
			IbPad(&body);
			memcpy((uint8_t*)b->t.cpu + at, body.dw, body.n * 4u);
			IbPush(ib, Pkt3Gfx(0x3fu, 3)); // INDIRECT_BUFFER
			IbPush(ib, (uint32_t)(t + at));
			IbPush(ib, (uint32_t)((t + at) >> 32));
			IbPush(ib, body.n | (1u << 23)); // IB_SIZE, VALID
		}
	}
	IbShadowReadSh(ib, t, 0x080);
	IbWriteData(ib, t + 0x100, 0x5ad0e001u);
	IbShadowOff(ib);
	IbWriteData(ib, t + 0x104, 0x5ad0e002u);
	IbPad(ib);
	__sync_synchronize();
}

static const char* ShadowLoaded(uint32_t v, size_t k) {
	return v == ShadowSentinel(k, 0)   ? "set A"
	       : v == ShadowSentinel(k, 1) ? "set B"
	       : v == ShadowPre(k)         ? "unchanged"
	                                   : "other";
}

// Where each test value landed in S against its place (in either region set with load_off); true when every register
// of a class with shadowing on was shadowed exactly at its place.
static bool ShadowCheck1(const ShadowBufs* b, uint32_t classes, bool load_off) {
	const uint32_t* s = (const uint32_t*)b->s.cpu;
	const uint32_t* t = (const uint32_t*)b->t.cpu;
	printf("  marker after the writes %s, after shadowing off %s\n", t[0x40] == 0x5ad0e001u ? "landed" : "MISSING",
	       t[0x41] == 0x5ad0e002u ? "landed" : "MISSING");
	printf("  SH registers after the LOADs (set A: loaded from S%s; unchanged: not loaded), then after the writes:\n",
	       load_off ? " with loading on, set B: with loading off" : "");
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (ShadowIsSh(&k_shadow_regs[k])) {
			printf("    %-26s 0x%08x %-9s 0x%08x%s\n", k_shadow_regs[k].name, t[k], ShadowLoaded(t[k], k), t[0x20 + k],
			       t[0x20 + k] == ShadowMagic(k) ? "" : " (not the written value)");
		}
	}
	bool     all       = t[0x40] == 0x5ad0e001u;
	uint32_t off_found = 0; // test values of a class with shadowing off found in S
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		const ShadowReg* r     = &k_shadow_regs[k];
		const uint32_t   at    = ShadowAt(r, 0);
		const uint32_t   at_b  = ShadowAt(r, SHADOW_SET_B);
		uint32_t         found = 0;
		uint32_t         first = 0;
		for (uint32_t i = 0; i < SHADOW_SIZE / 4; i++) {
			if (s[i] == ShadowMagic(k)) {
				first = found == 0 ? i * 4 : first;
				found++;
			}
		}
		char where[96];
		if (found == 0) {
			snprintf(where, sizeof(where), "not in S (its place +0x%05x holds 0x%08x)", at, s[at / 4]);
		} else if (found == 1 && first == at) {
			snprintf(where, sizeof(where), "at its place +0x%05x", at);
		} else if (found == 1 && first == at_b) {
			snprintf(where, sizeof(where), "at its place in set B +0x%06x", at_b);
		} else {
			snprintf(where, sizeof(where), "%u times, first at +0x%06x (its place +0x%05x)", found, first, at);
		}
		// A compute register written by a graphics-type 0x63: which SH enable governs it is open, so it counts only
		// when both agree.
		const uint32_t bits = r->cls == 'c' && r->how == 't' ? (CC_GFX_SH | CC_CS_SH) : ShadowBit(r->cls);
		const bool     on   = (classes & bits) == bits;
		const bool     off  = (classes & bits) == 0;
		printf("  %-26s %-3s %-8s shadow %-3s: %s\n", r->name, ShadowClassName(r->cls), ShadowHowName(r),
		       on ? "on" : off ? "off" : "mix", where);
		if (on) {
			all = all && found == 1 && (first == at || (load_off && first == at_b));
		} else if (off) {
			off_found += found;
		}
	}
	// SETs made with shadowing off: the SH values before the LOADs, the UCONFIG values put back at the end.
	uint32_t leaked = 0;
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		const ShadowReg* r = &k_shadow_regs[k];
		for (uint32_t i = 0; i < SHADOW_SIZE / 4 && ShadowIsSh(r); i++) {
			leaked += s[i] == ShadowPre(k);
		}
		leaked += r->cls == 'u' && r->rest != 0 && s[ShadowAt(r, 0) / 4] == r->rest;
	}
	printf("  SET values written with shadowing off found in S: %u, of classes with shadowing off: %u\n", leaked, off_found);
	ShadowOthers(b);
	return all && leaked == 0 && off_found == 0;
}

static int CtxAlloc(Device* d, uint32_t* id) {
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	int r     = Ioctl(d->fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
	if (r == 0) {
		*id = ctx.out.alloc.ctx_id;
	}
	return r;
}

static void CtxFree(Device* d, uint32_t id) {
	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op     = AMDGPU_CTX_OP_FREE_CTX;
	ctx.in.ctx_id = id;
	Ioctl(d->fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
}

// A restore job: the SH test registers clobbered with shadowing off, CONTEXT_CONTROL with `load` (CC_UPDATE | CC_ALL:
// every load on, as a PAL / Mesa preamble; 0: the load enables left as amdgpu's CONTEXT_CONTROL set them for this IB)
// and shadowing on, the LOADs from S, the SH registers read to T + at, a marker, shadowing off.
static void ShadowIbRestore(Ib* ib, const ShadowBufs* b, uint32_t load, uint32_t at, uint32_t clobber) {
	IbContextControl(ib, load, CC_UPDATE);
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		if (ShadowIsSh(&k_shadow_regs[k])) {
			IbShadowSet(ib, &k_shadow_regs[k], clobber | (uint32_t)k);
		}
	}
	IbContextControl(ib, load, CC_UPDATE | CC_ALL);
	IbShadowLoads(ib, b->s.va, 0);
	IbShadowReadSh(ib, b->t.va, at);
	IbWriteData(ib, b->t.va + at + 0x7c, 0x5ad0e003u);
	IbShadowOff(ib);
	IbPad(ib);
}

// 2: every SH test register holds what S holds, 1: every one kept the clobber (nothing loaded), 0: anything else.
static int ShadowRestored(const ShadowBufs* b, uint32_t at, uint32_t clobber, const char* what) {
	const uint32_t* s        = (const uint32_t*)b->s.cpu;
	const uint32_t* t        = (const uint32_t*)b->t.cpu + at / 4;
	uint32_t        n        = 0;
	uint32_t        same     = 0;
	uint32_t        kept     = 0;
	uint32_t        shadowed = 0;
	for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
		const ShadowReg* r = &k_shadow_regs[k];
		if (!ShadowIsSh(r)) {
			continue;
		}
		const uint32_t want = s[ShadowAt(r, 0) / 4];
		n++;
		same += t[k] == want;
		kept += t[k] == (clobber | (uint32_t)k);
		shadowed += t[k] == want && want == ShadowMagic(k);
	}
	printf("  %s: %u of %u SH registers hold S's value (%u of them written there by the CP), %u not loaded, marker %s\n",
	       what, same, n, shadowed, kept, t[0x1f] == 0x5ad0e003u ? "landed" : "MISSING");
	if (same != n && kept != n) {
		for (size_t k = 0; k < ARRAY_SIZE(k_shadow_regs); k++) {
			const ShadowReg* r = &k_shadow_regs[k];
			if (ShadowIsSh(r)) {
				printf("    %-26s 0x%08x (S 0x%08x)\n", r->name, t[k], s[ShadowAt(r, 0) / 4]);
			}
		}
	}
	return same == n ? 2 : kept == n ? 1 : 0;
}

// shadow-restore, after the shadowing job: a second job loads S back with every load on (a PAL / Mesa preamble) and
// reads the SH registers: do they hold what the CP shadowed? The context and UCONFIG test registers are loaded too but
// cannot be read back (a COPY_DATA read of either stops the ring); the marker after the LOADs shows the CP ran them.
// Then twice with the load enables left as amdgpu set them: its CONTEXT_CONTROL before each IB turns every load on
// only after another context ran on the ring (gfx_v10_0_ring_emit_cntxcntl, AMDGPU_HAVE_CTX_SWITCH), so the LOADs
// should load after a second DRM context's job and not right after our own.
static bool ShadowRestore(Device* d, Common* c, ShadowBufs* b) {
	Ib ib = {.n = 0};
	ShadowIbRestore(&ib, b, CC_UPDATE | CC_ALL, 0x800, 0xc10b0000u);
	ShadowRun(d, c, b, &ib, "restore job, every load on");
	const int explicit_loads = ShadowRestored(b, 0x800, 0xc10b0000u, "  loaded back");
	uint32_t  other          = 0;
	int       r              = CtxAlloc(d, &other);
	if (r != 0) {
		printf("  second context: %s\n", ErrStr(r));
		return explicit_loads == 2;
	}
	ib.n = 0;
	ShadowIbRestore(&ib, b, 0, 0x880, 0xc20b0000u);
	ShadowRun(d, c, b, &ib, "restore job, amdgpu's load enables, right after our own job");
	const int same_ctx = ShadowRestored(b, 0x880, 0xc20b0000u, "  loaded back");
	ib.n = 0;
	IbWriteData(&ib, b->t.va + 0x9fc, 0x0be0be0bu);
	IbPad(&ib);
	const uint32_t own = d->ctx;
	d->ctx             = other;
	ShadowRun(d, c, b, &ib, "a job of a second DRM context");
	d->ctx = own;
	ib.n   = 0;
	ShadowIbRestore(&ib, b, 0, 0x900, 0xc30b0000u);
	ShadowRun(d, c, b, &ib, "restore job, amdgpu's load enables, after the other context");
	const int switched = ShadowRestored(b, 0x900, 0xc30b0000u, "  loaded back");
	CtxFree(d, other);
	printf("  amdgpu's load enables: %s\n",
	       same_ctx == 1 && switched == 2   ? "the LOADs load only after another context ran"
	       : same_ctx == 2 && switched == 2 ? "the LOADs loaded both times (not gated, or another process ran in between)"
	                                        : "neither pattern (see above)");
	return explicit_loads == 2;
}

// shadow / shadow-restore. SHADOW_CLASSES (e.g. "ctx,uc"; default "ctx,sh,cs,uc") picks the shadow enables of the
// first job, SHADOW_LOAD=0 adds its second LOADs with loading off; shadow-restore uses neither.
static bool DangerShadow(Device* d, Common* c, bool restore) {
	const char* cls_env  = getenv("SHADOW_CLASSES");
	const char* load_env = getenv("SHADOW_LOAD");
	uint32_t    classes  = CC_ALL;
	if (cls_env != NULL && !restore) {
		classes = (strstr(cls_env, "ctx") != NULL ? CC_CONTEXT : 0u) | (strstr(cls_env, "sh") != NULL ? CC_GFX_SH : 0u) |
		          (strstr(cls_env, "cs") != NULL ? CC_CS_SH : 0u) | (strstr(cls_env, "uc") != NULL ? CC_UCONFIG : 0u);
	}
	const bool load_off = !restore && load_env != NULL && load_env[0] == '0';
	printf("== danger %s: CP register shadowing (CONTEXT_CONTROL shadow enables, LOAD_*_REG addresses) on the gfx ring\n",
	       restore ? "shadow-restore" : "shadow");
	ShadowBufs b;
	if (!ShadowSetup(d, &b)) {
		return false;
	}
	printf("  S at 0x%" PRIx64 " (SH +0x%x, context +0x%x, UCONFIG +0x%x), CONTEXT_CONTROL 0x%08x 0x%08x%s\n", b.s.va,
	       SHADOW_SH_AT, SHADOW_CTX_AT, SHADOW_UC_AT, CC_UPDATE | CC_ALL, CC_UPDATE | classes,
	       load_off ? ", then 0x80000000 (loading off) and LOADs from set B (+1 MiB)" : "");
	Ib ib = {.n = 0};
	ShadowIb1(&ib, &b, classes, load_off);
	ShadowRun(d, c, &b, &ib, "shadowing job");
	bool ok = ShadowCheck1(&b, classes, load_off);
	if (restore) {
		ok = ShadowRestore(d, c, &b) && ok;
	}
	BoDestroy(d, &b.t);
	BoDestroy(d, &b.s);
	return ok;
}

static void IbClearState(Ib* ib, uint32_t cmd) {
	IbPush(ib, Pkt3Gfx(0x12u, 1));
	IbPush(ib, cmd);
}

// shadow-pushpop: libSceAgc's context push and pop as it writes them (AgcDcbContextStateOp in src/libs/agc.cpp:
// CONTEXT_CONTROL 0 / 0x80018001 then CLEAR_STATE CMD 1 for a push, 0 / 0x80018003 then CMD 2 for a pop; AMD's F32
// PFP/ME packet lists name CMD 1-3 PUSH_STATE, POP_STATE, PUSH_CLEAR_STATE) inside a set-up shadow. Three blocks in one
// job, S copied out after each step (COPY_DATA from memory):
// 1. markers written into the context places of S right before the push, the pop and a plain CLEAR_STATE: does any
//    of them write S (a marker replaced), is a SET between push and pop shadowed, one after the pop?
// 2. the push's and the pop's CONTEXT_CONTROL without CLEAR_STATE: what the CONTEXT_CONTROLs do on their own.
// 3. a register that can be read: SPI_SHADER_USER_DATA_PS_0 set with gfx SH shadowing on, a push with gfx SH shadowing
//    off too (0x80008001), a new value, libSceAgc's pop, the register read: does the pop load registers back from S?
//    Then the same without CLEAR_STATE. Context registers cannot be read (a COPY_DATA of one stops the ring).
static bool DangerShadowPushPop(Device* d, Common* c) {
	printf("== danger shadow-pushpop: libSceAgc's CONTEXT_CONTROL + CLEAR_STATE push and pop inside a set-up shadow\n");
	ShadowBufs b;
	if (!ShadowSetup(d, &b)) {
		return false;
	}
	const ShadowReg* tl   = ShadowRegNamed("PA_SC_SCREEN_SCISSOR_TL");
	const ShadowReg* br   = ShadowRegNamed("PA_SC_SCREEN_SCISSOR_BR");
	const ShadowReg* ps0  = ShadowRegNamed("SPI_SHADER_USER_DATA_PS_0"); // an SH register: COPY_DATA may read it
	const uint64_t   stl  = b.s.va + ShadowAt(tl, 0);
	const uint64_t   sbr  = b.s.va + ShadowAt(br, 0);
	const uint64_t   sps0 = b.s.va + ShadowAt(ps0, 0);
	const uint64_t   copy = b.t.va + 0xa00;
	Ib               ib   = {.n = 0};
	IbContextControl(&ib, CC_UPDATE, CC_UPDATE);
	IbContextControl(&ib, CC_UPDATE | CC_ALL, CC_UPDATE | CC_ALL);
	IbShadowLoads(&ib, b.s.va, 0);
	// 1
	IbShadowSet(&ib, tl, 0x7e100001u);
	IbShadowSet(&ib, br, 0x7e100002u);
	IbCopyMemToMem(&ib, stl, copy + 0);
	IbCopyMemToMem(&ib, sbr, copy + 4);
	IbWriteData(&ib, stl, 0x7e1000a1u);
	IbWriteData(&ib, sbr, 0x7e1000a2u);
	IbContextControl(&ib, 0, 0x80018001u); // the push
	IbClearState(&ib, 1);
	IbCopyMemToMem(&ib, stl, copy + 8);
	IbCopyMemToMem(&ib, sbr, copy + 12);
	IbShadowSet(&ib, tl, 0x7e100003u);
	IbCopyMemToMem(&ib, stl, copy + 16);
	IbWriteData(&ib, stl, 0x7e1000b1u);
	IbWriteData(&ib, sbr, 0x7e1000b2u);
	IbContextControl(&ib, 0, 0x80018003u); // the pop
	IbClearState(&ib, 2);
	IbCopyMemToMem(&ib, stl, copy + 20);
	IbCopyMemToMem(&ib, sbr, copy + 24);
	IbShadowSet(&ib, br, 0x7e100004u);
	IbCopyMemToMem(&ib, sbr, copy + 28);
	IbWriteData(&ib, stl, 0x7e1000c1u);
	IbWriteData(&ib, sbr, 0x7e1000c2u);
	IbClearState(&ib, 0); // a plain CLEAR_STATE
	IbCopyMemToMem(&ib, stl, copy + 32);
	IbCopyMemToMem(&ib, sbr, copy + 36);
	// 2
	IbContextControl(&ib, CC_UPDATE | CC_ALL, CC_UPDATE | CC_ALL);
	IbShadowSet(&ib, tl, 0x7e100011u);
	IbContextControl(&ib, 0, 0x80018001u);
	IbShadowSet(&ib, tl, 0x7e100013u);
	IbCopyMemToMem(&ib, stl, copy + 40);
	IbContextControl(&ib, 0, 0x80018003u);
	IbShadowSet(&ib, br, 0x7e100014u);
	IbCopyMemToMem(&ib, sbr, copy + 44);
	// 3
	IbContextControl(&ib, CC_UPDATE | CC_ALL, CC_UPDATE | CC_ALL);
	IbShadowSet(&ib, ps0, 0x7e200001u);
	IbContextControl(&ib, 0, 0x80008001u); // the push with gfx SH shadowing off too
	IbClearState(&ib, 1);
	IbShadowSet(&ib, ps0, 0x7e200002u);
	IbContextControl(&ib, 0, 0x80018003u); // libSceAgc's pop
	IbClearState(&ib, 2);
	IbCopyRegToMem(&ib, ps0->reg, copy + 48);
	IbCopyMemToMem(&ib, sps0, copy + 52);
	IbContextControl(&ib, CC_UPDATE | CC_ALL, CC_UPDATE | CC_ALL);
	IbShadowSet(&ib, ps0, 0x7e200011u);
	IbContextControl(&ib, 0, 0x80008001u);
	IbShadowSet(&ib, ps0, 0x7e200012u);
	IbContextControl(&ib, 0, 0x80018003u);
	IbCopyRegToMem(&ib, ps0->reg, copy + 56);
	IbCopyMemToMem(&ib, sps0, copy + 60);
	IbWriteData(&ib, b.t.va + 0x100, 0x5ad0e001u);
	IbShadowOff(&ib);
	IbWriteData(&ib, b.t.va + 0x104, 0x5ad0e002u);
	IbPad(&ib);
	ShadowRun(d, c, &b, &ib, "push/pop job");
	const uint32_t* t = (const uint32_t*)b.t.cpu;
	const uint32_t* v = t + 0xa00 / 4;
	printf("  marker after the steps %s, after shadowing off %s\n", t[0x40] == 0x5ad0e001u ? "landed" : "MISSING",
	       t[0x41] == 0x5ad0e002u ? "landed" : "MISSING");
	printf("  1. S at %s / %s:\n", tl->name, br->name);
	printf("    after SETs of 0x7e100001 / 0x7e100002              0x%08x / 0x%08x\n", v[0], v[1]);
	printf("    marks 0x7e1000a1 / a2, then the push              0x%08x / 0x%08x\n", v[2], v[3]);
	printf("    after a SET of _TL to 0x7e100003                   0x%08x\n", v[4]);
	printf("    marks 0x7e1000b1 / b2, then the pop               0x%08x / 0x%08x\n", v[5], v[6]);
	printf("    after a SET of _BR to 0x7e100004                   0x%08x\n", v[7]);
	printf("    marks 0x7e1000c1 / c2, then a plain CLEAR_STATE   0x%08x / 0x%08x\n", v[8], v[9]);
	printf("    context shadowing before the push: %s; the push %s; a SET between push and pop %s; the pop %s;\n",
	       v[0] == 0x7e100001u && v[1] == 0x7e100002u ? "on" : "NOT seen",
	       v[2] == 0x7e1000a1u && v[3] == 0x7e1000a2u ? "left S" : "wrote S",
	       v[4] == 0x7e100003u ? "was shadowed" : "was not shadowed",
	       v[5] == 0x7e1000b1u && v[6] == 0x7e1000b2u ? "left S" : "wrote S");
	printf("    a SET after the pop %s; the plain CLEAR_STATE %s\n", v[7] == 0x7e100004u ? "was shadowed" : "was not shadowed",
	       v[8] == 0x7e1000c1u && v[9] == 0x7e1000c2u ? "left S" : "wrote S");
	printf("  2. the CONTEXT_CONTROLs alone: a SET after the push's %s, a SET after the pop's %s\n",
	       v[10] == 0x7e100013u ? "was shadowed" : v[10] == 0x7e100011u ? "was not shadowed" : "(other)",
	       v[11] == 0x7e100014u ? "was shadowed" : "was not shadowed");
	printf("  3. %s 0x7e200001 shadowed, push with gfx SH shadowing off, 0x7e200002, pop: register 0x%08x, S 0x%08x\n",
	       ps0->name, v[12], v[13]);
	printf("     the same without CLEAR_STATE (0x7e200011, then 0x7e200012): register 0x%08x, S 0x%08x\n", v[14], v[15]);
	printf("     the pop %s\n", v[12] == 0x7e200001u   ? "loaded the register back from S"
	                            : v[12] == 0x7e200002u ? "left the register"
	                                                   : "left something else in the register");
	ShadowOthers(&b);
	const bool ok = t[0x40] == 0x5ad0e001u && t[0x41] == 0x5ad0e002u;
	BoDestroy(d, &b.t);
	BoDestroy(d, &b.s);
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
	printf("danger cases (owner approval required, run alone): vmfault vmfault-read long dot2c dot2x gds gdsdma crossqueue uconfig tag342 mecforms atomicmem condib condexec exitguard labelvis clock regindex-{sh,ctx,uc,uc342,sh-nr,ctx-nr} shadow shadow-restore shadow-pushpop\n");
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
		} else if (strcmp(danger, "vmfault-read") == 0) {
			ok = DangerVmFaultRead(&d, &c);
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
		} else if (strcmp(danger, "condib") == 0) {
			ok = DangerCondIb(&d, &c);
		} else if (strcmp(danger, "condexec") == 0) {
			ok = DangerCondExec(&d, &c);
		} else if (strcmp(danger, "exitguard") == 0) {
			ok = DangerExitGuard(&d, &c);
		} else if (strcmp(danger, "atomicmem") == 0) {
			ok = DangerAtomicMem(&d, &c);
		} else if (strcmp(danger, "mecforms") == 0) {
			ok = DangerMecForms(&d, &c);
		} else if (strcmp(danger, "regindex-shc-nr") == 0) {
			ok = DangerRegIndexNoRead(&d, &c, 0x63u, (0xb900u - 0xb000u) / 4, "COMPUTE_USER_DATA_0");
		} else if (strcmp(danger, "regindex-ctx-nr") == 0) {
			ok = DangerRegIndexNoRead(&d, &c, 0x9fu, (0x28030u - 0x28000u) / 4, "PA_SC_SCREEN_SCISSOR_TL");
		} else if (strcmp(danger, "shadow") == 0) {
			ok = DangerShadow(&d, &c, false);
		} else if (strcmp(danger, "shadow-restore") == 0) {
			ok = DangerShadow(&d, &c, true);
		} else if (strcmp(danger, "shadow-pushpop") == 0) {
			ok = DangerShadowPushPop(&d, &c);
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
