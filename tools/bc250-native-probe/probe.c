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

// Copies ib into ib_bo and submits it with the given buffers resident. Returns the fence seq.
static int SubmitOn(Device* d, uint32_t ip, uint32_t ring, Bo* ib_bo, const Ib* ib, Bo* const* bos,
                    uint32_t num_bos, uint64_t* seq) {
	memcpy(ib_bo->cpu, ib->dw, ib->n * 4u);
	__sync_synchronize();

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
	ib_info.va_start = ib_bo->va;
	ib_info.ib_bytes = ib->n * 4u;
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
                     double timeout_sec) {
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
	Dispatch p     = {c->code.va, 64, 16, 2, false, 64, 1, ud};
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
	if (!RunLanes(d, c, k_dot2c_w64, sizeof(k_dot2c_w64), NULL, out, 15.0)) {
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
	if (!RunLanes(d, c, k_dot2x_w64, sizeof(k_dot2x_w64), NULL, out, 15.0)) {
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
	bool     ok = RunLanes(d, c, k_gds_w64, sizeof(k_gds_w64), &gds, out, 15.0);
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

static void Usage(const char* argv0) {
	printf("usage: %s [--device PATH] [--max-alloc-gib N] [--danger CASE] [probe...]\n", argv0);
	printf("probes: info va alloc dispatch identity bandwidth coherency dmabuf submit (default: all)\n");
	printf("danger cases (owner approval required, run alone): vmfault long dot2c dot2x gds crossqueue uconfig\n");
}

int main(int argc, char** argv) {
	const char* path        = "/dev/dri/renderD128";
	uint64_t    max_alloc   = 4 * GiB;
	const char* danger      = NULL;
	const char* selected[16];
	int         num_selected = 0;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
			path = argv[++i];
		} else if (strcmp(argv[i], "--max-alloc-gib") == 0 && i + 1 < argc) {
			max_alloc = strtoull(argv[++i], NULL, 10) * GiB;
		} else if (strcmp(argv[i], "--danger") == 0 && i + 1 < argc) {
			danger = argv[++i];
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
	if (danger != NULL) {
		bool ok;
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
		} else if (strcmp(danger, "crossqueue") == 0) {
			ok = DangerCrossQueue(&d, &c);
		} else if (strcmp(danger, "uconfig") == 0) {
			ok = DangerUconfig(&d, &c);
		} else {
			fprintf(stderr, "unknown danger case '%s'\n", danger);
			return 1;
		}
		BoDestroy(&d, &c.code);
		BoDestroy(&d, &c.ib);
		close(d.fd);
		return ok ? 0 : 2;
	}
	static const char* const order[] = {"info",      "va",        "alloc",  "dispatch", "identity",
	                                    "bandwidth", "coherency", "dmabuf", "submit"};
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
