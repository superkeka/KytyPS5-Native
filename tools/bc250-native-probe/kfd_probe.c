// KFD user-mode compute queue probe for the BC-250: can Kyty submit PS5 compute command buffers
// the PS5 way, by writing PM4 into a ring of its own and ringing a doorbell, with no CS ioctl?
//   1. a KFD PM4 compute queue in the GPU VM of a render-node file (AMDKFD_IOC_ACQUIRE_VM);
//   2. WRITE_DATA into KFD-allocated memory;
//   3. WRITE_DATA into a GEM buffer created on the render node (shared VM);
//   4. WAIT_REG_MEM on a label the CPU sets 50 ms later, then WRITE_DATA: wake-up latency;
//   5. submission cost: doorbell to completion of a single WRITE_DATA, many times;
//   6. KFD memory written by a render-node CS (compute ring) that does NOT list it: does KFD's
//      residency make guest memory usable without per-submission BO lists?
//   7. KFD memory exported as a dma-buf and mapped for the CPU at an inner offset;
//   8 ("gds" argument): 64 lanes of ds_add_rtn_u32 on GDS dword 0 from a KFD queue (needs the KFD
//      VMIDs' GDS partition, which kyty-lodhw writes: GDS_VMID8..15).
// Build: cc -O2 -o kfd_probe kfd_probe.c. Needs /dev/kfd (no root).
#define _GNU_SOURCE
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "shaders.h"

static int kfd = -1, drm = -1;
static uint32_t gpu_id;

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
static uint32_t Pkt3(uint32_t op, uint32_t body) { return 0xc0000000u | ((body - 1) << 16) | (op << 8) | 2u; } // shader type: compute

struct Mem {
	uint64_t va, size, handle;
	void*    cpu;
};

// KFD memory at `va`, mapped for the GPU and (GTT) for the CPU through the render node.
static int Alloc(struct Mem* m, uint64_t va, uint64_t size, uint32_t flags) {
	struct kfd_ioctl_alloc_memory_of_gpu_args a = {.va_addr = va, .size = size, .gpu_id = gpu_id, .flags = flags};
	int r = Ioctl(kfd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &a);
	if (r) {
		fprintf(stderr, "ALLOC_MEMORY_OF_GPU va 0x%llx size 0x%llx flags 0x%x: %s\n", (unsigned long long)va,
		        (unsigned long long)size, flags, strerror(-r));
		return r;
	}
	m->va = va;
	m->size = size;
	m->handle = a.handle;
	struct kfd_ioctl_map_memory_to_gpu_args mp = {.handle = a.handle, .device_ids_array_ptr = (uintptr_t)&gpu_id, .n_devices = 1};
	if ((r = Ioctl(kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &mp))) {
		fprintf(stderr, "MAP_MEMORY_TO_GPU: %s\n", strerror(-r));
		return r;
	}
	m->cpu = NULL;
	if (flags & KFD_IOC_ALLOC_MEM_FLAGS_GTT) {
		m->cpu = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm, a.mmap_offset);
		if (m->cpu == MAP_FAILED) {
			fprintf(stderr, "mmap of KFD memory through the render node: %s\n", strerror(errno));
			m->cpu = NULL;
			return -errno;
		}
	}
	return 0;
}

int main(int argc, char** argv) {
	// "hold N": keep the queue alive N seconds after the first test, for reading registers meanwhile.
	const int hold = argc > 2 && strcmp(argv[1], "hold") == 0 ? atoi(argv[2]) : 0;
	const int gds  = argc > 1 && strcmp(argv[1], "gds") == 0;
	// The GPU node: the one with SIMDs.
	char path[128];
	for (int n = 0; n < 8 && gpu_id == 0; n++) {
		snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/gpu_id", n);
		FILE* f = fopen(path, "r");
		if (f) {
			unsigned v = 0;
			if (fscanf(f, "%u", &v) == 1 && v != 0) {
				gpu_id = v;
			}
			fclose(f);
		}
	}
	kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
	drm = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	struct kfd_ioctl_get_version_args ver = {0};
	int r = Ioctl(kfd, AMDKFD_IOC_GET_VERSION, &ver);
	printf("kfd %d, render %d, gpu_id %u, KFD ioctl version %u.%u (%s)\n", kfd, drm, gpu_id, ver.major_version, ver.minor_version,
	       r ? strerror(-r) : "ok");
	if (kfd < 0 || drm < 0 || gpu_id == 0 || r) {
		return 1;
	}
	struct kfd_ioctl_acquire_vm_args acq = {.drm_fd = (uint32_t)drm, .gpu_id = gpu_id};
	r = Ioctl(kfd, AMDKFD_IOC_ACQUIRE_VM, &acq);
	printf("ACQUIRE_VM (KFD shares the render node's GPU VM): %s\n", r ? strerror(-r) : "ok");
	if (r) {
		return 1;
	}
	struct kfd_process_device_apertures ap[4];
	struct kfd_ioctl_get_process_apertures_new_args apn = {.kfd_process_device_apertures_ptr = (uintptr_t)ap, .num_of_nodes = 4};
	r = Ioctl(kfd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &apn);
	uint64_t base = 0;
	for (uint32_t i = 0; r == 0 && i < apn.num_of_nodes; i++) {
		if (ap[i].gpu_id == gpu_id) {
			printf("GPUVM aperture 0x%llx-0x%llx, scratch 0x%llx, lds 0x%llx\n", (unsigned long long)ap[i].gpuvm_base,
			       (unsigned long long)ap[i].gpuvm_limit, (unsigned long long)ap[i].scratch_base, (unsigned long long)ap[i].lds_base);
			base = ap[i].gpuvm_base;
		}
	}
	// Probe VAs well clear of guest memory and of Kyty's IB at 0x400000000000.
	uint64_t va = 0x500000000000ull;
	if (base > va) {
		va = base + (1ull << 32);
	}
	const uint32_t gtt = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
	                     KFD_IOC_ALLOC_MEM_FLAGS_COHERENT | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
	struct Mem ring, ptrs, eop, ctx, data;
	if (Alloc(&ring, va, 64 << 10, gtt) || Alloc(&ptrs, va + 0x100000, 4096, gtt) ||
	    Alloc(&eop, va + 0x200000, 4096, KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE) ||
	    Alloc(&data, va + 0x3000000, 4096, gtt)) {
		return 2;
	}
	memset(ring.cpu, 0, ring.size);
	memset(ptrs.cpu, 0, ptrs.size);
	memset(data.cpu, 0, data.size);
	volatile uint64_t* wptr = (volatile uint64_t*)ptrs.cpu;
	volatile uint64_t* rptr = (volatile uint64_t*)((uint8_t*)ptrs.cpu + 64);

	struct kfd_ioctl_create_queue_args q = {
	    .ring_base_address = ring.va,
	    .write_pointer_address = ptrs.va,
	    .read_pointer_address = ptrs.va + 64,
	    .ring_size = (uint32_t)ring.size,
	    .gpu_id = gpu_id,
	    .queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE,
	    .queue_percentage = 100,
	    .queue_priority = 7,
	    .eop_buffer_address = eop.va,
	    .eop_buffer_size = eop.size,
	    .ctx_save_restore_address = ctx.va,
	    .ctx_save_restore_size = 8368128,
	    .ctl_stack_size = 12288,
	};
	// The context save area must be mapped at exactly cwsr_size + the debugger area (32 bytes per wave),
	// page aligned; the wave count is the kernel's, so try the plausible ones.
	static const uint32_t kWaves[] = {768, 960, 480, 1536};
	r = -EINVAL;
	for (uint32_t i = 0; i < 4 && r != 0; i++) {
		const uint64_t debug = ((uint64_t)kWaves[i] * 32 + 63) & ~63ull;
		const uint64_t total = (8368128 + debug + 4095) & ~4095ull;
		if (Alloc(&ctx, va + 0x1000000 + (uint64_t)i * 0x1000000, total, gtt)) {
			continue;
		}
		q.ctx_save_restore_address = ctx.va;
		r = Ioctl(kfd, AMDKFD_IOC_CREATE_QUEUE, &q);
		printf("CREATE_QUEUE (PM4 compute) with a 0x%llx-byte save area (%u waves): %s\n", (unsigned long long)total, kWaves[i],
		       r ? strerror(-r) : "ok");
	}
	printf("queue %u, doorbell offset 0x%llx\n", q.queue_id, (unsigned long long)q.doorbell_offset);
	if (r) {
		return 3;
	}
	// The process doorbell slice on SOC15: 1024 doorbells of 8 bytes, mapped whole; the queue's doorbell
	// is the low bits of the offset KFD returned.
	const uint64_t slice = 8192;
	uint8_t*       db_page = mmap(NULL, slice, PROT_READ | PROT_WRITE, MAP_SHARED, kfd, q.doorbell_offset & ~(slice - 1));
	if (db_page == MAP_FAILED) {
		printf("doorbell mmap: %s\n", strerror(errno));
		return 4;
	}
	volatile uint64_t* doorbell = (volatile uint64_t*)(db_page + (q.doorbell_offset & (slice - 1)));
	uint32_t*          rb       = ring.cpu;
	uint64_t           w        = 0;
	const uint32_t     mask     = (uint32_t)(ring.size / 4 - 1);
#define PUSH(v) (rb[(w++) & mask] = (v))
	// WRITE_DATA: dst_sel 5 (memory), write confirm.
#define WRITE(addr, value)                         \
	do {                                           \
		PUSH(Pkt3(0x37, 4));                       \
		PUSH((5u << 8) | (1u << 20));              \
		PUSH((uint32_t)(addr));                    \
		PUSH((uint32_t)((uint64_t)(addr) >> 32));  \
		PUSH(value);                               \
	} while (0)
#define KICK()                    \
	do {                          \
		__sync_synchronize();     \
		*wptr    = w;             \
		__sync_synchronize();     \
		*doorbell = w;            \
	} while (0)

	// 2. WRITE_DATA into KFD memory.
	volatile uint32_t* d = data.cpu;
	WRITE(data.va, 0xcafe0001u);
	double t0 = Now();
	KICK();
	while (d[0] != 0xcafe0001u && Now() - t0 < 2.0) {
	}
	printf("WRITE_DATA to KFD memory: %s after %.1f us (rptr %llu, wptr %llu)\n", d[0] == 0xcafe0001u ? "landed" : "NOT landed",
	       (Now() - t0) * 1e6, (unsigned long long)*rptr, (unsigned long long)w);
	if (d[0] != 0xcafe0001u) {
		return 5;
	}
	if (hold > 0) {
		char fdinfo[64], line[128];
		snprintf(fdinfo, sizeof(fdinfo), "/proc/self/fdinfo/%d", drm);
		FILE* fi = fopen(fdinfo, "r");
		while (fi && fgets(line, sizeof(line), fi)) {
			if (strncmp(line, "pasid:", 6) == 0) {
				printf("render node %s", line);
			}
		}
		if (fi) {
			fclose(fi);
		}
		printf("holding the queue %d s\n", hold);
		fflush(stdout);
		sleep(hold);
	}

	if (gds) {
		// 8. A wave64 dispatch: s[0:1] = destination, each lane adds 1 to GDS dword 0 and stores what it got.
		struct Mem code, out;
		if (Alloc(&code, va + 0x6000000, 65536, gtt) || Alloc(&out, va + 0x7000000, 65536, gtt)) {
			return 7;
		}
		memcpy(code.cpu, k_gds_w64, sizeof(k_gds_w64));
		memset(out.cpu, 0xff, 256);
		d[5] = 0;
		__sync_synchronize();
#define SETSH(reg, n)                                   \
	do {                                                \
		PUSH(Pkt3(0x76, (n) + 1));                      \
		PUSH(((reg) - 0xb000u) / 4u);                   \
	} while (0)
		SETSH(0xb830, 2); PUSH((uint32_t)(code.va >> 8)); PUSH((uint32_t)(code.va >> 40));            // COMPUTE_PGM_LO/HI
		SETSH(0xb848, 2); PUSH(((16 + 3) / 4 - 1) | (0xc0u << 12) | (1u << 21) | (1u << 30)); PUSH(2u << 1); // RSRC1/2: 16 VGPRs, 2 user SGPRs
		SETSH(0xb8a0, 1); PUSH(0);                                                                      // RSRC3
		SETSH(0xb854, 1); PUSH(0);                                                                      // RESOURCE_LIMITS
		SETSH(0xb860, 1); PUSH(0);                                                                      // TMPRING_SIZE
		SETSH(0xb858, 2); PUSH(0xffffffffu); PUSH(0xffffffffu);                                         // STATIC_THREAD_MGMT_SE0/1
		SETSH(0xb864, 2); PUSH(0xffffffffu); PUSH(0xffffffffu);                                         // SE2/3
		SETSH(0xb810, 3); PUSH(0); PUSH(0); PUSH(0);                                                    // START_X/Y/Z
		SETSH(0xb81c, 3); PUSH(64); PUSH(1); PUSH(1);                                                   // NUM_THREAD_X/Y/Z
		SETSH(0xb900, 2); PUSH((uint32_t)out.va); PUSH((uint32_t)(out.va >> 32));                       // USER_DATA_0/1
		PUSH(Pkt3(0x15, 4)); PUSH(1); PUSH(1); PUSH(1); PUSH(1u | (1u << 2) | (1u << 6));              // DISPATCH_DIRECT, wave64
		PUSH(Pkt3(0x46, 1)); PUSH(7u | (4u << 8));                                                      // EVENT_WRITE CS_PARTIAL_FLUSH
		WRITE(data.va + 20, 0xcafe0008u);
		t0 = Now();
		KICK();
		while (d[5] != 0xcafe0008u && Now() - t0 < 5.0) {
		}
		// GDS at base 0 is shared with the render node's jobs, so dword 0 starts wherever they left it:
		// a working GDS gives 64 consecutive values from there.
		const uint32_t* o  = out.cpu;
		uint32_t        lo = o[0];
		for (int i = 1; i < 64; i++) {
			lo = o[i] < lo ? o[i] : lo;
		}
		uint64_t seen = 0;
		for (int i = 0; i < 64; i++) {
			if (o[i] - lo < 64) {
				seen |= 1ull << (o[i] - lo);
			}
		}
		printf("GDS dispatch on the KFD queue: %s after %.1f us; lane values 0x%x 0x%x 0x%x ... 64 consecutive: %s\n",
		       d[5] == 0xcafe0008u ? "done" : "NOT done", (Now() - t0) * 1e6, o[0], o[1], o[2], seen == ~0ull ? "YES" : "no");
		return d[5] == 0xcafe0008u && seen == ~0ull ? 0 : 8;
	}

	// 3. A GEM buffer of the render node, at a VA of the same VM.
	union drm_amdgpu_gem_create gc = {.in = {.bo_size = 65536, .alignment = 65536, .domains = AMDGPU_GEM_DOMAIN_GTT,
	                                         .domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED}};
	r = Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_CREATE, &gc);
	const uint64_t gem_va = va + 0x4000000;
	struct drm_amdgpu_gem_va gv = {.handle = gc.out.handle, .operation = AMDGPU_VA_OP_MAP,
	                               .flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE, .va_address = gem_va, .map_size = 65536};
	if (r == 0) {
		r = Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_VA, &gv);
	}
	union drm_amdgpu_gem_mmap gm = {.in = {.handle = gc.out.handle}};
	if (r == 0) {
		r = Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_MMAP, &gm);
	}
	volatile uint32_t* gem = r == 0 ? mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_SHARED, drm, gm.out.addr_ptr) : MAP_FAILED;
	if (gem == MAP_FAILED) {
		printf("GEM buffer setup: %s\n", strerror(-r));
		return 6;
	}
	gem[0] = 0;
	WRITE(gem_va, 0xcafe0002u);
	t0 = Now();
	KICK();
	while (gem[0] != 0xcafe0002u && Now() - t0 < 2.0) {
	}
	printf("WRITE_DATA to a render-node GEM buffer (shared VM): %s after %.1f us\n",
	       gem[0] == 0xcafe0002u ? "landed" : "NOT landed", (Now() - t0) * 1e6);

	// 4. WAIT_REG_MEM on a label the CPU sets 50 ms later (memory, equal, poll 4), then a write.
	d[1] = 0;
	d[2] = 0;
	PUSH(Pkt3(0x3c, 6));
	PUSH(3u | (1u << 4));
	PUSH((uint32_t)(data.va + 4));
	PUSH((uint32_t)((data.va + 4) >> 32));
	PUSH(1u);
	PUSH(0xffffffffu);
	PUSH(4u);
	WRITE(data.va + 8, 0xcafe0003u);
	KICK();
	usleep(50000);
	const int early = d[2] == 0xcafe0003u;
	t0 = Now();
	d[1] = 1;
	__sync_synchronize();
	while (d[2] != 0xcafe0003u && Now() - t0 < 2.0) {
	}
	printf("WAIT_REG_MEM on a CPU-set label: %s; released %.1f us after the CPU store\n",
	       early ? "passed EARLY (bad)" : "held until set", (Now() - t0) * 1e6);

	// 5. Doorbell to completion of one WRITE_DATA, 2000 times.
	double sum = 0, worst = 0;
	for (uint32_t i = 0; i < 2000; i++) {
		WRITE(data.va + 12, 0x10000u + i);
		t0 = Now();
		KICK();
		while (d[3] != 0x10000u + i && Now() - t0 < 1.0) {
		}
		const double dt = Now() - t0;
		sum += dt;
		worst = dt > worst ? dt : worst;
	}
	printf("submit to done, single WRITE_DATA: %.1f us average, %.1f us worst (CS ioctl path: ~400-1100 us)\n",
	       sum / 2000 * 1e6, worst * 1e6);

	// 6. A CS on render-node compute ring 0 writes KFD memory, listing only its own IB buffer.
	{
		union drm_amdgpu_ctx c = {.in = {.op = AMDGPU_CTX_OP_ALLOC_CTX}};
		r = Ioctl(drm, DRM_IOCTL_AMDGPU_CTX, &c);
		union drm_amdgpu_gem_create ic = {.in = {.bo_size = 65536, .alignment = 65536, .domains = AMDGPU_GEM_DOMAIN_GTT,
		                                         .domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED}};
		r = r ? r : Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_CREATE, &ic);
		const uint64_t ib_va = va + 0x5000000;
		struct drm_amdgpu_gem_va iv = {.handle = ic.out.handle, .operation = AMDGPU_VA_OP_MAP,
		                               .flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_EXECUTABLE, .va_address = ib_va, .map_size = 65536};
		r = r ? r : Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_VA, &iv);
		union drm_amdgpu_gem_mmap im = {.in = {.handle = ic.out.handle}};
		r = r ? r : Ioctl(drm, DRM_IOCTL_AMDGPU_GEM_MMAP, &im);
		uint32_t* ib = r ? MAP_FAILED : mmap(NULL, 65536, PROT_READ | PROT_WRITE, MAP_SHARED, drm, im.out.addr_ptr);
		if (ib == MAP_FAILED) {
			printf("render-node CS setup: %s\n", strerror(-r));
		} else {
			d[4] = 0;
			uint32_t n = 0;
			ib[n++] = Pkt3(0x37, 4);
			ib[n++] = (5u << 8) | (1u << 20);
			ib[n++] = (uint32_t)(data.va + 16);
			ib[n++] = (uint32_t)((data.va + 16) >> 32);
			ib[n++] = 0xcafe0006u;
			while (n & 7) {
				ib[n++] = 0xffff1000u;
			}
			struct drm_amdgpu_bo_list_entry e = {.bo_handle = ic.out.handle};
			struct drm_amdgpu_bo_list_in list = {.operation = ~0u, .list_handle = ~0u, .bo_number = 1, .bo_info_size = sizeof(e),
			                                     .bo_info_ptr = (uintptr_t)&e};
			struct drm_amdgpu_cs_chunk_ib ci = {.va_start = ib_va, .ib_bytes = n * 4, .ip_type = AMDGPU_HW_IP_COMPUTE};
			struct drm_amdgpu_cs_chunk ch[2] = {{AMDGPU_CHUNK_ID_IB, sizeof(ci) / 4, (uintptr_t)&ci},
			                                    {AMDGPU_CHUNK_ID_BO_HANDLES, sizeof(list) / 4, (uintptr_t)&list}};
			uint64_t           cp[2] = {(uintptr_t)&ch[0], (uintptr_t)&ch[1]};
			union drm_amdgpu_cs cs   = {.in = {.ctx_id = c.out.alloc.ctx_id, .num_chunks = 2, .chunks = (uintptr_t)cp}};
			t0 = Now();
			r = Ioctl(drm, DRM_IOCTL_AMDGPU_CS, &cs);
			while (r == 0 && d[4] != 0xcafe0006u && Now() - t0 < 2.0) {
			}
			printf("render-node CS (not listing it) writes KFD memory: CS %s, value %s\n", r ? strerror(-r) : "ok",
			       d[4] == 0xcafe0006u ? "LANDED" : "not landed");
		}
	}

	// 7. KFD memory as a dma-buf, mapped for the CPU at an inner offset (a guest view of a block).
	{
		struct Mem big;
		if (Alloc(&big, va + 0x8000000, 16 << 20, gtt) == 0) {
			struct kfd_ioctl_export_dmabuf_args ex = {.handle = big.handle, .flags = O_RDWR | O_CLOEXEC};
			r = Ioctl(kfd, AMDKFD_IOC_EXPORT_DMABUF, &ex);
			printf("EXPORT_DMABUF of KFD memory: %s\n", r ? strerror(-r) : "ok");
			if (r == 0) {
				volatile uint32_t* inner = mmap(NULL, 1 << 20, PROT_READ | PROT_WRITE, MAP_SHARED, ex.dmabuf_fd, 4 << 20);
				if (inner == MAP_FAILED) {
					printf("dma-buf mmap at inner offset 4 MiB: %s\n", strerror(errno));
				} else {
					((volatile uint32_t*)big.cpu)[(4 << 20) / 4 + 3] = 0x1234abcdu;
					printf("dma-buf mmap at inner offset 4 MiB: ok, aliasing %s\n", inner[3] == 0x1234abcdu ? "correct" : "WRONG");
				}
			}
		}
	}

	struct kfd_ioctl_destroy_queue_args dq = {.queue_id = q.queue_id};
	r = Ioctl(kfd, AMDKFD_IOC_DESTROY_QUEUE, &dq);
	printf("DESTROY_QUEUE: %s\n", r ? strerror(-r) : "ok");
	return 0;
}
