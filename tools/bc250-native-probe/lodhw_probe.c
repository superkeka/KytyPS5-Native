// Is the PRT LOD statistics reporter (CP_PRT_LOD_STATS_CNTL0..3, the block the PS5 kernel and ME drive
// for GET_LOD_STATS) alive on the BC-250?
//
// This half runs unprivileged: it reserves a VMID for its VM (the kernel then writes that VMID into
// RLC_SPM_MC_CNTL on the next flush), maps a 64 KiB report buffer filled with 0xff at a fixed GPU VA,
// submits one NOP IB so the VM is flushed with the reserved VMID, and polls the buffer. The root half
// (lodhw-root.sh) reads the VMID and asks the block for one report into the buffer, the way the PS5
// kernel's gc_stop_lod_report does (REPORT_AND_RESET, CACHE_POLICY, bit 29).
// Build: cc -O2 -o lodhw_probe lodhw_probe.c (needs the amdgpu uAPI headers).

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

#define KIB 1024ull

static int g_fd = -1;

static int Ioctl(unsigned long req, void* arg) {
	int r;
	do {
		r = ioctl(g_fd, req, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

static int Fail(const char* what, int r) {
	printf("FAIL %s: %s\n", what, strerror(-r));
	return 1;
}

static int GemCreate(uint64_t size, uint64_t flags, uint32_t* handle) {
	union drm_amdgpu_gem_create c;
	memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = 64 * KIB;
	c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
	c.in.domain_flags = flags;
	int r             = Ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE, &c);
	*handle           = c.out.handle;
	return r;
}

static int VaMap(uint32_t handle, uint64_t va, uint64_t size) {
	struct drm_amdgpu_gem_va v;
	memset(&v, 0, sizeof(v));
	v.handle     = handle;
	v.operation  = AMDGPU_VA_OP_MAP;
	v.flags      = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
	v.va_address = va;
	v.map_size   = size;
	return Ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &v);
}

static void* CpuMap(uint32_t handle, uint64_t size) {
	union drm_amdgpu_gem_mmap gm;
	memset(&gm, 0, sizeof(gm));
	gm.in.handle = handle;
	if (Ioctl(DRM_IOCTL_AMDGPU_GEM_MMAP, &gm) != 0) {
		return NULL;
	}
	void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_fd, (off_t)gm.out.addr_ptr);
	return p == MAP_FAILED ? NULL : p;
}

static void Dump(const volatile uint32_t* rep, uint32_t bytes) {
	uint32_t changed = 0, first = ~0u;
	for (uint32_t i = 0; i < bytes / 4; i++) {
		if (rep[i] != 0xffffffffu) {
			changed++;
			first = first == ~0u ? i : first;
		}
	}
	printf("  %u of %u dwords changed (first at +0x%x); +0x4 = 0x%08x, +0x83c = 0x%08x\n", changed, bytes / 4,
	       first == ~0u ? 0 : first * 4, rep[1], rep[0x83c / 4]);
	for (uint32_t row = 0; row < 0x60 / 4; row += 8) {
		printf("  +0x%03x:", row * 4);
		for (uint32_t i = 0; i < 8; i++) {
			printf(" %08x", rep[row + i]);
		}
		printf("\n");
	}
	for (uint32_t base = 0x820; base < 0x880; base += 0x20) {
		printf("  +0x%03x:", base);
		for (uint32_t i = 0; i < 8; i++) {
			printf(" %08x", rep[base / 4 + i]);
		}
		printf("\n");
	}
}

int main(int argc, char** argv) {
	const unsigned wait_s  = argc > 1 ? (unsigned)atoi(argv[1]) : 120;
	const int      reserve = !(argc > 2 && strcmp(argv[2], "noreserve") == 0);
	g_fd                  = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
	if (g_fd < 0) {
		printf("FAIL open: %s\n", strerror(errno));
		return 1;
	}
	int r = 0;
	if (reserve) {
		union drm_amdgpu_vm vm;
		memset(&vm, 0, sizeof(vm));
		vm.in.op = AMDGPU_VM_OP_RESERVE_VMID;
		if ((r = Ioctl(DRM_IOCTL_AMDGPU_VM, &vm)) != 0) {
			return Fail("VM_OP_RESERVE_VMID", r);
		}
		printf("ok   VMID reserved for this VM\n");
	}

	union drm_amdgpu_ctx ctx;
	memset(&ctx, 0, sizeof(ctx));
	ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	if ((r = Ioctl(DRM_IOCTL_AMDGPU_CTX, &ctx)) != 0) {
		return Fail("CTX alloc", r);
	}

	// The report buffer stays mapped and valid in the VM whether or not a CS lists it: the reporter
	// writes it outside any job.
	const uint64_t kRep = 64 * KIB, rep_va = 0x123400000000ull;
	uint32_t       rep_bo = 0;
	if ((r = GemCreate(kRep, AMDGPU_GEM_CREATE_VM_ALWAYS_VALID | AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED, &rep_bo)) != 0) {
		return Fail("report GEM_CREATE (VM_ALWAYS_VALID)", r);
	}
	if ((r = VaMap(rep_bo, rep_va, kRep)) != 0) {
		return Fail("report GEM_VA", r);
	}
	volatile uint32_t* rep = CpuMap(rep_bo, kRep);
	if (rep == NULL) {
		return Fail("report mmap", -errno);
	}
	memset((void*)rep, 0xff, kRep);
	__sync_synchronize();

	uint32_t ib_bo = 0;
	if ((r = GemCreate(64 * KIB, 0, &ib_bo)) != 0) {
		return Fail("IB GEM_CREATE", r);
	}
	const uint64_t ib_va = 0x123500000000ull;
	if ((r = VaMap(ib_bo, ib_va, 64 * KIB)) != 0) {
		return Fail("IB GEM_VA", r);
	}
	uint32_t* ib = CpuMap(ib_bo, 64 * KIB);
	if (ib == NULL) {
		return Fail("IB mmap", -errno);
	}
	// No register reads here: COPY_DATA from CP_ME0_PIPE0_VMID (and from CP_VMID) in a user IB hangs
	// the gfx ring on the BC-250. The VMID comes from IH_VMID_n_LUT (pasid) in the root half instead.
	uint32_t n = 0;
	ib[n++]    = 0xffff1000u;
	while (n & 7u) {
		ib[n++] = 0xffff1000u; // type-3 NOP filler
	}
	__sync_synchronize();

	struct drm_amdgpu_bo_list_entry entries[2];
	memset(entries, 0, sizeof(entries));
	entries[0].bo_handle = ib_bo;
	entries[1].bo_handle = rep_bo;
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
	if ((r = Ioctl(DRM_IOCTL_AMDGPU_CS, &cs)) != 0) {
		return Fail("CS", r);
	}
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
	printf("ok   NOP IB ran\n");

	// With a helper socket (argv[3]): ask kyty-lodhw for one report the way Kyty does, by PASID.
	if (argc > 3) {
		char     path[64];
		unsigned pasid = 0;
		snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", g_fd);
		FILE* fi = fopen(path, "r");
		char  line[128];
		while (fi != NULL && fgets(line, sizeof(line), fi) != NULL) {
			sscanf(line, "pasid: %u", &pasid);
		}
		if (fi != NULL) {
			fclose(fi);
		}
		const int s = socket(AF_UNIX, SOCK_SEQPACKET, 0);
		struct sockaddr_un a;
		memset(&a, 0, sizeof(a));
		a.sun_family = AF_UNIX;
		snprintf(a.sun_path, sizeof(a.sun_path), "%s", argv[3]);
		if (s < 0 || connect(s, (struct sockaddr*)&a, sizeof(a)) != 0) {
			return Fail("connect to the helper", -errno);
		}
		for (int k = 0; k < 2; k++) {
			struct {
				uint32_t op, size;
				uint64_t va;
			} q = {3u | (pasid << 8), 0x840, rep_va};
			struct {
				int32_t  status;
				uint32_t value;
			} rr = {0, 0};
			if (send(s, &q, sizeof(q), 0) != sizeof(q) || recv(s, &rr, sizeof(rr), 0) != sizeof(rr)) {
				return Fail("helper call", -errno);
			}
			printf("helper report %d (pasid %u): status %d (%s), VMID %u\n", k, pasid, rr.status,
			       rr.status != 0 ? strerror(-rr.status) : "ok", rr.value);
			Dump(rep, 0x840);
			memset((void*)rep, 0xff, kRep);
		}
		close(s);
		return 0;
	}

	FILE* f = fopen("/tmp/lodhw.va", "w");
	if (f != NULL) {
		fprintf(f, "0x%llx\n", (unsigned long long)rep_va);
		fclose(f);
	}
	printf("READY report buffer at GPU VA 0x%llx (in /tmp/lodhw.va); waiting %u s for the root half\n",
	       (unsigned long long)rep_va, wait_s);
	fflush(stdout);

	uint32_t seen = 0;
	for (unsigned t = 0; t < wait_s * 10; t++) {
		usleep(100000);
		uint32_t changed = 0;
		for (uint32_t i = 0; i < kRep / 4; i++) {
			changed += rep[i] != 0xffffffffu;
		}
		if (changed != seen) {
			printf("t=%.1f s: the buffer changed\n", t / 10.0);
			Dump(rep, 0x1040);
			fflush(stdout);
			seen = changed;
		}
	}
	printf("%s\n", seen != 0 ? "DONE the reporter wrote the buffer" : "DONE nothing written");
	return 0;
}
