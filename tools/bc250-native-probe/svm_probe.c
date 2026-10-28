// SVM probe for the BC-250: can a range of the process's own memory (anonymous or a memfd, as Kyty's
// guest memory is) be the GPU's at the same addresses through KFD's shared virtual memory
// (AMDKFD_IOC_SVM), the way PS5 memory is one memory for CPU and GPU, and do render-node submissions
// (gfx ring, amdgpu compute ring) on the VM KFD acquired see it? For each kind of memory and ring: the
// GPU copies a value the CPU wrote into a label and writes a value the CPU then reads; then the cost of
// an mprotect round trip on the range (Kyty write-protects guest pages to track writes) and whether the
// GPU still reaches the range after it. GPU accesses to an unmapped range fault without retry here, so
// a failure shows as a missing value, not a hang.
// Result on BC-250 (28.10.2026): both kinds of memory work from the gfx ring and the compute ring, the
// CPU and the GPU see each other's writes, an mprotect round trip on 2 MiB costs ~0.07 ms. But the
// GPU's write access follows the CPU's: a GPU write into a range the CPU write-protected faults (CPG,
// write) and stops the gfx ring (needs a reboot). Giving 1 GiB to the GPU takes ~1.5 s.
// Build: cc -O2 -o svm_probe svm_probe.c. Needs /dev/kfd (no root).
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

static int      kfd = -1, drm = -1;

static uint32_t gpu_id, ctx_id;

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
static uint32_t Pkt3(uint32_t op, uint32_t body, int compute) { return 0xc0000000u | ((body - 1) << 16) | (op << 8) | (compute ? 2u : 0u); }

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
static int Submit(uint32_t ip, uint32_t n, uint64_t* seq) {
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
	info.va_start = ib.va;
	info.ib_bytes = n * 4;
	info.ip_type  = ip;
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

static int WaitCs(uint32_t ip, uint64_t seq) {
	union drm_amdgpu_wait_cs w;
	memset(&w, 0, sizeof(w));
	w.in.handle  = seq;
	w.in.timeout = (uint64_t)((Now() + 2.0) * 1e9); // absolute CLOCK_MONOTONIC
	w.in.ip_type = ip;
	w.in.ctx_id  = ctx_id;
	int r        = Ioctl(drm, DRM_IOCTL_AMDGPU_WAIT_CS, &w);
	return r ? r : (w.out.status ? 1 : 0);
}


static uint32_t Pad(uint32_t n) {
	while (n & 7u) {
		((uint32_t*)ib.cpu)[n++] = 0xffff1000u;
	}
	return n;
}

// The range [addr, addr + size) to the GPU through SVM: system memory, GPU access, coherent, always mapped.
// Attribute sets from the fullest to GPU access alone: the first the kernel takes is used.
static int SvmSet(void* addr, uint64_t size, const struct kfd_ioctl_svm_attribute* attrs, uint32_t nattr) {
	struct {
		struct kfd_ioctl_svm_args      a;
		struct kfd_ioctl_svm_attribute attrs[4];
	} s;
	memset(&s, 0, sizeof(s));
	s.a.start_addr = (uintptr_t)addr;
	s.a.size       = size;
	s.a.op         = KFD_IOCTL_SVM_OP_SET_ATTR;
	s.a.nattr      = nattr;
	memcpy(s.attrs, attrs, nattr * sizeof(attrs[0]));
	// The ioctl's size covers the attributes too (the kernel checks it; libhsakmt does the same).
	return Ioctl(kfd, AMDKFD_IOC_SVM + ((unsigned long)(nattr * sizeof(attrs[0])) << _IOC_SIZESHIFT), &s);
}

static int SvmMap(void* addr, uint64_t size) {
	const uint32_t flags[] = {KFD_IOCTL_SVM_FLAG_HOST_ACCESS | KFD_IOCTL_SVM_FLAG_COHERENT | KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED,
	                          KFD_IOCTL_SVM_FLAG_HOST_ACCESS | KFD_IOCTL_SVM_FLAG_COHERENT, KFD_IOCTL_SVM_FLAG_HOST_ACCESS, 0};
	int r = -EINVAL;
	for (unsigned k = 0; k < sizeof(flags) / sizeof(flags[0]); k++) {
		struct kfd_ioctl_svm_attribute a[3] = {{KFD_IOCTL_SVM_ATTR_PREFERRED_LOC, KFD_IOCTL_SVM_LOCATION_SYSMEM},
		                                       {KFD_IOCTL_SVM_ATTR_ACCESS, gpu_id},
		                                       {KFD_IOCTL_SVM_ATTR_SET_FLAGS, flags[k]}};
		r = SvmSet(addr, size, a, flags[k] != 0 ? 3u : 2u);
		printf("  SVM set attributes (preferred sysmem, access, flags 0x%x): %s\n", flags[k], r ? strerror(-r) : "ok");
		if (r == 0) {
			return 0;
		}
	}
	struct kfd_ioctl_svm_attribute access = {KFD_IOCTL_SVM_ATTR_ACCESS, gpu_id};
	r = SvmSet(addr, size, &access, 1);
	printf("  SVM set attributes (access only): %s\n", r ? strerror(-r) : "ok");
	return r;
}

// On `ip`: COPY_DATA of buf[16] (the CPU's value) to the label, then WRITE_DATA of `value` to buf[0].
// Returns 1 when both landed.
static int GpuRoundTrip(uint32_t ip, volatile uint32_t* buf, uint32_t value, const char* what) {
	const int        compute = ip == AMDGPU_HW_IP_COMPUTE;
	volatile uint32_t* l     = (volatile uint32_t*)label.cpu;
	const uint32_t   cpu_v   = value ^ 0x00ff00ffu;
	buf[16]                  = cpu_v;
	buf[0]                   = 0;
	l[0]                     = 0;
	__sync_synchronize();
	uint32_t* d = (uint32_t*)ib.cpu;
	uint32_t  n = 0;
	const uint64_t src = (uintptr_t)(buf + 16), dst = (uintptr_t)buf;
	d[n++] = Pkt3(0x40, 5, compute); // COPY_DATA memory -> memory, write confirmed
	d[n++] = 1u | (5u << 8) | (1u << 20);
	d[n++] = (uint32_t)src;
	d[n++] = (uint32_t)(src >> 32);
	d[n++] = (uint32_t)label.va;
	d[n++] = (uint32_t)(label.va >> 32);
	d[n++] = Pkt3(0x37, 4, compute); // WRITE_DATA to memory, write confirmed
	d[n++] = (5u << 8) | (1u << 20);
	d[n++] = (uint32_t)dst;
	d[n++] = (uint32_t)(dst >> 32);
	d[n++] = value;
	uint64_t seq = 0;
	int      r   = Submit(ip, Pad(n), &seq);
	int      w   = r == 0 ? WaitCs(ip, seq) : r;
	usleep(1000);
	const int read_ok = l[0] == cpu_v, write_ok = buf[0] == value;
	printf("  %-30s %s: submit %s, GPU read the CPU's value %s, CPU sees the GPU's write %s\n", what, compute ? "compute" : "gfx",
	       r ? strerror(-r) : (w == 1 ? "TIMEOUT" : (w ? strerror(-w) : "done")), read_ok ? "yes" : "NO", write_ok ? "yes" : "NO");
	return read_ok && write_ok;
}

int main(void) {
	setvbuf(stdout, NULL, _IOLBF, 0);
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
	if (kfd < 0 || drm < 0 || gpu_id == 0) {
		fprintf(stderr, "no KFD or render node\n");
		return 1;
	}
	struct kfd_ioctl_acquire_vm_args acq = {.drm_fd = (uint32_t)drm, .gpu_id = gpu_id};
	int                              r   = Ioctl(kfd, AMDKFD_IOC_ACQUIRE_VM, &acq);
	if (r) {
		fprintf(stderr, "ACQUIRE_VM: %s\n", strerror(-r));
		return 1;
	}
	union drm_amdgpu_ctx c;
	memset(&c, 0, sizeof(c));
	c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
	if ((r = Ioctl(drm, DRM_IOCTL_AMDGPU_CTX, &c))) {
		fprintf(stderr, "CTX: %s\n", strerror(-r));
		return 1;
	}
	ctx_id = c.out.alloc.ctx_id;
	if ((r = BoAt(&ib, 0x510000000000ull, 65536)) || (r = BoAt(&label, 0x510000100000ull, 4096))) {
		fprintf(stderr, "buffers: %s\n", strerror(-r));
		return 2;
	}
	const uint64_t size = 2u << 20;
	int            bad  = 0;
	for (int kind = 0; kind < 2; kind++) {
		void* p;
		if (kind == 0) {
			p = mmap(NULL, size * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		} else {
			int fd = memfd_create("svm-probe", MFD_CLOEXEC);
			if (fd < 0 || ftruncate(fd, (off_t)(size * 2)) != 0) {
				perror("memfd");
				return 3;
			}
			p = mmap(NULL, size * 2, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		}
		if (p == MAP_FAILED) {
			perror("mmap");
			return 3;
		}
		volatile uint32_t* buf = (volatile uint32_t*)(((uintptr_t)p + size - 1) & ~(uintptr_t)(size - 1));
		memset((void*)buf, 0x11, size);
		const char* name = kind == 0 ? "anonymous" : "memfd (shared)";
		double      t0   = Now();
		r                = SvmMap((void*)buf, size);
		printf("%s at %p: SVM map %s (%.2f ms)\n", name, (void*)buf, r ? strerror(-r) : "ok", (Now() - t0) * 1e3);
		if (r) {
			bad++;
			continue;
		}
		bad += !GpuRoundTrip(AMDGPU_HW_IP_GFX, buf, 0x5a5a0001u + kind, "after the SVM map");
		bad += !GpuRoundTrip(AMDGPU_HW_IP_COMPUTE, buf, 0x5a5a0011u + kind, "after the SVM map");
		// Kyty write-protects guest pages to see the CPU's writes: an mprotect round trip on the range.
		const int rounds = 20;
		t0               = Now();
		for (int i = 0; i < rounds; i++) {
			mprotect((void*)buf, size, PROT_READ);
			mprotect((void*)buf, size, PROT_READ | PROT_WRITE);
		}
		printf("  mprotect round trip on 2 MiB: %.3f ms\n", (Now() - t0) * 1e3 / rounds);
		bad += !GpuRoundTrip(AMDGPU_HW_IP_GFX, buf, 0x5a5a0021u + kind, "after mprotect");
		// The GPU writes while the CPU side is write-protected (Kyty's write tracking): the PS5's GPU
		// writes regardless of the CPU's protection.
		mprotect((void*)buf, size, PROT_READ);
		{
			volatile uint32_t* l = (volatile uint32_t*)label.cpu;
			uint32_t*          d = (uint32_t*)ib.cpu;
			uint32_t           n = 0;
			const uint64_t     dst = (uintptr_t)(buf + 32);
			l[1]                   = 0;
			d[n++]                 = Pkt3(0x37, 4, 0);
			d[n++]                 = (5u << 8) | (1u << 20);
			d[n++]                 = (uint32_t)dst;
			d[n++]                 = (uint32_t)(dst >> 32);
			d[n++]                 = 0x77777777u;
			d[n++]                 = Pkt3(0x37, 4, 0); // a marker in the label: the IB went on
			d[n++]                 = (5u << 8) | (1u << 20);
			d[n++]                 = (uint32_t)(label.va + 4);
			d[n++]                 = (uint32_t)((label.va + 4) >> 32);
			d[n++]                 = 0x88888888u;
			uint64_t seq           = 0;
			int      rs            = Submit(AMDGPU_HW_IP_GFX, Pad(n), &seq);
			int      w             = rs == 0 ? WaitCs(AMDGPU_HW_IP_GFX, seq) : rs;
			usleep(1000);
			printf("  GPU write into the write-protected range: submit %s, landed %s, IB went on %s\n",
			       w == 1 ? "TIMEOUT" : (w ? strerror(-w) : "done"), buf[32] == 0x77777777u ? "yes" : "NO",
			       l[1] == 0x88888888u ? "yes" : "NO");
		}
		mprotect((void*)buf, size, PROT_READ | PROT_WRITE);
		// A CPU write to every page, then the GPU again.
		for (uint64_t o = 0; o < size / 4; o += 1024) {
			buf[o] = (uint32_t)o;
		}
		bad += !GpuRoundTrip(AMDGPU_HW_IP_GFX, buf, 0x5a5a0031u + kind, "after CPU writes");
	}
	// A large range: what giving 1 GiB to the GPU costs (pages touched by the CPU first, as a title's).
	{
		const uint64_t big = 1ull << 30;
		void*          p   = mmap(NULL, big, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (p != MAP_FAILED) {
			memset(p, 1, big);
			double t0 = Now();
			r         = SvmMap(p, big);
			printf("1 GiB anonymous: SVM map %s in %.1f ms\n", r ? strerror(-r) : "ok", (Now() - t0) * 1e3);
			if (r == 0) {
				bad += !GpuRoundTrip(AMDGPU_HW_IP_GFX, (volatile uint32_t*)p + (big / 8), 0x5a5a0041u, "1 GiB range, middle");
			}
			munmap(p, big);
		}
	}
	printf("%s\n", bad ? "FAILED" : "all as expected");
	return bad ? 4 : 0;
}
