// End-of-pipe interrupt probe for the BC-250: does a RELEASE_MEM with INT_SEL 2 in a render-node
// submission (gfx ring, amdgpu compute ring) wake the KFD event its context id names (bit 27 set),
// and does one with a title's context id reach the EOP log? Needs the patched amdgpu
// (bc250_gfx_eop_events=1, tools/bc250-amdgpu): stock KFD drops EOP interrupts of non-KFD VMIDs.
// This is how Kyty takes a title's EOP interrupts on the GPU, as the PS5 kernel does.
//   1. gfx ring: the event wakes, and how long after the label lands (busy-polled in a second pass);
//   2. INT_SEL 0 with the same context id: no wake (a 50 ms wait times out);
//   3. context id of event A, wait on event B: B does not wake, A does;
//   4. amdgpu compute ring 0: the event wakes;
//   5. one IB with 32 interrupting releases: the waits that follow see at least one wake, none hang;
//   6. EOP log (AMDKFD_IOC_BC250_EOP_LOG): a title's context ids from the gfx ring and compute ring 0
//      land in the signal page's log in order, with their ring; a KFD event's id still wakes it alone;
//   7. with bc250_gfx_eop_events=2: amdgpu's own fences stay out of the log, context id 0 from inside an
//      IB lands there with the IB's VMID.
// Build: cc -O2 -o eop_irq_probe eop_irq_probe.c. Needs /dev/kfd (no root).
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
static uint64_t event_page_offset;

// A context id that names KFD event `id` (bit 27); a title's ids keep to 27 bits.
static uint32_t EvCtx(uint32_t id) { return id | (1u << 27); }
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

// RELEASE_MEM, bottom of pipe with GL2 written back: DATA_SEL 1 of `value` to the label (DATA_SEL
// 0 and no address with INT_SEL 1, the interrupt alone), INT_SEL `int_sel`, context id `ctx`.
// Returns the next dword index.
static uint32_t Release(uint32_t* d, uint32_t i, int compute, uint32_t value, uint32_t int_sel, uint32_t ctx) {
	const int alone = int_sel == 1;
	d[i++] = Pkt3(0x49, 7, compute);
	d[i++] = (1u << 21) | (5u << 8) | 0x14u; // GL2_WB, EVENT_INDEX 5, CACHE_FLUSH_AND_INV_TS_EVENT
	d[i++] = (alone ? 0u : 1u << 29) | (int_sel << 24);
	d[i++] = alone ? 0u : (uint32_t)label.va;
	d[i++] = alone ? 0u : (uint32_t)(label.va >> 32);
	d[i++] = alone ? 0u : value;
	d[i++] = 0;
	d[i++] = ctx;
	return i;
}

static uint32_t Pad(uint32_t n) {
	while (n & 7u) {
		((uint32_t*)ib.cpu)[n++] = 0xffff1000u;
	}
	return n;
}

static uint32_t CreateEvent(void) {
	struct kfd_ioctl_create_event_args a;
	memset(&a, 0, sizeof(a));
	a.event_type = KFD_IOC_EVENT_SIGNAL;
	a.auto_reset = 1;
	int r        = Ioctl(kfd, AMDKFD_IOC_CREATE_EVENT, &a);
	if (r) {
		fprintf(stderr, "CREATE_EVENT: %s\n", strerror(-r));
		exit(3);
	}
	if (a.event_page_offset != 0) {
		event_page_offset = a.event_page_offset;
	}
	return a.event_id;
}

// 0: the event woke; 1: timeout.
static int WaitEvent(uint32_t id, uint32_t timeout_ms) {
	struct kfd_event_data          ev;
	struct kfd_ioctl_wait_events_args w;
	memset(&ev, 0, sizeof(ev));
	memset(&w, 0, sizeof(w));
	ev.event_id   = id;
	w.events_ptr  = (uintptr_t)&ev;
	w.num_events  = 1;
	w.timeout     = timeout_ms;
	int r         = Ioctl(kfd, AMDKFD_IOC_WAIT_EVENTS, &w);
	if (r) {
		fprintf(stderr, "WAIT_EVENTS: %s\n", strerror(-r));
		exit(3);
	}
	return w.wait_result == KFD_IOC_WAIT_RESULT_COMPLETE ? 0 : 1;
}

static int Cmp(const void* a, const void* b) {
	const double x = *(const double*)a, y = *(const double*)b;
	return x < y ? -1 : x > y;
}

// Single-release IBs on (ip): median and max of submit-to-wake (event) and submit-to-label (poll).
static int Latency(const char* name, uint32_t ip, uint32_t ev, int rounds) {
	const int compute = ip == AMDGPU_HW_IP_COMPUTE;
	double    irq[256], poll[256];
	int       lost = 0;
	volatile uint32_t* l = (volatile uint32_t*)label.cpu;
	for (int pass = 0; pass < 2; pass++) {
		for (int k = 0; k < rounds; k++) {
			const uint32_t value = 0x1000u + (uint32_t)(pass * rounds + k);
			uint32_t       n     = Release((uint32_t*)ib.cpu, 0, compute, value, pass == 0 ? 2u : 0u, EvCtx(ev));
			while (n & 7u) {
				((uint32_t*)ib.cpu)[n++] = 0xffff1000u;
			}
			__sync_synchronize();
			uint64_t     seq = 0;
			const double t0  = Now();
			if (Submit(ip, n, &seq)) {
				fprintf(stderr, "%s: CS failed\n", name);
				return -1;
			}
			if (pass == 0) {
				if (WaitEvent(ev, 500)) {
					lost++;
					irq[k] = 0.5;
					if (lost == 3 && k == 2) {
						WaitCs(ip, seq);
						printf("%s: the first 3 releases did not wake the event; giving up\n", name);
						return lost;
					}
				} else {
					irq[k] = Now() - t0;
				}
			} else {
				while (*l != value && Now() - t0 < 0.5) {
				}
				poll[k] = Now() - t0;
			}
			WaitCs(ip, seq);
		}
	}
	qsort(irq, rounds, sizeof(double), Cmp);
	qsort(poll, rounds, sizeof(double), Cmp);
	printf("%s: %d releases, %d without a wake; submit to wake median %.1f us max %.1f us; submit to label median %.1f us max %.1f us\n",
	       name, rounds, lost, irq[rounds / 2] * 1e6, irq[rounds - 1] * 1e6, poll[rounds / 2] * 1e6, poll[rounds - 1] * 1e6);
	return lost;
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
	FILE* p = fopen("/sys/module/amdgpu/parameters/bc250_gfx_eop_events", "r");
	int   on = -1;
	if (p) {
		if (fscanf(p, "%d", &on) != 1) {
			on = -1;
		}
		fclose(p);
	}
	printf("bc250_gfx_eop_events: %d%s\n", on, on < 0 ? " (stock amdgpu: no wakes expected)" : "");
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
	const uint32_t a = CreateEvent(), b = CreateEvent();
	printf("KFD signal events %u and %u\n", a, b);
	int bad = 0;

	// 1. gfx ring.
	bad += Latency("gfx", AMDGPU_HW_IP_GFX, a, 200) != 0;

	// 2. INT_SEL 0: no wake.
	{
		uint32_t n = Release((uint32_t*)ib.cpu, 0, 0, 7, 0, EvCtx(a));
		while (n & 7u) {
			((uint32_t*)ib.cpu)[n++] = 0xffff1000u;
		}
		uint64_t seq = 0;
		Submit(AMDGPU_HW_IP_GFX, n, &seq);
		WaitCs(AMDGPU_HW_IP_GFX, seq);
		const int woke = WaitEvent(a, 50) == 0;
		printf("INT_SEL 0 with event %u's context id: %s\n", a, woke ? "WOKE (unexpected)" : "no wake (right)");
		bad += woke;
	}

	// 3. Event A's context id, wait on B, then on A.
	{
		uint32_t n = Release((uint32_t*)ib.cpu, 0, 0, 8, 2, EvCtx(a));
		while (n & 7u) {
			((uint32_t*)ib.cpu)[n++] = 0xffff1000u;
		}
		uint64_t seq = 0;
		Submit(AMDGPU_HW_IP_GFX, n, &seq);
		WaitCs(AMDGPU_HW_IP_GFX, seq);
		const int woke_b = WaitEvent(b, 50) == 0;
		const int woke_a = WaitEvent(a, 50) == 0;
		printf("context id %u: event %u %s, event %u %s\n", a, b, woke_b ? "WOKE (unexpected)" : "stays (right)", a,
		       woke_a ? "woke (right)" : "DID NOT WAKE");
		bad += woke_b || !woke_a;
	}

	// 4. amdgpu compute ring 0.
	bad += Latency("compute", AMDGPU_HW_IP_COMPUTE, b, 100) != 0;

	// 5. 32 interrupting releases in one IB.
	{
		uint32_t n = 0;
		for (uint32_t k = 0; k < 32; k++) {
			n = Release((uint32_t*)ib.cpu, n, 0, 100 + k, 2, EvCtx(a));
		}
		while (n & 7u) {
			((uint32_t*)ib.cpu)[n++] = 0xffff1000u;
		}
		uint64_t seq = 0;
		Submit(AMDGPU_HW_IP_GFX, n, &seq);
		const int done  = WaitCs(AMDGPU_HW_IP_GFX, seq);
		int       wakes = 0;
		while (wakes < 64 && WaitEvent(a, 20) == 0) {
			wakes++;
		}
		printf("32 interrupts in one IB: IB %s, label %u, %d wakes after it\n", done == 0 ? "done" : "NOT DONE", *(volatile uint32_t*)label.cpu,
		       wakes);
		bad += done != 0 || wakes == 0;
	}

	// 6. The EOP log.
	{
		const uint32_t log = CreateEvent();
		volatile uint64_t* page =
		    event_page_offset ? mmap(NULL, 4096 * 8, PROT_READ | PROT_WRITE, MAP_SHARED, kfd, (off_t)event_page_offset) : MAP_FAILED;
		struct {
			uint32_t event_id, pad;
		} reg     = {log, 0};
		const int rr = page == MAP_FAILED ? -1 : Ioctl(kfd, _IOW('K', 0x80, reg), &reg);
		printf("EOP log: event %u, signal page %s, register %s\n", log, page == MAP_FAILED ? "NOT MAPPED" : "mapped",
		       rr == 0 ? "ok" : strerror(-rr));
		if (rr != 0) {
			bad++;
		} else {
			volatile uint64_t* head = page + 2048;
			volatile uint64_t* tail = page + 2049;
			volatile uint64_t* drop = page + 2050;
			volatile uint64_t* ring = page + 3072;
			static const uint32_t kIds[] = {0xabcd, 1, 2, 3, 0x5555, 0x7ffffff};
			uint32_t              n      = 0;
			for (uint32_t k = 0; k < 6; k++) {
				n = Release((uint32_t*)ib.cpu, n, 0, 200 + k, k == 4 ? 2u : 1u, kIds[k]);
			}
			n = Release((uint32_t*)ib.cpu, n, 0, 300, 2, EvCtx(a)); // Kyty's: the event alone
			uint64_t seq = 0;
			Submit(AMDGPU_HW_IP_GFX, Pad(n), &seq);
			WaitCs(AMDGPU_HW_IP_GFX, seq);
			const int woke = WaitEvent(log, 100) == 0;
			const int woke_a = WaitEvent(a, 50) == 0;
			n = Release((uint32_t*)ib.cpu, 0, 1, 400, 1, 0xa);
			Submit(AMDGPU_HW_IP_COMPUTE, Pad(n), &seq);
			WaitCs(AMDGPU_HW_IP_COMPUTE, seq);
			// The log's event may still be set from the gfx entries: wait for the compute one itself.
			int woke_c = 0;
			for (int k = 0; k < 10 && *head - *tail < 7; k++) {
				woke_c |= WaitEvent(log, 20) == 0;
			}
			woke_c |= *head - *tail == 7;
			const uint64_t h = *head, t = *tail;
			printf("log: woke %d (gfx) %d (compute), event %u woke %d; head %llu tail %llu dropped %llu\n", woke, woke_c, a, woke_a,
			       (unsigned long long)h, (unsigned long long)t, (unsigned long long)*drop);
			int order_ok = h - t == 7;
			for (uint64_t i = t; i < h; i++) {
				const uint64_t e = ring[i % 1024];
				printf("  ctx 0x%x vmid %u ring id 0x%02x (me %u pipe %u queue %u)\n", (uint32_t)e, (uint32_t)(e >> 40) & 0xf,
				       (uint32_t)(e >> 32) & 0xff, (uint32_t)(e >> 34) & 3, (uint32_t)(e >> 32) & 3, (uint32_t)(e >> 36) & 7);
				const uint32_t want = i - t < 6 ? kIds[i - t] : 0xa;
				order_ok &= (uint32_t)e == want;
			}
			*tail = h;
			printf("log %s\n", order_ok ? "in order, all there (right)" : "WRONG");
			bad += !woke || !woke_c || !woke_a || !order_ok;

			// 7. bc250_gfx_eop_events=2: context id 0. amdgpu's own fences (ring level, VMID 0) stay out of the
			// log; a release with id 0 inside an IB lands there, with the IB's VMID.
			FILE* pf    = fopen("/sys/module/amdgpu/parameters/bc250_gfx_eop_events", "r");
			int   param = 0;
			if (pf != NULL) {
				if (fscanf(pf, "%d", &param) != 1) {
					param = 0;
				}
				fclose(pf);
			}
			if (param >= 2) {
				for (int k = 0; k < 8; k++) { // plain submissions: only amdgpu's fences interrupt
					n = Release((uint32_t*)ib.cpu, 0, k & 1, 500 + k, 0, 0);
					Submit((k & 1) ? AMDGPU_HW_IP_COMPUTE : AMDGPU_HW_IP_GFX, Pad(n), &seq);
					WaitCs((k & 1) ? AMDGPU_HW_IP_COMPUTE : AMDGPU_HW_IP_GFX, seq);
				}
				usleep(50000);
				const uint64_t fences = *head - *tail;
				n = Release((uint32_t*)ib.cpu, 0, 0, 600, 1, 0);
				n = Release((uint32_t*)ib.cpu, n, 0, 601, 2, 0);
				Submit(AMDGPU_HW_IP_GFX, Pad(n), &seq);
				WaitCs(AMDGPU_HW_IP_GFX, seq);
				n = Release((uint32_t*)ib.cpu, 0, 1, 602, 2, 0);
				Submit(AMDGPU_HW_IP_COMPUTE, Pad(n), &seq);
				WaitCs(AMDGPU_HW_IP_COMPUTE, seq);
				for (int k = 0; k < 10 && *head - *tail < fences + 3; k++) {
					WaitEvent(log, 20);
				}
				const uint64_t h7 = *head, t7 = *tail;
				int            ok7 = fences == 0 && h7 - t7 == 3;
				printf("context id 0: %llu entries from 8 plain submissions (want 0), %llu from 3 id-0 releases in IBs (want 3)\n",
				       (unsigned long long)fences, (unsigned long long)(h7 - t7 - fences));
				for (uint64_t i = t7; i < h7; i++) {
					const uint64_t e = ring[i % 1024];
					printf("  ctx 0x%x vmid %u ring id 0x%02x\n", (uint32_t)e, (uint32_t)(e >> 40) & 0xf, (uint32_t)(e >> 32) & 0xff);
					ok7 &= (uint32_t)e == 0 && ((e >> 40) & 0xf) != 0;
				}
				*tail = h7;
				printf("context id 0 %s\n", ok7 ? "as expected" : "WRONG");
				bad += !ok7;
			} else {
				printf("context id 0: bc250_gfx_eop_events=%d, skipped (needs 2)\n", param);
			}
		}
	}
	printf("%s\n", bad ? "FAILED" : "all as expected");
	return bad ? 4 : 0;
}
