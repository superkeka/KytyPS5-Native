// LD_PRELOAD shim: time spent per thread in a few libc syscall wrappers, printed to stderr every
// 5 s for the threads that spent the most. Build: cc -O2 -shared -fPIC -o syscall-shim.so syscall-shim.c -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>

enum { C_IOCTL, C_MPROTECT, C_MMAP, C_MUNMAP, C_MADVISE, C_YIELD, C_FSTAT, C_N };
static const char* kNames[C_N] = {"ioctl", "mprotect", "mmap", "munmap", "madvise", "sched_yield", "fstat"};

#define MAX_T 4096
struct Slot { int tid; uint64_t ns[C_N]; uint64_t n[C_N]; uint64_t ioctl_ns[256]; };
static struct Slot g_slots[MAX_T];
static uint64_t g_last;

static uint64_t now_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static struct Slot* slot(void) {
	static __thread struct Slot* s;
	if (s == NULL) {
		int tid = (int)syscall(SYS_gettid);
		s = &g_slots[tid % MAX_T];
		s->tid = tid;
	}
	return s;
}
static void report(uint64_t t) {
	if (g_last == 0) { g_last = t; return; }
	if (t - g_last < 5000000000ull) return;
	uint64_t span = t - g_last;
	g_last = t;
	for (int i = 0; i < MAX_T; i++) {
		struct Slot* s = &g_slots[i];
		uint64_t tot = 0;
		for (int c = 0; c < C_N; c++) tot += s->ns[c];
		if (tot * 100 < span * 3) { memset(s->ns, 0, sizeof(s->ns)); memset(s->n, 0, sizeof(s->n)); memset(s->ioctl_ns, 0, sizeof(s->ioctl_ns)); continue; }
		char comm[32] = "?";
		char path[64];
		snprintf(path, sizeof(path), "/proc/self/task/%d/comm", s->tid);
		FILE* f = fopen(path, "r");
		if (f) { if (fgets(comm, sizeof(comm), f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
		fprintf(stderr, "shim: tid %d %s %.1f%% in wrappers:", s->tid, comm, 100.0 * tot / span);
		for (int c = 0; c < C_N; c++)
			if (s->n[c]) fprintf(stderr, " %s %.1f%% (%.0f/s)", kNames[c], 100.0 * s->ns[c] / span, s->n[c] * 1e9 / span);
		int top = -1;
		for (int k = 0; k < 256; k++) if (top < 0 || s->ioctl_ns[k] > s->ioctl_ns[top]) top = k;
		if (top >= 0 && s->ioctl_ns[top]) fprintf(stderr, " [top ioctl nr 0x%02x %.1f%%]", top, 100.0 * s->ioctl_ns[top] / span);
		fprintf(stderr, "\n");
		memset(s->ns, 0, sizeof(s->ns)); memset(s->n, 0, sizeof(s->n)); memset(s->ioctl_ns, 0, sizeof(s->ioctl_ns));
	}
}
#define WRAP_BEGIN uint64_t t0 = now_ns();
#define WRAP_END(c) do { uint64_t t1 = now_ns(); struct Slot* s = slot(); s->ns[c] += t1 - t0; s->n[c]++; if ((c) == C_IOCTL) s->ioctl_ns[_IOC_NR(req) & 0xff] += t1 - t0; report(t1); } while (0)

int ioctl(int fd, unsigned long req, ...) {
	static int (*real)(int, unsigned long, void*);
	if (!real) real = dlsym(RTLD_NEXT, "ioctl");
	va_list ap; va_start(ap, req); void* arg = va_arg(ap, void*); va_end(ap);
	WRAP_BEGIN int r = real(fd, req, arg); WRAP_END(C_IOCTL); return r;
}
int mprotect(void* a, size_t l, int p) {
	static int (*real)(void*, size_t, int);
	if (!real) real = dlsym(RTLD_NEXT, "mprotect");
	unsigned long req = 0; (void)req;
	WRAP_BEGIN int r = real(a, l, p); WRAP_END(C_MPROTECT); return r;
}
void* mmap(void* a, size_t l, int p, int fl, int fd, off_t o) {
	static void* (*real)(void*, size_t, int, int, int, off_t);
	if (!real) real = dlsym(RTLD_NEXT, "mmap");
	unsigned long req = 0; (void)req;
	WRAP_BEGIN void* r = real(a, l, p, fl, fd, o); WRAP_END(C_MMAP); return r;
}
int munmap(void* a, size_t l) {
	static int (*real)(void*, size_t);
	if (!real) real = dlsym(RTLD_NEXT, "munmap");
	unsigned long req = 0; (void)req;
	WRAP_BEGIN int r = real(a, l); WRAP_END(C_MUNMAP); return r;
}
int madvise(void* a, size_t l, int adv) {
	static int (*real)(void*, size_t, int);
	if (!real) real = dlsym(RTLD_NEXT, "madvise");
	unsigned long req = 0; (void)req;
	WRAP_BEGIN int r = real(a, l, adv); WRAP_END(C_MADVISE); return r;
}
int sched_yield(void) {
	static int (*real)(void);
	if (!real) real = dlsym(RTLD_NEXT, "sched_yield");
	unsigned long req = 0; (void)req;
	WRAP_BEGIN int r = real(); WRAP_END(C_YIELD); return r;
}
int fstat(int fd, struct stat* st) {
	static int (*real)(int, struct stat*);
	if (!real) real = dlsym(RTLD_NEXT, "fstat");
	unsigned long req = 0; (void)req;
	WRAP_BEGIN int r = real(fd, st); WRAP_END(C_FSTAT); return r;
}
