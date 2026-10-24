// LD_PRELOAD shim: where threads block. Times pthread_cond_wait/timedwait, pthread_mutex_lock,
// sem_wait, nanosleep/clock_nanosleep and sched_yield per thread and per call site (return
// address), and every 5 s prints the threads with the most blocked time with their top sites.
// Build: cc -O2 -shared -fPIC -o wait-shim.so wait-shim.c -ldl -lpthread
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>

#define MAX_T 4096
#define SITES 64
struct Site { void* ra; int kind; uint64_t ns, n; };
struct Slot { int tid; uint64_t total; struct Site s[SITES]; };
static struct Slot g_slots[MAX_T];
static uint64_t g_last;
static pthread_mutex_t g_report = PTHREAD_MUTEX_INITIALIZER;
static const char* kKinds[] = {"cond_wait", "cond_timedwait", "mutex_lock", "sem_wait", "nanosleep", "clock_nanosleep", "sched_yield"};
static __thread int t_in;

static uint64_t now_ns(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static void report(uint64_t t) {
	if (g_last == 0) { g_last = t; return; }
	if (t - g_last < 5000000000ull || pthread_mutex_trylock(&g_report) != 0) return;
	uint64_t span = t - g_last;
	g_last = t;
	for (int i = 0; i < MAX_T; i++) {
		struct Slot* sl = &g_slots[i];
		if (sl->total * 100 < span * 5) { memset(sl->s, 0, sizeof(sl->s)); sl->total = 0; continue; }
		char comm[32] = "?", path[64];
		snprintf(path, sizeof(path), "/proc/self/task/%d/comm", sl->tid);
		FILE* f = fopen(path, "r");
		if (f) { if (fgets(comm, sizeof(comm), f)) comm[strcspn(comm, "\n")] = 0; fclose(f); }
		fprintf(stderr, "wait: tid %d %s blocked %.1f%%:", sl->tid, comm, 100.0 * sl->total / span);
		for (int k = 0; k < 4; k++) {
			int best = -1;
			for (int j = 0; j < SITES; j++) if (sl->s[j].ns && (best < 0 || sl->s[j].ns > sl->s[best].ns)) best = j;
			if (best < 0) break;
			Dl_info di; const char* sym = "?"; uintptr_t off = 0;
			if (dladdr(sl->s[best].ra, &di) && di.dli_sname) { sym = di.dli_sname; off = (uintptr_t)sl->s[best].ra - (uintptr_t)di.dli_saddr; }
			fprintf(stderr, " | %s %.1f%% (%.0f/s) at %p %s+0x%lx", kKinds[sl->s[best].kind], 100.0 * sl->s[best].ns / span,
			        sl->s[best].n * 1e9 / span, sl->s[best].ra, sym, (unsigned long)off);
			sl->s[best].ns = 0;
		}
		fprintf(stderr, "\n");
		memset(sl->s, 0, sizeof(sl->s)); sl->total = 0;
	}
	pthread_mutex_unlock(&g_report);
}
static void account(void* ra, int kind, uint64_t t0) {
	uint64_t t1 = now_ns();
	static __thread struct Slot* sl;
	if (!sl) { int tid = (int)syscall(SYS_gettid); sl = &g_slots[tid % MAX_T]; sl->tid = tid; }
	uint64_t d = t1 - t0;
	sl->total += d;
	int h = (int)(((uintptr_t)ra >> 2) % SITES);
	for (int p = 0; p < SITES; p++) {
		struct Site* s = &sl->s[(h + p) % SITES];
		if (s->ra == ra && s->kind == kind) { s->ns += d; s->n++; break; }
		if (s->ns == 0) { s->ra = ra; s->kind = kind; s->ns = d; s->n = 1; break; }
	}
	report(t1);
}
#define REAL(ret, name, ...) static ret (*real)(__VA_ARGS__); if (!real) real = dlsym(RTLD_NEXT, #name)
#define TIMED(kind, call) do { if (t_in) return call; t_in = 1; uint64_t t0 = now_ns(); __typeof__(call) r = call; account(__builtin_return_address(0), kind, t0); t_in = 0; return r; } while (0)

int pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m) { REAL(int, pthread_cond_wait, pthread_cond_t*, pthread_mutex_t*); TIMED(0, real(c, m)); }
int pthread_cond_timedwait(pthread_cond_t* c, pthread_mutex_t* m, const struct timespec* a) { REAL(int, pthread_cond_timedwait, pthread_cond_t*, pthread_mutex_t*, const struct timespec*); TIMED(1, real(c, m, a)); }
int pthread_mutex_lock(pthread_mutex_t* m) {
	REAL(int, pthread_mutex_lock, pthread_mutex_t*);
	static int (*trylock)(pthread_mutex_t*);
	if (!trylock) trylock = dlsym(RTLD_NEXT, "pthread_mutex_trylock");
	if (trylock(m) == 0) return 0;
	TIMED(2, real(m));
}
int sem_wait(sem_t* s) { REAL(int, sem_wait, sem_t*); TIMED(3, real(s)); }
int nanosleep(const struct timespec* a, struct timespec* b) { REAL(int, nanosleep, const struct timespec*, struct timespec*); TIMED(4, real(a, b)); }
int clock_nanosleep(clockid_t c, int f, const struct timespec* a, struct timespec* b) { REAL(int, clock_nanosleep, clockid_t, int, const struct timespec*, struct timespec*); TIMED(5, real(c, f, a, b)); }
int sched_yield(void) { REAL(int, sched_yield, void); TIMED(6, real()); }
