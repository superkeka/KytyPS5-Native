// LD_PRELOAD heap-growth sampler: tracks ~1/64 of allocations (chosen by address hash) with their
// backtrace, forgets them when freed, and every 10 s prints the stacks holding the most sampled
// live bytes (multiply by 64 for an estimate) plus the executable's load address for addr2line.
// Build: cc -O2 -shared -fPIC -o leak-shim.so leak-shim.c -lpthread
#define _GNU_SOURCE
#include <execinfo.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

extern void* __libc_malloc(size_t);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
extern void  __libc_free(void*);
extern void* __libc_memalign(size_t, size_t);

#define DEPTH 10
#define TABLE (1u << 21)
#define STACKS 8192
struct Stack { uint64_t hash; void* pc[DEPTH]; int n; int64_t bytes; int64_t count; };
struct Entry { void* p; uint32_t size; uint32_t stack; };
static struct Entry g_table[TABLE];
static struct Stack g_stacks[STACKS];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread int t_busy;
static time_t g_last;

static int sampled(void* p) { uint64_t x = (uint64_t)(uintptr_t)p >> 4; x *= 0x9E3779B97F4A7C15ull; return (x >> 58) == 0; }
static uint32_t slot_of(void* p) { uint64_t x = (uint64_t)(uintptr_t)p * 0x9E3779B97F4A7C15ull; return (uint32_t)(x >> 43) & (TABLE - 1); }

static void dump(void) {
	FILE* m = fopen("/proc/self/maps", "r");
	char line[512];
	if (m) { while (fgets(line, sizeof(line), m)) if (strstr(line, "kyty_emulator") && strstr(line, "r-xp")) { fprintf(stderr, "leak: exe %s", line); break; } fclose(m); }
	for (int k = 0; k < 8; k++) {
		int best = -1;
		for (int i = 0; i < STACKS; i++) if (g_stacks[i].bytes > 0 && (best < 0 || g_stacks[i].bytes > g_stacks[best].bytes)) best = i;
		if (best < 0) break;
		struct Stack* s = &g_stacks[best];
		fprintf(stderr, "leak: #%d ~%lld MiB live in ~%lld allocs:", k, (long long)(s->bytes * 64 >> 20), (long long)(s->count * 64));
		for (int j = 0; j < s->n; j++) fprintf(stderr, " %p", s->pc[j]);
		fprintf(stderr, "\n");
		s->bytes = -s->bytes; // hide for this round
	}
	for (int i = 0; i < STACKS; i++) if (g_stacks[i].bytes < 0) g_stacks[i].bytes = -g_stacks[i].bytes;
}

static void track(void* p, size_t size) {
	if (!p || !sampled(p) || t_busy) return;
	t_busy = 1;
	void* pc[DEPTH + 2];
	int n = backtrace(pc, DEPTH + 2) - 2;
	if (n < 0) n = 0;
	uint64_t h = 1469598103934665603ull;
	for (int i = 0; i < n; i++) { h ^= (uint64_t)(uintptr_t)pc[i + 2]; h *= 1099511628211ull; }
	pthread_mutex_lock(&g_lock);
	uint32_t si = (uint32_t)(h % STACKS);
	for (int probe = 0; probe < STACKS; probe++, si = (si + 1) % STACKS) {
		if (g_stacks[si].hash == h) break;
		if (g_stacks[si].hash == 0) { g_stacks[si].hash = h; g_stacks[si].n = n; memcpy(g_stacks[si].pc, pc + 2, n * sizeof(void*)); break; }
	}
	g_stacks[si].bytes += size; g_stacks[si].count++;
	// Count the allocation only when it fits the table, so a full probe chain cannot fake a leak; a
	// tombstone slot is reused.
	uint32_t e = slot_of(p);
	int stored = 0;
	for (int probe = 0; probe < 64; probe++, e = (e + 1) & (TABLE - 1)) {
		void* q = g_table[e].p;
		if (q == NULL || q == (void*)1 || q == p) { g_table[e].p = p; g_table[e].size = (uint32_t)size; g_table[e].stack = si; stored = 1; break; }
	}
	if (!stored) { g_stacks[si].bytes -= size; g_stacks[si].count--; }
	time_t now = time(NULL);
	if (g_last == 0) g_last = now;
	if (now - g_last >= 10) { g_last = now; dump(); }
	pthread_mutex_unlock(&g_lock);
	t_busy = 0;
}
static void untrack(void* p) {
	if (!p || !sampled(p)) return;
	pthread_mutex_lock(&g_lock);
	uint32_t e = slot_of(p);
	for (int probe = 0; probe < 64; probe++, e = (e + 1) & (TABLE - 1)) {
		if (g_table[e].p == p) {
			g_stacks[g_table[e].stack].bytes -= g_table[e].size; g_stacks[g_table[e].stack].count--;
			g_table[e].p = (void*)1; // tombstone keeps probe chains intact
			break;
		}
		if (g_table[e].p == NULL) break;
	}
	pthread_mutex_unlock(&g_lock);
}

void* malloc(size_t n) { void* p = __libc_malloc(n); track(p, n); return p; }
void* calloc(size_t a, size_t b) { void* p = __libc_calloc(a, b); track(p, a * b); return p; }
void* realloc(void* q, size_t n) { untrack(q); void* p = __libc_realloc(q, n); track(p, n); return p; }
void free(void* p) { untrack(p); __libc_free(p); }
void* memalign(size_t a, size_t n) { void* p = __libc_memalign(a, n); track(p, n); return p; }
void* aligned_alloc(size_t a, size_t n) { void* p = __libc_memalign(a, n); track(p, n); return p; }
int posix_memalign(void** out, size_t a, size_t n) { void* p = __libc_memalign(a, n); if (!p) return ENOMEM; *out = p; track(p, n); return 0; }
