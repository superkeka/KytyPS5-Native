// kyty-lodhw: the privileged half of GET_LOD_STATS on the BC-250, the part the PS5 kernel and ME do.
//
// The GPU's PRT LOD statistics reporter (CP_PRT_LOD_STATS_CNTL0..3, absolute dwords 0x20ad..0x20b0)
// counts texture samples per counter bank for every T# with MIP_STATS_CNT_EN and, when CNTL3's
// REPORT_AND_RESET is set, writes a 0x40-byte header plus one 8-byte record per bank to a GPU VA in a
// given VMID and resets the counters. The registers are privileged MMIO, so Kyty asks this helper to
// do it at each GET_LOD_STATS packet. Registers are accessed through debugfs amdgpu_regs, which keeps
// GFX powered for the access. Besides the reporter it reads any register (op 4, for probing what the
// PS5 kernel programs) and writes the few registers kWritable lists (op 5): what the PS5 kernel sets
// differently from Linux. A golden table in /etc/kyty/ps5-golden.conf (lines "dword mask value name",
// kept out of the repository) can be applied entry by entry and restored (ops 7-9); its registers
// are writable through op 5 too. Nothing persists past a reboot. Several clients can be connected at
// once; register access is serialized.
//
// Protocol on the unix socket (default /run/kyty-lodhw.sock): the client sends 16 bytes
// {uint32 op, uint32 size, uint64 va}; the reply is 8 bytes {int32 status, uint32 value}.
//   op 1: the VMID the kernel last gave a reserved-VMID VM (RLC_SPM_MC_CNTL[3:0]); value = VMID.
//   op 2: report `size` bytes at GPU VA `va` in that VMID; status 0 once REPORT_AND_RESET cleared.
//   op 3 | pasid << 8: the same in whichever VMID the kernel last mapped to `pasid` on the gfx hub
//      (IH_VMID_n_LUT, written at every VMID assignment); value = that VMID. The client's PASID is the
//      "pasid:" line of its DRM fd's /proc fdinfo. No VMID has to be reserved.
//   op 4: read register `size` (absolute dword offset); value = its contents.
//   op 5: write the low 32 bits of `va` to register `size`, if kWritable lists it; value = read back.
//   op 6: jump table entry `va` of engine `size` (0 PFP, 1 ME, 2 CE) through CP_*_UCODE_ADDR/DATA.
//   op 7: apply golden entry `size` ((reg & ~mask) | (value & mask)), all with 0xffffffff; value = the
//      register afterwards (one entry) or how many entries changed (all). The old value is kept.
//   op 8: restore golden entry `size` (all with 0xffffffff) to the value it had before op 7.
//   op 9: golden entry `size`: status -ENOENT past the end; value = its register; `va` unused.
//   Op 7 with `va` 1 also restores what it applied when this client disconnects.
// Banked and masked writes for the PS5 per-VMID, per-pipe and per-queue state (ops 10-13). `sel` is
// GRBM_GFX_CNTL's layout, PIPEID[1:0] MEID[3:2] VMID[7:4] QUEUEID[10:8], plus bit 11 to select at all:
//   op 10 | sel << 8: register `size` = (reg & ~mask) | (value & mask), `va` = value | mask << 32, if
//      kBanked (or, unselected, kWritable or the golden table) allows it for that selection; value =
//      read back. The first old value per register and selection is kept and put back by op 11 or
//      when the client disconnects.
//   op 11 | sel << 8: restore register `size` (all of this client's with 0xffffffff); value = count.
//   op 12 | sel << 8: read register `size` with that selection.
//   op 13 | pasid << 8: value = the gfx VMID (1-7) the kernel last gave `pasid`; -ESRCH if none.
// Build: cc -O2 -pthread -o kyty-lodhw kyty-lodhw.c. Run as root: kyty-lodhw [socket path] [socket group gid]

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

enum {
	kCntl0      = 0x20ad,
	kCntl1      = 0x20ae,
	kCntl2      = 0x20af,
	kCntl3      = 0x20b0,
	kCpfStatus  = 0x1260 + 0x0e27,
	kSpmMcCntl  = 0xA000 + 0x4c71,
	kIhVmidLut  = 0x10A0, // IH_VMID_0_LUT (OSSSYS seg0 0x10A0): the PASID each gfx-hub VMID serves
	kReport     = 1u << 19,
	kCachePol   = 1u << 28,
	kEnable     = 1u << 29, // always set by the PS5 kernel
	kMaxReport  = 0x4000,
	kMmioDwords = 0x20000, // register MMIO aperture: 512 KiB
};

// GC registers the PS5 kernel programs differently from Linux (ps5-kernel-gc-findings, section 6).
// Absolute dword offsets: GC segment 0 is 0x1260.
static const struct {
	uint32_t    reg;
	const char* name;
} kWritable[] = {
    {0x1260 + 0x13ac, "DB_DEBUG"},       {0x1260 + 0x13af, "DB_DEBUG4"},   {0x1260 + 0x1423, "CB_HW_CONTROL_3"},
    {0x1260 + 0x1422, "CB_HW_CONTROL_4"}, {0x1260 + 0x12e2, "TA_CNTL_AUX"}, {0x1260 + 0x1588, "UTCL1_CTRL"},
    // The GDS partition of the VMIDs KFD gives user-mode compute queues (8-15). amdgpu programs it only
    // for the VMIDs of its own jobs; the PS5 kernel gives the game's queues a static one at base 0.
    {0x1260 + 0x20b0, "GDS_VMID8_BASE"},  {0x1260 + 0x20b1, "GDS_VMID8_SIZE"},  {0x1260 + 0x20b2, "GDS_VMID9_BASE"},
    {0x1260 + 0x20b3, "GDS_VMID9_SIZE"},  {0x1260 + 0x20b4, "GDS_VMID10_BASE"}, {0x1260 + 0x20b5, "GDS_VMID10_SIZE"},
    {0x1260 + 0x20b6, "GDS_VMID11_BASE"}, {0x1260 + 0x20b7, "GDS_VMID11_SIZE"}, {0x1260 + 0x20b8, "GDS_VMID12_BASE"},
    {0x1260 + 0x20b9, "GDS_VMID12_SIZE"}, {0x1260 + 0x20ba, "GDS_VMID13_BASE"}, {0x1260 + 0x20bb, "GDS_VMID13_SIZE"},
    {0x1260 + 0x20bc, "GDS_VMID14_BASE"}, {0x1260 + 0x20bd, "GDS_VMID14_SIZE"}, {0x1260 + 0x20be, "GDS_VMID15_BASE"},
    {0x1260 + 0x20bf, "GDS_VMID15_SIZE"},
};

// Registers ops 10-12 select through GRBM_GFX_CNTL: what the PS5 kernel programs per VMID (the game's
// VMIDs here: the gfx VMID amdgpu gives the VM, KFD's 8-15), per gfx pipe and per compute queue, and
// the global ones that go with them.
enum { kGlobal, kVmid, kGfxPipe, kHqd };
enum { kSel = 1u << 11, kSavedMax = 256 };
static const struct {
	uint32_t    reg;
	int         bank;
	const char* name;
} kBanked[] = {
    {0x1260 + 0x100d, kGfxPipe, "GFX_PIPE_CONTROL"},
    {0x1260 + 0x10aa, kVmid, "SH_MEM_BASES"},
    {0x1260 + 0x11df, kVmid, "SPI_USER_ACCUM_VMID_CNTL"},
    {0x1260 + 0x11ef, kGlobal, "SPI_CONFIG_CNTL_1"},
    {0x1260 + 0x1fad, kHqd, "CP_HQD_PERSISTENT_STATE"},
    {0x1260 + 0x1fba, kHqd, "CP_HQD_PQ_CONTROL"},
    {0x1260 + 0x1fdb, kHqd, "CP_HQD_GDS_RESOURCE_STATE"},
    // GWS and OA of KFD's VMIDs, which amdgpu leaves empty (the PS5 title's VMID has 32 and 8).
    {0x1260 + 0x20c8, kGlobal, "GDS_GWS_VMID8"},  {0x1260 + 0x20c9, kGlobal, "GDS_GWS_VMID9"},
    {0x1260 + 0x20ca, kGlobal, "GDS_GWS_VMID10"}, {0x1260 + 0x20cb, kGlobal, "GDS_GWS_VMID11"},
    {0x1260 + 0x20cc, kGlobal, "GDS_GWS_VMID12"}, {0x1260 + 0x20cd, kGlobal, "GDS_GWS_VMID13"},
    {0x1260 + 0x20ce, kGlobal, "GDS_GWS_VMID14"}, {0x1260 + 0x20cf, kGlobal, "GDS_GWS_VMID15"},
    {0x1260 + 0x20d8, kGlobal, "GDS_OA_VMID8"},   {0x1260 + 0x20d9, kGlobal, "GDS_OA_VMID9"},
    {0x1260 + 0x20da, kGlobal, "GDS_OA_VMID10"},  {0x1260 + 0x20db, kGlobal, "GDS_OA_VMID11"},
    {0x1260 + 0x20dc, kGlobal, "GDS_OA_VMID12"},  {0x1260 + 0x20dd, kGlobal, "GDS_OA_VMID13"},
    {0x1260 + 0x20de, kGlobal, "GDS_OA_VMID14"},  {0x1260 + 0x20df, kGlobal, "GDS_OA_VMID15"},
};
enum { kHqdActive = 0x1260 + 0x1fab, kHqdVmid = 0x1260 + 0x1fac };

// CP jump table access (GC segment 1 is 0xA000): {address register, data register} per engine.
static const uint32_t kJt[3][2] = {{0xA000 + 0x5814, 0xA000 + 0x5815}, {0xA000 + 0x5816, 0xA000 + 0x5817},
                                   {0xA000 + 0x5818, 0xA000 + 0x5819}};

static pthread_mutex_t g_hw = PTHREAD_MUTEX_INITIALIZER;

static uint32_t Rd(uint32_t reg);
static int      Wr(uint32_t reg, uint32_t v);
static uint32_t RdSel(uint32_t reg, uint32_t sel);
static int      WrSel(uint32_t reg, uint32_t sel, uint32_t v);

#define kGoldenPath "/etc/kyty/ps5-golden.conf"
static struct Golden {
	uint32_t reg, mask, value, saved;
	int      applied;
	char     name[40];
} g_golden[128];
static uint32_t g_golden_count;

// Entries whose register reads all ones (blocked for the host while PSP owns it) are skipped.
static void LoadGolden(void) {
	FILE* f = fopen(kGoldenPath, "re");
	if (f == NULL) {
		return;
	}
	char line[256];
	while (fgets(line, sizeof(line), f) != NULL && g_golden_count < sizeof(g_golden) / sizeof(g_golden[0])) {
		struct Golden e;
		memset(&e, 0, sizeof(e));
		if (line[0] == '#' || sscanf(line, "%x %x %x %39s", &e.reg, &e.mask, &e.value, e.name) < 3) {
			continue;
		}
		if (e.reg >= kMmioDwords || Rd(e.reg) == 0xffffffffu) {
			fprintf(stderr, "kyty-lodhw: golden %s 0x%05x not accessible, skipped\n", e.name, e.reg);
			continue;
		}
		g_golden[g_golden_count++] = e;
	}
	fclose(f);
	fprintf(stderr, "kyty-lodhw: %u golden entries from %s\n", g_golden_count, kGoldenPath);
}

static uint32_t ApplyGolden(struct Golden* e) {
	const uint32_t cur  = Rd(e->reg);
	const uint32_t want = (cur & ~e->mask) | (e->value & e->mask);
	if (!e->applied) {
		e->saved   = cur;
		e->applied = 1;
	}
	if (want != cur) {
		Wr(e->reg, want);
		fprintf(stderr, "kyty-lodhw: golden %s 0x%08x -> 0x%08x\n", e->name, cur, want);
	}
	return Rd(e->reg);
}

static void RestoreGolden(struct Golden* e) {
	if (e->applied) {
		Wr(e->reg, e->saved);
		e->applied = 0;
		fprintf(stderr, "kyty-lodhw: golden %s restored to 0x%08x\n", e->name, e->saved);
	}
}

static int g_regs = -1;

// amdgpu_regs takes a GRBM_GFX_CNTL selection in the file offset (bit 61; ME [33:24], PIPE [43:34],
// QUEUE [53:44], VMID [58:54]), applies it under srbm_mutex and selects 0 again afterwards.
static off_t Off(uint32_t reg, uint32_t sel) {
	uint64_t o = (uint64_t)reg * 4;
	if ((sel & kSel) != 0) {
		o |= 1ull << 61 | (uint64_t)((sel >> 2) & 3u) << 24 | (uint64_t)(sel & 3u) << 34 | (uint64_t)((sel >> 8) & 7u) << 44 |
		     (uint64_t)((sel >> 4) & 0xfu) << 54;
	}
	return (off_t)o;
}

static uint32_t RdSel(uint32_t reg, uint32_t sel) {
	uint32_t v = 0;
	if (pread(g_regs, &v, 4, Off(reg, sel)) != 4) {
		return 0xffffffffu;
	}
	return v;
}

static int WrSel(uint32_t reg, uint32_t sel, uint32_t v) {
	return pwrite(g_regs, &v, 4, Off(reg, sel)) == 4 ? 0 : -errno;
}

static uint32_t Rd(uint32_t reg) { return RdSel(reg, 0); }
static int      Wr(uint32_t reg, uint32_t v) { return WrSel(reg, 0, v); }

static double Now(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Waits for the previous report to finish (REPORT_AND_RESET clear), up to `seconds`.
static int Idle(double seconds) {
	const double end = Now() + seconds;
	for (;;) {
		const uint32_t c3 = Rd(kCntl3);
		if (c3 == 0xffffffffu) {
			return -EIO;
		}
		if ((c3 & kReport) == 0) {
			return 0;
		}
		if (Now() > end) {
			return -ETIMEDOUT;
		}
		usleep(20);
	}
}

static int Report(uint64_t va, uint32_t size, uint32_t vmid) {
	if (size < 0x40 || size > kMaxReport || (va & 0x3f) != 0 || (va >> 48) != 0 || vmid == 0 || vmid >= 15) {
		return -EINVAL;
	}
	int r = Idle(0.01);
	if (r != 0) {
		return r;
	}
	if ((r = Wr(kCntl0, size)) != 0 || (r = Wr(kCntl1, (uint32_t)(va >> 6))) != 0 ||
	    (r = Wr(kCntl2, (uint32_t)(va >> 38) & 0x3ffu)) != 0 ||
	    (r = Wr(kCntl3, kEnable | kCachePol | kReport | (vmid << 23))) != 0) {
		return r;
	}
	return Idle(0.01);
}

static const char* WritableName(uint32_t reg) {
	for (size_t i = 0; i < sizeof(kWritable) / sizeof(kWritable[0]); i++) {
		if (kWritable[i].reg == reg) {
			return kWritable[i].name;
		}
	}
	for (uint32_t i = 0; i < g_golden_count; i++) {
		if (g_golden[i].reg == reg) {
			return g_golden[i].name;
		}
	}
	return NULL;
}

// The name op 10 writes `reg` under for selection `sel`, or NULL when it may not: banked registers only
// with a selection their bank has (a compute queue only while it is KFD's, active in VMID 8-15), the
// others only unselected.
static const char* BankedName(uint32_t reg, uint32_t sel) {
	const uint32_t pipe = sel & 3u, me = (sel >> 2) & 3u, vmid = (sel >> 4) & 0xfu, queue = (sel >> 8) & 7u;
	if (sel >= 2 * kSel) {
		return NULL;
	}
	for (size_t i = 0; i < sizeof(kBanked) / sizeof(kBanked[0]); i++) {
		if (kBanked[i].reg != reg) {
			continue;
		}
		switch (kBanked[i].bank) {
			case kGlobal: return sel == 0 ? kBanked[i].name : NULL;
			case kVmid: return (sel & kSel) != 0 && pipe == 0 && me == 0 && queue == 0 && vmid != 0 ? kBanked[i].name : NULL;
			case kGfxPipe: return (sel & kSel) != 0 && pipe < 2 && me == 0 && queue == 0 && vmid == 0 ? kBanked[i].name : NULL;
			default:
				return (sel & kSel) != 0 && (me == 1 || me == 2) && vmid == 0 && (RdSel(kHqdActive, sel) & 1u) != 0 &&
				               (RdSel(kHqdVmid, sel) & 0xfu) >= 8
				           ? kBanked[i].name
				           : NULL;
		}
	}
	return sel == 0 ? WritableName(reg) : NULL;
}

struct Saved {
	uint32_t reg, sel, old, set;
};

// Puts one op 10 write back, unless it went to a compute queue that is no longer mapped (its next
// mapping loads the queue's MQD anyway). The value found first shows whether something rewrote it.
static void RestoreSaved(const struct Saved* e) {
	const char*    name = BankedName(e->reg, e->sel);
	const uint32_t cur  = RdSel(e->reg, e->sel);
	if (name == NULL) {
		fprintf(stderr, "kyty-lodhw: 0x%05x [sel 0x%03x] not restored (queue unmapped), reads 0x%08x\n", e->reg, e->sel, cur);
		return;
	}
	WrSel(e->reg, e->sel, e->old);
	fprintf(stderr, "kyty-lodhw: %s [sel 0x%03x] 0x%08x (set 0x%08x) restored to 0x%08x\n", name, e->sel, cur, e->set, e->old);
}

// One connected client. Only a client that issued reports leaves the reporter to be reset.
static void* Client(void* arg) {
	const int     c       = (int)(intptr_t)arg;
	unsigned long reports = 0, failures = 0;
	struct Saved* saved   = calloc(kSavedMax, sizeof(struct Saved));
	uint32_t      nsaved  = 0;
	int           golden  = 0; // op 7 asked to restore the golden table at disconnect
	for (;;) {
		struct {
			uint32_t op, size;
			uint64_t va;
		} q;
		if (recv(c, &q, sizeof(q), 0) != (ssize_t)sizeof(q)) {
			break;
		}
		struct {
			int32_t  status;
			uint32_t value;
		} rep = {0, 0};
		pthread_mutex_lock(&g_hw);
		if (q.op == 1) {
			rep.value = Rd(kSpmMcCntl) & 0xfu;
		} else if (q.op == 2) {
			rep.status = Report(q.va, q.size, Rd(kSpmMcCntl) & 0xfu);
			if (rep.status == 0) {
				reports++;
			} else if (failures++ < 8) {
				fprintf(stderr, "kyty-lodhw: report of 0x%x bytes at 0x%llx failed: %s\n", q.size,
				        (unsigned long long)q.va, strerror(-rep.status));
			}
		} else if ((q.op & 0xffu) == 3) {
			const uint32_t pasid = q.op >> 8;
			uint32_t       vmid  = 0;
			for (uint32_t v = 1; v < 15 && pasid != 0; v++) {
				if (Rd(kIhVmidLut + v) == pasid) {
					vmid = v;
					break;
				}
			}
			rep.value  = vmid;
			rep.status = vmid == 0 ? -ESRCH : Report(q.va, q.size, vmid);
			// The VMID could have gone to another VM meanwhile; the report then landed there.
			if (rep.status == 0 && Rd(kIhVmidLut + vmid) != pasid) {
				rep.status = -EAGAIN;
			}
			if (rep.status == 0) {
				reports++;
			} else if (failures++ < 8) {
				fprintf(stderr, "kyty-lodhw: report for pasid %u (VMID %u) at 0x%llx failed: %s\n", pasid, vmid,
				        (unsigned long long)q.va, strerror(-rep.status));
			}
		} else if (q.op == 4) {
			rep.status = q.size < kMmioDwords ? 0 : -EINVAL;
			rep.value  = rep.status == 0 ? Rd(q.size) : 0;
		} else if (q.op == 5) {
			const char* name = WritableName(q.size);
			if (name == NULL) {
				rep.status = -EPERM;
			} else {
				const uint32_t old = Rd(q.size);
				rep.status         = Wr(q.size, (uint32_t)q.va);
				rep.value          = Rd(q.size);
				fprintf(stderr, "kyty-lodhw: %s 0x%08x -> 0x%08x (reads 0x%08x)\n", name, old, (uint32_t)q.va, rep.value);
			}
		} else if (q.op == 6) {
			if (q.size > 2 || q.va >= 0x100) {
				rep.status = -EINVAL;
			} else {
				rep.status = Wr(kJt[q.size][0], (uint32_t)q.va);
				rep.value  = rep.status == 0 ? Rd(kJt[q.size][1]) : 0;
			}
		} else if (q.op == 7 || q.op == 8) {
			golden |= q.op == 7 && q.va == 1;
			if (q.size == 0xffffffffu) {
				for (uint32_t i = 0; i < g_golden_count; i++) {
					const uint32_t before = Rd(g_golden[i].reg);
					if (q.op == 7) {
						rep.value += ApplyGolden(&g_golden[i]) != before;
					} else {
						RestoreGolden(&g_golden[i]);
					}
				}
			} else if (q.size < g_golden_count) {
				if (q.op == 7) {
					rep.value = ApplyGolden(&g_golden[q.size]);
				} else {
					RestoreGolden(&g_golden[q.size]);
					rep.value = Rd(g_golden[q.size].reg);
				}
			} else {
				rep.status = -ENOENT;
			}
		} else if (q.op == 9) {
			rep.status = q.size < g_golden_count ? 0 : -ENOENT;
			rep.value  = rep.status == 0 ? g_golden[q.size].reg : 0;
		} else if ((q.op & 0xffu) == 10) {
			const uint32_t sel  = q.op >> 8;
			const uint32_t mask = (uint32_t)(q.va >> 32);
			const char*    name = BankedName(q.size, sel);
			uint32_t       i    = 0;
			while (i < nsaved && (saved[i].reg != q.size || saved[i].sel != sel)) {
				i++;
			}
			if (name == NULL || mask == 0) {
				rep.status = name == NULL ? -EPERM : -EINVAL;
			} else if (i == nsaved && (saved == NULL || nsaved == kSavedMax)) {
				rep.status = -ENOSPC;
			} else {
				const uint32_t old  = RdSel(q.size, sel);
				const uint32_t want = (old & ~mask) | ((uint32_t)q.va & mask);
				if (i == nsaved) {
					saved[nsaved++] = (struct Saved) {q.size, sel, old, want};
				}
				saved[i].set = want;
				rep.status   = want != old ? WrSel(q.size, sel, want) : 0;
				rep.value    = RdSel(q.size, sel);
				if (want != old) {
					fprintf(stderr, "kyty-lodhw: %s [sel 0x%03x] 0x%08x -> 0x%08x (reads 0x%08x)\n", name, sel, old, want, rep.value);
				}
			}
		} else if ((q.op & 0xffu) == 11) {
			const uint32_t sel = q.op >> 8;
			for (uint32_t i = nsaved; i-- > 0;) {
				if (q.size == 0xffffffffu || (saved[i].reg == q.size && saved[i].sel == sel)) {
					RestoreSaved(&saved[i]);
					memmove(&saved[i], &saved[i + 1], (nsaved - i - 1) * sizeof(saved[0]));
					nsaved--;
					rep.value++;
				}
			}
		} else if ((q.op & 0xffu) == 12) {
			const uint32_t sel = q.op >> 8;
			rep.status         = q.size < kMmioDwords && sel < 2 * kSel ? 0 : -EINVAL;
			rep.value          = rep.status == 0 ? RdSel(q.size, sel) : 0;
		} else if ((q.op & 0xffu) == 13) {
			const uint32_t pasid = q.op >> 8;
			rep.status           = -ESRCH;
			for (uint32_t v = 1; v < 8 && pasid != 0; v++) {
				if (Rd(kIhVmidLut + v) == pasid) {
					rep.value  = v;
					rep.status = 0;
					break;
				}
			}
		} else {
			rep.status = -EINVAL;
		}
		pthread_mutex_unlock(&g_hw);
		if (send(c, &rep, sizeof(rep), MSG_NOSIGNAL) != (ssize_t)sizeof(rep)) {
			break;
		}
	}
	close(c);
	if (nsaved != 0 || golden) {
		// The client is gone: what it changed goes back, last change first.
		pthread_mutex_lock(&g_hw);
		for (uint32_t i = nsaved; i-- > 0;) {
			RestoreSaved(&saved[i]);
		}
		for (uint32_t i = 0; golden && i < g_golden_count; i++) {
			RestoreGolden(&g_golden[i]);
		}
		pthread_mutex_unlock(&g_hw);
	}
	free(saved);
	if (reports != 0 || failures != 0) {
		// The client is gone: leave the reporter idle with no buffer behind it.
		pthread_mutex_lock(&g_hw);
		Idle(0.05);
		Wr(kCntl3, 0);
		Wr(kCntl2, 0);
		Wr(kCntl1, 0);
		Wr(kCntl0, 0);
		pthread_mutex_unlock(&g_hw);
		fprintf(stderr, "kyty-lodhw: client done, %lu reports, %lu failures\n", reports, failures);
	}
	return NULL;
}

int main(int argc, char** argv) {
	const char* path = argc > 1 ? argv[1] : "/run/kyty-lodhw.sock";
	const int   gid  = argc > 2 ? atoi(argv[2]) : -1;
	glob_t      g;
	if (glob("/sys/kernel/debug/dri/*/amdgpu_regs", 0, NULL, &g) != 0 || g.gl_pathc == 0) {
		fprintf(stderr, "kyty-lodhw: no debugfs amdgpu_regs (run as root, debugfs mounted)\n");
		return 1;
	}
	g_regs = open(g.gl_pathv[0], O_RDWR | O_CLOEXEC);
	if (g_regs < 0) {
		fprintf(stderr, "kyty-lodhw: open %s: %s\n", g.gl_pathv[0], strerror(errno));
		return 1;
	}
	globfree(&g);
	LoadGolden();

	const int s = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
	struct sockaddr_un a;
	memset(&a, 0, sizeof(a));
	a.sun_family = AF_UNIX;
	snprintf(a.sun_path, sizeof(a.sun_path), "%s", path);
	unlink(path);
	if (s < 0 || bind(s, (struct sockaddr*)&a, sizeof(a)) != 0 || listen(s, 4) != 0) {
		fprintf(stderr, "kyty-lodhw: socket %s: %s\n", path, strerror(errno));
		return 1;
	}
	if (gid >= 0) {
		if (chown(path, 0, (gid_t)gid) != 0 || chmod(path, 0660) != 0) {
			fprintf(stderr, "kyty-lodhw: chown/chmod %s: %s\n", path, strerror(errno));
		}
	} else {
		chmod(path, 0666);
	}
	fprintf(stderr, "kyty-lodhw: listening on %s (CNTL3 now 0x%08x, CP_CPF_STATUS 0x%08x)\n", path, Rd(kCntl3),
	        Rd(kCpfStatus));

	for (;;) {
		const int c = accept4(s, NULL, NULL, SOCK_CLOEXEC);
		if (c < 0) {
			continue;
		}
		pthread_t t;
		if (pthread_create(&t, NULL, Client, (void*)(intptr_t)c) != 0) {
			close(c);
			continue;
		}
		pthread_detach(t);
	}
}
