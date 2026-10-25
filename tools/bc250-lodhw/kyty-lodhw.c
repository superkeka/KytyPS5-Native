// kyty-lodhw: the privileged half of GET_LOD_STATS on the BC-250, the part the PS5 kernel and ME do.
//
// The GPU's PRT LOD statistics reporter (CP_PRT_LOD_STATS_CNTL0..3, absolute dwords 0x20ad..0x20b0)
// counts texture samples per counter bank for every T# with MIP_STATS_CNT_EN and, when CNTL3's
// REPORT_AND_RESET is set, writes a 0x40-byte header plus one 8-byte record per bank to a GPU VA in a
// given VMID and resets the counters. The registers are privileged MMIO, so Kyty asks this helper to
// do it at each GET_LOD_STATS packet. Only those four registers (and reads of the VMID and status)
// are touched, through debugfs amdgpu_regs, which keeps GFX powered for the access.
//
// Protocol on the unix socket (default /run/kyty-lodhw.sock): the client sends 16 bytes
// {uint32 op, uint32 size, uint64 va}; the reply is 8 bytes {int32 status, uint32 value}.
//   op 1: the VMID the kernel last gave a reserved-VMID VM (RLC_SPM_MC_CNTL[3:0]); value = VMID.
//   op 2: report `size` bytes at GPU VA `va` in that VMID; status 0 once REPORT_AND_RESET cleared.
//   op 3 | pasid << 8: the same in whichever VMID the kernel last mapped to `pasid` on the gfx hub
//      (IH_VMID_n_LUT, written at every VMID assignment); value = that VMID. The client's PASID is the
//      "pasid:" line of its DRM fd's /proc fdinfo. No VMID has to be reserved.
// Build: cc -O2 -o kyty-lodhw kyty-lodhw.c. Run as root: kyty-lodhw [socket path] [socket group gid]

#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
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
};

static int g_regs = -1;

static uint32_t Rd(uint32_t reg) {
	uint32_t v = 0;
	if (pread(g_regs, &v, 4, (off_t)reg * 4) != 4) {
		return 0xffffffffu;
	}
	return v;
}

static int Wr(uint32_t reg, uint32_t v) {
	return pwrite(g_regs, &v, 4, (off_t)reg * 4) == 4 ? 0 : -errno;
}

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
		unsigned long reports = 0, failures = 0;
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
			} else {
				rep.status = -EINVAL;
			}
			if (send(c, &rep, sizeof(rep), MSG_NOSIGNAL) != (ssize_t)sizeof(rep)) {
				break;
			}
		}
		close(c);
		// The client is gone: leave the reporter idle with no buffer behind it.
		Idle(0.05);
		Wr(kCntl3, 0);
		Wr(kCntl2, 0);
		Wr(kCntl1, 0);
		Wr(kCntl0, 0);
		fprintf(stderr, "kyty-lodhw: client done, %lu reports, %lu failures\n", reports, failures);
	}
}
