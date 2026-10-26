#!/usr/bin/env python3
# Client for kyty-lodhw's register operations (ops 4-9, 12), run as the user in the socket's group.
#   kyty-hw.py read REG [COUNT]      read registers (absolute dword offsets, hex or decimal)
#   kyty-hw.py write REG VALUE       write a register the helper allows (kWritable)
#   kyty-hw.py jt pfp|me|ce [COUNT]  jump table entries
#   kyty-hw.py cp                    CP microcode state: IC base/control of PFP, ME and CE, CP_ME_CNTL, CP_STAT
#   kyty-hw.py ps5                   the registers the PS5 kernel programs differently from Linux
#   kyty-hw.py golden FILE           compare registers with a golden table (lines: dword mask value name)
#   kyty-hw.py apply all|N...        apply the helper's golden entries (its /etc/kyty/ps5-golden.conf)
#   kyty-hw.py restore all|N...      put them back to the values they had before
#   kyty-hw.py entries               list the helper's golden entries
#   kyty-hw.py busy [SECONDS]        how busy the GPU really is: GRBM_STATUS sampled as fast as possible
#   kyty-hw.py vmids                 per-VMID state: SH_MEM_BASES/CONFIG, SPI_USER_ACCUM_VMID_CNTL, GDS/GWS/OA
#   kyty-hw.py hqds                  active compute queues: VMID, PQ_CONTROL, PERSISTENT_STATE, GDS resource state
import os
import socket
import struct
import sys

SOCK = os.environ.get("KYTY_LOD_HW_SOCK", "/tmp/kyty-lodhw.sock")
GC0, GC1 = 0x1260, 0xA000
CP_REGS = [
    ("CP_PFP_IC_BASE_LO", GC1 + 0x5840), ("CP_PFP_IC_BASE_HI", GC1 + 0x5841), ("CP_PFP_IC_BASE_CNTL", GC1 + 0x5842),
    ("CP_ME_IC_BASE_LO", GC1 + 0x5844), ("CP_ME_IC_BASE_HI", GC1 + 0x5845), ("CP_ME_IC_BASE_CNTL", GC1 + 0x5846),
    ("CP_CE_IC_BASE_LO", GC1 + 0x5848), ("CP_CE_IC_BASE_HI", GC1 + 0x5849), ("CP_CE_IC_BASE_CNTL", GC1 + 0x584a),
    ("CP_ME_CNTL", GC0 + 0xf56), ("CP_STAT", GC0 + 0xf40), ("GRBM_STATUS", GC0 + 0xda4),
    ("CP_ME_HEADER_DUMP", GC0 + 0xf41), ("CP_PFP_HEADER_DUMP", GC0 + 0xf42),
]
PS5_REGS = [
    ("DB_DEBUG", GC0 + 0x13ac), ("DB_DEBUG4", GC0 + 0x13af), ("CB_HW_CONTROL_3", GC0 + 0x1423),
    ("CB_HW_CONTROL_4", GC0 + 0x1422), ("TA_CNTL_AUX", GC0 + 0x12e2), ("UTCL1_CTRL", GC0 + 0x1588),
]


class Helper:
    def __init__(self):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.s.connect(SOCK)

    def op(self, op, size, va=0):
        self.s.send(struct.pack("<IIQ", op, size, va))
        status, value = struct.unpack("<iI", self.s.recv(8))
        if status != 0:
            raise OSError(-status, os.strerror(-status))
        return value

    def read(self, reg):
        return self.op(4, reg)

    def read_sel(self, reg, sel):
        # sel: GRBM_GFX_CNTL layout (PIPEID[1:0] MEID[3:2] VMID[7:4] QUEUEID[10:8]); bit 11 selects.
        return self.op(12 | (sel | 0x800) << 8, reg)


def num(s):
    return int(s, 0)


def golden_table(path):
    for line in open(path):
        f = line.split("#")[0].split()
        if len(f) >= 3:
            yield num(f[0]), num(f[1]), num(f[2]), f[3] if len(f) > 3 else ""


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__ or open(__file__).read().split("\n\n")[0])
        return 1
    h = Helper()
    if a[0] == "read":
        reg, count = num(a[1]), num(a[2]) if len(a) > 2 else 1
        for r in range(reg, reg + count):
            print(f"0x{r:05x}: 0x{h.read(r):08x}")
    elif a[0] == "write":
        print(f"reads back 0x{h.op(5, num(a[1]), num(a[2])):08x}")
    elif a[0] == "jt":
        engine = {"pfp": 0, "me": 1, "ce": 2}[a[1]]
        count = num(a[2]) if len(a) > 2 else 0x80
        vals = [h.op(6, engine, i) for i in range(count)]
        for i in range(0, count, 8):
            print(f"{i:3d}: " + " ".join(f"{v:08x}" for v in vals[i:i + 8]))
    elif a[0] == "golden":
        differ = 0
        for reg, mask, value, name in golden_table(a[1]):
            cur = h.read(reg)
            want = (cur & ~mask) | (value & mask)
            same = cur == want
            differ += not same
            print(f"{'  ' if same else '* '}{name:30s} 0x{reg:05x} now 0x{cur:08x}  golden 0x{want:08x}"
                  f"{'' if same else f'  (bits 0x{cur ^ want:08x})'}")
        print(f"{differ} differ")
    elif a[0] in ("apply", "restore"):
        op = 7 if a[0] == "apply" else 8
        for which in a[1:]:
            v = h.op(op, 0xffffffff if which == "all" else num(which))
            print(f"{a[0]} {which}: " + (f"{v} changed" if which == "all" and op == 7 else f"0x{v:08x}"))
    elif a[0] == "entries":
        i = 0
        while True:
            try:
                reg = h.op(9, i)
            except OSError:
                break
            print(f"{i:3d}: 0x{reg:05x} = 0x{h.read(reg):08x}")
            i += 1
    elif a[0] == "busy":
        import time
        secs = float(a[1]) if len(a) > 1 else 3.0
        bits = {"GUI_ACTIVE": 31, "CP_BUSY": 29, "CB_BUSY": 30, "SPI_BUSY": 22, "PA_BUSY": 25, "DB_BUSY": 26,
                "TA_BUSY": 14, "GE_BUSY": 21, "SC_BUSY": 24, "CP_COHERENCY_BUSY": 28}
        count = {k: 0 for k in bits}
        n = 0
        end = time.monotonic() + secs
        while time.monotonic() < end:
            v = h.read(GC0 + 0xda4)
            n += 1
            for k, b in bits.items():
                count[k] += (v >> b) & 1
        print(f"{n} samples in {secs:.1f} s ({n / secs:.0f}/s)")
        for k in bits:
            print(f"  {k:18s} {100.0 * count[k] / n:5.1f} %")
    elif a[0] == "vmids":
        for v in range(16):
            print(f"VMID {v:2d}: SH_MEM_BASES 0x{h.read_sel(GC0 + 0x10aa, v << 4):08x}"
                  f" SH_MEM_CONFIG 0x{h.read_sel(GC0 + 0x10ad, v << 4):08x}"
                  f" USER_ACCUM 0x{h.read_sel(GC0 + 0x11df, v << 4):x}"
                  f" GDS 0x{h.read(GC0 + 0x20a0 + 2 * v):04x}+0x{h.read(GC0 + 0x20a1 + 2 * v):05x}"
                  f" GWS 0x{h.read(GC0 + 0x20c0 + v):08x} OA 0x{h.read(GC0 + 0x20d0 + v):04x}")
        print(f"SPI_CONFIG_CNTL_1 0x{h.read(GC0 + 0x11ef):08x}  GFX_PIPE_CONTROL pipe 0 0x{h.read_sel(GC0 + 0x100d, 0):08x}"
              f" pipe 1 0x{h.read_sel(GC0 + 0x100d, 1):08x}")
    elif a[0] == "hqds":
        for me in (1, 2):
            for pipe in range(4):
                for queue in range(8):
                    sel = queue << 8 | me << 2 | pipe
                    if h.read_sel(GC0 + 0x1fab, sel) & 1:
                        print(f"ME{me} pipe {pipe} queue {queue}: VMID {h.read_sel(GC0 + 0x1fac, sel) & 0xf:2d}"
                              f" PQ_CONTROL 0x{h.read_sel(GC0 + 0x1fba, sel):08x}"
                              f" PERSISTENT_STATE 0x{h.read_sel(GC0 + 0x1fad, sel):08x}"
                              f" GDS_RESOURCE_STATE 0x{h.read_sel(GC0 + 0x1fdb, sel):08x}")
    elif a[0] in ("cp", "ps5"):
        for name, reg in CP_REGS if a[0] == "cp" else PS5_REGS:
            print(f"{name:22s} 0x{reg:05x} = 0x{h.read(reg):08x}")
    else:
        print("unknown command")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
