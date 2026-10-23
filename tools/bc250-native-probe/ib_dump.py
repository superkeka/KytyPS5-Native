#!/usr/bin/env python3
# Lists the PM4 packets of an IB or DCB written by KYTY_NATIVE_GPU_BREADCRUMBS (ib_<n>.bin, dcb_<n>.bin),
# with the fields that matter when a submission hangs the GPU: draws and dispatches with their
# arguments, SET_BASE, DMA_DATA, waits, and register writes summarized per packet.
# Usage: ib_dump.py FILE [--all]   (--all also prints every register-write packet)

import struct
import sys

NAMES = {
    0x10: "NOP", 0x11: "SET_BASE", 0x12: "CLEAR_STATE", 0x13: "INDEX_BUFFER_SIZE", 0x15: "DISPATCH_DIRECT",
    0x16: "DISPATCH_INDIRECT", 0x25: "DRAW_INDIRECT", 0x26: "INDEX_BASE", 0x27: "DRAW_INDEX_2",
    0x28: "CONTEXT_CONTROL", 0x2A: "INDEX_TYPE", 0x2D: "DRAW_INDEX_AUTO", 0x2F: "NUM_INSTANCES",
    0x35: "DRAW_INDEX_OFFSET_2", 0x37: "WRITE_DATA", 0x3C: "WAIT_REG_MEM", 0x40: "COPY_DATA",
    0x42: "PFP_SYNC_ME", 0x46: "EVENT_WRITE", 0x49: "RELEASE_MEM", 0x50: "DMA_DATA", 0x58: "ACQUIRE_MEM",
    0x63: "SET_SH_REG_INDIRECT", 0x64: "SET_UCONFIG_REG_INDIRECT", 0x69: "SET_CONTEXT_REG",
    0x76: "SET_SH_REG", 0x79: "SET_UCONFIG_REG", 0x7A: "SET_UCONFIG_REG_INDEX", 0x93: "WAIT_REG_MEM_64",
    0x9F: "SET_CONTEXT_REG_INDIRECT",
}
REGS = {0x69, 0x76, 0x79, 0x7A}


def main():
    data = open(sys.argv[1], "rb").read()
    show_all = "--all" in sys.argv
    dw = list(struct.unpack(f"<{len(data) // 4}I", data[: len(data) // 4 * 4]))
    i = 0
    counts = {}
    while i < len(dw):
        h = dw[i]
        if (h >> 30) != 3:
            print(f"{i:6d}: non-type-3 header 0x{h:08x}, stopping")
            break
        size = 1 if (h & 0x3FFFFF00) == 0x3FFF1000 else ((h >> 16) & 0x3FFF) + 2
        op = (h >> 8) & 0xFF
        body = dw[i + 1 : i + size]
        name = NAMES.get(op, f"op_0x{op:02x}")
        counts[name] = counts.get(name, 0) + 1
        if op in REGS:
            if show_all:
                print(f"{i:6d}: {name} base 0x{body[0]:x} x{len(body) - 1}")
        elif op == 0x10:
            pass
        else:
            extra = ""
            if op == 0x15 and len(body) >= 3:
                extra = f" groups {body[0]}x{body[1]}x{body[2]}"
            elif op == 0x16 and len(body) >= 1:
                extra = f" data_offset 0x{body[0]:x}"
            elif op == 0x11 and len(body) >= 3:
                extra = f" index {body[0]} addr 0x{(body[2] << 32) | body[1]:x}"
            elif op == 0x50 and len(body) >= 6:
                c = body[0]
                extra = (f" src_sel {(c >> 29) & 3} dst_sel {(c >> 20) & 3} src 0x{(body[2] << 32) | body[1]:x}"
                         f" dst 0x{(body[4] << 32) | body[3]:x} bytes {body[5] & 0x1FFFFF}")
            elif op in (0x27, 0x35, 0x2D, 0x25):
                extra = " " + " ".join(f"0x{v:x}" for v in body[:5])
            elif op == 0x58:
                extra = f" coher_cntl 0x{body[0]:x} gcr 0x{body[6] if len(body) > 6 else 0:x}"
            elif op in (0x3C, 0x93):
                extra = " " + " ".join(f"0x{v:x}" for v in body[:6])
            print(f"{i:6d}: {name}{extra}")
        i += size
    print("packets:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
