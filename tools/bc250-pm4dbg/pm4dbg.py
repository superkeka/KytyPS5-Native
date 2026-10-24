#!/usr/bin/env python3
# PM4 decoder and GPU-timing report for the BC-250 native backend of Kyty.
#
# Reads the capture files the emulator writes with KYTY_NATIVE_GPU_CAPTURE_DIR (see README.md):
#   pm4.bin    guest DCBs/ACBs, their translated IBs and the guest memory their user data points at
#   trace.bin  GPU timestamps of every buffer (and draw/dispatch) with the CPU-side pipeline times
# Breadcrumb files (KYTY_NATIVE_GPU_BREADCRUMBS ib_<n>.bin / dcb_<n>.bin) work too, as a single buffer.
#
#   pm4dbg.py list    pm4.bin                         buffers in the dump
#   pm4dbg.py packets pm4.bin -b 1234 [--raw] [--regs] every packet of a buffer's IB (--raw: the guest buffer)
#   pm4dbg.py draws   pm4.bin -b 1234                 its draws/dispatches: shaders, targets, counts
#   pm4dbg.py state   pm4.bin -b 1234 -i 17 [--all]   register state at item 17: shaders, user data with
#                                                    V#/T#/S#, render targets, depth, viewport (--all: every
#                                                    register with its fields)
#   pm4dbg.py table   pm4.bin -b 1234 0xADDR          a dumped table as V#, T# and S#
#   pm4dbg.py find    pm4.bin --shader 0xADDR         the items that use a shader
#   pm4dbg.py profile trace.bin [--top 25] [--frames] where each frame's GPU time goes
#
# -b takes a DCB number as the shadow log prints it ("frame 1234"), "acb:56" for an ACB, or "#7" for
# the 7th buffer in the file. Items count draws and dispatches together, in IB order (the numbering
# of trace.bin's items and of KYTY_NATIVE_GPU_DRAW_MARKERS).

import argparse
import bisect
import json
import os
import struct
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

OPS = {
    0x10: "NOP", 0x11: "SET_BASE", 0x12: "CLEAR_STATE", 0x13: "INDEX_BUFFER_SIZE", 0x15: "DISPATCH_DIRECT",
    0x16: "DISPATCH_INDIRECT", 0x1D: "ATOMIC_GDS", 0x1E: "ATOMIC_MEM", 0x1F: "OCCLUSION_QUERY",
    0x20: "SET_PREDICATION", 0x21: "REG_RMW", 0x22: "COND_EXEC", 0x23: "PRED_EXEC", 0x24: "DRAW_INDIRECT",
    0x25: "DRAW_INDEX_INDIRECT", 0x26: "INDEX_BASE", 0x27: "DRAW_INDEX_2", 0x28: "CONTEXT_CONTROL",
    0x2A: "INDEX_TYPE", 0x2C: "DRAW_INDIRECT_MULTI", 0x2D: "DRAW_INDEX_AUTO", 0x2F: "NUM_INSTANCES",
    0x30: "DRAW_INDEX_MULTI_AUTO", 0x33: "INDIRECT_BUFFER_CNST", 0x34: "STRMOUT_BUFFER_UPDATE",
    0x35: "DRAW_INDEX_OFFSET_2", 0x36: "DRAW_PREAMBLE", 0x37: "WRITE_DATA", 0x38: "DRAW_INDEX_INDIRECT_MULTI",
    0x3A: "MEM_SEMAPHORE", 0x3C: "WAIT_REG_MEM", 0x3F: "INDIRECT_BUFFER", 0x40: "COPY_DATA", 0x41: "CP_DMA",
    0x42: "PFP_SYNC_ME", 0x43: "SURFACE_SYNC", 0x45: "COND_WRITE", 0x46: "EVENT_WRITE", 0x47: "EVENT_WRITE_EOP",
    0x48: "EVENT_WRITE_EOS", 0x49: "RELEASE_MEM", 0x4A: "PREAMBLE_CNTL", 0x50: "DMA_DATA", 0x58: "ACQUIRE_MEM",
    0x5E: "REWIND", 0x63: "SET_SH_REG_INDIRECT", 0x64: "SET_UCONFIG_REG_INDIRECT", 0x68: "SET_CONFIG_REG",
    0x69: "SET_CONTEXT_REG", 0x76: "SET_SH_REG", 0x77: "SET_SH_REG_OFFSET", 0x79: "SET_UCONFIG_REG",
    0x7A: "SET_UCONFIG_REG_INDEX", 0x8E: "GET_LOD_STATS", 0x93: "WAIT_REG_MEM_64", 0x9B: "SET_SH_REG_INDEX",
    0x9F: "SET_CONTEXT_REG_INDIRECT",
}
DRAWS = {0x24, 0x25, 0x27, 0x2D, 0x35}
DISPATCHES = {0x15, 0x16}
# Kyty's IT_NOP markers (register field of the header).
NOP_MARKERS = {0x06: "R_WAIT_FLIP_DONE", 0x17: "R_FLIP", 0x18: "R_RELEASE_MEM", 0x1A: "R_CONTEXT_STATE"}
EVENTS = {0x14: "CACHE_FLUSH_AND_INV_TS", 0x19: "PIPELINESTAT_START", 0x1E: "SAMPLE_PIPELINESTAT",
          0x28: "BOTTOM_OF_PIPE_TS", 0x16: "CS_PARTIAL_FLUSH", 0x10: "PS_PARTIAL_FLUSH", 0x0F: "VS_PARTIAL_FLUSH",
          0x2F: "CS_DONE", 0x30: "PS_DONE", 0x2C: "FLUSH_AND_INV_DB_META", 0x2D: "FLUSH_AND_INV_CB_META",
          0x2E: "CACHE_FLUSH_AND_INV", 0x1A: "ZPASS_DONE", 0x15: "CACHE_FLUSH_AND_INV", 0x07: "VGT_FLUSH",
          0x31: "FLUSH_AND_INV_CB_PIXEL_DATA", 0x04: "CACHE_FLUSH_TS", 0x05: "CONTEXT_DONE",
          0x0A: "SQ_NON_EVENT", 0x18: "THREAD_TRACE_FINISH"}

# Stages: (name, SH offset of PGM_LO, of PGM_RSRC1 (RSRC2 follows), of USER_DATA_0, user-data count)
STAGES = [("PS", 0x008, 0x00A, 0x00C, 32), ("VS", 0x048, 0x04A, 0x04C, 32), ("GS", 0x088, 0x08A, 0x08C, 32),
          ("ES", 0x0C8, 0x0CA, 0x0CC, 32), ("HS", 0x108, 0x10A, 0x10C, 32), ("LS", 0x148, 0x14A, 0x14C, 32),
          ("CS", 0x20C, 0x212, 0x240, 16)]
DRAW_STAGES = ["PS", "VS", "GS", "ES", "HS", "LS"]


def u64(lo, hi):
    return (hi << 32) | lo


def plausible(a):
    return 0x100000000 <= a < 0x100000000000 and (a & 3) == 0


# --------------------------------------------------------------------------------------------------
# Register database (regdb.json, from gen_regdb.py)

class RegDb:
    def __init__(self, path=None):
        self.db = json.load(open(path or os.path.join(HERE, "regdb.json")))

    def entry(self, space, off):
        return self.db[space].get(f"0x{off:03x}")

    def name(self, space, off):
        e = self.entry(space, off)
        return e["name"] if e else f"{space}_0x{off:03x}"

    def fields(self, space, off, value, nonzero=False):
        e = self.entry(space, off)
        if not e or not e["fields"]:
            return ""
        out = []
        for fname, shift, width in e["fields"]:
            v = (value >> shift) & ((1 << width) - 1)
            if nonzero and v == 0:
                continue
            out.append(f"{fname}={v}" if v < 16 else f"{fname}=0x{v:x}")
        return " ".join(out)

    def find(self, space, name):
        for k, e in self.db[space].items():
            if e["name"] == name:
                return int(k, 16)
        return None


# --------------------------------------------------------------------------------------------------
# Files

class Buffer:
    def __init__(self, number, acb, flip, raw, ib, pos):
        self.number, self.acb, self.flip, self.raw, self.ib, self.pos = number, acb, flip, raw, ib, pos
        self.mem = {}  # addr -> bytes

    @property
    def label(self):
        return f"{'acb' if self.acb else 'dcb'}:{self.number}"

    def read(self, addr, size):
        """Up to `size` bytes at addr from the dumped tables (None when not dumped)."""
        best = None
        for base, data in self.mem.items():
            if base <= addr < base + len(data):
                got = data[addr - base:addr - base + size]
                if best is None or len(got) > len(best):
                    best = got
        return best


def records(data):
    pos = 0
    while pos + 8 <= len(data):
        rtype, size = struct.unpack_from("<II", data, pos)
        yield rtype, data[pos + 8:pos + 8 + size]
        pos += 8 + size


def dwords(b):
    return list(struct.unpack(f"<{len(b) // 4}I", b[:len(b) // 4 * 4]))


def load_dump(path):
    data = open(path, "rb").read()
    first = struct.unpack_from("<I", data, 0)[0] if len(data) >= 4 else 0
    if (first >> 30) == 3:  # a bare IB or DCB (breadcrumbs)
        d = dwords(data)
        return [Buffer(0, False, 0, d, d, 0)]
    bufs = []
    for rtype, p in records(data):
        if rtype == 16:
            number, flip, acb, raw_dw, ib_dw, _ = struct.unpack_from("<QQIIII", p, 0)
            raw = dwords(p[32:32 + raw_dw * 4])
            ib = dwords(p[32 + raw_dw * 4:32 + raw_dw * 4 + ib_dw * 4])
            bufs.append(Buffer(number, bool(acb), flip, raw, ib, len(bufs)))
        elif rtype == 17 and bufs:
            addr, nbytes, _ = struct.unpack_from("<QII", p, 0)
            bufs[-1].mem[addr] = p[16:16 + nbytes]
    return bufs


def pick(bufs, spec):
    if spec is None:
        if len(bufs) == 1:
            return bufs[0]
        sys.exit("several buffers in the file: choose one with -b (see `list`)")
    if spec.startswith("#"):
        return bufs[int(spec[1:])]
    acb = spec.startswith("acb:")
    n = int(spec.split(":")[-1], 0)
    for b in bufs:
        if b.acb == acb and b.number == n:
            return b
    sys.exit(f"buffer {spec} is not in the file")


# --------------------------------------------------------------------------------------------------
# Packets and state

def packets(ib):
    """(dword offset, opcode, header, body) for every type-3 packet."""
    i = 0
    while i < len(ib):
        h = ib[i]
        if (h >> 30) != 3:
            yield i, None, h, []
            return
        size = 1 if (h & 0x3FFFFF00) == 0x3FFF1000 else ((h >> 16) & 0x3FFF) + 2
        yield i, (h >> 8) & 0xFF, h, ib[i + 1:i + size]
        i += size


class State:
    """Register state while walking an IB, plus the draw-engine state packets set."""

    def __init__(self):
        self.cx, self.sh, self.uc = {}, {}, {}
        self.index_base = self.index_size = self.base = 0
        self.instances = 1
        self.cleared = False

    def apply(self, op, body):
        if op == 0x69 and body:
            for k, v in enumerate(body[1:]):
                self.cx[(body[0] & 0xFFFF) + k] = v
        elif op == 0x76 and body:
            for k, v in enumerate(body[1:]):
                self.sh[(body[0] & 0xFFFF) + k] = v
        elif op in (0x79, 0x7A) and body:
            for k, v in enumerate(body[1:]):
                self.uc[(body[0] & 0xFFFF) + k] = v
        elif op == 0x12:
            self.cx = {}
            self.cleared = True
        elif op == 0x26 and len(body) >= 2:
            self.index_base = u64(body[0], body[1] & 0xFFFF)
        elif op == 0x13 and body:
            self.index_size = body[0]
        elif op == 0x2F and body:
            self.instances = body[0]
        elif op == 0x11 and len(body) >= 3:
            self.base = u64(body[1], body[2] & 0xFFFF)

    def pgm(self, lo):
        return (self.sh.get(lo, 0) << 8) | (self.sh.get(lo + 1, 0) << 40)


def items(ib):
    """(item index, dw, op, body, state at the item) for every draw/dispatch; the state is live."""
    st = State()
    n = 0
    for dw, op, h, body in packets(ib):
        if op is None:
            break
        if op in DRAWS or op in DISPATCHES:
            yield n, dw, op, body, st
            n += 1
        else:
            st.apply(op, body)


def describe_packet(dw, op, h, body, reg, regs=False):
    name = OPS.get(op, f"op_0x{op:02x}") if op is not None else "?"
    if op == 0x10:
        m = (h >> 2) & 0x3F
        extra = NOP_MARKERS.get(m, "")
        return f"{dw:7d}: NOP{(' ' + extra) if extra else ''} ({len(body)} dw)" if (extra or regs) else None
    s = f"{dw:7d}: {name}"
    b = body
    if op in (0x69, 0x76, 0x79, 0x7A) and b:
        space = {0x69: "cx", 0x76: "sh", 0x79: "uc", 0x7A: "uc"}[op]
        base = b[0] & 0xFFFF
        if not regs:
            names = [reg.name(space, base + k) for k in range(len(b) - 1)]
            return f"{s} {names[0]}{' ..' + names[-1] if len(names) > 1 else ''} ({len(b) - 1})"
        lines = [s + (f" index {b[0] >> 28}" if op == 0x7A else "")]
        for k, v in enumerate(b[1:]):
            lines.append(f"           {reg.name(space, base + k):34s} = 0x{v:08x}  {reg.fields(space, base + k, v, True)}")
        return "\n".join(lines)
    if op == 0x27 and len(b) >= 5:
        s += f" max {b[0]} index_addr 0x{u64(b[1], b[2]):x} count {b[3]} initiator 0x{b[4]:x}"
    elif op == 0x35 and len(b) >= 4:
        s += f" max {b[0]} offset {b[1]} count {b[2]} initiator 0x{b[3]:x}"
    elif op == 0x2D and len(b) >= 2:
        s += f" count {b[0]} initiator 0x{b[1]:x}"
    elif op in (0x24, 0x25) and b:
        s += f" data_offset 0x{b[0]:x}" + (f" base_vtx_loc {b[1] & 0xFFFF} start_inst_loc {b[2] & 0xFFFF}" if len(b) >= 3 else "")
    elif op == 0x15 and len(b) >= 4:
        s += f" groups {b[0]}x{b[1]}x{b[2]} initiator 0x{b[3]:x}"
    elif op == 0x16 and b:
        s += f" data_offset 0x{b[0]:x}" + (f" initiator 0x{b[1]:x}" if len(b) > 1 else "")
    elif op == 0x11 and len(b) >= 3:
        s += f" index {b[0]} addr 0x{u64(b[1], b[2] & 0xFFFF):x}"
    elif op == 0x26 and len(b) >= 2:
        s += f" addr 0x{u64(b[0], b[1] & 0xFFFF):x}"
    elif op in (0x13, 0x2F) and b:
        s += f" {b[0]}"
    elif op == 0x50 and len(b) >= 6:
        c = b[0]
        s += (f" src_sel {(c >> 29) & 3} dst_sel {(c >> 20) & 3} src 0x{u64(b[1], b[2]):x} dst 0x{u64(b[3], b[4]):x}"
              f" bytes {b[5] & 0x1FFFFF}")
    elif op in (0x3C, 0x93) and len(b) >= 5:
        f = ["always", "<", "<=", "==", "!=", ">=", ">"]
        s += (f" {'mem' if b[0] & 0x10 else 'reg'} 0x{u64(b[1] & ~3, b[2]):x} {f[b[0] & 7] if (b[0] & 7) < 7 else '?'}"
              f" 0x{b[3]:x} mask 0x{b[4] if op == 0x3C else b[5]:x} engine {'pfp' if b[0] & 0x100 else 'me'}")
    elif op == 0x46 and b:
        s += f" {EVENTS.get(b[0] & 0x3F, hex(b[0] & 0x3F))} index {(b[0] >> 8) & 0xF}"
    elif op == 0x49 and len(b) >= 5:
        s += (f" {EVENTS.get(b[0] & 0x3F, hex(b[0] & 0x3F))} index {(b[0] >> 8) & 0xF} gcr 0x{(b[0] >> 12) & 0xFFFF:x}"
              f" data_sel {b[1] >> 29} int_sel {(b[1] >> 24) & 7} addr 0x{u64(b[2], b[3]):x}"
              + (f" data 0x{u64(b[4], b[5]):x}" if len(b) >= 6 else ""))
    elif op == 0x40 and len(b) >= 5:
        s += f" src_sel {b[0] & 0xF} dst_sel {(b[0] >> 8) & 0xF} src 0x{u64(b[1], b[2]):x} dst 0x{u64(b[3], b[4]):x}"
    elif op == 0x37 and len(b) >= 3:
        s += f" dst_sel {(b[0] >> 8) & 0xF} addr 0x{u64(b[1], b[2]):x} data " + " ".join(f"0x{v:x}" for v in b[3:7])
    elif op == 0x58 and len(b) >= 7:
        s += f" coher_cntl 0x{b[0]:x} size 0x{u64(b[1], b[2]):x} base 0x{u64(b[3], b[4]):x} gcr 0x{b[6]:x}"
    elif op in (0x9F, 0x63, 0x64) and len(b) >= 4:
        s += f" pairs at 0x{u64(b[0] & ~3, b[1]):x} count {b[3] & 0x3FFF}"
    elif op == 0x28 and len(b) >= 2:
        s += f" load 0x{b[0]:x} shadow 0x{b[1]:x}"
    else:
        s += " " + " ".join(f"0x{v:x}" for v in b[:6]) + (" ..." if len(b) > 6 else "")
    return s


# --------------------------------------------------------------------------------------------------
# Descriptors (gfx10 layouts, Mesa sid.h / PAL)

BUF_FMT = {0: "INVALID", 1: "8_UNORM", 2: "8_SNORM", 5: "8_UINT", 7: "16_UNORM", 13: "16_FLOAT", 20: "32_UINT",
           22: "32_FLOAT", 64: "32_32_32_32_UINT?", 77: "32_32_32_32_FLOAT"}
IMG_TYPES = {8: "1D", 9: "2D", 10: "3D", 11: "CUBE", 12: "1D_ARRAY", 13: "2D_ARRAY", 14: "2D_MSAA", 15: "2D_MSAA_ARRAY"}
SEL = "01??XYZW"
CLAMP = ["wrap", "mirror", "clamp_edge", "mirror_once_edge", "clamp_half_border", "mirror_once_half_border",
         "clamp_border", "mirror_once_border"]
FILT = ["point", "bilinear", "aniso_point", "aniso_linear"]


def dsel(v, shift):
    return "".join(SEL[(v >> (shift + 3 * k)) & 7] for k in range(4))


def vsharp(d):
    base = u64(d[0], d[1] & 0xFFFF)
    stride = (d[1] >> 16) & 0x3FFF
    fmt = (d[3] >> 12) & 0x7F
    return (f"V# base 0x{base:x} stride {stride} records {d[2]} ({d[2] * (stride or 1)} B) sel {dsel(d[3], 0)}"
            f" fmt {BUF_FMT.get(fmt, fmt)} oob {(d[3] >> 28) & 3} type {d[3] >> 30}")


def vsharp_ok(d):
    base = u64(d[0], d[1] & 0xFFFF)
    return (d[3] >> 30) == 0 and (d[3] & 0xFFF) != 0 and plausible(base) and 0 < d[2] < (1 << 31)


def tsharp(d):
    base = u64(d[0], d[1] & 0xFF) << 8
    width = (((d[1] >> 30) & 3) | ((d[2] & 0xFFF) << 2)) + 1
    height = ((d[2] >> 14) & 0x3FFF) + 1
    ty = (d[3] >> 28) & 0xF
    s = (f"T# base 0x{base:x} {IMG_TYPES.get(ty, ty)} {width}x{height} fmt {(d[1] >> 20) & 0x1FF} sel {dsel(d[3], 0)}"
         f" levels {(d[3] >> 12) & 0xF}-{(d[3] >> 16) & 0xF} swizzle {(d[3] >> 20) & 0x1F}")
    if len(d) >= 8:
        s += (f" depth {(d[4] & 0x1FFF) + 1} base_array {(d[4] >> 16) & 0x1FFF} meta 0x{((d[7] << 8) | (d[6] >> 24)) << 8:x}"
              f" compression {(d[6] >> 22) & 1}")
    return s


def tsharp_ok(d):
    return 8 <= ((d[3] >> 28) & 0xF) <= 15 and plausible(u64(d[0], d[1] & 0xFF) << 8)


def ssharp(d):
    return (f"S# clamp {CLAMP[d[0] & 7]}/{CLAMP[(d[0] >> 3) & 7]}/{CLAMP[(d[0] >> 6) & 7]} aniso {(d[0] >> 9) & 7}"
            f" cmp {(d[0] >> 12) & 7} mag {FILT[(d[2] >> 20) & 3]} min {FILT[(d[2] >> 22) & 3]} mip {(d[2] >> 26) & 3}"
            f" lod {(d[1] & 0xFFF) / 256:.2f}-{((d[1] >> 12) & 0xFFF) / 256:.2f} border {d[3] >> 30}")


def ssharp_ok(d):
    # Filters and clamps in range, no address-like dwords, border type in its two bits.
    return (d[3] & 0x3FFFF000) == 0 and ((d[2] >> 28) & 3) == 0 and d[0] < (1 << 31) and not plausible(u64(d[0], d[1] & 0xFFFF))


def classify(d, offset):
    """Descriptor interpretations of the dwords at d[offset:], the plausible ones."""
    out = []
    g = d[offset:offset + 8]
    if len(g) >= 8 and tsharp_ok(g):
        out.append(tsharp(g))
    if len(g) >= 4 and vsharp_ok(g[:4]):
        out.append(vsharp(g[:4]))
    return out


# --------------------------------------------------------------------------------------------------
# Commands: list / packets / draws / state / table / find

def cmd_list(a):
    bufs = load_dump(a.file)
    print(f"{len(bufs)} buffers")
    for b in bufs:
        n_draw = n_disp = 0
        for _, _, op, _, _ in items(b.ib):
            n_draw += op in DRAWS
            n_disp += op in DISPATCHES
        print(f"#{b.pos:<4d} {b.label:12s} flip {b.flip:6d}  guest {len(b.raw):7d} dw  ib {len(b.ib):7d} dw  "
              f"{n_draw:5d} draws {n_disp:4d} dispatches  {len(b.mem):5d} tables")


def cmd_packets(a):
    reg = RegDb()
    b = pick(load_dump(a.file), a.buffer)
    item = 0
    for dw, op, h, body in packets(b.raw if a.raw else b.ib):
        if op is None:
            print(f"{dw:7d}: not a type-3 header 0x{h:08x}, stopping")
            break
        s = describe_packet(dw, op, h, body, reg, a.regs)
        if s is not None:
            tag = ""
            if op in DRAWS or op in DISPATCHES:
                tag = f"   [item {item}]"
                item += 1
            print(s + tag)


def render_targets(st, reg):
    out = []
    mask = st.cx.get(0x08E, 0)
    for i in range(8):
        step = i * 0xF
        info = st.cx.get(0x31C + step, 0)
        base = (st.cx.get(0x318 + step, 0) << 8) | ((st.cx.get(0x390 + i, 0) & 0xFF) << 40)
        if ((info >> 2) & 0x1F) == 0 and ((mask >> (4 * i)) & 0xF) == 0:
            continue
        a2 = st.cx.get(0x3B0 + i, 0)
        a3 = st.cx.get(0x3B8 + i, 0)
        view = st.cx.get(0x31B + step, 0)
        cmask = (st.cx.get(0x31F + step, 0) << 8) | ((st.cx.get(0x398 + i, 0) & 0xFF) << 40)
        fmask = (st.cx.get(0x321 + step, 0) << 8) | ((st.cx.get(0x3A0 + i, 0) & 0xFF) << 40)
        dcc = (st.cx.get(0x325 + step, 0) << 8) | ((st.cx.get(0x3A8 + i, 0) & 0xFF) << 40)
        out.append(f"  MRT{i}: 0x{base:x} {((a2 >> 14) & 0x3FFF) + 1}x{(a2 & 0x3FFF) + 1} write mask {(mask >> (4 * i)) & 0xF:x}"
                   f" slices {view & 0x7FF}-{(view >> 13) & 0x7FF}\n"
                   f"        INFO {reg.fields('cx', 0x31C, info, True)}\n"
                   f"        ATTRIB3 {reg.fields('cx', 0x3B8, a3, True)}\n"
                   f"        cmask 0x{cmask:x} fmask 0x{fmask:x} dcc 0x{dcc:x}")
    return out


def depth(st, reg):
    zi, si = st.cx.get(0x010, 0), st.cx.get(0x011, 0)
    bases = [(st.cx.get(0x012 + k, 0) << 8) | ((st.cx.get(0x01A + k, 0) & 0xFF) << 40) for k in range(4)]
    htile = (st.cx.get(0x005, 0) << 8) | ((st.cx.get(0x01E, 0) & 0xFF) << 40)
    xy = st.cx.get(0x007, 0)
    return [f"  size {(xy & 0x3FFF) + 1}x{((xy >> 16) & 0x3FFF) + 1}  z read/write 0x{bases[0]:x}/0x{bases[2]:x}"
            f"  stencil read/write 0x{bases[1]:x}/0x{bases[3]:x}  htile 0x{htile:x}",
            f"  DB_Z_INFO {reg.fields('cx', 0x010, zi, True)}",
            f"  DB_STENCIL_INFO {reg.fields('cx', 0x011, si, True)}",
            f"  DB_DEPTH_CONTROL {reg.fields('cx', 0x200, st.cx.get(0x200, 0), True)}",
            f"  DB_RENDER_CONTROL {reg.fields('cx', 0x000, st.cx.get(0x000, 0), True)}"]


def f32(v):
    return struct.unpack("<f", struct.pack("<I", v & 0xFFFFFFFF))[0]


def user_data(b, st, rsrc1, ud0, count):
    rsrc2 = st.sh.get(rsrc1 + 1, 0)
    n_user = (rsrc2 >> 1) & 0x1F
    n_user = n_user if n_user else count
    vals = [st.sh.get(ud0 + k) for k in range(count)]
    lines = []
    shown = [v for v in vals[:n_user] if v is not None]
    if not shown:
        return lines
    lines.append(f"  user data ({n_user} SGPRs per RSRC2):")
    for k in range(min(n_user, count)):
        v = vals[k]
        if v is None:
            continue
        note = ""
        nxt = vals[k + 1] if k + 1 < count else None
        if nxt is not None and plausible(u64(v, nxt & 0xFFFF)):
            ptr = u64(v, nxt & 0xFFFF)
            data = b.read(ptr, 64)
            note = f" -> 0x{ptr:x}" + (" (table dumped)" if data else "")
        lines.append(f"    s[{k:2d}] 0x{v:08x}  {f32(v):<12.6g}{note}")
    # Inline descriptors in the SGPRs, and descriptors in the tables they point at.
    seen = set()
    for k in range(min(n_user, count)):
        g = [x for x in vals[k:k + 8] if x is not None]
        if len(g) >= 4 and k % 4 == 0:
            for c in classify(g, 0):
                lines.append(f"    s[{k}..] inline {c}")
        if k + 1 < count and vals[k] is not None and vals[k + 1] is not None:
            ptr = u64(vals[k], vals[k + 1] & 0xFFFF)
            if plausible(ptr) and ptr not in seen:
                seen.add(ptr)
                data = b.read(ptr, 512)
                if data:
                    d = dwords(data)
                    for off in range(0, len(d) - 3, 4):
                        for c in classify(d, off):
                            lines.append(f"    s[{k}:{k + 1}] 0x{ptr:x} +0x{off * 4:03x}: {c}")
    return lines


def cmd_state(a):
    reg = RegDb()
    b = pick(load_dump(a.file), a.buffer)
    for n, dw, op, body, st in items(b.ib):
        if n != a.item:
            continue
        dispatch = op in DISPATCHES
        print(f"{b.label} item {n} @ IB dw {dw}: " + describe_packet(dw, op, 0, body, reg).split(": ", 1)[1])
        if not dispatch:
            vgt = st.uc.get(0x242, 0)
            print(f"  primitive type {vgt & 0x3F}, index type {st.uc.get(0x243, 0) & 3}, instances {st.instances},"
                  f" index base 0x{st.index_base:x} size {st.index_size}, indirect base 0x{st.base:x}")
        elif op == 0x16:
            print(f"  indirect base 0x{st.base:x}")
        print("shaders:")
        for name, lo, rsrc1, ud0, count in STAGES:
            if (name == "CS") != dispatch:
                continue
            addr = st.pgm(lo)
            if addr == 0:
                continue
            print(f"  {name} 0x{addr:x}  RSRC1 {reg.fields('sh', rsrc1, st.sh.get(rsrc1, 0), True)}")
            print(f"       RSRC2 {reg.fields('sh', rsrc1 + 1, st.sh.get(rsrc1 + 1, 0), True)}")
            for line in user_data(b, st, rsrc1, ud0, count):
                print(line)
        if dispatch:
            for o in (0x204, 0x205, 0x206, 0x207, 0x212, 0x215):
                if o in st.sh:
                    print(f"  {reg.name('sh', o)} = 0x{st.sh[o]:08x} {reg.fields('sh', o, st.sh[o], True)}")
        else:
            print("render targets:")
            for line in render_targets(st, reg):
                print(line)
            print("depth/stencil:")
            for line in depth(st, reg):
                print(line)
            sc = [st.cx.get(o, 0) for o in (0x090, 0x091, 0x094, 0x095, 0x0C, 0x0D)]
            print(f"viewport 0: scale/offset x {f32(st.cx.get(0x10F, 0)):.1f}/{f32(st.cx.get(0x110, 0)):.1f}"
                  f" y {f32(st.cx.get(0x111, 0)):.1f}/{f32(st.cx.get(0x112, 0)):.1f}"
                  f" z {f32(st.cx.get(0x113, 0)):.3f}/{f32(st.cx.get(0x114, 0)):.3f};"
                  f" generic scissor ({sc[0] & 0x7FFF},{(sc[0] >> 16) & 0x7FFF})-({sc[1] & 0x7FFF},{(sc[1] >> 16) & 0x7FFF})"
                  f" vport scissor ({sc[2] & 0x7FFF},{(sc[2] >> 16) & 0x7FFF})-({sc[3] & 0x7FFF},{(sc[3] >> 16) & 0x7FFF})")
            for o in (0x08F, 0x202, 0x1E0, 0x204, 0x205, 0x206, 0x1B3, 0x1B4, 0x203, 0x292):
                print(f"  {reg.name('cx', o)} = 0x{st.cx.get(o, 0):08x} {reg.fields('cx', o, st.cx.get(o, 0), True)}")
        if a.all:
            for space, regs in (("cx", st.cx), ("sh", st.sh), ("uc", st.uc)):
                print(f"{space} registers ({len(regs)}):")
                for o in sorted(regs):
                    print(f"  {space} 0x{o:03x} {reg.name(space, o):40s} 0x{regs[o]:08x}  {reg.fields(space, o, regs[o], True)}")
        if not st.cleared:
            print("(no CLEAR_STATE before this item: registers not shown are whatever the ring held)")
        return
    sys.exit(f"{b.label} has no item {a.item}")


def cmd_draws(a):
    reg = RegDb()
    b = pick(load_dump(a.file), a.buffer)
    prev_rt = None
    for n, dw, op, body, st in items(b.ib):
        if op in DISPATCHES:
            groups = f"{body[0]}x{body[1]}x{body[2]}" if op == 0x15 and len(body) >= 3 else "indirect"
            print(f"item {n:5d} @{dw:7d} {OPS[op]:20s} CS 0x{st.pgm(0x20C):x} groups {groups}")
            continue
        count = {0x27: lambda: body[3], 0x35: lambda: body[2], 0x2D: lambda: body[0]}.get(op, lambda: "ind")()
        mask = st.cx.get(0x08E, 0)
        rts = []
        for i in range(8):
            base = (st.cx.get(0x318 + i * 0xF, 0) << 8) | ((st.cx.get(0x390 + i, 0) & 0xFF) << 40)
            if (mask >> (4 * i)) & 0xF and base:
                rts.append(f"{i}:0x{base:x}")
        zb = (st.cx.get(0x014, 0) << 8) | ((st.cx.get(0x01C, 0) & 0xFF) << 40)
        rt = " ".join(rts) + (f" z:0x{zb:x}" if zb and (st.cx.get(0x010, 0) & 3) else "")
        print(f"item {n:5d} @{dw:7d} {OPS[op]:20s} count {count!s:>7} inst {st.instances:4d} PS 0x{st.pgm(0x008):x}"
              f" ES 0x{st.pgm(0x0C8):x}" + (f"  RT {rt}" if rt != prev_rt else ""))
        prev_rt = rt


def cmd_table(a):
    b = pick(load_dump(a.file), a.buffer)
    addr = int(a.addr, 0)
    data = b.read(addr, 512)
    if data is None:
        sys.exit(f"0x{addr:x} was not dumped with {b.label}")
    d = dwords(data)
    for off in range(0, len(d) - 3, 4):
        g = d[off:off + 8]
        print(f"+0x{off * 4:03x}: " + " ".join(f"{v:08x}" for v in g[:4]))
        print(f"        {vsharp(g[:4])}{'' if vsharp_ok(g[:4]) else '  (unlikely)'}")
        if len(g) >= 8:
            print(f"        {tsharp(g)}{'' if tsharp_ok(g) else '  (unlikely)'}")
        print(f"        {ssharp(g[:4])}{'' if ssharp_ok(g[:4]) else '  (unlikely)'}")


def cmd_find(a):
    want = int(a.shader, 0)
    for b in load_dump(a.file):
        for n, dw, op, body, st in items(b.ib):
            hit = [s for s, lo, _, _, _ in STAGES if st.pgm(lo) == want and ((s == "CS") == (op in DISPATCHES))]
            if hit:
                print(f"{b.label} flip {b.flip} item {n} ({OPS[op]}) as {','.join(hit)}")


# --------------------------------------------------------------------------------------------------
# Profile report

class Trace:
    def __init__(self, path):
        self.khz = 100000
        self.calib, self.flips, self.batches, self.buffers, self.items, self.thens = [], [], {}, [], [], {}
        for rtype, p in records(open(path, "rb").read()):
            if rtype == 1:
                _, self.khz, _ = struct.unpack_from("<IIQ", p)
            elif rtype == 2:
                b, g, e = struct.unpack_from("<QQQ", p)
                if g:
                    self.calib.append(((b + e) / 2, g, e - b))
            elif rtype == 3:
                self.flips.append(struct.unpack_from("<QQ", p))
            elif rtype == 4:
                bid, seq, cb, ce, fence, nbuf, handles = struct.unpack_from("<QQQQQII", p)
                self.batches[bid] = dict(id=bid, seq=seq, cs_begin=cb, cs_end=ce, fence=fence, handles=handles)
            elif rtype == 5:
                f = struct.unpack_from("<QQQQQQIIIIII", p)
                self.buffers.append(dict(batch=f[0], number=f[1], enqueue=f[2], translated=f[3], issued=f[4], done=f[5],
                                         acb=f[6], draws=f[7], dispatches=f[8], flips=f[9], ib_dw=f[10], barriers=f[11]))
            elif rtype == 6:
                f = struct.unpack_from("<QQQQQIIII3II", p)
                self.items.append(dict(number=f[0], shader=f[1], es=f[2], issued=f[3], done=f[4], acb=f[5], index=f[6],
                                       dw=f[7], op=f[8], args=f[9:12]))
            elif rtype == 7:
                number, start, end, acb, _ = struct.unpack_from("<QQQII", p)
                self.thens[(acb, number)] = (start, end)
        # GPU clock -> CPU time: least squares over the calibration samples (the GPU clock may drift
        # against CLOCK_MONOTONIC, so a fit beats a single offset).
        cal = sorted(self.calib, key=lambda c: c[2])[:max(4, len(self.calib) // 2)] if self.calib else []
        if len(cal) >= 2:
            n = len(cal)
            mx = sum(c[1] for c in cal) / n
            my = sum(c[0] for c in cal) / n
            sxx = sum((c[1] - mx) ** 2 for c in cal)
            self.slope = sum((c[1] - mx) * (c[0] - my) for c in cal) / sxx if sxx else 1e6 / self.khz
            self.g0, self.c0 = mx, my
        elif cal:
            self.slope, self.g0, self.c0 = 1e6 / self.khz, cal[0][1], cal[0][0]
        else:
            self.slope = None

    def ns(self, ticks):
        return self.c0 + (ticks - self.g0) * self.slope


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def cmd_profile(a):
    t = Trace(a.file)
    if t.slope is None:
        sys.exit("no GPU clock calibration in the trace")
    ms = 1e-6
    print(f"GPU clock {t.khz} kHz; fitted {t.slope * t.khz / 1e6:.6f} x nominal over {len(t.calib)} samples; "
          f"{len(t.flips)} flips, {len(t.batches)} batches, {len(t.buffers)} buffers, {len(t.items)} items")
    bufs = [b for b in t.buffers if b["issued"] and b["done"] >= b["issued"]]
    lost = len(t.buffers) - len(bufs)
    for b in bufs:
        b["i"], b["d"] = t.ns(b["issued"]), t.ns(b["done"])
        bt = t.batches.get(b["batch"], {})
        b["cs_begin"], b["cs_end"], b["fence"] = bt.get("cs_begin", 0), bt.get("cs_end", 0), bt.get("fence", 0)
    bufs.sort(key=lambda b: b["i"])
    early = sum(1 for b in bufs if b["i"] + 50e3 < b["cs_end"])
    print(f"clock check: {early} of {len(bufs)} buffers appear to start >50 us before their CS returned"
          f"{' (calibration suspect)' if early > len(bufs) // 20 else ''}; {lost} buffers without timestamps")
    flips = [f[1] for f in t.flips]
    if len(flips) < 3:
        sys.exit("fewer than 3 flips in the trace")

    # Walk the GPU timeline once: busy intervals, and each idle gap split by what the next buffer was
    # doing on the CPU side at that time.
    cats = ["guest", "queue+translate", "batching", "cs_ioctl", "kernel->gpu", "in-batch"]
    frames = [dict(busy=0.0, dcb=0.0, acb=0.0, **{c: 0.0 for c in cats}) for _ in range(len(flips) - 1)]

    def add(kind, lo, hi):
        """Adds [lo, hi) to the frames it overlaps."""
        if hi <= lo:
            return
        k = max(0, bisect.bisect_right(flips, lo) - 1)
        while k < len(frames) and flips[k] < hi:
            ov = min(hi, flips[k + 1]) - max(lo, flips[k])
            if ov > 0:
                frames[k][kind] += ov
            k += 1

    busy_until = None
    gaps = []
    prev = None
    for b in bufs:
        start = b["i"] if busy_until is None else max(b["i"], busy_until)
        if b["d"] > start:
            add("busy", start, b["d"])
            add("acb" if b["acb"] else "dcb", start, b["d"])
        if busy_until is not None and b["i"] > busy_until:
            lo, hi = busy_until, b["i"]
            gaps.append((hi - lo, lo, prev, b))
            if prev is not None and prev["batch"] == b["batch"]:
                add("in-batch", lo, hi)
            else:
                # The CPU-side milestones of the buffer that ended the gap, clipped to the gap.
                marks = [lo, max(lo, min(hi, b["enqueue"])), max(lo, min(hi, b["translated"])),
                         max(lo, min(hi, b["cs_begin"])), max(lo, min(hi, b["cs_end"])), hi]
                for c, (x, y) in zip(cats[:5], zip(marks, marks[1:])):
                    add(c, x, max(x, y))
        busy_until = b["d"] if busy_until is None else max(busy_until, b["d"])
        prev = b

    first = 1 if len(frames) > 4 else 0  # the first frame may start mid-way
    use = frames[first:-1] if len(frames) > 3 else frames
    intervals = [(flips[k + 1] - flips[k]) * ms for k in range(first, first + len(use))]
    n = len(use)

    def avg(key):
        return sum(f[key] for f in use) * ms / n

    frame = sum(intervals) / n
    idle = frame - avg("busy")
    print(f"\n== {n} frames (flip to flip at the presenter) ==")
    print(f"frame interval   {frame:6.2f} ms  (p50 {pct(intervals, 50):.2f}, p90 {pct(intervals, 90):.2f}, max {max(intervals):.2f})")
    print(f"GPU busy         {avg('busy'):6.2f} ms  ({100 * avg('busy') / frame:.0f}%):  DCB {avg('dcb'):.2f} ms, ACB {avg('acb'):.2f} ms"
          f" (overlap counted once, for the buffer that was running first)")
    print(f"GPU idle         {idle:6.2f} ms  ({100 * idle / frame:.0f}%), what the buffer that ended each gap was waiting for:")
    expl = {"guest": "not submitted by the game yet",
            "queue+translate": "submitted, waiting for / in the translation thread",
            "batching": "translated, waiting for the rest of its batch",
            "cs_ioctl": "inside the CS ioctl (BO list validation)",
            "kernel->gpu": "CS returned, GPU had not started it (scheduler, other contexts on the ring)",
            "in-batch": "between two buffers of one batch (CP-side: preambles, barriers)"}
    for c in cats:
        print(f"  {c:16s} {avg(c):6.2f} ms  {expl[c]}")
    other = idle - sum(avg(c) for c in cats)
    print(f"  {'(unattributed)':16s} {other:6.2f} ms  edges of the window, frames without a gap end")

    # Fence and continuation latencies: GPU done -> completion thread saw it -> Kyty processed it.
    fence_lat = [(b["fence"] - b["d"]) * ms for b in bufs if b["fence"]]
    then_lat = []
    for b in bufs:
        th = t.thens.get((b["acb"], b["number"]))
        if th:
            then_lat.append((th[1] - b["d"]) * ms)
    sub_lat = [(b["i"] - b["enqueue"]) * ms for b in bufs if b["enqueue"]]
    print("\nlatencies (ms)                         p50     p90     max")
    for name, xs in (("guest submit -> GPU start", sub_lat), ("GPU done -> fence seen", fence_lat),
                     ("GPU done -> Kyty CP done (then)", then_lat)):
        if xs:
            print(f"  {name:34s} {pct(xs, 50):6.2f}  {pct(xs, 90):6.2f}  {max(xs):6.2f}")
    per_frame = defaultdict(int)
    for b in bufs:
        per_frame["acb" if b["acb"] else "dcb"] += 1
    print(f"buffers per frame: DCB {per_frame['dcb'] / max(1, len(frames)):.1f}, ACB {per_frame['acb'] / max(1, len(frames)):.1f};"
          f" batches per frame {len(t.batches) / max(1, len(frames)):.1f}")

    gaps.sort(key=lambda g: -g[0])
    print(f"\nlongest GPU idle gaps:")
    for dur, lo, p, b in gaps[:a.top // 2 or 10]:
        k = max(0, bisect.bisect_right(flips, lo) - 1)
        why = "same batch" if p and p["batch"] == b["batch"] else (
            f"next: enqueued {(b['enqueue'] - lo) * ms:+.2f}, translated {(b['translated'] - lo) * ms:+.2f},"
            f" CS {(b['cs_begin'] - lo) * ms:+.2f}..{(b['cs_end'] - lo) * ms:+.2f}")
        print(f"  {dur * ms:6.2f} ms in frame {k}: after {'acb' if p['acb'] else 'dcb'}:{p['number']}"
              f" before {'acb' if b['acb'] else 'dcb'}:{b['number']} ({why})")

    if a.frames:
        print("\nper frame (ms): interval busy dcb acb | idle: " + " ".join(cats))
        for k, f in enumerate(frames):
            print(f"  {k:4d} {(flips[k + 1] - flips[k]) * ms:6.2f} {f['busy'] * ms:6.2f} {f['dcb'] * ms:6.2f} {f['acb'] * ms:6.2f} | "
                  + " ".join(f"{f[c] * ms:6.2f}" for c in cats))

    # Buffers by cost.
    top_bufs = sorted(bufs, key=lambda b: b["d"] - b["i"], reverse=True)[:a.top // 2 or 10]
    print("\nlongest buffers (issued -> done):")
    for b in top_bufs:
        print(f"  {(b['d'] - b['i']) * ms:6.2f} ms {'acb' if b['acb'] else 'dcb'}:{b['number']}: {b['draws']} draws,"
              f" {b['dispatches']} dispatches, {b['barriers']} wait barriers, ib {b['ib_dw']} dw")

    if not t.items:
        print("\n(no per-item timestamps: run with KYTY_NATIVE_GPU_PROFILE_DRAWS=1 for the costliest draws/dispatches)")
        return
    # Item cost: how much later it finished than everything before it (end-of-pipe timestamps retire
    # in order), i.e. its share of the critical path.
    by_buf = defaultdict(list)
    for it in t.items:
        if it["issued"] and it["done"]:
            by_buf[(it["acb"], it["number"])].append(it)
    costs = []
    for key, its in by_buf.items():
        its.sort(key=lambda x: x["index"])
        last = None
        for it in its:
            d = t.ns(it["done"])
            i0 = t.ns(it["issued"])
            it["cost"] = d - (max(i0, last) if last is not None else i0)
            it["wait"] = (i0 - last) if last is not None and i0 > last else 0.0
            last = d if last is None else max(last, d)
            costs.append(it)
    total = sum(max(0.0, c["cost"]) for c in costs)
    print(f"\nitems: {len(costs)} timed, critical-path total {total * ms / max(1, len(frames)):.2f} ms/frame")
    by_shader = defaultdict(lambda: [0.0, 0, 0])
    for c in costs:
        s = by_shader[(c["acb"], c["op"] in DISPATCHES, c["shader"])]
        s[0] += max(0.0, c["cost"])
        s[1] += 1
    for kind in (0, 1):
        rows = sorted(((v, k) for k, v in by_shader.items() if k[0] == kind), key=lambda r: -r[0][0])[:a.top]
        if not rows:
            continue
        print(f"\ncostliest shaders in {'ACB' if kind else 'DCB'} buffers (critical-path ms per frame, items per frame):")
        for (tot, cnt, _), (_, disp, sh) in rows:
            print(f"  {tot * ms / max(1, len(frames)):6.3f} ms  {cnt / max(1, len(frames)):6.1f}x  {'CS' if disp else 'PS'} 0x{sh:x}")
    print(f"\ncostliest single items:")
    for c in sorted(costs, key=lambda c: -c["cost"])[:a.top]:
        what = (f"groups {c['args'][0]}x{c['args'][1]}x{c['args'][2]}" if c["op"] == 0x15 else
                f"count {c['args'][0]}" if c["op"] in (0x27, 0x35, 0x2D) else "indirect")
        print(f"  {c['cost'] * ms:6.3f} ms  {'acb' if c['acb'] else 'dcb'}:{c['number']} item {c['index']} {OPS.get(c['op'], c['op'])}"
              f" {what} {'CS' if c['op'] in DISPATCHES else 'PS'} 0x{c['shader']:x}"
              + (f" (issued {c['wait'] * ms:.3f} ms after the previous item finished)" if c["wait"] * ms > 0.05 else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("file")
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("packets")
    s.add_argument("file")
    s.add_argument("-b", "--buffer")
    s.add_argument("--raw", action="store_true", help="the guest buffer instead of the IB")
    s.add_argument("--regs", action="store_true", help="every register write with its fields")
    s.set_defaults(fn=cmd_packets)
    s = sub.add_parser("draws")
    s.add_argument("file")
    s.add_argument("-b", "--buffer")
    s.set_defaults(fn=cmd_draws)
    s = sub.add_parser("state")
    s.add_argument("file")
    s.add_argument("-b", "--buffer")
    s.add_argument("-i", "--item", type=int, required=True)
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_state)
    s = sub.add_parser("table")
    s.add_argument("file")
    s.add_argument("-b", "--buffer")
    s.add_argument("addr")
    s.set_defaults(fn=cmd_table)
    s = sub.add_parser("find")
    s.add_argument("file")
    s.add_argument("--shader", required=True)
    s.set_defaults(fn=cmd_find)
    s = sub.add_parser("profile")
    s.add_argument("file")
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--frames", action="store_true", help="a line per frame")
    s.set_defaults(fn=cmd_profile)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
