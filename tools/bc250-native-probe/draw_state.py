#!/usr/bin/env python3
# Replays the register writes of a native IB (KYTY_NATIVE_GPU_BREADCRUMBS ib_<n>.bin) and prints, for
# every draw, the state that decides which pixels it writes: color target, masks, depth/stencil,
# blending, scissors, culling and the pixel shader. Use it to tell apart draws into one target when
# only some of them come out wrong.
# Usage: draw_state.py IB [--target 0xADDR]   (--target: only draws into that CB_COLOR0 base address)

import struct
import sys

CX = {
    "CB0": 0x318, "CB0_EXT": 0x390, "CB0_INFO": 0x31C, "CB0_ATTRIB3": 0x3B8, "TARGET_MASK": 0x8E,
    "SHADER_MASK": 0x8F, "COLOR_CONTROL": 0x202, "BLEND0": 0x1E0, "DEPTH_CONTROL": 0x200,
    "STENCIL_CONTROL": 0x10B, "STENCILREFMASK": 0x10C, "DB_RENDER_CONTROL": 0x000, "DB_Z_INFO": 0x010,
    "DB_SHADER_CONTROL": 0x203, "SU_SC_MODE": 0x205, "CLIP_CNTL": 0x204, "WIN_SCISSOR_TL": 0x081,
    "WIN_SCISSOR_BR": 0x082, "GEN_SCISSOR_TL": 0x090, "GEN_SCISSOR_BR": 0x091, "VPORT_SCISSOR_TL": 0x094,
    "VPORT_SCISSOR_BR": 0x095, "SC_MODE_CNTL_0": 0x292, "SC_MODE_CNTL_1": 0x293, "PS_INPUT_ENA": 0x1B3,
    "ALPHA_TEST?": 0x103, "BLEND_RED": 0x105, "CB_BLEND_ALPHA": 0x108,
}
SH = {"PS_PGM_LO": 0x008, "PS_RSRC1": 0x00A, "PS_RSRC2": 0x00B, "GS_PGM_LO": 0x088}


def main():
    data = open(sys.argv[1], "rb").read()
    want = int(sys.argv[sys.argv.index("--target") + 1], 16) if "--target" in sys.argv else None
    dw = struct.unpack(f"<{len(data) // 4}I", data[: len(data) // 4 * 4])
    cx, sh = {}, {}
    i, draw = 0, 0
    prev = None
    while i < len(dw):
        h = dw[i]
        if (h >> 30) != 3:
            break
        size = 1 if (h & 0x3FFFFF00) == 0x3FFF1000 else ((h >> 16) & 0x3FFF) + 2
        op = (h >> 8) & 0xFF
        body = dw[i + 1 : i + size]
        if op == 0x69:  # SET_CONTEXT_REG
            for k, v in enumerate(body[1:]):
                cx[(body[0] & 0xFFFF) + k] = v
        elif op == 0x76:  # SET_SH_REG
            for k, v in enumerate(body[1:]):
                sh[(body[0] & 0xFFFF) + k] = v
        elif op in (0x27, 0x2D, 0x35):
            base = (cx.get(CX["CB0"], 0) << 8) | ((cx.get(CX["CB0_EXT"], 0) & 0xFF) << 40)
            if want is None or base == want:
                state = {n: cx.get(o, 0) for n, o in CX.items()}
                state.update({n: sh.get(o, 0) for n, o in SH.items()})
                count = body[3] if op == 0x27 else (body[0] if op == 0x2D else body[2])
                changed = {k: v for k, v in state.items() if prev is None or prev.get(k) != v}
                print(f"draw {draw:3d} @dw {i:6d} op 0x{op:02x} count {count:6d} target 0x{base:x}" +
                      "".join(f"\n      {k} 0x{v:08x}" for k, v in changed.items()))
                prev = state
            draw += 1
        i += size


if __name__ == "__main__":
    main()
