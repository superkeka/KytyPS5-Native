#!/usr/bin/env python3
# Compares native and Vulkan renderings of the same guest flips (BC-250 native shadow mode).
#
# Input: a KYTY_NATIVE_GPU_DUMP_DIR with flip_<n>.native.raw, flip_<n>.vulkan.raw and flip_<n>.txt
# (written when KYTY_NATIVE_GPU_SHADOW=1 KYTY_NATIVE_GPU_DUMP_FLIPS=... are set).
# Output: <dir>/diff/report.md plus, per flip, native/vulkan previews and an amplified diff map.
#
# Usage: frame_diff.py DUMP_DIR [--step N] [--threshold T] [--flips A,B] [--crop X0,Y0,X1,Y1]
#   --step N       sample every Nth pixel for statistics and previews (default 4)
#   --threshold T  a pixel "differs" when its max channel difference exceeds T (default 8)
#   --flips A,B    only these flips
#   --crop ...     only this rectangle (full-resolution pixels), e.g. a "diff box" from the report;
#                  combine with --step 1 for pixel-exact crops (written as flip_<n>.crop.*.png)
# Pure Python (no numpy) so it runs on SteamOS as is.

import argparse
import os
import re
import struct
import sys
import zlib

# SW_64KB_R_X, 32 bpp (Prospero kRenderTarget 0x1b): the in-block offset is an XOR of an x-only and
# a y-only term (tools/bc250-present-probe/detile_rt.comp), so two 128-entry tables cover it.
RT_X = [((x << 2) & 0x00c) ^ ((x << 5) & 0x380) ^ ((x << 4) & 0x400) ^ ((x << 6) & 0x800) ^ ((x << 9) & 0xa000)
        for x in range(128)]
RT_Y = [((y << 4) & 0x070) ^ ((y << 5) & 0xf00) ^ ((y << 9) & 0x1000) ^ ((y << 8) & 0x4000) for y in range(128)]
LINEAR = 0x00
RENDER_TARGET = 0x1B


def read_meta(path):
    meta = {}
    for line in open(path):
        key, _, value = line.strip().partition(" ")
        meta[key] = int(value, 0)
    return meta


def pixel_offsets(meta, step, crop):
    w, h = meta["width"], meta["height"]
    x0, y0, x1, y1 = crop or (0, 0, w, h)
    tile = meta["tile_mode"]
    if meta["bytes_per_pixel"] != 4 or tile not in (LINEAR, RENDER_TARGET):
        raise ValueError(f"unsupported surface: bpp {meta['bytes_per_pixel']} tile 0x{tile:x}")
    blocks_per_row = (w + 127) // 128
    for y in range(y0, min(y1, h), step):
        row = []
        for x in range(x0, min(x1, w), step):
            if tile == LINEAR:
                row.append((y * w + x) * 4)
            else:
                row.append(((y >> 7) * blocks_per_row + (x >> 7)) * 65536 + (RT_X[x & 127] ^ RT_Y[y & 127]))
        yield row


def write_png(path, width, height, rows):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def compare(dump_dir, n, step, threshold, crop, out_dir):
    meta = read_meta(os.path.join(dump_dir, f"flip_{n}.txt"))
    native = open(os.path.join(dump_dir, f"flip_{n}.native.raw"), "rb").read()
    vulkan = open(os.path.join(dump_dir, f"flip_{n}.vulkan.raw"), "rb").read()
    blocks = len(native) // 65536
    diff_blocks = sum(1 for b in range(blocks) if native[b * 65536:(b + 1) * 65536] != vulkan[b * 65536:(b + 1) * 65536])
    result = {"flip": n, "w": meta["width"], "h": meta["height"], "identical": native == vulkan,
              "diff_blocks": diff_blocks, "blocks": blocks}
    rows_n, rows_v, rows_d = [], [], []
    count = differing = total = 0
    worst = 0
    box = [None, None, None, None]
    for oy, row in enumerate(pixel_offsets(meta, step, crop)):
        rn, rv, rd = bytearray(), bytearray(), bytearray()
        for ox, off in enumerate(row):
            a = native[off:off + 3]
            b = vulkan[off:off + 3]
            d = max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))
            rn += a
            rv += b
            amp = min(255, d * 8)
            rd += bytes((amp, 0, 0)) if d > threshold else bytes((a[0] // 4, a[1] // 4, a[2] // 4))
            count += 1
            total += d
            worst = max(worst, d)
            if d > threshold:
                differing += 1
                box = [ox if box[0] is None else min(box[0], ox), oy if box[1] is None else min(box[1], oy),
                       ox if box[2] is None else max(box[2], ox), oy if box[3] is None else max(box[3], oy)]
        rows_n.append(rn)
        rows_v.append(rv)
        rows_d.append(rd)
    ow, oh = len(rows_n[0]) // 3, len(rows_n)
    result.update(mean=total / count, max=worst, pct=100.0 * differing / count,
                  box=None if box[0] is None else tuple(v * step + (crop[i % 2] if crop else 0)
                                                        for i, v in enumerate(box)))
    prefix = os.path.join(out_dir, f"flip_{n}.crop" if crop else f"flip_{n}")
    write_png(prefix + ".native.png", ow, oh, rows_n)
    write_png(prefix + ".vulkan.png", ow, oh, rows_v)
    write_png(prefix + ".diff.png", ow, oh, rows_d)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--threshold", type=int, default=8)
    ap.add_argument("--flips")
    ap.add_argument("--crop")
    args = ap.parse_args()
    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    wanted = {int(v) for v in args.flips.split(",")} if args.flips else None
    flips = sorted(int(m.group(1)) for f in os.listdir(args.dump_dir)
                   if (m := re.match(r"flip_(\d+)\.vulkan\.raw$", f)) and
                   os.path.exists(os.path.join(args.dump_dir, f"flip_{m.group(1)}.native.raw")))
    flips = [n for n in flips if wanted is None or n in wanted]
    if not flips:
        sys.exit("no flip_<n>.native.raw / flip_<n>.vulkan.raw pairs in " + args.dump_dir)
    out_dir = os.path.join(args.dump_dir, "diff")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for n in flips:
        r = compare(args.dump_dir, n, args.step, args.threshold, crop, out_dir)
        results.append(r)
        print(f"flip {n:6d}: {'IDENTICAL' if r['identical'] else 'differs'}  blocks {r['diff_blocks']}/{r['blocks']}  "
              f"mean {r['mean']:.2f}  max {r['max']}  >{args.threshold}: {r['pct']:.3f}%  box {r['box']}")
    first = next((r for r in results if r["pct"] > 0.5), None)
    if crop:
        return
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(f"# Native vs Vulkan frame comparison\n\n`{args.dump_dir}`, step {args.step}, threshold {args.threshold}\n\n")
        f.write(f"First flip with more than 0.5% differing pixels: {first['flip'] if first else 'none'}\n\n")
        f.write("| flip | identical | differing 64 KiB blocks | mean diff | max diff | pixels > threshold | diff box (x0,y0,x1,y1) |\n")
        f.write("|---:|:---:|---:|---:|---:|---:|---|\n")
        for r in results:
            f.write(f"| {r['flip']} | {'yes' if r['identical'] else 'no'} | {r['diff_blocks']}/{r['blocks']} | "
                    f"{r['mean']:.2f} | {r['max']} | {r['pct']:.3f}% | {r['box'] or '-'} |\n")
        f.write("\nPer flip: `flip_<n>.native.png`, `flip_<n>.vulkan.png`, `flip_<n>.diff.png` "
                "(differing pixels red, amplified 8x; the rest is the native frame dimmed).\n")
    print("report:", os.path.join(out_dir, "report.md"))


if __name__ == "__main__":
    main()
