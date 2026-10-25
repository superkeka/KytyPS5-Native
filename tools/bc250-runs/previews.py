import sys, os, glob
sys.path.insert(0, "/home/deck/dev/KytyPS5-bc250/tools/bc250-frame-diff")
import frame_diff as fd
d = sys.argv[1]; out = os.path.join(d, "prev"); os.makedirs(out, exist_ok=True)
step = 8
for meta_path in sorted(p for p in glob.glob(os.path.join(d, "flip_*.txt")) if p.count(".") == 1):
    n = meta_path.rsplit("_", 1)[1].split(".")[0]
    meta = fd.read_meta(meta_path)
    data = open(os.path.join(d, f"flip_{n}.native.raw"), "rb").read()
    rows = []
    for row in fd.pixel_offsets(meta, step, None):
        r = bytearray()
        for o in row:
            v = int.from_bytes(data[o:o + 4], "little")
            r += bytes((((v >> 20) & 0x3ff) >> 2, ((v >> 10) & 0x3ff) >> 2, ((v >> 0) & 0x3ff) >> 2))
        rows.append(r)
    fd.write_png(os.path.join(out, f"{n}.png"), len(rows[0]) // 3, len(rows), rows)
print("ok")
