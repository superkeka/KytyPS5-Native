#!/usr/bin/env python3
# Copies AMD GPU microcode files with the PSP signature wrapper cut off, as ps5-linux loads them directly
# (gfx_v10_0_patch_ucode: ucode_array_offset_bytes += 0x100, ucode_size_bytes -= 0x200). Only the two header
# fields change; size_bytes, the only field amdgpu_ucode_validate checks, stays the file size.
# fwstrip.py SRC_DIR DST_DIR NAME...   (SRC_DIR holds NAME.bin)
import os
import struct
import sys

src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(dst, exist_ok=True)
for name in names:
    data = bytearray(open(os.path.join(src, name + ".bin"), "rb").read())
    size_bytes = struct.unpack_from("<I", data, 0)[0]
    usize, uoff = struct.unpack_from("<II", data, 0x14)
    if size_bytes != len(data) or uoff != 0x100 or uoff + usize != len(data):
        sys.exit("%s: unexpected header (size %d, ucode 0x%x at 0x%x)" % (name, size_bytes, usize, uoff))
    struct.pack_into("<II", data, 0x14, usize - 0x200, uoff + 0x100)
    open(os.path.join(dst, name + ".bin"), "wb").write(data)
    print("%s: ucode 0x%x at 0x%x -> 0x%x at 0x%x" % (name, usize, uoff, usize - 0x200, uoff + 0x100))
