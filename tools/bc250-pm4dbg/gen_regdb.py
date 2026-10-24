#!/usr/bin/env python3
# Builds regdb.json, the register names and fields pm4dbg.py decodes with, from the amdgpu headers
# gc_10_1_0_offset.h and gc_10_1_0_sh_mask.h (Linux drivers/gpu/drm/amd/include/asic_reg/gc, MIT).
# Only the three spaces PM4 SET_* packets address are kept, as dword offsets within the space:
#   cx  context registers  (SET_CONTEXT_REG,  absolute 0xA000-0xBFFF)
#   sh  SH registers       (SET_SH_REG,       absolute 0x2C00-0x2FFF)
#   uc  UCONFIG registers  (SET_UCONFIG_REG,  absolute 0xC000-0xFFFF)
# Usage: gen_regdb.py HEADER_DIR > regdb.json

import json
import re
import sys

# GC_BASE segments of gfx10.1 (navi10_ip_offset.h): BASE_IDX 0 and 1.
SEGMENTS = {0: 0x1260, 1: 0xA000}
SPACES = {"cx": (0xA000, 0xC000), "sh": (0x2C00, 0x3000), "uc": (0xC000, 0x10000)}


def main():
    d = sys.argv[1]
    offs, idx = {}, {}
    for line in open(f"{d}/gc_10_1_0_offset.h"):
        m = re.match(r"#define mm(\w+?)(_BASE_IDX)?\s+(0x[0-9a-fA-F]+|\d+)", line)
        if not m:
            continue
        if m.group(2):
            idx[m.group(1)] = int(m.group(3), 0)
        else:
            offs[m.group(1)] = int(m.group(3), 0)
    fields = {}
    for line in open(f"{d}/gc_10_1_0_sh_mask.h"):
        # REG__FIELD__SHIFT and REG__FIELD_MASK
        m = re.match(r"#define (\w+?)__(\w+?)(__SHIFT|_MASK)\s+(0x[0-9a-fA-F]+|\d+)", line)
        if not m:
            continue
        f = fields.setdefault(m.group(1), {}).setdefault(m.group(2), [0, 0])
        f[0 if m.group(3) == "__SHIFT" else 1] = int(m.group(4), 0)
    db = {k: {} for k in SPACES}
    for name, off in offs.items():
        if name not in idx or idx[name] not in SEGMENTS:
            continue
        absolute = off + SEGMENTS[idx[name]]
        for space, (lo, hi) in SPACES.items():
            if lo <= absolute < hi:
                key = f"0x{absolute - lo:03x}"
                regfields = []
                for fname, (shift, mask) in sorted(fields.get(name, {}).items(), key=lambda kv: kv[1][0]):
                    if mask == 0:
                        continue
                    regfields.append([fname, shift, bin(mask).count("1")])
                entry = db[space].get(key)
                if entry is None:
                    db[space][key] = {"name": name, "fields": regfields}
                else:
                    entry.setdefault("aliases", []).append(name)
                    if not entry["fields"] and regfields:
                        entry["fields"] = regfields
    json.dump(db, sys.stdout, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
