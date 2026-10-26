#!/bin/bash
# Reproducible ASTRO BOT white-level run for A/B: replays white.rec (launch -> the white level) through a
# virtual 8BitDo from the moment the emulator starts, stays on the level 70 s more, and averages the
# native per-flip stats of that idle stretch. Unplug the real pad first.
# Usage: white-bench.sh TAG [ENV=VALUE ...]
set -u
TAG=$1; shift
R=/home/deck/dev/kyty-runs
rm -rf $R/$TAG
systemd-run --user --collect --unit=bench-$TAG -E EXTRA_ENV="KYTY_NATIVE_GPU_FLIP_STATS=1 KYTY_NATIVE_GPU_MAP_RESERVE_MIB=256 KYTY_NATIVE_GPU_BLOCK_MIB=16 KYTY_LOD_HW_SOCK=/tmp/kyty-lodhw.sock $*" \
	$R/nosteam-run.sh PPSA21567 primary 480 $TAG 30 >/dev/null
for i in $(seq 1 400); do pgrep -x kyty_emulator >/dev/null && break; sleep 0.25; done
python3 $R/padplay.py $R/white.rec $R/8bitdo.json 0
mark=$(wc -l < $R/$TAG/run.log)
sleep 70
python3 - $R/$TAG/run.log "$mark" <<"PY"
import re, sys
lines = open(sys.argv[1], errors="ignore").read().splitlines()[int(sys.argv[2]):]
def avg(pat):
    v = [float(m.group(1)) for l in lines for m in [re.search(pat, l)] if m]
    return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
for name, pat in [("interval", r"per flip: interval ([0-9.]+) ms"), ("GPU busy", r"GPU busy ([0-9.]+) ms"),
                  ("translate", r"translate ([0-9.]+) ms"), ("queued", r"queued ([0-9.]+) ms"),
                  ("kdwords", r"of ([0-9]+) dwords per flip")]:
    a, n = avg(pat)
    print(f"{name} {a/1000 if name == kdwords else a:.1f} (n={n})")
print("timeouts", sum("TIMED OUT" in l or "disabled" in l for l in lines))
PY
systemctl --user stop bench-$TAG
