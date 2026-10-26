#!/bin/sh
# KYTY_NATIVE_GPU_HANG_HOOK target: captures the GPU as it hangs, before the emulator releases anything:
# GRBM/CP/MEC state through kyty-lodhw and the waves through umr (NOPASSWD). Usage: hang-capture-now.sh OUTDIR
OUT=${1:-/home/deck/dev/kyty-runs/hang-now}
H=/home/deck/dev/kyty-hw.py
mkdir -p "$OUT"
{
	date +%T
	timeout 5 python3 $H busy 1
	for i in 1 2 3 4; do python3 $H cp | grep -E "CP_STAT|HEADER"; python3 $H read 0x2084 3; python3 $H read 0x208e 2; done
} > "$OUT/cp.txt" 2>&1
timeout 120 sudo -n umr -f navi10 -O halt_waves -wa 2>/dev/null | grep -v "ring debugfs" > "$OUT/waves.txt"
journalctl -k -b --no-pager --since "-10min" | grep -A3 "page fault" > "$OUT/faults.txt"
