#!/bin/sh
# Waits for the CP's "Illegal register access in command stream" (or a gfx ring timeout) in the kernel
# log, then captures the CP state at once: the PFP/ME packet header dumps, CP_STAT and the IB
# addresses (kyty-lodhw, no root), twice, plus the devcoredump if it is readable.
# Usage: illegal-watch.sh OUTDIR
OUT=$1
H=/home/deck/dev/kyty-hw.py
mkdir -p "$OUT"
journalctl -kf -n 0 --no-pager | grep -m1 -E "Illegal register access|ring gfx.*timeout" > "$OUT/trigger.txt"
{
	date +%T.%N
	for i in 1 2; do
		python3 $H cp
		python3 $H read 0x2084 3 # the IB registers hang-watch.sh reads
		python3 $H read 0x208e 2
	done
} > "$OUT/cp.txt" 2>&1
cp /sys/class/drm/card0/device/devcoredump/data "$OUT/devcoredump.txt" 2>/dev/null
