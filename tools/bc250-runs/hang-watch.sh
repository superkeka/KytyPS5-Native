#!/bin/sh
# Waits for a ring timeout in the emulator log, then captures what is running at once: GRBM/CP/MEC
# state (kyty-lodhw, no root) and the waves (umr with the navi10 register database, NOPASSWD).
# Usage: hang-watch.sh OUTDIR [RUNLOG]
OUT=$1; LOG=${2:-/home/deck/dev/kyty-runs/steam-PPSA21567/run.log}
H=/home/deck/dev/kyty-hw.py
mkdir -p "$OUT"
start=$(date +%s)
while :; do
	if [ "$(stat -c %Y "$LOG" 2>/dev/null || echo 0)" -ge "$start" ] && grep -aq "TIMED OUT" "$LOG"; then break; fi
	if ! pgrep -x kyty_emulator >/dev/null && [ $(( $(date +%s) - start )) -gt 120 ]; then exit 0; fi
	sleep 0.5
done
{
	date +%T
	timeout 5 python3 $H busy 1
	for i in 1 2 3 4; do python3 $H cp | grep -E "CP_STAT|HEADER"; python3 $H read 0x2084 3; python3 $H read 0x208e 2; done
} > "$OUT/cp.txt" 2>&1
timeout 120 sudo -n umr -f navi10 -O halt_waves -wa 2>/dev/null | grep -v "ring debugfs" > "$OUT/waves.txt"
journalctl -k -b --no-pager --since "-10min" | grep -A3 "page fault" > "$OUT/faults.txt"
echo "captured $OUT"
