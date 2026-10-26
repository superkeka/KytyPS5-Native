#!/bin/sh
# Without root: waits for TIMED OUT in the emulator log, then samples MEC state through kyty-lodhw.
# Usage: mec-watch.sh OUTFILE [RUNLOG]
OUT=$1; LOG=${2:-/home/deck/dev/kyty-runs/steam-PPSA24156/run.log}
H=/home/deck/dev/kyty-hw.py
start=$(date +%s)
while :; do
	if [ "$(stat -c %Y "$LOG" 2>/dev/null || echo 0)" -ge "$start" ] && grep -aq "TIMED OUT" "$LOG"; then break; fi
	sleep 1
done
{
	date +%T; grep -a "TIMED OUT" "$LOG" | tail -1
	for i in 1 2 3 4 5 6; do
		echo "-- sample $i"
		python3 $H read 0x2084 3     # CP_CPC_STATUS, CP_CPC_BUSY_STAT, CP_CPC_STALLED_STAT1
		python3 $H read 0x208a 1     # CP_CPC_BUSY_STAT2
		python3 $H read 0x208e 2     # CP_MEC_ME1_HEADER_DUMP, ME2
		python3 $H read 0x21a8 1     # CP_MEC1_INSTR_PNTR
		python3 $H read 0x2004 1     # GRBM_STATUS
		python3 $H read 0x2022 1     # GRBM_GFX_CNTL (current SRBM bank)
		python3 $H read 0x320b 3     # CP_HQD_ACTIVE, VMID (of that bank)
		python3 $H read 0x3211 3     # CP_HQD_PQ_BASE, _HI, PQ_RPTR
		python3 $H read 0x321b 3     # CP_HQD_IB_BASE_ADDR, _HI, IB_RPTR
		python3 $H read 0x323f 1     # CP_HQD_PQ_WPTR_LO
		sleep 0.3
	done
} > "$OUT" 2>&1
echo "captured $OUT"
