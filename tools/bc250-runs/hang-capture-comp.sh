#!/bin/sh
# Run under sudo before a native GPU run that may hang a compute ring. Waits until the emulator's log
# reports a TIMED OUT submission, then captures the CP/MEC state with umr.
# Usage: sudo hang-capture-comp.sh OUTDIR [RUNLOG]
OUT=${1:-/home/deck/dev/kyty-runs/hang-$(date +%H%M%S)}
LOG=${2:-/home/deck/dev/kyty-runs/steam-PPSA24156/run.log}
mkdir -p "$OUT"
start=$(date +%s)
echo "waiting for TIMED OUT in $LOG (newer than now), output to $OUT"
while :; do
	if [ "$(stat -c %Y "$LOG" 2>/dev/null || echo 0)" -ge "$start" ] && grep -aq "TIMED OUT" "$LOG"; then break; fi
	sleep 1
done
echo "timeout detected, capturing"
dmesg | tail -n 100 > "$OUT/dmesg.txt"
umr -go 0 > "$OUT/gfxoff.txt" 2>&1
for r in GRBM_STATUS GRBM_STATUS2 CP_STAT CP_CPC_STATUS CP_CPC_BUSY_STAT CP_CPC_STALLED_STAT1 CP_CPF_STATUS \
	CP_CPF_BUSY_STAT CP_CPF_STALLED_STAT1 CP_BUSY_STAT CP_STALLED_STAT1 CP_STALLED_STAT2 CP_STALLED_STAT3 \
	CP_IB1_BASE_LO CP_IB1_BASE_HI CP_RB0_RPTR CP_RB0_WPTR SPI_DEBUG_BUSY GDS_CS_CTXSW_STATUS; do
	echo "== $r"; umr -O bits -r "*.*.mm$r" 2>&1 | head -40
done > "$OUT/regs.txt" 2>&1
for i in 1 2 3 4 5 6 7 8; do umr -r "*.*.mmCP_MEC_ME1_HEADER_DUMP" 2>&1; umr -r "*.*.mmCP_MEC_ME2_HEADER_DUMP" 2>&1; done > "$OUT/mec_header_dumps.txt"
timeout 60 umr -cpc > "$OUT/cpc.txt" 2>&1
for pipe in 0 1 2 3; do for q in 0 1 2 3 4 5 6 7; do
	echo "== me1 pipe $pipe queue $q"
	for r in CP_HQD_ACTIVE CP_HQD_VMID CP_HQD_PQ_BASE CP_HQD_PQ_BASE_HI CP_HQD_PQ_RPTR CP_HQD_PQ_WPTR_LO \
		CP_HQD_IB_BASE_ADDR CP_HQD_IB_BASE_ADDR_HI CP_HQD_IB_RPTR CP_HQD_IB_CONTROL CP_HQD_DEQUEUE_REQUEST; do
		umr -sb 1 $pipe $q -r "*.*.mm$r" 2>&1 | head -2
	done
done; done > "$OUT/hqd.txt" 2>&1
for ring in $(ls /sys/kernel/debug/dri/*/ 2>/dev/null | grep -o "amdgpu_ring_comp_[0-9.]*" | sed "s/amdgpu_ring_//" | sort -u); do
	echo "== $ring"; timeout 30 umr -RS "$ring[.]" 2>&1 | head -300
done > "$OUT/rings.txt" 2>&1
timeout 60 umr -O halt_waves -wa > "$OUT/waves.txt" 2>&1
chown -R deck:deck "$OUT"
echo "captured to $OUT"; ls -l "$OUT"
