#!/bin/sh
# Run under sudo before a risky native GPU run. Waits for a gfx ring timeout in the kernel log
# (needs amdgpu.gpu_recovery=0 so the board survives it) and captures GPU state with umr.
# Usage: sudo hang-capture.sh OUTDIR
OUT=${1:-/home/deck/dev/kyty-runs/hang-$(date +%H%M%S)}
mkdir -p "$OUT"
base=$(dmesg | grep -c "ring gfx.*timeout")
echo "waiting for gfx ring timeout (seen so far: $base), output to $OUT"
while [ "$(dmesg | grep -c "ring gfx.*timeout")" -le "$base" ]; do sleep 1; done
echo "timeout detected, capturing"
sleep 2
dmesg | tail -n 200 > "$OUT/dmesg.txt"
umr -go 0 > "$OUT/gfxoff.txt" 2>&1
for r in GRBM_STATUS GRBM_STATUS2 GRBM_STATUS3 GRBM_STATUS_SE0 GRBM_STATUS_SE1 GRBM_STATUS_SE2 GRBM_STATUS_SE3 \
	CP_STAT CP_BUSY_STAT CP_STALLED_STAT1 CP_STALLED_STAT2 CP_STALLED_STAT3 CP_CPF_STATUS CP_CPF_BUSY_STAT \
	CP_CPF_STALLED_STAT1 CP_CPC_STATUS CP_ME_CNTL CP_IB1_BASE_LO CP_IB1_BASE_HI CP_IB1_BUFSZ CP_IB2_BASE_LO \
	CP_IB2_BASE_HI CP_IB2_BUFSZ CP_RB0_RPTR CP_RB0_WPTR CP_ME_HEADER_DUMP CP_PFP_HEADER_DUMP \
	SPI_DEBUG_BUSY GE_STATUS VGT_STATUS SQ_DEBUG_STS_GLOBAL DB_DEBUG RLC_STAT CB_HW_CONTROL; do
	echo "== $r"; umr -O bits -r "*.*.mm$r" 2>&1 | head -40
done > "$OUT/regs.txt" 2>&1
for i in 1 2 3 4 5 6 7 8; do umr -r "*.*.mmCP_ME_HEADER_DUMP" 2>&1; umr -r "*.*.mmCP_PFP_HEADER_DUMP" 2>&1; done > "$OUT/header_dumps.txt"
timeout 60 umr -O halt_waves -wa gfx_0.0.0 > "$OUT/waves.txt" 2>&1
timeout 60 umr -RS gfx_0.0.0 > "$OUT/ring.txt" 2>&1
cat /sys/class/drm/card*/device/devcoredump/data > "$OUT/devcoredump.txt" 2>&1
chown -R deck:deck "$OUT"
echo "captured to $OUT"; ls -l "$OUT"
