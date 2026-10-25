#!/bin/sh
# Root half of lodhw_probe: one PRT LOD statistics report into the probe's buffer, the way the PS5
# kernel's gc_stop_lod_report asks for one (CNTL3 = REPORT_AND_RESET | CACHE_POLICY | bit 29, with the
# VMID of the probe's VM). Registers by absolute dword address through debugfs amdgpu_regs.
# Run while lodhw_probe waits: sudo sh lodhw-root.sh
set -u
R=$(ls /sys/kernel/debug/dri/*/amdgpu_regs | head -1)
rd() { dd if=$R bs=4 skip=$(($1)) count=1 2>/dev/null | od -An -tx4 | tr -d ' \n'; }
wr() { printf "$(printf '\\%03o\\%03o\\%03o\\%03o' $(($2 & 255)) $((($2 >> 8) & 255)) $((($2 >> 16) & 255)) $((($2 >> 24) & 255)))" |
	dd of=$R bs=4 seek=$(($1)) count=1 conv=notrunc 2>/dev/null; }
CNTL0=0x20ad CNTL1=0x20ae CNTL2=0x20af CNTL3=0x20b0
CPF_STATUS=$((0x1260 + 0x0e27))
SPM_MC_CNTL=$((0xA000 + 0x4c71))

VA=$(cat /tmp/lodhw.va) || { echo "no /tmp/lodhw.va: start lodhw_probe first"; exit 1; }
SPM=0x$(rd $SPM_MC_CNTL)
VMID=$((SPM & 0xf))
echo "RLC_SPM_MC_CNTL $SPM -> VMID $VMID"
[ "$VMID" = 0 ] || [ "$VMID" = 15 ] && { echo "VMID $VMID looks like no reserved VMID was flushed; stopping"; exit 1; }
echo "before: CNTL0..3 $(rd $CNTL0) $(rd $CNTL1) $(rd $CNTL2) $(rd $CNTL3), CP_CPF_STATUS $(rd $CPF_STATUS)"

wr $CNTL0 0x1040
wr $CNTL1 $(((VA >> 6) & 0xffffffff))
wr $CNTL2 $(((VA >> 38) & 0x3ff))
wr $CNTL3 $((0x30080000 | (VMID << 23)))
echo "written: CNTL0..3 $(rd $CNTL0) $(rd $CNTL1) $(rd $CNTL2) $(rd $CNTL3)"
i=0
while [ $i -lt 50 ]; do
	c3=0x$(rd $CNTL3)
	[ $((c3 & 0x80000)) = 0 ] && break
	sleep 0.02; i=$((i + 1))
done
echo "after $i polls: CNTL3 $(rd $CNTL3) (REPORT_AND_RESET bit 19 should clear), CP_CPF_STATUS $(rd $CPF_STATUS) (PRT_BUSY bit 16)"
sleep 0.5
wr $CNTL3 0; wr $CNTL2 0; wr $CNTL1 0; wr $CNTL0 0
echo "cleared: CNTL0..3 $(rd $CNTL0) $(rd $CNTL1) $(rd $CNTL2) $(rd $CNTL3)"
