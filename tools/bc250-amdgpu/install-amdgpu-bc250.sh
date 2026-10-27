#!/bin/bash
# Installs the amdgpu module build-amdgpu-bc250.sh made (bc250-amdgpu.patch: PASID flushes through
# MMIO, runlist rebuild on KFD map/unmap under HWS, the KFD VMID flushed directly without it, legacy TLB
# flushes, wave save areas for all 40 CUs) next to the stock one, for the running SteamOS kernel only.
# It sets amdgpu.sched_policy=2: KFD's queues sit on their hardware queues for good, as the PS5's
# compute queues do.
# The stock module stays in place; uninstall-amdgpu-bc250.sh removes this one. Run with sudo.
# A SteamOS update replaces /usr, which drops this module as well (the board is then stock again).
set -eu
KVER=$(uname -r)
SRC=${AMDGPU_BC250_MODULE:-/home/deck/dev/amdgpu-bc250/amdgpu.ko.zst}
DST=/usr/lib/modules/$KVER/updates/amdgpu.ko.zst
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
modinfo -F vermagic "$SRC" | grep -q "^$KVER " || { echo "module is not for $KVER"; exit 1; }
steamos-readonly disable
install -D -m 644 "$SRC" "$DST"
cat > /etc/modprobe.d/bc250-amdgpu.conf <<'EOF'
# BC-250 TLB flush fixes (patched amdgpu in updates/, see /home/deck/dev/amdgpu-bc250)
options amdgpu sched_policy=2 bc250_tlb_flush_type=0 bc250_gfx_queues=2 vm_update_mode=0 bc250_flush_pasid_kiq=0 bc250_flush_by_runlist=3 gpu_recovery=0
EOF
depmod "$KVER"
modinfo -n amdgpu | grep -q /updates/ || { echo "depmod does not pick the patched module"; exit 1; }
mkinitcpio -P || echo "mkinitcpio reported errors (the stock steam-deck hook asks for blake2b_generic, built into 7.2): image built anyway"
steamos-readonly enable
echo "installed: $DST; reboot, then check: cat /sys/module/amdgpu/parameters/bc250_flush_pasid_kiq (0)"
