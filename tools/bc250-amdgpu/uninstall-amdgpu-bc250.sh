#!/bin/bash
# Removes the patched amdgpu module install-amdgpu-bc250.sh put in, back to the stock driver. Run with sudo.
set -eu
KVER=$(uname -r)
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
steamos-readonly disable
rm -f "/usr/lib/modules/$KVER/updates/amdgpu.ko.zst" /etc/modprobe.d/bc250-amdgpu.conf
depmod "$KVER"
mkinitcpio -P || echo "mkinitcpio reported errors (the stock steam-deck hook asks for blake2b_generic, built into 7.2): image built anyway"
steamos-readonly enable
echo "stock amdgpu restored ($(modinfo -n amdgpu)); reboot"
