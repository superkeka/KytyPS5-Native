#!/bin/bash
# Boots SteamOS once with extra kernel parameters, for trying a driver option that could hang the board
# at boot: a copy of the current SteamOS menu entry with the parameters appended goes into the ESP's
# custom.cfg (backed up first; its other entries stay), and grub's next_entry selects it for one boot
# only (grub clears next_entry before it loads the kernel). A hang costs a power cycle, which boots the
# normal entry. Run with sudo. Usage: oneshot-boot.sh "amdgpu.bc250_gfx_queues=2" [--no-reboot]
set -eu
PARAMS=$1
ESP=/efi/EFI/steamos
ID=bc250-oneshot
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
[ -f "$ESP/grub.cfg" ] || { echo "no $ESP/grub.cfg"; exit 1; }
[ -f "$ESP/custom.cfg" ] && [ ! -f "$ESP/custom.cfg.bak-before-$ID" ] && cp -p "$ESP/custom.cfg" "$ESP/custom.cfg.bak-before-$ID"
# The 'SteamOS' entry of grub.cfg, renamed, with an id and the parameters on its kernel line.
ENTRY=$(awk -v id="$ID" -v params="$PARAMS" '
	/^menuentry .SteamOS. / { on = 1; print "menuentry '\''SteamOS (one boot: " params ")'\'' --class steamos --id '\''" id "'\'' {"; next }
	on && /steamenv_boot[ \t]+linux / { print $0 " " params; next }
	on { print }
	on && /^}/ { exit }' "$ESP/grub.cfg")
echo "$ENTRY" | grep -q "steamenv_boot.*$PARAMS" || { echo "could not build the entry"; exit 1; }
# Drop an entry an earlier run left, then append this one.
if [ -f "$ESP/custom.cfg" ]; then
	awk -v id="$ID" '$0 ~ "--id .?" id { skip = 1 } !skip { print } skip && /^}/ { skip = 0 }' "$ESP/custom.cfg" > "$ESP/custom.cfg.new"
else
	: > "$ESP/custom.cfg.new"
fi
printf '%s\n' "$ENTRY" >> "$ESP/custom.cfg.new"
mv "$ESP/custom.cfg.new" "$ESP/custom.cfg"
echo "== $ESP/custom.cfg"; cat "$ESP/custom.cfg"
# next_entry in both places a grubenv can be read from: $prefix (the ESP directory) and /boot/grub.
grub-editenv "$ESP/grubenv" set next_entry="$ID" 2>/dev/null || grub-editenv "$ESP/grubenv" create && grub-editenv "$ESP/grubenv" set next_entry="$ID"
grub-reboot "$ID" 2>/dev/null || true # /boot/grub is read-only on SteamOS; the ESP grubenv is the one grub reads
echo "== next_entry: $(grub-editenv "$ESP/grubenv" list | grep next_entry) / $(grub-editenv list | grep next_entry)"
[ "${2:-}" = "--no-reboot" ] || systemctl reboot
