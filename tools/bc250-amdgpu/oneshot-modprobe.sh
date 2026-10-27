#!/bin/bash
# Module options for one boot only, for trying a driver option that could hang the board at boot
# (SteamOS's grub ignores next_entry, so a one-shot boot entry does not work). amdgpu is loaded by udev
# from the root file system, after systemd starts, so a unit ordered before udev's coldplug moves
# /etc/bc250-oneshot.conf to /run/modprobe.d: the options apply to that boot, and the file is gone
# from the disk before amdgpu loads. A hang costs a power cycle, which boots without them.
# Run with sudo.
#   oneshot-modprobe.sh arm "options amdgpu bc250_gfx_queues=2"   install the unit if needed, arm one boot
#   oneshot-modprobe.sh disarm | uninstall
set -eu
UNIT=/etc/systemd/system/bc250-oneshot-modprobe.service
FLAG=/etc/bc250-oneshot.conf
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
case "${1:-}" in
arm)
	[ -n "${2:-}" ] || { echo "usage: $0 arm \"options <module> <option>=<value>...\""; exit 1; }
	if [ ! -f "$UNIT" ]; then
		cat > "$UNIT" <<'EOF'
[Unit]
Description=BC-250: module options for this boot only (/etc/bc250-oneshot.conf)
DefaultDependencies=no
Before=systemd-udev-trigger.service systemd-modules-load.service
ConditionPathExists=/etc/bc250-oneshot.conf

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'mkdir -p /run/modprobe.d && cp /etc/bc250-oneshot.conf /run/modprobe.d/bc250-oneshot.conf && rm -f /etc/bc250-oneshot.conf && sync'

[Install]
WantedBy=sysinit.target
EOF
		systemctl daemon-reload
		systemctl enable bc250-oneshot-modprobe.service
	fi
	printf '%s\n' "$2" > "$FLAG"
	sync
	echo "armed for the next boot: $(cat "$FLAG")"
	;;
disarm)
	rm -f "$FLAG"
	echo "disarmed"
	;;
uninstall)
	rm -f "$FLAG"
	systemctl disable bc250-oneshot-modprobe.service 2>/dev/null || true
	rm -f "$UNIT"
	systemctl daemon-reload
	echo "removed"
	;;
*)
	echo "usage: $0 arm \"options ...\" | disarm | uninstall"
	exit 1
	;;
esac
