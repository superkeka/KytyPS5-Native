#!/bin/bash
# Firmware files and module options for one boot only, for trying GPU firmware that could keep the GPU from
# starting, e.g. microcode loaded directly (amdgpu.fw_load_type=0) with its PSP wrapper cut off, as ps5-linux
# loads it. amdgpu is loaded by udev from the root file system after systemd starts, so a unit ordered before
# udev's coldplug first renames /etc/bc250-oneshot-fw on the disk (no later boot sees it, whatever happens next),
# copies it to /run, points the firmware loader at /run/bc250-oneshot-fw/firmware (firmware_class.path is searched
# before /lib/firmware) and puts its modprobe.conf into /run/modprobe.d, then deletes the renamed copy. A hang costs
# a power cycle, which boots with the system's firmware and options. Run with sudo.
#   oneshot-firmware.sh arm DIR ["options amdgpu fw_load_type=0"]   DIR/firmware/amdgpu/*.bin
#   oneshot-firmware.sh disarm | uninstall
set -eu
UNIT=/etc/systemd/system/bc250-oneshot-firmware.service
RUNNER=/etc/bc250-oneshot-firmware.run
STAGE=/etc/bc250-oneshot-fw
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
case "${1:-}" in
arm)
	SRC=${2:-}
	[ -d "$SRC/firmware" ] || { echo "usage: $0 arm DIR [\"options ...\"] (DIR/firmware/ is missing)"; exit 1; }
	cat > "$RUNNER" <<'EOF'
#!/bin/sh
# bc250-oneshot-firmware.service: one boot with the staged firmware and options (oneshot-firmware.sh).
set -e
mv /etc/bc250-oneshot-fw /etc/bc250-oneshot-fw.used
sync
rm -rf /run/bc250-oneshot-fw
cp -a /etc/bc250-oneshot-fw.used /run/bc250-oneshot-fw
rm -rf /etc/bc250-oneshot-fw.used
sync
if [ -f /run/bc250-oneshot-fw/modprobe.conf ]; then
	mkdir -p /run/modprobe.d
	cp /run/bc250-oneshot-fw/modprobe.conf /run/modprobe.d/bc250-oneshot-fw.conf
fi
printf '%s' /run/bc250-oneshot-fw/firmware > /sys/module/firmware_class/parameters/path
EOF
	chmod 755 "$RUNNER"
	if [ ! -f "$UNIT" ]; then
		cat > "$UNIT" <<'EOF'
[Unit]
Description=BC-250: firmware and module options for this boot only (/etc/bc250-oneshot-fw)
DefaultDependencies=no
Before=systemd-udev-trigger.service systemd-modules-load.service
ConditionPathExists=/etc/bc250-oneshot-fw

[Service]
Type=oneshot
ExecStart=/bin/sh /etc/bc250-oneshot-firmware.run

[Install]
WantedBy=sysinit.target
EOF
		systemctl daemon-reload
		systemctl enable bc250-oneshot-firmware.service
	fi
	rm -rf "$STAGE.new" "$STAGE.used"
	cp -a "$SRC" "$STAGE.new"
	if [ -n "${3:-}" ]; then
		printf '%s\n' "$3" > "$STAGE.new/modprobe.conf"
	fi
	rm -rf "$STAGE"
	mv "$STAGE.new" "$STAGE"
	sync
	echo "armed for the next boot: $(find "$STAGE/firmware" -type f | wc -l) firmware files$( [ -f "$STAGE/modprobe.conf" ] && printf ', %s' "$(cat "$STAGE/modprobe.conf")")"
	;;
disarm)
	rm -rf "$STAGE" "$STAGE.new" "$STAGE.used"
	echo "disarmed"
	;;
uninstall)
	rm -rf "$STAGE" "$STAGE.new" "$STAGE.used"
	systemctl disable bc250-oneshot-firmware.service 2>/dev/null || true
	rm -f "$UNIT" "$RUNNER"
	systemctl daemon-reload
	echo "removed"
	;;
*)
	echo "usage: $0 arm DIR [\"options ...\"] | disarm | uninstall"
	exit 1
	;;
esac
