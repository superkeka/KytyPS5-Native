#!/bin/bash
# Tries the patched amdgpu's second graphics ring (bc250_gfx_queues=2: gfx pipe 0 queue 1, the PS5's
# "AGR" ring) by reloading the module on the running system, so a failure costs a reboot and nothing
# else: /etc/modprobe.d does not set the parameter, and a reboot loads the stock configuration again.
# Run with sudo over SSH. Prints the rings amdgpu brought up; bc250-cu-live-manager and bc250-smu-oc
# run again afterwards (a module reload drops what they set).
set -u
QUEUES=${1:-2}
USER_NAME=${SUDO_USER:-deck}
USER_ID=$(id -u "$USER_NAME")
as_user() { sudo -u "$USER_NAME" XDG_RUNTIME_DIR=/run/user/$USER_ID DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$USER_ID/bus "$@"; }
# GPU users: the display manager, the gamescope session (a user unit, it outlives sddm) and running
# podman containers (llama-server); the containers that ran come back afterwards.
CONTAINERS=$(as_user podman ps --format '{{.Names}}' 2>/dev/null)
restore() {
	for v in /sys/class/vtconsole/vtcon*; do echo 1 > "$v/bind" 2>/dev/null; done
	systemctl restart bc250-cu-live-manager.service bc250-smu-oc.service 2>/dev/null
	systemctl start kyty-lodhw.service
	systemctl isolate graphical.target
	for c in $CONTAINERS; do as_user podman start "$c" >/dev/null; done
}
systemctl isolate multi-user.target
systemctl stop kyty-lodhw.service # after the isolate, which starts it (multi-user.target wants it)
as_user systemctl --user stop gamescope-session.target gamescope-session.service steam-launcher.service gamescope-mangoapp.service
for c in $CONTAINERS; do as_user podman stop -t 10 "$c" >/dev/null; done
sleep 3
# The kernel console sits on amdgpu's framebuffer (fbcon) and holds the module too.
for v in /sys/class/vtconsole/vtcon*; do
	grep -q "frame buffer" "$v/name" && echo 0 > "$v/bind"
done
if ! modprobe -r amdgpu; then
	echo "amdgpu is still in use (refcnt $(cat /sys/module/amdgpu/refcnt)):"
	fuser -v /dev/dri/* /dev/kfd 2>&1 | head -20
	lsof /dev/dri/* /dev/kfd /sys/kernel/debug/dri/*/* 2>/dev/null | head -20
	restore
	exit 1
fi
modprobe amdgpu bc250_gfx_queues="$QUEUES"
sleep 2
echo "bc250_gfx_queues=$(cat /sys/module/amdgpu/parameters/bc250_gfx_queues)"
dmesg | grep -E "ring gfx_|gfx_0\.0\.[01]|ring test|amdgpu.*(fail|error)" | tail -15
ls /sys/kernel/debug/dri/*/ 2>/dev/null | grep -E "^amdgpu_ring_gfx" | sort -u
restore
