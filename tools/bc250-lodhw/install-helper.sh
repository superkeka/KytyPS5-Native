#!/bin/sh
# Installs kyty-lodhw as a system service: sudo ./install-helper.sh (from the release folder).
# The socket /run/kyty-lodhw.sock belongs to the group of the user who ran sudo.
# Remove: sudo ./install-helper.sh --remove
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
unit=/etc/systemd/system/kyty-lodhw.service
# SteamOS keeps /usr read-only and wipes it on updates; /var/usrlocal/bin survives.
bin=/usr/local/bin
[ -d /var/usrlocal ] && bin=/var/usrlocal/bin
if [ "${1:-}" = "--remove" ]; then
	systemctl disable --now kyty-lodhw.service 2>/dev/null || true
	rm -f "$unit" "$bin/kyty-lodhw"
	systemctl daemon-reload
	exit 0
fi
dir=$(dirname "$(readlink -f "$0")")
install -Dm755 "$dir/kyty-lodhw" "$bin/kyty-lodhw"
sed "s|@GID@|${SUDO_GID:-$(id -g)}|; s|@BIN@|$bin|" "$dir/kyty-lodhw.service" > "$unit"
systemctl daemon-reload
systemctl enable --now kyty-lodhw.service
systemctl --no-pager --lines=0 status kyty-lodhw.service
