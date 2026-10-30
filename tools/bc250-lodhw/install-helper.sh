#!/bin/sh
# Installs kyty-lodhw as a system service: sudo ./install-helper.sh (from the release folder).
# The socket /run/kyty-lodhw.sock belongs to the group of the user who ran sudo.
# Remove: sudo ./install-helper.sh --remove
set -eu
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
unit=/etc/systemd/system/kyty-lodhw.service
# SteamOS keeps /usr read-only and wipes it on updates: the helper goes under /var there
# (/var/usrlocal/bin where it exists, as on Bazzite).
bin=/usr/local/bin
if [ -d /var/usrlocal ]; then
	bin=/var/usrlocal/bin
elif command -v steamos-readonly >/dev/null; then
	bin=/var/lib/kyty-lodhw
fi
if [ "${1:-}" = "--remove" ]; then
	systemctl disable --now kyty-lodhw.service 2>/dev/null || true
	rm -f "$unit" "$bin/kyty-lodhw"
	systemctl daemon-reload
	exit 0
fi
# kyty-lodhw and its unit from KYTY_HELPER_FILES (the Decky plugin runs its own copy of this script).
dir=${KYTY_HELPER_FILES:-$(dirname "$(readlink -f "$0")")}
install -Dm755 "$dir/kyty-lodhw" "$bin/kyty-lodhw"
sed "s|@GID@|${SUDO_GID:-$(id -g)}|; s|@BIN@|$bin|" "$dir/kyty-lodhw.service" > "$unit"
systemctl daemon-reload
systemctl enable --now kyty-lodhw.service
systemctl --no-pager --lines=0 status kyty-lodhw.service
