#!/bin/bash
# Runs game-run.sh with the Steam client stopped: Steam and its web helpers hold 2-3 GB of RAM that a
# PS5 title with full residency needs. gamescope-session.target upholds steam-launcher.service, so the
# unit is masked for the run (runtime only: gone after a reboot) and unmasked and started again after.
# Usage: nosteam-run.sh <game-run.sh arguments>
restore() {
	systemctl --user unmask --runtime steam-launcher.service
	systemctl --user start steam-launcher.service
}
trap restore EXIT
# Steam stopped while it is still starting (a previous run just brought it back) hangs in its stop:
# let it have been up for 20 s first.
started=$(systemctl --user show steam-launcher.service -p ActiveEnterTimestampMonotonic --value)
if [ "$(systemctl --user is-active steam-launcher.service)" = active ] && [ "${started:-0}" -gt 0 ]; then
	up=$(( $(awk '{printf "%d", $1 * 1000000}' /proc/uptime) - started ))
	[ "$up" -lt 20000000 ] && sleep $(( (20000000 - up) / 1000000 + 1 ))
fi
systemctl --user mask --runtime steam-launcher.service
systemctl --user stop steam-launcher.service
/home/deck/dev/kyty-runs/game-run.sh "$@"
