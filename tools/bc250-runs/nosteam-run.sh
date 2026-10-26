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
systemctl --user mask --runtime steam-launcher.service
systemctl --user stop steam-launcher.service
/home/deck/dev/kyty-runs/game-run.sh "$@"
