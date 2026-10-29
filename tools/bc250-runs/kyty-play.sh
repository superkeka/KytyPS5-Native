#!/bin/bash
# Steam shortcut target: plays a PS5 title the way a user would. Steam is unloaded for the game's memory,
# the boot video plays, then the game with the in-game menu (PS button) and no performance overlay, the
# exit video when the user leaves from the menu, and Steam comes back on any exit (nosteam-run.sh).
# Usage (Steam launch options): kyty-play.sh TITLEID
# Log: /home/deck/dev/kyty-runs/play-TITLEID/run.log
set -u
TITLE=${1:?title id}
R=/home/deck/dev/kyty-runs
# A unit of its own: Steam is stopped by it, so it must not be in Steam's process tree. This script
# returns at once and Steam sees the shortcut as ended.
systemctl --user stop gamescope-mangoapp.service 2>/dev/null
systemctl --user reset-failed "kyty-play-$TITLE" 2>/dev/null
exec systemd-run --user --collect --quiet --unit="kyty-play-$TITLE" \
	-E KYTY_EMU="${KYTY_EMU:-/home/deck/dev/kyty-merge/_Build/linux-no-qt/kyty_emulator}" -E OVERLAY= \
	"$R/nosteam-run.sh" "$TITLE" primary 86400 "play-$TITLE" 3600
