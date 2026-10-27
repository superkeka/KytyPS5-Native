#!/bin/bash
# Runs scanout_probe inside the gamescope session the way game-run.sh runs Kyty: Xwayland :1 at the
# screen's mode, the window tagged STEAM_GAME and made the base layer, the desktop restored after.
# Usage: scanout-probe.sh [vram|gtt] [seconds]
HERE=$(dirname "$(realpath "$0")")
SCREEN=$(for st in /sys/class/drm/card*-*/status; do [ "$(cat "$st")" = connected ] && head -1 "$(dirname "$st")/modes" && break; done)
SCREEN=${SCREEN:-3840x2160}
DISPLAY=:0 xprop -root -f GAMESCOPE_XWAYLAND_MODE_CONTROL 32c -set GAMESCOPE_XWAYLAND_MODE_CONTROL "1,${SCREEN%x*},${SCREEN#*x},1"
restore() { DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 769; }
trap restore EXIT
(sleep 2; DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242) &
DISPLAY=:1 "$HERE/scanout_probe" "${1:-vram}" "${2:-20}"
