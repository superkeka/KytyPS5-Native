#!/bin/bash
# Runs a PS5 dump in Kyty on the TV with periodic screenshots, detached from the caller.
# Usage: game-run.sh TITLEID MODE SECONDS TAG [SHOT_EVERY]
#   MODE: vulkan (stock Kyty), shadow (native next to Vulkan), primary (fully native)
# Output in /home/deck/dev/kyty-runs/TAG: run.log, shot-<t>.png, fps.csv. Restores the desktop at the end.
set -u
TITLE=$1 MODE=$2 SECS=$3 TAG=$4 EVERY=${5:-15}
OUT=/home/deck/dev/kyty-runs/$TAG
EMU=${KYTY_EMU:-/home/deck/dev/KytyPS5-bc250/_Build/linux-no-qt/kyty_emulator}
mkdir -p "$OUT"
# Saves, trophies and download data (GT7 keeps its progress there) live in one place for every run
# (Kyty writes them into its working directory),
# so progress carries over between runs. KYTY_FRESH_SAVES=1 starts this run from an empty save.
SAVES=${KYTY_SAVES:-/home/deck/kyty-saves}
if [ "${KYTY_FRESH_SAVES:-0}" != 1 ]; then
  mkdir -p "$SAVES/_SaveData" "$SAVES/_Trophies" "$SAVES/_DownloadData"
  for d in _SaveData _Trophies _DownloadData; do
    # A tag reused from an old run may hold a directory of its own: kept aside, never deleted.
    [ -d "$OUT/$d" ] && [ ! -L "$OUT/$d" ] && mv "$OUT/$d" "$OUT/$d.old-$(date +%s)"
    [ -e "$OUT/$d" ] || ln -s "$SAVES/$d" "$OUT/$d"
  done
fi
case $MODE in
  vulkan)  ENVS="" ;;
  shadow)  ENVS="KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_SHADOW=1" ;;
  primary) ENVS="KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_PRIMARY=1" ;;
  *) echo "bad mode"; exit 1 ;;
esac
# The window is fullscreen at the connected display's native mode (KYTY_SCREEN=WxH overrides it), so
# the presenter's swapchain matches the screen instead of gamescope scaling up a 1280x720 window.
SCREEN=${KYTY_SCREEN:-$(for st in /sys/class/drm/card*-*/status; do
  [ "$(cat "$st")" = connected ] && head -1 "$(dirname "$st")/modes" && break; done)}
SCREEN=${SCREEN:-3840x2160}
# gamescope sizes the games' Xwayland per app when Steam starts a game; without Steam it stays at
# 1280x800 and the window is letterboxed. Set it the way Steam does.
DISPLAY=:0 xprop -root -f GAMESCOPE_XWAYLAND_MODE_CONTROL 32c -set GAMESCOPE_XWAYLAND_MODE_CONTROL "1,${SCREEN%x*},${SCREEN#*x},1"
restore() { DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 769; }
trap restore EXIT
# A run stopped from outside can leave the container half torn down ("creating temporary passwd file
# ... no such file or directory"); a clean stop and start brings it back.
distrobox enter kyty-dev -- true >/dev/null 2>&1 || { podman stop -t 5 kyty-dev; podman start kyty-dev; } >/dev/null 2>&1
(cd "$OUT" && distrobox enter kyty-dev -- env DISPLAY=:1 SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1 ${EXTRA_ENV:-} $ENVS timeout -s INT "$SECS" "$EMU" --game /home/deck/Games/$TITLE --screen-width ${SCREEN%x*} --screen-height ${SCREEN#*x} --fullscreen --no-splash --skip-notice-screen ${OVERLAY---perf-overlay} ${EXTRA_ARGS:-} > run.log 2>&1; echo "exit=$?" >> run.log) &
EMUPID=$!
WID=""
for i in $(seq 1 90); do sleep 1; WID=$(DISPLAY=:1 xdotool search --name "$TITLE" 2>/dev/null | head -1); [ -n "$WID" ] && break; kill -0 $EMUPID 2>/dev/null || break; done
echo "window=${WID:-none} after ${i}s" > "$OUT/window.txt"
if [ -n "$WID" ]; then
  DISPLAY=:1 xprop -id "$WID" -f STEAM_GAME 32c -set STEAM_GAME 424242
  DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
  t=0
  while kill -0 $EMUPID 2>/dev/null; do
    # Steam sometimes takes the base layer back during a run; the game is put back in front every 2 s.
    for ((k = 0; k < EVERY; k += 2)); do
      sleep 2
      DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
      kill -0 $EMUPID 2>/dev/null || break
    done
    t=$((t+EVERY))
    DISPLAY=:0 xprop -root -f GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 32c -set GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 1; sleep 2
    cp /tmp/gamescope.png "$OUT/shot-$t.png" 2>/dev/null
    echo "$t,$(DISPLAY=:1 xprop -id "$WID" WM_NAME 2>/dev/null | grep -oE "fps: [0-9.]+" | grep -oE "[0-9.]+")" >> "$OUT/fps.csv"
  done
fi
wait
