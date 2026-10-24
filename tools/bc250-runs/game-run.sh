#!/bin/bash
# Runs a PS5 dump in Kyty on the TV with periodic screenshots, detached from the caller.
# Usage: game-run.sh TITLEID MODE SECONDS TAG [SHOT_EVERY]
#   MODE: vulkan (stock Kyty), shadow (native next to Vulkan), primary (fully native)
# Output in /home/deck/dev/kyty-runs/TAG: run.log, shot-<t>.png, fps.csv. Restores the desktop at the end.
set -u
TITLE=$1 MODE=$2 SECS=$3 TAG=$4 EVERY=${5:-15}
OUT=/home/deck/dev/kyty-runs/$TAG
EMU=/home/deck/dev/KytyPS5-bc250/_Build/linux-no-qt/kyty_emulator
mkdir -p "$OUT"
case $MODE in
  vulkan)  ENVS="" ;;
  shadow)  ENVS="KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_SHADOW=1" ;;
  primary) ENVS="KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_PRIMARY=1" ;;
  *) echo "bad mode"; exit 1 ;;
esac
restore() { DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 769; }
trap restore EXIT
(cd "$OUT" && distrobox enter shadps4-dev -- env DISPLAY=:1 SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1 ${EXTRA_ENV:-} $ENVS timeout -s INT "$SECS" "$EMU" --game /home/deck/Games/$TITLE --no-splash --skip-notice-screen ${OVERLAY---perf-overlay} ${EXTRA_ARGS:-} > run.log 2>&1; echo "exit=$?" >> run.log) &
EMUPID=$!
WID=""
for i in $(seq 1 90); do sleep 1; WID=$(DISPLAY=:1 xdotool search --name "$TITLE" 2>/dev/null | head -1); [ -n "$WID" ] && break; kill -0 $EMUPID 2>/dev/null || break; done
echo "window=${WID:-none} after ${i}s" > "$OUT/window.txt"
if [ -n "$WID" ]; then
  DISPLAY=:1 xprop -id "$WID" -f STEAM_GAME 32c -set STEAM_GAME 424242
  DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
  t=0
  while kill -0 $EMUPID 2>/dev/null; do
    sleep "$EVERY"; t=$((t+EVERY))
    # Steam sometimes takes the base layer back during a run; the game stays in front.
    DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
    DISPLAY=:0 xprop -root -f GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 32c -set GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 1; sleep 2
    cp /tmp/gamescope.png "$OUT/shot-$t.png" 2>/dev/null
    echo "$t,$(DISPLAY=:1 xprop -id "$WID" WM_NAME 2>/dev/null | grep -oE "fps: [0-9.]+" | grep -oE "[0-9.]+")" >> "$OUT/fps.csv"
  done
fi
wait
