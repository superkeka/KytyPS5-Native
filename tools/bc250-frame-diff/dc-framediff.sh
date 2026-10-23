#!/bin/bash
# Dead Cells in native shadow mode, dumping native and Vulkan versions of a series of flips for
# tools/bc250-frame-diff. Usage: [EXTRA_ENV="VAR=1 ..."] dc-framediff.sh TAG [FLIPS]
#   -> /home/deck/dev/kyty-runs/TAG. FLIPS as in KYTY_NATIVE_GPU_DUMP_FLIPS (default 200-6000/100).
# Inputs: menu, three confirms into gameplay, walk/jump, pause. Each dumped flip is ~66 MB.
set -u
TAG=${1:-dc-framediff}
FLIPS=${2:-200-6000/100}
OUT=/home/deck/dev/kyty-runs/$TAG
EMU=/home/deck/dev/KytyPS5-bc250/_Build/linux-no-qt/kyty_emulator
mkdir -p "$OUT"
key() { DISPLAY=:1 xdotool keydown "$1"; sleep "${2:-0.15}"; DISPLAY=:1 xdotool keyup "$1"; }
restore() { DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 769; }
trap restore EXIT
(cd "$OUT" && distrobox enter shadps4-dev -- env DISPLAY=:1 ${EXTRA_ENV:-} KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_SHADOW=1 KYTY_NATIVE_GPU_DUMP_FLIPS="$FLIPS" KYTY_NATIVE_GPU_DUMP_DIR="$OUT" timeout -s INT 200 "$EMU" --game /home/deck/Games/PPSA15552 --no-splash --skip-notice-screen > run.log 2>&1; echo "exit=$?" >> run.log) &
WID=""
for i in $(seq 1 40); do sleep 1; WID=$(DISPLAY=:1 xdotool search --name "Dead Cells, PPSA15552" 2>/dev/null | head -1); [ -n "$WID" ] && break; done
[ -z "$WID" ] && { echo "no window"; wait; exit 1; }
DISPLAY=:1 xprop -id "$WID" -f STEAM_GAME 32c -set STEAM_GAME 424242
DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
sleep 15
key j; sleep 5; key j; sleep 5; key j; sleep 40
for i in 1 2 3 4; do key Right 2; key Left 2; key space; sleep 1; done
sleep 10
key Return; sleep 20
key Return
wait
