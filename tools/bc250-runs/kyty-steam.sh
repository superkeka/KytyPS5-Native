#!/bin/bash
# Launches a PS5 title in Kyty (native BC-250 backend) from a Steam shortcut.
# Usage (Steam launch options): kyty-steam.sh TITLEID [extra env assignments...]
# Log: /home/deck/dev/kyty-runs/steam-TITLEID/run.log
set -u
TITLE=${1:?title id}; shift
# Extra items: VAR=value goes to the environment, --option=value to the emulator (as --option value).
EXTRA=(); ARGS=()
for a in "$@"; do case $a in --*=*) ARGS+=("${a%%=*}" "${a#*=}");; *) EXTRA+=("$a");; esac; done
OUT=/home/deck/dev/kyty-runs/steam-$TITLE
EMU=/home/deck/dev/KytyPS5-bc250/_Build/linux-no-qt/kyty_emulator
SAVES=/home/deck/kyty-saves
mkdir -p "$OUT" "$SAVES/_SaveData" "$SAVES/_Trophies"
for d in _SaveData _Trophies; do [ -e "$OUT/$d" ] || ln -s "$SAVES/$d" "$OUT/$d"; done
ENVS="KYTY_NATIVE_GPU_GEM=1 KYTY_NATIVE_GPU_PRIMARY=1 KYTY_NATIVE_GPU_FLIP_STATS=1 KYTY_NATIVE_GPU_ACB_RINGS=1"
[ -S /tmp/kyty-lodhw.sock ] && ENVS="$ENVS KYTY_LOD_HW_SOCK=/tmp/kyty-lodhw.sock"
APPID=${SteamAppId:-${SteamGameId:-0}}
# Steam sizes the gamescope display from the per-app resolution: the swapchain follows it.
SIZE=$(DISPLAY="${DISPLAY:-:1}" xdpyinfo 2>/dev/null | sed -n "s/.*dimensions: *\([0-9]*\)x\([0-9]*\) pixels.*/\1 \2/p")
set -- $SIZE
W=${1:-3840}; H=${2:-2160}
cd "$OUT"
# The container must not start inside this launch: its conmon would become a child of Steam's reaper
# (a subreaper), which waits for it after the game exits, so Steam keeps the game "running".
if ! podman container inspect -f "{{.State.Running}}" shadps4-dev 2>/dev/null | grep -q true; then
	systemctl --user reset-failed shadps4-dev-container 2>/dev/null
	systemd-run --user --unit=shadps4-dev-container -p KillMode=process -p RemainAfterExit=yes --quiet \
		podman start shadps4-dev >/dev/null 2>&1
	for i in $(seq 1 30); do podman container inspect -f "{{.State.Running}}" shadps4-dev 2>/dev/null | grep -q true && break; sleep 1; done
fi
distrobox enter shadps4-dev -- env DISPLAY="${DISPLAY:-:1}" SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1 $ENVS "${EXTRA[@]}" "$EMU" --game /home/deck/Games/$TITLE --screen-width $W --screen-height $H --fullscreen --no-splash --skip-notice-screen --perf-overlay "${ARGS[@]}" > run.log 2>&1 &
PID=$!
# The emulator runs in the container, outside Steam's process tree: tag its window for gamescope.
for i in $(seq 1 60); do
	sleep 1
	WID=$(DISPLAY="${DISPLAY:-:1}" xdotool search --name "$TITLE" 2>/dev/null | head -1)
	[ -n "$WID" ] && break
	kill -0 $PID 2>/dev/null || break
done
[ -n "${WID:-}" ] && [ "$APPID" != 0 ] && DISPLAY="${DISPLAY:-:1}" xprop -id "$WID" -f STEAM_GAME 32c -set STEAM_GAME "$APPID"
echo "window=${WID:-none} appid=$APPID display=${DISPLAY:-}" > "$OUT/window.txt"
wait $PID
