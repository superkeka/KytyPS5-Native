#!/bin/bash
# Plays a PS5 game in KytyPS5-Native from a Steam shortcut (the Decky plugin adds them).
# Usage (Steam launch options): kytyps5-run.sh "<game folder or .zar>" [--keep-steam]
#
# By default Steam is unloaded while the game runs: Steam and its web helpers hold 2-3 GB of RAM that a
# big PS5 game needs (Gran Turismo 7 uses ~12.6 of 14.8 GB). It comes back on any exit. --keep-steam
# leaves Steam running (enough for smaller games; its overlay and menu stay).
# ~/.config/KytyPS5-Native/boot.webm plays before the game and exit.webm when the game is left from the
# in-game menu (PS button). A missing file plays nothing.
# Saves: ~/.local/share/KytyPS5-Native/data. Log: ~/.local/share/KytyPS5-Native/logs/<title id>.log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
EMU=$HERE/kyty_emulator
GAME=${1:?game folder}
KEEP_STEAM=0
[ "${2:-}" = --keep-steam ] && KEEP_STEAM=1
# The title id (the log and the unit are named by it): from the game's param.json, inside the archive for
# a .zar, else the folder name.
if [ -f "$GAME" ]; then
	PARAM=$("$EMU" --zar-cat "$GAME" sce_sys/param.json 2>/dev/null)
else
	PARAM=$(cat "$GAME/sce_sys/param.json" 2>/dev/null)
fi
TITLE=$(printf '%s' "$PARAM" | grep -o '"titleId" *: *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
TITLE=${TITLE:-$(basename "$GAME" .zar)}
CONFIG=${XDG_CONFIG_HOME:-$HOME/.config}/KytyPS5-Native
SHARE=${XDG_DATA_HOME:-$HOME/.local/share}/KytyPS5-Native
mkdir -p "$SHARE/data" "$SHARE/logs"
LOG=$SHARE/logs/$TITLE.log
# Games run on the gamescope Xwayland Steam gives them, the base layer is set on the root one (:0).
export DISPLAY=${DISPLAY:-:1}
ROOT_DISPLAY=${KYTYPS5_ROOT_DISPLAY:-:0}

# One game at a time: two Kyty on the native path fight over the same GPU queues. The lock is held
# for the whole session (the exit video too); a second launch leaves at once, Steam untouched.
exec 9>"$SHARE/.running"
if ! flock -n 9; then
	echo "== $(date) $GAME: another game is running" >> "$LOG"
	exit 1
fi

steam_unit_active() { [ "$(systemctl --user is-active steam-launcher.service 2>/dev/null)" = active ]; }

# Unloading Steam: the rest runs in a systemd unit of its own, outside Steam's process tree (Steam is
# stopped by it), and this call returns at once, so Steam sees the shortcut as ended.
if [ "$KEEP_STEAM" = 0 ] && [ -z "${KYTYPS5_DETACHED:-}" ] && steam_unit_active; then
	unit=kytyps5-$TITLE
	systemctl --user reset-failed "$unit" 2>/dev/null
	# The unit takes the lock itself: let it go here.
	exec 9>&-
	exec systemd-run --user --collect --quiet --unit="$unit" -E KYTYPS5_DETACHED=1 -E DISPLAY="$DISPLAY" \
		-E HOME="$HOME" "$0" "$@"
fi

APPID=${SteamAppId:-${SteamGameId:-0}}
if [ -n "${KYTYPS5_DETACHED:-}" ]; then
	restore_steam() {
		systemd-run --user --collect --quiet /bin/sh -c \
			'systemctl --user unmask --runtime steam-launcher.service; systemctl --user start steam-launcher.service'
	}
	trap restore_steam EXIT
	# A stop sends SIGTERM, on which bash may skip the EXIT trap: restore there too.
	trap 'trap - EXIT; restore_steam; exit 143' TERM INT HUP
	# Steam stopped while it is still starting hangs in its stop: let it have been up for 20 s first.
	started=$(systemctl --user show steam-launcher.service -p ActiveEnterTimestampMonotonic --value)
	if [ "${started:-0}" -gt 0 ]; then
		up=$(( $(awk '{printf "%d", $1 * 1000000}' /proc/uptime) - started ))
		[ "$up" -lt 20000000 ] && sleep $(( (20000000 - up) / 1000000 + 1 ))
	fi
	systemctl --user mask --runtime steam-launcher.service
	systemctl --user stop steam-launcher.service
	systemctl --user stop gamescope-mangoapp.service 2>/dev/null
fi

# The window is fullscreen at the connected display's native mode, so the swapchain matches the screen.
SCREEN=${KYTYPS5_SCREEN:-$(for st in /sys/class/drm/card*-*/status; do
	[ "$(cat "$st")" = connected ] && head -1 "$(dirname "$st")/modes" && break; done)}
SCREEN=${SCREEN:-1920x1080}
# Without Steam gamescope keeps the Xwayland at 1280x800: set it the way Steam does for a game.
[ -n "${KYTYPS5_DETACHED:-}" ] && DISPLAY=$ROOT_DISPLAY xprop -root -f GAMESCOPE_XWAYLAND_MODE_CONTROL 32c \
	-set GAMESCOPE_XWAYLAND_MODE_CONTROL "1,${SCREEN%x*},${SCREEN#*x},1" 2>/dev/null

# What gamescope shows: Kyty sets the app id in KYTY_GAMESCOPE_APPID on its window (gamescope shows only
# windows of the app in front). With Steam: the shortcut's app id. Without Steam: our own ids, and the
# base layer on the root window (:0) picks the one in front.
GAME_APPID=$APPID
VIDEO_APPID=$APPID
[ -n "${KYTYPS5_DETACHED:-}" ] && GAME_APPID=424242 VIDEO_APPID=424243
front() {
	[ -n "${KYTYPS5_DETACHED:-}" ] && DISPLAY=$ROOT_DISPLAY xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co \
		-set GAMESCOPECTRL_BASELAYER_APPID "$1" 2>/dev/null
}
# Present while a video plays: the game is not brought back in front over it.
VIDEO_FLAG=$SHARE/.video
play_video() {
	[ -f "$1" ] || return 0
	touch "$VIDEO_FLAG"
	front "$VIDEO_APPID"
	KYTY_GAMESCOPE_APPID=$VIDEO_APPID "$EMU" --play-video "$1" >> "$LOG" 2>&1
	rm -f "$VIDEO_FLAG"
}

cd "$SHARE/data"
play_video "$CONFIG/boot.webm"

# Notices Kyty shows in its trophy toast once the game is up, a line each.
NOTICE_TITLES=""
NOTICE_TEXTS=""
notice() {
	NOTICE_TITLES="${NOTICE_TITLES:+$NOTICE_TITLES$'\n'}$1"
	NOTICE_TEXTS="${NOTICE_TEXTS:+$NOTICE_TEXTS$'\n'}$2"
}
# On the PS5's APU: the kernel config the native backend wants (README, Requirements). The patched
# amdgpu (bc250_gfx_queues) brings its own queue options; GTT and the scheduler still count.
for dev in /sys/class/drm/card*/device; do
	case "$(cat "$dev/vendor" 2>/dev/null):$(cat "$dev/device" 2>/dev/null)" in
	0x1002:0x13fe | 0x1002:0x13da)
		param() { cat "/sys/module/amdgpu/parameters/$1" 2>/dev/null; }
		off=""
		grep -q 'ttm.pages_limit=3670016' /proc/cmdline || off="$off, GTT 14 GiB"
		[ "$(param sched_policy)" = 2 ] || off="$off, sched_policy=2"
		if [ ! -e /sys/module/amdgpu/parameters/bc250_gfx_queues ]; then
			[ "$(param vm_update_mode)" = 0 ] || off="$off, vm_update_mode=0"
			[ "$(param cwsr_enable)" = 0 ] || off="$off, cwsr_enable=0"
		fi
		[ -n "$off" ] && notice "Not the recommended kernel config" "Off: ${off#, }. Kernel config in the Decky plugin"
		break ;;
	esac
done
# The Decky plugin leaves this when a newer release is out.
[ -s "$CONFIG/update-available" ] &&
	notice "KytyPS5-Native update available" "$(head -1 "$CONFIG/update-available"): update it in the Decky plugin"
[ -n "$NOTICE_TITLES" ] && export KYTY_NOTICE_TITLE="$NOTICE_TITLES" KYTY_NOTICE_TEXT="$NOTICE_TEXTS"

MARKER=$SHARE/data/.exit-$TITLE
rm -f "$MARKER" "$VIDEO_FLAG"
front "$GAME_APPID"
{
	echo "== $(date) $GAME"
	SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1 KYTY_EXIT_MARKER="$MARKER" KYTY_GAMESCOPE_APPID=$GAME_APPID "$EMU" --game "$GAME" \
		--screen-width "${SCREEN%x*}" --screen-height "${SCREEN#*x}" --fullscreen --no-splash --skip-notice-screen
	echo "== exit=$?"
} >> "$LOG" 2>&1 &
EMUPID=$!
# The exit video starts as soon as the user leaves from the menu, while the game shuts down.
( while :; do
	[ -e "$MARKER" ] && { play_video "$CONFIG/exit.webm"; rm -f "$MARKER"; break; }
	kill -0 $EMUPID 2>/dev/null || break
	sleep 0.1
done ) &
WATCH=$!
# Without Steam nothing else sets the base layer; keep the game in front unless the user left or a
# video plays.
while kill -0 $EMUPID 2>/dev/null; do
	sleep 2
	[ ! -e "$MARKER" ] && [ ! -e "$VIDEO_FLAG" ] && front "$GAME_APPID"
done
wait $EMUPID
wait $WATCH
