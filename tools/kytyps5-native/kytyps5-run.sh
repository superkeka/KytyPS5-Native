#!/bin/bash
# Plays a PS5 game in KytyPS5-Native from a Steam shortcut (the Decky plugin adds them).
# Usage (Steam launch options): kytyps5-run.sh "<game folder>" [--keep-steam]
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
# The title id (the window name has it): from the game's param.json, else the folder name.
TITLE=$(grep -o '"titleId" *: *"[^"]*"' "$GAME/sce_sys/param.json" 2>/dev/null | head -1 | sed 's/.*"\([^"]*\)"$/\1/')
TITLE=${TITLE:-$(basename "$GAME")}
CONFIG=${XDG_CONFIG_HOME:-$HOME/.config}/KytyPS5-Native
SHARE=${XDG_DATA_HOME:-$HOME/.local/share}/KytyPS5-Native
mkdir -p "$SHARE/data" "$SHARE/logs"
LOG=$SHARE/logs/$TITLE.log
# Games run on the gamescope Xwayland Steam gives them, the base layer is set on the root one (:0).
export DISPLAY=${DISPLAY:-:1}
ROOT_DISPLAY=${KYTYPS5_ROOT_DISPLAY:-:0}

steam_unit_active() { [ "$(systemctl --user is-active steam-launcher.service 2>/dev/null)" = active ]; }

# Unloading Steam: the rest runs in a systemd unit of its own, outside Steam's process tree (Steam is
# stopped by it), and this call returns at once, so Steam sees the shortcut as ended.
if [ "$KEEP_STEAM" = 0 ] && [ -z "${KYTYPS5_DETACHED:-}" ] && steam_unit_active; then
	unit=kytyps5-$TITLE
	systemctl --user reset-failed "$unit" 2>/dev/null
	exec systemd-run --user --collect --quiet --unit="$unit" -E KYTYPS5_DETACHED=1 -E DISPLAY="$DISPLAY" \
		-E HOME="$HOME" "$0" "$@"
fi

APPID=${SteamAppId:-${SteamGameId:-0}}
front_appid=${APPID}
if [ -n "${KYTYPS5_DETACHED:-}" ]; then
	front_appid=424242
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

# What gamescope shows. Without Steam: the base layer app id. With Steam: the shortcut's app id on
# every window of ours (gamescope only shows windows of the game Steam started).
show() { # window id, app id
	DISPLAY=$DISPLAY xprop -id "$1" -f STEAM_GAME 32c -set STEAM_GAME "$2" 2>/dev/null
	[ -n "${KYTYPS5_DETACHED:-}" ] && DISPLAY=$ROOT_DISPLAY xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co \
		-set GAMESCOPECTRL_BASELAYER_APPID "$2" 2>/dev/null
}
find_window() { # name regex, pid to watch
	local wid=""
	for _ in $(seq 1 600); do
		wid=$(xdotool search --name "$1" 2>/dev/null | head -1)
		[ -n "$wid" ] && break
		kill -0 "$2" 2>/dev/null || break
		sleep 0.1
	done
	echo "$wid"
}
play_video() {
	[ -f "$1" ] || return 0
	"$EMU" --play-video "$1" >> "$LOG" 2>&1 &
	local pid=$!
	local wid
	wid=$(find_window '^KytyPS5-Native video$' $pid)
	if [ -n "$wid" ]; then
		if [ -n "${KYTYPS5_DETACHED:-}" ]; then show "$wid" 424243; else show "$wid" "$APPID"; fi
	fi
	wait $pid
}

cd "$SHARE/data"
play_video "$CONFIG/boot.webm"

# The Decky plugin leaves this when a newer release is out: Kyty says so in its trophy toast.
if [ -s "$CONFIG/update-available" ]; then
	export KYTY_NOTICE_TITLE="KytyPS5-Native update available"
	export KYTY_NOTICE_TEXT="$(head -1 "$CONFIG/update-available"): update it in the Decky plugin"
fi

MARKER=$SHARE/data/.exit-$TITLE
rm -f "$MARKER"
{
	echo "== $(date) $GAME"
	SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1 KYTY_EXIT_MARKER="$MARKER" "$EMU" --game "$GAME" \
		--screen-width "${SCREEN%x*}" --screen-height "${SCREEN#*x}" --fullscreen --no-splash --skip-notice-screen
	echo "== exit=$?"
} >> "$LOG" 2>&1 &
EMUPID=$!
# The exit video starts as soon as the user leaves from the menu, while the game shuts down.
( while :; do
	[ -e "$MARKER" ] && { rm -f "$MARKER"; play_video "$CONFIG/exit.webm"; break; }
	kill -0 $EMUPID 2>/dev/null || break
	sleep 0.1
done ) &
WATCH=$!
GAME_WID=$(find_window "$TITLE" $EMUPID)
if [ -n "$GAME_WID" ]; then
	show "$GAME_WID" "$front_appid"
	# Without Steam nothing else sets the base layer; keep the game in front until a video takes it.
	while kill -0 $EMUPID 2>/dev/null; do
		sleep 2
		[ -n "${KYTYPS5_DETACHED:-}" ] && [ ! -e "$MARKER" ] && ! xdotool search --name '^KytyPS5-Native video$' >/dev/null 2>&1 &&
			show "$GAME_WID" 424242
	done
fi
wait $EMUPID
wait $WATCH
