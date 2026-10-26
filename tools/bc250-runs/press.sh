#!/bin/sh
# Presses a key in the Kyty window: press.sh DELAY KEY COUNT GAP. Only into the Kyty window; exits
# when the emulator or its window is gone.
EMU="kyty_emulator"
sleep "$1"
for i in $(seq 1 "$3"); do
	W=$(DISPLAY=:1 xdotool search --name ${PRESS_TITLE:-PPSA21567} 2>/dev/null | head -1)
	{ [ -n "$W" ] && pgrep -f "$EMU" >/dev/null; } || exit 0
	# Keep the game on gamescope's base layer, so a key never lands in Steam if it took the screen.
	# PRESS_BASELAYER=0 leaves it: with Steam Input the base-layer app id also routes the user's own pad.
	[ "${PRESS_BASELAYER:-1}" = 0 ] || DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
	DISPLAY=:1 xdotool windowactivate --sync "$W" 2>/dev/null
	DISPLAY=:1 xdotool keydown --window "$W" "$2"; sleep 0.15; DISPLAY=:1 xdotool keyup --window "$W" "$2"
	echo "$(date +%T) pressed $2 in $W"
	sleep "$4"
done
