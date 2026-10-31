#!/bin/sh
# Presses a pad button in a running Kyty, through the keyboard Kyty maps to player 1: key.sh TITLE KEY [COUNT] [GAP] [HOLD]
# J Cross, L Circle, I Triangle, K Square, Q/E L1/R1, Z/C L2/R2, Return Options, arrows the D-pad, WASD the left stick,
# TFGH the right one. HOLD (s, default 0.15): Demon's Souls skips its opening film on a long Cross (2.5).
# Only while Steam is unloaded (kytyps5-run.sh without --keep-steam): a key never lands in Steam.
TITLE=$1 KEY=$2
W=$(DISPLAY=:1 xdotool search --name "$TITLE" 2>/dev/null | head -1)
[ -n "$W" ] || { echo "no Kyty window for $TITLE"; exit 1; }
systemctl --user is-active -q steam-launcher && { echo "Steam is up: not pressing"; exit 1; }
DISPLAY=:0 xprop -root -f GAMESCOPECTRL_BASELAYER_APPID 32co -set GAMESCOPECTRL_BASELAYER_APPID 424242
for i in $(seq 1 "${3:-1}"); do
	DISPLAY=:1 xdotool windowactivate --sync "$W" 2>/dev/null
	DISPLAY=:1 xdotool keydown --window "$W" "$KEY"; sleep "${5:-0.15}"; DISPLAY=:1 xdotool keyup --window "$W" "$KEY"
	sleep "${4:-0.5}"
done
echo "pressed $KEY x${3:-1}"
