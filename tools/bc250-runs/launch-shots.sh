#!/bin/bash
# Starts a game the way its Steam shortcut does (kytyps5-run.sh, Steam unloaded), a gamescope screenshot every 10 s
# for SECS seconds, then leaves as the in-game menu does (exit marker, SIGTERM) and Steam comes back.
# launch-shots.sh GAME TAG SECS TITLE -> /home/deck/kyty-loadtest/TAG: run.log, t<s>.png
GAME=$1 TAG=$2 SECS=${3:-120} T=${4:-PPSA02100}
OUT=/home/deck/kyty-loadtest/$TAG
SHARE=/home/deck/.local/share/KytyPS5-Native
mkdir -p "$OUT"
: > "$SHARE/logs/$T.log"
"$SHARE/kyty/kytyps5-run.sh" "$GAME"
for i in $(seq 1 60); do pgrep -x kyty_emulator >/dev/null && break; sleep 0.5; done
t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt "$SECS" ]; do
	sleep 8
	DISPLAY=:0 xprop -root -f GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 32c -set GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 1; sleep 2
	cp /tmp/gamescope.png "$OUT/t$(printf %03d $(( $(date +%s) - t0 ))).png" 2>/dev/null
done
touch "$SHARE/data/.exit-$T"
for p in $(pgrep -f -- "kyty_emulator --game"); do kill -TERM "$p"; done
for i in $(seq 1 60); do pgrep -f kytyps5-run.sh >/dev/null || break; sleep 1; done
cp "$SHARE/logs/$T.log" "$OUT/run.log"
grep -a -E "tracker stuck|TIMED OUT|disabling" "$OUT/run.log" | head -20 | cut -c1-160
echo "steam: $(systemctl --user is-active steam-launcher)"
