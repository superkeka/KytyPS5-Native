#!/bin/bash
# Regression over the titles with one Kyty binary, the Steam client stopped for each.
# Only Cross goes into the Kyty window: ASTRO and Demon's Souls reach gameplay, the others stay at their first screens.
# Usage: regress.sh BIN TAG [SECONDS] [TITLES...]; output in kyty-runs/TAG-<title> and kyty-runs/TAG.progress
set -u
BIN=$1 TAG=$2 SECS=${3:-150}
shift $(( $# < 3 ? $# : 3 ))
TITLES=("$@")
[ ${#TITLES[@]} -gt 0 ] || TITLES=(PPSA21567 PPSA01341 PPSA02100 PPSA01317)
R=/home/deck/dev/kyty-runs
P=$R/$TAG.progress
: > "$P"
say() { echo "$(date +%H:%M:%S) $*" >> "$P"; }

for T in "${TITLES[@]}"; do
	OUT=$R/$TAG-$T
	say "start $T"
	KEYS=0
	case $T in
	PPSA21567)
		(PRESS_TITLE=$T "$R/press.sh" 55 j 8 6 >> "$R/$TAG-$T.keys" 2>&1) &
		;;
	PPSA15552)
		(PRESS_TITLE=$T "$R/press.sh" 30 j 10 6 >> "$R/$TAG-$T.keys" 2>&1) &
		;;
	PPSA01341)
		(sleep 22; "$R/key.sh" "$T" j 1 0.5 2.5; sleep 6; "$R/key.sh" "$T" j; sleep 5; "$R/key.sh" "$T" j
			sleep 14; "$R/key.sh" "$T" j) >> "$R/$TAG-$T.keys" 2>&1 &
		;;
	esac
	[ "$T" = PPSA02100 ] || [ "$T" = PPSA01317 ] || KEYS=$!
	if [ -f "/home/deck/Games/$T.zar" ]; then export KYTY_GAME_PATH=/home/deck/Games/$T.zar; else unset KYTY_GAME_PATH; fi
	KYTY_EMU=$BIN "$R/nosteam-run.sh" "$T" primary "$SECS" "$TAG-$T" 20
	RC=$?
	[ "$KEYS" = 0 ] || kill "$KEYS" 2>/dev/null
	say "end $T rc=$RC last-fps $(tail -3 "$OUT/fps.csv" 2>/dev/null | paste -sd' ')"
	sleep 25
done
say "DONE"
