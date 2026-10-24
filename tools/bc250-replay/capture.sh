#!/bin/bash
# Captures one frame of a game on the board (KYTY_NATIVE_GPU_CAPTURE) through tools/bc250-runs/try.sh.
# Usage: capture.sh TAG FLIP [SECONDS] ["MORE ENV=VALUE ..."]
# The capture lands on the board in /home/deck/dev/kyty-runs/TAG/cap/capture_FLIP. PRESS="DELAY KEY
# COUNT GAP" (default "55 j 3 5": ASTRO BOT's memory card animation after the second X) is pressed
# into the Kyty window by kyty-runs/press.sh during the run; the shared GPU lock is held from before
# the presses until the run ends, so no key lands in another session's game. TRY_EMU as in try.sh.
set -u
TAG=$1 FLIP=$2 SECS=${3:-200} MORE=${4:-}
H=${BC250_HOST:-deck@bc250.local}
RUNS=/home/deck/dev/kyty-runs
HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=${TRY_LOCK:-/tmp/bc250-gpu.lock}
exec 8>"$LOCK"
if command -v flock >/dev/null 2>&1; then
	flock 8
else
	until mkdir "$LOCK.d" 2>/dev/null; do sleep 5; done
	trap 'rmdir "$LOCK.d" 2>/dev/null' EXIT
fi
ssh -o BatchMode=yes $H "mkdir -p $RUNS/$TAG/cap; systemd-run --user --collect --unit=press-$TAG $RUNS/press.sh ${PRESS:-55 j 3 5}" > /dev/null 2>&1
# try.sh takes a lock of its own; this one is already held.
TRY_CRUMBS=0 TRY_LOCK=/tmp/bc250-capture-$TAG.lock "$HERE/../bc250-runs/try.sh" "$TAG" "$SECS" \
	"KYTY_NATIVE_GPU_CAPTURE=$FLIP KYTY_NATIVE_GPU_CAPTURE_DIR=$RUNS/$TAG/cap $MORE"
ssh -o BatchMode=yes $H "grep 'native capture' $RUNS/$TAG/run.log | tail -12; du -sh $RUNS/$TAG/cap/capture_$FLIP"
