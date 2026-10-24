#!/bin/bash
# Runs bc250-replay on the board through tools/bc250-runs/try.sh (one GPU job at a time, hang
# recovery) and copies its output back.
# Usage: replay.sh TAG CAPTURE_DIR_ON_BOARD [REPLAY ARGS...]
# Output on the board in /home/deck/dev/kyty-runs/TAG/out, here in ${TRY_OUT:-/tmp/bc250-try}/TAG/out.
# REPLAY_BIN overrides the binary (default: the kyty-replay build on the board).
set -u
TAG=$1 CAP=$2
shift 2
H=${BC250_HOST:-deck@bc250.local}
BIN=${REPLAY_BIN:-/home/deck/dev/kyty-replay/_Build/linux-no-qt/bc250_replay}
OUT=/home/deck/dev/kyty-runs/$TAG/out
HERE=$(cd "$(dirname "$0")" && pwd)
TRY_CRUMBS=0 TRY_CMD="$BIN $CAP --out $OUT $*" "$HERE/../bc250-runs/try.sh" "$TAG" "${REPLAY_SECS:-300}" ""
LOCAL=${TRY_OUT:-/tmp/bc250-try}/$TAG
cat "$LOCAL/run.log" 2>/dev/null | tail -40
scp -q -r "$H:$OUT" "$LOCAL/" 2>/dev/null && echo "output: $LOCAL/out"
