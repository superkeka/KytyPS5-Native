#!/bin/bash
# Runs one command on the board for try.sh (TRY_CMD), detached from the caller: inside the
# shadps4-dev distrobox, in /home/deck/dev/kyty-runs/TAG, with EXTRA_ENV set and a time limit.
# Usage: CMD="..." cmd-run.sh SECONDS TAG. Output in TAG/run.log, ending with "exit=<status>".
set -u
SECS=$1 TAG=$2
OUT=/home/deck/dev/kyty-runs/$TAG
mkdir -p "$OUT"
cd "$OUT" && distrobox enter shadps4-dev -- env ${EXTRA_ENV:-} timeout -s INT "$SECS" sh -c "$CMD" > run.log 2>&1
echo "exit=$?" >> run.log
