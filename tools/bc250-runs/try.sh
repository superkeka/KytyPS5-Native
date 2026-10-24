#!/bin/bash
# Runs one native experiment on the BC-250 from the Mac and recovers the board on its own.
# Usage: try.sh TAG SECONDS "ENV=VALUE ..." [TITLE] ["EMULATOR ARGS"]
# TRY_CMD="..." runs that command on the board instead of a game (cmd-run.sh: inside the distrobox,
# in $RUNS/TAG, with the ENV list set), e.g. the frame replay tool, under the same lock and hang
# recovery.
# Prints HANG or DONE. On a gfx ring hang (needs amdgpu.gpu_recovery=0) it saves the breadcrumbs and
# the devcoredump next to the run, reboots the board (sudoers: NOPASSWD systemctl reboot) and waits
# until SSH answers again.
set -u
TAG=$1 SECS=$2 ENVS=$3 TITLE=${4:-PPSA21567} ARGS=${5:-}
# TRY_CRUMBS=0 runs without breadcrumbs (they sync every submission to disk: slow).
CRUMBS='KYTY_NATIVE_GPU_BREADCRUMBS=$C'
[ "${TRY_CRUMBS:-1}" = 0 ] && CRUMBS=""
H=${BC250_HOST:-deck@bc250.local}
RUNS=/home/deck/dev/kyty-runs
LOCAL=${TRY_OUT:-/tmp/bc250-try}/$TAG
mkdir -p "$LOCAL"
# One GPU experiment at a time on the board, across sessions: hold a lock on the Mac side for the
# whole run (TRY_LOCK overrides the path; several sessions share it).
LOCK=${TRY_LOCK:-/tmp/bc250-gpu.lock}
exec 9>"$LOCK"
if command -v flock >/dev/null 2>&1; then
	flock 9
else
	until mkdir "$LOCK.d" 2>/dev/null; do sleep 5; done
	trap 'rmdir "$LOCK.d" 2>/dev/null' EXIT
fi
ssh -o BatchMode=yes -o ServerAliveInterval=2 -o ServerAliveCountMax=5 $H 'journalctl -kf -n 0 --no-pager -o short-monotonic' > "$LOCAL/kernel.log" 2>&1 &
KLOG=$!
sleep 2
if [ -n "${TRY_CMD:-}" ]; then
	ssh -o BatchMode=yes $H "C=$RUNS/crumbs-$TAG; mkdir -p \$C; systemd-run --user --collect --unit=kyty-$TAG --setenv=EXTRA_ENV=\"$ENVS\" --setenv=CMD=\"$TRY_CMD\" $RUNS/cmd-run.sh $SECS $TAG" > /dev/null 2>&1
else
	ssh -o BatchMode=yes $H "C=$RUNS/crumbs-$TAG; mkdir -p \$C; systemd-run --user --collect --unit=kyty-$TAG --setenv=EXTRA_ENV=\"$ENVS $CRUMBS\" --setenv=EXTRA_ARGS=\"$ARGS\" ${TRY_NO_OVERLAY:+--setenv=OVERLAY=} ${TRY_EMU:+--setenv=KYTY_EMU=$TRY_EMU} $RUNS/game-run.sh $TITLE primary $SECS $TAG ${TRY_SHOT:-30}" > /dev/null 2>&1
fi
result=TIMEOUT
for i in $(seq 1 $(( (SECS + 60) / 5 ))); do
	sleep 5
	if grep -q "ring gfx.*timeout" "$LOCAL/kernel.log"; then result=HANG; break; fi
	if ssh -o ConnectTimeout=5 $H "grep -q '^exit=' $RUNS/$TAG/run.log" 2>/dev/null; then result=DONE; break; fi
done
sleep 3
scp -q -o ConnectTimeout=5 -o ServerAliveInterval=3 -o ServerAliveCountMax=3 -r $H:$RUNS/crumbs-$TAG "$LOCAL/" 2>/dev/null
scp -q -o ConnectTimeout=5 -o ServerAliveInterval=3 -o ServerAliveCountMax=3 $H:$RUNS/$TAG/run.log "$LOCAL/" 2>/dev/null
kill $KLOG 2>/dev/null
echo "$result $(cat "$LOCAL/crumbs-$TAG/last.txt" 2>/dev/null)"
if [ $result != DONE ]; then
	ssh -o ConnectTimeout=5 $H 'sudo -n /usr/bin/systemctl reboot -ff || sudo -n /usr/bin/systemctl reboot' > /dev/null 2>&1
	sleep 30
	# A reboot after a GPU hang sometimes stalls in shutdown (network up, sshd gone): give up after
	# TRY_BOOT_WAIT seconds so the caller can report it instead of waiting forever.
	waited=0
	until ssh -o ConnectTimeout=5 -o BatchMode=yes $H true 2>/dev/null; do
		sleep 5; waited=$((waited + 5))
		if [ $waited -ge ${TRY_BOOT_WAIT:-600} ]; then echo "BOARD DOWN (needs a power cycle)"; exit 2; fi
	done
	sleep 20 # let the session and gamescope come up before the next run
	echo "board back"
fi
