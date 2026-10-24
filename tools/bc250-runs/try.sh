#!/bin/bash
# Runs one native experiment on the BC-250 from the Mac and recovers the board on its own.
# Usage: try.sh TAG SECONDS "ENV=VALUE ..." [TITLE] ["EMULATOR ARGS"]
# Prints HANG or DONE. On a gfx ring hang (needs amdgpu.gpu_recovery=0) it saves the breadcrumbs and
# the devcoredump next to the run, reboots the board (sudoers: NOPASSWD systemctl reboot) and waits
# until SSH answers again.
set -u
TAG=$1 SECS=$2 ENVS=$3 TITLE=${4:-PPSA21567} ARGS=${5:-}
H=${BC250_HOST:-deck@bc250.local}
RUNS=/home/deck/dev/kyty-runs
LOCAL=${TRY_OUT:-/tmp/bc250-try}/$TAG
mkdir -p "$LOCAL"
ssh -o BatchMode=yes -o ServerAliveInterval=2 -o ServerAliveCountMax=5 $H 'journalctl -kf -n 0 --no-pager -o short-monotonic' > "$LOCAL/kernel.log" 2>&1 &
KLOG=$!
sleep 2
ssh -o BatchMode=yes $H "C=$RUNS/crumbs-$TAG; mkdir -p \$C; systemd-run --user --collect --unit=kyty-$TAG --setenv=EXTRA_ENV=\"$ENVS KYTY_NATIVE_GPU_BREADCRUMBS=\$C\" --setenv=EXTRA_ARGS=\"$ARGS\" $RUNS/game-run.sh $TITLE primary $SECS $TAG ${TRY_SHOT:-30}" > /dev/null 2>&1
result=TIMEOUT
for i in $(seq 1 $(( (SECS + 60) / 5 ))); do
	sleep 5
	if grep -q "ring gfx.*timeout" "$LOCAL/kernel.log"; then result=HANG; break; fi
	if ssh -o ConnectTimeout=5 $H "grep -q '^exit=' $RUNS/$TAG/run.log" 2>/dev/null; then result=DONE; break; fi
done
sleep 3
scp -q -r $H:$RUNS/crumbs-$TAG "$LOCAL/" 2>/dev/null
scp -q $H:$RUNS/$TAG/run.log "$LOCAL/" 2>/dev/null
kill $KLOG 2>/dev/null
echo "$result $(cat "$LOCAL/crumbs-$TAG/last.txt" 2>/dev/null)"
if [ $result != DONE ]; then
	ssh -o ConnectTimeout=5 $H 'sudo -n /usr/bin/systemctl reboot -ff || sudo -n /usr/bin/systemctl reboot' > /dev/null 2>&1
	sleep 30
	until ssh -o ConnectTimeout=5 -o BatchMode=yes $H true 2>/dev/null; do sleep 5; done
	sleep 20 # let the session and gamescope come up before the next run
	echo "board back"
fi
