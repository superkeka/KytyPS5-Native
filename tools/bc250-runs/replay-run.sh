#!/bin/bash
# Launches ASTRO BOT, replays the pool recording through the control socket, dumps flips at the pool.
# replay-run.sh TAG "EXTRA ENV" [DUMP_AT_S] [DUMPS]
TAG=$1; ENVX=$2; AT=${3:-122}; N=${4:-12}
R=/home/deck/dev/kyty-runs
rm -rf $R/water-d; mkdir -p $R/water-d
T0=$(date +%s.%N)
systemd-run --user --collect --unit=kyty-$TAG /bin/bash -c "KYTY_EMU=/home/deck/dev/KytyPS5-bc250/_Build/linux-no-qt/kyty_emulator EXTRA_ENV=\"KYTY_NATIVE_GPU_FLIP_STATS=1 KYTY_NATIVE_GPU_ACB_RINGS=1 KYTY_LOD_HW_SOCK=/tmp/kyty-lodhw.sock KYTY_NATIVE_GPU_CTL=1 KYTY_NATIVE_GPU_DUMP_DIR=$R/water-d $ENVX\" $R/game-run.sh PPSA21567 primary 400 $TAG 400" >/dev/null
until echo ping | socat - UNIX-CONNECT:/tmp/kyty-ctl.sock 2>/dev/null | grep -q pong; do sleep 0.5; done
E=$(python3 -c "import time; print(round(time.time() - $T0, 2))")
echo "pad play $R/pool.rec $E" | socat - UNIX-CONNECT:/tmp/kyty-ctl.sock
sleep $(python3 -c "print(max(0, $AT - $E))")
echo "dump $N" | socat - UNIX-CONNECT:/tmp/kyty-ctl.sock
until [ $(grep -ac "dumped flip" $R/$TAG/run.log) -ge $N ]; do sleep 2; done
python3 -I $R/previews.py $R/water-d >/dev/null 2>&1
systemctl --user stop kyty-$TAG
echo done
