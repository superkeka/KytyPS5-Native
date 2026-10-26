#!/bin/bash
# My First Gran Turismo through its Steam shortcut, screenshots every 15 s, Return for the PSN dialog, then J every 15 s;
# after SECS the game gets SIGTERM and the shortcut its launch options back.
# Usage: gt-steam.sh TAG SECS [ENV=VALUE ...]
set -u
# GT_J / GT_J_GAP: how many J presses and how far apart (default 100, 15 s).
TAG=$1 SECS=$2; shift 2
R=/home/deck/dev/kyty-runs
GAMEID=18339203229432152064 APPID=4269928492
L=$R/steam-PPSA24156/run.log
OUT=$R/$TAG
pkill -x kyty_emulator; sleep 3
rm -rf $OUT; mkdir -p $OUT
python3 $R/steamcdp.py "SteamClient.Apps.SetShortcutLaunchOptions($APPID,\"PPSA24156 KYTY_NATIVE_GPU_FLIP_STATS=1 $*\")" >/dev/null
# GT_REC=<file>: replays a padrec.py recording (made from the launch) through a virtual pad instead of
# the key presses below; the virtual pad presses first, so it holds player 1.
[ -n "${GT_REC:-}" ] && systemd-run --user --collect --unit=rec-$TAG python3 $R/padplay.py "$GT_REC" $R/8bitdo.json 0 >/dev/null
# Steam ignores RunGame while it still counts the previous launch as running: retry.
for try in 1 2 3 4; do
	python3 $R/steamcdp.py "SteamClient.Apps.RunGame(\"$GAMEID\",\"\",-1,100)" >/dev/null
	for i in $(seq 1 60); do sleep 1; pgrep -x kyty_emulator >/dev/null && break 2; done
done
# Return (Options in the game) closes Kyty's PSN sign-in dialog once; J (Cross) goes through the menus.
if [ -z "${GT_REC:-}" ]; then
	systemd-run --user --collect --unit=pad-$TAG -E PRESS_TITLE=PPSA24156 -E PRESS_BASELAYER=0 $R/press.sh 35 Return 1 8 >/dev/null
	systemd-run --user --collect --unit=press-$TAG -E PRESS_TITLE=PPSA24156 -E PRESS_BASELAYER=0 $R/press.sh 50 j ${GT_J:-100} ${GT_J_GAP:-15} >/dev/null
fi
for t in $(seq 15 15 $SECS); do
	sleep 15
	DISPLAY=:0 xprop -root -f GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 32c -set GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 1; sleep 1
	cp /tmp/gamescope.png "$OUT/shot-$t.png" 2>/dev/null
	[ $t -lt 45 ] || pgrep -x kyty_emulator >/dev/null || { echo "emulator gone at $t s"; break; }
done
systemctl --user stop pad-$TAG press-$TAG rec-$TAG 2>/dev/null
pkill -x kyty_emulator; sleep 4
python3 $R/steamcdp.py "SteamClient.Apps.SetShortcutLaunchOptions($APPID,\"PPSA24156\")" >/dev/null
cp $L $OUT/run.log
grep -a "per flip: interval\|disabled\|WATCHDOG\|TIMED OUT" $OUT/run.log | tail -5 | cut -c1-220
grep -ac "native shadow: flip" $OUT/run.log | sed "s/^/flip stat lines: /"
