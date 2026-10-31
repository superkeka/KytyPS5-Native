#!/bin/bash
# Demon's Souls (PPSA01341) from its Steam shortcut's launcher to gameplay with the saved character: the opening film
# skipped with a long Cross, PRESS ANY BUTTON, CONTINUE, then "Continue Offline" on the server notice. The game stays up
# (leave it from the in-game menu or with: systemctl --user stop kytyps5-PPSA01341). Env as for kytyps5-run.sh, e.g.
# set with systemctl --user set-environment before this. des-gameplay.sh [GAME_DIR]
GAME=${1:-/home/deck/Games/PPSA01341}
K=/home/deck/dev/kyty-runs/key.sh
systemd-run --user --collect --quiet /home/deck/.local/share/KytyPS5-Native/kyty/kytyps5-run.sh "$GAME"
sleep 22
$K PPSA01341 j 1 0.5 2.5 # the film
sleep 6
$K PPSA01341 j          # PRESS ANY BUTTON
sleep 5
$K PPSA01341 j          # CONTINUE
sleep 14
$K PPSA01341 j          # Unable to connect to the server: Continue Offline
sleep 40
/home/deck/dev/kyty-runs/shot.sh des-gameplay
