#!/bin/sh
# A gamescope screenshot of what the TV shows: shot.sh NAME -> /home/deck/kyty-watch/NAME.png
mkdir -p /home/deck/kyty-watch
rm -f /tmp/gamescope.png
DISPLAY=:0 xprop -root -f GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 32c -set GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 1
for i in $(seq 1 30); do [ -s /tmp/gamescope.png ] && break; sleep 0.2; done
sleep 0.5
cp /tmp/gamescope.png "/home/deck/kyty-watch/$1.png" && echo "/home/deck/kyty-watch/$1.png"
