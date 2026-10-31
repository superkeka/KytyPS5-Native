#!/bin/bash
# N gamescope screenshots back to back (about 0.7 s apart at 4K) and a 4-wide collage of them, numbered, to see
# what changes from frame to frame (flicker): burst.sh NAME [N] -> /home/deck/kyty-watch/NAME.png
N=${2:-12}
OUT=$(mktemp -d /home/deck/kyty-watch/burst-XXXXXX) || exit 1
for n in $(seq 1 "$N"); do
	i=$(printf %02d "$n")
	rm -f /tmp/gamescope.png
	DISPLAY=:0 xprop -root -f GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 32c -set GAMESCOPECTRL_DEBUG_REQUEST_SCREENSHOT 1
	for k in $(seq 1 100); do [ -s /tmp/gamescope.png ] && break; sleep 0.02; done
	s1=0
	for k in $(seq 1 50); do
		s2=$(stat -c %s /tmp/gamescope.png 2>/dev/null || echo 0)
		[ "$s2" = "$s1" ] && [ "$s2" != 0 ] && break
		s1=$s2; sleep 0.05
	done
	cp /tmp/gamescope.png "$OUT/f$i.png"
done
ffmpeg -loglevel error -y -framerate 1 -i "$OUT/f%02d.png" \
	-vf "scale=640:360,drawtext=text=%{frame_num}:x=8:y=8:fontsize=28:fontcolor=yellow:start_number=1,tile=4x$(( (N + 3) / 4 ))" \
	-frames:v 1 "/home/deck/kyty-watch/$1.png"
echo "/home/deck/kyty-watch/$1.png ($OUT)"
