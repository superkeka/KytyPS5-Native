#!/bin/bash
# Samples system memory while the emulator runs: memwatch.sh OUT_CSV [EVERY_SECONDS]
# Columns: t, used MiB (MemTotal - MemAvailable), MemAvailable, swap used, GTT used, emulator RSS (MiB),
# compilers running (a build next to the run skews the figures).
# Waits up to 2 minutes for kyty_emulator to start and stops when it exits. Start it next to try.sh:
#   ssh deck@bc250.local 'systemd-run --user --collect --unit=memwatch /home/deck/dev/kyty-runs/memwatch.sh OUT.csv 5'
OUT=$1 EVERY=${2:-5}
GTT=/sys/class/drm/renderD128/device/mem_info_gtt_used
pid=""
for i in $(seq 1 120); do pid=$(pgrep -x kyty_emulator | head -1); [ -n "$pid" ] && break; sleep 1; done
[ -z "$pid" ] && exit 1
echo "t,used_mib,avail_mib,swap_used_mib,gtt_used_mib,rss_mib,compilers" > "$OUT"
t=0
while kill -0 "$pid" 2>/dev/null; do
	awk -v t=$t -v gtt=$(cat $GTT) -v rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null || echo 0) \
		-v cc=$(pgrep -c -f 'clang|cc1plus') '
		/^MemTotal:/{tot=$2} /^MemAvailable:/{av=$2} /^SwapTotal:/{st=$2} /^SwapFree:/{sf=$2}
		END{printf "%d,%d,%d,%d,%d,%d,%d\n", t, (tot-av)/1024, av/1024, (st-sf)/1024, gtt/1048576, rss/1024, cc}' /proc/meminfo >> "$OUT"
	sleep "$EVERY"; t=$((t + EVERY))
done
