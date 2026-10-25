# Records a gamepad's evdev events with their time since start: padrec.py OUT [name substring]
import sys, time, evdev
name = sys.argv[2] if len(sys.argv) > 2 else "8BitDo"
dev = next(evdev.InputDevice(p) for p in evdev.list_devices() if name in evdev.InputDevice(p).name)
t0 = time.monotonic()
with open(sys.argv[1], "w") as f:
    f.write(f"# {dev.name}\n")
    for e in dev.read_loop():
        if e.type in (evdev.ecodes.EV_KEY, evdev.ecodes.EV_ABS, evdev.ecodes.EV_SYN):
            f.write(f"{time.monotonic() - t0:.4f} {e.type} {e.code} {e.value}\n")
            f.flush()
