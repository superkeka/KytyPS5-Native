# Saves a gamepad's identity and capabilities for padplay.py: padcaps.py OUT.json [name substring]
import sys, json, evdev
name = sys.argv[2] if len(sys.argv) > 2 else "8BitDo"
dev = next(evdev.InputDevice(p) for p in evdev.list_devices() if name in evdev.InputDevice(p).name)
caps = {}
for typ, codes in dev.capabilities(absinfo=True).items():
    if typ in (evdev.ecodes.EV_SYN,):
        continue
    caps[typ] = [[c[0], list(c[1])] if isinstance(c, tuple) else c for c in codes]
json.dump({"name": dev.name, "vendor": dev.info.vendor, "product": dev.info.product, "version": dev.info.version,
           "bustype": dev.info.bustype, "caps": caps}, open(sys.argv[1], "w"))
print("saved", dev.name, hex(dev.info.vendor), hex(dev.info.product))
