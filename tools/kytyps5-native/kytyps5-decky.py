#!/usr/bin/env python3
"""Turns the Decky plugins off while a game runs, and back on after it.

Usage: kytyps5-decky.py disable | enable | list

Decky Loader keeps every plugin's backend running, with Steam or without it, and a big PS5 game needs
that memory. This asks Decky itself to disable each plugin, the same call as its plugin menu makes, so a
plugin stops the way it does there (no process is killed, Decky is not restarted). Enable undoes exactly
what disable did: a plugin that was already off stays off.

What it turned off is kept in ~/.local/share/KytyPS5-Native/plugins-disabled.json until it is turned back
on, so a game that ended badly (a crash, a power cut) is followed by an enable at the next launch.

~/.config/KytyPS5-Native/keep-plugins lists plugin names to leave on, one a line (a line starting with #
is a comment); a line with * leaves every plugin on, which turns this off. The plugin's Plugins tab writes
it. KytyPS5-Native is always left on.

It never fails a launch: any error is printed and the exit status stays 0.
"""
import asyncio
import json
import os
import sys

import aiohttp

DECKY = os.environ.get("KYTYPS5_DECKY_URL", "http://127.0.0.1:1337")
ALWAYS_KEEP = {"KytyPS5-Native"}
SHARE = os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share"), "KytyPS5-Native")
CONFIG = os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"), "KytyPS5-Native")
STATE = os.path.join(SHARE, "plugins-disabled.json")
KEEP = os.path.join(CONFIG, "keep-plugins")

# Decky's websocket messages (decky_loader/wsrouter.py).
CALL, REPLY, DISCARD, RECEIVED, ERROR = 0, 1, 2, 3, -1


def say(text):
    print(f"decky: {text}", flush=True)


def read_keep():
    keep = set(ALWAYS_KEEP)
    try:
        with open(KEEP) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    keep.add(line)
    except OSError:
        pass
    return keep


def read_state():
    try:
        with open(STATE) as f:
            names = json.load(f)
        return [n for n in names if isinstance(n, str)]
    except (OSError, ValueError):
        return []


def write_state(names):
    if not names:
        try:
            os.remove(STATE)
        except OSError:
            pass
        return
    os.makedirs(SHARE, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(names, f)
    os.replace(tmp, STATE)


class Decky:
    """One websocket to Decky Loader, the way its own frontend talks to it."""

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=3))
        last = None
        # Right after a boot Decky may still be starting.
        for _ in range(10):
            try:
                async with self.session.get(f"{DECKY}/auth/token", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    token = (await r.text()).strip()
                self.ws = await self.session.ws_connect(f"{DECKY}/ws", params={"auth": token})
                self.next_id = 0
                return self
            except (aiohttp.ClientError, OSError) as e:  # a timeout is an OSError
                last = e
                await asyncio.sleep(1)
        await self.session.close()
        raise ConnectionError(f"Decky is not reachable ({last})")

    async def __aexit__(self, *exc):
        await self.ws.close()
        await self.session.close()

    async def call(self, route, *args, timeout=30):
        self.next_id += 1
        call_id = self.next_id
        await self.ws.send_json({"type": CALL, "id": call_id, "route": route, "args": list(args)})

        async def answer():
            async for msg in self.ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = msg.json()
                # Events Decky emits on its own (plugin loaded, disabled...) are not the answer.
                if data.get("id") != call_id or data.get("type") not in (REPLY, DISCARD, ERROR):
                    continue
                if data["type"] == ERROR:
                    raise RuntimeError(data.get("error", {}).get("message") or data.get("error"))
                await self.ws.send_json({"type": RECEIVED, "id": call_id})
                return data.get("result")
            raise ConnectionError("Decky closed the connection")

        return await asyncio.wait_for(answer(), timeout)


async def list_plugins(decky):
    return await decky.call("loader/get_plugins")


async def off_now(decky, plugins):
    """The plugins that are off at this moment, whoever turned them off.

    Decky's own list (disabled) only knows the ones that were off when Decky started: a plugin turned off
    from its menu since is written in its disabled_plugins setting and nowhere else. Both are read, and
    an error is not caught: not knowing what is off, nothing is turned off (a plugin the user turned off
    must never be turned back on).
    """
    off = {p["name"] for p in plugins if p.get("disabled")}
    off.update(await decky.call("utilities/settings/get", "disabled_plugins", []))
    return off


async def disable():
    keep = read_keep()
    if "*" in keep:
        say("keep-plugins has *: every plugin stays on")
        return
    # What a game that ended badly left off is turned back on first, so the snapshot below is the user's.
    await enable()
    async with Decky() as decky:
        plugins = await list_plugins(decky)
        off = await off_now(decky, plugins)
        state = read_state()
        for p in plugins:
            name = p["name"]
            if name in off or name in keep:
                continue
            # Written before the call: a stop half done is still turned back on after the game.
            if name not in state:
                state.append(name)
                write_state(state)
            try:
                await decky.call("utilities/disable_plugin", name)
                say(f"{name}: off")
            except Exception as e:  # noqa: BLE001 - one plugin must not stop the others
                say(f"{name}: could not turn off ({e})")


async def sync_flags(decky):
    """Makes Decky's plugin list say what its disabled_plugins setting says.

    Steam's frontend shows a plugin as off only if that list says so when the frontend starts, and Decky's
    disable call (its menu's, as ours) does not update the list. Steam starts again after every game, so
    a plugin the user had turned off would show as on, as if it had been started again. Reloading a plugin
    that is in the setting makes Decky build it again as disabled, without starting it.
    """
    off = await decky.call("utilities/settings/get", "disabled_plugins", [])
    for p in await list_plugins(decky):
        if p["name"] in off and not p.get("disabled"):
            await decky.call("loader/reload_plugin", p["name"])
            say(f"{p['name']}: kept off")


async def enable():
    state = read_state()
    async with Decky() as decky:
        for name in list(state):
            try:
                await decky.call("utilities/enable_plugin", name)
                say(f"{name}: on")
            except RuntimeError as e:
                # Decky answered: a plugin uninstalled since is not coming back, do not keep asking.
                say(f"{name}: could not turn on ({e})")
            state.remove(name)
            write_state(state)
        await sync_flags(decky)


async def show():
    async with Decky() as decky:
        plugins = await list_plugins(decky)
        off = await off_now(decky, plugins)
        for p in plugins:
            say(f"{p['name']}: {'off' if p['name'] in off else 'on'}")
    say(f"turned off by us: {read_state() or 'none'}")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    work = {"disable": disable, "enable": enable, "list": show}.get(command)
    if work is None:
        print(__doc__)
        return
    try:
        asyncio.run(work())
    except Exception as e:  # noqa: BLE001 - never fail a launch
        say(f"{command}: {e}")


if __name__ == "__main__":
    main()
