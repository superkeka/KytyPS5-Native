# KytyPS5-Native Decky plugin, the root side: kernel settings, KytyPS5-Native and plugin updates from
# GitHub releases, the games folder, boot and exit videos. Steam shortcuts are made by the frontend
# (SteamClient lives there); this side remembers which shortcut belongs to which game.
import asyncio
import base64
import glob
import json
import os
import re
import shutil
import ssl
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

import decky

REPO = "superkeka/KytyPS5-Native"
USER = decky.DECKY_USER
HOME = decky.DECKY_USER_HOME
SHARE = os.path.join(HOME, ".local/share/KytyPS5-Native")
KYTY_DIR = os.path.join(SHARE, "kyty")
CONFIG_DIR = os.path.join(HOME, ".config/KytyPS5-Native")
SETTINGS = os.path.join(decky.DECKY_PLUGIN_SETTINGS_DIR, "settings.json")
ART_CACHE = os.path.join(decky.DECKY_PLUGIN_RUNTIME_DIR, "art")
# Sony's PS5 boot and shutdown animations: not ours to ship, fetched from where fans host them.
VIDEOS = {
    "boot": "https://steamdeckrepo.com/post/download/YWW81",
    "exit": "https://steamdeckrepo.com/post/download/n8v11",
}
# Recommended kernel settings (README, tools/bc250-amdgpu): GTT 14 GiB, KFD queues without CWSR.
GRUB_CFG = "/etc/default/grub.d/kytyps5-native.cfg"
MODPROBE_CONF = "/etc/modprobe.d/kytyps5-native.conf"
GTT_PAGES = "3670016"
AMDGPU = {"sched_policy": "2", "vm_update_mode": "0", "cwsr_enable": "0"}


def _ids():
    import pwd
    pw = pwd.getpwnam(USER)
    return pw.pw_uid, pw.pw_gid


def _chown(path):
    uid, gid = _ids()
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            os.lchown(os.path.join(root, name), uid, gid)
    os.lchown(path, uid, gid)


def _mkdir_user(path):
    os.makedirs(path, exist_ok=True)
    # Every level under the home belongs to the user, not root.
    uid, gid = _ids()
    p = path
    while p.startswith(HOME + "/"):
        os.lchown(p, uid, gid)
        p = os.path.dirname(p)


def _load():
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(s):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, SETTINGS)


def _run(*cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    decky.logger.info("%s -> %d %s", " ".join(cmd), p.returncode, out[-500:])
    return p.returncode, out


def _ssl_context():
    # Decky's bundled Python has no CA certificates of its own: the system's bundle.
    for ca in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem"):
        if os.path.exists(ca):
            return ssl.create_default_context(cafile=ca)
    return ssl.create_default_context()


def _get(url, path=None):
    req = urllib.request.Request(url, headers={"User-Agent": "KytyPS5-Native-Decky"})
    with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
        if path is None:
            return r.read()
        with open(path, "wb") as f:
            shutil.copyfileobj(r, f)
    return None


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _ps5_apu():
    """The PS5's APU (BC-250 1002:13fe, PS5 1002:13da): the native backend, its kernel settings and the
    root helper. Elsewhere Kyty runs on its Vulkan path and needs none of them."""
    for dev in glob.glob("/sys/class/drm/card*/device"):
        if _read(f"{dev}/vendor") == "0x1002" and _read(f"{dev}/device") in ("0x13fe", "0x13da"):
            return True
    return False


def _game_running():
    return _run("pgrep", "-x", "kyty_emulator")[0] == 0


# --- kernel ---------------------------------------------------------------------------------------


def _grub_lines():
    text = ""
    for p in ["/etc/default/grub"] + sorted(glob.glob("/etc/default/grub.d/*")):
        text += _read(p) + "\n"
    return text


def _modprobe_options():
    opts = {}
    for p in sorted(glob.glob("/etc/modprobe.d/*.conf")):
        for line in _read(p).splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "options" and parts[1] == "amdgpu":
                for kv in parts[2:]:
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        opts[k] = (v, p)
    return opts


def _kernel_status():
    cmdline = _read("/proc/cmdline")
    m = re.search(r"ttm\.pages_limit=(\d+)", cmdline)
    gtt_active = m.group(1) if m else ""
    g = re.search(r"ttm\.pages_limit=(\d+)", _grub_lines())
    gtt_conf = g.group(1) if g else ""
    if not gtt_conf and shutil.which("rpm-ostree"):
        gtt_conf = gtt_active
    items = [{
        "key": "gtt", "label": "GTT 14 GiB", "want": GTT_PAGES,
        "active": gtt_active, "configured": gtt_conf,
    }]
    opts = _modprobe_options()
    for k, v in AMDGPU.items():
        items.append({
            "key": k, "label": f"amdgpu {k}={v}", "want": v,
            "active": _read(f"/sys/module/amdgpu/parameters/{k}"),
            "configured": opts.get(k, ("", ""))[0],
            "file": opts.get(k, ("", ""))[1],
        })
    # Reverted in this boot: the old values stay active until a reboot.
    reverted = _load().get("reverted_boot") == _read("/proc/sys/kernel/random/boot_id")
    for it in items:
        it["ok"] = it["active"] == it["want"]
        it["pending"] = not it["ok"] and it["configured"] == it["want"]
        it["reverting"] = reverted and it["ok"] and it["configured"] != it["want"]
    return {
        "items": items,
        "ours": os.path.exists(GRUB_CFG) or os.path.exists(MODPROBE_CONF),
        "reboot": any(it["pending"] or it["reverting"] for it in items),
        "all_ok": all(it["ok"] for it in items),
    }


def _update_boot():
    """Regenerates the boot config after a kernel command line change."""
    if shutil.which("steamos-readonly"):
        was_ro = "enabled" in _run("steamos-readonly", "status")[1]
        if was_ro:
            _run("steamos-readonly", "disable")
        try:
            return _run("update-grub")
        finally:
            if was_ro:
                _run("steamos-readonly", "enable")
    if shutil.which("update-grub"):
        return _run("update-grub")
    if shutil.which("grub-mkconfig") and os.path.exists("/boot/grub/grub.cfg"):
        return _run("grub-mkconfig", "-o", "/boot/grub/grub.cfg")
    return 1, "no update-grub or grub-mkconfig: add ttm.pages_limit=%s to the kernel command line" % GTT_PAGES


# --- games ----------------------------------------------------------------------------------------


def _title(param):
    lp = param.get("localizedParameters") or {}
    names = lp.get(lp.get("defaultLanguage", "")) or lp.get("en-US") or {}
    name = names.get("titleName") or param.get("titleId") or ""
    name = re.sub(r"[®™©]", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _scan(folder):
    games = []
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return games
    for e in entries:
        path = os.path.join(folder, e)
        try:
            with open(os.path.join(path, "sce_sys/param.json"), encoding="utf-8") as f:
                param = json.load(f)
        except (OSError, ValueError):
            continue
        tid = param.get("titleId") or e
        games.append({"title_id": tid, "name": _title(param) or tid, "path": path})
    return games


class Plugin:
    async def _main(self):
        decky.logger.info("KytyPS5-Native plugin %s", decky.DECKY_PLUGIN_VERSION)
        asyncio.get_event_loop().create_task(self._default_videos())

    # The videos are on by default: fetched once, unless the user turned one off. A few tries, the
    # network may come up after Decky.
    async def _default_videos(self):
        for _ in range(10):
            off = set(_load().get("videos_off", []))
            missing = [k for k in VIDEOS if k not in off and not os.path.exists(os.path.join(CONFIG_DIR, k + ".webm"))]
            if not missing:
                return
            for k in missing:
                if os.path.exists(os.path.join(CONFIG_DIR, k + ".webm.off")):
                    continue
                r = await self.set_video(k, True)
                decky.logger.info("default %s video: %s", k, r)
            await asyncio.sleep(60)

    async def _unload(self):
        pass

    # --- state ------------------------------------------------------------------------------------

    async def get_state(self):
        s = _load()
        folder = s.get("games_dir") or os.path.join(HOME, "Games")
        games = _scan(folder)
        known = s.get("games", {})
        for g in games:
            g.update({"appid": 0, "keep_steam": False, **known.get(g["title_id"], {})})
        return {
            "kyty_dir": KYTY_DIR,
            "runner": os.path.join(KYTY_DIR, "kytyps5-run.sh"),
            "kyty_version": _read(os.path.join(KYTY_DIR, "VERSION")),
            "kyty_installed": os.access(os.path.join(KYTY_DIR, "kyty_emulator"), os.X_OK),
            "helper": _run("systemctl", "is-active", "kyty-lodhw.service")[1] == "active",
            "plugin_version": decky.DECKY_PLUGIN_VERSION,
            "plugin_release": _read(os.path.join(decky.DECKY_PLUGIN_DIR, "RELEASE")),
            "games_dir": folder,
            "games": games,
            "gone": {k: v for k, v in known.items() if k not in {g["title_id"] for g in games} and v.get("appid")},
            "videos": {k: os.path.exists(os.path.join(CONFIG_DIR, k + ".webm")) for k in VIDEOS},
            "ps5_apu": _ps5_apu(),
            "kernel": _kernel_status(),
            "game_running": _game_running(),
        }

    async def latest_release(self):
        def fetch():
            rel = json.loads(_get(f"https://api.github.com/repos/{REPO}/releases/latest"))
            assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
            kyty = next((u for n, u in assets.items() if n.endswith("-Linux-x86_64.tar.gz")), "")
            plugin = next((u for n, u in assets.items() if n.endswith("-Decky.zip")), "")
            return {"tag": rel.get("tag_name", ""), "kyty_url": kyty, "plugin_url": plugin}
        try:
            return await asyncio.to_thread(fetch)
        except Exception as e:
            decky.logger.error("latest release: %s", e)
            return {"tag": "", "error": str(e)}

    # --- KytyPS5-Native ---------------------------------------------------------------------------

    async def install_kyty(self, url, tag):
        if _game_running():
            return {"ok": False, "log": "a game is running"}

        def work():
            _mkdir_user(SHARE)
            with tempfile.TemporaryDirectory(dir=SHARE) as tmp:
                archive = os.path.join(tmp, "kyty.tar.gz")
                _get(url, archive)
                new = os.path.join(tmp, "kyty")
                with tarfile.open(archive) as t:
                    try:
                        t.extractall(new, filter="data")
                    except TypeError:
                        t.extractall(new)
                with open(os.path.join(new, "VERSION"), "w") as f:
                    f.write(tag + "\n")
                # Kyty keeps saves, trophies and download data in its working folder, its own folder when
                # its launcher starts it. They live in data/ (an update replaces kyty/ whole), linked here
                # so Kyty run that way and the shortcuts (kytyps5-run.sh works in data/) share them.
                for d in ("_SaveData", "_Trophies", "_DownloadData"):
                    os.makedirs(os.path.join(SHARE, "data", d), exist_ok=True)
                    if not os.path.lexists(os.path.join(new, d)):
                        os.symlink(os.path.join("..", "data", d), os.path.join(new, d))
                _chown(os.path.join(SHARE, "data"))
                _chown(new)
                old = KYTY_DIR + ".old"
                shutil.rmtree(old, ignore_errors=True)
                if os.path.exists(KYTY_DIR):
                    os.rename(KYTY_DIR, old)
                os.rename(new, KYTY_DIR)
                shutil.rmtree(old, ignore_errors=True)
            if not _ps5_apu():
                return {"ok": True, "log": ""}
            # The root helper: the socket's group is the user's.
            env = dict(os.environ, SUDO_GID=str(_ids()[1]))
            p = subprocess.run([os.path.join(KYTY_DIR, "install-helper.sh")], capture_output=True, text=True, env=env)
            return {"ok": True, "log": (p.stdout + p.stderr).strip()[-400:]}
        try:
            return await asyncio.to_thread(work)
        except Exception as e:
            decky.logger.exception("install")
            return {"ok": False, "log": str(e)}

    async def install_plugin(self, url, tag):
        def work():
            with tempfile.TemporaryDirectory() as tmp:
                z = os.path.join(tmp, "plugin.zip")
                _get(url, z)
                with zipfile.ZipFile(z) as f:
                    f.extractall(tmp)
                src = next(os.path.join(tmp, d) for d in os.listdir(tmp)
                           if os.path.exists(os.path.join(tmp, d, "plugin.json")))
                with open(os.path.join(src, "RELEASE"), "w") as f:
                    f.write(tag + "\n")
                dst = decky.DECKY_PLUGIN_DIR
                old = dst + ".old"
                shutil.rmtree(old, ignore_errors=True)
                shutil.copytree(src, dst + ".new")
                os.rename(dst, old)
                os.rename(dst + ".new", dst)
                shutil.rmtree(old, ignore_errors=True)
            # Decky loads the new version on its restart; this process goes with it.
            subprocess.Popen(["sh", "-c", "sleep 2; systemctl restart plugin_loader"], start_new_session=True)
            return {"ok": True}
        try:
            return await asyncio.to_thread(work)
        except Exception as e:
            decky.logger.exception("plugin update")
            return {"ok": False, "log": str(e)}

    # --- kernel -----------------------------------------------------------------------------------

    async def kernel_apply(self):
        def work():
            log = []
            st = _kernel_status()
            gtt = st["items"][0]
            if gtt["configured"] != GTT_PAGES and not gtt["ok"]:
                if shutil.which("rpm-ostree"):
                    log.append(_run("rpm-ostree", "kargs", f"--append-if-missing=ttm.pages_limit={GTT_PAGES}")[1])
                else:
                    with open(GRUB_CFG, "w") as f:
                        f.write("# KytyPS5-Native (Decky plugin): GTT 14 GiB for PS5 games.\n"
                                f'GRUB_CMDLINE_LINUX="${{GRUB_CMDLINE_LINUX}} ttm.pages_limit={GTT_PAGES}"\n')
                    code, out = _update_boot()
                    log.append(out[-300:])
                    if code != 0:
                        return {"ok": False, "log": "\n".join(log)}
            mine = {}
            for it in st["items"][1:]:
                if it["configured"] and it["configured"] != it["want"] and it.get("file") != MODPROBE_CONF:
                    log.append(f"{it['key']}={it['configured']} is set in {it['file']}: left as it is")
                elif it["configured"] != it["want"] or it.get("file") == MODPROBE_CONF:
                    mine[it["key"]] = it["want"]
            s = _load()
            s.pop("reverted_boot", None)
            _save(s)
            if mine:
                with open(MODPROBE_CONF, "w") as f:
                    f.write("# KytyPS5-Native (Decky plugin): KFD queues for PS5 games.\n"
                            "options amdgpu " + " ".join(f"{k}={v}" for k, v in mine.items()) + "\n")
            return {"ok": True, "log": "\n".join(x for x in log if x)}
        return await asyncio.to_thread(work)

    async def kernel_revert(self):
        def work():
            log = []
            if os.path.exists(MODPROBE_CONF):
                os.remove(MODPROBE_CONF)
            if shutil.which("rpm-ostree"):
                log.append(_run("rpm-ostree", "kargs", f"--delete-if-present=ttm.pages_limit={GTT_PAGES}")[1])
            elif os.path.exists(GRUB_CFG):
                os.remove(GRUB_CFG)
                log.append(_update_boot()[1][-300:])
            s = _load()
            s["reverted_boot"] = _read("/proc/sys/kernel/random/boot_id")
            _save(s)
            return {"ok": True, "log": "\n".join(x for x in log if x)}
        return await asyncio.to_thread(work)

    async def reboot(self):
        subprocess.Popen(["sh", "-c", "sleep 1; systemctl reboot"], start_new_session=True)

    # --- games ------------------------------------------------------------------------------------

    async def set_games_dir(self, path):
        s = _load()
        s["games_dir"] = path
        _save(s)

    async def set_game(self, title_id, appid, keep_steam):
        s = _load()
        games = s.setdefault("games", {})
        if appid:
            games[title_id] = {"appid": appid, "keep_steam": bool(keep_steam)}
        else:
            games.pop(title_id, None)
        _save(s)

    # Artwork made once per game (the frontend composes it) and kept: a shortcut added again reuses it.
    async def get_cached_art(self, title_id):
        def work():
            out = {}
            for kind in ("capsule", "hero", "wide"):
                try:
                    with open(os.path.join(ART_CACHE, title_id, kind + ".png"), "rb") as f:
                        out[kind] = base64.b64encode(f.read()).decode()
                except OSError:
                    return {}
            return out
        return await asyncio.to_thread(work)

    async def save_art(self, title_id, kind, data):
        def work():
            d = os.path.join(ART_CACHE, title_id)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, kind + ".png"), "wb") as f:
                f.write(base64.b64decode(data))
        await asyncio.to_thread(work)

    async def get_art(self, path):
        def b64(name):
            try:
                with open(os.path.join(path, "sce_sys", name), "rb") as f:
                    return base64.b64encode(f.read()).decode()
            except OSError:
                return ""
        return await asyncio.to_thread(lambda: {"icon": b64("icon0.png"), "pic": b64("pic0.png")})

    # --- videos -----------------------------------------------------------------------------------

    async def set_video(self, kind, on):
        path = os.path.join(CONFIG_DIR, kind + ".webm")
        off = path + ".off"

        s = _load()
        off = set(s.get("videos_off", []))
        off.discard(kind) if on else off.add(kind)
        s["videos_off"] = sorted(off)
        _save(s)

        def work():
            _mkdir_user(CONFIG_DIR)
            if not on:
                if os.path.exists(path):
                    os.replace(path, off)
                return {"ok": True}
            if os.path.exists(off):
                os.replace(off, path)
            elif not os.path.exists(path):
                tmp = path + ".part"
                _get(VIDEOS[kind], tmp)
                os.replace(tmp, path)
            _chown(CONFIG_DIR)
            return {"ok": True}
        try:
            return await asyncio.to_thread(work)
        except Exception as e:
            decky.logger.exception("video")
            return {"ok": False, "log": str(e)}
