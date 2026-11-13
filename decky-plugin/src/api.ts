// What the plugin's panel and its page share: the backend's calls, the state they show and the
// shortcut sync.
import { callable, toaster } from "@decky/api";
import { useEffect, useState } from "react";
import { ART_VERSION, Game, languageChanged, needsSync, syncAll } from "./shortcuts";

export interface KernelItem {
  key: string;
  label: string;
  ok: boolean;
  pending: boolean;
  reverting: boolean;
  configured: string;
  file?: string;
}

export interface State {
  kyty_dir: string;
  runner: string;
  kyty_version: string;
  kyty_installed: boolean;
  helper: boolean;
  plugin_version: string;
  plugin_release: string;
  home: string;
  games_dir: string;
  games_dir_state: "ok" | "empty" | "missing";
  art_version: number;
  ps5_apu: boolean;
  games: Game[];
  gone: Record<string, { appid: number }>;
  videos: Record<string, boolean>;
  hdr: boolean;
  kernel: { items: KernelItem[]; method: "files" | "rpm-ostree" | "limine" | "unsupported"; ours: boolean; reboot: boolean; all_ok: boolean };
  game_running: boolean;
}

export interface Release {
  tag: string;
  kyty_url?: string;
  plugin_url?: string;
  error?: string;
}

export interface Result {
  ok: boolean;
  log?: string;
}

// An add-on (DLC or licence) of a game: found in the game's executable, or a line the user wrote
// into the game's add-ons file (with the folder of its data, if it has any).
export interface Addon {
  label: string;
  on: boolean;
  found: boolean;
  folder: string;
}

export interface GameInfo {
  title_id: string;
  name: string;
  path: string;
  size: number;
  version: string;
  addons: Addon[];
  addons_known: boolean;
  game_running: boolean;
}

export const getState = callable<[], State>("get_state");
export const latestRelease = callable<[], Release>("latest_release");
export const installKyty = callable<[url: string, tag: string], Result>("install_kyty");
const installPlugin = callable<[url: string, tag: string], Result>("install_plugin");
export const kernelApply = callable<[], Result>("kernel_apply");
export const kernelRevert = callable<[], Result>("kernel_revert");
export const reboot = callable<[], void>("reboot");
export const setGamesDir = callable<[path: string], void>("set_games_dir");
export const setVideo = callable<[kind: string, on: boolean], Result>("set_video");
export const setHdr = callable<[on: boolean], Result>("set_hdr");
const setArtVersion = callable<[version: number], void>("set_art_version");
export const getGame = callable<[titleId: string], GameInfo | null>("get_game");
export const setAddon = callable<[titleId: string, label: string, on: boolean], Result>("set_addon");
export const findAddons = callable<[titleId: string], Result>("find_addons");

export const toast = (body: string, title = "KytyPS5-Native") => toaster.toast({ title, body });
// Release tags without the project name: 2026-10-29-6957c86.
export const short = (tag: string) => tag.replace(/^KytyPS5-Native-/, "");
export const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;

// Decky's own installer (as its store): asks, swaps the files and reloads this plugin alone. Without
// it, the backend swaps them and restarts Decky.
export async function updatePlugin(release: Release) {
  const decky = (window as any).DeckyBackend;
  if (decky?.call) {
    await decky.call("utilities/install_plugin", release.plugin_url!, "KytyPS5-Native", release.tag, "", 2);
    // This panel is the old version's: wait for the new one (the user confirms Decky's prompt first)
    // and have Decky show it in its place, as when the panel is opened.
    for (let i = 0; i < 60; i++) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        if ((await getState()).plugin_release === release.tag) {
          toast(`Plugin updated to ${short(release.tag)}`);
          const decky = (window as any).DeckyPluginLoader?.deckyState;
          if (decky?.publicState?.().activePlugin?.name === "KytyPS5-Native") decky.setActivePlugin("KytyPS5-Native");
          return;
        }
      } catch {
        // the backend restarts with the new version
      }
    }
    return;
  }
  const r = await installPlugin(release.plugin_url!, release.tag);
  toast(r.ok ? "Updated, Decky restarts" : r.log || "failed");
}

let syncRunning = false;
async function sync(s: State) {
  if (syncRunning) return;
  syncRunning = true;
  try {
    const redoArt = s.art_version !== ART_VERSION;
    await syncAll(s.games, s.gone, s.runner, s.kyty_dir, redoArt);
    if (redoArt) await setArtVersion(ART_VERSION);
  } catch (e) {
    console.error("KytyPS5-Native sync", e);
  }
  syncRunning = false;
}

const wantsSync = async (s: State) =>
  s.kyty_installed && (needsSync(s.games, s.gone) || s.art_version !== ART_VERSION || (await languageChanged()));

// Also when nothing of the plugin is open: at start and every 10 minutes.
export async function autoSync() {
  const s = await getState();
  if (await wantsSync(s)) await sync(s);
}

// The plugin's state for a view: read when the view opens, and again after each action.
export function usePlugin() {
  const [state, setState] = useState<State | null>(null);
  const [release, setRelease] = useState<Release | null>(null);
  const [busy, setBusy] = useState("");
  const [syncing, setSyncing] = useState(false);

  // New games in the folder (and the folder itself) get their shortcuts on their own.
  const refresh = async () => {
    const s = await getState();
    setState(s);
    if ((await wantsSync(s)) && !syncRunning) {
      setSyncing(true);
      await sync(s);
      setSyncing(false);
      setState(await getState());
    }
  };
  useEffect(() => {
    refresh();
    latestRelease().then(setRelease);
  }, []);

  const run = async (what: string, work: () => Promise<void>) => {
    setBusy(what);
    try {
      await work();
    } catch (e) {
      toast(String(e), what);
    }
    setBusy("");
    await refresh();
  };

  return { state, release, setRelease, busy, syncing, refresh, run };
}

export type PluginView = ReturnType<typeof usePlugin>;
