import {
  ButtonItem,
  ConfirmModal,
  Field,
  PanelSection,
  PanelSectionRow,
  ToggleField,
  showModal,
  staticClasses,
} from "@decky/ui";
import { callable, definePlugin, openFilePicker, toaster } from "@decky/api";
import { useEffect, useState } from "react";
import { FaPlaystation } from "react-icons/fa";
import { Game, needsSync, syncAll } from "./shortcuts";

interface KernelItem {
  key: string;
  label: string;
  ok: boolean;
  pending: boolean;
  reverting: boolean;
  configured: string;
  file?: string;
}

interface State {
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
  ps5_apu: boolean;
  games: Game[];
  gone: Record<string, { appid: number }>;
  videos: Record<string, boolean>;
  kernel: { items: KernelItem[]; ours: boolean; reboot: boolean; all_ok: boolean };
  game_running: boolean;
}

interface Release {
  tag: string;
  kyty_url?: string;
  plugin_url?: string;
  error?: string;
}

interface Result {
  ok: boolean;
  log?: string;
}

const getState = callable<[], State>("get_state");
const latestRelease = callable<[], Release>("latest_release");
const installKyty = callable<[url: string, tag: string], Result>("install_kyty");
const installPlugin = callable<[url: string, tag: string], Result>("install_plugin");
const kernelApply = callable<[], Result>("kernel_apply");
const kernelRevert = callable<[], Result>("kernel_revert");
const reboot = callable<[], void>("reboot");
const setGamesDir = callable<[path: string], void>("set_games_dir");
const setVideo = callable<[kind: string, on: boolean], Result>("set_video");

const toast = (body: string, title = "KytyPS5-Native") => toaster.toast({ title, body });
// Release tags without the project name: 2026-10-29-6957c86.
const short = (tag: string) => tag.replace(/^KytyPS5-Native-/, "");

// Decky's own installer (as its store): asks, swaps the files and reloads this plugin alone. Without
// it, the backend swaps them and restarts Decky.
async function updatePlugin(release: Release) {
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

const confirm = (title: string, text: string, ok: string, action: () => void) =>
  showModal(<ConfirmModal strTitle={title} strDescription={text} strOKButtonText={ok} onOK={action} />);

function Content() {
  const [state, setState] = useState<State | null>(null);
  const [release, setRelease] = useState<Release | null>(null);
  const [busy, setBusy] = useState("");

  const [syncing, setSyncing] = useState(false);
  // New games in the folder (and the folder itself) get their shortcuts on their own.
  const refresh = async () => {
    const s = await getState();
    setState(s);
    if (s.kyty_installed && needsSync(s.games, s.gone) && !syncRunning) {
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

  if (!state) return <PanelSection><PanelSectionRow><Field label="Loading..." /></PanelSectionRow></PanelSection>;

  const kytyUpdate = !!release?.kyty_url && release.tag !== state.kyty_version;
  const pluginUpdate = !!release?.plugin_url && !!state.plugin_release && release.tag !== state.plugin_release;
  const kernelText = (it: KernelItem) =>
    it.reverting ? "on until reboot" : it.ok ? "on" : it.pending ? "after reboot" : it.configured ? `set to ${it.configured}` : "off";

  return (
    <>
      <PanelSection title="KytyPS5-Native">
        <PanelSectionRow>
          <Field
            label="Version"
            description={
              !state.ps5_apu ? "Vulkan path (native mode needs the PS5's APU)" : state.helper ? "Native mode, root helper running" : "Root helper not running"
            }
          >
            {state.kyty_installed ? short(state.kyty_version) || "installed" : "not installed"}
          </Field>
        </PanelSectionRow>
        {!state.kyty_installed && (
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              disabled={!!busy || state.game_running}
              onClick={() =>
                run("Installing KytyPS5-Native", async () => {
                  const r = await installKyty(release!.kyty_url!, release!.tag);
                  toast(r.ok ? `Installed ${short(release!.tag)}` : r.log || "failed");
                })
              }
            >
              {busy === "Installing KytyPS5-Native" ? "Installing..." : "Install"}
            </ButtonItem>
          </PanelSectionRow>
        )}
        {state.ps5_apu && (
          <PanelSectionRow>
            {state.kernel.reboot ? (
              <ButtonItem
                layout="below"
                description="Reboot to apply the new config"
                onClick={() => confirm("Reboot", "Reboot now for the kernel config to take effect?", "Reboot", () => reboot())}
              >
                Kernel config
              </ButtonItem>
            ) : !state.kernel.all_ok ? (
              <ButtonItem
                layout="below"
                description="Not set: press to apply the recommended config"
                disabled={!!busy}
                onClick={() =>
                  confirm(
                    "Recommended kernel config",
                    "Kernel options only, the kernel stays the same: GTT 14 GiB (ttm.pages_limit, big games need it) and " +
                      "amdgpu sched_policy=2 vm_update_mode=0 cwsr_enable=0 (the GPU queues PS5 games use). " +
                      "Now: " + state.kernel.items.map((it) => `${it.label} ${kernelText(it)}`).join(", ") + ". " +
                      "Files: /etc/default/grub.d/kytyps5-native.cfg, /etc/modprobe.d/kytyps5-native.conf. " +
                      "Takes effect after a reboot; Revert removes them.",
                    "Apply",
                    () =>
                      run("Kernel", async () => {
                        const r = await kernelApply();
                        toast(r.ok ? "Applied, reboot to use it" : r.log || "failed");
                      }),
                  )
                }
              >
                Kernel config
              </ButtonItem>
            ) : state.kernel.ours ? (
              <ButtonItem
                layout="below"
                description="Recommended: press to revert"
                disabled={!!busy}
                onClick={() =>
                  confirm("Revert", "Remove the kernel options this plugin added? Takes effect after a reboot.", "Revert", () =>
                    run("Kernel", async () => {
                      const r = await kernelRevert();
                      toast(r.ok ? "Reverted, reboot to finish" : r.log || "failed");
                    }),
                  )
                }
              >
                Kernel config
              </ButtonItem>
            ) : (
              <Field label="Kernel config" description="Recommended (set by hand)" focusable />
            )}
          </PanelSectionRow>
        )}
        {state.kyty_installed && (
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              description={
                kytyUpdate || pluginUpdate
                  ? `${short(release!.tag)} is out: press to update`
                  : `Plugin ${short(state.plugin_release) || state.plugin_version}: press to check`
              }
              disabled={!!busy || ((kytyUpdate || pluginUpdate) && state.game_running)}
              onClick={() =>
                kytyUpdate || pluginUpdate
                  ? run("Update", async () => {
                      if (kytyUpdate) {
                        const r = await installKyty(release!.kyty_url!, release!.tag);
                        if (!r.ok) {
                          toast(r.log || "failed");
                          return;
                        }
                        toast(`KytyPS5-Native ${short(release!.tag)} installed`);
                      }
                      if (pluginUpdate) await updatePlugin(release!);
                    })
                  : run("Check", async () => {
                      const r = await latestRelease();
                      setRelease(r);
                      const kyty = !!r.kyty_url && r.tag !== state.kyty_version;
                      const plugin = !!r.plugin_url && !!state.plugin_release && r.tag !== state.plugin_release;
                      toast(r.error ? `No connection to GitHub: ${r.error}` : kyty || plugin ? `${short(r.tag)} is out` : "Up to date");
                    })
              }
            >
              {busy === "Update" ? "Updating..." : busy === "Check" ? "Checking..." : "Updates"}
            </ButtonItem>
          </PanelSectionRow>
        )}
        {release?.error && (
          <PanelSectionRow>
            <Field label="No connection to GitHub" description={release.error} />
          </PanelSectionRow>
        )}
      </PanelSection>


      <PanelSection title="Games">
        <PanelSectionRow>
          <ButtonItem
            layout="below"
            description={state.games_dir}
            disabled={!!busy}
            onClick={async () => {
              const start = state.games_dir_state === "missing" ? state.home : state.games_dir;
              try {
                const r = await openFilePicker(1 as any, start, false, true);
                await setGamesDir(r.realpath || r.path);
              } catch {
                return; // closed without a choice
              }
              await refresh();
            }}
          >
            Games folder
          </ButtonItem>
        </PanelSectionRow>
        <PanelSectionRow>
          <Field
            label={
              !state.kyty_installed
                ? "Install KytyPS5-Native first"
                : syncing
                  ? "Adding to Steam..."
                  : `${state.games.length} ${state.games.length === 1 ? "game" : "games"} in the library`
            }
            description={
              state.games_dir_state === "missing"
                ? "Games folder not found"
                : state.games_dir_state === "empty"
                  ? "No games in the folder"
                  : state.games.length
                    ? `Synced to the Non-Steam library on their own. Total ${(state.games.reduce((sum, g) => sum + g.size, 0) / 1e9).toFixed(1)} GB`
                    : "No games found: a game is a folder with sce_sys/param.json in it"
            }
          />
        </PanelSectionRow>
      </PanelSection>

      <PanelSection title="Animations">
        {(["boot", "exit"] as const).map((kind) => (
          <PanelSectionRow key={kind}>
            <ToggleField
              label={kind === "boot" ? "Boot animation" : "Exit animation"}
              description={kind === "boot" ? "Before the game starts" : "When you leave the game"}
              checked={state.videos[kind]}
              disabled={!!busy}
              onChange={(on) =>
                run("Video", async () => {
                  const r = await setVideo(kind, on);
                  if (!r.ok) toast(r.log || "failed");
                })
              }
            />
          </PanelSectionRow>
        ))}
      </PanelSection>

    </>
  );
}

let syncRunning = false;
async function sync(s: State) {
  if (syncRunning) return;
  syncRunning = true;
  try {
    await syncAll(s.games, s.gone, s.runner, s.kyty_dir);
  } catch (e) {
    console.error("KytyPS5-Native sync", e);
  }
  syncRunning = false;
}

// Also when the panel is closed: at start and every 10 minutes.
async function autoSync() {
  const s = await getState();
  if (s.kyty_installed && needsSync(s.games, s.gone)) await sync(s);
}

export default definePlugin(() => {
  const auto = () => autoSync().catch((e) => console.error("KytyPS5-Native sync", e));
  setTimeout(auto, 15000);
  const timer = setInterval(auto, 10 * 60 * 1000);
  return {
    name: "KytyPS5-Native",
    titleView: <div className={staticClasses.Title}>KytyPS5-Native</div>,
    content: <Content />,
    icon: <FaPlaystation />,
    onDismount() {
      clearInterval(timer);
    },
  };
});
