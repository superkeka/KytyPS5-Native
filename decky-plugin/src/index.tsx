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
import { Game, needsSync, syncAll, syncGame } from "./shortcuts";

interface KernelItem {
  key: string;
  label: string;
  ok: boolean;
  pending: boolean;
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
  games_dir: string;
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

const confirm = (title: string, text: string, ok: string, action: () => void) =>
  showModal(<ConfirmModal strTitle={title} strDescription={text} strOKButtonText={ok} onOK={action} />);

function Content() {
  const [state, setState] = useState<State | null>(null);
  const [release, setRelease] = useState<Release | null>(null);
  const [busy, setBusy] = useState("");

  const refresh = async () => setState(await getState());
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
    it.ok ? "on" : it.pending ? "after reboot" : it.configured ? `set to ${it.configured}` : "off";

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
            {state.kyty_installed ? state.kyty_version || "installed" : "not installed"}
          </Field>
        </PanelSectionRow>
        {(kytyUpdate || !state.kyty_installed) && (
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              disabled={!!busy || state.game_running}
              onClick={() =>
                run("Installing KytyPS5-Native", async () => {
                  const r = await installKyty(release!.kyty_url!, release!.tag);
                  toast(r.ok ? `Installed ${release!.tag}` : r.log || "failed");
                })
              }
            >
              {busy === "Installing KytyPS5-Native" ? "Installing..." : state.kyty_installed ? `Update to ${release?.tag}` : "Install"}
            </ButtonItem>
          </PanelSectionRow>
        )}
        {release?.error && (
          <PanelSectionRow>
            <Field label="No connection to GitHub" description={release.error} />
          </PanelSectionRow>
        )}
      </PanelSection>

      {state.ps5_apu && <PanelSection title="Kernel">
        {state.kernel.items.map((it) => (
          <PanelSectionRow key={it.key}>
            <Field label={it.label} description={!it.ok && it.file && !it.pending ? it.file : undefined}>
              {kernelText(it)}
            </Field>
          </PanelSectionRow>
        ))}
        {!state.kernel.all_ok && !state.kernel.reboot && (
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              disabled={!!busy}
              onClick={() =>
                confirm(
                  "Recommended kernel settings",
                  "Sets GTT to 14 GiB (big games need it) and amdgpu sched_policy=2 vm_update_mode=0 cwsr_enable=0 " +
                    "(the GPU queues PS5 games use). Files: /etc/default/grub.d/kytyps5-native.cfg, " +
                    "/etc/modprobe.d/kytyps5-native.conf. Takes effect after a reboot; Revert removes them.",
                  "Apply",
                  () =>
                    run("Kernel", async () => {
                      const r = await kernelApply();
                      toast(r.ok ? "Applied, reboot to use them" : r.log || "failed");
                    }),
                )
              }
            >
              Apply recommended settings
            </ButtonItem>
          </PanelSectionRow>
        )}
        {state.kernel.reboot && (
          <PanelSectionRow>
            <ButtonItem layout="below" onClick={() => confirm("Reboot", "Reboot now to use the new kernel settings?", "Reboot", () => reboot())}>
              Reboot
            </ButtonItem>
          </PanelSectionRow>
        )}
        {state.kernel.ours && (
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              disabled={!!busy}
              onClick={() =>
                confirm("Revert", "Remove the kernel settings this plugin added? Takes effect after a reboot.", "Revert", () =>
                  run("Kernel", async () => {
                    const r = await kernelRevert();
                    toast(r.ok ? "Reverted, reboot to finish" : r.log || "failed");
                  }),
                )
              }
            >
              Revert kernel settings
            </ButtonItem>
          </PanelSectionRow>
        )}
      </PanelSection>}

      <PanelSection title="Games">
        <PanelSectionRow>
          <ButtonItem
            layout="below"
            description={state.games_dir}
            disabled={!!busy}
            onClick={async () => {
              const r = await openFilePicker(1 as any, state.games_dir, false, true);
              await setGamesDir(r.realpath || r.path);
              await refresh();
            }}
          >
            Games folder
          </ButtonItem>
        </PanelSectionRow>
        <PanelSectionRow>
          <ButtonItem
            layout="below"
            disabled={!!busy || !state.kyty_installed}
            onClick={() =>
              run("Shortcuts", async () => {
                await syncAll(state.games, state.gone, state.runner, state.kyty_dir);
                toast(`${state.games.length} games in Steam`);
              })
            }
          >
            {busy === "Shortcuts" ? "Adding..." : "Add games to Steam"}
          </ButtonItem>
        </PanelSectionRow>
        {state.games.map((g) => (
          <PanelSectionRow key={g.title_id}>
            <ToggleField
              label={g.name}
              description="Unload Steam while playing (frees 2-3 GB)"
              checked={!g.keep_steam}
              disabled={!!busy || !g.appid}
              onChange={(unload) =>
                run("Shortcut", async () => {
                  await syncGame({ ...g, keep_steam: !unload }, state.runner, state.kyty_dir);
                })
              }
            />
          </PanelSectionRow>
        ))}
        {state.games.length === 0 && (
          <PanelSectionRow>
            <Field label="No games found" description="A game is a folder with sce_sys/param.json in it." />
          </PanelSectionRow>
        )}
      </PanelSection>

      <PanelSection title="Videos">
        {(["boot", "exit"] as const).map((kind) => (
          <PanelSectionRow key={kind}>
            <ToggleField
              label={kind === "boot" ? "Boot video" : "Exit video"}
              description={kind === "boot" ? "PS5 boot animation before the game" : "PS5 shutdown animation when you leave the game"}
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

      <PanelSection title="Plugin">
        <PanelSectionRow>
          <Field label="Version">{state.plugin_release || state.plugin_version}</Field>
        </PanelSectionRow>
        {pluginUpdate && (
          <PanelSectionRow>
            <ButtonItem
              layout="below"
              disabled={!!busy}
              onClick={() =>
                run("Plugin", async () => {
                  const r = await installPlugin(release!.plugin_url!, release!.tag);
                  toast(r.ok ? "Updated, Decky restarts" : r.log || "failed");
                })
              }
            >
              Update plugin
            </ButtonItem>
          </PanelSectionRow>
        )}
      </PanelSection>
    </>
  );
}

// New games in the folder get their shortcut on their own (at start and every 10 minutes).
async function autoSync() {
  try {
    const s = await getState();
    if (s.kyty_installed && needsSync(s.games, s.gone)) await syncAll(s.games, s.gone, s.runner, s.kyty_dir);
  } catch (e) {
    console.error("KytyPS5-Native sync", e);
  }
}

export default definePlugin(() => {
  setTimeout(autoSync, 15000);
  const timer = setInterval(autoSync, 10 * 60 * 1000);
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
