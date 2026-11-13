import { ButtonItem, ConfirmModal, Field, PanelSection, PanelSectionRow, showModal, staticClasses } from "@decky/ui";
import { definePlugin, routerHook } from "@decky/api";
import { FaPlaystation } from "react-icons/fa";
import {
  KernelItem,
  autoSync,
  installKyty,
  kernelApply,
  kernelRevert,
  latestRelease,
  reboot,
  short,
  toast,
  updatePlugin,
  usePlugin,
} from "./api";
import { GAME_PAGE, GamePage, PAGE_ROUTE, Page, openPage } from "./page";

const confirm = (title: string, text: string, ok: string, action: () => void) =>
  showModal(<ConfirmModal strTitle={title} strDescription={text} strOKButtonText={ok} onOK={action} />);

// The side panel: KytyPS5-Native itself (version, kernel config, updates) and the way to the plugin's
// page, where the games and the display settings are.
function Content() {
  const { state, release, setRelease, busy, syncing, run } = usePlugin();

  if (!state) return <PanelSection><PanelSectionRow><Field label="Loading..." /></PanelSectionRow></PanelSection>;

  const kytyUpdate = !!release?.kyty_url && release.tag !== state.kyty_version;
  const pluginUpdate = !!release?.plugin_url && !!state.plugin_release && release.tag !== state.plugin_release;
  const kernelText = (it: KernelItem) =>
    it.reverting ? "on until reboot" : it.ok ? "on" : it.pending ? "after reboot" : it.configured ? `set to ${it.configured}` : "off";

  return (
    <>
      <PanelSection>
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
            {state.kernel.method === "unsupported" ? (
              <ButtonItem layout="below" description="Not supported on this OS: set it by hand (README)" disabled>
                Kernel config
              </ButtonItem>
            ) : state.kernel.reboot ? (
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
                description="Recommended config is off. Press to apply it"
                disabled={!!busy}
                onClick={() =>
                  confirm(
                    "Recommended kernel config",
                    "Kernel options only, the kernel stays the same: GTT 14 GiB (ttm.pages_limit, big games need it) and " +
                      "amdgpu sched_policy=2 vm_update_mode=0 cwsr_enable=0 (the GPU queues PS5 games use). " +
                      "Now: " + state.kernel.items.map((it) => `${it.label} ${kernelText(it)}`).join(", ") + ". " +
                      (state.kernel.method === "rpm-ostree"
                        ? "Set as kernel arguments with rpm-ostree kargs (a new deployment). "
                        : state.kernel.method === "limine"
                          ? "Set as kernel arguments in /etc/limine-entry-tool.d/kytyps5-native.conf, then limine-update. "
                          : "Files: /etc/default/grub.d/kytyps5-native.cfg, /etc/modprobe.d/kytyps5-native.conf. ") +
                      "Takes effect after a reboot; Revert removes them.",
                    "Apply",
                    () =>
                      run("Kernel", async () => {
                        const r = await kernelApply();
                        toast(r.ok ? ["Applied, reboot to use it", r.log].filter(Boolean).join(". ") : r.log || "failed");
                      }),
                  )
                }
              >
                Kernel config
              </ButtonItem>
            ) : state.kernel.ours ? (
              <ButtonItem
                layout="below"
                description="Recommended config is on. Press to revert it"
                disabled={!!busy}
                onClick={() =>
                  confirm("Revert", "Remove the kernel options this plugin added (only those)? Takes effect after a reboot.", "Revert", () =>
                    run("Kernel", async () => {
                      const r = await kernelRevert();
                      toast(r.ok ? ["Reverted, reboot to finish", r.log].filter(Boolean).join(". ") : r.log || "failed");
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
            description={
              !state.kyty_installed
                ? "Install KytyPS5-Native first"
                : syncing
                  ? "Adding to Steam..."
                  : `${state.games.length} ${state.games.length === 1 ? "game" : "games"} in the library: each one's menu, the games folder`
            }
            onClick={() => openPage()}
          >
            Games
          </ButtonItem>
        </PanelSectionRow>
        <PanelSectionRow>
          <ButtonItem layout="below" description="HDR, boot and exit animations" onClick={() => openPage("display")}>
            Display
          </ButtonItem>
        </PanelSectionRow>
      </PanelSection>
    </>
  );
}

export default definePlugin(() => {
  const auto = () => autoSync().catch((e) => console.error("KytyPS5-Native sync", e));
  setTimeout(auto, 15000);
  const timer = setInterval(auto, 10 * 60 * 1000);
  routerHook.addRoute(PAGE_ROUTE, Page, { exact: true });
  routerHook.addRoute(GAME_PAGE, GamePage, { exact: true });
  return {
    name: "KytyPS5-Native",
    titleView: <div className={staticClasses.Title}>KytyPS5-Native</div>,
    content: <Content />,
    icon: <FaPlaystation />,
    onDismount() {
      clearInterval(timer);
      routerHook.removeRoute(PAGE_ROUTE);
      routerHook.removeRoute(GAME_PAGE);
    },
  };
});
