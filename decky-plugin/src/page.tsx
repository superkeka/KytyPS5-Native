// The plugin's own page: tabs for the games and the display. Each game has a page of its own
// (game.tsx); the system side (version, updates, kernel config) is in the side panel.
import {
  DialogBody,
  DialogButton,
  DialogControlsSection,
  DialogControlsSectionHeader,
  DropdownItem,
  Field,
  Tabs,
  ToggleField,
  useParams,
} from "@decky/ui";
import { openFilePicker } from "@decky/api";
import { ReactNode, useState } from "react";
import { FaChevronRight } from "react-icons/fa";
import {
  PluginView,
  State,
  gigabytes,
  setAudioDefault,
  setGamesDir,
  setHdr,
  setVideo,
  toast,
  usePlugin,
} from "./api";
import { openGame } from "./routes";

// Under Steam's header, by the size Steam sets for it. A page that scrolls by itself also ends above
// Steam's footer (the button hints): a row under it could not be scrolled into view. Tabs leave that
// room on their own.
const Screen = ({ children, tabs }: { children: ReactNode; tabs?: boolean }) => (
  <div
    style={{
      marginTop: "var(--basicui-header-height, 40px)",
      height: tabs
        ? "calc(100% - var(--basicui-header-height, 40px))"
        : "calc(100% - var(--basicui-header-height, 40px) - var(--gamepadui-current-footer-height, 42px))",
    }}
  >
    {children}
  </div>
);
const Scroll = ({ children }: { children: ReactNode }) => (
  <div style={{ height: "100%", overflowY: "auto", padding: "0 2.8vw 2em", boxSizing: "border-box" }}>
    <DialogBody>{children}</DialogBody>
  </div>
);

// A row with its action on a button at the right.
function Action(props: { label: string; description?: string; button: string; disabled?: boolean; onClick: () => void }) {
  return (
    <Field label={props.label} description={props.description} childrenContainerWidth="fixed">
      <DialogButton disabled={props.disabled} onClick={props.onClick} style={{ minWidth: "11em" }}>
        {props.button}
      </DialogButton>
    </Field>
  );
}

function GamesTab({ view, state }: { view: PluginView; state: State }) {
  const total = state.games.reduce((sum, g) => sum + g.size, 0);
  return (
    <Scroll>
      <DialogControlsSection>
        <DialogControlsSectionHeader>
          {view.syncing
            ? "Adding to Steam..."
            : state.games.length
              ? `${state.games.length} ${state.games.length === 1 ? "game" : "games"}, ${gigabytes(total)}`
              : "Games"}
        </DialogControlsSectionHeader>
        {!state.kyty_installed && <Field label="Install KytyPS5-Native first" description="In the plugin's side panel" focusable />}
        {state.games.map((g) => (
          <Field
            key={g.title_id}
            label={g.name}
            description={
              `${g.title_id} · ${gigabytes(g.size)}` +
              (g.addons ? ` · ${g.addons_on === g.addons ? g.addons : `${g.addons_on} of ${g.addons}`} ${g.addons === 1 ? "add-on" : "add-ons"} on` : "") +
              (g.patches_on ? ` · ${g.patches_on} ${g.patches_on === 1 ? "patch" : "patches"} on` : "")
            }
            focusable
            highlightOnFocus
            onActivate={() => openGame(g.title_id)}
            onClick={() => openGame(g.title_id)}
          >
            <FaChevronRight style={{ opacity: 0.5 }} />
          </Field>
        ))}
        {!state.games.length && (
          <Field
            label={state.games_dir_state === "missing" ? "Games folder not found" : "No games in the folder"}
            description="A game is a folder with sce_sys/param.json in it, or a .zar archive of one"
            focusable
          />
        )}
      </DialogControlsSection>
      <DialogControlsSection>
        <DialogControlsSectionHeader>Games folder</DialogControlsSectionHeader>
        <Action
          label={state.games_dir}
          description="Games in it get their Steam shortcuts on their own"
          button="Change"
          disabled={!!view.busy}
          onClick={async () => {
            const start = state.games_dir_state === "missing" ? state.home : state.games_dir;
            try {
              const r = await openFilePicker(1 as any, start, false, true);
              await setGamesDir(r.realpath || r.path);
            } catch {
              return; // closed without a choice
            }
            await view.refresh();
          }}
        />
      </DialogControlsSection>
    </Scroll>
  );
}

// The audio output's channels offered for every game: [what the wrapper takes, its name here].
const AUDIO_MODES: [string, string][] = [
  ["auto", "Auto (the output device)"],
  ["2", "Stereo"],
  ["6", "5.1"],
  ["8", "7.1"],
];

function DisplayTab({ view, state }: { view: PluginView; state: State }) {
  const { busy, run } = view;
  return (
    <Scroll>
      <DialogControlsSection>
        <DialogControlsSectionHeader>Display</DialogControlsSectionHeader>
        <ToggleField
          label="HDR"
          description="HDR10 when Steam's display outputs HDR (on by default). Applies from the next game start"
          checked={state.hdr}
          disabled={!!busy}
          onChange={(on) =>
            run("HDR", async () => {
              const r = await setHdr(on);
              if (!r.ok) toast(r.log || "failed");
            })
          }
        />
      </DialogControlsSection>
      <DialogControlsSection>
        <DialogControlsSectionHeader>Animations</DialogControlsSectionHeader>
        {(["boot", "exit"] as const).map((kind) => (
          <ToggleField
            key={kind}
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
        ))}
      </DialogControlsSection>
    </Scroll>
  );
}

function AudioTab({ view, state }: { view: PluginView; state: State }) {
  const { busy, run } = view;
  return (
    <Scroll>
      <DialogControlsSection>
        <DialogControlsSectionHeader>Audio</DialogControlsSectionHeader>
        <DropdownItem
          label="Output channels"
          description={
            "How many speakers games are told the output has, for every game (a game's page can set its own). " +
            "Auto follows the output device, so Steam's 5.1 or 7.1 setting. Applies from the next game start"
          }
          rgOptions={AUDIO_MODES.map(([data, label]) => ({ data, label }))}
          selectedOption={state.audio}
          disabled={!!busy}
          onChange={(option) =>
            run("Audio", async () => {
              const r = await setAudioDefault(option.data);
              if (!r.ok) toast(r.log || "failed");
            })
          }
        />
      </DialogControlsSection>
    </Scroll>
  );
}

export function Page() {
  const view = usePlugin();
  const asked = useParams<{ tab?: string }>().tab;
  const [tab, setTab] = useState(asked === "display" ? asked : "games");
  const state = view.state;
  if (!state)
    return (
      <Screen>
        <Scroll>
          <Field label="Loading..." />
        </Scroll>
      </Screen>
    );
  return (
    <Screen tabs>
      <Tabs
        activeTab={tab}
        onShowTab={setTab}
        tabs={[
          { id: "games", title: "Games", content: <GamesTab view={view} state={state} /> },
          { id: "display", title: "Display", content: <DisplayTab view={view} state={state} /> },
          { id: "audio", title: "Audio", content: <AudioTab view={view} state={state} /> },
        ]}
      />
    </Screen>
  );
}
