// The plugin's own page: tabs for the games and the display, and a menu per game. The system side
// (version, updates, kernel config) is in the side panel.
import {
  DialogBody,
  DialogButton,
  DialogControlsSection,
  DialogControlsSectionHeader,
  Field,
  Navigation,
  Tabs,
  ToggleField,
  useParams,
} from "@decky/ui";
import { openFilePicker } from "@decky/api";
import { ReactNode, useEffect, useState } from "react";
import { FaChevronRight } from "react-icons/fa";
import {
  GameInfo,
  PluginView,
  State,
  findAddons,
  getGame,
  gigabytes,
  setAddon,
  setGamesDir,
  setHdr,
  setVideo,
  toast,
  usePlugin,
} from "./api";

export const PAGE = "/kytyps5-native";
// The page opens on the tab named after it (games, display), on the games without one.
export const PAGE_ROUTE = `${PAGE}/:tab?`;
export const GAME_PAGE = `${PAGE}/game/:titleId`;

export const openPage = (tab = "") => {
  Navigation.Navigate(tab ? `${PAGE}/${tab}` : PAGE);
  Navigation.CloseSideMenus();
};

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
              (g.addons ? ` · ${g.addons_on === g.addons ? g.addons : `${g.addons_on} of ${g.addons}`} ${g.addons === 1 ? "add-on" : "add-ons"} on` : "")
            }
            focusable
            highlightOnFocus
            onActivate={() => Navigation.Navigate(`${PAGE}/game/${g.title_id}`)}
            onClick={() => Navigation.Navigate(`${PAGE}/game/${g.title_id}`)}
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

function DisplayTab({ view, state }: { view: PluginView; state: State }) {
  const { busy, run } = view;
  return (
    <Scroll>
      <DialogControlsSection>
        <DialogControlsSectionHeader>Display</DialogControlsSectionHeader>
        <ToggleField
          label="HDR"
          description="HDR10 when Steam's display outputs HDR. Applies from the next game start"
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
        ]}
      />
    </Screen>
  );
}

// One game's menu: what it is, and its add-ons.
export function GamePage() {
  const { titleId } = useParams<{ titleId: string }>();
  const [game, setGame] = useState<GameInfo | null | undefined>(undefined);
  const [looking, setLooking] = useState(false);

  // While the game's executable is still being read for the first time, ask again.
  useEffect(() => {
    let open = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      const g = await getGame(titleId);
      if (!open) return;
      setGame(g);
      if (g && !g.addons_known) timer = setTimeout(load, 1500);
    };
    load();
    return () => {
      open = false;
      clearTimeout(timer);
    };
  }, [titleId]);

  if (game === undefined)
    return (
      <Screen>
        <Scroll>
          <Field label="Loading..." />
        </Scroll>
      </Screen>
    );
  if (game === null)
    return (
      <Screen>
        <Scroll>
          <Field label="This game is not in the games folder any more" description={titleId} focusable />
        </Scroll>
      </Screen>
    );

  const on = game.addons.filter((a) => a.on).length;
  const switchAddon = async (label: string, value: boolean) => {
    setGame({ ...game, addons: game.addons.map((a) => (a.label === label ? { ...a, on: value } : a)) });
    const r = await setAddon(game.title_id, label, value);
    if (!r.ok) {
      toast(r.log || "failed", game.name);
      setGame(await getGame(titleId));
    }
  };

  return (
    <Screen>
      <Scroll>
        <DialogControlsSection>
          <DialogControlsSectionHeader>{game.name}</DialogControlsSectionHeader>
          <Field label="Title ID" focusable>
            {game.title_id}
          </Field>
          {game.version && <Field label="Version">{game.version}</Field>}
          <Field label="Size" description={game.path}>
            {gigabytes(game.size)}
          </Field>
        </DialogControlsSection>
        <DialogControlsSection>
          <DialogControlsSectionHeader>
            {game.addons.length ? `Add-ons: ${on} of ${game.addons.length} on` : "Add-ons"}
          </DialogControlsSectionHeader>
          {!game.addons_known ? (
            <Field label="Looking for add-ons..." description="Reading the game's executable" focusable />
          ) : game.addons.length ? (
            <Field
              label="DLC and licences the game asks about"
              description={
                "Found in the game's executable, all on to begin with: Kyty tells the game it has them. " +
                "Switch off the ones it should not have. Applies from the next start of the game"
              }
              focusable
            />
          ) : (
            <Field
              label="None found"
              description={`The game's executable names none. One can be added by hand: a line with its label in ~/.config/KytyPS5-Native/addcont/${game.title_id}`}
              focusable
            />
          )}
          {game.addons.map((a) => (
            <ToggleField
              key={a.label}
              label={a.label}
              description={a.found ? undefined : a.folder ? `Added by hand, its data in ${a.folder}` : "Added by hand"}
              checked={a.on}
              onChange={(value) => switchAddon(a.label, value)}
            />
          ))}
          {game.addons_known && (
            <Action
              label="Look again"
              description="After the game was updated or replaced"
              button={looking ? "Looking..." : "Look"}
              disabled={looking}
              onClick={async () => {
                setLooking(true);
                await findAddons(game.title_id);
                setGame(await getGame(titleId));
                setLooking(false);
              }}
            />
          )}
        </DialogControlsSection>
      </Scroll>
    </Screen>
  );
}
