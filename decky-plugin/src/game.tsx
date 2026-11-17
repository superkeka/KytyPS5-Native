// A game's page, laid out as Steam's own page of a game: the artwork on top with the name over it,
// then what Kyty adds to the game in tabs: its patches, its add-ons, what it is.
import {
  DialogBody,
  DialogButton,
  DialogControlsSection,
  Field,
  Focusable,
  GamepadButton,
  NavEntryPositionPreferences,
  ScrollPanel,
  ScrollPanelGroup,
  Spinner,
  ToggleField,
  useParams,
} from "@decky/ui";
import { Dispatch, SetStateAction, useEffect, useRef, useState } from "react";
import { GameInfo, Patches, findAddons, getGame, getPatches, gigabytes, setAddon, setPatch, toast } from "./api";

// Sizes and colours are those of Steam's page of a game, so the two read as one.
const CSS = `
.kyty-game { position: relative; height: calc(100% - var(--gamepadui-current-footer-height, 42px)); overflow: hidden; background: #0e141b; }
.kyty-backdrop { position: absolute; inset: -12%; background-size: cover; background-position: center; filter: blur(64px) saturate(1.5); }
.kyty-backdrop::after { content: ""; position: absolute; inset: 0; background: rgba(14,20,27,0.8); }
.kyty-scroll { position: relative; height: 100%; }
.kyty-hero { position: relative; height: 38vh; min-height: 220px; background-size: cover; background-position: center 22%; background-color: #17202b; }
.kyty-hero::after { content: ""; position: absolute; inset: 0; background: linear-gradient(to bottom, rgba(14,20,27,0.62) 0, rgba(14,20,27,0) 26%, rgba(14,20,27,0) 44%, rgba(14,20,27,0.8) 100%); }
.kyty-hero-text { position: absolute; left: 2.8vw; right: 2.8vw; bottom: 18px; z-index: 1; }
.kyty-title { font-size: 40px; font-weight: 700; line-height: 1.1; color: #fff; text-shadow: 0 2px 14px rgba(0,0,0,0.65); }
.kyty-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.kyty-chip { padding: 4px 12px; border-radius: 64px; background: rgba(255,255,255,0.16); color: #fff; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; }
.kyty-tabs { display: flex; justify-content: center; gap: 2px; padding: 16px 0 12px; }
.kyty-pill { padding: 6px 16px; border-radius: 64px; color: #dcdedf; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; white-space: nowrap; }
.kyty-pill.kyty-on { background: rgba(255,255,255,0.15); color: #fff; }
.kyty-pill.kyty-focus { background: #fff; color: #0e141b; }
.kyty-pill span { margin-left: 8px; opacity: 0.6; }
.kyty-body { padding: 0 2.8vw 2.5em; }
`;

type TabId = "patches" | "addons" | "about";
const TABS: TabId[] = ["patches", "addons", "about"];

const steamStore = () => (window as any).appStore;
// The game's Steam shortcut, as Steam knows it.
function shortcut(appid: number): any {
  try {
    return (appid && steamStore()?.GetAppOverviewByAppID(appid)) || null;
  } catch {
    return null;
  }
}
// The artwork the shortcut got from the game (its picture under its icon), as Steam serves it.
function heroOf(app: any): string {
  try {
    const url = app ? steamStore().GetCustomHeroImageURLs(app)?.[0] : "";
    return url ? (url.startsWith("/") ? `https://steamloopback.host${url}` : url) : "";
  } catch {
    return "";
  }
}
const Scroll = ScrollPanel as any;
const ScrollGroup = ScrollPanelGroup as any;

export function GamePage() {
  const { titleId } = useParams<{ titleId: string }>();
  const [game, setGame] = useState<GameInfo | null | undefined>(undefined);
  const [patches, setPatches] = useState<Patches | undefined>(undefined);
  const [tab, setTab] = useState<TabId | undefined>(undefined);
  const pills = useRef<Partial<Record<TabId, HTMLDivElement | null>>>({});
  // Set once the user has picked a tab: the page does not pick for them after that.
  const picked = useRef(false);

  // While the game's executable is still being read for the first time, ask again.
  useEffect(() => {
    let open = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      const g = await getGame(titleId);
      if (!open) return;
      setGame(g);
      // The page opens on what the game has: its patches, else its add-ons.
      if (g) setTab((t) => t ?? (g.patches || !g.addons.length ? "patches" : "addons"));
      if (g && !g.addons_known) timer = setTimeout(load, 1500);
    };
    load();
    getPatches(titleId, false).then((p) => {
      if (!open) return;
      setPatches(p);
      if (p.mods.length && !picked.current) setTab("patches");
    });
    return () => {
      open = false;
      clearTimeout(timer);
    };
  }, [titleId]);

  if (!game)
    return (
      <div className="kyty-game">
        <style>{CSS}</style>
        <div className="kyty-body" style={{ paddingTop: "calc(var(--basicui-header-height, 40px) + 12px)" }}>
          <DialogBody>
            {game === undefined ? (
              <Field label="Loading..." />
            ) : (
              <Field label="This game is not in the games folder any more" description={titleId} focusable />
            )}
          </DialogBody>
        </div>
      </div>
    );

  const app = shortcut(game.appid);
  const hero = heroOf(app);
  const patchesOn = patches ? patches.mods.filter((m) => m.on).length : game.patches_on;
  const patchesAll = patches ? patches.mods.length : game.patches;
  const addonsOn = game.addons.filter((a) => a.on).length;
  const shown = tab ?? "patches";
  const titles: Record<TabId, [string, string]> = {
    patches: ["Patches", patchesAll ? `${patchesOn}/${patchesAll}` : ""],
    addons: ["Add-ons", game.addons.length ? `${addonsOn}/${game.addons.length}` : ""],
    about: ["About", ""],
  };

  // With a bumper the focus goes to the tab's name: what it was on is gone with the tab left.
  const pick = (id: TabId) => {
    picked.current = true;
    setTab(id);
  };
  const step = (by: number) => {
    const next = TABS[(TABS.indexOf(shown) + by + TABS.length) % TABS.length];
    pick(next);
    requestAnimationFrame(() => pills.current[next]?.focus());
  };

  return (
    <Focusable
      className="kyty-game"
      onButtonDown={(e) => {
        if (e.detail.button === GamepadButton.BUMPER_LEFT) step(-1);
        if (e.detail.button === GamepadButton.BUMPER_RIGHT) step(1);
      }}
      actionDescriptionMap={{ [GamepadButton.BUMPER_LEFT]: "Previous tab", [GamepadButton.BUMPER_RIGHT]: "Next tab" }}
    >
      <style>{CSS}</style>
      {hero && <div className="kyty-backdrop" style={{ backgroundImage: `url("${hero}")` }} />}
      <ScrollGroup focusable={false} className="kyty-scroll">
        <Scroll>
          <div className="kyty-hero" style={hero ? { backgroundImage: `url("${hero}")` } : undefined}>
            <div className="kyty-hero-text">
              <div className="kyty-title">{game.name}</div>
              <div className="kyty-chips">
                <span className="kyty-chip">{game.title_id}</span>
                {game.version && <span className="kyty-chip">{game.version}</span>}
                <span className="kyty-chip">{gigabytes(game.size)}</span>
              </div>
            </div>
          </div>

          {/* The row is entered at the tab that is open, and moving along it opens the next. */}
          <Focusable
            className="kyty-tabs"
            flow-children="horizontal"
            navEntryPreferPosition={NavEntryPositionPreferences.PREFERRED_CHILD}
          >
            {TABS.map((id) => (
              <Focusable
                key={id}
                ref={(el) => {
                  pills.current[id] = el;
                }}
                className={`kyty-pill${shown === id ? " kyty-on" : ""}`}
                focusClassName="kyty-focus"
                noFocusRing
                preferredFocus={shown === id}
                onActivate={() => pick(id)}
                onClick={() => pick(id)}
                onGamepadFocus={() => pick(id)}
              >
                {titles[id][0]}
                {titles[id][1] && <span>{titles[id][1]}</span>}
              </Focusable>
            ))}
          </Focusable>

          <div className="kyty-body">
            <DialogBody>
              {shown === "patches" && <PatchesTab game={game} patches={patches} setPatches={setPatches} />}
              {shown === "addons" && <AddonsTab game={game} setGame={setGame} />}
              {shown === "about" && <AboutTab game={game} app={app} />}
            </DialogBody>
          </div>
        </Scroll>
      </ScrollGroup>
    </Focusable>
  );
}

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

function PatchesTab(props: { game: GameInfo; patches?: Patches; setPatches: Dispatch<SetStateAction<Patches | undefined>> }) {
  const { game, patches, setPatches } = props;
  const [looking, setLooking] = useState(false);
  const again = async () => {
    setLooking(true);
    setPatches(await getPatches(game.title_id, true));
    setLooking(false);
  };
  const change = async (name: string, on: boolean) => {
    // From the state as it is now: two switches flipped at once both count.
    setPatches((p) => p && { ...p, mods: p.mods.map((m) => (name && m.name !== name ? m : { ...m, on })) });
    const r = await setPatch(game.title_id, name, on);
    if (!r.ok) {
      toast(r.log || "failed", game.name);
      setPatches(await getPatches(game.title_id, false));
    }
  };

  if (!patches)
    return (
      <DialogControlsSection>
        <Field label="Looking for patches..." description={`For version ${game.version} of the game`} focusable>
          <Spinner style={{ width: 24, height: 24 }} />
        </Field>
      </DialogControlsSection>
    );
  if (patches.state !== "ready")
    return (
      <DialogControlsSection>
        <Action
          label={
            patches.state === "offline"
              ? "The patch collection could not be reached"
              : patches.others.length
                ? "No patches for this version"
                : "No patches for this game"
          }
          description={
            patches.state === "offline"
              ? patches.error
              : patches.others.length
                ? `The collection has them for ${patches.others.join(", ")}, not for ${patches.version || "this one"}`
                : "The HEN Cheats Collection has none for it"
          }
          button={looking ? "Looking..." : patches.state === "none" ? "Look again" : "Try again"}
          disabled={looking}
          onClick={again}
        />
      </DialogControlsSection>
    );

  const on = patches.mods.filter((m) => m.on).length;
  return (
    <>
      <DialogControlsSection>
        <Action
          label={`${patches.mods.length} ${patches.mods.length === 1 ? "patch" : "patches"} for version ${patches.version}`}
          button="All off"
          disabled={!on}
          onClick={() => change("", false)}
        />
        {patches.mods.map((m) => (
          <ToggleField
            key={m.name}
            label={m.name}
            description={m.hint || undefined}
            checked={m.on}
            onChange={(value) => change(m.name, value)}
          />
        ))}
      </DialogControlsSection>
      <DialogControlsSection>
        <Action
          label="Look again"
          description={
            (patches.credits.length ? `Made by ${patches.credits.join("; ")}. ` : "") +
            "From the HEN Cheats Collection (github.com/TeeKay87/HEN-Cheats-Collection)" +
            (patches.error ? `. Could not be reached now: ${patches.error}` : "")
          }
          button={looking ? "Looking..." : "Look"}
          disabled={looking}
          onClick={again}
        />
      </DialogControlsSection>
    </>
  );
}

function AddonsTab(props: { game: GameInfo; setGame: Dispatch<SetStateAction<GameInfo | null | undefined>> }) {
  const { game, setGame } = props;
  const [looking, setLooking] = useState(false);
  const change = async (label: string, on: boolean) => {
    setGame((g) => g && { ...g, addons: g.addons.map((a) => (a.label === label ? { ...a, on } : a)) });
    const r = await setAddon(game.title_id, label, on);
    if (!r.ok) {
      toast(r.log || "failed", game.name);
      setGame(await getGame(game.title_id));
    }
  };
  return (
    <DialogControlsSection>
      {!game.addons_known ? (
        <Field label="Looking for add-ons..." description="Reading the game's executable" focusable>
          <Spinner style={{ width: 24, height: 24 }} />
        </Field>
      ) : game.addons.length ? (
        <Field
          label="DLC and licences the game asks about"
          description={
            "Found in the game's executable, all on to begin with: Kyty tells the game it has them. " +
            "Switch off the ones it should not have. Applies from the game's next start"
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
          onChange={(value) => change(a.label, value)}
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
            setGame(await getGame(game.title_id));
            setLooking(false);
          }}
        />
      )}
    </DialogControlsSection>
  );
}

function AboutTab({ game, app }: { game: GameInfo; app: any }) {
  return (
    <DialogControlsSection>
      <Field label="Title ID" focusable>
        {game.title_id}
      </Field>
      {game.version && (
        <Field label="Version" focusable>
          {game.version}
        </Field>
      )}
      <Field label="Size" description={game.path} focusable>
        {gigabytes(game.size)}
      </Field>
      <Field label="Steam shortcut" focusable>
        {app ? app.display_name : "Not made yet"}
      </Field>
      <Field label="Log of the last start" description={`~/.local/share/KytyPS5-Native/logs/${game.title_id}.log`} focusable />
    </DialogControlsSection>
  );
}
