// Steam shortcuts for the games: made through SteamClient (only the frontend has it), with artwork
// composed from the game's own sce_sys/icon0.png and pic0.png.
import { callable } from "@decky/api";

export interface Game {
  title_id: string;
  name: string;
  path: string;
  size: number;
  appid: number;
  keep_steam: boolean;
}

const getArt = callable<[path: string], { icon: string; pic: string }>("get_art");
type Art = { capsule: string; hero: string; wide: string };
const getCachedArt = callable<[titleId: string], Partial<Art>>("get_cached_art");
const saveArt = callable<[titleId: string, kind: string, data: string], void>("save_art");
const setGame = callable<[titleId: string, appid: number, keepSteam: boolean], void>("set_game");

const steam = () => (window as any).SteamClient;
const appStore = () => (window as any).appStore;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export const shortcutName = (g: Game) => `${g.name} (PS5)`;

function exists(appid: number): boolean {
  return !!appid && !!appStore()?.GetAppOverviewByAppID(appid);
}

async function image(b64: string): Promise<HTMLImageElement | null> {
  if (!b64) return null;
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  try {
    await img.decode();
    return img;
  } catch {
    return null;
  }
}

// Draws `img` over the whole w x h canvas, cropping what does not fit (as CSS object-fit: cover).
function cover(ctx: CanvasRenderingContext2D, img: HTMLImageElement, w: number, h: number, pad = 0) {
  const s = Math.max((w + 2 * pad) / img.width, (h + 2 * pad) / img.height);
  const dw = img.width * s;
  const dh = img.height * s;
  ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

function canvas(w: number, h: number, draw: (ctx: CanvasRenderingContext2D) => void): string {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  draw(c.getContext("2d")!);
  return c.toDataURL("image/png").split(",")[1];
}

// Steam's custom artwork kinds: 0 capsule (600x900), 1 hero (1920x620), 3 wide capsule (920x430).
async function composeArt(g: Game): Promise<Art | null> {
  const art = await getArt(g.path);
  const icon = await image(art.icon);
  const pic = (await image(art.pic)) ?? icon;
  if (!icon || !pic) return null;
  return {
    capsule: canvas(600, 900, (ctx) => {
      ctx.filter = "blur(12px) brightness(0.85)";
      cover(ctx, pic, 600, 900, 24);
      ctx.filter = "none";
      ctx.drawImage(icon, 70, 220, 460, 460);
    }),
    hero: canvas(1920, 620, (ctx) => cover(ctx, pic, 1920, 620)),
    wide: canvas(920, 430, (ctx) => {
      ctx.filter = "blur(12px) brightness(0.85)";
      cover(ctx, pic, 920, 430, 24);
      ctx.filter = "none";
      ctx.drawImage(icon, 310, 65, 300, 300);
    }),
  };
}

// Raised when the artwork's look changes: shortcuts get it again, the cache keeps one per version.
export const ART_VERSION = 2;

async function setArtwork(appid: number, g: Game) {
  const key = `${g.title_id}-v${ART_VERSION}`;
  let art = (await getCachedArt(key)) as Art;
  if (!art.capsule) {
    const made = await composeArt(g);
    if (!made) return;
    art = made;
    for (const kind of ["capsule", "hero", "wide"] as const) await saveArt(key, kind, art[kind]);
  }
  await steam().Apps.SetCustomArtworkForApp(appid, art.capsule, "png", 0);
  await steam().Apps.SetCustomArtworkForApp(appid, art.hero, "png", 1);
  await steam().Apps.SetCustomArtworkForApp(appid, art.wide, "png", 3);
}

// A shortcut already in Steam with this name (one the settings lost, or made by hand): taken over
// instead of adding another.
function byName(name: string): number {
  const app = [...(appStore()?.allApps ?? [])].find((a: any) => a.app_type === 1073741824 && a.display_name === name);
  return app?.appid ?? 0;
}

// Adds or updates the shortcut of one game; returns its app id.
export async function syncGame(g: Game, runner: string, kytyDir: string, redoArt = false): Promise<number> {
  let appid = exists(g.appid) ? g.appid : byName(shortcutName(g));
  const fresh = !appid;
  if (fresh) {
    appid = await steam().Apps.AddShortcut(shortcutName(g), `"${runner}"`, "", "");
    await sleep(500);
  }
  steam().Apps.SetShortcutName(appid, shortcutName(g));
  steam().Apps.SetShortcutExe(appid, `"${runner}"`);
  steam().Apps.SetShortcutStartDir(appid, `"${kytyDir}/"`);
  steam().Apps.SetShortcutLaunchOptions(appid, `"${g.path}"${g.keep_steam ? " --keep-steam" : ""}`);
  steam().Apps.SetShortcutIcon(appid, `${g.path}/sce_sys/icon0.png`);
  if (fresh || redoArt) await setArtwork(appid, g);
  await setGame(g.title_id, appid, g.keep_steam);
  return appid;
}

// Shortcuts of games that are no longer in the folder.
export async function removeGone(gone: Record<string, { appid: number }>) {
  for (const [titleId, { appid }] of Object.entries(gone)) {
    if (exists(appid)) steam().Apps.RemoveShortcut(appid);
    await setGame(titleId, 0, false);
  }
}

export async function syncAll(
  games: Game[],
  gone: Record<string, { appid: number }>,
  runner: string,
  kytyDir: string,
  redoArt = false,
) {
  for (const g of games) await syncGame(g, runner, kytyDir, redoArt);
  await removeGone(gone);
}

// True when a game has no shortcut yet or one was removed: an automatic sync is worth it.
export function needsSync(games: Game[], gone: Record<string, { appid: number }>): boolean {
  return Object.keys(gone).some((t) => exists(gone[t].appid)) || games.some((g) => !exists(g.appid));
}
