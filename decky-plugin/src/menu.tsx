// The game's page from the game's own menu in the library (the one on the Options button): an item
// above "Properties...". Steam has no interface for this, so its menu component is looked up among
// its modules and what it renders gets one more item. Whatever does not look as expected leaves the
// menu as it was: the page stays reachable from the plugin's panel.
import { MenuItem, afterPatch, fakeRenderComponent, findInReactTree, findModuleByExport } from "@decky/ui";
import { openGame } from "./routes";

const KEY = "kytyps5-native-game";

// The games' shortcuts, by Steam's app id.
let titles = new Map<number, string>();
export const setMenuGames = (games: { appid: number; title_id: string }[]) => {
  titles = new Map(games.filter((g) => g.appid).map((g) => [g.appid, g.title_id]));
};

function libraryMenu(): any {
  const module = findModuleByExport((e: any) => e?.toString?.().includes("().LibraryContextMenu"));
  const wrapper = module && Object.values(module).find((e: any) => e?.toString?.().includes("navigator:"));
  return wrapper ? fakeRenderComponent(wrapper as any)?.type : undefined;
}

// Returns what takes the item out again.
export function patchLibraryMenu(): () => void {
  try {
    const menu = libraryMenu();
    if (!menu?.prototype?.render) throw new Error("the library's menu was not found");
    const patch = afterPatch(menu.prototype, "render", function (this: any, _: any[], ret: any) {
      try {
        const titleId = titles.get(this.props?.overview?.appid);
        const items = ret?.props?.children;
        // One game of ours, not a selection of several.
        if (!titleId || !Array.isArray(items) || items.some((i: any) => i?.key === KEY)) return ret;
        if ((this.GetTargetApps?.()?.length ?? 1) > 1) return ret;
        const properties = items.findIndex(
          (i: any) => !!i && !!findInReactTree(i, (x: any) => x?.props?.onSelected?.toString().includes("AppProperties")),
        );
        items.splice(
          properties < 0 ? items.length : properties,
          0,
          <MenuItem key={KEY} onSelected={() => openGame(titleId)}>
            KytyPS5-Native...
          </MenuItem>,
        );
      } catch (e) {
        console.error("KytyPS5-Native: the game's menu", e);
      }
      return ret;
    });
    return () => patch.unpatch();
  } catch (e) {
    console.error("KytyPS5-Native: no item in the games' menu", e);
    return () => {};
  }
}
