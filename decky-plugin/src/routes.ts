// Where the plugin's pages are in Steam's router, and the ways to open them.
import { Navigation } from "@decky/ui";

export const PAGE = "/kytyps5-native";
// The page opens on the tab named after it (games, display), on the games without one.
export const PAGE_ROUTE = `${PAGE}/:tab?`;
export const GAME_PAGE = `${PAGE}/game/:titleId`;

export const openPage = (tab = "") => {
  Navigation.Navigate(tab ? `${PAGE}/${tab}` : PAGE);
  Navigation.CloseSideMenus();
};

export const openGame = (titleId: string) => {
  Navigation.Navigate(`${PAGE}/game/${titleId}`);
  Navigation.CloseSideMenus();
};
