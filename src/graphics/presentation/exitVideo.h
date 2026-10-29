#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_EXITVIDEO_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_EXITVIDEO_H_

struct SDL_Window;

namespace Libs::Graphics {

// `kyty_emulator --play-video <file>`: the launch wrapper plays the exit video with it while the game
// shuts down. A missing file plays nothing.
int PlayExitVideo(const char* path);

// Called when the user leaves the game from the in-game menu: creates the file named by KYTY_EXIT_MARKER,
// so the wrapper can start the exit video at once. Nothing without the variable.
void MarkUserExit();

// Under gamescope: sets STEAM_GAME on the window to the app id the launch wrapper names in
// KYTY_GAMESCOPE_APPID (gamescope shows only the windows of the app in front), so the wrapper needs no
// tool to find the window. Nothing without the variable or off X11.
void MarkGamescopeWindow(SDL_Window* window);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_EXITVIDEO_H_
