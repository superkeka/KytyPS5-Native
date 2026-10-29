#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_GAMEMENU_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_GAMEMENU_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

union SDL_Event;

namespace Libs::Graphics {

// The in-game menu (KytyPS5-Native): the PS button, Share and Options together, or F10 opens it over
// the game, as the PS5's own menu. Resume, monitoring (the performance overlay) and leaving the game,
// all from a gamepad, with nothing of Steam running.

// Window thread. True when the event was the menu's (the game does not get it). Button releases
// always go on to the game, so nothing stays held there while the menu is open.
bool ProcessGameMenuInput(const SDL_Event& event);

[[nodiscard]] bool     GameMenuOpen() noexcept;
[[nodiscard]] uint64_t GameMenuRevision() noexcept;

// Present thread, between ImGui::NewFrame() and ImGui::Render().
void DrawGameMenu(vk::Extent2D extent);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_GAMEMENU_H_
