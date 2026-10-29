#include "graphics/presentation/gameMenu.h"

#include "graphics/native_gpu/nativeShadow.h"
#include "graphics/presentation/exitVideo.h"
#include "graphics/presentation/performanceOverlay.h"
#include "imgui.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>

namespace Libs::Graphics {

namespace {

enum class Page : uint8_t { Closed, Main, ConfirmExit };

enum MainItem : int { ItemResume, ItemMonitoring, ItemExit, MainItemCount };
enum ConfirmItem : int { ConfirmExitGame, ConfirmCancel, ConfirmItemCount };

struct MenuState {
	Page page     = Page::Closed;
	int  selected = 0;
};

std::mutex            g_mutex;
MenuState             g_state;
std::atomic<bool>     g_open {false};
std::atomic<uint64_t> g_revision {0};

// Window thread only: Share and Options held, for the combination.
bool g_back_down  = false;
bool g_start_down = false;

MenuState GetState() {
	std::scoped_lock lock(g_mutex);
	return g_state;
}

void SetState(MenuState state) {
	{
		std::scoped_lock lock(g_mutex);
		g_state = state;
	}
	g_open.store(state.page != Page::Closed, std::memory_order_release);
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

int ItemCount(Page page) {
	return page == Page::ConfirmExit ? ConfirmItemCount : MainItemCount;
}

const char* MonitoringLabel(int mode) {
	switch (mode) {
		case 1: return "FPS";
		case 2: return "Detailed";
		default: return "Off";
	}
}

void StepMonitoring(int step) {
	SetPerformanceOverlayMode((PerformanceOverlayMode() + step + 3) % 3);
	g_revision.fetch_add(1, std::memory_order_acq_rel);
}

void LeaveGame() {
	SDL_Event quit {};
	quit.type = SDL_EVENT_QUIT;
	SDL_PushEvent(&quit);
}

void Activate(const MenuState& state) {
	if (state.page == Page::Main) {
		switch (state.selected) {
			case ItemResume: SetState({}); break;
			case ItemMonitoring: StepMonitoring(1); break;
			case ItemExit: SetState({Page::ConfirmExit, ConfirmExitGame}); break;
			default: break;
		}
		return;
	}
	if (state.selected == ConfirmExitGame) {
		SetState({});
		MarkUserExit();
		LeaveGame();
	} else {
		SetState({Page::Main, ItemExit});
	}
}

void Back(const MenuState& state) {
	SetState(state.page == Page::ConfirmExit ? MenuState {Page::Main, ItemExit} : MenuState {});
}

// Navigation shared by the gamepad and the keyboard. True when the menu took the input.
bool Navigate(int up_down, int left_right, bool accept, bool back) {
	MenuState state = GetState();
	if (state.page == Page::Closed) {
		return false;
	}
	if (up_down != 0) {
		const int count = ItemCount(state.page);
		state.selected  = (state.selected + up_down + count) % count;
		SetState(state);
	} else if (left_right != 0 && state.page == Page::Main && state.selected == ItemMonitoring) {
		StepMonitoring(left_right);
	} else if (accept) {
		Activate(state);
	} else if (back) {
		Back(state);
	}
	return true;
}

} // namespace

bool ProcessGameMenuInput(const SDL_Event& event) {
	if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN || event.type == SDL_EVENT_GAMEPAD_BUTTON_UP) {
		const bool down   = event.gbutton.down;
		const auto button = static_cast<SDL_GamepadButton>(event.gbutton.button);
		if (button == SDL_GAMEPAD_BUTTON_BACK) {
			g_back_down = down;
		} else if (button == SDL_GAMEPAD_BUTTON_START) {
			g_start_down = down;
		}
		if (!down) {
			return false; // releases go on to the game: nothing stays held there
		}
		const bool combination = (button == SDL_GAMEPAD_BUTTON_BACK && g_start_down) ||
		                         (button == SDL_GAMEPAD_BUTTON_START && g_back_down);
		if (button == SDL_GAMEPAD_BUTTON_GUIDE || combination) {
			SetState(GameMenuOpen() ? MenuState {} : MenuState {Page::Main, ItemResume});
			return true;
		}
		switch (button) {
			case SDL_GAMEPAD_BUTTON_DPAD_UP: return Navigate(-1, 0, false, false);
			case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return Navigate(1, 0, false, false);
			case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return Navigate(0, -1, false, false);
			case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return Navigate(0, 1, false, false);
			case SDL_GAMEPAD_BUTTON_SOUTH: return Navigate(0, 0, true, false);
			case SDL_GAMEPAD_BUTTON_EAST: return Navigate(0, 0, false, true);
			default: return GameMenuOpen();
		}
	}
	if (event.type == SDL_EVENT_GAMEPAD_AXIS_MOTION) {
		return GameMenuOpen();
	}
	if (event.type == SDL_EVENT_KEY_DOWN && event.key.key == SDLK_F10) {
		if (!event.key.repeat) {
			SetState(GameMenuOpen() ? MenuState {} : MenuState {Page::Main, ItemResume});
		}
		return true;
	}
	if (event.type == SDL_EVENT_KEY_DOWN && GameMenuOpen()) {
		switch (event.key.key) {
			case SDLK_UP: return Navigate(-1, 0, false, false);
			case SDLK_DOWN: return Navigate(1, 0, false, false);
			case SDLK_LEFT: return Navigate(0, -1, false, false);
			case SDLK_RIGHT: return Navigate(0, 1, false, false);
			case SDLK_RETURN:
			case SDLK_KP_ENTER: return Navigate(0, 0, true, false);
			case SDLK_ESCAPE:
			case SDLK_BACKSPACE: return Navigate(0, 0, false, true);
			default: return true;
		}
	}
	return false;
}

bool GameMenuOpen() noexcept {
	return g_open.load(std::memory_order_acquire);
}

uint64_t GameMenuRevision() noexcept {
	return g_revision.load(std::memory_order_acquire);
}

void DrawGameMenu(vk::Extent2D extent) {
	const MenuState state = GetState();
	if (state.page == Page::Closed) {
		return;
	}
	const ImVec2 display(static_cast<float>(extent.width), static_cast<float>(extent.height));
	const float  scale = std::max(std::min(display.x / 1280.0f, display.y / 720.0f), 0.5f);
	ImGui::GetBackgroundDrawList()->AddRectFilled({0.0f, 0.0f}, display, IM_COL32(0, 0, 0, 170));
	ImGui::SetNextWindowPos({display.x * 0.5f, display.y * 0.5f}, ImGuiCond_Always, {0.5f, 0.5f});
	ImGui::SetNextWindowSize({std::max(std::min(460.0f * scale, display.x - 32.0f), 1.0f), 0.0f},
	                         ImGuiCond_Always);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {24.0f * scale, 22.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {12.0f * scale, 12.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 10.0f * scale);
	ImGui::PushFont(nullptr, 22.0f * scale);
	constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
	                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize |
	                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
	ImGui::Begin("##KytyGameMenu", nullptr, flags);
	ImGui::TextUnformatted("KytyPS5-Native");
	ImGui::SameLine();
	ImGui::TextDisabled("%s", NativeGpu::PrimaryRendering() ? "native PS5 GPU path" : "Vulkan path");
	ImGui::Separator();
	const float row_height = 44.0f * scale;
	auto        row        = [&](int index, const char* label, const char* value) {
        const bool selected = state.selected == index;
        const ImVec2 start  = ImGui::GetCursorScreenPos();
        const float  width  = ImGui::GetContentRegionAvail().x;
        if (selected) {
            ImGui::GetWindowDrawList()->AddRectFilled(start, {start.x + width, start.y + row_height},
                                                      IM_COL32(255, 255, 255, 40), 6.0f * scale);
        }
        const float text_y = start.y + (row_height - ImGui::GetTextLineHeight()) * 0.5f;
        ImGui::GetWindowDrawList()->AddText({start.x + 12.0f * scale, text_y}, IM_COL32(240, 240, 245, 255), label);
        if (value != nullptr) {
            const float value_width = ImGui::CalcTextSize(value).x;
            ImGui::GetWindowDrawList()->AddText({start.x + width - value_width - 12.0f * scale, text_y},
                                                IM_COL32(170, 200, 255, 255), value);
        }
        ImGui::Dummy({width, row_height});
	};
	if (state.page == Page::Main) {
		row(ItemResume, "Resume", nullptr);
		row(ItemMonitoring, "Monitoring", MonitoringLabel(PerformanceOverlayMode()));
		row(ItemExit, "Exit game", nullptr);
	} else {
		ImGui::TextUnformatted("Exit the game?");
		ImGui::TextDisabled("Progress the game has not saved is lost.");
		row(ConfirmExitGame, "Exit game", nullptr);
		row(ConfirmCancel, "Cancel", nullptr);
	}
	ImGui::Separator();
	ImGui::PushFont(nullptr, 17.0f * scale);
	ImGui::TextDisabled("Cross / Enter: select    Circle / Esc: back    PS / F10: close");
	ImGui::PopFont();
	ImGui::End();
	ImGui::PopFont();
	ImGui::PopStyleVar(3);
}

} // namespace Libs::Graphics
