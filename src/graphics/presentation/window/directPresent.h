#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_DIRECTPRESENT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_DIRECTPRESENT_H_

#include "common/common.h"

#include <cstdint>
#include <functional>
#include <memory>

struct SDL_Window;

namespace Libs::Graphics {

// How the display took a direct present (Present CompleteNotify mode). Flip: the buffer itself is on
// screen (Xwayland handed it to gamescope, which scans it out or composites it). Copy: Xwayland
// copied it into the window, which costs a blit on the shared gfx ring. Skip: a newer present
// replaced it before its MSC. Lost: it never reached the display (the connection died or the X server
// refused the request); its idle follows at once.
enum class DirectCompleteMode : uint8_t { Flip, Copy, Skip, Lost };

struct DirectPresentCallbacks {
	// All run on the DirectPresenter event thread, one at a time. They may take VideoOut locks; they
	// must not call SetCallbacks or wait for another Present to complete. For one token, complete always
	// comes before idle: the X server sends a copied or skipped pixmap's IdleNotify ahead of its
	// CompleteNotify, and that idle is held back until the complete has been delivered.
	std::function<void(uint64_t token, uint64_t msc, uint64_t ust_us, DirectCompleteMode mode)> complete;
	std::function<void(uint64_t token)>                  idle;    // the pixmap of that present was released
	std::function<void(uint32_t width, uint32_t height)> resized; // window size changed (Present ConfigureNotify)
	std::function<void()> lost; // connection error; after complete(Lost)+idle for everything in flight
};

struct DirectSurface {
	int      dmabuf_fd; // borrowed; AddBuffer dup()s it (xcb takes ownership of the fd it sends)
	uint64_t offset;    // DRI3 offset0
	uint32_t width, height;
	uint32_t stride; // bytes per row: pitch_in_elements * bytes_per_element (3840 -> 15360)
	uint8_t  depth;  // 24 (XRGB8888) or 30 (XRGB2101010); bpp 32
};

// The PS5 display engine scans the title's display buffer itself out: a flip queues the buffer for
// the next vblank and costs no GPU work, and the buffer's label goes to 0 when the display has
// released it. Here the display is Xwayland, then gamescope, then a DCN plane. Each display buffer
// lives in one VRAM BO (GemBacking's carve) in the GPU's own SW_64KB_R_X layout, which is the AMD GFX10
// 64K_R_X modifier without DCC, so it becomes a DRI3 pixmap of SDL's X11 window once, at registration,
// and every flip is one PresentPixmap. CompleteNotify is the flip, IdleNotify the release.
class DirectPresenter final {
public:
	// Null when: KYTY_DIRECT_PRESENT is not 1 (NativeGpu::DirectPresentWanted()); the SDL window is not
	// X11 (SDL_PROP_WINDOW_X11_WINDOW_NUMBER == 0); DRI3 < 1.2 or Present < 1.2; the server lists no
	// 0x0200000000801b02 for depth 24 (GetSupportedModifiers, window and screen lists).
	static std::unique_ptr<DirectPresenter> Create(SDL_Window* window);
	~DirectPresenter(); // stops the event thread, frees pixmaps, disconnects
	KYTY_CLASS_NO_COPY(DirectPresenter);

	void SetCallbacks(DirectPresentCallbacks callbacks); // waits for a callback in progress
	[[nodiscard]] bool    Usable() const;                // false after lost, Disable(), or HDR ever used
	void                  Disable(const char* why);      // logs once; Usable() false from then on
	[[nodiscard]] uint8_t DepthFor(uint64_t video_out_pixel_format) const; // 0x8000000000000000 -> 24,
	                                       // 0x8100000000000000 -> 30 (only if the screen has depth 30), else 0
	[[nodiscard]] bool WindowIs(uint32_t width, uint32_t height) const; // current X window size
	uint32_t AddBuffer(const DirectSurface& surface); // pixmap XID or 0; PixmapFromBuffers_checked (round trip)
	void     RemoveBuffer(uint32_t pixmap);           // xcb_free_pixmap; presents in flight still complete/idle
	// Non-blocking, thread-safe. target_msc 0 = next MSC. async = PRESENT_OPTION_ASYNC. False only when
	// the request could not be sent (connection lost); no callback comes for that token then. A present
	// sent after Disable() still goes out and completes.
	bool Present(uint32_t pixmap, uint64_t token, uint64_t target_msc, bool async);
	[[nodiscard]] uint64_t LastMsc() const;  // msc of the last CompleteNotify (or NotifyMSC)
	[[nodiscard]] uint32_t InFlight() const; // presents sent, CompleteNotify not yet seen

private:
	struct Impl;
	explicit DirectPresenter(std::unique_ptr<Impl> impl);
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_DIRECTPRESENT_H_
