#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_

#include "common/common.h"

#include <memory>
#include <span>

namespace Libs::Graphics {

class CommandBuffer;
class DirectPresenter;
class RenderContext;
struct ImageInfo;
struct WindowContext;

class Presenter final {
public:
	struct Frame;
	struct Layer {
		Frame* frame;
		int    bus;
		bool   premultiplied_alpha;
	};

	explicit Presenter(WindowContext& window);
	~Presenter();
	KYTY_CLASS_NO_COPY(Presenter);

	[[nodiscard]] Frame&         PrepareFrame(CommandBuffer& command, const ImageInfo& info,
	                                          bool hdr = false, bool headset = false);
	[[nodiscard]] Frame&         PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
	                                               CommandBuffer* producer = nullptr);
	[[nodiscard]] bool           PresentLastFrame();
	[[nodiscard]] bool           IsGuestPaused() const noexcept;
	[[nodiscard]] bool           IsHdrSupported() const noexcept;
	[[nodiscard]] bool           NeedsSystemOverlayRefresh() const noexcept;
	[[nodiscard]] RenderContext& Renderer() const noexcept;
	void                         Present(Frame& frame);
	void                         Present(std::span<const Layer> layers);
	void                         ClearLayer(int bus);
	void                         Discard(Frame& frame);

	// Direct scan-out flips (window/directPresent.h). Null when unavailable: KYTY_DIRECT_PRESENT not 1,
	// not an X11 window, or the X server cannot take the display buffers.
	[[nodiscard]] DirectPresenter* Direct() const noexcept;
	// Any thread, lock-free: a direct flip completed as FLIP. Any Vulkan present resets the streak.
	void OnDirectShown() noexcept;
	// VideoOut PresentThread, every tick; takes present_mutex. Once direct flips have run for about
	// half a second with no overlay to draw, the Vulkan swapchain is released, which ends the gamescope
	// WSI layer's override of the window's content and frees its images. The next Vulkan present
	// (menu, toast, fallback flip) creates it again.
	void MaintainDirect();
	// Any thread; takes present_mutex. Makes `frame` the main layer without presenting it, so the next
	// PresentLastFrame draws it under the system overlay. The previous main-layer frame goes back to the pool.
	void SetMainLayer(Frame& frame);
	// True once a swapchain was created in HDR10 this session; direct flips are off from then on.
	[[nodiscard]] bool HdrSwapchainEverUsed() const noexcept;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
