#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_

#include "common/common.h"
// #include "common/subsystems.h"

#include "common/abi.h"
#include "kernel/eventQueue.h"

#include <memory>

namespace Libs::Graphics {
class CommandBuffer;
class Presenter;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

struct VideoOutBufferAttribute2;
struct VideoOutFlipStatus;
struct VideoOutVblankStatus;
struct VideoOutOutputStatus;
struct VideoOutOutputOptions;
struct VideoOutBuffers;
struct VideoOutColorSettings;

class VideoOutDriver final {
public:
	struct Impl;

	VideoOutDriver(uint32_t width, uint32_t height, Graphics::Presenter& presenter);
	~VideoOutDriver();
	KYTY_CLASS_NO_COPY(VideoOutDriver);

	int  SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index, int flip_mode,
	                       int64_t flip_arg, uint64_t& request_id);
	void PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer);
	void CompleteFlip(uint64_t request_id);
	// `on_gpu_thread`: the GPU thread waits (PrepareVideoOutFlip), where a port full of CPU flips
	// that only it can prepare is a deadlock; another thread just waits.
	void WaitForSubmitSlot(int handle, bool on_gpu_thread = true);
	// `before`: only flip requests older than this id count (NextFlipRequestId() when the command
	// buffer that waits was submitted). `eop_mark`: an EOP flip reserved by SubmitEopFlip counts only
	// when its end-of-pipe release ran by then (EopFlipOrder() when the wait was reached; never
	// while its release has not run), as the PS5's wait on the buffer label sees it.
	[[nodiscard]] bool     IsSafeForRendering(int handle, int index, uint64_t before = UINT64_MAX,
	                                          uint64_t eop_mark = UINT64_MAX);
	[[nodiscard]] uint64_t NextFlipRequestId();
	// How many end-of-pipe releases of flips reserved by SubmitEopFlip have run (arrived at EopFlip).
	[[nodiscard]] uint64_t EopFlipOrder();

	// sceVideoOutSubmitEopFlip: reserves a flip that the GPU's end-of-pipe interrupt with context
	// `*ctx` will submit (EopFlip). With `wait_for_slot`, a full queue waits for a slot instead of
	// returning VIDEO_OUT_ERROR_FLIP_QUEUE_FULL.
	int SubmitEopFlip(int handle, int index, int flip_mode, int64_t flip_arg, uint32_t* ctx,
	                  bool wait_for_slot);
	// sceVideoOutGetBufferLabelAddress: the port's 16 display buffer labels (1 while the buffer is
	// or will be on screen, 0 once it left it), stable while the port is open.
	int GetBufferLabelAddress(int handle, uint64_t** labels);
	// The PS5 kernel's handling of an end-of-pipe interrupt: true when `ctx` is the context of a
	// reserved EOP flip, whose preparation is then queued on the GPU thread (false too when the GPU
	// is shutting down and drops it).
	bool EopFlip(uint32_t ctx);
	// Whether `addr` lies in some port's buffer labels, and which.
	bool IsBufferLabel(uint64_t addr, int* handle, int* index);

	[[nodiscard]] Impl& State() noexcept;

private:
	std::unique_ptr<Impl> m_impl;
};

[[nodiscard]] VideoOutDriver& VideoOutInit(uint32_t width, uint32_t height,
                                           Graphics::Presenter& presenter);
void                          VideoOutShutdown();

KYTY_SYSV_ABI int  VideoOutOpen(int user_id, int bus_type, int index, const void* param);
KYTY_SYSV_ABI int  VideoOutClose(int handle);
KYTY_SYSV_ABI int  VideoOutSetFlipMaster(int slave_handle, int master_handle);
KYTY_SYSV_ABI void VideoOutAddBufferAttributeOption(VideoOutBufferAttribute2* attribute,
                                                   uint64_t option);
KYTY_SYSV_ABI void VideoOutSetBufferAttribute2(VideoOutBufferAttribute2* attribute,
                                               uint64_t pixel_format, uint32_t tiling_mode,
                                               uint32_t width, uint32_t height, uint64_t option,
                                               uint32_t dcc_control,
                                               uint64_t dcc_cb_register_clear_color);
KYTY_SYSV_ABI int  VideoOutSetFlipRate(int handle, int rate);
KYTY_SYSV_ABI int  VideoOutAddFlipEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                        void* udata);
KYTY_SYSV_ABI int  VideoOutAddVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                          void* udata);
KYTY_SYSV_ABI int VideoOutAddPreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                                 void* udata);
KYTY_SYSV_ABI int VideoOutAddOutputModeEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                             void* udata);
KYTY_SYSV_ABI int VideoOutDeleteFlipEvent(LibKernel::EventQueue::KernelEqueue eq, int handle);
KYTY_SYSV_ABI int VideoOutDeleteVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle);
KYTY_SYSV_ABI int VideoOutDeletePreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq,
                                                    int                                 handle);
KYTY_SYSV_ABI int VideoOutRegisterBuffers2(int handle, int set_index, int buffer_index_start,
                                           const VideoOutBuffers* buffers, int buffer_num,
                                           const VideoOutBufferAttribute2* attribute, int category,
                                           void* option);
KYTY_SYSV_ABI int VideoOutSubmitChangeBufferAttribute2(int handle, int set_index,
                                                       const VideoOutBufferAttribute2* attribute,
                                                       void*                           option);
KYTY_SYSV_ABI int VideoOutUnregisterBuffers(int handle, int set_index);
KYTY_SYSV_ABI int VideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg);
KYTY_SYSV_ABI int VideoOutSubmitEopFlip(int handle, int index, int flip_mode, int64_t flip_arg,
                                        uint32_t* ctx);
KYTY_SYSV_ABI int VideoOutGetBufferLabelAddress(int handle, void** address);
KYTY_SYSV_ABI int VideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status);
KYTY_SYSV_ABI int VideoOutIsFlipPending(int handle);
KYTY_SYSV_ABI int VideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status);
KYTY_SYSV_ABI int VideoOutGetEventId(const LibKernel::EventQueue::KernelEvent* ev);
KYTY_SYSV_ABI int VideoOutGetEventData(const LibKernel::EventQueue::KernelEvent* ev, int64_t* data);
KYTY_SYSV_ABI int VideoOutGetEventCount(const LibKernel::EventQueue::KernelEvent* ev);
KYTY_SYSV_ABI int VideoOutWaitVblank(int handle);
KYTY_SYSV_ABI int VideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status);
KYTY_SYSV_ABI int VideoOutAllowOutputResolutionWqhdDetection(int handle);
KYTY_SYSV_ABI int VideoOutVrrPegToFixedRate(int handle, uint64_t arg1, uint64_t arg2);
KYTY_SYSV_ABI int VideoOutVrrUnpegFromFixedRate(int handle);
KYTY_SYSV_ABI int VideoOutInitializeOutputOptions(VideoOutOutputOptions* options);
KYTY_SYSV_ABI int VideoOutIsOutputSupported(int handle, uint64_t mode,
                                            const VideoOutOutputOptions* options,
                                            void* reserved_ptr, uint64_t reserved);
KYTY_SYSV_ABI int VideoOutConfigureOutput(int handle, uint64_t mode,
                                          const VideoOutOutputOptions* options, void* reserved_ptr,
                                          uint64_t reserved);
KYTY_SYSV_ABI int VideoOutSetWindowModeMargins(int handle, int top, int bottom);
KYTY_SYSV_ABI int VideoOutLatencyControlWaitBeforeInput(int handle);
KYTY_SYSV_ABI int VideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point);
KYTY_SYSV_ABI int VideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma);
KYTY_SYSV_ABI int VideoOutAdjustColor(int handle, const VideoOutColorSettings* settings);

} // namespace Libs::VideoOut

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_VIDEOOUT_H_ */
