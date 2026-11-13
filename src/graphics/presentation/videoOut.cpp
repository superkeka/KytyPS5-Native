#include "graphics/presentation/videoOut.h"

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/native_gpu/nativeShadow.h"
#include "graphics/presentation/performanceOverlay.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/renderDoc.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/window/directPresent.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "libs/hmd2.h"
#include "libs/libs.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <cstdio>
#include <array>
#include <atomic>
#include <chrono>
#include <fmt/format.h>
#include <list>
#include <span>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::VideoOut {

LIB_NAME("VideoOut", "VideoOut");

namespace EventQueue = LibKernel::EventQueue;

constexpr int      VIDEO_OUT_EVENT_FLIP                                 = 0;
constexpr int      VIDEO_OUT_EVENT_VBLANK                               = 1;
constexpr int      VIDEO_OUT_EVENT_PRE_VBLANK_START                     = 2;
constexpr int      VIDEO_OUT_EVENT_SET_MODE                             = 8;
constexpr int      VIDEO_OUT_TRUE                                       = 1;
constexpr int      VIDEO_OUT_FALSE                                      = 0;
constexpr int      VIDEO_OUT_BUS_TYPE_MAIN                              = 0;
constexpr int      VIDEO_OUT_BUS_TYPE_OVERLAY                           = 1;
constexpr int      VIDEO_OUT_BUS_TYPE_SUB                               = 2;
constexpr int      VIDEO_OUT_BUS_TYPE_HMD2_MAIN                          = 32;
constexpr int      VIDEO_OUT_FLIP_MODE_VSYNC                            = 1;
constexpr int      VIDEO_OUT_FLIP_MODE_HSYNC                            = 2;
constexpr int      VIDEO_OUT_FLIP_MODE_VSYNC_MULTI                      = 4;
constexpr int      VIDEO_OUT_FLIP_MODE_SLAVE                            = 8;
constexpr int      VIDEO_OUT_FLIP_MODE_MASTER                           = 9;
constexpr int      VIDEO_OUT_BUFFER_INDEX_BLACK                         = -2;
constexpr int      VIDEO_OUT_BUFFER_INDEX_BLANK                         = -1;
constexpr int      VIDEO_OUT_BUFFER_NUM_MAX                             = 16;
constexpr size_t   VIDEO_OUT_FLIP_QUEUE_CAPACITY                        = 16;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX                   = 4;
constexpr uint64_t VIDEO_OUT_OUTPUT_MODE_DEFAULT                        = 0x0000000000000001ULL;
constexpr uint64_t VIDEO_OUT_OUTPUT_MODE_119_88HZ                       = 0x000000000000000FULL;
constexpr uint64_t VIDEO_OUT_REFRESH_RATE_59_94HZ                       = 3;
constexpr uint64_t VIDEO_OUT_REFRESH_RATE_119_88HZ                      = 13;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED     = 0;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED       = 1;
constexpr int      VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_HMD2              = 0x10;
constexpr uint64_t VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_STRICT_COLORIMETRY = 8;
constexpr uint64_t VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_ALPHA_PREMULTIPLIED = 32;

enum class VideoOutEventKind : uintptr_t {
	Flip           = VIDEO_OUT_EVENT_FLIP,
	Vblank         = VIDEO_OUT_EVENT_VBLANK,
	PreVblankStart = VIDEO_OUT_EVENT_PRE_VBLANK_START,
	OutputMode     = VIDEO_OUT_EVENT_SET_MODE,
};

enum class FlipRequestSource { Cpu, GpuEop };

struct VideoOutEventState;

struct VideoOutEventRegistration {
	EventQueue::KernelEqueue            handle = EventQueue::KERNEL_EQUEUE_INVALID;
	std::shared_ptr<VideoOutEventState> state;
	VideoOutEventKind                   kind = VideoOutEventKind::Flip;
};

using VideoOutEventRegistrationRef = std::shared_ptr<VideoOutEventRegistration>;
using VideoOutEventQueues          = std::vector<VideoOutEventRegistrationRef>;

struct VideoOutEventState {
	Common::Mutex       mutex;
	VideoOutEventQueues flip;
	VideoOutEventQueues pre_vblank;
	VideoOutEventQueues vblank;
	VideoOutEventQueues output_mode;
};

struct VideoOutBufferAttribute2 {
	uint32_t reserved0;
	uint32_t tiling_mode;
	uint32_t aspect_ratio;
	uint32_t width;
	uint32_t height;
	uint32_t pitch_in_pixel;
	uint64_t option;
	uint64_t pixel_format;
	uint64_t dcc_cb_register_clear_color;
	uint32_t dcc_control;
	uint32_t pad0;
	uint64_t reserved1[3];
};

// PS5 layout
struct VideoOutFlipStatus {
	uint64_t count                    = 0;
	uint64_t processTime              = 0;
	uint64_t reserved0                = 0;
	int64_t  flipArg                  = 0;
	uint64_t reserved1                = 0;
	uint64_t processTimeCounter       = 0;
	int32_t  gcQueueNum               = 0;
	int32_t  flipPendingNum           = 0;
	int32_t  currentBuffer            = 0;
	uint32_t reserved2                = 0;
	uint64_t submitProcessTimeCounter = 0;
	uint64_t reserved3[7]             = {};
};

// PS5 layout
struct VideoOutVblankStatus {
	uint64_t count              = 0;
	uint64_t processTime        = 0;
	uint64_t reserved           = 0;
	uint64_t processTimeCounter = 0;
	uint8_t  flags              = 0;
	uint8_t  phase              = 0;
	uint8_t  pad1[6]            = {};
};

struct VideoOutOutputStatus {
	uint32_t resolution   = 0;
	uint32_t dynamicRange = 0;
	uint64_t refreshRate  = 0;
	uint64_t flags        = 0;
	uint64_t reserved[3]  = {};
};
static_assert(sizeof(VideoOutOutputStatus) == 0x30); // Also used by VideoOutVrrStatus.

struct VideoOutOutputOptions {
	uint32_t internalData[16] = {};
};

struct VideoOutColorSettings {
	float    gamma       = 1.0f;
	uint32_t reserved[3] = {};
};

struct VideoOutBuffers {
	const void* data;
	const void* metadata;
	const void* reserved[2];
};

struct VideoOutBuffer {
	int      group_index      = -1;
	uint64_t data_address     = 0;
	uint64_t metadata_address = 0;

	[[nodiscard]] bool Occupied() const noexcept { return group_index >= 0; }
};

struct BufferAttributeGroup {
	VideoOutBufferAttribute2 attribute {};
	int                      category = VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED;
	bool                     occupied = false;

	[[nodiscard]] Graphics::ImageInfo ImageInfo(const VideoOutBuffer& buffer) const;
};

// Direct presentation (KYTY_DIRECT_PRESENT=1). On the PS5 the display engine scans the display buffer
// itself out; here a display buffer that became one VRAM scan-out BO at registration
// (NativeGpu::ShadowCarveDisplayBuffer) is a DRI3 pixmap of the window, and its flips are
// PresentPixmaps (Graphics::DirectPresenter) instead of copies through Kyty's Vulkan presenter.
struct DirectSlot {
	uint32_t pixmap  = 0; // 0: the buffer's flips take the Vulkan path
	uint32_t width   = 0;
	uint32_t height  = 0;
	uint64_t address = 0; // the carved range, released at unregistration (size 0: not carved)
	uint64_t size    = 0;
};

// On the PS5 the display engine reads a buffer from the vblank that shows it until the vblank that
// shows another, and only then may the title render into it again. Here the reader is the display
// (Xwayland, gamescope, a DCN plane) from the buffer's PresentPixmap to its IdleNotify.
struct DisplayHold {
	uint32_t          presents  = 0; // PresentPixmaps of the buffer whose IdleNotify has not come
	uint64_t          flip_id   = 0; // the request of the last of them, as IsSafeForRendering counts it
	FlipRequestSource source    = FlipRequestSource::Cpu;
	uint64_t          eop_order = 0;
	double            since     = 0; // monotonic seconds: the last present, or when it left the screen
};

struct VideoOutConfig {
	Common::Mutex                       mutex;
	Common::CondVar                     vblank_cond;
	std::shared_ptr<VideoOutEventState> events      = std::make_shared<VideoOutEventState>();
	uint64_t                            generation  = 0; // Bumped on every open and close.
	bool                                opened      = false;
	int                                 bus         = 0;
	VideoOutConfig*                     master      = nullptr;
	uint32_t                            slave_count = 0;
	int                                 flip_rate   = 0;
	uint64_t                            output_mode = VIDEO_OUT_OUTPUT_MODE_DEFAULT;
	float                               gamma       = 1.0f;
	uint64_t                            pre_vblanks = 0;
	uint64_t                            flip_vblank = 0; // vblank_status.count at the last flip.
	VideoOutFlipStatus                  flip_status;
	uint64_t                            current_flip_id = 0; // the request that put flip_status.currentBuffer on screen
	// That request's source and, for an EOP flip reserved by SubmitEopFlip, the order in which its
	// end-of-pipe release ran (FlipQueue::TakeEopFlip; 0: not such a flip). See IsSafeForRendering.
	FlipRequestSource                   current_flip_source    = FlipRequestSource::Cpu;
	uint64_t                            current_flip_eop_order = 0;
	VideoOutVblankStatus                vblank_status;
	std::array<VideoOutBuffer, VIDEO_OUT_BUFFER_NUM_MAX>                 buffers;
	std::array<BufferAttributeGroup, VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX> groups;
	// sceVideoOutGetBufferLabelAddress: one 64-bit label per display buffer. The GPU's flip release
	// (AgcDcbSetFlip) writes 1, and FlipQueue::Flip writes 0 once another buffer is on screen; a
	// WaitUntilSafeForRendering waits for 0. Plain host memory the guest shares, zeroed at Open. It
	// lives as long as the driver, so a release still in flight after Close never writes freed memory.
	alignas(64) std::array<uint64_t, VIDEO_OUT_BUFFER_NUM_MAX> labels {};
	// Direct presentation, per buffer index from its registration. `holds` outlive the registration
	// and the port: they end at the display's IdleNotify (FlipQueue::OnDirectIdle).
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX>  direct {};
	std::array<DisplayHold, VIDEO_OUT_BUFFER_NUM_MAX> holds {};
	uint64_t direct_last_target = 0;     // target MSC of the port's last direct present (0: none yet)
	bool     last_shown_direct  = false; // the buffer on screen was shown directly, not by Kyty's presenter
	uint64_t direct_shown       = 0;     // direct flips shown on this port
};

class FlipQueue {
public:
	FlipQueue(Graphics::Presenter& presenter, uint32_t width, uint32_t height)
	    : m_presenter(presenter), m_width(width), m_height(height), m_direct(presenter.Direct()) {}
	~FlipQueue();
	KYTY_CLASS_NO_COPY(FlipQueue);

	bool Reserve(VideoOutConfig& cfg, int index, int flip_mode, int64_t flip_arg, FlipRequestSource source,
	             uint64_t& request_id);
	void Cancel(VideoOutConfig& cfg);
	void Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer);
	void Complete(uint64_t request_id);
	// `on_gpu_thread`: the caller is the GPU thread, which alone moves CPU flips into m_requests, so
	// a port full of CPU flips with nothing in m_requests can never drain there.
	void WaitForSubmitSlot(VideoOutConfig& cfg, bool on_gpu_thread);
	bool Flip();
	void GetFlipStatus(VideoOutConfig& cfg, VideoOutFlipStatus& out);
	bool     IsSafeForRendering(VideoOutConfig& cfg, int index, uint64_t before, uint64_t eop_mark);
	// The reserved, not yet submitted EOP flip whose interrupt context is `ctx`; it then gets the
	// next end-of-pipe order, so a repeated interrupt does not prepare it twice.
	bool TakeEopFlip(uint32_t ctx, uint64_t& request_id);
	void DropEopOrder(uint64_t request_id);
	uint64_t NextRequestId() {
		Common::LockGuard lock(m_mutex);
		return m_next_request_id;
	}
	// How many EOP flip releases have run so far (the last order TakeEopFlip gave).
	uint64_t EopOrder() {
		Common::LockGuard lock(m_mutex);
		return m_eop_order;
	}

	// Direct presentation (KYTY_DIRECT_PRESENT=1; null presenter otherwise, and nothing below runs).
	[[nodiscard]] Graphics::DirectPresenter* Direct() const noexcept { return m_direct; }
	// Off for the session: the display stopped answering, it copies instead of flipping, or the
	// connection is lost. Flips requested from then on take the Vulkan path.
	[[nodiscard]] bool DirectOff() {
		Common::LockGuard lock(m_mutex);
		return m_direct == nullptr || m_direct_off;
	}
	// The PS5 kernel's queueing of a flip to the display: a reserved request whose buffer is a
	// directly shown display buffer goes to the display now (at the EOP interrupt or the CPU submit),
	// with no GPU work. False, and nothing changed, when the request takes the Vulkan path.
	// `complete_follows`: the caller will still call Complete for it (SubmitFlipFromGpu).
	bool TryDirect(uint64_t request_id, bool complete_follows);
	// DirectPresenter callbacks, on its event thread.
	void OnDirectComplete(uint64_t token, uint64_t msc, uint64_t ust_us, Graphics::DirectCompleteMode mode);
	void OnDirectIdle(uint64_t token);
	void OnDirectLost();
	// The vblank thread, every tick: a display that stopped answering does not hang the title.
	void DirectWatchdog(double now);

private:
	// A direct request goes Reserved -> Ready (TryDirect) -> Presenting (sent to the display) and is
	// erased at its CompleteNotify (OnDirectComplete).
	enum class RequestState { Reserved, Recording, Ready, Presenting };

	struct Request {
		uint64_t                    id;
		uint64_t                    group;
		VideoOutConfig*             cfg;
		VideoOutConfig*             master;
		uint64_t                    generation;
		int                         index;
		int64_t                     flip_arg;
		uint64_t                    submit_ptc;
		FlipRequestSource           source;
		RequestState                state;
		Graphics::Presenter::Frame* frame;
		bool                        premultiplied_alpha;
		// EOP flip reserved by SubmitEopFlip: the order in which its end-of-pipe release ran (its
		// interrupt arrived and queued the preparation); 0 while it has not run.
		uint64_t eop_order;
		int      flip_mode;
		bool     direct;      // shown by a PresentPixmap of `pixmap`, not by the Vulkan presenter
		uint32_t pixmap;
		double   direct_sent;     // monotonic seconds when it was sent to the display
		double   direct_deadline; // the watchdog's limit for its CompleteNotify
	};
	void SignalDoneLocked();
	// Whether a pending request of the buffer holds it (IsSafeForRendering).
	static bool HoldsBuffer(const Request& r, uint64_t before, uint64_t eop_mark);
	// Whether a request of r's port older than r and outside r's group is still pending.
	bool OlderPendingOnPort(const Request& r) const;
	// Whether a request of r's port older than r has not reached the display yet: one for the Vulkan
	// presenter (until Flip shows it), or a direct one not sent yet.
	bool OlderUnsentOnPort(const Request& r) const;
	// The buffer's label goes to 0 (WaitUntilSafeForRendering may pass) once it is off the screen, the
	// display no longer holds it, and no pending request other than `except_id` that shows it again
	// has written 1 (an EOP flip whose release has not run has written nothing).
	void MaybeClearLabelLocked(VideoOutConfig& cfg, int index, uint64_t except_id);
	// Sends the port's ready direct requests to the display in request order, each after every older
	// request of the port reached it. Holds cfg.mutex and m_mutex.
	void PumpDirectLocked(VideoOutConfig& cfg, double now);
	// The same from a thread that holds no lock.
	void PumpDirect();
	// What Flip does when it shows a request, for a direct one: the flip status, the Flip event, the
	// label of the buffer it took off the screen. Holds its cfg.mutex and m_mutex; erases it.
	std::list<Request>::iterator CompleteDirectLocked(std::list<Request>::iterator it, double now);

	Graphics::Presenter& m_presenter;
	const uint32_t       m_width;
	const uint32_t       m_height;
	Common::Mutex        m_mutex;
	Common::CondVar      m_submit_slot_cond_var;
	Common::CondVar      m_done_cond_var;
	std::list<Request>   m_requests;
	std::list<Request>   m_cpu_requests;
	std::list<Request>   m_cancelled_requests;
	bool                 m_gpu_waiting     = false;
	uint64_t             m_next_request_id = 1;
	uint64_t             m_eop_order       = 0;

	struct DirectToken {
		VideoOutConfig* cfg;
		int             index;
	};
	Graphics::DirectPresenter* m_direct = nullptr;
	// Every PresentPixmap until its IdleNotify: which buffer the display holds for it.
	std::unordered_map<uint64_t, DirectToken> m_direct_tokens;
	// Direct requests from SubmitFlipFromGpu, whose caller still calls Complete.
	std::unordered_set<uint64_t> m_direct_noop_completes;
	// Ports other than the main one that have a frame on screen: Kyty's presenter composes them.
	std::unordered_set<const VideoOutConfig*> m_side_shown;
	uint32_t m_copy_streak   = 0;
	bool     m_direct_off    = false; // no new direct requests this session
	bool     m_direct_failed = false; // nothing more is sent; what was sent completes as skipped
	uint64_t m_direct_skips  = 0;
};

struct VideoOutDriver::Impl {
public:
	static constexpr std::array VIDEO_OUT_BUSES {VIDEO_OUT_BUS_TYPE_MAIN,
	                                             VIDEO_OUT_BUS_TYPE_OVERLAY, VIDEO_OUT_BUS_TYPE_SUB,
	                                             VIDEO_OUT_BUS_TYPE_HMD2_MAIN};
	static constexpr int VIDEO_OUT_NUM_MAX = VIDEO_OUT_BUSES.size() + 1;

	Impl(uint32_t width, uint32_t height, Graphics::Presenter& presenter)
	    : width(width), height(height), m_renderer(presenter.Renderer()), m_presenter(presenter),
	      m_flip_queue(presenter, width, height) {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		RegisterDirectCallbacks();
		m_present_thread = std::jthread([this](std::stop_token token) { PresentThread(token); });
	}
	~Impl();
	KYTY_CLASS_NO_COPY(Impl);

	int             Open(int bus_type, int index);
	bool            Close(int handle);
	VideoOutConfig* Get(int handle);
	int             SetFlipMaster(int slave_handle, int master_handle);
	// The labels are members of the fixed port array, so finding one needs no lock.
	bool FindLabel(uint64_t addr, int& handle, int& index) const;

	FlipQueue&               GetFlipQueue() { return m_flip_queue; }
	Graphics::RenderContext& Renderer() const noexcept { return m_renderer; }
	Graphics::Presenter&     GetPresenter() const noexcept { return m_presenter; }
	bool                     IsHdrSupported() const noexcept { return m_presenter.IsHdrSupported(); }

	void VblankBegin();
	void VblankEnd();
	void PresentThread(std::stop_token token);

	const uint32_t width; // The output size of every port.
	const uint32_t height;

private:
	void RegisterDirectCallbacks();
	// Direct presentation's vblank-thread work: the watchdog and the swapchain's release.
	void MaintainDirect();
	// Before Kyty's presenter draws the system overlay over its last frame (see the definition).
	bool OverlayFrameReady();
	[[nodiscard]] bool IsOpenedLocked(int handle) const {
		return handle > 0 && handle < VIDEO_OUT_NUM_MAX && m_video_out_ctx[handle].opened;
	}

	Common::Mutex            m_mutex;
	VideoOutConfig           m_video_out_ctx[VIDEO_OUT_NUM_MAX];
	Graphics::RenderContext& m_renderer;
	Graphics::Presenter&     m_presenter;
	FlipQueue                m_flip_queue;
	// The overlay's copy of a directly shown buffer (OverlayFrameReady), all on the vblank thread but
	// the hand-over: for which direct flip of the main port it was asked and when, the copy the GPU
	// thread's completion hands over, which one is the presenter's main layer, and over which one the
	// overlay was presented.
	struct OverlayCopy {
		Common::Mutex               mutex;
		uint64_t                    serial = 0;
		Graphics::Presenter::Frame* frame  = nullptr;
	};
	uint64_t                     m_overlay_asked_serial = 0;
	double                       m_overlay_asked_time   = 0;
	std::shared_ptr<OverlayCopy> m_overlay_copy         = std::make_shared<OverlayCopy>();
	uint64_t                     m_overlay_ready_serial = 0;
	uint64_t                     m_overlay_shown_serial = 0;
	std::jthread             m_present_thread;
};

static std::unique_ptr<VideoOutDriver> g_video_out_driver;

static VideoOutDriver::Impl& DriverState() {
	EXIT_IF(g_video_out_driver == nullptr);
	return g_video_out_driver->State();
}

static uintptr_t VideoOutEventId(VideoOutEventKind kind) {
	return static_cast<uintptr_t>(kind);
}

static VideoOutEventQueues& VideoOutEventQueuesFor(VideoOutEventState& state,
                                                   VideoOutEventKind   kind) {
	switch (kind) {
		case VideoOutEventKind::Flip: return state.flip;
		case VideoOutEventKind::Vblank: return state.vblank;
		case VideoOutEventKind::PreVblankStart: return state.pre_vblank;
		case VideoOutEventKind::OutputMode: return state.output_mode;
	}
	EXIT("unsupported video-out event kind\n");
	return state.flip;
}

static intptr_t MakeVideoOutEventData(intptr_t current_data, void* trigger_data) {
	const uint64_t old_data = static_cast<uint64_t>(current_data);
	uint64_t       counter  = (old_data >> 12u) & 0xfu;
	if (counter != 0xfu) {
		counter++;
	}

	const uint64_t time    = LibKernel::KernelReadTsc() & 0xfffu;
	const uint64_t payload = static_cast<uint64_t>(reinterpret_cast<intptr_t>(trigger_data));

	return static_cast<intptr_t>(time | (counter << 12u) |
	                             ((payload & 0x0000ffffffffffffULL) << 16u));
}

static void ResetVideoOutEvent(EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void TriggerVideoOutEvent(EventQueue::KernelEqueueEvent* event, void* trigger_data) {
	EXIT_IF(event == nullptr);

	// One event counts every occurrence until it is delivered.
	event->event.fflags = std::min(event->event.fflags + 1u, 0xfu);
	event->event.data   = MakeVideoOutEventData(event->event.data, trigger_data);
	event->triggered    = true;
}

static void RemoveVideoOutEventQueue(EventQueue::KernelEqueue       eq,
                                     EventQueue::KernelEqueueEvent* event) {
	if (event == nullptr || event->filter.data == nullptr) {
		return;
	}

	auto* registration = static_cast<VideoOutEventRegistration*>(event->filter.data);
	auto  state        = registration->state;
	if (registration->handle != eq || !state) {
		return;
	}
	auto&             queues = VideoOutEventQueuesFor(*state, registration->kind);
	Common::LockGuard lock(state->mutex);
	const auto        entry =
	    std::find_if(queues.begin(), queues.end(), [registration](const auto& candidate) {
		    return candidate.get() == registration;
	    });
	if (entry != queues.end()) {
		queues.erase(entry);
	}
}

static void TriggerVideoOutEvents(VideoOutConfig& video_out, VideoOutEventKind kind,
                                  void* trigger_data) {
	VideoOutEventQueues queues;
	{
		Common::LockGuard lock(video_out.events->mutex);
		queues = VideoOutEventQueuesFor(*video_out.events, kind);
	}
	static const bool log = [] {
		const char* v = std::getenv("KYTY_LOG_FLIPS");
		return v != nullptr && v[0] == '1';
	}();
	if (log && !queues.empty()) {
		struct timespec ts {};
		clock_gettime(CLOCK_MONOTONIC, &ts);
		std::fprintf(stderr, "vo [%.2f] event %d data %" PRIu64 " queues %zu\n",
		             static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6, static_cast<int>(kind),
		             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(trigger_data)), queues.size());
	}
	for (const auto& registration: queues) {
		const auto result =
		    EventQueue::KernelTriggerEvent(registration->handle, VideoOutEventId(kind),
		                                   EventQueue::KERNEL_EVFILT_VIDEO_OUT, trigger_data);
		EXIT_NOT_IMPLEMENTED(result != OK && result != LibKernel::KERNEL_ERROR_EBADF &&
		                     result != LibKernel::KERNEL_ERROR_ENOENT);
	}
}

static void DeleteVideoOutEvents(const VideoOutEventQueues& queues, VideoOutEventKind kind) {
	for (const auto& registration: queues) {
		const auto result = EventQueue::KernelDeleteEvent(
		    registration->handle, VideoOutEventId(kind), EventQueue::KERNEL_EVFILT_VIDEO_OUT);
		EXIT_NOT_IMPLEMENTED(result != OK && result != LibKernel::KERNEL_ERROR_EBADF &&
		                     result != LibKernel::KERNEL_ERROR_ENOENT);
	}
}

static int RegisterVideoOutEvent(int handle, EventQueue::KernelEqueue eq, VideoOutEventKind kind,
                                 void* udata) {
	auto* video_out = DriverState().Get(handle);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	Common::LockGuard lock(video_out->mutex);
	if (!video_out->opened) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (kind == VideoOutEventKind::OutputMode) {
		LOGF("\t eq     = 0x%016" PRIx64 "\n"
		     "\t handle = %d\n"
		     "\t udata  = 0x%016" PRIx64 "\n",
		     static_cast<uint64_t>(eq), handle, reinterpret_cast<uint64_t>(udata));
	}
	if (eq == EventQueue::KERNEL_EQUEUE_INVALID) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	if (!EventQueue::KernelPinEqueue(eq)) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	auto        event_state         = video_out->events;
	auto&       queues              = VideoOutEventQueuesFor(*event_state, kind);
	const bool  initially_triggered = kind == VideoOutEventKind::OutputMode;
	void* const initial_trigger_data =
	    initially_triggered ? reinterpret_cast<void*>(video_out->output_mode) : nullptr;

	EventQueue::KernelEqueueEvent event {};
	event.triggered    = initially_triggered;
	event.event.ident  = VideoOutEventId(kind);
	event.event.filter = EventQueue::KERNEL_EVFILT_VIDEO_OUT;
	event.event.udata  = udata;
	event.event.fflags = initially_triggered ? 1u : 0u;
	event.event.data   = initially_triggered ? MakeVideoOutEventData(0, initial_trigger_data) : 0;
	event.filter.delete_event_func = RemoveVideoOutEventQueue;
	event.filter.reset_func        = ResetVideoOutEvent;
	event.filter.trigger_func      = TriggerVideoOutEvent;

	VideoOutEventRegistrationRef registration;
	bool                         add_queue = false;
	{
		Common::LockGuard event_lock(event_state->mutex);
		const auto        existing =
		    std::find_if(queues.begin(), queues.end(),
			             [eq](const auto& candidate) { return candidate->handle == eq; });
		if (existing != queues.end()) {
			registration = *existing;
		} else {
			registration = std::make_shared<VideoOutEventRegistration>(
			    VideoOutEventRegistration {.handle = eq, .state = event_state, .kind = kind});
			queues.push_back(registration);
			add_queue = true;
		}
	}
	event.filter.data  = registration.get();
	event.filter.owner = registration;
	const int result   = EventQueue::KernelAddEvent(eq, event);
	if (result != OK && add_queue) {
		Common::LockGuard event_lock(event_state->mutex);
		const auto        added = std::find(queues.begin(), queues.end(), registration);
		if (added != queues.end()) {
			queues.erase(added);
		}
	}
	return result == LibKernel::KERNEL_ERROR_EBADF ? VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE : result;
}

static int DeleteVideoOutEvent(int handle, EventQueue::KernelEqueue eq, VideoOutEventKind kind) {
	auto* video_out = DriverState().Get(handle);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	Common::LockGuard lock(video_out->mutex);
	if (!video_out->opened) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (!EventQueue::KernelPinEqueue(eq)) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	const int result = EventQueue::KernelDeleteEvent(eq, VideoOutEventId(kind),
	                                                 EventQueue::KERNEL_EVFILT_VIDEO_OUT);
	if (result == LibKernel::KERNEL_ERROR_EBADF) {
		return VIDEO_OUT_ERROR_INVALID_EVENT_QUEUE;
	}
	return result == LibKernel::KERNEL_ERROR_ENOENT ? OK : result;
}

// A flip is due once flip_rate + 1 vblanks have passed since the last flip, so a late frame flips
// at the next vblank.
static bool IsFlipDueLocked(const VideoOutConfig& cfg) {
	const auto interval = static_cast<uint64_t>(cfg.flip_rate) + 1;

	return interval <= 1 || cfg.vblank_status.count - cfg.flip_vblank >= interval;
}

static bool IsValidBufferIndex(int index) {
	return index >= VIDEO_OUT_BUFFER_INDEX_BLACK && index < VIDEO_OUT_BUFFER_NUM_MAX;
}

static bool IsSpecialBufferIndex(int index) {
	return index == VIDEO_OUT_BUFFER_INDEX_BLANK || index == VIDEO_OUT_BUFFER_INDEX_BLACK;
}

static bool IsValidFlipMode(int mode) {
	return (mode >= VIDEO_OUT_FLIP_MODE_VSYNC && mode <= VIDEO_OUT_FLIP_MODE_VSYNC_MULTI) ||
	       mode == VIDEO_OUT_FLIP_MODE_SLAVE || mode == VIDEO_OUT_FLIP_MODE_MASTER;
}

// The end-of-pipe interrupt context of an EOP flip request (sceVideoOutSubmitEopFlip). Bit 27 marks a
// kernel context as the PS5 kernel's do, so the interrupt goes to VideoOut and not to the title's
// event queues (RenderContext::TriggerInterrupt). Bits 8..13 stay 0, so libSceAgc's flip marker
// decodes it with p = 0. At most 16 requests per port are pending, so the low byte of the request id
// tells them apart.
constexpr uint32_t VIDEO_OUT_EOP_FLIP_CONTEXT = 0x08000000u;

static uint32_t EopFlipContext(uint64_t request_id) {
	return VIDEO_OUT_EOP_FLIP_CONTEXT | static_cast<uint32_t>(request_id & 0xffu);
}

// Set once the headset port shows eye images. The window then shows them instead of the main port,
// so the picture is visible without a headset.
static std::atomic<bool> g_headset_mirror {false};

// Display buffers that stop being shown directly: their pixmaps go, and their carves stop being
// registered display buffers (the VRAM BO and the guest mapping stay; GemBacking reuses a carve when
// the range is registered again). Called without VideoOut locks: freeing a pixmap talks to the X
// server, and the carve's release takes GemBacking's lock, which no VideoOut lock may be held under.
static void ReleaseDirectSlots(Graphics::DirectPresenter* direct, std::span<const DirectSlot> slots) {
	for (const auto& slot: slots) {
		if (slot.pixmap != 0 && direct != nullptr) {
			direct->RemoveBuffer(slot.pixmap);
		}
		if (slot.size != 0) {
			Graphics::NativeGpu::ShadowReleaseDisplayBuffer(slot.address, slot.size);
		}
	}
}

static int ReserveFlipRequest(VideoOutDriver::Impl& driver, int handle, int index, int flip_mode,
                              int64_t flip_arg, FlipRequestSource source, uint64_t& request_id) {
	auto* video_out = driver.Get(handle);
	if (video_out == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (!IsValidFlipMode(flip_mode)) {
		return VIDEO_OUT_ERROR_INVALID_FLIP_MODE;
	}
	if (!IsValidBufferIndex(index)) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	Common::LockGuard lock(video_out->mutex);
	if ((video_out->master != nullptr) != (flip_mode == VIDEO_OUT_FLIP_MODE_SLAVE) ||
	    (video_out->slave_count != 0) != (flip_mode == VIDEO_OUT_FLIP_MODE_MASTER)) {
		return VIDEO_OUT_ERROR_INVALID_FLIP_MODE;
	}
	// The headset port can flip before its reprojection buffers are registered; those show blank.
	if (!video_out->opened ||
	    (!IsSpecialBufferIndex(index) && !video_out->buffers[index].Occupied() &&
	     video_out->bus != VIDEO_OUT_BUS_TYPE_HMD2_MAIN)) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}
	if (!driver.GetFlipQueue().Reserve(*video_out, index, flip_mode, flip_arg, source, request_id)) {
		return VIDEO_OUT_ERROR_FLIP_QUEUE_FULL;
	}
	return OK;
}

Graphics::ImageInfo BufferAttributeGroup::ImageInfo(const VideoOutBuffer& buffer) const {
	auto compression = Graphics::ClassifyVideoOutCompression(
	    category == VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED, buffer.metadata_address,
	    attribute.dcc_control, attribute.dcc_cb_register_clear_color);
	if (attribute.reserved0 != 0 || attribute.aspect_ratio != 0 || attribute.width == 0 ||
	    attribute.height == 0 || attribute.width > 16384 || attribute.height > 16384 ||
	    attribute.pitch_in_pixel != 0 ||
	    (attribute.option & ~(VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_STRICT_COLORIMETRY |
	                          VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_ALPHA_PREMULTIPLIED)) != 0 ||
	    attribute.tiling_mode != 0 || attribute.pad0 != 0 || attribute.reserved1[0] != 0 ||
	    attribute.reserved1[1] != 0 || attribute.reserved1[2] != 0 || buffer.data_address == 0 ||
	    compression == Graphics::VideoOutCompression::Unsupported) {
		EXIT("unsupported or invalid video-out surface attributes\n");
	}
	Graphics::VideoOutPixelFormatInfo pixel_format {};
	if (!Graphics::DecodeVideoOutPixelFormat(attribute.pixel_format, pixel_format)) {
		EXIT("unsupported video-out pixel format: 0x%016" PRIx64 "\n", attribute.pixel_format);
	}
	const auto          tile_mode = Graphics::Prospero::TileMode::kRenderTarget;
	Graphics::ImageInfo info {};
	info.data            = {buffer.data_address, 0};
	info.pixel_format    = pixel_format.format;
	info.guest_format    = pixel_format.guest_format;
	info.type            = Graphics::Prospero::ImageType::kColor2D;
	info.extent          = {attribute.width, attribute.height, 1};
	info.resources       = {1, 1};
	info.bytes_per_block = pixel_format.bytes_per_element;
	info.samples         = 1;
	info.tile_mode       = tile_mode;
	info.bgra16          = pixel_format.bgra16;
	info.UpdateSize();
	if (info.data.size == 0 || info.tiling.block.block_size != 65536 ||
	    (buffer.data_address & 65535u) != 0) {
		EXIT("invalid video-out surface footprint or alignment\n");
	}
	// Natively rendered display buffers hold plain pixels (ShadowDisplayWithoutDcc).
	if (compression != Graphics::VideoOutCompression::Uncompressed &&
	    Graphics::NativeGpu::ShadowDisplayWithoutDcc(buffer.data_address, info.data.size)) {
		compression = Graphics::VideoOutCompression::Uncompressed;
	}
	if (compression != Graphics::VideoOutCompression::Uncompressed) {
		Graphics::TileSizeAlign dcc_size {};
		if (!Graphics::TileGetDccSize(attribute.width, attribute.height, 1,
		                              pixel_format.bytes_per_element, 1, tile_mode, dcc_size)) {
			EXIT("invalid video-out DCC footprint\n");
		}
		info.metadata.range       = {buffer.metadata_address, dcc_size.size};
		info.metadata.kind        = Graphics::ImageMetadataKind::Dcc;
		info.metadata.compression = compression;
	}
	Graphics::ImageOps::Validate(info);
	if (!Graphics::IsSupportedVideoOutFormat(info)) {
		EXIT("unsupported normalized video-out format\n");
	}
	return info;
}

VideoOutDriver::VideoOutDriver(uint32_t width, uint32_t height, Graphics::Presenter& presenter)
    : m_impl(std::make_unique<Impl>(width, height, presenter)) {}

VideoOutDriver::~VideoOutDriver() = default;

VideoOutDriver::Impl& VideoOutDriver::State() noexcept {
	return *m_impl;
}

VideoOutDriver& VideoOutInit(uint32_t width, uint32_t height, Graphics::Presenter& presenter) {
	EXIT_IF(g_video_out_driver != nullptr);
	g_video_out_driver = std::make_unique<VideoOutDriver>(width, height, presenter);
	return *g_video_out_driver;
}

void VideoOutShutdown() {
	g_video_out_driver.reset();
}

VideoOutDriver::Impl::~Impl() {
	// No DirectPresenter callback runs into the queue from here on (SetCallbacks waits for one in
	// progress); presents still in flight are never completed, which no one waits for any more.
	if (auto* direct = m_flip_queue.Direct(); direct != nullptr) {
		direct->SetCallbacks({});
	}
	if (m_present_thread.joinable()) {
		m_present_thread.request_stop();
		m_present_thread.join();
	}
	for (int handle = 1; handle < VIDEO_OUT_NUM_MAX; handle++) {
		if (m_video_out_ctx[handle].master != nullptr) {
			(void)Close(handle);
		}
	}
	for (int handle = 1; handle < VIDEO_OUT_NUM_MAX; handle++) {
		(void)Close(handle);
	}
}

int VideoOutDriver::Impl::Open(int bus_type, int index) {
	const auto bus = std::find(VIDEO_OUT_BUSES.begin(), VIDEO_OUT_BUSES.end(), bus_type);
	if (bus == VIDEO_OUT_BUSES.end()) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}
	EXIT_NOT_IMPLEMENTED(index != 0);
	Common::LockGuard lock(m_mutex);

	const int handle = static_cast<int>(bus - VIDEO_OUT_BUSES.begin()) + 1;
	if (m_video_out_ctx[handle].opened) {
		return VIDEO_OUT_ERROR_RESOURCE_BUSY;
	}
	auto&             config = m_video_out_ctx[handle];
	Common::LockGuard config_lock(config.mutex);

	{
		Common::LockGuard event_lock(config.events->mutex);
		EXIT_IF(!config.events->flip.empty());
		EXIT_IF(!config.events->pre_vblank.empty());
		EXIT_IF(!config.events->vblank.empty());
		EXIT_IF(!config.events->output_mode.empty());
	}
	EXIT_IF(config.flip_rate != 0);
	for (const auto& buffer: config.buffers) {
		EXIT_IF(buffer.Occupied());
	}
	for (const auto& group: config.groups) {
		EXIT_IF(group.occupied);
	}

	config.opened = true;
	config.bus    = bus_type;
	if (++config.generation == 0) {
		EXIT("video-out port generation wrapped\n");
	}
	config.output_mode               = VIDEO_OUT_OUTPUT_MODE_DEFAULT;
	config.flip_status               = VideoOutFlipStatus();
	config.flip_status.flipArg       = -1;
	config.flip_status.currentBuffer = -1;
	config.pre_vblanks               = 0;
	config.flip_vblank               = 0;
	config.vblank_status             = VideoOutVblankStatus();
	for (auto& label: config.labels) {
		std::atomic_ref<uint64_t>(label).store(0, std::memory_order_release);
	}

	return handle;
}

bool VideoOutDriver::Impl::Close(int handle) {
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> direct_slots {};
	const bool closed = [&] {
		Common::LockGuard lock(m_mutex);

		if (!IsOpenedLocked(handle)) {
			return false;
		}

		auto&               config = m_video_out_ctx[handle];
		VideoOutEventQueues flip_events;
		VideoOutEventQueues pre_vblank_events;
		VideoOutEventQueues vblank_events;
		VideoOutEventQueues output_mode_events;
		VideoOutConfig* master = nullptr;
		{
			Common::LockGuard config_lock(config.mutex);
			EXIT_IF(config.slave_count != 0);
			config.opened = false;
			if (++config.generation == 0) {
				EXIT("video-out port generation wrapped\n");
			}
			{
				Common::LockGuard event_lock(config.events->mutex);
				flip_events        = std::move(config.events->flip);
				pre_vblank_events  = std::move(config.events->pre_vblank);
				vblank_events      = std::move(config.events->vblank);
				output_mode_events = std::move(config.events->output_mode);
			}
			config.flip_rate   = 0;
			master             = config.master;
			config.master      = nullptr;

			for (const auto& buffer: config.buffers) {
				if (buffer.Occupied() &&
				    (buffer.group_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
				     buffer.data_address == 0 || !config.groups[buffer.group_index].occupied)) {
					EXIT("inconsistent registered video-out buffer state\n");
				}
			}
			for (auto& buffer: config.buffers) {
				buffer = VideoOutBuffer {};
			}
			for (auto& group: config.groups) {
				group = BufferAttributeGroup {};
			}
			direct_slots = config.direct;
			config.direct.fill(DirectSlot {});
			config.vblank_cond.SignalAll();
		}
		if (master != nullptr) {
			Common::LockGuard master_lock(master->mutex);
			master->slave_count--;
		}
		m_flip_queue.Cancel(config);
		if (config.bus == VIDEO_OUT_BUS_TYPE_HMD2_MAIN) {
			g_headset_mirror = false;
		}
		m_presenter.ClearLayer(config.bus);
		DeleteVideoOutEvents(flip_events, VideoOutEventKind::Flip);
		DeleteVideoOutEvents(pre_vblank_events, VideoOutEventKind::PreVblankStart);
		DeleteVideoOutEvents(vblank_events, VideoOutEventKind::Vblank);
		DeleteVideoOutEvents(output_mode_events, VideoOutEventKind::OutputMode);
		return true;
	}();
	// Outside VideoOut's locks: freeing a pixmap talks to the X server, and the carve's release takes
	// GemBacking's lock, which is never taken under a VideoOut lock. Presents of these pixmaps still in
	// flight complete and go idle as usual.
	ReleaseDirectSlots(m_flip_queue.Direct(), direct_slots);
	return closed;
}

VideoOutConfig* VideoOutDriver::Impl::Get(int handle) {
	Common::LockGuard lock(m_mutex);

	return IsOpenedLocked(handle) ? m_video_out_ctx + handle : nullptr;
}

bool VideoOutDriver::Impl::FindLabel(uint64_t addr, int& handle, int& index) const {
	for (int h = 1; h < VIDEO_OUT_NUM_MAX; h++) {
		const auto base = reinterpret_cast<uint64_t>(m_video_out_ctx[h].labels.data());
		if (addr >= base && addr < base + sizeof(m_video_out_ctx[h].labels)) {
			handle = h;
			index  = static_cast<int>((addr - base) / sizeof(uint64_t));
			return true;
		}
	}
	return false;
}

int VideoOutDriver::Impl::SetFlipMaster(int slave_handle, int master_handle) {
	Common::LockGuard lock(m_mutex);
	if (!IsOpenedLocked(slave_handle) || !IsOpenedLocked(master_handle)) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	auto& slave  = m_video_out_ctx[slave_handle];
	auto& master = m_video_out_ctx[master_handle];
	if (&slave == &master) {
		return VIDEO_OUT_ERROR_UNSUPPORTED_OPERATION;
	}
	Common::LockGuard first_lock(m_video_out_ctx[std::min(slave_handle, master_handle)].mutex);
	Common::LockGuard second_lock(m_video_out_ctx[std::max(slave_handle, master_handle)].mutex);
	if (slave.master != nullptr || slave.slave_count != 0 || master.master != nullptr) {
		return VIDEO_OUT_ERROR_UNSUPPORTED_OPERATION;
	}
	if (slave.flip_status.flipPendingNum != 0 || master.flip_status.flipPendingNum != 0) {
		return VIDEO_OUT_ERROR_RESOURCE_BUSY;
	}
	slave.master = &master;
	master.slave_count++;
	return OK;
}

void VideoOutDriver::Impl::VblankBegin() {
	Common::LockGuard lock(m_mutex);

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		auto& ctx = m_video_out_ctx[i];
		if (ctx.opened) {
			ctx.mutex.Lock();
			TriggerVideoOutEvents(ctx, VideoOutEventKind::PreVblankStart,
			                      reinterpret_cast<void*>(++ctx.pre_vblanks));
			ctx.mutex.Unlock();
		}
	}
}

void VideoOutDriver::Impl::VblankEnd() {
	Common::LockGuard lock(m_mutex);

	for (int i = 1; i < VIDEO_OUT_NUM_MAX; i++) {
		auto& ctx = m_video_out_ctx[i];
		if (ctx.opened) {
			ctx.mutex.Lock();
			ctx.vblank_status.count++;
			ctx.vblank_status.processTime        = LibKernel::KernelGetProcessTime();
			ctx.vblank_status.reserved           = LibKernel::KernelReadTsc();
			ctx.vblank_status.processTimeCounter = LibKernel::KernelGetProcessTimeCounter();

			TriggerVideoOutEvents(ctx, VideoOutEventKind::Vblank,
			                      reinterpret_cast<void*>(ctx.vblank_status.count));
			ctx.vblank_cond.SignalAll();
			ctx.mutex.Unlock();
		}
	}
}

void VideoOutDriver::Impl::PresentThread(std::stop_token token) {
	using Clock       = std::chrono::steady_clock;
	const auto period = Clock::duration(std::chrono::seconds(1)) / Config::GetVblankFrequency();

	std::chrono::nanoseconds total_wait {};
	while (!token.stop_requested()) {
		const auto sleep_begin = Clock::now();
		if (total_wait.count() > 0) {
			Common::Thread::SleepNano(static_cast<uint64_t>(total_wait.count()));
		}
		if (token.stop_requested()) {
			break;
		}
		const auto frame_begin = Clock::now();
		total_wait -= frame_begin - sleep_begin;

		VblankBegin();
		bool presented = m_flip_queue.Flip();
		if (!presented && m_presenter.NeedsSystemOverlayRefresh() && OverlayFrameReady()) {
			if (!m_presenter.PresentLastFrame()) {
				m_presenter.Present(m_presenter.PrepareBlankFrame(width, height, true));
			}
			presented = true;
		}
		if (!presented && total_wait.count() < 0) {
			bool any_open = false;
			{
				Common::LockGuard lock(m_mutex);
				for (int handle = 1; handle < VIDEO_OUT_NUM_MAX; handle++) {
					any_open |= m_video_out_ctx[handle].opened;
				}
			}
			if (!any_open) {
				m_presenter.Present(m_presenter.PrepareBlankFrame(width, height, true));
			}
		}
		VblankEnd();
		MaintainDirect();

		total_wait += period - (Clock::now() - frame_begin);
		if (total_wait.count() < 0) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::LateVblanks);
		}
	}
}

// KYTY_LOG_FLIPS=1: every flip request through the queue (reserve, prepare, complete, present) and
// each refusal of IsSafeForRendering, to find what holds a display buffer.
// KYTY_LOG_FLIPS=1: each flip request's life and every refusal to render into a buffer, on stderr.
static bool LogFlips() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_LOG_FLIPS");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

// Milliseconds on the monotonic clock, for the flip log's timeline.
static double FlipLogMs() {
	struct timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}

// KYTY_LOG_FLIPS=1 also logs what the title asks about flips and vblanks (a status call only when its
// answer changed), so a frame the title starts late shows what it was waiting for.
static void VoLog(const char* what, int64_t a = 0, int64_t b = 0, int64_t c = 0) {
	std::fprintf(stderr, "vo [%.2f] %s %" PRId64 " %" PRId64 " %" PRId64 "\n", FlipLogMs(), what, a, b, c);
}

// Monotonic seconds, for direct presentation's watchdog and display holds.
static double DirectNow() {
	return FlipLogMs() * 1e-3;
}

void VideoOutDriver::Impl::RegisterDirectCallbacks() {
	auto* direct = m_flip_queue.Direct();
	if (direct == nullptr) {
		return;
	}
	Graphics::DirectPresentCallbacks callbacks;
	callbacks.complete = [this](uint64_t token, uint64_t msc, uint64_t ust_us, Graphics::DirectCompleteMode mode) {
		m_flip_queue.OnDirectComplete(token, msc, ust_us, mode);
	};
	callbacks.idle = [this](uint64_t token) { m_flip_queue.OnDirectIdle(token); };
	// Every direct flip checks the window's size (DirectPresenter::WindowIs), so a resize needs nothing here.
	callbacks.resized = [](uint32_t width, uint32_t height) {
		if (LogFlips()) {
			std::fprintf(stderr, "flip [%.2f]: direct present: window is %ux%u\n", FlipLogMs(), width, height);
		}
	};
	callbacks.lost = [this] { m_flip_queue.OnDirectLost(); };
	direct->SetCallbacks(std::move(callbacks));
}

void VideoOutDriver::Impl::MaintainDirect() {
	if (m_flip_queue.Direct() == nullptr) {
		return;
	}
	m_flip_queue.DirectWatchdog(DirectNow());
	m_presenter.MaintainDirect();
}

// Kyty's presenter draws the system overlay (game menu, toasts, cursor) over the last frame it
// presented itself. When the main port's last flip was shown directly, that frame is older than what
// is on screen, so the buffer on screen is first copied into a presenter frame on the GPU thread, as a
// CPU flip is prepared, once per directly shown flip. True when the presenter may present now: always
// when the last flip was not direct (as without direct presentation); otherwise once that copy is
// the presenter's main layer, or 200 ms after it was asked for (the overlay then goes over the older
// frame rather than not at all). With no overlay visible over the direct flip on screen, nothing
// needs repainting: false.
bool VideoOutDriver::Impl::OverlayFrameReady() {
	if (m_flip_queue.Direct() == nullptr) {
		return true;
	}
	// A copy the GPU thread has finished.
	Graphics::Presenter::Frame* copy        = nullptr;
	uint64_t                    copy_serial = 0;
	{
		Common::LockGuard lock(m_overlay_copy->mutex);
		copy        = std::exchange(m_overlay_copy->frame, nullptr);
		copy_serial = m_overlay_copy->serial;
	}
	auto&               main    = m_video_out_ctx[1]; // VIDEO_OUT_BUSES[0], the main bus
	const double        now     = DirectNow();
	const bool          overlay = Graphics::GetSystemOverlayVisualState().active;
	uint64_t            serial  = 0;
	bool                direct  = false;
	Graphics::ImageInfo info;
	bool                hdr = false;
	bool                ask = false;
	{
		Common::LockGuard lock(main.mutex);
		direct = main.opened && main.last_shown_direct;
		serial = main.direct_shown;
		// Only for an overlay: a copy per direct flip would be the work direct flips save.
		if (direct && overlay && m_overlay_asked_serial != serial) {
			m_overlay_asked_serial = serial;
			m_overlay_asked_time   = now;
			const int index        = main.flip_status.currentBuffer;
			if (index >= 0 && index < VIDEO_OUT_BUFFER_NUM_MAX && main.buffers[index].Occupied() &&
			    main.buffers[index].group_index < VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX &&
			    main.groups[main.buffers[index].group_index].occupied) {
				const auto& group = main.groups[main.buffers[index].group_index];
				info              = group.ImageInfo(main.buffers[index]);
				hdr               = Graphics::IsHdrVideoOutPixelFormat(group.attribute.pixel_format);
				ask               = true;
			}
		}
	}
	// It becomes the main layer only while its flip is still the last one shown. Vulkan flips are
	// presented on this thread, so none comes between this check and SetMainLayer; a direct flip that
	// completes meanwhile only makes the next call ask for a newer copy.
	if (copy != nullptr) {
		if (direct && copy_serial == serial) {
			m_presenter.SetMainLayer(*copy);
			m_overlay_ready_serial = serial;
		} else {
			m_presenter.Discard(*copy);
		}
	}
	if (ask) {
		auto&      presenter = m_presenter;
		const bool queued    = m_renderer.GetGpu().SubmitHostTask(
		    [&presenter, info, hdr](Graphics::CommandBuffer& command) -> void* {
			    // As FlipQueue::Prepare: in primary mode Kyty never rendered the buffer.
			    if (Graphics::NativeGpu::ShadowShowsNativeFrames()) {
				    (void)presenter.Renderer().InvalidateMemory(info.data.address, info.data.size,
				                                                Graphics::NativeGpu::PrimaryRendering());
			    }
			    return &presenter.PrepareFrame(command, info, hdr, false);
		    },
		    [&presenter, slot = m_overlay_copy, serial](void* frame) {
			    Common::LockGuard lock(slot->mutex);
			    if (slot->frame != nullptr) {
				    presenter.Discard(*slot->frame);
			    }
			    slot->frame  = static_cast<Graphics::Presenter::Frame*>(frame);
			    slot->serial = serial;
		    });
		if (!queued) {
			m_overlay_asked_time = now - 1.0;
		}
	}
	if (!direct) {
		return true;
	}
	// The window shows the direct flip itself; a present is needed only to draw the overlay, or to
	// take it away again after it was drawn over this flip (over its copy, made then).
	if (!overlay && m_overlay_shown_serial != serial) {
		return false;
	}
	const bool ready = m_overlay_ready_serial == serial || now - m_overlay_asked_time > 0.2;
	if (ready) {
		m_overlay_shown_serial = serial;
	}
	return ready;
}

bool FlipQueue::Reserve(VideoOutConfig& cfg, int index, int flip_mode, int64_t flip_arg, FlipRequestSource source,
                        uint64_t& request_id) {
	Common::LockGuard lock(m_mutex);

	// A newer slave request replaces the one not yet captured by a master flip.
	if (cfg.master != nullptr) {
		for (auto* queue: {&m_requests, &m_cpu_requests}) {
			for (auto it = queue->begin(); it != queue->end();) {
				if (it->cfg != &cfg || it->group != 0) {
					++it;
					continue;
				}
				cfg.flip_status.flipPendingNum--;
				cfg.flip_status.gcQueueNum -= it->source == FlipRequestSource::GpuEop;
				if (it->state == RequestState::Ready) {
					m_presenter.Discard(*it->frame);
					it = queue->erase(it);
				} else {
					auto cancelled = it++;
					m_cancelled_requests.splice(m_cancelled_requests.end(), *queue, cancelled);
				}
			}
		}
		SignalDoneLocked();
	}
	if (cfg.flip_status.flipPendingNum >= VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::FlipsRejected);
		return false;
	}
	auto& pending = source == FlipRequestSource::GpuEop ? m_requests : m_cpu_requests;

	Request r {};
	r.id         = m_next_request_id++;
	r.group      = cfg.master == nullptr ? r.id : 0;
	r.cfg        = &cfg;
	r.master     = cfg.master;
	r.generation = cfg.generation;
	r.index      = index;
	r.flip_arg   = flip_arg;
	r.submit_ptc = LibKernel::KernelGetProcessTimeCounter();
	r.source     = source;
	r.state      = RequestState::Reserved;
	r.flip_mode  = flip_mode;
	if (cfg.slave_count != 0) {
		for (auto* queue: {&m_requests, &m_cpu_requests}) {
			for (auto& slave: *queue) {
				if (slave.master == &cfg && slave.group == 0) {
					slave.group = r.id;
				}
			}
		}
	}

	pending.push_back(r);
	request_id = r.id;
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": reserved buffer %d from the %s (pending %d, current %d)\n", FlipLogMs(), r.id, index,
		     source == FlipRequestSource::GpuEop ? "GPU" : "CPU", cfg.flip_status.flipPendingNum + 1,
		     cfg.flip_status.currentBuffer);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::FlipsRequested);

	cfg.flip_status.flipPendingNum++;
	cfg.flip_status.submitProcessTimeCounter = r.submit_ptc;
	if (source == FlipRequestSource::GpuEop) {
		cfg.flip_status.gcQueueNum++;
	}

	return true;
}

FlipQueue::~FlipQueue() {
	for (auto* queue: {&m_requests, &m_cpu_requests, &m_cancelled_requests}) {
		for (auto& request: *queue) {
			if (request.frame != nullptr) {
				m_presenter.Discard(*request.frame);
			}
		}
	}
}

void FlipQueue::Cancel(VideoOutConfig& cfg) {
	std::vector<Graphics::Presenter::Frame*> frames;
	m_mutex.Lock();
	// A direct request at the display is not waited for: its CompleteNotify may never come. It is kept
	// with the cancelled ones until it does (or the watchdog drops it); its IdleNotify still ends the
	// display's hold through the token.
	while (std::any_of(m_requests.begin(), m_requests.end(), [&cfg](const auto& r) {
		return r.cfg == &cfg && r.state == RequestState::Presenting && !r.direct;
	})) {
		m_done_cond_var.Wait(&m_mutex);
	}
	for (auto* queue: {&m_requests, &m_cpu_requests}) {
		for (auto it = queue->begin(); it != queue->end();) {
			if (it->cfg != &cfg) {
				++it;
				continue;
			}
			if (it->state == RequestState::Reserved || it->state == RequestState::Recording ||
			    (it->direct && it->state == RequestState::Presenting)) {
				auto cancelled = it++;
				m_cancelled_requests.splice(m_cancelled_requests.end(), *queue, cancelled);
				continue;
			}
			if (it->frame != nullptr) {
				frames.push_back(it->frame);
			}
			it = queue->erase(it);
		}
	}
	m_side_shown.erase(&cfg);
	SignalDoneLocked();
	m_submit_slot_cond_var.SignalAll();
	m_mutex.Unlock();
	for (auto* frame: frames) {
		m_presenter.Discard(*frame);
	}
	Common::LockGuard lock(cfg.mutex);
	cfg.flip_status.flipPendingNum = 0;
	cfg.flip_status.gcQueueNum     = 0;
	cfg.direct_last_target         = 0;
	cfg.last_shown_direct          = false;
}

void FlipQueue::Prepare(uint64_t request_id, Graphics::CommandBuffer& buffer) {
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": prepare\n", FlipLogMs(), request_id);
	}
	VideoOutConfig* cfg        = nullptr;
	uint64_t        generation = 0;
	int             index      = 0;
	{
		Common::LockGuard lock(m_mutex);
		// A GPU request is already in m_requests: reserved just before by SubmitFlipFromGpu, or by
		// SubmitEopFlip when the command buffer was written and taken by its interrupt (TakeEopFlip).
		// A CPU request moves there now.
		auto request = std::find_if(m_requests.begin(), m_requests.end(),
		                            [request_id](const auto& r) { return r.id == request_id; });
		if (request == m_requests.end()) {
			auto pending = std::find_if(m_cpu_requests.begin(), m_cpu_requests.end(),
			                            [request_id](const auto& r) { return r.id == request_id; });
			if (pending == m_cpu_requests.end()) {
				return; // Cancelled before recording; Complete releases it.
			}
			// In request order: an EOP flip reserved when its command buffer was written may already
			// wait there, and a flip submitted before it must not be shown after it.
			const auto position = std::find_if(m_requests.begin(), m_requests.end(),
			                                   [request_id](const auto& r) { return r.id > request_id; });
			request = m_requests.insert(position, *pending);
			m_cpu_requests.erase(pending);
		}
		// Shown directly (TryDirect): nothing to record.
		if (request->direct) {
			return;
		}
		if (request->state != RequestState::Reserved) {
			EXIT("cannot prepare video-out request id=%" PRIu64 "\n", request_id);
		}
		request->state = RequestState::Recording;
		cfg            = request->cfg;
		generation     = request->generation;
		index          = request->index;
	}

	bool                special = IsSpecialBufferIndex(index);
	Graphics::ImageInfo source_info;
	bool                current = false;
	bool                premultiplied_alpha = false;
	bool                hdr                 = false;
	bool                headset             = false;
	{
		Common::LockGuard lock(cfg->mutex);
		current = cfg->generation == generation;
		headset = cfg->bus == VIDEO_OUT_BUS_TYPE_HMD2_MAIN;
		special = special || (headset && !cfg->buffers[index].Occupied());
		if (current && !special) {
			const auto& surface = cfg->buffers[index];
			if (!surface.Occupied()) {
				EXIT("cannot prepare flip from an unregistered surface, id=%" PRIu64
				     " index=%d\n",
				     request_id, index);
			}
			const auto& group   = cfg->groups[surface.group_index];
			source_info         = group.ImageInfo(surface);
			hdr                 = Graphics::IsHdrVideoOutPixelFormat(group.attribute.pixel_format);
			premultiplied_alpha = (group.attribute.option &
			                       VIDEO_OUT_BUFFER_ATTRIBUTE_OPTION_ALPHA_PREMULTIPLIED) != 0;
		}
	}
	if (!current) {
		Common::LockGuard lock(m_mutex);
		const auto        request =
		    std::find_if(m_requests.begin(), m_requests.end(),
		                 [request_id](const auto& r) { return r.id == request_id; });
		if (request != m_requests.end()) {
			m_cancelled_requests.splice(m_cancelled_requests.end(), m_requests, request);
		}
		return;
	}
	Graphics::Presenter::Frame* frame = nullptr;
	if (special) {
		frame = &m_presenter.PrepareBlankFrame(m_width, m_height,
		                                       index == VIDEO_OUT_BUFFER_INDEX_BLACK, &buffer);
	} else {
		// BC-250 native GPU shadow path: dump and optionally show the natively rendered surface,
		// which the shadow submit already left in guest memory. No-op unless shadow mode is on.
		Graphics::NativeGpu::ShadowOnFlip(source_info.data.address, source_info.data.size,
		                                  source_info.extent.width, source_info.extent.height,
		                                  source_info.bytes_per_block,
		                                  static_cast<uint32_t>(source_info.pixel_format),
		                                  static_cast<uint32_t>(source_info.tile_mode),
		                                  [&](uint64_t addr, uint64_t bytes) {
			                                  return m_presenter.Renderer().DownloadMemoryForDebug(addr, bytes);
		                                  });
		if (Graphics::NativeGpu::ShadowShowsNativeFrames()) {
			// In primary mode Kyty never rendered this surface, so its cached copies are stale and
			// must not be written back over the native result.
			(void)m_presenter.Renderer().InvalidateMemory(source_info.data.address, source_info.data.size,
			                                              Graphics::NativeGpu::PrimaryRendering());
		}
		const auto prepare_t0 = std::chrono::steady_clock::now();
		frame = &m_presenter.PrepareFrame(buffer, source_info, hdr, headset);
		Graphics::NativeGpu::ShadowFlipStats(
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - prepare_t0).count());
		if (headset) {
			g_headset_mirror = true;
		}
	}

	Common::LockGuard lock(m_mutex);
	auto prepared = std::find_if(m_requests.begin(), m_requests.end(),
	                             [request_id](const auto& r) { return r.id == request_id; });
	if (prepared == m_requests.end()) {
		prepared = std::find_if(m_cancelled_requests.begin(), m_cancelled_requests.end(),
		                        [request_id](const auto& r) { return r.id == request_id; });
	}
	prepared->frame               = frame;
	prepared->premultiplied_alpha = premultiplied_alpha;
}

void FlipQueue::Complete(uint64_t request_id) {
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": recording complete\n", FlipLogMs(), request_id);
	}
	m_mutex.Lock();
	// SubmitFlipFromGpu's request went to the display directly; it may be gone already.
	if (m_direct_noop_completes.erase(request_id) != 0) {
		m_mutex.Unlock();
		return;
	}
	auto request = std::find_if(m_requests.begin(), m_requests.end(),
	                            [request_id](const auto& r) { return r.id == request_id; });
	if (request != m_requests.end()) {
		request->state = RequestState::Ready;
		m_mutex.Unlock();
		return;
	}
	auto  cancelled = std::find_if(m_cancelled_requests.begin(), m_cancelled_requests.end(),
	                               [request_id](const auto& r) { return r.id == request_id; });
	auto* frame     = cancelled->frame;
	m_cancelled_requests.erase(cancelled);
	m_done_cond_var.SignalAll();
	m_mutex.Unlock();
	if (frame != nullptr) {
		m_presenter.Discard(*frame);
	}
}

void FlipQueue::WaitForSubmitSlot(VideoOutConfig& cfg, bool on_gpu_thread) {
	Common::LockGuard lock(m_mutex);
	auto belongs_to_port = [&cfg](const auto& request) { return request.cfg == &cfg; };
	while (std::count_if(m_requests.begin(), m_requests.end(), belongs_to_port) +
	           std::count_if(m_cpu_requests.begin(), m_cpu_requests.end(), belongs_to_port) >=
	       VIDEO_OUT_FLIP_QUEUE_CAPACITY) {
		// Another thread (the title's, recording a SetFlip) waits while the GPU thread prepares and
		// the video out shows the CPU flips.
		if (on_gpu_thread && m_requests.empty()) {
			EXIT("video-out queue is saturated by CPU flips queued behind the current EOP\n");
		}
		m_submit_slot_cond_var.Wait(&m_mutex);
	}
}

void FlipQueue::SignalDoneLocked() {
	m_done_cond_var.SignalAll();
	// A stalled GPU cannot drain and shut down without re-checking under m_mutex, so it is alive here.
	if (m_gpu_waiting) {
		m_gpu_waiting = false;
		m_presenter.Renderer().GetGpu().Wake();
	}
}

// A display buffer is safe for rendering once it is neither displayed nor waiting to be. Only flips
// requested before `before` count: on the PS5 the GPU checks this before the frame's end-of-pipe
// event lets the title request the next flip, so a later request of the same buffer (possible here,
// where the check runs after the frame's GPU work, or behind it) cannot hold the wait forever. That
// holds for the buffer on screen too: when a later flip has already shown it (the check ran late,
// after the frame's own flip), the PS5's GPU passed the wait before that flip existed. Holding it
// until the next vblank instead made each frame miss one, and My First GT stayed at 30 fps.
//
// An EOP flip reserved when its command buffer was written (SubmitEopFlip, libSceAgc's SetFlip)
// counts by the GPU order of its release instead: on the PS5 the wait reads the buffer's label at its
// own place in the GPU's work, where only flips whose release ran before it have written 1. So such a
// flip holds the buffer when its release ran by the wait's `eop_mark` (EopFlipOrder() when the
// command processor reached the wait), and never while its release has not run.
bool FlipQueue::HoldsBuffer(const Request& r, uint64_t before, uint64_t eop_mark) {
	if (r.source == FlipRequestSource::GpuEop && r.eop_order != 0) {
		return r.eop_order <= eop_mark;
	}
	if (r.source == FlipRequestSource::GpuEop && r.state == RequestState::Reserved) {
		return false;
	}
	// A CPU flip, or a GPU flip reserved at its end of pipe and prepared at once (the R_FLIP path).
	return r.id < before;
}

bool FlipQueue::IsSafeForRendering(VideoOutConfig& cfg, int index, uint64_t before, uint64_t eop_mark) {
	Common::LockGuard lock(m_mutex);
	auto matches = [&cfg, index, before, eop_mark](const auto& r) {
		return r.cfg == &cfg && r.index == index && HoldsBuffer(r, before, eop_mark);
	};
	const bool shown_holds =
	    cfg.flip_status.currentBuffer == index &&
	    (cfg.current_flip_source == FlipRequestSource::GpuEop && cfg.current_flip_eop_order != 0
	         ? cfg.current_flip_eop_order <= eop_mark
	         : cfg.current_flip_id < before);
	// A directly shown buffer stays held after it left the screen, until the display's IdleNotify
	// (the PS5's display engine reading it until the next vblank). Counted like the buffer on screen:
	// a wait placed before the release of the flip that showed it is not held by it.
	const auto& hold          = cfg.holds[index >= 0 && index < VIDEO_OUT_BUFFER_NUM_MAX ? index : 0];
	const bool  display_holds = index >= 0 && index < VIDEO_OUT_BUFFER_NUM_MAX && hold.presents != 0 &&
	                           (hold.source == FlipRequestSource::GpuEop && hold.eop_order != 0
	                                ? hold.eop_order <= eop_mark
	                                : hold.flip_id < before);
	const bool safe = !shown_holds && !display_holds && std::none_of(m_requests.begin(), m_requests.end(), matches) &&
	                  std::none_of(m_cpu_requests.begin(), m_cpu_requests.end(), matches);
	m_gpu_waiting = !safe;
	if (LogFlips()) {
		// The first check that holds a buffer, and the one that lets it go.
		static int      last_index    = -1;
		static uint64_t last_before   = 0;
		static uint64_t last_eop_mark = 0;
		static bool     last_safe     = true;
		if (safe != last_safe || index != last_index || before != last_before || eop_mark != last_eop_mark) {
			last_index    = index;
			last_before   = before;
			last_eop_mark = eop_mark;
			last_safe     = safe;
			std::fprintf(stderr,
			             "flip [%.2f]: buffer %d %s for rendering (current %d from flip %" PRIu64 " eop %" PRIu64
			             ", flips before %" PRIu64 ", releases up to %" PRIu64 ", display holds %u from flip %" PRIu64 ")\n",
			             FlipLogMs(), index, safe ? "safe" : "not safe", cfg.flip_status.currentBuffer, cfg.current_flip_id,
			             cfg.current_flip_eop_order, before, eop_mark, display_holds ? hold.presents : 0u,
			             display_holds ? hold.flip_id : 0);
			for (const auto* queue: {&m_requests, &m_cpu_requests}) {
				for (const auto& r: *queue) {
					if (safe) {
						break;
					}
					std::fprintf(stderr, "\t %s request %" PRIu64 ": buffer %d, state %d, eop %" PRIu64 "\n",
					             queue == &m_requests ? "GPU-side" : "CPU", r.id, r.index, static_cast<int>(r.state), r.eop_order);
				}
			}
		}
	}
	return safe;
}

// A flip whose preparation was refused (the GPU thread is gone) no longer holds its buffer.
void FlipQueue::DropEopOrder(uint64_t request_id) {
	Common::LockGuard lock(m_mutex);
	for (auto* queue: {&m_requests, &m_cancelled_requests}) {
		for (auto& r: *queue) {
			if (r.id == request_id && r.state == RequestState::Reserved) {
				r.eop_order = 0;
			}
		}
	}
}

bool FlipQueue::TakeEopFlip(uint32_t ctx, uint64_t& request_id) {
	if ((ctx & 0x0fffff00u) != VIDEO_OUT_EOP_FLIP_CONTEXT) {
		return false;
	}
	Common::LockGuard lock(m_mutex);
	// A port closed (or a slave request replaced) before the interrupt left the request cancelled;
	// it is still prepared, which retires it.
	for (auto* queue: {&m_requests, &m_cancelled_requests}) {
		const auto request = std::find_if(queue->begin(), queue->end(), [ctx](const auto& r) {
			return r.source == FlipRequestSource::GpuEop && r.state == RequestState::Reserved &&
			       r.eop_order == 0 && EopFlipContext(r.id) == (ctx & 0x0fffffffu);
		});
		if (request != queue->end()) {
			request->eop_order = ++m_eop_order;
			request_id         = request->id;
			// The release wrote 1 into the label before its interrupt; a Flip that took the buffer off the screen in
			// between saw no EOP order yet and cleared it. With the order given, Flip leaves the label alone.
			if (queue == &m_requests && request->index >= 0 && request->index < VIDEO_OUT_BUFFER_NUM_MAX) {
				std::atomic_ref<uint64_t>(request->cfg->labels[request->index]).store(1, std::memory_order_release);
			}
			if (LogFlips()) {
				std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": end-of-pipe interrupt 0x%08" PRIx32 "\n", FlipLogMs(),
				             request_id, ctx);
			}
			return true;
		}
	}
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f]: end-of-pipe interrupt 0x%08" PRIx32 " matches no reserved flip\n", FlipLogMs(), ctx);
	}
	return false;
}

bool FlipQueue::OlderPendingOnPort(const Request& r) const {
	auto older = [&r](const Request& o) { return o.cfg == r.cfg && o.id < r.id && o.group != r.group; };
	return std::any_of(m_requests.begin(), m_requests.end(), older) ||
	       std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), older);
}

bool FlipQueue::Flip() {
	KYTY_PROFILER_BLOCK("FlipQueue::Flip");

	m_mutex.Lock();
	// The first group that can be shown: every request of it is ready (none still on the CPU side),
	// and no older flip of any of its ports is pending outside it, so each port shows its flips in
	// request order. A group that is not ready holds back only its own ports: an EOP flip reserved
	// when its command buffer was written (SubmitEopFlip) stays Reserved for the frame's whole GPU
	// time, and the other ports' flips do not wait for it.
	// A direct request is not Kyty's to present (TryDirect sent it or will), but it still holds back
	// the newer flips of its port until the display completed it (OlderPendingOnPort).
	uint64_t group = 0;
	for (const auto& owner: m_requests) {
		if (owner.group != owner.id || owner.direct) {
			continue;
		}
		const uint64_t candidate = owner.id;
		if (std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(),
		                [candidate](const auto& r) { return r.group == candidate; }) ||
		    std::any_of(m_requests.begin(), m_requests.end(), [this, candidate](const auto& r) {
			    return r.group == candidate && (r.state != RequestState::Ready || OlderPendingOnPort(r));
		    })) {
			continue;
		}
		group = candidate;
		break;
	}
	if (group == 0) {
		m_mutex.Unlock();
		return false;
	}
	std::array<Request, VideoOutDriver::Impl::VIDEO_OUT_BUSES.size()> requests {};
	size_t count = 0;
	for (auto& request: m_requests) {
		if (request.group == group) {
			EXIT_IF(count == requests.size());
			request.state = RequestState::Presenting;
			requests[count++] = request;
		}
	}
	m_mutex.Unlock();

	// Present and complete the whole group at one Vblank, taking the port locks in bus order. The
	// present runs without them: Cancel waits for Presenting requests, so their frames and ports
	// stay valid.
	std::sort(requests.begin(), requests.begin() + count,
	          [](const auto& a, const auto& b) { return a.cfg->bus < b.cfg->bus; });
	bool due = true;
	std::array<Graphics::Presenter::Layer, VideoOutDriver::Impl::VIDEO_OUT_BUSES.size()> layers {};
	size_t                      layer_count = 0;
	Graphics::Presenter::Frame* hidden      = nullptr; // The main port behind the headset image.
	for (size_t i = 0; i < count; i++) {
		auto&             r = requests[i];
		Common::LockGuard lock(r.cfg->mutex);
		due &= r.cfg->generation == r.generation;
		if (r.id == group) {
			due &= IsFlipDueLocked(*r.cfg);
		}
		if (r.cfg->bus == VIDEO_OUT_BUS_TYPE_MAIN &&
		    g_headset_mirror.load(std::memory_order_relaxed)) {
			hidden = r.frame;
		} else {
			const bool headset    = r.cfg->bus == VIDEO_OUT_BUS_TYPE_HMD2_MAIN;
			layers[layer_count++] = {r.frame, headset ? VIDEO_OUT_BUS_TYPE_MAIN : r.cfg->bus,
			                         r.premultiplied_alpha};
		}
	}
	if (due) {
		if (hidden != nullptr) {
			m_presenter.Discard(*hidden);
		}
		if (layer_count != 0) {
			m_presenter.Present(std::span(layers.data(), layer_count));
		}
		if (requests[0].cfg->bus == 0) {
			Graphics::PerformanceOverlayRecordFlip(m_width, m_height);
		}
	}

	for (size_t i = 0; i < count; i++) {
		requests[i].cfg->mutex.Lock();
	}
	m_mutex.Lock();
	for (auto it = m_requests.begin(); it != m_requests.end();) {
		if (it->group != group) {
			++it;
			continue;
		}
		if (!due) {
			it++->state = RequestState::Ready;
			continue;
		}
		auto&     r        = *it;
		const int previous = r.cfg->flip_status.currentBuffer;
		r.cfg->flip_vblank = r.cfg->vblank_status.count;
		r.cfg->flip_status.count++;
		r.cfg->flip_status.processTime              = LibKernel::KernelGetProcessTime();
		r.cfg->flip_status.processTimeCounter       = LibKernel::KernelGetProcessTimeCounter();
		r.cfg->flip_status.submitProcessTimeCounter = r.submit_ptc;
		r.cfg->flip_status.flipArg                  = r.flip_arg;
		r.cfg->flip_status.currentBuffer            = r.index;
		r.cfg->current_flip_id                      = r.id;
		r.cfg->current_flip_source                  = r.source;
		r.cfg->current_flip_eop_order               = r.eop_order;
		r.cfg->last_shown_direct                    = false;
		if (r.cfg->bus != VIDEO_OUT_BUS_TYPE_MAIN) {
			m_side_shown.insert(r.cfg);
		}
		// The buffer this flip takes off the screen is safe for rendering again (MaybeClearLabelLocked);
		// one shown directly before stays held until the display's IdleNotify.
		if (previous >= 0 && previous < VIDEO_OUT_BUFFER_NUM_MAX && previous != r.index) {
			if (r.cfg->holds[previous].presents != 0) {
				r.cfg->holds[previous].since = DirectNow();
			}
			MaybeClearLabelLocked(*r.cfg, previous, r.id);
		}
		if (LogFlips()) {
			std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": buffer %d displayed\n", FlipLogMs(), r.id, r.index);
		}
		r.cfg->flip_status.flipPendingNum--;
		r.cfg->flip_status.gcQueueNum -= r.source == FlipRequestSource::GpuEop;
		TriggerVideoOutEvents(*r.cfg, VideoOutEventKind::Flip, reinterpret_cast<void*>(r.flip_arg));
		it = m_requests.erase(it);
	}
	SignalDoneLocked();
	m_submit_slot_cond_var.SignalAll();
	m_mutex.Unlock();
	for (size_t i = count; i != 0; i--) {
		requests[i - 1].cfg->mutex.Unlock();
	}
	// Direct requests that waited behind this one go to the display now.
	if (due && m_direct != nullptr) {
		PumpDirect();
	}
	if (due) {
		Graphics::RenderDocOnGuestFlip(m_presenter.Renderer());
		if (Config::GraphicsDebugDumpEnabled() &&
		    Config::GetPrintfDirection() != Config::LogDirection::Silent) {
			LOGF("Flip done: %d\n", requests[0].index);
		}
	}
	return due;
}

bool FlipQueue::OlderUnsentOnPort(const Request& r) const {
	auto older = [&r](const Request& o) {
		return o.cfg == r.cfg && o.id < r.id && (!o.direct || o.state == RequestState::Ready);
	};
	return std::any_of(m_requests.begin(), m_requests.end(), older) ||
	       std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), older);
}

void FlipQueue::MaybeClearLabelLocked(VideoOutConfig& cfg, int index, uint64_t except_id) {
	if (index < 0 || index >= VIDEO_OUT_BUFFER_NUM_MAX || index == cfg.flip_status.currentBuffer ||
	    cfg.holds[index].presents != 0) {
		return;
	}
	auto shows = [&cfg, index, except_id](const Request& o) {
		return o.cfg == &cfg && o.index == index && o.id != except_id &&
		       !(o.source == FlipRequestSource::GpuEop && o.state == RequestState::Reserved && o.eop_order == 0);
	};
	if (std::any_of(m_requests.begin(), m_requests.end(), shows) ||
	    std::any_of(m_cpu_requests.begin(), m_cpu_requests.end(), shows)) {
		return;
	}
	std::atomic_ref<uint64_t>(cfg.labels[index]).store(0, std::memory_order_release);
}

// Whether the system overlay or a debug path needs Kyty's presenter for this flip. On the PS5 the
// system software draws its UI on a display plane of its own; here Kyty's presenter draws it into
// the frame, so while it is visible the frame is a Vulkan copy.
static bool DirectBlockedByHost(Graphics::Presenter& presenter) {
	// DirectPresentWanted() is false too while a flip dump is wanted (KYTY_NATIVE_GPU_DUMP_FLIPS, a
	// pending "dump N" on the control socket): dumps run in ShadowOnFlip, at the Vulkan path's preparation.
	return !Graphics::NativeGpu::DirectPresentWanted() || Graphics::GetSystemOverlayVisualState().active ||
	       Graphics::PerformanceOverlayEnabled() || presenter.HdrSwapchainEverUsed();
}

bool FlipQueue::TryDirect(uint64_t request_id, bool complete_follows) {
	if (m_direct == nullptr || DirectBlockedByHost(m_presenter)) {
		return false;
	}
	VideoOutConfig* cfg = nullptr;
	{
		Common::LockGuard lock(m_mutex);
		if (m_direct_off || m_direct_failed) {
			return false;
		}
		for (const auto* queue: {&m_requests, &m_cpu_requests}) {
			for (const auto& r: *queue) {
				if (r.id == request_id) {
					cfg = r.cfg;
				}
			}
		}
		if (cfg == nullptr) {
			return false; // cancelled: the Vulkan path retires it
		}
	}
	Common::LockGuard cfg_lock(cfg->mutex);
	Common::LockGuard lock(m_mutex);
	std::list<Request>* queue = &m_requests;
	auto it = std::find_if(m_requests.begin(), m_requests.end(), [request_id](const auto& r) { return r.id == request_id; });
	if (it == m_requests.end()) {
		queue = &m_cpu_requests;
		it    = std::find_if(m_cpu_requests.begin(), m_cpu_requests.end(),
		                     [request_id](const auto& r) { return r.id == request_id; });
		if (it == m_cpu_requests.end()) {
			return false;
		}
	}
	const auto& r = *it;
	if (r.state != RequestState::Reserved || r.cfg != cfg || m_direct_off || m_direct_failed || !m_direct->Usable() ||
	    !cfg->opened || cfg->generation != r.generation || r.index < 0 ||
	    r.index >= VIDEO_OUT_BUFFER_NUM_MAX) {
		return false;
	}
	const auto& slot = cfg->direct[r.index];
	// Shown 1:1 on the window, alone, as the title wrote it: a registered direct buffer of the main
	// port, no flip master or slave, no headset image or other port composed over it, no colour change.
	if (slot.pixmap == 0 || !cfg->buffers[r.index].Occupied() || cfg->buffers[r.index].data_address != slot.address ||
	    cfg->bus != VIDEO_OUT_BUS_TYPE_MAIN || cfg->master != nullptr || cfg->slave_count != 0 || r.group != r.id ||
	    g_headset_mirror.load(std::memory_order_relaxed) || !m_side_shown.empty() || cfg->gamma != 1.0f ||
	    (cfg->output_mode != VIDEO_OUT_OUTPUT_MODE_DEFAULT && cfg->output_mode != VIDEO_OUT_OUTPUT_MODE_119_88HZ) ||
	    !m_direct->WindowIs(slot.width, slot.height)) {
		return false;
	}
	if (queue == &m_cpu_requests) {
		// In request order, as Prepare moves it.
		const auto position = std::find_if(m_requests.begin(), m_requests.end(),
		                                   [request_id](const auto& o) { return o.id > request_id; });
		it = m_requests.insert(position, *it);
		m_cpu_requests.erase(std::find_if(m_cpu_requests.begin(), m_cpu_requests.end(),
		                                  [request_id](const auto& o) { return o.id == request_id; }));
	}
	it->state  = RequestState::Ready;
	it->direct = true;
	it->frame  = nullptr;
	it->pixmap = slot.pixmap;
	if (complete_follows) {
		m_direct_noop_completes.insert(request_id);
	}
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": buffer %d goes to the display directly (mode %d)\n", FlipLogMs(),
		             request_id, it->index, it->flip_mode);
	}
	PumpDirectLocked(*cfg, DirectNow());
	return true;
}

void FlipQueue::PumpDirectLocked(VideoOutConfig& cfg, double now) {
	for (auto it = m_requests.begin(); it != m_requests.end();) {
		auto& r = *it;
		if (r.cfg != &cfg || !r.direct || r.state != RequestState::Ready) {
			++it;
			continue;
		}
		// The port is closing: Cancel retires its requests.
		if (!cfg.opened || cfg.generation != r.generation || OlderUnsentOnPort(r)) {
			break;
		}
		if (cfg.direct[r.index].pixmap != r.pixmap) {
			// The buffer was unregistered or its attribute changed after this flip was requested: its
			// pixmap is gone. The flip counts as shown and skipped.
			std::fprintf(stderr, "direct present: flip %" PRIu64 " of buffer %d skipped, its pixmap is gone\n", r.id,
			             r.index);
			it = CompleteDirectLocked(it, now);
			continue;
		}
		r.state           = RequestState::Presenting;
		r.direct_sent     = now;
		r.direct_deadline = now;
		if (m_direct_failed) {
			// Not sent; the watchdog completes it as skipped on its next tick.
			++it;
			continue;
		}
		// FIFO by target MSC: each flip stays on screen for at least `interval` vblanks (the flip
		// rate), the next free vblank when nothing is queued; an immediate (HSYNC) flip goes at once.
		const uint64_t interval = static_cast<uint64_t>(cfg.flip_rate) + 1;
		const bool     async    = r.flip_mode == VIDEO_OUT_FLIP_MODE_HSYNC;
		const uint64_t last_msc = m_direct->LastMsc();
		uint64_t       target   = 0;
		if (!async && cfg.direct_last_target != 0) {
			target = std::max(cfg.direct_last_target + interval, last_msc + 1);
		}
		if (!m_direct->Present(r.pixmap, r.id, target, async)) {
			std::fprintf(stderr, "direct present: sending flip %" PRIu64 " failed; flips take the Vulkan path from now on\n",
			             r.id);
			m_direct_off    = true;
			m_direct_failed = true;
			++it;
			continue;
		}
		cfg.direct_last_target = target != 0 ? target : last_msc + 1;
		// 250 ms after the vblank it is due at (counted at 30 Hz, so a 120 Hz output is not shorter).
		r.direct_deadline = now + 0.25 + static_cast<double>(target > last_msc ? target - last_msc : 1) / 30.0;
		auto& hold        = cfg.holds[r.index];
		hold.presents++;
		hold.flip_id   = r.id;
		hold.source    = r.source;
		hold.eop_order = r.eop_order;
		hold.since     = now;
		m_direct_tokens[r.id] = {&cfg, r.index};
		if (LogFlips()) {
			std::fprintf(stderr,
			             "flip [%.2f] %" PRIu64 ": buffer %d sent to the display (target msc %" PRIu64 "%s, last %" PRIu64
			             ", held by %u presents)\n",
			             FlipLogMs(), r.id, r.index, target, async ? ", async" : "", last_msc, hold.presents);
		}
		++it;
	}
}

void FlipQueue::PumpDirect() {
	VideoOutConfig* cfg = nullptr;
	{
		Common::LockGuard lock(m_mutex);
		const auto        ready = std::find_if(m_requests.begin(), m_requests.end(), [](const auto& r) {
            return r.direct && r.state == RequestState::Ready;
        });
		if (ready == m_requests.end()) {
			return;
		}
		cfg = ready->cfg;
	}
	Common::LockGuard cfg_lock(cfg->mutex);
	Common::LockGuard lock(m_mutex);
	PumpDirectLocked(*cfg, DirectNow());
}

std::list<FlipQueue::Request>::iterator FlipQueue::CompleteDirectLocked(std::list<Request>::iterator it, double now) {
	auto&     r        = *it;
	auto&     cfg      = *r.cfg;
	const int previous = cfg.flip_status.currentBuffer;
	cfg.flip_vblank    = cfg.vblank_status.count;
	cfg.flip_status.count++;
	cfg.flip_status.processTime              = LibKernel::KernelGetProcessTime();
	cfg.flip_status.processTimeCounter       = LibKernel::KernelGetProcessTimeCounter();
	cfg.flip_status.submitProcessTimeCounter = r.submit_ptc;
	cfg.flip_status.flipArg                  = r.flip_arg;
	cfg.flip_status.currentBuffer            = r.index;
	cfg.current_flip_id                      = r.id;
	cfg.current_flip_source                  = r.source;
	cfg.current_flip_eop_order               = r.eop_order;
	cfg.last_shown_direct                    = true;
	cfg.direct_shown++;
	cfg.flip_status.flipPendingNum--;
	cfg.flip_status.gcQueueNum -= r.source == FlipRequestSource::GpuEop;
	// The buffer it took off the screen: one shown by Kyty's presenter is free now; one shown directly
	// is still the display's until its IdleNotify.
	if (previous >= 0 && previous < VIDEO_OUT_BUFFER_NUM_MAX && previous != r.index) {
		if (cfg.holds[previous].presents != 0) {
			cfg.holds[previous].since = now;
		}
		MaybeClearLabelLocked(cfg, previous, r.id);
	}
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": buffer %d displayed directly\n", FlipLogMs(), r.id, r.index);
	}
	TriggerVideoOutEvents(cfg, VideoOutEventKind::Flip, reinterpret_cast<void*>(r.flip_arg));
	auto next = m_requests.erase(it);
	SignalDoneLocked();
	m_submit_slot_cond_var.SignalAll();
	return next;
}

void FlipQueue::OnDirectComplete(uint64_t token, uint64_t msc, uint64_t ust_us, Graphics::DirectCompleteMode mode) {
	(void)ust_us; // Stage 2: the Flip event's and the vblank's times from the display's.
	VideoOutConfig* cfg = nullptr;
	{
		Common::LockGuard lock(m_mutex);
		// Its port was closed meanwhile.
		if (const auto cancelled = std::find_if(m_cancelled_requests.begin(), m_cancelled_requests.end(),
		                                        [token](const auto& r) { return r.id == token && r.direct; });
		    cancelled != m_cancelled_requests.end()) {
			m_cancelled_requests.erase(cancelled);
			return;
		}
		const auto t = m_direct_tokens.find(token);
		if (t == m_direct_tokens.end()) {
			return;
		}
		cfg = t->second.cfg;
	}
	bool     disable = false;
	bool     main    = false;
	uint32_t width   = 0;
	uint32_t height  = 0;
	{
		Common::LockGuard cfg_lock(cfg->mutex);
		Common::LockGuard lock(m_mutex);
		const auto        it = std::find_if(m_requests.begin(), m_requests.end(), [token](const auto& r) {
            return r.id == token && r.direct && r.state == RequestState::Presenting;
        });
		if (it == m_requests.end()) {
			return; // the watchdog completed it already
		}
		const double now = DirectNow();
		if (LogFlips()) {
			std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": the display completed it (mode %d, msc %" PRIu64 ", %.2f ms after sending)\n",
			             FlipLogMs(), token, static_cast<int>(mode), msc, (now - it->direct_sent) * 1e3);
		}
		CompleteDirectLocked(it, now);
		switch (mode) {
			case Graphics::DirectCompleteMode::Flip: m_copy_streak = 0; break;
			case Graphics::DirectCompleteMode::Copy:
				// Xwayland copies the pixmap into the window instead of handing it to gamescope: a blit on
				// the shared gfx pipe, more work than Kyty's own path.
				if (++m_copy_streak >= 3 && !m_direct_off) {
					m_direct_off = true;
					disable      = true;
				}
				break;
			case Graphics::DirectCompleteMode::Skip:
			case Graphics::DirectCompleteMode::Lost:
				// A lost present (the connection died, or the X server refused it) counts as shown, so
				// the title goes on; the flips after it take the Vulkan path.
				if (mode == Graphics::DirectCompleteMode::Lost) {
					m_direct_off = true;
				}
				if (++m_direct_skips <= 10 || m_direct_skips % 600 == 0) {
					std::fprintf(stderr, "direct present: flip %" PRIu64 " %s (%" PRIu64 " so far)\n", token,
					             mode == Graphics::DirectCompleteMode::Skip ? "skipped by the display" : "lost",
					             m_direct_skips);
				}
				break;
		}
		main   = cfg->bus == VIDEO_OUT_BUS_TYPE_MAIN;
		width  = m_width;
		height = m_height;
		PumpDirectLocked(*cfg, now);
	}
	if (disable) {
		m_direct->Disable("Xwayland copies the buffers instead of flipping them");
	}
	if (main) {
		Graphics::PerformanceOverlayRecordFlip(width, height);
	}
	Graphics::NativeGpu::ShadowOnDirectFlip();
	if (mode == Graphics::DirectCompleteMode::Flip) {
		m_presenter.OnDirectShown();
	}
}

void FlipQueue::OnDirectIdle(uint64_t token) {
	VideoOutConfig* cfg = nullptr;
	{
		Common::LockGuard lock(m_mutex);
		const auto        t = m_direct_tokens.find(token);
		if (t == m_direct_tokens.end()) {
			return;
		}
		cfg = t->second.cfg;
	}
	Common::LockGuard cfg_lock(cfg->mutex);
	Common::LockGuard lock(m_mutex);
	const auto        t = m_direct_tokens.find(token);
	if (t == m_direct_tokens.end()) {
		return; // the watchdog dropped the hold
	}
	const int index = t->second.index;
	m_direct_tokens.erase(t);
	auto& hold = cfg->holds[index];
	if (hold.presents != 0) {
		hold.presents--;
	}
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f] %" PRIu64 ": the display released buffer %d (still held by %u)\n", FlipLogMs(),
		             token, index, hold.presents);
	}
	if (hold.presents == 0) {
		hold = DisplayHold {};
		MaybeClearLabelLocked(*cfg, index, 0);
	}
	// A GPU thread parked in a wait for this buffer (WaitFlipDone) checks again.
	SignalDoneLocked();
}

void FlipQueue::OnDirectLost() {
	Common::LockGuard lock(m_mutex);
	if (!m_direct_failed) {
		std::fprintf(stderr, "direct present: the connection to the display is lost; flips take the Vulkan path\n");
	}
	m_direct_off    = true;
	m_direct_failed = true;
}

void FlipQueue::DirectWatchdog(double now) {
	if (m_direct == nullptr) {
		return;
	}
	// The ports with direct requests or display holds (only the main port has them).
	std::array<VideoOutConfig*, VideoOutDriver::Impl::VIDEO_OUT_NUM_MAX> cfgs {};
	size_t                                                              count = 0;
	auto add = [&cfgs, &count](VideoOutConfig* cfg) {
		if (std::find(cfgs.begin(), cfgs.begin() + count, cfg) == cfgs.begin() + count && count < cfgs.size()) {
			cfgs[count++] = cfg;
		}
	};
	{
		Common::LockGuard lock(m_mutex);
		// Requests of a closed port whose CompleteNotify never came.
		std::erase_if(m_cancelled_requests, [now](const auto& r) {
			return r.direct && r.state == RequestState::Presenting && now > r.direct_deadline;
		});
		for (const auto& r: m_requests) {
			if (r.direct) {
				add(r.cfg);
			}
		}
		for (const auto& [token, held]: m_direct_tokens) {
			add(held.cfg);
		}
	}
	bool disable = false;
	for (size_t i = 0; i < count; i++) {
		auto&             cfg = *cfgs[i];
		Common::LockGuard cfg_lock(cfg.mutex);
		Common::LockGuard lock(m_mutex);
		// A flip the display has not completed 250 ms after its vblank never will be: the display
		// stopped answering. Direct presentation ends for the session and what was sent counts as
		// shown and skipped, so the title, its flip status and its waits go on.
		const bool expired = m_direct_failed || std::any_of(m_requests.begin(), m_requests.end(), [&cfg, now](const auto& r) {
			                     return r.cfg == &cfg && r.direct && r.state == RequestState::Presenting &&
			                            now > r.direct_deadline;
		                     });
		if (expired) {
			if (!m_direct_failed) {
				std::fprintf(stderr, "direct present: the display did not complete a flip 250 ms after its vblank; flips "
				                     "take the Vulkan path from now on\n");
				disable = true;
			}
			m_direct_off    = true;
			m_direct_failed = true;
			for (auto it = m_requests.begin(); it != m_requests.end();) {
				if (it->cfg == &cfg && it->direct && it->state == RequestState::Presenting) {
					it = CompleteDirectLocked(it, now);
				} else {
					++it;
				}
			}
			// The rest, once the requests before them have reached the display.
			PumpDirectLocked(cfg, now);
		}
		// A buffer off the screen whose IdleNotify has not come within 1 s: the display let it go
		// without saying so, or never will.
		for (int index = 0; index < VIDEO_OUT_BUFFER_NUM_MAX; index++) {
			auto& hold = cfg.holds[index];
			if (hold.presents == 0 || index == cfg.flip_status.currentBuffer || now - hold.since <= 1.0) {
				continue;
			}
			std::fprintf(stderr, "direct present: buffer %d had no IdleNotify for %.0f ms after it left the screen; "
			                     "released\n",
			             index, (now - hold.since) * 1e3);
			std::erase_if(m_direct_tokens, [&cfg, index](const auto& entry) {
				return entry.second.cfg == &cfg && entry.second.index == index;
			});
			hold = DisplayHold {};
			MaybeClearLabelLocked(cfg, index, 0);
			SignalDoneLocked();
		}
	}
	if (disable) {
		m_direct->Disable("the display stopped answering");
	}
}

void FlipQueue::GetFlipStatus(VideoOutConfig& cfg, VideoOutFlipStatus& out) {
	Common::LockGuard lock(cfg.mutex);

	out = cfg.flip_status;
}

KYTY_SYSV_ABI int VideoOutSetFlipMaster(int slave_handle, int master_handle) {
	PRINT_NAME();
	return DriverState().SetFlipMaster(slave_handle, master_handle);
}

KYTY_SYSV_ABI void VideoOutAddBufferAttributeOption(VideoOutBufferAttribute2* attribute,
                                                   uint64_t option) {
	PRINT_NAME();
	EXIT_IF(attribute == nullptr);
	attribute->option |= option;
}

KYTY_SYSV_ABI int VideoOutOpen(int user_id, int bus_type, int index, const void* param) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(user_id != 255 && user_id != 0);
	LOGF("\t param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(param));

	return DriverState().Open(bus_type, index);
}

KYTY_SYSV_ABI int VideoOutClose(int handle) {
	PRINT_NAME();

	return DriverState().Close(handle) ? OK : VIDEO_OUT_ERROR_INVALID_HANDLE;
}

KYTY_SYSV_ABI void VideoOutSetBufferAttribute2(VideoOutBufferAttribute2* attribute,
                                               uint64_t pixel_format, uint32_t tiling_mode,
                                               uint32_t width, uint32_t height, uint64_t option,
                                               uint32_t dcc_control,
                                               uint64_t dcc_cb_register_clear_color) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(attribute == nullptr);

	LOGF("\t pixel_format                = %016" PRIx64 "\n"
	     "\t tiling_mode                 = %" PRIu32 "\n"
	     "\t width                       = %" PRIu32 "\n"
	     "\t height                      = %" PRIu32 "\n"
	     "\t option                      = %016" PRIx64 "\n"
	     "\t dcc_control                 = %08" PRIx32 "\n"
	     "\t dcc_cb_register_clear_color = %016" PRIx64 "\n",
	     pixel_format, tiling_mode, width, height, option, dcc_control,
	     dcc_cb_register_clear_color);

	memset(attribute, 0, sizeof(VideoOutBufferAttribute2));

	attribute->tiling_mode                 = tiling_mode;
	attribute->aspect_ratio                = 0;
	attribute->width                       = width;
	attribute->height                      = height;
	attribute->pitch_in_pixel              = 0;
	attribute->option                      = option;
	attribute->pixel_format                = pixel_format;
	attribute->dcc_cb_register_clear_color = dcc_cb_register_clear_color;
	attribute->dcc_control                 = dcc_control;
}

KYTY_SYSV_ABI int VideoOutSetFlipRate(int handle, int rate) {
	PRINT_NAME();

	LOGF("\trate = %d\n", rate);

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (rate < 0 || rate > 2) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	Common::LockGuard lock(ctx->mutex);
	ctx->flip_rate = rate;
	if (LogFlips()) {
		std::fprintf(stderr, "flip [%.2f]: rate %d (a flip every %d vblanks)\n", FlipLogMs(), rate, rate + 1);
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutDeleteFlipEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::Flip);
}

KYTY_SYSV_ABI int VideoOutAddFlipEvent(EventQueue::KernelEqueue eq, int handle, void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::Flip, udata);
}

KYTY_SYSV_ABI int VideoOutDeleteVblankEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::Vblank);
}

KYTY_SYSV_ABI int VideoOutDeletePreVblankStartEvent(EventQueue::KernelEqueue eq, int handle) {
	PRINT_NAME();
	return DeleteVideoOutEvent(handle, eq, VideoOutEventKind::PreVblankStart);
}

KYTY_SYSV_ABI int VideoOutAddVblankEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                         void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::Vblank, udata);
}

KYTY_SYSV_ABI int VideoOutAddPreVblankStartEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                                 void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::PreVblankStart, udata);
}

KYTY_SYSV_ABI int VideoOutAddOutputModeEvent(LibKernel::EventQueue::KernelEqueue eq, int handle,
                                             void* udata) {
	PRINT_NAME();
	return RegisterVideoOutEvent(handle, eq, VideoOutEventKind::OutputMode, udata);
}

// Why the display buffer `info` cannot be shown directly (KYTY_DIRECT_PRESENT=1), or nullptr when it
// can; `depth` gets the X depth of its pixmap. The display engine reads it as it is, so it must be a
// layout and size the window shows 1:1: the main port, Prospero's 64 KiB render-target tiling (the AMD
// 64K_R_X modifier), 4 bytes per pixel with an X depth, no DCC, SDR, the window's size.
static const char* DirectIneligibility(Graphics::DirectPresenter& direct, Graphics::Presenter& presenter, int bus,
                                       const VideoOutBufferAttribute2& attribute, const Graphics::ImageInfo& info,
                                       uint8_t* depth) {
	*depth = 0;
	if (!Graphics::NativeGpu::DirectPresentWanted() || !direct.Usable()) {
		return "direct presentation is off";
	}
	if (bus != VIDEO_OUT_BUS_TYPE_MAIN) {
		return "not the main port";
	}
	if (info.tile_mode != Graphics::Prospero::TileMode::kRenderTarget || info.bytes_per_block != 4) {
		return "not a 4-byte render-target layout";
	}
	if (info.HasMetadata()) {
		return "compressed (DCC)";
	}
	if (Graphics::IsHdrVideoOutPixelFormat(attribute.pixel_format) || presenter.HdrSwapchainEverUsed()) {
		return "HDR";
	}
	*depth = direct.DepthFor(attribute.pixel_format);
	if (*depth == 0) {
		return "no X depth for the pixel format";
	}
	if (info.pitch % 128 != 0) {
		return "pitch not a multiple of 128";
	}
	if (!direct.WindowIs(attribute.width, attribute.height)) {
		return "size differs from the window";
	}
	return nullptr;
}

// The pixmap of a carved display buffer (0 when the X server refuses it).
static uint32_t AddDirectPixmap(Graphics::DirectPresenter& direct, const Graphics::NativeGpu::NativeDisplayBuffer& bo,
                                const Graphics::ImageInfo& info, uint8_t depth) {
	Graphics::DirectSurface surface {};
	surface.dmabuf_fd = bo.dmabuf_fd;
	surface.offset    = bo.bo_offset;
	surface.width     = info.extent.width;
	surface.height    = info.extent.height;
	surface.stride    = info.pitch * info.bytes_per_block;
	surface.depth     = depth;
	return direct.AddBuffer(surface);
}

// A display buffer being registered (sceVideoOutRegisterBuffers2), with direct presentation on: when
// the window can show it 1:1, its guest range becomes one VRAM scan-out BO at the same address
// (NativeGpu::ShadowCarveDisplayBuffer) and that BO a pixmap of the window, as the PS5's display
// engine scans the registered buffer out itself. Done now, before the title renders into it, on the
// title's thread without VideoOut locks (the copy and the X round trip take tens of milliseconds, at
// load time). One log line per buffer says which way its flips go.
static DirectSlot MakeDirectSlot(Graphics::DirectPresenter& direct, Graphics::Presenter& presenter, int bus,
                                 const BufferAttributeGroup& group, const VideoOutBuffer& buffer, int index,
                                 uint64_t label) {
	DirectSlot  slot {};
	const auto  info  = group.ImageInfo(buffer);
	uint8_t     depth = 0;
	const char* why   = DirectIneligibility(direct, presenter, bus, group.attribute, info, &depth);
	Graphics::NativeGpu::NativeDisplayBuffer bo {};
	if (why == nullptr) {
		if (Graphics::NativeGpu::ShadowCarveDisplayBuffer(info.data.address, info.data.size, &bo)) {
			slot.address = info.data.address;
			slot.size    = info.data.size;
			slot.width   = info.extent.width;
			slot.height  = info.extent.height;
			slot.pixmap  = AddDirectPixmap(direct, bo, info, depth);
			if (slot.pixmap == 0) {
				why = "the X server refused its pixmap";
			}
		} else {
			why = "its memory could not become a VRAM scan-out buffer";
		}
	}
	if (why != nullptr) {
		std::fprintf(stderr, "direct present: buffer %d (0x%016" PRIx64 ", %ux%u) takes the Vulkan path: %s\n", index,
		             info.data.address, info.extent.width, info.extent.height, why);
		return slot;
	}
	std::fprintf(stderr,
	             "direct present: buffer %d (0x%016" PRIx64 ", %ux%u, depth %u) is shown directly, pixmap 0x%x, BO offset "
	             "0x%" PRIx64 "\n",
	             index, info.data.address, info.extent.width, info.extent.height, depth, slot.pixmap, bo.bo_offset);
	// GPU writes between the carve's copy and its remap are lost; a buffer the title already flipped
	// (its release wrote the label) may have had some.
	if (label != 0) {
		std::fprintf(stderr, "direct present: buffer %d was registered after its flip release ran\n", index);
	}
	return slot;
}

KYTY_SYSV_ABI int VideoOutRegisterBuffers2(int handle, int set_index, int buffer_index_start,
                                           const VideoOutBuffers* buffers, int buffer_num,
                                           const VideoOutBufferAttribute2* attribute, int category,
                                           void* option) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (buffers == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (attribute == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}

	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX ||
	    buffer_index_start < 0 || buffer_index_start >= VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_num < 1 || buffer_num > VIDEO_OUT_BUFFER_NUM_MAX ||
	    buffer_index_start + buffer_num > VIDEO_OUT_BUFFER_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	LOGF("\t start_index    = %d\n"
	     "\t buffer_num     = %d\n"
	     "\t set_index      = %d\n"
	     "\t pixel_format   = 0x%016" PRIx64 "\n"
	     "\t tiling_mode    = %" PRIu32 "\n"
	     "\t aspect_ratio   = %" PRIu32 "\n"
	     "\t width          = %" PRIu32 "\n"
	     "\t height         = %" PRIu32 "\n"
	     "\t pitch_in_pixel = %" PRIu32 "\n"
	     "\t option         = %" PRIu64 "\n"
	     "\t category       = %d\n",
	     buffer_index_start, buffer_num, set_index, attribute->pixel_format, attribute->tiling_mode,
	     attribute->aspect_ratio, attribute->width, attribute->height, attribute->pitch_in_pixel,
	     attribute->option, category);

	if (option != nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}
	// The headset port takes reprojection buffers with a zeroed attribute. Each names its eye
	// texture, which is registered and flipped like a regular surface.
	const bool headset = ctx->bus == VIDEO_OUT_BUS_TYPE_HMD2_MAIN;
	if (headset != (category == VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_HMD2)) {
		return VIDEO_OUT_ERROR_INVALID_CATEGORY;
	}
	VideoOutBufferAttribute2                              headset_attribute {};
	std::array<VideoOutBuffers, VIDEO_OUT_BUFFER_NUM_MAX> headset_buffers {};
	if (headset) {
		for (int i = 0; i < buffer_num; i++) {
			Hmd2::ReprojectionSurface surface {};
			if (buffers[i].metadata != nullptr ||
			    !Hmd2::GetReprojectionSurface(buffers[i].data, surface) ||
			    (i != 0 && (surface.width != headset_attribute.width ||
			                surface.height != headset_attribute.height ||
			                surface.pixel_format != headset_attribute.pixel_format))) {
				return VIDEO_OUT_ERROR_INVALID_VALUE;
			}
			headset_attribute.pixel_format = surface.pixel_format;
			headset_attribute.width        = surface.width;
			headset_attribute.height       = surface.height;
			headset_buffers[i].data        = reinterpret_cast<const void*>(surface.address);
		}
		attribute = &headset_attribute;
		buffers   = headset_buffers.data();
		category  = VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED;
	}
	if (category != VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_UNCOMPRESSED &&
	    category != VIDEO_OUT_BUFFER_ATTRIBUTE_CATEGORY_COMPRESSED) {
		return VIDEO_OUT_ERROR_INVALID_CATEGORY;
	}

	BufferAttributeGroup group {
	    .attribute = *attribute,
	    .category  = category,
	    .occupied  = true,
	};
	std::vector<VideoOutBuffer> registrations;
	registrations.reserve(static_cast<size_t>(buffer_num));

	for (int i = 0; i < buffer_num; i++) {
		LOGF("\t buffers[%d]: data=%p metadata=%p\n", i, buffers[i].data, buffers[i].metadata);
		if (buffers[i].reserved[0] != nullptr || buffers[i].reserved[1] != nullptr) {
			LOGF("\t buffers[%d]: ignoring reserved fields {%p, %p}\n", i, buffers[i].reserved[0],
			     buffers[i].reserved[1]);
		}
		const auto data_address     = reinterpret_cast<uint64_t>(buffers[i].data);
		const auto metadata_address = reinterpret_cast<uint64_t>(buffers[i].metadata);
		registrations.push_back({
		    .group_index      = set_index,
		    .data_address     = data_address,
		    .metadata_address = metadata_address,
		});
		(void)group.ImageInfo(registrations.back());
	}

	// Direct presentation (KYTY_DIRECT_PRESENT=1): before the port's lock (MakeDirectSlot).
	auto*                                            direct = DriverState().GetFlipQueue().Direct();
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> direct_slots {};
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> stale_slots {};
	if (direct != nullptr && !headset) {
		int bus = 0;
		{
			Common::LockGuard lock(ctx->mutex);
			bus = ctx->bus;
		}
		for (int i = 0; i < buffer_num; i++) {
			const int index = buffer_index_start + i;
			direct_slots[static_cast<size_t>(i)] =
			    MakeDirectSlot(*direct, DriverState().GetPresenter(), bus, group, registrations[static_cast<size_t>(i)], index,
			                   std::atomic_ref<uint64_t>(ctx->labels[index]).load(std::memory_order_acquire));
		}
	}

	const int result = [&] {
		Common::LockGuard lock(ctx->mutex);
		if (!ctx->opened) {
			return VIDEO_OUT_ERROR_INVALID_HANDLE;
		}
		if (ctx->groups[set_index].occupied) {
			return VIDEO_OUT_ERROR_INVALID_INDEX;
		}
		for (int i = 0; i < buffer_num; i++) {
			if (ctx->buffers[buffer_index_start + i].Occupied()) {
				return VIDEO_OUT_ERROR_SLOT_OCCUPIED;
			}
		}

		ctx->groups[set_index] = group;
		for (int i = 0; i < buffer_num; i++) {
			ctx->buffers[buffer_index_start + i] = registrations[static_cast<size_t>(i)];
			stale_slots[static_cast<size_t>(i)]  = ctx->direct[buffer_index_start + i];
			ctx->direct[buffer_index_start + i]  = direct_slots[static_cast<size_t>(i)];
			const auto& buffer                   = registrations[static_cast<size_t>(i)];
			LOGF("\tbuffers[%d] = %016" PRIx64 " metadata = %016" PRIx64 " dcc = %08" PRIx32 "\n",
			     buffer_index_start + i, buffer.data_address, buffer.metadata_address,
			     attribute->dcc_control);
		}
		return OK;
	}();
	// A refused registration keeps nothing; a slot left over at a free index (none is expected) goes.
	ReleaseDirectSlots(direct, std::span(result == OK ? stale_slots : direct_slots).first(static_cast<size_t>(buffer_num)));

	return result;
}

KYTY_SYSV_ABI int VideoOutSubmitChangeBufferAttribute2(int handle, int set_index,
                                                       const VideoOutBufferAttribute2* attribute,
                                                       void*                           option) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (attribute == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}
	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	if (option != nullptr) {
		return VIDEO_OUT_ERROR_INVALID_OPTION;
	}

	// Direct presentation: the set's pixmaps describe the old attribute. They go now, and flips
	// requested from here on use pixmaps made for the new one; a carved buffer stays carved, and one
	// that was not carved at registration is not carved now, after the title may have rendered into it.
	auto*                                            direct = DriverState().GetFlipQueue().Direct();
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> old_pixmaps {};
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> carved {};
	std::array<VideoOutBuffer, VIDEO_OUT_BUFFER_NUM_MAX> carved_buffers {};
	BufferAttributeGroup                             replacement {};
	int                                              bus = 0;
	{
		Common::LockGuard lock(ctx->mutex);
		const auto&       current = ctx->groups[set_index];
		if (!ctx->opened || !current.occupied) {
			return VIDEO_OUT_ERROR_INVALID_INDEX;
		}

		replacement = BufferAttributeGroup {
		    .attribute = *attribute,
		    .category  = current.category,
		    .occupied  = true,
		};
		for (const auto& buffer: ctx->buffers) {
			if (buffer.group_index == set_index) {
				(void)replacement.ImageInfo(buffer);
			}
		}
		ctx->groups[set_index] = replacement;
		bus                    = ctx->bus;
		for (int i = 0; i < VIDEO_OUT_BUFFER_NUM_MAX; i++) {
			auto& slot = ctx->direct[i];
			if (ctx->buffers[i].group_index != set_index || (slot.pixmap == 0 && slot.size == 0)) {
				continue;
			}
			old_pixmaps[i].pixmap = slot.pixmap;
			slot.pixmap           = 0;
			carved[i]             = slot;
			carved_buffers[i]     = ctx->buffers[i];
		}
	}
	if (direct == nullptr) {
		return OK;
	}
	ReleaseDirectSlots(direct, old_pixmaps);
	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> unused {};
	for (int i = 0; i < VIDEO_OUT_BUFFER_NUM_MAX; i++) {
		if (carved[i].size == 0) {
			continue;
		}
		const auto  info  = replacement.ImageInfo(carved_buffers[i]);
		uint8_t     depth = 0;
		const char* why   = DirectIneligibility(*direct, DriverState().GetPresenter(), bus, replacement.attribute, info, &depth);
		Graphics::NativeGpu::NativeDisplayBuffer bo {};
		uint32_t                                 pixmap = 0;
		if (why == nullptr && (info.data.address != carved[i].address || info.data.size > carved[i].size)) {
			why = "the new layout is larger than its VRAM scan-out buffer";
		}
		// The range lies in its carve, so this only looks the carve up.
		if (why == nullptr && !Graphics::NativeGpu::ShadowCarveDisplayBuffer(info.data.address, info.data.size, &bo)) {
			why = "its VRAM scan-out buffer is gone";
		}
		if (why == nullptr) {
			pixmap = AddDirectPixmap(*direct, bo, info, depth);
			if (pixmap == 0) {
				why = "the X server refused its pixmap";
			}
		}
		std::fprintf(stderr, "direct present: buffer %d (%ux%u) after its attribute change %s%s\n", i, info.extent.width,
		             info.extent.height, why == nullptr ? "is shown directly" : "takes the Vulkan path: ",
		             why == nullptr ? "" : why);
		if (pixmap == 0) {
			continue;
		}
		Common::LockGuard lock(ctx->mutex);
		auto&             slot = ctx->direct[i];
		if (!ctx->opened || ctx->buffers[i].group_index != set_index ||
		    ctx->buffers[i].data_address != carved_buffers[i].data_address || slot.pixmap != 0 ||
		    slot.address != carved[i].address) {
			unused[i].pixmap = pixmap; // unregistered or changed again meanwhile
			continue;
		}
		slot.pixmap = pixmap;
		slot.width  = info.extent.width;
		slot.height = info.extent.height;
	}
	ReleaseDirectSlots(direct, unused);

	return OK;
}

KYTY_SYSV_ABI int VideoOutUnregisterBuffers(int handle, int set_index) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	if (set_index < 0 || set_index >= VIDEO_OUT_BUFFER_ATTRIBUTE_NUM_MAX) {
		return VIDEO_OUT_ERROR_INVALID_INDEX;
	}

	std::array<DirectSlot, VIDEO_OUT_BUFFER_NUM_MAX> released {};
	{
		Common::LockGuard lock(ctx->mutex);
		if (!ctx->opened) {
			return VIDEO_OUT_ERROR_INVALID_HANDLE;
		}
		if (!ctx->groups[set_index].occupied) {
			return VIDEO_OUT_ERROR_INVALID_INDEX;
		}
		ctx->groups[set_index] = BufferAttributeGroup {};
		for (int i = 0; i < VIDEO_OUT_BUFFER_NUM_MAX; i++) {
			if (ctx->buffers[i].group_index == set_index) {
				ctx->buffers[i] = VideoOutBuffer {};
				released[i]     = ctx->direct[i];
				ctx->direct[i]  = DirectSlot {};
			}
		}
	}
	// The display may still hold these buffers: their holds end at their IdleNotify as usual.
	ReleaseDirectSlots(DriverState().GetFlipQueue().Direct(), released);

	return OK;
}

KYTY_SYSV_ABI int VideoOutSubmitFlip(int handle, int index, int flip_mode, int64_t flip_arg) {
	PRINT_NAME();

	auto&     driver     = DriverState();
	uint64_t  request_id = 0;
	const int result     = ReserveFlipRequest(driver, handle, index, flip_mode, flip_arg,
	                                          FlipRequestSource::Cpu, request_id);
	if (result == VIDEO_OUT_ERROR_INVALID_FLIP_MODE) {
		LOGF("\t unsupported flip_mode = %d\n", flip_mode);
	}
	if (result != OK) {
		return result;
	}
	// On the PS5 a CPU flip goes to the display at the next vblank whatever the GPU is doing.
	if (driver.GetFlipQueue().TryDirect(request_id, false)) {
		return OK;
	}
	driver.Renderer().GetGpu().SubmitFlipPreparation(request_id);

	return OK;
}

KYTY_SYSV_ABI int VideoOutSubmitEopFlip(int handle, int index, int flip_mode, int64_t flip_arg,
                                        uint32_t* ctx) {
	PRINT_NAME();
	EXIT_IF(g_video_out_driver == nullptr);

	const int result = g_video_out_driver->SubmitEopFlip(handle, index, flip_mode, flip_arg, ctx, false);
	if (result == VIDEO_OUT_ERROR_INVALID_FLIP_MODE) {
		LOGF("\t unsupported flip_mode = %d\n", flip_mode);
	}
	return result;
}

KYTY_SYSV_ABI int VideoOutGetBufferLabelAddress(int handle, void** address) {
	PRINT_NAME();

	if (address == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}
	EXIT_IF(g_video_out_driver == nullptr);
	uint64_t* labels = nullptr;
	const int result = g_video_out_driver->GetBufferLabelAddress(handle, &labels);
	if (result == OK) {
		*address = labels;
	}
	return result;
}

int VideoOutDriver::SubmitFlipFromGpu(Graphics::CommandBuffer& buffer, int handle, int index,
                                      int flip_mode, int64_t flip_arg, uint64_t& request_id) {
	EXIT_IF(buffer.IsInvalid());

	const int result = ReserveFlipRequest(*m_impl, handle, index, flip_mode, flip_arg,
	                                      FlipRequestSource::GpuEop, request_id);
	if (result != OK) {
		return result;
	}
	// The R_FLIP point the GPU has passed: a direct buffer goes to the display now, and the caller's
	// deferred CompleteFlip finds nothing to complete.
	if (m_impl->GetFlipQueue().TryDirect(request_id, true)) {
		return OK;
	}
	m_impl->GetFlipQueue().Prepare(request_id, buffer);

	return OK;
}

void VideoOutDriver::PrepareFlip(uint64_t request_id, Graphics::CommandBuffer& buffer) {
	m_impl->GetFlipQueue().Prepare(request_id, buffer);
}

void VideoOutDriver::CompleteFlip(uint64_t request_id) {
	m_impl->GetFlipQueue().Complete(request_id);
}

void VideoOutDriver::WaitForSubmitSlot(int handle, bool on_gpu_thread) {
	auto* cfg = m_impl->Get(handle);
	EXIT_IF(cfg == nullptr);
	m_impl->GetFlipQueue().WaitForSubmitSlot(*cfg, on_gpu_thread);
}

uint64_t VideoOutDriver::NextFlipRequestId() {
	return m_impl->GetFlipQueue().NextRequestId();
}

uint64_t VideoOutDriver::EopFlipOrder() {
	return m_impl->GetFlipQueue().EopOrder();
}

bool VideoOutDriver::IsSafeForRendering(int handle, int index, uint64_t before, uint64_t eop_mark) {
	auto* ctx = m_impl->Get(handle);
	EXIT_IF(ctx == nullptr);

	EXIT_NOT_IMPLEMENTED(!IsValidBufferIndex(index));
	return m_impl->GetFlipQueue().IsSafeForRendering(*ctx, index, before, eop_mark);
}

// The flip is reserved now, when the command buffer is written, as the PS5 does; the GPU's
// end-of-pipe interrupt with `*ctx` submits it later (EopFlip). Until then it holds its place in the
// queue, so flips submitted after it are shown after it.
int VideoOutDriver::SubmitEopFlip(int handle, int index, int flip_mode, int64_t flip_arg,
                                  uint32_t* ctx, bool wait_for_slot) {
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}
	for (;;) {
		uint64_t  request_id = 0;
		const int result     = ReserveFlipRequest(*m_impl, handle, index, flip_mode, flip_arg,
		                                          FlipRequestSource::GpuEop, request_id);
		if (result == OK) {
			EXIT_IF(request_id == 0);
			*ctx = EopFlipContext(request_id);
			return OK;
		}
		if (result != VIDEO_OUT_ERROR_FLIP_QUEUE_FULL || !wait_for_slot) {
			return result;
		}
		// The title's thread, while it writes a command buffer: the GPU thread drains the queue.
		WaitForSubmitSlot(handle, false);
	}
}

int VideoOutDriver::GetBufferLabelAddress(int handle, uint64_t** labels) {
	if (labels == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}
	auto* ctx = m_impl->Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	*labels = ctx->labels.data();
	return OK;
}

// On the PS5 the kernel takes the flip's end-of-pipe interrupt and submits the flip; the title never
// sees it. Here the flip is prepared on the GPU thread ahead of queued submissions, as the R_FLIP
// tracker point's flip is (GuestGpu::SubmitEopFlip). An interrupt that arrives while the GPU shuts
// down is dropped: the request goes back to holding nothing (no EOP order), and VideoOutShutdown discards it.
bool VideoOutDriver::EopFlip(uint32_t ctx) {
	uint64_t request_id = 0;
	if (!m_impl->GetFlipQueue().TakeEopFlip(ctx, request_id)) {
		return false;
	}
	// As the PS5 kernel does from this interrupt: a direct buffer is queued to the display here, with
	// no GPU work and no hop to the GPU thread.
	if (m_impl->GetFlipQueue().TryDirect(request_id, false)) {
		return true;
	}
	if (!m_impl->Renderer().GetGpu().SubmitEopFlipRequest(request_id)) {
		m_impl->GetFlipQueue().DropEopOrder(request_id);
		return false;
	}
	return true;
}

bool VideoOutDriver::IsBufferLabel(uint64_t addr, int* handle, int* index) {
	int h = 0;
	int i = 0;
	if (!m_impl->FindLabel(addr, h, i)) {
		return false;
	}
	if (handle != nullptr) {
		*handle = h;
	}
	if (index != nullptr) {
		*index = i;
	}
	return true;
}

KYTY_SYSV_ABI int VideoOutGetFlipStatus(int handle, VideoOutFlipStatus* status) {
	PRINT_NAME();

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	DriverState().GetFlipQueue().GetFlipStatus(*ctx, *status);
	if (LogFlips()) {
		static uint64_t last_count = ~0ull;
		static int      last_pending = -1;
		if (status->count != last_count || status->flipPendingNum != last_pending) {
			last_count   = status->count;
			last_pending = status->flipPendingNum;
			VoLog("GetFlipStatus count/pending/current", static_cast<int64_t>(status->count), status->flipPendingNum,
			      status->currentBuffer);
		}
	}

	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t processTimeCounter = %" PRIu64 "\n"
	     "\t submitProcessTimeCounter = %" PRIu64 "\n"
	     "\t flipArg = %" PRId64 "\n"
	     "\t gcQueueNum = %d\n"
	     "\t flipPendingNum = %d\n"
	     "\t currentBuffer = %d\n",
	     status->count, status->processTime, status->processTimeCounter,
	     status->submitProcessTimeCounter, status->flipArg, status->gcQueueNum,
	     status->flipPendingNum, status->currentBuffer);

	return OK;
}

KYTY_SYSV_ABI int VideoOutIsFlipPending(int handle) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	VideoOutFlipStatus status {};
	DriverState().GetFlipQueue().GetFlipStatus(*ctx, status);
	if (LogFlips()) {
		static int last = -1;
		if (status.flipPendingNum != last) {
			last = status.flipPendingNum;
			VoLog("IsFlipPending", status.flipPendingNum);
		}
	}

	if (Config::GraphicsDebugDumpEnabled()) {
		LOGF("\t flipPendingNum = %d\n", status.flipPendingNum);
	}

	return status.flipPendingNum;
}

KYTY_SYSV_ABI int VideoOutGetVblankStatus(int handle, VideoOutVblankStatus* status) {
	PRINT_NAME();

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	*status = ctx->vblank_status;
	ctx->mutex.Unlock();
	if (LogFlips()) {
		static uint64_t last = ~0ull;
		if (status->count != last) {
			last = status->count;
			VoLog("GetVblankStatus count", static_cast<int64_t>(status->count));
		}
	}

	LOGF("\t count = %" PRIu64 "\n"
	     "\t processTime = %" PRIu64 "\n"
	     "\t processTimeCounter = %" PRIu64 "\n",
	     status->count, status->processTime, status->processTimeCounter);

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetEventId(const EventQueue::KernelEvent* ev) {
	PRINT_NAME();

	if (ev == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	switch (ev->ident) {
		case VIDEO_OUT_EVENT_FLIP:
		case VIDEO_OUT_EVENT_VBLANK:
		case VIDEO_OUT_EVENT_PRE_VBLANK_START:
		case VIDEO_OUT_EVENT_SET_MODE: return static_cast<int>(ev->ident);
		default: return VIDEO_OUT_ERROR_INVALID_EVENT;
	}
}

KYTY_SYSV_ABI int VideoOutGetEventData(const EventQueue::KernelEvent* ev, int64_t* data) {
	PRINT_NAME();

	if (ev == nullptr || data == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	uint64_t event_data = static_cast<uint64_t>(ev->data) >> 16u;
	if (ev->ident == VIDEO_OUT_EVENT_FLIP &&
	    (static_cast<uint64_t>(ev->data) & 0x8000000000000000ULL) != 0) {
		event_data |= 0xffff000000000000ULL;
	}

	*data = static_cast<int64_t>(event_data);

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetEventCount(const EventQueue::KernelEvent* ev) {
	PRINT_NAME();

	if (ev == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (ev->filter != EventQueue::KERNEL_EVFILT_VIDEO_OUT) {
		return VIDEO_OUT_ERROR_INVALID_EVENT;
	}

	return static_cast<int>((static_cast<uint64_t>(ev->data) >> 12u) & 0xfu);
}

KYTY_SYSV_ABI int VideoOutWaitVblank(int handle) {
	PRINT_NAME();

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	Common::LockGuard lock(ctx->mutex);
	const auto        count = ctx->vblank_status.count;
	if (LogFlips()) {
		VoLog("WaitVblank enter", static_cast<int64_t>(count));
	}
	while (ctx->opened && ctx->vblank_status.count == count) {
		ctx->vblank_cond.Wait(&ctx->mutex);
	}
	if (LogFlips()) {
		VoLog("WaitVblank exit", static_cast<int64_t>(ctx->vblank_status.count));
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutGetOutputStatus(int handle, VideoOutOutputStatus* status) {
	PRINT_NAME();

	if (status == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto& driver = DriverState();
	auto* ctx    = driver.Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	int32_t attribute3 = 0;
	Loader::SystemContentParamSfoGetInt("ATTRIBUTE3", &attribute3);
	int32_t attribute = 0;
	const bool hdr_allowed = !Loader::SystemContentParamSfoGetInt("ATTRIBUTE", &attribute) ||
	                         (attribute & 0x20000000) != 0;
	ctx->mutex.Lock();
	// Primary output reports 4K unless param.json Video-out Info enables resolution detection.
	status->resolution =
	    ((attribute3 & 4) != 0 && driver.width < 3840 && driver.height < 2160 ? 1u : 2u);
	status->dynamicRange = hdr_allowed && driver.IsHdrSupported() ? 2u : 1u;
	status->refreshRate =
	    (ctx->output_mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ || Config::GetVblankFrequency() >= 119
	         ? VIDEO_OUT_REFRESH_RATE_119_88HZ
	         : VIDEO_OUT_REFRESH_RATE_59_94HZ);
	status->flags       = 0;
	status->reserved[0] = 0;
	status->reserved[1] = 0;
	status->reserved[2] = 0;
	ctx->mutex.Unlock();

	return OK;
}

KYTY_SYSV_ABI int VideoOutAllowOutputResolutionWqhdDetection(int handle) {
	if (DriverState().Get(handle) == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	return OK;
}

KYTY_SYSV_ABI int VideoOutVrrPegToFixedRate(int handle, uint64_t arg1, uint64_t arg2) {
	PRINT_NAME();

	static std::atomic_bool logged {false};
	if (!logged.exchange(true, std::memory_order_relaxed)) {
		LOGF("\t handle = %d\n"
		     "\t arg1   = 0x%016" PRIx64 "\n"
		     "\t arg2   = 0x%016" PRIx64 "\n",
		     handle, arg1, arg2);
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutVrrUnpegFromFixedRate(int handle) {
	PRINT_NAME();

	static std::atomic_bool logged {false};
	if (!logged.exchange(true, std::memory_order_relaxed)) {
		LOGF("\t handle = %d\n", handle);
	}

	return OK;
}

static int ValidateOutputConfig(int handle, uint64_t mode, const VideoOutOutputOptions* options,
                                void* reserved_ptr, uint64_t reserved) {
	if (DriverState().Get(handle) == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	if (reserved_ptr != nullptr || reserved != 0) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	if (options != nullptr) {
		for (auto v: options->internalData) {
			if (v != 0) {
				return VIDEO_OUT_ERROR_INVALID_OPTION;
			}
		}
	}

	if (mode != VIDEO_OUT_OUTPUT_MODE_DEFAULT && mode != VIDEO_OUT_OUTPUT_MODE_119_88HZ &&
	    mode != Hmd2::OUTPUT_MODE_89_91HZ && mode != Hmd2::OUTPUT_MODE_119_88HZ) {
		return VIDEO_OUT_ERROR_UNSUPPORTED_OUTPUT_MODE;
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutInitializeOutputOptions(VideoOutOutputOptions* options) {
	PRINT_NAME();

	if (options == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	memset(options, 0, sizeof(VideoOutOutputOptions));

	return OK;
}

KYTY_SYSV_ABI int VideoOutIsOutputSupported(int handle, uint64_t mode,
                                            const VideoOutOutputOptions* options,
                                            void* reserved_ptr, uint64_t reserved) {
	PRINT_NAME();

	LOGF("\t mode = 0x%016" PRIx64 "\n", mode);

	int result = ValidateOutputConfig(handle, mode, options, reserved_ptr, reserved);
	if (result != OK) {
		return result;
	}

	if (mode == VIDEO_OUT_OUTPUT_MODE_119_88HZ) {
		return (Config::GetVblankFrequency() >= 119 ? VIDEO_OUT_TRUE : VIDEO_OUT_FALSE);
	}

	return VIDEO_OUT_TRUE;
}

KYTY_SYSV_ABI int VideoOutConfigureOutput(int handle, uint64_t mode,
                                          const VideoOutOutputOptions* options, void* reserved_ptr,
                                          uint64_t reserved) {
	PRINT_NAME();

	LOGF("\t mode = 0x%016" PRIx64 "\n", mode);

	int result = VideoOutIsOutputSupported(handle, mode, options, reserved_ptr, reserved);
	if (result < 0) {
		return result;
	}
	if (result == VIDEO_OUT_FALSE) {
		return VIDEO_OUT_ERROR_UNAVAILABLE_OUTPUT_MODE;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	ctx->mutex.Lock();
	ctx->output_mode = mode;
	TriggerVideoOutEvents(*ctx, VideoOutEventKind::OutputMode,
	                      reinterpret_cast<void*>(ctx->output_mode));
	ctx->mutex.Unlock();

	return OK;
}

KYTY_SYSV_ABI int VideoOutSetWindowModeMargins(int handle, int top, int bottom) {
	PRINT_NAME();

	[[maybe_unused]] auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	LOGF("\t top    = %d\n"
	     "\t bottom = %d\n",
	     top, bottom);

	return OK;
}

KYTY_SYSV_ABI int VideoOutLatencyControlWaitBeforeInput(int handle) {
	PRINT_NAME();
	if (LogFlips()) {
		VoLog("LatencyControlWaitBeforeInput", handle);
	}

	[[maybe_unused]] auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	return OK;
}

KYTY_SYSV_ABI int VideoOutLatencyMeasureSetStartPoint(int handle, uint32_t point) {
	PRINT_NAME();

	[[maybe_unused]] auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}

	LOGF("\t point = %" PRIu32 "\n", point);

	return OK;
}

KYTY_SYSV_ABI int VideoOutColorSettingsSetGamma(VideoOutColorSettings* settings, float gamma) {
	PRINT_NAME();

	if (settings == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	if (gamma < 0.1f || gamma > 2.0f) {
		return VIDEO_OUT_ERROR_INVALID_VALUE;
	}

	settings->gamma = gamma;
	return OK;
}

KYTY_SYSV_ABI int VideoOutAdjustColor(int handle, const VideoOutColorSettings* settings) {
	PRINT_NAME();

	if (settings == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_ADDRESS;
	}

	auto* ctx = DriverState().Get(handle);
	if (ctx == nullptr) {
		return VIDEO_OUT_ERROR_INVALID_HANDLE;
	}
	ctx->mutex.Lock();
	ctx->gamma = settings->gamma;
	ctx->mutex.Unlock();

	return OK;
}

} // namespace Libs::VideoOut
