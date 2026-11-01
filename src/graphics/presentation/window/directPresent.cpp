#include "graphics/presentation/window/directPresent.h"

#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/native_gpu/nativeShadow.h"

#include <SDL3/SDL.h>

#include <cstdio>

#if defined(KYTY_DIRECT_PRESENT) && KYTY_DIRECT_PRESENT
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <iterator>
#include <mutex>
#include <set>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xcb/dri3.h>
#include <xcb/present.h>
#include <xcb/xcb.h>
#endif

namespace Libs::Graphics {

#if defined(KYTY_DIRECT_PRESENT) && KYTY_DIRECT_PRESENT

namespace {

// The one modifier a display buffer can be shown with: AMD, GFX10 tiling, SW_64KB_R_X, PIPE_XOR_BITS 4,
// no DCC. It is the layout the PS5's GPU writes a Prospero kRenderTarget surface in (the tiler and the
// DRI3 probe agree on it), so the buffer is shown as written; any other modifier would mean retiling.
constexpr uint64_t kModifier = 0x0200000000801b02ull;

// NotifyMSC serials. Pixmap serials stay below them, so a NotifyMSC completion is never taken for a flip.
constexpr uint32_t kPrimeSerial = 0xfffffffeu;
constexpr uint32_t kStopSerial  = 0xffffffffu;
constexpr uint32_t kSerialLimit = 0xfffffff0u;

// Completions between two summary lines (10 s at 60 Hz).
constexpr uint64_t kSummaryEvery = 600;

uint64_t MonotonicUs() {
	timespec ts {};
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<uint64_t>(ts.tv_sec) * 1000000u + static_cast<uint64_t>(ts.tv_nsec) / 1000u;
}

bool Lists(const uint64_t* list, int n) {
	return std::find(list, list + std::max(n, 0), kModifier) != list + std::max(n, 0);
}

void LogList(const char* what, const uint64_t* list, int n) {
	std::fprintf(stderr, "direct present:   %s modifiers (%d):", what, n);
	for (int i = 0; i < n && i < 32; i++) {
		std::fprintf(stderr, " 0x%016llx", static_cast<unsigned long long>(list[i]));
	}
	std::fprintf(stderr, "\n");
}

// Whether the server lists kModifier for pixmaps of `depth` (bpp 32) on `window`, in the window's
// list (what the compositor takes for this window, Xwayland's scan-out tranche) or the screen's.
bool ServerTakes(xcb_connection_t* conn, xcb_window_t window, uint8_t depth, bool log_lists) {
	auto* reply =
	    xcb_dri3_get_supported_modifiers_reply(conn, xcb_dri3_get_supported_modifiers(conn, window, depth, 32), nullptr);
	if (reply == nullptr) {
		return false;
	}
	const uint64_t* win    = xcb_dri3_get_supported_modifiers_window_modifiers(reply);
	const int       win_n  = xcb_dri3_get_supported_modifiers_window_modifiers_length(reply);
	const uint64_t* scr    = xcb_dri3_get_supported_modifiers_screen_modifiers(reply);
	const int       scr_n  = xcb_dri3_get_supported_modifiers_screen_modifiers_length(reply);
	const bool      listed = Lists(win, win_n) || Lists(scr, scr_n);
	if (!listed && log_lists) {
		LogList("window", win, win_n);
		LogList("screen", scr, scr_n);
	}
	std::free(reply);
	return listed;
}

// Depth 30 pixmaps also need a depth-30, 32 bpp pixmap format on the server.
bool HasPixmapFormat(xcb_connection_t* conn, uint8_t depth) {
	const xcb_setup_t* setup = xcb_get_setup(conn);
	for (auto it = xcb_setup_pixmap_formats_iterator(setup); it.rem > 0; xcb_format_next(&it)) {
		if (it.data->depth == depth && it.data->bits_per_pixel == 32) {
			return true;
		}
	}
	return false;
}

DirectCompleteMode ModeOf(uint8_t mode) {
	switch (mode) {
		case XCB_PRESENT_COMPLETE_MODE_FLIP: return DirectCompleteMode::Flip;
		case XCB_PRESENT_COMPLETE_MODE_SKIP: return DirectCompleteMode::Skip;
		case XCB_PRESENT_COMPLETE_MODE_COPY:
		case XCB_PRESENT_COMPLETE_MODE_SUBOPTIMAL_COPY:
		default: return DirectCompleteMode::Copy;
	}
}

} // namespace

struct DirectPresenter::Impl {
	// One PresentPixmap, from the request to its IdleNotify. The X server sends a copied or skipped
	// pixmap's IdleNotify before its CompleteNotify; such an idle waits here (`idled`) until the complete
	// has been delivered, so VideoOut always sees the flip before the release.
	struct Pending {
		uint64_t token     = 0;
		uint64_t sent_us   = 0;
		uint32_t sequence  = 0; // X request sequence, to match an error to the request
		bool     completed = false;
		bool     idled     = false;
	};

	xcb_connection_t*    conn    = nullptr;
	xcb_window_t         window  = 0;
	xcb_special_event_t* events  = nullptr;
	bool                 depth30 = false;

	std::atomic<uint32_t> win_w {0};
	std::atomic<uint32_t> win_h {0};
	std::atomic<uint64_t> last_msc {0};
	std::atomic<uint32_t> in_flight {0};
	std::atomic<bool>     disabled {false};
	std::atomic<bool>     lost {false};
	std::atomic<bool>     stop {false};

	// Leaf lock: the Present request is built, sent and flushed under it, so serials follow the order
	// in which VideoOut queues flips, whichever thread sends them.
	std::mutex                            mutex;
	uint32_t                              serial = 0;
	std::unordered_map<uint32_t, Pending> pending;
	std::set<uint32_t>                    pixmaps;

	std::mutex             callback_mutex;
	DirectPresentCallbacks callbacks;

	std::mutex              exit_mutex;
	std::condition_variable exit_cv;
	bool                    exited = false;
	std::thread             thread;

	// Event thread only.
	uint64_t completes        = 0;
	uint32_t modes[4]         = {};
	uint32_t latency[6]       = {}; // send -> complete: < 8, < 17, < 34, < 50, < 100 ms, more
	uint64_t last_complete_us = 0;
	double   idle_sum_ms      = 0.0;
	double   idle_max_ms      = 0.0;
	uint32_t idle_count       = 0;
	uint32_t errors_logged    = 0;

	Impl() = default;
	~Impl();
	KYTY_CLASS_NO_COPY(Impl);

	uint32_t NextSerialLocked() {
		serial = serial + 1 >= kSerialLimit ? 1 : serial + 1;
		return serial;
	}

	void Run();
	void OnComplete(const xcb_present_complete_notify_event_t& e);
	void OnIdle(const xcb_present_idle_notify_event_t& e);
	void OnConfigure(const xcb_present_configure_notify_event_t& e);
	void OnLost();
	void DrainErrors();
	void Count(const Pending& p, const xcb_present_complete_notify_event_t& e, DirectCompleteMode mode);
};

DirectPresenter::Impl::~Impl() {
	// Nothing may call back into VideoOut from here on; this also waits for a callback in progress.
	{
		std::lock_guard lock(callback_mutex);
		callbacks = {};
	}
	if (thread.joinable()) {
		// A NotifyMSC with divisor 0 completes at the current MSC, which wakes the event thread. If the
		// display does not answer (window gone, compositor stopped), shutting the socket down ends the
		// wait as a connection error.
		stop.store(true);
		xcb_present_notify_msc(conn, window, kStopSerial, 0, 0, 0);
		xcb_flush(conn);
		{
			std::unique_lock lock(exit_mutex);
			if (!exit_cv.wait_for(lock, std::chrono::milliseconds(250), [this] { return exited; })) {
				::shutdown(xcb_get_file_descriptor(conn), SHUT_RDWR);
			}
		}
		thread.join();
	}
	if (conn != nullptr) {
		if (xcb_connection_has_error(conn) == 0) {
			std::lock_guard lock(mutex);
			for (const uint32_t pixmap: pixmaps) {
				xcb_free_pixmap(conn, pixmap);
			}
			xcb_flush(conn);
		}
		if (events != nullptr) {
			xcb_unregister_for_special_event(conn, events);
		}
		// The server drops whatever this client still owns; a pixmap on screen stays referenced by
		// Present until the window shows something else.
		xcb_disconnect(conn);
	}
}

void DirectPresenter::Impl::Run() {
	KYTY_PROFILER_THREAD("DirectPresent");
	// Completions arrive in bursts, one per vblank; at the lowest P-state the flip event reaches the
	// title later than on the PS5, where the kernel's flip interrupt handler runs at once.
	Common::Thread::RunAtTopClock("KYTY_UCLAMP");
	for (;;) {
		xcb_generic_event_t* ev = xcb_wait_for_special_event(conn, events);
		if (ev == nullptr) {
			// Only a connection error ends the wait.
			if (!stop.load()) {
				OnLost();
			}
			break;
		}
		const auto* ge = reinterpret_cast<const xcb_present_generic_event_t*>(ev);
		switch (ge->evtype) {
			case XCB_PRESENT_EVENT_COMPLETE_NOTIFY:
				OnComplete(*reinterpret_cast<const xcb_present_complete_notify_event_t*>(ev));
				break;
			case XCB_PRESENT_EVENT_IDLE_NOTIFY:
				OnIdle(*reinterpret_cast<const xcb_present_idle_notify_event_t*>(ev));
				break;
			case XCB_PRESENT_EVENT_CONFIGURE_NOTIFY:
				OnConfigure(*reinterpret_cast<const xcb_present_configure_notify_event_t*>(ev));
				break;
			default: break;
		}
		std::free(ev);
		DrainErrors();
		if (stop.load()) {
			break;
		}
	}
	{
		std::lock_guard lock(exit_mutex);
		exited = true;
	}
	exit_cv.notify_all();
}

void DirectPresenter::Impl::OnComplete(const xcb_present_complete_notify_event_t& e) {
	if (e.kind == XCB_PRESENT_COMPLETE_KIND_NOTIFY_MSC) {
		if (e.msc != 0) {
			last_msc.store(e.msc);
		}
		return;
	}
	Pending p;
	bool    idle_too = false;
	{
		std::lock_guard lock(mutex);
		auto            it = pending.find(e.serial);
		if (it == pending.end() || it->second.completed) {
			return;
		}
		it->second.completed = true;
		in_flight.fetch_sub(1);
		p = it->second;
		if (p.idled) {
			idle_too = true;
			pending.erase(it);
		}
	}
	// A skipped present carries no MSC; the counter only moves forward from real completions.
	if (e.msc != 0) {
		last_msc.store(e.msc);
	}
	const DirectCompleteMode mode = ModeOf(e.mode);
	Count(p, e, mode);
	std::lock_guard lock(callback_mutex);
	if (callbacks.complete) {
		callbacks.complete(p.token, e.msc, e.ust, mode);
	}
	if (idle_too && callbacks.idle) {
		callbacks.idle(p.token);
	}
}

void DirectPresenter::Impl::OnIdle(const xcb_present_idle_notify_event_t& e) {
	uint64_t token = 0;
	{
		std::lock_guard lock(mutex);
		auto            it = pending.find(e.serial);
		if (it == pending.end()) {
			return;
		}
		if (!it->second.completed) {
			it->second.idled = true;
			return;
		}
		token = it->second.token;
		pending.erase(it);
	}
	// The buffer shown before is released once the next one is on screen: about one vblank after
	// that one's completion when the compositor scans out, more when it composites.
	if (last_complete_us != 0) {
		const double ms = static_cast<double>(MonotonicUs() - last_complete_us) / 1000.0;
		idle_sum_ms += ms;
		idle_max_ms = std::max(idle_max_ms, ms);
		idle_count++;
	}
	std::lock_guard lock(callback_mutex);
	if (callbacks.idle) {
		callbacks.idle(token);
	}
}

void DirectPresenter::Impl::OnConfigure(const xcb_present_configure_notify_event_t& e) {
	const uint32_t w      = e.width;
	const uint32_t h      = e.height;
	const uint32_t old_w  = win_w.exchange(w);
	const uint32_t old_h  = win_h.exchange(h);
	if (old_w == w && old_h == h) {
		return;
	}
	std::fprintf(stderr, "direct present: window is %ux%u\n", w, h);
	std::lock_guard lock(callback_mutex);
	if (callbacks.resized) {
		callbacks.resized(w, h);
	}
}

void DirectPresenter::Impl::OnLost() {
	lost.store(true);
	std::vector<std::pair<uint32_t, Pending>> all;
	{
		std::lock_guard lock(mutex);
		all.assign(pending.begin(), pending.end());
		pending.clear();
		in_flight.store(0);
	}
	std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) { return a.second.token < b.second.token; });
	std::fprintf(stderr, "direct present: X connection lost (error %d), %zu presents released, presenting through Vulkan\n",
	             xcb_connection_has_error(conn), all.size());
	// VideoOut must never wait for an event that cannot come: every flip completes and every buffer
	// is released here.
	std::lock_guard lock(callback_mutex);
	for (const auto& [s, p]: all) {
		if (!p.completed && callbacks.complete) {
			callbacks.complete(p.token, 0, 0, DirectCompleteMode::Lost);
		}
	}
	for (const auto& [s, p]: all) {
		if (callbacks.idle) {
			callbacks.idle(p.token);
		}
	}
	if (callbacks.lost) {
		callbacks.lost();
	}
}

// Errors of the requests sent without a check (PresentPixmap, FreePixmap) land in the connection's
// ordinary event queue, which nothing else reads. A refused PresentPixmap gets neither a complete nor an
// idle, so it is completed here as Lost, and direct presents stop for the session.
void DirectPresenter::Impl::DrainErrors() {
	while (xcb_generic_event_t* ev = xcb_poll_for_queued_event(conn)) {
		if (ev->response_type == 0) {
			const auto* error = reinterpret_cast<const xcb_generic_error_t*>(ev);
			Pending     p;
			bool        refused = false;
			{
				std::lock_guard lock(mutex);
				for (auto it = pending.begin(); it != pending.end(); ++it) {
					if (!it->second.completed && it->second.sequence == error->full_sequence) {
						p       = it->second;
						refused = true;
						pending.erase(it);
						in_flight.fetch_sub(1);
						break;
					}
				}
			}
			if (errors_logged < 16 || refused) {
				errors_logged++;
				std::fprintf(stderr, "direct present: X error %u (request %u.%u, sequence %u)%s\n", error->error_code,
				             error->major_code, error->minor_code, error->full_sequence,
				             refused ? ", PresentPixmap refused" : "");
			}
			if (refused) {
				if (!disabled.exchange(true)) {
					std::fprintf(stderr, "direct present: off for this session (the X server refused a PresentPixmap), "
					                     "presenting through Vulkan\n");
				}
				std::lock_guard lock(callback_mutex);
				if (callbacks.complete) {
					callbacks.complete(p.token, 0, 0, DirectCompleteMode::Lost);
				}
				if (callbacks.idle) {
					callbacks.idle(p.token);
				}
			}
		}
		std::free(ev);
	}
}

void DirectPresenter::Impl::Count(const Pending& p, const xcb_present_complete_notify_event_t& e, DirectCompleteMode mode) {
	const uint64_t now = MonotonicUs();
	// The server's UST is CLOCK_MONOTONIC in microseconds, the clock of `sent_us`. A skipped present
	// has no UST; then the time the completion arrived stands in.
	const uint64_t at = e.ust >= p.sent_us && e.ust <= now + 1000000u ? e.ust : now;
	const uint64_t us = at >= p.sent_us ? at - p.sent_us : 0;
	const uint32_t bucket = us < 8000 ? 0 : us < 17000 ? 1 : us < 34000 ? 2 : us < 50000 ? 3 : us < 100000 ? 4 : 5;
	latency[bucket]++;
	modes[static_cast<uint8_t>(mode)]++;
	last_complete_us = now;
	if (++completes % kSummaryEvery != 0) {
		return;
	}
	std::fprintf(stderr,
	             "direct present: %llu presents: FLIP %u COPY %u SKIP %u LOST %u; send->complete <8 ms %u, <17 %u, <34 %u, "
	             "<50 %u, <100 %u, more %u; previous buffer complete->idle avg %.1f ms max %.1f ms (%u); MSC %llu\n",
	             static_cast<unsigned long long>(completes), modes[static_cast<uint8_t>(DirectCompleteMode::Flip)],
	             modes[static_cast<uint8_t>(DirectCompleteMode::Copy)], modes[static_cast<uint8_t>(DirectCompleteMode::Skip)],
	             modes[static_cast<uint8_t>(DirectCompleteMode::Lost)], latency[0], latency[1], latency[2], latency[3], latency[4],
	             latency[5], idle_count != 0 ? idle_sum_ms / idle_count : 0.0, idle_max_ms, idle_count,
	             static_cast<unsigned long long>(last_msc.load()));
	std::fill(std::begin(modes), std::end(modes), 0u);
	std::fill(std::begin(latency), std::end(latency), 0u);
	idle_sum_ms = 0.0;
	idle_max_ms = 0.0;
	idle_count  = 0;
}

std::unique_ptr<DirectPresenter> DirectPresenter::Create(SDL_Window* window) {
	if (window == nullptr || !NativeGpu::DirectPresentWanted()) {
		return nullptr;
	}
	const auto xid = static_cast<xcb_window_t>(
	    SDL_GetNumberProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0));
	if (xid == 0) {
		std::fprintf(stderr, "direct present: not an X11 window, presenting through Vulkan\n");
		return nullptr;
	}
	auto impl    = std::make_unique<Impl>();
	impl->window = xid;
	// The same $DISPLAY as SDL's own connection, so the same server and the same window.
	impl->conn = xcb_connect(nullptr, nullptr);
	if (xcb_connection_has_error(impl->conn) != 0) {
		std::fprintf(stderr, "direct present: cannot connect to the X server, presenting through Vulkan\n");
		return nullptr;
	}
	xcb_connection_t* conn = impl->conn;

	const auto* dri3_ext    = xcb_get_extension_data(conn, &xcb_dri3_id);
	const auto* present_ext = xcb_get_extension_data(conn, &xcb_present_id);
	if (dri3_ext == nullptr || dri3_ext->present == 0 || present_ext == nullptr || present_ext->present == 0) {
		std::fprintf(stderr, "direct present: the X server lacks DRI3 or Present, presenting through Vulkan\n");
		return nullptr;
	}
	auto* dri3    = xcb_dri3_query_version_reply(conn, xcb_dri3_query_version(conn, 1, 2), nullptr);
	auto* present = xcb_present_query_version_reply(conn, xcb_present_query_version(conn, 1, 2), nullptr);
	const auto at_least_1_2 = [](uint32_t major, uint32_t minor) { return major > 1 || (major == 1 && minor >= 2); };
	const bool versions_ok  = dri3 != nullptr && present != nullptr && at_least_1_2(dri3->major_version, dri3->minor_version) &&
	                         at_least_1_2(present->major_version, present->minor_version);
	if (!versions_ok) {
		std::fprintf(stderr, "direct present: the X server has DRI3 %u.%u and Present %u.%u, 1.2 needed, presenting through Vulkan\n",
		             dri3 != nullptr ? dri3->major_version : 0, dri3 != nullptr ? dri3->minor_version : 0,
		             present != nullptr ? present->major_version : 0, present != nullptr ? present->minor_version : 0);
	}
	std::free(dri3);
	std::free(present);
	if (!versions_ok) {
		return nullptr;
	}

	auto* geometry = xcb_get_geometry_reply(conn, xcb_get_geometry(conn, xid), nullptr);
	if (geometry == nullptr) {
		std::fprintf(stderr, "direct present: window 0x%x not found, presenting through Vulkan\n", xid);
		return nullptr;
	}
	impl->win_w.store(geometry->width);
	impl->win_h.store(geometry->height);
	std::free(geometry);

	if (!ServerTakes(conn, xid, 24, true)) {
		std::fprintf(stderr, "direct present: the X server lists no modifier 0x%016llx for depth 24, presenting through Vulkan\n",
		             static_cast<unsigned long long>(kModifier));
		return nullptr;
	}
	impl->depth30 = HasPixmapFormat(conn, 30) && ServerTakes(conn, xid, 30, false);

	const xcb_present_event_t eid = xcb_generate_id(conn);
	impl->events                  = xcb_register_for_special_xge(conn, &xcb_present_id, eid, nullptr);
	const uint32_t mask           = XCB_PRESENT_EVENT_MASK_COMPLETE_NOTIFY | XCB_PRESENT_EVENT_MASK_IDLE_NOTIFY |
	                      XCB_PRESENT_EVENT_MASK_CONFIGURE_NOTIFY;
	if (xcb_generic_error_t* error = xcb_request_check(conn, xcb_present_select_input_checked(conn, eid, xid, mask));
	    error != nullptr) {
		std::fprintf(stderr, "direct present: Present events of window 0x%x refused (error %u), presenting through Vulkan\n", xid,
		             error->error_code);
		std::free(error);
		return nullptr;
	}
	// The first NotifyMSC gives LastMsc() a value before the first flip.
	xcb_present_notify_msc(conn, xid, kPrimeSerial, 0, 0, 0);
	xcb_flush(conn);

	Impl* raw    = impl.get();
	impl->thread = std::thread([raw] { raw->Run(); });
	std::fprintf(stderr, "direct present: on (window 0x%x %ux%u, modifier 0x%016llx depth 24%s)\n", xid, impl->win_w.load(),
	             impl->win_h.load(), static_cast<unsigned long long>(kModifier), impl->depth30 ? "/30" : "");
	return std::unique_ptr<DirectPresenter>(new DirectPresenter(std::move(impl)));
}

void DirectPresenter::SetCallbacks(DirectPresentCallbacks callbacks) {
	std::lock_guard lock(m_impl->callback_mutex);
	m_impl->callbacks = std::move(callbacks);
}

bool DirectPresenter::Usable() const {
	return !m_impl->disabled.load() && !m_impl->lost.load();
}

void DirectPresenter::Disable(const char* why) {
	if (!m_impl->disabled.exchange(true)) {
		std::fprintf(stderr, "direct present: off for this session (%s), presenting through Vulkan\n", why);
	}
}

uint8_t DirectPresenter::DepthFor(uint64_t video_out_pixel_format) const {
	switch (video_out_pixel_format) {
		case 0x8000000000000000ull: return 24;                          // B8G8R8A8 sRGB: XRGB8888
		case 0x8100000000000000ull: return m_impl->depth30 ? 30 : 0;     // A2R10G10B10: XRGB2101010
		default: return 0;
	}
}

bool DirectPresenter::WindowIs(uint32_t width, uint32_t height) const {
	return m_impl->win_w.load() == width && m_impl->win_h.load() == height;
}

uint32_t DirectPresenter::AddBuffer(const DirectSurface& s) {
	auto& d = *m_impl;
	if (d.lost.load() || xcb_connection_has_error(d.conn) != 0 || s.dmabuf_fd < 0) {
		return 0;
	}
	if (s.depth != 24 && !(s.depth == 30 && d.depth30)) {
		std::fprintf(stderr, "direct present: no pixmap of depth %u\n", s.depth);
		return 0;
	}
	// DCN scans a 64K_R_X surface in whole 64 KiB blocks (128x128 at 4 bytes): the surface starts on a
	// block, and the kernel's AddFB2 wants the pitch in whole blocks and the BO to hold every block row,
	// partial last row included.
	const uint64_t rows = (static_cast<uint64_t>(s.height) + 127u) & ~uint64_t {127};
	const uint64_t need = s.offset + static_cast<uint64_t>(s.stride) * rows;
	const off_t    bo   = ::lseek(s.dmabuf_fd, 0, SEEK_END);
	if (s.width == 0 || s.height == 0 || s.width > 0xffffu || s.height > 0xffffu || s.stride % 512u != 0 ||
	    s.stride < s.width * 4u || s.offset % 0x10000u != 0 || s.offset > 0xffffffffu || (bo > 0 && need > static_cast<uint64_t>(bo))) {
		std::fprintf(stderr,
		             "direct present: buffer %ux%u stride %u offset 0x%llx (needs 0x%llx bytes, dma-buf 0x%llx) cannot be "
		             "scanned out\n",
		             s.width, s.height, s.stride, static_cast<unsigned long long>(s.offset), static_cast<unsigned long long>(need),
		             static_cast<unsigned long long>(bo));
		return 0;
	}
	struct stat st {};
	const uint64_t inode = ::fstat(s.dmabuf_fd, &st) == 0 ? static_cast<uint64_t>(st.st_ino) : 0;
	// xcb closes the descriptors it sends.
	const int fd = ::fcntl(s.dmabuf_fd, F_DUPFD_CLOEXEC, 0);
	if (fd < 0) {
		return 0;
	}
	int32_t                 fds[1] = {fd};
	const xcb_pixmap_t      pixmap = xcb_generate_id(d.conn);
	const xcb_void_cookie_t cookie = xcb_dri3_pixmap_from_buffers_checked(
	    d.conn, pixmap, d.window, 1, static_cast<uint16_t>(s.width), static_cast<uint16_t>(s.height), s.stride,
	    static_cast<uint32_t>(s.offset), 0, 0, 0, 0, 0, 0, s.depth, 32, kModifier, fds);
	if (xcb_generic_error_t* error = xcb_request_check(d.conn, cookie); error != nullptr) {
		std::fprintf(stderr, "direct present: DRI3 pixmap %ux%u depth %u stride %u offset 0x%llx (dma-buf inode %llu) refused: error %u\n",
		             s.width, s.height, s.depth, s.stride, static_cast<unsigned long long>(s.offset),
		             static_cast<unsigned long long>(inode), error->error_code);
		std::free(error);
		return 0;
	}
	if (xcb_connection_has_error(d.conn) != 0) {
		return 0;
	}
	{
		std::lock_guard lock(d.mutex);
		d.pixmaps.insert(pixmap);
	}
	std::fprintf(stderr, "direct present: pixmap 0x%x: %ux%u depth %u stride %u offset 0x%llx, dma-buf inode %llu (%llu KiB)\n",
	             pixmap, s.width, s.height, s.depth, s.stride, static_cast<unsigned long long>(s.offset),
	             static_cast<unsigned long long>(inode), static_cast<unsigned long long>(bo > 0 ? bo / 1024 : 0));
	return pixmap;
}

void DirectPresenter::RemoveBuffer(uint32_t pixmap) {
	auto& d = *m_impl;
	if (pixmap == 0) {
		return;
	}
	std::lock_guard lock(d.mutex);
	if (d.pixmaps.erase(pixmap) == 0 || xcb_connection_has_error(d.conn) != 0) {
		return;
	}
	// Present keeps its own reference to a pixmap it still shows or has queued, so the presents in
	// flight still complete and go idle.
	xcb_free_pixmap(d.conn, pixmap);
	xcb_flush(d.conn);
}

bool DirectPresenter::Present(uint32_t pixmap, uint64_t token, uint64_t target_msc, bool async) {
	auto& d = *m_impl;
	if (d.lost.load()) {
		return false;
	}
	std::lock_guard lock(d.mutex);
	const uint32_t          serial  = d.NextSerialLocked();
	const uint32_t          options = async ? XCB_PRESENT_OPTION_ASYNC : XCB_PRESENT_OPTION_NONE;
	const xcb_void_cookie_t cookie  = xcb_present_pixmap(d.conn, d.window, pixmap, serial, 0, 0, 0, 0, 0, 0, 0, options,
	                                                     target_msc, 0, 0, 0, nullptr);
	Impl::Pending           p;
	p.token           = token;
	p.sent_us         = MonotonicUs();
	p.sequence        = cookie.sequence;
	d.pending[serial] = p;
	d.in_flight.fetch_add(1);
	xcb_flush(d.conn);
	if (xcb_connection_has_error(d.conn) == 0) {
		return true;
	}
	// The event thread reports what was in flight before; this one is the caller's to fall back.
	d.pending.erase(serial);
	d.in_flight.fetch_sub(1);
	return false;
}

uint64_t DirectPresenter::LastMsc() const {
	return m_impl->last_msc.load();
}

uint32_t DirectPresenter::InFlight() const {
	return m_impl->in_flight.load();
}

#else

// Built without xcb-dri3 / xcb-present: every flip goes through the Vulkan presenter.
struct DirectPresenter::Impl {};

std::unique_ptr<DirectPresenter> DirectPresenter::Create(SDL_Window* window) {
	if (window != nullptr && NativeGpu::DirectPresentWanted()) {
		std::fprintf(stderr, "direct present: not built in (no xcb-dri3 / xcb-present), presenting through Vulkan\n");
	}
	return nullptr;
}

void DirectPresenter::SetCallbacks(DirectPresentCallbacks /*callbacks*/) {}

bool DirectPresenter::Usable() const {
	return false;
}

void DirectPresenter::Disable(const char* /*why*/) {}

uint8_t DirectPresenter::DepthFor(uint64_t /*video_out_pixel_format*/) const {
	return 0;
}

bool DirectPresenter::WindowIs(uint32_t /*width*/, uint32_t /*height*/) const {
	return false;
}

uint32_t DirectPresenter::AddBuffer(const DirectSurface& /*surface*/) {
	return 0;
}

void DirectPresenter::RemoveBuffer(uint32_t /*pixmap*/) {}

bool DirectPresenter::Present(uint32_t /*pixmap*/, uint64_t /*token*/, uint64_t /*target_msc*/, bool /*async*/) {
	return false;
}

uint64_t DirectPresenter::LastMsc() const {
	return 0;
}

uint32_t DirectPresenter::InFlight() const {
	return 0;
}

#endif

DirectPresenter::DirectPresenter(std::unique_ptr<Impl> impl): m_impl(std::move(impl)) {}

DirectPresenter::~DirectPresenter() = default;

} // namespace Libs::Graphics
