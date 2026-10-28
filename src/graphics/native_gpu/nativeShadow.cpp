#include "graphics/native_gpu/nativeShadow.h"

#include "graphics/native_gpu/nativePm4.h"
#include "graphics/native_gpu/nativeSubmit.h" // KYTY_GEM_BACKING_SUPPORTED + NativeSubmitter
#include "graphics/native_gpu/shaderPatcher.h"

#include "common/threads.h"
#include "kernel/memory.h"
#include "kernel/syncOnAddress.h"
#include "libs/agc.h"
#include "libs/controller.h"
#include "libs/libs.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <chrono>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <array>
#include <deque>
#include <functional>
#include <ctime>
#include <map>
#include <mutex>
#if defined(__GLIBC__)
#include <malloc.h>
#endif
#include <set>
#include <tuple>
#include <unordered_set>
#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <unordered_map>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <ucontext.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

namespace Libs::Graphics::NativeGpu {

namespace {

double NowSecShadow() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}

// Held pieces (a guest wait the GPU could not take on its own) released since the last flip report:
// how many, how many had to wait for a later pump, and how long those waited in total.
std::atomic<uint64_t> g_held_released {0};
std::atomic<uint64_t> g_held_waited {0};
std::atomic<uint64_t> g_held_wait_us {0};
// By why the piece was first held: 0 cross-ring writer not submitted yet, 1 the write ahead does not
// satisfy the wait (value or function), 2 the write ahead is a CPU (tracker) write, 3 partial cover,
// 4 no write ahead and the label not in memory yet, 5 no native form of the wait.
constexpr int kHoldReasons = 6;
std::atomic<uint64_t> g_hold_reason_n[kHoldReasons] {};
std::atomic<uint64_t> g_hold_reason_us[kHoldReasons] {};

bool EnvOn(const char* name) {
	const char* v = std::getenv(name);
	return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// True when `name` is set to exactly `value`.
bool EnvIs(const char* name, const char* value) {
	const char* v = std::getenv(name);
	return v != nullptr && std::strcmp(v, value) == 0;
}

// The native GPU workers run at the CPU's top clock (Common::Thread::RunAtTopClock): the same DCB of
// My First GT took 1.5 ms or 4 to translate as the worker woke on a fast or a slow core.
// KYTY_NATIVE_GPU_UCLAMP=0 leaves them to the governor.
void RunAtTopClock() {
	if (!Common::Thread::RunAtTopClock("KYTY_NATIVE_GPU_UCLAMP")) {
		static std::atomic<bool> logged {false};
		if (!logged.exchange(true)) {
			std::fprintf(stderr, "native shadow: uclamp_min for the GPU workers refused (errno %d)\n", errno);
		}
	}
}

// KYTY_NATIVE_GPU_CTL=1: a debugging control socket, /tmp/kyty-ctl.sock (KYTY_NATIVE_GPU_CTL_SOCK
// overrides it). One command per line, one reply line each:
//   dump [N]  dump the next N flips (default 8) into KYTY_NATIVE_GPU_DUMP_DIR, render targets included
//   ping      "pong"
//   pad play FILE [ELAPSED]  replay a gamepad recording as player 1 (PlayPadRecording)
std::atomic<uint32_t>& DemandedDumps() {
	static std::atomic<uint32_t> n {0};
	return n;
}

// Control socket "crumbs N": the next N translated buffers (DCBs and ACBs) are written to
// KYTY_NATIVE_GPU_DUMP_DIR as <kind><n>.dcb.bin / .ib.bin, without slowing every frame like
// KYTY_NATIVE_GPU_BREADCRUMBS.
std::atomic<uint32_t>& DemandedCrumbs() {
	static std::atomic<uint32_t> n {0};
	return n;
}

// "pad play FILE [ELAPSED]": replays a padrec.py recording (evdev events of an 8BitDo in DInput mode,
// "t type code value" with t in seconds since the game was launched) as player 1's input. ELAPSED is
// how long the game has already been running, so the recording keeps its timing.
void PlayPadRecording(std::string path, double elapsed) {
	std::thread([path, elapsed] {
		namespace C = Libs::Controller;
		std::FILE* f = std::fopen(path.c_str(), "r");
		if (f == nullptr) {
			std::fprintf(stderr, "pad play: cannot open %s\n", path.c_str());
			return;
		}
		const double start = NowSecShadow() - elapsed;
		char         line[128];
		uint32_t     events = 0;
		while (std::fgets(line, sizeof(line), f) != nullptr) {
			double t = 0;
			int    type = 0, code = 0, value = 0;
			if (line[0] == '#' || std::sscanf(line, "%lf %d %d %d", &t, &type, &code, &value) != 4) {
				continue;
			}
			if (const double wait = t - (NowSecShadow() - start); wait > 0) {
				std::this_thread::sleep_for(std::chrono::duration<double>(wait));
			}
			const int id = C::HOST_INPUT_CONTROLLER_ID;
			if (type == 1) {
				uint32_t button = 0;
				switch (code) {
					case 304: button = C::PAD_BUTTON_CROSS; break;
					case 305: button = C::PAD_BUTTON_CIRCLE; break;
					case 307: button = C::PAD_BUTTON_SQUARE; break;
					case 308: button = C::PAD_BUTTON_TRIANGLE; break;
					case 310: button = C::PAD_BUTTON_L1; break;
					case 311: button = C::PAD_BUTTON_R1; break;
					case 312: C::SetAxis(id, C::Axis::TriggerLeft, value != 0 ? 255 : 0); break;
					case 313: C::SetAxis(id, C::Axis::TriggerRight, value != 0 ? 255 : 0); break;
					case 315: button = C::PAD_BUTTON_OPTIONS; break;
					case 317: button = C::PAD_BUTTON_L3; break;
					case 318: button = C::PAD_BUTTON_R3; break;
					default: break;
				}
				if (button != 0) {
					C::SetButton(id, button, value != 0);
				}
			} else if (type == 3) {
				switch (code) {
					case 0: C::SetAxis(id, C::Axis::LeftX, value); break;
					case 1: C::SetAxis(id, C::Axis::LeftY, value); break;
					case 2: C::SetAxis(id, C::Axis::RightX, value); break;
					case 5: C::SetAxis(id, C::Axis::RightY, value); break;
					case 9: C::SetAxis(id, C::Axis::TriggerRight, value); break;
					case 10: C::SetAxis(id, C::Axis::TriggerLeft, value); break;
					case 16:
						C::SetButton(id, C::PAD_BUTTON_LEFT, value < 0);
						C::SetButton(id, C::PAD_BUTTON_RIGHT, value > 0);
						break;
					case 17:
						C::SetButton(id, C::PAD_BUTTON_UP, value < 0);
						C::SetButton(id, C::PAD_BUTTON_DOWN, value > 0);
						break;
					default: break;
				}
			}
			events++;
		}
		std::fclose(f);
		std::fprintf(stderr, "pad play: %s done, %u events\n", path.c_str(), events);
	}).detach();
}

void StartControlSocket() {
	const char* path = std::getenv("KYTY_NATIVE_GPU_CTL_SOCK");
	path             = path != nullptr && path[0] != '\0' ? path : "/tmp/kyty-ctl.sock";
	const int fd     = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
	struct sockaddr_un a {};
	a.sun_family = AF_UNIX;
	std::snprintf(a.sun_path, sizeof(a.sun_path), "%s", path);
	::unlink(path);
	if (fd < 0 || ::bind(fd, reinterpret_cast<struct sockaddr*>(&a), sizeof(a)) != 0 || ::listen(fd, 4) != 0) {
		std::fprintf(stderr, "native shadow: control socket %s failed: %s\n", path, std::strerror(errno));
		return;
	}
	std::fprintf(stderr, "native shadow: control socket %s (dump [N], crumbs N, ping)\n", path);
	std::thread([fd] {
		for (;;) {
			const int c = ::accept4(fd, nullptr, nullptr, SOCK_CLOEXEC);
			if (c < 0) {
				continue;
			}
			std::string line;
			char        ch = 0;
			for (;;) {
				const ssize_t got = ::recv(c, &ch, 1, 0);
				if (got <= 0) {
					break;
				}
				if (ch != '\n') {
					line.push_back(ch);
					continue;
				}
				char     reply[128];
				unsigned count = 0;
				if (line == "ping") {
					std::snprintf(reply, sizeof(reply), "pong\n");
				} else if (char file[256]; std::sscanf(line.c_str(), "pad play %255s", file) == 1) {
					double elapsed = 0;
					std::sscanf(line.c_str(), "pad play %*s %lf", &elapsed);
					PlayPadRecording(file, elapsed);
					std::snprintf(reply, sizeof(reply), "ok playing %s from %.1f s\n", file, elapsed);
				} else if (std::sscanf(line.c_str(), "crumbs %u", &count) == 1) {
					DemandedCrumbs().store(count);
					std::snprintf(reply, sizeof(reply), "ok writing the next %u buffers\n", count);
				} else if (line == "dump" || std::sscanf(line.c_str(), "dump %u", &count) == 1) {
					count = count != 0 ? count : 8u;
					DemandedDumps().store(count);
					std::snprintf(reply, sizeof(reply), "ok dumping the next %u flips\n", count);
				} else {
					std::snprintf(reply, sizeof(reply), "unknown command; try: dump [N], crumbs N, pad play FILE [ELAPSED], ping\n");
				}
				::send(c, reply, std::strlen(reply), MSG_NOSIGNAL);
				line.clear();
			}
			::close(c);
		}
	}).detach();
}

// GET_LOD_STATS the way the PS5 does it: the GPU's PRT LOD statistics reporter counts texture samples
// per counter bank and writes the report itself. Its registers are privileged, so a root helper
// (tools/bc250-lodhw/kyty-lodhw, the part the PS5 kernel and ME play) programs them; this is the
// client. The helper finds this VM's current VMID by its PASID in the kernel's IH_VMID_n_LUT, so no
// VMID is reserved (a reserved VMID makes gfx and compute jobs of one VM wait for each other before
// a flush, which deadlocks against cross-ring waits). KYTY_LOD_HW_SOCK names the socket
// (default /run/kyty-lodhw.sock); KYTY_LOD_STATS_MODE=usage or all keeps the CPU-side answers.
class LodStatsHw {
public:
	static LodStatsHw& Instance() {
		static LodStatsHw h;
		return h;
	}
	// Connects to the helper; true when reports can go through it. Without KYTY_LOD_HW_SOCK it tries
	// /run/kyty-lodhw.sock, then /tmp/kyty-lodhw.sock (where the BC-250's service listens).
	bool Connect() {
		const char* path = std::getenv("KYTY_LOD_HW_SOCK");
		if (path != nullptr && path[0] != '\0') {
			return Connect(path);
		}
		return Connect("/run/kyty-lodhw.sock") || Connect("/tmp/kyty-lodhw.sock");
	}
	bool Connect(const char* path) {
		const int fd = ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
		if (fd < 0) {
			return false;
		}
		struct sockaddr_un a {};
		a.sun_family = AF_UNIX;
		std::snprintf(a.sun_path, sizeof(a.sun_path), "%s", path);
		if (::connect(fd, reinterpret_cast<struct sockaddr*>(&a), sizeof(a)) != 0) {
			::close(fd);
			return false;
		}
		m_fd = fd;
		return true;
	}
	[[nodiscard]] bool On() const { return m_fd >= 0; }
	void Disconnect() {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (m_fd >= 0) {
			::close(m_fd);
			m_fd = -1;
		}
	}
	// The PASID of the VM on `drm_fd` (its /proc fdinfo "pasid:" line); 0 when not found.
	static uint32_t Pasid(int drm_fd) {
		char path[64];
		std::snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", drm_fd);
		FILE*    f     = std::fopen(path, "re");
		uint32_t pasid = 0;
		char     line[128];
		while (f != nullptr && std::fgets(line, sizeof(line), f) != nullptr) {
			unsigned v = 0;
			if (std::sscanf(line, "pasid: %u", &v) == 1) {
				pasid = v;
			}
		}
		if (f != nullptr) {
			std::fclose(f);
		}
		return pasid;
	}
	void SetPasid(uint32_t pasid) { m_pasid = pasid; }
	// One report of `size` bytes at `va` in whichever VMID serves this VM now, finished when this
	// returns 0. -ESRCH: the VM holds no VMID at the moment (nothing of it on the GPU).
	int Report(uint64_t va, uint32_t size) { return Call(3u | (m_pasid << 8u), size, va, nullptr); }
	// Writes a register the helper allows (op 5); 0 when it reads back `value`.
	int WriteReg(uint32_t reg, uint32_t value) {
		uint32_t back = 0;
		const int r   = Call(5, reg, value, &back);
		return r != 0 ? r : (back == value ? 0 : -EIO);
	}

	// Banked registers (ops 10-13): `sel` is GRBM_GFX_CNTL's layout plus kSel, which selects at all.
	static constexpr uint32_t kSel = 1u << 11u;
	static constexpr uint32_t SelVmid(uint32_t vmid) { return kSel | vmid << 4u; }
	static constexpr uint32_t SelGfxPipe(uint32_t pipe) { return kSel | pipe; }
	static constexpr uint32_t SelHqd(uint32_t me, uint32_t pipe, uint32_t queue) { return kSel | queue << 8u | me << 2u | pipe; }
	int ReadReg(uint32_t reg, uint32_t sel, uint32_t* value) { return Call(12u | sel << 8u, reg, 0, value); }
	// Invalidates the gfx hub TLB of KFD's VMIDs 8-15 (op 14); 0 once acknowledged.
	int InvalidateKfdTlb() { return Call(14u | 0xff00u << 8u, 0, 0, nullptr); }
	// reg = (reg & ~mask) | (value & mask) if the helper allows it; `now` is the register afterwards.
	// The helper keeps the old value and puts it back on RestoreRegs or when this connection closes.
	int WriteRegMasked(uint32_t reg, uint32_t sel, uint32_t value, uint32_t mask, uint32_t* now) {
		return Call(10u | sel << 8u, reg, static_cast<uint64_t>(mask) << 32u | value, now);
	}
	// RCIU CAM `entry` the way the PS5 kernel programs it (phase 0 = register | pipe enables, then mask
	// and value); the helper puts the old entry back when this connection closes. 0 or -errno.
	// GFXOFF off while this connection is open (the helper's request on debugfs amdgpu_gfxoff). 0 or -errno.
	int KeepGfxPowered() {
		uint32_t v = 0;
		return Call(16u, 0, 0, &v);
	}
	int SetRciu(uint32_t entry, uint32_t phase0, uint32_t mask, uint32_t value) {
		uint32_t v = 0;
		return Call(15u | entry << 8u, phase0, mask | static_cast<uint64_t>(value) << 32u, &v);
	}
	// Puts back what WriteRegMasked changed in `reg` (everything with 0xffffffff); how many registers.
	int RestoreRegs(uint32_t reg = 0xffffffffu, uint32_t sel = 0) {
		uint32_t  n = 0;
		const int r = Call(11u | sel << 8u, reg, 0, &n);
		return r != 0 ? r : static_cast<int>(n);
	}
	// The gfx VMID (1-7) the kernel last gave the VM with `pasid`; 0 when none.
	uint32_t GfxVmid(uint32_t pasid) {
		uint32_t v = 0;
		return pasid != 0 && Call(13u | pasid << 8u, 0, 0, &v) == 0 ? v : 0;
	}
	// Applies the helper's local golden table, put back when this connection closes; how many changed.
	int ApplyGolden() {
		uint32_t  n = 0;
		const int r = Call(7, 0xffffffffu, 1, &n);
		return r != 0 ? r : static_cast<int>(n);
	}

private:
	int Call(uint32_t op, uint32_t size, uint64_t va, uint32_t* value) {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (m_fd < 0) {
			return -ENOTCONN;
		}
		struct {
			uint32_t op, size;
			uint64_t va;
		} q {op, size, va};
		struct {
			int32_t  status;
			uint32_t value;
		} r {};
		if (::send(m_fd, &q, sizeof(q), MSG_NOSIGNAL) != static_cast<ssize_t>(sizeof(q)) ||
		    ::recv(m_fd, &r, sizeof(r), 0) != static_cast<ssize_t>(sizeof(r))) {
			::close(m_fd);
			m_fd = -1;
			return -EPIPE;
		}
		if (value != nullptr) {
			*value = r.value;
		}
		return r.status;
	}

	std::mutex m_mutex;
	int        m_fd    = -1;
	uint32_t   m_pasid = 0;
};

// What native primary mode needs to run games correctly is on there by default; setting the switch
// to 0 turns a feature off (and setting it turns it on outside primary mode).
bool PrimaryFeature(const char* name) {
	const char* v = std::getenv(name);
	if (v != nullptr && v[0] != '\0') {
		return v[0] != '0';
	}
	return EnvOn("KYTY_NATIVE_GPU_PRIMARY");
}

// KYTY_LOD_STATS_LOG=N: every Nth scan of the LOD statistics banks (one per GET_LOD_STATS).
void LogLodScan(const LodBankScan& scan) {
	static const unsigned long every = [] {
		const char* v = std::getenv("KYTY_LOD_STATS_LOG");
		return v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
	}();
	static uint64_t scans  = 0;
	static uint32_t logged = 0;
	if (every == 0 || scans++ % every != 0) {
		return;
	}
	auto ranges = [](const std::bitset<256>& b) {
		std::string out;
		for (uint32_t i = 0; i < 256; i++) {
			if (!b.test(i)) {
				continue;
			}
			uint32_t j = i;
			while (j + 1 < 256 && b.test(j + 1)) {
				j++;
			}
			out += (out.empty() ? "" : ",") + std::to_string(i) + (j != i ? "-" + std::to_string(j) : "");
			i = j;
		}
		return out.empty() ? std::string("none") : out;
	};
	std::string warn;
	for (const auto& [w, n]: scan.min_lod_warn) {
		char b[32];
		std::snprintf(b, sizeof(b), " 0x%03x:%u", w, n);
		warn += b;
	}
	std::fprintf(stderr,
	             "native shadow: LOD scan %llu buf 0x%llx: %u T#s, %u counting (ud %u, table %u, table-in-table %u) in %u "
	             "tables, furthest at dword %u; %zu banks: %s; gfx9 layout: %u counting, banks %s; MIN_LOD_WARN:%s\n",
	             static_cast<unsigned long long>(scans - 1), static_cast<unsigned long long>(scan.addr), scan.tsharps,
	             scan.counting, scan.inline_ud, scan.depth1, scan.depth2, scan.tables, scan.max_word, scan.banks.count(),
	             ranges(scan.banks).c_str(), scan.alt_counting, ranges(scan.alt_banks).c_str(), warn.c_str());
	if (logged++ < 3) {
		for (const auto& t: scan.samples) {
			std::fprintf(stderr,
			             "native shadow:   T# %08x %08x %08x %08x | w4 %08x w5 %08x w6 %08x w7 %08x (warn 0x%03x bank %u; "
			             "gfx9: warn 0x%03x bank %u en %u)\n",
			             t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], (t[5] >> 8u) & 0xfffu, t[6] & 0xffu, t[6] & 0xfffu,
			             (t[6] >> 12u) & 0xffu, (t[6] >> 20u) & 1u);
		}
	}
}

// The host side native labels signal (ShadowSetHost).
struct Host {
	std::function<void(int, uint32_t)>           interrupt;
	std::function<uint64_t()>                    reference_clock;
	std::function<void(int, int, int, int64_t)> flip; // ShadowSetFlipHost
};
Host& HostCallbacks() {
	static Host host;
	return host;
}

// Shadow-mode state, built lazily on the first DCB so it sees the final env and the created GEM
// backing. One instance for the process; submit_dcb is driven from the guest graphics thread, but
// a mutex keeps it safe if that ever changes.
class Shadow {
public:
	static Shadow& Instance() {
		static Shadow s;
		return s;
	}

	// True once shadow submission is set up and has not disabled itself.
	// Lock-free: Kyty's command processor asks for every draw and dispatch, while the pipeline's
	// worker may hold m_mutex for a whole submission.
	bool Active() const { return m_active.load(std::memory_order_acquire); }

	// The draws and dispatches translated since the last call (KYTY_NATIVE_GPU_CTL), for the flip dump.
	struct FrameDraw {
		bool       acb;
		uint32_t   buffer; // submission number within the frame
		DrawRecord draw;
	};
	std::vector<FrameDraw> TakeFrameDraws() {
		std::lock_guard<std::mutex> lock(m_mutex);
		std::vector<FrameDraw>      out;
		out.swap(m_frame_draws);
		m_frame_buffers = 0;
		return out;
	}

	// The color targets the DCBs bound since the last call, for the flip dump.
	std::vector<RenderTargetInfo> TakeFrameTargets() {
		std::lock_guard<std::mutex>   lock(m_mutex);
		std::vector<RenderTargetInfo> out;
		for (const auto& [addr, t]: m_frame_targets) {
			out.push_back(t);
		}
		m_frame_targets.clear();
		return out;
	}

	// Reads guest memory through the backing (clipped to the view holding `addr`); 0 when unavailable.
	uint64_t ReadGuest(uint64_t addr, void* dst, uint64_t size) {
#if KYTY_GEM_BACKING_SUPPORTED
		return m_gem != nullptr ? m_gem->ReadGuest(addr, dst, size) : 0;
#else
		(void)addr;
		(void)dst;
		(void)size;
		return 0;
#endif
	}

	// `acb`: an asynchronous compute buffer. ACBs go through their own translator (their own shadow
	// of the compute registers) and run on the same gfx ring as the DCBs, in guest submission order.
	// With `out` (native labels) the buffer is translated and made resident, and the result goes to
	// the caller, who schedules its segments; false means it is not run natively (refused, or the
	// native path is off) and Kyty has to run it.
	// `source`: where `dcb`'s dwords sit in guest memory when it is not the guest's buffer itself (a chunk of
	// one LabelPipeline flattened), for direct runs.
	bool Submit(std::span<const uint32_t> dcb, bool acb = false, Pm4TranslateResult* out = nullptr,
	            const std::vector<SourcePiece>* source = nullptr) {
		const double                t_enter = NowSecShadow();
		std::lock_guard<std::mutex> lock(m_mutex);
		const double                t_locked = NowSecShadow();
		m_wait_lock_s += t_locked - t_enter;
		struct BusyScope {
			double& total;
			double& trace_total;
			double  start;
			~BusyScope() {
				const double d = NowSecShadow() - start;
				total += d;
				trace_total += d;
			}
		} busy {m_busy_s, m_trace_busy_s, t_locked};
		if (!m_initialized) {
			Init();
			m_initialized = true;
		}
		if (!m_enabled || m_disabled || (acb && m_acb_mode == AcbMode::Off)) {
			return false;
		}

		SyncCollectRefs();
		const uint64_t     frame = acb ? m_acbs++ : m_frames++;
		const char* const  kind  = acb ? "acb" : "frame";
		Pm4Translator&     tr    = acb ? m_acb_translator : m_translator;
		if (!acb) {
			LoadAgcStartState();
		}
		SetStage(Stage::Translate, frame);
		const double       t_translate = NowSecShadow();
		// Command buffers written in chunks (INDIRECT_BUFFER chains, REWIND) are translated as the CP
		// fetches them.
		std::vector<uint32_t>    flat;
		std::string              flat_why;
		std::vector<SourcePiece> pieces;
		const bool               direct = m_direct_min_dw != 0 && !acb;
		if (!FlattenIndirectBuffers(dcb, &flat, &flat_why, direct ? &pieces : nullptr) && !flat_why.empty() &&
		    m_flatten_failed++ < 8) {
			std::fprintf(stderr, "native shadow: %s %llu not inlined: %s\n", kind, static_cast<unsigned long long>(frame),
			             flat_why.c_str());
		}
		if (!flat.empty()) {
			dcb = flat;
			if (m_flattened++ == 0) {
				std::fprintf(stderr, "native shadow: INDIRECT_BUFFER chains and calls inlined (first in %s %llu)\n", kind,
				             static_cast<unsigned long long>(frame));
			}
		}
		if (direct) {
			if (source != nullptr) {
				pieces = *source;
			} else if (flat.empty()) {
				pieces = {SourcePiece {0, static_cast<uint32_t>(dcb.size()), reinterpret_cast<uint64_t>(dcb.data())}};
			}
#if KYTY_GEM_BACKING_SUPPORTED
			// Only stretches in memory the GPU VM maps (guest memory in the backing); a buffer elsewhere
			// is copied as before.
			std::erase_if(pieces, [&](const SourcePiece& piece) {
				return m_gem == nullptr || !m_gem->IsGpuMapped(piece.va) || !m_gem->IsGpuMapped(piece.va + 4ull * piece.len - 4);
			});
#else
			pieces.clear();
#endif
			tr.SetSource(pieces);
		}
		// KYTY_NATIVE_GPU_TRANSLATE_BENCH=1 (or =N, the least DCB size in dwords; 1 means 400000): every
		// 120th DCB that large is translated again from a copy of the translator in variants, to see what
		// each part of the translation costs on the same buffer.
		static const size_t bench_min = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_TRANSLATE_BENCH");
			const long  n = v == nullptr ? 0 : std::atol(v);
			return n <= 0 ? 0 : (n == 1 ? 400000 : static_cast<size_t>(n));
		}();
		std::unique_ptr<Pm4Translator> bench_snap;
		if (bench_min != 0 && !acb && out != nullptr && dcb.size() >= bench_min && (m_bench_count++ % 120) == 0) {
			bench_snap = tr.Clone();
		}
		Pm4TranslateResult result      = tr.Translate(dcb);
		if (bench_snap != nullptr) {
			auto run = [&](bool refs, bool direct_runs) {
				double best = 1e9;
				for (int k = 0; k < 3; k++) {
					auto         t  = bench_snap->Clone();
					const double t0 = NowSecShadow();
					t->SetCollectRefs(refs);
					if (direct_runs) {
						t->SetSource(pieces);
					} else {
						t->SetDirectRuns(0);
					}
					(void)t->Translate(dcb);
					best = std::min(best, (NowSecShadow() - t0) * 1e3);
				}
				return best;
			};
			const double t0 = NowSecShadow();
			(void)bench_snap->Clone();
			const double clone_ms = (NowSecShadow() - t0) * 1e3;
			std::fprintf(stderr,
			             "native bench: DCB of %zu dwords: translate %.2f ms, without references %.2f, without direct runs %.2f "
			             "(a copy of the translator %.2f ms, included)\n",
			             dcb.size(), run(m_collect_refs, true), run(false, true), run(m_collect_refs, false), clone_ms);
		}
		const double       translate_ms = (NowSecShadow() - t_translate) * 1e3;
		for (uint32_t i = 0; i < result.stats.event_type.size(); i++) {
			m_event_type_hist[i] += result.stats.event_type[i];
		}
		m_verbatim_dw += result.stats.verbatim_dw;
		m_total_dw += result.stats.total_dw;
		m_verbatim_runs += result.stats.verbatim_runs;
		m_labels_native += result.stats.labels_native;
		m_write_data_dw += result.stats.write_data_dw;
		m_direct_ib2 += result.stats.direct_ib2;
		m_direct_dw += result.stats.direct_dw;
		for (size_t i = 0; i < 256; i++) {
			m_changed_ops[acb ? 1 : 0][i] += result.stats.changed_ops[i];
			m_ops[acb ? 1 : 0][i] += result.stats.ops[i];
		}
		for (size_t i = 0; i < m_run_hist.size(); i++) {
			m_run_hist[i] += result.stats.run_hist[i];
		}
		{
			CutStats& c = m_cut_stats[acb ? 1 : 0];
			c.total_dw += result.stats.total_dw;
			m_trace_dw[acb ? 1 : 0] += result.stats.total_dw;
			c.cut_runs += result.stats.cut_runs;
			for (size_t i = 0; i < c.cut_ops.size(); i++) {
				c.cut_ops[i] += result.stats.cut_ops[i];
				c.inserted_ops[i] += result.stats.inserted_ops[i];
			}
		}
		for (uint32_t i = 0; i < result.stats.event_index.size(); i++) {
			m_event_index_hist[i] += result.stats.event_index[i];
		}
		for (const auto& h: result.find_hits) {
			const auto key = std::make_tuple(h.shader, h.user_reg, h.depth, h.word);
			if (m_find_seen.insert(key).second && m_find_seen.size() <= 64) {
				std::fprintf(stderr,
				             "native shadow: FIND %s %llu %s %u: shader 0x%llx (es 0x%llx) user reg 0x%03x depth %u table 0x%llx "
				             "word %u -> %s 0x%llx\n",
				             kind, static_cast<unsigned long long>(frame), h.dispatch ? "dispatch" : "draw", h.index,
				             static_cast<unsigned long long>(h.shader), static_cast<unsigned long long>(h.es), h.user_reg,
				             h.depth, static_cast<unsigned long long>(h.table), h.word, h.image ? "T#" : "ptr",
				             static_cast<unsigned long long>(h.target));
			}
		}
		if (!result.lod_scans.empty()) {
			// Whichever buffer carried the packet, the report covers every DCB and ACB translated
			// since the previous one. The pipeline thread publishes it here; the command processor
			// (the completion thread when submissions are pipelined, otherwise this same thread,
			// after Submit returns) takes it by buffer address and writes the usage records.
			Pm4Translator& other = acb ? m_translator : m_acb_translator;
			for (auto& scan: result.lod_scans) {
				scan.Merge(other.TakeLodScan());
				LogLodScan(scan);
				LodStatsReports().Push(scan.addr, scan.banks);
			}
		}
		// KYTY_NATIVE_GPU_LOG_WAITS=N: report the waits of the first N buffers that were not met when
		// translated, i.e. work the native path runs before what the guest made it wait for.
		static const unsigned long log_waits = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_LOG_WAITS");
			return v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
		}();
		if (log_waits != 0 && m_waits_logged < log_waits && !result.unmet_waits.empty()) {
			m_waits_logged++;
			for (const auto& w: result.unmet_waits) {
				std::fprintf(stderr,
				             "native shadow: WAIT %s %llu dw %u (after %u draws, %u dispatches): [0x%llx] = 0x%llx, "
				             "needs func %u ref 0x%llx\n",
				             kind, static_cast<unsigned long long>(frame), w.dw, w.draws, w.dispatches,
				             static_cast<unsigned long long>(w.addr), static_cast<unsigned long long>(w.value), w.function,
				             static_cast<unsigned long long>(w.ref));
			}
		}
		// KYTY_NATIVE_GPU_LOG_DATA_WRITES=N: report the first N WRITE_DATA carrying more than one dword,
		// which Kyty performs only after the whole batch (data, not labels).
		static const unsigned long log_data_writes = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_LOG_DATA_WRITES");
			return v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
		}();
		for (const auto& [a, n]: result.data_writes) {
			if (m_data_writes_logged >= log_data_writes) {
				break;
			}
			m_data_writes_logged++;
			std::fprintf(stderr, "native shadow: DATA WRITE %s %llu: 0x%llx, %u dwords\n", kind,
			             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(a), n);
		}
		if (acb && SkipAcb(frame, result)) {
			return false;
		}

		// A wait the GPU runs past (guest waits become barriers) is only safe when the label's writer
		// was submitted before it. Report writers that come later: their work runs after the waiter here.
		if (m_check_order) {
			auto satisfies = [](const Pm4TranslateResult::UnmetWait& w, uint64_t v) {
				switch (w.function) {
					case 1: return v < w.ref;
					case 2: return v <= w.ref;
					case 4: return v != w.ref;
					case 5: return v >= w.ref;
					case 6: return v > w.ref;
					default: return v == w.ref;
				}
			};
			for (const auto& [a, v]: result.label_writes) {
				if (auto it = m_unmet_by_addr.find(a); it != m_unmet_by_addr.end() && satisfies(it->second.wait, v)) {
					m_late_writers++;
					if (m_late_writers <= 24) {
						std::fprintf(stderr,
						             "native shadow: ORDER label 0x%llx (needs 0x%llx) waited for by %s %llu is written by %s %llu, "
						             "submitted later\n",
						             static_cast<unsigned long long>(a), static_cast<unsigned long long>(it->second.wait.ref),
						             it->second.acb ? "acb" : "frame", static_cast<unsigned long long>(it->second.buffer), kind,
						             static_cast<unsigned long long>(frame));
					}
					m_unmet_by_addr.erase(it);
				}
				m_last_label[a] = v;
			}
			for (const auto& w: result.unmet_waits) {
				// Already satisfied by a write in an earlier buffer, which runs first: no hazard.
				if (auto it = m_last_label.find(w.addr); it != m_last_label.end() && satisfies(w, it->second)) {
					continue;
				}
				if (m_unmet_by_addr.size() < 4096) {
					m_unmet_by_addr[w.addr] = {w, acb, frame};
				}
			}
		}
		for (const auto& [off, bytes]: result.gds_dma) {
			if (m_gds_seen.insert((static_cast<uint64_t>(acb) << 63u) | (static_cast<uint64_t>(off) << 20u) | bytes).second &&
			    m_gds_seen.size() <= 64) {
				std::fprintf(stderr, "native shadow: GDS DMA %s %s offset 0x%x, %u bytes\n", acb ? "acb" : "dcb",
				             (off & 0x80000000u) != 0 ? "write" : "read", off & 0x7fffffffu, bytes);
			}
		}
		(acb ? m_gds_dma_acb : m_gds_dma_dcb)[0] += result.stats.dma_gds_dst;
		(acb ? m_gds_dma_acb : m_gds_dma_dcb)[1] += result.stats.dma_gds_src;
		DetectShaders(result.shader_addrs);
		// KYTY_NATIVE_GPU_INSPECT_AFTER_S: only inspect draws this many seconds after start (past the title).
		static const double inspect_after = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_INSPECT_AFTER_S");
			return v != nullptr ? std::strtod(v, nullptr) : 0.0;
		}();
		static const double start_s = NowSecShadow();
		for (const auto& snap: result.inspected) {
			if (NowSecShadow() - start_s < inspect_after) {
				break;
			}
			InspectDraw(frame, snap);
		}
		// A shader that uses GDS hung the board in ASTRO BOT even with a GDS partition; until that is
		// understood such DCBs stay on Kyty unless KYTY_NATIVE_GPU_GDS_SHADERS=1.
		// Skip mode learns GDS shaders as they appear: the DCB that introduces one was translated
		// before the scan, so it goes to Kyty whole and the translator drops that shader's draws from
		// the next DCB on.
		if (result.ok && !acb && m_skip_gds_shaders && m_gds_shaders.size() != m_skipped_shader_count) {
			m_skipped_shader_count = m_gds_shaders.size();
			std::set<uint64_t> skip = m_gds_shaders;
			skip.insert(m_skip_ps.begin(), m_skip_ps.end());
			m_translator.SetSkippedPixelShaders(skip);
			m_refused++;
			return false;
		}
		if (result.ok && !m_allow_gds_shaders && !m_skip_gds_shaders) {
			for (uint64_t va: result.shader_addrs) {
				if (m_gds_shaders.count(va) != 0) {
					if (m_gds_refused++ < 4) {
						std::fprintf(stderr, "native shadow: frame %llu left to Kyty: shader 0x%llx uses GDS\n",
						             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(va));
					}
					m_refused++;
					return false;
				}
			}
		}
		if (!result.draw_log.empty()) {
			const uint32_t buffer = m_frame_buffers++;
			for (const auto& d: result.draw_log) {
				m_frame_draws.push_back(FrameDraw {acb, buffer, d});
			}
		}
		if (m_collect_targets) {
			for (const auto& t: result.render_targets) {
				auto& acc = m_frame_targets[t.addr];
				const uint32_t draws = acc.draws + t.draws;
				acc       = t;
				acc.draws = draws;
			}
		}
		for (uint32_t off: result.skipped_uconfig) {
			if (m_skipped_uconfig.insert(off).second) {
				std::fprintf(stderr, "native shadow: dropping non-whitelisted UCONFIG 0x%05x (offset 0x%03x)\n",
				             0x30000u + off * 4u, off);
			}
		}

		if (!result.ok) {
			for (uint32_t op: result.unknown_ops) {
				if (m_unknown_ops.insert(op).second) {
					std::fprintf(stderr, "native shadow: untranslatable packet op 0x%02x (first in %s %llu)\n", op, kind,
					             static_cast<unsigned long long>(frame));
				}
			}
			if (m_refused < 8) {
				std::fprintf(stderr,
				             "native shadow: %s %llu refused, unknown packet op 0x%02x at dw %u "
				             "(%u packets translated)%s",
				             kind, static_cast<unsigned long long>(frame), result.unknown_op,
				             result.unknown_dw, result.stats.packets, result.unknown_detail.empty() ? "\n" : ", body");
				for (uint32_t v: result.unknown_detail) {
					std::fprintf(stderr, " 0x%08x", v);
				}
				if (!result.unknown_detail.empty()) {
					std::fprintf(stderr, "\n");
				}
			}
			m_refused++;
			return false;
		}

#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter == nullptr) {
			return false;
		}
		// Guest direct memory is resident on the GPU only where needed (gemBacking.h): before the IB
		// runs, promote what the CPU has written and what this DCB references.
		const double t_residency = NowSecShadow();
		SetStage(Stage::Residency, frame);
		MakeResident(result);
		m_residency_ms_total += (NowSecShadow() - t_residency) * 1e3;
		SetStage(Stage::Idle, frame);
		if (out != nullptr) {
			// Only packet forms that have run on the board before go to the GPU; anything else and the
			// buffer is Kyty's.
			const NativeIbKind ib_kind = acb && m_acb_rings ? NativeIbKind::Compute : NativeIbKind::Gfx;
			if (uint32_t bad = 0; m_validate != 0 && !ValidateNativeIb(result.ib, NativeLodStats(), &bad, ib_kind)) {
				if (m_validate_refused++ < 16) {
					std::fprintf(stderr,
					             "native shadow: %s %llu left to Kyty by the packet check: op 0x%02x (header 0x%08x) at dw %u\n",
					             kind, static_cast<unsigned long long>(frame), (result.ib[bad] >> 8u) & 0xffu, result.ib[bad], bad);
				}
				m_refused++;
				return false;
			}
			if (!m_breadcrumbs.empty()) {
				WriteBreadcrumb(acb ? "acb" : "", frame, dcb, result, result.ib);
			}
			WriteDemandedCrumb(acb, frame, dcb, result.ib);
			if (m_trace_gpu) {
				NoteLabelWriters(acb, result);
			}
			m_translate_ms_total += translate_ms;
			m_labels_stats[0] += result.stats.labels_native;
			m_labels_stats[1] += result.stats.labels_cpu;
			m_labels_stats[2] += result.stats.interrupts;
			m_irq_gpu += result.stats.interrupts_gpu;
			m_labels_stats[3] += result.stats.waits_native;
			m_labels_stats[4] += result.stats.waits_split;
			m_labels_stats[5] += result.stats.wait_barriers;
			m_labels_stats[6] += result.stats.timestamps;
			m_labels_stats[7] += result.stats.timestamps_native;
			// The first labels the GPU may not write, and why (diagnostics for the tracker's share).
			for (const auto& seg: result.segments) {
				for (const auto& p: seg.points) {
					if (p.kind == TrackerPoint::Kind::CpuWrite && !p.mirror && m_cpu_labels_logged < 8 &&
					    m_cpu_label_pages.insert(p.addr >> 12u).second) {
						m_cpu_labels_logged++;
						std::fprintf(stderr, "native shadow: label 0x%llx written by the tracker: %s\n", static_cast<unsigned long long>(p.addr),
						             m_gem->IsBacked(p.addr) ? "direct memory not mapped for the GPU"
						                                     : (m_gem->IsGpuMapped(p.addr) ? "read-only for the GPU" : "not GPU memory"));
					}
				}
			}
			*out = std::move(result);
			return true;
		}

		// Every GEM buffer the GPU may touch must be resident in the submission: the kernel only
		// validates and fences buffers that are listed, so a userptr one left out could have its pages
		// migrated under a live GPU access. CreatedHandles() is the resident blocks, the dummy and the
		// readable external slices (CPU stacks are never bound, so they are not here).
		std::vector<uint32_t> handles = m_gem->CreatedHandles();

		// KYTY_NATIVE_GPU_STATS=1: bracket the IB with pipeline-statistics samples written into the
		// submitter's scratch area, to see how far each frame gets (vertices, primitives, pixel
		// shader invocations) when the output stays empty.
		// Every IB is bracketed with two GPU timestamps (COPY_DATA from the GPU clock) in the
		// scratch area, so the log can split submit->fence time into the CS ioctl, the GPU work and
		// the rest. With KYTY_NATIVE_GPU_STATS=1 pipeline-statistics samples are added as well.
		const uint64_t        sva = m_submitter->ScratchVa();
		std::vector<uint32_t> wrapped;
		wrapped.reserve(result.ib.size() + 32);
		EmitTimestamp(wrapped, sva + kTimestampOffset);
		if (m_stats) {
			for (uint32_t i = 0; i < 2 * kStatSlots * 8; i++) {
				m_submitter->ScratchCpu()[i] = 0;
			}
			wrapped.push_back(StatPkt3(kItEventWrite, 1));
			wrapped.push_back(kEventPipelineStatStart);
			EmitSample(wrapped, sva);
		}
		wrapped.insert(wrapped.end(), result.ib.begin(), result.ib.end());
		if (m_stats) {
			EmitSample(wrapped, sva + kStatSlots * 8);
		}
		EmitTimestamp(wrapped, sva + kTimestampOffset + 8);
		while ((wrapped.size() & 7u) != 0) {
			wrapped.push_back(0xffff1000u);
		}
		if (!m_breadcrumbs.empty()) {
			WriteBreadcrumb(acb ? "acb" : "", frame, dcb, result, wrapped);
		}
		WriteDemandedCrumb(acb, frame, dcb, wrapped);
		// Pipelined: the IB joins the batch the pipeline submits with one CS once it has taken every
		// queued buffer (FlushBatch). One BO-list validation then covers them all.
		if (m_pipelined) {
			// KYTY_NATIVE_GPU_ACB_DEFER=1: an ACB that waits for labels no submitted DCB has written yet
			// goes into the batch after the DCB that writes them: on the ring its waits are barriers, so
			// in guest submission order it would read what the previous frame left there. Kyty's compute
			// queue only completes the ACB after those labels, i.e. after the writer ran, so the ACB may
			// wait across batches. ACBs keep their order among themselves.
			if (m_acb_defer) {
				ReleaseDeferredAcbsLocked();
			}
			if (acb && m_acb_defer && (!m_deferred_acbs.empty() || !result.unmet_waits.empty())) {
				DeferredAcb d;
				d.ib = std::move(wrapped);
				for (const auto& w: result.unmet_waits) {
					// A DCB already submitted writes it: that one runs first on the ring.
					if (w.addr != 0 && m_inflight_labels.count(w.addr) == 0) {
						d.needs.push_back(w);
					}
				}
				if (!d.needs.empty() || !m_deferred_acbs.empty()) {
					m_deferred_acbs.push_back(std::move(d));
					m_acb_deferred++;
					return true;
				}
				wrapped = std::move(d.ib);
			}
			AppendToBatchLocked(wrapped);
			if (!acb && m_acb_defer) {
				for (const auto& [a, v]: result.label_writes) {
					if (a != 0) {
						m_inflight_labels[a] = static_cast<uint32_t>(v);
					}
				}
				for (auto& d: m_deferred_acbs) {
					std::erase_if(d.needs, [&](const Pm4TranslateResult::UnmetWait& w) {
						return std::any_of(result.label_writes.begin(), result.label_writes.end(),
						                   [&](const auto& lw) { return lw.first == w.addr; });
					});
					d.dcbs_since++;
				}
				ReleaseDeferredAcbsLocked();
			}
			return true;
		}
		std::span<const uint32_t> ib = wrapped;
		SetStage(Stage::Gpu, frame);
		NativeSubmitter::Result r = m_submitter->Submit(ib, handles, m_timeout_sec);
		// ENOENT: a buffer in the list was closed meanwhile (another guest thread unmapped an external
		// range between CreatedHandles and the ioctl). Take a fresh list and try again.
		for (int retry = 0; !r.submitted && r.err == -ENOENT && retry < 3; retry++) {
			handles = m_gem->CreatedHandles();
			r       = m_submitter->Submit(ib, handles, m_timeout_sec);
		}
		SetStage(Stage::Idle, frame);

		const bool detail = frame < 3 || (frame % 120 == 0);
		if (!r.submitted) {
			std::fprintf(stderr,
			             "native shadow: %s %llu submit failed: %s (after %d retries, %zu BOs)\n", kind,
			             static_cast<unsigned long long>(frame), std::strerror(-r.err), r.retries,
			             handles.size());
			Disable("submit error");
			return false;
		}
		m_last_seq = r.seq;
		if (r.timed_out || r.reset || !r.completed) {
			std::fprintf(stderr,
			             "native shadow: %s %llu %s after %.1f ms (ib %zu dw, %u draws); "
			             "disabling native path\n",
			             kind, static_cast<unsigned long long>(frame),
			             r.timed_out ? "TIMED OUT" : (r.reset ? "context RESET" : "did not complete"),
			             r.seconds * 1e3, result.ib.size(), result.stats.draws);
			Disable("timeout or reset");
			return false;
		}

		if (acb) {
			m_acbs_completed++;
			if (m_acbs_completed % 30 == 0 && WaitIdle() && m_gem->DummyWritten() && m_acb_dummy_writes++ < 8) {
				std::fprintf(stderr,
				             "native shadow: acb %llu: the GPU wrote to guest memory that was not resident "
				             "(a reference the translator missed)\n",
				             static_cast<unsigned long long>(frame));
			}
			if (m_acbs_completed <= 3 || m_acbs_completed % 600 == 0) {
				std::fprintf(stderr, "native shadow: acb %llu ok (%llu done), %.2f ms, ib %zu dw, %u dispatches\n",
				             static_cast<unsigned long long>(frame), static_cast<unsigned long long>(m_acbs_completed),
				             r.seconds * 1e3, result.ib.size(), result.stats.dispatches);
			}
			return true;
		}
		m_completed++;
		if (m_completed % 30 == 0 && WaitIdle() && m_gem->DummyWritten()) {
			m_dummy_writes++;
			if (m_dummy_writes <= 8) {
				std::fprintf(stderr,
				             "native shadow: frame %llu: the GPU wrote to guest memory that was not "
				             "resident (a reference the translator missed)\n",
				             static_cast<unsigned long long>(frame));
			}
		}
		m_gpu_ms_total += r.seconds * 1e3;
		const double gpu_ms = GpuMs();
		m_cs_ms_total += r.cs_seconds * 1e3;
		m_gpu_only_ms_total += gpu_ms;
		m_translate_ms_total += translate_ms;
		if (m_stats && result.stats.draws > 0 && m_stats_logged < 12) {
			LogStats(frame, result.stats.draws);
			m_stats_logged++;
		}
		if (m_dump_regs_frame >= 0 && frame == static_cast<uint64_t>(m_dump_regs_frame)) {
			DumpRegs(frame);
		}
		if (detail) {
			std::fprintf(stderr,
			             "native shadow: frame %llu ok, gpu(submit->fence) %.2f ms, %d cs-retries, "
			             "ib %zu dw (preamble cx/sh/uc %u/%u/%u), %u draws, %u/%u/%u loads, %u dispatch, "
			             "%u acquire, %u uc-skipped, %u/%u waits unmet/dropped\n",
			             static_cast<unsigned long long>(frame), r.seconds * 1e3, r.retries,
			             result.ib.size(), result.stats.preamble_cx, result.stats.preamble_sh,
			             result.stats.preamble_uc, result.stats.draws, result.stats.loaded_cx_regs,
			             result.stats.loaded_sh_regs, result.stats.loaded_uc_regs,
			             result.stats.dispatches, result.stats.acquire_mem, result.stats.uconfig_skipped,
			             result.stats.waits_unmet, result.stats.waits_dropped);
		}
		if (m_completed > 0 && m_completed % 120 == 0) {
			const double n = static_cast<double>(m_completed);
			std::fprintf(stderr,
			             "native shadow: avg per frame: translate %.2f ms, residency %.2f ms, CS ioctl %.2f ms, "
			             "GPU %.2f ms, submit->fence %.2f ms; %llu MiB of guest memory resident, %u dummy "
			             "writes seen\n",
			             m_translate_ms_total / n, m_residency_ms_total / n, m_cs_ms_total / n,
			             m_gpu_only_ms_total / n, m_gpu_ms_total / n,
			             static_cast<unsigned long long>(m_gem->ResidentBytes() >> 20u), m_dummy_writes);
			std::fprintf(stderr,
			             "native shadow: blocks made resident by: cpu writes %llu, shaders %llu, user-data pointers %llu, "
			             "direct refs %llu, table buffers %llu, table images %llu\n",
			             static_cast<unsigned long long>(m_promoted_by[0]), static_cast<unsigned long long>(m_promoted_by[1]),
			             static_cast<unsigned long long>(m_promoted_by[2]), static_cast<unsigned long long>(m_promoted_by[3]),
			             static_cast<unsigned long long>(m_promoted_by[4]), static_cast<unsigned long long>(m_promoted_by[5]));
			if (m_stats) {
				const auto evname = [](uint32_t t) -> const char* {
					switch (t) {
						case 0x04: return "PS_PARTIAL_FLUSH";
						case 0x14: return "CACHE_FLUSH_AND_INV_TS";
						case 0x15: return "BOTTOM_OF_PIPE_TS";
						case 0x2c: return "FLUSH_AND_INV_DB_META";
						case 0x2e: return "FLUSH_AND_INV_CB_META";
						default: return "";
					}
				};
				std::fprintf(stderr, "native shadow: EVENT_WRITE/RELEASE_MEM EVENT_TYPE histogram:");
				for (uint32_t t = 0; t < 64; t++) {
					if (m_event_type_hist[t] != 0) {
						std::fprintf(stderr, " 0x%02x%s%s=%llu", t, evname(t)[0] != '\0' ? ":" : "", evname(t),
						             static_cast<unsigned long long>(m_event_type_hist[t]));
					}
				}
				std::fprintf(stderr, "\nnative shadow: EVENT_INDEX histogram:");
				for (uint32_t i = 0; i < 16; i++) {
					if (m_event_index_hist[i] != 0) {
						std::fprintf(stderr, " %u=%llu", i, static_cast<unsigned long long>(m_event_index_hist[i]));
					}
				}
				std::fprintf(stderr, "\n");
			}
			std::fprintf(stderr,
			             "native shadow: %llu frames completed, avg gpu(submit->fence) %.2f ms, "
			             "%u shaders scanned, %u v_dot2c_f32_f16 sites\n",
			             static_cast<unsigned long long>(m_completed),
			             m_gpu_ms_total / static_cast<double>(m_completed), m_shaders_scanned,
			             m_dot2c_sites);
		}
#else
		(void)result;
		(void)out;
#endif
		return true;
	}

	void AddDisplayRange(uint64_t addr, uint64_t size) {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_translator.AddDisplayRange(addr, size)) {
			return;
		}
		std::fprintf(stderr, "native shadow: video-out buffer 0x%llx (%llu MiB) drawn without DCC\n", static_cast<unsigned long long>(addr),
		             static_cast<unsigned long long>(size >> 20u));
	}

	// Display buffers are written by the GPU (often by compute through a descriptor the translator
	// cannot see), so whatever is flipped is made resident; until then such writes land in the dummy.
	void KeepResident(uint64_t addr, uint64_t size) {
#if KYTY_GEM_BACKING_SUPPORTED
		std::lock_guard<std::mutex> lock(m_mutex);
		if (m_gem == nullptr || !m_flipped.insert(addr).second) {
			return;
		}
		const uint32_t promoted = m_gem->PromoteGuestRange(addr, size);
		// The presenter imports the buffer's dma-bufs: always-valid blocks under it become ordinary ones.
		const uint32_t exported = m_gem->ExportGuestRange(addr, size, [this] { WaitIdle(); });
		std::fprintf(stderr, "native shadow: display buffer 0x%llx (%llu MiB) kept resident, %u block(s) promoted, %u made exportable\n",
		             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size >> 20u), promoted, exported);
#else
		(void)addr;
		(void)size;
#endif
	}

	// The dummy check needs an idle GPU; in async mode that means waiting for what is in flight.
	bool WaitIdle() {
#if KYTY_GEM_BACKING_SUPPORTED
		return !m_submitter->Async() || m_submitter->WaitIdle(m_timeout_sec);
#else
		return true;
#endif
	}

	// Per-flip timing (KYTY_NATIVE_GPU_FLIP_STATS=1): native time spent in Submit and waiting for its
	// lock between flips, with the presenter's share passed in.
	uint64_t ResidentMiB() const {
#if KYTY_GEM_BACKING_SUPPORTED
		return m_gem != nullptr ? m_gem->ResidentBytes() >> 20u : 0;
#else
		return 0;
#endif
	}

	// Lock-free: the native labels tracker calls this, and m_mutex is held for whole translations.
	void AddGpuBusy(double seconds) {
		m_gpu_busy_s.fetch_add(seconds, std::memory_order_relaxed);
		m_trace_gpu_busy_s.fetch_add(seconds, std::memory_order_relaxed);
	}

	// One pipelined buffer finished: `queued` waiting for the worker, `latency` from the guest's
	// submit to its continuation being done, `then` the continuation (Kyty's command processor).
	void AddChain(double queued, double latency, double then) {
		// Its own lock: the native labels tracker calls this, and m_mutex is held for whole translations.
		std::lock_guard<std::mutex> lock(m_chain_mutex);
		m_chain_n++;
		m_chain_queued_s += queued;
		m_chain_latency_s += latency;
		m_chain_then_s += then;
		m_chain_max_s = std::max(m_chain_max_s, latency);
	}

	// KYTY_NATIVE_GPU_FLIP_TRACE=gpu: when each buffer writing a label was translated, and when each DCB
	// whose GPU waits are timed was, to tell a GPU wait for compute work from one for our own latency.
	void NoteLabelWriters(bool acb, const Pm4TranslateResult& result) {
		const double now = NowSecShadow();
		if (m_label_writer.size() > 262144) {
			m_label_writer.clear();
		}
		for (const auto& [addr, value]: result.label_writes) {
			m_label_writer[addr] = LabelWriter {now, acb};
		}
		if (!acb) {
			for (; m_wait_stamped < m_translator.GpuWaitsNext(); m_wait_stamped++) {
				m_wait_dcb_time[m_wait_stamped % kGpuTimesPairs] = now;
			}
		}
	}

	// KYTY_NATIVE_GPU_FLIP_TRACE=1 (or =gpu, with GPU clocks): one line per flip in flips.csv (the working
	// directory) with what happened since the previous one, to see what the late frames have that the
	// others do not.
	void FlipTrace(double now, double present_s) {
		static const bool on = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_FLIP_TRACE");
			return v != nullptr && (v[0] == '1' || std::strcmp(v, "gpu") == 0);
		}();
		if (!on) {
			return;
		}
		struct Mark {
			double   t = 0, translate_ms = 0, busy_s = 0, gpu_s = 0;
			uint64_t dcbs = 0, acbs = 0, dcb_dw = 0, acb_dw = 0, irqs = 0, waits = 0, labels = 0;
		};
		static std::FILE* f    = nullptr;
		static Mark       prev;
		static uint64_t   flip = 0;
		// GPU clock pairs finished since the last flip, in order; a pair without its begin (a dropped first
		// segment) or long unfinished (a refused buffer) is skipped.
		static uint64_t span_from = 0, wait_from = 0;
		// GPU wait time by who writes the label (KYTY_NATIVE_GPU_FLIP_TRACE=gpu), printed every 600 flips:
		// an ACB translated before the DCB that waits on it, one translated after it (by how long), the
		// DCB itself, or nothing seen.
		struct WaitAgg {
			double   ms = 0, max_ms = 0, lag_ms = 0;
			uint64_t n = 0, met = 0;
		};
		static std::map<std::string, WaitAgg> by_writer;
		auto take = [&](uint64_t offset, uint64_t& from, uint64_t next, uint32_t* n, bool label_waits) {
			double ms = 0;
			for (; from < next; from++) {
				auto* p = reinterpret_cast<volatile uint64_t*>(m_submitter->ScratchCpu() + offset + 16ull * (from % kGpuTimesPairs));
				const uint64_t b = p[0], e = p[1];
				if (e == 0 && next - from < 64) {
					break; // still on the GPU
				}
				if (b != 0 && e >= b) {
					const double one = static_cast<double>(e - b) * 1e-5; // 100 MHz
					ms += one;
					(*n)++;
					if (label_waits) {
						const auto&  info = m_translator.GpuWait(from);
						const double t    = m_wait_dcb_time[from % kGpuTimesPairs];
						const auto   w    = m_label_writer.find(info.addr);
						double       lag  = 0;
						std::string  key  = "no writer seen";
						if (w != m_label_writer.end()) {
							lag = (w->second.t - t) * 1e3;
							if (!w->second.acb) {
								key = "the DCB (or another DCB)";
							} else if (lag <= 0) {
								key = "an ACB translated before the DCB";
							} else {
								key = lag < 1 ? "an ACB translated after the DCB, <1 ms"
								      : lag < 4 ? "an ACB translated after the DCB, 1-4 ms"
								      : lag < 10 ? "an ACB translated after the DCB, 4-10 ms"
								                 : "an ACB translated after the DCB, 10+ ms";
							}
						}
						WaitAgg& agg = by_writer[key];
						agg.ms += one;
						agg.max_ms = std::max(agg.max_ms, one);
						agg.lag_ms += lag;
						agg.n++;
						agg.met += info.met ? 1 : 0;
					}
				}
				p[0] = 0;
				p[1] = 0;
			}
			return ms;
		};
		uint32_t     spans = 0, waits = 0;
		const double span_ms = take(kGpuTimesOffset, span_from, m_translator.GpuSpansNext(), &spans, false);
		const double wait_ms = take(kGpuTimesOffset + 16ull * kGpuTimesPairs, wait_from, m_translator.GpuWaitsNext(), &waits, true);
		if (flip % 600 == 599 && !by_writer.empty()) {
			double total = 0;
			for (const auto& [key, agg]: by_writer) {
				total += agg.ms;
			}
			std::fprintf(stderr, "native trace: gfx GPU waits over 600 flips: %.2f ms a flip, by the label's writer:\n", total / 600.0);
			for (const auto& [key, agg]: by_writer) {
				std::fprintf(stderr,
				             "native trace:   %s: %.2f ms a flip, %.1f waits a flip, %.3f ms each (max %.2f), writer %+.2f ms after "
				             "the DCB, met at translation %.0f%%\n",
				             key.c_str(), agg.ms / 600.0, static_cast<double>(agg.n) / 600.0, agg.ms / static_cast<double>(agg.n),
				             agg.max_ms, agg.lag_ms / static_cast<double>(agg.n),
				             100.0 * static_cast<double>(agg.met) / static_cast<double>(agg.n));
			}
			by_writer.clear();
		}
		const Mark cur {now, m_translate_ms_total, m_trace_busy_s, m_trace_gpu_busy_s.load(std::memory_order_relaxed),
		                m_frames, m_acbs, m_trace_dw[0], m_trace_dw[1], m_labels_stats[2], m_labels_stats[3], m_labels_stats[0]};
		if (f == nullptr) {
			f = std::fopen("flips.csv", "w");
			if (f == nullptr) {
				return;
			}
			std::fprintf(f, "flip,t_s,interval_ms,translate_ms,submit_ms,gpu_busy_ms,present_ms,dcbs,dcb_kdw,acbs,acb_kdw,"
			                "interrupts,gpu_waits,labels,gfx_gpu_ms,gfx_buffers,gfx_wait_ms,gfx_waits\n");
		} else {
			std::fprintf(f, "%llu,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%llu,%.1f,%llu,%.1f,%llu,%llu,%llu,%.2f,%u,%.2f,%u\n",
			             static_cast<unsigned long long>(flip), now, (now - prev.t) * 1e3, cur.translate_ms - prev.translate_ms,
			             (cur.busy_s - prev.busy_s) * 1e3, (cur.gpu_s - prev.gpu_s) * 1e3, present_s * 1e3,
			             static_cast<unsigned long long>(cur.dcbs - prev.dcbs), (cur.dcb_dw - prev.dcb_dw) / 1e3,
			             static_cast<unsigned long long>(cur.acbs - prev.acbs), (cur.acb_dw - prev.acb_dw) / 1e3,
			             static_cast<unsigned long long>(cur.irqs - prev.irqs), static_cast<unsigned long long>(cur.waits - prev.waits),
			             static_cast<unsigned long long>(cur.labels - prev.labels), span_ms, spans, wait_ms, waits);
			if ((flip & 63u) == 0) {
				std::fflush(f);
			}
		}
		prev = cur;
		flip++;
	}

	void FlipStats(double present_s) {
		std::lock_guard<std::mutex> lock(m_mutex);
		const double now = NowSecShadow();
		FlipTrace(now, present_s);
		m_flips++;
		m_present_s += present_s;
		// Flip-to-flip intervals by display refreshes: on time (one 60 Hz period), one late (two), worse.
		if (m_flip_last != 0.0) {
			const double ms = (now - m_flip_last) * 1e3;
			m_flip_hist[ms < 20.0 ? 0 : ms < 37.0 ? 1 : 2]++;
		}
		m_flip_last = now;
		if (m_flip_t0 == 0.0) {
			m_flip_t0 = now;
			m_busy_s = m_wait_lock_s = m_present_s = 0.0;
			m_flips  = 0;
			for (uint32_t i = 0; i < 8; i++) {
				m_labels_mark[i] = m_labels_stats[i];
			}
			return;
		}
		if (m_flips == 120) {
			const double per = 1e3 / static_cast<double>(m_flips);
			std::fprintf(stderr,
			             "native shadow: per flip: interval %.1f ms, native submit %.1f ms, waiting for the native "
			             "lock %.1f ms, presenter %.1f ms, GPU busy %.1f ms; %llu MiB guest memory resident\n",
			             (now - m_flip_t0) * per, m_busy_s * per, m_wait_lock_s * per, m_present_s * per,
			             m_gpu_busy_s.load(std::memory_order_relaxed) * per, static_cast<unsigned long long>(ResidentMiB()));
			std::fprintf(stderr, "native shadow: flip intervals: %u under 20 ms, %u 20-37 ms, %u longer\n", m_flip_hist[0],
			             m_flip_hist[1], m_flip_hist[2]);
			std::fprintf(stderr, "native shadow: pipeline per flip: translate %.2f ms, CS ioctl %.2f ms\n",
			             (m_translate_ms_total - m_translate_mark) / m_flips, (m_cs_ms_total - m_cs_mark) / m_flips);
			m_translate_mark = m_translate_ms_total;
			m_cs_mark        = m_cs_ms_total;
#if KYTY_GEM_BACKING_SUPPORTED
			if (m_gem != nullptr) {
				const auto [ext_n, ext_bytes] = m_gem->ExternalBuffers();
				std::fprintf(stderr, "native shadow: %llu userptr buffers, %llu MiB\n", static_cast<unsigned long long>(ext_n),
				             static_cast<unsigned long long>(ext_bytes >> 20u));
			}
#endif
			m_flip_hist = {};
			if (m_total_dw != 0) {
				std::fprintf(stderr,
				             "native shadow: verbatim (directly executable) %.1f%% of %llu dwords per flip, %.1f runs per flip "
				             "(<16: %llu, <64: %llu, <256: %llu, longer: %llu)\n",
				             100.0 * static_cast<double>(m_verbatim_dw) / static_cast<double>(m_total_dw),
				             static_cast<unsigned long long>(m_total_dw / m_flips), static_cast<double>(m_verbatim_runs) / m_flips,
				             static_cast<unsigned long long>(m_run_hist[0]), static_cast<unsigned long long>(m_run_hist[1]),
				             static_cast<unsigned long long>(m_run_hist[2]), static_cast<unsigned long long>(m_run_hist[3]));
				// Per opcode, DCB and ACB apart: packets per flip and how many of them did not go through as
				// written, by the number changed.
				for (uint32_t k = 0; k < 2; k++) {
					std::vector<std::tuple<uint64_t, uint64_t, uint32_t>> ops; // changed, all, opcode
					for (uint32_t i = 0; i < 256; i++) {
						if (m_ops[k][i] != 0) {
							ops.emplace_back(m_changed_ops[k][i], m_ops[k][i], i);
						}
					}
					std::sort(ops.rbegin(), ops.rend());
					std::fprintf(stderr, "native shadow: %s packets per flip by op (changed/all):", k == 0 ? "DCB" : "ACB");
					for (size_t i = 0; i < ops.size() && i < 14; i++) {
						const uint32_t op = std::get<2>(ops[i]);
						std::fprintf(stderr, " 0x%02x:%.0f/%.0f", op, static_cast<double>(m_changed_ops[k][op]) / m_flips,
						             static_cast<double>(m_ops[k][op]) / m_flips);
					}
					std::fprintf(stderr, "\n");
				}
				if (Libs::CountHleList() != nullptr) {
					Libs::DumpHleCounts(stderr, static_cast<double>(m_flips), 48);
				}
				for (uint32_t k = 0; k < 2; k++) {
					CutStats& c = m_cut_stats[k];
					if (c.total_dw == 0) {
						continue;
					}
					auto top = [&](const std::array<uint64_t, 256>& a, const char* what) {
						std::vector<std::pair<uint64_t, uint32_t>> v;
						for (uint32_t i = 0; i < a.size(); i++) {
							if (a[i] != 0) {
								v.emplace_back(a[i], i);
							}
						}
						std::sort(v.rbegin(), v.rend());
						std::fprintf(stderr, " %s", what);
						for (size_t i = 0; i < v.size() && i < 8; i++) {
							std::fprintf(stderr, " 0x%02x:%.0f", v[i].second, static_cast<double>(v[i].first) / m_flips);
						}
					};
					std::fprintf(stderr, "native shadow: direct execution, %s: %llu dwords and %.0f cuts per flip;",
					             k == 0 ? "DCBs" : "ACBs", static_cast<unsigned long long>(c.total_dw / m_flips),
					             static_cast<double>(c.cut_runs) / m_flips);
					top(c.cut_ops, "replaced");
					std::fprintf(stderr, ";");
					top(c.inserted_ops, "ours inserted before");
					std::fprintf(stderr, "\n");
					c = {};
				}
				std::fprintf(stderr, "native shadow: GPU label writes per flip: %.0f, %.0f dwords of WRITE_DATA\n",
				             static_cast<double>(m_labels_native) / m_flips, static_cast<double>(m_write_data_dw) / m_flips);
				m_verbatim_dw = m_total_dw = m_verbatim_runs = 0;
				m_labels_native = m_write_data_dw = 0;
				if (m_direct_min_dw != 0) {
					std::fprintf(stderr, "native shadow: direct runs per flip: %.0f IB2 calls, %.0f dwords run in place\n",
					             static_cast<double>(m_direct_ib2) / m_flips, static_cast<double>(m_direct_dw) / m_flips);
				}
				m_direct_ib2 = m_direct_dw = 0;
				m_run_hist    = {};
				m_changed_ops = {};
				m_ops         = {};
			}
			const uint64_t held   = g_held_released.exchange(0, std::memory_order_relaxed);
			const uint64_t waited = g_held_waited.exchange(0, std::memory_order_relaxed);
			const uint64_t wait_us = g_held_wait_us.exchange(0, std::memory_order_relaxed);
			std::fprintf(stderr, "native shadow: held pieces per flip: %.1f released, %.1f after waiting, %.2f ms waited\n",
			             static_cast<double>(held) / m_flips, static_cast<double>(waited) / m_flips,
			             static_cast<double>(wait_us) / 1e3 / m_flips);
			static const char* const kReason[kHoldReasons] = {"cross-ring unsubmitted", "write does not satisfy", "CPU writer",
			                                                  "partial cover", "no writer yet", "no native form"};
			for (int r = 0; r < kHoldReasons; r++) {
				const uint64_t n  = g_hold_reason_n[r].exchange(0, std::memory_order_relaxed);
				const uint64_t us = g_hold_reason_us[r].exchange(0, std::memory_order_relaxed);
				if (n != 0) {
					std::fprintf(stderr, "native shadow:   held for %s: %.1f per flip, %.2f ms\n", kReason[r],
					             static_cast<double>(n) / m_flips, static_cast<double>(us) / 1e3 / m_flips);
				}
			}
#if defined(__GLIBC__)
			if (EnvOn("KYTY_NATIVE_GPU_MALLOC_STATS")) {
				const struct mallinfo2 mi = mallinfo2();
				std::fprintf(stderr, "native shadow: host malloc: %zu MiB in arenas, %zu MiB mmapped, %zu MiB in use\n",
				             mi.arena >> 20u, mi.hblkhd >> 20u, (mi.uordblks + mi.hblkhd) >> 20u);
			}
#endif
			// The pipelined path never reaches the per-submission check: look at the dummy here. The GPU
			// may be writing it right now; a write that lands during the clear is caught next time.
			if (m_gem != nullptr && m_gem->DummyWritten()) {
				m_dummy_writes++;
			}
			if (m_dummy_writes > 0) {
				std::fprintf(stderr, "native shadow: %u checks found GPU writes to non-resident guest memory\n",
				             m_dummy_writes);
			}
			if (m_gds_dma_dcb[0] + m_gds_dma_dcb[1] + m_gds_dma_acb[0] + m_gds_dma_acb[1] > 0) {
				std::fprintf(stderr, "native shadow: DMA_DATA with GDS: dcb %llu to / %llu from, acb %llu to / %llu from\n",
				             static_cast<unsigned long long>(m_gds_dma_dcb[0]), static_cast<unsigned long long>(m_gds_dma_dcb[1]),
				             static_cast<unsigned long long>(m_gds_dma_acb[0]), static_cast<unsigned long long>(m_gds_dma_acb[1]));
			}
			if (m_acb_deferred > 0) {
				std::fprintf(stderr, "native shadow: ACBs deferred behind their label writers: %llu, %llu placed, %llu not\n",
				             static_cast<unsigned long long>(m_acb_deferred), static_cast<unsigned long long>(m_acb_resolved),
				             static_cast<unsigned long long>(m_acb_unresolved));
			}
			if (m_labels) {
				const double irq_per = m_flips != 0 ? static_cast<double>(m_labels_stats[2] - m_labels_mark[2]) /
				                                           static_cast<double>(m_flips)
				                                     : 0.0;
				std::fprintf(stderr,
				             "native shadow: native labels so far: %llu GPU label writes, %llu by the tracker, "
				             "timestamps %llu on the GPU / %llu by the tracker, %llu interrupts (%.1f per flip); "
				             "waits: %llu native, %llu cut, %llu barriers\n",
				             static_cast<unsigned long long>(m_labels_stats[0]), static_cast<unsigned long long>(m_labels_stats[1]),
				             static_cast<unsigned long long>(m_labels_stats[7]), static_cast<unsigned long long>(m_labels_stats[6]),
				             static_cast<unsigned long long>(m_labels_stats[2]), irq_per,
				             static_cast<unsigned long long>(m_labels_stats[3]), static_cast<unsigned long long>(m_labels_stats[4]),
				             static_cast<unsigned long long>(m_labels_stats[5]));
				for (uint32_t i = 0; i < 8; i++) {
					m_labels_mark[i] = m_labels_stats[i];
				}
				if (m_irq_event != 0) {
					std::fprintf(stderr, "native shadow: tracker interrupts so far: %llu waits woken, %llu ran out (10 ms)\n",
					             static_cast<unsigned long long>(m_irq_wakes.load()),
					             static_cast<unsigned long long>(m_irq_timeouts.load()));
				}
#if KYTY_GEM_BACKING_SUPPORTED
				if (m_eop_log_open) {
					std::fprintf(stderr,
					             "native shadow: title interrupts raised by the GPU so far: %llu emitted, %llu from the EOP log, "
					             "%llu dropped by it, %llu from other rings\n",
					             static_cast<unsigned long long>(m_irq_gpu.load()), static_cast<unsigned long long>(m_irq_logged.load()),
					             static_cast<unsigned long long>(m_eop_log.Dropped()), static_cast<unsigned long long>(m_irq_stray.load()));
				}
#endif
			}
			std::lock_guard<std::mutex> chain_lock(m_chain_mutex);
			if (m_chain_n > 0) {
				const double per_buf = 1e3 / static_cast<double>(m_chain_n);
				std::fprintf(stderr,
				             "native shadow: per buffer (%.1f per flip): queued %.2f ms, submit to done %.2f ms "
				             "(max %.1f), continuation %.2f ms\n",
				             static_cast<double>(m_chain_n) / static_cast<double>(m_flips), m_chain_queued_s * per_buf,
				             m_chain_latency_s * per_buf, m_chain_max_s * 1e3, m_chain_then_s * per_buf);
			}
			m_chain_n = 0;
			m_chain_queued_s = m_chain_latency_s = m_chain_then_s = m_chain_max_s = 0.0;
			m_flip_t0 = now;
			m_busy_s = m_wait_lock_s = m_present_s = 0.0;
			m_gpu_busy_s.store(0.0, std::memory_order_relaxed);
			m_flips  = 0;
		}
	}

	bool NativeAcb() const { return Active() && m_acb_mode == AcbMode::On; }

	bool DmabufPieces(uint64_t vaddr, uint64_t size, std::vector<NativeDmabufPiece>& out) {
#if KYTY_GEM_BACKING_SUPPORTED
		std::vector<GemBacking::DmabufPiece> pieces;
		if (m_gem == nullptr || !m_gem->DmabufPieces(vaddr, size, pieces)) {
			return false;
		}
		out.clear();
		for (const auto& p: pieces) {
			out.push_back(NativeDmabufPiece {p.fd, p.bo_size, p.bo_offset, p.len});
		}
		return true;
#else
		(void)vaddr;
		(void)size;
		(void)out;
		return false;
#endif
	}

	// Native labels: the translators write labels natively and report tracker points that write the
	// tracker slot in the scratch area; `oracle` decides the guest waits. False when the native path
	// cannot run (nothing is translated then).
	bool EnableLabels(LabelOracle* oracle) {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_initialized) {
			Init();
			m_initialized = true;
		}
#if KYTY_GEM_BACKING_SUPPORTED
		if (!m_enabled || m_disabled || m_submitter == nullptr || !m_submitter->Async()) {
			return false;
		}
		for (uint32_t slot = 0; slot < kTrackerSlots; slot++) {
			*reinterpret_cast<volatile uint64_t*>(m_submitter->ScratchCpu() + kTrackerOffset + slot * kTrackerStride) = 0;
		}
		m_validate = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_VALIDATE");
			return v == nullptr ? 1 : std::atoi(v);
		}();
		m_compute_mask = m_submitter->ComputeRingMask();
		// Primary mode puts ACBs on compute rings (KFD queues where KfdQueuesWanted) unless =0.
		m_acb_rings = EnvOn("KYTY_NATIVE_GPU_ACB_RINGS") ||
		              (EnvOn("KYTY_NATIVE_GPU_PRIMARY") && !EnvIs("KYTY_NATIVE_GPU_ACB_RINGS", "0"));
		if (m_acb_rings && m_compute_mask == 0) {
			std::fprintf(stderr, "native shadow: KYTY_NATIVE_GPU_ACB_RINGS set but the kernel exposes no compute ring; "
			                     "ACBs stay on the gfx ring\n");
			m_acb_rings = false;
		}
		if (m_acb_rings && KfdQueuesWanted()) {
			EnableKfdQueues();
		}
		// Tracker points as the PS5 takes a title's EOP interrupts: each tracker release raises one,
		// and the tracker sleeps on it instead of polling its slots every millisecond. Needs the
		// patched amdgpu; KYTY_NATIVE_GPU_IRQ=0 keeps the polling.
		if (!EnvIs("KYTY_NATIVE_GPU_IRQ", "0")) {
			m_irq_event = KfdEopEvent::Create(m_gem->KfdFd());
			// A title's interrupts with context ids reach it through the EOP log (patched amdgpu), raised
			// by the GPU where its graphics queue's releases run. KYTY_NATIVE_GPU_GUEST_IRQ=0 keeps them
			// tracker points.
			if (m_irq_event != 0 && !EnvIs("KYTY_NATIVE_GPU_GUEST_IRQ", "0") && m_eop_log.Open(m_gem->KfdFd())) {
				m_eop_log_open = true;
				m_translator.SetGuestInterruptsOnGpu(true);
				if (m_kfd_queues) {
					m_submitter->SetKfdCalibration(kKfdCalibrationCtx); // ACBs' interrupts too (UseAcbTracker)
				}
				std::thread([this] { ReadEopLog(); }).detach();
			}
			m_irq_ctx = m_irq_event | (m_eop_log_open ? KfdEopLog::kEventBit : 0u);
			if (m_irq_event != 0) {
				std::fprintf(stderr, "native shadow: tracker points raise the EOP interrupt (KFD event %u)%s\n", m_irq_event,
				             m_eop_log_open ? "; the graphics queue's interrupts with context ids are raised by the GPU (EOP log)"
				                            : "");
			} else {
				std::fprintf(stderr, "native shadow: no EOP interrupts for tracker points (needs KFD and the patched amdgpu, "
				                     "bc250_gfx_eop_events=1); the tracker polls\n");
			}
		}
		const uint64_t va      = m_submitter->ScratchVa() + kTrackerOffset;
		const char*    native  = std::getenv("KYTY_NATIVE_GPU_NATIVE_WAITS");
		const bool     waits   = native == nullptr || native[0] != '0';
		const bool     cpu_ts  = EnvOn("KYTY_NATIVE_GPU_CPU_TIMESTAMPS");
		// KYTY_NATIVE_GPU_TRACKER_WRITES=1: the tracker mirrors GPU label stores on the CPU (see
		// Pm4Translator::SetTrackerWrites).
		const bool     mirrors = EnvOn("KYTY_NATIVE_GPU_TRACKER_WRITES");
		m_label_oracle = oracle;
		m_translator.SetNativeLabels(oracle, va);
		// DCB flips at the end of pipe (tracker points); KYTY_NATIVE_GPU_EOP_FLIPS=0 leaves them to
		// Kyty's continuation after the whole buffer.
		m_translator.SetFlipsAtEop(HostCallbacks().flip != nullptr && !EnvIs("KYTY_NATIVE_GPU_EOP_FLIPS", "0"));
		m_acb_translator.SetNativeLabels(oracle, va);
		m_translator.SetNativeWaits(waits);
		m_acb_translator.SetNativeWaits(waits);
		m_translator.SetCpuTimestamps(cpu_ts);
		m_acb_translator.SetCpuTimestamps(cpu_ts);
		m_translator.SetTrackerWrites(mirrors);
		m_acb_translator.SetTrackerWrites(mirrors);
		if (m_acb_rings) {
			// MEC packet forms. The gfx translator is unchanged.
			m_acb_translator.SetComputeRing(true);
		}
		// KYTY_NATIVE_GPU_FLIP_TRACE=gpu: the trace also takes GPU clocks around every gfx buffer and GPU
		// wait. A clock read on an idle gfx ring stopped it (probe clock): this hung ASTRO BOT in a light
		// scene, where the ring idles between buffers. Diagnostics only.
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_FLIP_TRACE"); v != nullptr && std::strcmp(v, "gpu") == 0) {
			for (uint64_t i = kGpuTimesOffset; i < kGpuTimesOffset + 2 * 16ull * kGpuTimesPairs; i++) {
				m_submitter->ScratchCpu()[i] = 0;
			}
			const uint64_t va = m_submitter->ScratchVa() + kGpuTimesOffset;
			m_translator.SetGpuTimes(va, va + 16ull * kGpuTimesPairs, kGpuTimesPairs);
			m_trace_gpu = true;
			m_wait_dcb_time.assign(kGpuTimesPairs, 0.0);
		}
		// Direct runs: DCB stretches passed on as written run from the title's buffer as IB2s
		// (Pm4Translator::SetDirectRuns), as the PS5's CP runs the title's own buffer. An IB2 costs the CP
		// about 0.3 us (probe ib2), the dwords it saves copying and validating about 1 ns each.
		// KYTY_NATIVE_GPU_DIRECT_RUNS=N: stretches of N+ dwords (default 256); =0 copies everything.
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_DIRECT_RUNS"); v == nullptr || v[0] != '0') {
			const unsigned long n = v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
			m_direct_min_dw       = n > 1 ? static_cast<uint32_t>(n) : 256u;
			m_translator.SetDirectRuns(m_direct_min_dw);
			m_submitter->SetGfxMemSync(true);
			std::fprintf(stderr, "native shadow: direct runs: DCB stretches of %u+ dwords run from the title's buffer\n",
			             m_direct_min_dw);
		}
		m_labels = true;
		std::fprintf(stderr, "native shadow: native labels (the GPU writes labels, a tracker raises interrupts%s%s)\n",
		             waits ? ", guest waits pass through" : "; guest waits stay barriers",
		             mirrors ? "; the tracker mirrors GPU label stores" : "");
		if (m_acb_rings) {
			std::fprintf(stderr,
			             "native shadow: ACBs on compute rings (mask 0x%x); a guest wait is a GPU wait only once its "
			             "writer has been submitted\n",
			             m_compute_mask);
		}
		return true;
#else
		(void)oracle;
		return false;
#endif
	}

	// The tracker slot: the (low 32 bits of the) sequence number of the last tracker point the GPU has
	// passed.
	uint32_t TrackerValue(uint32_t slot = 0) const {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter == nullptr || slot >= kTrackerSlots) {
			return 0;
		}
		return *reinterpret_cast<const volatile uint32_t*>(m_submitter->ScratchCpu() + kTrackerOffset + slot * kTrackerStride);
#else
		(void)slot;
		return 0;
#endif
	}
	// True once the GPU has passed tracker point `seq` on `slot` (wrap-safe on the 32-bit slot).
	bool Reached(uint64_t seq, uint32_t slot = 0) const {
		return static_cast<int32_t>(TrackerValue(slot) - static_cast<uint32_t>(seq)) >= 0;
	}

	bool AcbRings() const { return m_acb_rings; }
	uint32_t ComputeRingMask() const { return m_compute_mask; }
	// Guest compute queues run on KFD user-mode queues of their own (EnableKfdQueues).
	bool KfdQueuesOn() const { return m_kfd_queues; }
	// LabelPipeline::SlotOf gave guest compute queue `queue` KFD queue `index`.
	void NoteKfdGuestQueue(uint32_t index, uint32_t queue) {
		if (index < m_kfd_guest_queue.size()) {
			const uint32_t prev = m_kfd_guest_queue[index].load();
			m_kfd_guest_queue[index] = prev == 0 ? queue + 1u : kSharedKfdQueue;
		}
	}

	// The KFD event the tracker's releases raise (INT_SEL 2); 0 when they do not.
	uint32_t IrqEvent() const { return m_irq_event; }
	// The context id that names it.
	uint32_t IrqCtx() const { return m_irq_ctx; }
	// Sleeps until the tracker's interrupt fires or `timeout_ms` passes.
	void WaitIrq(uint32_t timeout_ms) {
#if KYTY_GEM_BACKING_SUPPORTED
		if (KfdEopEvent::Wait(m_gem->KfdFd(), m_irq_event, timeout_ms)) {
			m_irq_wakes++;
		} else {
			m_irq_timeouts++;
		}
#else
		(void)timeout_ms;
#endif
	}
	// Tracker slots: 0 is the gfx ring, kGfx1Slot the second gfx ring when amdgpu has one (the PS5's
	// "AGR" ring: the BC-250's patched amdgpu with bc250_gfx_queues=2), any other n the compute ring or
	// KFD queue n - 1 (KFD indexes stay below 31).
	static constexpr uint32_t kGfx1Slot = 32;
	// A guest graphics queue's waits go to the GPU (while each has a gfx ring of its own) when amdgpu
	// updates compute VMs' page tables with SDMA (amdgpu.vm_update_mode=0). With its default on this
	// board, the CPU writes them and first waits for every fence of the VM: a guest GEM_VA then sat in
	// the kernel behind a gfx job waiting on the GPU for an ACB, holding GemBacking's lock that ACB's
	// submission needed (My First Gran Turismo stopped at once). KYTY_NATIVE_GPU_GFX_GPU_WAITS=1/0
	// decides instead.
	bool GfxQueueWaitsOnGpu() const {
		static const bool on = [] {
			if (const char* v = std::getenv("KYTY_NATIVE_GPU_GFX_GPU_WAITS"); v != nullptr && v[0] != '\0') {
				return v[0] == '1';
			}
			FILE* f    = std::fopen("/sys/module/amdgpu/parameters/vm_update_mode", "re");
			int   mode = -1;
			if (f != nullptr) {
				mode = std::fscanf(f, "%d", &mode) == 1 ? mode : -1;
				std::fclose(f);
			}
			return mode == 0;
		}();
		return on && m_acb_rings;
	}
	static bool GfxSlot(uint32_t slot) { return slot == 0 || slot == kGfx1Slot; }
	bool SecondGfxRing() const {
#if KYTY_GEM_BACKING_SUPPORTED
		return m_submitter != nullptr && (m_submitter->GfxRingMask() & 2u) != 0;
#else
		return false;
#endif
	}
	uint32_t KfdQueueCount() const { return std::max(m_submitter != nullptr ? m_submitter->KfdQueueCount() : 0u, 1u); }

	// The per-VMID PS5 settings follow this VM's gfx VMID, which amdgpu may change between jobs: once
	// per flip they move to the VMID it holds now and the one it left gets the Linux values back.
	void Ps5RegsOnFlip() {
#if KYTY_GEM_BACKING_SUPPORTED
		if ((m_ps5_regs & (kPs5Accum | kPs5ShMem)) == 0) {
			return;
		}
		std::lock_guard<std::mutex> lock(m_ps5_mutex);
		const uint32_t vmid = m_ps5_hw.GfxVmid(m_ps5_pasid);
		if (vmid == 0 || vmid == m_ps5_gfx_vmid) {
			return;
		}
		if (m_ps5_gfx_vmid != 0) {
			for (uint32_t reg: {kPs5ShMemBases, kPs5UserAccumVmid}) {
				m_ps5_hw.RestoreRegs(reg, LodStatsHw::SelVmid(m_ps5_gfx_vmid));
			}
		}
		std::fprintf(stderr, "native shadow: PS5 per-VMID registers on gfx VMID %u (was %u)\n", vmid, m_ps5_gfx_vmid);
		m_ps5_gfx_vmid = vmid;
		Ps5VmidRegs(vmid);
#endif
	}

	// The ACB translator's tracker slot, one per compute ring (slot 0 is the gfx ring), and its barrier
	// fence, the slot's last dword. Compute rings run at once, and a barrier releases its own value into
	// the fence and waits for exactly that value: a fence shared with another ring can be overwritten in
	// between, and the wait never ends. Called on the label-pipeline thread before that queue is translated.
	void UseAcbTracker(uint32_t slot) {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_label_oracle == nullptr || m_submitter == nullptr || slot >= kTrackerSlots) {
			return;
		}
		const uint64_t va = m_submitter->ScratchVa() + kTrackerOffset + static_cast<uint64_t>(slot) * kTrackerStride;
		m_acb_translator.SetNativeLabels(m_label_oracle, va);
		m_acb_translator.SetFenceAddress(va + 12);
		// A guest compute queue with a KFD queue of its own: its interrupts with context ids are raised
		// by the GPU too, and the EOP log tells its hardware queue apart (ReadEopLog).
		const bool own = m_eop_log_open && m_kfd_queues && slot >= 1 && slot - 1 < m_kfd_guest_queue.size() &&
		                 m_kfd_guest_queue[slot - 1].load() != 0 && m_kfd_guest_queue[slot - 1].load() != kSharedKfdQueue &&
		                 !EnvIs("KYTY_NATIVE_GPU_ACB_IRQ", "0");
		m_acb_translator.SetGuestInterruptsOnGpu(own);
#else
		(void)slot;
#endif
	}

	// KYTY_NATIVE_GPU_VALIDATE (1 unless set): 0 no packet check, 1 every translated IB goes through
	// ValidateNativeIb, 2 also each segment again after LabelPipeline patched it. The patches are filler
	// NOPs over the translator's own wait and barrier packets and sequence values in label data dwords,
	// which keep a checked IB valid; checking twice took a tenth of the submit thread in My First GT.
	bool ValidatingPatched() const { return m_validate >= 2; }
	void DisableNow(const char* reason) {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_disabled) {
			Disable(reason);
		}
	}

	// The dword at `addr` through the guest alias, so a label mirror can tell a GPU store that
	// already landed from one the guest has replaced. Called while translating.
	bool SampleDword(uint64_t addr, uint32_t* out) const {
#if KYTY_GEM_BACKING_SUPPORTED
		return out != nullptr && m_gem != nullptr && m_gem->ReadGuest(addr, out, 4) == 4;
#else
		(void)addr;
		(void)out;
		return false;
#endif
	}

	// Whether the GPU may write a label at `addr`. Called while translating (the lock is held).
	bool GpuWritable(uint64_t addr) const {
#if KYTY_GEM_BACKING_SUPPORTED
		// The range of the last answer, kept while the views stay the same: labels sit in a few views,
		// and each asked the backing under its lock (~8000 a frame in ASTRO BOT).
		if (m_gem == nullptr) {
			return false;
		}
		const uint64_t gen = m_gem->ViewsGeneration();
		if (gen == m_writable_gen && addr >= m_writable_lo && addr < m_writable_hi) {
			return m_writable;
		}
		uint64_t lo = 0, hi = 0;
		const bool writable = m_gem->GpuWritableRange(addr, &lo, &hi);
		m_writable_gen      = gen;
		m_writable_lo       = lo;
		m_writable_hi       = hi;
		m_writable          = writable;
		return writable;
#else
		(void)addr;
		return false;
#endif
	}

	double TimeoutSec() const { return m_timeout_sec; }

	// Native labels: appends a ready segment to the batch FlushBatch submits. `slot` 0 is the gfx
	// ring; with ACB rings, slot n > 0 is compute ring n - 1.
	void AppendBatch(std::span<const uint32_t> ib, uint32_t slot = 0) {
#if KYTY_GEM_BACKING_SUPPORTED
		std::lock_guard<std::mutex> lock(m_mutex);
		std::vector<uint32_t>& batch = m_acb_rings ? m_ring_batch[slot < kTrackerSlots ? slot : 0] : m_batch;
		if ((batch.size() + ib.size()) * 4 > NativeSubmitter::kSlotSize) {
			SubmitBufferLocked(batch, m_acb_rings ? slot : 0);
		}
		batch.insert(batch.end(), ib.begin(), ib.end());
#else
		(void)ib;
		(void)slot;
#endif
	}

	// Pipeline mode: Submit batches IBs; FlushBatch submits them with one CS.
	void SetPipelined() {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_pipelined = true;
	}
	void FlushBatch() {
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_acb_rings) {
			SubmitBatchLocked();
			return;
		}
		// Gfx first, then each compute ring. PlanRingWait does not put a cross-ring GPU wait into a
		// batch until that writer's CS has returned, so a waiter is never submitted before its writer.
		for (uint32_t slot = 0; slot < kTrackerSlots; slot++) {
			SubmitBufferLocked(m_ring_batch[slot], slot);
		}
	}

	// Waits for the last CS submitted to tracker `slot`. True when that ring is idle (or nothing was
	// submitted there). Each ring has its own fence. A timeout disables the native path.
	bool WaitSlot(uint32_t slot) {
#if KYTY_GEM_BACKING_SUPPORTED
		SlotFence fence;
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			if (slot >= kTrackerSlots) {
				return false;
			}
			fence = m_slot_fence[slot];
		}
		if (fence.seq == 0) {
			return true;
		}
		if (m_submitter != nullptr &&
		    m_submitter->WaitSeq(fence.seq, m_timeout_sec, {fence.ip_type, fence.ring})) {
			return true;
		}
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_disabled) {
			std::fprintf(stderr, "native shadow: ring slot %u submission %llu TIMED OUT; disabling native path %s\n", slot,
			             static_cast<unsigned long long>(fence.seq), m_submitter->QueueState({fence.ip_type, fence.ring}).c_str());
			DumpHangLocked();
			RunHangHook();
			Disable("timeout");
		}
#else
		(void)slot;
#endif
		return false;
	}

	// KYTY_NATIVE_GPU_HANG_DUMP=<dir>: on a ring timeout, writes each ring's submissions that have not
	// signaled (the stuck one first) and each compute ring's last 32 as slot<S>_seq<N>[_done].bin, and the
	// gfx ring's last ones, so a hang can be read with ib_dump.py without breadcrumbs slowing every frame.
	void DumpHangLocked() {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_hang_dump.empty() || m_submitter == nullptr) {
			return;
		}
		for (uint32_t slot = 0; slot < kTrackerSlots; slot++) {
			const NativeSubmitter::RingTarget target = SlotTarget(slot);
			uint32_t written = 0;
			const auto& kept = m_hang_kept[slot];
			for (size_t k = 0; k < kept.size(); k++) {
				const auto& [seq, ib] = kept[k];
				const bool done = m_submitter->WaitSeq(seq, 0.0, target);
				// Compute rings: also the last 32 that signaled (waves can outlive the fence that ends their
				// submission), and up to 8 that did not.
				const bool recent = k + 32 >= kept.size();
				if ((done && !GfxSlot(slot) && !recent) || written >= 40) {
					continue;
				}
				const std::string path = m_hang_dump + "/slot" + std::to_string(slot) + "_seq" + std::to_string(seq) +
				                         (done ? "_done" : "") + ".bin";
				if (std::FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
					std::fwrite(ib.data(), 4, ib.size(), f);
					std::fclose(f);
					written++;
				}
			}
			if (written != 0) {
				std::fprintf(stderr, "native shadow: hang dump: %u submissions of slot %u in %s (tracker %u)\n", written, slot,
				             m_hang_dump.c_str(), TrackerValue(slot));
			}
		}
#endif
	}

	// KYTY_NATIVE_GPU_HANG_HOOK=<command>: run on a ring timeout before anything is released, so a
	// capture (tools/bc250-runs/hang-capture-now.sh: CP state and waves) sees the GPU as it hung.
	static void RunHangHook() {
		if (const char* hook = std::getenv("KYTY_NATIVE_GPU_HANG_HOOK"); hook != nullptr && hook[0] != '\0') {
			std::fprintf(stderr, "native shadow: running the hang hook: %s\n", hook);
			const int r = std::system(hook);
			std::fprintf(stderr, "native shadow: hang hook done (%d)\n", r);
		}
	}

	// Blocks until `slot`'s latest fence signals or `timeout_sec` passes. A timeout does not disable
	// the path: the tracker uses this instead of polling the slot. True when the ring is idle or the
	// fence signaled.
	bool WaitSlotBrief(uint32_t slot, double timeout_sec) {
#if KYTY_GEM_BACKING_SUPPORTED
		SlotFence fence;
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			if (slot >= kTrackerSlots) {
				return false;
			}
			fence = m_slot_fence[slot];
		}
		if (fence.seq == 0 || m_submitter == nullptr) {
			return fence.seq == 0;
		}
		return m_submitter->WaitSeq(fence.seq, timeout_sec, {fence.ip_type, fence.ring});
#else
		(void)slot;
		(void)timeout_sec;
		return true;
#endif
	}

	// The submission the last Submit queued in async mode (0: none), for the pipeline to wait on.
	uint64_t TakeLastSeq() {
		std::lock_guard<std::mutex> lock(m_mutex);
		return std::exchange(m_last_seq, 0);
	}

	// Waits for an async submission; false on a timeout (the native path then disables itself).
	bool WaitSubmission(uint64_t seq) {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter != nullptr && m_submitter->WaitSeq(seq, m_timeout_sec)) {
			return true;
		}
		std::lock_guard<std::mutex> lock(m_mutex);
		if (!m_disabled) {
			std::fprintf(stderr, "native shadow: submission %llu TIMED OUT; disabling native path\n",
			             static_cast<unsigned long long>(seq));
			Disable("timeout");
		}
#else
		(void)seq;
#endif
		return false;
	}

private:
	// SIGTERM, SIGINT, SIGHUP: the GPU's pending waits are released before the process goes. Killed
	// with jobs on the rings, the jobs not yet scheduled are dropped, and a submitted one waiting for
	// a label they would have written spins forever: amdgpu cannot preempt a WAIT_REG_MEM on its own
	// rings, and with GPU recovery off the board hangs. A thread does it (the handler only wakes it):
	// it writes each recent native wait a value that satisfies it until the rings are idle, then exits.
	void InstallStopDrain() {
#if KYTY_GEM_BACKING_SUPPORTED
		static int fds[2] = {-1, -1};
		if (fds[0] >= 0 || ::pipe2(fds, O_CLOEXEC) != 0) {
			return;
		}
		struct sigaction sa {};
		sa.sa_handler = [](int sig) {
			const char c = static_cast<char>(sig);
			(void)!::write(fds[1], &c, 1);
		};
		sigemptyset(&sa.sa_mask);
		for (int sig: {SIGTERM, SIGINT, SIGHUP}) {
			::sigaction(sig, &sa, nullptr);
		}
		std::thread([this] {
			char sig = 0;
			while (::read(fds[0], &sig, 1) != 1) {
			}
			DrainAndExit(sig);
		}).detach();
#endif
	}

	// The EOP log's reader: each entry is a title's interrupt the GPU raised. The DCB translator only
	// leaves those of guest queue 0 to the GPU, and that queue owns gfx ring 0 (LabelPipeline::SlotOf),
	// so an entry from the gfx micro engine (me 0) is queue 0's.
	void ReadEopLog() {
#if KYTY_GEM_BACKING_SUPPORTED
		::prctl(PR_SET_TIMERSLACK, 1ul, 0ul, 0ul, 0ul);
		// A compute entry names its hardware queue (the IV ring id); each KFD queue's calibration
		// interrupt says which index that is. Entries from a queue not calibrated yet wait for it.
		std::array<int32_t, 256>                     ring_index;
		std::deque<std::pair<uint32_t, uint32_t>>    waiting;
		ring_index.fill(-1);
		auto deliver = [this, &ring_index](uint32_t ctx, uint32_t ring) {
			const uint32_t queue = m_kfd_guest_queue[static_cast<uint32_t>(ring_index[ring])].load();
			if (queue == 0 || queue == kSharedKfdQueue) {
				m_irq_stray++;
				return;
			}
			m_irq_logged++;
			if (HostCallbacks().interrupt) {
				HostCallbacks().interrupt(static_cast<int>(queue - 1u), ctx);
			}
		};
		for (;;) {
			m_eop_log.Drain(100, [&](uint32_t ctx, uint32_t source) {
				const uint32_t ring = source & 0xffu;
				if (((source >> 2u) & 0x3u) == 0u) {
					m_irq_logged++;
					if (HostCallbacks().interrupt) {
						HostCallbacks().interrupt(0, ctx);
					}
				} else if ((ctx & ~0x1fu) == kKfdCalibrationCtx) {
					ring_index[ring] = static_cast<int32_t>(ctx & 0x1fu);
					std::fprintf(stderr, "native shadow: KFD queue %u runs on hardware queue me %u pipe %u queue %u\n", ctx & 0x1fu,
					             (ring >> 2u) & 3u, ring & 3u, (ring >> 4u) & 7u);
					for (auto it = waiting.begin(); it != waiting.end();) {
						if (it->second == ring) {
							deliver(it->first, ring);
							it = waiting.erase(it);
						} else {
							++it;
						}
					}
				} else if (ring_index[ring] >= 0) {
					deliver(ctx, ring);
				} else if (waiting.size() < 4096) {
					waiting.emplace_back(ctx, ring);
				} else {
					m_irq_stray++;
				}
			});
		}
#endif
	}

	[[noreturn]] void DrainAndExit(int sig) {
#if KYTY_GEM_BACKING_SUPPORTED
		{
			std::lock_guard<std::mutex> lock(m_mutex);
			m_disabled = true; // nothing more is submitted
		}
		uint32_t released = 0;
		bool     idle     = m_submitter == nullptr;
		for (int round = 0; round < 40 && !idle; round++) {
			for (const WaitCond& w: RecentGpuWaits()) {
				uint64_t cur = 0;
				if (m_gem->ReadGuest(w.addr, &cur, w.wide ? 8 : 4) == 0 || w.Met(cur)) {
					continue;
				}
				const uint64_t v = SatisfyingValue(w, cur);
				if (round == 0) {
					std::fprintf(stderr, "native shadow: pending GPU wait on 0x%llx (%s, func %u, ref 0x%llx, mask 0x%llx): holds 0x%llx\n",
					             static_cast<unsigned long long>(w.addr), w.wide ? "64-bit" : "32-bit", w.function,
					             static_cast<unsigned long long>(w.ref), static_cast<unsigned long long>(w.mask),
					             static_cast<unsigned long long>(cur));
				}
				m_gem->WriteGuest(w.addr, &v, w.wide ? 8 : 4);
				released++;
			}
			idle = m_submitter->WaitIdle(0.05);
		}
		std::fprintf(stderr, "native shadow: signal %d: %u pending GPU waits released, rings %s; exiting\n", sig, released,
		             idle ? "idle" : "STILL BUSY");
		if (m_ps5_regs != 0) {
			// The helper would do it when the connection closes; here it shows in this log too.
			std::lock_guard<std::mutex> lock(m_ps5_mutex);
			std::fprintf(stderr, "native shadow: %d PS5 registers put back (kyty-lodhw logs each)\n", m_ps5_hw.RestoreRegs());
		}
#endif
		::_exit(128 + sig);
	}

	// KYTY_NATIVE_GPU_KFD_QUEUES=1: the compute rings become KFD user-mode queues, rung by doorbell
	// like the PS5's compute queues. Their VMIDs (8-15, KFD's) get the same GDS partition as the CS jobs
	// (base 0), which amdgpu leaves empty there; the kyty-lodhw helper writes it, as the PS5 kernel
	// gives the title's VMID a static one. Without the helper, GDS users stay on CS compute rings.
	void EnableKfdQueues() {
#if KYTY_GEM_BACKING_SUPPORTED
		if (m_submitter->HasGds()) {
			if (!LodStatsHw::Instance().On()) {
				LodStatsHw::Instance().Connect();
			}
			for (uint32_t vmid = 8; vmid < 16; vmid++) {
				const uint32_t reg = 0x1260 + 0x20a0 + vmid * 2;
				if (LodStatsHw::Instance().WriteReg(reg, 0) != 0 || LodStatsHw::Instance().WriteReg(reg + 1, m_submitter->GdsSize()) != 0) {
					std::fprintf(stderr, "native shadow: GDS of the KFD VMIDs not writable (kyty-lodhw helper); ACBs stay on CS "
					                     "compute rings\n");
					return;
				}
			}
		}
		// Every guest compute queue gets a KFD queue of its own (LabelPipeline::SlotOf hands out the
		// indexes). KYTY_NATIVE_GPU_KFD_QUEUE_RINGS=<mask>: only these indexes do (0: none; the others
		// stay CS compute rings).
		uint32_t rings = (1u << NativeSubmitter::MaxKfdQueues()) - 1u;
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_KFD_QUEUE_RINGS"); v != nullptr && v[0] != '\0') {
			rings &= static_cast<uint32_t>(std::strtoul(v, nullptr, 0));
		}
		if (!m_submitter->EnableKfdQueues(rings)) {
			std::fprintf(stderr, "native shadow: KFD queues unavailable; ACBs stay on CS compute rings\n");
			return;
		}
		m_kfd_queues = true;
		std::fprintf(stderr, "native shadow: ACBs on %u KFD user-mode queues, one per guest queue (doorbell submission, GDS %u bytes)\n",
		             m_submitter->KfdQueueCount(), m_submitter->GdsSize());
		if (LodStatsHw::Instance().On() && LodStatsHw::Instance().InvalidateKfdTlb() == 0) {
			m_submitter->SetKfdTlbFlush([] { return LodStatsHw::Instance().InvalidateKfdTlb() == 0; });
			std::fprintf(stderr, "native shadow: KFD TLB flushes by the kyty-lodhw helper after GEM_VA changes\n");
		} else {
			std::fprintf(stderr, "native shadow: no KFD TLB flush through kyty-lodhw (helper missing or older than op 14)\n");
		}
		{
			std::lock_guard<std::mutex> lock(m_ps5_mutex);
			Ps5KfdRegs();
		}
		// The wave save area stock KFD gives a queue holds the CUs amdgpu counted at init; on a board with
		// more CUs enabled after it (the BC-250's 40 of 24), a preempted queue would overflow it. ACBs
		// stay on those CUs unless the save area holds every CU (the BC-250's patched amdgpu) or
		// KYTY_NATIVE_GPU_KFD_CU_CLAMP says otherwise (1: always, 0: never).
		const char* clamp_env = std::getenv("KYTY_NATIVE_GPU_KFD_CU_CLAMP");
		const bool  clamp     = clamp_env != nullptr && clamp_env[0] != '\0' ? clamp_env[0] != '0'
		                                                                     : !KfdQueue::SaveAreaCoversLateCus(m_gem->KfdGpuId());
		if (!clamp) {
			std::fprintf(stderr, "native shadow: ACBs on every CU (KFD's wave save areas hold them all)\n");
		} else {
			const auto masks = m_submitter->InitCuMasks();
			m_acb_translator.SetComputeCuClamp(masks);
			std::fprintf(stderr, "native shadow: ACB CU masks clamped to amdgpu's init CUs 0x%08x 0x%08x 0x%08x 0x%08x\n", masks[0],
			             masks[1], masks[2], masks[3]);
		}
#endif
	}

	// GET_LOD_STATS through the GPU's own reporter when the kyty-lodhw helper runs (and the mode is
	// not a CPU-side one). The helper puts this VM's VMID into CNTL3, as the PS5 kernel does with the
	// process VMID; it finds it by the VM's PASID.
	void InitLodStatsHw() {
#if KYTY_GEM_BACKING_SUPPORTED
		const char* mode = std::getenv("KYTY_LOD_STATS_MODE");
		if (mode != nullptr && mode[0] != '\0' && std::strcmp(mode, "hw") != 0 && std::strcmp(mode, "usage") != 0) {
			return;
		}
		if (!LodStatsHw::Instance().Connect()) {
			if (mode != nullptr && std::strcmp(mode, "hw") == 0) {
				std::fprintf(stderr, "native shadow: KYTY_LOD_STATS_MODE=hw but the kyty-lodhw helper is not running\n");
			}
			return;
		}
		const uint32_t pasid = LodStatsHw::Pasid(m_gem->DrmFd());
		if (pasid == 0) {
			std::fprintf(stderr, "native shadow: no PASID in the DRM fdinfo; not using the LOD statistics reporter\n");
			LodStatsHw::Instance().Disconnect();
			return;
		}
		LodStatsHw::Instance().SetPasid(pasid);
		std::fprintf(stderr, "native shadow: GET_LOD_STATS by the GPU's LOD statistics reporter (kyty-lodhw helper, PASID %u)\n", pasid);
#endif
	}

	// KYTY_NATIVE_GPU_PS5_REGS: GPU state the PS5 kernel sets up for the title's queues and amdgpu does
	// not, written through the kyty-lodhw helper (root). A comma list of:
	//   gfx_pipe  GFX_PIPE_CONTROL.CONTEXT_SUSPEND_EN on both gfx pipes
	//   accum     SPI_CONFIG_CNTL_1.EN_USER_ACCUM, and SPI_USER_ACCUM_VMID_CNTL = 0xf on the game's VMIDs
	//   kfd_gds   GWS 0-31 and OA 0-7 for KFD's VMIDs 8-15 (the PS5 title VMID's; GDS itself is
	//             EnableKfdQueues'; the gfx VMID's comes from KYTY_NATIVE_GPU_GDS)
	//   shmem     SH_MEM_BASES = 0x80007000 (the PS5 aperture bases) on the game's VMIDs
	//   hqd       on each active KFD compute queue: TUNNEL_DISPATCH, the PS5 PERSISTENT_STATE bits and
	//             the GDS resource state (OA and 48 GWS); written while it runs, and KFD's scheduler
	//             may reload the queue from its MQD on any remap
	//   golden    the helper's local golden table (/etc/kyty/ps5-golden.conf)
	// "all" (or 1) is gfx_pipe,accum,kfd_gds. The game's VMIDs are this VM's gfx VMID (by PASID,
	// followed per flip) and, with KFD queues, 8-15. Nothing is written before a setting is asked for;
	// the helper puts every register back when the connection closes (exit or crash).
	static constexpr uint32_t kPs5GfxPipe = 1u, kPs5Accum = 2u, kPs5KfdGds = 4u, kPs5ShMem = 8u, kPs5Hqd = 16u, kPs5Golden = 32u;
	// Absolute dword offsets (GC segment 0 is 0x1260).
	static constexpr uint32_t kPs5GfxPipeControl = 0x1260 + 0x100d, kPs5ShMemBases = 0x1260 + 0x10aa,
	                          kPs5UserAccumVmid = 0x1260 + 0x11df, kPs5SpiConfigCntl1 = 0x1260 + 0x11ef,
	                          kPs5HqdActive = 0x1260 + 0x1fab, kPs5HqdVmid = 0x1260 + 0x1fac,
	                          kPs5HqdPersistentState = 0x1260 + 0x1fad, kPs5HqdPqControl = 0x1260 + 0x1fba,
	                          kPs5HqdGdsResourceState = 0x1260 + 0x1fdb, kPs5GdsGwsVmid0 = 0x1260 + 0x20c0,
	                          kPs5GdsOaVmid0 = 0x1260 + 0x20d0;

	void InitPs5Regs() {
#if KYTY_GEM_BACKING_SUPPORTED
		const char* v = std::getenv("KYTY_NATIVE_GPU_PS5_REGS");
		if (v == nullptr || v[0] == '\0' || std::strcmp(v, "0") == 0) {
			return;
		}
		uint32_t    regs = 0;
		std::string list = v;
		for (size_t pos = 0; pos <= list.size();) {
			size_t end = list.find(',', pos);
			end        = end == std::string::npos ? list.size() : end;
			const std::string name = list.substr(pos, end - pos);
			pos                    = end + 1;
			static const std::pair<const char*, uint32_t> kNames[] = {
			    {"all", kPs5GfxPipe | kPs5Accum | kPs5KfdGds}, {"1", kPs5GfxPipe | kPs5Accum | kPs5KfdGds},
			    {"gfx_pipe", kPs5GfxPipe}, {"accum", kPs5Accum}, {"kfd_gds", kPs5KfdGds}, {"shmem", kPs5ShMem},
			    {"hqd", kPs5Hqd}, {"golden", kPs5Golden}};
			const auto* it = std::find_if(std::begin(kNames), std::end(kNames), [&](const auto& n) { return name == n.first; });
			if (it != std::end(kNames)) {
				regs |= it->second;
			} else if (!name.empty()) {
				std::fprintf(stderr, "native shadow: KYTY_NATIVE_GPU_PS5_REGS: unknown setting '%s'\n", name.c_str());
			}
		}
		uint32_t probe = 0;
		if (regs == 0 || !m_ps5_hw.Connect()) {
			std::fprintf(stderr, "native shadow: KYTY_NATIVE_GPU_PS5_REGS set but %s; no PS5 registers written\n",
			             regs == 0 ? "no known setting" : "the kyty-lodhw helper is not running");
			return;
		}
		if (m_ps5_hw.ReadReg(kPs5SpiConfigCntl1, 0, &probe) != 0) {
			std::fprintf(stderr, "native shadow: the kyty-lodhw helper has no banked register ops (rebuild it); no PS5 registers written\n");
			m_ps5_hw.Disconnect();
			return;
		}
		std::lock_guard<std::mutex> lock(m_ps5_mutex);
		m_ps5_regs  = regs;
		m_ps5_pasid = LodStatsHw::Pasid(m_gem->DrmFd());
		std::fprintf(stderr, "native shadow: PS5 registers (0x%x) through the kyty-lodhw helper, PASID %u\n", regs, m_ps5_pasid);
		if ((regs & kPs5GfxPipe) != 0) {
			for (uint32_t pipe = 0; pipe < 2; pipe++) {
				Ps5Reg("GFX_PIPE_CONTROL", "gfx pipe " + std::to_string(pipe), kPs5GfxPipeControl, LodStatsHw::SelGfxPipe(pipe),
				       1u << 16u, 1u << 16u);
			}
		}
		if ((regs & kPs5Accum) != 0) {
			Ps5Reg("SPI_CONFIG_CNTL_1", "global", kPs5SpiConfigCntl1, 0, 1u << 21u, 1u << 21u);
		}
		if ((regs & kPs5Golden) != 0) {
			std::fprintf(stderr, "native shadow: PS5 golden table: %d registers changed (kyty-lodhw logs each)\n", m_ps5_hw.ApplyGolden());
		}
		if ((regs & (kPs5Accum | kPs5ShMem)) != 0 && m_ps5_pasid == 0) {
			std::fprintf(stderr, "native shadow: no PASID in the DRM fdinfo; the gfx VMID gets no per-VMID PS5 registers\n");
		}
		if ((regs & (kPs5Accum | kPs5KfdGds | kPs5ShMem | kPs5Hqd)) != 0) {
			// KFD queues are created on their first submission; each gets its settings then.
			m_submitter->SetKfdQueueHook([this](uint32_t ring) {
				std::lock_guard<std::mutex> lock(m_ps5_mutex);
				std::fprintf(stderr, "native shadow: PS5 registers after KFD queue %u was created\n", ring);
				Ps5KfdRegs();
			});
		}
#endif
	}

	// One PS5 setting: (reg & ~mask) | (value & mask), logged with the old and new value when it changes.
	void Ps5Reg(const char* name, const std::string& where, uint32_t reg, uint32_t sel, uint32_t value, uint32_t mask) {
		uint32_t old = 0;
		uint32_t now = 0;
		if (m_ps5_hw.ReadReg(reg, sel, &old) == 0 && (old & mask) == (value & mask)) {
			return;
		}
		const int r = m_ps5_hw.WriteRegMasked(reg, sel, value, mask, &now);
		if (r != 0) {
			std::fprintf(stderr, "native shadow: PS5 register %s (%s) not written: %s\n", name, where.c_str(), std::strerror(-r));
			return;
		}
		std::fprintf(stderr, "native shadow: PS5 register %s (%s) 0x%08x -> 0x%08x%s\n", name, where.c_str(), old, now,
		             (now & mask) != (value & mask) ? " (does not hold)" : "");
	}

	// The per-VMID settings of one of the game's VMIDs. Callers hold m_ps5_mutex.
	void Ps5VmidRegs(uint32_t vmid) {
		const std::string where = "VMID " + std::to_string(vmid);
		if ((m_ps5_regs & kPs5Accum) != 0) {
			Ps5Reg("SPI_USER_ACCUM_VMID_CNTL", where, kPs5UserAccumVmid, LodStatsHw::SelVmid(vmid), 0xfu, 0xfu);
		}
		if ((m_ps5_regs & kPs5ShMem) != 0) {
			// PRIVATE_BASE 0x7000, SHARED_BASE 0x8000; amdgpu has 0x1000/0x2000 for gfx VMIDs, KFD 0x2/0x1.
			Ps5Reg("SH_MEM_BASES", where, kPs5ShMemBases, LodStatsHw::SelVmid(vmid), 0x80007000u, 0xffffffffu);
		}
	}

	// KFD's VMIDs (8-15) and, with hqd, its active compute queues. Runs again after each KFD queue is
	// created (a new runlist remaps the process, which can rewrite SH_MEM_*). Callers hold m_ps5_mutex.
	void Ps5KfdRegs() {
		if (m_ps5_regs == 0 || !m_kfd_queues) {
			return;
		}
		for (uint32_t vmid = 8; vmid < 16; vmid++) {
			if ((m_ps5_regs & kPs5KfdGds) != 0) {
				// GDS_GWS_VMIDn: BASE [5:0] 0, SIZE [22:16] 32. GDS_OA_VMIDn: MASK [15:0] counters 0-7.
				Ps5Reg("GDS_GWS_VMID", "VMID " + std::to_string(vmid), kPs5GdsGwsVmid0 + vmid, 0, 32u << 16u, 0xffffffffu);
				Ps5Reg("GDS_OA_VMID", "VMID " + std::to_string(vmid), kPs5GdsOaVmid0 + vmid, 0, 0xffu, 0xffffffffu);
			}
			Ps5VmidRegs(vmid);
		}
		if ((m_ps5_regs & kPs5Hqd) == 0) {
			return;
		}
		// The hardware queues KFD mapped: active, in a KFD VMID (amdgpu's own rings and the KIQ are VMID 0).
		for (uint32_t me = 1; me <= 2; me++) {
			for (uint32_t pipe = 0; pipe < 4; pipe++) {
				for (uint32_t queue = 0; queue < 8; queue++) {
					const uint32_t sel    = LodStatsHw::SelHqd(me, pipe, queue);
					uint32_t       active = 0;
					uint32_t       vmid   = 0;
					if (m_ps5_hw.ReadReg(kPs5HqdActive, sel, &active) != 0 || (active & 1u) == 0 ||
					    m_ps5_hw.ReadReg(kPs5HqdVmid, sel, &vmid) != 0 || (vmid & 0xfu) < 8) {
						continue;
					}
					char where[64];
					std::snprintf(where, sizeof(where), "ME%u pipe %u queue %u, VMID %u", me, pipe, queue, vmid & 0xfu);
					Ps5Reg("CP_HQD_PQ_CONTROL", where, kPs5HqdPqControl, sel, 1u << 29u, 1u << 29u); // TUNNEL_DISPATCH
					// PRELOAD_REQ, PRELOAD_SIZE 0x53, the WPP/IQ/IB/EOP/PQ switch QOS enables and
					// CACHE_FULL_PACKET_EN; KFD's QSWITCH_MODE (CWSR) stays.
					Ps5Reg("CP_HQD_PERSISTENT_STATE", where, kPs5HqdPersistentState, sel, 0x0be05301u, 0x0be0ff01u);
					// OA_REQUIRED, GWS_SIZE 48.
					Ps5Reg("CP_HQD_GDS_RESOURCE_STATE", where, kPs5HqdGdsResourceState, sel, 0x301u, 0x3f1u);
				}
			}
		}
	}

	// Deferred ACBs leave in order, once nothing they wait for is outstanding: a submitted DCB wrote
	// the label, Kyty already did (the writer has run), or 64 DCBs went by (a writer the translator
	// does not see; counted as unresolved). Labels Kyty has written stop counting as in flight.
	void ReleaseDeferredAcbsLocked() {
		std::erase_if(m_inflight_labels, [](const auto& e) {
			return *reinterpret_cast<const volatile uint32_t*>(e.first) == e.second;
		});
		while (!m_deferred_acbs.empty()) {
			DeferredAcb& d = m_deferred_acbs.front();
			std::erase_if(d.needs, [](const Pm4TranslateResult::UnmetWait& w) {
				const uint32_t v = *reinterpret_cast<const volatile uint32_t*>(w.addr);
				return WaitMet(w, v);
			});
			if (!d.needs.empty() && d.dcbs_since < 64) {
				break;
			}
			(d.needs.empty() ? m_acb_resolved : m_acb_unresolved)++;
			AppendToBatchLocked(d.ib);
			m_deferred_acbs.pop_front();
		}
	}

	static bool WaitMet(const Pm4TranslateResult::UnmetWait& w, uint64_t v) {
		switch (w.function) {
			case 1: return v < w.ref;
			case 2: return v <= w.ref;
			case 4: return v != w.ref;
			case 5: return v >= w.ref;
			case 6: return v > w.ref;
			default: return v == w.ref;
		}
	}

	void AppendToBatchLocked(const std::vector<uint32_t>& ib) {
		if ((m_batch.size() + ib.size()) * 4 > NativeSubmitter::kSlotSize) {
			SubmitBatchLocked();
		}
		m_batch.insert(m_batch.end(), ib.begin(), ib.end());
	}

	void SubmitBatchLocked() { SubmitBufferLocked(m_batch, 0); }

#if KYTY_GEM_BACKING_SUPPORTED
	// Slot 0 is the gfx ring, kGfx1Slot the second gfx ring, any other slot the compute ring `slot - 1`.
	static NativeSubmitter::RingTarget SlotTarget(uint32_t slot) {
		NativeSubmitter::RingTarget target;
		if (slot == kGfx1Slot) {
			target.ring = 1;
		} else if (slot != 0) {
			target.ip_type = AMDGPU_HW_IP_COMPUTE;
			target.ring    = slot - 1;
		}
		return target;
	}
#endif

	void SubmitBufferLocked(std::vector<uint32_t>& batch, uint32_t slot) {
#if KYTY_GEM_BACKING_SUPPORTED
		if (batch.empty() || m_disabled || m_submitter == nullptr) {
			batch.clear();
			return;
		}
		const NativeSubmitter::RingTarget target = SlotTarget(slot);
		std::vector<uint32_t>   handles = m_gem->CreatedHandles();
		NativeSubmitter::Result r       = m_submitter->Submit(batch, handles, m_timeout_sec, target);
		if (!m_hang_dump.empty() && r.submitted && slot < kTrackerSlots) {
			// Older submissions on the gfx ring are large and a hang there is caught early; a hung compute
			// ring keeps receiving work behind the stuck one, so it keeps many more.
			auto& kept = m_hang_kept[slot];
			kept.emplace_back(r.seq, batch);
			while (kept.size() > (GfxSlot(slot) ? 8u : 4096u)) {
				kept.pop_front();
			}
		}
		for (int retry = 0; !r.submitted && r.err == -ENOENT && retry < 3; retry++) {
			handles = m_gem->CreatedHandles();
			r       = m_submitter->Submit(batch, handles, m_timeout_sec, target);
		}
		batch.clear();
		if (!r.submitted) {
			std::fprintf(stderr, "native shadow: batch submit failed: %s (%zu BOs, ip %u ring %u)\n", std::strerror(-r.err),
			             handles.size(), target.ip_type, target.ring);
			Disable("submit error");
			return;
		}
		m_cs_ms_total += r.cs_seconds * 1e3;
		m_last_seq = r.seq;
		if (slot < kTrackerSlots) {
			m_slot_fence[slot] = SlotFence {r.seq, target.ip_type, target.ring};
		}
#else
		(void)batch;
		(void)slot;
#endif
	}

	// KYTY_NATIVE_GPU_ACB: "1" runs async compute buffers natively (and Kyty skips their dispatches),
	// "dry" only translates them and reports what the translator lacks.
	enum class AcbMode { Off, Dry, On };

	// Dry mode: count and report, never submit. Returns true when the ACB is done with.
	bool SkipAcb(uint64_t n, const Pm4TranslateResult& result) {
		if (m_acb_mode != AcbMode::Dry) {
			return false;
		}
		if (result.ok) {
			m_acbs_completed++;
		} else {
			m_acbs_refused++;
			for (uint32_t op: result.unknown_ops) {
				if (m_unknown_acb_ops.insert(op).second) {
					std::fprintf(stderr, "native shadow: acb dry run: untranslatable packet op 0x%02x (first in acb %llu)\n",
					             op, static_cast<unsigned long long>(n));
				}
			}
		}
		if (n < 3 || n % 600 == 0) {
			std::fprintf(stderr, "native shadow: acb dry run: acb %llu %s, %u dispatches; %llu ok, %llu refused so far\n",
			             static_cast<unsigned long long>(n), result.ok ? "ok" : "refused", result.stats.dispatches,
			             static_cast<unsigned long long>(m_acbs_completed), static_cast<unsigned long long>(m_acbs_refused));
		}
		return true;
	}

	// Watchdog (KYTY_NATIVE_GPU_WATCHDOG=1): every 2 s it checks whether the shadow path is stuck inside a
	// submission (which stage, for how long) or whether the game has stopped submitting DCBs at all, so an
	// intermittent freeze shows where it sits without a debugger.
	enum class Stage : int { Idle = 0, Translate = 1, Residency = 2, Gpu = 3 };
	void SetStage(Stage stage, uint64_t frame) {
		m_stage.store(static_cast<int>(stage), std::memory_order_relaxed);
		m_stage_frame.store(frame, std::memory_order_relaxed);
		m_stage_since.store(NowSecShadow(), std::memory_order_relaxed);
	}
	void StartWatchdog() {
		std::thread([this] {
			static const char* const kNames[] = {"idle (waiting for the next DCB)", "translate", "residency", "GPU submit/wait"};
			double reported = -1.0;
			for (;;) {
				struct timespec nap = {2, 0};
				nanosleep(&nap, nullptr);
				const double since = m_stage_since.load(std::memory_order_relaxed);
				const double stuck = NowSecShadow() - since;
				if (since > 0.0 && stuck >= 10.0 && since != reported) {
					reported = since;
					const int stage = m_stage.load(std::memory_order_relaxed);
					std::fprintf(stderr, "native shadow: WATCHDOG: %.0f s in stage %s at frame %llu (promotion in progress: %s)\n",
					             stuck, kNames[stage], static_cast<unsigned long long>(m_stage_frame.load()),
					             PromotionInProgress() ? "yes" : "no");
				}
			}
		}).detach();
	}
	static bool PromotionInProgress() {
#if KYTY_GEM_BACKING_SUPPORTED
		return GemBacking::WaitIfPromotingWouldWait();
#else
		return false;
#endif
	}
	std::atomic<int>      m_stage {0};
	std::atomic<uint64_t> m_stage_frame {0};
	std::atomic<double>   m_stage_since {0.0};

#if KYTY_GEM_BACKING_SUPPORTED
	// KYTY_NATIVE_GPU_BREADCRUMBS: a GPU hang on this board can take the whole machine down before
	// any log reaches the disk, so before each submission the guest DCB, the IB about to run and a
	// summary are written and synced: after a power cycle, last.txt names the submission that hung
	// and dcb_<n>.bin / ib_<n>.bin (the last four, n = frame % 4) hold it.
	// Control socket "crumbs N" (DemandedCrumbs): this buffer and its IB, while the count lasts.
	static void WriteDemandedCrumb(bool acb, uint64_t frame, std::span<const uint32_t> dcb, std::span<const uint32_t> ib) {
		if (uint32_t left = DemandedCrumbs().load(std::memory_order_relaxed);
		    left != 0 && DemandedCrumbs().compare_exchange_strong(left, left - 1)) {
			const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
			const std::string base = std::string(dir != nullptr && dir[0] != '\0' ? dir : ".") + "/" +
			                         (acb ? "acb" : "dcb") + std::to_string(frame);
			if (std::FILE* f = std::fopen((base + ".dcb.bin").c_str(), "wb"); f != nullptr) {
				std::fwrite(dcb.data(), 4, dcb.size(), f);
				std::fclose(f);
			}
			if (std::FILE* f = std::fopen((base + ".ib.bin").c_str(), "wb"); f != nullptr) {
				std::fwrite(ib.data(), 4, ib.size(), f);
				std::fclose(f);
			}
	}
	}

	void WriteBreadcrumb(const std::string& prefix, uint64_t frame, std::span<const uint32_t> dcb,
	                     const Pm4TranslateResult& result, std::span<const uint32_t> ib) const {
		auto write_file = [&](const std::string& name, const void* data, size_t bytes) {
			const std::string path = m_breadcrumbs + "/" + name;
			const int         fd   = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
			if (fd < 0) {
				return;
			}
			const auto* p = static_cast<const char*>(data);
			while (bytes > 0) {
				const ssize_t n = ::write(fd, p, bytes);
				if (n <= 0) {
					break;
				}
				p += n;
				bytes -= static_cast<size_t>(n);
			}
			::fdatasync(fd);
			::close(fd);
		};
		// KYTY_NATIVE_GPU_BREADCRUMB_SLOTS: how many recent submissions to keep (default 4).
		static const unsigned slots = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_BREADCRUMB_SLOTS");
			const unsigned long n = v != nullptr ? std::strtoul(v, nullptr, 0) : 4ul;
			return static_cast<unsigned>(n == 0 ? 4ul : n);
		}();
		const unsigned slot = static_cast<unsigned>(frame % slots);
		write_file(prefix + "dcb_" + std::to_string(slot) + ".bin", dcb.data(), dcb.size_bytes());
		write_file(prefix + "ib_" + std::to_string(slot) + ".bin", ib.data(), ib.size_bytes());
		char summary[512];
		const int n = std::snprintf(summary, sizeof(summary),
		                            "%s %llu slot %u: dcb %zu dw, ib %zu dw, %u draws, %u dispatches, %u dma, "
		                            "%u set_base, %llu MiB resident\n",
		                            prefix.empty() ? "frame" : prefix.c_str(), static_cast<unsigned long long>(frame), slot, dcb.size(), ib.size(),
		                            result.stats.draws, result.stats.dispatches, result.stats.dma_data,
		                            result.stats.set_base, static_cast<unsigned long long>(m_gem->ResidentBytes() >> 20u));
		write_file(prefix + "last.txt", summary, static_cast<size_t>(n));
	}

	// Makes resident, before `result`'s IB runs: every block the CPU has written, every shader, and
	// every reference the translator reported that lies in guest direct memory. Pointer candidates
	// from shader user data are followed two levels deep (resource tables, and tables of tables),
	// scanning each table for 48-bit buffer pointers (V#, whose dwords 1-2 give stride and record
	// count, so the whole buffer is promoted) and 40-bit image bases (T#).
	// KYTY_NATIVE_GPU_LOG_UNMAPPED=1: report addresses found in shader user data and resource tables
	// that fall in the guest address range but are mapped for the GPU nowhere (a GPU access there is a
	// page fault, which hangs this board). Candidates are heuristic, so some reports are just data.
	void NoteUnmapped(uint64_t addr, const char* kind, uint64_t table) {
		if (!m_log_unmapped || addr < 0x100000000ull || addr >= 0x10000000000ull || m_unmapped_seen.size() >= 4096 ||
		    !m_unmapped_seen.insert(addr >> 16u).second || m_gem->IsGpuMapped(addr)) {
			return;
		}
		if (m_unmapped_logged++ < 40) {
			std::fprintf(stderr, "native shadow: %s 0x%llx (from table 0x%llx) is not mapped for the GPU\n", kind,
			             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(table));
		}
	}

	// KYTY_NATIVE_GPU_INSPECT_PS: for a few draws with that pixel shader, log both stages' user data and
	// decode the descriptors around them: inline V#s in the user data, and V#/T# in the 0x200 bytes
	// behind every user-data pointer, each with whether its base is mapped for the GPU.
	void InspectDraw(uint64_t frame, const DrawSnapshot& snap) {
		if (m_inspected++ >= 40) {
			return;
		}
		auto mapped = [&](uint64_t a) { return a == 0 ? "null" : (m_gem->IsGpuMapped(a) ? "mapped" : "NOT MAPPED"); };
		std::fprintf(stderr, "native shadow: INSPECT frame %llu ps 0x%llx es 0x%llx\n", static_cast<unsigned long long>(frame),
		             static_cast<unsigned long long>(snap.ps), static_cast<unsigned long long>(snap.es));
		auto dump_stage = [&](const char* stage, const uint32_t* ud) {
			for (uint32_t i = 0; i < 32; i++) {
				std::fprintf(stderr, "%s%s ud[%2u]=0x%08x", i % 8 == 0 ? "\n  " : " ", stage, i, ud[i]);
			}
			std::fprintf(stderr, "\n");
			for (uint32_t i = 0; i + 1 < 32; i++) {
				const uint64_t ptr = ud[i] | (static_cast<uint64_t>(ud[i + 1] & 0xffffu) << 32u);
				if (ptr < 0x100000000ull || ptr >= 0x10000000000ull) {
					continue;
				}
				std::fprintf(stderr, "  %s ud[%u:%u] -> 0x%llx (%s)\n", stage, i, i + 1, static_cast<unsigned long long>(ptr), mapped(ptr));
				uint32_t     t[128];
				const size_t n = m_gem->ReadGuest(ptr, t, sizeof(t)) / 4;
				for (size_t k = 0; k + 3 < n; k += 4) {
					const uint64_t vbase = t[k] | (static_cast<uint64_t>(t[k + 1] & 0xffffu) << 32u);
					const uint64_t tbase = (t[k] | (static_cast<uint64_t>(t[k + 1] & 0xffu) << 32u)) << 8u;
					if ((vbase >= 0x100000000ull && vbase < 0x10000000000ull) || (tbase >= 0x100000000ull && tbase < 0x10000000000ull)) {
						std::fprintf(stderr, "    +0x%03zx: %08x %08x %08x %08x  as V# 0x%llx (%s) stride %u records %u | as T# 0x%llx (%s)\n",
						             k * 4, t[k], t[k + 1], t[k + 2], t[k + 3], static_cast<unsigned long long>(vbase), mapped(vbase),
						             (t[k + 1] >> 16u) & 0x3fffu, t[k + 2], static_cast<unsigned long long>(tbase), mapped(tbase));
						// An image descriptor (TYPE 8 and up in word 3): its second half and fields.
						if ((t[k + 3] >> 28u) >= 8u && k + 7 < n) {
							std::fprintf(stderr, "      T# %08x %08x %08x %08x: %ux%u format %u swizzle %u dw6 0x%08x meta 0x%llx\n",
							             t[k + 4], t[k + 5], t[k + 6], t[k + 7], ((t[k + 1] >> 30u) | ((t[k + 2] & 0x3fffu) << 2u)) + 1u,
							             ((t[k + 2] >> 14u) & 0xffffu) + 1u, (t[k + 1] >> 20u) & 0x1ffu, (t[k + 3] >> 20u) & 0x1fu,
							             t[k + 6], static_cast<unsigned long long>(((static_cast<uint64_t>(t[k + 7]) << 8u) | (t[k + 6] >> 24u)) << 8u));
						}
						// Small buffers (constants, exposure and other per-frame scalars): their first values.
						const uint32_t stride = (t[k + 1] >> 16u) & 0x3fffu;
						const uint64_t bytes  = static_cast<uint64_t>(t[k + 2]) * (stride != 0 ? stride : 1u);
						if (vbase >= 0x100000000ull && vbase < 0x10000000000ull && bytes != 0 && bytes <= 4096) {
							float f[8] = {};
							const size_t got = m_gem->ReadGuest(vbase, f, std::min<uint64_t>(sizeof(f), bytes)) / 4;
							std::fprintf(stderr, "      V# 0x%llx data:", static_cast<unsigned long long>(vbase));
							for (size_t q = 0; q < got; q++) {
								std::fprintf(stderr, " %g", static_cast<double>(f[q]));
							}
							std::fprintf(stderr, "\n");
						}
					}
				}
			}
		};
		dump_stage("ps", snap.ps_user_data);
		dump_stage("gs", snap.gs_user_data);
	}

	// Per-draw references are collected only while map-time residency is incomplete (see
	// GemBacking::ResidencyAtMap): once a block stayed out of it, submissions make memory resident.
	void SyncCollectRefs() {
#if KYTY_GEM_BACKING_SUPPORTED
		const bool collect = m_gem == nullptr || !m_gem->MapComplete();
		if (collect == m_collect_refs) {
			return;
		}
		m_collect_refs = collect;
		m_translator.SetCollectRefs(collect);
		m_acb_translator.SetCollectRefs(collect);
		std::fprintf(stderr, collect ? "native shadow: residency by reference: per-draw references and the CPU-write scan are on\n"
		                             : "native shadow: residency at map time (PS5): no per-draw references, no CPU-write scan\n");
#endif
	}

	void MakeResident(const Pm4TranslateResult& result) {
		constexpr uint64_t kTableSpan = 4096;
		constexpr uint64_t kImageSpan = 32ull << 20u;
		constexpr uint64_t kBufferCap = 64ull << 20u;
		constexpr size_t   kMaxTables = 2048;
		constexpr uint32_t kBucketShift = 24; // 16 MiB guest-address buckets, the block size
		if (m_gem->MapComplete()) {
			return; // mappings with GPU access were made resident when the guest mapped them
		}
		m_promoted_by[0] += m_gem->PromoteTouched();

		// Membership tests run thousands of times per frame: binary search over a cached copy of the
		// view ranges, refreshed when the views change (which also forgets which buckets are known
		// to be resident).
		if (const uint64_t gen = m_gem->ViewsGeneration(); gen != m_views_generation) {
			m_view_ranges      = m_gem->ViewRanges();
			m_views_generation = gen;
			m_resident_buckets.clear();
		}
		auto backed = [&](uint64_t addr) {
			auto it = std::upper_bound(m_view_ranges.begin(), m_view_ranges.end(),
			                           std::make_pair(addr, ~0ull));
			return it != m_view_ranges.begin() && addr < std::prev(it)->second;
		};
		// `kind` counts which reference made blocks resident (m_promoted_by, logged with the averages).
		auto promote = [&](uint64_t addr, uint64_t size, int kind) {
			for (uint64_t b = addr >> kBucketShift; b <= (addr + size - 1) >> kBucketShift; b++) {
				if (m_resident_buckets.insert(b).second) {
					m_promoted_by[kind] += m_gem->PromoteGuestRange(b << kBucketShift, 1ull << kBucketShift);
				}
			}
		};

		for (uint64_t addr: result.shader_addrs) {
			if (backed(addr)) {
				promote(addr, 64u * 1024u, 1);
			}
		}
		for (uint32_t b: result.ref_buckets) {
			promote(static_cast<uint64_t>(b) << kBucketShift, 1ull << kBucketShift, 3);
		}
		std::vector<uint64_t> tables;
		for (const auto& ref: result.refs) {
			if (ref.pointer) {
				NoteUnmapped(ref.addr, "user-data pointer", 0);
			}
			// An index buffer the GPU cannot reach at all is a page fault, and a fault hangs this board.
			if (ref.index && !backed(ref.addr) && !m_gem->IsGpuMapped(ref.addr) && m_unmapped_logged++ < 16) {
				std::fprintf(stderr, "native shadow: index buffer 0x%llx (0x%llx bytes) is not mapped for the GPU\n",
				             static_cast<unsigned long long>(ref.addr), static_cast<unsigned long long>(ref.size));
			}
			if (backed(ref.addr)) {
				promote(ref.addr, ref.size != 0 ? ref.size : kTableSpan, ref.pointer ? 2 : 3);
				if (ref.pointer) {
					tables.push_back(ref.addr);
				}
			}
		}
		// Following descriptor tables guesses: any dword pair that looks like a guest address makes its
		// whole block resident. In ASTRO BOT that was 336 of 712 blocks (5.4 GB) and nothing needed them
		// (what the GPU reads was already resident through CPU writes and direct references), so tables
		// are only followed with KYTY_NATIVE_GPU_TABLE_PROMOTE=1. A GPU write into memory left out still
		// shows as a dummy write in the log.
		static const bool follow_tables = EnvOn("KYTY_NATIVE_GPU_TABLE_PROMOTE");
		if (!follow_tables) {
			tables.clear();
		}
		std::set<uint64_t> seen;
		for (int depth = 0; depth < 2 && !tables.empty(); depth++) {
			std::vector<uint64_t> next;
			for (uint64_t table: tables) {
				if (seen.size() >= kMaxTables || !seen.insert(table).second) {
					continue;
				}
				uint32_t     d[64];
				const size_t n = m_gem->ReadGuest(table, d, sizeof(d)) / 4;
				for (size_t i = 0; i + 1 < n; i++) {
					const uint64_t ptr = d[i] | (static_cast<uint64_t>(d[i + 1] & 0xffffu) << 32u);
					NoteUnmapped(ptr, "buffer", table);
					if (ptr != 0 && backed(ptr)) {
						const uint64_t stride  = (d[i + 1] >> 16u) & 0x3fffu;
						const uint64_t records = i + 2 < n ? d[i + 2] : 0;
						const uint64_t bytes   = stride != 0 ? stride * records : records;
						promote(ptr, std::clamp<uint64_t>(bytes, kTableSpan, kBufferCap), 4);
						next.push_back(ptr);
					}
					const uint64_t image = (d[i] | (static_cast<uint64_t>(d[i + 1] & 0xffu) << 32u)) << 8u;
					NoteUnmapped(image, "image", table);
					if (image != 0 && backed(image)) {
						promote(image, kImageSpan, 5);
					}
				}
			}
			tables = std::move(next);
		}
	}
#endif

	// Pipeline statistics: 11 64-bit counters per sample on GFX10, in hardware slot order
	// (RADV pipeline_statistics_indices): PS invocations, clipper primitives, clipper invocations,
	// VS invocations, GS invocations, GS primitives, IA primitives, IA vertices, HS, DS, CS.
	static constexpr uint32_t kStatSlots              = 11;
	static constexpr uint32_t kItEventWrite           = 0x46;
	static constexpr uint32_t kEventPipelineStatStart = 0x19;
	static constexpr uint32_t kEventSamplePipelineStat = 0x1e | (2u << 8u); // EVENT_INDEX 2

	static uint32_t StatPkt3(uint32_t op, uint32_t body_dw) {
		return (3u << 30u) | (((body_dw - 1u) & 0x3fffu) << 16u) | (op << 8u);
	}

	// COPY_DATA SRC_SEL 9 (GPU clock, gpu_counter_freq = 100 MHz on BC-250) -> memory, 64 bit.
	static constexpr uint32_t kItCopyData       = 0x40;
	static constexpr uint32_t kTimestampOffset  = 512;
	// KYTY_NATIVE_GPU_FLIP_TRACE: the GPU clock pair rings past the 4 KiB of scratch slots.
	static constexpr uint64_t kGpuTimesOffset = 4096;
	static constexpr uint32_t kGpuTimesPairs  = 2048;
	// Native labels: the 32-bit slot tracker points write and, 8 bytes on, the dword flush-only releases
	// write (the fences are at 1024 and 1032; with ACB rings each ring's is its slot's last dword). ACB
	// rings give each ring its own slot, stride 16.
	// Slot 0 is the gfx ring; slot n > 0 is compute ring n - 1. 33 slots fit in the scratch page.
	static constexpr uint32_t kTrackerOffset    = 1040;
	static constexpr uint32_t kTrackerStride    = 16;
	static constexpr uint32_t kTrackerSlots     = 33;
	static constexpr double   kGpuClockHz       = 100e6;

	static void EmitTimestamp(std::vector<uint32_t>& ib, uint64_t va) {
		ib.push_back(StatPkt3(kItCopyData, 5));
		ib.push_back(9u | (5u << 8u) | (1u << 16u) | (1u << 20u)); // SRC TIMESTAMP, DST MEM, 64-bit, confirm
		ib.push_back(0);
		ib.push_back(0);
		ib.push_back(static_cast<uint32_t>(va));
		ib.push_back(static_cast<uint32_t>(va >> 32u));
	}

	double GpuMs() const {
#if KYTY_GEM_BACKING_SUPPORTED
		const volatile uint8_t* base  = m_submitter->ScratchCpu() + kTimestampOffset;
		uint64_t                begin = 0;
		uint64_t                end   = 0;
		for (uint32_t b = 0; b < 8; b++) {
			begin |= static_cast<uint64_t>(base[b]) << (8u * b);
			end |= static_cast<uint64_t>(base[8 + b]) << (8u * b);
		}
		return end > begin ? static_cast<double>(end - begin) * 1e3 / kGpuClockHz : 0.0;
#else
		return 0.0;
#endif
	}

	static void EmitSample(std::vector<uint32_t>& ib, uint64_t va) {
		ib.push_back(StatPkt3(kItEventWrite, 3));
		ib.push_back(kEventSamplePipelineStat);
		ib.push_back(static_cast<uint32_t>(va));
		ib.push_back(static_cast<uint32_t>(va >> 32u));
	}

	void LogStats(uint64_t frame, uint32_t draws) const {
#if KYTY_GEM_BACKING_SUPPORTED
		static const char* const kNames[kStatSlots] = {"ps", "c_prim", "c_inv", "vs", "gs", "gs_prim",
		                                               "ia_prim", "ia_vert", "hs", "ds", "cs"};
		const volatile uint8_t*  base               = m_submitter->ScratchCpu();
		std::fprintf(stderr, "native shadow: frame %llu pipeline stats (%u draws):",
		             static_cast<unsigned long long>(frame), draws);
		for (uint32_t i = 0; i < kStatSlots; i++) {
			uint64_t begin = 0;
			uint64_t end   = 0;
			for (uint32_t b = 0; b < 8; b++) {
				begin |= static_cast<uint64_t>(base[i * 8 + b]) << (8u * b);
				end |= static_cast<uint64_t>(base[kStatSlots * 8 + i * 8 + b]) << (8u * b);
			}
			std::fprintf(stderr, " %s=%llu", kNames[i], static_cast<unsigned long long>(end - begin));
		}
		std::fprintf(stderr, "\n");
#else
		(void)frame;
		(void)draws;
#endif
	}

	void DumpRegs(uint64_t frame) const {
		const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
		char        path[512];
		std::snprintf(path, sizeof(path), "%s/shadow_regs_%llu.txt",
		              (dir != nullptr && dir[0] != '\0') ? dir : ".", static_cast<unsigned long long>(frame));
		if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
			m_translator.DumpShadow(f);
			std::fclose(f);
			std::fprintf(stderr, "native shadow: dumped shadow registers of frame %llu to %s\n",
			             static_cast<unsigned long long>(frame), path);
		}
	}

	// The first native CS pays a large one-off cost while amdgpu validates the userptr ranges and
	// places the freshly created GTT chunks, so the wait must outlast it or a
	// successful-but-slow submit is misread as a hang. Overridable with KYTY_NATIVE_GPU_TIMEOUT_SEC.
	static constexpr double kDefaultTimeoutSec = 12.0;

	// libSceAgc runs its context start buffer once, from AgcInit: every register of the register
	// defaults of the version the title passed (public then internal; context, SH, UCONFIG). Called
	// before each graphics translation until that version is known. KYTY_NATIVE_GPU_AGC_DEFAULTS=v11ctx
	// keeps only the version 11 public context table.
	void LoadAgcStartState() {
		if (m_agc_start_loaded || EnvOn("KYTY_NATIVE_GPU_NO_PS5_DEFAULTS")) {
			return;
		}
		static const bool v11ctx = [] {
			const char* mode = std::getenv("KYTY_NATIVE_GPU_AGC_DEFAULTS");
			return mode != nullptr && std::strcmp(mode, "v11ctx") == 0;
		}();
		const uint32_t ver = Gen5::AgcRegisterDefaultsVersion();
		if (v11ctx || ver == UINT32_MAX) {
			return;
		}
		m_agc_start_loaded     = true;
		const auto     state   = Gen5::AgcStartStateDefaults(ver);
		const uint32_t skipped = m_translator.LoadStartState(state.context, state.sh, state.uconfig);
		std::fprintf(stderr,
		             "native shadow: AgcInit start state, register defaults version %u (title asked for %u): %zu context, %zu SH, "
		             "%zu UCONFIG (%u UCONFIG left out)\n",
		             state.version, ver, state.context.size(), state.sh.size(), state.uconfig.size() - skipped, skipped);
	}

	void Init() {
		if (!EnvOn("KYTY_NATIVE_GPU_SHADOW") && !EnvOn("KYTY_NATIVE_GPU_PRIMARY")) {
			return;
		}
		m_translator.SeedPs5Defaults();
		// The rest of the PS5 context state the system sets up before a title runs: the graphics
		// library's register defaults. amdgpu's CLEAR_STATE leaves many of these at other values
		// (rasterizer and depth controls, scissors, guard band, blend and DCC defaults).
		// Until the title calls AgcInit, the version 11 public context table stands in; LoadAgcStartState
		// then loads what libSceAgc's start buffer loads at that point.
		if (!EnvOn("KYTY_NATIVE_GPU_NO_PS5_DEFAULTS")) {
			const auto defaults = Gen5::AgcContextRegisterDefaults();
			m_translator.SeedContextDefaults(defaults);
			std::fprintf(stderr, "native shadow: seeded %zu PS5 context register defaults (version 11 public)\n", defaults.size());
		}
		LoadAgcStartState();
		m_stats = EnvOn("KYTY_NATIVE_GPU_STATS");
		m_dump_shaders = EnvOn("KYTY_NATIVE_GPU_DUMP_SHADERS");
		m_log_unmapped = EnvOn("KYTY_NATIVE_GPU_LOG_UNMAPPED");
		if (EnvOn("KYTY_NATIVE_GPU_NO_DCC")) {
			m_translator.SetNoDcc(true);
			std::fprintf(stderr, "native shadow: render targets without DCC (experiment)\n");
		}
		if (PrimaryFeature("KYTY_NATIVE_GPU_NO_CMASK_CLEAR")) {
			// Only MRT4 by default: ASTRO BOT's CMASK-only fast clear there hangs the CB. Turning it off
			// everywhere leaves other fast-cleared targets stale (8x8 squares around moving objects).
			const char*    v     = std::getenv("KYTY_NATIVE_GPU_NO_CMASK_CLEAR_SLOTS");
			const uint32_t slots = v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0x10u;
			m_translator.SetNoCmaskClear(true, slots);
			std::fprintf(stderr, "native shadow: no CMASK fast clear on targets without DCC in MRT slots 0x%02x\n", slots);
		}
		if (const char* mask = std::getenv("KYTY_NATIVE_GPU_TARGET_MASK"); mask != nullptr) {
			m_translator.SetTargetMask(static_cast<uint32_t>(std::strtoul(mask, nullptr, 0)));
			std::fprintf(stderr, "native shadow: CB_TARGET_MASK limited to %s (experiment)\n", mask);
		}
		// KYTY_NATIVE_GPU_INSPECT_PS=0xADDR[,0xADDR...]: pixel or compute shaders whose draws and dispatches
		// InspectDraw logs.
		if (const char* list = std::getenv("KYTY_NATIVE_GPU_INSPECT_PS"); list != nullptr) {
			std::set<uint64_t> shaders;
			for (const char* p = list; *p != '\0';) {
				char* end = nullptr;
				shaders.insert(std::strtoull(p, &end, 0));
				if (end == p) {
					break;
				}
				p = (*end == ',') ? end + 1 : end;
			}
			m_translator.SetInspectedShaders(shaders);
		}
		// KYTY_NATIVE_GPU_SKIP_PS=0xADDR[,0xADDR...]: leave draws with these pixel shaders out of the IB.
		if (const char* list = std::getenv("KYTY_NATIVE_GPU_SKIP_PS"); list != nullptr) {
			for (const char* p = list; *p != '\0';) {
				char* end = nullptr;
				m_skip_ps.insert(std::strtoull(p, &end, 0));
				if (end == p) {
					break;
				}
				p = (*end == ',') ? end + 1 : end;
			}
			m_translator.SetSkippedPixelShaders(m_skip_ps);
			std::fprintf(stderr, "native shadow: leaving draws with %zu pixel shader(s) out: %s\n", m_skip_ps.size(), list);
		}
		if (const char* ops = std::getenv("KYTY_NATIVE_GPU_SKIP_OPS"); ops != nullptr) {
			std::set<uint32_t> skip;
			for (const char* p = ops; *p != '\0';) {
				char* end = nullptr;
				skip.insert(static_cast<uint32_t>(std::strtoul(p, &end, 0)));
				p = (*end == ',') ? end + 1 : end;
				if (end == p && *p != '\0') {
					break;
				}
			}
			m_translator.SetSkippedOpcodes(skip);
			m_acb_translator.SetSkippedOpcodes(skip);
			std::fprintf(stderr, "native shadow: leaving %zu packet type(s) out of the IB: %s\n", skip.size(), ops);
		}
		// KYTY_NATIVE_GPU_GDS_SHADERS=1 runs draws with GDS shaders natively, =skip leaves just those
		// draws out of the IB (the rest of their DCB runs natively); unset leaves whole DCBs to Kyty.
		if (const char* g = std::getenv("KYTY_NATIVE_GPU_GDS_SHADERS"); g != nullptr) {
			m_skip_gds_shaders  = std::strcmp(g, "skip") == 0;
			m_allow_gds_shaders = !m_skip_gds_shaders && g[0] == '1';
		} else {
			m_allow_gds_shaders = EnvOn("KYTY_NATIVE_GPU_PRIMARY"); // primary mode has the GDS partition
		}
		m_collect_targets = std::getenv("KYTY_NATIVE_GPU_DUMP_FLIPS") != nullptr || EnvOn("KYTY_NATIVE_GPU_CTL");
		if (EnvOn("KYTY_NATIVE_GPU_CTL")) {
			StartControlSocket();
			m_translator.SetDrawLog(true);
			m_acb_translator.SetDrawLog(true);
		}
		if (EnvOn("KYTY_NATIVE_GPU_WATCHDOG")) {
			StartWatchdog();
		}
		if (const char* dir = std::getenv("KYTY_NATIVE_GPU_HANG_DUMP"); dir != nullptr && dir[0] != '\0') {
			m_hang_dump = dir;
			std::fprintf(stderr, "native shadow: unfinished submissions are written to %s on a ring timeout\n", dir);
		}
		if (const char* dir = std::getenv("KYTY_NATIVE_GPU_BREADCRUMBS"); dir != nullptr && dir[0] != '\0') {
			m_breadcrumbs = dir;
			std::fprintf(stderr, "native shadow: breadcrumbs (every submission synced to disk) in %s\n", dir);
		}

		if (const char* d = std::getenv("KYTY_NATIVE_GPU_DUMP_REGS"); d != nullptr && d[0] != '\0') {
			m_dump_regs_frame = std::atoll(d);
		}
		if (const char* t = std::getenv("KYTY_NATIVE_GPU_TIMEOUT_SEC"); t != nullptr && t[0] != '\0') {
			const double v = std::atof(t);
			if (v > 0.0) {
				m_timeout_sec = v;
			}
		}
		if (!EnvOn("KYTY_NATIVE_GPU_GEM")) {
			std::fprintf(stderr,
			             "native shadow: KYTY_NATIVE_GPU_SHADOW set but KYTY_NATIVE_GPU_GEM is not; "
			             "native path needs GEM-backed guest memory. Staying off.\n");
			return;
		}
#if KYTY_GEM_BACKING_SUPPORTED
		m_gem = LibKernel::Memory::GetGemBacking();
		if (m_gem == nullptr) {
			std::fprintf(stderr, "native shadow: GEM backing unavailable; staying off.\n");
			return;
		}
		m_submitter = NativeSubmitter::Create(m_gem->DrmFd(), m_gem);
		if (m_submitter == nullptr) {
			std::fprintf(stderr, "native shadow: submitter init failed; staying off.\n");
			return;
		}
		InitLodStatsHw();
		InitPs5Regs();
		InstallStopDrain();
		m_translator.SetGdsAvailable(m_submitter->HasGds());
		m_acb_translator.SetGdsAvailable(m_submitter->HasGds());
		m_acb_translator.SetComputeQueue(true);
		// Residency only needs the 16 MiB buckets the GPU may touch, unless descriptor tables are
		// followed or unmapped references logged: those need every reference.
		if (!EnvOn("KYTY_NATIVE_GPU_TABLE_PROMOTE") && !m_log_unmapped) {
			m_translator.SetCompactRefs(true);
			m_acb_translator.SetCompactRefs(true);
		}
		SyncCollectRefs();
		// The guest's WAIT_REG_MEM / WAIT_REG_MEM_64 run as written on the gfx ring (probe waitforms: the
		// BC-250's CP runs them as the PS5's does). KYTY_NATIVE_GPU_WAITS_VERBATIM=0 emits our form.
		if (!EnvIs("KYTY_NATIVE_GPU_WAITS_VERBATIM", "0")) {
			m_translator.SetWaitsVerbatim(true);
		}
		if (EnvOn("KYTY_NATIVE_GPU_PFP_SYNC_INDIRECT")) {
			m_translator.SetPfpSyncIndirect(true);
			m_acb_translator.SetPfpSyncIndirect(true);
			std::fprintf(stderr, "native shadow: PFP_SYNC_ME before indirect draws and dispatches\n");
		}
		// KYTY_NATIVE_GPU_WRITE_DATA_NATIVE=1|2: WRITE_DATA also runs in the native IB (see SetWriteDataNative).
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_WRITE_DATA_NATIVE"); v != nullptr && v[0] != '0') {
			m_translator.SetWriteDataNative(static_cast<uint32_t>(std::strtoul(v, nullptr, 0)));
			m_acb_translator.SetWriteDataNative(static_cast<uint32_t>(std::strtoul(v, nullptr, 0)));
			std::fprintf(stderr, "native shadow: WRITE_DATA runs natively too (mode %s)\n", v);
		}
		if (NativeLodStats()) {
			m_translator.SetLodStats(true);
			m_acb_translator.SetLodStats(true);
			std::fprintf(stderr, "native shadow: GET_LOD_STATS runs on the GPU (experiment)\n");
		}
		m_acb_defer = EnvOn("KYTY_NATIVE_GPU_ACB_DEFER");
		// The PS5 kernel's RCIU CAM entry 19 (on in primary mode; KYTY_NATIVE_GPU_PS5_REG_OVERRIDES=0 off).
		// Entry 18 (UCONFIG 0xc035 bit 0) and the per-process UCONFIG 0xc035..0xc03c are Oberon registers
		// gfx1013 does not define, so nothing is written for them.
		// GFX stays powered between submissions, as a PS5 title's GPU does: with GFXOFF the gfx block powers
		// down whenever the ring idles, and ASTRO BOT's heavy view missed five times as many refreshes (13.9
		// late frames in 120 instead of 2.8). Through the kyty-lodhw helper (root), which lets GFXOFF back
		// on when Kyty exits; KYTY_NATIVE_GPU_GFXOFF=1 leaves it on.
		if (!EnvOn("KYTY_NATIVE_GPU_GFXOFF")) {
			const int r = m_rciu_hw.On() || m_rciu_hw.Connect() ? m_rciu_hw.KeepGfxPowered() : -ENOTCONN;
			std::fprintf(stderr, "native shadow: GFXOFF %s\n", r == 0 ? "off while Kyty runs" : std::strerror(-r));
		}
		// With the kyty-lodhw helper (root) the CAM entry itself is programmed, as on the PS5, and register
		// writes reach the CP as the title made them; without it the translator sets the bit in each write.
		if (PrimaryFeature("KYTY_NATIVE_GPU_PS5_REG_OVERRIDES")) {
			const int r = m_rciu_hw.On() || m_rciu_hw.Connect() ? m_rciu_hw.SetRciu(19, 0xa310u | 1u << 24u | 1u << 25u, 4u, 4u)
			                                                           : -ENOTCONN;
			if (r == 0) {
				std::fprintf(stderr, "native shadow: RCIU CAM entry 19 programmed as on the PS5 (PA_SC_SHADER_CONTROL.LOAD_COLLISION_WAVEID)\n");
			} else {
				m_translator.SetPs5RegOverrides(true);
				std::fprintf(stderr,
				             "native shadow: PS5 register overrides in the translator (PA_SC_SHADER_CONTROL.LOAD_COLLISION_WAVEID; "
				             "RCIU CAM through kyty-lodhw: %s)\n",
				             std::strerror(-r));
			}
		}
		if (EnvOn("KYTY_NATIVE_GPU_ACQUIRE_FULL")) {
			m_translator.SetAcquireFull(true);
			m_acb_translator.SetAcquireFull(true);
			std::fprintf(stderr, "native shadow: every ACQUIRE_MEM is a full cache writeback and invalidate\n");
		}
		if (EnvOn("KYTY_NATIVE_GPU_GDS_DMA_SYNC")) {
			m_translator.SetGdsDmaSync(true);
			m_acb_translator.SetGdsDmaSync(true);
			std::fprintf(stderr, "native shadow: DMA_DATA with GDS gets CP_SYNC, copies out of GDS a CS_PARTIAL_FLUSH\n");
		}
		// The LOD statistics banks of every draw and dispatch, for Kyty's GET_LOD_STATS
		// (KYTY_LOD_STATS_MODE=usage, the default, or =banks; also when KYTY_LOD_STATS_LOG is set).
		// KYTY_LOD_STATS_SCAN_DWORDS: how much of each first-level table is read.
		if (LodBankScanOn() && !LodStatsHw::Instance().On()) {
			const char*    v      = std::getenv("KYTY_LOD_STATS_SCAN_DWORDS");
			const uint32_t dwords = v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 1024u;
			m_translator.SetLodBankScan(true, dwords);
			m_acb_translator.SetLodBankScan(true, dwords);
			std::fprintf(stderr, "native shadow: scanning T#s for LOD statistics banks (%u dwords per table)\n", dwords);
		}
		// KYTY_NATIVE_GPU_GUESS_SPAN_KIB: residency for each user-data value that may be an image base.
		if (const char* v = std::getenv("KYTY_NATIVE_GPU_GUESS_SPAN_KIB"); v != nullptr) {
			const uint64_t bytes = std::strtoull(v, nullptr, 0) << 10u;
			m_translator.SetGuessSpan(bytes);
			m_acb_translator.SetGuessSpan(bytes);
			std::fprintf(stderr, "native shadow: image-base guesses make %llu KiB resident\n",
			             static_cast<unsigned long long>(bytes >> 10u));
		}
		// KYTY_NATIVE_GPU_ASYNC=1: submissions return without waiting for the GPU (experiment; the
		// guest's labels are still written by Kyty's CPU command processor, possibly too early).
		if (EnvOn("KYTY_NATIVE_GPU_ASYNC") || ShadowThreaded()) {
			m_submitter->SetAsync(true);
			std::fprintf(stderr, "native shadow: asynchronous submission (experiment)\n");
		}
		if (PrimaryFeature("KYTY_NATIVE_GPU_WAIT_BARRIERS")) {
			m_translator.SetFenceAddress(m_submitter->ScratchVa() + 1024);
			m_acb_translator.SetFenceAddress(m_submitter->ScratchVa() + 1032);
			std::fprintf(stderr, "native shadow: guest waits become GPU barriers\n");
		}
		// KYTY_NATIVE_GPU_FIND=0xADDR:0xSIZE: log the draws/dispatches whose resources reach that range.
		if (const char* find = std::getenv("KYTY_NATIVE_GPU_FIND"); find != nullptr) {
			char*          end  = nullptr;
			const uint64_t addr = std::strtoull(find, &end, 0);
			const uint64_t size = (end != nullptr && *end == ':') ? std::strtoull(end + 1, nullptr, 0) : 1;
			m_translator.SetFindRange(addr, size);
			m_acb_translator.SetFindRange(addr, size);
			std::fprintf(stderr, "native shadow: looking for references to [0x%llx, +0x%llx)\n",
			             static_cast<unsigned long long>(addr), static_cast<unsigned long long>(size));
		}
		if (const char* acb = std::getenv("KYTY_NATIVE_GPU_ACB"); acb == nullptr ? EnvOn("KYTY_NATIVE_GPU_PRIMARY")
		                                                                       : (acb[0] != '\0' && acb[0] != '0')) {
			m_acb_mode = acb != nullptr && std::strcmp(acb, "dry") == 0 ? AcbMode::Dry : AcbMode::On;
			const char* where = "run natively on the gfx ring";
			if (m_acb_mode != AcbMode::Dry && EnvOn("KYTY_NATIVE_GPU_ACB_RINGS")) {
				where = "run natively, on compute rings when native labels are on";
			}
			std::fprintf(stderr, "native shadow: async compute buffers %s\n",
			             m_acb_mode == AcbMode::Dry ? "translated only (dry run)" : where);
		}
		if (const char* depth = std::getenv("KYTY_NATIVE_GPU_DEPTH"); depth != nullptr) {
			const bool hiz_off  = std::strcmp(depth, "hiz_off") == 0;
			const bool no_htile = std::strcmp(depth, "no_htile") == 0;
			m_translator.SetDepthDebugMode(hiz_off ? Pm4Translator::DepthDebugMode::HizOff
			                                       : (no_htile ? Pm4Translator::DepthDebugMode::NoHtile
			                                                   : Pm4Translator::DepthDebugMode::None));
			std::fprintf(stderr, "native shadow: depth experiment %s\n", depth);
		}
		m_enabled = true;
		m_active.store(true, std::memory_order_release);
		std::fprintf(stderr, "native shadow: enabled (gfx-ring shadow submission, %.1f s timeout)\n",
		             m_timeout_sec);
#else
		std::fprintf(stderr, "native shadow: not supported on this platform; staying off.\n");
#endif
	}

	void Disable(const char* reason) {
		m_disabled = true;
		m_active.store(false, std::memory_order_release);
#if KYTY_GEM_BACKING_SUPPORTED
		// The pipeline's completion thread may still wait on submissions of this submitter.
		if (m_submitter != nullptr && !m_submitter->Async()) {
			delete m_submitter;
			m_submitter = nullptr;
		}
#endif
		std::fprintf(stderr,
		             "native shadow: disabled (%s) after %llu completed frames, avg %.2f ms, "
		             "%u v_dot2c_f32_f16 sites in %u shaders\n",
		             reason, static_cast<unsigned long long>(m_completed),
		             m_completed > 0 ? m_gpu_ms_total / static_cast<double>(m_completed) : 0.0,
		             m_dot2c_sites, m_shaders_scanned);
	}

	void DetectShaders(const std::vector<uint64_t>& addrs) {
		for (uint64_t va: addrs) {
			if (va == 0 || !m_seen_shaders.insert(va).second) {
				continue;
			}
			// Guest VA == host VA in Kyty, so the ISA is readable through the pointer.
			const auto* code = reinterpret_cast<const uint32_t*>(va);
			Dot2cScan   scan = CountDot2c(std::span<const uint32_t>(code, 1u << 16u));
			m_shaders_scanned++;
			m_dot2c_sites += scan.sites;
			m_dot_ops += scan.dot_ops;
			if (m_shaders_scanned % 500u == 0) {
				std::fprintf(stderr, "native shadow: %u shaders scanned so far: %u v_dot2c_f32_f16 sites, %u other dot ops\n",
				             m_shaders_scanned, m_dot2c_sites, m_dot_ops);
			}
			if (scan.decode_error && m_decode_stops++ < 16) {
				std::fprintf(stderr, "native shadow: shader 0x%llx: sweep stopped at an undecodable word 0x%08x after %u instrs\n",
				             static_cast<unsigned long long>(va), code[scan.dwords], scan.instructions);
			}
			if (scan.sites > 0 || scan.dot_ops > 0) {
				std::fprintf(stderr,
				             "native shadow: shader 0x%llx has %u v_dot2c_f32_f16 site(s) and %u other dot op(s) "
				             "(%u instrs, %s); patching is future work\n",
				             static_cast<unsigned long long>(va), scan.sites, scan.dot_ops, scan.instructions,
				             scan.ended ? "end" : (scan.decode_error ? "decode stop" : "capped"));
			}
			// The PS5 kernel gives the game VMID SH_MEM_BASES 0x80007000, Linux 0x20001000: a shader that
			// reads the apertures adapts, one that hardcodes PS5 aperture addresses in FLAT accesses does not.
			if (scan.flat_ops + scan.aperture_ops > 0) {
				m_flat_shaders++;
				if (m_flat_shaders <= 32) {
					std::fprintf(stderr, "native shadow: shader 0x%llx has %u FLAT accesses, %u aperture reads (%u instrs)\n",
					             static_cast<unsigned long long>(va), scan.flat_ops, scan.aperture_ops, scan.instructions);
				}
			}
			if (scan.gds_ops + scan.ordered_ops + scan.gws_ops > 0) {
				m_gds_shaders.insert(va);
				std::fprintf(stderr,
				             "native shadow: shader 0x%llx uses GDS: %u gds, %u ds_ordered_count, %u ds_gws (%u instrs, %s)\n",
				             static_cast<unsigned long long>(va), scan.gds_ops, scan.ordered_ops, scan.gws_ops,
				             scan.instructions, scan.ended ? "s_endpgm" : (scan.decode_error ? "decode stop" : "capped"));
			}
			// Keep the code for disassembly (llvm-mc -arch=amdgcn -mcpu=gfx1013 --disassemble): GDS
			// shaders always, every shader with KYTY_NATIVE_GPU_DUMP_SHADERS=1.
			if (scan.gds_ops + scan.ordered_ops + scan.gws_ops > 0 || m_dump_shaders) {
				if (const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR"); dir != nullptr && scan.dwords != 0) {
					char path[512];
					std::snprintf(path, sizeof(path), "%s/shader_%llx.bin", dir, static_cast<unsigned long long>(va));
					// KYTY_NATIVE_GPU_DUMP_TAIL=<dwords> also keeps code past the first s_endpgm, where
					// compilers put out-of-line blocks such as the all-lanes-killed epilogue.
					const char*  tail  = std::getenv("KYTY_NATIVE_GPU_DUMP_TAIL");
					const size_t extra = tail != nullptr ? std::strtoul(tail, nullptr, 0) : 0;
					if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
						std::fwrite(code, 4, scan.dwords + extra, f);
						std::fclose(f);
					}
				}
			}
		}
	}

	std::mutex    m_mutex;
	bool          m_initialized = false;
	bool          m_agc_start_loaded = false; // LoadAgcStartState ran
	bool          m_enabled     = false;
	bool          m_disabled    = false;
	Pm4Translator m_translator;
	Pm4Translator m_acb_translator; // async compute buffers (KYTY_NATIVE_GPU_ACB)
	AcbMode       m_acb_mode = AcbMode::Off;
	uint64_t      m_acbs           = 0;
	uint64_t      m_acbs_completed = 0;
	uint64_t      m_acbs_refused   = 0;
	std::set<uint32_t> m_unknown_acb_ops;
	uint32_t      m_acb_dummy_writes = 0;
	uint64_t      m_promoted_by[6]   = {};
	uint64_t      m_event_type_hist[64]  = {}; // EVENT_WRITE/RELEASE_MEM EVENT_TYPE counts, all frames
	uint64_t      m_event_index_hist[16] = {}; // EVENT_WRITE/RELEASE_MEM EVENT_INDEX counts, all frames
	uint64_t      m_last_seq         = 0;
	bool          m_pipelined        = false;
	bool          m_labels           = false;
	int           m_validate         = 1;
	uint32_t      m_validate_refused = 0;
	uint64_t      m_labels_stats[8] = {}; // GPU labels, tracker labels, interrupts, native/cut/barrier waits, CPU timestamps, GPU timestamps
	uint64_t      m_labels_mark[8]  = {}; // m_labels_stats at the previous flip report (per-flip deltas)
	uint32_t      m_cpu_labels_logged = 0;
	std::set<uint64_t> m_cpu_label_pages;
	std::vector<uint32_t> m_batch;
	// Per-ring batches and the last fence on each (KYTY_NATIVE_GPU_ACB_RINGS). Slot 0 is gfx.
	struct SlotFence {
		uint64_t seq     = 0;
		uint32_t ip_type = 0;
		uint32_t ring    = 0;
	};
	bool                  m_acb_rings    = false;
	uint32_t              m_compute_mask = 0;
	bool                  m_kfd_queues   = false; // EnableKfdQueues succeeded
	uint32_t              m_irq_event    = 0;     // KfdEopEvent of the tracker's releases (0: it polls)
	uint32_t              m_irq_ctx      = 0;     // their context id: the event, with KfdEopLog::kEventBit once the log is open
#if KYTY_GEM_BACKING_SUPPORTED
	KfdEopLog             m_eop_log;              // a title's interrupts the GPU raises (SetGuestInterruptsOnGpu)
#endif
	bool                  m_eop_log_open = false;
	std::atomic<uint64_t> m_irq_gpu {0};          // title interrupts the translator left to the GPU
	std::atomic<uint64_t> m_irq_logged {0};       // and those the EOP log delivered
	std::atomic<uint64_t> m_irq_stray {0};        // log entries from a ring with no guest queue here
	// KFD queue index -> its guest compute queue + 1 (0: none yet, kSharedKfdQueue: more than one).
	std::array<std::atomic<uint32_t>, 32> m_kfd_guest_queue {};
	static constexpr uint32_t             kSharedKfdQueue = 0xffffffffu;
	// Context ids the KFD queues raise once created (NativeSubmitter::SetKfdCalibration): this | index.
	static constexpr uint32_t kKfdCalibrationCtx = 0x07ffff00u;
	std::atomic<uint64_t> m_irq_wakes {0};        // tracker waits the interrupt ended
	std::atomic<uint64_t> m_irq_timeouts {0};     // and those that ran out instead
	// KYTY_NATIVE_GPU_PS5_REGS (InitPs5Regs): its own helper connection, so closing it (exit, crash)
	// puts the registers back, and the LOD statistics client is not turned on by it.
	LodStatsHw            m_ps5_hw;
	LodStatsHw            m_rciu_hw; // holds the RCIU CAM entry and GFXOFF off while Kyty runs
	std::mutex            m_ps5_mutex;
	uint32_t              m_ps5_regs     = 0; // kPs5* bits
	uint32_t              m_ps5_pasid    = 0;
	uint32_t              m_ps5_gfx_vmid = 0; // the gfx VMID the per-VMID settings are on now
	LabelOracle*          m_label_oracle = nullptr;
	std::vector<uint32_t> m_ring_batch[kTrackerSlots];
	SlotFence             m_slot_fence[kTrackerSlots];
	struct DeferredAcb {
		std::vector<uint32_t>                       ib;
		std::vector<Pm4TranslateResult::UnmetWait> needs;      // waits no submitted DCB satisfies yet
		uint32_t                                    dcbs_since = 0;
	};
	bool                                   m_acb_defer = false;
	std::deque<DeferredAcb>                m_deferred_acbs;
	std::unordered_map<uint64_t, uint32_t> m_inflight_labels; // label -> value a submitted DCB writes, until Kyty has written it
	uint64_t                 m_acb_deferred   = 0;
	uint64_t                 m_acb_resolved   = 0;
	uint64_t                 m_acb_unresolved = 0;
	std::atomic<bool> m_active {false};
	unsigned long m_waits_logged     = 0;
	std::set<uint64_t> m_flipped; // display buffers already made resident
	double        m_busy_s      = 0.0;
	double        m_wait_lock_s = 0.0;
	double        m_present_s   = 0.0;
	std::atomic<double> m_gpu_busy_s {0.0};
	// GpuWritable's last answer (translation runs under m_mutex).
	mutable uint64_t m_writable_gen = ~0ull, m_writable_lo = 0, m_writable_hi = 0;
	mutable bool     m_writable     = false;
	// KYTY_NATIVE_GPU_FLIP_TRACE=1: never-reset totals FlipTrace takes per-flip differences of.
	double                  m_trace_busy_s = 0.0;
	std::atomic<double>     m_trace_gpu_busy_s {0.0};
	std::array<uint64_t, 2> m_trace_dw {};
	bool          m_check_order = EnvOn("KYTY_NATIVE_GPU_CHECK_ORDER");
	struct PendingWait {
		Pm4TranslateResult::UnmetWait wait;
		bool                          acb;
		uint64_t                      buffer;
	};
	std::map<uint64_t, PendingWait> m_unmet_by_addr; // label -> the wait no earlier buffer satisfies
	std::map<uint64_t, uint64_t>    m_last_label;    // label -> last value written by a translated buffer
	uint64_t      m_late_writers = 0;
	std::set<uint64_t> m_gds_seen; // distinct (queue, direction, offset, size) of GDS DMA, for the log
	uint64_t      m_gds_dma_dcb[2]  = {}; // DMA_DATA to / from GDS in graphics buffers
	uint64_t      m_gds_dma_acb[2]  = {}; // and in async compute buffers
	unsigned long m_data_writes_logged = 0;
	uint64_t      m_flat_shaders       = 0;
	std::mutex    m_chain_mutex;
	uint64_t      m_chain_n         = 0; // pipelined buffers finished since the last flip report
	double        m_chain_queued_s  = 0.0;
	double        m_chain_latency_s = 0.0;
	double        m_chain_then_s    = 0.0;
	double        m_chain_max_s     = 0.0;
	double        m_flip_t0     = 0.0;
	uint32_t      m_flips       = 0;
	double                  m_flip_last = 0.0;
	std::array<uint32_t, 3> m_flip_hist {}; // flip intervals: on time, one refresh late, later
	std::set<std::tuple<uint64_t, uint32_t, uint32_t, uint32_t>> m_find_seen; // KYTY_NATIVE_GPU_FIND hits logged
	bool          m_stats          = false;
	uint32_t      m_stats_logged   = 0;
	int64_t       m_dump_regs_frame = -1;
	std::string   m_breadcrumbs; // KYTY_NATIVE_GPU_BREADCRUMBS directory, empty when off
	std::string   m_hang_dump;   // KYTY_NATIVE_GPU_HANG_DUMP directory, empty when off
	std::array<std::deque<std::pair<uint64_t, std::vector<uint32_t>>>, kTrackerSlots> m_hang_kept; // per tracker slot
	bool          m_collect_targets = false;           // flip dumps compare render targets too
	bool          m_collect_refs    = true;            // the translators collect per-draw references (SyncCollectRefs)
	std::map<uint64_t, RenderTargetInfo> m_frame_targets; // color targets bound since the last flip dump
	std::vector<FrameDraw>               m_frame_draws;   // KYTY_NATIVE_GPU_CTL: draws since the last flip
	uint32_t                             m_frame_buffers = 0;
	uint64_t      m_frames         = 0;
	uint64_t      m_completed      = 0;
	uint32_t      m_refused        = 0;
	uint64_t      m_flattened      = 0; // buffers FlattenIndirectBuffers inlined
	uint64_t                m_verbatim_dw   = 0; // Pm4TranslateStats::verbatim_dw over the flips being summed
	uint64_t                m_total_dw      = 0;
	uint64_t                m_verbatim_runs = 0;
	uint64_t                m_labels_native = 0; // Pm4TranslateStats::labels_native and write_data_dw, same flips
	uint64_t                m_write_data_dw = 0;
	uint64_t                m_bench_count   = 0; // KYTY_NATIVE_GPU_TRANSLATE_BENCH
	// KYTY_NATIVE_GPU_FLIP_TRACE=gpu (NoteLabelWriters).
	struct LabelWriter {
		double t   = 0;
		bool   acb = false;
	};
	bool                                      m_trace_gpu = false;
	std::unordered_map<uint64_t, LabelWriter> m_label_writer;
	std::vector<double>                       m_wait_dcb_time;
	uint64_t                                  m_wait_stamped = 0;
	uint64_t                m_direct_ib2    = 0; // Pm4TranslateStats::direct_ib2 and direct_dw, same flips
	uint64_t                m_direct_dw     = 0;
	uint32_t                m_direct_min_dw = 0; // KYTY_NATIVE_GPU_DIRECT_RUNS
	std::array<uint64_t, 4> m_run_hist {};
	std::array<std::array<uint64_t, 256>, 2> m_changed_ops {}; // DCB, ACB
	std::array<std::array<uint64_t, 256>, 2> m_ops {};
	// Direct-execution view, [0] DCBs and [1] ACBs: dwords, stretches between cuts, replaced packets and
	// packets with an insertion of ours before them (Pm4TranslateStats::cut_*).
	struct CutStats {
		uint64_t                  total_dw = 0;
		uint64_t                  cut_runs = 0;
		std::array<uint64_t, 256> cut_ops {};
		std::array<uint64_t, 256> inserted_ops {};
	};
	std::array<CutStats, 2> m_cut_stats {};
	uint64_t      m_flatten_failed = 0; // and could not
	double        m_gpu_ms_total   = 0.0;
	double        m_cs_ms_total    = 0.0;
	double        m_cs_mark        = 0.0; // m_cs_ms_total at the last flip stats
	double        m_translate_mark = 0.0;
	double        m_gpu_only_ms_total = 0.0;
	double        m_translate_ms_total = 0.0;
	double        m_residency_ms_total = 0.0;
#if KYTY_GEM_BACKING_SUPPORTED
	uint64_t                                   m_views_generation = ~0ull;
	std::vector<std::pair<uint64_t, uint64_t>> m_view_ranges;
	std::set<uint64_t>                         m_resident_buckets; // 16 MiB guest buckets already promoted
#endif
	uint32_t      m_dummy_writes       = 0;
	uint32_t      m_unmapped_logged    = 0;
	bool          m_dump_shaders       = false;
	bool          m_log_unmapped       = false;
	uint32_t      m_inspected          = 0;
	std::set<uint64_t> m_unmapped_seen;     // 64 KiB pages already checked by NoteUnmapped
	std::set<uint64_t> m_skip_ps;           // KYTY_NATIVE_GPU_SKIP_PS
	std::set<uint32_t> m_unknown_ops;
	std::set<uint64_t> m_gds_shaders;         // shaders seen using GDS instructions
	bool               m_allow_gds_shaders = false;
	bool               m_skip_gds_shaders  = false;
	size_t             m_skipped_shader_count = 0;
	uint32_t           m_gds_refused       = 0;
	double        m_timeout_sec    = kDefaultTimeoutSec;
	uint32_t      m_shaders_scanned = 0;
	uint32_t      m_dot2c_sites    = 0;
	uint32_t      m_dot_ops = 0; // dot instructions other than v_dot2c_f32_f16 in scanned shaders
	uint32_t      m_decode_stops   = 0;
	std::unordered_set<uint64_t> m_seen_shaders;
	std::set<uint32_t> m_skipped_uconfig;
#if KYTY_GEM_BACKING_SUPPORTED
	GemBacking*      m_gem       = nullptr;
	NativeSubmitter* m_submitter = nullptr;
#endif
};

} // namespace

void ShadowSubmitDcb(const uint32_t* dcb, uint32_t size_in_dwords) {
	if (dcb == nullptr || size_in_dwords == 0) {
		return;
	}
	Shadow::Instance().Submit(std::span<const uint32_t>(dcb, size_in_dwords));
}

namespace {

// KYTY_NATIVE_GPU_THREAD=1: guest submissions leave the game thread at once. A worker translates and
// submits them in order without waiting for the GPU; a completion thread waits for each one and only
// then hands the buffer to Kyty's command processor, so the labels, EOP events and flips Kyty
// produces still follow the GPU work they stand for.
thread_local bool t_pipeline_thread = false;

class Pipeline {
public:
	static Pipeline& Instance() {
		static Pipeline p;
		return p;
	}

	void Enqueue(const uint32_t* buffer, uint32_t size_dw, bool acb, std::function<void()> then) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_jobs.push_back(Job {buffer, size_dw, acb, std::move(then), NowSecShadow()});
		m_cv.notify_one();
	}

private:
	struct Job {
		const uint32_t*       buffer;
		uint32_t              size_dw;
		bool                  acb;
		std::function<void()> then;
		double                enqueued; // when the guest submitted it
	};
	struct Done {
		uint64_t              seq;
		std::function<void()> then;
		double                submitted; // when the batch went to the GPU
		double                enqueued;
		double                picked;    // when the worker took it
	};

	Pipeline() {
		Shadow::Instance().SetPipelined();
		std::thread([this] {
			RunAtTopClock();
			Work();
		}).detach();
		std::thread([this] {
			RunAtTopClock();
			Complete();
		}).detach();
	}

	// Takes every queued buffer, translates them into one batch and submits it with a single CS. Their
	// continuations all wait for that last submission: the gfx ring runs submissions in order.
	void Work() {
		for (;;) {
			std::deque<Job> jobs;
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				m_cv.wait(lock, [this] { return !m_jobs.empty(); });
				jobs.swap(m_jobs);
			}
			const double picked = NowSecShadow();
			for (auto& job: jobs) {
				if (job.buffer != nullptr) { // nullptr: only keeps `then` in order
					Shadow::Instance().Submit(std::span<const uint32_t>(job.buffer, job.size_dw), job.acb);
				}
			}
			Shadow::Instance().FlushBatch();
			const uint64_t seq = Shadow::Instance().TakeLastSeq();
			const double   now = NowSecShadow();
			std::lock_guard<std::mutex> lock(m_done_mutex);
			for (auto& job: jobs) {
				m_done.push_back(Done {seq, std::move(job.then), now, job.enqueued, picked});
			}
			m_done_cv.notify_one();
		}
	}

	void Complete() {
		t_pipeline_thread = true;
		for (;;) {
			Done done;
			{
				std::unique_lock<std::mutex> lock(m_done_mutex);
				m_done_cv.wait(lock, [this] { return !m_done.empty(); });
				done = std::move(m_done.front());
				m_done.pop_front();
			}
			if (done.seq != 0 && done.seq != m_waited_seq) {
				(void)Shadow::Instance().WaitSubmission(done.seq);
				// GPU busy time: from the batch's submission (or the previous batch's end) to its fence.
				const double end = NowSecShadow();
				Shadow::Instance().AddGpuBusy(end - std::max(done.submitted, m_last_end));
				m_last_end   = end;
				m_waited_seq = done.seq;
			}
			const double then_start = NowSecShadow();
			done.then();
			const double then_end = NowSecShadow();
			Shadow::Instance().AddChain(done.picked - done.enqueued, then_end - done.enqueued, then_end - then_start);
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::deque<Job>         m_jobs;
	std::mutex              m_done_mutex;
	std::condition_variable m_done_cv;
	std::deque<Done>        m_done;
	uint64_t                m_waited_seq = 0;
	double                  m_last_end   = 0.0;
};

// Native labels (KYTY_NATIVE_GPU_LABELS, on in primary mode with the pipeline): the GPU writes the
// guest's labels and Kyty's command processor no longer reads the guest buffers. A worker translates
// each buffer as it arrives, while the guest cannot have reused it yet, and puts its segments on its
// queue (graphics, or the guest compute queue of an ACB). A segment that starts at a guest wait nobody
// is known to satisfy is held, and so is everything behind it on that queue, until a write ahead in
// the ring satisfies the wait (then the guest wait runs natively) or the label is seen written in
// memory (then a barrier replaces it). Other queues go on meanwhile, which is what lets an ACB wait
// for a label a later DCB writes, as PS5 titles do. Ready segments join one batch and one CS.
// KYTY_NATIVE_GPU_ACB_RINGS=1 puts each compute queue on its own amdgpu compute ring (DCBs stay on
// gfx). A wait for a label the other ring writes stays on the CPU until that ring's CS has been
// accepted, and only then becomes a real WAIT_REG_MEM, so neither ring can block on a writer that
// was never submitted. Each ring has its own tracker slot and its own fence; a buffer's continuation
// runs when its ring passes the buffer's end point.
//
// A tracker thread blocks on the ring fence and performs what each point stands for as soon as the
// GPU passes it: EOP interrupts, labels the GPU cannot write (and, for the ones it wrote, a wake of a
// guest futex-waiting on the address; a CPU mirror of the store only with
// KYTY_NATIVE_GPU_TRACKER_WRITES=1), the end of a buffer (its continuation:
// Kyty's command processor runs the few packets it still needs, flips and waits for flips, copied
// out at translation; refused buffers whole).
class LabelPipeline final: public LabelOracle {
public:
	using Then = std::function<void(std::span<const uint32_t>)>;

	static LabelPipeline& Instance() {
		static LabelPipeline p;
		return p;
	}

	void Enqueue(const uint32_t* buffer, uint32_t size_dw, uint32_t queue, Then then) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_jobs.push_back(Job {buffer, size_dw, queue, std::move(then), NowSecShadow()});
		m_cv.notify_one();
	}

	bool GpuWritable(uint64_t addr) override { return Shadow::Instance().GpuWritable(addr); }

	bool SampleDword(uint64_t addr, uint32_t* out) override { return Shadow::Instance().SampleDword(addr, out); }

	// A wait in a buffer of m_translating's queue that no earlier write of that buffer decides: the
	// pieces of the same queue still held come before it, then writes ahead in the ring that have not
	// landed yet, then guest memory.
	Plan PlanWait(const WaitCond& wait) override {
		// A guest compute queue has a hardware queue of its own (KFD): its wait is what the title wrote,
		// run by the GPU as on PS5, and holds up nothing but that queue. So does a guest graphics queue
		// while each has a gfx ring of its own (two with the BC-250's patched amdgpu, as on the PS5).
		// When guest graphics queues share an in-order ring, a gfx wait whose writer is not known to be
		// ahead stays on the CPU, or another guest queue's writer behind it never runs (My First Gran
		// Turismo stopped at once); one memory already meets runs as itself, not as a barrier.
		const bool gpu_waits = AllWaitsOnGpu();
		const uint32_t slot = SlotOf(m_translating);
		if (gpu_waits && (!Shadow::GfxSlot(slot) || !m_gfx_shared)) {
			return Plan::Native;
		}
		const Plan plan = PlanWaitHeld(wait);
		return gpu_waits && plan == Plan::Met ? Plan::Native : plan;
	}

private:
	Plan PlanWaitHeld(const WaitCond& wait) {
		auto decide = [&](const LabelWrite& w) -> std::optional<Plan> {
			uint64_t  v = 0;
			const int c = w.Covers(wait.addr, wait.wide, &v, wait.mask);
			if (c == 0) {
				return std::nullopt;
			}
			return c == 1 && wait.Met(v) && w.gpu ? Plan::Native : Plan::Split;
		};
		if (auto q = m_pending.find(m_translating); q != m_pending.end()) {
			for (auto p = q->second.rbegin(); p != q->second.rend(); ++p) {
				for (auto w = p->seg.writes.rbegin(); w != p->seg.writes.rend(); ++w) {
					if (auto plan = decide(*w)) {
						return *plan;
					}
				}
			}
		}
		if (const RingWrite* w = PendingRingWrite(wait.addr)) {
			if (auto plan = decide(w->write)) {
				// Another ring's write is not ahead of this waiter. A real GPU wait before that CS
				// has returned can block the ring on a label nobody has been asked to write.
				// Nor while memory already satisfies it: that value may be stale (see PlanRingWait).
				if (*plan == Plan::Native && m_acb_rings && w->slot != SlotOf(m_translating) &&
				    (!Submitted(*w) || wait.Met(ReadLabel(wait)))) {
					return Plan::Split;
				}
				return *plan;
			}
		}
		return wait.Met(ReadLabel(wait)) ? Plan::Met : Plan::Split;
	}

private:
	struct Job {
		const uint32_t* buffer;
		uint32_t        size_dw;
		uint32_t        queue;
		Then            then;
		double          enqueued;
	};
	// What runs once a buffer is done.
	struct Done {
		Then                  then;
		const uint32_t*       buffer  = nullptr;
		uint32_t              size_dw = 0;
		bool                  whole   = false; // Kyty runs the guest buffer itself (refused, Kyty's own submissions)
		std::vector<uint32_t> tail;            // otherwise the packets it still needs
		double                enqueued = 0.0;
		double                picked   = 0.0;
	};
	// KYTY_NATIVE_GPU_CHUNK_DW=N (experiment, off): graphics buffers longer than N dwords go to the GPU
	// in chunks of about N (TranslateChunks). ASTRO BOT hung twice with it on the way into a level.
	uint32_t m_chunk_dw = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_CHUNK_DW");
		return v != nullptr ? static_cast<uint32_t>(std::strtoul(v, nullptr, 0)) : 0u;
	}();
	uint64_t                   m_chunks         = 0;
	uint32_t                   m_chunk_refused  = 0;
	std::map<uint32_t, double> m_hold_since; // queue -> when its front piece was first found held
	uint32_t                   m_stale_logged = 0;
	struct Piece {
		std::vector<uint32_t> ib;  // empty: no GPU work (refused buffers, Kyty's own submissions)
		Pm4Segment            seg; // offsets relative to `ib`
		std::shared_ptr<Done> done; // on the last piece of a buffer
		uint64_t              made = 0; // m_emits when it was translated: later emits carry writes it may wait for
	};
	struct Entry {
		uint64_t              seq;
		uint32_t              queue;
		uint32_t              slot; // tracker slot of the ring that runs this point
		TrackerPoint          point;
		std::shared_ptr<Done> done;
	};
	struct RingWrite {
		LabelWrite write;
		uint64_t   landed;    // the tracker value once its segment is done
		uint32_t   slot  = 0; // which ring wrote it
		uint64_t   batch = 0; // the m_batch it was appended in: its CS ioctl has returned once that is flushed
	};
	enum class Release { No, Plain, Native, Observed };

	bool Submitted(const RingWrite& w) const { return w.batch < m_batch; }

	LabelPipeline() {
		m_native = Shadow::Instance().EnableLabels(this);
		m_acb_rings = Shadow::Instance().AcbRings();
		m_compute_mask = Shadow::Instance().ComputeRingMask();
		m_timeout = Shadow::Instance().TimeoutSec();
		std::thread([this] {
			RunAtTopClock();
			Work();
		}).detach();
		std::thread([this] {
			RunAtTopClock();
			Track();
		}).detach();
	}

	// With KFD queues every guest wait stays a GPU wait, as on the PS5: each guest queue has a hardware
	// queue of its own, so a wait holds up only its queue. ASTRO BOT's white level: GPU busy 13.8 ms a
	// frame instead of 33.4 with the CPU holds and the barriers. KYTY_NATIVE_GPU_ALL_GPU_WAITS=0 brings
	// those back.
	bool AllWaitsOnGpu() const {
		static const bool on = !EnvIs("KYTY_NATIVE_GPU_ALL_GPU_WAITS", "0");
		return on && Shadow::Instance().KfdQueuesOn();
	}

	// Slot 0 is the gfx ring. A guest compute queue (0x20 + n) maps onto a kernel compute ring, or,
	// with KFD queues, onto a KFD queue of its own: the first queues seen take the free indexes in
	// order, as the PS5 maps each of a title's compute queues onto one hardware queue (more queues
	// than indexes share them).
	uint32_t SlotOf(uint32_t queue) const {
		if (queue < 0x20u || queue >= 0x58u) {
			// Guest graphics queues take the gfx rings in the order they appear, as the PS5 gives a title
			// its main ring and its "AGR" ring; a further one shares the first ring.
			auto [it, fresh] = m_gfx_index.try_emplace(queue, static_cast<uint32_t>(m_gfx_index.size()));
			const bool own = it->second == 0 || (it->second == 1 && m_acb_rings && Shadow::Instance().SecondGfxRing());
			if (fresh) {
				// Decided when the first one appears: by then the submitter knows the gfx rings.
				m_gfx_shared = it->second == 0 ? !Shadow::Instance().GfxQueueWaitsOnGpu() : (m_gfx_shared || !own);
				std::fprintf(stderr, "native shadow: guest graphics queue 0x%x runs on gfx ring %u%s\n", queue, own ? it->second : 0u,
				             m_gfx_shared ? " (gfx waits with no known writer ahead stay held on the CPU)" : " (gfx waits on the GPU)");
			}
			return own && it->second == 1 ? Shadow::kGfx1Slot : 0;
		}
		if (!m_acb_rings) {
			return 0;
		}
		if (!Shadow::Instance().KfdQueuesOn()) {
			return ComputeRingForQueue(queue, m_compute_mask) + 1u;
		}
		auto [it, fresh] = m_queue_index.try_emplace(queue, static_cast<uint32_t>(m_queue_index.size()) % Shadow::Instance().KfdQueueCount());
		if (fresh) {
			std::fprintf(stderr, "native shadow: guest compute queue 0x%x runs on KFD queue %u\n", queue, it->second);
			Shadow::Instance().NoteKfdGuestQueue(it->second, queue);
		}
		return it->second + 1u;
	}

	static uint64_t ReadLabel(const WaitCond& w) {
		return w.wide ? *reinterpret_cast<const volatile uint64_t*>(w.addr) : *reinterpret_cast<const volatile uint32_t*>(w.addr);
	}

	// The latest write to `addr` in the ring whose segment the GPU has not finished yet (after that,
	// guest memory says more), or nullptr.
	const RingWrite* PendingRingWrite(uint64_t addr) {
		auto it = m_ring.find(addr);
		if (it == m_ring.end()) {
			return nullptr;
		}
		if (Shadow::Instance().Reached(it->second.landed, it->second.slot)) {
			m_ring.erase(it);
			return nullptr;
		}
		return &it->second;
	}

	void Work() {
		for (;;) {
			std::deque<Job> jobs;
			{
				std::unique_lock<std::mutex> lock(m_mutex);
				if (m_held > 0) {
					// A label with no GPU writer shows up in memory when the tracker writes it, or when
					// the GPU does and the fence (or this timeout) says so. Not a 50 us poll.
					m_cv.wait_for(lock, std::chrono::milliseconds(2), [this] { return !m_jobs.empty() || m_wake; });
					m_wake = false;
				} else {
					m_cv.wait(lock, [this] { return !m_jobs.empty(); });
				}
				jobs.swap(m_jobs);
			}
			const double picked = NowSecShadow();
			for (auto& job: jobs) {
				Translate(job, picked);
			}
			// A cross-ring wait stays on the CPU until the writer's CS ioctl has returned, then the
			// next pump puts a real WAIT_REG_MEM on the waiter's ring. Same-ring waits are appended
			// behind the writer in one batch, so they do not need that round trip. Loop until a pump
			// adds nothing: a chain of waits across rings drains one hop per flush.
			for (;;) {
				Pump();
				if (!m_appended) {
					break;
				}
				Shadow::Instance().FlushBatch();
				m_batch++; // everything appended so far is submitted, without walking m_ring
				m_last_cs.store(Shadow::Instance().TakeLastSeq(), std::memory_order_release);
				m_appended = false;
			}
		}
	}

	void Translate(Job& job, double picked) {
		auto done      = std::make_shared<Done>();
		done->then     = std::move(job.then);
		done->buffer   = job.buffer;
		done->size_dw  = job.size_dw;
		done->enqueued = job.enqueued;
		done->picked   = picked;
		auto& queue    = m_pending[job.queue];
		Pm4TranslateResult r;
		m_translating = job.queue;
		if (m_acb_rings && job.queue >= 0x20u) {
			Shadow::Instance().UseAcbTracker(SlotOf(job.queue));
		}
		if (job.buffer != nullptr && m_native && job.queue == 0 && m_chunk_dw != 0 && job.size_dw > m_chunk_dw &&
		    TranslateChunks(job, done, queue)) {
			return;
		}
		if (job.buffer == nullptr || !m_native ||
		    !Shadow::Instance().Submit(std::span<const uint32_t>(job.buffer, job.size_dw), job.queue != 0, &r) ||
		    r.segments.empty()) {
			done->whole = true;
			queue.push_back(Piece {{}, {}, std::move(done)});
			return;
		}
		done->tail = std::move(r.cpu_tail);
		PushPieces(r, done, queue);
	}

	// A translated buffer as pieces, one per segment; `done` (when given) on the last.
	void PushPieces(Pm4TranslateResult& r, const std::shared_ptr<Done>& done, std::deque<Piece>& queue) {
		for (size_t k = 0; k < r.segments.size(); k++) {
			Piece p;
			p.made = m_emits;
			p.seg = std::move(r.segments[k]);
			p.ib.assign(r.ib.begin() + p.seg.begin, r.ib.begin() + p.seg.end);
			for (auto& point: p.seg.points) {
				point.dw -= p.seg.begin;
			}
			p.seg.barrier_dw -= p.seg.held ? p.seg.begin : 0;
			p.seg.wait_dw -= p.seg.held ? p.seg.begin : 0;
			m_held += p.seg.held ? 1 : 0;
			if (k + 1 == r.segments.size()) {
				p.done = done;
			}
			queue.push_back(std::move(p));
		}
	}

	// Pumps the pieces ready so far to the GPU, as Work does after each round.
	void PumpAndFlush() {
		for (;;) {
			Pump();
			if (!m_appended) {
				break;
			}
			Shadow::Instance().FlushBatch();
			m_batch++;
			m_last_cs.store(Shadow::Instance().TakeLastSeq(), std::memory_order_release);
			m_appended = false;
		}
	}

	// A large graphics buffer (ASTRO BOT's main DCB: ~900K dwords, ~10 ms to translate) goes to the GPU in
	// chunks of about m_chunk_dw dwords as each is translated, instead of after the whole of it: the
	// gfx ring otherwise idled until the end, and frames missed the display refresh whenever the
	// translation ran long. Each chunk is translated like a buffer of its own (the shadow carries the
	// state across, as between buffers); `then` comes after the last. A chunk never ends inside a
	// COND_EXEC's range. False (nothing done) when the first chunk is refused: the buffer is Kyty's.
	bool TranslateChunks(Job& job, std::shared_ptr<Done>& done, std::deque<Piece>& queue) {
		std::vector<uint32_t>     flat;
		std::vector<SourcePiece> source;
		std::span<const uint32_t> whole(job.buffer, job.size_dw);
		if (FlattenIndirectBuffers(whole, &flat, nullptr, &source)) {
			whole = flat;
		} else {
			source = {SourcePiece {0, job.size_dw, reinterpret_cast<uint64_t>(job.buffer)}};
		}
		std::vector<std::pair<uint32_t, uint32_t>> chunks; // [begin, end) at packet boundaries
		uint32_t                                   begin = 0, cond_until = 0;
		for (uint32_t dw = 0; dw < whole.size();) {
			const uint32_t header = whole[dw];
			const uint32_t size   = (header >> 30u) == 3u ? ((header >> 16u) & 0x3fffu) + 2u : 1u;
			if ((header >> 30u) == 3u && ((header >> 8u) & 0xffu) == 0x22u && size >= 5) { // COND_EXEC: EXEC_COUNT dwords follow
				cond_until = std::max(cond_until, dw + size + (whole[dw + 4] & 0x3fffu));
			}
			dw += size;
			if (dw - begin >= m_chunk_dw && dw >= cond_until && dw < whole.size()) {
				chunks.emplace_back(begin, dw);
				begin = dw;
			}
		}
		chunks.emplace_back(begin, static_cast<uint32_t>(whole.size()));
		if (chunks.size() < 2) {
			return false;
		}
		for (size_t c = 0; c < chunks.size(); c++) {
			const auto [cb, ce] = chunks[c];
			std::vector<SourcePiece> part;
			for (const SourcePiece& piece: source) {
				const uint32_t lo = std::max(piece.dw, cb), hi = std::min(piece.dw + piece.len, ce);
				if (lo < hi) {
					part.push_back({lo - cb, hi - lo, piece.va + 4ull * (lo - piece.dw)});
				}
			}
			Pm4TranslateResult r;
			if (!Shadow::Instance().Submit(whole.subspan(cb, ce - cb), false, &r, &part) || r.segments.empty()) {
				if (c == 0) {
					return false;
				}
				// Earlier chunks are on the GPU already: the rest of this frame is lost.
				if (m_chunk_refused++ < 8) {
					std::fprintf(stderr, "native shadow: chunk %zu of %zu of a DCB refused; the rest of it is dropped\n", c,
					             chunks.size());
				}
				queue.push_back(Piece {{}, {}, std::move(done)});
				return true;
			}
			done->tail.insert(done->tail.end(), r.cpu_tail.begin(), r.cpu_tail.end());
			PushPieces(r, c + 1 == chunks.size() ? done : nullptr, queue);
			if (c + 1 < chunks.size()) {
				PumpAndFlush(); // this chunk runs while the next is translated
			}
		}
		m_chunks += chunks.size();
		return true;
	}

	Release Ready(uint32_t queue, const Piece& p) {
		if (!p.seg.held) {
			return Release::Plain;
		}
		if (!Shadow::Instance().Active()) {
			return Release::Observed; // nothing runs any more; let the continuations through
		}
		const WaitCond& wait = p.seg.wait;
		int             cover = 0;
		bool            satisfies = false;
		bool            gpu = false;
		bool            same = false;
		bool            submitted = false;
		const uint32_t  slot = SlotOf(queue);
		bool            gfx_writer = false; // see PlanRingWait
		if (const RingWrite* w = PendingRingWrite(wait.addr)) {
			uint64_t v = 0;
			cover = w->write.Covers(wait.addr, wait.wide, &v, wait.mask);
			if (cover == 1) {
				satisfies = wait.Met(v);
			}
			gpu        = w->write.gpu;
			same       = w->slot == slot;
			submitted  = Submitted(*w);
			gfx_writer = m_early_waits && !Shadow::GfxSlot(slot) && Shadow::GfxSlot(w->slot);
		} else if (auto last = m_last_write.find(wait.addr); last != m_last_write.end() && last->second.second > p.made) {
			// A write emitted after this waiter was translated has already landed. On PS5 the waiting
			// queue would have seen it, even if the title has reset the label since (ASTRO BOT does that
			// when a level loads, faster than a held piece is looked at again): the wait is met.
			uint64_t v = 0;
			if (last->second.first.Covers(wait.addr, wait.wide, &v, wait.mask) == 1 && wait.Met(v)) {
				return Release::Observed;
			}
		}
		// Hold until a writer is guaranteed to run (PlanRingWait). A label already in memory, with
		// no pending write about to replace it, is a barrier: the GPU never waits for a writer that
		// will not be submitted.
		const bool native = p.seg.wait_len != 0 && m_native_waits;
		// A compute queue of its own (KFD) just waits, as on PS5: whatever writes the label later, on
		// whichever queue, nothing can be stuck behind this one.
		if (native && (!Shadow::GfxSlot(slot) || !m_gfx_shared) && Shadow::Instance().KfdQueuesOn()) {
			return Release::Native;
		}
		// With independent queues the wait reads memory as on PS5: a label that already holds a
		// satisfying value lets it through, even if a write to it is still on its way. Holding for that
		// write (against a stale value) deadlocked the gfx ring against a KFD queue that waited for the
		// gfx ring's later write.
		if (native && Shadow::Instance().KfdQueuesOn() && wait.Met(ReadLabel(wait))) {
			return Release::Native;
		}
		switch (PlanRingWait(cover, satisfies, gpu, same, submitted, wait.Met(ReadLabel(wait)), native, gfx_writer)) {
			case RingWaitRelease::GpuWait: return Release::Native;
			case RingWaitRelease::Barrier: return Release::Observed;
			case RingWaitRelease::Hold:
				m_last_hold_reason = cover == 0 ? 4 : cover != 1 ? 3 : !gpu ? 2 : !satisfies ? 1 : !native ? 5 : 0;
				return Release::No;
		}
		return Release::No;
	}

	// KYTY_NATIVE_GPU_LOG_HOLDS=N: the first N releases of a hold with no known writer: who waited, on what,
	// and what let it go (a writer that appeared in a later buffer, on which ring, or memory that changed
	// with no writer the translator saw: the CPU, or a GPU write it does not track).
	void LogHoldRelease(uint32_t queue, const Piece& p, Release how, int reason, uint64_t us) {
		static const unsigned long limit = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_LOG_HOLDS");
			return v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
		}();
		if (reason != 4 || us < 500 || m_holds_logged >= limit) {
			return;
		}
		m_holds_logged++;
		const WaitCond&  w  = p.seg.wait;
		const RingWrite* rw = PendingRingWrite(w.addr);
		char             writer[96] = "no writer";
		if (rw != nullptr) {
			std::snprintf(writer, sizeof(writer), "writer on slot %u (%s, gpu %d, value 0x%llx)", rw->slot,
			              Submitted(*rw) ? "submitted" : "not submitted", rw->write.gpu ? 1 : 0,
			              static_cast<unsigned long long>(rw->write.value));
		}
		std::fprintf(stderr,
		             "hold: queue 0x%x (slot %u) %.2f ms on 0x%llx %s ref 0x%llx mask 0x%llx func %u: memory 0x%llx -> 0x%llx, "
		             "released %s, %s, %zu pieces behind\n",
		             queue, SlotOf(queue), static_cast<double>(us) / 1e3, static_cast<unsigned long long>(w.addr), w.wide ? "64" : "32",
		             static_cast<unsigned long long>(w.ref), static_cast<unsigned long long>(w.mask), w.function,
		             static_cast<unsigned long long>(m_hold_start_value[queue]), static_cast<unsigned long long>(ReadLabel(w)),
		             how == Release::Native ? "as a GPU wait" : "on memory", writer, m_pending[queue].size() - 1);
	}

	// KYTY_NATIVE_GPU_LABEL_TRACE=<lo>-<hi> (hex guest addresses): the last 1024 label events in that
	// range, dumped with the first stale holds: W a GPU write emitted (C: one the CPU side makes), H a
	// hold begins, G / M it ends as a GPU wait / on memory; each with the queue, the value (the wait's
	// reference for H, G, M) and what memory held at that moment.
	struct LabelEvent {
		double   t;
		char     kind;
		uint32_t queue;
		uint64_t addr;
		uint64_t value;
		uint64_t memory;
	};
	void TraceLabel(char kind, uint32_t queue, uint64_t addr, uint64_t value) {
		static const std::pair<uint64_t, uint64_t> range = [] {
			const char* v = std::getenv("KYTY_NATIVE_GPU_LABEL_TRACE");
			uint64_t    lo = 0, hi = 0;
			if (v != nullptr) {
				char* end = nullptr;
				lo        = std::strtoull(v, &end, 16);
				hi        = end != nullptr && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : lo + 8;
			}
			return std::make_pair(lo, hi);
		}();
		if (addr < range.first || addr >= range.second) {
			return;
		}
		const uint64_t memory = *reinterpret_cast<const volatile uint32_t*>(addr);
		m_label_trace[m_label_trace_next++ % m_label_trace.size()] = LabelEvent {NowSecShadow(), kind, queue, addr, value, memory};
	}
	void DumpLabelTrace() const {
		const size_t n = std::min<uint64_t>(m_label_trace_next, m_label_trace.size());
		for (size_t i = 0; i < n; i++) {
			const LabelEvent& e = m_label_trace[(m_label_trace_next - n + i) % m_label_trace.size()];
			std::fprintf(stderr, "label trace %.4f %c q%x %llx v%llx mem %llx\n", e.t, e.kind, e.queue, static_cast<unsigned long long>(e.addr),
			             static_cast<unsigned long long>(e.value), static_cast<unsigned long long>(e.memory));
		}
	}

	// A piece held for more than 2 s: what it waits for and what memory holds (the first 16).
	void NoteStaleHold(uint32_t queue, const Piece& p) {
		const double now   = NowSecShadow();
		auto [it, fresh]   = m_hold_since.try_emplace(queue, now);
		if (fresh || now - it->second < 2.0 || m_stale_logged >= 16 || it->second < 0) {
			return;
		}
		const double held = now - it->second;
		it->second        = -1.0; // logged for this hold
		m_stale_logged++;
		const WaitCond&  w  = p.seg.wait;
		const RingWrite* rw = PendingRingWrite(w.addr);
		std::fprintf(stderr,
		             "native labels: queue 0x%x held %.1f s on WAIT_REG_MEM%s 0x%llx (mask 0x%llx func %u ref 0x%llx), memory 0x%llx, "
		             "pending ring write %s, %zu pieces queued behind\n",
		             queue, held, w.wide ? "_64" : "", static_cast<unsigned long long>(w.addr),
		             static_cast<unsigned long long>(w.mask), w.function, static_cast<unsigned long long>(w.ref),
		             static_cast<unsigned long long>(ReadLabel(w)), rw != nullptr ? (Submitted(*rw) ? "submitted" : "not submitted") : "none",
		             m_pending[queue].size());
		// The first stalls: every queue's pending pieces in order, as the label waits that hold them
		// ("wait") and the label writes they carry, to see which queue would write what another waits for.
		if (m_pending_dumps < 3) {
			m_pending_dumps++;
			for (const auto& [q, pieces]: m_pending) {
				std::string line;
				size_t      items = 0;
				for (const auto& piece: pieces) {
					if (items >= 40) {
						line += " ...";
						break;
					}
					char text[64];
					if (piece.seg.held) {
						std::snprintf(text, sizeof(text), " wait %llx=%llx", static_cast<unsigned long long>(piece.seg.wait.addr),
						              static_cast<unsigned long long>(piece.seg.wait.ref));
						line += text;
						items++;
					}
					for (const auto& lw: piece.seg.writes) {
						std::snprintf(text, sizeof(text), " %llx<-%llx%s", static_cast<unsigned long long>(lw.addr),
						              static_cast<unsigned long long>(lw.value), lw.gpu ? "" : "(cpu)");
						line += text;
						items++;
					}
				}
				std::fprintf(stderr, "native labels:   queue 0x%x (slot %u), %zu pieces:%s\n", q, SlotOf(q), pieces.size(), line.c_str());
			}
			DumpLabelTrace();
		}
	}

	// Emits what is ready, queue by queue, until nothing more is.
	void Pump() {
		for (bool progress = true; progress;) {
			progress = false;
			for (auto& [queue, pieces]: m_pending) {
				while (!pieces.empty()) {
					const Release how = Ready(queue, pieces.front());
					if (how == Release::No) {
						if (m_hold_since.count(queue) == 0) {
							m_hold_reason[queue] = m_last_hold_reason;
							m_hold_start_value[queue] = ReadLabel(pieces.front().seg.wait);
							TraceLabel('H', queue, pieces.front().seg.wait.addr, pieces.front().seg.wait.ref);
						}
						NoteStaleHold(queue, pieces.front());
						break;
					}
					if (pieces.front().seg.held) {
						g_held_released.fetch_add(1, std::memory_order_relaxed);
						if (auto h = m_hold_since.find(queue); h != m_hold_since.end() && h->second > 0.0) {
							const auto us = static_cast<uint64_t>((NowSecShadow() - h->second) * 1e6);
							const int  r  = m_hold_reason[queue];
							LogHoldRelease(queue, pieces.front(), how, r, us);
							g_held_waited.fetch_add(1, std::memory_order_relaxed);
							g_held_wait_us.fetch_add(us, std::memory_order_relaxed);
							g_hold_reason_n[r].fetch_add(1, std::memory_order_relaxed);
							g_hold_reason_us[r].fetch_add(us, std::memory_order_relaxed);
						}
					}
					m_hold_since.erase(queue);
					if (pieces.front().seg.held) {
						TraceLabel(how == Release::Native ? 'G' : 'M', queue, pieces.front().seg.wait.addr, pieces.front().seg.wait.ref);
					}
					Emit(queue, pieces.front(), how);
					pieces.pop_front();
					progress = true;
				}
			}
		}
	}

	void Emit(uint32_t queue, Piece& p, Release how) {
		if (p.seg.held) {
			m_held--;
			if (how == Release::Native) {
				NopOut(p.ib, p.seg.barrier_dw, p.seg.barrier_len); // the write ahead in the ring satisfies the wait
				m_released_native++;
			} else {
				NopOut(p.ib, p.seg.wait_dw, p.seg.wait_len); // the label is there; keep the barrier
				m_released_observed++;
			}
		}
		const uint32_t     slot = SlotOf(queue);
		std::vector<Entry> entries;
		uint32_t           packet = ~0u;
		for (auto& point: p.seg.points) {
			if (point.dw != packet) { // points sharing a packet share its number
				packet         = point.dw;
				p.ib[point.dw] = static_cast<uint32_t>(++m_seq);
				m_slot_seq[slot] = m_seq;
				if (Shadow::Instance().IrqEvent() != 0) {
					// RELEASE_MEM: control 3 dwords before the data, context id 2 after it.
					p.ib[point.dw - 3] |= 2u << 24u; // INT_SEL 2: interrupt once the write is confirmed
					p.ib[point.dw + 2] = Shadow::Instance().IrqCtx();
				}
			}
			const bool last = &point == &p.seg.points.back();
			entries.push_back(Entry {m_seq, queue, slot, std::move(point), last ? p.done : nullptr});
		}
		if (p.ib.empty()) {
			// No GPU work: done once everything ahead of it on its own ring is.
			entries.push_back(Entry {m_slot_seq[slot], queue, slot, TrackerPoint {}, p.done});
		}
		if (m_last_write.size() > 65536) {
			m_last_write.clear();
		}
		m_emits++;
		// Who writes a label matters only to a wait that may be held on the CPU. With every wait on the
		// GPU (compute queues on KFD, gfx waits while each guest gfx queue has its own ring) none is,
		// and recording ~5000 label writes a frame in these maps took a fifth of this thread.
		const bool holds_possible = !AllWaitsOnGpu() || m_gfx_shared;
		for (const auto& w: p.seg.writes) {
			TraceLabel(w.gpu ? 'W' : 'C', queue, w.addr, w.value);
			if (!holds_possible) {
				continue;
			}
			m_last_write[w.addr] = {w, m_emits};
			if (w.span_dw > 2) { // a block of data: each dword touched, its value unknown here
				for (uint32_t i = 0; i < std::min<uint32_t>(w.span_dw, 64u); i++) {
					m_ring[w.addr + 4ull * i] = RingWrite {LabelWrite {w.addr + 4ull * i, 0, false, w.gpu, false}, m_seq, slot, m_batch};
				}
				continue;
			}
			m_ring[w.addr] = RingWrite {w, m_seq, slot, m_batch};
			if (w.wide) {
				m_ring[w.addr + 4] =
				    RingWrite {LabelWrite {w.addr + 4, w.value >> 32u, w.known, w.gpu, false}, m_seq, slot, m_batch};
			}
		}
		// Forget writes that have landed once the map is large. The next sweep waits for it to double:
		// writes of a queue that waits on the GPU for long (a KFD queue) stay, and sweeping all of them
		// at every emit took most of this thread in ASTRO BOT's heavier levels.
		if (m_ring.size() > m_ring_sweep_at) {
			for (auto it = m_ring.begin(); it != m_ring.end();) {
				it = Shadow::Instance().Reached(it->second.landed, it->second.slot) ? m_ring.erase(it) : std::next(it);
			}
			m_ring_sweep_at = std::max<size_t>(16384, m_ring.size() * 2);
		}
		if (!p.ib.empty()) {
			// The translated IB passed the packet check; KYTY_NATIVE_GPU_VALIDATE=2 checks the patched one too.
			const NativeIbKind kind = Shadow::GfxSlot(slot) ? NativeIbKind::Gfx : NativeIbKind::Compute;
			if (uint32_t bad = 0; Shadow::Instance().ValidatingPatched() && !ValidateNativeIb(p.ib, NativeLodStats(), &bad, kind)) {
				std::fprintf(stderr, "native shadow: a patched segment fails the packet check at dw %u (0x%08x); disabling\n", bad,
				             p.ib[bad]);
				Shadow::Instance().DisableNow("packet check");
			} else {
				Shadow::Instance().AppendBatch(p.ib, slot);
				m_appended = true;
			}
		}
		std::lock_guard<std::mutex> lock(m_track_mutex);
		for (auto& e: entries) {
			m_entries.push_back(std::move(e));
		}
		m_track_cv.notify_one();
	}

	void Track() {
		t_pipeline_thread = true;
#if defined(__linux__)
		::prctl(PR_SET_TIMERSLACK, 1ul, 0ul, 0ul, 0ul);
#endif
		double stuck_since = 0.0;
		double busy_since  = 0.0; // GPU busy for the flip stats: while tracker points are outstanding
		for (;;) {
			std::vector<Entry> ready;
			{
				std::unique_lock<std::mutex> lock(m_track_mutex);
				if (m_entries.empty() && busy_since != 0.0) {
					Shadow::Instance().AddGpuBusy(NowSecShadow() - busy_since);
					busy_since = 0.0;
				}
				m_track_cv.wait(lock, [this] { return !m_entries.empty(); });
				busy_since = busy_since == 0.0 ? NowSecShadow() : busy_since;
				// In order within a ring, independently across rings: a compute queue waiting on the GPU
				// for a label (a KFD queue does, as on PS5) must not hold up the gfx ring's points, whose
				// end-of-frame events are what lets the title submit the buffer that writes that label.
				uint64_t blocked = 0; // rings with a point not reached yet
				for (auto it = m_entries.begin(); it != m_entries.end();) {
					const uint64_t bit = 1ull << (it->slot & 63u);
					if ((blocked & bit) == 0 && (!m_native || m_drain || Shadow::Instance().Reached(it->seq, it->slot))) {
						ready.push_back(std::move(*it));
						it = m_entries.erase(it);
					} else {
						blocked |= bit;
						++it;
					}
				}
			}
			if (!ready.empty()) {
				stuck_since = 0.0;
				for (auto& e: ready) {
					Run(e);
				}
				{
					std::lock_guard<std::mutex> lock(m_mutex);
					m_wake = true;
				}
				m_cv.notify_one();
				continue;
			}
			const double now = NowSecShadow();
			stuck_since      = stuck_since == 0.0 ? now : stuck_since;
			if (now - stuck_since > m_timeout) {
				// The GPU has not passed a point for the whole timeout: wait for what was submitted. If
				// that completes, the point was lost (run it anyway); if not, the path disables itself
				// and everything outstanding runs.
				const uint32_t stuck_slot = [&] {
					std::lock_guard<std::mutex> lock(m_track_mutex);
					return m_entries.empty() ? 0u : m_entries.front().slot;
				}();
				// This ring's own fence, not the other ring's. A compute wait must not be declared
				// idle because the gfx ring has finished, or the other way around.
				const bool idle = Shadow::Instance().WaitSlot(stuck_slot);
				std::fprintf(stderr, "native shadow: tracker stuck at %u (slot %u) for %.0f s; %s\n",
				             Shadow::Instance().TrackerValue(stuck_slot), stuck_slot, now - stuck_since,
				             idle ? "the ring is idle, running the next point" : "draining");
				{
					// Every ring's outstanding points: how many, the first one's sequence, and where its
					// tracker is, so a stall shows which ring the rest wait behind.
					std::lock_guard<std::mutex> lock(m_track_mutex);
					std::map<uint32_t, std::pair<size_t, uint64_t>> per_slot;
					for (const auto& e: m_entries) {
						auto [it, fresh] = per_slot.try_emplace(e.slot, 0, e.seq);
						it->second.first++;
					}
					for (const auto& [s, v]: per_slot) {
						std::fprintf(stderr, "native shadow:   slot %u: %zu points outstanding, first seq %llu, tracker at %u\n", s, v.first,
						             static_cast<unsigned long long>(v.second), Shadow::Instance().TrackerValue(s));
					}
				}
				std::lock_guard<std::mutex> lock(m_track_mutex);
				if (idle && !m_entries.empty()) {
					m_entries.front().seq = Shadow::Instance().TrackerValue(m_entries.front().slot);
				} else {
					m_drain = true;
				}
				stuck_since = 0.0;
				continue;
			}
			// With EOP interrupts every tracker release raises one: sleep until the next. An interrupt that
			// fired since the scan above left the event set, so this returns at once. The 10 ms cap is
			// a backstop that also lets a point whose submission has not run yet be looked at again.
			if (Shadow::Instance().IrqEvent() != 0) {
				Shadow::Instance().WaitIrq(10);
				continue;
			}
			// Block on a ring's fence: the gfx ring's when it has points outstanding (compute queues may
			// wait on labels for long), else the oldest point's. The 2 ms cap covers a tracker dword
			// written before the CS ends, and a point whose CS has not been submitted yet. A timeout
			// does not disable the path. If the fence was already signaled, WaitSeq returns at once:
			// sleep so that case does not spin.
			uint32_t slot = 0;
			{
				std::lock_guard<std::mutex> lock(m_track_mutex);
				if (!m_entries.empty()) {
					slot = std::any_of(m_entries.begin(), m_entries.end(), [](const Entry& e) { return e.slot == 0; })
					           ? 0u
					           : m_entries.front().slot;
				}
			}
			Shadow::Instance().WaitSlotBrief(slot, 0.002);
			bool pending = false;
			{
				std::lock_guard<std::mutex> lock(m_track_mutex);
				pending = !m_entries.empty() && std::none_of(m_entries.begin(), m_entries.end(), [](const Entry& e) {
					          return Shadow::Instance().Reached(e.seq, e.slot);
				          });
			}
			if (pending) {
				struct timespec nap = {0, 1000000};
				nanosleep(&nap, nullptr);
			}
		}
	}

	// sync-on-address sleeps in FUTEX_WAIT on a label's word. A store alone, the GPU's or the CPU's,
	// does not wake it; the 10 ms poll would, but the waiter returns as soon as it is woken and then
	// reads. Most labels have no sleeper: skip the syscall then (~20k per second in ASTRO BOT).
	static void WakeAddress(uint64_t addr) {
		if (LibKernel::SyncOnAddress::HasWaiters(reinterpret_cast<void*>(addr))) {
#if defined(__linux__)
			syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr, nullptr, 0);
#endif
		}
	}

	void Run(Entry& e) {
		const TrackerPoint& p = e.point;
		switch (p.kind) {
			case TrackerPoint::Kind::Flip:
				if (HostCallbacks().flip && p.data.size() == 5) {
					HostCallbacks().flip(static_cast<int>(p.data[0]), static_cast<int>(p.data[1]), static_cast<int>(p.data[2]),
					                     static_cast<int64_t>(p.data[3] | (static_cast<uint64_t>(p.data[4]) << 32u)));
				}
				break;
			case TrackerPoint::Kind::Interrupt:
				// The queue's EOP event: the graphics queue's is 0, an ACB's its guest queue number.
				if (HostCallbacks().interrupt) {
					HostCallbacks().interrupt(static_cast<int>(e.queue), p.ctx);
				}
				break;
			case TrackerPoint::Kind::CpuWrite: {
				auto* dst = reinterpret_cast<volatile uint32_t*>(p.addr);
				// A mirror publishes only while the guest still holds the dword from translate time.
				// The GPU store landing (the dword already equals `data`) and a later guest store both
				// leave the word alone. An empty `prior` is a label the GPU cannot write: always store.
				const bool conditional = p.mirror && !p.prior.empty();
				bool       published   = !conditional && !p.data.empty();
				if (!conditional) {
					for (size_t i = 0; i < p.data.size(); i++) {
						dst[p.one_addr ? 0 : i] = p.data[i];
					}
				} else if (p.one_addr) {
					const uint32_t cur = dst[0];
					if (cur == p.prior[0]) {
						for (uint32_t word: p.data) {
							dst[0] = word;
						}
						published = true;
					} else {
						published = !p.data.empty() && cur == p.data.back();
					}
				} else {
					published = !p.data.empty();
					for (size_t i = 0; i < p.data.size(); i++) {
						const uint32_t cur = dst[i];
						if (cur == p.data[i]) {
							continue;
						}
						if (i < p.prior.size() && cur == p.prior[i]) {
							dst[i] = p.data[i];
							continue;
						}
						published = false;
					}
				}
				if (published) {
					WakeAddress(p.addr);
				}
				break;
			}
			case TrackerPoint::Kind::Wake: WakeAddress(p.addr); break;
			case TrackerPoint::Kind::Timestamp: {
				const uint64_t clock = HostCallbacks().reference_clock ? HostCallbacks().reference_clock() : 0;
				std::memcpy(reinterpret_cast<void*>(p.addr), &clock, std::min<uint32_t>(p.size, 8));
				break;
			}
			case TrackerPoint::Kind::LodStats: {
				// As Kyty's command processor fills it (CpOpGetLodStats): in usage mode the banks the
				// translator published for this buffer, otherwise (or with no report) every texture fully
				// used; the first dword 1.
				static const int fill = [] {
					const char* v = std::getenv("KYTY_LOD_STATS_FILL");
					return v != nullptr ? static_cast<int>(std::strtol(v, nullptr, 0)) : 0xff;
				}();
				static const LodStatsMode   mode      = ParseLodStatsMode(std::getenv("KYTY_LOD_STATS_MODE"));
				static const unsigned long log_every = [] {
					const char* v = std::getenv("KYTY_LOD_STATS_LOG");
					return v != nullptr ? std::strtoul(v, nullptr, 0) : 0ul;
				}();
				static uint64_t  reports  = 0;
				static uint32_t  no_scan  = 0;
				if (LodStatsHw::Instance().On()) {
					// The reporter counted every sample since the previous report; it writes the header
					// and the records itself and resets the counters, as the PS5's ME does for the packet.
					static uint32_t failed = 0;
					const int       r      = LodStatsHw::Instance().Report(p.addr, p.size);
					if (r == 0) {
						break;
					}
					if (failed++ < 8) {
						std::fprintf(stderr, "LOD stats: hardware report at 0x%016llx failed: %s\n",
						             static_cast<unsigned long long>(p.addr), std::strerror(-r));
					}
				}
				std::bitset<256> banks;
				if (mode == LodStatsMode::Usage && TakeLodStatsBanks(p.addr, banks)) {
					const uint32_t used = WriteLodUsageReport(reinterpret_cast<void*>(p.addr), p.size, banks);
					if (log_every != 0 && reports % log_every == 0) {
						char ranges[2048];
						FormatLodBankRanges(banks, ranges, sizeof(ranges));
						std::fprintf(stderr, "LOD usage %llu (tracker): buffer 0x%016llx %zu banks used (%s); %u records used, the rest no-data\n",
						             static_cast<unsigned long long>(reports), static_cast<unsigned long long>(p.addr), banks.count(),
						             ranges, used);
					}
					reports++;
					break;
				}
				if (mode == LodStatsMode::Usage && no_scan++ < 8) {
					std::fprintf(stderr, "LOD stats (tracker): no bank scan for buffer 0x%016llx, filling every record\n",
					             static_cast<unsigned long long>(p.addr));
				}
				std::memset(reinterpret_cast<void*>(p.addr), fill, p.size);
				if (p.size >= 4) {
					*reinterpret_cast<volatile uint32_t*>(p.addr) = 1;
				}
				break;
			}
			case TrackerPoint::Kind::End: break;
		}
		if (e.done != nullptr) {
			Finish(*e.done);
		}
	}

	void Finish(Done& done) {
		const double start = NowSecShadow();
		if (done.whole) {
			done.then(std::span<const uint32_t>(done.buffer, done.size_dw));
		} else if (!done.tail.empty()) {
			// Kyty keeps a span to the packets until its command processor has run them: keep the last
			// tails alive, far more than it ever has queued.
			m_tails.push_back(std::move(done.tail));
			if (m_tails.size() > 1024) {
				m_tails.pop_front();
			}
			done.then(std::span<const uint32_t>(m_tails.back()));
		}
		const double end = NowSecShadow();
		Shadow::Instance().AddChain(done.picked - done.enqueued, end - done.enqueued, end - start);
	}

	// Worker side.
	std::mutex                              m_mutex;
	std::condition_variable                 m_cv;
	std::deque<Job>                         m_jobs;
	bool                                    m_native  = false;
	bool                                    m_acb_rings = false;
	uint32_t                                m_compute_mask = 0;
	uint64_t                                m_slot_seq[33] = {};
	bool                                    m_native_waits = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_NATIVE_WAITS");
		return v == nullptr || v[0] != '0';
	}();
	// KYTY_NATIVE_GPU_EARLY_WAITS=0: a compute wait on a gfx-written label stays held on the CPU until
	// the writer's CS has returned (see PlanRingWait).
	bool                                    m_early_waits = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_EARLY_WAITS");
		return v == nullptr || v[0] != '0';
	}();
	double                                  m_timeout = 12.0;
	std::map<uint32_t, std::deque<Piece>>   m_pending; // by queue
	uint32_t                                m_translating = 0;
	uint64_t                                m_held        = 0;
	uint32_t                                m_pending_dumps = 0; // NoteStaleHold's queue dumps so far
	std::unordered_map<uint64_t, std::pair<LabelWrite, uint64_t>> m_last_write; // label -> its last write emitted, and when
	uint64_t                                m_emits = 0; // Emit calls so far
	mutable std::map<uint32_t, uint32_t>    m_queue_index; // guest compute queue -> its KFD queue index (SlotOf)
	mutable std::map<uint32_t, uint32_t>    m_gfx_index;   // guest graphics queue -> the order it appeared in (SlotOf)
	mutable bool                            m_gfx_shared = true; // gfx waits with no known writer ahead held (SlotOf decides)
	std::array<LabelEvent, 1024>            m_label_trace {}; // KYTY_NATIVE_GPU_LABEL_TRACE ring
	uint64_t                                m_label_trace_next = 0;
	bool                                    m_wake        = false; // the tracker moved; recheck held labels
	uint64_t                                m_seq         = 0;
	bool                                    m_appended    = false;
	std::unordered_map<uint64_t, RingWrite> m_ring; // label -> the latest write emitted into the ring
	size_t                                  m_ring_sweep_at = 16384; // m_ring size that triggers its next sweep
	uint64_t                                m_batch = 1; // the batch being appended to (RingWrite::batch)
	uint64_t                                m_released_native   = 0;
	int                                     m_last_hold_reason  = 0;
	std::map<uint32_t, uint64_t>            m_hold_start_value; // the label when the queue's current hold began
	unsigned long                           m_holds_logged = 0; // KYTY_NATIVE_GPU_LOG_HOLDS
	std::map<uint32_t, int>                 m_hold_reason; // queue -> why its front piece was first held
	uint64_t                                m_released_observed = 0;
	std::atomic<uint64_t>                   m_last_cs {0};
	// Tracker side.
	std::mutex                              m_track_mutex;
	std::condition_variable                 m_track_cv;
	std::deque<Entry>                       m_entries;
	bool                                    m_drain = false;
	std::deque<std::vector<uint32_t>>       m_tails;
};

} // namespace

bool ShadowNativeLabels() {
	static const bool on = EnvOn("KYTY_NATIVE_GPU_PRIMARY") && ShadowThreaded() && PrimaryFeature("KYTY_NATIVE_GPU_LABELS");
	return on;
}

void ShadowSetFlipHost(std::function<void(int, int, int, int64_t)> flip) {
	HostCallbacks().flip = std::move(flip);
}

void ShadowSetHost(std::function<void(int, uint32_t)> interrupt, std::function<uint64_t()> reference_clock) {
	HostCallbacks().interrupt       = std::move(interrupt);
	HostCallbacks().reference_clock = std::move(reference_clock);
}

bool ShadowOnPipelineThread() {
	return t_pipeline_thread;
}

bool ShadowThreaded() {
	static const bool on = PrimaryFeature("KYTY_NATIVE_GPU_THREAD");
	return on;
}

void ShadowEnqueue(const uint32_t* buffer, uint32_t size_in_dwords, uint32_t queue,
                   std::function<void(std::span<const uint32_t>)> then) {
	if (ShadowNativeLabels()) {
		LabelPipeline::Instance().Enqueue(buffer, size_in_dwords, queue, std::move(then));
		return;
	}
	Pipeline::Instance().Enqueue(buffer, size_in_dwords, queue >= 0x20u && queue < 0x58u,
	                             [then = std::move(then), buffer, size_in_dwords] {
		                             then(std::span<const uint32_t>(buffer, size_in_dwords));
	                             });
}

bool ShadowDmabufPieces(uint64_t vaddr, uint64_t size, std::vector<NativeDmabufPiece>& out) {
	static const bool on = !EnvOn("KYTY_NATIVE_GPU_NO_ZERO_COPY");
	return on && PrimaryRendering() && Shadow::Instance().DmabufPieces(vaddr, size, out);
}

bool ShadowDisplayWithoutDcc(uint64_t addr, uint64_t size) {
	static const bool on = EnvOn("KYTY_NATIVE_GPU_GEM") && EnvOn("KYTY_NATIVE_GPU_PRIMARY") && !EnvOn("KYTY_NATIVE_GPU_DISPLAY_DCC");
	if (!on) {
		return false;
	}
	Shadow::Instance().AddDisplayRange(addr, size);
	return true;
}

void ShadowSetTessellationRings(uint64_t tf_ring_base, uint32_t tf_ring_bytes, uint64_t offchip_granularity,
                                uint64_t offchip_buffering) {
	// VGT_HS_OFFCHIP_PARAM: OFFCHIP_BUFFERING [8:0], OFFCHIP_GRANULARITY [10:9].
	const auto param = static_cast<uint32_t>((offchip_buffering & 0x1ffu) | ((offchip_granularity & 0x3u) << 9u));
	SetTessellationRings(tf_ring_base, tf_ring_bytes, param);
}

void ShadowFlipStats(double present_seconds) {
	static const bool on = EnvOn("KYTY_NATIVE_GPU_FLIP_STATS");
	if (on && Shadow::Instance().Active()) {
		Shadow::Instance().FlipStats(present_seconds);
	}
}

void ShadowSubmitAcb(const uint32_t* acb, uint32_t size_in_dwords) {
	if (acb == nullptr || size_in_dwords == 0) {
		return;
	}
	Shadow::Instance().Submit(std::span<const uint32_t>(acb, size_in_dwords), true);
}

bool NativeAsyncCompute() {
	return PrimaryRendering() && Shadow::Instance().NativeAcb();
}

namespace {

std::atomic<bool>& ShowNative() {
	static std::atomic<bool> show {EnvOn("KYTY_NATIVE_GPU_SHOW")};
	return show;
}

} // namespace

bool NativeLodStats() {
	static const bool on = EnvOn("KYTY_NATIVE_GPU_LOD_STATS");
	return on;
}

bool LodBankScanOn() {
	static const bool on = [] {
		const char*      mode   = std::getenv("KYTY_LOD_STATS_MODE");
		const LodStatsMode parsed = ParseLodStatsMode(mode);
		// Usage is the default. An explicit diagnostic mode other than "banks" does not scan
		// unless the log is on.
		if (mode != nullptr && mode[0] != '\0' && parsed != LodStatsMode::Usage && parsed != LodStatsMode::Banks) {
			return EnvOn("KYTY_LOD_STATS_LOG");
		}
		return true;
	}();
	return on;
}

bool TakeLodStatsBanks(uint64_t addr, std::bitset<256>& banks) {
	return LodBankScanOn() && Shadow::Instance().Active() && LodStatsReports().Take(addr, banks);
}

namespace {

// KYTY_LOD_STATS_TRAP: pages of LOD statistics buffers made inaccessible. Each CPU access is logged
// and single-stepped (EFLAGS.TF) with the page open, then the page is closed again, for up to
// kLodTrapAccesses accesses per armed buffer. The code around each new PC goes to
// lodstats-code-<pc>.bin (pc - 0x400 .. pc + 0x400) in the working directory.
constexpr uint32_t kLodTrapAccesses = 48;
struct LodTrap {
	std::atomic<uint64_t> page {0};
	std::atomic<uint64_t> bytes {0};
	std::atomic<uint64_t> buffer {0};
	std::atomic<uint64_t> size {0};
	std::atomic<uint32_t> left {0};
};
LodTrap                g_lod_traps[4];
thread_local LodTrap*  t_lod_stepping = nullptr;
std::atomic<uint32_t>  g_lod_trap_outside {0};
std::atomic<uint32_t>  g_lod_trap_lines {0};
uint64_t               g_lod_code_pcs[16] = {};
std::atomic<uint32_t>  g_lod_code_count {0};
struct sigaction       g_lod_prev_trap {};

void LodTrapStep(int sig, siginfo_t* info, void* native_context) {
	auto* uc = static_cast<ucontext_t*>(native_context);
	if (LodTrap* t = t_lod_stepping; t != nullptr) {
		t_lod_stepping = nullptr;
		uc->uc_mcontext.gregs[REG_EFL] &= ~0x100ll;
		if (t->left.load() > 0) {
			::mprotect(reinterpret_cast<void*>(t->page.load()), t->bytes.load(), PROT_NONE);
		} else {
			t->page.store(0);
		}
		return;
	}
	if ((g_lod_prev_trap.sa_flags & SA_SIGINFO) != 0 && g_lod_prev_trap.sa_sigaction != nullptr) {
		g_lod_prev_trap.sa_sigaction(sig, info, native_context);
	} else if (g_lod_prev_trap.sa_handler != SIG_DFL && g_lod_prev_trap.sa_handler != SIG_IGN && g_lod_prev_trap.sa_handler != nullptr) {
		g_lod_prev_trap.sa_handler(sig);
	}
}

void LodTrapDumpCode(uint64_t pc) {
	const uint32_t n = g_lod_code_count.load();
	for (uint32_t i = 0; i < n && i < 16; i++) {
		if (g_lod_code_pcs[i] == pc) {
			return;
		}
	}
	if (n >= 16) {
		return;
	}
	g_lod_code_pcs[n] = pc;
	g_lod_code_count.store(n + 1);
	char name[64];
	std::snprintf(name, sizeof(name), "lodstats-code-%llx.bin", static_cast<unsigned long long>(pc));
	const int fd = ::open(name, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
	if (fd >= 0) {
		// Code pages are readable; a page that is not would fault here, so stay inside pc's own page
		// and the neighbours the instruction stream needs anyway.
		(void)::write(fd, reinterpret_cast<const void*>(pc - 0x400), 0x800);
		::close(fd);
	}
}

} // namespace

void LodStatsTrapArm(uint64_t addr, uint64_t size) {
	static const bool installed = [] {
		struct sigaction sa {};
		sa.sa_sigaction = LodTrapStep;
		sa.sa_flags     = SA_SIGINFO | SA_ONSTACK;
		sigemptyset(&sa.sa_mask);
		return ::sigaction(SIGTRAP, &sa, &g_lod_prev_trap) == 0;
	}();
	if (!installed) {
		return;
	}
	const uint64_t page_size = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
	const uint64_t page      = addr & ~(page_size - 1);
	const uint64_t bytes     = ((addr + size + page_size - 1) & ~(page_size - 1)) - page;
	for (auto& t: g_lod_traps) {
		uint64_t expected = 0;
		if (t.page.compare_exchange_strong(expected, page)) {
			t.bytes.store(bytes);
			t.buffer.store(addr);
			t.size.store(size);
			t.left.store(kLodTrapAccesses);
			if (::mprotect(reinterpret_cast<void*>(page), bytes, PROT_NONE) != 0) {
				t.page.store(0);
			}
			return;
		}
	}
}

bool LodStatsTrapFault(uint64_t vaddr, uint64_t pc, bool write, void* native_context) {
	for (auto& t: g_lod_traps) {
		const uint64_t page  = t.page.load();
		const uint64_t bytes = t.bytes.load();
		if (page == 0 || vaddr < page || vaddr >= page + bytes) {
			continue;
		}
		::mprotect(reinterpret_cast<void*>(page), bytes, PROT_READ | PROT_WRITE);
		const uint64_t buffer = t.buffer.load();
		// Other data sharing the pages is only stepped over (up to a limit), not counted or logged.
		const bool inside = vaddr >= buffer && vaddr < buffer + t.size.load();
		if (!inside) {
			if (native_context != nullptr && g_lod_trap_outside.fetch_add(1) < 200000) {
				t_lod_stepping = &t;
				static_cast<ucontext_t*>(native_context)->uc_mcontext.gregs[REG_EFL] |= 0x100ll;
			} else {
				t.left.store(0);
				t.page.store(0);
			}
			return true;
		}
		if (g_lod_trap_lines.fetch_add(1) < 400) {
			char      line[192];
			const int n = std::snprintf(line, sizeof(line),
			                            "native shadow: LOD stats TRAP %s at 0x%llx (buffer 0x%llx +%lld) by pc 0x%llx\n",
			                            write ? "write" : "read", static_cast<unsigned long long>(vaddr),
			                            static_cast<unsigned long long>(buffer), static_cast<long long>(vaddr - buffer),
			                            static_cast<unsigned long long>(pc));
			if (n > 0) {
				(void)::write(2, line, static_cast<size_t>(n));
			}
		}
		LodTrapDumpCode(pc);
		// Step over the access with the page open, then close it again (LodTrapStep).
		if (t.left.fetch_sub(1) > 1 && native_context != nullptr) {
			t_lod_stepping = &t;
			static_cast<ucontext_t*>(native_context)->uc_mcontext.gregs[REG_EFL] |= 0x100ll;
		} else {
			t.left.store(0);
			t.page.store(0);
		}
		return true;
	}
	return false;
}

bool PrimaryRendering() {
	static const bool primary = EnvOn("KYTY_NATIVE_GPU_PRIMARY");
	// KYTY_NATIVE_GPU_NULL=1: nothing renders at all, natively or through Vulkan, while the game and
	// Kyty's command processor run on. For measuring what a game does on the CPU side (e.g. how much
	// guest memory it really touches) when rendering is what stops it.
	static const bool null_rendering = EnvOn("KYTY_NATIVE_GPU_NULL");
	return null_rendering || (primary && Shadow::Instance().Active());
}

bool ShadowShowsNativeFrames() {
	return (ShowNative().load(std::memory_order_relaxed) || PrimaryRendering()) &&
	       Shadow::Instance().Active();
}

void ShadowToggleNativeFrames() {
	const bool now = !ShowNative().load(std::memory_order_relaxed);
	ShowNative().store(now, std::memory_order_relaxed);
	std::fprintf(stderr, "native shadow: showing %s frames\n", now ? "NATIVE" : "Vulkan");
}

namespace {

// KYTY_NATIVE_GPU_DUMP_FLIPS: comma-separated flip numbers or ranges "first-last" with an optional
// "/step" ("300,600" or "300-2000/10").
bool FlipWanted(const char* list, uint64_t n) {
	for (const char* p = list; *p != '\0';) {
		char*                    end   = nullptr;
		const unsigned long long first = std::strtoull(p, &end, 10);
		if (end == p) {
			return false;
		}
		unsigned long long last = first;
		unsigned long long step = 1;
		p                       = end;
		if (*p == '-') {
			last = std::strtoull(p + 1, &end, 10);
			p    = end;
		}
		if (*p == '/') {
			step = std::strtoull(p + 1, &end, 10);
			p    = end;
		}
		if (n >= first && n <= last && step != 0 && (n - first) % step == 0) {
			return true;
		}
		if (*p == ',') {
			p++;
		} else if (*p != '\0') {
			return false;
		}
	}
	return false;
}

void WriteFile(const char* path, const void* data, uint64_t size) {
	if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
		std::fwrite(data, 1, size, f);
		std::fclose(f);
	}
}

} // namespace

namespace {

// Bytes per pixel of a CB_COLORn_INFO.FORMAT, 0 when unknown.
uint32_t ColorFormatBytes(uint32_t info) {
	switch ((info >> 2u) & 0x1fu) {
		case 1: return 1;                                            // 8
		case 2: case 3: case 16: case 17: case 18: case 19: return 2; // 16, 8_8, 5_6_5, 1_5_5_5, 5_5_5_1, 4_4_4_4
		case 4: case 5: case 6: case 7: case 8: case 9: case 10: case 20: case 21: return 4;
		case 11: case 12: case 22: return 8;                         // 32_32, 16_16_16_16, X24_8_32
		case 14: return 16;                                          // 32_32_32_32
		default: return 0;
	}
}

// For each color target bound since the last dump: guest memory as the native path left it, then
// after Kyty's Vulkan copy is written back over it. Differing targets are written out; a summary of
// all of them goes to flip_<n>.rts.txt, so the first target whose contents diverge can be found.
void DumpRenderTargets(const char* dir, unsigned long long flip,
                       const std::function<uint32_t(uint64_t, uint64_t)>& download_region) {
	auto       targets = Shadow::Instance().TakeFrameTargets();
	char       path[512];
	std::snprintf(path, sizeof(path), "%s/flip_%llu.rts.txt", dir, flip);
	std::FILE* summary = std::fopen(path, "w");
	if (summary == nullptr) {
		return;
	}
	std::fprintf(summary, "# addr width height format swizzle draws bytes vulkan_images differing_bytes\n");
	for (const auto& t: targets) {
		const uint32_t bpp = ColorFormatBytes(t.info);
		// The surface itself, padded to whole 64 KiB swizzle blocks.
		const uint64_t want = bpp != 0 ? (static_cast<uint64_t>(t.width) * t.height * bpp + 0xffffu) & ~0xffffull
		                               : std::min<uint64_t>(t.size, 64ull << 20u);
		std::vector<uint8_t> native(want), vulkan(want);
		const uint64_t       got = Shadow::Instance().ReadGuest(t.addr, native.data(), want);
		if (got == 0) {
			continue;
		}
		const uint32_t images = download_region ? download_region(t.addr, got) : 0;
		Shadow::Instance().ReadGuest(t.addr, vulkan.data(), got);
		uint64_t differing = 0;
		for (uint64_t i = 0; i < got; i++) {
			differing += native[i] != vulkan[i] ? 1u : 0u;
		}
		std::fprintf(summary, "0x%llx %u %u %u 0x%x %u %llu %u %llu\n", static_cast<unsigned long long>(t.addr),
		             t.width, t.height, (t.info >> 2u) & 0x1fu, t.attrib3 & 0x1fu, t.draws,
		             static_cast<unsigned long long>(got), images, static_cast<unsigned long long>(differing));
		if (!download_region) { // primary mode: nothing to compare with, keep every native target
			std::snprintf(path, sizeof(path), "%s/flip_%llu.rt_%llx.native.raw", dir, flip,
			              static_cast<unsigned long long>(t.addr));
			WriteFile(path, native.data(), got);
		} else if (differing != 0) {
			std::snprintf(path, sizeof(path), "%s/flip_%llu.rt_%llx.native.raw", dir, flip,
			              static_cast<unsigned long long>(t.addr));
			WriteFile(path, native.data(), got);
			std::snprintf(path, sizeof(path), "%s/flip_%llu.rt_%llx.vulkan.raw", dir, flip,
			              static_cast<unsigned long long>(t.addr));
			WriteFile(path, vulkan.data(), got);
		}
	}
	std::fclose(summary);
}

} // namespace

void ShadowOnFlip(uint64_t address, uint64_t size, uint32_t width, uint32_t height,
                  uint32_t bytes_per_pixel, uint32_t pixel_format, uint32_t tile_mode,
                  const std::function<uint32_t(uint64_t, uint64_t)>& download_region) {
	static std::atomic<uint64_t> flips {0};
	const uint64_t               n    = flips.fetch_add(1, std::memory_order_relaxed);
	if (address != 0 && size != 0 && Shadow::Instance().Active()) {
		Shadow::Instance().KeepResident(address, size);
	}
	if (Shadow::Instance().Active()) {
		Shadow::Instance().Ps5RegsOnFlip();
	}
	// KYTY_NATIVE_GPU_CTL=1: "dump N" on the control socket dumps the next N flips, render targets
	// included (see ControlSocket).
	static const bool on_demand   = EnvOn("KYTY_NATIVE_GPU_CTL");
	static uint32_t   demand_left = 0;
	if (on_demand && demand_left == 0) {
		if (const uint32_t count = DemandedDumps().exchange(0); count != 0) {
			demand_left = std::min(count, 120u);
			std::fprintf(stderr, "native shadow: on-demand dump of %u flips from flip %llu\n", demand_left,
			             static_cast<unsigned long long>(n));
		}
	}
	const char* list   = std::getenv("KYTY_NATIVE_GPU_DUMP_FLIPS");
	const bool  wanted = demand_left > 0 || (list != nullptr && FlipWanted(list, n));
	if ((list == nullptr && !on_demand) || address == 0 || size == 0 || !Shadow::Instance().Active()) {
		return;
	}
	if (!wanted) {
		(void)Shadow::Instance().TakeFrameTargets(); // keep only the dumped frame's targets
		(void)Shadow::Instance().TakeFrameDraws();
		return;
	}
	if (demand_left > 0) {
		demand_left--;
	}
	const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
	if (dir == nullptr || dir[0] == '\0') {
		dir = ".";
	}
	const auto flip = static_cast<unsigned long long>(n);
	char       path[512];
	// The native result first: it is what guest memory holds right now.
	std::snprintf(path, sizeof(path), "%s/flip_%llu.native.raw", dir, flip);
	WriteFile(path, reinterpret_cast<const void*>(address), size);
	// Then Kyty's Vulkan rendering of the same DCBs, written back over the same guest bytes.
	uint32_t vulkan_images = 0;
	if (!PrimaryRendering() && download_region) {
		vulkan_images = download_region(address, size);
		if (vulkan_images > 0) {
			std::snprintf(path, sizeof(path), "%s/flip_%llu.vulkan.raw", dir, flip);
			WriteFile(path, reinterpret_cast<const void*>(address), size);
		}
		DumpRenderTargets(dir, flip, download_region);
	} else if (PrimaryRendering() && (EnvOn("KYTY_NATIVE_GPU_DUMP_RTS") || on_demand)) {
		DumpRenderTargets(dir, flip, nullptr);
	}
	// KYTY_NATIVE_GPU_DUMP_RANGES=0xADDR:0xSIZE[,...]: guest ranges saved next to the flip, e.g. a
	// game's intermediate targets when the presented surface stays empty.
	if (const char* ranges = std::getenv("KYTY_NATIVE_GPU_DUMP_RANGES"); ranges != nullptr) {
		for (const char* p = ranges; *p != '\0';) {
			char*          end   = nullptr;
			const uint64_t raddr = std::strtoull(p, &end, 0);
			if (end == p || *end != ':') {
				break;
			}
			const uint64_t rsize = std::strtoull(end + 1, &end, 0);
			std::vector<uint8_t> bytes(rsize);
			const uint64_t       got = Shadow::Instance().ReadGuest(raddr, bytes.data(), rsize);
			std::snprintf(path, sizeof(path), "%s/flip_%llu.range_%llx.raw", dir, flip, static_cast<unsigned long long>(raddr));
			WriteFile(path, bytes.data(), got);
			p = (*end == ',') ? end + 1 : end;
		}
	}
	// Every draw and dispatch of the frame, with the arguments of indirect ones as the GPU left them.
	if (const auto draws = Shadow::Instance().TakeFrameDraws(); !draws.empty()) {
		std::snprintf(path, sizeof(path), "%s/flip_%llu.draws.txt", dir, flip);
		if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
			std::fprintf(f, "# q buf idx op count inst ps gs cb0 zbase ud_hash gs_hash indirect args\n");
			for (const auto& fd: draws) {
				const auto& d = fd.draw;
				uint32_t    args[5] = {};
				if (d.op == 0x46u && d.indirect != 0) {
					// An occlusion dump: 16 RBs x 64-bit counters 16 bytes apart (bit 63 = written). args: RBs
					// written, their counter sum (low 32 bits), and the first counter's low and high dwords.
					uint64_t c[32] = {};
					Shadow::Instance().ReadGuest(d.indirect, c, sizeof(c));
					for (int rb = 1; rb < 16; rb++) {
						c[rb] = c[rb * 2];
					}
					uint64_t sum = 0;
					for (int rb = 0; rb < 16; rb++) {
						const uint64_t v = c[rb];
						args[0] += static_cast<uint32_t>(v >> 63u);
						sum += v & 0x7fffffffffffffffull;
					}
					args[1] = static_cast<uint32_t>(sum);
					args[2] = static_cast<uint32_t>(c[0]);
					args[3] = static_cast<uint32_t>(c[0] >> 32u);
				} else if (d.indirect != 0) {
					Shadow::Instance().ReadGuest(d.indirect, args, sizeof(args));
				}
				std::fprintf(f, "%s %u %u 0x%02x %u %u 0x%llx 0x%llx 0x%llx 0x%llx %08x %08x 0x%llx %u,%u,%u,%u,%u\n", fd.acb ? "acb" : "dcb",
				             fd.buffer, d.index, d.op, d.count, d.instances, static_cast<unsigned long long>(d.ps),
				             static_cast<unsigned long long>(d.gs), static_cast<unsigned long long>(d.cb0),
				             static_cast<unsigned long long>(d.zbase), d.ud_hash, d.gs_hash, static_cast<unsigned long long>(d.indirect),
				             args[0], args[1], args[2], args[3], args[4]);
			}
			std::fclose(f);
		}
	}
	std::snprintf(path, sizeof(path), "%s/flip_%llu.txt", dir, flip);
	if (std::FILE* f = std::fopen(path, "w"); f != nullptr) {
		std::fprintf(f,
		             "address 0x%llx\nsize %llu\nwidth %u\nheight %u\nbytes_per_pixel %u\n"
		             "pixel_format %u\ntile_mode 0x%x\nvulkan_images %u\n",
		             static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
		             width, height, bytes_per_pixel, pixel_format, tile_mode, vulkan_images);
		std::fclose(f);
	}
	std::fprintf(stderr, "native shadow: dumped flip %llu (%ux%u) native%s to %s\n", flip, width,
	             height, vulkan_images > 0 ? " + vulkan" : "", dir);
}

} // namespace Libs::Graphics::NativeGpu
