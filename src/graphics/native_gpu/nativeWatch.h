#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEWATCH_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEWATCH_H_

// The hang watchdog's decisions (Shadow::WatchLoop does the I/O): samples in, a trigger out. Pure, so the
// thresholds are unit-tested without a GPU (tests/NativeGpuWatchTests.cpp).
//
// On the PS5 the kernel owns the title's queues: a GPU timeout or a page fault resets the GPU or the queue and the
// title is terminated, while the system's compositor on the other gfx pipe stays up. On the BC-250 amdgpu owns the
// one gfx ring, and its own recovery is what kills the board: a per-queue reset whose ring test fails (shadowing
// left on by a job that never reached its end, waves that never finish) escalates to MODE1, which the BC-250 does
// not come back from. So Kyty plays the PS5 kernel's part before amdgpu's timer fires: it notices a stuck gfx job or
// a fault well inside amdgpu's gfx lockup_timeout, stops the title's GPU work as a title's end does
// (Shadow::StopGpu: exit guards, KFD queues destroyed, guest waits released) and ends the process.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>

namespace Libs::Graphics::NativeGpu {

// amdgpu's gfx job timeout in ms from the lockup_timeout module parameter ("[GFX[,Compute[,SDMA[,Video]]]]"), as
// amdgpu_device_get_job_timeout_settings reads it: a missing or empty parameter, or a 0 field, is the 2000 ms
// default; a negative one disables the timeout (returned as -1). A single value applies to every engine, so the first
// field is the gfx one either way. Bad syntax keeps the default (amdgpu refuses to load with it anyway).
inline int64_t ParseGfxLockupTimeoutMs(std::string_view text) {
	constexpr int64_t kDefault = 2000;
	const size_t      comma    = text.find(',');
	std::string       field(text.substr(0, comma));
	while (!field.empty() && (field.back() == '\n' || field.back() == ' ' || field.back() == '\t')) {
		field.pop_back();
	}
	if (field.empty()) {
		return kDefault;
	}
	char*           end = nullptr;
	const long long v   = std::strtoll(field.c_str(), &end, 0);
	if (end == field.c_str() || *end != '\0') {
		return kDefault;
	}
	return v == 0 ? kDefault : (v < 0 ? -1 : static_cast<int64_t>(v));
}

// GCVM_L2_PROTECTION_FAULT_STATUS as AMDGPU_INFO_GPUVM_FAULT returns it (gc_10_1_0_sh_mask.h): the faulting client
// (CID 17:9), read or write (RW 18) and the VMID (23:20; 1-7 a gfx or CS job's, 8-15 a KFD queue's).
struct GpuvmFault {
	uint64_t addr   = 0;
	uint32_t status = 0;
	uint32_t vmhub  = 0;
};
constexpr uint32_t FaultVmid(uint32_t status) { return (status >> 20u) & 0xfu; }
constexpr uint32_t FaultCid(uint32_t status) { return (status >> 9u) & 0x1ffu; }
constexpr bool     FaultWrite(uint32_t status) { return ((status >> 18u) & 1u) != 0; }

struct WatchConfig {
	int64_t t_gfx_ms  = 2000;   // amdgpu's gfx timeout (ParseGfxLockupTimeoutMs); < 0: none
	double  gfx_ms    = 600.0;  // T1: no gfx progress for this long
	double  stream_ms = 3000.0; // T1s: the CP parked on a streamed DCB's next chunk for this long
	double  margin_ms = 1000.0; // the stop must be done this long before amdgpu's timer fires

	// T1's default: 30% of amdgpu's timeout, between 0.5 and 1.5 s. The audit (PR-9) saw gfx waits up to 54 ms and
	// jobs up to 69 ms, so 1.5 s is a 20x margin; 1.5 s also leaves the stop 2.5 s of a 5 s timeout.
	static double DefaultGfxMs(int64_t t_gfx_ms) {
		return t_gfx_ms < 0 ? 1500.0 : std::clamp(0.3 * static_cast<double>(t_gfx_ms), 500.0, 1500.0);
	}
	// A streamed DCB's head job may not live longer than this (KYTY_NATIVE_GPU_STREAM_MAX_MS): amdgpu times the CS,
	// not the progress inside it. 2 s under the timeout, half of a short one; 0 (no bound) without a timeout.
	static double DefaultStreamMaxMs(int64_t t_gfx_ms) {
		if (t_gfx_ms < 0) {
			return 0.0;
		}
		return t_gfx_ms > 4000 ? static_cast<double>(t_gfx_ms - 2000) : 0.5 * static_cast<double>(t_gfx_ms);
	}
};

// One poll of the watchdog. Times in ms on one clock.
struct WatchSample {
	double                  now_ms        = 0.0;
	bool                    gfx_pending   = false; // a gfx job of ours has not signalled
	double                  gfx_start_ms  = 0.0;   // the oldest one's start estimate (submit, or the one before it signalling)
	std::array<uint32_t, 3> progress {};           // the gfx rings' tracker words and the stream's chunk counter
	bool                    stream_parked = false; // the CP has entered the last published chunk of an open stream
	bool                    fault_valid   = false; // AMDGPU_INFO_GPUVM_FAULT answered
	GpuvmFault              fault;
	bool                    reset = false; // our fence came back with an error, or the context is guilty / lost VRAM
};

enum class WatchTrigger {
	None,
	StreamParkLog, // the CP waits for a chunk the worker owes: logged once, a CPU-side stall
	GfxStall,      // T1
	StreamStall,   // T1s
	VmFault,       // T2
	Reset,         // T4: amdgpu has reset our job already
};

inline const char* WatchTriggerName(WatchTrigger t) {
	switch (t) {
		case WatchTrigger::StreamParkLog: return "stream park";
		case WatchTrigger::GfxStall: return "gfx stall";
		case WatchTrigger::StreamStall: return "stream stall";
		case WatchTrigger::VmFault: return "GPU page fault";
		case WatchTrigger::Reset: return "amdgpu reset";
		default: return "none";
	}
}

class WatchDecider {
public:
	explicit WatchDecider(const WatchConfig& c): m_c(c) {}

	WatchTrigger Feed(const WatchSample& s) {
		if (!m_have_progress || s.progress != m_progress) {
			m_progress      = s.progress;
			m_t_progress    = s.now_ms;
			m_have_progress = true;
			m_park_logged   = false;
		}
		if (s.reset) {
			return WatchTrigger::Reset;
		}
		// amdgpu keeps the VM's last fault and never clears it: the first answer is the baseline, a change is a new one.
		if (s.fault_valid) {
			const bool fresh = m_have_fault && s.fault.status != 0 && (s.fault.addr != m_fault.addr || s.fault.status != m_fault.status);
			m_fault          = s.fault;
			m_have_fault     = true;
			if (fresh) {
				return WatchTrigger::VmFault;
			}
		}
		if (!s.gfx_pending) {
			m_park_logged = false;
			return WatchTrigger::None;
		}
		m_start = s.gfx_start_ms;
		// Stuck: the oldest job has been the oldest that long, and no progress word moved that long.
		const double stalled = s.now_ms - std::max(m_t_progress, s.gfx_start_ms);
		if (s.stream_parked) {
			if (stalled >= m_c.stream_ms) {
				return WatchTrigger::StreamStall;
			}
			if (stalled >= m_c.gfx_ms && !m_park_logged) {
				m_park_logged = true;
				return WatchTrigger::StreamParkLog;
			}
			return WatchTrigger::None;
		}
		return stalled >= m_c.gfx_ms ? WatchTrigger::GfxStall : WatchTrigger::None;
	}

	// When amdgpu's timer fires for the oldest job at the latest, less the margin: it may run from the job's start
	// (whether gamescope's later submissions re-arm it is not known), so that is assumed. Infinite without a timeout.
	[[nodiscard]] double DeadlineMs() const {
		if (m_c.t_gfx_ms < 0) {
			return std::numeric_limits<double>::infinity();
		}
		return m_start + static_cast<double>(m_c.t_gfx_ms) - m_c.margin_ms;
	}
	[[nodiscard]] double LastProgressMs() const { return m_t_progress; }
	[[nodiscard]] double StartMs() const { return m_start; }
	[[nodiscard]] const GpuvmFault& Fault() const { return m_fault; }
	[[nodiscard]] const WatchConfig& Config() const { return m_c; }

private:
	WatchConfig             m_c;
	double                  m_start         = 0.0;
	std::array<uint32_t, 3> m_progress {};
	double                  m_t_progress    = 0.0;
	bool                    m_have_progress = false;
	bool                    m_park_logged   = false;
	GpuvmFault              m_fault;
	bool                    m_have_fault = false;
};

// The gfx VMID a reservation got, from the IH LUT masks (bit n: VMID n serves our PASID) read before and after the
// priming job: the one new bit, or with none new (the reserved VMID served us already) the only bit. 0 when that is
// not one gfx VMID (1-7): stale entries of VMIDs this VM held before leave it unclear.
inline uint32_t ReservedGfxVmid(uint32_t before, uint32_t after) {
	after &= 0xfeu;
	uint32_t pick = after & ~before;
	if (pick == 0) {
		pick = after;
	}
	if (pick == 0 || (pick & (pick - 1u)) != 0) {
		return 0;
	}
	uint32_t vmid = 1;
	while ((pick & (1u << vmid)) == 0) {
		vmid++;
	}
	return vmid;
}

} // namespace Libs::Graphics::NativeGpu

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVEWATCH_H_
