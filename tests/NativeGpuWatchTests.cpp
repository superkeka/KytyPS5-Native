#include "graphics/native_gpu/nativeWatch.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using Libs::Graphics::NativeGpu::FaultCid;
using Libs::Graphics::NativeGpu::FaultVmid;
using Libs::Graphics::NativeGpu::FaultWrite;
using Libs::Graphics::NativeGpu::ParseGfxLockupTimeoutMs;
using Libs::Graphics::NativeGpu::ReservedGfxVmid;
using Libs::Graphics::NativeGpu::WatchConfig;
using Libs::Graphics::NativeGpu::WatchDecider;
using Libs::Graphics::NativeGpu::WatchSample;
using Libs::Graphics::NativeGpu::WatchTrigger;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuWatchTests: failed: %s\n", message);
		std::abort();
	}
}

// amdgpu_device_get_job_timeout_settings' reading of lockup_timeout, gfx field.
void TestLockupTimeout() {
	Check(ParseGfxLockupTimeoutMs("5000,10000,10000,5000\n") == 5000, "the BC-250's cmdline: 5 s for gfx");
	Check(ParseGfxLockupTimeoutMs("") == 2000 && ParseGfxLockupTimeoutMs("\n") == 2000, "missing: amdgpu's 2 s default");
	Check(ParseGfxLockupTimeoutMs("0,10000") == 2000, "0: the default");
	Check(ParseGfxLockupTimeoutMs("-1") == -1, "negative: no timeout");
	Check(ParseGfxLockupTimeoutMs("10000") == 10000, "one value for every engine");
	Check(ParseGfxLockupTimeoutMs("0x1388,1") == 5000, "kstrtol base 0");
	Check(ParseGfxLockupTimeoutMs("abc") == 2000, "bad syntax: the default");
	Check(WatchConfig::DefaultGfxMs(5000) == 1500.0 && WatchConfig::DefaultGfxMs(2000) == 600.0 && WatchConfig::DefaultGfxMs(1000) == 500.0 &&
	          WatchConfig::DefaultGfxMs(-1) == 1500.0,
	      "T1 default: clamp(0.3 x T, 500, 1500)");
	Check(WatchConfig::DefaultStreamMaxMs(5000) == 3000.0 && WatchConfig::DefaultStreamMaxMs(2000) == 1000.0 &&
	          WatchConfig::DefaultStreamMaxMs(-1) == 0.0,
	      "stream bound: T - 2 s, half of a short T, none without");
}

void TestFaultDecode() {
	// A gfx-ring CPG write fault in VMID 3, CID 0x40 (made-up client).
	const uint32_t status = (3u << 20u) | (1u << 18u) | (0x40u << 9u) | 1u;
	Check(FaultVmid(status) == 3 && FaultWrite(status) && FaultCid(status) == 0x40u, "status fields: VMID 23:20, RW 18, CID 17:9");
}

WatchSample Sample(double now, bool pending, double start, uint32_t p0 = 0) {
	WatchSample s;
	s.now_ms       = now;
	s.gfx_pending  = pending;
	s.gfx_start_ms = start;
	s.progress     = {p0, 0u, 0u};
	return s;
}

void TestStall() {
	WatchConfig c;
	c.t_gfx_ms  = 5000;
	c.gfx_ms    = 1500;
	c.stream_ms = 3000;
	c.margin_ms = 1000;
	WatchDecider d(c);
	Check(d.Feed(Sample(0, false, 0)) == WatchTrigger::None, "idle");
	Check(d.Feed(Sample(100, true, 100)) == WatchTrigger::None, "a job just submitted");
	Check(d.Feed(Sample(1500, true, 100)) == WatchTrigger::None, "1.4 s: under T1");
	Check(d.Feed(Sample(1600, true, 100)) == WatchTrigger::GfxStall, "1.5 s with no progress: T1");
	Check(d.DeadlineMs() == 100.0 + 5000.0 - 1000.0, "deadline: start + T - margin");
	// Progress resets it.
	WatchDecider e(c);
	e.Feed(Sample(0, true, 0, 1));
	Check(e.Feed(Sample(1400, true, 0, 2)) == WatchTrigger::None, "a tracker word moved");
	Check(e.Feed(Sample(2800, true, 0, 2)) == WatchTrigger::None, "1.4 s since it moved");
	Check(e.Feed(Sample(2900, true, 0, 2)) == WatchTrigger::GfxStall, "1.5 s since it moved");
	// A new oldest job (the previous one signalled) restarts the clock even without progress words.
	WatchDecider f(c);
	f.Feed(Sample(0, true, 0));
	Check(f.Feed(Sample(1400, true, 1300)) == WatchTrigger::None, "the oldest job changed at 1.3 s");
	Check(f.Feed(Sample(2799, true, 1300)) == WatchTrigger::None && f.Feed(Sample(2800, true, 1300)) == WatchTrigger::GfxStall,
	      "T1 counts from the later of the two");
	Check(std::isinf(WatchDecider(WatchConfig {-1, 1500, 3000, 1000}).DeadlineMs()), "no amdgpu timeout: no deadline");
}

void TestStreamPark() {
	WatchConfig c;
	c.t_gfx_ms  = 5000;
	c.gfx_ms    = 1500;
	c.stream_ms = 3000;
	WatchDecider d(c);
	WatchSample  s = Sample(0, true, 0);
	s.stream_parked = true;
	Check(d.Feed(s) == WatchTrigger::None, "parked, young");
	s.now_ms = 1600;
	Check(d.Feed(s) == WatchTrigger::StreamParkLog, "parked past T1: logged once, not stopped");
	s.now_ms = 2000;
	Check(d.Feed(s) == WatchTrigger::None, "logged once");
	s.now_ms = 3000;
	Check(d.Feed(s) == WatchTrigger::StreamStall, "parked 3 s: T1s");
}

void TestFaultAndReset() {
	WatchConfig  c;
	WatchDecider d(c);
	WatchSample  s = Sample(0, false, 0);
	s.fault_valid  = true;
	s.fault        = {0x1000, 0x00300001u, 0};
	Check(d.Feed(s) == WatchTrigger::None, "the first answer is the baseline (an old fault)");
	s.now_ms = 20;
	Check(d.Feed(s) == WatchTrigger::None, "the same fault again");
	s.fault = {0x2000, 0x00300001u, 0};
	Check(d.Feed(s) == WatchTrigger::VmFault && FaultVmid(d.Fault().status) == 3, "a new one: T2");
	Check(d.Feed(s) == WatchTrigger::None, "once");
	s.reset = true;
	Check(d.Feed(s) == WatchTrigger::Reset, "a fence error or a guilty context: T4");
	WatchDecider z(c);
	WatchSample  zs = Sample(0, false, 0);
	zs.fault_valid  = true;
	Check(z.Feed(zs) == WatchTrigger::None, "no fault yet");
	zs.fault = {0x3000, 0x00800001u, 0};
	Check(z.Feed(zs) == WatchTrigger::VmFault, "the first fault after an empty baseline (a KFD VMID)");
}

void TestReservedVmid() {
	Check(ReservedGfxVmid(0, 1u << 5) == 5, "nothing before: the primed VMID");
	Check(ReservedGfxVmid(1u << 3, (1u << 3) | (1u << 6)) == 6, "a stale entry and a new one: the new one");
	Check(ReservedGfxVmid(1u << 4, 1u << 4) == 4, "the reserved VMID served us already, alone");
	Check(ReservedGfxVmid((1u << 2) | (1u << 4), (1u << 2) | (1u << 4)) == 0, "two old entries, none new: unclear");
	Check(ReservedGfxVmid(0, (1u << 2) | (1u << 4)) == 0, "two new entries: unclear");
	Check(ReservedGfxVmid(0, 0) == 0, "no entry");
	Check(ReservedGfxVmid(0, 1u) == 0, "VMID 0 is the kernel's");
	Check(ReservedGfxVmid(1u << 7, 1u << 7) == 7, "VMID 7");
}

} // namespace

int main() {
	TestLockupTimeout();
	TestFaultDecode();
	TestStall();
	TestStreamPark();
	TestFaultAndReset();
	TestReservedVmid();
	std::fprintf(stderr, "NativeGpuWatchTests: all passed\n");
	return 0;
}
