// Exercise production output scheduling with a deterministic clock and consuming SDL queues.
#include "common/threads.h"
#include "libs/dualSenseHaptics.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace {
void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AudioOutTimingTests: %s\n", message);
		std::abort();
	}
}

struct Stream {
	SDL_AudioSpec spec;
	uint64_t      frames = 0; // Millionths of a frame retain exact simulated consumption.
	uint64_t      updated;
	std::vector<uint8_t> pcm;
	uint64_t             opened = 0;
	double               pulled = 0; // Device frames taken so far, whole periods if it has one.
};

uint64_t                             now           = 1000000;
double                               device_speed  = 1.0; // Its real rate over its nominal one.
uint32_t                             device_period = 0;   // Frames it takes at once (0: smoothly).
uint32_t                             oversleep = 0, processing = 0;
std::vector<uint32_t>                sleeps;
std::vector<std::unique_ptr<Stream>> streams;
bool                                 fail_open = false, fail_put = false, stalled = false;
bool                                 pad_connected = false, pad_bluetooth = false;
uint64_t                             pad_queue_us = 0;
const float*                         queued_pad_gains = nullptr;
int                                  clears = 0;

Stream& GetStream(SDL_AudioStream* stream) {
	return *reinterpret_cast<Stream*>(stream);
}

void Drain(Stream& stream) {
	// Device frames since the stream opened.
	const double device =
	    static_cast<double>(now - stream.opened) * stream.spec.freq * device_speed / 1e6;
	const double due =
	    device_period != 0 ? std::floor(device / device_period) * device_period : device;
	if (!stalled && device_speed == 1.0 && device_period == 0) {
		// Exact in millionths of a frame at the nominal rate.
		const auto consumed = (now - stream.updated) * stream.spec.freq;
		stream.frames -= std::min(stream.frames, consumed);
	} else if (!stalled && due > stream.pulled) {
		const auto consumed = static_cast<uint64_t>((due - stream.pulled) * 1e6);
		stream.frames -= std::min(stream.frames, consumed);
	}
	stream.pulled  = std::max(stream.pulled, due);
	stream.updated = now;
}
} // namespace

namespace Fake {
bool InitSubSystem(SDL_InitFlags) {
	return true;
}
void QuitSubSystem(SDL_InitFlags) {}
bool ResumeAudioStreamDevice(SDL_AudioStream*) {
	return true;
}

SDL_AudioStream* OpenAudioDeviceStream(SDL_AudioDeviceID, const SDL_AudioSpec* spec,
                                       SDL_AudioStreamCallback, void*) {
	if (fail_open) {
		return nullptr;
	}

	streams.push_back(std::make_unique<Stream>(Stream {*spec, 0, now, {}, now}));
	return reinterpret_cast<SDL_AudioStream*>(streams.back().get());
}

void DestroyAudioStream(SDL_AudioStream* stream) {
	const auto it = std::find_if(streams.begin(), streams.end(), [stream](const auto& value) {
		return value.get() == &GetStream(stream);
	});
	Check(it != streams.end(), "unknown stream destroyed");
	streams.erase(it);
}

int GetAudioStreamQueued(SDL_AudioStream* stream) {
	auto& state = GetStream(stream);
	Drain(state);
	return static_cast<int>(state.frames / 1000000 * SDL_AUDIO_BYTESIZE(state.spec.format) *
	                        state.spec.channels);
}

bool PutAudioStreamData(SDL_AudioStream* stream, const void* data, int bytes) {
	now += processing;
	if (fail_put) {
		return false;
	}
	auto& state = GetStream(stream);
	const auto* pcm = static_cast<const uint8_t*>(data);
	state.pcm.assign(pcm, pcm + bytes);
	Drain(state);
	state.frames += static_cast<uint64_t>(bytes) * 1000000 /
	                (SDL_AUDIO_BYTESIZE(state.spec.format) * state.spec.channels);
	return true;
}

SDL_AudioDeviceID GetAudioStreamDevice(SDL_AudioStream*) {
	return 1;
}

bool GetAudioDeviceFormat(SDL_AudioDeviceID, SDL_AudioSpec* spec, int* sample_frames) {
	*spec          = {SDL_AUDIO_F32, 2, 48000};
	*sample_frames = device_period != 0 ? static_cast<int>(device_period) : 1024;
	return true;
}

bool ClearAudioStream(SDL_AudioStream* stream) {
	auto& state   = GetStream(stream);
	state.frames  = 0;
	state.updated = now;
	clears++;
	return true;
}
} // namespace Fake

namespace Common::TimingThread {
using namespace Thread;
inline void SleepMicro(uint32_t micros) {
	sleeps.push_back(micros);
	now += micros + oversleep;
}
} // namespace Common::TimingThread

#define Thread                      TimingThread
#define SDL_InitSubSystem           Fake::InitSubSystem
#define SDL_QuitSubSystem           Fake::QuitSubSystem
#define SDL_OpenAudioDeviceStream   Fake::OpenAudioDeviceStream
#define SDL_ResumeAudioStreamDevice Fake::ResumeAudioStreamDevice
#define SDL_DestroyAudioStream      Fake::DestroyAudioStream
#define SDL_GetAudioStreamQueued    Fake::GetAudioStreamQueued
#define SDL_PutAudioStreamData      Fake::PutAudioStreamData
#define SDL_ClearAudioStream        Fake::ClearAudioStream
#define SDL_GetAudioStreamDevice    Fake::GetAudioStreamDevice
#define SDL_GetAudioDeviceFormat    Fake::GetAudioDeviceFormat
#include "libs/audio.cpp"
#undef Thread
#undef SDL_InitSubSystem
#undef SDL_QuitSubSystem
#undef SDL_OpenAudioDeviceStream
#undef SDL_ResumeAudioStreamDevice
#undef SDL_DestroyAudioStream
#undef SDL_GetAudioStreamQueued
#undef SDL_PutAudioStreamData
#undef SDL_ClearAudioStream
#undef SDL_GetAudioStreamDevice
#undef SDL_GetAudioDeviceFormat

namespace Libs::Controller {
int GetActiveControllerId() {
	return 0;
}
float GetSettingScale(Setting) {
	return 1.0f;
}

namespace DualSenseHaptics {
Stream* Open(uint32_t, bool) {
	static int pad_stream;
	return pad_connected ? reinterpret_cast<Stream*>(&pad_stream) : nullptr;
}
void     Close(Stream*) {}
bool UsesBluetooth(const Stream* stream) {
	return stream != nullptr && pad_bluetooth;
}
uint64_t Queue(Stream* stream, int, const void*, uint32_t, uint32_t, bool, const int*, float, const float* gains) {
	queued_pad_gains = gains;
	return stream != nullptr ? pad_queue_us : 0;
}
} // namespace DualSenseHaptics
} // namespace Libs::Controller

namespace Libs::LibKernel {
uint64_t KYTY_SYSV_ABI KernelGetProcessTime() {
	return now;
}
} // namespace Libs::LibKernel

namespace Loader::Timer {
double GetTimeMs() {
	return static_cast<double>(now) / 1000;
}
} // namespace Loader::Timer

namespace {
using Audio = Libs::Audio::Audio;

struct Fixture {
	Audio                   audio;
	std::array<float, 2048> pcm {};

	Fixture() {
		Check(streams.empty(), "stream leaked between tests");
		now        = 1000000;
		processing = oversleep = 0;
		clears                 = 0;
		fail_open = fail_put = stalled = false;
		pad_connected = pad_bluetooth = false;
		pad_queue_us = 0;
		queued_pad_gains = nullptr;
		device_speed                         = 1.0;
		device_period                        = 0;
		Libs::Audio::g_device_restart_logged = false;
		sleeps.clear();
	}

	Audio::Id Open(uint32_t frames = 256, int type = 0) {
		const auto id = audio.AudioOutOpen(type, frames, 48000, Audio::Format::FloatStereo);
		Check(id.IsValid(), "port open failed");
		return id;
	}

	void Output(Audio::Id port, bool blocking = true) {
		Audio::OutputParam param {port, pcm.data()};
		audio.AudioOutOutputs(&param, 1, blocking);
	}

	void Prime(Audio::Id port) {
		// The cushion: 40 ms of blocks, or the device's period and two blocks if that is more.
		const auto period = device_period != 0 ? device_period : 1024u;
		const int  blocks = std::max<int>(8, static_cast<int>((period + 255) / 256) + 2);
		const auto start  = now;
		for (int i = 0; i < blocks; i++) {
			Output(port);
		}
		Check(now == start, "priming was paced before the 40 ms cushion filled");
		Output(port);
		Check(now - start >= 5333 && now - start <= 5334,
		      "priming accumulated a multi-block deadline");
		Drain(*streams.front());
		// One block was due by then, or one whole pull of a device that takes them.
		const uint64_t due = device_period != 0 ? device_period : 256;
		Check(streams.front()->frames >= (uint64_t(blocks) * 256 - due) * 1000000,
		      "first paced output consumed the priming cushion");
	}
};

void TestPrimingAndSteadyCadence() {
	Fixture    f;
	const auto port = f.Open();
	f.Prime(port);
	processing = 125;
	oversleep  = 175;
	for (int i = 0; i < 10; i++) {
		f.Output(port);
	}
	const auto start = now;
	for (int i = 0; i < 3000; i++) {
		f.Output(port);
	}
	Check(now - start >= 15999999 && now - start <= 16000001,
	      "processing, sleep overshoot, or fractional periods accumulated timing drift");
}

void TestUnderrunAndStalledQueueRecovery() {
	Fixture    f;
	const auto port = f.Open();
	f.Prime(port);
	now += 1000000;
	f.Prime(port);

	// A stopped device's stale queue is cleared and starts over at its cushion, in silence; the
	// guest keeps its 48 kHz cadence throughout instead of waiting on the queue or priming ahead of
	// it.
	stalled                = true;
	const auto stall_start = now;
	for (int i = 0; i < 60 && clears == 0; i++) {
		f.Output(port);
	}
	Check(clears == 1, "stalled playback queue was not cleared");
	Check(now - stall_start >= 200000 && now - stall_start <= 211000,
	      "stalled device was not cleared after 200 ms, or the guest waited on it");
	const auto start = now;
	for (int i = 0; i < 375; i++) { // 2 s more of the stall, then 2 s of a healthy device
		f.Output(port);
	}
	stalled = false;
	for (int i = 0; i < 375; i++) {
		f.Output(port);
	}
	Check(now - start >= 3999999 && now - start <= 4000001,
	      "a stalled device changed the guest's cadence");
}

// Outputs `blocks` 256-frame blocks after `work` us of the guest's own work each; how long they
// took and the most the queue held.
struct DeviceRun {
	uint64_t elapsed = 0, queued_max = 0;
};
DeviceRun RunDevice(Fixture& f, Audio::Id port, int blocks, uint32_t work = 0) {
	DeviceRun  run;
	const auto start = now;
	for (int i = 0; i < blocks; i++) {
		now += work;
		f.Output(port);
		Drain(*streams.front());
		run.queued_max = std::max(run.queued_max, streams.front()->frames / 1000000);
	}
	run.elapsed = now - start;
	return run;
}

// A device that plays slower or faster than its nominal rate (BC-250 DisplayPort audio ran ~14%
// slow with a wrong audio clock), or takes audio in large pulls, must not set the guest's pace: the
// 48 kHz clock does. A slow one's backlog is dropped and a fast one's dry queue starts over in
// silence; a device at its rate is never restarted.
void TestDriftingDeviceDoesNotPaceGuest() {
	for (const double speed: {0.857, 0.95, 0.999, 1.0, 1.0001, 1.05}) {
		for (const uint32_t period: {0u, 1024u, 4096u}) {
			Fixture f;
			device_speed    = speed;
			device_period   = period;
			const auto port = f.Open();
			f.Prime(port);
			const auto run = RunDevice(f, port, 5625); // 30 s
			if (getenv("AUDIO_TEST_VERBOSE") != nullptr) {
				std::fprintf(stderr,
				             "speed %.4f period %u: 30 s took %llu us, clears %d, queue max %llu\n",
				             speed, period, static_cast<unsigned long long>(run.elapsed), clears,
				             static_cast<unsigned long long>(run.queued_max));
			}
			Check(run.elapsed >= 29999999 && run.elapsed <= 30000001,
			      "a drifting device changed the guest's 48 kHz cadence");
			// The cushion, the backlog margin (two pulls and 30 ms) and a pull in flight.
			const uint64_t pull = std::max(period, 1024u);
			const uint64_t most = std::max<uint64_t>(8 * 256, (pull + 255) / 256 * 256 + 512) +
			                      2 * pull + 1440 + pull + 256;
			Check(run.queued_max <= most, "a drifting device's queue grew past its backlog margin");
			if (std::abs(speed - 1.0) < 0.002) {
				Check(clears == 0, "a device near its rate had its queue restarted");
			} else {
				Check(clears > 0 && Libs::Audio::g_device_restart_logged,
				      "a drifting device's queue was never restarted, or not reported");
			}
		}
	}
}

// A device that skips pulls for 100 ms (an xrun) leaves a backlog: it is dropped once, not kept as
// latency, and the guest's cadence does not change.
void TestDeviceGapIsDropped() {
	Fixture f;
	device_period   = 1024;
	const auto port = f.Open();
	f.Prime(port);
	RunDevice(f, port, 1875);
	stalled = true;
	RunDevice(f, port, 19); // ~100 ms
	stalled        = false;
	const auto run = RunDevice(f, port, 3750);
	Check(run.elapsed >= 19999999 && run.elapsed <= 20000001,
	      "a device gap changed the guest's cadence");
	Check(clears == 1, "a device gap's backlog was not dropped, or more than once");
}

// A guest that produces audio late underruns and primes again, as before: its queue is not
// restarted and the device is not reported. Work within a block's time is not lateness: the clock
// sleeps less.
void TestLateGuestUnderruns() {
	Fixture f;
	device_period   = 1024;
	const auto port = f.Open();
	f.Prime(port);
	const auto run = RunDevice(f, port, 1875, 4000);
	Check(run.elapsed >= 9999999 && run.elapsed <= 10000001,
	      "work within a block's time changed the cadence");
	RunDevice(f, port, 3750, 5600); // longer than the 5333 us block: 95% of real time
	Check(clears == 0 && !Libs::Audio::g_device_restart_logged,
	      "a late guest's underruns were taken for a fast device");
}

void TestFallbackClockAndLongGap() {
	Fixture f;
	fail_open       = true;
	const auto port = f.Open();
	f.Output(port);
	const auto start = now;
	oversleep        = 175;
	for (int i = 0; i < 3000; i++) {
		now += 125;
		f.Output(port);
	}
	Check(now - start >= 16000174 && now - start <= 16000176,
	      "device-free output accumulated timing drift");
	now += 1000000;
	f.Output(port);
	const auto resumed = now;
	f.Output(port);
	Check(now - resumed >= 5333 && now - resumed <= 5509,
	      "long interruption caused a burst of device-free catch-up submissions");
}

void TestAsyncDoesNotAccumulateDeadlines() {
	Fixture    f;
	const auto port  = f.Open();
	const auto start = now;
	for (int i = 0; i < 8; i++) {
		f.Output(port, false);
	}
	Check(now == start, "asynchronous output slept");
	f.Output(port);
	Check(now - start <= 5334, "asynchronous submissions accumulated a blocking deadline");
	for (int i = 0; i < 3; i++) {
		f.Output(port);
	}
	now += 1000000;
	f.Output(port, false);
	const auto resumed = now;
	for (int i = 0; i < 7; i++) {
		f.Output(port);
	}
	Check(now == resumed, "async-to-blocking transition did not reset an emptied queue");
}

void TestSynchronizedBatchAndInactivePorts() {
	Fixture            f;
	const auto         vibration  = f.Open(256, 10);
	const auto         main       = f.Open();
	const auto         background = f.Open();
	Audio::OutputParam batch[] {
	    {vibration, f.pcm.data()}, {main, f.pcm.data()}, {background, f.pcm.data()}};
	const auto start = now;
	for (int i = 0; i < 8; i++) {
		f.audio.AudioOutOutputs(batch, 3);
	}
	Check(now == start, "vibration fallback slowed a device-backed batch's priming");
	for (int i = 0; i < 3; i++) {
		sleeps.clear();
		f.audio.AudioOutOutputs(batch, 3);
		Check(sleeps.size() == 1 && sleeps.front() >= 5333 && sleeps.front() <= 5334,
		      "synchronized streams accumulated separate pacing waits");
	}

	Audio::OutputParam inactive[] {{main, nullptr}, {vibration, f.pcm.data()}};
	now += 1000000;
	f.audio.AudioOutOutputs(inactive, 2);
	const auto resumed = now;
	f.audio.AudioOutOutputs(inactive, 2);
	Check(now - resumed >= 5333 && now - resumed <= 5334,
	      "a device without PCM disabled fallback pacing");
}

void TestFallbackUsesEachPortsPeriod() {
	Fixture f;
	fail_open                     = true;
	const auto         short_port = f.Open(128);
	const auto         long_port  = f.Open(256);
	Audio::OutputParam batch[] {{short_port, f.pcm.data()}, {long_port, f.pcm.data()}};
	f.audio.AudioOutOutputs(batch, 2);
	const auto start = now;
	f.audio.AudioOutOutputs(batch, 2);
	Check(now - start >= 5333 && now - start <= 5334,
	      "fallback batch used the first port's period for every port");
}

void TestFailedQueueUsesFallbackClock() {
	Fixture    f;
	const auto port = f.Open();
	fail_put        = true;
	f.Output(port);
	const auto start = now;
	for (int i = 0; i < 3; i++) {
		f.Output(port);
	}
	Check(now - start == 16000, "repeated queue failures reset fallback pacing");
}

void TestControllerSpeakerPacing() {
	Fixture f;
	fail_open     = true; // No PC audio device paces the pad speaker.
	pad_connected = true;
	pad_queue_us  = 80000;
	const auto speaker = f.Open(256, 4);

	pad_bluetooth = true;
	f.Output(speaker);
	const auto bluetooth_start = now;
	f.Output(speaker);
	Check(now - bluetooth_start >= 5333 && now - bluetooth_start <= 5334,
	      "Bluetooth speaker waited on its HID queue instead of the sample clock");

	pad_bluetooth = false;
	const auto usb_start = now;
	f.Output(speaker);
	Check(now - usb_start >= 5333 && now - usb_start <= 5334,
	      "stalled USB speaker waited longer than one audio block");
}

void TestInvalidBatchSize() {
	Libs::Audio::AudioOut::AudioOutOutputParam param {};
	for (const auto count: {0u, 33u}) {
		Check(Libs::Audio::AudioOut::AudioOutOutputs(&param, count) ==
		          Libs::Audio::AUDIO_OUT_ERROR_INVALID_SIZE,
		      "invalid batch size was not rejected before accessing ports");
	}
}

void TestZeroOutputFrequency() {
	Fixture f;
	Check(Libs::Audio::AudioOut::AudioOutOpen(0, 0, 0, 256, 0, 4) ==
	          Libs::Audio::AUDIO_OUT_ERROR_INVALID_SAMPLE_FREQ,
	      "public output open accepted zero frequency");
	Check(!f.audio.AudioOutOpen(10, 256, 0, Audio::Format::FloatStereo).IsValid(),
	      "internal output open accepted zero frequency");
	Check(f.Open().ToInt() == 1, "rejected output port occupied a handle");
}

template <typename T>
void CheckOutputSamples(std::initializer_list<T> expected, const char* message) {
	const auto& pcm = streams.back()->pcm;
	Check(pcm.size() == expected.size() * sizeof(T) &&
	          std::memcmp(pcm.data(), expected.begin(), pcm.size()) == 0, message);
}

void TestChannelGains() {
	Fixture f;
	const std::array pcm {0.25f, -0.5f, 0.75f, -1.0f};
	const auto stereo = f.audio.AudioOutOpen(0, 2, 48000, Audio::Format::FloatStereo);
	std::array gains {2.0f, 0.5f};
	Audio::OutputParam output {stereo, pcm.data(), gains.data()};
	for (int push = 0; push < 2; push++) {
		f.audio.AudioOutOutputs(&output, 1, false);
		CheckOutputSamples({0.5f, -0.25f, 1.5f, -0.5f}, "float channel gains were not applied independently");
	}
	const int volume[] {16384, 32768};
	f.audio.AudioOutSetVolume(stereo, 3, volume);
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples({0.25f, -0.25f, 0.75f, -0.5f}, "channel gain did not combine with port volume");
	gains = {0.0f, 0.0f};
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples({0.0f, -0.0f, 0.0f, -0.0f}, "zero channel gains did not mute");
	output.gains = nullptr;
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples({0.125f, -0.5f, 0.375f, -1.0f}, "channel scaling changed original float PCM");

	const std::array<int16_t, 2> integer_pcm {20000, -20000};
	const auto integer = f.audio.AudioOutOpen(0, 1, 48000, Audio::Format::Signed16bitStereo);
	output = {integer, integer_pcm.data(), gains.data()};
	for (const auto gain: {2.0f, std::numeric_limits<float>::max()}) {
		gains = {gain, gain};
		f.audio.AudioOutOutputs(&output, 1, false);
		CheckOutputSamples<int16_t>({32767, -32768}, "integer gain overflowed instead of saturating");
	}
	gains = {0.5f, 0.25f};
	f.audio.AudioOutOutputs(&output, 1, false);
	CheckOutputSamples<int16_t>({10000, -5000}, "integer channel gains changed original PCM");

	const std::array surround_pcm {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
	for (const auto format: {Audio::Format::Float8Ch, Audio::Format::Float12Ch}) {
		const auto surround = f.audio.AudioOutOpen(0, 1, 48000, format);
		output = {surround, surround_pcm.data(), surround_pcm.data()};
		f.audio.AudioOutOutputs(&output, 1, false);
		if (format == Audio::Format::Float8Ch) {
			CheckOutputSamples({1.0f, 4.0f, 9.0f, 16.0f, 49.0f, 64.0f, 25.0f, 36.0f},
			                   "channel reordering used destination channel gains");
		} else {
			CheckOutputSamples({82.0f, 104.0f, 9.0f, 16.0f, 170.0f, 208.0f, 25.0f, 36.0f},
			                   "height downmix lost per-channel gains");
		}
	}
	pad_connected = true;
	const auto speaker = f.Open(2, 4);
	gains = {0.75f, 0.25f};
	output = {speaker, pcm.data(), gains.data()};
	f.audio.AudioOutOutputs(&output, 1, false);
	Check(queued_pad_gains != nullptr && queued_pad_gains[0] == gains[0] && queued_pad_gains[1] == gains[1],
	      "connected controller did not receive channel gains");
}
} // namespace

int main() {
	TestPrimingAndSteadyCadence();
	TestUnderrunAndStalledQueueRecovery();
	TestDriftingDeviceDoesNotPaceGuest();
	TestDeviceGapIsDropped();
	TestLateGuestUnderruns();
	TestFallbackClockAndLongGap();
	TestAsyncDoesNotAccumulateDeadlines();
	TestSynchronizedBatchAndInactivePorts();
	TestFallbackUsesEachPortsPeriod();
	TestFailedQueueUsesFallbackClock();
	TestControllerSpeakerPacing();
	TestInvalidBatchSize();
	TestZeroOutputFrequency();
	TestChannelGains();
	Check(streams.empty(), "output stream leaked");
	std::puts("AudioOutTimingTests: all cases passed");
}
