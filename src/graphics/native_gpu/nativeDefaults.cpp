#include "graphics/native_gpu/nativeShadow.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#if defined(__linux__)
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace Libs::Graphics::NativeGpu {

// Native mode is the default on the PS5's APU: the BC-250 (PCI 1002:13fe) and a PS5 running Linux
// (1002:13da). It sets KYTY_NATIVE_GPU_GEM and KYTY_NATIVE_GPU_PRIMARY unless they are set already;
// KYTY_NATIVE_GPU=0 keeps Kyty's Vulkan path. Runs first in main, before guest memory is set up.
namespace {

#if defined(__linux__)
std::string ReadLine(const std::string& path) {
	std::string line;
	if (std::FILE* f = std::fopen(path.c_str(), "r"); f != nullptr) {
		char buf[128];
		if (std::fgets(buf, sizeof(buf), f) != nullptr) {
			line = buf;
			while (!line.empty() && (line.back() == '\n' || line.back() == ' ')) {
				line.pop_back();
			}
		}
		std::fclose(f);
	}
	return line;
}

// One line on what the system gives the native backend, so a log shows the mode a game ran in: the
// kernel, CPUs, the BIOS VRAM split, GTT and the amdgpu queue settings (README, Requirements).
void PrintSystem(const std::string& device_dir) {
	auto mib = [&](const char* name) { return std::strtoull(ReadLine(device_dir + name).c_str(), nullptr, 10) >> 20u; };
	auto param = [](const char* name) {
		std::string v = ReadLine(std::string("/sys/module/amdgpu/parameters/") + name);
		return v.empty() ? std::string("?") : v;
	};
	utsname uts {};
	uname(&uts);
	const bool        patched = !ReadLine("/sys/module/amdgpu/parameters/bc250_gfx_queues").empty();
	const auto        gtt     = mib("mem_info_gtt_total");
	const std::string sched   = param("sched_policy");
	const std::string vm      = param("vm_update_mode");
	const std::string cwsr    = param("cwsr_enable");
	const bool recommended = gtt >= 14336 && sched == "2" && (patched || (vm == "0" && cwsr == "0"));
	std::fprintf(stderr,
	             "system: kernel %s, %ld CPUs (max %llu MHz), VRAM %llu MiB, GTT %llu MiB, amdgpu sched_policy=%s "
	             "vm_update_mode=%s cwsr_enable=%s (%s amdgpu), kernel config %s\n",
	             uts.release, sysconf(_SC_NPROCESSORS_ONLN),
	             std::strtoull(ReadLine("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").c_str(), nullptr, 10) / 1000,
	             mib("mem_info_vram_total"), gtt, sched.c_str(), vm.c_str(), cwsr.c_str(), patched ? "patched" : "stock",
	             recommended ? "recommended" : "NOT the recommended one");
}
#endif

} // namespace

// Native mode is the default on the PS5's APU: the BC-250 (PCI 1002:13fe) and a PS5 running Linux
// (1002:13da). It sets KYTY_NATIVE_GPU_GEM and KYTY_NATIVE_GPU_PRIMARY unless they are set already;
// KYTY_NATIVE_GPU=0 keeps Kyty's Vulkan path. Runs first in main, before guest memory is set up.
void ApplyNativeDefaults() {
#if defined(__linux__)
	for (int card = 0; card < 16; card++) {
		const std::string dir    = "/sys/class/drm/card" + std::to_string(card) + "/device/";
		const unsigned    device = std::strtoul(ReadLine(dir + "device").c_str(), nullptr, 16);
		if (std::strtoul(ReadLine(dir + "vendor").c_str(), nullptr, 16) != 0x1002u || (device != 0x13feu && device != 0x13dau)) {
			continue;
		}
		PrintSystem(dir);
		const char* name = device == 0x13feu ? "BC-250" : "PS5";
		if (const char* v = std::getenv("KYTY_NATIVE_GPU"); v != nullptr && v[0] == '0') {
			std::fprintf(stderr, "native GPU: %s (1002:%04x), native mode off (KYTY_NATIVE_GPU=0)\n", name, device);
			return;
		}
		setenv("KYTY_NATIVE_GPU_GEM", "1", 0);
		setenv("KYTY_NATIVE_GPU_PRIMARY", "1", 0);
		std::fprintf(stderr, "native GPU: %s (1002:%04x), native mode on (KYTY_NATIVE_GPU=0 turns it off)\n", name, device);
		return;
	}
#endif
}

} // namespace Libs::Graphics::NativeGpu
