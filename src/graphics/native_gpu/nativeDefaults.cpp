#include "graphics/native_gpu/nativeShadow.h"

#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::NativeGpu {

// Native mode is the default on the PS5's APU: the BC-250 (PCI 1002:13fe) and a PS5 running Linux
// (1002:13da). It sets KYTY_NATIVE_GPU_GEM and KYTY_NATIVE_GPU_PRIMARY unless they are set already;
// KYTY_NATIVE_GPU=0 keeps Kyty's Vulkan path. Runs first in main, before guest memory is set up.
void ApplyNativeDefaults() {
#if defined(__linux__)
	if (const char* v = std::getenv("KYTY_NATIVE_GPU"); v != nullptr && v[0] == '0') {
		return;
	}
	auto read_hex = [](int card, const char* name) {
		char path[96];
		std::snprintf(path, sizeof(path), "/sys/class/drm/card%d/device/%s", card, name);
		unsigned value = 0;
		if (std::FILE* f = std::fopen(path, "r"); f != nullptr) {
			if (std::fscanf(f, "%x", &value) != 1) {
				value = 0;
			}
			std::fclose(f);
		}
		return value;
	};
	for (int card = 0; card < 16; card++) {
		const unsigned device = read_hex(card, "device");
		if (read_hex(card, "vendor") != 0x1002u || (device != 0x13feu && device != 0x13dau)) {
			continue;
		}
		setenv("KYTY_NATIVE_GPU_GEM", "1", 0);
		setenv("KYTY_NATIVE_GPU_PRIMARY", "1", 0);
		std::fprintf(stderr, "native GPU: %s (1002:%04x), native mode on (KYTY_NATIVE_GPU=0 turns it off)\n",
		             device == 0x13feu ? "BC-250" : "PS5", device);
		return;
	}
#endif
}

} // namespace Libs::Graphics::NativeGpu
