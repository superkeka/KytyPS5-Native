// The emulator side of the shader scanner (shaderScan.h): which size and hash a shader has, from shader.cpp's
// registry of AgcCreateShader, and one reporter for the process. Kept out of shaderScan.cpp so the unit tests
// need neither shader.cpp nor xxHash.

#include "graphics/native_gpu/shaderScan.h"

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/shader.h"

#include <cstdlib>
#include <mutex>
#include <xxhash.h>

namespace Libs::Graphics::NativeGpu {

namespace {

// KYTY_NATIVE_GPU_SHADER_SCAN: 0 off, 1 Warn and Error findings once per code hash (default), 2 also Info
// findings, a per-shader line and shader_hist.txt in KYTY_NATIVE_GPU_DUMP_DIR.
int ScanLevel() {
	static const int level = [] {
		const char* v = std::getenv("KYTY_NATIVE_GPU_SHADER_SCAN");
		return v == nullptr || v[0] == '\0' ? 1 : std::atoi(v);
	}();
	return level;
}

uint64_t Xxh3(const void* data, size_t bytes) {
	return XXH3_64bits(data, bytes);
}

struct ScanState {
	std::mutex         mutex;
	ShaderScanReporter reporter {ScanLevel(), Xxh3};
};

// Never destroyed: the exit summary may run while another thread is still translating.
ScanState& State() {
	static ScanState* state = [] {
		auto* s = new ScanState;
		std::atexit([] {
			ScanState&                  st = State();
			std::lock_guard<std::mutex> lock(st.mutex);
			if (st.reporter.Shaders() != 0) {
				st.reporter.PrintSummary("exit");
				if (ScanLevel() >= 2) {
					st.reporter.WriteHistogram(std::getenv("KYTY_NATIVE_GPU_DUMP_DIR"));
				}
			}
		});
		return s;
	}();
	return *state;
}

} // namespace

void ShaderScanDetect(uint64_t va, uint32_t swept_dwords, const std::function<bool(uint64_t addr)>& readable) {
	const int level = ScanLevel();
	if (level <= 0 || va == 0) {
		return;
	}
	ScanState&                  s = State();
	std::lock_guard<std::mutex> lock(s.mutex);

	// A shader AgcCreateShader registered has its exact size and the hash Kyty's recompiler dumps use; code the
	// title points PGM_LO/HI at without it (baked command buffers, library shaders) is walked in the same 64 Ki
	// dword window CountDot2c sweeps. Guest VA == host VA in Kyty, so the ISA is readable through the pointer.
	uint32_t                       size = 0;
	uint64_t                       hash = 0;
	Prospero::ShaderBinaryType     type {};
	ShaderScanReporter::Registered reg;
	const bool known = ShaderFindMapped(va, &size, &hash, &type) && size >= 4u && size <= (16u << 20u);
	if (known) {
		reg = {size, hash, static_cast<int>(type)};
	}
	const auto* code = reinterpret_cast<const uint32_t*>(va);
	const std::span<const uint32_t> window(code, known ? size / 4u : (1u << 16u));
	std::function<bool(uint32_t)>   dw_readable;
	if (readable) {
		dw_readable = [&](uint32_t dw) { return readable(va + static_cast<uint64_t>(dw) * 4u); };
	}
	const char* dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
	if (s.reporter.Detect(va, window, known ? &reg : nullptr, swept_dwords, dw_readable, dir) &&
	    s.reporter.Shaders() % 500u == 0) {
		s.reporter.PrintSummary("so far");
		if (level >= 2) {
			s.reporter.WriteHistogram(dir);
		}
	}
}

} // namespace Libs::Graphics::NativeGpu
