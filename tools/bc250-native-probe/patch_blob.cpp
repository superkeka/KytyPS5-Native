// Applies the native_gpu shader patcher to a raw shader binary (the .text of one shader).
// Build: c++ -std=c++20 -I../../src patch_blob.cpp ../../src/graphics/native_gpu/shaderPatcher.cpp

#include "graphics/native_gpu/shaderPatcher.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
	if (argc != 4) {
		std::fprintf(stderr, "usage: %s IN.bin OUT.bin NUM_VGPRS\n", argv[0]);
		std::fprintf(stderr, "  NUM_VGPRS: VGPRs the shader allocates; the stubs use v[NUM_VGPRS] as scratch.\n");
		return 1;
	}
	const uint32_t num_vgprs = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
	std::FILE*     in        = std::fopen(argv[1], "rb");
	if (in == nullptr) {
		std::perror(argv[1]);
		return 1;
	}
	std::vector<uint32_t> code;
	uint32_t              w = 0;
	while (std::fread(&w, 4, 1, in) == 1) {
		code.push_back(w);
	}
	std::fclose(in);

	const auto r = Libs::Graphics::NativeGpu::PatchDot2c(code, num_vgprs);
	std::printf("%zu dwords in, %zu out, %u patched, %u unsupported, +%u scratch vgpr (dispatch with >= %u vgprs)%s\n",
	            code.size(), r.code.size(), r.patched_sites, r.unsupported_sites, r.scratch_vgprs,
	            num_vgprs + r.scratch_vgprs, r.decode_error ? ", DECODE ERROR" : "");
	if (r.decode_error) {
		std::printf("undecodable instruction at dword %u: 0x%08x\n", r.decode_error_dw, code[r.decode_error_dw]);
		return 2;
	}
	std::FILE* out = std::fopen(argv[2], "wb");
	if (out == nullptr) {
		std::perror(argv[2]);
		return 1;
	}
	std::fwrite(r.code.data(), 4, r.code.size(), out);
	std::fclose(out);
	return 0;
}
