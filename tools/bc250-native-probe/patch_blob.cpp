// Applies the native_gpu shader patcher to a raw shader binary (the .text of one shader).
// Build: c++ -std=c++20 -I../../src patch_blob.cpp ../../src/graphics/native_gpu/shaderPatcher.cpp

#include "graphics/native_gpu/shaderPatcher.h"

#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
	if (argc != 3) {
		std::fprintf(stderr, "usage: %s IN.bin OUT.bin\n", argv[0]);
		return 1;
	}
	std::FILE* in = std::fopen(argv[1], "rb");
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

	const auto r = Libs::Graphics::NativeGpu::PatchDot2c(code);
	std::printf("%zu dwords in, %zu out, %u patched, %u unsupported%s\n", code.size(), r.code.size(),
	            r.patched_sites, r.unsupported_sites, r.decode_error ? ", DECODE ERROR" : "");
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
