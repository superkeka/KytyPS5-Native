#include "graphics/native_gpu/shaderPatcher.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Libs::Graphics::NativeGpu::InstructionDwords;
using Libs::Graphics::NativeGpu::PatchDot2c;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuShaderPatcherTests: failed: %s\n", message);
		std::abort();
	}
}

constexpr uint32_t SEndpgm  = 0xbf810000u;
constexpr uint32_t SCodeEnd = 0xbf9f0000u;

uint32_t Branch(uint32_t from_dw, uint32_t to_dw) {
	return 0xbf820000u | (static_cast<uint32_t>(static_cast<int32_t>(to_dw) - static_cast<int32_t>(from_dw) - 1) & 0xffffu);
}

// Encodings below come from llvm-mc -triple=amdgcn -mcpu=gfx1013 (gfx1030 for v_dot2c).
void TestInstructionLengths() {
	struct Case {
		std::vector<uint32_t> words;
		uint32_t              dwords;
		const char*           what;
	};
	const Case cases[] = {
	    {{SEndpgm}, 1, "s_endpgm"},
	    {{0xbf8c3f70u}, 1, "s_waitcnt"},
	    {{0x7e0202ffu, 0x40003e00u}, 2, "v_mov_b32 with literal"},
	    {{0x4a1402ffu, 0x00001000u}, 2, "VOP2 with literal"},
	    {{0x42020702u, 0x40000000u}, 2, "v_madak_f32"},
	    {{0x060206f9u, 0x06051602u}, 2, "VOP2 SDWA"},
	    {{0x040604fau, 0xff00e401u}, 2, "v_dot2c DPP"},
	    {{0x040604ffu, 0x40003e00u}, 2, "v_dot2c with literal"},
	    {{0x04060501u}, 1, "v_dot2c"},
	    {{0x8000ff00u, 0x00004000u}, 2, "SOP2 with literal"},
	    {{0xba80f801u, 0x00000001u}, 2, "s_setreg_imm32_b32"},
	    {{0xf4940100u, 0x00000000u}, 2, "SMEM s_memrealtime"},
	    {{0xd76d0001u, 0x040e04ffu, 0x00001234u}, 3, "VOP3 with literal"},
	    {{0xcc200003u, 0x1c0e0501u}, 2, "VOP3P v_fma_mix_f32"},
	    {{0xc8000001u}, 1, "VINTRP"},
	    {{0xd8820000u, 0x03000201u}, 2, "DS gds"},
	    {{0xdc708000u, 0x00000201u}, 2, "GLOBAL store"},
	    {{0xe0300000u, 0x80000100u}, 2, "MUBUF"},
	    {{0xe8b00000u, 0x80000100u}, 2, "MTBUF"},
	    {{0xf0800f0au, 0x00400004u, 0x00000006u}, 3, "MIMG with NSA"},
	    {{0xf800000fu, 0x03020100u}, 2, "EXP"},
	};
	for (const auto& c: cases) {
		std::vector<uint32_t> padded = c.words;
		padded.push_back(SEndpgm);
		Check(InstructionDwords(padded) == c.dwords, c.what);
	}
	Check(InstructionDwords(std::vector<uint32_t> {0xe4000000u}) == 0, "unknown encoding has no length");
	Check(InstructionDwords(std::vector<uint32_t> {0x7e0202ffu}) == 0, "truncated literal has no length");
}

void TestPatchesVgprAndSgprSites() {
	// v_mov v1, v2, v3 (literals); v_dot2c v3, v1, v2; v_dot2c v4, s1, v2; s_endpgm
	const std::vector<uint32_t> in = {
	    0x7e0202ffu, 0x40003e00u, 0x7e0402ffu, 0x44004200u, 0x7e0602ffu, 0x41200000u,
	    0x04060501u, 0x04080401u, SEndpgm,
	};
	const auto r = PatchDot2c(in);
	Check(!r.decode_error, "clean shader reported a decode error");
	Check(r.patched_sites == 2 && r.unsupported_sites == 0, "expected two patched sites");

	const uint32_t n = static_cast<uint32_t>(in.size());
	for (uint32_t i = 0; i < n; i++) {
		if (i != 6 && i != 7) {
			Check(r.code[i] == in[i], "non-site word changed");
		}
	}
	Check(r.code[6] == Branch(6, n), "first site does not branch to its stub");
	Check(r.code[7] == Branch(7, n + 5), "second site does not branch to its stub");

	const std::vector<uint32_t> stub0 = {0xcc200003u, 0x1c0e0501u, 0xcc201803u, 0x1c0e0501u, Branch(n + 4, 7)};
	const std::vector<uint32_t> stub1 = {0xcc200004u, 0x1c120401u, 0xcc201804u, 0x1c120401u, Branch(n + 9, 8)};
	for (uint32_t i = 0; i < 5; i++) {
		Check(r.code[n + i] == stub0[i], "VGPR stub differs from the llvm-mc encoding");
		Check(r.code[n + 5 + i] == stub1[i], "SGPR stub differs from the expected encoding");
	}
	Check((r.code.size() & 15u) == 0, "patched code is not padded to 64 bytes");
	for (size_t i = n + 10; i < r.code.size(); i++) {
		Check(r.code[i] == SCodeEnd, "padding is not s_code_end");
	}
}

void TestLeavesUnsupportedSitesAlone() {
	const std::vector<uint32_t> in = {
	    0x040604fau, 0xff00e401u, // DPP
	    0x040604ffu, 0x40003e00u, // literal src0
	    0x04020301u,              // v_dot2c v1, v1, v1: vdst aliases both inputs
	    SEndpgm,
	};
	const auto r = PatchDot2c(in);
	Check(!r.decode_error, "unsupported forms reported a decode error");
	Check(r.patched_sites == 0 && r.unsupported_sites == 3, "expected three unsupported sites");
	Check(r.code == in, "unsupported sites were modified");
}

void TestLiteralThatLooksLikeDot2c() {
	// v_madak_f32 v1, v2, v3, <literal equal to a v_dot2c encoding>: the literal must be skipped.
	const std::vector<uint32_t> in = {0x42020702u, 0x04060501u, SEndpgm};
	const auto                  r = PatchDot2c(in);
	Check(r.patched_sites == 0 && r.unsupported_sites == 0 && r.code == in, "a literal was patched");
}

void TestDecodeErrorPatchesNothing() {
	const std::vector<uint32_t> in = {0x04060501u, 0xe4000000u, SEndpgm};
	const auto                  r  = PatchDot2c(in);
	Check(r.decode_error && r.decode_error_dw == 1, "decode error not reported at the bad word");
	Check(r.patched_sites == 0 && r.code == in, "code changed despite a decode error");
}

} // namespace

int main() {
	TestInstructionLengths();
	TestPatchesVgprAndSgprSites();
	TestLeavesUnsupportedSitesAlone();
	TestLiteralThatLooksLikeDot2c();
	TestDecodeErrorPatchesNothing();
	std::puts("NativeGpuShaderPatcherTests: all cases passed");
	return 0;
}
