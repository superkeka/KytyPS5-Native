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
	const auto r = PatchDot2c(in, 16);
	Check(!r.decode_error, "clean shader reported a decode error");
	Check(r.patched_sites == 2 && r.unsupported_sites == 0, "expected two patched sites");
	Check(r.scratch_vgprs == 0, "plain non-aliasing sites should not need scratch");

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

// v_dot2c v3, 0x40003e00, v2 : src0 is a packed-f16 literal materialised into the scratch VGPR.
void TestPatchesLiteralSrc0() {
	const std::vector<uint32_t> in = {0x040604ffu, 0x40003e00u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 16);
	Check(!r.decode_error && r.patched_sites == 1 && r.unsupported_sites == 0, "literal src0 not patched");
	Check(r.scratch_vgprs == 1, "literal src0 must report one scratch VGPR");

	const uint32_t n = static_cast<uint32_t>(in.size());
	Check(r.code[0] == Branch(0, n), "literal site does not branch to its stub");
	Check(r.code[1] == 0x40003e00u && r.code[2] == SEndpgm, "literal site corrupted trailing words");
	// v_mov_b32 v16, 0x40003e00 ; fma lo ; fma hi ; s_branch back past both dwords.
	const std::vector<uint32_t> stub = {0x7e2002ffu, 0x40003e00u, 0xcc200003u, 0x1c0e0510u,
	                                    0xcc201803u, 0x1c0e0510u, Branch(n + 6, 2)};
	for (uint32_t i = 0; i < stub.size(); i++) {
		Check(r.code[n + i] == stub[i], "literal stub differs from the llvm-mc encoding");
	}
}

// v_dot2c_f32_f16_dpp v3, v1, v2 quad_perm:[1,0,3,2] : the DPP swizzle is reproduced by
// v_mov_b32_dpp into the scratch VGPR, copying the original DPP control word verbatim.
void TestPatchesDppSrc0() {
	const std::vector<uint32_t> in = {0x040604fau, 0xff00b101u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 16);
	Check(!r.decode_error && r.patched_sites == 1 && r.unsupported_sites == 0, "DPP src0 not patched");
	Check(r.scratch_vgprs == 1, "DPP src0 must report one scratch VGPR");

	const uint32_t n = static_cast<uint32_t>(in.size());
	Check(r.code[0] == Branch(0, n), "DPP site does not branch to its stub");
	// v_mov_b32_dpp v16, v1 <same dpp word> ; fma lo ; fma hi ; s_branch back.
	const std::vector<uint32_t> stub = {0x7e2002fau, 0xff00b101u, 0xcc200003u, 0x1c0e0510u,
	                                    0xcc201803u, 0x1c0e0510u, Branch(n + 6, 2)};
	for (uint32_t i = 0; i < stub.size(); i++) {
		Check(r.code[n + i] == stub[i], "DPP stub differs from the llvm-mc encoding");
	}

	// DPP8 uses the 0xe9 marker; otherwise identical.
	const std::vector<uint32_t> in8 = {0x040604e9u, 0xfac68801u, SEndpgm};
	const auto                  r8  = PatchDot2c(in8, 16);
	Check(r8.patched_sites == 1 && r8.scratch_vgprs == 1, "DPP8 src0 not patched");
	Check(r8.code[3] == 0x7e2002e9u && r8.code[4] == 0xfac68801u, "DPP8 stub marker/word wrong");
}

// Copying a DPP16 control word onto v_mov_b32_dpp is only equivalent for a subset of words; the
// rest must stay unsupported. Control words below are from llvm-mc (v_mov/v_add_f32 _dpp).
void TestDpp16ReductionGate() {
	// main word: v_dot2c_f32_f16_dpp v3, v1, v2 (src0 = 0xfa marker, vsrc1 = v2, vdst = v3).
	const uint32_t main = 0x040604fau;
	struct Reject {
		uint32_t    ctrl;
		const char* what;
	};
	const Reject rejects[] = {
	    {0x3f00b101u, "row_mask != 0xf"},          // quad_perm[1,0,3,2] row_mask:0x3
	    {0xf500b101u, "bank_mask != 0xf"},         // quad_perm bank_mask:0x5
	    {0xff10b101u, "src0_neg (bit 20)"},        // per-half neg can't be reproduced on v_mov
	    {0xff20b101u, "src0_abs (bit 21)"},
	    {0xff40b101u, "src1_neg (bit 22)"},        // the FMAs read vsrc1 without modifiers
	    {0xff80b101u, "src1_abs (bit 23)"},
	    {0xff010101u, "row_shl with bound_ctrl 0"}, // OOB lanes are disabled, not zeroed
	};
	for (const auto& rj: rejects) {
		const std::vector<uint32_t> in = {main, rj.ctrl, SEndpgm};
		const auto                  r  = PatchDot2c(in, 16);
		Check(r.patched_sites == 0 && r.unsupported_sites == 1, rj.what);
		Check(r.scratch_vgprs == 0 && r.code == in, "a non-reducible DPP16 site was modified");
	}

	// A shift/rotate with bound_ctrl 1 writes every lane (OOB reads 0), so it reduces.
	const std::vector<uint32_t> in = {main, 0xff092201u, SEndpgm}; // row_ror:2 bound_ctrl:1
	const auto                  r  = PatchDot2c(in, 16);
	Check(r.patched_sites == 1 && r.unsupported_sites == 0 && r.scratch_vgprs == 1, "bound_ctrl row_ror not patched");
	const uint32_t n = static_cast<uint32_t>(in.size());
	Check(r.code[n] == 0x7e2002fau && r.code[n + 1] == 0xff092201u, "bound_ctrl DPP16 stub copied the wrong word");
}

// v_dot2c v1, v1, v1 : vdst aliases both inputs, so a scratch copy of vdst feeds both FMA operands.
void TestPatchesAliasingSites() {
	const std::vector<uint32_t> in = {0x04020301u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 16);
	Check(!r.decode_error && r.patched_sites == 1 && r.unsupported_sites == 0, "aliasing site not patched");
	Check(r.scratch_vgprs == 1, "aliasing site must report one scratch VGPR");

	const uint32_t n = static_cast<uint32_t>(in.size());
	Check(r.code[0] == Branch(0, n), "aliasing site does not branch to its stub");
	// v_mov_b32 v16, v1 ; fma lo (v16,v16) ; fma hi ; s_branch back.
	const std::vector<uint32_t> stub = {0x7e200301u, 0xcc200001u, 0x1c062110u,
	                                    0xcc201801u, 0x1c062110u, Branch(n + 5, 1)};
	for (uint32_t i = 0; i < stub.size(); i++) {
		Check(r.code[n + i] == stub[i], "aliasing stub differs from the llvm-mc encoding");
	}
}

// The scratch register index is v[num_vgprs]; a larger count moves it and every stub operand.
void TestScratchIndexFollowsVgprCount() {
	const std::vector<uint32_t> in = {0x040604ffu, 0x40003e00u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 20);
	Check(r.patched_sites == 1 && r.scratch_vgprs == 1, "literal src0 not patched at num_vgprs=20");
	const uint32_t n = static_cast<uint32_t>(in.size());
	// v_mov_b32 v20, 0x40003e00 ; fma using v20.
	Check(r.code[n] == 0x7e2802ffu, "scratch v_mov did not move to v20");
	Check(r.code[n + 3] == 0x1c0e0514u && r.code[n + 5] == 0x1c0e0514u, "fma src0 did not move to v20");
}

void TestLeavesUnsupportedSitesAlone() {
	const std::vector<uint32_t> in = {
	    0x040604f9u, 0x06050602u, // SDWA src0: dst_sel/sub-dword semantics can't be reproduced
	    0x040404ffu, 0x40003e00u, // literal src0 with vsrc1 == vdst (v2): would need a second scratch
	    0x04000102u,              // v_dot2c v0, v2, v0: plain aliasing site, still patched
	    SEndpgm,
	};
	// Only the first two forms are unsupported; the third is an ordinary aliasing site (vsrc1 == vdst).
	const auto r = PatchDot2c(in, 16);
	Check(!r.decode_error, "unsupported forms reported a decode error");
	Check(r.unsupported_sites == 2, "expected two unsupported sites (SDWA, literal+alias)");
	Check(r.patched_sites == 1, "the plain aliasing site should still be patched");
	// The two unsupported sites and their words are left byte-for-byte unchanged.
	for (uint32_t i = 0; i < 4; i++) {
		Check(r.code[i] == in[i], "an unsupported site was modified");
	}
}

void TestUnsupportedWhenNoScratchRoom() {
	// num_vgprs == 256 leaves no VGPR index for scratch, so scratch-needing forms stay unsupported.
	const std::vector<uint32_t> in = {0x040604ffu, 0x40003e00u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 256);
	Check(r.patched_sites == 0 && r.unsupported_sites == 1, "literal must be unsupported with no scratch room");
	Check(r.scratch_vgprs == 0 && r.code == in, "no-scratch-room case changed the code");
}

void TestLiteralThatLooksLikeDot2c() {
	// v_madak_f32 v1, v2, v3, <literal equal to a v_dot2c encoding>: the literal must be skipped.
	const std::vector<uint32_t> in = {0x42020702u, 0x04060501u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 16);
	Check(r.patched_sites == 0 && r.unsupported_sites == 0 && r.code == in, "a literal was patched");
}

void TestDecodeErrorPatchesNothing() {
	const std::vector<uint32_t> in = {0x04060501u, 0xe4000000u, SEndpgm};
	const auto                  r  = PatchDot2c(in, 16);
	Check(r.decode_error && r.decode_error_dw == 1, "decode error not reported at the bad word");
	Check(r.patched_sites == 0 && r.scratch_vgprs == 0 && r.code == in, "code changed despite a decode error");
}

} // namespace

int main() {
	TestInstructionLengths();
	TestPatchesVgprAndSgprSites();
	TestPatchesLiteralSrc0();
	TestPatchesDppSrc0();
	TestDpp16ReductionGate();
	TestPatchesAliasingSites();
	TestScratchIndexFollowsVgprCount();
	TestLeavesUnsupportedSitesAlone();
	TestUnsupportedWhenNoScratchRoom();
	TestLiteralThatLooksLikeDot2c();
	TestDecodeErrorPatchesNothing();
	std::puts("NativeGpuShaderPatcherTests: all cases passed");
	return 0;
}
