#include "graphics/native_gpu/shaderPatcher.h"
#include "graphics/native_gpu/shaderScan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <string>
#include <vector>

// Synthetic, hand-encoded gfx10 words only, no shader bytes from any title. They decode with llvm-mc -mcpu=gfx1013
// (gfx1011 for the dot ops), except where a case says the word is invalid on purpose (an undefined DPP control or
// SDWA select, src1_sel on VOP1, a register DPP marker in v_madak_f32's src0).

using namespace Libs::Graphics::NativeGpu;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuShaderScanTests: failed: %s\n", message);
		std::abort();
	}
}

constexpr uint32_t SEndpgm  = 0xbf810000u;
constexpr uint32_t SNop     = 0xbf800000u;
constexpr uint32_t SCodeEnd = 0xbf9f0000u;
constexpr uint32_t VMov     = 0x7e020302u; // v_mov_b32 v1, v2
constexpr uint32_t VDot2c   = 0x04060501u; // v_dot2c_f32_f16 v3, v1, v2 (gfx1011)
constexpr uint32_t SSetpc   = 0xbe802000u; // s_setpc_b64 s[0:1]
constexpr uint32_t SSwappc  = 0xbe822100u; // s_swappc_b64 s[2:3], s[0:1]
constexpr uint32_t SRfe     = 0xbe802200u; // s_rfe_b64 s[0:1]
constexpr uint32_t SGetpc   = 0xbe801f00u; // s_getpc_b64 s[0:1]
constexpr uint32_t SAddLit  = 0x8000ff00u; // s_add_u32 s0, s0, literal
constexpr uint32_t SMovLit  = 0xbe8003ffu; // s_mov_b32 s0, literal

// SOPP with a 16-bit branch offset from `from` to `to`.
uint32_t Sopp(uint32_t op, uint32_t from, uint32_t to) {
	return 0xbf800000u | (op << 16u) | (static_cast<uint32_t>(static_cast<int32_t>(to) - static_cast<int32_t>(from) - 1) & 0xffffu);
}
uint32_t Sopk(uint32_t op, uint32_t sdst, uint32_t simm) {
	return 0xb0000000u | (op << 23u) | (sdst << 16u) | (simm & 0xffffu);
}
uint32_t Getreg(uint32_t id) {
	return Sopk(0x12u, 1, 0xf800u | id); // s_getreg_b32 s1, hwreg(id, 0, 32)
}
uint32_t Setreg(uint32_t id) {
	return Sopk(0x13u, 0, 0xf800u | id); // s_setreg_b32 hwreg(id, 0, 32), s0
}

ShaderScanResult Scan(const std::vector<uint32_t>& code, uint32_t known = 0, bool ps5_apertures = true) {
	ShaderScanOptions opt;
	opt.known_dwords  = known;
	opt.ps5_apertures = ps5_apertures;
	return ScanShader(code, opt);
}

void TestTable() {
	Check(Gfx1013HasOpcode(ScanEnc::VOP3P, 0x20u), "v_fma_mix_f32 is gfx1013");
	Check(Gfx1013HasOpcode(ScanEnc::MIMG, 0xe6u) && Gfx1013HasOpcode(ScanEnc::MIMG, 0xe7u), "image_bvh*_intersect_ray are gfx1013");
	Check(Gfx1013HasOpcode(ScanEnc::MIMG, 0x80u), "image_msaa_load is gfx1013");
	Check(Gfx1013HasOpcode(ScanEnc::SMEM, 0x10u), "s_store_dword is gfx1013");
	Check(Gfx1013HasOpcode(ScanEnc::VOP2, 0x1fu), "v_mac_f32 is gfx1013");
	Check(!Gfx1013HasOpcode(ScanEnc::VOP2, 0x02u) && !Gfx1013HasOpcode(ScanEnc::VOP2, 0x0du), "no VOP2 dot on gfx1013");
	for (uint16_t op = 0x13u; op <= 0x19u; op++) {
		Check(!Gfx1013HasOpcode(ScanEnc::VOP3P, op), "no VOP3P dot on gfx1013");
	}
	Check(!Gfx1013HasOpcode(ScanEnc::GLOBAL, 0x16u) && !Gfx1013HasOpcode(ScanEnc::GLOBAL, 0x17u) &&
	          !Gfx1013HasOpcode(ScanEnc::GLOBAL, 0x34u) && !Gfx1013HasOpcode(ScanEnc::MUBUF, 0x34u),
	      "the gfx10.3 addtid/csub ops are not gfx1013");
	Check(!Gfx1013HasOpcode(ScanEnc::VOP2, 0x3eu) && !Gfx1013HasOpcode(ScanEnc::VOP2, 0x3fu), "0x3e/0x3f are not VOP2");
	Check(std::string(Gfx1013Mnemonic(ScanEnc::SOPP, 0x01u)) == "s_endpgm", "s_endpgm mnemonic");
	Check(Gfx1013Mnemonic(ScanEnc::VOP2, 0x02u) == nullptr, "no mnemonic for v_dot2c on gfx1013");
	Check(!Gfx1013HasOpcode(ScanEnc::Invalid, 0), "Invalid encoding has no opcodes");
}

void TestDecodeOpcode() {
	struct Case {
		std::vector<uint32_t> words;
		ScanEnc               enc;
		uint16_t              op;
	};
	const Case cases[] = {
	    {{VMov}, ScanEnc::VOP1, 0x01u},
	    {{VDot2c}, ScanEnc::VOP2, 0x02u},
	    {{0x7c040302u}, ScanEnc::VOPC, 0x02u},
	    {{0xd76d0001u, 0x040e0501u}, ScanEnc::VOP3, 0x36du},
	    {{0xcc200003u, 0x1c0e0501u}, ScanEnc::VOP3P, 0x20u},
	    {{SSetpc}, ScanEnc::SOP1, 0x20u},
	    {{SAddLit, 4}, ScanEnc::SOP2, 0x00u},
	    {{Getreg(1)}, ScanEnc::SOPK, 0x12u},
	    {{0xbf068000u}, ScanEnc::SOPC, 0x06u},
	    {{SEndpgm}, ScanEnc::SOPP, 0x01u},
	    {{0xf4940100u, 0}, ScanEnc::SMEM, 0x25u},
	    {{0xd8820000u, 0x03000201u}, ScanEnc::DS, 0x20u},
	    {{0xdc300000u, 0x017d0002u}, ScanEnc::FLAT, 0x0cu},
	    {{0xdc308000u, 0x017d0002u}, ScanEnc::GLOBAL, 0x0cu},
	    {{0xdc304000u, 0x017d0002u}, ScanEnc::SCRATCH, 0x0cu},
	    {{0xe0d00000u, 0x80000100u}, ScanEnc::MUBUF, 0x34u},
	    {{0xe2000000u, 0x80000100u}, ScanEnc::MUBUF, 0x80u},
	    {{0xe8b00000u, 0x80200100u}, ScanEnc::MTBUF, 0x08u},
	    {{0xf1989f01u, 0x00000400u}, ScanEnc::MIMG, 0xe6u},
	    {{0xc8000001u}, ScanEnc::VINTRP, 0x00u},
	    {{0xf800000fu, 0x03020100u}, ScanEnc::EXP, 0x00u},
	    {{0xdc30c000u, 0}, ScanEnc::Invalid, 0x0cu},
	};
	for (const auto& c: cases) {
		uint16_t op = 0xffffu;
		Check(ScanDecodeOpcode(c.words, &op) == c.enc && op == c.op, "encoding and opcode");
	}
}

void TestOneKindEach() {
	struct Case {
		std::vector<uint32_t> words;
		ScanKind              kind;
		ScanSeverity          sev;
		const char*           what;
	};
	const Case cases[] = {
	    {{VDot2c, SEndpgm}, ScanKind::Dot2c, ScanSeverity::Error, "v_dot2c_f32_f16"},
	    {{0x1a000501u, SEndpgm}, ScanKind::DotOther, ScanSeverity::Error, "v_dot4c_i32_i8"},
	    {{0xcc134000u, 0x1c0e0501u, SEndpgm}, ScanKind::DotOther, ScanSeverity::Error, "v_dot2_f32_f16"},
	    {{0xdc588000u, 0x007d0001u, SEndpgm}, ScanKind::UnknownOpcode, ScanSeverity::Error, "global_load_dword_addtid"},
	    {{0xe0d00000u, 0x80000100u, SEndpgm}, ScanKind::UnknownOpcode, ScanSeverity::Error, "buffer_atomic_csub"},
	    {{Getreg(23), SEndpgm}, ScanKind::Hwreg, ScanSeverity::Warn, "HW_ID1 read"},
	    {{0x060004fau, 0xff013001u, SEndpgm}, ScanKind::InvalidDpp, ScanSeverity::Warn, "wave_shl dpp"},
	    {{0x060004f9u, 0x06060701u, SEndpgm}, ScanKind::InvalidSdwa, ScanSeverity::Warn, "SDWA dst_sel 7"},
	    {{0xf1989f01u, 0x00000400u, SEndpgm}, ScanKind::Bvh, ScanSeverity::Warn, "image_bvh_intersect_ray"},
	    {{0xf0000f01u, 0x00000400u, SEndpgm}, ScanKind::MsaaLoad, ScanSeverity::Info, "image_msaa_load"},
	    {{0xbf900001u, SEndpgm}, ScanKind::SendMsg, ScanSeverity::Warn, "s_sendmsg INTERRUPT"},
	    {{0xbf920002u, SEndpgm}, ScanKind::Trap, ScanSeverity::Warn, "s_trap"},
	    {{0xbf8d0001u, SEndpgm}, ScanKind::Trap, ScanSeverity::Warn, "s_sethalt"},
	    {{0xd8820000u, 0x03000201u, SEndpgm}, ScanKind::Gds, ScanSeverity::Info, "ds with GDS"},
	    {{0xd8fe0004u, 0x01000002u, SEndpgm}, ScanKind::Gds, ScanSeverity::Info, "ds_ordered_count"},
	    {{0xf4940100u, 0, SEndpgm}, ScanKind::Timer, ScanSeverity::Info, "s_memrealtime"},
	    {{SSwappc, SEndpgm}, ScanKind::IndirectBranch, ScanSeverity::Info, "s_swappc_b64"},
	    {{SSetpc}, ScanKind::IndirectBranch, ScanSeverity::Info, "s_setpc_b64"},
	    {{SRfe}, ScanKind::IndirectBranch, ScanSeverity::Warn, "s_rfe_b64"},
	    {{SCodeEnd}, ScanKind::CodeEndReached, ScanSeverity::Warn, "s_code_end"},
	    {{0xe4000000u, SEndpgm}, ScanKind::DecodeError, ScanSeverity::Error, "undecodable word"},
	};
	for (const auto& c: cases) {
		const auto r = Scan(c.words);
		bool       ok = r.findings.size() == 1 && r.findings[0].kind == c.kind && r.findings[0].sev == c.sev && r.findings[0].count == 1;
		if (!ok) {
			std::fprintf(stderr, "case %s: %zu findings\n", c.what, r.findings.size());
			for (const auto& f: r.findings) {
				std::fprintf(stderr, "  %s %s\n", ScanKindName(f.kind), ScanSeverityName(f.sev));
			}
		}
		Check(ok, c.what);
	}
	// Clean code has no findings: render messages and wave-local hwregs included.
	const auto clean = Scan({VMov, 0xbf900003u, 0xbf900002u, Getreg(1), Getreg(5), Setreg(1), SEndpgm});
	Check(clean.findings.empty() && clean.ended && clean.instructions == 7, "clean code has no findings");
	Check(clean.Worst() == ScanSeverity::Info, "Worst of no findings is Info");
}

void TestHwregs() {
	const auto sev_of = [](uint32_t word, bool ps5 = true) {
		const auto r = Scan({word, SEndpgm}, 0, ps5);
		return r.findings.empty() ? -1 : static_cast<int>(r.findings[0].sev);
	};
	Check(sev_of(Getreg(1)) == -1 && sev_of(Getreg(5)) == -1 && sev_of(Getreg(2)) == -1 && sev_of(Getreg(7)) == -1,
	      "wave-local hwreg reads are not flagged");
	Check(sev_of(Getreg(23)) == 1 && sev_of(Getreg(24)) == 1 && sev_of(Getreg(16)) == 1, "HW_ID1, HW_ID2, TBA reads are Warn");
	Check(sev_of(Getreg(15), true) == 0 && sev_of(Getreg(15), false) == 1, "SH_MEM_BASES: Info with PS5 apertures, else Warn");
	Check(sev_of(Getreg(4)) == 2 && sev_of(Getreg(30)) == 2 && sev_of(Getreg(29)) == 2, "unnamed hwreg ids are Error");
	Check(sev_of(Setreg(1)) == -1 && sev_of(Setreg(3)) == -1, "MODE and TRAPSTS writes are not flagged");
	Check(sev_of(Setreg(5)) == 2 && sev_of(Setreg(15)) == 2, "other hwreg writes are Error");
	// s_setreg_imm32_b32 hwreg(HW_REG_MODE), literal and hwreg(id 5), literal.
	const auto imm_mode = Scan({0xba80f801u, 0x00000000u, SEndpgm});
	const auto imm_gpr  = Scan({0xba80f805u, 0x00000000u, SEndpgm});
	Check(imm_mode.findings.empty(), "s_setreg_imm32 of MODE is not flagged");
	Check(imm_gpr.findings.size() == 1 && imm_gpr.findings[0].sev == ScanSeverity::Error && imm_gpr.findings[0].detail == 5,
	      "s_setreg_imm32 of GPR_ALLOC is Error with the id as detail");
}

void TestDppSdwa() {
	const auto dpp = [](uint32_t ctrl) {
		return Scan({0x060004fau, 0xff000001u | (ctrl << 8u), SEndpgm}).Count(ScanKind::InvalidDpp);
	};
	Check(dpp(0x0e4u) == 0 && dpp(0x101u) == 0 && dpp(0x15fu) == 0 && dpp(0x141u) == 0 && dpp(0x16fu) == 0, "valid gfx10 DPP16 controls");
	Check(dpp(0x130u) == 1 && dpp(0x142u) == 1 && dpp(0x100u) == 1 && dpp(0x110u) == 1 && dpp(0x13cu) == 1, "invalid DPP16 controls");
	Check(Scan({0x060004e9u, 0x00fac688u, SEndpgm}).findings.empty(), "DPP8 is always valid");
	const auto sdwa = [](uint32_t first, uint32_t word1) { return Scan({first, word1, SEndpgm}).Count(ScanKind::InvalidSdwa); };
	Check(sdwa(0x060004f9u, 0x06060601u) == 0, "SDWA all DWORD selects");
	Check(sdwa(0x060004f9u, 0x06061e01u) == 1, "SDWA dst_unused 3");
	Check(sdwa(0x060004f9u, 0x07060601u) == 1, "SDWA src1_sel 7 on VOP2");
	Check(sdwa(0x7e0202f9u, 0x07060601u) == 0, "VOP1 has no src1_sel");
	Check(sdwa(0x7c0404f9u, 0x06060f01u) == 0, "VOPC has sdst in bits 15:8, not dst_sel/dst_unused");
	Check(sdwa(0x7c0404f9u, 0x06070001u) == 1, "VOPC SDWA src0_sel 7");
	// v_madak_f32: the literal is an operand; a word that looks like a DPP control is not checked.
	Check(Scan({0x420206fau, 0x00013000u, SEndpgm}).findings.empty(), "v_madak_f32 literal is not a DPP word");
}

void TestApertureLiteral() {
	// s_mov_b32 s0, 0x80000000; flat_load_dword v1, v[2:3]; s_endpgm.
	const std::vector<uint32_t> flat = {SMovLit, 0x80000000u, 0xdc300000u, 0x017d0002u, SEndpgm};
	Check(Scan(flat, 0, true).Count(ScanKind::ApertureLiteral, ScanSeverity::Info) == 1, "aperture literal with FLAT, PS5 apertures: Info");
	Check(Scan(flat, 0, false).Count(ScanKind::ApertureLiteral, ScanSeverity::Warn) == 1, "aperture literal with FLAT, Linux apertures: Warn");
	// The same literal without FLAT accesses (global_load_dword) is not an aperture.
	const std::vector<uint32_t> global = {SMovLit, 0x70000000u, 0xdc308000u, 0x017d0002u, SEndpgm};
	Check(Scan(global, 0, false).findings.empty(), "aperture literal without FLAT ops is not flagged");
}

void TestReachability() {
	// Code after s_endpgm reached by s_cbranch_execz is scanned and counted as past the first end.
	{
		const std::vector<uint32_t> code = {Sopp(0x08u, 0, 3), VMov, SEndpgm, VDot2c, SEndpgm};
		const auto                  r    = Scan(code);
		Check(r.Count(ScanKind::Dot2c) == 1 && r.Count(ScanKind::PastFirstEnd) == 2 && r.first_end_dw == 2, "branch past s_endpgm");
		Check(r.IsStart(3) && r.IsStart(4) && r.instructions == 5 && r.image_dwords == 5, "out-of-line block decoded");
	}
	// Unreached data after s_endpgm that looks like v_dot2c is not code; CountDot2c agrees.
	{
		const std::vector<uint32_t> code = {VMov, SEndpgm, VDot2c, 0x12345678u};
		const auto                  r    = Scan(code);
		Check(r.findings.empty() && r.image_dwords == 2 && !r.IsStart(2), "data after s_endpgm is not scanned");
		Check(CountDot2c(code).sites == 0, "CountDot2c also stops at s_endpgm");
	}
	// s_branch jumps over data.
	{
		const std::vector<uint32_t> code = {Sopp(0x02u, 0, 2), VDot2c, SEndpgm};
		const auto                  r    = Scan(code);
		Check(r.findings.empty() && !r.IsStart(1) && r.instructions == 2, "s_branch skips the word it jumps over");
	}
	// s_setpc_b64 ends the path.
	{
		const auto r = Scan({SSetpc, VDot2c});
		Check(r.Count(ScanKind::Dot2c) == 0 && r.legacy.ended && !r.ended, "s_setpc_b64 stops the path");
	}
	// s_call_b64 and s_subvector_loop_* targets are followed.
	{
		const std::vector<uint32_t> code = {Sopk(0x16u, 0, 2), SEndpgm, SNop, VDot2c, SSetpc};
		Check(Scan(code).Count(ScanKind::Dot2c) == 1, "s_call_b64 target followed");
		const std::vector<uint32_t> loop = {Sopk(0x1bu, 0, 3), VMov, Sopk(0x1cu, 0, static_cast<uint32_t>(-2)), SEndpgm, VDot2c, SEndpgm};
		const auto                  r    = Scan(loop);
		Check(r.Count(ScanKind::Dot2c) == 1 && r.findings.size() == 2, "s_subvector_loop_begin target followed");
	}
	// A backward loop terminates.
	{
		const auto r = Scan({VMov, Sopp(0x04u, 1, 0), SEndpgm});
		Check(r.findings.empty() && r.instructions == 3 && r.ended, "backward loop");
	}
	// A branch into the middle of a literal.
	{
		const auto r = Scan({SMovLit, 0x0u, Sopp(0x04u, 2, 1), SEndpgm});
		Check(r.Count(ScanKind::OverlapDecode) == 1, "branch into a literal");
	}
	// An earlier start decoded inside a later, longer instruction: the reverse overlap.
	{
		// [0] s_cbranch_scc0 -> 2; [1] s_mov_b32 s0, literal: the branch target is that literal.
		const auto r = Scan({Sopp(0x04u, 0, 2), SMovLit, VMov, SEndpgm});
		Check(r.Count(ScanKind::OverlapDecode) == 1, "a decoded start inside a later instruction");
	}
	// Out-of-range targets are never read: the span ends at the limit (run under ASan).
	{
		const std::vector<uint32_t> code = {Sopp(0x02u, 0, 5)};
		const auto                  r    = Scan(code);
		Check(r.Count(ScanKind::BranchOutOfRange) == 1 && r.instructions == 1, "forward target past the limit");
		const std::vector<uint32_t> back = {VMov, Sopp(0x04u, 1, static_cast<uint32_t>(-3)), SEndpgm};
		Check(Scan(back).Count(ScanKind::BranchOutOfRange) == 1, "target before dword 0");
		// A known size bounds the walk below the span's end.
		const std::vector<uint32_t> big = {Sopp(0x02u, 0, 3), SEndpgm, SEndpgm, VDot2c, SEndpgm};
		Check(Scan(big, 2).Count(ScanKind::BranchOutOfRange) == 1 && Scan(big, 2).Count(ScanKind::Dot2c) == 0, "known size bounds targets");
	}
	// Running into the limit is a truncated window, also mid-instruction; not a decode error.
	{
		const auto r = Scan({VMov, VMov});
		Check(r.truncated && !r.decode_error && r.findings.empty(), "no s_endpgm in the window");
		const auto v = Scan({VMov, 0xd76d0001u});
		Check(v.truncated && !v.decode_error && v.instructions == 1, "VOP3 cut by the limit");
	}
}

void TestReadablePages() {
	alignas(4096) static uint32_t mem[2048];
	for (uint32_t& w: mem) {
		w = SEndpgm;
	}
	mem[0]    = Sopp(0x02u, 0, 1500);
	mem[1500] = VDot2c;
	std::vector<uint32_t> asked;
	ShaderScanOptions     opt;
	opt.readable = [&](uint32_t dw) {
		asked.push_back(dw);
		return dw < 1024;
	};
	auto r = ScanShader(std::span<const uint32_t>(mem, 2048), opt);
	Check(r.Count(ScanKind::BranchOutOfRange) == 1 && r.Count(ScanKind::Dot2c) == 0, "target on an unreadable page is not read");
	Check(asked.size() == 2 && asked[0] == 0 && asked[1] == 1500, "readable asked once per page, at the first word read");
	asked.clear();
	opt.safe_dwords = 2048;
	r               = ScanShader(std::span<const uint32_t>(mem, 2048), opt);
	Check(asked.empty() && r.Count(ScanKind::Dot2c) == 1, "words below safe_dwords need no question");
	asked.clear();
	opt.safe_dwords  = 0;
	opt.known_dwords = 2048;
	r                = ScanShader(std::span<const uint32_t>(mem, 2048), opt);
	Check(asked.empty() && r.Count(ScanKind::Dot2c) == 1, "a registered size needs no question");
}

void TestPcRel() {
	// s_getpc_b64 s[0:1]; s_add_u32 s0, s0, lit; s_endpgm: the address is (0 + 1) * 4 + lit.
	const std::vector<uint32_t> inside  = {SGetpc, SAddLit, 8, SEndpgm};
	const std::vector<uint32_t> outside = {SGetpc, SAddLit, 0x100, SEndpgm};
	auto                        r       = Scan(inside, 4);
	Check(r.findings.empty() && r.pcrel_min == 12 && r.pcrel_max == 12, "pc-relative address inside the image");
	r = Scan(outside, 4);
	Check(r.Count(ScanKind::PcRelOutside) == 1 && r.pcrel_max == 0x104, "pc-relative address outside the image");
	r = Scan(outside);
	Check(r.findings.empty() && r.pcrel_max == 0x104, "without a registered size only the offsets are kept");
	// Too far from the s_getpc_b64.
	r = Scan({SGetpc, VMov, VMov, VMov, VMov, SAddLit, 0x100, SEndpgm}, 8);
	Check(r.findings.empty() && r.pcrel_max == INT64_MIN, "s_add_u32 five instructions later is not tracked");
}

void CheckLegacyParity(const std::vector<uint32_t>& code, const char* what) {
	const Dot2cScan a = CountDot2c(code);
	const Dot2cScan b = Scan(code).legacy;
	const bool ok = a.sites == b.sites && a.dot_ops == b.dot_ops && a.instructions == b.instructions && a.gds_ops == b.gds_ops &&
	                a.ordered_ops == b.ordered_ops && a.gws_ops == b.gws_ops && a.flat_ops == b.flat_ops &&
	                a.aperture_ops == b.aperture_ops && a.dwords == b.dwords && a.ended == b.ended &&
	                a.decode_error == b.decode_error && a.getreg == b.getreg && a.setreg == b.setreg &&
	                a.hwid_simm == b.hwid_simm && a.hwid_simm_n == b.hwid_simm_n && a.dot_by_op == b.dot_by_op;
	if (!ok) {
		std::fprintf(stderr, "parity %s: instr %u/%u dwords %u/%u ap %u/%u\n", what, a.instructions, b.instructions, a.dwords,
		             b.dwords, a.aperture_ops, b.aperture_ops);
	}
	Check(ok, what);
}

void TestLegacyParity() {
	// The CountDot2c vectors of NativeGpuShaderPatcherTests.cpp, plus GDS, ordered append and GWS.
	CheckLegacyParity({0xdc300000u, 0x017d0002u, 0xdc308000u, 0x017d0002u, 0xbe8003ebu, 0xb901f80fu, SEndpgm}, "flat and apertures");
	CheckLegacyParity({0x04000501u, 0xcc134000u, 0x1c0e0501u, 0xcc164000u, 0x1c0e0501u, 0xcc184000u, 0x1c0e0501u, SEndpgm},
	                  "dot ops");
	CheckLegacyParity({0xb901f817u, 0xb901f818u, 0xb901f817u, 0xb9011817u, 0xb980f801u, 0xba80f801u, 0x00000000u, 0x1a000501u, SEndpgm},
	                  "hwregs");
	CheckLegacyParity({0xd8820000u, 0x03000201u, 0xd8fe0004u, 0x01000002u, 0xd8660000u, 0x00000001u, SEndpgm}, "gds, ordered, gws");
	CheckLegacyParity({VMov, 0xd76d0001u, 0x040e04ffu, 0x00001234u, SSetpc, VDot2c}, "s_setpc return");
	CheckLegacyParity({VDot2c, 0xe4000000u, SEndpgm}, "decode error");
	CheckLegacyParity({0x7e0002ecu, 0xd5030000u, 0x000204edu, SEndpgm}, "aperture sources in VOP1 and VOP3");
}

uint64_t Fnv(const void* data, size_t bytes) {
	uint64_t h = 1469598103934665603ull;
	for (size_t i = 0; i < bytes; i++) {
		h = (h ^ static_cast<const uint8_t*>(data)[i]) * 1099511628211ull;
	}
	return h;
}

std::string ReadAll(std::FILE* f) {
	std::fflush(f);
	std::rewind(f);
	std::string s;
	char        buf[512];
	size_t      n = 0;
	while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
		s.append(buf, n);
	}
	return s;
}

void TestReporter() {
	std::FILE* out = std::tmpfile();
	Check(out != nullptr, "tmpfile");
	ShaderScanReporter rep(1, Fnv, out);
	const std::vector<uint32_t> clean = {VMov, SEndpgm};
	const std::vector<uint32_t> dot   = {VDot2c, SEndpgm, 0xdeadbeefu};
	const std::vector<uint32_t> info  = {0xf4940100u, 0, SEndpgm};
	// The same code at two addresses is reported once; changed code at an address is new.
	Check(rep.Detect(0x1000, clean, nullptr, 0, {}, nullptr), "first sighting scanned");
	Check(!rep.Detect(0x2000, clean, nullptr, 0, {}, nullptr), "same code at another address is a duplicate");
	Check(rep.Detect(0x1000, dot, nullptr, 0, {}, nullptr), "changed code at the same address is new");
	// The unregistered hash covers the reachable image only: trailing data does not make a new shader.
	std::vector<uint32_t> dot2 = dot;
	dot2[2]                    = 0x12345678u;
	Check(!rep.Detect(0x3000, dot2, nullptr, 0, {}, nullptr), "data past the image does not change the hash");
	// Registered shaders go by shader.cpp's hash, before anything is read.
	ShaderScanReporter::Registered reg {12, 0xabcdu, 1};
	Check(rep.Detect(0x4000, info, &reg, 0, {}, nullptr), "registered shader scanned");
	Check(!rep.Detect(0x5000, dot, &reg, 0, {}, nullptr), "same registered hash elsewhere is a duplicate");
	Check(rep.Shaders() == 3 && rep.Duplicates() == 3 && rep.Flagged() == 1, "shader, duplicate and flagged counts");
	Check(rep.KindInstructions(ScanKind::Dot2c) == 1 && rep.KindShaders(ScanKind::Timer) == 1, "per-kind totals");
	std::string text = ReadAll(out);
	Check(text.find("error dot2c - ") == std::string::npos && text.find("error dot2c ") != std::string::npos, "dot2c line");
	Check(text.find("timer") == std::string::npos, "Info findings are not logged at level 1");
	Check(rep.Lines() == 1, "one line for the one flagged shader");
	rep.PrintSummary("test");
	text = ReadAll(out);
	Check(text.find("3 shaders (1 registered), 3 duplicates, 1 flagged") != std::string::npos, "summary counts");
	Check(text.find("dot2c:1/1") != std::string::npos && text.find("timer:1/1") != std::string::npos, "summary kinds");
	std::fclose(out);

	// Level 2 logs Info, keeps the histogram; flagged shaders are dumped as exactly their image.
	out = std::tmpfile();
	ShaderScanReporter verbose(2, Fnv, out);
	const auto dir = std::filesystem::temp_directory_path() / ("kyty_shader_scan_test_" + std::to_string(Fnv(&out, sizeof(out))));
	std::filesystem::create_directories(dir);
	Check(verbose.Detect(0x1000, info, nullptr, 0, {}, dir.string().c_str()), "verbose scan");
	Check(verbose.Detect(0x2000, dot, nullptr, 0, {}, dir.string().c_str()), "verbose flagged scan");
	text = ReadAll(out);
	Check(text.find("info timer s_memrealtime") != std::string::npos, "Info findings are logged at level 2");
	char name[64];
	std::snprintf(name, sizeof(name), "shader_%016llx.bin", static_cast<unsigned long long>(Fnv(dot.data(), 8)));
	Check(std::filesystem::exists(dir / name) && std::filesystem::file_size(dir / name) == 8, "flagged image dumped, 2 dwords");
	Check(std::filesystem::directory_iterator(dir) != std::filesystem::directory_iterator() &&
	          std::distance(std::filesystem::directory_iterator(dir), std::filesystem::directory_iterator()) == 1,
	      "only the flagged shader is dumped");
	verbose.WriteHistogram(dir.string().c_str());
	std::FILE* hist = std::fopen((dir / "shader_hist.txt").string().c_str(), "r");
	Check(hist != nullptr, "shader_hist.txt written");
	text = ReadAll(hist);
	std::fclose(hist);
	Check(text.find("SOPP 0x001 s_endpgm 2 2") != std::string::npos, "histogram counts s_endpgm in both shaders");
	Check(text.find("VOP2 0x002 (not-gfx1013) 1 1") != std::string::npos, "histogram marks opcodes outside gfx1013");
	std::filesystem::remove_all(dir);
	std::fclose(out);
}

} // namespace

int main() {
	TestTable();
	TestDecodeOpcode();
	TestOneKindEach();
	TestHwregs();
	TestDppSdwa();
	TestApertureLiteral();
	TestReachability();
	TestReadablePages();
	TestPcRel();
	TestLegacyParity();
	TestReporter();
	std::puts("NativeGpuShaderScanTests: all cases passed");
	return 0;
}
