#include "graphics/native_gpu/shaderPatcher.h"

namespace Libs::Graphics::NativeGpu {

namespace {

constexpr uint32_t SrcLiteral = 0xffu;
constexpr uint32_t SrcDpp16   = 0xfau;
constexpr uint32_t SrcDpp8    = 0xe9u;
constexpr uint32_t SrcDpp8Fi  = 0xeau;
constexpr uint32_t SrcSdwa    = 0xf9u;
constexpr uint32_t SrcVgpr0   = 0x100u;

constexpr uint32_t OpVop2Dot2cF32F16 = 0x02u;

constexpr uint32_t SBranch  = 0xbf820000u;
constexpr uint32_t SCodeEnd = 0xbf9f0000u;

// v_fma_mix_f32 vdst, src0, src1, vdst: VOP3P opcode 0x20, src2 = vdst as f32.
constexpr uint32_t Vop3pFmaMixF32 = 0xcc200000u;
constexpr uint32_t OpSelLoHalves  = 0u;
constexpr uint32_t OpSelHiHalves  = 3u << 11; // op_sel[0..1]: high f16 of src0 and src1
constexpr uint32_t OpSelHiMixF16  = 3u << 27; // op_sel_hi[0..1]: src0 and src1 are f16

// VOP1 v_mov_b32 with vdst and src0 cleared: used to materialise a stub's scratch register. A
// literal or DPP16/DPP8 src0 carries into the appended word verbatim (same src0 marker and, for
// DPP, the same control word as the original v_dot2c), which is how the scratch form is built.
constexpr uint32_t Vop1MovB32 = 0x7e000200u;

uint32_t VopExtraDwords(uint32_t src0) {
	switch (src0) {
		case SrcLiteral:
		case SrcDpp16:
		case SrcDpp8:
		case SrcDpp8Fi:
		case SrcSdwa: return 1;
		default: return 0;
	}
}

bool IsVop2AlwaysLiteral(uint32_t op) {
	// v_madmk_f32, v_madak_f32, v_fmamk_f32, v_fmaak_f32, v_fmamk_f16, v_fmaak_f16
	return op == 0x20u || op == 0x21u || op == 0x2cu || op == 0x2du || op == 0x37u || op == 0x38u;
}

// True when a DPP16 control word can be reproduced by copying it onto v_mov_b32_dpp. The mov
// swizzles src0 into scratch and the two FMAs then write vdst in every active lane, so the form is
// only equivalent when the original also writes every active lane with an unmodified src0:
//   - row_mask and bank_mask (bits 31:28 / 27:24) must be 0xf, else masked-out lanes keep vdst in
//     the original but would be overwritten from stale scratch here;
//   - src0/src1 neg and abs (bits 23:20) must be clear: on v_mov_b32 neg/abs act on the whole f32,
//     not per f16 half as v_dot2c needs, and src1's modifiers are dropped by the FMAs entirely;
//   - the lane select must be a quad_perm (dpp_ctrl <= 0xff, always in-range) or bound_ctrl (bit
//     19) must be set, so a shift/rotate/broadcast that reads an invalid lane substitutes 0 (and
//     writes) instead of disabling the lane (and leaving vdst) as bound_ctrl == 0 does.
bool Dpp16Reducible(uint32_t ctrl) {
	const bool full_masks = (ctrl >> 24) == 0xffu;
	const bool no_mods    = (ctrl & 0x00f00000u) == 0u;
	const bool quad_perm  = ((ctrl >> 8) & 0x1ffu) <= 0xffu;
	const bool bound_ctrl = ((ctrl >> 19) & 1u) != 0u;
	return full_masks && no_mods && (quad_perm || bound_ctrl);
}

uint32_t Vop3LiteralDwords(uint32_t word1) {
	bool literal = (word1 & 0x1ffu) == SrcLiteral || ((word1 >> 9) & 0x1ffu) == SrcLiteral ||
	               ((word1 >> 18) & 0x1ffu) == SrcLiteral;
	return literal ? 1 : 0;
}

uint32_t Branch(uint32_t from_dw, uint32_t to_dw, bool* ok) {
	int64_t offset = static_cast<int64_t>(to_dw) - static_cast<int64_t>(from_dw) - 1;
	if (offset < INT16_MIN || offset > INT16_MAX) {
		*ok = false;
	}
	return SBranch | (static_cast<uint32_t>(offset) & 0xffffu);
}

} // namespace

uint32_t InstructionDwords(std::span<const uint32_t> code) {
	if (code.empty()) {
		return 0;
	}
	const uint32_t w = code[0];
	uint32_t       n = 0;

	if ((w >> 31) == 0u) {
		const uint32_t enc7 = w >> 25;
		const uint32_t src0 = w & 0x1ffu;
		if (enc7 == 0x3fu || enc7 == 0x3eu) { // VOP1, VOPC
			n = 1 + VopExtraDwords(src0);
		} else { // VOP2
			n = 1 + (IsVop2AlwaysLiteral(enc7) ? 1 : VopExtraDwords(src0));
		}
	} else if ((w >> 30) == 0x2u) {
		const uint32_t enc9 = w >> 23;
		if (enc9 == 0x17du) { // SOP1
			n = 1 + ((w & 0xffu) == SrcLiteral ? 1 : 0);
		} else if (enc9 == 0x17eu) { // SOPC
			n = 1 + (((w & 0xffu) == SrcLiteral || ((w >> 8) & 0xffu) == SrcLiteral) ? 1 : 0);
		} else if (enc9 == 0x17fu) { // SOPP
			n = 1;
		} else if ((w >> 28) == 0xbu) { // SOPK, s_setreg_imm32_b32 carries a literal
			n = 1 + (((w >> 23) & 0x1fu) == 0x15u ? 1 : 0);
		} else { // SOP2
			n = 1 + (((w & 0xffu) == SrcLiteral || ((w >> 8) & 0xffu) == SrcLiteral) ? 1 : 0);
		}
	} else {
		switch (w >> 26) {
			case 0x32u: n = 1; break; // VINTRP
			case 0x35u:               // VOP3, VOP3B
				n = code.size() >= 2 ? 2 + Vop3LiteralDwords(code[1]) : 0;
				break;
			case 0x33u: // VOP3P (bits 31:23 == 0x198)
				n = ((w >> 23) == 0x198u && code.size() >= 2) ? 2 + Vop3LiteralDwords(code[1]) : 0;
				break;
			case 0x36u: // DS
			case 0x37u: // FLAT, GLOBAL, SCRATCH
			case 0x38u: // MUBUF
			case 0x3au: // MTBUF
			case 0x3du: // SMEM
			case 0x3eu: n = 2; break; // EXP
			case 0x3cu: n = 2 + ((w >> 1) & 0x3u); break; // MIMG with NSA address dwords
			default: n = 0; break;
		}
	}
	return n <= code.size() ? n : 0;
}

ShaderPatchResult PatchDot2c(std::span<const uint32_t> code, uint32_t num_vgprs) {
	ShaderPatchResult r;
	r.code.assign(code.begin(), code.end());

	// v[num_vgprs] is the one VGPR past the shader's allocation; the literal/DPP/aliasing stubs
	// use it as scratch. A site that needs scratch is only supported when it fits a VGPR index.
	const uint32_t scratch     = num_vgprs;
	const uint32_t scratch_src = SrcVgpr0 | scratch;
	const bool     scratch_ok  = scratch <= 0xffu;

	struct Site {
		uint32_t dw;
		uint32_t len;
		uint32_t vdst;
		uint32_t src0;
		uint32_t src1;
		uint32_t extra;   // code[dw + 1] for the 2-dword forms (literal value or DPP control word)
		bool     literal; // 8-byte literal src0
		bool     dpp;     // DPP16/DPP8 src0
		bool     scratch; // stub needs the scratch VGPR
	};
	std::vector<Site> sites;
	bool              any_scratch = false;

	for (uint32_t dw = 0; dw < code.size();) {
		const uint32_t len = InstructionDwords(code.subspan(dw));
		if (len == 0) {
			r.code.assign(code.begin(), code.end());
			r.decode_error    = true;
			r.decode_error_dw = dw;
			return r;
		}
		const uint32_t w = code[dw];
		if ((w >> 31) == 0u && (w >> 25) == OpVop2Dot2cF32F16) {
			const uint32_t src0     = w & 0x1ffu;
			const uint32_t src1     = SrcVgpr0 | ((w >> 9) & 0xffu);
			const uint32_t vdst     = (w >> 17) & 0xffu;
			const uint32_t vdst_src = SrcVgpr0 | vdst;
			const bool     reg_src0 = src0 >= SrcVgpr0 || src0 <= 107u;
			const bool     literal  = src0 == SrcLiteral && len == 2;
			const bool     dpp16    = src0 == SrcDpp16 && len == 2;
			const bool     dpp8     = (src0 == SrcDpp8 || src0 == SrcDpp8Fi) && len == 2;
			const bool     dpp      = dpp16 || dpp8;

			bool supported = false;
			bool scratch   = false;
			if (literal || dpp) {
				// v_fma_mix_f32 takes neither a literal nor DPP, so src0 is materialised into the
				// scratch VGPR. vsrc1 is read straight by both FMAs, so it must not alias vdst
				// (that would need a second scratch to preserve it); such a combo stays unsupported.
				// DPP8 has no masks or modifiers and copies verbatim; DPP16 only when it reduces.
				supported = src1 != vdst_src && (!dpp16 || Dpp16Reducible(code[dw + 1]));
				scratch   = true;
			} else if (reg_src0 && len == 1) {
				// Plain VOP2 with a VGPR/SGPR (incl. VCC) src0. The FMAs accumulate into vdst, so an
				// input that aliases vdst is clobbered before the second FMA; a scratch copy of vdst
				// preserves it (one copy serves both operands for v_dot2c vN, vN, vN).
				supported = true;
				scratch   = src0 == vdst_src || src1 == vdst_src;
			}
			// SDWA (src0 == 0xf9) and inline-constant src0 are left unsupported.
			if (supported && scratch && !scratch_ok) {
				supported = false;
			}
			if (supported) {
				sites.push_back({dw, len, vdst, src0, src1, len == 2 ? code[dw + 1] : 0, literal, dpp, scratch});
				any_scratch = any_scratch || scratch;
			} else {
				r.unsupported_sites++;
			}
		}
		dw += len;
	}

	bool ok = true;
	for (const auto& s: sites) {
		const auto stub = static_cast<uint32_t>(r.code.size());
		r.code[s.dw]    = Branch(s.dw, stub, &ok);

		uint32_t eff_src0 = s.src0;
		uint32_t eff_src1 = s.src1;
		if (s.literal) {
			r.code.push_back(Vop1MovB32 | (scratch << 17) | SrcLiteral); // v_mov_b32 scratch, literal
			r.code.push_back(s.extra);
			eff_src0 = scratch_src;
		} else if (s.dpp) {
			r.code.push_back(Vop1MovB32 | (scratch << 17) | s.src0); // v_mov_b32_dpp scratch, (dpp)
			r.code.push_back(s.extra);                               // DPP control word, verbatim
			eff_src0 = scratch_src;
		} else if (s.scratch) {
			r.code.push_back(Vop1MovB32 | (scratch << 17) | (SrcVgpr0 | s.vdst)); // v_mov_b32 scratch, vdst
			if (s.src0 == (SrcVgpr0 | s.vdst)) {
				eff_src0 = scratch_src;
			}
			if (s.src1 == (SrcVgpr0 | s.vdst)) {
				eff_src1 = scratch_src;
			}
		}

		const uint32_t word1 = eff_src0 | (eff_src1 << 9) | ((SrcVgpr0 | s.vdst) << 18) | OpSelHiMixF16;
		r.code.push_back(Vop3pFmaMixF32 | OpSelLoHalves | s.vdst);
		r.code.push_back(word1);
		r.code.push_back(Vop3pFmaMixF32 | OpSelHiHalves | s.vdst);
		r.code.push_back(word1);
		// Branch past the whole original instruction; for a 2-dword form its second word is dead.
		r.code.push_back(Branch(static_cast<uint32_t>(r.code.size()), s.dw + s.len, &ok));
	}
	if (!ok) {
		r.code.assign(code.begin(), code.end());
		r.unsupported_sites += static_cast<uint32_t>(sites.size());
		return r;
	}
	r.patched_sites = static_cast<uint32_t>(sites.size());
	r.scratch_vgprs = any_scratch ? 1 : 0;
	if (r.patched_sites > 0) {
		// The instruction prefetcher may read past the last stub.
		do {
			r.code.push_back(SCodeEnd);
		} while ((r.code.size() & 15u) != 0);
	}
	return r;
}

} // namespace Libs::Graphics::NativeGpu
