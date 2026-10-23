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

ShaderPatchResult PatchDot2c(std::span<const uint32_t> code) {
	ShaderPatchResult r;
	r.code.assign(code.begin(), code.end());

	struct Site {
		uint32_t dw;
		uint32_t vdst;
		uint32_t src0;
		uint32_t src1;
	};
	std::vector<Site> sites;

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
			const uint32_t src0 = w & 0x1ffu;
			const uint32_t src1 = SrcVgpr0 | ((w >> 9) & 0xffu);
			const uint32_t vdst = (w >> 17) & 0xffu;
			// Plain VOP2 with a VGPR or SGPR (incl. VCC) src0 that does not alias vdst: the two
			// FMAs update vdst in place, so an aliased input would be clobbered before the second.
			const bool reg_src0 = src0 >= SrcVgpr0 || src0 <= 107u;
			if (len == 1 && reg_src0 && src0 != (SrcVgpr0 | vdst) && src1 != (SrcVgpr0 | vdst)) {
				sites.push_back({dw, vdst, src0, src1});
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

		const uint32_t word1 = s.src0 | (s.src1 << 9) | ((SrcVgpr0 | s.vdst) << 18) | OpSelHiMixF16;
		r.code.push_back(Vop3pFmaMixF32 | OpSelLoHalves | s.vdst);
		r.code.push_back(word1);
		r.code.push_back(Vop3pFmaMixF32 | OpSelHiHalves | s.vdst);
		r.code.push_back(word1);
		r.code.push_back(Branch(static_cast<uint32_t>(r.code.size()), s.dw + 1, &ok));
	}
	if (!ok) {
		r.code.assign(code.begin(), code.end());
		r.unsupported_sites += static_cast<uint32_t>(sites.size());
		return r;
	}
	r.patched_sites = static_cast<uint32_t>(sites.size());
	if (r.patched_sites > 0) {
		// The instruction prefetcher may read past the last stub.
		do {
			r.code.push_back(SCodeEnd);
		} while ((r.code.size() & 15u) != 0);
	}
	return r;
}

} // namespace Libs::Graphics::NativeGpu
