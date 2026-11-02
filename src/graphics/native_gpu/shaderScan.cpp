#include "graphics/native_gpu/shaderScan.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <iterator>
#include <map>

namespace Libs::Graphics::NativeGpu {

namespace {

struct IsaOpName {
	uint16_t    op;
	const char* name;
};

// kIsa<ENC>[] sorted by opcode and kIsaByEnc[ScanEnc]: every opcode llvm-mc decodes for gfx1013.
#include "graphics/native_gpu/shaderIsaGfx1013.inc"

static_assert(std::size(kIsaByEnc) == static_cast<size_t>(ScanEnc::Count), "shaderIsaGfx1013.inc encoding order");

const IsaOpName* FindOp(ScanEnc enc, uint16_t op) {
	if (enc >= ScanEnc::Count) {
		return nullptr;
	}
	const auto table = kIsaByEnc[static_cast<size_t>(enc)];
	const auto  it    = std::lower_bound(table.begin(), table.end(), op, [](const IsaOpName& e, uint16_t v) { return e.op < v; });
	return it != table.end() && it->op == op ? &*it : nullptr;
}

bool IsVop2AlwaysLiteral(uint16_t op) {
	// v_madmk_f32, v_madak_f32, v_fmamk_f32, v_fmaak_f32, v_fmamk_f16, v_fmaak_f16: the literal is an operand,
	// so src0 0xfa/0xf9 there is a register, not a DPP or SDWA marker.
	return op == 0x20u || op == 0x21u || op == 0x2cu || op == 0x2du || op == 0x37u || op == 0x38u;
}

// The DPP16 controls gfx10 defines (llvm-mc -mcpu=gfx1013): quad_perm, row_shl/shr/ror 1..15, row_mirror,
// row_half_mirror, row_share and row_xmask. The gfx9 wave_* shifts and row_bcast are gone; what the SQ does with
// them on gfx10 is undefined.
bool Dpp16CtrlValid(uint32_t ctrl) {
	return ctrl <= 0xffu || (ctrl >= 0x101u && ctrl <= 0x10fu) || (ctrl >= 0x111u && ctrl <= 0x11fu) ||
	       (ctrl >= 0x121u && ctrl <= 0x12fu) || ctrl == 0x140u || ctrl == 0x141u || (ctrl >= 0x150u && ctrl <= 0x16fu);
}

uint32_t Bit(const std::vector<uint8_t>& bits, uint32_t i) {
	return (bits[i / 8u] >> (i % 8u)) & 1u;
}
void SetBit(std::vector<uint8_t>& bits, uint32_t i) {
	bits[i / 8u] = static_cast<uint8_t>(bits[i / 8u] | (1u << (i % 8u)));
}

enum class Flow : uint8_t { Next, Stop, BranchStop, BranchNext };

class Walker {
public:
	Walker(std::span<const uint32_t> mem, const ShaderScanOptions& opt, ShaderScanResult* r): m_opt(opt), m_r(*r) {
		const uint32_t want = opt.known_dwords != 0 ? opt.known_dwords : opt.max_dwords;
		m_limit             = static_cast<uint32_t>(std::min<size_t>(mem.size(), want));
		m_mem               = mem.first(m_limit);
		m_r.insn_start.assign((m_limit + 7u) / 8u, 0);
		m_interior.assign((m_limit + 7u) / 8u, 0);
	}

	void Run() {
		if (m_limit != 0) {
			m_work.push_back(0);
		}
		while (!m_work.empty()) {
			const uint32_t start = m_work.back();
			m_work.pop_back();
			Path(start);
		}
		Finish();
	}

private:
	void Add(ScanKind kind, ScanSeverity sev, ScanEnc enc, uint16_t op, uint16_t detail, uint32_t dw, uint32_t word,
	         uint32_t count = 1) {
		for (ScanFinding& f: m_r.findings) {
			if (f.kind == kind && f.sev == sev && f.enc == enc && f.op == op && f.detail == detail) {
				f.count += count;
				if (dw < f.dw) {
					f.dw   = dw;
					f.word = word;
				}
				return;
			}
		}
		m_r.findings.push_back({kind, sev, enc, op, detail, dw, word, count});
	}

	// Without a registered size only pages the caller vouches for are read: a mis-followed branch must not
	// fault the emulator on an unmapped guest page.
	bool PageOk(uint32_t dw) {
		if (m_opt.known_dwords != 0 || !m_opt.readable || dw < m_opt.safe_dwords) {
			return true;
		}
		const uintptr_t page = reinterpret_cast<uintptr_t>(m_mem.data() + dw) >> 12u;
		for (const auto& [p, ok]: m_pages) {
			if (p == page) {
				return ok;
			}
		}
		const bool ok = m_opt.readable(dw);
		m_pages.emplace_back(page, ok);
		return ok;
	}

	bool IsStart(uint32_t dw) const { return Bit(m_r.insn_start, dw) != 0; }
	bool IsInterior(uint32_t dw) const { return Bit(m_interior, dw) != 0; }

	void PushTarget(uint32_t dw, uint32_t w, ScanEnc enc, uint16_t op) {
		const int64_t t = static_cast<int64_t>(dw) + 1 + static_cast<int16_t>(w & 0xffffu);
		if (t < 0 || t >= m_limit) {
			Add(ScanKind::BranchOutOfRange, ScanSeverity::Warn, enc, op, 0, dw, w);
			return;
		}
		const auto target = static_cast<uint32_t>(t);
		if (IsInterior(target)) {
			Add(ScanKind::OverlapDecode, ScanSeverity::Warn, enc, op, 0, dw, w);
			return;
		}
		if (!IsStart(target)) {
			m_work.push_back(target);
		}
	}

	void Path(uint32_t dw) {
		// s_getpc_b64 seen on this path: its SGPR pair, its dword and how many instructions may still add to it.
		int      pc_reg  = -1;
		uint32_t pc_dw   = 0;
		uint32_t pc_left = 0;
		for (;;) {
			if (dw >= m_limit) {
				m_r.truncated = true;
				return;
			}
			if (IsStart(dw)) {
				return; // joins a path already decoded
			}
			if (IsInterior(dw)) {
				Add(ScanKind::OverlapDecode, ScanSeverity::Warn, ScanEnc::Invalid, 0, 0, dw, 0);
				return;
			}
			if (!PageOk(dw) || (dw + 1 < m_limit && !PageOk(dw + 1))) {
				Add(ScanKind::BranchOutOfRange, ScanSeverity::Warn, ScanEnc::Invalid, 0, 0, dw, 0);
				return;
			}
			const auto     at  = m_mem.subspan(dw);
			const uint32_t len = InstructionDwords(at);
			if (len == 0) {
				// An instruction that runs past the limit is a truncated window, not an undecodable word.
				std::array<uint32_t, 5> pad {};
				std::copy_n(at.begin(), std::min<size_t>(at.size(), pad.size()), pad.begin());
				if (at.size() < pad.size() && InstructionDwords(pad) > at.size()) {
					m_r.truncated = true;
				} else {
					m_r.decode_error        = true;
					m_r.legacy.decode_error = true;
					Add(ScanKind::DecodeError, ScanSeverity::Error, ScanEnc::Invalid, 0, 0, dw, at[0]);
				}
				return;
			}
			if (!PageOk(dw + len - 1)) {
				Add(ScanKind::BranchOutOfRange, ScanSeverity::Warn, ScanEnc::Invalid, 0, 0, dw, at[0]);
				return;
			}
			for (uint32_t k = 1; k < len; k++) {
				if (IsStart(dw + k)) {
					Add(ScanKind::OverlapDecode, ScanSeverity::Warn, ScanEnc::Invalid, 0, 0, dw, at[0]);
					return;
				}
			}
			SetBit(m_r.insn_start, dw);
			for (uint32_t k = 1; k < len; k++) {
				SetBit(m_interior, dw + k);
			}
			m_r.instructions++;
			m_end = std::max(m_end, dw + len);

			const auto insn = at.first(len);
			uint16_t   op   = 0;
			const auto enc  = ScanDecodeOpcode(insn, &op);
			const uint32_t w = insn[0];

			// s_getpc_b64 s[n:n+1] then s_add_u32 sn, sn, literal: a pc-relative address (data in the image,
			// a jump table, the embedded constants).
			if (pc_left > 0) {
				pc_left--;
				const auto s0 = static_cast<int>(w & 0xffu);
				const auto s1 = static_cast<int>((w >> 8u) & 0xffu);
				if (enc == ScanEnc::SOP2 && op == 0x00u && len == 2 && static_cast<int>((w >> 16u) & 0x7fu) == pc_reg &&
				    ((s0 == pc_reg && s1 == 0xff) || (s0 == 0xff && s1 == pc_reg))) {
					const int64_t off = (static_cast<int64_t>(pc_dw) + 1) * 4 + static_cast<int32_t>(insn[1]);
					m_r.pcrel_min     = std::min(m_r.pcrel_min, off);
					m_r.pcrel_max     = std::max(m_r.pcrel_max, off);
					if (m_opt.known_dwords != 0 && (off < 0 || off >= static_cast<int64_t>(m_opt.known_dwords) * 4)) {
						Add(ScanKind::PcRelOutside, ScanSeverity::Info, enc, op, 0, dw, w);
					}
					pc_left = 0;
				}
			}
			if (enc == ScanEnc::SOP1 && op == 0x1fu) {
				pc_reg  = static_cast<int>((w >> 16u) & 0x7fu);
				pc_dw   = dw;
				pc_left = 4;
			}

			const Flow flow = Classify(insn, dw, enc, op);
			switch (flow) {
				case Flow::Next: break;
				case Flow::Stop: return;
				case Flow::BranchStop: PushTarget(dw, w, enc, op); return;
				case Flow::BranchNext: PushTarget(dw, w, enc, op); break;
			}
			dw += len;
		}
	}

	void Legacy(std::span<const uint32_t> insn, ScanEnc enc, uint16_t op) {
		// CountDot2c's counters, kept for the GDS policy and the audit (shaderPatcher.cpp).
		Dot2cScan&     s = m_r.legacy;
		const uint32_t w = insn[0];
		if (enc == ScanEnc::VOP2 && op == 0x02u) {
			s.sites++;
			s.dot_by_op[0]++;
		} else if (enc == ScanEnc::VOP2 && op == 0x0du) {
			s.dot_ops++;
			s.dot_by_op[1]++;
		} else if (enc == ScanEnc::VOP3P && op >= 0x13u && op <= 0x19u) {
			s.dot_ops++;
			s.dot_by_op[2u + op - 0x13u]++;
		}
		if (enc == ScanEnc::DS) {
			if (op == 0x3fu) {
				s.ordered_ops++;
			} else if (op >= 0x18u && op <= 0x1du) {
				s.gws_ops++;
			} else if (((w >> 17u) & 1u) != 0u) {
				s.gds_ops++;
			}
		}
		if (enc == ScanEnc::FLAT) {
			s.flat_ops++;
		}
		const auto aperture_src = [](uint32_t src) { return src >= 235u && src <= 238u; };
		if (enc == ScanEnc::SOPK && op == 0x12u) {
			const uint32_t id = w & 0x3fu;
			s.getreg[id]++;
			const auto simm = static_cast<uint16_t>(w & 0xffffu);
			const auto end  = s.hwid_simm.begin() + s.hwid_simm_n;
			if ((id == 4u || id == 23u || id == 24u) && s.hwid_simm_n < s.hwid_simm.size() &&
			    std::find(s.hwid_simm.begin(), end, simm) == end) {
				s.hwid_simm[s.hwid_simm_n++] = simm;
			}
			if (id == 15u) {
				s.aperture_ops++;
			}
		} else if (enc == ScanEnc::SOPK && (op == 0x13u || op == 0x15u)) {
			s.setreg[w & 0x3fu]++;
		} else if ((enc == ScanEnc::SOP1 || enc == ScanEnc::SOP2 || enc == ScanEnc::SOPC) &&
		           (aperture_src(w & 0xffu) || aperture_src((w >> 8u) & 0xffu))) {
			s.aperture_ops++;
		} else if ((enc == ScanEnc::VOP1 || enc == ScanEnc::VOP2 || enc == ScanEnc::VOPC) && aperture_src(w & 0x1ffu)) {
			s.aperture_ops++;
		} else if (enc == ScanEnc::VOP3 && (aperture_src(insn[1] & 0x1ffu) || aperture_src((insn[1] >> 9u) & 0x1ffu) ||
		                                    aperture_src((insn[1] >> 18u) & 0x1ffu))) {
			s.aperture_ops++;
		}
	}

	Flow Classify(std::span<const uint32_t> insn, uint32_t dw, ScanEnc enc, uint16_t op) {
		const uint32_t w = insn[0];
		if (enc < ScanEnc::Count) {
			m_r.by_enc[static_cast<size_t>(enc)]++;
		}
		Legacy(insn, enc, op);

		// Opcodes outside gfx1013. The dot instructions get their own kinds: v_dot2c_f32_f16 is the one the
		// BC-250 is known to execute as 0, the others were never probed.
		if (enc == ScanEnc::VOP2 && op == 0x02u) {
			Add(ScanKind::Dot2c, ScanSeverity::Error, enc, op, 0, dw, w);
		} else if ((enc == ScanEnc::VOP2 && op == 0x0du) || (enc == ScanEnc::VOP3P && op >= 0x13u && op <= 0x19u)) {
			Add(ScanKind::DotOther, ScanSeverity::Error, enc, op, 0, dw, w);
		} else if (!Gfx1013HasOpcode(enc, op)) {
			Add(ScanKind::UnknownOpcode, ScanSeverity::Error, enc, op, 0, dw, w);
		}

		// DPP16 controls and SDWA selects gfx10 does not define, and the VGPR range in use.
		if (enc == ScanEnc::VOP1 || enc == ScanEnc::VOP2 || enc == ScanEnc::VOPC) {
			const uint32_t src0 = w & 0x1ffu;
			if (enc != ScanEnc::VOPC) {
				m_r.max_vgpr_field = std::max(m_r.max_vgpr_field, (w >> 17u) & 0xffu);
			}
			if (!(enc == ScanEnc::VOP2 && IsVop2AlwaysLiteral(op)) && insn.size() >= 2) {
				const uint32_t x = insn[1];
				if (src0 == 0xfau) {
					const uint32_t ctrl = (x >> 8u) & 0x1ffu;
					if (!Dpp16CtrlValid(ctrl)) {
						Add(ScanKind::InvalidDpp, ScanSeverity::Warn, enc, op, static_cast<uint16_t>(ctrl), dw, w);
					}
				} else if (src0 == 0xf9u) {
					// detail: 1 dst_sel, 2 dst_unused, 3 src0_sel, 4 src1_sel. VOPC has sdst/SD in bits 15:8 instead of
					// the destination fields, and VOP1 has no src1.
					uint16_t bad = 0;
					if (enc != ScanEnc::VOPC && ((x >> 8u) & 7u) > 6u) {
						bad = 1;
					} else if (enc != ScanEnc::VOPC && ((x >> 11u) & 3u) == 3u) {
						bad = 2;
					} else if (((x >> 16u) & 7u) > 6u) {
						bad = 3;
					} else if (enc != ScanEnc::VOP1 && ((x >> 24u) & 7u) > 6u) {
						bad = 4;
					}
					if (bad != 0) {
						Add(ScanKind::InvalidSdwa, ScanSeverity::Warn, enc, op, bad, dw, w);
					}
				}
			}
		} else if (enc == ScanEnc::VOP3 || enc == ScanEnc::VOP3P) {
			m_r.max_vgpr_field = std::max(m_r.max_vgpr_field, w & 0xffu);
		}

		// Literals equal to the high dword of the PS5's LDS or scratch aperture base (SH_MEM_BASES 0x80007000):
		// a FLAT address built from them only works where the kernel set the PS5's apertures.
		uint32_t literal     = 0;
		bool     has_literal = false;
		if ((enc == ScanEnc::SOP1 || enc == ScanEnc::SOP2 || enc == ScanEnc::SOPC) && insn.size() == 2) {
			literal     = insn[1];
			has_literal = true;
		} else if ((enc == ScanEnc::VOP1 || enc == ScanEnc::VOP2 || enc == ScanEnc::VOPC) && insn.size() == 2 &&
		           ((w & 0x1ffu) == 0xffu || (enc == ScanEnc::VOP2 && IsVop2AlwaysLiteral(op)))) {
			literal     = insn[1];
			has_literal = true;
		} else if ((enc == ScanEnc::VOP3 || enc == ScanEnc::VOP3P) && insn.size() == 3) {
			literal     = insn[2];
			has_literal = true;
		}
		if (has_literal && (literal == 0x80000000u || literal == 0x70000000u)) {
			if (m_aperture_literals++ == 0 || dw < m_aperture_dw) {
				m_aperture_dw   = dw;
				m_aperture_word = w;
				m_aperture_lit  = literal;
			}
		}

		switch (enc) {
			case ScanEnc::SOPK:
				if (op == 0x12u || op == 0x13u || op == 0x15u) {
					ClassifyHwreg(w, dw, op);
				} else if (op == 0x16u || op == 0x1bu || op == 0x1cu) {
					return Flow::BranchNext; // s_call_b64, s_subvector_loop_begin/end
				}
				break;
			case ScanEnc::SOPP:
				switch (op) {
					case 0x01u: // s_endpgm
					case 0x1bu: // s_endpgm_saved
					case 0x1eu: // s_endpgm_ordered_ps_done
						m_r.ended        = true;
						m_r.legacy.ended = true;
						m_r.first_end_dw = std::min(m_r.first_end_dw, dw);
						return Flow::Stop;
					case 0x02u: return Flow::BranchStop; // s_branch
					case 0x1fu: // s_code_end
						Add(ScanKind::CodeEndReached, ScanSeverity::Warn, enc, op, 0, dw, w);
						return Flow::Stop;
					case 0x04u:
					case 0x05u:
					case 0x06u:
					case 0x07u:
					case 0x08u:
					case 0x09u: return Flow::BranchNext; // s_cbranch_scc0/scc1/vccz/vccnz/execz/execnz
					case 0x17u:
					case 0x18u:
					case 0x19u:
					case 0x1au: // s_cbranch_cdbg*: branch on the debugger's flags
						Add(ScanKind::Trap, ScanSeverity::Warn, enc, op, 0, dw, w);
						return Flow::BranchNext;
					case 0x0du: // s_sethalt
					case 0x12u: // s_trap
					case 0x16u: // s_ttracedata
					case 0x28u: // s_ttracedata_imm
						// Neither the render node nor KFD with cwsr_enable=0 has a trap handler, and on the BC-250 a
						// halted wave hangs the gfx ring.
						Add(ScanKind::Trap, ScanSeverity::Warn, enc, op, 0, dw, w);
						break;
					case 0x10u: // s_sendmsg
					case 0x11u: { // s_sendmsghalt
						// GS, GS_DONE, ORDERED_PS_DONE, EARLY_PRIM_DEALLOC and GS_ALLOC_REQ are render traffic the SPI
						// handles the same on both boards; INTERRUPT goes to a kernel (amdgpu ignores it), SAVEWAVE,
						// STALL_WAVE_GEN, HALT_WAVES and SYSMSG are debug or halt, GET_DOORBELL/GET_DDID return host values.
						const uint32_t id = w & 0xfu;
						if (id != 2u && id != 3u && id != 7u && id != 8u && id != 9u) {
							Add(ScanKind::SendMsg, ScanSeverity::Warn, enc, op, static_cast<uint16_t>(id), dw, w);
						}
						break;
					}
					default: break;
				}
				break;
			case ScanEnc::SOP1:
				if (op == 0x20u) { // s_setpc_b64: a subroutine return, or a jump through a register
					m_r.legacy.ended = true;
					Add(ScanKind::IndirectBranch, ScanSeverity::Info, enc, op, 0, dw, w);
					return Flow::Stop;
				}
				if (op == 0x22u) { // s_rfe_b64: return from a trap handler, which no title has
					Add(ScanKind::IndirectBranch, ScanSeverity::Warn, enc, op, 0, dw, w);
					return Flow::Stop;
				}
				if (op == 0x21u) { // s_swappc_b64: a call through a register (fetch shaders); it returns here
					Add(ScanKind::IndirectBranch, ScanSeverity::Info, enc, op, 0, dw, w);
				}
				break;
			case ScanEnc::MIMG:
				if (op == 0xe6u || op == 0xe7u) {
					Add(ScanKind::Bvh, ScanSeverity::Warn, enc, op, 0, dw, w);
				} else if (op == 0x80u) {
					Add(ScanKind::MsaaLoad, ScanSeverity::Info, enc, op, 0, dw, w);
				}
				break;
			case ScanEnc::SMEM:
				if (op == 0x24u || op == 0x25u) { // s_memtime, s_memrealtime: the clock rates may differ
					Add(ScanKind::Timer, ScanSeverity::Info, enc, op, 0, dw, w);
				}
				break;
			case ScanEnc::DS:
				if (op == 0x3fu || (op >= 0x18u && op <= 0x1du) || ((w >> 17u) & 1u) != 0u) {
					Add(ScanKind::Gds, ScanSeverity::Info, enc, op, 0, dw, w);
				}
				break;
			default: break;
		}
		return Flow::Next;
	}

	void ClassifyHwreg(uint32_t w, uint32_t dw, uint16_t op) {
		const uint32_t id   = w & 0x3fu;
		ScanSeverity   sev  = ScanSeverity::Error;
		bool           safe = false;
		if (op == 0x12u) {
			// MODE, STATUS, TRAPSTS, GPR_ALLOC, LDS_ALLOC and IB_STS are wave-local: the same on the same silicon.
			// SH_MEM_BASES is the kernel's (PS5 0x80007000); TBA/TMA, FLAT_SCR, XNACK_MASK, HW_ID1/2 (harvest,
			// VMID, queue) and POPS_PACKER depend on the kernel, the queue or the harvest.
			if (id == 1u || id == 2u || id == 3u || id == 5u || id == 6u || id == 7u) {
				safe = true;
			} else if (id == 15u) {
				sev = m_opt.ps5_apertures ? ScanSeverity::Info : ScanSeverity::Warn;
			} else if (id >= 16u && id <= 25u) {
				sev = ScanSeverity::Warn;
			}
		} else if (id == 1u || id == 3u) {
			safe = true; // MODE, and TRAPSTS's exception-clear fields
		}
		if (!safe) {
			Add(ScanKind::Hwreg, sev, ScanEnc::SOPK, op, static_cast<uint16_t>(id), dw, w);
		}
	}

	void Finish() {
		m_r.image_dwords  = m_opt.known_dwords != 0 ? m_opt.known_dwords : m_end;
		m_r.legacy.dwords = m_end;
		m_r.legacy.instructions = m_r.instructions;
		if (m_aperture_literals > 0 && m_r.legacy.flat_ops > 0) {
			Add(ScanKind::ApertureLiteral, m_opt.ps5_apertures ? ScanSeverity::Info : ScanSeverity::Warn, ScanEnc::Invalid, 0,
			    static_cast<uint16_t>(m_aperture_lit >> 16u), m_aperture_dw, m_aperture_word, m_aperture_literals);
		}
		if (m_r.first_end_dw != UINT32_MAX) {
			uint32_t n     = 0;
			uint32_t first = 0;
			for (uint32_t dw = m_r.first_end_dw + 1; dw < m_end; dw++) {
				if (IsStart(dw)) {
					first = n == 0 ? dw : first;
					n++;
				}
			}
			if (n > 0) {
				Add(ScanKind::PastFirstEnd, ScanSeverity::Info, ScanEnc::Invalid, 0, 0, first, m_mem[first], n);
			}
		}
	}

	const ShaderScanOptions&   m_opt;
	ShaderScanResult&          m_r;
	std::span<const uint32_t>  m_mem;
	uint32_t                   m_limit = 0;
	uint32_t                   m_end   = 0;
	std::vector<uint8_t>       m_interior;
	std::vector<uint32_t>      m_work;
	std::vector<std::pair<uintptr_t, bool>> m_pages;
	uint32_t                   m_aperture_literals = 0;
	uint32_t                   m_aperture_dw       = 0;
	uint32_t                   m_aperture_word     = 0;
	uint32_t                   m_aperture_lit      = 0;
};

const char* const kKindNames[] = {
    "dot2c",          "dot-other", "unknown-opcode", "hwreg",       "invalid-dpp",      "invalid-sdwa",  "bvh",
    "msaa-load",      "sendmsg",   "trap",           "aperture-literal", "gds",         "timer",         "indirect-branch",
    "branch-out-of-range", "overlap-decode", "code-end-reached", "past-first-end", "pcrel-outside", "decode-error",
};
static_assert(std::size(kKindNames) == static_cast<size_t>(ScanKind::Count), "kKindNames");

const char* const kEncNames[] = {"SOP1", "SOP2", "SOPK", "SOPC", "SOPP", "SMEM",  "VOP1",  "VOP2",    "VOPC",   "VOP3",
                                 "VOP3P", "VINTRP", "DS", "MUBUF", "MTBUF", "MIMG", "FLAT", "SCRATCH", "GLOBAL", "EXP"};
static_assert(std::size(kEncNames) == static_cast<size_t>(ScanEnc::Count), "kEncNames");

} // namespace

ScanEnc ScanDecodeOpcode(std::span<const uint32_t> code, uint16_t* op) {
	*op = 0;
	if (code.empty()) {
		return ScanEnc::Invalid;
	}
	const uint32_t w = code[0];
	if ((w >> 31u) == 0u) {
		const uint32_t enc7 = w >> 25u;
		if (enc7 == 0x3fu) {
			*op = static_cast<uint16_t>((w >> 9u) & 0xffu);
			return ScanEnc::VOP1;
		}
		if (enc7 == 0x3eu) {
			*op = static_cast<uint16_t>((w >> 17u) & 0xffu);
			return ScanEnc::VOPC;
		}
		*op = static_cast<uint16_t>(enc7);
		return ScanEnc::VOP2;
	}
	if ((w >> 30u) == 0x2u) {
		const uint32_t enc9 = w >> 23u;
		if (enc9 == 0x17du) {
			*op = static_cast<uint16_t>((w >> 8u) & 0xffu);
			return ScanEnc::SOP1;
		}
		if (enc9 == 0x17eu) {
			*op = static_cast<uint16_t>((w >> 16u) & 0x7fu);
			return ScanEnc::SOPC;
		}
		if (enc9 == 0x17fu) {
			*op = static_cast<uint16_t>((w >> 16u) & 0x7fu);
			return ScanEnc::SOPP;
		}
		if ((w >> 28u) == 0xbu) {
			*op = static_cast<uint16_t>((w >> 23u) & 0x1fu);
			return ScanEnc::SOPK;
		}
		*op = static_cast<uint16_t>((w >> 23u) & 0x7fu);
		return ScanEnc::SOP2;
	}
	switch (w >> 26u) {
		case 0x32u: *op = static_cast<uint16_t>((w >> 16u) & 0x3u); return ScanEnc::VINTRP;
		case 0x35u: *op = static_cast<uint16_t>((w >> 16u) & 0x3ffu); return ScanEnc::VOP3;
		case 0x33u:
			if ((w >> 23u) != 0x198u) {
				return ScanEnc::Invalid;
			}
			*op = static_cast<uint16_t>((w >> 16u) & 0x7fu);
			return ScanEnc::VOP3P;
		case 0x36u: *op = static_cast<uint16_t>((w >> 18u) & 0xffu); return ScanEnc::DS;
		case 0x37u: {
			*op                = static_cast<uint16_t>((w >> 18u) & 0x7fu);
			const uint32_t seg = (w >> 14u) & 0x3u;
			return seg == 0u ? ScanEnc::FLAT : seg == 1u ? ScanEnc::SCRATCH : seg == 2u ? ScanEnc::GLOBAL : ScanEnc::Invalid;
		}
		case 0x38u: *op = static_cast<uint16_t>(((w >> 18u) & 0x7fu) | (((w >> 25u) & 1u) << 7u)); return ScanEnc::MUBUF;
		case 0x3au:
			*op = static_cast<uint16_t>(((w >> 16u) & 0x7u) | (code.size() >= 2 ? ((code[1] >> 21u) & 1u) << 3u : 0u));
			return ScanEnc::MTBUF;
		case 0x3cu: *op = static_cast<uint16_t>(((w >> 18u) & 0x7fu) | ((w & 1u) << 7u)); return ScanEnc::MIMG;
		case 0x3du: *op = static_cast<uint16_t>((w >> 18u) & 0xffu); return ScanEnc::SMEM;
		case 0x3eu: return ScanEnc::EXP;
		default: return ScanEnc::Invalid;
	}
}

bool Gfx1013HasOpcode(ScanEnc enc, uint16_t op) {
	return FindOp(enc, op) != nullptr;
}

const char* Gfx1013Mnemonic(ScanEnc enc, uint16_t op) {
	const IsaOpName* e = FindOp(enc, op);
	return e != nullptr ? e->name : nullptr;
}

const char* ScanEncName(ScanEnc enc) {
	return enc < ScanEnc::Count ? kEncNames[static_cast<size_t>(enc)] : "?";
}

const char* ScanKindName(ScanKind kind) {
	return kind < ScanKind::Count ? kKindNames[static_cast<size_t>(kind)] : "?";
}

const char* ScanSeverityName(ScanSeverity sev) {
	return sev == ScanSeverity::Error ? "error" : sev == ScanSeverity::Warn ? "warn" : "info";
}

uint32_t ShaderScanResult::Count(ScanKind k) const {
	uint32_t n = 0;
	for (const ScanFinding& f: findings) {
		n += f.kind == k ? f.count : 0;
	}
	return n;
}

uint32_t ShaderScanResult::Count(ScanKind k, ScanSeverity s) const {
	uint32_t n = 0;
	for (const ScanFinding& f: findings) {
		n += f.kind == k && f.sev == s ? f.count : 0;
	}
	return n;
}

ScanSeverity ShaderScanResult::Worst() const {
	ScanSeverity s = ScanSeverity::Info;
	for (const ScanFinding& f: findings) {
		s = std::max(s, f.sev);
	}
	return s;
}

ShaderScanResult ScanShader(std::span<const uint32_t> mem, const ShaderScanOptions& opt) {
	ShaderScanResult r;
	Walker(mem, opt, &r).Run();
	return r;
}

// ---------------------------------------------------------------------------------------------------------------

ShaderScanReporter::ShaderScanReporter(int level, Hasher hasher, std::FILE* out): m_level(level), m_hasher(hasher), m_out(out) {}

namespace {

const char* StageName(int type) {
	// Prospero::ShaderBinaryType.
	static const char* const kNames[] = {"cs", "ps", "gs", "hs", "gs-front", "hs-front", "gs-back", "hs-back", "fs"};
	return type >= 0 && type < static_cast<int>(std::size(kNames)) ? kNames[type] : "?";
}

} // namespace

bool ShaderScanReporter::Detect(uint64_t va, std::span<const uint32_t> window, const Registered* reg, uint32_t safe_dwords,
                                const std::function<bool(uint32_t dw)>& readable, const char* dump_dir) {
	// Code AgcCreateShader registered is identified by shader.cpp's hash before anything is read; other code
	// (registers a title baked into its command buffers, library shaders) by the hash of its reachable image.
	if (reg != nullptr && m_hashes.count(reg->hash) != 0) {
		m_duplicates++;
		return false;
	}
	const auto        t0 = std::chrono::steady_clock::now();
	ShaderScanOptions opt;
	opt.known_dwords = reg != nullptr ? reg->size_bytes / 4u : 0u;
	opt.safe_dwords  = safe_dwords;
	opt.readable     = readable;
	const ShaderScanResult r = ScanShader(window, opt);
	// An unregistered image is hashed and dumped only as far as it is readable: the walk read only reachable
	// words, the gaps between them (data, padding) are checked page by page here.
	uint32_t image = std::min<uint32_t>(r.image_dwords, static_cast<uint32_t>(window.size()));
	if (reg == nullptr && readable) {
		for (uint32_t dw = safe_dwords; dw < image;) {
			if (!readable(dw)) {
				image = dw;
				break;
			}
			const uintptr_t addr = reinterpret_cast<uintptr_t>(window.data() + dw);
			dw += static_cast<uint32_t>((((addr | 0xfffu) + 1u) - addr) / 4u);
		}
	}
	const uint64_t hash = reg != nullptr ? reg->hash : (m_hasher != nullptr ? m_hasher(window.data(), size_t {image} * 4u) : va);
	m_scan_ns += static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
	if (!m_hashes.insert(hash).second) {
		m_duplicates++;
		return false;
	}
	m_shaders++;
	m_registered += reg != nullptr ? 1u : 0u;

	std::array<bool, static_cast<size_t>(ScanKind::Count)> seen {};
	for (const ScanFinding& f: r.findings) {
		m_kind_insns[static_cast<size_t>(f.kind)] += f.count;
		seen[static_cast<size_t>(f.kind)] = true;
	}
	for (size_t k = 0; k < seen.size(); k++) {
		m_kind_shaders[k] += seen[k] ? 1u : 0u;
	}
	const bool flagged = r.Worst() != ScanSeverity::Info;
	if (flagged) {
		m_flagged++;
		m_flagged_registered += reg != nullptr ? 1u : 0u;
	}

	// Errors first, at most 8 lines a shader; Info only at level 2.
	std::vector<const ScanFinding*> lines;
	for (const ScanFinding& f: r.findings) {
		if (f.sev != ScanSeverity::Info || m_level >= 2) {
			lines.push_back(&f);
		}
	}
	std::stable_sort(lines.begin(), lines.end(), [](const ScanFinding* a, const ScanFinding* b) { return a->sev > b->sev; });
	const size_t shown = std::min<size_t>(lines.size(), 8);
	for (size_t i = 0; i < shown; i++) {
		const ScanFinding& f = *lines[i];
		char               what[64];
		if (const char* name = Gfx1013Mnemonic(f.enc, f.op); name != nullptr) {
			std::snprintf(what, sizeof(what), "%s", name);
		} else if (f.enc < ScanEnc::Count) {
			std::snprintf(what, sizeof(what), "%s:0x%x", ScanEncName(f.enc), f.op);
		} else {
			std::snprintf(what, sizeof(what), "-");
		}
		char detail[24] = "";
		if (f.kind == ScanKind::Hwreg || f.kind == ScanKind::SendMsg || f.kind == ScanKind::InvalidDpp ||
		    f.kind == ScanKind::InvalidSdwa || f.kind == ScanKind::ApertureLiteral) {
			std::snprintf(detail, sizeof(detail), " (0x%x)", f.detail);
		}
		std::fprintf(m_out, "native shader scan: %016" PRIx64 " va 0x%" PRIx64 " %u dw (%s) %s: %s %s %s%s x%u first +%u word 0x%08x\n",
		             hash, va, r.image_dwords, reg != nullptr ? "agc" : "swept", reg != nullptr ? StageName(reg->type) : "?",
		             ScanSeverityName(f.sev), ScanKindName(f.kind), what, detail, f.count, f.dw, f.word);
		m_lines++;
	}
	if (lines.size() > shown) {
		std::fprintf(m_out, "native shader scan: %016" PRIx64 " ... %zu more finding(s)\n", hash, lines.size() - shown);
		m_lines++;
	}
	if (m_level >= 2) {
		std::fprintf(m_out,
		             "native shader scan: %016" PRIx64 " va 0x%" PRIx64 " %u dw (%s) %s: %u instrs, %s%s, pcrel [%" PRId64 ", %" PRId64 "]\n",
		             hash, va, r.image_dwords, reg != nullptr ? "agc" : "swept", reg != nullptr ? StageName(reg->type) : "?",
		             r.instructions, r.ended ? "ended" : "no s_endpgm", r.truncated ? ", truncated" : "",
		             r.pcrel_min == INT64_MAX ? int64_t {0} : r.pcrel_min, r.pcrel_max == INT64_MIN ? int64_t {0} : r.pcrel_max);
		m_lines++;
		// The opcode histogram: what the PS5 compiler emits, against shaderIsaGfx1013.inc.
		std::unordered_set<uint32_t> in_shader;
		for (uint32_t dw = 0; dw < image; dw++) {
			if (!r.IsStart(dw)) {
				continue;
			}
			uint16_t       op  = 0;
			const ScanEnc  enc = ScanDecodeOpcode(window.subspan(dw, std::min<size_t>(2, window.size() - dw)), &op);
			const uint32_t key = (static_cast<uint32_t>(enc) << 16u) | op;
			auto&          h   = m_hist[key];
			h.first++;
			if (in_shader.insert(key).second) {
				h.second++;
			}
		}
	}

	// Flagged code goes to KYTY_NATIVE_GPU_DUMP_DIR as shader_<hash>.bin, exactly the image, for llvm-mc
	// (-mcpu=gfx1013 --disassemble). It is game content: it stays on the machine.
	if (flagged && dump_dir != nullptr && image != 0) {
		char path[512];
		std::snprintf(path, sizeof(path), "%s/shader_%016" PRIx64 ".bin", dump_dir, hash);
		if (std::FILE* f = std::fopen(path, "wb"); f != nullptr) {
			std::fwrite(window.data(), 4, image, f);
			std::fclose(f);
		}
	}
	return true;
}

void ShaderScanReporter::PrintSummary(const char* why) {
	std::string kinds;
	for (size_t k = 0; k < m_kind_insns.size(); k++) {
		if (m_kind_insns[k] != 0) {
			char part[64];
			std::snprintf(part, sizeof(part), " %s:%u/%u", kKindNames[k], m_kind_insns[k], m_kind_shaders[k]);
			kinds += part;
		}
	}
	std::fprintf(m_out,
	             "native shader scan (%s): %u shaders (%u registered), %u duplicates, %u flagged (%u registered), %.1f ms;%s\n",
	             why, m_shaders, m_registered, m_duplicates, m_flagged, m_flagged_registered,
	             static_cast<double>(m_scan_ns) / 1e6, kinds.empty() ? " no findings" : kinds.c_str());
}

void ShaderScanReporter::WriteHistogram(const char* dir) const {
	if (dir == nullptr || m_hist.empty()) {
		return;
	}
	char path[512];
	std::snprintf(path, sizeof(path), "%s/shader_hist.txt", dir);
	std::FILE* f = std::fopen(path, "w");
	if (f == nullptr) {
		return;
	}
	std::fprintf(f, "# enc op mnemonic instructions shaders (%u shaders)\n", m_shaders);
	const std::map<uint32_t, std::pair<uint64_t, uint32_t>> sorted(m_hist.begin(), m_hist.end());
	for (const auto& [key, h]: sorted) {
		const auto        enc  = static_cast<ScanEnc>(key >> 16u);
		const auto        op   = static_cast<uint16_t>(key & 0xffffu);
		const char*       name = Gfx1013Mnemonic(enc, op);
		std::fprintf(f, "%s 0x%03x %s %" PRIu64 " %u\n", ScanEncName(enc), op, name != nullptr ? name : "(not-gfx1013)", h.first,
		             h.second);
	}
	std::fclose(f);
}

} // namespace Libs::Graphics::NativeGpu
