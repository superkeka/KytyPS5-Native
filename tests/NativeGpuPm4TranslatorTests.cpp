#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

using Libs::Graphics::NativeGpu::GpuRef;
using Libs::Graphics::NativeGpu::Pm4TranslateResult;
using Libs::Graphics::NativeGpu::Pm4Translator;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuPm4TranslatorTests: failed: %s\n", message);
		std::abort();
	}
}

// Type-3 opcodes used in the tests (pm4.h).
constexpr uint32_t IT_NOP                      = 0x10;
constexpr uint32_t IT_CLEAR_STATE              = 0x12;
constexpr uint32_t IT_DISPATCH_DIRECT          = 0x15;
constexpr uint32_t IT_DRAW_INDEX_2             = 0x27;
constexpr uint32_t IT_NUM_INSTANCES            = 0x2F;
constexpr uint32_t IT_CONTEXT_CONTROL          = 0x28;
constexpr uint32_t IT_EVENT_WRITE              = 0x46;
constexpr uint32_t IT_ACQUIRE_MEM              = 0x58;
constexpr uint32_t IT_SET_SH_REG_INDIRECT      = 0x63;
constexpr uint32_t IT_SET_UCONFIG_REG_INDIRECT = 0x64;
constexpr uint32_t IT_SET_CONTEXT_REG          = 0x69;
constexpr uint32_t IT_SET_SH_REG               = 0x76;
constexpr uint32_t IT_SET_UCONFIG_REG          = 0x79;
constexpr uint32_t IT_WRITE_DATA               = 0x37;
constexpr uint32_t IT_SET_UCONFIG_REG_INDEX    = 0x7A;
constexpr uint32_t IT_SET_CONTEXT_REG_INDIRECT = 0x9F;

constexpr uint32_t R_FLIP = 0x17;

uint32_t Pkt3(uint32_t op, uint32_t body_dw, bool compute, uint32_t marker = 0) {
	return (3u << 30u) | (((body_dw - 1u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) |
	       ((marker & 0x3fu) << 2u) | (compute ? (1u << 1u) : 0u);
}

// Mirrors the decode in nativePm4.cpp / pm4.h.
uint32_t PacketSizeDw(uint32_t header) {
	return (header & 0x3fffff00u) == 0x3fff1000u ? 1u : (((header >> 16u) & 0x3fffu) + 2u);
}

struct Packet {
	uint32_t              op;
	bool                  compute;
	std::vector<uint32_t> body;
};

std::vector<Packet> Decode(const std::vector<uint32_t>& ib) {
	std::vector<Packet> out;
	uint32_t            dw = 0;
	while (dw < ib.size()) {
		const uint32_t header = ib[dw];
		Check((header >> 30u) == 3u, "IB packet is type-3");
		const uint32_t size = PacketSizeDw(header);
		Check(size >= 1 && dw + size <= ib.size(), "IB packet fits");
		const uint32_t op = (header >> 8u) & 0xffu;
		if (op == IT_NOP) { // filler padding
			dw += size;
			continue;
		}
		Packet p;
		p.op      = op;
		p.compute = (header & 0x2u) != 0;
		for (uint32_t i = 1; i < size; i++) {
			p.body.push_back(ib[dw + i]);
		}
		out.push_back(p);
		dw += size;
	}
	return out;
}

// Appends a packet (header + non-empty body) to a DCB.
void Append(std::vector<uint32_t>& dcb, uint32_t op, const std::vector<uint32_t>& body, bool compute,
            uint32_t marker = 0) {
	Check(!body.empty(), "test packet has a body");
	dcb.push_back(Pkt3(op, static_cast<uint32_t>(body.size()), compute, marker));
	for (uint32_t v: body) {
		dcb.push_back(v);
	}
}

// An indirect register-load packet pointing at offset/value pairs (which the test owns).
void AppendIndirect(std::vector<uint32_t>& dcb, uint32_t op, const std::vector<uint32_t>& pairs) {
	const auto     addr = reinterpret_cast<uint64_t>(pairs.data());
	const uint32_t num  = static_cast<uint32_t>(pairs.size() / 2);
	std::vector<uint32_t> body = {static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 0u,
	                              num};
	dcb.push_back(Pkt3(op, 4, false));
	for (uint32_t v: body) {
		dcb.push_back(v);
	}
}

bool IbAligned(const std::vector<uint32_t>& ib) { return (ib.size() & 7u) == 0; }

const Packet* FindFirst(const std::vector<Packet>& pkts, uint32_t op) {
	for (const auto& p: pkts) {
		if (p.op == op) {
			return &p;
		}
	}
	return nullptr;
}

uint32_t Count(const std::vector<Packet>& pkts, uint32_t op) {
	uint32_t n = 0;
	for (const auto& p: pkts) {
		n += (p.op == op) ? 1u : 0u;
	}
	return n;
}

// Every IB must open with CONTEXT_CONTROL then CLEAR_STATE.
void CheckPreambleHead(const std::vector<Packet>& pkts) {
	Check(pkts.size() >= 2, "IB has a preamble");
	Check(pkts[0].op == IT_CONTEXT_CONTROL, "IB starts with CONTEXT_CONTROL");
	Check(pkts[1].op == IT_CLEAR_STATE, "CLEAR_STATE follows CONTEXT_CONTROL");
}

void TestNopsOnly() {
	Pm4Translator        t;
	std::vector<uint32_t> dcb;
	Append(dcb, IT_NOP, {0x68750017u}, false, R_FLIP); // a marker NOP with a body
	dcb.push_back(0xffff1000u);                         // a filler NOP
	auto r = t.Translate(dcb);
	Check(r.ok, "nop-only DCB translates");
	Check(r.stats.nops == 2, "both NOPs counted");
	Check(r.stats.marker_flip == 1, "flip marker counted");
	Check(IbAligned(r.ib), "IB padded to 8 dwords");
	auto pkts = Decode(r.ib);
	CheckPreambleHead(pkts);
	Check(pkts.size() == 2, "no body packets from NOPs");
}

void TestContextShadowAcrossDcbs() {
	Pm4Translator t;
	std::vector<uint32_t> dcb1;
	Append(dcb1, IT_SET_CONTEXT_REG, {0x5u, 0xaaaau}, false); // reg 0x5 = 0xaaaa
	Append(dcb1, IT_DRAW_INDEX_2, {1u, 2u, 3u, 0u}, false);
	auto r1 = t.Translate(dcb1);
	Check(r1.ok && r1.stats.set_context == 1 && r1.stats.draws == 1, "dcb1 counts");
	auto p1 = Decode(r1.ib);
	CheckPreambleHead(p1);
	Check(Count(p1, IT_DRAW_INDEX_2) == 1, "draw passed through");

	// Second DCB sets nothing new; its preamble must still replay reg 0x5 from the shadow.
	std::vector<uint32_t> dcb2;
	Append(dcb2, IT_DRAW_INDEX_2, {4u, 5u, 6u, 0u}, false);
	auto r2 = t.Translate(dcb2);
	Check(r2.ok, "dcb2 translates");
	Check(r2.stats.preamble_cx == 1, "shadowed context reg replayed in preamble");
	auto p2  = Decode(r2.ib);
	bool seen = false;
	for (const auto& p: p2) {
		if (p.op == IT_SET_CONTEXT_REG && p.body.size() == 2 && p.body[0] == 0x5u &&
		    p.body[1] == 0xaaaau) {
			seen = true;
		}
	}
	Check(seen, "preamble carries reg 0x5 = 0xaaaa");
}

void TestShaderDetection() {
	Pm4Translator t;
	std::vector<uint32_t> dcb;
	// SPI_SHADER_PGM_LO_PS (0x8) and HI (0x9) set in one SET_SH_REG → base VA (lo<<8)|(hi<<40).
	Append(dcb, IT_SET_SH_REG, {0x8u, 0x00123400u, 0x00000012u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.set_sh == 1, "sh set counted");
	Check(r.shader_addrs.size() == 1, "one shader address collected");
	const uint64_t want = (static_cast<uint64_t>(0x00123400u) << 8u) | (static_cast<uint64_t>(0x12u) << 40u);
	Check(r.shader_addrs[0] == want, "shader VA from PGM_LO/HI");
}

void TestIndirectContextExpansion() {
	Pm4Translator t;
	std::vector<uint32_t> pairs = {0x10u, 0x1111u, 0x20u, 0x2222u}; // two (offset,value) pairs
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs);
	auto r = t.Translate(dcb);
	Check(r.ok, "indirect cx translates");
	Check(r.stats.load_context == 1 && r.stats.loaded_cx_regs == 2, "indirect cx expanded to 2");
	auto     pkts = Decode(r.ib);
	uint32_t sets = 0;
	for (const auto& p: pkts) {
		if (p.op == IT_SET_CONTEXT_REG && p.body.size() == 2) {
			if (p.body[0] == 0x10u) {
				Check(p.body[1] == 0x1111u, "pair 0 value");
				sets++;
			}
			if (p.body[0] == 0x20u) {
				Check(p.body[1] == 0x2222u, "pair 1 value");
				sets++;
			}
		}
	}
	// Each appears twice: once in the preamble (shadow) and once in the translated body.
	Check(sets == 4, "expanded context regs present in preamble and body");
}

void TestIndirectShComputeBit() {
	Pm4Translator t;
	std::vector<uint32_t> pairs = {0x20Cu, 0xdeadu, 0x20Du, 0x0012u}; // COMPUTE_PGM_LO/HI
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_SH_REG_INDIRECT, pairs);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.loaded_sh_regs == 2, "indirect sh expanded");
	Check(r.shader_addrs.size() == 1, "compute shader VA collected");
	auto pkts = Decode(r.ib);
	for (const auto& p: pkts) {
		if (p.op == IT_SET_SH_REG && !p.body.empty() && p.body[0] >= 0x200u) {
			Check(p.compute, "compute SH register uses the shader-type bit");
		}
	}
}

void TestUconfigWhitelist() {
	Pm4Translator t;
	// 0x243 (VGT_INDEX_TYPE) is allowed; 0x300 is not.
	std::vector<uint32_t> pairs = {0x243u, 0x2u, 0x300u, 0x9u};
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_UCONFIG_REG_INDIRECT, pairs);
	auto r = t.Translate(dcb);
	Check(r.ok, "indirect uc translates");
	Check(r.stats.loaded_uc_regs == 1, "only the whitelisted uc reg kept");
	Check(r.stats.uconfig_skipped == 1, "non-whitelisted uc reg skipped");
	auto pkts = Decode(r.ib);
	bool index_type = false;
	for (const auto& p: pkts) {
		Check(p.op != IT_SET_UCONFIG_REG || (!p.body.empty() && p.body[0] != 0x300u),
		      "non-whitelisted uc reg emitted");
		index_type = index_type || (p.op == IT_SET_UCONFIG_REG_INDEX && p.body[0] == 0x20000243u);
	}
	Check(index_type, "VGT_INDEX_TYPE emitted with SET_UCONFIG_REG_INDEX index 2");
	Check(Pm4Translator::UconfigAllowed(0x24Au) && Pm4Translator::UconfigAllowed(0x24Bu) &&
	          Pm4Translator::UconfigAllowed(0x25Fu),
	      "per-draw GE registers (index offset, primitive restart, stereo) allowed");
	Check(!Pm4Translator::UconfigAllowed(0x342u), "unknown UCONFIG 0x342 stays dropped");
}

// GFX9+ only updates the CP's copy of VGT_PRIMITIVE_TYPE / VGT_INDEX_TYPE through
// SET_UCONFIG_REG_INDEX (index 1 / 2), in the body and in the preamble of the next IB alike.
void TestUconfigIndexedRegisters() {
	Pm4Translator         t;
	std::vector<uint32_t> pairs = {0x242u, 0x4u, 0x243u, 0x1u, 0x25Bu, 0x8040u};
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_UCONFIG_REG_INDIRECT, pairs);
	Append(dcb, IT_SET_UCONFIG_REG, {0x242u, 0x5u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.loaded_uc_regs == 3, "indirect uc regs expanded");
	uint32_t prim_idx1 = 0, type_idx2 = 0, ge_plain = 0, prim_plain = 0;
	for (const auto& p: Decode(r.ib)) {
		if (p.op == IT_SET_UCONFIG_REG_INDEX && p.body[0] == 0x10000242u) {
			prim_idx1++;
		}
		if (p.op == IT_SET_UCONFIG_REG_INDEX && p.body[0] == 0x20000243u) {
			type_idx2++;
		}
		if (p.op == IT_SET_UCONFIG_REG && p.body[0] == 0x25Bu) {
			ge_plain++;
		}
		if (p.op == IT_SET_UCONFIG_REG && p.body[0] == 0x242u) {
			prim_plain++;
		}
	}
	// The preamble replays the committed shadow too, so each register shows up there and in the body.
	Check(prim_idx1 == 3 && prim_plain == 0, "VGT_PRIMITIVE_TYPE must use index 1, also when the guest wrote it plain");
	Check(type_idx2 == 2, "VGT_INDEX_TYPE must use index 2");
	Check(ge_plain == 2, "GE_CNTL stays a plain SET_UCONFIG_REG");

	// The next IB's preamble replays the shadow with the same encodings.
	std::vector<uint32_t> next;
	Append(next, IT_NOP, {0u}, false);
	auto r2 = t.Translate(next);
	uint32_t pre_prim = 0, pre_type = 0;
	for (const auto& p: Decode(r2.ib)) {
		pre_prim += (p.op == IT_SET_UCONFIG_REG_INDEX && p.body[0] == 0x10000242u && p.body[1] == 0x5u) ? 1 : 0;
		pre_type += (p.op == IT_SET_UCONFIG_REG_INDEX && p.body[0] == 0x20000243u && p.body[1] == 0x1u) ? 1 : 0;
	}
	Check(pre_prim == 1 && pre_type == 1, "preamble replays indexed uc regs with their last values");
}

void TestPs5Defaults() {
	Pm4Translator t;
	t.SeedPs5Defaults();
	std::vector<uint32_t> dcb;
	// The guest's own value wins over a seeded default.
	Append(dcb, IT_SET_CONTEXT_REG, {0x202u, 0x00cc0000u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.preamble_cx == 5, "five seeded defaults in the preamble");
	bool vte = false, cb_guest = false;
	for (const auto& p: Decode(r.ib)) {
		vte = vte || (p.op == IT_SET_CONTEXT_REG && p.body[0] == 0x206u && p.body[1] == 0x43fu);
		cb_guest = cb_guest || (p.op == IT_SET_CONTEXT_REG && p.body[0] == 0x202u && p.body[1] == 0x00cc0000u);
	}
	Check(vte, "PA_CL_VTE_CNTL seeded with the PS5 viewport-transform default");
	Check(cb_guest, "guest CB_COLOR_CONTROL overrides the seeded default");
}

void TestWaitRegMemDropped() {
	alignas(8) static volatile uint64_t label = 5;
	const auto addr = reinterpret_cast<uint64_t>(&label);
	Pm4Translator t;
	std::vector<uint32_t> dcb;
	// WAIT_REG_MEM_64, memory space, function ">=": met (5 >= 5) and unmet (5 >= 6).
	Append(dcb, 0x93u, {0x15u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 5u, 0u,
	                    0xffffffffu, 0xffffffffu, 10u}, false);
	Append(dcb, 0x93u, {0x15u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 6u, 0u,
	                    0xffffffffu, 0xffffffffu, 10u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.waits_dropped == 2 && r.stats.waits_unmet == 1, "waits counted and dropped");
	for (const auto& p: Decode(r.ib)) {
		Check(p.op != 0x93u, "WAIT_REG_MEM_64 must not reach the shadow IB");
	}
	Check(r.stats.pfp_syncs_emitted == 0 && Count(Decode(r.ib), 0x42u) == 0, "ME waits leave no PFP_SYNC_ME");

	// A PFP wait (ENGINE_SEL bit 8, as Dead Cells does after each cache flush) becomes PFP_SYNC_ME.
	std::vector<uint32_t> pfp;
	Append(pfp, 0x93u, {0x113u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 0u,
	                    0xffffffffu, 0xffffffffu, 10u}, false);
	auto rp = t.Translate(pfp);
	Check(rp.ok && rp.stats.pfp_syncs_emitted == 1 && Count(Decode(rp.ib), 0x42u) == 1,
	      "a dropped PFP wait becomes PFP_SYNC_ME");

	std::vector<uint32_t> reg_wait;
	Append(reg_wait, 0x93u, {0x05u, 0x1234u, 0u, 0u, 0u, 0u, 0u, 10u}, false);
	Check(!t.Translate(reg_wait).ok, "register wait refused");
}

void TestDmaData() {
	Pm4Translator t;
	std::vector<uint32_t> dcb;
	// memory -> memory (SRC_SEL 0, DST_SEL 0, SAS/DAS clear), 256 bytes.
	Append(dcb, 0x50u, {0x0u, 0x1000u, 0x2u, 0x2000u, 0x2u, 256u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.dma_data == 1, "memory DMA_DATA passed through");
	std::vector<uint32_t> to_reg;
	Append(to_reg, 0x50u, {0x0u, 0x1000u, 0x2u, 0x2c00u, 0x0u, 4u | (1u << 27u)}, false);
	Check(!t.Translate(to_reg).ok, "DMA_DATA into the register space refused");
	std::vector<uint32_t> gds;
	Append(gds, 0x50u, {1u << 20u, 0x1000u, 0x2u, 0x0u, 0x0u, 4u}, false);
	Check(!t.Translate(gds).ok, "DMA_DATA into GDS refused without a GDS partition");
	t.SetGdsAvailable(true);
	auto rg = t.Translate(gds);
	Check(rg.ok && rg.stats.dma_data == 1, "DMA_DATA into GDS passed through with a GDS partition");
	for (const auto& ref: rg.refs) {
		Check(ref.addr != 0x0u, "a GDS destination is not a memory reference");
	}
}

void TestWriteData() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	Append(dcb, IT_WRITE_DATA, {(5u << 8u) | (1u << 20u), 0x1000u, 0x2u, 0xabcdu}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.write_data_dropped == 1, "WRITE_DATA to memory accepted");
	const auto pkts = Decode(r.ib);
	Check(FindFirst(pkts, IT_WRITE_DATA) == nullptr, "WRITE_DATA left to Kyty, not in the native IB");

	std::vector<uint32_t> to_reg;
	Append(to_reg, IT_WRITE_DATA, {0u, 0x2c00u, 0x0u, 0x1u}, false);
	auto r2 = t.Translate(to_reg);
	Check(!r2.ok && r2.unknown_op == IT_WRITE_DATA, "WRITE_DATA to a register refused");
}

void TestUconfigIndexPassthrough() {
	Pm4Translator t;
	std::vector<uint32_t> dcb;
	// SET_UCONFIG_REG_INDEX VGT_INDEX_TYPE with index 2 in the high nibble.
	Append(dcb, IT_SET_UCONFIG_REG_INDEX, {0x20000243u, 0x0u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.set_uconfig_index == 1, "uconfig index passed through");
	auto pkts = Decode(r.ib);
	const Packet* p = FindFirst(pkts, IT_SET_UCONFIG_REG_INDEX);
	Check(p != nullptr && p->body[0] == 0x20000243u, "index bits preserved on passthrough");
}

void TestPassthroughPackets() {
	Pm4Translator t;
	std::vector<uint32_t> dcb;
	Append(dcb, IT_ACQUIRE_MEM, {0, 0, 0, 0, 0, 0}, false);
	Append(dcb, IT_DISPATCH_DIRECT, {1, 1, 1, 0}, true);
	Append(dcb, IT_EVENT_WRITE, {0, 0}, false);
	Append(dcb, IT_CLEAR_STATE, {0}, false);
	Append(dcb, IT_NUM_INSTANCES, {4}, false);
	Append(dcb, 0x26u /* IT_INDEX_BASE */, {0x10000000u, 0x2u}, false);
	Append(dcb, 0x13u /* IT_INDEX_BUFFER_SIZE */, {600u}, false);
	Append(dcb, 0x11u /* IT_SET_BASE */, {1u, 0x20000000u, 0x2u}, false);
	Append(dcb, 0x16u /* IT_DISPATCH_INDIRECT */, {0x40u, 1u}, true);
	Append(dcb, 0x42u /* IT_PFP_SYNC_ME */, {0u}, false);
	Append(dcb, 0x35u /* IT_DRAW_INDEX_OFFSET_2 */, {600u, 0u, 300u, 0u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok, "passthrough DCB translates");
	Check(r.stats.acquire_mem == 1 && r.stats.dispatches == 2 && r.stats.event_write == 1 &&
	          r.stats.clear_state == 1 && r.stats.num_instances == 1 && r.stats.index_state == 2 &&
	          r.stats.set_base == 1 && r.stats.pfp_sync_me == 1 && r.stats.draws == 1,
	      "passthrough packets counted");
	auto pkts = Decode(r.ib);
	Check(Count(pkts, IT_ACQUIRE_MEM) == 1 && Count(pkts, IT_DISPATCH_DIRECT) == 1 &&
	          Count(pkts, IT_NUM_INSTANCES) == 1 && FindFirst(pkts, IT_NUM_INSTANCES)->body[0] == 4,
	      "present in IB");
	bool index_ref = false;
	for (const auto& ref: r.refs) {
		index_ref = index_ref || (ref.addr == 0x210000000ull && ref.size == 2400u);
	}
	Check(index_ref, "INDEX_BASE + INDEX_BUFFER_SIZE reported as a 2400-byte reference");
	bool base_ref = false;
	for (const auto& ref: r.refs) {
		base_ref = base_ref || ref.addr == 0x220000000ull;
	}
	Check(base_ref, "SET_BASE address reported as a reference");
}

void TestUnknownRefused() {
	Pm4Translator t;
	// Establish a shadow first.
	std::vector<uint32_t> dcb0;
	Append(dcb0, IT_SET_CONTEXT_REG, {0x7u, 0x1234u}, false);
	Check(t.Translate(dcb0).ok, "setup DCB ok");

	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x9u, 0x5u}, false);
	Append(dcb, 0x50u /* IT_DMA_DATA, not handled */, {0, 0, 0, 0, 0}, false);
	auto r = t.Translate(dcb);
	Check(!r.ok, "unknown opcode refused");
	Check(r.unknown_op == 0x50u, "reports the offending opcode");
	Check(r.ib.empty(), "no IB on refusal");

	// The refusal must not corrupt the shadow: the next DCB's preamble still has only reg 0x7.
	std::vector<uint32_t> dcb2;
	Append(dcb2, IT_DRAW_INDEX_2, {0, 0, 0, 0}, false);
	auto r2 = t.Translate(dcb2);
	Check(r2.ok && r2.stats.preamble_cx == 1, "shadow intact after refusal");

	// Every unknown opcode of a refused DCB is reported, not just the first.
	std::vector<uint32_t> many;
	Append(many, 0xA1u, {0}, false);
	Append(many, IT_SET_CONTEXT_REG, {0x9u, 0x5u}, false);
	Append(many, 0xB2u, {0, 0}, false);
	Append(many, 0xA1u, {0}, false);
	auto r3 = t.Translate(many);
	Check(!r3.ok && r3.unknown_op == 0xA1u && r3.unknown_dw == 0, "first unknown opcode and offset reported");
	Check(r3.unknown_ops.size() == 2 && r3.unknown_ops.count(0xB2u) == 1, "all unknown opcodes collected");
	Check(t.Translate(dcb2).stats.preamble_cx == 1, "shadow intact after a multi-unknown refusal");
}

} // namespace

bool HasRef(const std::vector<GpuRef>& refs, uint64_t addr, uint64_t min_size, bool pointer) {
	for (const auto& r: refs) {
		if (r.addr == addr && r.size >= min_size && r.pointer == pointer) {
			return true;
		}
	}
	return false;
}

void TestDrawIndexOffset() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_UCONFIG_REG, {0x243u, 1u}, false);                // VGT_INDEX_TYPE = 32-bit
	Append(dcb, 0x26u /* IT_INDEX_BASE */, {0x10000000u, 0x2u}, false); // 0x2_1000_0000
	Append(dcb, 0x35u /* IT_DRAW_INDEX_OFFSET_2 */, {0x33cu, 0x75cu, 0x33cu, 0u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.draws == 1, "DRAW_INDEX_OFFSET_2 translated");
	const auto pkts = Decode(r.ib);
	Check(Count(pkts, 0x35u) == 0, "DRAW_INDEX_OFFSET_2 does not reach the IB");
	const Packet* d = FindFirst(pkts, IT_DRAW_INDEX_2);
	const uint64_t addr = 0x210000000ull + 0x75cull * 4u;
	Check(d != nullptr && d->body.size() == 5 && d->body[0] == 0x33cu && d->body[1] == static_cast<uint32_t>(addr) &&
	          d->body[2] == static_cast<uint32_t>(addr >> 32u) && d->body[3] == 0x33cu,
	      "DRAW_INDEX_OFFSET_2 becomes DRAW_INDEX_2 at INDEX_BASE + offset * index size");
}

void TestFmaskFix() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	// MRT0 and MRT4 bound and writable; MRT4 has no FMASK (0), MRT0 already points it at its base.
	Append(dcb, IT_SET_CONTEXT_REG, {0x08Eu, 0xf000fu}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x50fc300u}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x321u, 0x50fc300u}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u + 4u * 0xFu, 0x5125700u}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x321u + 4u * 0xFu, 0x0u}, false);
	Append(dcb, 0x2Du /* IT_DRAW_INDEX_AUTO */, {3u, 2u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.fmask_fixed == 1, "one target without FMASK fixed");
	uint32_t fmask4 = 0;
	size_t   fix_at = 0, draw_at = 0, i = 0;
	for (const auto& p: Decode(r.ib)) {
		if (p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2 && p.body[0] == 0x321u + 4u * 0xFu) {
			fmask4 = p.body[1];
			fix_at = i;
		}
		if (p.op == 0x2Du) {
			draw_at = i;
		}
		i++;
	}
	Check(fmask4 == 0x5125700u && fix_at < draw_at, "CB_COLOR4_FMASK = CB_COLOR4_BASE before the draw");
}

void TestContextDefaults() {
	Pm4Translator t;
	t.SeedPs5Defaults();
	const std::pair<uint32_t, uint32_t> defaults[] = {{0x206u, 0x0u}, {0x2DEu, 0x1E9u}, {0x2F9u, 0x2Du}};
	t.SeedContextDefaults(defaults);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x2F9u, 0x1u}, false); // the guest's own value replaces a default
	auto r = t.Translate(dcb);
	bool vte = false, seeded = false, guest = false;
	for (const auto& p: Decode(r.ib)) {
		if (p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2) {
			vte    = vte || (p.body[0] == 0x206u && p.body[1] == 0x43Fu);
			seeded = seeded || (p.body[0] == 0x2DEu && p.body[1] == 0x1E9u);
			guest  = guest || (p.body[0] == 0x2F9u && p.body[1] == 0x1u);
		}
	}
	Check(vte, "library defaults do not override SeedPs5Defaults");
	Check(seeded, "a seeded default reaches the preamble");
	Check(guest, "a guest write replaces the seeded default");
	Check(r.stats.preamble_cx == 7, "five SeedPs5Defaults plus two library defaults in the preamble");
}

void TestGpuRefs() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	// CB_COLOR0 at 0x2_0000_0000 (BASE = addr >> 8, BASE_EXT = 0), 1920x1080 via ATTRIB2.
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x02000000u}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x3B0u, ((1920u - 1u) << 14u) | (1080u - 1u)}, false);
	// A user-data pair holding the 48-bit pointer 0x2_1234_5600 (SPI_SHADER_USER_DATA_PS_0/1).
	Append(dcb, IT_SET_SH_REG, {0x0Cu, 0x12345600u, 0x00000002u}, false);
	// DRAW_INDEX_2: MAX_SIZE 300 indices at 0x2_4000_0000.
	Append(dcb, IT_DRAW_INDEX_2, {300u, 0x40000000u, 0x2u, 300u, 0u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok, "refs DCB translated");
	Check(HasRef(r.refs, 0x200000000ull, 1920ull * 1080ull * 16ull, false), "render target ref sized from ATTRIB2");
	Check(HasRef(r.refs, 0x212345600ull, 0, true), "user-data pair reported as a pointer candidate");
	Check(HasRef(r.refs, 0x240000000ull, 1200, false), "index buffer ref from DRAW_INDEX_2");
	Check(std::is_sorted(r.refs.begin(), r.refs.end()), "refs sorted");
	Check(r.render_targets.empty(), "a target with INFO.FORMAT 0 is not reported as bound");

	// With a format, the target is reported with its size and the draws that used it.
	std::vector<uint32_t> bound;
	Append(bound, IT_SET_CONTEXT_REG, {0x31Cu, 0xAu << 2u}, false); // CB_COLOR0_INFO.FORMAT = 10
	Append(bound, IT_DRAW_INDEX_2, {3u, 0x40000000u, 0x2u, 3u, 0u}, false);
	Append(bound, IT_DRAW_INDEX_2, {3u, 0x40000000u, 0x2u, 3u, 0u}, false);
	auto rb = t.Translate(bound);
	Check(rb.render_targets.size() == 1 && rb.render_targets[0].addr == 0x200000000ull &&
	          rb.render_targets[0].width == 1920 && rb.render_targets[0].height == 1080 &&
	          rb.render_targets[0].draws == 2,
	      "bound render target reported with dimensions and draw count");

	// A second DCB with no register change still reports the live render target (state carries over),
	// and only once per DCB.
	std::vector<uint32_t> again;
	Append(again, IT_DRAW_INDEX_2, {3u, 0x40000000u, 0x2u, 3u, 0u}, false);
	Append(again, IT_DRAW_INDEX_2, {3u, 0x40000000u, 0x2u, 3u, 0u}, false);
	auto r2 = t.Translate(again);
	size_t rt = 0;
	for (const auto& ref: r2.refs) {
		rt += ref.addr == 0x200000000ull && !ref.pointer ? 1u : 0u;
	}
	Check(rt == 1, "carried-over render target reported once");
}

int main() {
	TestNopsOnly();
	TestContextShadowAcrossDcbs();
	TestShaderDetection();
	TestIndirectContextExpansion();
	TestIndirectShComputeBit();
	TestUconfigWhitelist();
	TestUconfigIndexPassthrough();
	TestUconfigIndexedRegisters();
	TestWriteData();
	TestPs5Defaults();
	TestWaitRegMemDropped();
	TestDmaData();
	TestPassthroughPackets();
	TestUnknownRefused();
	TestGpuRefs();
	TestContextDefaults();
	TestDrawIndexOffset();
	TestFmaskFix();
	std::puts("NativeGpuPm4TranslatorTests: all passed");
	return 0;
}
