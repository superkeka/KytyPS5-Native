#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

using Libs::Graphics::NativeGpu::GpuRef;
using Libs::Graphics::NativeGpu::LabelOracle;
using Libs::Graphics::NativeGpu::Pm4Segment;
using Libs::Graphics::NativeGpu::TrackerPoint;
using Libs::Graphics::NativeGpu::WaitCond;
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

void TestComputeQueueDispatchIndirect() {
	Pm4Translator t;
	t.SetComputeQueue(true);
	std::vector<uint32_t> acb;
	Append(acb, 0x16u /* IT_DISPATCH_INDIRECT */, {0x074063c0u, 0x5u, 0x41u}, false); // ADDR_LO, ADDR_HI, initiator
	auto r = t.Translate(acb);
	Check(r.ok && r.stats.dispatches == 1, "compute-queue DISPATCH_INDIRECT translated");
	const auto     pkts = Decode(r.ib);
	const Packet*  base = FindFirst(pkts, 0x11u);
	const Packet*  disp = FindFirst(pkts, 0x16u);
	Check(base != nullptr && base->body.size() == 3 && base->body[0] == 1u && base->body[1] == 0x074063c0u &&
	          base->body[2] == 0x5u,
	      "SET_BASE(1) points at the argument address");
	Check(disp != nullptr && disp->body.size() == 2 && disp->body[0] == 0u && disp->body[1] == 0x41u,
	      "DISPATCH_INDIRECT at offset 0 keeps the initiator");
}

void TestIndirectDraws() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	Append(dcb, 0x11u /* IT_SET_BASE */, {1u, 0x10000000u, 0x2u}, false);
	Append(dcb, 0x25u /* IT_DRAW_INDEX_INDIRECT */, {0x40u, 0u, 0u, 0u}, false);
	Append(dcb, 0x8Eu /* IT_GET_LOD_STATS */, {0u, 0x10000000u, 0x2u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.draws == 1 && r.stats.lod_stats_dropped == 1, "DRAW_INDEX_INDIRECT and GET_LOD_STATS translated");
	const auto pkts = Decode(r.ib);
	Check(Count(pkts, 0x25u) == 1 && Count(pkts, 0x8Eu) == 0, "indirect draw kept, LOD statistics dropped");
}

void TestContextStatePushPop() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	auto marker = [&](uint32_t operation) {
		// Kyty's R_CONTEXT_STATE NOP: type-3 NOP whose header bits 7:2 carry the marker.
		dcb.push_back((3u << 30u) | (1u << 16u) | (0x10u << 8u) | (0x1Au << 2u));
		dcb.push_back(operation);
		dcb.push_back(0u);
	};
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x1111u}, false); // CB_COLOR0_BASE before the push
	marker(1);                                                // push
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x2222u}, false); // an inner pass changes it
	marker(2);                                                // pop
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.context_ops == 2, "context push/pop markers applied");
	uint32_t last = 0;
	for (const auto& p: Decode(r.ib)) {
		if (p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2 && p.body[0] == 0x318u) {
			last = p.body[1];
		}
	}
	Check(last == 0x1111u, "the pop restores the pushed CB_COLOR0_BASE");
}

void TestWaitBarrier() {
	Pm4Translator t;
	t.SetFenceAddress(0x2000001000ull);
	static uint32_t label = 0; // unmet: the guest waits for 1
	std::vector<uint32_t> dcb;
	const uint64_t        addr = reinterpret_cast<uint64_t>(&label);
	Append(dcb, 0x3Cu /* IT_WAIT_REG_MEM */,
	       {3u | (1u << 4u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 0xffffffffu, 4u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.wait_barriers == 1 && r.stats.waits_unmet == 1, "guest wait becomes a barrier");
	const auto    pkts = Decode(r.ib);
	const Packet* rel  = FindFirst(pkts, 0x49u);
	const Packet* wait = FindFirst(pkts, 0x3Cu);
	Check(rel != nullptr && rel->body.size() == 7 && rel->body[2] == 0x1000u && rel->body[3] == 0x20u && rel->body[4] == 1u,
	      "RELEASE_MEM writes fence 1 to the fence address");
	Check(wait != nullptr && wait->body.size() == 6 && wait->body[1] == 0x1000u && wait->body[3] == 1u,
	      "WAIT_REG_MEM waits for that fence");
}

void TestCompactRefs() {
	Pm4Translator t;
	t.SetCompactRefs(true);
	std::vector<uint32_t> dcb;
	// CB_COLOR0 at 0x2_0100_0000, 1920x1080: one 16 MiB bucket (0x201).
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x02010000u}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x3B0u, ((1920u - 1u) << 14u) | (1080u - 1u)}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x08Eu, 0xfu}, false);
	Append(dcb, 0x2Du /* IT_DRAW_INDEX_AUTO */, {3u, 2u}, false);
	auto r = t.Translate(dcb);
	bool found = false;
	for (uint32_t b: r.ref_buckets) {
		found = found || b == 0x201u;
	}
	Check(r.ok && found && r.refs.empty(), "compact refs report the render target's bucket, no GpuRef list");
}

void TestGdsReleaseMem() {
	Pm4Translator t;
	t.SetGdsAvailable(true);
	std::vector<uint32_t> dcb;
	// Kyty's R_RELEASE_MEM marker: NOP header + hardware RELEASE_MEM body, DATA_SEL 5 (GDS), INT_SEL 2.
	dcb.push_back((3u << 30u) | (6u << 16u) | (0x10u << 8u) | (0x18u << 2u));
	dcb.push_back(0x2fu | (6u << 8u));
	dcb.push_back((5u << 29u) | (2u << 24u));
	dcb.push_back(0x00001000u);
	dcb.push_back(0x2u);
	dcb.push_back(0xc64u | (1u << 16u));
	dcb.push_back(0u);
	dcb.push_back(0x55u);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.gds_releases == 1, "GDS release marker translated");
	const auto    pkts = Decode(r.ib);
	const Packet* rel  = FindFirst(pkts, 0x49u);
	Check(rel != nullptr && rel->body.size() == 7 && ((rel->body[1] >> 24u) & 0x7u) == 0u &&
	          ((rel->body[1] >> 29u) & 0x7u) == 5u && rel->body[4] == (0xc64u | (1u << 16u)) && rel->body[6] == 0u,
	      "native RELEASE_MEM sends the GDS dword, without the interrupt");
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


// Native labels: a backend stand-in that answers every wait the same way.
class TestOracle: public LabelOracle {
public:
	bool GpuWritable(uint64_t addr) override { return writable && addr != unwritable; }
	Plan PlanWait(const WaitCond& wait) override {
		asked++;
		last = wait;
		return plan;
	}
	bool     writable   = true;
	uint64_t unwritable = 0;
	Plan     plan       = Plan::Split;
	uint32_t asked      = 0;
	WaitCond last;
};

constexpr uint64_t kTrackerVa = 0x400000fff410ull;
constexpr uint64_t kFenceVa   = 0x400000fff400ull;

// The barrier's RELEASE_MEM event (CACHE_FLUSH_AND_INV_TS, end of pipe, full GCR): the only form
// native labels use.
constexpr uint32_t kFlushEvent = 0x14u | (5u << 8u) | (1u << 12u) | (1u << 13u) | (1u << 14u) | (1u << 15u) | (1u << 20u) | (1u << 21u);

// The tracker RELEASE_MEM a point's sequence value sits in.
bool IsTrackerPacket(const std::vector<uint32_t>& ib, uint32_t dw) {
	return dw >= 5 && dw + 2 < ib.size() && ib[dw - 5] == 0xc0064900u && ib[dw - 4] == kFlushEvent && ib[dw - 3] == (1u << 29u) &&
	       ib[dw - 2] == static_cast<uint32_t>(kTrackerVa) && ib[dw - 1] == static_cast<uint32_t>(kTrackerVa >> 32u);
}

// A label release in the barrier's form writing `value` at `addr`.
bool HasLabelRelease(const std::vector<Packet>& pkts, uint64_t addr, uint32_t value) {
	for (const auto& p: pkts) {
		if (p.op == 0x49u && p.body[0] == kFlushEvent && p.body[1] == (1u << 29u) && p.body[2] == static_cast<uint32_t>(addr) &&
		    p.body[3] == static_cast<uint32_t>(addr >> 32u) && p.body[4] == value) {
			return true;
		}
	}
	return false;
}

// Kyty's R_RELEASE_MEM marker around a hardware RELEASE_MEM body.
void AppendReleaseMarker(std::vector<uint32_t>& dcb, uint32_t data_sel, uint32_t int_sel, uint64_t dst, uint64_t value,
                         uint32_t ctx) {
	dcb.push_back((3u << 30u) | (6u << 16u) | (IT_NOP << 8u) | (0x18u << 2u));
	dcb.push_back(0x28u | (5u << 8u));
	dcb.push_back((data_sel << 29u) | (int_sel << 24u));
	dcb.push_back(static_cast<uint32_t>(dst));
	dcb.push_back(static_cast<uint32_t>(dst >> 32u));
	dcb.push_back(static_cast<uint32_t>(value));
	dcb.push_back(static_cast<uint32_t>(value >> 32u));
	dcb.push_back(ctx);
}

void AppendWait(std::vector<uint32_t>& dcb, uint64_t addr, uint32_t ref) {
	Append(dcb, 0x3Cu, {3u | (1u << 4u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), ref, 0xffffffffu, 4u},
	       false);
}

void TestLabelsWriteData() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(dcb, IT_WRITE_DATA, {5u << 8u, 0x02000100u, 0x2u, 7u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 1 && !r.segments[0].held, "one segment, not held");
	const auto pkts = Decode(r.ib);
	Check(FindFirst(pkts, IT_WRITE_DATA) == nullptr && HasLabelRelease(pkts, 0x202000100ull, 7u),
	      "the label WRITE_DATA runs on the GPU as a release in the barrier's form");
	const auto& s = r.segments[0];
	Check(s.points.size() == 1 && s.points[0].kind == TrackerPoint::Kind::End && IsTrackerPacket(r.ib, s.points[0].dw),
	      "the segment ends with a tracker point");
	Check(s.writes.size() == 1 && s.writes[0].addr == 0x202000100ull && s.writes[0].value == 7u && s.writes[0].gpu,
	      "the write is reported as a GPU label write");
	Check(s.begin == 0 && s.end == r.ib.size() && IbAligned(r.ib), "the segment spans the aligned IB");
	Check(r.stats.labels_native == 1 && r.cpu_tail.empty(), "native label counted, nothing left for Kyty");

	// Memory the GPU cannot write: the tracker writes it.
	o.unwritable = 0x202000200ull;
	std::vector<uint32_t> cpu;
	Append(cpu, IT_WRITE_DATA, {5u << 8u, 0x02000200u, 0x2u, 9u}, false);
	auto rc = t.Translate(cpu);
	Check(FindFirst(Decode(rc.ib), IT_WRITE_DATA) == nullptr, "an unwritable label is not written by the GPU");
	const auto& sc = rc.segments[0];
	Check(sc.points.size() == 2 && sc.points[0].kind == TrackerPoint::Kind::CpuWrite && sc.points[0].addr == 0x202000200ull &&
	          sc.points[0].data.size() == 1 && sc.points[0].data[0] == 9u && IsTrackerPacket(rc.ib, sc.points[0].dw) &&
	          sc.points[1].dw == sc.points[0].dw,
	      "the tracker writes it at its point, which the end shares");
	Check(sc.writes.size() == 1 && !sc.writes[0].gpu, "reported as a CPU write");
}

void TestLabelsReleaseMem() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	std::vector<uint32_t> dcb;
	AppendReleaseMarker(dcb, 1, 2, 0x202000300ull, 5, 0x1234);
	AppendReleaseMarker(dcb, 2, 1, 0x202000308ull, 6, 0x55); // gfx INT_SEL 1: Kyty writes no data
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 1, "release markers translated");
	std::vector<const Packet*> rels;
	const auto pkts = Decode(r.ib);
	for (const auto& p: pkts) {
		if (p.op == 0x49u && p.body[2] != static_cast<uint32_t>(kTrackerVa)) {
			rels.push_back(&p);
		}
	}
	Check(rels.size() == 1 && HasLabelRelease(pkts, 0x202000300ull, 5u),
	      "the first writes its 32-bit label in the barrier's form; INT_SEL 1 on the graphics queue writes nothing");
	const auto& pts = r.segments[0].points;
	Check(pts.size() == 3 && pts[0].kind == TrackerPoint::Kind::Interrupt && pts[0].ctx == 0x1234 &&
	          pts[1].kind == TrackerPoint::Kind::Interrupt && pts[1].ctx == 0x55 && pts[2].kind == TrackerPoint::Kind::End,
	      "each interrupt is a tracker point, in order, before the end");
	Check(pts[1].dw == pts[0].dw && pts[2].dw == pts[0].dw, "points with nothing between them share one packet");

	// A 64-bit label: two 32-bit releases, the high dword first.
	std::vector<uint32_t> wide;
	AppendReleaseMarker(wide, 2, 0, 0x202000320ull, 0x0000000900000008ull, 0);
	auto       rw    = t.Translate(wide);
	const auto wpkts = Decode(rw.ib);
	Check(HasLabelRelease(wpkts, 0x202000324ull, 9u) && HasLabelRelease(wpkts, 0x202000320ull, 8u), "64-bit label as two halves");
	Check(r.stats.interrupts == 2 && r.stats.labels_native == 1, "stats");

	// A timestamp goes through the tracker (Kyty writes its reference clock).
	std::vector<uint32_t> ts;
	AppendReleaseMarker(ts, 3, 0, 0x202000310ull, 0, 0);
	Append(ts, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(ts, 1, 2, 0x202000318ull, 1, 7);
	auto        rt  = t.Translate(ts);
	const auto& tsp = rt.segments[0].points;
	Check(tsp.size() == 3 && tsp[0].kind == TrackerPoint::Kind::Timestamp && tsp[0].addr == 0x202000310ull &&
	          tsp[1].kind == TrackerPoint::Kind::Interrupt && tsp[0].dw == tsp[1].dw && tsp[2].kind == TrackerPoint::Kind::End,
	      "a timestamp is written by the tracker at the next point, without a packet of its own");
}

void TestLabelsLocalWait() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	static uint32_t label = 3; // stale, the guest waits for 1
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 1, 0, addr, 1, 0);
	AppendWait(dcb, addr, 1);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 1 && o.asked == 0, "a wait on a label written earlier in the buffer is decided locally");
	const auto    pkts = Decode(r.ib);
	const Packet* wait = FindFirst(pkts, 0x3Cu);
	Check(wait != nullptr && wait->body[1] == static_cast<uint32_t>(addr) && r.stats.waits_native == 1 && r.stats.wait_barriers == 0,
	      "and passes through natively");

	// The same wait with the label reset to 0 after the release: nobody satisfies it, so the IB is cut.
	std::vector<uint32_t> reset;
	Append(reset, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(reset, 1, 0, addr, 1, 0);
	Append(reset, IT_WRITE_DATA, {5u << 8u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 0u}, false);
	AppendWait(reset, addr, 1);
	auto rr = t.Translate(reset);
	Check(rr.segments.size() == 2 && rr.segments[1].held && o.asked == 0, "a write that does not satisfy the wait cuts the IB");
}

void TestLabelsSplit() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	static uint32_t label = 0;
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_SH_REG, {0x240u, 0x11u}, true);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(dcb, IT_SET_SH_REG, {0x240u, 0x22u}, true);
	AppendWait(dcb, addr, 1);
	Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 2 && o.asked == 1 && o.last.addr == addr && o.last.ref == 1u, "the oracle cuts the IB");
	const Pm4Segment& a = r.segments[0];
	const Pm4Segment& b = r.segments[1];
	Check(!a.held && b.held && a.end == b.begin && b.end == r.ib.size() && (b.begin & 7u) == 0, "two aligned segments");
	Check(a.points.size() == 1 && a.points[0].kind == TrackerPoint::Kind::End && b.points.size() == 1 &&
	          b.points[0].kind == TrackerPoint::Kind::End && IsTrackerPacket(r.ib, a.points[0].dw) &&
	          IsTrackerPacket(r.ib, b.points[0].dw) && a.points[0].dw < a.end && b.points[0].dw >= b.begin,
	      "each segment ends with its own tracker point");
	std::vector<uint32_t> second(r.ib.begin() + b.begin, r.ib.begin() + b.end);
	const auto            pkts = Decode(second);
	CheckPreambleHead(pkts);
	bool user_data = false;
	for (const auto& p: pkts) {
		user_data = user_data || (p.op == IT_SET_SH_REG && p.body[0] == 0x240u && p.body[1] == 0x22u);
	}
	Check(user_data, "the second segment's preamble holds the state at the cut");
	Check(b.barrier_len == 15 && ((r.ib[b.barrier_dw] >> 8u) & 0xffu) == 0x49u, "the barrier block comes first");
	Check(b.wait_len == 7 && b.wait_dw == b.barrier_dw + b.barrier_len && ((r.ib[b.wait_dw] >> 8u) & 0xffu) == 0x3Cu &&
	          r.ib[b.wait_dw + 2] == static_cast<uint32_t>(addr),
	      "then the guest wait");
	Check(Count(Decode(std::vector<uint32_t>(r.ib.begin() + b.wait_dw + b.wait_len, r.ib.begin() + b.end)), IT_DISPATCH_DIRECT) == 1,
	      "and the rest of the buffer");

	// NopOut turns either one into filler NOPs.
	Libs::Graphics::NativeGpu::NopOut(r.ib, b.barrier_dw, b.barrier_len);
	bool filler = true;
	for (uint32_t i = 0; i < b.barrier_len; i++) {
		filler = filler && r.ib[b.barrier_dw + i] == 0xffff1000u;
	}
	Check(filler, "fillers in place of the barrier");

	// A wait before any work: the empty first segment is left out.
	std::vector<uint32_t> first;
	Append(first, IT_SET_SH_REG, {0x240u, 0x33u}, true);
	AppendWait(first, addr, 1);
	Append(first, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	auto rf = t.Translate(first);
	Check(rf.segments.size() == 1 && rf.segments[0].held && rf.segments[0].begin == 0, "a buffer that starts with the wait is held whole");

	// Met: a barrier, as without native labels.
	o.plan = LabelOracle::Plan::Met;
	auto rm = t.Translate(first);
	Check(rm.segments.size() == 1 && !rm.segments[0].held && rm.stats.wait_barriers == 1, "a met wait is a barrier");
	// Native: passed through.
	o.plan  = LabelOracle::Plan::Native;
	auto rn = t.Translate(first);
	Check(rn.segments.size() == 1 && rn.stats.waits_native == 1 && rn.stats.wait_barriers == 0, "a native wait passes through");
	t.SetNativeWaits(false);
	auto rb = t.Translate(first);
	Check(rb.stats.waits_native == 0 && rb.stats.wait_barriers == 1, "KYTY_NATIVE_GPU_NATIVE_WAITS=0: a barrier instead");
}

void TestLabelsCpuTail() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_NOP, {0u, 1u, 2u, 3u, 4u}, false, R_FLIP);
	Append(dcb, IT_NOP, {0x68750778u, 0x02000400u, 0x2u, 1u}, false);
	Append(dcb, IT_NOP, {0x68750004u}, false); // a user-data marker: Kyty's own state, not needed
	auto r = t.Translate(dcb);
	Check(r.ok && r.cpu_tail.size() == 6 + 5 && r.cpu_tail[0] == dcb[0] && r.cpu_tail[6] == dcb[6],
	      "flips are copied for Kyty's command processor");
	Check(r.segments.size() == 1 && r.segments[0].writes.size() == 1 && !r.segments[0].writes[0].gpu &&
	          r.segments[0].writes[0].addr == 0x202000400ull,
	      "the flip's label is a CPU write");
}

void TestLabelWriteCovers() {
	using Libs::Graphics::NativeGpu::LabelWrite;
	uint64_t   v = 0;
	LabelWrite w32 {0x1000, 5, true, true, false};
	LabelWrite w64 {0x2000, 0x0000000700000006ull, true, true, true};
	Check(w32.Covers(0x1000, false, &v) == 1 && v == 5, "32-bit write, 32-bit wait");
	Check(w32.Covers(0x1000, true, &v) == 2, "32-bit write, 64-bit wait: unknown");
	Check(w64.Covers(0x2000, false, &v) == 1 && v == 6, "64-bit write, low half");
	Check(w64.Covers(0x2004, false, &v) == 1 && v == 7, "64-bit write, high half");
	Check(w64.Covers(0x2000, true, &v) == 1 && v == 0x0000000700000006ull, "64-bit write, 64-bit wait");
	Check(w32.Covers(0x1008, false, &v) == 0, "unrelated");
}

void TestLabelsWait64() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	static uint64_t label = 0;
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	auto wait64 = [&](std::vector<uint32_t>& dcb, uint32_t mask_hi) {
		Append(dcb, 0x93u, {3u | (1u << 4u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 0u, 0xffffffffu, mask_hi, 4u},
		       false);
	};
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	wait64(dcb, 0);
	o.plan = LabelOracle::Plan::Native;
	auto r = t.Translate(dcb);
	const auto    pkts = Decode(r.ib);
	const Packet* w32  = FindFirst(pkts, 0x3Cu);
	Check(r.segments.size() == 1 && FindFirst(pkts, 0x93u) == nullptr && w32 != nullptr && w32->body[1] == static_cast<uint32_t>(addr) &&
	          w32->body[3] == 1u && w32->body[4] == 0xffffffffu && w32->body[5] == 4u,
	      "a native 64-bit wait with zero high halves becomes WAIT_REG_MEM (the gfx ring has no WAIT_REG_MEM_64)");

	std::vector<uint32_t> both;
	Append(both, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	wait64(both, 1);
	auto rb = t.Translate(both);
	Check(rb.segments.size() == 2 && rb.segments[1].held && rb.segments[1].wait_len == 0 && FindFirst(Decode(rb.ib), 0x93u) == nullptr,
	      "one on both halves is cut and has no native form");
	o.plan  = LabelOracle::Plan::Split;
	auto rs = t.Translate(dcb);
	Check(rs.segments.size() == 2 && rs.segments[1].wait_len == 7, "a held 64-bit wait carries its 32-bit form");
}

void TestValidateNativeIb() {
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	// Everything the native-labels translator emits passes.
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	static uint64_t label = 0; // read as 64 bits by the WAIT_REG_MEM_64 too
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_SH_REG, {0x240u, 0x11u}, true);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(dcb, IT_WRITE_DATA, {(2u << 8u) | (1u << 20u), 0x02000100u, 0x2u, 7u}, false);
	AppendReleaseMarker(dcb, 1, 2, 0x202000300ull, 5, 0x1234);
	AppendReleaseMarker(dcb, 2, 0, 0x202000308ull, 6, 0);
	dcb.push_back((3u << 30u) | (6u << 16u) | (IT_NOP << 8u) | (0x18u << 2u)); // a cache flush alone
	dcb.push_back(0x30e62fu);
	dcb.push_back(1u << 16u);
	for (int i = 0; i < 5; i++) {
		dcb.push_back(0u);
	}
	AppendWait(dcb, addr, 1);
	Append(dcb, 0x93u, {3u | (1u << 4u) | (1u << 8u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 0u, 0xffffffffu, 0u, 0x19u},
	       false);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	for (auto plan: {LabelOracle::Plan::Split, LabelOracle::Plan::Native, LabelOracle::Plan::Met}) {
		o.plan = plan;
		auto r = t.Translate(dcb);
		Check(r.ok && ValidateNativeIb(r.ib, false), "the native-labels IB passes the packet check");
		Check(FindFirst(Decode(r.ib), IT_WRITE_DATA) == nullptr && FindFirst(Decode(r.ib), 0x93u) == nullptr, "no guest forms in it");
	}

	auto rejects = [&](std::vector<uint32_t> ib, const char* what) {
		while ((ib.size() & 7u) != 0) {
			ib.push_back(0xffff1000u);
		}
		uint32_t bad = ~0u;
		Check(!ValidateNativeIb(ib, false, &bad) && bad == 0u, what);
	};
	rejects({0xc0079300u, 0x13u, 0u, 0u, 1u, 0u, ~0u, 0u, 4u}, "WAIT_REG_MEM_64 rejected");
	rejects({0xc0033700u, 0x100200u, 0x1000u, 0u, 1u}, "WRITE_DATA rejected");
	rejects({0xc0064902u, kFlushEvent, 1u << 29u, 0x1000u, 0u, 1u, 0u, 0u}, "RELEASE_MEM with the compute bit rejected");
	rejects({0xc0064900u, 0x528u, 2u << 29u, 0x1000u, 0u, 1u, 0u, 0u}, "a guest-form RELEASE_MEM rejected");
	rejects({0xc0064900u, kFlushEvent, (1u << 29u) | (2u << 24u), 0x1000u, 0u, 1u, 0u, 0u}, "a RELEASE_MEM interrupt rejected");
	rejects({0xc0053c00u, 0x4000013u, 0x1000u, 0u, 1u, ~0u, 0x19u}, "a guest-form WAIT_REG_MEM rejected");
	rejects({0xc0053c02u, 0x13u, 0x1000u, 0u, 1u, ~0u, 4u}, "WAIT_REG_MEM with the compute bit rejected");
	rejects({0xc0011000u, 0u, 0u}, "a NOP that is not a filler rejected");
	rejects({0xc0028e00u, 64u, 0x1000u, 0u}, "GET_LOD_STATS rejected unless enabled");
	rejects({0x12345678u}, "not a type-3 packet");
	std::vector<uint32_t> lod = {0xc0028e00u, 64u, 0x1000u, 0u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u};
	Check(ValidateNativeIb(lod, true), "GET_LOD_STATS passes when enabled");
	std::vector<uint32_t> gds = {0xc0064900u, 0x2fu | (6u << 8u), 5u << 29u, 0x1000u, 2u, 0xc64u, 0u, 0u};
	Check(ValidateNativeIb(gds, false), "a GDS release as always run passes");
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
	TestComputeQueueDispatchIndirect();
	TestIndirectDraws();
	TestContextStatePushPop();
	TestWaitBarrier();
	TestCompactRefs();
	TestGdsReleaseMem();
	TestLabelsWriteData();
	TestLabelsReleaseMem();
	TestLabelsLocalWait();
	TestLabelsSplit();
	TestLabelsCpuTail();
	TestLabelWriteCovers();
	TestLabelsWait64();
	TestValidateNativeIb();
	std::puts("NativeGpuPm4TranslatorTests: all passed");
	return 0;
}
