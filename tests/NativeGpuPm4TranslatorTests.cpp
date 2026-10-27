#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

using Libs::Graphics::NativeGpu::GpuRef;
using Libs::Graphics::NativeGpu::LabelOracle;
using Libs::Graphics::NativeGpu::Pm4Segment;
using Libs::Graphics::NativeGpu::TrackerPoint;
using Libs::Graphics::NativeGpu::WaitCond;
using Libs::Graphics::NativeGpu::LodStatsHandoff;
using Libs::Graphics::NativeGpu::LodStatsMode;
using Libs::Graphics::NativeGpu::ParseLodStatsMode;
using Libs::Graphics::NativeGpu::Pm4TranslateResult;
using Libs::Graphics::NativeGpu::ComputeRingForQueue;
using Libs::Graphics::NativeGpu::FlattenIndirectBuffers;
using Libs::Graphics::NativeGpu::NativeIbKind;
using Libs::Graphics::NativeGpu::PlanRingWait;
using Libs::Graphics::NativeGpu::Pm4Translator;
using Libs::Graphics::NativeGpu::RingWaitRelease;
using Libs::Graphics::NativeGpu::WriteLodUsageReport;

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
constexpr uint32_t IT_DRAW_INDEX_AUTO          = 0x2d;
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
// data_format: dword 3 of the packet; libSceAgc (agc.cpp) writes 0x80000000 (offset and data pairs).
void AppendIndirect(std::vector<uint32_t>& dcb, uint32_t op, const std::vector<uint32_t>& pairs,
                    uint32_t data_format = 0u) {
	const auto     addr = reinterpret_cast<uint64_t>(pairs.data());
	const uint32_t num  = static_cast<uint32_t>(pairs.size() / 2);
	std::vector<uint32_t> body = {static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), data_format,
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

// Register tables in libSceAgc's form reach the gfx CP as written when every pair would be emitted
// unchanged; the shadow copy still takes their values. A stub entry, a patched value or
// VGT_PRIMITIVE_TYPE keeps the expansion.
void TestIndirectTablesVerbatim() {
	Pm4Translator         t;
	std::vector<uint32_t> cx_pairs = {0x10u, 0x1111u, 0x20u, 0x2222u};
	std::vector<uint32_t> sh_pairs = {0x0Cu, 0x5555u}; // SPI_SHADER_USER_DATA_PS_0
	std::vector<uint32_t> uc_pairs = {0x25Bu, 0x8040u};
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, cx_pairs, 0x80000000u);
	AppendIndirect(dcb, IT_SET_SH_REG_INDIRECT, sh_pairs, 0x80000000u);
	AppendIndirect(dcb, IT_SET_UCONFIG_REG_INDIRECT, uc_pairs, 0x80000000u);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.loads_verbatim == 3, "three tables run as written");
	Check(std::search(r.ib.begin(), r.ib.end(), dcb.begin(), dcb.end()) != r.ib.end(), "the packets are in the IB unchanged");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "a gfx IB with tables passes the check");
	// The shadow took the values: the next buffer's preamble replays them.
	std::vector<uint32_t> next;
	Append(next, IT_SET_CONTEXT_REG, {0x30u, 0x3u}, false);
	auto     r2   = t.Translate(next);
	uint32_t seen = 0;
	for (const auto& p: Decode(r2.ib)) {
		seen += p.op == IT_SET_CONTEXT_REG && p.body.size() == 2 && p.body[0] == 0x10u && p.body[1] == 0x1111u ? 1u : 0u;
	}
	Check(r2.ok && seen == 1, "table values replayed from the shadow");

	std::vector<uint32_t> stub_pairs = {0xffffffffu, 0u, 0x25Bu, 0x8040u};
	std::vector<uint32_t> prim_pairs = {0x242u, 0x4u};
	std::vector<uint32_t> held;
	AppendIndirect(held, IT_SET_UCONFIG_REG_INDIRECT, stub_pairs, 0x80000000u);
	AppendIndirect(held, IT_SET_UCONFIG_REG_INDIRECT, prim_pairs, 0x80000000u);
	AppendIndirect(held, IT_SET_CONTEXT_REG_INDIRECT, cx_pairs); // another data format
	auto r3 = t.Translate(held);
	Check(r3.ok && r3.stats.loads_verbatim == 0 && r3.stats.loaded_uc_regs == 2 && r3.stats.loaded_cx_regs == 2,
	      "stub entries, VGT_PRIMITIVE_TYPE and other data formats are expanded");
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

void TestComputeCuClamp() {
	Pm4Translator t;
	t.SetComputeCuClamp({0x003f003fu, 0x003f003fu, 0u, 0u});
	std::vector<uint32_t> dcb;
	// COMPUTE_STATIC_THREAD_MGMT_SE0, SE1 and the reserved word between SE1 and SE2, all ones.
	Append(dcb, IT_SET_SH_REG, {0x216u, 0xffffffffu, 0xffffffffu, 0xffffffffu}, true);
	auto r = t.Translate(dcb);
	Check(r.ok, "clamped DCB translates");
	bool seen = false;
	for (const auto& p: Decode(r.ib)) {
		if (p.op == IT_SET_SH_REG && p.body.size() == 4 && p.body[0] == 0x216u) {
			seen = true;
			Check(p.body[1] == 0x003f003fu && p.body[2] == 0x003f003fu && p.body[3] == 0xffffffffu,
			      "CU masks ANDed with the clamp, other registers untouched");
		}
	}
	Check(seen, "the SET_SH_REG passes through");
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

// libSceAgc's internal-data tags (SET_UCONFIG_REG of 0x342, header bit 2 set in DCBs) reach the gfx
// CP as written; the other trace registers stay dropped.
void TestAgcTagsVerbatim() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb = {0xc0017904u, 0x342u, 0xc8010000u, 0xc0017904u, 0x342u, 0xc8000000u};
	Append(dcb, IT_SET_UCONFIG_REG, {0x343u, 0x22u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.uconfig_skipped == 1 && r.skipped_uconfig.count(0x343u) == 1, "0x343 still dropped on gfx");
	Check(r.ib.size() >= 6 && std::search(r.ib.begin(), r.ib.end(), dcb.begin(), dcb.begin() + 6) != r.ib.end(),
	      "the two 0x342 tags are in the gfx IB as written");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "a tagged IB passes the gfx-ring check");
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
	t.SetGdsDmaSync(true);
	std::vector<uint32_t> from_gds;
	Append(from_gds, 0x50u, {1u << 29u, 0x0u, 0x0u, 0x1000u, 0x2u, 4u}, false);
	const auto ps = Decode(t.Translate(from_gds).ib);
	const Packet* ev  = FindFirst(ps, 0x46u);
	const Packet* dma = FindFirst(ps, 0x50u);
	Check(ev != nullptr && ev->body.size() == 1 && ev->body[0] == ((4u << 8u) | 7u) && dma != nullptr &&
	          (dma->body[0] >> 31u) == 1u,
	      "a copy out of GDS gets CS_PARTIAL_FLUSH and CP_SYNC");
	const auto pg = Decode(t.Translate(gds).ib);
	Check(FindFirst(pg, 0x46u) == nullptr && (FindFirst(pg, 0x50u)->body[0] >> 31u) == 1u,
	      "a copy into GDS gets CP_SYNC only");
	t.SetGdsDmaSync(false);
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

	std::vector<uint32_t> data;
	Append(data, IT_WRITE_DATA, {(5u << 8u) | (1u << 20u), 0x1000u, 0x2u, 0x1u, 0x2u, 0x3u}, false);
	auto rd = t.Translate(data);
	Check(rd.ok && rd.data_writes.size() == 1 && rd.data_writes[0].second == 3, "a multi-dword WRITE_DATA reported");
	Check(FindFirst(Decode(rd.ib), IT_WRITE_DATA) == nullptr, "multi-dword WRITE_DATA left to Kyty by default");
	t.SetWriteDataNative(1);
	Check(FindFirst(Decode(t.Translate(data).ib), IT_WRITE_DATA) != nullptr, "mode 1 keeps a multi-dword WRITE_DATA");
	Check(FindFirst(Decode(t.Translate(dcb).ib), IT_WRITE_DATA) == nullptr, "mode 1 still leaves a label to Kyty");
	t.SetWriteDataNative(2);
	Check(FindFirst(Decode(t.Translate(dcb).ib), IT_WRITE_DATA) != nullptr, "mode 2 keeps every WRITE_DATA");
	t.SetWriteDataNative(0);

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

void TestPs5RegOverrides() {
	auto shader_control = [](const std::vector<Packet>& pkts, uint32_t& value) {
		bool found = false;
		for (const auto& p: pkts) {
			if (p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2 && p.body[0] == 0x310u) {
				value = p.body[1];
				found = true;
			}
		}
		return found;
	};
	Pm4Translator t;
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x310u, 0x1u}, false);
	uint32_t v = 0;
	Check(shader_control(Decode(t.Translate(dcb).ib), v) && v == 0x1u, "PA_SC_SHADER_CONTROL as written by default");
	t.SetPs5RegOverrides(true);
	Check(shader_control(Decode(t.Translate(dcb).ib), v) && v == 0x5u, "LOAD_COLLISION_WAVEID forced like the PS5 CAM");
	Pm4Translator fresh;
	fresh.SetPs5RegOverrides(true);
	std::vector<uint32_t> nops;
	Append(nops, IT_NOP, {0x0u}, false);
	Check(shader_control(Decode(fresh.Translate(nops).ib), v) && v == 0x4u,
	      "the preamble sets LOAD_COLLISION_WAVEID before the game writes the register");
}

void TestAcquireFull() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	Append(dcb, 0x58u, {0x0u, 0x1000u, 0x0u, 0x40000u, 0x0u, 0xau, 0x4000u}, false);
	const Packet* a = nullptr;
	auto          p0 = Decode(t.Translate(dcb).ib);
	a                = FindFirst(p0, 0x58u);
	Check(a != nullptr && a->body[6] == 0x4000u && a->body[1] == 0x1000u, "ACQUIRE_MEM passes through by default");
	t.SetAcquireFull(true);
	auto p1 = Decode(t.Translate(dcb).ib);
	a       = FindFirst(p1, 0x58u);
	Check(a != nullptr && a->body[1] == 0xffffffffu && a->body[2] == 0xffu && a->body[3] == 0u &&
	          (a->body[6] & 0xc3f1u) == 0xc3f1u,
	      "full ACQUIRE_MEM covers all memory and every cache level");
	bool cb = false, db = false;
	for (const auto& p: p1) {
		cb = cb || (p.op == 0x46u && p.body.size() == 1 && p.body[0] == 0x2eu);
		db = db || (p.op == 0x46u && p.body.size() == 1 && p.body[0] == 0x2cu);
	}
	Check(cb && db, "the gfx ring flushes CB and DB metadata before a full ACQUIRE_MEM");
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
// The tracker's stores: bottom of pipe, GLM and GL2 written back, nothing invalidated.
constexpr uint32_t kTrackerEvent = 0x28u | (5u << 8u) | (1u << 12u) | (1u << 21u);
// A label release: the guest's event and cache actions plus GLM and GL2 write-back (LabelEventCntl),
// or the barrier's form.
bool IsLabelEvent(uint32_t event) {
	constexpr uint32_t wb = (1u << 12u) | (1u << 21u);
	return event == kFlushEvent || (event & wb) == wb;
}

// The tracker RELEASE_MEM a point's sequence value sits in.
bool IsTrackerPacket(const std::vector<uint32_t>& ib, uint32_t dw) {
	return dw >= 5 && dw + 2 < ib.size() && ib[dw - 5] == 0xc0064900u && ib[dw - 4] == kTrackerEvent && ib[dw - 3] == (1u << 29u) &&
	       ib[dw - 2] == static_cast<uint32_t>(kTrackerVa) && ib[dw - 1] == static_cast<uint32_t>(kTrackerVa >> 32u);
}

// A label release writing `value` at `addr`.
bool HasLabelRelease(const std::vector<Packet>& pkts, uint64_t addr, uint32_t value) {
	for (const auto& p: pkts) {
		if (p.op == 0x49u && IsLabelEvent(p.body[0]) && p.body[1] == (1u << 29u) && p.body[2] == static_cast<uint32_t>(addr) &&
		    p.body[3] == static_cast<uint32_t>(addr >> 32u) && p.body[4] == value) {
			return true;
		}
	}
	return false;
}

// RELEASE_MEM DATA_SEL 3: the GPU clock, data dwords zero.
bool HasGpuTimestamp(const std::vector<Packet>& pkts, uint64_t addr) {
	for (const auto& p: pkts) {
		if (p.op == 0x49u && p.body.size() >= 7 && p.body[0] == kFlushEvent && p.body[1] == (3u << 29u) &&
		    p.body[2] == static_cast<uint32_t>(addr) && p.body[3] == static_cast<uint32_t>(addr >> 32u) && p.body[4] == 0u &&
		    p.body[5] == 0u && p.body[6] == 0u) {
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

// SetDirectRuns: a long stretch passed on as written becomes an IB2 into the guest's buffer, split
// where the buffer's guest memory is not contiguous; short stretches are copied; tracker points after
// a run keep pointing at their packets.
void TestDirectRuns() {
	std::vector<uint32_t> dcb;
	for (uint32_t i = 0; i < 100; i++) {
		Append(dcb, IT_SET_CONTEXT_REG, {0x10u, 0x1000u + i}, false);
	}
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	const uint32_t run_dw = static_cast<uint32_t>(dcb.size()); // the dispatch is passed on as written too
	constexpr uint64_t kVa = 0x123400000ull;
	auto has_ib2 = [](const std::vector<uint32_t>& ib, uint64_t va, uint32_t dwords) {
		for (size_t k = 0; k + 3 < ib.size(); k++) {
			if (ib[k] == 0xc0023f00u && ib[k + 1] == static_cast<uint32_t>(va) && ib[k + 2] == static_cast<uint32_t>(va >> 32u) &&
			    ib[k + 3] == (dwords | (1u << 23u))) {
				return true;
			}
		}
		return false;
	};
	{
		Pm4Translator t;
		t.SetDirectRuns(64);
		t.SetSource({{0u, static_cast<uint32_t>(dcb.size()), kVa}});
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.direct_ib2 == 1 && r.stats.direct_dw == run_dw, "one IB2 for the 305-dword stretch");
		Check(has_ib2(r.ib, kVa, run_dw), "the IB2 points at the stretch in guest memory");
		Check(r.ib.size() < dcb.size(), "the stretch is not copied");
		uint32_t bad = 0;
		Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "an IB with IB2s passes the gfx check");
		Check(!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Compute), "not on a compute ring");
		// SetSource is for one buffer: the next one is copied.
		auto r2 = t.Translate(dcb);
		Check(r2.ok && r2.stats.direct_ib2 == 0, "without a source the next buffer is copied");
	}
	{
		Pm4Translator t;
		t.SetDirectRuns(1000);
		t.SetSource({{0u, static_cast<uint32_t>(dcb.size()), kVa}});
		Check(t.Translate(dcb).stats.direct_ib2 == 0, "a stretch under the minimum is copied");
	}
	{
		// Two pieces: dwords [0, 150) at kVa, the rest at another address.
		Pm4Translator t;
		t.SetDirectRuns(64);
		t.SetSource({{0u, 150u, kVa}, {150u, static_cast<uint32_t>(dcb.size()) - 150u, 0x200000000ull}});
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.direct_ib2 == 2 && has_ib2(r.ib, kVa, 150u) && has_ib2(r.ib, 0x200000000ull, run_dw - 150u),
		      "the stretch is split where its guest memory is");
	}
	{
		// Native labels: the segment's tracker store after the run still holds its points.
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetDirectRuns(64);
		t.SetSource({{0u, static_cast<uint32_t>(dcb.size()), kVa}});
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.direct_ib2 == 1 && r.segments.size() == 1, "one segment with one IB2");
		const auto& pts = r.segments[0].points;
		Check(!pts.empty() && IsTrackerPacket(r.ib, pts.back().dw) && r.segments[0].end == r.ib.size() && IbAligned(r.ib),
		      "the end point moves with the IB");
	}
}

// A WRITE_DATA of a block of data is one write: a wait on a dword inside it that the block meets runs
// on the GPU (the write is ahead of it in the ring); one it does not meet is not; the segment keeps the
// block as touched, without the buffer's memory.
void TestLabelsWriteDataBlock() {
	alignas(8) static uint32_t mem[8] = {};
	const uint64_t             addr   = reinterpret_cast<uint64_t>(&mem[0]);
	for (uint32_t ref: {7u, 9u}) {
		std::vector<uint32_t> dcb;
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		Append(dcb, IT_WRITE_DATA,
		       {5u << 8u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 2u, 7u, 4u}, false);
		AppendWait(dcb, addr + 8u, ref);
		Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
		Pm4Translator t;
		TestOracle    o;
		o.plan = LabelOracle::Plan::Native;
		t.SetNativeLabels(&o, kTrackerVa);
		auto r = t.Translate(dcb);
		Check(r.ok && o.asked == 0, "the block decides the wait inside it, the backend is not asked");
		if (ref == 7u) {
			Check(r.stats.waits_native == 1, "a wait the block meets runs on the GPU");
		} else {
			Check(r.stats.waits_native == 0, "a wait the block does not meet stays off the GPU");
		}
		Check(r.stats.write_data_dw == 4, "four data dwords counted");
		const auto& w = r.segments[0].writes;
		Check(w.size() == 1 && w[0].addr == addr && w[0].span_dw == 4 && w[0].data == nullptr,
		      "one block write, kept without the buffer's memory");
		uint64_t v = 0;
		Check(w[0].Covers(addr + 8u, false, &v) == 2 && w[0].Covers(addr + 16u, false, &v) == 0,
		      "afterwards the block is touched but unknown, and ends where it ends");
	}
}

void TestDisplayDccRestore() {
	// A draw into a video-out buffer loses DCC; a later target with the same format that has no DCC
	// of its own keeps its INFO as the guest wrote it.
	Pm4Translator t;
	t.AddDisplayRange(0x200000000ull, 0x2000000ull);
	constexpr uint32_t kInfoDcc = 0x10008024u;
	constexpr uint32_t kInfo    = 0x00008024u;
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x02000000u}, false); // CB_COLOR0_BASE: the display buffer
	Append(dcb, IT_SET_CONTEXT_REG, {0x31cu, kInfoDcc}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x08eu, 0xfu}, false);
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 2u}, false);
	Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x122f7a00u}, false); // another target, no DCC
	Append(dcb, IT_SET_CONTEXT_REG, {0x31cu, kInfo}, false);
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 2u}, false);
	const auto r = t.Translate(dcb);
	// The INFO values sent after the display draw (the guest's own DCC write for the display target
	// passes through before it).
	std::vector<uint32_t> infos;
	bool                  drawn = false;
	for (const auto& p: Decode(r.ib)) {
		drawn = drawn || p.op == IT_DRAW_INDEX_AUTO;
		if (drawn && p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2 && (p.body[0] & 0xffffu) == 0x31cu) {
			infos.push_back(p.body[1]);
		}
	}
	Check(r.ok && !infos.empty() && infos.back() == kInfo, "the second target's own INFO is kept without DCC");
	Check(std::find(infos.begin(), infos.end(), kInfoDcc) == infos.end(), "DCC is never turned on for it");
}

void TestSetPredication() {
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	Pm4Translator t;
	// The gfx10 form passes as it is; the older one (address low, flags with address bits 39:32) is
	// rewritten into it.
	std::vector<uint32_t> dcb;
	Append(dcb, 0x20u, {(2u << 16u) | (1u << 8u), 0x00345670u, 0x12u}, false);
	Append(dcb, 0x20u, {0x00345680u, (3u << 16u) | 0x12u}, false);
	const auto r = t.Translate(dcb);
	std::vector<const Packet*> preds;
	const auto                 pkts = Decode(r.ib);
	for (const auto& p: pkts) {
		if (p.op == 0x20u) {
			preds.push_back(&p);
		}
	}
	Check(r.ok && preds.size() == 2, "SET_PREDICATION translated");
	Check(preds[0]->body == std::vector<uint32_t>({(2u << 16u) | (1u << 8u), 0x00345670u, 0x12u}), "gfx10 form kept");
	Check(preds[1]->body == std::vector<uint32_t>({3u << 16u, 0x00345680u, 0x12u}), "older form rewritten");
	Check(ValidateNativeIb(r.ib, false), "it passes the gfx packet check");
	Pm4Translator c;
	c.SetComputeRing(true);
	std::vector<uint32_t> acb;
	Append(acb, 0x20u, {2u << 16u, 0x1000u, 0u}, true);
	Check(!c.Translate(acb).ok, "not on a compute ring");
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
	const auto  pkts = Decode(r.ib);
	const auto* wd   = FindFirst(pkts, IT_WRITE_DATA);
	Check(wd != nullptr && wd->body.size() == 4 && wd->body[1] == 0x02000100u && wd->body[3] == 7u &&
	          (wd->body[0] & (1u << 20u)) != 0,
	      "the label WRITE_DATA runs on the GPU as the guest's WRITE_DATA, in order, write confirmed");
	const auto& s = r.segments[0];
	Check(s.points.size() == 2 && s.points[0].kind == TrackerPoint::Kind::Wake && s.points[0].addr == 0x202000100ull &&
	          s.points[0].data.empty() && s.points[1].kind == TrackerPoint::Kind::End && s.points[0].dw == s.points[1].dw &&
	          IsTrackerPacket(r.ib, s.points[0].dw),
	      "the GPU label only wakes its sleepers at the segment's tracker point; the CPU stores nothing");
	Check(s.writes.size() == 1 && s.writes[0].addr == 0x202000100ull && s.writes[0].value == 7u && s.writes[0].gpu,
	      "the write is reported as a GPU label write");
	Check(s.begin == 0 && s.end == r.ib.size() && IbAligned(r.ib), "the segment spans the aligned IB");
	Check(r.stats.labels_native == 1 && r.stats.labels_cpu == 0 && r.cpu_tail.empty(),
	      "native label counted, nothing left for Kyty or the tracker");

	// A long WRITE_DATA (a dispatch's constants) stays one in-order GPU write, and on a compute ring
	// it is a compute packet on the micro engine.
	{
		Pm4Translator tc;
		TestOracle    oc;
		tc.SetNativeLabels(&oc, kTrackerVa);
		tc.SetComputeRing(true);
		std::vector<uint32_t> body = {(5u << 8u) | (1u << 30u), 0x02000400u, 0x2u};
		for (uint32_t i = 0; i < 40; i++) {
			body.push_back(i);
		}
		std::vector<uint32_t> acb;
		Append(acb, IT_WRITE_DATA, body, true);
		Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
		const auto rl    = tc.Translate(acb);
		const auto pl    = Decode(rl.ib);
		const auto* lw   = FindFirst(pl, IT_WRITE_DATA);
		const auto* disp = FindFirst(pl, IT_DISPATCH_DIRECT);
		Check(rl.ok && lw != nullptr && lw->compute && lw->body.size() == 43 && lw->body[42] == 39u &&
		          (lw->body[0] >> 30u) == 0u && disp != nullptr && lw < disp,
		      "a 40-dword WRITE_DATA on a compute ring is one ME WRITE_DATA ahead of the dispatch");
	}

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
	Check(pts.size() == 4 && pts[0].kind == TrackerPoint::Kind::Wake && pts[0].addr == 0x202000300ull && pts[0].data.empty() &&
	          pts[1].kind == TrackerPoint::Kind::Interrupt && pts[1].ctx == 0x1234 &&
	          pts[2].kind == TrackerPoint::Kind::Interrupt && pts[2].ctx == 0x55 && pts[3].kind == TrackerPoint::Kind::End,
	      "the 32-bit label's sleepers are woken (no CPU store), then each interrupt, then the end");
	Check(pts[1].dw == pts[0].dw && pts[2].dw == pts[0].dw && pts[3].dw == pts[0].dw,
	      "points with nothing between them share one packet");

	// A 64-bit label: two 32-bit releases, the high dword first.
	std::vector<uint32_t> wide;
	AppendReleaseMarker(wide, 2, 0, 0x202000320ull, 0x0000000900000008ull, 0);
	auto       rw    = t.Translate(wide);
	const auto wpkts = Decode(rw.ib);
	Check(HasLabelRelease(wpkts, 0x202000324ull, 9u) && HasLabelRelease(wpkts, 0x202000320ull, 8u), "64-bit label as two halves");
	const auto& wpts = rw.segments[0].points;
	Check(wpts.size() == 2 && wpts[0].kind == TrackerPoint::Kind::Wake && wpts[0].addr == 0x202000320ull &&
	          wpts[1].kind == TrackerPoint::Kind::End,
	      "the 64-bit label is the GPU's alone; the tracker only wakes it");
	Check(r.stats.interrupts == 2 && r.stats.labels_native == 1 && r.stats.labels_cpu == 0 && rw.stats.labels_cpu == 0, "stats");

	// A timestamp the GPU can reach is a DATA_SEL 3 release, not a tracker write. The interrupt
	// after it still shares the segment's one tracker packet with the end.
	std::vector<uint32_t> ts;
	AppendReleaseMarker(ts, 3, 0, 0x202000310ull, 0, 0);
	Append(ts, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(ts, 1, 2, 0x202000318ull, 1, 7);
	auto        rt  = t.Translate(ts);
	const auto& tsp = rt.segments[0].points;
	const auto  tpk = Decode(rt.ib);
	Check(rt.ok && rt.stats.timestamps_native == 1 && rt.stats.timestamps == 0 && HasGpuTimestamp(tpk, 0x202000310ull),
	      "a reachable timestamp is a GPU clock release");
	Check(tsp.size() == 3 && tsp[0].kind == TrackerPoint::Kind::Wake && tsp[0].addr == 0x202000318ull &&
	          tsp[1].kind == TrackerPoint::Kind::Interrupt && tsp[1].ctx == 7 && tsp[2].kind == TrackerPoint::Kind::End &&
	          tsp[0].dw == tsp[2].dw,
	      "the label's wake and the interrupt share the segment's one tracker packet with the end");
	Check(HasRef(rt.refs, 0x202000310ull, 8, false), "the timestamp address is made resident");
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
	Check(HasRef(r.refs, addr, 4, false), "the GPU label is a residency reference, so its block is promoted");

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
	// A 64-bit wait that masks off the high half reads only the low one.
	Check(w32.Covers(0x1000, true, &v, 0xffffffffull) == 1 && v == 5, "32-bit write, 64-bit wait on the low half");
	const LabelWrite high {0x1004, 9, true, true, false};
	Check(high.Covers(0x1000, true, &v, 0xffffffffull) == 0, "a write to the masked-off high half does not decide it");
	Check(high.Covers(0x1000, true, &v) == 2, "a write to the high half of a full 64-bit wait: unknown");
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
	auto       rb    = t.Translate(both);
	const auto pkts2 = Decode(rb.ib);
	uint32_t   halves = 0;
	for (const auto& p: pkts2) {
		if (p.op == 0x3Cu && (p.body[1] == static_cast<uint32_t>(addr) || p.body[1] == static_cast<uint32_t>(addr + 4u))) {
			halves++;
		}
	}
	Check(rb.segments.size() == 1 && FindFirst(pkts2, 0x93u) == nullptr && halves == 2,
	      "one on both halves becomes a WAIT_REG_MEM per half");
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
		const auto pkts = Decode(r.ib);
		const auto* wd   = FindFirst(pkts, IT_WRITE_DATA);
		Check(wd == nullptr || (wd->body[0] & (1u << 20u)) != 0, "a WRITE_DATA in it is write-confirmed");
		Check(FindFirst(pkts, 0x93u) == nullptr, "no 64-bit guest wait in it");
	}

	auto rejects = [&](std::vector<uint32_t> ib, const char* what) {
		while ((ib.size() & 7u) != 0) {
			ib.push_back(0xffff1000u);
		}
		uint32_t bad = ~0u;
		Check(!ValidateNativeIb(ib, false, &bad) && bad == 0u, what);
	};
	rejects({0xc0079300u, 0x13u, 0u, 0u, 1u, 0u, ~0u, 0u, 4u}, "WAIT_REG_MEM_64 rejected");
	rejects({0xc0033700u, 0x200u, 0x1000u, 0u, 1u}, "WRITE_DATA without write confirm rejected");
	rejects({0xc0033700u, 0x100000u, 0x1000u, 0u, 1u}, "WRITE_DATA to a register rejected");
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
	std::vector<uint32_t> wd = {0xc0033700u, 0x100500u, 0x1000u, 0u, 1u, 0xffff1000u, 0xffff1000u, 0xffff1000u};
	Check(ValidateNativeIb(wd, false), "a write-confirmed WRITE_DATA to memory passes");
	uint32_t bad = ~0u;
	Check(!ValidateNativeIb(wd, false, &bad, NativeIbKind::Compute), "on a compute ring it needs the compute bit");
	wd[0] = 0xc0033702u;
	Check(ValidateNativeIb(wd, false, &bad, NativeIbKind::Compute), "with it, it passes there");
}

void TestComputeRingPackets() {
	Pm4Translator t;
	TestOracle    o;
	t.SetComputeQueue(true);
	t.SetComputeRing(true);
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	o.plan = LabelOracle::Plan::Native;
	static uint32_t label = 0;
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	std::vector<uint32_t> acb;
	Append(acb, IT_SET_SH_REG, {0x240u, 0x11u}, true);
	Append(acb, 0x16u /* IT_DISPATCH_INDIRECT */, {0x074063c0u, 0x5u, 0x41u}, true);
	Append(acb, IT_WRITE_DATA, {5u << 8u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u}, true);
	AppendWait(acb, addr, 1);
	auto r = t.Translate(acb);
	Check(r.ok && r.stats.dispatches == 1, "compute-ring ACB translated");
	const auto pkts = Decode(r.ib);
	Check(FindFirst(pkts, IT_CONTEXT_CONTROL) == nullptr && FindFirst(pkts, IT_CLEAR_STATE) == nullptr,
	      "a compute ring gets no graphics preamble");
	Check(FindFirst(pkts, 0x42u) == nullptr, "no PFP_SYNC_ME on a compute ring");
	Check(FindFirst(pkts, 0x11u) == nullptr, "DISPATCH_INDIRECT is not rewritten through SET_BASE");
	const Packet* disp = FindFirst(pkts, 0x16u);
	Check(disp != nullptr && disp->compute && disp->body.size() == 3 && disp->body[0] == 0x074063c0u && disp->body[1] == 0x5u &&
	          disp->body[2] == 0x41u,
	      "the guest DISPATCH_INDIRECT is kept, with the compute shader bit");
	const Packet* sh = nullptr;
	for (const auto& p: pkts) {
		if (p.op == IT_SET_SH_REG && p.body.size() >= 2 && p.body[0] == 0x240u) {
			sh = &p;
		}
	}
	Check(sh != nullptr && sh->compute && sh->body[1] == 0x11u, "compute SH state is replayed with the shader bit");
	bool release_cs = false;
	bool wait_cs    = false;
	for (const auto& p: pkts) {
		release_cs = release_cs || (p.op == 0x49u && p.compute && IsLabelEvent(p.body[0]));
		wait_cs    = wait_cs || (p.op == 0x3Cu && p.compute && p.body[1] == static_cast<uint32_t>(addr));
	}
	Check(release_cs && wait_cs && r.stats.waits_native == 1, "label release and the guest wait use the compute shader bit");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Compute), "the IB passes the compute-ring check");
	Check(!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "a compute-bit release is not a gfx-ring packet");

	std::vector<uint32_t> draw;
	Append(draw, IT_DRAW_INDEX_2, {3u, 0x40000000u, 0x2u, 3u, 0u}, false);
	Check(!t.Translate(draw).ok, "a draw in a compute-ring buffer is refused");

	// SQ_THREAD_TRACE_USERDATA_2/3 (dwords 0xc342/0xc343, offsets 0x342/0x343) and the rest of
	// USERDATA_0..7. ASTRO BOT's ACBs write only these; refusing the opcode refused every ACB.
	std::vector<uint32_t> trace;
	Append(trace, IT_SET_UCONFIG_REG, {0x342u, 0x11u}, true);
	Append(trace, IT_SET_UCONFIG_REG, {0x343u, 0x22u}, true);
	Append(trace, IT_SET_UCONFIG_REG_INDEX, {0x340u, 0x1u}, true);
	Append(trace, IT_SET_UCONFIG_REG, {0x347u, 0x3u}, true);
	Append(trace, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
	auto rt = t.Translate(trace);
	Check(rt.ok && rt.stats.dispatches == 1 && rt.stats.uconfig_skipped == 4 && rt.skipped_uconfig.count(0x342u) == 1 &&
	          rt.skipped_uconfig.count(0x343u) == 1 && rt.skipped_uconfig.count(0x340u) == 1,
	      "profiler UCONFIG is dropped and the ACB still translates");
	const auto trace_pkts = Decode(rt.ib);
	Check(FindFirst(trace_pkts, IT_SET_UCONFIG_REG) == nullptr && FindFirst(trace_pkts, IT_SET_UCONFIG_REG_INDEX) == nullptr,
	      "dropped UCONFIG is not in the compute IB");
	Check(ValidateNativeIb(rt.ib, false, &bad, NativeIbKind::Compute), "an ACB that only marked the profiler passes the check");

	std::vector<uint32_t> ge;
	Append(ge, IT_SET_UCONFIG_REG, {0x242u, 0x5u}, true); // VGT_PRIMITIVE_TYPE
	Append(ge, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
	auto rg = t.Translate(ge);
	Check(!rg.ok && rg.unknown_op == IT_SET_UCONFIG_REG, "geometry-engine UCONFIG still refuses a compute-ring buffer");
	std::vector<uint32_t> ge_index;
	Append(ge_index, IT_SET_UCONFIG_REG_INDEX, {0x20000243u, 0u}, true); // VGT_INDEX_TYPE, index 2
	Check(!t.Translate(ge_index).ok, "indexed geometry-engine UCONFIG still refuses a compute-ring buffer");
}

// A label the GPU cannot reach must not become a WAIT_REG_MEM or a RELEASE_MEM. The oracle saying
// "native" is not enough: that packet would read or write the dummy page.
bool WaitsOn(const std::vector<Packet>& pkts, uint64_t addr) {
	for (const auto& p: pkts) {
		if (p.op == 0x3Cu && p.body.size() >= 3 && p.body[1] == static_cast<uint32_t>(addr) &&
		    p.body[2] == static_cast<uint32_t>(addr >> 32u)) {
			return true;
		}
	}
	return false;
}

void TestUnreachableLabel() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	static uint32_t label = 0;
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	o.unwritable          = addr;
	o.plan                = LabelOracle::Plan::Native;

	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(dcb, IT_WRITE_DATA, {5u << 8u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u}, false);
	AppendWait(dcb, addr, 1);
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 2 && r.segments[1].held && r.segments[1].wait_len == 0 && o.asked == 0,
	      "a wait on a label this buffer writes on the CPU is held, with no native wait");
	const auto pkts = Decode(r.ib);
	Check(!HasLabelRelease(pkts, addr, 1u) && !WaitsOn(pkts, addr), "no GPU release and no GPU wait on an unreachable label");
	Check(!HasRef(r.refs, addr, 1, false), "an unreachable label is not added for residency");
	bool cpu = false;
	for (const auto& p: r.segments[0].points) {
		cpu = cpu || (p.kind == TrackerPoint::Kind::CpuWrite && p.addr == addr && p.data.size() == 1 && p.data[0] == 1u);
	}
	Check(cpu && r.segments[0].writes.size() == 1 && !r.segments[0].writes[0].gpu, "the tracker writes it");

	std::vector<uint32_t> bare;
	Append(bare, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendWait(bare, addr, 1);
	auto rb = t.Translate(bare);
	Check(rb.ok && o.asked == 1 && rb.segments.size() == 2 && rb.segments[1].held && rb.segments[1].wait_len == 0 &&
	          !WaitsOn(Decode(rb.ib), addr),
	      "an oracle native wait on an unreachable label stays on the CPU");
}

void TestGpuTimestamp() {
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);

	std::vector<uint32_t> dcb;
	AppendReleaseMarker(dcb, 3, 2, 0x202000400ull, 0, 0x42); // clock + an interrupt
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 3, 2, 0x202000408ull, 0, 0x43);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.timestamps_native == 2 && r.stats.timestamps == 0 && r.stats.interrupts == 2, "two GPU timestamps");
	const auto pkts = Decode(r.ib);
	Check(HasGpuTimestamp(pkts, 0x202000400ull) && HasGpuTimestamp(pkts, 0x202000408ull), "both clocks are DATA_SEL 3");
	bool gfx_bit = true;
	uint32_t tracker = 0;
	for (const auto& p: pkts) {
		if (p.op == 0x49u && p.body[1] == (3u << 29u)) {
			gfx_bit = gfx_bit && !p.compute;
		}
		tracker += (p.op == 0x49u && p.body[2] == static_cast<uint32_t>(kTrackerVa)) ? 1u : 0u;
	}
	Check(gfx_bit && tracker == 1, "gfx timestamp has no shader bit; both interrupts share one tracker packet");
	Check(r.segments[0].points.size() == 3 && r.segments[0].points[0].dw == r.segments[0].points[2].dw &&
	          r.segments[0].points[0].kind == TrackerPoint::Kind::Interrupt &&
	          r.segments[0].points[1].kind == TrackerPoint::Kind::Interrupt,
	      "interrupts with a dispatch between them still share the end packet");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "a GPU timestamp passes the gfx check");

	t.SetComputeRing(true);
	std::vector<uint32_t> acb;
	AppendReleaseMarker(acb, 3, 0, 0x202000410ull, 0, 0);
	Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
	auto rc = t.Translate(acb);
	Check(rc.ok && rc.stats.timestamps_native == 1, "compute-ring timestamp translated");
	bool cs_bit = false;
	for (const auto& p: Decode(rc.ib)) {
		cs_bit = cs_bit || (p.op == 0x49u && p.body[1] == (3u << 29u) && p.compute);
	}
	Check(cs_bit && ValidateNativeIb(rc.ib, false, &bad, NativeIbKind::Compute), "compute timestamp carries the shader bit");
	Check(!ValidateNativeIb(rc.ib, false, &bad, NativeIbKind::Gfx), "that shader bit is not a gfx packet");

	// Not 8-byte aligned, or the GPU cannot reach it, or the CPU clock is requested: the tracker writes it.
	t.SetComputeRing(false);
	o.unwritable = 0x202000420ull;
	std::vector<uint32_t> cpu;
	AppendReleaseMarker(cpu, 3, 0, 0x202000420ull, 0, 0);
	AppendReleaseMarker(cpu, 3, 0, 0x202000424ull, 0, 0); // not qword aligned
	auto ru = t.Translate(cpu);
	Check(ru.stats.timestamps == 2 && ru.stats.timestamps_native == 0 && !HasGpuTimestamp(Decode(ru.ib), 0x202000420ull),
	      "an unreachable or unaligned timestamp stays on the tracker");
	bool saw_ts = false;
	for (const auto& p: ru.segments[0].points) {
		saw_ts = saw_ts || p.kind == TrackerPoint::Kind::Timestamp;
	}
	Check(saw_ts, "the tracker point is still there");

	o.unwritable = 0;
	t.SetCpuTimestamps(true);
	std::vector<uint32_t> host;
	AppendReleaseMarker(host, 3, 0, 0x202000430ull, 0, 0);
	auto rh = t.Translate(host);
	Check(rh.stats.timestamps == 1 && rh.stats.timestamps_native == 0 && !HasGpuTimestamp(Decode(rh.ib), 0x202000430ull),
	      "KYTY_NATIVE_GPU_CPU_TIMESTAMPS keeps the host clock on the tracker");

	std::vector<uint32_t> bad_data = {0xc0064900u, kFlushEvent, 3u << 29u, 0x1000u, 0u, 1u, 0u, 0u,
	                                  0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u,
	                                  0xffff1000u, 0xffff1000u};
	Check(!ValidateNativeIb(bad_data, false, &bad, NativeIbKind::Gfx), "a timestamp with a data dword is rejected");
	std::vector<uint32_t> bad_align = {0xc0064900u, kFlushEvent, 3u << 29u, 0x1004u, 0u, 0u, 0u, 0u,
	                                   0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u,
	                                   0xffff1000u, 0xffff1000u};
	Check(!ValidateNativeIb(bad_align, false, &bad, NativeIbKind::Gfx), "an unaligned GPU timestamp is rejected");
}

// Index of the first decoded packet that satisfies `pred`, or the packet count.
template <typename Pred>
size_t IndexOf(const std::vector<Packet>& pkts, Pred pred) {
	for (size_t i = 0; i < pkts.size(); i++) {
		if (pred(pkts[i])) {
			return i;
		}
	}
	return pkts.size();
}

void TestLabelsGpuOnly() {
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	// A label the GPU can reach is written by the GPU alone, in stream order, as on PS5. A CPU store at
	// the segment's tracker point would land after later writes to the same label (the next use's reset,
	// another ring) and could put the old value back.
	static uint32_t label = 0;
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	const uint32_t  lo    = static_cast<uint32_t>(addr);
	const uint32_t  hi    = static_cast<uint32_t>(addr >> 32u);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 1, 2, addr, 1, 0x21);                       // the label, with an interrupt
	Append(dcb, IT_WRITE_DATA, {(5u << 8u) | (2u << 30u), lo, hi, 0u}, false); // reset by the constant engine
	AppendWait(dcb, addr, 0);
	Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 1, 0, addr, 2, 0);

	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 1 && o.asked == 0 && r.stats.waits_native == 1,
	      "the wait on the constant engine's reset passes through: the GPU writes it ahead of the wait");
	Check(r.stats.labels_native == 3 && r.stats.labels_cpu == 0 && r.stats.interrupts == 1,
	      "three GPU label writes, none by the tracker");
	const auto pkts = Decode(r.ib);
	const auto first = IndexOf(pkts, [&](const Packet& p) {
		return p.op == 0x49u && IsLabelEvent(p.body[0]) && p.body[1] == (1u << 29u) && p.body[2] == lo && p.body[4] == 1u;
	});
	const auto reset = IndexOf(pkts, [&](const Packet& p) { return p.op == IT_WRITE_DATA && p.body[1] == lo; });
	const auto wait  = IndexOf(pkts, [&](const Packet& p) { return p.op == 0x3Cu && p.body[1] == lo; });
	const auto last  = IndexOf(pkts, [&](const Packet& p) {
		return p.op == 0x49u && IsLabelEvent(p.body[0]) && p.body[1] == (1u << 29u) && p.body[2] == lo && p.body[4] == 2u;
	});
	Check(first < reset && reset < wait && wait < last && last < pkts.size(), "the GPU writes the label in stream order");
	Check((pkts[reset].body[0] >> 30u) == 0u && (pkts[reset].body[0] & (1u << 20u)) != 0 && pkts[reset].body[3] == 0u,
	      "the constant-engine WRITE_DATA runs on the micro engine, write confirmed");
	const auto& pts   = r.segments[0].points;
	uint32_t    wakes = 0;
	uint32_t    irqs  = 0;
	bool        cpu   = false;
	for (const auto& p: pts) {
		wakes += p.kind == TrackerPoint::Kind::Wake && p.addr == addr ? 1u : 0u;
		irqs += p.kind == TrackerPoint::Kind::Interrupt && p.ctx == 0x21 ? 1u : 0u;
		cpu = cpu || p.kind == TrackerPoint::Kind::CpuWrite || !p.data.empty();
		// Points ahead of a GPU wait are flushed by a tracker store before it, the rest at the end.
		Check(IsTrackerPacket(r.ib, p.dw), "every point sits at a tracker store");
	}
	Check(!cpu && wakes == 3 && irqs == 1 && pts.back().kind == TrackerPoint::Kind::End,
	      "the tracker wakes the label and raises the interrupt, and stores nothing");
	bool all_gpu = !r.segments[0].writes.empty();
	for (const auto& w: r.segments[0].writes) {
		all_gpu = all_gpu && w.gpu;
	}
	Check(all_gpu, "every write is reported as the GPU's");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "the IB passes the gfx packet check");

	// On a compute ring the constant-engine write is a compute-bit micro engine WRITE_DATA.
	{
		Pm4Translator tc;
		TestOracle    oc;
		tc.SetComputeQueue(true);
		tc.SetComputeRing(true);
		tc.SetNativeLabels(&oc, kTrackerVa);
		tc.SetFenceAddress(kFenceVa);
		std::vector<uint32_t> acb;
		Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
		Append(acb, IT_WRITE_DATA, {(5u << 8u) | (2u << 30u), lo, hi, 4u}, true);
		const auto rc = tc.Translate(acb);
		const auto pc = Decode(rc.ib);
		const auto* wd = FindFirst(pc, IT_WRITE_DATA);
		Check(rc.ok && wd != nullptr && wd->compute && (wd->body[0] >> 30u) == 0u && rc.stats.labels_cpu == 0 &&
		          ValidateNativeIb(rc.ib, false, &bad, NativeIbKind::Compute),
		      "a compute-ring constant-engine write is the GPU's and passes the compute check");
	}

	// KYTY_NATIVE_GPU_TRACKER_WRITES=1: the old split. The tracker mirrors the GPU's label stores and
	// writes the constant engine's on its own, so the wait on that write is cut.
	Pm4Translator tm;
	TestOracle    om;
	tm.SetNativeLabels(&om, kTrackerVa);
	tm.SetFenceAddress(kFenceVa);
	tm.SetTrackerWrites(true);
	auto rm = tm.Translate(dcb);
	Check(rm.ok && rm.segments.size() == 2 && rm.segments[1].held && FindFirst(Decode(rm.ib), IT_WRITE_DATA) == nullptr,
	      "with tracker writes the constant-engine write is the tracker's and the wait is held");
	bool mirror = false;
	bool ce     = false;
	for (const auto& p: rm.segments[0].points) {
		mirror = mirror || (p.kind == TrackerPoint::Kind::CpuWrite && p.mirror && p.data.size() == 1 && p.data[0] == 1u);
		ce     = ce || (p.kind == TrackerPoint::Kind::CpuWrite && !p.mirror && p.data.size() == 1 && p.data[0] == 0u);
	}
	Check(mirror && ce && rm.stats.labels_cpu == 3, "the GPU label is mirrored and the reset written on the CPU");
}

void TestPlanRingWait() {
	// No pending write, label already in memory: a barrier. The GPU is not left waiting for a writer
	// that no submitted buffer will produce.
	Check(PlanRingWait(0, false, false, false, false, true, true) == RingWaitRelease::Barrier,
	      "a label already in memory is a barrier");
	Check(PlanRingWait(0, false, false, false, false, false, true) == RingWaitRelease::Hold,
	      "nothing known about the label: hold the segment on the CPU");
	// Same ring: the writer is already ahead in the batch, so the wait may go to the GPU before the CS.
	Check(PlanRingWait(1, true, true, true, false, false, true) == RingWaitRelease::GpuWait,
	      "same ring, writer ahead: real GPU wait");
	// Other ring: only after that writer's ioctl has returned.
	Check(PlanRingWait(1, true, true, false, false, false, true) == RingWaitRelease::Hold,
	      "other ring, writer not submitted yet: hold");
	Check(PlanRingWait(1, true, true, false, true, false, true) == RingWaitRelease::GpuWait,
	      "other ring, writer submitted: real GPU wait");
	// A pending write owns the address. Memory matching now is not enough: that write lands later.
	Check(PlanRingWait(1, false, true, false, true, true, true) == RingWaitRelease::Hold,
	      "a pending write that does not satisfy keeps the segment held");
	Check(PlanRingWait(2, false, true, true, true, true, true) == RingWaitRelease::Hold,
	      "a pending write of unknown value keeps the segment held");
	Check(PlanRingWait(1, true, true, false, true, false, false) == RingWaitRelease::Hold,
	      "no native wait form: do not tell the GPU to wait");
	// Other ring, writer submitted, but memory already holds a satisfying value: possibly stale from
	// the label's previous use, which the GPU wait would pass before the writer's ring runs.
	Check(PlanRingWait(1, true, true, false, true, true, true) == RingWaitRelease::Hold,
	      "other ring, memory already satisfies: hold until the writer lands");
	Check(PlanRingWait(1, true, true, true, false, true, true) == RingWaitRelease::GpuWait,
	      "same ring, memory already satisfies: the ring order still holds, real GPU wait");
	// A compute waiter whose pending writer is the gfx ring waits on the GPU before that CS.
	Check(PlanRingWait(0, false, false, false, false, false, true, true) == RingWaitRelease::Hold,
	      "no write emitted yet: hold, whoever wrote the label before");
	Check(PlanRingWait(1, true, true, false, false, false, true, true) == RingWaitRelease::GpuWait,
	      "gfx writer still in its batch: compute waits on the GPU");
	Check(PlanRingWait(1, true, true, false, false, true, true, true) == RingWaitRelease::Hold,
	      "gfx writer pending, memory already satisfies (stale): hold until it lands");
	Check(PlanRingWait(1, false, true, false, false, false, true, true) == RingWaitRelease::Hold,
	      "a gfx write that does not satisfy keeps the segment held");
}

void TestSplitIbAtPackets() {
	using Libs::Graphics::NativeGpu::SplitIbAtPackets;
	std::vector<uint32_t> ib;
	Append(ib, IT_SET_CONTEXT_REG, {0x318u, 1u, 2u, 3u}, false); // 5 dwords
	ib.push_back(0xffff1000u);                                   // 1-dword filler
	Append(ib, IT_DRAW_INDEX_AUTO, {3u, 2u}, false);              // 3 dwords
	Append(ib, IT_SET_CONTEXT_REG, {0x31cu, 4u}, false);          // 3 dwords
	const auto pieces = SplitIbAtPackets(ib, 5);
	Check(pieces.size() == 3 && pieces[0] == std::make_pair<size_t, size_t>(0, 5) &&
	          pieces[1] == std::make_pair<size_t, size_t>(5, 4) && pieces[2] == std::make_pair<size_t, size_t>(9, 3),
	      "an IB is cut only between packets, fillers count as one dword");
	Check(SplitIbAtPackets(ib, 64).size() == 1, "a short IB stays whole");
	Check(SplitIbAtPackets(ib, 4).empty(), "a packet longer than a piece cannot be cut");
}

void TestComputeRingForQueue() {
	Check(ComputeRingForQueue(0x20, 0x5u) == 0u && ComputeRingForQueue(0x21, 0x5u) == 2u &&
	          ComputeRingForQueue(0x22, 0x5u) == 0u,
	      "guest queues spread across the rings the kernel exposes");
	Check(ComputeRingForQueue(0x27, 0x1u) == 0u, "one compute ring: every queue shares it");
	Check(ComputeRingForQueue(0x20, 0u) == 0u, "no compute ring reported: ring 0");
}

void TestFlattenIndirectBuffers() {
	auto ib = [](const std::vector<uint32_t>& target, bool chain) {
		const auto va = reinterpret_cast<uint64_t>(target.data());
		return std::vector<uint32_t> {0xc0023f00u, static_cast<uint32_t>(va), static_cast<uint32_t>(va >> 32u),
		                              static_cast<uint32_t>(target.size()) | (chain ? (1u << 20u) : 0u) | 0x0f200000u};
	};
	const std::vector<uint32_t> nop_a = {0xc0001000u, 0xaaaau};
	const std::vector<uint32_t> nop_b = {0xc0001000u, 0xbbbbu};
	std::vector<uint32_t>       out;

	Check(!FlattenIndirectBuffers(nop_a, &out) && out.empty(), "no IB, no rewind: nothing to inline");

	std::vector<uint32_t> called = {0xc0001000u, 0x1111u};
	std::vector<uint32_t> buf    = ib(called, false);
	buf.insert(buf.end(), nop_b.begin(), nop_b.end());
	Check(FlattenIndirectBuffers(buf, &out) && out == std::vector<uint32_t>({0xc0001000u, 0x1111u, 0xc0001000u, 0xbbbbu}),
	      "a call runs the target, then the rest");

	// The chunked layout titles use: a valid REWIND, then a chain to the next chunk; nothing after
	// the chain runs.
	std::vector<uint32_t> next = nop_b;
	buf                        = nop_a;
	buf.insert(buf.end(), {0xc0005900u, 0x80000000u});
	const auto chain = ib(next, true);
	buf.insert(buf.end(), chain.begin(), chain.end());
	buf.insert(buf.end(), {0xc0001000u, 0xdeadu});
	Check(FlattenIndirectBuffers(buf, &out) && out == std::vector<uint32_t>({0xc0001000u, 0xaaaau, 0xc0001000u, 0xbbbbu}),
	      "a valid REWIND is dropped and a chain replaces the rest");

	buf = nop_a;
	buf.insert(buf.end(), {0xc0005900u, 0x00000000u});
	Check(!FlattenIndirectBuffers(buf, &out) && out.empty(), "a REWIND still waiting for the CPU is not inlined");

	// A command buffer of many chunks, each ending in a valid REWIND and a chain to the next.
	std::vector<std::vector<uint32_t>> chunks(20);
	for (size_t i = chunks.size(); i-- > 0;) {
		chunks[i] = {0xc0001000u, static_cast<uint32_t>(i)};
		if (i + 1 < chunks.size()) {
			chunks[i].insert(chunks[i].end(), {0xc0005900u, 0x80000000u});
			const auto next_chain = ib(chunks[i + 1], true);
			chunks[i].insert(chunks[i].end(), next_chain.begin(), next_chain.end());
		}
	}
	Check(FlattenIndirectBuffers(chunks[0], &out) && out.size() == 40 && out[1] == 0u && out[39] == 19u,
	      "a chain of 20 chunks is followed to its end");
}

// Gfx10 T# that counts into `bank` (word5 bit 25, word6[7:0]). The other words only have to pass
// the translator's plausibility checks.
std::array<uint32_t, 8> CountingT(uint32_t bank) {
	return {0x01234567u, 0x00a00000u, 0u, 0x90000000u, 0u, 0x02000000u, bank, 0u};
}

void AppendImage(std::vector<uint32_t>& dcb, uint32_t sh_offset, uint32_t bank) {
	const auto img = CountingT(bank);
	Append(dcb, IT_SET_SH_REG,
	       {sh_offset, img[0], img[1], img[2], img[3], img[4], img[5], img[6], img[7]}, false);
}

void TestLodUsageReport() {
	Check(ParseLodStatsMode(nullptr) == LodStatsMode::Usage && ParseLodStatsMode("") == LodStatsMode::Usage &&
	          ParseLodStatsMode("usage") == LodStatsMode::Usage,
	      "usage is the default LOD stats mode");
	Check(ParseLodStatsMode("all") == LodStatsMode::All && ParseLodStatsMode("fill") == LodStatsMode::All,
	      "all and fill keep the old fill");
	Check(ParseLodStatsMode("banks") == LodStatsMode::Banks && ParseLodStatsMode("even") == LodStatsMode::Even &&
	          ParseLodStatsMode("lo") == LodStatsMode::Lo && ParseLodStatsMode("hi") == LodStatsMode::Hi,
	      "the diagnostic modes stay selectable");

	std::bitset<256> banks;
	banks.set(0);
	banks.set(7);
	banks.set(255);
	alignas(8) uint8_t buf[0x840];
	std::memset(buf, 0xa5, sizeof(buf));
	const uint32_t used = WriteLodUsageReport(buf, sizeof(buf), banks);
	Check(used == 3, "three used records");
	Check(*reinterpret_cast<uint32_t*>(buf) == 1u, "header dword 0 is the report-present label");
	bool header_rest = true;
	for (uint32_t i = 1; i < 16; i++) {
		header_rest = header_rest && reinterpret_cast<uint32_t*>(buf)[i] == 0u;
	}
	Check(header_rest, "the rest of the 64-byte header is zero");
	auto record = [&](uint32_t bank) {
		uint64_t v = 0;
		std::memcpy(&v, buf + 64 + bank * 8, 8);
		return v;
	};
	Check(record(0) == 1ull && record(7) == 1ull && record(255) == 1ull, "a used bank is count 1, min mip 0");
	const uint8_t* used_bytes = buf + 64 + 7 * 8;
	Check(used_bytes[0] == 1 && used_bytes[1] == 0 && used_bytes[2] == 0 && used_bytes[3] == 0 && used_bytes[4] == 0 &&
	          used_bytes[5] == 0 && used_bytes[6] == 0 && used_bytes[7] == 0,
	      "a used record is the integer 1 (id byte and min-mip nibble clear)");
	Check(record(1) == 0x0F00000000000000ull && record(254) == 0x0F00000000000000ull,
	      "an unused bank is no-data with a zero sample count");
	const uint8_t* none = buf + 64 + 8;
	Check(none[0] == 0 && none[1] == 0 && none[2] == 0 && none[3] == 0 && none[4] == 0 && none[5] == 0 && none[6] == 0 &&
	          none[7] == 0x0F,
	      "no-data is nibble 59:56 = 0xF and nothing else");

	// A short buffer still gets a label and must not be written past.
	alignas(8) uint8_t tiny[64 + 16 + 4];
	std::memset(tiny, 0xa5, sizeof(tiny));
	std::bitset<256> one;
	one.set(1);
	Check(WriteLodUsageReport(tiny, 64 + 16, one) == 1, "one record fits in a two-record buffer");
	Check(tiny[64 + 16] == 0xa5 && tiny[64 + 17] == 0xa5, "the report stops at the caller's size");
	uint64_t first = 0;
	uint64_t second = 0;
	std::memcpy(&first, tiny + 64, 8);
	std::memcpy(&second, tiny + 72, 8);
	Check(first == 0x0F00000000000000ull && second == 1ull, "bank 1 is the second record");
}

void TestLodStatsHandoff() {
	LodStatsHandoff  handoff;
	std::bitset<256> banks;
	banks.set(4);
	handoff.Push(0x210000040ull, banks);
	std::bitset<256> got;
	Check(handoff.Take(0x210000040ull, got) && got.count() == 1 && got.test(4), "take returns the published banks");
	Check(!handoff.Take(0x210000040ull, got), "a report is delivered once");
	Check(!handoff.Take(0x210000080ull, got), "an unknown buffer has no report");

	banks.set(5);
	handoff.Push(0x1000, banks);
	std::bitset<256> later;
	later.set(9);
	handoff.Push(0x1000, later);
	Check(handoff.Take(0x1000, got) && got.test(4) && got.test(5) && !got.test(9), "the older report for an address is first");
	Check(handoff.Take(0x1000, got) && got.test(9) && !got.test(4), "then the newer one");

	// The translator publishes on the pipeline thread; the command processor takes on its own thread.
	LodStatsHandoff        shared;
	std::atomic<uint32_t>  matches {0};
	std::thread            producer([&] {
		for (uint32_t i = 0; i < 200; i++) {
			std::bitset<256> b;
			b.set(i % 256);
			shared.Push(static_cast<uint64_t>(i + 1), b);
		}
	});
	std::thread            consumer([&] {
		for (uint32_t i = 0; i < 200; i++) {
			std::bitset<256> b;
			bool             got_one = false;
			for (uint32_t spin = 0; spin < 10000000u && !got_one; spin++) {
				got_one = shared.Take(static_cast<uint64_t>(i + 1), b);
			}
			matches += got_one && b.test(i % 256) ? 1u : 0u;
		}
	});
	producer.join();
	consumer.join();
	Check(matches.load() == 200, "concurrent publish and take do not drop or tear a report");
}

void TestLodBankScan() {
	Pm4Translator t;
	t.SetLodBankScan(true);
	// A 2D T# at 0x123456700 counting into bank 7 (word5 bit 25, MIN_LOD_WARN 0x100) in PS user data
	// 0-7, and the same image without MIP_STATS_CNT_EN (bank 9) in GS user data 0-7.
	const uint32_t counting[8] = {0x01234567u, 0x00a00000u, 0u, 0x90000000u, 0u, 0x02010000u, 0x7u, 0u};
	const uint32_t silent[8]   = {0x01234567u, 0x00a00000u, 0u, 0x90000000u, 0u, 0x00010000u, 0x9u, 0u};
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_SH_REG, {0x00Cu, counting[0], counting[1], counting[2], counting[3], counting[4], counting[5], counting[6], counting[7]}, false);
	Append(dcb, IT_SET_SH_REG, {0x08Cu, silent[0], silent[1], silent[2], silent[3], silent[4], silent[5], silent[6], silent[7]}, false);
	Append(dcb, 0x2Du /* IT_DRAW_INDEX_AUTO */, {3u, 2u}, false);
	Append(dcb, 0x8Eu /* IT_GET_LOD_STATS */, {0x840u, 0x10000040u, 0x2u, 0x80000u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.lod_scans.size() == 1, "one LOD bank scan per GET_LOD_STATS");
	const auto& s = r.lod_scans[0];
	Check(s.addr == 0x210000040ull, "scan names the packet's buffer");
	Check(s.banks.count() == 1 && s.banks.test(7) && s.counting == 1 && s.inline_ud == 1, "only the counting T#'s bank");
	Check(s.min_lod_warn.size() == 1 && s.min_lod_warn.count(0x100u) == 1, "MIN_LOD_WARN from word5[19:8]");
	// Banks start over after the packet.
	std::vector<uint32_t> next;
	Append(next, 0x8Eu /* IT_GET_LOD_STATS */, {0x840u, 0x10000040u, 0x2u, 0x80000u}, false);
	auto r2 = t.Translate(next);
	Check(r2.lod_scans.size() == 1 && r2.lod_scans[0].banks.none(), "no draws, no banks");
}

void TestLodBankSources() {
	Pm4Translator t;
	t.SetLodBankScan(true);
	// A zero-size GET_LOD_STATS is the per-frame disable. It is not a report the game reads, so the
	// banks sampled before it stay in the window until the next real report.
	std::vector<uint32_t> first;
	AppendImage(first, 0x00Cu, 3);
	Append(first, 0x2Du /* IT_DRAW_INDEX_AUTO */, {3u, 2u}, false);
	Append(first, 0x8Eu /* IT_GET_LOD_STATS */, {0u, 0u, 0u, 0u}, false);
	auto r0 = t.Translate(first);
	Check(r0.ok && r0.lod_scans.empty(), "a zero-size GET_LOD_STATS does not close the window");

	// VS user data loaded indirectly, plus ES/HS/LS inline, plus a T# in a descriptor table and one
	// in the table that table points at, plus the PS USER_DATA_ADDR spill table.
	const auto vs = CountingT(21);
	std::vector<uint32_t> pairs;
	for (uint32_t i = 0; i < 8; i++) {
		pairs.push_back(0x04Cu + i);
		pairs.push_back(vs[i]);
	}
	const auto     nested = CountingT(28);
	const uint32_t nested_words[8] = {nested[0], nested[1], nested[2], nested[3], nested[4], nested[5], nested[6], nested[7]};
	const uint32_t table_words[8]  = {0x00000400u, 0x2u, 0u, 0u, 0u, 0u, 0u, 0u}; // -> 0x200000400
	const auto     spill_img = CountingT(27);
	const uint32_t spill_words[8] = {spill_img[0], spill_img[1], spill_img[2], spill_img[3],
	                                 spill_img[4], spill_img[5], spill_img[6], spill_img[7]};
	const uint32_t silent[8] = {0x01234567u, 0x00a00000u, 0u, 0x90000000u, 0u, 0x00010000u, 0x9u, 0u};

	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_SH_REG_INDIRECT, pairs);
	AppendImage(dcb, 0x0CCu, 22); // ES
	AppendImage(dcb, 0x10Cu, 23); // HS
	AppendImage(dcb, 0x14Cu, 24); // LS
	// PS user data 0-1 is the descriptor table; user data 8-15 is an inline T#.
	const auto inline_ps = CountingT(30);
	Append(dcb, IT_SET_SH_REG,
	       {0x00Cu, 0x00000100u, 0x2u, 0u, 0u, 0u, 0u, 0u, 0u, inline_ps[0], inline_ps[1], inline_ps[2], inline_ps[3],
	        inline_ps[4], inline_ps[5], inline_ps[6], inline_ps[7]},
	       false);
	Append(dcb, IT_SET_SH_REG, {0x002u, 0x00010000u, 0x2u}, false); // USER_DATA_ADDR -> 0x200010000
	Append(dcb, IT_SET_SH_REG, {0x08Cu, silent[0], silent[1], silent[2], silent[3], silent[4], silent[5], silent[6], silent[7]}, false);
	Append(dcb, 0x2Du /* IT_DRAW_INDEX_AUTO */, {3u, 2u}, false);
	// Compute user data, sampled by the dispatch in the same window.
	AppendImage(dcb, 0x240u, 25);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 0u}, true);
	Append(dcb, 0x8Eu /* IT_GET_LOD_STATS */, {0x840u, 0x10000040u, 0x2u, 0x80000u}, false);

	struct Tables {
		uint64_t        addr[3];
		const uint32_t* words[3];
		size_t          count[3];
	};
	Tables tables {{0x200000100ull, 0x200000400ull, 0x200010000ull},
	                     {table_words, nested_words, spill_words},
	                     {8, 8, 8}};
	t.SetLodGuestRead(
	    [](uint64_t addr, uint32_t* dst, size_t dwords, void* user) -> size_t {
		    const auto* tabs = static_cast<const Tables*>(user);
		    for (int i = 0; i < 3; i++) {
			    if (addr == tabs->addr[i]) {
				    const size_t n = std::min(dwords, tabs->count[i]);
				    std::memcpy(dst, tabs->words[i], n * sizeof(uint32_t));
				    return n;
			    }
		    }
		    return 0;
	    },
	    &tables);
	auto r = t.Translate(dcb);
	Check(r.ok && r.lod_scans.size() == 1, "the real report closes the window");
	const auto& s = r.lod_scans[0];
	Check(s.banks.test(3), "banks from the previous DCB are still in the window");
	Check(s.banks.test(21), "a T# loaded into VS user data by SET_SH_REG_INDIRECT counts");
	Check(s.banks.test(22) && s.banks.test(23) && s.banks.test(24), "ES, HS and LS user data count");
	Check(s.banks.test(30), "an inline T# beside a table pointer counts");
	Check(s.banks.test(28), "a T# in a table reached through another table counts");
	Check(s.banks.test(27), "a T# in the USER_DATA_ADDR spill table counts");
	Check(s.banks.test(25), "a dispatch's COMPUTE_USER_DATA T# counts");
	Check(s.banks.count() == 9, "exactly the sampled banks");
	Check(!s.banks.test(9), "a T# without MIP_STATS_CNT_EN is not a sampled bank");
}

int main() {
	// The cases count the packets a DCB turns into; the preamble's start values (PIXEL_PIPE_STAT_CONTROL,
	// the CP draw state) are board settings with their own switches, left out here.
	setenv("KYTY_NATIVE_GPU_PIXEL_PIPE_STAT", "0", 0);
	setenv("KYTY_NATIVE_GPU_CP_STATE", "0", 0);
	TestNopsOnly();
	TestContextShadowAcrossDcbs();
	TestShaderDetection();
	TestIndirectContextExpansion();
	TestIndirectShComputeBit();
	TestComputeCuClamp();
	TestUconfigWhitelist();
	TestIndirectTablesVerbatim();
	TestAgcTagsVerbatim();
	TestUconfigIndexPassthrough();
	TestUconfigIndexedRegisters();
	TestWriteData();
	TestSetPredication();
	TestDisplayDccRestore();
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
	TestPs5RegOverrides();
	TestAcquireFull();
	TestLabelsWriteData();
	TestLabelsWriteDataBlock();
	TestDirectRuns();
	TestLabelsReleaseMem();
	TestLabelsLocalWait();
	TestLabelsSplit();
	TestLabelsCpuTail();
	TestLabelWriteCovers();
	TestLabelsWait64();
	TestValidateNativeIb();
	TestComputeRingPackets();
	TestUnreachableLabel();
	TestGpuTimestamp();
	TestLabelsGpuOnly();
	TestPlanRingWait();
	TestComputeRingForQueue();
	TestSplitIbAtPackets();
	TestFlattenIndirectBuffers();
	TestLodBankScan();
	TestLodUsageReport();
	TestLodStatsHandoff();
	TestLodBankSources();
	std::puts("NativeGpuPm4TranslatorTests: all passed");
	return 0;
}
