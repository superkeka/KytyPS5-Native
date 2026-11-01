#include "graphics/native_gpu/nativePm4.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
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
using Libs::Graphics::NativeGpu::kFlipPadMagic;
using Libs::Graphics::NativeGpu::kWaitFlipPadMagic;
using Libs::Graphics::NativeGpu::kWaitFlipPadEopMark;
using Libs::Graphics::NativeGpu::kWaitFlipPadToken;
using Libs::Graphics::NativeGpu::WaitFlipGroupDw;
using Libs::Graphics::NativeGpu::SetWaitFlipMarks;
using Libs::Graphics::NativeGpu::AppendKernelReleases;

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
// unchanged; the shadow copy still takes their values. VGT_PRIMITIVE_TYPE in a table runs as written
// too (AgcCreatePrimState's); a stub entry or a patched value keeps the expansion.
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
	Check(r3.ok && r3.stats.loads_verbatim == 1 && r3.stats.loaded_uc_regs == 2 && r3.stats.loaded_cx_regs == 2,
	      "stub entries and other data formats are expanded, a primitive-type table runs as written");

	// Compute registers under a graphics-type header (My First GT's DCB dispatches), duplicates included.
	std::vector<uint32_t> cs_pairs = {0x215u, 0u, 0x20Cu, 0x905593cu, 0x20Du, 0u, 0x22Au, 0x28000100u, 0x22Au, 0x1157a6b7u};
	std::vector<uint32_t> cs;
	AppendIndirect(cs, IT_SET_SH_REG_INDIRECT, cs_pairs, 0x80000000u);
	auto r4 = t.Translate(cs);
	Check(r4.ok && r4.stats.loads_verbatim == 1 && r4.stats.changed_ops[IT_SET_SH_REG_INDIRECT] == 0 &&
	          std::search(r4.ib.begin(), r4.ib.end(), cs.begin(), cs.end()) != r4.ib.end(),
	      "a compute table under a graphics-type header runs as written on the gfx ring");
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
	Check(Pm4Translator::UconfigAllowed(0x248u) && Pm4Translator::UconfigAllowed(0x383u) &&
	          Pm4Translator::UconfigAllowed(0x384u),
	      "AgcCreatePrimState's VGT_OBJECT_ID and TEXTURE_GRADIENT_FACTORS/CONTROL allowed");
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

	// A 64-bit wait's begin tag: 0x342 and 0x343 in one packet.
	Pm4Translator         t64;
	std::vector<uint32_t> wide = {0xc0027904u, 0x342u, 0xc8020010u, 0x00ee89e0u};
	auto                  r64  = t64.Translate(wide);
	Check(r64.ok && r64.stats.uconfig_skipped == 0 && r64.stats.changed_ops[IT_SET_UCONFIG_REG] == 0 &&
	          std::search(r64.ib.begin(), r64.ib.end(), wide.begin(), wide.end()) != r64.ib.end(),
	      "the two-register begin tag is in the gfx IB as written");

	// A push marker as libSceAgcDriver writes it: tag and first 4 characters to 0x342, the rest with
	// WRITE_DATA to the register 0xc343. As written on the gfx ring, dropped on a compute ring.
	std::vector<uint32_t> marker = {0xc0027904u, 0x342u, 0xc100c000u, 0x6c6c6568u,
	                                0xc0043704u, 0x06010000u, 0xc343u, 0u, 0x726f776fu, 0x0000646cu};
	Pm4Translator tm;
	auto          rm = tm.Translate(marker);
	Check(rm.ok && rm.stats.changed_ops[IT_SET_UCONFIG_REG] == 0 && rm.stats.changed_ops[IT_WRITE_DATA] == 0 &&
	          std::search(rm.ib.begin(), rm.ib.end(), marker.begin(), marker.end()) != rm.ib.end() &&
	          ValidateNativeIb(rm.ib, false, nullptr, NativeIbKind::Gfx),
	      "a push marker reaches the gfx CP as written");
	Pm4Translator tc;
	tc.SetComputeQueue(true);
	tc.SetComputeRing(true);
	auto rc = tc.Translate(marker);
	Check(rc.ok && std::find(rc.ib.begin(), rc.ib.end(), 0xc343u) == rc.ib.end() &&
	          ValidateNativeIb(rc.ib, false, nullptr, NativeIbKind::Compute),
	      "a compute ring drops it");
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
	{
		Pm4Translator off;
		const auto    r0 = off.Translate(dcb);
		bool       set_fmask = false;
		for (const auto& p: Decode(r0.ib)) {
			set_fmask |= p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2 && p.body[0] == 0x321u + 4u * 0xFu && p.body[1] != 0u;
		}
		Check(r0.ok && r0.stats.fmask_fixed == 0 && !set_fmask, "by default the title's FMASK 0 runs as written");
	}
	t.SetFmaskFix(true);
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
	Check(r.stats.pfp_syncs_emitted == 1 && Count(pkts, 0x42u) == 1 && FindFirst(pkts, 0x42u) < base,
	      "PFP_SYNC_ME first: the PS5 runs the buffer on MEC, where no prefetch parser reads the arguments early");
	Pm4Translator         d;
	std::vector<uint32_t> dcb;
	Append(dcb, 0x11u /* IT_SET_BASE */, {1u, 0x10000000u, 0x2u}, false);
	Append(dcb, 0x16u, {0x40u, 0x41u}, false);
	auto rd = d.Translate(dcb);
	Check(rd.ok && rd.stats.pfp_syncs_emitted == 0, "a DCB's indirect dispatch runs as written");
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

void TestIndirectMultiDraws() {
	Pm4Translator         t;
	std::vector<uint32_t> dcb;
	Append(dcb, 0x11u /* IT_SET_BASE */, {1u, 0x10000000u, 0x2u}, false);
	// AgcDcbDrawIndexIndirectMulti: offset 0x40, at most 16 records 20 bytes apart, the count at 0x2_3000_0000.
	Append(dcb, 0x38u /* IT_DRAW_INDEX_INDIRECT_MULTI */,
	       {0x40u, 0x280u, 0x280u, 0x280u | (1u << 30u), 16u, 0x30000000u, 0x2u, 20u, 0u}, false);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.draws == 1, "DRAW_INDEX_INDIRECT_MULTI translated");
	const auto    pkts = Decode(r.ib);
	const Packet* p    = FindFirst(pkts, 0x38u);
	Check(p != nullptr && p->body.size() == 9 && p->body[3] == (0x280u | (1u << 30u)) && p->body[4] == 16u && p->body[7] == 20u,
	      "multi indirect draw kept as written");
	bool args_ref  = false;
	bool count_ref = false;
	for (const auto& ref: r.refs) {
		args_ref  = args_ref || (ref.addr == 0x210000040ull && ref.size == 320u);
		count_ref = count_ref || (ref.addr == 0x230000000ull && ref.size == 4u);
	}
	Check(args_ref && count_ref, "argument records and the count reported as references");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "a multi indirect draw passes the gfx-ring check");
	Check(!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Compute), "and is refused on a compute ring");
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

// AgcDcbResetQueue op 0x400 (libSceAgc's form): COND_EXEC on the push flag over CLEAR_STATE 2 and the flag's swap back.
// The COND_EXEC is left out; the pop restores a pushed context and does nothing without one.
void TestResetQueuePop() {
	auto reset_pop = [](std::vector<uint32_t>& dcb) {
		dcb.insert(dcb.end(), {0xc0032200u, 0x0025e000u, 0x2u, 0u, 0x0000000bu, 0xc0001200u, 2u, 0xc0071e00u, 0x40000267u, 0x0025e000u, 0x2u, 0u, 1u,
		                       0u, 0u, 6u});
	};
	for (const bool pushed: {true, false}) {
		Pm4Translator         t;
		std::vector<uint32_t> dcb;
		Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x1111u}, false);
		if (pushed) {
			Append(dcb, 0x12u /* IT_CLEAR_STATE */, {1u}, false); // push
		}
		Append(dcb, IT_SET_CONTEXT_REG, {0x318u, 0x2222u}, false);
		reset_pop(dcb);
		auto     r    = t.Translate(dcb);
		uint32_t last = 0;
		for (const auto& p: Decode(r.ib)) {
			if (p.op == IT_SET_CONTEXT_REG && p.body.size() >= 2 && p.body[0] == 0x318u) {
				last = p.body[1];
			}
		}
		Check(r.ok && r.stats.context_checks == 1 && Count(Decode(r.ib), 0x22u) == 0 && Count(Decode(r.ib), 0x1Eu) == 1,
		      "the reset's pop translates: COND_EXEC left out, the flag swap kept");
		Check(last == (pushed ? 0x1111u : 0x2222u), pushed ? "after a push it restores the pushed context" : "without a push it changes nothing");
	}
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

// A DMA_DATA without CP_SYNC gets a zero-byte DMA_DATA with CP_SYNC before the job ends (the IB's end, or with native
// labels before the segment's tracker release); one with CP_SYNC, none, and KYTY_NATIVE_GPU_CP_DMA_IDLE=0 none.
void TestCpDmaIdle() {
	constexpr uint32_t kIdleControl = (1u << 31u) | (3u << 20u) | (3u << 29u);
	auto is_idle = [&](const Packet& p) { return p.op == 0x50u && p.body == std::vector<uint32_t>({kIdleControl, 0u, 0u, 0u, 0u, 0u}); };
	auto idles   = [&](const std::vector<Packet>& pkts) { return static_cast<uint32_t>(std::count_if(pkts.begin(), pkts.end(), is_idle)); };
	std::vector<uint32_t> plain;
	Append(plain, 0x50u, {0x0u, 0x1000u, 0x2u, 0x2000u, 0x2u, 256u}, false);
	Append(plain, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	std::vector<uint32_t> synced;
	Append(synced, 0x50u, {1u << 31u, 0x1000u, 0x2u, 0x2000u, 0x2u, 256u}, false);
	{
		Pm4Translator t;
		auto          r    = t.Translate(plain);
		const auto    pkts = Decode(r.ib);
		Check(r.ok && r.stats.cp_dma_idle == 1 && idles(pkts) == 1,
		      "a DMA without CP_SYNC: the IB ends with the CP DMA idle wait");
		size_t at = 0;
		for (size_t i = 0; i < pkts.size(); i++) {
			at = is_idle(pkts[i]) ? i : at;
		}
		Check(FindFirst(pkts, IT_DISPATCH_DIRECT) < &pkts[at], "after the rest of the job");
		Check(ValidateNativeIb(r.ib, false, nullptr, NativeIbKind::Gfx), "it passes the gfx packet check");
		auto rs = t.Translate(synced);
		Check(rs.ok && rs.stats.cp_dma_idle == 0 && idles(Decode(rs.ib)) == 0, "a DMA with CP_SYNC needs none");
	}
	{
		Pm4Translator t;
		t.SetCpDmaIdle(false);
		auto r = t.Translate(plain);
		Check(r.ok && r.stats.cp_dma_idle == 0 && idles(Decode(r.ib)) == 0, "off: none");
	}
	{
		Pm4Translator t;
		t.SetComputeRing(true);
		std::vector<uint32_t> acb;
		Append(acb, 0x50u, {0x0u, 0x1000u, 0x2u, 0x2000u, 0x2u, 256u}, true);
		auto r = t.Translate(acb);
		Check(r.ok && r.stats.cp_dma_idle == 1 && ValidateNativeIb(r.ib, true, nullptr, NativeIbKind::Compute), "on a compute ring with the shader-type bit");
	}
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		auto r = t.Translate(plain);
		Check(r.ok && r.segments.size() == 1 && r.stats.cp_dma_idle == 1, "native labels: one idle wait");
		const auto& s       = r.segments[0];
		uint32_t    idle_dw = 0;
		for (uint32_t dw = 0; dw + 7 <= r.ib.size(); dw++) {
			if (r.ib[dw] == Pkt3(0x50u, 6, false) && r.ib[dw + 1] == kIdleControl) {
				idle_dw = dw;
			}
		}
		Check(!s.points.empty() && idle_dw != 0 && idle_dw < s.points.back().dw && IsTrackerPacket(r.ib, s.points.back().dw),
		      "before the segment's tracker release");
	}
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
	{
		// A title's NOPs (AgcCbNop) run in place: one stretch.
		std::vector<uint32_t> nops = dcb;
		nops.insert(nops.begin() + 99, {0xc0021000u, 0u, 0u, 0u}); // 4 dwords
		nops.insert(nops.begin() + 202, {0xc0001000u, 0u});         // 2 dwords
		Pm4Translator t;
		t.SetDirectRuns(64);
		t.SetSource({{0u, static_cast<uint32_t>(nops.size()), kVa}});
		auto r = t.Translate(nops);
		Check(r.ok && r.stats.direct_ib2 == 1 && r.stats.direct_dw == nops.size() && r.stats.changed_ops[IT_NOP] == 0,
		      "NOPs inside a stretch run in place");
		uint32_t bad = 0;
		Check(ValidateNativeIb(nops, false, &bad, NativeIbKind::Gfx), "a title's NOP passes the gfx check");
		Check(!ValidateNativeIb(nops, false, &bad, NativeIbKind::Compute), "not the compute check");
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

// libSceAgc's own RELEASE_MEM (agc.cpp writes the hardware packet), as ASTRO BOT sends it: CACHE_FLUSH_AND_INV_TS
// at end of pipe, cache policy 3, data to memory (DST_SEL 0; `dst_sel` 1 is the PS5 library's L2).
void AppendGuestRelease(std::vector<uint32_t>& dcb, uint32_t data_sel, uint32_t int_sel, uint64_t dst, uint64_t value,
                        uint32_t dst_sel = 0, uint32_t event = 0x06000514u, uint32_t ctx = 0) {
	Append(dcb, 0x49u,
	       {event, (dst_sel << 16u) | (int_sel << 24u) | (data_sel << 29u), static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u),
	        static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32u), ctx},
	       false);
}

void TestLabelsAsWritten() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	std::vector<uint32_t> dcb;
	AppendGuestRelease(dcb, 1, 0, 0x202000300ull, 5);
	AppendGuestRelease(dcb, 2, 0, 0x202000308ull, 0x0000000900000008ull);
	AppendGuestRelease(dcb, 3, 0, 0x202000310ull, 0);
	AppendGuestRelease(dcb, 0, 0, 0, 0);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.changed_ops[0x49] == 1 && r.stats.ops[0x49] == 4,
	      "labels without an interrupt go as the title wrote them (32-bit, clock, cache only), 64-bit ones not by default");
	const auto pkts = Decode(r.ib);
	Check(HasLabelRelease(pkts, 0x20200030cull, 9u) && HasLabelRelease(pkts, 0x202000308ull, 8u), "the 64-bit label as two halves");

	// Data to L2 (libSceAgc's labels) never reaches the CPU on this board: written to memory with write-backs.
	std::vector<uint32_t> l2;
	AppendGuestRelease(l2, 1, 0, 0x202000300ull, 5, 1);
	auto rl = t.Translate(l2);
	Check(rl.stats.changed_ops[0x49] == 1 && HasLabelRelease(Decode(rl.ib), 0x202000300ull, 5u), "a label sent to L2 is rewritten");

	// GL2_WB without GLM_WB: rewritten with GLM_WB added.
	std::vector<uint32_t> wb;
	AppendGuestRelease(wb, 0, 0, 0, 0, 0, 0x00200528u);
	auto rw = t.Translate(wb);
	Check(rw.stats.changed_ops[0x49] == 1, "a GL2 write-back without GLM_WB is not passed as written");

	// An interrupt stays the tracker's.
	std::vector<uint32_t> irq;
	AppendGuestRelease(irq, 1, 2, 0x202000300ull, 5);
	auto ri = t.Translate(irq);
	Check(ri.stats.changed_ops[0x49] == 1 && ri.stats.interrupts == 1, "a release with an interrupt is not passed as written");
}

// With the EOP log, the GPU raises a title's interrupts that carry a context id where they run.
void TestGuestInterruptsOnGpu() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetGuestInterruptsOnGpu(true);
	std::vector<uint32_t> dcb;
	AppendGuestRelease(dcb, 0, 1, 0, 0, 1, 0x06000514u, 0xabcdu);         // the interrupt alone, ctx 0xabcd
	AppendGuestRelease(dcb, 1, 2, 0x202000300ull, 5, 1, 0x06000514u, 0xa); // a label, then the interrupt
	AppendGuestRelease(dcb, 2, 2, 0x202000308ull, 7, 1);                   // ctx 0: the tracker's
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.interrupts_gpu == 2 && r.stats.interrupts == 1, "context ids 0xabcd and 0xa on the GPU, ctx 0 on the tracker");
	const auto pkts  = Decode(r.ib);
	size_t     alone = pkts.size(), after = pkts.size(), label = pkts.size();
	for (size_t i = 0; i < pkts.size(); i++) {
		const auto& b = pkts[i].body;
		if (pkts[i].op != 0x49u) {
			continue;
		}
		alone = b[6] == 0xabcdu ? i : alone;
		after = b[6] == 0xau ? i : after;
		label = b[2] == 0x02000300u && b[4] == 5u ? i : label;
	}
	Check(alone < pkts.size() && ((pkts[alone].body[1] >> 24u) & 7u) == 1u && (pkts[alone].body[1] >> 29u) == 0u,
	      "the interrupt alone goes as the title wrote it");
	Check(after < pkts.size() && ((pkts[after].body[1] >> 24u) & 7u) == 2u &&
	          pkts[after].body[2] == static_cast<uint32_t>(kTrackerVa + 8) && label < after,
	      "after a label the interrupt follows it, into the scratch dword");
	Check(ValidateNativeIb(r.ib, false, nullptr, NativeIbKind::Gfx), "and the packet check takes both");
	const auto& pts = r.segments[0].points;
	Check(std::count_if(pts.begin(), pts.end(), [](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Interrupt; }) == 1,
	      "one tracker interrupt point");

	// SetGuestInterruptsCtx0 (bc250_gfx_eop_events=2): context id 0 is raised by the GPU too.
	Pm4Translator t0;
	t0.SetNativeLabels(&o, kTrackerVa);
	t0.SetGuestInterruptsOnGpu(true);
	t0.SetGuestInterruptsCtx0(true);
	auto r0 = t0.Translate(dcb);
	Check(r0.ok && r0.stats.interrupts_gpu == 3 && r0.stats.interrupts == 0, "context id 0 on the GPU as well");
	Check(ValidateNativeIb(r0.ib, false, nullptr, NativeIbKind::Gfx), "the packet check takes it");

	// A compute queue with a KFD queue of its own: the interrupt alone goes in our form, compute bit set.
	Pm4Translator acb;
	acb.SetNativeLabels(&o, kTrackerVa);
	acb.SetComputeQueue(true);
	acb.SetComputeRing(true);
	acb.SetGuestInterruptsOnGpu(true);
	std::vector<uint32_t> acb_buf;
	AppendGuestRelease(acb_buf, 0, 1, 0, 0, 1, 0x06000514u, 0xau);
	auto       ra   = acb.Translate(acb_buf);
	const auto apkt = Decode(ra.ib);
	const auto air  = std::find_if(apkt.begin(), apkt.end(), [](const Packet& p) { return p.op == 0x49u && p.body[6] == 0xau; });
	Check(ra.ok && ra.stats.interrupts_gpu == 1 && ra.stats.interrupts == 0 && air != apkt.end() && air->compute &&
	          ((air->body[1] >> 24u) & 7u) == 2u,
	      "an ACB's interrupt is raised by the GPU in our form");
	Check(ValidateNativeIb(ra.ib, false, nullptr, NativeIbKind::Compute), "which the compute packet check takes");

	// Off by default.
	Pm4Translator plain;
	plain.SetNativeLabels(&o, kTrackerVa);
	auto rp = plain.Translate(dcb);
	Check(rp.stats.interrupts_gpu == 0 && rp.stats.interrupts == 3, "without it every interrupt is the tracker's");
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

	// SetFlipsAtEop: AgcDcbSetFlip's flip is a tracker point; the flip with a label stays Kyty's.
	Pm4Translator e;
	e.SetNativeLabels(&o, kTrackerVa);
	e.SetFlipsAtEop(true);
	auto re = e.Translate(dcb);
	Check(re.ok && re.cpu_tail.size() == 5 && re.cpu_tail[0] == dcb[6], "the plain flip leaves the continuation");
	const auto& pts  = re.segments[0].points;
	const auto  flip = std::find_if(pts.begin(), pts.end(), [](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Flip; });
	Check(flip != pts.end() && flip->data == std::vector<uint32_t>({0u, 1u, 2u, 3u, 4u}), "and is a flip point with its five dwords");
}

// Kyty's bookkeeping in the padding of libSceAgc's SetFlip and WaitUntilSafeForRendering (pm4.h): kFlipPadMagic,
// kWaitFlipPadMagic and the padding's layout come from nativePm4.h.

// libSceAgc's marker (agc.cpp): SET_UCONFIG 0x342 with the value, a payload's first dword in 0x343 and the rest in a
// WRITE_DATA to SQ_THREAD_TRACE_USERDATA_3 (0xc343). Header bit 2 is set in a DCB, clear in an ACB.
void AppendAgcMarker(std::vector<uint32_t>& cb, bool acb, uint32_t value, const std::vector<uint32_t>& payload) {
	const uint32_t r = acb ? 0u : 4u;
	if (payload.empty()) {
		cb.insert(cb.end(), {0xc0017900u | r, 0x342u, value});
		return;
	}
	cb.insert(cb.end(), {0xc0027900u | r, 0x342u, value, payload[0]});
	if (payload.size() > 1) {
		const auto rem = static_cast<uint32_t>(payload.size() - 1);
		cb.insert(cb.end(), {0xc0003700u | ((rem + 2u) << 16u) | r, 0x06010000u, 0x0000c343u, 0u});
		cb.insert(cb.end(), payload.begin() + 1, payload.end());
	}
}

// AgcDcbSetFlip / AgcAcbSetFlip in libSceAgc's form, 64 dwords: the marker with its 12-byte payload, the RELEASE_MEM
// that writes 1 into the display buffer's label (none for a negative index) and raises the interrupt with the context id
// sceVideoOutSubmitEopFlip gave, and the padding NOP with Kyty's bookkeeping.
void AppendSonyFlip(std::vector<uint32_t>& cb, bool acb, uint32_t handle, int idx, uint32_t mode, uint64_t arg, uint64_t label,
                    uint32_t ctx) {
	const size_t   start = cb.size();
	const auto     index = static_cast<uint32_t>(idx);
	const bool     has   = idx >= 0;
	const uint64_t l     = has ? label : 0u;
	// The context's bits 8..13 are 0 (Kyty's VideoOut), so libSceAgc's p and t are 0.
	AppendAgcMarker(cb, acb, 0xc7010000u | ((mode & 0xfu) << 8u) | ((index & 0x1fu) << 3u),
	                {0u, static_cast<uint32_t>(arg), static_cast<uint32_t>(arg >> 32u)});
	cb.insert(cb.end(), {0xc0064900u, acb ? 0x0620062fu : 0x06200504u, 0x02010000u | (has ? 1u << 30u : 0u), static_cast<uint32_t>(l) & ~3u,
	                     static_cast<uint32_t>(l >> 32u), has ? 1u : 0u, 0u, ctx & 0x0fffffffu});
	cb.push_back(0xc02c1000u);
	std::vector<uint32_t> pad(45, 0u);
	const std::vector<uint32_t> kyty = {kFlipPadMagic, handle, index, mode, static_cast<uint32_t>(arg), static_cast<uint32_t>(arg >> 32u), ctx};
	std::copy(kyty.begin(), kyty.end(), pad.begin());
	cb.insert(cb.end(), pad.begin(), pad.end());
	Check(cb.size() - start == 64, "test SetFlip is 64 dwords");
}

// AgcDcbWaitUntilSafeForRendering / AgcAcbWaitUntilSafeForRendering in libSceAgc's form, 32 dwords: the begin marker, the
// WAIT_REG_MEM_64 until the label is 0 (a 9-dword NOP for a negative index), the end marker and the padding NOP with
// Kyty's bookkeeping (the EOP flip order not set yet: 0xffffffff twice).
void AppendSonyWaitUntilSafe(std::vector<uint32_t>& cb, bool acb, uint32_t handle, int idx, uint64_t label, uint64_t watermark) {
	const size_t start = cb.size();
	const auto   index = static_cast<uint32_t>(idx);
	AppendAgcMarker(cb, acb, 0xcb000000u | (index & 0x1fu), {});
	if (idx >= 0) {
		cb.insert(cb.end(), {0xc0079300u, 0x06000113u, static_cast<uint32_t>(label) & ~7u, static_cast<uint32_t>(label >> 32u) & 0x3ffffu, 0u, 0u,
		                     0xffffffffu, 0xffffffffu, 0x40u});
	} else {
		cb.push_back(0xc0071000u);
		cb.resize(cb.size() + 8u, 0u);
	}
	AppendAgcMarker(cb, acb, 0xcb000020u | (index & 0x1fu), {});
	cb.push_back(0xc00f1000u);
	std::vector<uint32_t> pad(16, 0u);
	const std::vector<uint32_t> kyty = {kWaitFlipPadMagic, handle, index, static_cast<uint32_t>(watermark), static_cast<uint32_t>(watermark >> 32u),
	                                    0xffffffffu, 0xffffffffu};
	std::copy(kyty.begin(), kyty.end(), pad.begin());
	cb.insert(cb.end(), pad.begin(), pad.end());
	Check(cb.size() - start == 32, "test WaitUntilSafeForRendering is 32 dwords");
}

// SetWorkloadsActive (18 dwords) and SetWorkloadComplete (12 dwords) in libSceAgc's form: the marker, then the ATOMIC_MEM
// on the stream's 64-bit word (a swap with the mask, an AND with all but the workload's bit; engine PFP in a DCB).
void AppendSonyWorkloads(std::vector<uint32_t>& cb, bool acb, uint32_t stream, uint64_t mask, uint32_t id, uint64_t word) {
	const uint32_t engine = acb ? 0u : 1u << 30u;
	AppendAgcMarker(cb, acb, 0xcc000000u | stream, {static_cast<uint32_t>(mask), static_cast<uint32_t>(mask >> 32u)});
	cb.insert(cb.end(), {0xc0071e00u, engine | 0x267u, static_cast<uint32_t>(word), static_cast<uint32_t>(word >> 32u), static_cast<uint32_t>(mask),
	                     static_cast<uint32_t>(mask >> 32u), 0u, 0u, 0u});
	const uint64_t keep = ~(1ull << id);
	AppendAgcMarker(cb, acb, 0xcd000000u | (id << 5u) | stream, {});
	cb.insert(cb.end(), {0xc0071e00u, engine | 0x275u, static_cast<uint32_t>(word), static_cast<uint32_t>(word >> 32u), static_cast<uint32_t>(keep),
	                     static_cast<uint32_t>(keep >> 32u), 0u, 0u, 0u});
}

bool PassesBothChecks(const std::vector<uint32_t>& ib, NativeIbKind kind) {
	using Libs::Graphics::NativeGpu::SetNativeIbValidation;
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	SetNativeIbValidation(false);
	const bool blacklist = ValidateNativeIb(ib, false, nullptr, kind);
	SetNativeIbValidation(true);
	const bool strict = ValidateNativeIb(ib, false, nullptr, kind);
	return blacklist && strict;
}

// SetFlip in libSceAgc's form on either ring: its markers pass, the label the GPU cannot write is the tracker's write
// of 1, and the interrupt that follows it keeps the kernel context id (bit 27) for the host's dispatch.
void TestSonyFlip() {
	static uint64_t labels[16] = {};
	const uint64_t  label      = reinterpret_cast<uint64_t>(&labels[5]);
	constexpr uint32_t kCtx    = 0x08000005u;
	auto label_then_irq = [&](const Pm4TranslateResult& r) {
		Check(r.segments.size() == 1, "one segment");
		const auto& pts   = r.segments[0].points;
		const auto  write = std::find_if(pts.begin(), pts.end(), [&](const TrackerPoint& p) {
			return p.kind == TrackerPoint::Kind::CpuWrite && p.addr == label && p.data == std::vector<uint32_t>({1u, 0u});
		});
		const auto  irq   = std::find_if(pts.begin(), pts.end(), [](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Interrupt; });
		return write != pts.end() && irq != pts.end() && write < irq && irq->ctx == kCtx &&
		       std::none_of(pts.begin(), pts.end(), [](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Flip; });
	};
	for (const bool eop_flips: {false, true}) {
		Pm4Translator t;
		TestOracle    o;
		o.unwritable = label;
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetFlipsAtEop(eop_flips);
		std::vector<uint32_t> dcb;
		AppendSonyFlip(dcb, false, 1u, 5, 1u, 0x1122334455667788ull, label, kCtx);
		auto r = t.Translate(dcb);
		Check(r.ok && r.unknown_ops.empty() && r.stats.interrupts == 1 && r.stats.labels_cpu == 1 && r.cpu_tail.empty(),
		      "a DCB's SetFlip translates: the label on the tracker, the interrupt on the tracker, nothing for Kyty's CP");
		Check(label_then_irq(r), "the label's write of 1, then the interrupt with all 28 bits of the context id");
		const auto pkts = Decode(r.ib);
		bool       tag = false, user_data = false;
		for (const auto& p: pkts) {
			tag       = tag || (p.op == IT_SET_UCONFIG_REG && p.body.size() == 3 && p.body[0] == 0x342u && (p.body[1] >> 24u) == 0xc7u);
			user_data = user_data || (p.op == IT_WRITE_DATA && p.body.size() == 5 && p.body[0] == 0x06010000u && p.body[1] == 0xc343u &&
			                          p.body[3] == 0x55667788u && p.body[4] == 0x11223344u);
		}
		Check(tag && user_data, "the marker and its payload go to the gfx CP as written");
		Check(!HasLabelRelease(pkts, label, 1u), "no GPU release on a label the GPU cannot write");
		Check(PassesBothChecks(r.ib, NativeIbKind::Gfx), "the IB passes both packet checks");
	}
	// A label the GPU may write: the GPU's release (two halves), then the interrupt.
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		std::vector<uint32_t> dcb;
		AppendSonyFlip(dcb, false, 1u, 5, 1u, 0, label, kCtx);
		auto       r    = t.Translate(dcb);
		const auto pkts = Decode(r.ib);
		Check(r.ok && HasLabelRelease(pkts, label, 1u) && HasLabelRelease(pkts, label + 4u, 0u), "the GPU writes a label it can reach");
		const auto& pts = r.segments[0].points;
		Check(std::count_if(pts.begin(), pts.end(), [&](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Interrupt && p.ctx == kCtx; }) == 1,
		      "and the interrupt keeps bit 27");
		Check(PassesBothChecks(r.ib, NativeIbKind::Gfx), "the IB passes both packet checks");
	}
	// A negative index: no label, the interrupt alone.
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		std::vector<uint32_t> dcb;
		AppendSonyFlip(dcb, false, 1u, -1, 0u, 0, 0, 0x08000007u);
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.interrupts == 1 && r.stats.labels_cpu == 0 && r.stats.labels_native == 0, "SetFlip of index -1: the interrupt alone");
		const auto& pts = r.segments[0].points;
		Check(std::any_of(pts.begin(), pts.end(), [](const TrackerPoint& p) { return p.kind == TrackerPoint::Kind::Interrupt && p.ctx == 0x08000007u; }),
		      "with its kernel context id");
	}
	// An ACB's SetFlip (CS_DONE, end of shader) on a compute ring: the markers are dropped, label and interrupt as on gfx.
	{
		Pm4Translator t;
		TestOracle    o;
		o.unwritable = label;
		t.SetComputeQueue(true);
		t.SetComputeRing(true);
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetFenceAddress(kFenceVa);
		std::vector<uint32_t> acb;
		AppendSonyFlip(acb, true, 1u, 5, 1u, 0x1122334455667788ull, label, kCtx);
		auto       r    = t.Translate(acb);
		const auto pkts = Decode(r.ib);
		Check(r.ok && r.unknown_ops.empty() && label_then_irq(r), "an ACB's SetFlip on a compute ring: label, then interrupt");
		Check(FindFirst(pkts, IT_SET_UCONFIG_REG) == nullptr && FindFirst(pkts, IT_WRITE_DATA) == nullptr, "its markers stay off MEC");
		Check(PassesBothChecks(r.ib, NativeIbKind::Compute), "the IB passes both compute packet checks");
	}
}

// WaitUntilSafeForRendering in libSceAgc's form: the GPU does not wait on the label (the flip is a Vulkan copy on the
// same gfx pipe); its padding goes to Kyty's command processor, which waits there.
void TestSonyWaitUntilSafe() {
	static uint64_t labels[16] = {};
	const uint64_t  label      = reinterpret_cast<uint64_t>(&labels[3]);
	labels[3]                  = 1; // on screen: a GPU wait here would block
	for (const bool acb: {false, true}) {
		Pm4Translator t;
		TestOracle    o;
		if (acb) {
			t.SetComputeQueue(true);
			t.SetComputeRing(true);
		}
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetFenceAddress(kFenceVa);
		std::vector<uint32_t> cb;
		AppendSonyWaitUntilSafe(cb, acb, 1u, 3, label, 0x0000000700000042ull);
		Append(cb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, acb);
		auto       r    = t.Translate(cb);
		const auto pkts = Decode(r.ib);
		Check(r.ok && r.unknown_ops.empty() && r.segments.size() == 1 && o.asked == 0 && r.stats.waits_flip == 1 && r.stats.marker_wait_flip == 1,
		      acb ? "an ACB's WaitUntilSafeForRendering: no held segment, the oracle not asked"
		          : "a DCB's WaitUntilSafeForRendering: no held segment, the oracle not asked");
		Check(Count(pkts, 0x93u) == 0 && Count(pkts, 0x3Cu) == 0, "no GPU wait on the label");
		Check(r.cpu_tail.size() == 17 && r.cpu_tail[0] == 0xc00f1000u && r.cpu_tail[1] == kWaitFlipPadMagic && r.cpu_tail[2] == 1u &&
		          r.cpu_tail[3] == 3u && r.cpu_tail[4] == 0x42u && r.cpu_tail[5] == 7u,
		      "the padding NOP, with handle, index and watermark, is Kyty's CP's");
		Check(PassesBothChecks(r.ib, acb ? NativeIbKind::Compute : NativeIbKind::Gfx), "the IB passes both packet checks");
		// Where the GPU reaches the padding, a FlipMark point notes which EOP flips have released; its token in the
		// tail's copy lets the tracker put that order into the padding before Kyty's CP runs it.
		const uint32_t token = r.cpu_tail[1u + kWaitFlipPadToken];
		const auto&    pts   = r.segments[0].points;
		Check(token != 0 && std::count_if(pts.begin(), pts.end(), [&](const TrackerPoint& p) {
			      return p.kind == TrackerPoint::Kind::FlipMark && p.ctx == token;
		      }) == 1,
		      "one FlipMark point, its token in the tail's padding");
		Check(r.cpu_tail[1u + kWaitFlipPadEopMark] == 0xffffffffu && r.cpu_tail[2u + kWaitFlipPadEopMark] == 0xffffffffu,
		      "the order is not set by the translator");
		uint32_t asked = 0;
		Check(SetWaitFlipMarks(r.cpu_tail, [&](uint32_t t) {
			      asked = t;
			      return 0x0000000900000011ull;
		      }) == 1 &&
		          asked == token && r.cpu_tail[1u + kWaitFlipPadEopMark] == 0x11u && r.cpu_tail[2u + kWaitFlipPadEopMark] == 9u,
		      "SetWaitFlipMarks puts the order for the padding's token into it");
		Check(SetWaitFlipMarks(r.cpu_tail, [](uint32_t) { return 0ull; }) == 0 && r.cpu_tail[1u + kWaitFlipPadEopMark] == 0x11u,
		      "and leaves a padding whose order is set");
	}
	// A negative index has no wait; the padding still goes to Kyty's CP, which decides on the index.
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		std::vector<uint32_t> dcb;
		AppendSonyWaitUntilSafe(dcb, false, 1u, -1, 0, 0x42u);
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.waits_flip == 0 && r.cpu_tail.size() == 17 && r.cpu_tail[1] == kWaitFlipPadMagic, "index -1: no wait, the padding");
	}
	// A WAIT_REG_MEM_64 on its own is a guest wait as before.
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetFenceAddress(kFenceVa);
		std::vector<uint32_t> dcb;
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		dcb.insert(dcb.end(), {0xc0079300u, 0x06000113u, static_cast<uint32_t>(label) & ~7u, static_cast<uint32_t>(label >> 32u) & 0x3ffffu, 0u, 0u,
		                       0xffffffffu, 0xffffffffu, 0x40u});
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.waits_flip == 0 && o.asked == 1 && r.segments.size() == 2 && r.segments[1].held, "a plain 64-bit wait is held");
	}
	// Without native labels (shadow mode, Kyty's CP runs the whole buffer) the wait is still left out.
	{
		Pm4Translator         t;
		std::vector<uint32_t> dcb;
		AppendSonyWaitUntilSafe(dcb, false, 1u, 3, label, 0x42u);
		auto r = t.Translate(dcb);
		Check(r.ok && r.stats.waits_flip == 1 && r.cpu_tail.empty() && Count(Decode(r.ib), 0x93u) == 0, "shadow mode: no wait, no tail");
	}
}

// A WaitUntilSafeForRendering's wait, end marker and padding are one group a chunk must not cut (WaitFlipGroupDw); the
// releases of SetFlip in a part of a buffer that does not run still go to Kyty's CP (AppendKernelReleases).
void TestWaitFlipGroupAndKernelReleases() {
	static uint64_t labels[16] = {};
	const uint64_t  label      = reinterpret_cast<uint64_t>(&labels[2]);
	{
		std::vector<uint32_t> dcb;
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		const size_t start = dcb.size();
		AppendSonyWaitUntilSafe(dcb, false, 1u, 2, label, 0x42u);
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		const size_t wait = start + 3u;
		Check(WaitFlipGroupDw(dcb, wait) == 9u + 3u + 17u, "the wait, the end marker and the padding are one group");
		Check(WaitFlipGroupDw(dcb, 0) == 0 && WaitFlipGroupDw(dcb, start) == 0 && WaitFlipGroupDw(dcb, wait + 9u) == 0,
		      "other packets start none");
		Check(WaitFlipGroupDw(std::span<const uint32_t>(dcb).first(wait + 12u), wait) == 0, "nor a wait whose padding is cut off");
		std::vector<uint32_t> neg;
		AppendSonyWaitUntilSafe(neg, false, 1u, -1, 0, 0x42u);
		Check(WaitFlipGroupDw(neg, 3u) == 0, "index -1 has no wait");
		std::vector<uint32_t> acb;
		AppendSonyWaitUntilSafe(acb, true, 1u, 2, label, 0x42u);
		Check(WaitFlipGroupDw(acb, 3u) == 29u, "nor does an ACB's form differ");
	}
	{
		std::vector<uint32_t> dcb;
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		const size_t first = dcb.size();
		AppendSonyFlip(dcb, false, 1u, 2, 1u, 0x1122334455667788ull, label, 0x08000005u);
		// A title's own end-of-pipe interrupt (no kernel context) stays dropped with its part.
		const std::vector<uint32_t> title = {0xc0064900u, 0x06200504u, 0x02010000u, 0u, 0u, 0u, 0u, 0x5u};
		dcb.insert(dcb.end(), title.begin(), title.end());
		const size_t second = dcb.size();
		AppendSonyFlip(dcb, true, 1u, -1, 0u, 0, 0, 0x08000007u);
		std::vector<uint32_t> out = {0xdeadbeefu};
		Check(AppendKernelReleases(dcb, &out) == 2 && out.size() == 17u && out[0] == 0xdeadbeefu, "both SetFlip releases, after what was there");
		Check(std::equal(out.begin() + 1, out.begin() + 9, dcb.begin() + static_cast<std::ptrdiff_t>(first + 10u)) &&
		          std::equal(out.begin() + 9, out.end(), dcb.begin() + static_cast<std::ptrdiff_t>(second + 10u)),
		      "as written");
		std::vector<uint32_t> none;
		Check(AppendKernelReleases(title, &none) == 0 && none.empty(), "a title's release is not a kernel one");
	}
}

// SetWorkloadsActive and SetWorkloadComplete in libSceAgc's form: on the gfx ring the ATOMIC_MEMs run as written where the
// GPU may write the word; otherwise, and on a compute ring, the tracker performs them.
void TestSonyWorkloads() {
	static uint64_t words[32] = {};
	const uint64_t  word      = reinterpret_cast<uint64_t>(&words[4]);
	const uint64_t  mask      = (1ull << 3u) | (1ull << 40u);
	auto atomics = [&](const Pm4TranslateResult& r) {
		std::vector<std::vector<uint32_t>> out;
		for (const auto& s: r.segments) {
			for (const auto& p: s.points) {
				if (p.kind == TrackerPoint::Kind::Atomic && p.addr == word) {
					out.push_back(p.data);
				}
			}
		}
		return out;
	};
	const std::vector<uint32_t> swap  = {(1u << 30u) | 0x267u, static_cast<uint32_t>(mask), static_cast<uint32_t>(mask >> 32u)};
	const std::vector<uint32_t> clear = {(1u << 30u) | 0x275u, ~(1u << 3u), 0xffffffffu};
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		std::vector<uint32_t> dcb;
		AppendSonyWorkloads(dcb, false, 4u, mask, 3u, word);
		auto       r    = t.Translate(dcb);
		const auto pkts = Decode(r.ib);
		Check(r.ok && r.unknown_ops.empty() && Count(pkts, 0x1Eu) == 2 && r.stats.atomics_cpu == 0 && atomics(r).empty(),
		      "a DCB's workload swap and AND run on the gfx ring as written");
		Check(HasRef(r.refs, word, 8, false), "the workload word is made resident");
		Check(PassesBothChecks(r.ib, NativeIbKind::Gfx), "the IB passes both packet checks");
	}
	{
		Pm4Translator t;
		TestOracle    o;
		o.unwritable = word;
		t.SetNativeLabels(&o, kTrackerVa);
		std::vector<uint32_t> dcb;
		AppendSonyWorkloads(dcb, false, 4u, mask, 3u, word);
		auto r = t.Translate(dcb);
		Check(r.ok && Count(Decode(r.ib), 0x1Eu) == 0 && r.stats.atomics_cpu == 2 && atomics(r) == std::vector<std::vector<uint32_t>>({swap, clear}),
		      "on a word the GPU cannot write the tracker performs both, in order");
	}
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetComputeQueue(true);
		t.SetComputeRing(true);
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetFenceAddress(kFenceVa);
		std::vector<uint32_t> acb;
		AppendSonyWorkloads(acb, true, 4u, mask, 3u, word);
		Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
		auto r          = t.Translate(acb);
		const auto pkts = Decode(r.ib);
		const std::vector<uint32_t> acb_swap  = {0x267u, swap[1], swap[2]};
		const std::vector<uint32_t> acb_clear = {0x275u, clear[1], clear[2]};
		Check(r.ok && r.unknown_ops.empty() && Count(pkts, 0x1Eu) == 0 && atomics(r) == std::vector<std::vector<uint32_t>>({acb_swap, acb_clear}),
		      "an ACB's on a compute ring: the tracker performs them");
		Check(FindFirst(pkts, IT_SET_UCONFIG_REG) == nullptr && FindFirst(pkts, IT_WRITE_DATA) == nullptr, "its markers stay off MEC");
		Check(PassesBothChecks(r.ib, NativeIbKind::Compute), "the IB passes both compute packet checks");
	}
	{
		Pm4Translator         t;
		std::vector<uint32_t> dcb;
		AppendSonyWorkloads(dcb, false, 4u, mask, 3u, word);
		auto r = t.Translate(dcb);
		Check(r.ok && Count(Decode(r.ib), 0x1Eu) == 2, "without native labels as written on the gfx ring, as before");
	}
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
	t.SetWaitsVerbatim(true);
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
	const Packet* w64  = FindFirst(pkts, 0x93u);
	Check(r.segments.size() == 1 && w64 != nullptr && w64->body.size() == 8 && w64->body[1] == static_cast<uint32_t>(addr) &&
	          w64->body[3] == 1u && w64->body[5] == 0xffffffffu && w64->body[7] == 4u && FindFirst(pkts, 0x3Cu) == nullptr,
	      "a native 64-bit wait reaches the gfx ring as written (the CP compares all 64 bits, probe waitforms)");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "a guest WAIT_REG_MEM_64 passes the gfx-ring check");

	// A PFP wait (ENGINE_SEL 1) as written too, without our PFP_SYNC_ME.
	std::vector<uint32_t> pfp;
	Append(pfp, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(pfp, 0x3Cu, {3u | (1u << 4u) | (1u << 8u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 0xffffffffu, 10u},
	       false);
	auto       rp    = t.Translate(pfp);
	const auto pkts3 = Decode(rp.ib);
	const Packet* wp = FindFirst(pkts3, 0x3Cu);
	Check(wp != nullptr && wp->body[0] == (3u | (1u << 4u) | (1u << 8u)) && wp->body[5] == 10u && FindFirst(pkts3, 0x42u) == nullptr,
	      "a PFP wait reaches the gfx ring as written, no PFP_SYNC_ME");

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
		Check(plan == LabelOracle::Plan::Native || FindFirst(pkts, 0x93u) == nullptr,
		      "a 64-bit guest wait only where it runs on the GPU as written");
	}

	auto rejects = [&](std::vector<uint32_t> ib, const char* what) {
		while ((ib.size() & 7u) != 0) {
			ib.push_back(0xffff1000u);
		}
		uint32_t bad = ~0u;
		Check(!ValidateNativeIb(ib, false, &bad) && bad == 0u, what);
	};
	rejects({0xc0079300u, 0x03u, 0u, 0u, 1u, 0u, ~0u, 0u, 4u}, "WAIT_REG_MEM_64 on a register rejected");
	rejects({0xc0079300u, 0x53u, 0u, 0u, 1u, 0u, ~0u, 0u, 4u}, "WAIT_REG_MEM_64 with an operation rejected");
	{
		std::vector<uint32_t> ib = {0xc0079300u, 0x113u, 0u, 0u, 1u, 0u, ~0u, 0u, 4u};
		while ((ib.size() & 7u) != 0) {
			ib.push_back(0xffff1000u);
		}
		Check(ValidateNativeIb(ib, false, nullptr, NativeIbKind::Gfx), "a guest PFP WAIT_REG_MEM_64 on memory passes on gfx");
		Check(!ValidateNativeIb(ib, false, nullptr, NativeIbKind::Compute), "and is rejected on a compute ring");
	}
	rejects({0xc0033700u, 0x200u, 0x1000u, 0u, 1u}, "WRITE_DATA without write confirm rejected");
	rejects({0xc0033700u, 0x100000u, 0x1000u, 0u, 1u}, "WRITE_DATA to a register rejected");
	rejects({0xc0064902u, kFlushEvent, 1u << 29u, 0x1000u, 0u, 1u, 0u, 0u}, "RELEASE_MEM with the compute bit rejected");
	rejects({0xc0064900u, 0x528u, 2u << 29u, 0x1000u, 0u, 1u, 0u, 0u}, "a guest-form RELEASE_MEM rejected");
	rejects({0xc0064900u, kFlushEvent, (1u << 29u) | (1u << 24u), 0x1000u, 0u, 1u, 0u, 0u}, "a RELEASE_MEM interrupt alone rejected");
	rejects({0xc0064900u, kFlushEvent, 1u << 29u, 0x1000u, 0u, 1u, 0u, 5u}, "a context id without an interrupt rejected");
	{
		// The tracker's release with its EOP interrupt: INT_SEL 2, the KFD event's id as context id.
		std::vector<uint32_t> ib = {0xc0064900u, kFlushEvent, (1u << 29u) | (2u << 24u), 0x1000u, 0u, 1u, 0u, 5u};
		Check(ValidateNativeIb(ib, false, nullptr, NativeIbKind::Gfx), "a tracker release raising the EOP interrupt passes");
	}
	rejects({0xc0053c00u, 0x4000003u, 0x1000u, 0u, 1u, ~0u, 0x19u}, "a WAIT_REG_MEM on a register rejected");
	{
		std::vector<uint32_t> ib = {0xc0053c00u, 0x4000013u, 0x1000u, 0u, 1u, ~0u, 0x19u, 0xffff1000u};
		Check(ValidateNativeIb(ib, false, nullptr, NativeIbKind::Gfx), "a guest-form WAIT_REG_MEM (cache policy, poll 0x19) passes on gfx");
	}
	rejects({0xc0053c02u, 0x13u, 0x1000u, 0u, 1u, ~0u, 4u}, "WAIT_REG_MEM with the compute bit rejected");
	{
		// A title's NOP: skipped by the gfx CP, not taken on a compute ring.
		std::vector<uint32_t> ib = {0xc0011000u, 0u, 0u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u, 0xffff1000u};
		uint32_t              bad = ~0u;
		Check(ValidateNativeIb(ib, false, nullptr, NativeIbKind::Gfx) && !ValidateNativeIb(ib, false, &bad, NativeIbKind::Compute) &&
		          bad == 0u,
		      "a NOP that is not a filler passes on gfx only");
	}
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

// AgcDcbContextStateOp as libSceAgc writes it: a push and a pop around a register write. The shadow does
// the push and pop (CLEAR_STATE CMD 1 and 2), the COND_EXEC + RELEASE_MEM checks are left out, the
// ATOMIC_MEM flag swaps go as written, and the pop puts the register back.
void TestContextStateOpPackets() {
	static uint32_t flags[2] = {0, 1};
	const uint64_t  fa       = reinterpret_cast<uint64_t>(flags);
	auto            piece    = [&](std::vector<uint32_t>& d, uint32_t op) {
        const uint64_t f = fa + (op == 1 ? 0u : 4u);
        d.insert(d.end(), {0xc0032200u, static_cast<uint32_t>(f), static_cast<uint32_t>(f >> 32u), 0u, 8u});
        d.insert(d.end(), {0xc0064900u, 0x28u | (5u << 8u) | (3u << 25u), (1u << 16u) | (4u << 24u), 0u, 0u, 0u, 0u, 0x08100000u | op});
        d.insert(d.end(), {0xc0071e00u, (1u << 30u) | (2u << 8u) | 0x67u, static_cast<uint32_t>(fa), static_cast<uint32_t>(fa >> 32u),
                           op == 1 ? 1u : 0u, op == 1 ? 0u : 1u, 0u, 0u, 6u});
	};
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x10u, 0x1111u}, false);
	piece(dcb, 1);
	dcb.insert(dcb.end(), {0xc0012800u, 0u, 0x80018001u, 0xc0001200u, 1u});
	Append(dcb, IT_SET_CONTEXT_REG, {0x10u, 0x2222u}, false);
	dcb.insert(dcb.end(), {0xc0012800u, 0u, 0x80018003u});
	piece(dcb, 2);
	dcb.insert(dcb.end(), {0xc0001200u, 2u});
	Pm4Translator t;
	auto          r = t.Translate(dcb);
	Check(r.ok && r.stats.context_ops == 2 && r.stats.context_checks == 2, "push and pop done by the shadow, checks left out");
	uint32_t atomics = 0, last_10 = 0, clear_states = 0;
	for (const auto& p: Decode(r.ib)) {
		atomics += p.op == 0x1Eu ? 1u : 0u;
		clear_states += p.op == IT_CLEAR_STATE && !p.body.empty() && p.body[0] != 0u ? 1u : 0u; // the preamble's has CMD 0
		if (p.op == IT_SET_CONTEXT_REG && p.body.size() == 2 && p.body[0] == 0x10u) {
			last_10 = p.body[1];
		}
	}
	Check(atomics == 2 && clear_states == 0, "the flag swaps go as written, no PUSH_STATE / POP_STATE reaches the GPU");
	Check(last_10 == 0x1111u, "the pop puts the register back");
	Check(ValidateNativeIb(r.ib, false, nullptr, NativeIbKind::Gfx), "the packet check takes it");
}

// libSceAgc's ACB packets without the shader-type bit go to MEC as written (SET_SH_REG, DISPATCH_DIRECT,
// ACQUIRE_MEM, DMA_DATA); the compute-ring check takes them.
void TestComputeRingPlainHeaders() {
	Pm4Translator t;
	t.SetComputeQueue(true);
	t.SetComputeRing(true);
	std::vector<uint32_t> acb;
	Append(acb, IT_SET_SH_REG, {0x240u, 0x11u}, false);
	Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 0x41u}, false);
	auto r = t.Translate(acb);
	Check(r.ok && r.stats.changed_ops[IT_SET_SH_REG] == 0 && r.stats.changed_ops[IT_DISPATCH_DIRECT] == 0 &&
	          std::search(r.ib.begin(), r.ib.end(), acb.begin(), acb.end()) != r.ib.end(),
	      "an ACB's SET_SH_REG and DISPATCH_DIRECT reach MEC as written");
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Compute), "the compute-ring check takes them");
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
	bool wait_plain = false;
	for (const auto& p: pkts) {
		release_cs = release_cs || (p.op == 0x49u && p.compute && IsLabelEvent(p.body[0]));
		wait_plain = wait_plain || (p.op == 0x3Cu && !p.compute && p.body[1] == static_cast<uint32_t>(addr));
	}
	Check(release_cs && wait_plain && r.stats.waits_native == 1,
	      "the label release uses the compute shader bit, the guest wait goes to MEC as written");
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
	Check(gfx_bit && tracker == 2, "gfx timestamp has no shader bit; the first interrupt's tracker packet, then the end's");
	{
		const auto& ip = r.segments[0].points;
		uint32_t    dispatch_dw = 0;
		for (uint32_t dw = 0; dw < r.ib.size(); dw++) {
			if ((r.ib[dw] >> 30u) == 3u && ((r.ib[dw] >> 8u) & 0xffu) == IT_DISPATCH_DIRECT) {
				dispatch_dw = dw;
			}
		}
		Check(ip.size() == 3 && ip[0].kind == TrackerPoint::Kind::Interrupt && ip[1].kind == TrackerPoint::Kind::Interrupt &&
		          ip[0].dw < dispatch_dw && ip[1].dw == ip[2].dw && ip[1].dw > dispatch_dw,
		      "an interrupt fires before the GPU work after its release; the last shares the end packet");
	}
	{
		Pm4Translator late;
		TestOracle    lo;
		late.SetNativeLabels(&lo, kTrackerVa);
		late.SetFenceAddress(kFenceVa);
		late.SetInterruptsAtRelease(false);
		const auto  rl = late.Translate(dcb);
		const auto& lp = rl.segments[0].points;
		Check(rl.ok && lp.size() == 3 && lp[0].dw == lp[2].dw, "off: interrupts with a dispatch between them share the end packet");
	}
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

// Direct runs with native labels: WRITE_DATA blocks the title wrote write-confirmed by the micro engine
// run in place without a copy; a long series becomes an IB2, a short one between packets of ours is
// copied from the buffer at its place; the segment's points still sit at their tracker stores.
void TestDirectRunsWriteData() {
	alignas(8) static uint32_t mem[1024] = {};
	const uint64_t             addr      = reinterpret_cast<uint64_t>(&mem[0]);
	auto write_block = [&](std::vector<uint32_t>& dcb, uint32_t at, uint32_t dwords) {
		std::vector<uint32_t> b = {(5u << 8u) | (1u << 20u), static_cast<uint32_t>(addr + 4u * at), static_cast<uint32_t>(addr >> 32u)};
		for (uint32_t i = 0; i < dwords; i++) {
			b.push_back(i);
		}
		Append(dcb, IT_WRITE_DATA, b, false);
	};
	std::vector<uint32_t> dcb;
	for (uint32_t k = 0; k < 8; k++) {
		write_block(dcb, 64u * k, 60u); // 8 x 63 dwords
	}
	const uint32_t series = static_cast<uint32_t>(dcb.size());
	AppendReleaseMarker(dcb, 1, 2, addr + 4000u, 1, 0x21); // ours: a label with an interrupt
	const uint32_t short_at = static_cast<uint32_t>(dcb.size());
	write_block(dcb, 1000u, 2u);                            // 5 dwords between packets of ours
	const uint32_t short_end = static_cast<uint32_t>(dcb.size());
	AppendReleaseMarker(dcb, 1, 2, addr + 4008u, 2, 0x22);
	constexpr uint64_t kVa = 0x123400000ull;
	Pm4Translator      t;
	TestOracle         o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetDirectRuns(256);
	t.SetSource({{0u, static_cast<uint32_t>(dcb.size()), kVa}});
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.direct_ib2 == 1 && r.stats.direct_dw == series, "the series of blocks is one IB2");
	bool ib2 = false;
	for (size_t k = 0; k + 3 < r.ib.size(); k++) {
		ib2 = ib2 || (r.ib[k] == 0xc0023f00u && r.ib[k + 1] == static_cast<uint32_t>(kVa) && r.ib[k + 3] == (series | (1u << 23u)));
	}
	Check(ib2, "the IB2 points at the series");
	Check(std::search(r.ib.begin(), r.ib.end(), dcb.begin() + short_at, dcb.begin() + short_end) != r.ib.end(),
	      "the short block is copied from the buffer");
	const auto pkts  = Decode(r.ib);
	const auto first = IndexOf(pkts, [&](const Packet& p) { return p.op == 0x49u && p.body[2] == static_cast<uint32_t>(addr + 4000u); });
	const auto block = IndexOf(pkts, [&](const Packet& p) { return p.op == IT_WRITE_DATA && p.body[1] == static_cast<uint32_t>(addr + 4000u); });
	const auto last  = IndexOf(pkts, [&](const Packet& p) { return p.op == 0x49u && p.body[2] == static_cast<uint32_t>(addr + 4008u); });
	Check(first < block && block < last && last < pkts.size(), "the short block stays between the two labels");
	for (const auto& seg: r.segments) {
		for (const auto& p: seg.points) {
			Check(IsTrackerPacket(r.ib, p.dw), "every point sits at a tracker store");
		}
	}
	uint32_t bad = 0;
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "the IB passes the gfx check");
}


void TestLabelWriteBackOnlyForCpu() {
	// A label the same buffer waits on is a GPU sync: its release keeps only the guest's cache actions.
	// One nobody in the buffer waits on keeps the added GLM and GL2 write-backs, so the CPU sees the
	// GPU's results by the time it sees the label. KYTY_NATIVE_GPU_LABEL_WB set picks another mode.
	if (const char* v = std::getenv("KYTY_NATIVE_GPU_LABEL_WB"); v != nullptr && v[0] != '\0') {
		return;
	}
	static uint32_t sync  = 0;
	static uint32_t done  = 0;
	const uint64_t  a     = reinterpret_cast<uint64_t>(&sync);
	const uint64_t  b     = reinterpret_cast<uint64_t>(&done);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 1, 0, a, 1, 0);
	AppendWait(dcb, a, 1);
	Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 1, 0, b, 1, 0);

	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.waits_native == 1 && r.stats.label_wb_stripped == 1, "the waited label's write-back is taken off");
	const auto pkts = Decode(r.ib);
	constexpr uint32_t wb      = (1u << 12u) | (1u << 21u);
	const auto         release = [&](uint64_t addr) {
		return IndexOf(pkts, [&](const Packet& p) {
			return p.op == 0x49u && p.body[1] == (1u << 29u) && p.body[2] == static_cast<uint32_t>(addr) && p.body[4] == 1u;
		});
	};
	const auto ra = release(a);
	const auto rb = release(b);
	Check(ra < pkts.size() && (pkts[ra].body[0] & wb) == 0u && (pkts[ra].body[0] & 0x3fu) == 0x28u,
	      "the GPU-waited label keeps the guest's event without write-backs");
	Check(rb < pkts.size() && (pkts[rb].body[0] & wb) == wb, "the label nobody here waits on keeps the write-backs");
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
	// A ring two guest queues share (the AGR queue in queue 0's gfx ring): another ring's writer may itself
	// wait for the other guest queue's work behind this waiter, so the wait stays held until it lands.
	Check(PlanRingWait(1, true, true, false, true, false, true, false, true) == RingWaitRelease::Hold,
	      "shared ring, other ring's writer submitted: hold until it lands");
	Check(PlanRingWait(1, true, true, true, false, false, true, false, true) == RingWaitRelease::GpuWait,
	      "shared ring, writer ahead in the same ring: real GPU wait");
	Check(PlanRingWait(0, false, false, false, false, true, true, false, true) == RingWaitRelease::Barrier,
	      "shared ring, the other ring's write landed and memory holds it: a barrier");
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

// SetExitGuard: a stop leaves the title's GPU work where it stands. On the gfx ring every segment starts
// with COND_EXEC on the abort word over a chain to the empty IB, and every guest wait the GPU gets is
// followed by one (an ME wait by PFP_SYNC_ME first: the PFP decides a COND_EXEC ahead of the ME). On a
// compute ring each guard skips to the next one or the segment's end on the run word.
void TestExitGuards() {
	constexpr uint64_t kAbort = 0x400000fff808ull, kRun = 0x400000fff80cull, kStub = 0x400000ffffe0ull;
	static uint32_t    label = 0;
	const uint64_t     addr  = reinterpret_cast<uint64_t>(&label);
	auto guards_at = [&](const std::vector<uint32_t>& ib) {
		std::vector<uint32_t> at;
		for (uint32_t dw = 0; dw + 9 <= ib.size(); dw++) {
			if (ib[dw] == Pkt3(0x22u, 4, false) && ib[dw + 1] == static_cast<uint32_t>(kAbort) && ib[dw + 4] == 4u &&
			    ib[dw + 5] == Pkt3(0x3Fu, 3, false) && ib[dw + 6] == static_cast<uint32_t>(kStub) &&
			    ib[dw + 8] == (8u | (1u << 20u) | (1u << 23u))) {
				at.push_back(dw);
			}
		}
		return at;
	};
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetWaitsVerbatim(true);
		t.SetFenceAddress(kFenceVa);
		t.SetExitGuard(kAbort, kRun, kStub);
		o.plan = LabelOracle::Plan::Native;
		std::vector<uint32_t> dcb;
		Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
		AppendWait(dcb, addr, 1);
		Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
		Append(dcb, 0x3Cu, {3u | (1u << 4u) | (1u << 8u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 2u, 0xffffffffu, 4u},
		       false);
		Append(dcb, IT_DISPATCH_DIRECT, {3u, 1u, 1u, 1u}, false);
		auto r = t.Translate(dcb);
		Check(r.ok && r.segments.size() == 1, "guarded buffer translated");
		const auto at = guards_at(r.ib);
		Check(at.size() == 3 && at[0] == r.segments[0].begin, "a guard opens the segment and one follows each GPU wait");
		Check(r.ib[at[1] - 2] == ((3u << 30u) | (0x42u << 8u)) && r.ib[at[1] - 9] == Pkt3(0x3Cu, 6, false),
		      "an ME wait, then PFP_SYNC_ME, then its guard");
		Check(r.ib[at[2] - 7] == Pkt3(0x3Cu, 6, false) && r.ib[at[2] - 6] == (3u | (1u << 4u) | (1u << 8u)), "a PFP wait's guard follows it at once");
		Check(r.segments[0].gpu_waits.size() == 2 && r.segments[0].gpu_waits[0].ref == 1u && r.segments[0].gpu_waits[1].ref == 2u,
		      "the segment names the waits it gives the GPU, in order");
		uint32_t bad = 0;
		Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "guarded IB passes the gfx-ring check");

		// A held wait's segment opens with a guard too, and its native wait is followed by one.
		o.plan  = LabelOracle::Plan::Split;
		auto rs = t.Translate(dcb);
		Check(rs.segments.size() >= 2, "held segments");
		const auto ats = guards_at(rs.ib);
		for (const auto& seg: rs.segments) {
			Check(std::find(ats.begin(), ats.end(), seg.begin) != ats.end(), "every segment starts with a guard");
		}
		Check(rs.segments[1].held && std::find(ats.begin(), ats.end(), rs.segments[1].wait_dw + rs.segments[1].wait_len + 2u) != ats.end(),
		      "the held wait is followed by PFP_SYNC_ME and a guard");
		Check(ValidateNativeIb(rs.ib, false, &bad, NativeIbKind::Gfx), "held guarded IB passes the gfx-ring check");

		// Without SetExitGuard nothing changes.
		Pm4Translator plain;
		plain.SetNativeLabels(&o, kTrackerVa);
		o.plan = LabelOracle::Plan::Native;
		Check(guards_at(plain.Translate(dcb).ib).empty(), "no guards unless asked for");
	}
	{
		Pm4Translator t;
		TestOracle    o;
		t.SetComputeQueue(true);
		t.SetComputeRing(true);
		t.SetNativeLabels(&o, kTrackerVa);
		t.SetFenceAddress(kFenceVa);
		t.SetExitGuard(kAbort, kRun, kStub);
		o.plan = LabelOracle::Plan::Native;
		std::vector<uint32_t> acb;
		Append(acb, IT_SET_SH_REG, {0x240u, 0x11u}, true);
		AppendWait(acb, addr, 1);
		for (uint32_t i = 0; i < 2000; i++) { // 8000 dwords: a spacer guard every few thousand
			Append(acb, IT_SET_SH_REG, {0x241u, i, i}, true);
		}
		Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
		auto r = t.Translate(acb);
		Check(r.ok && r.segments.size() == 1, "guarded compute buffer translated");
		const auto& g = r.segments[0].guards;
		Check(g.size() >= 4 && g[0] == r.segments[0].begin + 4u, "a compute segment opens with a guard, spacers follow");
		for (size_t i = 0; i < g.size(); i++) {
			Check(r.ib[g[i] - 4u] == Pkt3(0x22u, 4, true) && r.ib[g[i] - 3u] == static_cast<uint32_t>(kRun), "a compute guard is COND_EXEC on the run word");
			const uint32_t until = i + 1 < g.size() ? g[i + 1] - 4u : r.segments[0].end;
			Check(r.ib[g[i]] == until - g[i] - 1u && r.ib[g[i]] <= Libs::Graphics::NativeGpu::kExitGuardMaxSkip,
			      "each compute guard skips exactly to the next guard or the segment's end");
		}
		Check(FindFirst(Decode(r.ib), 0x42u) == nullptr && FindFirst(Decode(r.ib), 0x3Fu) == nullptr, "no PFP_SYNC_ME and no chain on MEC");
		uint32_t bad = 0;
		Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Compute), "guarded compute IB passes the compute-ring check");

		// The splitter keeps a COND_EXEC with what it skips.
		using Libs::Graphics::NativeGpu::SplitIbAtPackets;
		const auto parts = SplitIbAtPackets(r.ib, 6000);
		for (const auto& [from, len]: parts) {
			for (uint32_t gi: g) {
				Check(!(gi - 4u >= from && gi - 4u < from + len && gi + 1u + r.ib[gi] > from + len), "no part ends inside a guard's skip");
			}
		}
	}
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
	// libSceAgc's one-dword NOP (AgcCbNop(buf, 1): count 0x3fff), as its queue resets begin, is one dword.
	std::vector<uint32_t> filler = {0xffff1000u, 0xc0001000u, 0x2222u};
	buf                          = ib(filler, false);
	out.clear();
	Check(FlattenIndirectBuffers(buf, &out) && out == filler, "a called buffer that starts with the one-dword NOP");

	// libSceAgc's branch with compare function 0 (AgcCbBranch, 14 dwords; Stray's calls): its then target is
	// called. A branch on memory (function 3) is not inlined.
	auto branch = [](const std::vector<uint32_t>& target, uint32_t function) {
		const auto va = reinterpret_cast<uint64_t>(target.data());
		return std::vector<uint32_t> {0xc00c3f00u, 1u | (function << 8u), 0x1000u, 0u, 0u, 0u, 0u, 0u, static_cast<uint32_t>(va),
		                              static_cast<uint32_t>(va >> 32u), static_cast<uint32_t>(target.size()), 0u, 0u, 0u};
	};
	buf = branch(called, 0);
	buf.insert(buf.end(), nop_b.begin(), nop_b.end());
	Check(FlattenIndirectBuffers(buf, &out) && out == std::vector<uint32_t>({0xc0001000u, 0x1111u, 0xc0001000u, 0xbbbbu}),
	      "an always-taken branch calls its then target");
	buf = branch(called, 3);
	out.clear();
	Check(!FlattenIndirectBuffers(buf, &out), "a branch on memory is not inlined");

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

	// Where each stretch of the flattened buffer sits in memory: before the call, the call, after it.
	buf = nop_a;
	const auto call = ib(called, false);
	buf.insert(buf.end(), call.begin(), call.end());
	buf.insert(buf.end(), nop_b.begin(), nop_b.end());
	buf.insert(buf.end(), nop_b.begin(), nop_b.end());
	std::vector<Libs::Graphics::NativeGpu::SourcePiece> pieces;
	const auto                                          va = [](const uint32_t* p) { return reinterpret_cast<uint64_t>(p); };
	Check(FlattenIndirectBuffers(buf, &out, nullptr, &pieces) && out.size() == 8 && pieces.size() == 3 &&
	          pieces[0].dw == 0 && pieces[0].len == 2 && pieces[0].va == va(buf.data()) && pieces[1].dw == 2 &&
	          pieces[1].len == 2 && pieces[1].va == va(called.data()) && pieces[2].dw == 4 && pieces[2].len == 4 &&
	          pieces[2].va == va(buf.data() + 6),
	      "source pieces of a flattened call");
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

// CP register shadowing (Pm4Translator::SetCpShadow).
using Libs::Graphics::NativeGpu::CpShadowBatch;
using Libs::Graphics::NativeGpu::CpShadowMode;
using Libs::Graphics::NativeGpu::CpShadowPiece;
using Libs::Graphics::NativeGpu::CpShadowTailOk;
using Libs::Graphics::NativeGpu::EmitCpShadowOff;
using Libs::Graphics::NativeGpu::kCpShadowBytes;
using Libs::Graphics::NativeGpu::kCpShadowContextDw;
using Libs::Graphics::NativeGpu::kCpShadowDwords;
using Libs::Graphics::NativeGpu::kCpShadowOff;
using Libs::Graphics::NativeGpu::kCpShadowShDw;
using Libs::Graphics::NativeGpu::kCpShadowUconfigDw;
using Libs::Graphics::NativeGpu::SetCpShadowValidation;

constexpr uint64_t kTestShadowVa = 0x400003F80000ull; // where the backend puts S (NativeSubmitter::CpShadowVa)
constexpr uint32_t IT_PFP_SYNC_ME = 0x42;
constexpr uint32_t IT_LOAD_UC     = 0x5E;
constexpr uint32_t IT_LOAD_SH     = 0x5F;
constexpr uint32_t IT_LOAD_CX     = 0x61;

using Regs = std::vector<std::pair<uint32_t, uint32_t>>;
// libSceAgc's start state (Pm4Translator::LoadStartState) of a test.
void Start(Pm4Translator& t, const Regs& cx, const Regs& sh, const Regs& uc) {
	t.LoadStartState(cx, sh, uc);
}

// A toy gfx CP for CP shadowing: the registers of the three classes, S at kTestShadowVa in Mesa's layout,
// CONTEXT_CONTROL's load and shadow enables, and a shadow base per class (context, UCONFIG, gfx SH, CS SH) that a
// LOAD with that class's loading on sets. Every job starts without bases, which is stricter than the board, and
// a shadowed write of a class without a base fails the test (on the board it goes near VA 0: a VM fault).
// CLEAR_STATE CMD 0 resets the context to a fake clear state; any other command fails (it never reaches the GPU).
// A COND_EXEC is false (no stop armed): the exit guards' chains are skipped.
struct ToyCp {
	enum Cls { kCtx = 0, kUc = 1, kGsh = 2, kCsh = 3 };
	std::array<uint32_t, 0x400> cx {};
	std::array<uint32_t, 0x400> sh {};
	std::array<uint32_t, 0x400> uc {};
	std::vector<uint32_t>       s = std::vector<uint32_t>(Libs::Graphics::NativeGpu::kCpShadowReserveBytes / 4u, 0u); // S, then P
	uint32_t                    load   = 0;
	uint32_t                    shadow = 0;
	std::array<bool, 4>         base {};
	uint32_t                    loads = 0; // LOADs that loaded
	uint32_t                    dmas  = 0; // DMA_DATA copies

	bool InS(uint64_t va) const { return va >= kTestShadowVa && va < kTestShadowVa + 4ull * s.size(); }
	static uint32_t Bit(int c) { return c == kCtx ? 1u << 1u : c == kUc ? 1u << 15u : c == kGsh ? 1u << 16u : 1u << 24u; }
	static uint32_t Region(int c) { return c == kCtx ? kCpShadowContextDw : c == kUc ? kCpShadowUconfigDw : kCpShadowShDw; }
	static uint32_t Csb(uint32_t off) { return 0xc5b00000u | off; }
	std::array<uint32_t, 0x400>& Regs(int c) { return c == kCtx ? cx : c == kUc ? uc : sh; }
	void Write(int c, uint32_t off, uint32_t v) {
		Check(off < 0x400u, "toy CP: a register below 0x400");
		Regs(c)[off] = v;
		if ((shadow & Bit(c)) != 0) {
			Check(base[c], "a shadowed write has its class's base in this job (else a VM fault near VA 0)");
			s[Region(c) + off] = v;
		}
	}
	void Run(std::span<const uint32_t> ib) {
		for (size_t dw = 0; dw < ib.size();) {
			const uint32_t h = ib[dw];
			Check((h >> 30u) == 3u, "toy CP: a type-3 packet");
			const uint32_t size = PacketSizeDw(h);
			Check(dw + size <= ib.size(), "toy CP: the packet fits");
			const uint32_t  op = (h >> 8u) & 0xffu;
			const uint32_t* b  = ib.data() + dw + 1;
			const uint32_t  n  = size - 1u;
			const bool      cs = (h & 2u) != 0;
			switch (op) {
				case IT_CONTEXT_CONTROL:
					load   = (b[0] >> 31u) != 0 ? b[0] & 0x7fffffffu : load;
					shadow = (b[1] >> 31u) != 0 ? b[1] & 0x7fffffffu : shadow;
					break;
				case IT_CLEAR_STATE:
					Check(n >= 1 && b[0] == 0u, "toy CP: only CLEAR_STATE 0 reaches the GPU");
					for (uint32_t i = 0; i < 0x400u; i++) {
						cx[i] = Csb(i); // S untouched (probe shadow-pushpop)
					}
					break;
				case IT_SET_CONTEXT_REG:
					for (uint32_t i = 1; i < n; i++) {
						Write(kCtx, (b[0] & 0xffffu) + i - 1u, b[i]);
					}
					break;
				case IT_SET_SH_REG:
					for (uint32_t i = 1; i < n; i++) {
						Write(cs ? kCsh : kGsh, (b[0] & 0xffffu) + i - 1u, b[i]); // the class follows the header (probe shadow)
					}
					break;
				case IT_SET_UCONFIG_REG:
				case IT_SET_UCONFIG_REG_INDEX:
					for (uint32_t i = 1; i < n; i++) {
						Write(kUc, (b[0] & 0xffffu) + i - 1u, b[i]);
					}
					break;
				case IT_SET_CONTEXT_REG_INDIRECT:
				case IT_SET_SH_REG_INDIRECT:
				case IT_SET_UCONFIG_REG_INDIRECT: {
					const uint64_t at    = (static_cast<uint64_t>(b[1]) << 32u) | (b[0] & ~3u);
					const auto*    pairs = InS(at) ? s.data() + (at - kTestShadowVa) / 4u : reinterpret_cast<const uint32_t*>(at); // a table of ours in S's area
					const int   c     = op == IT_SET_CONTEXT_REG_INDIRECT ? kCtx : op == IT_SET_UCONFIG_REG_INDIRECT ? kUc : (cs ? kCsh : kGsh);
					for (uint32_t k = 0; k < (b[3] & 0x3fffu); k++) {
						Write(c, pairs[2 * k] & 0xffffu, pairs[2 * k + 1]); // read when the CP gets here
					}
					break;
				}
				case IT_LOAD_UC:
				case IT_LOAD_SH:
				case IT_LOAD_CX: {
					const int c = op == IT_LOAD_CX ? kCtx : op == IT_LOAD_UC ? kUc : (cs ? kCsh : kGsh);
					if ((load & Bit(c)) == 0) {
						break; // loading off: neither loads nor moves the base (probe shadow, SHADOW_LOAD=0)
					}
					const uint64_t va = b[0] | (static_cast<uint64_t>(b[1]) << 32u);
					Check(va == kTestShadowVa + 4ull * Region(c), "toy CP: a LOAD from its class's region of S");
					base[c] = true;
					loads++;
					for (uint32_t i = 2; i + 1 < n; i += 2) {
						for (uint32_t j = 0; j < b[i + 1]; j++) {
							Check(b[i] + j < 0x400u, "toy CP: a LOAD range below 0x400");
							Regs(c)[b[i] + j] = s[Region(c) + b[i] + j];
						}
					}
					break;
				}
				case 0x3Fu: // INDIRECT_BUFFER: an IB2 call runs and comes back, a chain ends this IB
					if ((b[2] & (1u << 20u)) != 0) {
						return;
					}
					Run(std::span<const uint32_t>(reinterpret_cast<const uint32_t*>((static_cast<uint64_t>(b[1]) << 32u) | b[0]),
					                              b[2] & 0xfffffu));
					break;
				case 0x50u: { // DMA_DATA: memory to memory, only inside S and P here
					const uint64_t src = b[1] | (static_cast<uint64_t>(b[2]) << 32u);
					const uint64_t dst = b[3] | (static_cast<uint64_t>(b[4]) << 32u);
					const uint32_t len = b[5] & 0x3ffffffu;
					Check(src >= kTestShadowVa && dst >= kTestShadowVa && src + len <= kTestShadowVa + 4ull * s.size() &&
					          dst + len <= kTestShadowVa + 4ull * s.size(),
					      "toy CP: a DMA_DATA inside S and P");
					std::copy_n(s.begin() + static_cast<std::ptrdiff_t>((src - kTestShadowVa) / 4u), len / 4u,
					            s.begin() + static_cast<std::ptrdiff_t>((dst - kTestShadowVa) / 4u));
					dmas++;
					break;
				}
				case 0x1Eu: { // ATOMIC_MEM: the display-DCC fix's AND on a word in S's area
					const uint64_t at = b[1] | (static_cast<uint64_t>(b[2]) << 32u);
					Check((b[0] & 0x7fu) == 0x55u && InS(at), "toy CP: an ATOMIC_MEM AND in S's area");
					s[(at - kTestShadowVa) / 4u] &= b[3];
					break;
				}
				case 0x22u: { // COND_EXEC: on a word in S's area (the FMASK fix) as the CP does; an exit guard's abort
				             // word is 0 (no stop armed), so the dwords it guards are skipped
					const uint64_t at = b[0] | (static_cast<uint64_t>(b[1]) << 32u);
					if (!InS(at) || s[(at - kTestShadowVa) / 4u] == 0u) {
						dw += b[3];
					}
					break;
				}
				default: break;
			}
			dw += size;
		}
	}
	// One job of ours: no bases at its start, shadowing off at its end.
	void Job(std::span<const uint32_t> ib) {
		base = {};
		Run(ib);
		Check(shadow == 0u, "a job ends with shadowing off (else another process's SETs land at our VA)");
	}
	// gamescope between our jobs: RADV's preamble, then its own state.
	void Foreign() {
		std::vector<uint32_t> ib = {0xc0012800u, 0x80000000u, 0x80000000u, 0xc0001200u, 0u};
		Append(ib, IT_SET_CONTEXT_REG, {0x10Fu, 0xdead0001u}, false);
		Append(ib, IT_SET_SH_REG, {0x00Cu, 0xdead0002u}, false);
		Append(ib, IT_SET_SH_REG, {0x240u, 0xdead0003u}, true);
		Append(ib, IT_SET_UCONFIG_REG, {0x24Au, 0xdead0004u}, false);
		Job(ib);
	}
	// S as the backend seeds it when CP shadowing comes on.
	void Seed(Pm4Translator& t) {
		std::fill(s.begin(), s.end(), 0u);
		t.WriteCpShadowImage(std::span<uint32_t>(s.data(), kCpShadowDwords),
		                     std::span<uint32_t>(s.data() + Libs::Graphics::NativeGpu::kCpShadowSaveOffset / 4u,
		                                         Libs::Graphics::NativeGpu::kCpShadowSaveBytes / 4u));
		const auto fix = Libs::Graphics::NativeGpu::CpShadowFixTables();
		std::copy(fix.begin(), fix.end(), s.begin() + Libs::Graphics::NativeGpu::kCpShadowFixOffset / 4u);
	}
};

// A job from a piece of ours: the piece and the end every gfx job gets.
std::vector<uint32_t> AsJob(std::span<const uint32_t> piece) {
	std::vector<uint32_t> job(piece.begin(), piece.end());
	EmitCpShadowOff(job);
	return job;
}

std::vector<uint32_t> Segment(const Pm4TranslateResult& r, size_t k) {
	return std::vector<uint32_t>(r.ib.begin() + r.segments[k].begin, r.ib.begin() + r.segments[k].end);
}

// The LOAD ranges of the first LOAD of `op` (with the compute header when `cs`) in `pkts`, as (offset, count).
std::vector<std::pair<uint32_t, uint32_t>> LoadRanges(const std::vector<Packet>& pkts, uint32_t op, bool cs) {
	std::vector<std::pair<uint32_t, uint32_t>> out;
	for (const auto& p: pkts) {
		if (p.op == op && p.compute == cs) {
			for (size_t i = 2; i + 1 < p.body.size(); i += 2) {
				out.emplace_back(p.body[i], p.body[i + 1]);
			}
			break;
		}
	}
	return out;
}

bool SetsReg(const std::vector<Packet>& pkts, uint32_t op, uint32_t off) {
	for (const auto& p: pkts) {
		if (p.op == op && !p.body.empty() && (p.body[0] & 0xffffu) <= off && off < (p.body[0] & 0xffffu) + p.body.size() - 1u) {
			return true;
		}
	}
	return false;
}

// The end of every gfx job, and the tail check the submitter runs on each one.
void TestCpShadowExits() {
	Check(kCpShadowOff[0] == Pkt3(IT_CONTEXT_CONTROL, 2, false) && kCpShadowOff[1] == 0x80000000u && kCpShadowOff[2] == 0x80000000u &&
	          std::all_of(kCpShadowOff.begin() + 3, kCpShadowOff.end(), [](uint32_t v) { return v == 0xffff1000u; }),
	      "the postamble, empty trampoline and exit stub: CONTEXT_CONTROL off/off and five fillers");
	std::vector<uint32_t> ib;
	Append(ib, IT_SET_CONTEXT_REG, {0x10u, 1u}, false);
	Check(!CpShadowTailOk(ib) && !CpShadowTailOk({}), "an IB without the postamble is refused");
	EmitCpShadowOff(ib);
	Check(ib.size() == 11 && CpShadowTailOk(ib), "with it, it passes");
	ib.push_back(0xffff1000u);
	Check(CpShadowTailOk(ib), "padding after it too");
	std::vector<uint32_t> chain = {0xc0053c00u, 0x115u, 0x1000u, 0u, 1u, ~0u, 4u, Pkt3(0x3Fu, 3, false), 0x2000u, 0x4000u, 8u | (1u << 20u) | (1u << 23u)};
	Check(CpShadowTailOk(chain), "a stream head's wait and chain passes: its job ends where the stream does");
	std::vector<uint32_t> guard = {Pkt3(0x22u, 4, false), 0x808u, 0x4000u, 0u, 4u, Pkt3(0x3Fu, 3, false), 0x1000u, 0x4000u,
	                               8u | (1u << 20u) | (1u << 23u)};
	Check(!CpShadowTailOk(guard), "an exit guard's chain does not: without a stop the IB goes on behind it");
	std::vector<uint32_t> call = {Pkt3(0x3Fu, 3, false), 0x2000u, 0x4000u, 8u | (1u << 23u)};
	Check(!CpShadowTailOk(call), "nor does an IB2 call");
}

// The shadow preamble: its packets in order, the LOAD runs from the copy's keys, what is SET instead, the anchors,
// and the packet check.
// Gated load mode: no CLEAR_STATE, the load control left as amdgpu set it for the job (bit 31 clear) and shadowing on,
// then the LOADs, the context's taking every register a CLEAR_STATE resets; a forced preamble (after an AGR buffer) is
// the full one once.
// The default packet check, a blacklist: the board's known killers refused, anything else as the title wrote it.
void TestValidateBlacklist() {
	using Libs::Graphics::NativeGpu::SetNativeIbValidation;
	SetNativeIbValidation(false);
	auto ok = [](std::vector<uint32_t> ib, NativeIbKind kind) { return ValidateNativeIb(ib, false, nullptr, kind); };
	Check(ok({Pkt3(0x35u, 4, false), 1u, 0u, 3u, 0u}, NativeIbKind::Gfx) && ok({0xc0032200u, 0x1000u, 0x2u, 0u, 4u}, NativeIbKind::Gfx) &&
	          ok({Pkt3(0x2Au, 1, false), 0u}, NativeIbKind::Gfx),
	      "a draw, a title's COND_EXEC and a packet the strict check never saw pass");
	Check(!ok({Pkt3(0x2Du, 2, false), 3u, 0u}, NativeIbKind::Compute) && !ok({Pkt3(0x3Fu, 3, true), 0x1000u, 0x2u, 8u | (1u << 20u) | (1u << 23u)}, NativeIbKind::Compute),
	      "a draw or an IB on MEC");
	Check(!ok({Pkt3(0x98u, 1, false), 0u}, NativeIbKind::Gfx) && !ok({Pkt3(0xA2u, 1, false), 0u}, NativeIbKind::Gfx) &&
	          !ok({Pkt3(0x68u, 2, false), 0x10u, 0u}, NativeIbKind::Gfx),
	      "kernel and queue-management packets");
	Check(!ok({Pkt3(0x37u, 4, false), 0u, 0x2c00u, 0u, 1u}, NativeIbKind::Gfx) && !ok({Pkt3(0x40u, 5, false), 0u | (5u << 8u), 0x2c00u, 0u, 0x1000u, 0x2u}, NativeIbKind::Gfx) &&
	          !ok({Pkt3(0x3Cu, 6, false), 3u, 0x2c00u, 0u, 1u, ~0u, 4u}, NativeIbKind::Gfx),
	      "register writes, register reads and register waits");
	Check(!ok({Pkt3(0x45u, 8, false), 0x10u, 0x1000u, 0x2u, 1u, ~0u, 0x2c00u, 0u, 1u}, NativeIbKind::Gfx) &&
	          !ok({Pkt3(0x45u, 8, false), 0x100u, 0x2c00u, 0u, 1u, ~0u, 0x1000u, 0x2u, 1u}, NativeIbKind::Gfx) &&
	          ok({Pkt3(0x45u, 8, false), 0x110u, 0x1000u, 0x2u, 1u, ~0u, 0x2000u, 0x2u, 1u}, NativeIbKind::Gfx),
	      "COND_WRITE: a register write or poll refused, memory to memory (libSceAgc's form) passes");
	Check(!ok({Pkt3(0x3Fu, 3, false), 0u, 0u, 8u | (1u << 23u)}, NativeIbKind::Gfx), "an IB at VA 0");
	Check(!ok({Pkt3(0x28u, 2, false), 0x81018002u, 0x81018002u}, NativeIbKind::Gfx), "shadowing enables with no shadow buffer");
	Check(ok({Pkt3(0x3Cu, 6, false), 3u | (1u << 4u), 0x1000u, 0x2u, 1u, ~0u, 4u}, NativeIbKind::Compute), "a title's wait on MEC as written");
	SetNativeIbValidation(true);
}

void TestCpShadowGatedPreamble() {
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	t.SetCpShadowGatedLoads(true);
	std::vector<uint32_t> draw;
	Append(draw, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto       r    = t.Translate(draw);
	const auto pkts = Decode(r.ib);
	Check(r.ok && r.stats.cp_shadow_gated == 1 && pkts.size() >= 6 && pkts[0].op == IT_PFP_SYNC_ME && pkts[1].op == IT_CONTEXT_CONTROL &&
	          pkts[1].body == std::vector<uint32_t>({0u, 0x81018002u}) && Count(pkts, IT_CLEAR_STATE) == 0 && pkts[2].op == IT_LOAD_UC &&
	          pkts[3].op == IT_LOAD_CX,
	      "gated: PFP_SYNC_ME, shadowing on with amdgpu's load control, no CLEAR_STATE, then the LOADs");
	SetCpShadowValidation(kTestShadowVa);
	Check(ValidateNativeIb(r.ib, false, nullptr, NativeIbKind::Gfx), "the gated preamble passes the gfx packet check");
	SetCpShadowValidation(0);
	using Ranges = std::vector<std::pair<uint32_t, uint32_t>>;
	const Ranges cx = LoadRanges(pkts, IT_LOAD_CX, false);
	uint32_t     n  = 0;
	for (const auto& [first, count]: cx) {
		n += count;
	}
	Check(!cx.empty() && cx.front() == std::make_pair(0u, 0xd7u) && n > 0x300u, "the context LOAD takes amdgpu's clear-state registers");
	std::vector<uint32_t> s(kCpShadowDwords, 0xdeadbeefu), save(Libs::Graphics::NativeGpu::kCpShadowSaveBytes / 4u, 0u);
	t.WriteCpShadowImage(s, save);
	Check(s[kCpShadowContextDw + 0x08eu] == 0xffffffffu && s[kCpShadowContextDw + 0x000u] == 0u && s[kCpShadowContextDw + 0x0b5u] == 0x3f800000u,
	      "S starts from amdgpu's clear state (CB_TARGET_MASK, DB_RENDER_CONTROL, a viewport's ZMAX)");
	t.ForceCpShadowLoad();
	const auto pf = Decode(t.Translate(draw).ib);
	Check(Count(pf, IT_CLEAR_STATE) == 1 && pf[0].op == IT_CONTEXT_CONTROL && pf[0].body == std::vector<uint32_t>({0x80000000u, 0x80000000u}),
	      "forced: the full preamble once");
	const auto pn = Decode(t.Translate(draw).ib);
	Check(Count(pn, IT_CLEAR_STATE) == 0, "then gated again");
}

void TestCpShadowPreamble() {
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	// A buffer from an empty copy: every class LOADs its anchor.
	std::vector<uint32_t> draw;
	Append(draw, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto r0 = t.Translate(draw);
	auto p0 = Decode(r0.ib);
	Check(r0.ok && r0.cp_shadow && r0.stats.cp_shadow_anchors == 4 && LoadRanges(p0, IT_LOAD_UC, false) == decltype(LoadRanges(p0, 0, false)) {{0x24Au, 1u}} &&
	          LoadRanges(p0, IT_LOAD_CX, false) == decltype(LoadRanges(p0, 0, false)) {{0x080u, 1u}} &&
	          LoadRanges(p0, IT_LOAD_SH, false) == decltype(LoadRanges(p0, 0, false)) {{0x00Cu, 1u}} &&
	          LoadRanges(p0, IT_LOAD_SH, true) == decltype(LoadRanges(p0, 0, false)) {{0x240u, 1u}},
	      "a class without a key LOADs an anchor, so it has a base");
	Start(t, {{0x10u, 1u}, {0x11u, 2u}, {0x12u, 3u}, {0x20u, 4u}, {0x310u, 0x5u}},
	                 {{0x00Cu, 5u}, {0x00Du, 6u}, {0x00Eu, 7u}, {0x00Fu, 8u}, {0x007u, 9u}, {0x240u, 10u}, {0x216u, 0xffffu}, {0x280u, 1u}},
	                 {{0x242u, 4u}, {0x24Au, 0u}, {0x24Bu, 1u}, {0x25Bu, 2u}});
	auto r = t.Translate(draw);
	Check(r.ok && r.cp_shadow && r.stats.cp_shadow_preambles == 1 && r.stats.cp_shadow_anchors == 0, "a shadow preamble");
	const auto pkts = Decode(r.ib);
	Check(pkts.size() >= 9 && pkts[0].op == IT_CONTEXT_CONTROL && pkts[0].body == std::vector<uint32_t>({0x80000000u, 0x80000000u}) &&
	          pkts[1].op == IT_CLEAR_STATE && pkts[1].body[0] == 0u && pkts[2].op == IT_PFP_SYNC_ME && pkts[3].op == IT_CONTEXT_CONTROL &&
	          pkts[3].body == std::vector<uint32_t>({0x81018002u, 0x81018002u}) && pkts[4].op == IT_LOAD_UC && !pkts[4].compute &&
	          pkts[5].op == IT_LOAD_CX && pkts[6].op == IT_LOAD_SH && !pkts[6].compute && pkts[7].op == IT_LOAD_SH && pkts[7].compute,
	      "shadowing off, CLEAR_STATE, PFP_SYNC_ME, the enables, then the four LOADs back to back");
	Check(pkts[4].body[0] == static_cast<uint32_t>(kTestShadowVa + 0x9000u) && pkts[4].body[1] == 0x4000u &&
	          pkts[5].body[0] == static_cast<uint32_t>(kTestShadowVa + 0x1000u) && pkts[6].body[0] == static_cast<uint32_t>(kTestShadowVa) &&
	          pkts[7].body[0] == static_cast<uint32_t>(kTestShadowVa),
	      "each LOAD names its class's region of S");
	using Ranges = std::vector<std::pair<uint32_t, uint32_t>>;
	Check(LoadRanges(pkts, IT_LOAD_UC, false) == Ranges({{0x24Au, 2u}, {0x25Bu, 1u}}) && LoadRanges(pkts, IT_LOAD_CX, false) == Ranges({{0x10u, 3u}, {0x20u, 1u}}) &&
	          LoadRanges(pkts, IT_LOAD_SH, false) == Ranges({{0x00Cu, 4u}}) && LoadRanges(pkts, IT_LOAD_SH, true) == Ranges({{0x240u, 1u}}),
	      "the keys in runs, SH split at 0x200, the registers S does not restore left out");
	Check(r.stats.cp_shadow_load_regs == 3 + 4 + 4 + 1 && r.stats.cp_shadow_load_dw == 7 + 7 + 5 + 5, "LOAD counts");
	Check(!SetsReg(pkts, IT_SET_CONTEXT_REG, 0x10u) && !SetsReg(pkts, IT_SET_SH_REG, 0x00Cu) && !SetsReg(pkts, IT_SET_SH_REG, 0x240u) &&
	          !SetsReg(pkts, IT_SET_UCONFIG_REG, 0x24Au),
	      "no SET of a register it LOADs");
	bool rsrc3 = false, mgmt = false, nop = false, prim = false, sc = false;
	size_t first_set = pkts.size();
	for (size_t i = 0; i < pkts.size(); i++) {
		const auto& p = pkts[i];
		if (p.op == IT_SET_CONTEXT_REG || p.op == IT_SET_SH_REG || p.op == IT_SET_UCONFIG_REG || p.op == IT_SET_UCONFIG_REG_INDEX) {
			first_set = std::min(first_set, i);
		}
		rsrc3 = rsrc3 || (p.op == IT_SET_SH_REG && !p.compute && p.body == std::vector<uint32_t>({0x007u, 9u}));
		mgmt  = mgmt || (p.op == IT_SET_SH_REG && p.compute && p.body == std::vector<uint32_t>({0x216u, 0xffffu}));
		nop   = nop || (p.op == IT_SET_SH_REG && p.compute && p.body == std::vector<uint32_t>({0x280u, 1u}));
		prim  = prim || (p.op == IT_SET_UCONFIG_REG_INDEX && p.body == std::vector<uint32_t>({0x10000242u, 4u}));
		sc    = sc || (p.op == IT_SET_CONTEXT_REG && p.body == std::vector<uint32_t>({0x310u, 5u}));
	}
	Check(rsrc3 && mgmt && nop && prim && sc && first_set > 7, "RSRC3, STATIC_THREAD_MGMT, SH 0x280, the INDEX form and "
	                                                         "PA_SC_SHADER_CONTROL are SET, after the LOADs");
	Check(r.stats.preamble_cx == 1 && r.stats.preamble_sh == 3 && r.stats.preamble_uc == 1, "only those are counted as SETs");
	// The packet check: LOADs and the enables only from the registered S, on the gfx ring.
	uint32_t bad = 0;
	Check(!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx) && ((r.ib[bad] >> 8u) & 0xffu) == IT_CONTEXT_CONTROL,
	      "no shadow buffer registered: the shadowing CONTEXT_CONTROL is refused");
	SetCpShadowValidation(kTestShadowVa);
	Check(ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx), "registered: the shadow preamble passes on the gfx ring");
	Check(!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Compute), "and not on a compute ring");
	SetCpShadowValidation(kTestShadowVa + 0x100000u);
	Check(!ValidateNativeIb(r.ib, false, &bad, NativeIbKind::Gfx) && ((r.ib[bad] >> 8u) & 0xffu) == IT_LOAD_UC, "nor from another S");
	SetCpShadowValidation(kTestShadowVa);
	auto rejects = [&](std::vector<uint32_t> ib, const char* what) {
		uint32_t at = ~0u;
		Check(!ValidateNativeIb(ib, false, &at, NativeIbKind::Gfx) && at == 0u, what);
	};
	const uint32_t sh_lo = static_cast<uint32_t>(kTestShadowVa), cx_lo = sh_lo + 0x1000u;
	rejects({Pkt3(IT_LOAD_CX, 4, false), cx_lo, 0x4000u, 0x3F0u, 0x20u}, "a LOAD range past 0x400 is refused");
	rejects({Pkt3(IT_LOAD_CX, 4, false), cx_lo, 0x4000u, 0x10u, 0u}, "an empty range is refused");
	rejects({Pkt3(IT_LOAD_CX, 2, false), cx_lo, 0x4000u}, "a LOAD without ranges is refused");
	rejects({Pkt3(IT_LOAD_SH, 4, true), sh_lo, 0x4000u, 0x00Cu, 1u}, "LOAD_SH with the compute header below 0x200 is refused");
	rejects({Pkt3(IT_LOAD_SH, 4, false), sh_lo, 0x4000u, 0x1F0u, 0x20u}, "LOAD_SH with the graphics header past 0x200 is refused");
	rejects({Pkt3(IT_LOAD_CX, 4, true), cx_lo, 0x4000u, 0x10u, 1u}, "LOAD_CONTEXT_REG with the compute header is refused");
	rejects({Pkt3(IT_CONTEXT_CONTROL, 2, false), 0x80018003u, 0x80018003u}, "any other CONTEXT_CONTROL is refused");
	SetCpShadowValidation(0);
}

// S's seed: every key below 0x400 at its place, with the values the IB writes.
void TestCpShadowImage() {
	Pm4Translator t;
	t.SetNoCmaskClear(true, 0x10u); // MRT4 without DCC loses FAST_CLEAR
	Start(t, {{0x358u, (1u << 13u) | 0x40u}, {0x10u, 3u}}, {{0x00Cu, 5u}, {0x241u, 6u}}, {{0x24Au, 7u}});
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x410u, 9u}, false); // above the dense range
	Check(t.Translate(dcb).ok, "a buffer with a context register above 0x400");
	std::vector<uint32_t> s(kCpShadowDwords, 0u);
	t.WriteCpShadowImage(s);
	Check(s[kCpShadowContextDw + 0x358u] == 0x40u && s[kCpShadowContextDw + 0x10u] == 3u, "context at S + 0x1000, PatchContext applied");
	Check(s[kCpShadowShDw + 0x00Cu] == 5u && s[kCpShadowShDw + 0x241u] == 6u && s[kCpShadowUconfigDw + 0x24Au] == 7u,
	      "SH at S + 0, compute SH in the same region, UCONFIG at S + 0x9000");
	// Besides the keys, the context region holds amdgpu's clear state: 85 registers it sets to other than 0, none of
	// them a key here.
	uint32_t nonzero = 0, context = 0;
	for (uint32_t i = 0; i < s.size(); i++) {
		const bool cx = i >= kCpShadowContextDw && i < kCpShadowContextDw + 0x400u;
		nonzero += s[i] != 0 && !cx ? 1u : 0u;
		context += s[i] != 0 && cx ? 1u : 0u;
	}
	Check(nonzero == 3 && context == 2 + 85, "nothing else, offsets of 0x400 and up included");
}

// The Demon's Souls mechanism: a verbatim table the title fills after the buffer was translated. The CP runs
// what the table holds by then; the next buffer's restore takes it from S in load mode, and today's replay of
// the copy (record and off) puts the translate-time value back.
void TestCpShadowLateTable() {
	for (const CpShadowMode mode: {CpShadowMode::Load, CpShadowMode::Record, CpShadowMode::Off}) {
		Pm4Translator t;
		t.SetCpShadow(mode == CpShadowMode::Off ? 0u : kTestShadowVa, mode);
		Start(t, {{0x10Fu, 0x11u}}, {}, {});
		ToyCp cp;
		cp.Seed(t);
		std::vector<uint32_t> pairs = {0x10Fu, 0xAu}; // PA_CL_VPORT_XSCALE = A when translated
		std::vector<uint32_t> dcb1;
		AppendIndirect(dcb1, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
		Append(dcb1, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
		auto r1 = t.Translate(dcb1);
		Check(r1.ok && r1.stats.loads_verbatim == 1, "the table runs as written");
		pairs[1] = 0xBu; // filled behind a GPU gate, after the translation
		cp.Job(AsJob(r1.ib));
		Check(cp.cx[0x10F] == 0xBu, "the CP draws with what the table holds when it runs");
		Check(mode == CpShadowMode::Off || cp.s[kCpShadowContextDw + 0x10Fu] == 0xBu, "and shadows it");
		cp.Foreign();
		std::vector<uint32_t> dcb2;
		Append(dcb2, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
		auto r2 = t.Translate(dcb2);
		cp.Job(AsJob(r2.ib));
		Check(cp.cx[0x10F] == (mode == CpShadowMode::Load ? 0xBu : 0xAu),
		      mode == CpShadowMode::Load ? "load mode: the next buffer restores the GPU's value from S"
		                                 : "record and off: the copy's replay restores the stale value, as today");
	}
}

// The AGR queue in queue 0's gfx ring, as Shadow sets it up (CreateAgrTranslatorLocked): a copy of the DCB
// translator made before CP shadowing comes on, which stays queue 0's. Between queue 0's jobs the AGR queue's
// run with shadowing off; each queue draws with its own registers, S keeps queue 0's.
void TestAgrQueueSharesRing0() {
	Pm4Translator main;
	auto          agr = main.Clone();
	main.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	agr->SetCpShadow(0, CpShadowMode::Off);
	agr->SetPreambleFromStart(true);
	for (Pm4Translator* t: {&main, agr.get()}) {
		Start(*t, {{0x10Fu, 0x11u}}, {{0x00Cu, 0x21u}}, {{0x24Au, 0x31u}});
	}
	ToyCp cp;
	cp.Seed(main);
	std::vector<uint32_t> set_main;
	Append(set_main, IT_SET_CONTEXT_REG, {0x10Fu, 0xAu}, false);
	Append(set_main, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	std::vector<uint32_t> set_agr;
	Append(set_agr, IT_SET_CONTEXT_REG, {0x10Fu, 0x7u}, false);
	Append(set_agr, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	std::vector<uint32_t> draw;
	Append(draw, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	const auto r1 = main.Translate(set_main);
	const auto a1 = agr->Translate(set_agr);
	Check(r1.ok && r1.cp_shadow && a1.ok && !a1.cp_shadow, "queue 0 translates with shadow preambles, the AGR queue without");
	cp.Job(AsJob(r1.ib));
	Check(cp.cx[0x10F] == 0xAu && cp.s[kCpShadowContextDw + 0x10Fu] == 0xAu, "queue 0 draws with its register, and S holds it");
	cp.Job(AsJob(a1.ib));
	Check(cp.cx[0x10F] == 0x7u && cp.s[kCpShadowContextDw + 0x10Fu] == 0xAu, "the AGR queue draws with its own, S keeps queue 0's");
	cp.Job(AsJob(main.Translate(draw).ib));
	Check(cp.cx[0x10F] == 0xAu, "queue 0's next buffer restores its register from S");
	cp.Job(AsJob(agr->Translate(draw).ib));
	Check(cp.cx[0x10F] == 0x7u && cp.s[kCpShadowContextDw + 0x10Fu] == 0xAu, "the AGR queue's next buffer restores its own");
	// With native labels, as the backend runs it, its preamble holds the registers where the buffer starts: a SET
	// after the first draw is not that draw's.
	TestOracle o;
	agr->SetNativeLabels(&o, kTrackerVa);
	std::vector<uint32_t> late_set;
	Append(late_set, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	Append(late_set, IT_SET_CONTEXT_REG, {0x10Fu, 0x9u}, false);
	Append(late_set, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	const auto a3          = agr->Translate(late_set);
	uint32_t   first_draws = 0;
	for (const auto& p: Decode(a3.ib)) {
		if (p.op == IT_DRAW_INDEX_AUTO) {
			break;
		}
		const uint32_t start = p.body.empty() ? 0u : p.body[0] & 0xffffu;
		if (p.op == IT_SET_CONTEXT_REG && start <= 0x10Fu && 0x10Fu < start + p.body.size() - 1u) {
			first_draws = p.body[1u + 0x10Fu - start];
		}
	}
	Check(a3.ok && !a3.segments.empty() && first_draws == 0x7u,
	      "the AGR queue's first draw of a buffer has the value it starts with, not the one SET after it");
}

// Record mode: the CP keeps S, the copy still restores every register as before.
void TestCpShadowRecord() {
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Record);
	Start(t, {{0x10u, 1u}, {0x11u, 2u}}, {{0x00Cu, 5u}, {0x240u, 6u}}, {{0x24Au, 7u}});
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x12u, 3u}, false);
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto       r    = t.Translate(dcb);
	const auto pkts = Decode(r.ib);
	Check(r.ok && FindFirst(pkts, IT_LOAD_CX) != nullptr && SetsReg(pkts, IT_SET_CONTEXT_REG, 0x10u) && SetsReg(pkts, IT_SET_SH_REG, 0x00Cu) &&
	          SetsReg(pkts, IT_SET_SH_REG, 0x240u) && SetsReg(pkts, IT_SET_UCONFIG_REG, 0x24Au),
	      "the LOADs, then today's replay of every key");
	// Today's state choice: a whole buffer's head replays the state where it ends.
	Check(r.stats.preamble_cx == 3 && LoadRanges(pkts, IT_LOAD_CX, false) == decltype(LoadRanges(pkts, 0, false)) {{0x10u, 3u}},
	      "a whole buffer: the end state, as today");
	ToyCp cp;
	cp.Seed(t);
	cp.Job(AsJob(r.ib));
	Check(cp.cx[0x10] == 1u && cp.cx[0x12] == 3u && cp.sh[0x240] == 6u && cp.uc[0x24A] == 7u && cp.s[kCpShadowContextDw + 0x12u] == 3u,
	      "the registers end where they do today, and S follows");
}

// Held segments restore the keys at the cut from S.
void TestCpShadowHeldSegment() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	static uint32_t       label = 0;
	std::vector<uint32_t> pairs = {0x00Cu, 0xAu};
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_SH_REG_INDIRECT, pairs, 0x80000000u);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendWait(dcb, reinterpret_cast<uint64_t>(&label), 1);
	Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
	ToyCp cp;
	cp.Seed(t);
	auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 2 && r.segments[0].cp_shadow_head && r.segments[1].held && r.segments[1].cp_shadow_head,
	      "both segments start with a shadow preamble");
	const auto held = Decode(Segment(r, 1));
	Check(LoadRanges(held, IT_LOAD_SH, false) == decltype(LoadRanges(held, 0, false)) {{0x00Cu, 1u}}, "the held one LOADs the keys at the cut");
	pairs[1] = 0xBu;
	cp.Job(AsJob(Segment(r, 0)));
	cp.Foreign();
	cp.Job(AsJob(Segment(r, 1)));
	Check(cp.sh[0x00C] == 0xBu, "after another process ran, the held segment restores the GPU's value");
}

// A first segment that only writes registers stays: a LOAD from S would not see its writes otherwise.
void TestCpShadowKeepsFirstSegment() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	static uint32_t       label = 0;
	std::vector<uint32_t> first;
	Append(first, IT_SET_SH_REG, {0x240u, 0x33u}, true);
	AppendWait(first, reinterpret_cast<uint64_t>(&label), 1);
	Append(first, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	ToyCp cp;
	cp.Seed(t);
	auto r = t.Translate(first);
	Check(r.ok && r.segments.size() == 2 && !r.segments[0].held && r.segments[1].held, "the register-only first segment is kept");
	cp.Job(AsJob(Segment(r, 0)));
	cp.Foreign();
	cp.Job(AsJob(Segment(r, 1)));
	Check(cp.sh[0x240] == 0x33u, "and its write reaches the GPU and S");
}

// A streamed chunk has no preamble; its resume is the one it would have had, and the batch puts it in front only
// where the chunk does not carry on from a job of ours with shadowing on.
void TestCpShadowResume() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	Start(t, {{0x10u, 1u}}, {{0x00Cu, 2u}}, {{0x24Au, 3u}});
	static uint32_t       label = 0;
	std::vector<uint32_t> chunk;
	Append(chunk, IT_SET_CONTEXT_REG, {0x11u, 4u}, false);
	Append(chunk, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	AppendWait(chunk, reinterpret_cast<uint64_t>(&label), 1);
	Append(chunk, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	t.SetPreambleFromStart(true);
	t.SetNoPreamble(true);
	auto r = t.Translate(chunk);
	t.SetNoPreamble(false);
	t.SetPreambleFromStart(false);
	const auto seg0 = Decode(Segment(r, 0));
	Check(r.ok && r.segments.size() == 2 && !r.segments[0].cp_shadow_head && FindFirst(seg0, IT_CONTEXT_CONTROL) == nullptr &&
	          r.segments[1].cp_shadow_head,
	      "no preamble in the first segment, the held one keeps its own");
	const auto res = Decode(r.cp_shadow_resume);
	Check(res.size() == 8 && res[3].op == IT_CONTEXT_CONTROL && LoadRanges(res, IT_LOAD_CX, false) == decltype(LoadRanges(res, 0, false)) {{0x10u, 1u}},
	      "the resume restores the state where the chunk starts");
	Check((r.cp_shadow_resume.size() & 7u) == 0, "in whole 8-dword units, as the pieces it goes in front of");
	Check(r.stats.cp_shadow_preambles == 1, "only the held segment's preamble is in the IB");

	// The batch rules.
	const std::vector<uint32_t> head = AsJob(std::span<const uint32_t>(r.cp_shadow_resume)); // an own piece with a head
	const std::vector<uint32_t> foreign = {0xc0012800u, 0x80000000u, 0x80000000u, 0xc0001200u, 0u, Pkt3(IT_SET_SH_REG, 2, true), 0x241u, 0xdeadu};
	const std::vector<uint32_t> piece   = Segment(r, 0);
	const CpShadowPiece         own_head {true, true, {}};
	const CpShadowPiece         own_cont {true, false, r.cp_shadow_resume};
	const CpShadowPiece         other {false, false, {}};
	{
		CpShadowBatch         b;
		std::vector<uint32_t> batch;
		b.Append(batch, head, own_head);
		b.Append(batch, foreign, other);
		Check(batch.size() == head.size() + 8 + foreign.size() && std::equal(kCpShadowOff.begin(), kCpShadowOff.end(), batch.begin() + head.size()) &&
		          b.guards == 1,
		      "shadowing off in front of another translator's piece");
		const size_t at = batch.size();
		b.Append(batch, piece, own_cont);
		Check(batch.size() == at + r.cp_shadow_resume.size() + piece.size() && b.resumes == 1 && b.EndOn(false) &&
		          std::equal(r.cp_shadow_resume.begin(), r.cp_shadow_resume.end(), batch.begin() + at),
		      "and the resume in front of a chunk of ours after it");
		Check(b.Front(false, false).empty(), "nothing kept for the flush");
		ToyCp cp;
		cp.Seed(t);
		cp.Job(AsJob(batch));
		Check(cp.sh[0x241] == 0xdeadu && cp.s[kCpShadowShDw + 0x241u] == 0u && cp.cx[0x11] == 4u && cp.cx[0x10] == 1u,
		      "on the CP: the foreign piece's SET is not shadowed, the chunk runs with its state restored");
	}
	{
		CpShadowBatch         b;
		std::vector<uint32_t> batch;
		b.Append(batch, piece, own_cont);
		Check(batch.size() == piece.size() && b.Extra(own_head) == r.cp_shadow_resume.size(), "at the front the resume is kept");
		Check(b.Front(true, true).empty(), "a stream part that carries on with shadowing on needs none");
		Check(b.Front(true, false).size() == r.cp_shadow_resume.size() && b.Front(false, false).size() == r.cp_shadow_resume.size(),
		      "after a stream break, or as a CS of its own, the resume goes first");
		std::vector<uint32_t> job(b.Front(false, false).begin(), b.Front(false, false).end());
		job.insert(job.end(), batch.begin(), batch.end());
		ToyCp cp;
		cp.Seed(t);
		cp.Job(AsJob(job));
		Check(cp.cx[0x11] == 4u, "a CS that starts with a chunk of ours runs it with every base set");
		b.Reset();
		Check(b.Empty() && b.Front(false, false).empty(), "a flushed batch keeps nothing");
	}
	{
		CpShadowBatch         b;
		std::vector<uint32_t> batch;
		b.Append(batch, foreign, other);
		Check(batch.size() == 8 + foreign.size() && !b.EndOn(true), "a foreign piece at the front gets shadowing off first");
		b.HoldsForeign();
		b.Append(batch, foreign, other);
		Check(batch.size() == 8 + 2 * foreign.size(), "but not again while it is off");
	}
}

// After a title's CLEAR_STATE 0 the keys are the defaults: a register outside them keeps the clear state.
void TestCpShadowClearState() {
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load, Libs::Graphics::NativeGpu::kCpShadowClasses, true);
	Start(t, {{0x10u, 1u}}, {}, {});
	ToyCp cp;
	cp.Seed(t);
	std::vector<uint32_t> dcb1;
	Append(dcb1, IT_SET_CONTEXT_REG, {0x30u, 5u}, false);
	dcb1.insert(dcb1.end(), {0xc0001200u, 0u});
	Append(dcb1, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto r1 = t.Translate(dcb1);
	const auto p1 = Decode(r1.ib);
	size_t clear = 0, rearm = 0;
	for (size_t i = 0; i < p1.size(); i++) {
		if (p1[i].op == IT_CLEAR_STATE) {
			clear = i; // the title's is the last one
		}
	}
	rearm = clear + 1;
	Check(r1.ok && rearm < p1.size() && p1[rearm].op == IT_CONTEXT_CONTROL && p1[rearm].body == std::vector<uint32_t>({0x81018002u, 0x81018002u}),
	      "KYTY_NATIVE_GPU_CP_SHADOW_REARM: the enables again right after the title's CLEAR_STATE");
	cp.Job(AsJob(r1.ib));
	Check(cp.s[kCpShadowContextDw + 0x30u] == 5u && cp.cx[0x30] == ToyCp::Csb(0x30u), "the clear resets the register, S keeps the SET");
	cp.Foreign();
	std::vector<uint32_t> dcb2;
	Append(dcb2, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto r2 = t.Translate(dcb2);
	Check(LoadRanges(Decode(r2.ib), IT_LOAD_CX, false) == decltype(LoadRanges(p1, 0, false)) {{0x10u, 1u}}, "the next restore LOADs the defaults only");
	cp.Job(AsJob(r2.ib));
	Check(cp.cx[0x30] == ToyCp::Csb(0x30u) && cp.cx[0x10] == 1u, "so the cleared register stays at the clear state, as today");
}

// A compute ring never shadows: MEC has no CONTEXT_CONTROL.
void TestCpShadowComputeRing() {
	Pm4Translator t;
	t.SetComputeQueue(true);
	t.SetComputeRing(true);
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	std::vector<uint32_t> acb;
	Append(acb, IT_SET_SH_REG, {0x240u, 0x11u}, true);
	Append(acb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, true);
	t.Translate(acb);
	auto       r    = t.Translate(acb);
	const auto pkts = Decode(r.ib);
	Check(r.ok && !r.cp_shadow && t.CpShadow() == CpShadowMode::Off && FindFirst(pkts, IT_CONTEXT_CONTROL) == nullptr &&
	          FindFirst(pkts, IT_LOAD_SH) == nullptr && r.stats.preamble_sh == 1,
	      "the compute-ring translator keeps its SH replay");
}

// KYTY_NATIVE_GPU_CP_SHADOW_CLASSES: a class left out is replayed from the copy after the LOADs (which it still gets).
void TestCpShadowClasses() {
	using namespace Libs::Graphics::NativeGpu;
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load, kCpShadowGfxSh | kCpShadowCsSh | kCpShadowUconfig);
	Start(t, {{0x10u, 1u}, {0x11u, 2u}}, {{0x00Cu, 5u}}, {{0x24Au, 7u}});
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto       r    = t.Translate(dcb);
	const auto pkts = Decode(r.ib);
	Check(r.ok && FindFirst(pkts, IT_LOAD_CX) != nullptr && SetsReg(pkts, IT_SET_CONTEXT_REG, 0x10u) && SetsReg(pkts, IT_SET_CONTEXT_REG, 0x11u) &&
	          !SetsReg(pkts, IT_SET_SH_REG, 0x00Cu) && !SetsReg(pkts, IT_SET_UCONFIG_REG, 0x24Au),
	      "context from the replay, the rest from S");
}

// The checker: S against the copy, by who last wrote each register.
void TestCpShadowCompare() {
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Record);
	Start(t, {{0x20u, 2u}}, {{0x007u, 3u}}, {});
	std::vector<uint32_t> pairs = {0x10Fu, 0xAu};
	std::vector<uint32_t> dcb;
	AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
	Append(dcb, IT_SET_SH_REG, {0x007u, 4u}, false);
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	ToyCp cp;
	cp.Seed(t);
	auto r = t.Translate(dcb);
	pairs[1] = 0xBu; // late
	cp.Job(AsJob(r.ib));
	std::vector<uint32_t> s(cp.s.begin(), cp.s.begin() + kCpShadowDwords);
	s[kCpShadowShDw + 0x007u]      = 3u;          // as if the CP did not shadow the SET of RSRC3
	s[kCpShadowContextDw + 0x20u]  = 0x99u;       // a register the translator saw no write of since the seed
	s[kCpShadowContextDw + 0x150u] = 0x77u;       // one only a late table wrote
	s[kCpShadowUconfigDw + 0x24Eu] = 0x1234u;     // the tess ring the preamble writes
	const auto c = t.CompareCpShadow(s, nullptr);
	Check(c.late == 1 && c.unshadowed == 1 && c.unseen == 1 && c.outside == 1 && c.keys == 3 && c.equal == 0,
	      "a late table, a write not shadowed, an unseen write and a register outside the keys");
	std::vector<uint32_t> same(cp.s.begin(), cp.s.begin() + kCpShadowDwords);
	same[kCpShadowContextDw + 0x10Fu] = 0xAu;
	const auto c2 = t.CompareCpShadow(same, nullptr);
	Check(c2.keys == 3 && c2.equal == 3 && c2.outside == 0, "S that matches the copy");
}

// Load mode: a whole buffer's head restores the state where the buffer starts, as on the PS5, not the
// registers the same buffer sets later (today's replay of the end state).
void TestCpShadowStartKeys() {
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	Start(t, {{0x20u, 2u}}, {}, {});
	ToyCp cp;
	cp.Seed(t);
	std::vector<uint32_t> dcb1;
	Append(dcb1, IT_SET_CONTEXT_REG, {0x10u, 0x11u}, false);
	Append(dcb1, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto r1 = t.Translate(dcb1);
	using Ranges = std::vector<std::pair<uint32_t, uint32_t>>;
	Check(r1.ok && LoadRanges(Decode(r1.ib), IT_LOAD_CX, false) == Ranges({{0x20u, 1u}}), "the first buffer LOADs the start keys only");
	cp.Job(AsJob(r1.ib));
	cp.Foreign();
	std::vector<uint32_t> dcb2;
	Append(dcb2, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	Append(dcb2, IT_SET_CONTEXT_REG, {0x30u, 0x33u}, false);
	Append(dcb2, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	auto r2 = t.Translate(dcb2);
	Check(r2.ok && LoadRanges(Decode(r2.ib), IT_LOAD_CX, false) == Ranges({{0x10u, 1u}, {0x20u, 1u}}),
	      "the next one LOADs the keys the previous buffer left, not 0x30 it sets itself");
	cp.Job(AsJob(r2.ib));
	Check(cp.cx[0x10] == 0x11u && cp.cx[0x20] == 2u && cp.cx[0x30] == 0x33u && cp.s[kCpShadowContextDw + 0x30u] == 0x33u,
	      "after another process ran: the earlier registers from S, the new one from its SET");
}

// libSceAgc's push/pop stays on the CPU with CP shadowing: no PUSH_STATE / POP_STATE reaches the GPU, the only
// CONTEXT_CONTROLs are the preamble's, and the pop's SET is shadowed like any other.
void TestCpShadowPushPopCpu() {
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	static uint32_t flags[2] = {0, 1};
	const uint64_t  fa       = reinterpret_cast<uint64_t>(flags);
	auto            piece    = [&](std::vector<uint32_t>& d, uint32_t op) {
        const uint64_t f = fa + (op == 1 ? 0u : 4u);
        d.insert(d.end(), {0xc0032200u, static_cast<uint32_t>(f), static_cast<uint32_t>(f >> 32u), 0u, 8u});
        d.insert(d.end(), {0xc0064900u, 0x28u | (5u << 8u) | (3u << 25u), (1u << 16u) | (4u << 24u), 0u, 0u, 0u, 0u, 0x08100000u | op});
	};
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_CONTEXT_REG, {0x10u, 0x1111u}, false);
	piece(dcb, 1);
	dcb.insert(dcb.end(), {0xc0012800u, 0u, 0x80018001u, 0xc0001200u, 1u});
	Append(dcb, IT_SET_CONTEXT_REG, {0x10u, 0x2222u}, false);
	dcb.insert(dcb.end(), {0xc0012800u, 0u, 0x80018003u});
	piece(dcb, 2);
	dcb.insert(dcb.end(), {0xc0001200u, 2u});
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 0u}, false);
	Pm4Translator t;
	t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
	Start(t, {{0x10u, 0u}}, {}, {});
	ToyCp cp;
	cp.Seed(t);
	auto r = t.Translate(dcb);
	Check(r.ok && r.stats.context_ops == 2, "push and pop done on the CPU");
	uint32_t controls = 0;
	for (const auto& p: Decode(r.ib)) {
		if (p.op == IT_CONTEXT_CONTROL) {
			controls++;
			Check(p.body == std::vector<uint32_t>({0x80000000u, 0x80000000u}) || p.body == std::vector<uint32_t>({0x81018002u, 0x81018002u}),
			      "only the preamble's two CONTEXT_CONTROLs, never the title's");
		}
	}
	Check(controls == 2, "one shadow preamble");
	SetCpShadowValidation(kTestShadowVa);
	Check(ValidateNativeIb(r.ib, false, nullptr, NativeIbKind::Gfx), "the packet check takes it");
	SetCpShadowValidation(0);
	cp.Job(AsJob(r.ib)); // the toy CP fails on a CLEAR_STATE other than 0
	Check(cp.cx[0x10] == 0x1111u && cp.s[kCpShadowContextDw + 0x10u] == 0x1111u, "the pop's SET reaches the register and S");
}

// libSceAgc's push and pop on the GPU through S (SetCpShadowPushPop): the packets, the Demon's Souls case (a late
// table inside the window that the CPU pop cannot see), a window across buffers and another process, a register
// first written inside the window, a pop whose push came before shadowing, and the packet check of the DMA_DATAs.
void TestCpShadowPushPopGpu() {
	using Libs::Graphics::NativeGpu::kCpShadowDmaCommand;
	using Libs::Graphics::NativeGpu::kCpShadowDmaControl;
	using Libs::Graphics::NativeGpu::kCpShadowSaveOffset;
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	constexpr uint32_t IT_DMA_DATA = 0x50;
	const uint64_t     cx_va       = kTestShadowVa + 0x1000u;
	const uint64_t     p_va        = kTestShadowVa + kCpShadowSaveOffset;
	auto push = [](std::vector<uint32_t>& d) { d.insert(d.end(), {0xc0001200u, 1u}); };
	auto pop  = [](std::vector<uint32_t>& d) { d.insert(d.end(), {0xc0001200u, 2u}); };
	auto draw = [](std::vector<uint32_t>& d) { Append(d, IT_DRAW_INDEX_AUTO, {3u, 0u}, false); };
	auto dma  = [](const Packet& p, uint64_t from, uint64_t to) {
		return p.op == IT_DMA_DATA && !p.compute &&
		       p.body == std::vector<uint32_t>({kCpShadowDmaControl, static_cast<uint32_t>(from), static_cast<uint32_t>(from >> 32u),
		                                        static_cast<uint32_t>(to), static_cast<uint32_t>(to >> 32u), kCpShadowDmaCommand});
	};
	auto make = [&](bool gpu) {
		auto t = std::make_unique<Pm4Translator>();
		t->SetCpShadow(kTestShadowVa, CpShadowMode::Load);
		t->SetCpShadowPushPop(gpu);
		Start(*t, {{0x10Fu, 0xAu}, {0x20u, 2u}}, {}, {});
		return t;
	};

	// One buffer: push, a table that sets the viewport to what the copy already holds (A when translated, B when
	// it runs), a draw, pop, a draw. The CPU pop sees no difference and leaves B; the GPU pop restores A.
	for (const bool gpu: {true, false}) {
		auto                  t = make(gpu);
		ToyCp                 cp;
		cp.Seed(*t);
		std::vector<uint32_t> pairs = {0x10Fu, 0xAu};
		std::vector<uint32_t> dcb;
		push(dcb);
		AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
		draw(dcb);
		pop(dcb);
		draw(dcb);
		auto r = t->Translate(dcb);
		Check(r.ok && r.stats.loads_verbatim == 1, "the table runs as written");
		pairs[1] = 0xBu;
		const auto pkts = Decode(r.ib);
		if (gpu) {
			Check(r.stats.cp_shadow_pushes == 1 && r.stats.cp_shadow_pops == 1, "one GPU push and one GPU pop");
			size_t i = 0;
			while (i < pkts.size() && pkts[i].op != IT_DMA_DATA) {
				i++;
			}
			Check(i >= 1 && i + 1 < pkts.size() && pkts[i - 1].op == IT_PFP_SYNC_ME && dma(pkts[i], cx_va, p_va),
			      "push: PFP_SYNC_ME, DMA_DATA of S's context region to P");
			size_t j = i + 1;
			while (j < pkts.size() && pkts[j].op != IT_DMA_DATA) {
				j++;
			}
			Check(j + 4 < pkts.size() && pkts[j - 1].op == IT_PFP_SYNC_ME && dma(pkts[j], p_va, cx_va) && pkts[j + 1].op == IT_PFP_SYNC_ME &&
			          pkts[j + 2].op == IT_CONTEXT_CONTROL && pkts[j + 2].body == std::vector<uint32_t>({0x81018002u, 0x81018002u}) &&
			          pkts[j + 3].op == IT_LOAD_CX && pkts[j + 3].body[0] == static_cast<uint32_t>(cx_va),
			      "pop: PFP_SYNC_ME, DMA_DATA of P back, PFP_SYNC_ME, the enables, LOAD_CONTEXT_REG from S");
			const std::vector<std::pair<uint32_t, uint32_t>> ranges(LoadRanges({pkts[j + 3]}, IT_LOAD_CX, false));
			Check(ranges == std::vector<std::pair<uint32_t, uint32_t>>({{0x20u, 1u}, {0x10Fu, 1u}}), "over the registers S has a value for");
			bool set_after = false;
			for (size_t k = j + 4; k < pkts.size(); k++) {
				set_after = set_after || pkts[k].op == IT_SET_CONTEXT_REG;
			}
			Check(!set_after, "and no CPU diff SET");
			SetCpShadowValidation(kTestShadowVa);
			Check(ValidateNativeIb(r.ib, false, nullptr, NativeIbKind::Gfx), "the packet check takes it");
			SetCpShadowValidation(0);
		}
		cp.Job(AsJob(r.ib));
		// The CPU pop in load mode SETs a register a verbatim table wrote even when the copy sees no change.
		Check(cp.cx[0x10F] == 0xAu, gpu ? "GPU pop: the pushed viewport comes back" : "load mode's CPU pop: SET again, as the copy has it");
		Check(!gpu || (cp.dmas == 2 && cp.s[kCpShadowContextDw + 0x10Fu] == 0xAu && cp.s[kCpShadowSaveOffset / 4u + 0x10Fu] == 0xAu),
		      "two copies: P holds the pushed value, and S holds it again after the pop");
	}

	// A second push inside a window does nothing; a pop without a push does nothing.
	{
		auto                  t = make(true);
		std::vector<uint32_t> dcb;
		pop(dcb);
		push(dcb);
		Append(dcb, IT_SET_CONTEXT_REG, {0x10Fu, 0xCu}, false);
		push(dcb);
		pop(dcb);
		draw(dcb);
		auto     r    = t->Translate(dcb);
		uint32_t dmas = 0;
		for (const auto& p: Decode(r.ib)) {
			dmas += p.op == IT_DMA_DATA ? 1u : 0u;
		}
		Check(r.ok && dmas == 2 && r.stats.cp_shadow_pushes == 1 && r.stats.cp_shadow_pops == 1,
		      "a pop with nothing pushed and a second push emit nothing");
		ToyCp cp;
		auto  t2 = make(true);
		cp.Seed(*t2);
		auto r2 = t2->Translate(dcb);
		cp.Job(AsJob(r2.ib));
		Check(cp.cx[0x10F] == 0xAu, "the first pushed state comes back");
	}

	// A window across two buffers with another process between them: the second buffer's preamble LOADs the inner
	// state from S, its pop the pushed one from P.
	{
		auto                  t = make(true);
		ToyCp                 cp;
		cp.Seed(*t);
		std::vector<uint32_t> pairs = {0x10Fu, 0xAu};
		std::vector<uint32_t> dcb1, dcb2;
		push(dcb1);
		AppendIndirect(dcb1, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
		draw(dcb1);
		draw(dcb2);
		pop(dcb2);
		draw(dcb2);
		auto r1 = t->Translate(dcb1);
		pairs[1] = 0xBu;
		cp.Job(AsJob(r1.ib));
		cp.Foreign();
		auto r2 = t->Translate(dcb2);
		Check(cp.s[kCpShadowContextDw + 0x10Fu] == 0xBu && cp.s[kCpShadowSaveOffset / 4u + 0x10Fu] == 0xAu,
		      "between the buffers S holds the inner value, P the pushed one");
		// Up to the pop's first packet (its PFP_SYNC_ME): the preamble and the draw see the inner value.
		size_t until = 0;
		for (size_t dw = 0; dw < r2.ib.size(); dw += PacketSizeDw(r2.ib[dw])) {
			if (r2.ib[dw] == Pkt3(IT_DMA_DATA, 6, false)) {
				break;
			}
			until = dw;
		}
		ToyCp mid = cp;
		mid.Job(AsJob(std::span<const uint32_t>(r2.ib.data(), until)));
		Check(mid.cx[0x10F] == 0xBu, "the next buffer's preamble LOADs the inner value inside the window");
		cp.Job(AsJob(r2.ib));
		Check(cp.cx[0x10F] == 0xAu && r2.stats.cp_shadow_pops == 1, "the pop in the next buffer restores the pushed value");
	}

	// A register first written inside the window: no pushed value in P, the CPU pop's 0.
	{
		auto                  t = make(true);
		ToyCp                 cp;
		cp.Seed(*t);
		std::vector<uint32_t> dcb;
		push(dcb);
		Append(dcb, IT_SET_CONTEXT_REG, {0x30u, 5u}, false);
		draw(dcb);
		pop(dcb);
		draw(dcb);
		auto r = t->Translate(dcb);
		cp.Job(AsJob(r.ib));
		Check(cp.cx[0x30] == 0u && cp.s[kCpShadowContextDw + 0x30u] == 0u && cp.cx[0x10F] == 0xAu, "it is set to 0 after the LOAD");
	}

	// A push translated before shadowing came on: P is seeded with the pushed state, and the pop restores it.
	{
		Pm4Translator t;
		Start(t, {{0x10Fu, 0xAu}}, {}, {});
		std::vector<uint32_t> dcb1;
		push(dcb1);
		Append(dcb1, IT_SET_CONTEXT_REG, {0x10Fu, 0xDu}, false);
		draw(dcb1);
		Check(t.Translate(dcb1).ok, "a CPU push");
		t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
		t.SetCpShadowPushPop(true);
		ToyCp cp;
		cp.Seed(t);
		Check(cp.s[kCpShadowContextDw + 0x10Fu] == 0xDu && cp.s[kCpShadowSaveOffset / 4u + 0x10Fu] == 0xAu, "S holds the inner state, P the pushed");
		std::vector<uint32_t> dcb2;
		pop(dcb2);
		draw(dcb2);
		auto r = t.Translate(dcb2);
		cp.Job(AsJob(r.ib));
		Check(r.stats.cp_shadow_pops == 1 && cp.cx[0x10F] == 0xAu, "the GPU pop restores it");
	}

	// The packet check: a DMA_DATA at S or P only in the push's and the pop's form, and only on the gfx ring.
	{
		SetCpShadowValidation(kTestShadowVa);
		auto ib_of = [](uint32_t control, uint64_t from, uint64_t to, uint32_t command) {
			return std::vector<uint32_t> {Pkt3(IT_DMA_DATA, 6, false), control, static_cast<uint32_t>(from), static_cast<uint32_t>(from >> 32u),
			                              static_cast<uint32_t>(to), static_cast<uint32_t>(to >> 32u), command};
		};
		Check(ValidateNativeIb(ib_of(kCpShadowDmaControl, cx_va, p_va, kCpShadowDmaCommand), false, nullptr, NativeIbKind::Gfx) &&
		          ValidateNativeIb(ib_of(kCpShadowDmaControl, p_va, cx_va, kCpShadowDmaCommand), false, nullptr, NativeIbKind::Gfx),
		      "the push's and the pop's copies pass");
		Check(!ValidateNativeIb(ib_of(kCpShadowDmaControl, cx_va, p_va, kCpShadowDmaCommand | 0x10u), false, nullptr, NativeIbKind::Gfx) &&
		          !ValidateNativeIb(ib_of(kCpShadowDmaControl, 0x1000000u, kTestShadowVa + 0x9000u, kCpShadowDmaCommand), false, nullptr,
		                            NativeIbKind::Gfx) &&
		          !ValidateNativeIb(ib_of(0u, 0x1000000u, p_va + 0xff0u, 0x40u), false, nullptr, NativeIbKind::Gfx),
		      "another size, or another copy into S or P, is refused");
		std::vector<uint32_t> cs = ib_of(kCpShadowDmaControl, cx_va, p_va, kCpShadowDmaCommand);
		cs[0] |= 2u;
		Check(!ValidateNativeIb(cs, false, nullptr, NativeIbKind::Compute), "and on a compute ring");
		Check(ValidateNativeIb(ib_of(0u, 0x1000000u, 0x2000000u, 0x40u), false, nullptr, NativeIbKind::Gfx),
		      "a title's DMA_DATA elsewhere still passes");
		SetCpShadowValidation(0);
	}
}

// Load mode: the CPU's inserts on CB registers the copy took from a verbatim table (the CP reads the table when it
// runs: here the pairs change after the translation, as a GPU write would change them). The FMASK fix runs on the GPU
// from the values the CP holds (a late BASE; a late FMASK that is not 0 stays), once until they change, with the
// zero word for a register S has no value for; the GPU's FMASK is SET again by a context switch, and the fix follows
// a GPU pop to what the GPU holds again. With CP shadowing off (or in record mode) it stays on the CPU as before.
// And the packet check of the fix's DMA_DATA and COND_EXEC. The display-DCC insert is left out on a late INFO only.
void TestCpShadowCbInserts() {
	using Libs::Graphics::NativeGpu::kCpShadowDmaControl;
	using Libs::Graphics::NativeGpu::kCpShadowDmaWordCommand;
	using Libs::Graphics::NativeGpu::kCpShadowDccAnd;
	using Libs::Graphics::NativeGpu::kCpShadowDccFixOffset;
	using Libs::Graphics::NativeGpu::kCpShadowFixOffset;
	using Libs::Graphics::NativeGpu::kCpShadowZeroOffset;
	using Libs::Graphics::NativeGpu::ValidateNativeIb;
	constexpr uint32_t IT_DMA_DATA = 0x50, IT_COND_EXEC = 0x22, IT_ATOMIC_MEM = 0x1E;
	constexpr uint32_t kBase0 = 0x318, kFmask0 = 0x321, kInfo0 = 0x31C, kMask = 0x08E, kBaseExt0 = 0x390, kFmaskExt0 = 0x3A0;
	const uint64_t     cx_va   = kTestShadowVa + 0x1000u;
	const uint64_t     zero_va = kTestShadowVa + kCpShadowZeroOffset;
	const uint64_t     fix_va  = kTestShadowVa + kCpShadowFixOffset;
	auto draw  = [](std::vector<uint32_t>& d) { Append(d, IT_DRAW_INDEX_AUTO, {3u, 0u}, false); };
	auto op    = [](std::vector<uint32_t>& d, uint32_t o) { d.insert(d.end(), {0xc0001200u, o}); }; // 1 push, 2 pop, 3 push + clear
	auto count = [](const std::vector<Packet>& pkts, uint32_t code) {
		return static_cast<uint32_t>(std::count_if(pkts.begin(), pkts.end(), [&](const Packet& p) { return p.op == code; }));
	};
	auto dmas_from = [](const std::vector<Packet>& pkts, uint64_t from) {
		std::vector<uint64_t> to;
		for (const auto& p: pkts) {
			if (p.op == 0x50u && p.body.size() == 6 && (p.body[1] | (static_cast<uint64_t>(p.body[2]) << 32u)) == from) {
				to.push_back(p.body[3] | (static_cast<uint64_t>(p.body[4]) << 32u));
			}
		}
		return to;
	};
	auto valid = [](const std::vector<uint32_t>& ib) {
		SetCpShadowValidation(kTestShadowVa);
		const bool ok = ValidateNativeIb(ib, false, nullptr, NativeIbKind::Gfx);
		SetCpShadowValidation(0);
		return ok;
	};
	const Regs start = {{kMask, 0xFu}, {kBase0, 0x1000u}, {kFmask0, 0u}, {kBaseExt0, 0u}, {kFmaskExt0, 0u}};

	// A late BASE (0x100 when translated, 0x2000 when run), and the second time a late FMASK (0, then 0x3000).
	for (const CpShadowMode mode: {CpShadowMode::Load, CpShadowMode::Record, CpShadowMode::Off}) {
		for (const bool late_fmask: {false, true}) {
			Pm4Translator t;
			t.SetFmaskFix(true);
			if (mode != CpShadowMode::Off) {
				t.SetCpShadow(kTestShadowVa, mode);
			}
			Start(t, start, {}, {});
			ToyCp cp;
			cp.Seed(t);
			std::vector<uint32_t> pairs = {kBase0, 0x100u};
			if (late_fmask) {
				pairs.insert(pairs.end(), {kFmask0, 0u});
			}
			std::vector<uint32_t> dcb;
			AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
			draw(dcb);
			draw(dcb);
			auto r = t.Translate(dcb);
			pairs[1] = 0x2000u;
			if (late_fmask) {
				pairs[3] = 0x3000u;
			}
			const auto pkts = Decode(r.ib);
			if (mode != CpShadowMode::Load) {
				Check(r.ok && count(pkts, IT_DMA_DATA) == 0 && SetsReg(pkts, IT_SET_CONTEXT_REG, kFmask0) && r.stats.fmask_fixed == 1 &&
				          r.stats.cp_shadow_fmask_gpu == 0,
				      mode == CpShadowMode::Off ? "CP shadowing off: the CPU's FMASK fix, as before" : "record mode: the CPU's FMASK fix, as before");
				continue;
			}
			// Without a late FMASK the copy's 0 is the GPU's: no check, two copies (BASE, BASE_EXT); with one, four.
			Check(r.ok && r.stats.cp_shadow_fmask_gpu == 1 && r.stats.fmask_fixed == 0 && count(pkts, IT_DMA_DATA) == (late_fmask ? 4u : 2u) &&
			          count(pkts, IT_COND_EXEC) == (late_fmask ? 1u : 0u) && !SetsReg(pkts, IT_SET_CONTEXT_REG, kFmask0),
			      "load mode: one FMASK fix on the GPU for both draws, no SET from the copy");
			Check(dmas_from(pkts, cx_va + 4u * kBase0) == std::vector<uint64_t>({fix_va + 4u}) &&
			          dmas_from(pkts, cx_va + 4u * kBaseExt0) == std::vector<uint64_t>({fix_va + 12u}),
			      "BASE and BASE_EXT from S into the first table's values");
			Check(valid(r.ib), "the packet check takes the fix");
			cp.Job(AsJob(r.ib));
			Check(cp.cx[kBase0] == 0x2000u && cp.cx[kFmask0] == (late_fmask ? 0x3000u : 0x2000u) && cp.s[kCpShadowContextDw + kFmask0] == cp.cx[kFmask0],
			      late_fmask ? "a late FMASK that is not 0 stays" : "FMASK = the BASE the CP holds, not the copy's, in S too");
		}
	}

	// A BASE_EXT S has no value for (no key: amdgpu's clear state, 0, on the GPU): the zero word is copied for it.
	{
		Pm4Translator t;
		t.SetFmaskFix(true);
		t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
		Start(t, {{kMask, 0xFu}, {kBase0, 0x1000u}, {kFmask0, 0u}}, {}, {});
		ToyCp cp;
		cp.Seed(t);
		std::vector<uint32_t> pairs = {kBase0, 0x100u};
		std::vector<uint32_t> dcb;
		AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
		draw(dcb);
		auto r = t.Translate(dcb);
		pairs[1] = 0x2000u;
		const auto pkts = Decode(r.ib);
		Check(r.ok && dmas_from(pkts, zero_va) == std::vector<uint64_t>({fix_va + 12u}) && dmas_from(pkts, cx_va + 4u * kBaseExt0).empty(),
		      "BASE_EXT without a value in S: the zero word");
		Check(valid(r.ib), "the packet check takes a copy of the zero word");
		cp.Job(AsJob(r.ib));
		Check(cp.cx[kFmask0] == 0x2000u && cp.cx[kFmaskExt0] == 0u, "FMASK = BASE, FMASK_EXT = 0");
	}

	// The Demon's Souls case: a late BASE, a draw, a push, the title's own target inside the window, a draw, the pop,
	// a draw. The GPU pop gives back the late BASE and the FMASK the GPU set for it; the copy has neither, so the
	// last fix is the GPU's again. The CPU pop SETs the copy's BASE, and the fix follows it on the CPU.
	for (const bool gpu: {true, false}) {
		Pm4Translator t;
		t.SetFmaskFix(true);
		t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
		t.SetCpShadowPushPop(gpu);
		Start(t, start, {}, {});
		ToyCp cp;
		cp.Seed(t);
		std::vector<uint32_t> pairs = {kBase0, 0x100u};
		std::vector<uint32_t> dcb;
		AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
		draw(dcb);
		op(dcb, 1u);
		Append(dcb, IT_SET_CONTEXT_REG, {kBase0, 0x4000u}, false);
		Append(dcb, IT_SET_CONTEXT_REG, {kFmask0, 0u}, false);
		draw(dcb);
		op(dcb, 2u);
		draw(dcb);
		auto r = t.Translate(dcb);
		pairs[1] = 0x2000u;
		// Inside the window FMASK_EXT is still the one the GPU's first fix set, so that fix is the GPU's too (on the
		// copy's BASE and FMASK, no check). The CPU pop SETs it back, and the last fix is the CPU's.
		Check(r.ok && r.stats.cp_shadow_fmask_gpu == (gpu ? 3u : 2u) && r.stats.fmask_fixed == (gpu ? 0u : 1u),
		      gpu ? "GPU pop: every fix on the GPU" : "CPU pop: the fixes before it on the GPU, the one after on the CPU");
		Check(valid(r.ib), "the packet check takes it");
		cp.Job(AsJob(r.ib));
		Check(cp.cx[kBase0] == (gpu ? 0x2000u : 0x100u) && cp.cx[kFmask0] == cp.cx[kBase0],
		      gpu ? "GPU pop: FMASK = the late BASE the pop gave back" : "CPU pop: FMASK = the BASE it SET");
	}

	// A push with a clear after a GPU fix: the clear SETs the late BASE and the GPU's FMASK to the defaults although
	// the copy holds those values already; the pop in the next buffer gives back both, and the fix after it is the GPU's.
	{
		Pm4Translator t;
		t.SetFmaskFix(true);
		t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
		t.SetCpShadowPushPop(true);
		Start(t, start, {}, {});
		ToyCp cp;
		cp.Seed(t);
		std::vector<uint32_t> pairs = {kBase0, 0x1000u};
		std::vector<uint32_t> dcb;
		AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
		draw(dcb);
		op(dcb, 3u);
		auto r = t.Translate(dcb);
		pairs[1] = 0x2000u;
		const auto pkts = Decode(r.ib);
		Check(r.ok && r.stats.cp_shadow_fmask_gpu == 1 && SetsReg(pkts, IT_SET_CONTEXT_REG, kBase0) && SetsReg(pkts, IT_SET_CONTEXT_REG, kFmask0) &&
		          SetsReg(pkts, IT_SET_CONTEXT_REG, kFmaskExt0),
		      "the clear SETs BASE, FMASK and FMASK_EXT although the copy sees no change");
		cp.Job(AsJob(r.ib));
		Check(cp.cx[kBase0] == 0x1000u && cp.cx[kFmask0] == 0u && cp.cx[kFmaskExt0] == 0u, "the GPU holds the defaults after the clear");
		std::vector<uint32_t> dcb2;
		op(dcb2, 2u);
		draw(dcb2);
		auto r2 = t.Translate(dcb2);
		Check(r2.ok && r2.stats.cp_shadow_pops == 1 && r2.stats.cp_shadow_fmask_gpu == 1 && r2.stats.fmask_fixed == 0,
		      "the pop, then the GPU's fix (the pushed BASE came from a table)");
		Check(valid(r2.ib), "the packet check takes it");
		cp.Job(AsJob(r2.ib));
		Check(cp.cx[kBase0] == 0x2000u && cp.cx[kFmask0] == 0x2000u, "after the pop: the late BASE and its FMASK");
	}

	// The packet check: the fix's COND_EXEC only on a value dword over one of its tables, its DMA_DATA only from a
	// context register of S or the zero word into a value dword.
	{
		auto cond = [&](uint64_t on, uint64_t table) {
			std::vector<uint32_t> ib = {Pkt3(IT_COND_EXEC, 4, false), static_cast<uint32_t>(on), static_cast<uint32_t>(on >> 32u), 0u, 5u,
			                            Pkt3(IT_SET_CONTEXT_REG_INDIRECT, 4, false), static_cast<uint32_t>(table), static_cast<uint32_t>(table >> 32u),
			                            0x80000000u, 2u};
			return valid(ib);
		};
		auto word = [&](uint64_t from, uint64_t to) {
			std::vector<uint32_t> ib = {Pkt3(IT_DMA_DATA, 6, false), kCpShadowDmaControl, static_cast<uint32_t>(from), static_cast<uint32_t>(from >> 32u),
			                            static_cast<uint32_t>(to), static_cast<uint32_t>(to >> 32u), kCpShadowDmaWordCommand};
			return valid(ib);
		};
		Check(cond(fix_va + 20u, fix_va + 16u) && cond(fix_va + 32u * 7u + 20u, fix_va + 32u * 7u + 16u), "the fix's COND_EXEC passes");
		Check(!cond(fix_va + 16u, fix_va + 16u) && !cond(fix_va + 20u, fix_va + 8u) && !cond(fix_va + 20u, fix_va + 0xF8u) &&
		          !cond(cx_va + 4u * kFmask0, fix_va + 16u),
		      "on an offset dword, over a table not at an entry or out of the area, or on S itself: refused");
		std::vector<uint32_t> other = {Pkt3(IT_COND_EXEC, 4, false), static_cast<uint32_t>(fix_va + 20u), static_cast<uint32_t>((fix_va + 20u) >> 32u),
		                               0u, 5u};
		Append(other, IT_SET_CONTEXT_REG, {0x10u, 1u, 2u, 3u}, false);
		Check(!valid(other), "a COND_EXEC on the fix tables over anything else is refused");
		Check(word(cx_va + 4u * kBase0, fix_va + 4u) && word(zero_va, fix_va + 12u) && word(cx_va + 4u * 0x3FFu, fix_va + 0xFCu),
		      "the fix's DMA_DATA passes");
		Check(!word(cx_va + 4u * kBase0, fix_va) && !word(cx_va + 4u * kBase0, fix_va + 8u) && !word(cx_va + 4u * kBase0, fix_va + 0x184u) &&
		          !word(cx_va + 0x1000u, fix_va + 4u) && !word(zero_va + 4u, fix_va + 4u) && !word(cx_va + 4u * kBase0, zero_va) &&
		          !word(cx_va + 4u * kBase0, cx_va + 4u * kFmask0),
		      "into an offset dword, past the tables, from outside S's context region, into the zero word or into S: refused");
	}

	// Display DCC: a target in a video-out buffer is drawn without DCC. Off, and in load mode with INFO the title's SET,
	// the CPU clears DCC in the copy's INFO. In load mode with INFO from a table the CP may hold another INFO: the GPU
	// clears DCC in its own (DMA_DATA of S's INFO into the slot's tables, ATOMIC_MEM AND, the table SETs it), which the
	// packet check takes; a slot that leaves the video-out buffer gets the CP's INFO back from the second table.
	for (const CpShadowMode mode: {CpShadowMode::Load, CpShadowMode::Off}) {
		for (const bool info_table: {true, false}) {
			Pm4Translator t;
			if (mode != CpShadowMode::Off) {
				t.SetCpShadow(kTestShadowVa, mode);
			}
			t.AddDisplayRange(0x200000000ull, 0x2000000ull);
			Start(t, {{kMask, 0xFu}, {kFmask0, 0x1u}}, {}, {});
			std::vector<uint32_t> pairs = {kBase0, 0x2000000u};
			std::vector<uint32_t> dcb;
			if (info_table) {
				pairs.insert(pairs.end(), {kInfo0, (1u << 28u) | 0x40u});
			} else {
				Append(dcb, IT_SET_CONTEXT_REG, {kInfo0, (1u << 28u) | 0x40u}, false);
			}
			AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pairs, 0x80000000u);
			draw(dcb);
			const auto r    = t.Translate(dcb);
			const bool gpu  = mode == CpShadowMode::Load && info_table;
			const auto pkts = Decode(r.ib);
			// SETs of INFO without DCC: the CPU's insert (and, off, the preamble's replay of the copy, which has it).
			uint32_t sets = 0;
			for (const auto& p: pkts) {
				sets += p.op == IT_SET_CONTEXT_REG && !p.body.empty() && (p.body[0] & 0xffffu) == kInfo0 && p.body.size() == 2 &&
				                p.body[1] == 0x40u
				            ? 1u
				            : 0u;
			}
			const uint64_t dcc_va = kTestShadowVa + kCpShadowDccFixOffset; // slot 0's tables
			const auto     to     = dmas_from(pkts, cx_va + 4u * kInfo0);
			Check(r.ok && r.stats.cp_shadow_dcc_gpu == (gpu ? 1u : 0u) && (sets != 0) == !gpu && count(pkts, IT_ATOMIC_MEM) == (gpu ? 1u : 0u) &&
			          (!gpu || (to.size() == 2 && to[0] == dcc_va + 4u && to[1] == dcc_va + 12u)),
			      gpu ? "load mode, INFO from a table: the GPU clears DCC in the CP's INFO" : "the CPU clears DCC in the copy's INFO");
			if (!gpu) {
				continue;
			}
			Check(valid(r.ib), "the packet check takes the display-DCC fix");
			ToyCp cp;
			cp.Seed(t);
			cp.Job(AsJob(r.ib));
			Check(cp.cx[kInfo0] == 0x40u && cp.s[kCpShadowContextDw + kInfo0] == 0x40u, "the CP draws without DCC, and S holds that INFO");
			// The slot leaves the video-out buffer (BASE the title's SET, INFO untouched): the CP's INFO comes back.
			std::vector<uint32_t> away;
			Append(away, IT_SET_CONTEXT_REG, {kBase0, 0x1000u}, false);
			draw(away);
			const auto r2 = t.Translate(away);
			cp.Job(AsJob(r2.ib));
			Check(r2.ok && valid(r2.ib) && cp.cx[kInfo0] == ((1u << 28u) | 0x40u), "off the video-out buffer the CP's INFO comes back, DCC and all");
			// An AND anywhere else, or of another mask, fails the packet check.
			std::vector<uint32_t> bad = {Pkt3(IT_ATOMIC_MEM, 8, false), kCpShadowDccAnd, static_cast<uint32_t>(dcc_va + 12u),
			                             static_cast<uint32_t>((dcc_va + 12u) >> 32u), ~(1u << 28u), 0u, 0u, 0u, 6u};
			std::vector<uint32_t> mask = {Pkt3(IT_ATOMIC_MEM, 8, false), kCpShadowDccAnd, static_cast<uint32_t>(dcc_va + 4u),
			                              static_cast<uint32_t>((dcc_va + 4u) >> 32u), 0u, 0u, 0u, 0u, 6u};
			Check(!valid(bad) && !valid(mask), "the packet check refuses another AND");
		}
	}

	// A GPU push, the display target bound by a table inside the window (the GPU clears its DCC), the pop: the CP holds
	// the pushed target's INFO again, with its own format, and nothing is given back on it.
	{
		Pm4Translator t;
		t.SetCpShadow(kTestShadowVa, CpShadowMode::Load);
		t.SetCpShadowPushPop(true);
		t.AddDisplayRange(0x200000000ull, 0x2000000ull);
		Start(t, {{kMask, 0xFu}, {kFmask0, 0x1u}}, {}, {});
		ToyCp cp;
		cp.Seed(t);
		const std::vector<uint32_t> pushed  = {kBase0, 0x1000u, kInfo0, (1u << 28u) | 0x44u};
		const std::vector<uint32_t> display = {kBase0, 0x2000000u, kInfo0, (1u << 28u) | 0x40u};
		std::vector<uint32_t>       dcb;
		AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, pushed, 0x80000000u);
		draw(dcb);
		op(dcb, 1u);
		AppendIndirect(dcb, IT_SET_CONTEXT_REG_INDIRECT, display, 0x80000000u);
		draw(dcb);
		op(dcb, 2u);
		draw(dcb);
		const auto r = t.Translate(dcb);
		Check(r.ok && r.stats.cp_shadow_pops == 1 && r.stats.cp_shadow_dcc_gpu == 1 && r.stats.cp_shadow_dcc_back == 0,
		      "the GPU clears DCC inside the window and gives nothing back after the pop");
		Check(valid(r.ib), "the packet check takes it");
		cp.Job(AsJob(r.ib));
		Check(cp.cx[kBase0] == 0x1000u && cp.cx[kInfo0] == ((1u << 28u) | 0x44u), "after the pop: the pushed target and its own INFO");
	}
}

// KYTY_NATIVE_GPU_AUDIT (SetAudit): the guest's sync packets field by field, from its own dwords.
size_t Form(Libs::Graphics::NativeGpu::AuditSync s) {
	return static_cast<size_t>(s);
}

void TestAuditSyncFields() {
	using Libs::Graphics::NativeGpu::AuditSync;
	Pm4Translator t;
	t.SetAudit(true);
	t.SetFenceAddress(kFenceVa);
	static uint32_t label   = 1;
	static uint64_t label64 = 1;
	const uint64_t  a       = reinterpret_cast<uint64_t>(&label);
	const uint64_t  a64     = reinterpret_cast<uint64_t>(&label64);
	std::vector<uint32_t> dcb;
	// WAIT_REG_MEM: EQUAL on memory by the PFP, CACHE_POLICY 1. WAIT_REG_MEM_64: GREATER_EQUAL, a reserved bit (12) set.
	Append(dcb, 0x3Cu, {3u | (1u << 4u) | (1u << 8u) | (1u << 25u), static_cast<uint32_t>(a), static_cast<uint32_t>(a >> 32u), 1u, 0xffffffffu, 4u},
	       false);
	Append(dcb, 0x93u,
	       {5u | (1u << 4u) | (1u << 12u), static_cast<uint32_t>(a64), static_cast<uint32_t>(a64 >> 32u), 1u, 0u, 0xffffffffu, 0u, 4u}, false);
	// libSceAgc's RELEASE_MEM (BOTTOM_OF_PIPE_TS, end of pipe, GLM_WB, CACHE_POLICY 2, 32-bit data to L2) and Kyty's marker.
	AppendGuestRelease(dcb, 1, 0, 0x202000300ull, 5, 1, 0x28u | (5u << 8u) | (1u << 12u) | (2u << 25u));
	AppendReleaseMarker(dcb, 1, 0, 0x202000304ull, 6, 0);
	// WRITE_DATA to memory with WR_CONFIRM and CACHE_POLICY 3, and a libSceAgc debug-marker string.
	Append(dcb, IT_WRITE_DATA, {(5u << 8u) | (1u << 20u) | (3u << 25u), 0x02020400u, 0x2u, 7u}, false);
	Append(dcb, IT_WRITE_DATA, {0u, 0xc343u, 0u, 0x41414141u}, false);
	// ACQUIRE_MEM: the engine bit, GL2_INV and GL2_WB.
	Append(dcb, IT_ACQUIRE_MEM, {1u << 31u, 0xffffffffu, 0xffu, 0u, 0u, 4u, (1u << 14u) | (1u << 15u)}, false);
	const auto  r = t.Translate(dcb);
	const auto& s = r.sync;
	Check(r.ok && r.refusals.empty(), "the audited buffer translates");
	Check(s.packets[Form(AuditSync::Wait)] == 1 && s.policy[Form(AuditSync::Wait)][1] == 1 && s.engine[Form(AuditSync::Wait)][1] == 1 &&
	          s.wait_function[0][3] == 1 && s.wait_mem_space[0][1] == 1 && s.wait_operation[0][0] == 1 && s.odd[Form(AuditSync::Wait)] == 0,
	      "WAIT_REG_MEM's fields");
	Check(s.packets[Form(AuditSync::Wait64)] == 1 && s.wait_function[1][5] == 1 && s.odd[Form(AuditSync::Wait64)] == 1 &&
	          s.odd_bits[Form(AuditSync::Wait64)] == (1u << 12u),
	      "WAIT_REG_MEM_64's fields, the reserved bit reported");
	Check(s.policy_lost[Form(AuditSync::Wait)] == 1 && s.policy_lost[Form(AuditSync::Wait64)] == 0,
	      "the barrier a wait becomes drops its CACHE_POLICY");
	Check(s.packets[Form(AuditSync::Release)] == 1 && s.policy[Form(AuditSync::Release)][2] == 1 && s.release_event[0][0x28] == 1 &&
	          s.release_gcr[0][0] == 1 && s.release_dst_sel[0][1] == 1 && s.release_data_sel[0][1] == 1 && s.release_int_sel[0][0] == 1,
	      "libSceAgc's RELEASE_MEM fields");
	Check(s.packets[Form(AuditSync::ReleaseMarker)] == 1 && s.release_event[1][0x28] == 1 && s.policy[Form(AuditSync::ReleaseMarker)][0] == 1,
	      "Kyty's marker counted apart");
	Check(s.packets[Form(AuditSync::WriteData)] == 1 && s.write_markers == 1 && s.write_dst_sel[5] == 1 && s.write_confirm == 1 &&
	          s.policy[Form(AuditSync::WriteData)][3] == 1,
	      "WRITE_DATA's fields, debug-marker strings apart");
	Check(s.packets[Form(AuditSync::Acquire)] == 1 && s.engine[Form(AuditSync::Acquire)][1] == 1 && s.acquire_gcr[14] == 1 &&
	          s.acquire_gcr[15] == 1 && s.odd[Form(AuditSync::Acquire)] == 0,
	      "ACQUIRE_MEM's engine and GCR bits");
	// Summed as the backend does, into 64-bit counts.
	Libs::Graphics::NativeGpu::SyncAuditT<uint64_t> sum;
	sum.Add(s);
	sum.Add(s);
	Check(sum.packets[Form(AuditSync::Wait)] == 2 && sum.release_gcr[0][0] == 2 && sum.odd_bits[Form(AuditSync::Wait64)] == (1u << 12u),
	      "sums add the counts and OR the bits");
	Pm4Translator off;
	off.SetFenceAddress(kFenceVa);
	const auto ro = off.Translate(dcb);
	Check(ro.ok && ro.sync.packets[Form(AuditSync::Wait)] == 0 && ro.sync.packets[Form(AuditSync::Release)] == 0,
	      "without the audit nothing is recorded");
}

// policy_lost: the title asked a CACHE_POLICY, and the GPU runs a packet of the translator's instead of the title's.
void TestAuditPolicyLost() {
	using Libs::Graphics::NativeGpu::AuditSync;
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	t.SetAudit(true);
	o.plan = LabelOracle::Plan::Native;
	static uint32_t label = 0;
	const uint64_t  addr  = reinterpret_cast<uint64_t>(&label);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(dcb, 0x3Cu, {3u | (1u << 4u) | (2u << 25u), static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u), 1u, 0xffffffffu, 4u},
	       false);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	const auto r = t.Translate(dcb);
	Check(r.ok && r.stats.waits_native == 1 && r.sync.policy_lost[Form(AuditSync::Wait)] == 1, "the wait in our form loses its policy");
	t.SetWaitsVerbatim(true);
	const auto rv = t.Translate(dcb);
	Check(rv.ok && rv.stats.waits_native == 1 && rv.sync.policy_lost[Form(AuditSync::Wait)] == 0, "as written it keeps it");

	// Releases in the guest's event keep it (CACHE_POLICY 3, libSceAgc's), the barrier's event (EVENT_INDEX 0, which
	// LabelEventCntl does not keep) and an interrupt the tracker raises do not.
	std::vector<uint32_t> rel;
	AppendGuestRelease(rel, 1, 0, 0x202000300ull, 5, 1);
	AppendGuestRelease(rel, 1, 0, 0x202000304ull, 5, 0, 0x06000014u);
	AppendGuestRelease(rel, 0, 2, 0, 0);
	AppendGuestRelease(rel, 1, 0, 0x202000308ull, 5);
	const auto rr = t.Translate(rel);
	Check(rr.ok && rr.sync.packets[Form(AuditSync::Release)] == 4 && rr.sync.policy[Form(AuditSync::Release)][3] == 4 &&
	          rr.sync.policy_lost[Form(AuditSync::Release)] == 2,
	      "two of four releases lose their policy");
}

// audit_waits: each wait a segment runs as a GPU wait or a barrier, with the first point after it.
void TestAuditWaitsRecorded() {
	Pm4Translator t;
	TestOracle    o;
	t.SetNativeLabels(&o, kTrackerVa);
	t.SetFenceAddress(kFenceVa);
	t.SetAudit(true);
	static uint32_t label = 0;
	static uint32_t other = 0;
	const uint64_t  a     = reinterpret_cast<uint64_t>(&label);
	const uint64_t  b     = reinterpret_cast<uint64_t>(&other);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(dcb, 1, 2, a, 1, 5); // the GPU writes the label (a wake point), then an interrupt (another)
	AppendWait(dcb, a, 1);                   // decided by that write: a GPU wait
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendWait(dcb, b, 3); // the oracle: met, a barrier
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	o.plan       = LabelOracle::Plan::Met;
	const auto r = t.Translate(dcb);
	Check(r.ok && r.segments.size() == 1 && r.segments[0].audit_waits.size() == 2 && r.segments[0].points.size() == 3,
	      "both waits recorded in their segment");
	const auto& w = r.segments[0].audit_waits;
	const auto& p = r.segments[0].points;
	Check(!w[0].barrier && w[0].local && !w[0].met && w[0].wait.addr == a && w[0].wait.ref == 1 && w[0].next_point == 2 &&
	          p[0].kind == TrackerPoint::Kind::Wake && p[1].kind == TrackerPoint::Kind::Interrupt && p[0].dw == p[1].dw,
	      "the GPU wait comes after the points flushed before it");
	Check(w[1].barrier && !w[1].local && w[1].wait.addr == b && w[1].next_point == 2 && p[2].kind == TrackerPoint::Kind::End && p[2].dw > p[1].dw,
	      "the barrier comes before the segment's end");
	// A held wait is not among them: it is the next segment's.
	o.plan        = LabelOracle::Plan::Split;
	const auto rs = t.Translate(dcb);
	Check(rs.ok && rs.segments.size() == 2 && rs.segments[1].held && rs.segments[0].audit_waits.size() == 1 && rs.segments[1].audit_waits.empty(),
	      "the held wait is the segment's `wait`");
}

// refusals: what refused the buffer, each opcode and reason once.
void TestAuditRefusalReasons() {
	using Libs::Graphics::NativeGpu::RefuseWhy;
	Pm4Translator t;
	t.SetAudit(true);
	t.SetFenceAddress(kFenceVa);
	std::vector<uint32_t> dcb;
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(dcb, 0x3Cu, {3u, 0x1234u, 0u, 1u, 0xffffffffu, 4u}, false); // MEM_SPACE 0: a register wait
	const auto r = t.Translate(dcb);
	Check(!r.ok && r.refusals.size() == 1 && r.refusals[0].why == RefuseWhy::RegisterWait && r.refusals[0].op == 0x3Cu &&
	          r.refusals[0].dw == 5 && r.refusals[0].count == 1 && r.refusals[0].nd == 5 && r.refusals[0].d[0] == dcb[5] &&
	          r.refusals[0].d[1] == 3u,
	      "a register wait refuses the buffer with its reason and dwords");
	// Soft refusals: each packet once, the end of the walk not again.
	std::vector<uint32_t> soft;
	Append(soft, 0xA1u, {0u}, false);
	Append(soft, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	Append(soft, 0xA1u, {1u}, false);
	Append(soft, 0xB2u, {0u, 0u}, false);
	const auto rs = t.Translate(soft);
	Check(!rs.ok && rs.refusals.size() == 2 && rs.refusals[0].op == 0xA1u && rs.refusals[0].why == RefuseWhy::Unknown &&
	          rs.refusals[0].count == 2 && rs.refusals[0].dw == 0 && rs.refusals[1].op == 0xB2u && rs.refusals[1].count == 1,
	      "unknown opcodes once per packet");
	Check(std::string(Libs::Graphics::NativeGpu::RefuseWhyName(RefuseWhy::RegisterWait)) == "register wait", "reasons have names");
	// A GDS release without a partition is a soft refusal that names its packet now (the log said op 0x00 at dw 0).
	Pm4Translator g;
	TestOracle    o;
	g.SetNativeLabels(&o, kTrackerVa);
	g.SetAudit(true);
	std::vector<uint32_t> gds;
	Append(gds, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendReleaseMarker(gds, 5, 0, 0x1000u, 0, 0);
	const auto rg = g.Translate(gds);
	Check(!rg.ok && rg.unknown_op == IT_NOP && rg.unknown_dw == 5 && rg.refusals.size() == 1 && rg.refusals[0].why == RefuseWhy::ReleaseGds,
	      "the GDS release names its packet");
}

// With the audit on, the GPU gets the same IB, segments and points, and the same buffers are refused.
void TestAuditIbUnchanged() {
	static uint32_t labels[4] = {0u, 1u, 0u, 0u};
	const auto      at        = [](int i) { return reinterpret_cast<uint64_t>(&labels[i]); };
	std::vector<uint32_t> dcb;
	Append(dcb, IT_SET_SH_REG, {0x240u, 0x11u}, true);
	Append(dcb, IT_DISPATCH_DIRECT, {1u, 1u, 1u, 1u}, false);
	AppendGuestRelease(dcb, 1, 0, at(0), 1);
	AppendWait(dcb, at(0), 1); // local
	AppendReleaseMarker(dcb, 1, 2, at(2), 7, 3);
	Append(dcb, 0x3Cu, {3u | (1u << 4u) | (3u << 25u), static_cast<uint32_t>(at(1)), static_cast<uint32_t>(at(1) >> 32u), 1u, 0xffffffffu, 4u},
	       false);
	Append(dcb, IT_WRITE_DATA, {(5u << 8u) | (1u << 20u), static_cast<uint32_t>(at(3)), static_cast<uint32_t>(at(3) >> 32u), 9u}, false);
	Append(dcb, IT_WRITE_DATA, {0u, 0xc343u, 0u, 0x41414141u}, false);
	Append(dcb, IT_ACQUIRE_MEM, {0u, 0xffffffffu, 0xffu, 0u, 0u, 4u, 1u << 14u}, false);
	Append(dcb, IT_DRAW_INDEX_AUTO, {3u, 2u}, false);
	AppendWait(dcb, at(2), 8); // the oracle decides
	Append(dcb, IT_DISPATCH_DIRECT, {2u, 1u, 1u, 1u}, false);
	std::vector<uint32_t> refused = dcb;
	Append(refused, 0xA1u, {0u}, false);
	for (const auto plan: {LabelOracle::Plan::Native, LabelOracle::Plan::Met, LabelOracle::Plan::Split}) {
		for (const bool compute: {false, true}) {
			Pm4Translator on, off;
			TestOracle    o_on, o_off;
			const std::pair<Pm4Translator*, TestOracle*> both[2] = {{&on, &o_on}, {&off, &o_off}};
			for (const auto& [t, o]: both) {
				o->plan = plan;
				t->SetNativeLabels(o, kTrackerVa);
				t->SetFenceAddress(kFenceVa);
				t->SetComputeQueue(compute);
				t->SetComputeRing(compute);
			}
			on.SetAudit(true);
			for (const auto* buffer: {&dcb, &refused, &dcb}) {
				const auto a = on.Translate(*buffer);
				const auto b = off.Translate(*buffer);
				Check(a.ok == b.ok && a.ib == b.ib && a.cpu_tail == b.cpu_tail && a.segments.size() == b.segments.size() &&
				          std::memcmp(&a.stats, &b.stats, sizeof(a.stats)) == 0,
				      "the audit leaves the IB and the stats alone");
				for (size_t k = 0; k < a.segments.size(); k++) {
					const Pm4Segment& x = a.segments[k];
					const Pm4Segment& y = b.segments[k];
					bool              same = x.begin == y.begin && x.end == y.end && x.held == y.held && x.barrier_dw == y.barrier_dw &&
					            x.barrier_len == y.barrier_len && x.wait_dw == y.wait_dw && x.wait_len == y.wait_len &&
					            x.points.size() == y.points.size() && x.gpu_waits.size() == y.gpu_waits.size() && y.audit_waits.empty();
					for (size_t i = 0; same && i < x.points.size(); i++) {
						same = x.points[i].dw == y.points[i].dw && x.points[i].kind == y.points[i].kind;
					}
					Check(same, "and the segments and their points");
				}
				Check(b.refusals.empty() && (a.ok || !a.refusals.empty()), "refusals are recorded with the audit only");
				Check(compute || buffer == &refused || (a.ok && (plan != LabelOracle::Plan::Split || a.segments.size() == 3)),
				      "the gfx buffer translates (cut at both of its waits the oracle decides when it splits)");
			}
		}
	}
}

int main() {
	// The cases count the packets a DCB turns into; the preamble's start values (PIXEL_PIPE_STAT_CONTROL,
	// the CP draw state) are board settings with their own switches, left out here.
	setenv("KYTY_NATIVE_GPU_PIXEL_PIPE_STAT", "0", 0);
	// The cases check the translator's forms with the strict packet check; TestValidateBlacklist the default one.
	Libs::Graphics::NativeGpu::SetNativeIbValidation(true);
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
	TestCpDmaIdle();
	TestIndirectDraws();
	TestIndirectMultiDraws();
	TestContextStatePushPop();
	TestResetQueuePop();
	TestWaitBarrier();
	TestCompactRefs();
	TestGdsReleaseMem();
	TestPs5RegOverrides();
	TestAcquireFull();
	TestLabelsWriteData();
	TestLabelsWriteDataBlock();
	TestDirectRuns();
	TestDirectRunsWriteData();
	TestLabelsReleaseMem();
	TestLabelsAsWritten();
	TestGuestInterruptsOnGpu();
	TestLabelsLocalWait();
	TestLabelsSplit();
	TestLabelsCpuTail();
	TestSonyFlip();
	TestSonyWaitUntilSafe();
	TestWaitFlipGroupAndKernelReleases();
	TestSonyWorkloads();
	TestLabelWriteCovers();
	TestLabelsWait64();
	TestValidateNativeIb();
	TestComputeRingPackets();
	TestComputeRingPlainHeaders();
	TestContextStateOpPackets();
	TestUnreachableLabel();
	TestGpuTimestamp();
	TestLabelsGpuOnly();
	TestLabelWriteBackOnlyForCpu();
	TestPlanRingWait();
	TestComputeRingForQueue();
	TestSplitIbAtPackets();
	TestExitGuards();
	TestFlattenIndirectBuffers();
	TestLodBankScan();
	TestLodUsageReport();
	TestLodStatsHandoff();
	TestLodBankSources();
	TestCpShadowExits();
	TestCpShadowPreamble();
	TestCpShadowGatedPreamble();
	TestValidateBlacklist();
	TestCpShadowImage();
	TestCpShadowLateTable();
	TestAgrQueueSharesRing0();
	TestCpShadowRecord();
	TestCpShadowHeldSegment();
	TestCpShadowKeepsFirstSegment();
	TestCpShadowResume();
	TestCpShadowClearState();
	TestCpShadowComputeRing();
	TestCpShadowClasses();
	TestCpShadowCompare();
	TestCpShadowStartKeys();
	TestCpShadowPushPopCpu();
	TestCpShadowPushPopGpu();
	TestCpShadowCbInserts();
	TestAuditSyncFields();
	TestAuditPolicyLost();
	TestAuditWaitsRecorded();
	TestAuditRefusalReasons();
	TestAuditIbUnchanged();
	std::puts("NativeGpuPm4TranslatorTests: all passed");
	return 0;
}
