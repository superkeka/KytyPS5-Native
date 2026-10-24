#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/native_gpu/nativeCapture.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace Libs::Graphics::NativeGpu::Capture;
namespace Prospero = Libs::Graphics::Prospero;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuCaptureTests: failed: %s\n", message);
		std::abort();
	}
}

void SetContext(std::vector<uint32_t>& ib, uint32_t reg, std::initializer_list<uint32_t> values) {
	ib.push_back(Pkt3(IT_SET_CONTEXT_REG, static_cast<uint32_t>(values.size()) + 1));
	ib.push_back(reg);
	ib.insert(ib.end(), values);
}

void SetSh(std::vector<uint32_t>& ib, uint32_t reg, std::initializer_list<uint32_t> values) {
	ib.push_back(Pkt3(IT_SET_SH_REG, static_cast<uint32_t>(values.size()) + 1));
	ib.push_back(reg);
	ib.insert(ib.end(), values);
}

void Draw(std::vector<uint32_t>& ib) {
	ib.push_back(Pkt3(IT_DRAW_INDEX_AUTO, 2));
	ib.push_back(3);
	ib.push_back(2);
}

// An IB in the translator's shape: preamble, two draws into MRT0 with a WRITE_DATA between, a dispatch.
std::vector<uint32_t> SampleIb() {
	std::vector<uint32_t> ib;
	ib.push_back(Pkt3(IT_CONTEXT_CONTROL, 2));
	ib.push_back(0x80000000u);
	ib.push_back(0x80000000u);
	ib.push_back(Pkt3(IT_CLEAR_STATE, 1));
	ib.push_back(0);
	SetContext(ib, kCbTargetMask, {0xfu});
	SetContext(ib, kCbColor0Base, {0x12345600u >> 8u});
	SetContext(ib, kCbColor0Info, {(12u << 2u) | (7u << 8u) | (1u << 28u)}); // 16_16_16_16 FLOAT, DCC
	SetContext(ib, kCbColor0Attrib2, {(2159u) | (3839u << 14u)});
	SetContext(ib, kCbColor0Attrib3, {(0x18u << 14u) | (1u << 30u)});
	SetSh(ib, kShPgmLoPs, {0x1000u, 0x2u});
	Draw(ib); // #0
	ib.push_back(Pkt3(IT_WRITE_DATA, 4));
	ib.push_back(5u << 8u);
	ib.push_back(0x1000);
	ib.push_back(0);
	ib.push_back(7);
	SetContext(ib, kCbColor0Base, {0x22000000u >> 8u});
	Draw(ib); // #1
	SetSh(ib, kShComputePgmLo, {0x3000u, 0u});
	ib.push_back(Pkt3(IT_DISPATCH_DIRECT, 4, true));
	ib.push_back(1);
	ib.push_back(1);
	ib.push_back(1);
	ib.push_back(1);
	ib.push_back(kNopPad);
	return ib;
}

void TestActions() {
	const auto          ib = SampleIb();
	std::vector<Action> actions;
	RegState            state;
	ListActions(ib, 3, {Part {0, static_cast<uint32_t>(ib.size()), false, 9}}, &actions, &state);
	Check(actions.size() == 3, "two draws and a dispatch");
	Check(actions[0].cs == 3 && actions[0].op == IT_DRAW_INDEX_AUTO, "first draw");
	Check(actions[0].shader == (0x1000ull << 8u | (2ull << 40u)), "pixel shader address");
	Check(actions[0].slots == 1u && actions[0].slot_base[0] == 0x12345600ull, "draw 0 writes MRT0 at its base");
	Check(actions[1].slot_base[0] == 0x22000000ull, "draw 1 sees the new base");
	Check(IsDispatch(actions[2].op) && actions[2].shader == (0x3000ull << 8u), "dispatch and its shader");
	Check(actions[2].end_dw + 1 == ib.size(), "dispatch end");
}

void TestSlice() {
	const auto          ib = SampleIb();
	std::vector<Action> actions;
	RegState            state;
	ListActions(ib, 0, {}, &actions, &state);
	// The head up to draw #0, then the rest.
	const auto head = SliceIb(ib, 0, actions[0].end_dw);
	Check(head.size() % 8 == 0, "padded to 8 dwords");
	uint32_t draws = 0;
	ForEachPacket(head, [&](const Packet& p) { draws += IsDraw(p.op) ? 1u : 0u; });
	Check(draws == 1, "the head has draw #0 only");
	const auto tail = SliceIb(ib, actions[0].end_dw, static_cast<uint32_t>(ib.size()));
	uint32_t   writes = 0, dispatches = 0, clears = 0;
	draws             = 0;
	ForEachPacket(tail, [&](const Packet& p) {
		draws += IsDraw(p.op) ? 1u : 0u;
		dispatches += IsDispatch(p.op) ? 1u : 0u;
		writes += p.op == IT_WRITE_DATA ? 1u : 0u;
		clears += p.op == IT_CLEAR_STATE ? 1u : 0u;
	});
	Check(draws == 1 && dispatches == 1 && writes == 1, "the tail redoes nothing of the head but keeps its own work");
	Check(clears == 1, "the tail rebuilds the state of the head");
	const RegState at = StateAt(tail, static_cast<uint32_t>(tail.size()));
	Check(ColorTargetAt(at, 0).base == 0x22000000ull, "state at the end of the tail");
	// Dropping the head's actions: a slice starting after the WRITE_DATA has no WRITE_DATA.
	const auto late = SliceIb(ib, actions[1].dw, static_cast<uint32_t>(ib.size()));
	writes          = 0;
	ForEachPacket(late, [&](const Packet& p) { writes += p.op == IT_WRITE_DATA ? 1u : 0u; });
	Check(writes == 0, "actions before the split are dropped");
}

void TestTsharp() {
	const auto  ib    = SampleIb();
	RegState    state = StateAt(ib, static_cast<uint32_t>(ib.size()));
	state.ctx[kCbColor0Base]     = 0x12345600u >> 8u;
	state.ctx[kCbColor0DccBase]  = 0x40000000u >> 8u;
	state.ctx[kCbColor0DccCtl]   = (2u << 5u);
	const ColorTarget t   = ColorTargetAt(state, 0);
	const auto        enc = Prospero::ResolveRenderTargetFormat(static_cast<Prospero::ChannelLayout>(t.Format()),
	                                                            static_cast<Prospero::ChannelType>(t.NumberType()));
	Check(enc.buffer_format == Prospero::BufferFormat::k16_16_16_16Float && enc.components == 4, "RGBA16F resolves");
	uint32_t    d[8];
	std::string why;
	Check(TsharpForColorTarget(t, static_cast<uint32_t>(enc.buffer_format), enc.components, d, &why), "T# built");
	Check(TsharpBase(d) == 0x12345600ull, "T# base");
	Check(TsharpWidth(d) == 3840 && TsharpHeight(d) == 2160, "T# size");
	Check(TsharpFormat(d) == 71, "T# format");
	Check(TsharpType(d) == 9 && TsharpSwizzle(d) == 0x18, "T# 2D, swizzle");
	Check((d[3] & 0xfffu) == (kSelX | (kSelY << 3u) | (kSelZ << 6u) | (kSelW << 9u)), "identity swizzle");
	Check(TsharpCompressed(d) && TsharpMeta(d) == 0x40000000ull, "DCC metadata");
	Check(((d[6] >> 17u) & 3u) == 2 && ((d[6] >> 19u) & 1u) == 1, "DCC block size and pipe alignment");
	// ASTRO BOT's own T# of its 3840x2160 RGBA16F target 0x520440000 (DCC, SW_64KB_R_X, meta 0x5718ff000).
	ColorTarget astro = t;
	astro.base        = 0x520440000ull;
	astro.attrib3     = (0x1bu << 14u) | (1u << 30u);
	astro.dcc_base    = 0x5718ff000ull;
	astro.dcc_control = (2u << 2u) | (1u << 5u);
	const uint32_t game[8] = {0x05204400u, 0xc4700000u, 0x021bc3bfu, 0x91b00facu, 0, 0x00700000u, 0xf07b0000u, 0x0005718fu};
	Check(TsharpForColorTarget(astro, 71, 4, d, &why), "ASTRO BOT target");
	for (int i = 0; i < 8; i++) {
		Check(d[i] == game[i], "the T# the game itself uses");
	}
	Check(LooksLikeTsharp(d, 0x520440000ull) && !LooksLikeTsharp(d, 0x520450000ull), "T# search");
	Check(DstSelForSwap(4, 1) == (kSelZ | (kSelY << 3u) | (kSelX << 6u) | (kSelW << 9u)), "BGRA undone");
}

void TestManifest() {
	Manifest m;
	m.gds_spec   = "gds=4096,gws=0,oa=0";
	m.gds_size   = 4096;
	m.start_flip = 3098;
	m.end_flip   = 3100;
	m.regions.push_back(Region {0, RegionKind::Scratch, 0, 4096, 0, true, "mem/r0.bin"});
	m.regions.push_back(Region {1, RegionKind::External, 0x900000000ull, 0x20000, 2, false, "mem/r1.bin"});
	m.maps = {"map_0.txt", "map_1.txt"};
	Cs cs;
	cs.ib_file     = "ib_0.bin";
	cs.ib_dw       = 1024;
	cs.delta_file  = "delta_0.bin";
	cs.map_version = 1;
	cs.parts       = {Part {0, 512, false, 7}, Part {512, 512, true, 3}};
	m.cs.push_back(cs);
	m.flips.push_back(Flip {3099, 1, 0xa000000, 0x2000000, 3840, 2160, 4, 0x80000000u, 0x1b});
	m.final_delta       = "delta_final.bin";
	m.final_map_version = 1;
	const std::string path = "native_gpu_capture_test_manifest.txt";
	Check(m.Write(path), "manifest written");
	Manifest    r;
	std::string error;
	Check(r.Read(path, &error), error.c_str());
	std::remove(path.c_str());
	Check(r.gds_spec == m.gds_spec && r.gds_size == 4096 && r.start_flip == 3098 && r.end_flip == 3100, "header");
	Check(r.regions.size() == 2 && r.regions[1].kind == RegionKind::External && r.regions[1].key == 0x900000000ull &&
	          r.regions[1].since == 2 && !r.regions[1].writable,
	      "regions");
	Check(r.cs.size() == 1 && r.cs[0].parts.size() == 2 && r.cs[0].parts[1].acb && r.cs[0].parts[1].offset_dw == 512 &&
	          r.cs[0].map_version == 1,
	      "CS and parts");
	Check(r.flips.size() == 1 && r.flips[0].n == 3099 && r.flips[0].pixel_format == 0x80000000u && r.flips[0].tile_mode == 0x1b,
	      "flips");
	Check(r.final_delta == "delta_final.bin" && r.final_map_version == 1, "final");

	const std::vector<Mapping> maps = {{0x100000000ull, 0x1000000, 3, 0}, {0x200000000ull, 0x4000, 1, 0x2000}};
	const std::string          mpath = "native_gpu_capture_test_map.txt";
	Check(WriteMappings(mpath, maps), "mapping written");
	std::vector<Mapping> back;
	Check(ReadMappings(mpath, &back) && back == maps, "mapping read back");
	std::remove(mpath.c_str());
}

void TestHash() {
	std::vector<uint8_t> a(kPageSize, 0), b(kPageSize, 0);
	Check(PageIsZero(a.data()), "zero page");
	b[4095] = 1;
	Check(!PageIsZero(b.data()) && HashPage(a.data()) != HashPage(b.data()), "last byte changes the hash");
	b[4095] = 0;
	b[8]    = 1;
	const uint64_t h8 = HashPage(b.data());
	b[8]              = 0;
	b[16]             = 1;
	Check(h8 != HashPage(b.data()), "lanes are told apart");
}

} // namespace

int main() {
	TestActions();
	TestSlice();
	TestTsharp();
	TestManifest();
	TestHash();
	std::printf("NativeGpuCaptureTests: all passed\n");
	return 0;
}
