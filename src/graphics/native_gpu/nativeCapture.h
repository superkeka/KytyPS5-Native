#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVECAPTURE_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVECAPTURE_H_

// Frame capture for the BC-250 native backend: the on-disk format shared by the capture in the
// emulator (nativeCaptureWriter.h, KYTY_NATIVE_GPU_CAPTURE) and the replay tool (tools/bc250-replay),
// plus the PM4 walking both need. Portable (no amdgpu headers), so it is unit-tested everywhere.
//
// A capture is a directory:
//   manifest.txt        what follows, one record per line (Manifest::Write / Read)
//   mem/r<id>.bin       a memory region as it was when it first appeared (sparse: zero pages are holes)
//   delta_<n>.bin       pages that changed since the previous submission, taken right before CS n
//                       (GPU idle) and applied by the replay right before it runs CS n
//   delta_final.bin     the same after the last captured CS, for checking the replay
//   ib_<n>.bin          the IB of CS n exactly as it went to the gfx ring
//   map_<v>.txt         GPU VA -> region mapping, version v (a new one whenever the mapping changes)
//
// Regions are what the GPU VM of the emulator maps: each resident 16 MiB block of guest direct memory
// (one GEM buffer, possibly mapped at several guest addresses), the dummy buffer behind non-resident
// guest memory, each external (userptr) range at its guest address, the submitter's scratch area, and
// the GDS partition (read back through a staging buffer).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::NativeGpu::Capture {

constexpr uint32_t kPageSize = 4096;
// GPU VAs above every guest range, next to the submitter's IB (0x400000000000).
constexpr uint64_t kStagingVa  = 0x410000000000ull; // GDS staging buffer (capture and replay)
constexpr uint64_t kReadbackVa = 0x420000000000ull; // replay: readback output
constexpr uint64_t kCodeVa     = 0x430000000000ull; // replay: readback shader code

constexpr uint32_t kDeltaMagic = 0x3144434bu; // "KCD1"

enum class RegionKind : uint32_t { Block, Dummy, External, Scratch, Gds };

inline const char* RegionKindName(RegionKind k) {
	switch (k) {
		case RegionKind::Block: return "block";
		case RegionKind::Dummy: return "dummy";
		case RegionKind::External: return "ext";
		case RegionKind::Scratch: return "scratch";
		case RegionKind::Gds: return "gds";
	}
	return "?";
}

inline bool ParseRegionKind(const char* s, RegionKind* out) {
	for (RegionKind k: {RegionKind::Block, RegionKind::Dummy, RegionKind::External, RegionKind::Scratch, RegionKind::Gds}) {
		if (std::strcmp(s, RegionKindName(k)) == 0) {
			*out = k;
			return true;
		}
	}
	return false;
}

struct Region {
	uint32_t   id       = 0;
	RegionKind kind     = RegionKind::Block;
	uint64_t   key      = 0; // block index, or the guest VA of an external range
	uint64_t   size     = 0;
	uint32_t   since    = 0; // the CS before which it first appeared (its file holds it then)
	bool       writable = true;
	std::string file;
};

// One GPU mapping: [va, va + size) maps region bytes [offset, offset + size).
struct Mapping {
	uint64_t va     = 0;
	uint64_t size   = 0;
	uint32_t region = 0;
	uint64_t offset = 0;

	bool operator==(const Mapping&) const = default;
	bool operator<(const Mapping& o) const {
		if (va != o.va) return va < o.va;
		if (size != o.size) return size < o.size;
		if (region != o.region) return region < o.region;
		return offset < o.offset;
	}
};

// A guest buffer inside a CS (a pipelined CS carries several).
struct Part {
	uint32_t offset_dw = 0;
	uint32_t size_dw   = 0;
	bool     acb       = false;
	uint64_t buffer    = 0; // the shadow path's DCB or ACB counter
};

struct Cs {
	std::string       ib_file;
	uint32_t          ib_dw = 0;
	std::string       delta_file;
	uint32_t          map_version = 0;
	std::vector<Part> parts;
};

struct Flip {
	uint64_t n        = 0;
	uint32_t after_cs = 0; // CS submitted before the flip was presented
	uint64_t addr     = 0;
	uint64_t size     = 0;
	uint32_t width = 0, height = 0, bytes_per_pixel = 0, pixel_format = 0, tile_mode = 0;
};

struct Manifest {
	std::string              gds_spec; // KYTY_NATIVE_GPU_GDS value the submitter was created with
	uint32_t                 gds_size   = 0;
	uint64_t                 start_flip = 0;
	uint64_t                 end_flip   = 0;
	std::vector<Region>      regions;
	std::vector<std::string> maps; // map file per version
	std::vector<Cs>          cs;
	std::vector<Flip>        flips;
	std::string              final_delta;
	uint32_t                 final_map_version = 0;

	bool Write(const std::string& path) const {
		std::FILE* f = std::fopen(path.c_str(), "w");
		if (f == nullptr) {
			return false;
		}
		std::fprintf(f, "kyty-native-capture 1\n");
		std::fprintf(f, "gds %u %s\n", gds_size, gds_spec.empty() ? "-" : gds_spec.c_str());
		std::fprintf(f, "flips %llu %llu\n", static_cast<unsigned long long>(start_flip),
		             static_cast<unsigned long long>(end_flip));
		for (const auto& r: regions) {
			std::fprintf(f, "region %u %s 0x%llx 0x%llx %u %u %s\n", r.id, RegionKindName(r.kind),
			             static_cast<unsigned long long>(r.key), static_cast<unsigned long long>(r.size), r.since,
			             r.writable ? 1u : 0u, r.file.c_str());
		}
		for (size_t v = 0; v < maps.size(); v++) {
			std::fprintf(f, "map %zu %s\n", v, maps[v].c_str());
		}
		for (size_t i = 0; i < cs.size(); i++) {
			const Cs& c = cs[i];
			std::fprintf(f, "cs %zu %s %u %s %u\n", i, c.ib_file.c_str(), c.ib_dw, c.delta_file.c_str(), c.map_version);
			for (const auto& p: c.parts) {
				std::fprintf(f, "part %zu %u %u %s %llu\n", i, p.offset_dw, p.size_dw, p.acb ? "acb" : "dcb",
				             static_cast<unsigned long long>(p.buffer));
			}
		}
		for (const auto& fl: flips) {
			std::fprintf(f, "flip %llu %u 0x%llx 0x%llx %u %u %u %u 0x%x\n", static_cast<unsigned long long>(fl.n),
			             fl.after_cs, static_cast<unsigned long long>(fl.addr), static_cast<unsigned long long>(fl.size),
			             fl.width, fl.height, fl.bytes_per_pixel, fl.pixel_format, fl.tile_mode);
		}
		if (!final_delta.empty()) {
			std::fprintf(f, "final %s %u\n", final_delta.c_str(), final_map_version);
		}
		return std::fclose(f) == 0;
	}

	bool Read(const std::string& path, std::string* error) {
		std::FILE* f = std::fopen(path.c_str(), "r");
		if (f == nullptr) {
			*error = "cannot open " + path;
			return false;
		}
		*this = Manifest {};
		char line[1024];
		bool ok    = std::fgets(line, sizeof(line), f) != nullptr && std::strncmp(line, "kyty-native-capture 1", 21) == 0;
		int  lineno = 1;
		if (!ok) {
			*error = "not a capture manifest";
		}
		while (ok && std::fgets(line, sizeof(line), f) != nullptr) {
			lineno++;
			char               word[32] = {}, a[512] = {}, b[512] = {};
			unsigned long long x = 0, y = 0, z = 0;
			unsigned           u0 = 0, u1 = 0, u2 = 0, u3 = 0, u4 = 0, u5 = 0;
			if (std::sscanf(line, "%31s", word) != 1) {
				continue;
			}
			const std::string w = word;
			if (w == "gds") {
				ok = std::sscanf(line, "gds %u %511s", &u0, a) == 2;
				gds_size = u0;
				gds_spec = std::strcmp(a, "-") == 0 ? "" : a;
			} else if (w == "flips") {
				ok = std::sscanf(line, "flips %llu %llu", &x, &y) == 2;
				start_flip = x;
				end_flip   = y;
			} else if (w == "region") {
				char kind[32] = {};
				ok = std::sscanf(line, "region %u %31s %llx %llx %u %u %511s", &u0, kind, &x, &y, &u1, &u2, a) == 7;
				Region r;
				ok = ok && ParseRegionKind(kind, &r.kind) && u0 == regions.size();
				r.id = u0;
				r.key = x;
				r.size = y;
				r.since = u1;
				r.writable = u2 != 0;
				r.file = a;
				regions.push_back(r);
			} else if (w == "map") {
				ok = std::sscanf(line, "map %u %511s", &u0, a) == 2 && u0 == maps.size();
				maps.emplace_back(a);
			} else if (w == "cs") {
				Cs c;
				ok = std::sscanf(line, "cs %u %511s %u %511s %u", &u0, a, &u1, b, &u2) == 5 && u0 == cs.size();
				c.ib_file     = a;
				c.ib_dw       = u1;
				c.delta_file  = b;
				c.map_version = u2;
				cs.push_back(c);
			} else if (w == "part") {
				char kind[8] = {};
				ok = std::sscanf(line, "part %u %u %u %7s %llu", &u0, &u1, &u2, kind, &x) == 5 && u0 < cs.size();
				if (ok) {
					cs[u0].parts.push_back(Part {u1, u2, std::strcmp(kind, "acb") == 0, x});
				}
			} else if (w == "flip") {
				Flip fl;
				ok = std::sscanf(line, "flip %llu %u %llx %llx %u %u %u %u %x", &x, &u0, &y, &z, &u1, &u2, &u3, &u4, &u5) == 9;
				fl.n = x;
				fl.after_cs = u0;
				fl.addr = y;
				fl.size = z;
				fl.width = u1;
				fl.height = u2;
				fl.bytes_per_pixel = u3;
				fl.pixel_format = u4;
				fl.tile_mode = u5;
				flips.push_back(fl);
			} else if (w == "final") {
				ok = std::sscanf(line, "final %511s %u", a, &u0) == 2;
				final_delta       = a;
				final_map_version = u0;
			}
			if (!ok) {
				*error = "manifest line " + std::to_string(lineno) + ": " + line;
			}
		}
		std::fclose(f);
		return ok;
	}
};

inline bool WriteMappings(const std::string& path, const std::vector<Mapping>& maps) {
	std::FILE* f = std::fopen(path.c_str(), "w");
	if (f == nullptr) {
		return false;
	}
	for (const auto& m: maps) {
		std::fprintf(f, "0x%llx 0x%llx %u 0x%llx\n", static_cast<unsigned long long>(m.va),
		             static_cast<unsigned long long>(m.size), m.region, static_cast<unsigned long long>(m.offset));
	}
	return std::fclose(f) == 0;
}

inline bool ReadMappings(const std::string& path, std::vector<Mapping>* out) {
	std::FILE* f = std::fopen(path.c_str(), "r");
	if (f == nullptr) {
		return false;
	}
	out->clear();
	unsigned long long va = 0, size = 0, offset = 0;
	unsigned           region = 0;
	while (std::fscanf(f, "%llx %llx %u %llx", &va, &size, &region, &offset) == 4) {
		out->push_back(Mapping {va, size, region, offset});
	}
	std::fclose(f);
	return true;
}

// 64-bit hash of one page, four independent lanes so it runs at memory speed.
inline uint64_t HashPage(const void* page) {
	const auto* w = static_cast<const uint64_t*>(page);
	uint64_t    h[4] = {0x9e3779b97f4a7c15ull, 0xc2b2ae3d27d4eb4full, 0x165667b19e3779f9ull, 0x27d4eb2f165667c5ull};
	for (uint32_t i = 0; i < kPageSize / 8; i += 4) {
		for (uint32_t l = 0; l < 4; l++) {
			h[l] = (h[l] ^ w[i + l]) * 0x100000001b3ull;
			h[l] ^= h[l] >> 29u;
		}
	}
	return (h[0] ^ (h[1] << 1u) ^ (h[2] << 2u) ^ (h[3] << 3u)) * 0x9e3779b97f4a7c15ull;
}

inline bool PageIsZero(const void* page) {
	const auto* w = static_cast<const uint64_t*>(page);
	uint64_t    acc = 0;
	for (uint32_t i = 0; i < kPageSize / 8; i++) {
		acc |= w[i];
	}
	return acc == 0;
}

// --- PM4 walking (gfx10 type-3 packets, as the translator emits them) ---

constexpr uint32_t IT_NOP                    = 0x10;
constexpr uint32_t IT_SET_BASE               = 0x11;
constexpr uint32_t IT_CLEAR_STATE            = 0x12;
constexpr uint32_t IT_INDEX_BUFFER_SIZE      = 0x13;
constexpr uint32_t IT_DISPATCH_DIRECT        = 0x15;
constexpr uint32_t IT_DISPATCH_INDIRECT      = 0x16;
constexpr uint32_t IT_DRAW_INDIRECT          = 0x24;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT    = 0x25;
constexpr uint32_t IT_INDEX_BASE             = 0x26;
constexpr uint32_t IT_DRAW_INDEX_2           = 0x27;
constexpr uint32_t IT_CONTEXT_CONTROL        = 0x28;
constexpr uint32_t IT_INDEX_TYPE             = 0x2A;
constexpr uint32_t IT_DRAW_INDIRECT_MULTI    = 0x2C;
constexpr uint32_t IT_DRAW_INDEX_AUTO        = 0x2D;
constexpr uint32_t IT_NUM_INSTANCES          = 0x2F;
constexpr uint32_t IT_DRAW_INDEX_MULTI_AUTO  = 0x30;
constexpr uint32_t IT_DRAW_INDEX_OFFSET_2    = 0x35;
constexpr uint32_t IT_WRITE_DATA             = 0x37;
constexpr uint32_t IT_DRAW_INDEX_INDIRECT_MULTI = 0x38;
constexpr uint32_t IT_WAIT_REG_MEM           = 0x3C;
constexpr uint32_t IT_COPY_DATA              = 0x40;
constexpr uint32_t IT_PFP_SYNC_ME            = 0x42;
constexpr uint32_t IT_EVENT_WRITE            = 0x46;
constexpr uint32_t IT_RELEASE_MEM            = 0x49;
constexpr uint32_t IT_DMA_DATA               = 0x50;
constexpr uint32_t IT_ACQUIRE_MEM            = 0x58;
constexpr uint32_t IT_SET_CONTEXT_REG        = 0x69;
constexpr uint32_t IT_SET_SH_REG             = 0x76;
constexpr uint32_t IT_SET_UCONFIG_REG        = 0x79;
constexpr uint32_t IT_SET_UCONFIG_REG_INDEX  = 0x7A;
constexpr uint32_t IT_GET_LOD_STATS          = 0x8E;
constexpr uint32_t IT_WAIT_REG_MEM_64        = 0x93;

constexpr uint32_t kNopPad = 0xffff1000u; // single-dword NOP (type 3, count 0x3fff)

constexpr uint32_t Pkt3(uint32_t op, uint32_t body_dw, bool compute = false) {
	return (3u << 30u) | (((body_dw - 1u) & 0x3fffu) << 16u) | ((op & 0xffu) << 8u) | (compute ? 2u : 0u);
}

inline bool IsDraw(uint32_t op) {
	switch (op) {
		case IT_DRAW_INDIRECT:
		case IT_DRAW_INDEX_INDIRECT:
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDIRECT_MULTI:
		case IT_DRAW_INDEX_AUTO:
		case IT_DRAW_INDEX_MULTI_AUTO:
		case IT_DRAW_INDEX_OFFSET_2:
		case IT_DRAW_INDEX_INDIRECT_MULTI: return true;
		default: return false;
	}
}

inline bool IsDispatch(uint32_t op) { return op == IT_DISPATCH_DIRECT || op == IT_DISPATCH_INDIRECT; }

// Packets that only set state: a continuation IB (ContinuationIb) keeps them and drops everything
// else that came before the split, so the register state at the split is rebuilt without redoing work.
inline bool IsStateOnly(uint32_t op) {
	switch (op) {
		case IT_NOP:
		case IT_SET_BASE:
		case IT_CLEAR_STATE:
		case IT_INDEX_BUFFER_SIZE:
		case IT_INDEX_BASE:
		case IT_CONTEXT_CONTROL:
		case IT_INDEX_TYPE:
		case IT_NUM_INSTANCES:
		case IT_SET_CONTEXT_REG:
		case IT_SET_SH_REG:
		case IT_SET_UCONFIG_REG:
		case IT_SET_UCONFIG_REG_INDEX: return true;
		default: return false;
	}
}

inline const char* OpName(uint32_t op) {
	switch (op) {
		case IT_DISPATCH_DIRECT: return "DISPATCH_DIRECT";
		case IT_DISPATCH_INDIRECT: return "DISPATCH_INDIRECT";
		case IT_DRAW_INDIRECT: return "DRAW_INDIRECT";
		case IT_DRAW_INDEX_INDIRECT: return "DRAW_INDEX_INDIRECT";
		case IT_DRAW_INDEX_2: return "DRAW_INDEX_2";
		case IT_DRAW_INDIRECT_MULTI: return "DRAW_INDIRECT_MULTI";
		case IT_DRAW_INDEX_AUTO: return "DRAW_INDEX_AUTO";
		case IT_DRAW_INDEX_MULTI_AUTO: return "DRAW_INDEX_MULTI_AUTO";
		case IT_DRAW_INDEX_OFFSET_2: return "DRAW_INDEX_OFFSET_2";
		case IT_DRAW_INDEX_INDIRECT_MULTI: return "DRAW_INDEX_INDIRECT_MULTI";
		default: return "?";
	}
}

struct Packet {
	uint32_t dw   = 0; // offset of the header
	uint32_t size = 0; // dwords, header included
	uint32_t op   = 0; // type-3 opcode, ~0u for other packet types
};

// Calls cb(Packet) for each packet of `ib`; stops at a packet that runs past the end.
template <typename Cb>
void ForEachPacket(std::span<const uint32_t> ib, Cb&& cb) {
	uint32_t dw = 0;
	while (dw < ib.size()) {
		const uint32_t h    = ib[dw];
		const uint32_t type = h >> 30u;
		Packet         p;
		p.dw = dw;
		if (h == kNopPad) {
			p.size = 1;
			p.op   = IT_NOP;
		} else if (type == 3u) {
			p.size = ((h >> 16u) & 0x3fffu) + 2u;
			p.op   = (h >> 8u) & 0xffu;
		} else if (type == 2u) {
			p.size = 1;
			p.op   = ~0u;
		} else {
			p.size = ((h >> 16u) & 0x3fffu) + 2u; // type 0: register writes
			p.op   = ~0u;
		}
		if (dw + p.size > ib.size()) {
			return;
		}
		cb(p);
		dw += p.size;
	}
}

// Register state seen while walking an IB, enough to tell what a draw renders to.
constexpr uint32_t kContextRegs = 0x400;
constexpr uint32_t kShRegs      = 0x400;

struct RegState {
	uint32_t ctx[kContextRegs] = {};
	uint32_t sh[kShRegs]       = {};

	void Apply(std::span<const uint32_t> ib, const Packet& p) {
		const uint32_t* body = ib.data() + p.dw + 1;
		const uint32_t  n    = p.size - 1;
		if (p.op == IT_CLEAR_STATE) {
			std::fill(std::begin(ctx), std::end(ctx), 0u);
			return;
		}
		if ((p.op != IT_SET_CONTEXT_REG && p.op != IT_SET_SH_REG) || n < 2) {
			return;
		}
		uint32_t* regs  = p.op == IT_SET_CONTEXT_REG ? ctx : sh;
		uint32_t  limit = p.op == IT_SET_CONTEXT_REG ? kContextRegs : kShRegs;
		uint32_t  reg   = body[0] & 0xffffu;
		for (uint32_t i = 1; i < n && reg < limit; i++, reg++) {
			regs[reg] = body[i];
		}
	}
};

// Context register offsets (dwords from 0xA000) and SH offsets (from 0x2C00).
constexpr uint32_t kCbTargetMask    = 0x08E;
constexpr uint32_t kCbColorControl  = 0x202;
constexpr uint32_t kCbColor0Base    = 0x318;
constexpr uint32_t kCbColorStride   = 0xF;
constexpr uint32_t kCbColor0View    = 0x31B;
constexpr uint32_t kCbColor0Info    = 0x31C;
constexpr uint32_t kCbColor0Attrib  = 0x31D;
constexpr uint32_t kCbColor0DccCtl  = 0x31E;
constexpr uint32_t kCbColor0Cmask   = 0x31F;
constexpr uint32_t kCbColor0Fmask   = 0x321;
constexpr uint32_t kCbColor0Clear0  = 0x323;
constexpr uint32_t kCbColor0DccBase = 0x325;
constexpr uint32_t kCbColor0BaseExt = 0x390;
constexpr uint32_t kCbColor0CmaskExt = 0x398;
constexpr uint32_t kCbColor0DccExt  = 0x3A8;
constexpr uint32_t kCbColor0Attrib2 = 0x3B0;
constexpr uint32_t kCbColor0Attrib3 = 0x3B8;
constexpr uint32_t kShPgmLoPs       = 0x008;
constexpr uint32_t kShUserDataPs    = 0x00C;
constexpr uint32_t kShPgmLoGs       = 0x088;
constexpr uint32_t kShUserDataGs    = 0x08C;
constexpr uint32_t kShComputePgmLo  = 0x20C;
constexpr uint32_t kShComputeUserData = 0x240;

struct ColorTarget {
	uint32_t slot = 0;
	uint64_t base = 0; // byte address
	uint32_t view = 0, info = 0, attrib = 0, attrib2 = 0, attrib3 = 0, dcc_control = 0;
	uint64_t dcc_base   = 0;
	uint64_t cmask_base = 0;
	uint32_t clear[2]   = {};

	[[nodiscard]] uint32_t Width() const { return ((attrib2 >> 14u) & 0x3fffu) + 1u; }
	[[nodiscard]] uint32_t Height() const { return (attrib2 & 0x3fffu) + 1u; }
	[[nodiscard]] uint32_t MaxMip() const { return (attrib2 >> 28u) & 0xfu; }
	[[nodiscard]] uint32_t Format() const { return (info >> 2u) & 0x1fu; }
	[[nodiscard]] uint32_t NumberType() const { return (info >> 8u) & 0x7u; }
	[[nodiscard]] uint32_t CompSwap() const { return (info >> 11u) & 0x3u; }
	[[nodiscard]] bool     FastClear() const { return ((info >> 13u) & 1u) != 0; }
	[[nodiscard]] bool     Dcc() const { return ((info >> 28u) & 1u) != 0; }
	[[nodiscard]] uint32_t SwizzleMode() const { return (attrib3 >> 14u) & 0x1fu; }
	[[nodiscard]] uint32_t Slices() const { return (attrib3 & 0x1fffu) + 1u; }
	[[nodiscard]] uint32_t ResourceType() const { return (attrib3 >> 24u) & 0x3u; }
	[[nodiscard]] uint32_t Samples() const { return 1u << ((attrib >> 12u) & 0x7u); }
};

inline ColorTarget ColorTargetAt(const RegState& s, uint32_t slot) {
	const uint32_t o = slot * kCbColorStride;
	ColorTarget    t;
	t.slot        = slot;
	t.base        = (static_cast<uint64_t>(s.ctx[kCbColor0Base + o]) | (static_cast<uint64_t>(s.ctx[kCbColor0BaseExt + slot] & 0xffu) << 32u)) << 8u;
	t.view        = s.ctx[kCbColor0View + o];
	t.info        = s.ctx[kCbColor0Info + o];
	t.attrib      = s.ctx[kCbColor0Attrib + o];
	t.dcc_control = s.ctx[kCbColor0DccCtl + o];
	t.attrib2     = s.ctx[kCbColor0Attrib2 + slot];
	t.attrib3     = s.ctx[kCbColor0Attrib3 + slot];
	t.dcc_base    = (static_cast<uint64_t>(s.ctx[kCbColor0DccBase + o]) | (static_cast<uint64_t>(s.ctx[kCbColor0DccExt + slot] & 0xffu) << 32u)) << 8u;
	t.cmask_base  = (static_cast<uint64_t>(s.ctx[kCbColor0Cmask + o]) | (static_cast<uint64_t>(s.ctx[kCbColor0CmaskExt + slot] & 0xffu) << 32u)) << 8u;
	t.clear[0]    = s.ctx[kCbColor0Clear0 + o];
	t.clear[1]    = s.ctx[kCbColor0Clear0 + o + 1];
	return t;
}

// Slots a draw writes: a target with a format and a non-zero CB_TARGET_MASK nibble.
inline uint32_t BoundColorSlots(const RegState& s) {
	uint32_t slots = 0;
	for (uint32_t i = 0; i < 8; i++) {
		const ColorTarget t = ColorTargetAt(s, i);
		if (t.Format() != 0 && t.base != 0 && ((s.ctx[kCbTargetMask] >> (4u * i)) & 0xfu) != 0) {
			slots |= 1u << i;
		}
	}
	return slots;
}

inline uint64_t ShaderAddress(const RegState& s, uint32_t lo) {
	return (static_cast<uint64_t>(s.sh[lo]) << 8u) | (static_cast<uint64_t>(s.sh[lo + 1] & 0xffu) << 40u);
}

// Every draw and dispatch of a capture, in GPU order.
struct Action {
	uint32_t cs     = 0;
	uint32_t dw     = 0; // header offset in the CS's IB
	uint32_t end_dw = 0; // one past the packet
	uint32_t op     = 0;
	uint32_t part   = 0; // which guest buffer of the CS
	uint64_t shader = 0; // pixel shader of a draw, compute shader of a dispatch
	uint32_t slots  = 0; // color targets a draw writes
	uint64_t slot_base[8] = {};
	uint32_t slot_info[8] = {}; // CB_COLORn_INFO
};

// Appends the draws and dispatches of IB `ib` (CS `cs`, guest buffers `parts`) to `out`; `state` is
// the register state the walk leaves behind (each IB sets all of it again in its preamble).
inline void ListActions(std::span<const uint32_t> ib, uint32_t cs, const std::vector<Part>& parts,
                        std::vector<Action>* out, RegState* state) {
	ForEachPacket(ib, [&](const Packet& p) {
		state->Apply(ib, p);
		if (!IsDraw(p.op) && !IsDispatch(p.op)) {
			return;
		}
		Action a;
		a.cs     = cs;
		a.dw     = p.dw;
		a.end_dw = p.dw + p.size;
		a.op     = p.op;
		for (uint32_t i = 0; i < parts.size(); i++) {
			if (p.dw >= parts[i].offset_dw && p.dw < parts[i].offset_dw + parts[i].size_dw) {
				a.part = i;
			}
		}
		if (IsDispatch(p.op)) {
			a.shader = ShaderAddress(*state, kShComputePgmLo);
		} else {
			a.shader = ShaderAddress(*state, kShPgmLoPs);
			a.slots  = BoundColorSlots(*state);
			for (uint32_t i = 0; i < 8; i++) {
				const ColorTarget t = ColorTargetAt(*state, i);
				a.slot_base[i]      = t.base;
				a.slot_info[i]      = t.info;
			}
		}
		out->push_back(a);
	});
}

// The register state right before the packet at `dw`.
inline RegState StateAt(std::span<const uint32_t> ib, uint32_t dw) {
	RegState s;
	ForEachPacket(ib, [&](const Packet& p) {
		if (p.dw < dw) {
			s.Apply(ib, p);
		}
	});
	return s;
}

inline void PadIb(std::vector<uint32_t>* ib) {
	while ((ib->size() & 7u) != 0) {
		ib->push_back(kNopPad);
	}
}

// The packets of `ib` in [begin_dw, end_dw), preceded by every state-only packet before begin_dw
// (the actions there have already run): the rest of an IB split at a draw, as its own submission.
inline std::vector<uint32_t> SliceIb(std::span<const uint32_t> ib, uint32_t begin_dw, uint32_t end_dw) {
	std::vector<uint32_t> out;
	out.reserve(std::min<size_t>(ib.size(), end_dw) + 8);
	ForEachPacket(ib, [&](const Packet& p) {
		if (p.dw >= end_dw || (p.dw < begin_dw && !IsStateOnly(p.op))) {
			return;
		}
		out.insert(out.end(), ib.begin() + p.dw, ib.begin() + p.dw + p.size);
	});
	PadIb(&out);
	return out;
}

// --- Image descriptors ---

// SQ_SEL values of a T# DST_SEL field.
constexpr uint32_t kSel0 = 0, kSel1 = 1, kSelX = 4, kSelY = 5, kSelZ = 6, kSelW = 7;

// DST_SEL (x | y << 3 | z << 6 | w << 9) that undoes a color target's COMP_SWAP, so the shader sees
// the components in the order the pixel shader wrote them.
inline uint32_t DstSelForSwap(uint32_t components, uint32_t swap) {
	uint32_t sel[4] = {kSel0, kSel0, kSel0, kSel1};
	switch (components) {
		case 1:
			if (swap == 3) {
				sel[3] = kSelX;
			} else {
				sel[0] = kSelX;
			}
			break;
		case 2:
			if (swap == 0) { sel[0] = kSelX; sel[1] = kSelY; }
			if (swap == 1) { sel[0] = kSelX; sel[3] = kSelY; }
			if (swap == 2) { sel[0] = kSelY; sel[1] = kSelX; }
			if (swap == 3) { sel[0] = kSelY; sel[3] = kSelX; }
			break;
		case 3:
			if (swap == 2) { sel[0] = kSelZ; sel[1] = kSelY; sel[2] = kSelX; }
			else { sel[0] = kSelX; sel[1] = kSelY; sel[2] = kSelZ; }
			break;
		default:
			if (swap == 0) { sel[0] = kSelX; sel[1] = kSelY; sel[2] = kSelZ; sel[3] = kSelW; }
			if (swap == 1) { sel[0] = kSelZ; sel[1] = kSelY; sel[2] = kSelX; sel[3] = kSelW; }
			if (swap == 2) { sel[0] = kSelW; sel[1] = kSelZ; sel[2] = kSelY; sel[3] = kSelX; }
			if (swap == 3) { sel[0] = kSelY; sel[1] = kSelZ; sel[2] = kSelW; sel[3] = kSelX; }
			break;
	}
	return sel[0] | (sel[1] << 3u) | (sel[2] << 6u) | (sel[3] << 9u);
}

// A gfx10 T# that reads color target `t` the way a game's sampler view of it would (ASTRO BOT's own
// T#s of its FP16 targets have exactly these fields): `img_format` is
// the image format of CB FORMAT + NUMBER_TYPE (Prospero::ResolveRenderTargetFormat), `components` its
// channel count. DCC is described to the texture unit so it decodes compressed blocks itself. False
// (with a reason) for what it cannot describe.
inline bool TsharpForColorTarget(const ColorTarget& t, uint32_t img_format, uint32_t components, uint32_t out[8],
                                 std::string* why) {
	if (t.base == 0 || t.Format() == 0) {
		*why = "no color target bound";
		return false;
	}
	if (img_format == 0) {
		*why = "unknown format";
		return false;
	}
	if (t.Samples() > 1) {
		*why = "MSAA target (pass its T# instead)";
		return false;
	}
	const uint32_t w     = t.Width() - 1u;
	const uint32_t h     = t.Height() - 1u;
	const uint32_t mip   = (t.view >> 26u) & 0xfu;
	const uint32_t slice = t.view & 0x1fffu;
	const bool     three_d = t.ResourceType() == 2u;
	const uint32_t type  = three_d ? 10u : (t.Slices() > 1 ? 13u : 9u); // SQ_RSRC_IMG_3D / 2D_ARRAY / 2D
	std::memset(out, 0, 8 * sizeof(uint32_t));
	out[0] = static_cast<uint32_t>(t.base >> 8u);
	out[1] = static_cast<uint32_t>((t.base >> 40u) & 0xffu) | ((img_format & 0x1ffu) << 20u) | ((w & 3u) << 30u);
	out[2] = ((w >> 2u) & 0xfffu) | ((h & 0x3fffu) << 14u);
	out[3] = DstSelForSwap(components, t.CompSwap()) | (mip << 12u) | (mip << 16u) | (t.SwizzleMode() << 20u) |
	         (type << 28u);
	out[4] = (t.Slices() - 1u) | (three_d ? 0u : (slice << 16u));
	out[5] = (t.MaxMip() << 4u) | (7u << 20u); // MAX_MIP, PERF_MOD 7
	if (t.Dcc() && t.dcc_base != 0) {
		const uint32_t max_uncomp = (t.dcc_control >> 2u) & 3u;
		const uint32_t max_comp   = (t.dcc_control >> 5u) & 3u;
		const bool     pipe_aligned = ((t.attrib3 >> 30u) & 1u) != 0;
		const uint32_t swap         = t.CompSwap();
		const bool alpha_msb = components == 1 ? swap == 3 : (swap != 2 && swap != 3); // Mesa vi_alpha_is_on_msb
		out[6] = (max_uncomp << 15u) | (max_comp << 17u) | ((pipe_aligned ? 1u : 0u) << 19u) | (1u << 20u) | (1u << 21u) |
		         ((alpha_msb ? 1u : 0u) << 22u) | (static_cast<uint32_t>((t.dcc_base >> 8u) & 0xffu) << 24u);
		out[7] = static_cast<uint32_t>(t.dcc_base >> 16u);
	}
	return true;
}

inline uint64_t TsharpBase(const uint32_t t[8]) {
	return ((t[0] | (static_cast<uint64_t>(t[1] & 0xffu) << 32u))) << 8u;
}
inline uint32_t TsharpFormat(const uint32_t t[8]) { return (t[1] >> 20u) & 0x1ffu; }
inline uint32_t TsharpWidth(const uint32_t t[8]) { return (((t[1] >> 30u) & 3u) | ((t[2] & 0xfffu) << 2u)) + 1u; }
inline uint32_t TsharpHeight(const uint32_t t[8]) { return ((t[2] >> 14u) & 0x3fffu) + 1u; }
inline uint32_t TsharpType(const uint32_t t[8]) { return t[3] >> 28u; }
inline uint32_t TsharpSwizzle(const uint32_t t[8]) { return (t[3] >> 20u) & 0x1fu; }
inline uint32_t TsharpDepth(const uint32_t t[8]) { return (t[4] & 0x1fffu) + 1u; }
inline bool     TsharpCompressed(const uint32_t t[8]) { return ((t[6] >> 21u) & 1u) != 0; }
inline uint64_t TsharpMeta(const uint32_t t[8]) {
	return ((static_cast<uint64_t>((t[6] >> 24u) & 0xffu)) | (static_cast<uint64_t>(t[7]) << 8u)) << 8u;
}

// Plausible gfx10 T# whose base is `base` (for finding a game's own descriptor of an image in memory).
inline bool LooksLikeTsharp(const uint32_t t[8], uint64_t base) {
	const uint32_t type = TsharpType(t);
	return TsharpBase(t) == base && type >= 8u && type <= 15u && TsharpFormat(t) != 0 && TsharpFormat(t) < 160u;
}

} // namespace Libs::Graphics::NativeGpu::Capture

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVECAPTURE_H_
