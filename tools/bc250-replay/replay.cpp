// Replays a frame captured by the BC-250 native backend (KYTY_NATIVE_GPU_CAPTURE, format in
// src/graphics/native_gpu/nativeCapture.h) on the GPU without the game, and reads any image back
// through the texture unit. See README.md.

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/native_gpu/nativeCapture.h"
#include "graphics/native_gpu/nativeSubmit.h"

#include "readback_shaders.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace Libs::Graphics::NativeGpu;
using namespace Libs::Graphics::NativeGpu::Capture;
namespace Prospero = Libs::Graphics::Prospero;

namespace {

[[noreturn]] void Fail(const std::string& message) {
	std::fprintf(stderr, "bc250-replay: %s\n", message.c_str());
	std::exit(1);
}

std::vector<uint8_t> ReadFile(const std::string& path) {
	std::vector<uint8_t> out;
	std::FILE*           f = std::fopen(path.c_str(), "rb");
	if (f == nullptr) {
		Fail("cannot read " + path);
	}
	std::fseek(f, 0, SEEK_END);
	out.resize(static_cast<size_t>(std::ftell(f)));
	std::fseek(f, 0, SEEK_SET);
	if (!out.empty() && std::fread(out.data(), 1, out.size(), f) != out.size()) {
		Fail("short read of " + path);
	}
	std::fclose(f);
	return out;
}

void WriteFile(const std::string& path, const void* data, size_t bytes) {
	std::FILE* f = std::fopen(path.c_str(), "wb");
	if (f == nullptr || std::fwrite(data, 1, bytes, f) != bytes) {
		Fail("cannot write " + path);
	}
	std::fclose(f);
}

uint64_t ParseU64(const std::string& s) {
	char*          end = nullptr;
	const uint64_t v   = std::strtoull(s.c_str(), &end, 0);
	if (end == s.c_str()) {
		Fail("bad number '" + s + "'");
	}
	return v;
}

std::vector<std::string> Split(const std::string& s, char sep) {
	std::vector<std::string> out;
	size_t                   at = 0;
	while (at <= s.size()) {
		const size_t next = s.find(sep, at);
		out.push_back(s.substr(at, next == std::string::npos ? std::string::npos : next - at));
		if (next == std::string::npos) {
			break;
		}
		at = next + 1;
	}
	return out;
}

// --- options ---

struct TsharpSource {
	enum Kind { Inline, AtVa, UserData } kind = Inline;
	uint32_t    dwords[8] = {};
	uint64_t    va        = 0;
	uint32_t    stage_lo  = 0; // SH offset of the stage's USER_DATA_0
	uint32_t    reg       = 0;
	uint64_t    offset    = 0;
	bool        inline_ud = false; // the T# is in the user data registers themselves
	std::string label;
};

struct Options {
	std::string              dir;
	std::string              out;
	std::string              device = "/dev/dri/renderD128";
	bool                     list   = false;
	std::string              at; // steps
	std::string              rt = "auto";
	std::vector<TsharpSource> tsharps;
	std::vector<uint64_t>    find_tsharp;
	std::vector<std::pair<uint64_t, uint64_t>> dumps;
	uint32_t                 slice = 0;
	uint32_t                 mip   = 0;
	int64_t                  crop[4] = {-1, -1, -1, -1};
	float                    scale = 1.0f;
	bool                     srgb  = false;
	bool                     alpha = false;
	bool                     raw   = false;
	bool                     no_deltas = false;
	bool                     verify    = false;
	double                   timeout   = 10.0;
};

void Usage() {
	std::fprintf(stderr,
	             "usage: bc250-replay CAPTURE_DIR [options]\n"
	             "  --list                 every CS, guest buffer and draw/dispatch (global index, op, shader, targets)\n"
	             "  --at STEPS             read back after these draws/dispatches: N, A-B, A-B/S, comma lists;\n"
	             "                         cs:N after the whole CS N; end after the last CS (default)\n"
	             "  --rt SLOTS             color targets per step: auto (those the draw writes, default), all, none, 0,2,...\n"
	             "  --tsharp D0,...,D7     also read an image through this T# (hex dwords)\n"
	             "  --tsharp-at VA         ... through the T# stored at VA\n"
	             "  --tsharp-ud S:R[+OFF]  ... through the T# in the table user data R/R+1 of stage S (ps|gs|cs) point to\n"
	             "                         (+OFF bytes); S:Ri takes the T# from the user data registers R..R+7 themselves\n"
	             "  --find-tsharp VA       list T#s in memory whose base is VA\n"
	             "  --dump VA:SIZE         save replay memory [VA, VA+SIZE) at each step\n"
	             "  --slice N --mip N      array slice / mip level to read\n"
	             "  --crop X,Y,W,H         read only that rectangle\n"
	             "  --scale F              multiply float and integer formats before the PNG (default 1)\n"
	             "  --srgb                 sRGB-encode float formats for the PNG; --alpha keeps alpha (else opaque)\n"
	             "  --raw                  also write the RGBA32 texels (.rgba32, with .txt)\n"
	             "  --no-deltas            apply only the first capture point, later CSs see what the replay left\n"
	             "  --verify               before each CS, count the captured changed pages the replay reproduced\n"
	             "  --out DIR              output directory (default CAPTURE_DIR/replay)\n");
	std::exit(2);
}

uint32_t StageUserData(const std::string& stage) {
	if (stage == "ps") {
		return kShUserDataPs;
	}
	if (stage == "gs") {
		return kShUserDataGs;
	}
	if (stage == "cs") {
		return kShComputeUserData;
	}
	Fail("unknown stage '" + stage + "' (ps, gs, cs)");
}

Options ParseOptions(int argc, char** argv) {
	Options o;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto              value = [&]() -> std::string {
            if (i + 1 >= argc) {
                Usage();
            }
            return argv[++i];
		};
		if (a == "--list") {
			o.list = true;
		} else if (a == "--at") {
			o.at = value();
		} else if (a == "--rt") {
			o.rt = value();
		} else if (a == "--tsharp") {
			TsharpSource t;
			const auto   words = Split(value(), ',');
			if (words.size() != 8) {
				Fail("--tsharp needs 8 dwords");
			}
			for (int k = 0; k < 8; k++) {
				t.dwords[k] = static_cast<uint32_t>(std::strtoul(words[k].c_str(), nullptr, 16));
			}
			t.label = "tsharp";
			o.tsharps.push_back(t);
		} else if (a == "--tsharp-at") {
			TsharpSource t;
			t.kind  = TsharpSource::AtVa;
			t.va    = ParseU64(value());
			t.label = "at";
			o.tsharps.push_back(t);
		} else if (a == "--tsharp-ud") {
			TsharpSource t;
			t.kind             = TsharpSource::UserData;
			const std::string v = value();
			const size_t      colon = v.find(':');
			if (colon == std::string::npos) {
				Fail("--tsharp-ud STAGE:REG[+OFF]");
			}
			t.stage_lo        = StageUserData(v.substr(0, colon));
			std::string rest  = v.substr(colon + 1);
			if (const size_t plus = rest.find('+'); plus != std::string::npos) {
				t.offset = ParseU64(rest.substr(plus + 1));
				rest     = rest.substr(0, plus);
			}
			if (!rest.empty() && rest.back() == 'i') {
				t.inline_ud = true;
				rest.pop_back();
			}
			t.reg   = static_cast<uint32_t>(ParseU64(rest));
			t.label = "ud_" + v.substr(0, colon) + std::to_string(t.reg);
			o.tsharps.push_back(t);
		} else if (a == "--find-tsharp") {
			o.find_tsharp.push_back(ParseU64(value()));
		} else if (a == "--dump") {
			const auto p = Split(value(), ':');
			if (p.size() != 2) {
				Fail("--dump VA:SIZE");
			}
			o.dumps.emplace_back(ParseU64(p[0]), ParseU64(p[1]));
		} else if (a == "--slice") {
			o.slice = static_cast<uint32_t>(ParseU64(value()));
		} else if (a == "--mip") {
			o.mip = static_cast<uint32_t>(ParseU64(value()));
		} else if (a == "--crop") {
			const auto p = Split(value(), ',');
			if (p.size() != 4) {
				Fail("--crop X,Y,W,H");
			}
			for (int k = 0; k < 4; k++) {
				o.crop[k] = static_cast<int64_t>(ParseU64(p[k]));
			}
		} else if (a == "--scale") {
			o.scale = std::strtof(value().c_str(), nullptr);
		} else if (a == "--srgb") {
			o.srgb = true;
		} else if (a == "--alpha") {
			o.alpha = true;
		} else if (a == "--raw") {
			o.raw = true;
		} else if (a == "--no-deltas") {
			o.no_deltas = true;
		} else if (a == "--verify") {
			o.verify = true;
		} else if (a == "--out") {
			o.out = value();
		} else if (a == "--device") {
			o.device = value();
		} else if (a == "--timeout") {
			o.timeout = std::strtod(value().c_str(), nullptr);
		} else if (!a.empty() && a[0] != '-' && o.dir.empty()) {
			o.dir = a;
		} else {
			Usage();
		}
	}
	if (o.dir.empty()) {
		Usage();
	}
	if (o.out.empty()) {
		o.out = o.dir + "/replay";
	}
	return o;
}

// --- the capture in memory ---

struct Loaded {
	Manifest                            manifest;
	std::vector<std::vector<uint32_t>>  ibs;
	std::vector<std::vector<Mapping>>   maps;
	std::vector<Action>                 actions;
};

Loaded Load(const std::string& dir) {
	Loaded      l;
	std::string error;
	if (!l.manifest.Read(dir + "/manifest.txt", &error)) {
		Fail(error);
	}
	for (const auto& file: l.manifest.maps) {
		std::vector<Mapping> m;
		if (!ReadMappings(dir + "/" + file, &m)) {
			Fail("cannot read " + file);
		}
		l.maps.push_back(std::move(m));
	}
	for (uint32_t i = 0; i < l.manifest.cs.size(); i++) {
		const auto            bytes = ReadFile(dir + "/" + l.manifest.cs[i].ib_file);
		std::vector<uint32_t> ib(bytes.size() / 4);
		std::memcpy(ib.data(), bytes.data(), ib.size() * 4);
		RegState state;
		ListActions(ib, i, l.manifest.cs[i].parts, &l.actions, &state);
		l.ibs.push_back(std::move(ib));
	}
	return l;
}

// "16_16_16_16f" style name of a CB FORMAT and NUMBER_TYPE.
std::string FormatName(uint32_t format, uint32_t number) {
	static const char* const kFormats[32] = {"inv", "8", "16", "8_8", "32", "16_16", "10_11_11", "11_11_10", "10_10_10_2",
	                                         "2_10_10_10", "8_8_8_8", "32_32", "16_16_16_16", "f13", "32_32_32_32", "f15",
	                                         "5_6_5", "1_5_5_5", "5_5_5_1", "4_4_4_4", "8_24", "24_8", "x24_8_32", "f23",
	                                         "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31"};
	static const char* const kNumbers[8] = {"un", "sn", "n2", "n3", "ui", "si", "srgb", "f"};
	return std::string(kFormats[format & 31u]) + kNumbers[number & 7u];
}

void List(const Loaded& l) {
	const Manifest& m = l.manifest;
	std::printf("capture of flips %llu-%llu: %zu CS, %zu regions, %zu mapping versions, GDS %u bytes\n",
	            static_cast<unsigned long long>(m.start_flip), static_cast<unsigned long long>(m.end_flip), m.cs.size(),
	            m.regions.size(), m.maps.size(), m.gds_size);
	uint64_t bytes = 0;
	for (const auto& r: m.regions) {
		bytes += r.size;
	}
	std::printf("regions: %llu MiB\n", static_cast<unsigned long long>(bytes >> 20u));
	for (const auto& f: m.flips) {
		std::printf("flip %llu presented after CS %u: 0x%llx %ux%u fmt %u tile 0x%x\n", static_cast<unsigned long long>(f.n),
		            f.after_cs, static_cast<unsigned long long>(f.addr), f.width, f.height, f.pixel_format, f.tile_mode);
	}
	size_t next = 0;
	for (uint32_t i = 0; i < m.cs.size(); i++) {
		std::printf("CS %u: %u dw, map %u", i, m.cs[i].ib_dw, m.cs[i].map_version);
		for (const auto& p: m.cs[i].parts) {
			std::printf(", %s %llu", p.acb ? "acb" : "dcb", static_cast<unsigned long long>(p.buffer));
		}
		std::printf("\n");
		for (; next < l.actions.size() && l.actions[next].cs == i; next++) {
			const Action& a = l.actions[next];
			const Part* part = a.part < m.cs[i].parts.size() ? &m.cs[i].parts[a.part] : nullptr;
			std::printf("  #%zu dw %u [%s %llu] %s %s 0x%llx", next, a.dw, part != nullptr && part->acb ? "acb" : "dcb",
			            static_cast<unsigned long long>(part != nullptr ? part->buffer : 0), OpName(a.op),
			            IsDispatch(a.op) ? "cs" : "ps", static_cast<unsigned long long>(a.shader));
			for (uint32_t s = 0; s < 8; s++) {
				if ((a.slots >> s) & 1u) {
					const uint32_t info = a.slot_info[s];
					std::printf(" rt%u=0x%llx:%s%s", s, static_cast<unsigned long long>(a.slot_base[s]),
					            FormatName((info >> 2u) & 0x1fu, (info >> 8u) & 0x7u).c_str(), ((info >> 28u) & 1u) != 0 ? ":dcc" : "");
				}
			}
			std::printf("\n");
		}
	}
}

// A step: read back after the packet ending at `end_dw` of CS `cs`.
struct Step {
	uint32_t    cs     = 0;
	uint32_t    end_dw = 0;
	int64_t     action = -1; // -1: the end of the CS
	std::string name;

	bool operator<(const Step& o) const { return cs != o.cs ? cs < o.cs : end_dw < o.end_dw; }
};

std::vector<Step> ParseSteps(const Loaded& l, const std::string& spec) {
	std::set<Step> steps;
	auto           cs_end = [&](uint32_t cs) {
        if (cs >= l.ibs.size()) {
            Fail("no CS " + std::to_string(cs));
        }
        steps.insert(Step {cs, static_cast<uint32_t>(l.ibs[cs].size()), -1, "cs" + std::to_string(cs)});
	};
	if (spec.empty() || spec == "end") {
		if (!l.ibs.empty()) {
			cs_end(static_cast<uint32_t>(l.ibs.size() - 1));
		}
		return {steps.begin(), steps.end()};
	}
	for (const auto& item: Split(spec, ',')) {
		if (item == "end") {
			cs_end(static_cast<uint32_t>(l.ibs.size() - 1));
			continue;
		}
		if (item.rfind("cs:", 0) == 0) {
			cs_end(static_cast<uint32_t>(ParseU64(item.substr(3))));
			continue;
		}
		uint64_t     first = 0, last = 0, stride = 1;
		const size_t slash = item.find('/');
		const auto   range = item.substr(0, slash);
		if (slash != std::string::npos) {
			stride = ParseU64(item.substr(slash + 1));
		}
		const size_t dash = range.find('-');
		first             = ParseU64(range.substr(0, dash));
		last              = dash == std::string::npos ? first : ParseU64(range.substr(dash + 1));
		for (uint64_t k = first; k <= last && stride != 0; k += stride) {
			if (k >= l.actions.size()) {
				Fail("no draw/dispatch #" + std::to_string(k) + " (" + std::to_string(l.actions.size()) + " in the capture)");
			}
			char name[32];
			std::snprintf(name, sizeof(name), "a%05llu", static_cast<unsigned long long>(k));
			steps.insert(Step {l.actions[k].cs, l.actions[k].end_dw, static_cast<int64_t>(k), name});
		}
	}
	return {steps.begin(), steps.end()};
}

// --- the GPU side ---

int Ioctl(int fd, unsigned long request, void* arg) {
	int r = -1;
	do {
		r = ::ioctl(fd, request, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

struct Bo {
	uint32_t handle = 0;
	uint8_t* cpu    = nullptr;
	uint64_t size   = 0;
};

class Replayer {
public:
	Replayer(const Options& o, const Loaded& l): m_opt(o), m_cap(l) {}

	void Run() {
		::mkdir(m_opt.out.c_str(), 0755);
		m_steps = ParseSteps(m_cap, m_opt.at);
		Open();
		const uint32_t last_cs = m_steps.empty() ? 0 : m_steps.back().cs;
		size_t         next    = 0;
		for (uint32_t cs = 0; cs <= last_cs; cs++) {
			Prepare(cs);
			uint32_t begin = 0;
			for (; next < m_steps.size() && m_steps[next].cs == cs; next++) {
				RunSlice(cs, begin, m_steps[next].end_dw);
				begin = m_steps[next].end_dw;
				AtStep(m_steps[next]);
			}
			if (cs < last_cs && begin < m_cap.ibs[cs].size()) {
				RunSlice(cs, begin, static_cast<uint32_t>(m_cap.ibs[cs].size()));
			}
			SaveScratch();
		}
		const bool whole = !m_steps.empty() && m_steps.back().end_dw == m_cap.ibs[last_cs].size();
		if (m_opt.verify && whole && last_cs + 1 == m_cap.ibs.size() && !m_cap.manifest.final_delta.empty()) {
			ApplyDelta(m_cap.manifest.final_delta, "final", false);
		}
		std::fprintf(stderr, "bc250-replay: done, output in %s\n", m_opt.out.c_str());
	}

private:
	void Open() {
		m_fd = ::open(m_opt.device.c_str(), O_RDWR | O_CLOEXEC);
		if (m_fd < 0) {
			Fail("cannot open " + m_opt.device);
		}
		// The same GDS partition as in the game, else none.
		::setenv("KYTY_NATIVE_GPU_GDS", m_cap.manifest.gds_spec.empty() ? "0" : m_cap.manifest.gds_spec.c_str(), 1);
		::unsetenv("KYTY_NATIVE_GPU_PRIMARY");
		m_sub = NativeSubmitter::Create(m_fd);
		if (m_sub == nullptr) {
			Fail("submitter setup failed");
		}
		if (m_sub->GdsSize() != m_cap.manifest.gds_size) {
			std::fprintf(stderr, "bc250-replay: warning: GDS is %u bytes here, %u in the capture\n", m_sub->GdsSize(),
			             m_cap.manifest.gds_size);
		}
		const auto& regions = m_cap.manifest.regions;
		m_region_bo.resize(regions.size());
		uint64_t total = 0;
		for (const auto& r: regions) {
			if (r.kind == RegionKind::Scratch) {
				continue;
			}
			Bo& bo = m_region_bo[r.id];
			bo     = CreateBo(r.kind == RegionKind::Gds ? std::max<uint64_t>(r.size, 65536) : r.size);
			if (r.kind == RegionKind::Gds) {
				MapVa(bo, kStagingVa, 0, bo.size, true);
			}
			if (r.since == 0) {
				LoadRegion(r);
			}
			total += bo.size;
		}
		m_code    = CreateBo(65536);
		std::memcpy(m_code.cpu, kReadback_2d, sizeof(kReadback_2d));
		std::memcpy(m_code.cpu + 0x1000, kReadback_2d_array, sizeof(kReadback_2d_array));
		std::memcpy(m_code.cpu + 0x2000, kReadback_3d, sizeof(kReadback_3d));
		MapVa(m_code, kCodeVa, 0, m_code.size, false);
		std::fprintf(stderr, "bc250-replay: %zu regions, %llu MiB in GTT\n", regions.size(),
		             static_cast<unsigned long long>(total >> 20u));
	}

	Bo CreateBo(uint64_t size) {
		size = (size + kPageSize - 1) & ~static_cast<uint64_t>(kPageSize - 1);
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size   = size;
		c.in.alignment = size >= (2u << 20u) ? (2u << 20u) : kPageSize;
		c.in.domains   = AMDGPU_GEM_DOMAIN_GTT;
		if (Ioctl(m_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c) != 0) {
			Fail("GEM_CREATE of " + std::to_string(size >> 10u) + " KiB failed");
		}
		union drm_amdgpu_gem_mmap m;
		std::memset(&m, 0, sizeof(m));
		m.in.handle = c.out.handle;
		if (Ioctl(m_fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &m) != 0) {
			Fail("GEM_MMAP failed");
		}
		void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, static_cast<off_t>(m.out.addr_ptr));
		if (p == MAP_FAILED) {
			Fail("mmap of a GEM buffer failed");
		}
		m_handles.push_back(c.out.handle);
		return Bo {c.out.handle, static_cast<uint8_t*>(p), size};
	}

	void MapVa(const Bo& bo, uint64_t va, uint64_t offset, uint64_t size, bool writable, uint32_t op = AMDGPU_VA_OP_MAP) {
		struct drm_amdgpu_gem_va v;
		std::memset(&v, 0, sizeof(v));
		v.handle       = bo.handle;
		v.operation    = op;
		v.flags        = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_EXECUTABLE | (writable ? AMDGPU_VM_PAGE_WRITEABLE : 0u);
		v.va_address   = va;
		v.offset_in_bo = offset;
		v.map_size     = size;
		if (Ioctl(m_fd, DRM_IOCTL_AMDGPU_GEM_VA, &v) != 0) {
			char msg[128];
			std::snprintf(msg, sizeof(msg), "GEM_VA op %u at 0x%llx (+0x%llx) failed", op, static_cast<unsigned long long>(va),
			              static_cast<unsigned long long>(size));
			Fail(msg);
		}
	}

	void ClearVa(uint64_t va, uint64_t size) {
		struct drm_amdgpu_gem_va v;
		std::memset(&v, 0, sizeof(v));
		v.operation  = AMDGPU_VA_OP_CLEAR;
		v.va_address = va;
		v.map_size   = size;
		Ioctl(m_fd, DRM_IOCTL_AMDGPU_GEM_VA, &v);
	}

	// A region file is sparse: copy its data extents only.
	void LoadRegion(const Region& r) {
		uint8_t* dst = RegionCpu(r.id);
		const int fd = ::open((m_opt.dir + "/" + r.file).c_str(), O_RDONLY | O_CLOEXEC);
		if (fd < 0) {
			Fail("cannot open " + r.file);
		}
		std::memset(dst, 0, r.size);
		off_t pos = 0;
		while (static_cast<uint64_t>(pos) < r.size) {
			const off_t data = ::lseek(fd, pos, SEEK_DATA);
			if (data < 0) {
				break;
			}
			off_t hole = ::lseek(fd, data, SEEK_HOLE);
			if (hole < 0 || static_cast<uint64_t>(hole) > r.size) {
				hole = static_cast<off_t>(r.size);
			}
			for (off_t at = data; at < hole;) {
				const ssize_t n = ::pread(fd, dst + at, static_cast<size_t>(hole - at), at);
				if (n <= 0) {
					Fail("short read of " + r.file);
				}
				at += n;
			}
			pos = hole;
		}
		::close(fd);
	}

	uint8_t* RegionCpu(uint32_t id) {
		if (m_cap.manifest.regions[id].kind == RegionKind::Scratch) {
			return m_scratch;
		}
		return m_region_bo[id].cpu;
	}

	void ApplyMapping(uint32_t version) {
		if (version == m_map_version) {
			return;
		}
		const auto&      next = m_cap.maps[version];
		std::set<Mapping> keep;
		if (m_map_version != ~0u) {
			std::set<Mapping> want(next.begin(), next.end());
			for (const auto& m: m_cap.maps[m_map_version]) {
				if (want.count(m) == 0) {
					ClearVa(m.va, m.size);
				} else {
					keep.insert(m);
				}
			}
		}
		for (const auto& m: next) {
			if (keep.count(m) == 0) {
				const Region& r = m_cap.manifest.regions[m.region];
				MapVa(m_region_bo[m.region], m.va, m.offset, m.size, r.writable, AMDGPU_VA_OP_REPLACE);
			}
		}
		m_map_version = version;
	}

	// Everything CS `cs` saw in the game: the mapping, regions that appeared, the changed pages,
	// the scratch area and GDS.
	void Prepare(uint32_t cs) {
		const Cs& c = m_cap.manifest.cs[cs];
		ApplyMapping(c.map_version);
		for (const auto& r: m_cap.manifest.regions) {
			if (r.since == cs && cs != 0 && r.kind != RegionKind::Scratch) {
				LoadRegion(r);
			}
		}
		if (cs == 0 || !m_opt.no_deltas) {
			ApplyDelta(c.delta_file, "CS " + std::to_string(cs), true);
		}
		std::memcpy(const_cast<uint8_t*>(m_sub->ScratchCpu()), m_scratch, sizeof(m_scratch));
		// Caches may still hold what the CPU has just overwritten; then GDS from the staging copy.
		std::vector<uint32_t> ib;
		EmitInvalidate(&ib);
		for (const auto& r: m_cap.manifest.regions) {
			if (r.kind == RegionKind::Gds && m_sub->GdsSize() != 0) {
				ib.push_back(Pkt3(IT_DMA_DATA, 6));
				ib.push_back((1u << 31u) | (3u << 29u) | (1u << 20u)); // CP_SYNC, SRC_SEL TC_L2, DST_SEL GDS
				ib.push_back(static_cast<uint32_t>(kStagingVa));
				ib.push_back(static_cast<uint32_t>(kStagingVa >> 32u));
				ib.push_back(0);
				ib.push_back(0);
				ib.push_back(static_cast<uint32_t>(std::min<uint64_t>(r.size, m_sub->GdsSize())));
			}
		}
		Submit(ib, "prologue of CS " + std::to_string(cs));
	}

	void SaveScratch() { std::memcpy(m_scratch, const_cast<const uint8_t*>(m_sub->ScratchCpu()), sizeof(m_scratch)); }

	// ACQUIRE_MEM invalidating (and writing back) every cache level.
	static void EmitInvalidate(std::vector<uint32_t>* ib) {
		ib->push_back(Pkt3(IT_ACQUIRE_MEM, 7));
		ib->push_back(0);
		ib->push_back(0xffffffffu);
		ib->push_back(0x00ffffffu);
		ib->push_back(0);
		ib->push_back(0);
		ib->push_back(0x0000000au);
		// GCR_CNTL: GLI_INV, GLM_WB, GLM_INV, GLK_INV, GLV_INV, GL1_INV, GL2_INV, GL2_WB.
		ib->push_back(1u | (1u << 4u) | (1u << 5u) | (1u << 7u) | (1u << 8u) | (1u << 9u) | (1u << 14u) | (1u << 15u));
	}

	void ApplyDelta(const std::string& file, const std::string& what, bool apply) {
		const auto bytes = ReadFile(m_opt.dir + "/" + file);
		uint32_t   header[2];
		if (bytes.size() < sizeof(header)) {
			Fail(file + " is truncated");
		}
		std::memcpy(header, bytes.data(), sizeof(header));
		if (header[0] != kDeltaMagic) {
			Fail(file + " is not a delta");
		}
		const size_t rec = 8 + kPageSize;
		if (bytes.size() < sizeof(header) + static_cast<size_t>(header[1]) * rec) {
			Fail(file + " is truncated");
		}
		uint32_t                     same = 0;
		std::map<uint32_t, uint32_t> differ_by_region;
		uint32_t                     differ_by_kind[5] = {};
		for (uint32_t i = 0; i < header[1]; i++) {
			const uint8_t* p = bytes.data() + sizeof(header) + static_cast<size_t>(i) * rec;
			uint32_t       region = 0, page = 0;
			std::memcpy(&region, p, 4);
			std::memcpy(&page, p + 4, 4);
			if (region >= m_cap.manifest.regions.size()) {
				Fail(file + ": bad region");
			}
			const Region&  r      = m_cap.manifest.regions[region];
			const uint64_t at     = static_cast<uint64_t>(page) * kPageSize;
			if (at >= r.size) {
				continue;
			}
			const uint64_t len = std::min<uint64_t>(kPageSize, r.size - at);
			uint8_t*       dst = RegionCpu(region) + at;
			if (m_opt.verify) {
				if (std::memcmp(dst, p + 8, len) == 0) {
					same++;
				} else {
					differ_by_region[region]++;
					differ_by_kind[static_cast<uint32_t>(r.kind)]++;
				}
			}
			if (apply) {
				std::memcpy(dst, p + 8, len);
			}
		}
		if (m_opt.verify) {
			std::fprintf(stderr, "bc250-replay: %s: %u changed pages captured, %u already equal here; differing:", what.c_str(),
			             header[1], same);
			for (uint32_t k = 0; k < 5; k++) {
				if (differ_by_kind[k] != 0) {
					std::fprintf(stderr, " %s %u", RegionKindName(static_cast<RegionKind>(k)), differ_by_kind[k]);
				}
			}
			std::vector<std::pair<uint32_t, uint32_t>> top;
			for (const auto& [region, n]: differ_by_region) {
				top.emplace_back(n, region);
			}
			std::sort(top.rbegin(), top.rend());
			for (size_t i = 0; i < top.size() && i < 6; i++) {
				const Region& r  = m_cap.manifest.regions[top[i].second];
				uint64_t      va = 0;
				for (const auto& m: m_cap.maps[m_map_version]) {
					if (m.region == top[i].second && m.offset == 0) {
						va = m.va;
						break;
					}
				}
				std::fprintf(stderr, "%s r%u (%s 0x%llx at VA 0x%llx) %u", i == 0 ? "; most in" : ",", top[i].second,
				             RegionKindName(r.kind), static_cast<unsigned long long>(r.key), static_cast<unsigned long long>(va),
				             top[i].first);
			}
			std::fprintf(stderr, "\n");
		}
	}

	void Submit(std::vector<uint32_t>& ib, const std::string& what) {
		PadIb(&ib);
		const NativeSubmitter::Result r = m_sub->Submit(ib, m_handles, m_opt.timeout);
		if (!r.submitted || !r.completed || r.reset) {
			std::fprintf(stderr, "bc250-replay: %s: %s (err %d)\n", what.c_str(),
			             r.timed_out ? "TIMED OUT" : (r.reset ? "context RESET" : "failed"), r.err);
			std::exit(3);
		}
	}

	void RunSlice(uint32_t cs, uint32_t begin, uint32_t end) {
		const auto& ib = m_cap.ibs[cs];
		std::vector<uint32_t> slice;
		if (begin == 0 && end == ib.size()) {
			slice = ib;
		} else {
			slice = SliceIb(ib, begin, end);
		}
		Submit(slice, "CS " + std::to_string(cs) + " [" + std::to_string(begin) + ", " + std::to_string(end) + ")");
	}

	// GPU VA -> CPU bytes through the current mapping, clipped to that mapping.
	uint8_t* VaCpu(uint64_t va, uint64_t* avail) {
		for (const auto& m: m_cap.maps[m_map_version]) {
			if (va >= m.va && va < m.va + m.size) {
				*avail = m.va + m.size - va;
				return RegionCpu(m.region) + m.offset + (va - m.va);
			}
		}
		return nullptr;
	}

	bool ReadVa(uint64_t va, void* dst, uint64_t size) {
		auto* out = static_cast<uint8_t*>(dst);
		while (size > 0) {
			uint64_t       avail = 0;
			const uint8_t* src   = VaCpu(va, &avail);
			if (src == nullptr) {
				return false;
			}
			const uint64_t n = std::min(avail, size);
			std::memcpy(out, src, n);
			out += n;
			va += n;
			size -= n;
		}
		return true;
	}

	void AtStep(const Step& step) {
		const RegState state = StateAt(m_cap.ibs[step.cs], step.end_dw);
		std::fprintf(stderr, "bc250-replay: step %s (CS %u dw %u)\n", step.name.c_str(), step.cs, step.end_dw);
		// Color targets.
		uint32_t slots = 0;
		if (m_opt.rt == "auto") {
			slots = step.action >= 0 ? m_cap.actions[static_cast<size_t>(step.action)].slots : 0u;
		} else if (m_opt.rt == "all") {
			for (uint32_t s = 0; s < 8; s++) {
				const ColorTarget t = ColorTargetAt(state, s);
				slots |= (t.Format() != 0 && t.base != 0) ? (1u << s) : 0u;
			}
		} else if (m_opt.rt != "none") {
			for (const auto& s: Split(m_opt.rt, ',')) {
				slots |= 1u << (ParseU64(s) & 7u);
			}
		}
		for (uint32_t s = 0; s < 8; s++) {
			if (((slots >> s) & 1u) == 0) {
				continue;
			}
			const ColorTarget t   = ColorTargetAt(state, s);
			const auto        enc = Prospero::ResolveRenderTargetFormat(static_cast<Prospero::ChannelLayout>(t.Format()),
			                                                            static_cast<Prospero::ChannelType>(t.NumberType()));
			uint32_t          tsharp[8];
			std::string       why;
			char              name[64];
			std::snprintf(name, sizeof(name), "%s_rt%u_%llx", step.name.c_str(), s, static_cast<unsigned long long>(t.base));
			if (!TsharpForColorTarget(t, static_cast<uint32_t>(enc.buffer_format), enc.components, tsharp, &why)) {
				std::fprintf(stderr, "  rt%u: %s\n", s, why.c_str());
				continue;
			}
			char desc[256];
			std::snprintf(desc, sizeof(desc), "rt%u fmt %u num %u swap %u sw 0x%x %s%s", s, t.Format(), t.NumberType(), t.CompSwap(),
			              t.SwizzleMode(), t.Dcc() ? "dcc " : "", t.FastClear() ? "fast-clear" : "");
			Readback(tsharp, name, desc);
		}
		// Explicit T#s.
		for (size_t k = 0; k < m_opt.tsharps.size(); k++) {
			const TsharpSource& src = m_opt.tsharps[k];
			uint32_t            tsharp[8];
			uint64_t            from = 0;
			if (src.kind == TsharpSource::Inline) {
				std::memcpy(tsharp, src.dwords, sizeof(tsharp));
			} else if (src.kind == TsharpSource::UserData && src.inline_ud) {
				for (uint32_t d = 0; d < 8; d++) {
					tsharp[d] = state.sh[src.stage_lo + src.reg + d];
				}
			} else {
				from = src.kind == TsharpSource::AtVa
				           ? src.va
				           : (state.sh[src.stage_lo + src.reg] | (static_cast<uint64_t>(state.sh[src.stage_lo + src.reg + 1] & 0xffffu) << 32u)) +
				                 src.offset;
				if (!ReadVa(from, tsharp, sizeof(tsharp))) {
					std::fprintf(stderr, "  %s: 0x%llx is not mapped\n", src.label.c_str(), static_cast<unsigned long long>(from));
					continue;
				}
			}
			char name[64];
			std::snprintf(name, sizeof(name), "%s_t%zu_%s_%llx", step.name.c_str(), k, src.label.c_str(),
			              static_cast<unsigned long long>(TsharpBase(tsharp)));
			char desc[96];
			std::snprintf(desc, sizeof(desc), "%s from 0x%llx", src.label.c_str(), static_cast<unsigned long long>(from));
			Readback(tsharp, name, desc);
		}
		for (uint64_t base: m_opt.find_tsharp) {
			FindTsharps(base);
		}
		for (const auto& [va, size]: m_opt.dumps) {
			std::vector<uint8_t> bytes(size);
			if (!ReadVa(va, bytes.data(), size)) {
				std::fprintf(stderr, "  dump 0x%llx: not mapped (in full)\n", static_cast<unsigned long long>(va));
				continue;
			}
			char name[96];
			std::snprintf(name, sizeof(name), "%s/%s_mem_%llx.bin", m_opt.out.c_str(), step.name.c_str(), static_cast<unsigned long long>(va));
			WriteFile(name, bytes.data(), bytes.size());
		}
	}

	void FindTsharps(uint64_t base) {
		// Region bytes -> guest VAs, through the current mapping.
		uint32_t found = 0;
		for (const auto& m: m_cap.maps[m_map_version]) {
			if (m_cap.manifest.regions[m.region].kind == RegionKind::Dummy) {
				continue; // mapped wherever guest memory is not resident
			}
			const uint8_t* p = RegionCpu(m.region) + m.offset;
			for (uint64_t at = 0; at + 32 <= m.size; at += 16) {
				uint32_t t[8];
				std::memcpy(t, p + at, sizeof(t));
				if (LooksLikeTsharp(t, base) && found++ < 32) {
					std::printf("T# at 0x%llx: %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x (%ux%u fmt %u type %u sw 0x%x%s)\n",
					            static_cast<unsigned long long>(m.va + at), t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7],
					            TsharpWidth(t), TsharpHeight(t), TsharpFormat(t), TsharpType(t), TsharpSwizzle(t),
					            TsharpCompressed(t) ? " dcc" : "");
				}
			}
		}
		std::printf("%u T#(s) with base 0x%llx\n", found, static_cast<unsigned long long>(base));
	}

	// Reads the image `t` describes with the readback shader into linear RGBA32 and writes a PNG.
	void Readback(uint32_t t_in[8], const std::string& name, const std::string& desc) {
		uint32_t t[8];
		std::memcpy(t, t_in, sizeof(t));
		if (m_opt.mip != 0) {
			t[3] = (t[3] & ~0xff000u) | (m_opt.mip << 12u) | (m_opt.mip << 16u);
		}
		const uint32_t type = TsharpType(t);
		const uint32_t full_w = std::max(1u, TsharpWidth(t) >> m_opt.mip);
		const uint32_t full_h = std::max(1u, TsharpHeight(t) >> m_opt.mip);
		uint32_t       x0 = 0, y0 = 0, w = full_w, h = full_h;
		if (m_opt.crop[0] >= 0) {
			x0 = std::min<uint32_t>(static_cast<uint32_t>(m_opt.crop[0]), full_w - 1);
			y0 = std::min<uint32_t>(static_cast<uint32_t>(m_opt.crop[1]), full_h - 1);
			w  = std::min<uint32_t>(static_cast<uint32_t>(m_opt.crop[2]), full_w - x0);
			h  = std::min<uint32_t>(static_cast<uint32_t>(m_opt.crop[3]), full_h - y0);
		}
		uint64_t code = kCodeVa; // 2D
		if (type == 13u || type == 12u || type == 11u) {
			code = kCodeVa + 0x1000; // arrays and cubes: slice as the third coordinate
		} else if (type == 10u) {
			code = kCodeVa + 0x2000;
		} else if (type != 9u) {
			std::fprintf(stderr, "  %s: image type %u is not supported\n", name.c_str(), type);
			return;
		}
		const uint64_t bytes = static_cast<uint64_t>(w) * h * 16u;
		if (m_out.size < bytes) {
			if (m_out.handle != 0) {
				ClearVa(kReadbackVa, m_out.size);
				::munmap(m_out.cpu, m_out.size);
				struct drm_gem_close c;
				std::memset(&c, 0, sizeof(c));
				c.handle = m_out.handle;
				Ioctl(m_fd, DRM_IOCTL_GEM_CLOSE, &c);
				m_handles.erase(std::find(m_handles.begin(), m_handles.end(), m_out.handle));
			}
			m_out = CreateBo(bytes);
			MapVa(m_out, kReadbackVa, 0, m_out.size, true);
		}
		std::memset(m_out.cpu, 0xff, bytes); // NaN where nothing was written

		std::vector<uint32_t> ib;
		EmitInvalidate(&ib);
		auto set_sh = [&](uint32_t reg, std::initializer_list<uint32_t> values) {
			ib.push_back(Pkt3(IT_SET_SH_REG, static_cast<uint32_t>(values.size()) + 1, true));
			ib.push_back(reg);
			ib.insert(ib.end(), values);
		};
		// wave64, 16 VGPRs, 16 user SGPRs, group id x/y, thread id x/y.
		const uint32_t rsrc1 = 3u | (0xc0u << 12u) | (1u << 21u) | (1u << 30u);
		const uint32_t rsrc2 = (16u << 1u) | (1u << 7u) | (1u << 8u) | (1u << 11u);
		set_sh(0x20C, {static_cast<uint32_t>(code >> 8u), static_cast<uint32_t>(code >> 40u)});
		set_sh(0x212, {rsrc1, rsrc2});
		set_sh(0x228, {0u});
		set_sh(0x215, {0u});
		set_sh(0x218, {0u});
		set_sh(0x216, {0xffffffffu, 0xffffffffu});
		set_sh(0x219, {0xffffffffu, 0xffffffffu});
		set_sh(0x204, {0u, 0u, 0u});
		set_sh(0x207, {8u, 8u, 1u});
		set_sh(0x240, {t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], static_cast<uint32_t>(kReadbackVa),
		               static_cast<uint32_t>(kReadbackVa >> 32u), w, h, m_opt.slice, x0, y0, 0u});
		ib.push_back(Pkt3(IT_DISPATCH_DIRECT, 4, true));
		ib.push_back((w + 7) / 8);
		ib.push_back((h + 7) / 8);
		ib.push_back(1);
		ib.push_back(1u | (1u << 2u) | (1u << 6u)); // COMPUTE_SHADER_EN, FORCE_START_AT_000, ORDER_MODE
		EmitInvalidate(&ib);
		Submit(ib, "readback " + name);

		const auto  format = static_cast<Prospero::BufferFormat>(TsharpFormat(t));
		const auto  num    = Prospero::SampledTextureNumericClass(format);
		const auto* texels = reinterpret_cast<const float*>(m_out.cpu);
		const auto* ints   = reinterpret_cast<const int32_t*>(m_out.cpu);
		const auto* uints  = reinterpret_cast<const uint32_t*>(m_out.cpu);
		const bool  is_srgb = format == Prospero::BufferFormat::k8Srgb || format == Prospero::BufferFormat::k8_8Srgb ||
		                     format == Prospero::BufferFormat::k8_8_8_8Srgb;
		std::vector<uint8_t> png(static_cast<size_t>(w) * h * 4);
		double   lo[4] = {INFINITY, INFINITY, INFINITY, INFINITY}, hi[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
		uint64_t bad   = 0;
		for (uint64_t i = 0; i < static_cast<uint64_t>(w) * h; i++) {
			for (uint32_t c = 0; c < 4; c++) {
				double v = 0.0;
				switch (num) {
					case Prospero::TextureNumericClass::Uint: v = uints[i * 4 + c]; break;
					case Prospero::TextureNumericClass::Sint: v = ints[i * 4 + c]; break;
					default: v = texels[i * 4 + c]; break;
				}
				if (!std::isfinite(v)) {
					bad++;
					v = 0.0;
				}
				lo[c] = std::min(lo[c], v);
				hi[c] = std::max(hi[c], v);
				double out = v * m_opt.scale;
				if (num == Prospero::TextureNumericClass::Uint || num == Prospero::TextureNumericClass::Sint) {
					out = std::clamp(out, 0.0, 255.0) / 255.0;
				} else if ((m_opt.srgb || is_srgb) && c < 3) {
					out = std::clamp(out, 0.0, 1.0);
					out = out <= 0.0031308 ? out * 12.92 : 1.055 * std::pow(out, 1.0 / 2.4) - 0.055;
				}
				if (c == 3 && !m_opt.alpha) {
					out = 1.0;
				}
				png[i * 4 + c] = static_cast<uint8_t>(std::lround(std::clamp(out, 0.0, 1.0) * 255.0));
			}
		}
		const std::string base = m_opt.out + "/" + name;
		stbi_write_png((base + ".png").c_str(), static_cast<int>(w), static_cast<int>(h), 4, png.data(), static_cast<int>(w * 4));
		if (m_opt.raw) {
			WriteFile(base + ".rgba32", m_out.cpu, bytes);
		}
		char info[1024];
		const int n = std::snprintf(
		    info, sizeof(info),
		    "%s\nT# %08x %08x %08x %08x %08x %08x %08x %08x\nbase 0x%llx %ux%u (read %ux%u at %u,%u) format %u type %u swizzle 0x%x%s meta 0x%llx\n"
		    "min %g %g %g %g\nmax %g %g %g %g\nnon-finite %llu\n",
		    desc.c_str(), t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], static_cast<unsigned long long>(TsharpBase(t)), full_w,
		    full_h, w, h, x0, y0, TsharpFormat(t), type, TsharpSwizzle(t), TsharpCompressed(t) ? " dcc" : "",
		    static_cast<unsigned long long>(TsharpMeta(t)), lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3],
		    static_cast<unsigned long long>(bad));
		WriteFile(base + ".txt", info, static_cast<size_t>(n));
		std::fprintf(stderr, "  %s: %s, %ux%u fmt %u, min %.4g/%.4g/%.4g/%.4g max %.4g/%.4g/%.4g/%.4g%s\n", name.c_str(), desc.c_str(), w,
		             h, TsharpFormat(t), lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3], bad != 0 ? " NON-FINITE" : "");
	}

	const Options&        m_opt;
	const Loaded&         m_cap;
	std::vector<Step>     m_steps;
	int                   m_fd  = -1;
	NativeSubmitter*      m_sub = nullptr;
	std::vector<Bo>       m_region_bo;
	std::vector<uint32_t> m_handles; // every buffer, listed in every submission
	uint8_t               m_scratch[NativeSubmitter::kScratchSize] = {};
	uint32_t              m_map_version = ~0u;
	Bo                    m_code;
	Bo                    m_out;
};

} // namespace

int main(int argc, char** argv) {
	const Options opt = ParseOptions(argc, argv);
	const Loaded  cap = Load(opt.dir);
	if (opt.list) {
		List(cap);
		return 0;
	}
	Replayer(opt, cap).Run();
	return 0;
}
