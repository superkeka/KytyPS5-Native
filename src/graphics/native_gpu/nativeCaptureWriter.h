#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVECAPTUREWRITER_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVECAPTUREWRITER_H_

#include "graphics/native_gpu/nativeCapture.h"
#include "graphics/native_gpu/nativeSubmit.h" // KYTY_GEM_BACKING_SUPPORTED, GemBacking, NativeSubmitter

// Deterministic frame capture for tools/bc250-replay (format in nativeCapture.h).
//
// KYTY_NATIVE_GPU_CAPTURE=<flip>: from flip <flip> - KYTY_NATIVE_GPU_CAPTURE_FRAMES (default 2, so
// that work the pipeline submitted ahead of the flip before it is not missed) until flip <flip> is
// presented, every CS the shadow path submits is recorded, in KYTY_NATIVE_GPU_CAPTURE_DIR (default
// KYTY_NATIVE_GPU_DUMP_DIR, then ".") as capture_<flip>/. Flips are numbered as in
// KYTY_NATIVE_GPU_DUMP_FLIPS. While capturing every CS runs synchronously: before it, with the GPU
// idle, all memory the GPU VM maps is compared page by page with the previous point and what changed
// (the CPU's writes and the previous CS's results alike) is saved; the first point saves everything.
// The replay applies the same pages before each CS, so every CS starts from exactly the memory it
// had in the game. GDS is read back through a staging buffer at each point too.
// KYTY_NATIVE_GPU_CAPTURE_MAX_CS (default 400) ends a capture that runs away.

#if KYTY_GEM_BACKING_SUPPORTED

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include <sys/stat.h>
#include <sys/uio.h>

namespace Libs::Graphics::NativeGpu {

class FrameCaptureWriter {
public:
	void Configure(NativeSubmitter* submitter) {
		const char* flip = std::getenv("KYTY_NATIVE_GPU_CAPTURE");
		if (flip == nullptr || flip[0] == '\0') {
			return;
		}
		m_end_flip = std::strtoull(flip, nullptr, 0);
		uint64_t frames = 2;
		if (const char* f = std::getenv("KYTY_NATIVE_GPU_CAPTURE_FRAMES"); f != nullptr && f[0] != '\0') {
			frames = std::strtoull(f, nullptr, 0);
		}
		m_start_flip = m_end_flip >= frames ? m_end_flip - frames : 0;
		if (const char* m = std::getenv("KYTY_NATIVE_GPU_CAPTURE_MAX_CS"); m != nullptr && m[0] != '\0') {
			m_max_cs = static_cast<uint32_t>(std::strtoul(m, nullptr, 0));
		}
		const char* dir = std::getenv("KYTY_NATIVE_GPU_CAPTURE_DIR");
		if (dir == nullptr || dir[0] == '\0') {
			dir = std::getenv("KYTY_NATIVE_GPU_DUMP_DIR");
		}
		m_dir = std::string(dir != nullptr && dir[0] != '\0' ? dir : ".") + "/capture_" + std::to_string(m_end_flip);
		m_manifest.start_flip = m_start_flip;
		m_manifest.end_flip   = m_end_flip;
		m_manifest.gds_size   = submitter->GdsSize();
		const char* gds_spec  = std::getenv("KYTY_NATIVE_GPU_GDS");
		m_manifest.gds_spec   = gds_spec != nullptr ? gds_spec : (m_manifest.gds_size != 0 ? "gds=" + std::to_string(m_manifest.gds_size) + ",gws=0,oa=0" : "");
		m_state = State::Waiting;
		std::fprintf(stderr, "native capture: flips %llu-%llu into %s\n", static_cast<unsigned long long>(m_start_flip),
		             static_cast<unsigned long long>(m_end_flip), m_dir.c_str());
	}

	// True while the next CS must be captured (the caller then waits for the GPU before BeforeCs and
	// after the CS).
	[[nodiscard]] bool Active() const { return m_state == State::Armed || m_state == State::Capturing; }

	// At every presented flip `fl.n`, with the GPU idle when Active().
	void OnFlip(GemBacking* gem, NativeSubmitter* sub, const Capture::Flip& fl) {
		if (m_state == State::Waiting && fl.n >= m_start_flip) {
			m_state = State::Armed;
			std::fprintf(stderr, "native capture: armed at flip %llu\n", static_cast<unsigned long long>(fl.n));
		}
		if (m_state == State::Capturing) {
			Capture::Flip f = fl;
			f.after_cs      = static_cast<uint32_t>(m_manifest.cs.size());
			m_manifest.flips.push_back(f);
			if (fl.n >= m_end_flip) {
				Finish(gem, sub);
			}
		}
	}

	// Right before CS `ib` (made of guest buffers `parts`) is submitted, with the GPU idle.
	void BeforeCs(GemBacking* gem, NativeSubmitter* sub, std::span<const uint32_t> ib, const std::vector<Capture::Part>& parts) {
		if (m_state == State::Armed) {
			if (!Begin(gem, sub)) {
				m_state = State::Off;
				return;
			}
			m_state = State::Capturing;
		}
		if (m_state != State::Capturing) {
			return;
		}
		const auto     n = static_cast<uint32_t>(m_manifest.cs.size());
		Capture::Cs    cs;
		cs.delta_file  = "delta_" + std::to_string(n) + ".bin";
		cs.map_version = Snapshot(gem, sub, n, cs.delta_file);
		cs.ib_file     = "ib_" + std::to_string(n) + ".bin";
		cs.ib_dw       = static_cast<uint32_t>(ib.size());
		cs.parts       = parts;
		WriteWhole(m_dir + "/" + cs.ib_file, ib.data(), ib.size_bytes());
		m_manifest.cs.push_back(cs);
		if (m_manifest.cs.size() >= m_max_cs) {
			std::fprintf(stderr, "native capture: %u CS captured, stopping early\n", m_max_cs);
			Finish(gem, sub);
		}
	}

private:
	enum class State { Off, Waiting, Armed, Capturing, Done };

	struct RegionData {
		std::vector<uint64_t> hashes; // per page, empty until the region was first saved
	};

	// Pages of one point, appended by the workers: a header {magic, count} and records
	// {region, page, 4096 bytes}.
	class DeltaFile {
	public:
		explicit DeltaFile(const std::string& path): m_file(std::fopen(path.c_str(), "wb")) {
			const uint32_t header[2] = {Capture::kDeltaMagic, 0};
			if (m_file != nullptr) {
				std::fwrite(header, sizeof(header), 1, m_file);
			}
		}
		void Add(uint32_t region, uint32_t page, const uint8_t* data) {
			std::lock_guard<std::mutex> lock(m_lock);
			if (m_file == nullptr) {
				return;
			}
			const uint32_t rec[2] = {region, page};
			std::fwrite(rec, sizeof(rec), 1, m_file);
			std::fwrite(data, Capture::kPageSize, 1, m_file);
			m_count++;
		}
		uint64_t Close() {
			if (m_file != nullptr) {
				std::fseek(m_file, 4, SEEK_SET);
				std::fwrite(&m_count, sizeof(m_count), 1, m_file);
				std::fclose(m_file);
				m_file = nullptr;
			}
			return static_cast<uint64_t>(m_count) * Capture::kPageSize;
		}
		[[nodiscard]] uint32_t Count() const { return m_count; }
		~DeltaFile() { Close(); }

	private:
		std::FILE* m_file;
		std::mutex m_lock;
		uint32_t   m_count = 0;
	};

	bool Begin(GemBacking* gem, NativeSubmitter* sub) {
		::mkdir(m_dir.c_str(), 0755);
		::mkdir((m_dir + "/mem").c_str(), 0755);
		struct stat st;
		if (::stat((m_dir + "/mem").c_str(), &st) != 0) {
			std::fprintf(stderr, "native capture: cannot create %s\n", m_dir.c_str());
			return false;
		}
		m_t0 = Now();
		// Fixed regions: the submitter's scratch area and the GDS partition.
		m_scratch_region = AddRegion(Capture::RegionKind::Scratch, 0, NativeSubmitter::kScratchSize, true);
		if (sub->GdsSize() != 0) {
			m_gds_region = AddRegion(Capture::RegionKind::Gds, 0, sub->GdsSize(), true);
			if (!CreateStaging(gem->DrmFd(), sub->GdsSize())) {
				return false;
			}
		}
		(void)gem;
		return true;
	}

	void Finish(GemBacking* gem, NativeSubmitter* sub) {
		const auto n                  = static_cast<uint32_t>(m_manifest.cs.size());
		m_manifest.final_delta        = "delta_final.bin";
		m_manifest.final_map_version  = Snapshot(gem, sub, n, m_manifest.final_delta);
		const bool ok                 = m_manifest.Write(m_dir + "/manifest.txt");
		m_state                       = State::Done;
		std::fprintf(stderr,
		             "native capture: %s: %u CS, %zu regions (%llu MiB saved in full), %llu MiB of deltas, %.1f s\n",
		             ok ? "done" : "MANIFEST WRITE FAILED", n, m_manifest.regions.size(),
		             static_cast<unsigned long long>(m_full_bytes >> 20u), static_cast<unsigned long long>(m_delta_bytes >> 20u),
		             Now() - m_t0);
	}

	uint32_t AddRegion(Capture::RegionKind kind, uint64_t key, uint64_t size, bool writable) {
		Capture::Region r;
		r.id       = static_cast<uint32_t>(m_manifest.regions.size());
		r.kind     = kind;
		r.key      = key;
		r.size     = size;
		r.since    = static_cast<uint32_t>(m_manifest.cs.size());
		r.writable = writable;
		r.file     = "mem/r" + std::to_string(r.id) + ".bin";
		m_manifest.regions.push_back(r);
		m_data.emplace_back();
		m_region_by_key[std::make_tuple(static_cast<uint32_t>(kind), key, size, writable)] = r.id;
		return r.id;
	}

	uint32_t RegionFor(Capture::RegionKind kind, uint64_t key, uint64_t size, bool writable) {
		auto it = m_region_by_key.find(std::make_tuple(static_cast<uint32_t>(kind), key, size, writable));
		return it != m_region_by_key.end() ? it->second : AddRegion(kind, key, size, writable);
	}

	// Saves the state before CS `n` (or the final state): the mapping (a new version when it changed),
	// regions that are new in full, and the pages of the others that changed into `delta_file`.
	// Returns the mapping version.
	uint32_t Snapshot(GemBacking* gem, NativeSubmitter* sub, uint32_t n, const std::string& delta_file) {
		const double t0 = Now();
		std::vector<Capture::Mapping> maps;
		for (const auto& m: gem->GpuMappings()) {
			uint32_t region = 0;
			switch (m.target) {
				case GemBacking::GpuTarget::Block: region = RegionFor(Capture::RegionKind::Block, m.index, GemBacking::kBlockSize, true); break;
				case GemBacking::GpuTarget::Dummy: region = RegionFor(Capture::RegionKind::Dummy, 0, GemBacking::kBlockSize, true); break;
				case GemBacking::GpuTarget::External: region = RegionFor(Capture::RegionKind::External, m.index, m.size, m.writable); break;
			}
			maps.push_back(Capture::Mapping {m.vaddr, m.size, region, m.offset});
		}
		if (m_manifest.maps.empty() || maps != m_last_maps) {
			const std::string file = "map_" + std::to_string(m_manifest.maps.size()) + ".txt";
			Capture::WriteMappings(m_dir + "/" + file, maps);
			m_manifest.maps.push_back(file);
			m_last_maps = std::move(maps);
		}
		// The block buffers' real sizes (the last block of the backing can be short).
		for (auto& r: m_manifest.regions) {
			if (r.kind == Capture::RegionKind::Block) {
				uint64_t size = 0;
				if (gem->ResidentBlockCpu(r.key, &size) != nullptr && m_data[r.id].hashes.empty()) {
					r.size = size;
				}
			}
		}
		std::memcpy(m_scratch.data(), const_cast<const uint8_t*>(sub->ScratchCpu()), NativeSubmitter::kScratchSize);
		if (m_gds_region != ~0u) {
			ReadGds(sub);
		}

		DeltaFile delta(m_dir + "/" + delta_file);
		std::atomic<uint32_t> next {0};
		std::atomic<uint64_t> full {0};
		const uint32_t        regions = static_cast<uint32_t>(m_manifest.regions.size());
		auto work = [&] {
			std::vector<uint8_t> chunk(kChunk);
			for (uint32_t id = next++; id < regions; id = next++) {
				full += SaveRegion(gem, id, n, chunk, delta);
			}
		};
		std::vector<std::thread> workers;
		for (uint32_t i = 0; i < kWorkers; i++) {
			workers.emplace_back(work);
		}
		for (auto& w: workers) {
			w.join();
		}
		m_full_bytes += full;
		m_delta_bytes += delta.Close();
		std::fprintf(stderr, "native capture: point %u: %u pages changed, %llu MiB new regions, %.2f s\n", n, delta.Count(),
		             static_cast<unsigned long long>(full.load() >> 20u), Now() - t0);
		return static_cast<uint32_t>(m_manifest.maps.size() - 1);
	}

	// Saves region `id`: in full when it has no hashes yet (it appeared before CS `n`), else its
	// changed pages into `delta`. Returns the bytes saved in full.
	uint64_t SaveRegion(GemBacking* gem, uint32_t id, uint32_t n, std::vector<uint8_t>& chunk, DeltaFile& delta) {
		Capture::Region& r     = m_manifest.regions[id];
		RegionData&      d     = m_data[id];
		const bool       fresh = d.hashes.empty();
		const uint8_t*   src   = nullptr;
		uint64_t         size  = r.size;
		switch (r.kind) {
			case Capture::RegionKind::Block: src = gem->ResidentBlockCpu(r.key, &size); break;
			case Capture::RegionKind::Dummy: src = gem->DummyCpu(); break;
			case Capture::RegionKind::Scratch: src = m_scratch.data(); break;
			case Capture::RegionKind::Gds: src = m_gds.data(); break;
			case Capture::RegionKind::External: break; // read with process_vm_readv: it may have holes
		}
		if (r.kind == Capture::RegionKind::Block && src == nullptr) {
			return 0; // no longer resident cannot happen (blocks are never demoted); keep the old pages
		}
		if (fresh) {
			r.since = n;
			d.hashes.assign((size + Capture::kPageSize - 1) / Capture::kPageSize, 0);
		}
		int fd = -1;
		if (fresh) {
			fd = ::open((m_dir + "/" + r.file).c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
			if (fd >= 0 && ::ftruncate(fd, static_cast<off_t>(size)) != 0) {
				std::fprintf(stderr, "native capture: cannot size %s\n", r.file.c_str());
			}
		}
		uint64_t saved = 0;
		for (uint64_t at = 0; at < size; at += kChunk) {
			const uint64_t len = std::min<uint64_t>(kChunk, size - at);
			if (r.kind == Capture::RegionKind::External) {
				ReadSelf(r.key + at, chunk.data(), len);
			} else {
				std::memcpy(chunk.data(), src + at, len);
			}
			if (len % Capture::kPageSize != 0) {
				std::memset(chunk.data() + len, 0, Capture::kPageSize - len % Capture::kPageSize);
			}
			for (uint64_t off = 0; off < len; off += Capture::kPageSize) {
				const uint8_t* page  = chunk.data() + off;
				const uint64_t index = (at + off) / Capture::kPageSize;
				const uint64_t hash  = Capture::HashPage(page);
				if (fresh) {
					d.hashes[index] = hash;
					if (fd >= 0 && !Capture::PageIsZero(page)) {
						const uint64_t bytes = std::min<uint64_t>(Capture::kPageSize, size - (at + off));
						if (::pwrite(fd, page, bytes, static_cast<off_t>(at + off)) == static_cast<ssize_t>(bytes)) {
							saved += bytes;
						}
					}
				} else if (d.hashes[index] != hash) {
					d.hashes[index] = hash;
					delta.Add(id, static_cast<uint32_t>(index), page);
				}
			}
		}
		if (fd >= 0) {
			::close(fd);
		}
		return saved;
	}

	// Guest memory outside the backing is host memory at the same address, but parts of it may be
	// unreadable by now: those read as zero.
	static void ReadSelf(uint64_t addr, uint8_t* dst, uint64_t len) {
		uint64_t done = 0;
		while (done < len) {
			struct iovec  local  = {dst + done, len - done};
			struct iovec  remote = {reinterpret_cast<void*>(addr + done), len - done};
			const ssize_t got    = ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0);
			if (got <= 0) {
				const uint64_t skip = std::min<uint64_t>(Capture::kPageSize - ((addr + done) % Capture::kPageSize), len - done);
				std::memset(dst + done, 0, skip);
				done += skip;
			} else {
				done += static_cast<uint64_t>(got);
			}
		}
	}

	static void WriteWhole(const std::string& path, const void* data, size_t bytes) {
		if (std::FILE* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
			std::fwrite(data, 1, bytes, f);
			std::fclose(f);
		}
	}

	static double Now() {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
	}

	static int Ioctl(int fd, unsigned long request, void* arg) {
		int r = -1;
		do {
			r = ::ioctl(fd, request, arg);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		return r == -1 ? -errno : 0;
	}

	// A small GTT buffer at kStagingVa that GDS is copied into.
	bool CreateStaging(int drm_fd, uint32_t gds_size) {
		const uint64_t size = (static_cast<uint64_t>(gds_size) + 0xffffu) & ~0xffffull;
		union drm_amdgpu_gem_create c;
		std::memset(&c, 0, sizeof(c));
		c.in.bo_size   = size;
		c.in.alignment = 64 * 1024;
		c.in.domains   = AMDGPU_GEM_DOMAIN_GTT;
		if (Ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c) != 0) {
			std::fprintf(stderr, "native capture: staging GEM_CREATE failed\n");
			return false;
		}
		m_staging = c.out.handle;
		union drm_amdgpu_gem_mmap m;
		std::memset(&m, 0, sizeof(m));
		m.in.handle = m_staging;
		if (Ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &m) != 0) {
			return false;
		}
		void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, static_cast<off_t>(m.out.addr_ptr));
		if (p == MAP_FAILED) {
			return false;
		}
		m_staging_cpu = static_cast<uint8_t*>(p);
		struct drm_amdgpu_gem_va v;
		std::memset(&v, 0, sizeof(v));
		v.handle     = m_staging;
		v.operation  = AMDGPU_VA_OP_MAP;
		v.flags      = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
		v.va_address = Capture::kStagingVa;
		v.map_size   = size;
		if (Ioctl(drm_fd, DRM_IOCTL_AMDGPU_GEM_VA, &v) != 0) {
			std::fprintf(stderr, "native capture: staging GEM_VA failed\n");
			return false;
		}
		m_gds.assign(gds_size, 0);
		return true;
	}

	// GDS -> staging with one DMA_DATA (SRC_SEL GDS, DST_SEL TC_L2, CP_SYNC), then into m_gds.
	void ReadGds(NativeSubmitter* sub) {
		std::vector<uint32_t> ib;
		ib.push_back(Capture::Pkt3(Capture::IT_DMA_DATA, 6));
		ib.push_back((1u << 31u) | (1u << 29u) | (3u << 20u));
		ib.push_back(0);
		ib.push_back(0);
		ib.push_back(static_cast<uint32_t>(Capture::kStagingVa));
		ib.push_back(static_cast<uint32_t>(Capture::kStagingVa >> 32u));
		ib.push_back(static_cast<uint32_t>(m_gds.size()));
		Capture::PadIb(&ib);
		std::memset(m_staging_cpu, 0, m_gds.size());
		NativeSubmitter::Result r = sub->Submit(ib, {m_staging}, 5.0);
		const bool ok = r.submitted && (!sub->Async() ? r.completed : sub->WaitIdle(5.0));
		if (!ok) {
			std::fprintf(stderr, "native capture: GDS readback failed (%d)\n", r.err);
		}
		std::memcpy(m_gds.data(), m_staging_cpu, m_gds.size());
	}

	static constexpr uint64_t kChunk   = 1ull << 20u;
	static constexpr uint32_t kWorkers = 8;

	State                         m_state      = State::Off;
	uint64_t                      m_start_flip = 0;
	uint64_t                      m_end_flip   = 0;
	uint32_t                      m_max_cs     = 400;
	std::string                   m_dir;
	Capture::Manifest             m_manifest;
	std::vector<RegionData>       m_data;
	std::map<std::tuple<uint32_t, uint64_t, uint64_t, bool>, uint32_t> m_region_by_key;
	std::vector<Capture::Mapping> m_last_maps;
	std::vector<uint8_t>          m_scratch = std::vector<uint8_t>(NativeSubmitter::kScratchSize);
	std::vector<uint8_t>          m_gds;
	uint32_t                      m_scratch_region = ~0u;
	uint32_t                      m_gds_region     = ~0u;
	uint32_t                      m_staging        = 0;
	uint8_t*                      m_staging_cpu    = nullptr;
	uint64_t                      m_full_bytes     = 0;
	uint64_t                      m_delta_bytes    = 0;
	double                        m_t0             = 0.0;
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_NATIVECAPTUREWRITER_H_
