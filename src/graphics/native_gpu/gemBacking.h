#ifndef EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_
#define EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_

#include <cstdint>

// amdgpu GEM backing for guest "physical" memory on the BC-250 native GPU backend. Only real
// Linux with the amdgpu uAPI headers has it; everywhere else this compiles to an empty header so
// the opt-in hook in src/kernel/memoryAddressSpace.inc keeps upstream behaviour unchanged.
#if defined(__linux__) && !defined(__APPLE__)
#define KYTY_GEM_BACKING_SUPPORTED 1
#else
#define KYTY_GEM_BACKING_SUPPORTED 0
#endif

#if KYTY_GEM_BACKING_SUPPORTED

#include <cerrno>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <drm/amdgpu_drm.h>
#include <drm/drm.h>

namespace Libs::Graphics::NativeGpu {

// Backs guest direct memory so every guest direct-memory mapping is also mapped into the GPU VM
// at the same virtual address, on /dev/dri/renderD128.
//
// The CPU side stays a memfd with a single contiguous writable alias, exactly like the upstream
// backing: the stock amdgpu driver can only CPU-map a GEM buffer whole and from its start (a
// non-zero inner offset is EINVAL), so a real GEM buffer cannot provide the contiguous alias and
// the arbitrary-offset guest views this backing needs. Instead each 256 MiB slice of the alias is
// registered as a userptr GEM buffer (REGISTER, no ANONONLY — the pages are shared memfd pages),
// created lazily on first touch; a job never pins more than it uses and no buffer approaches the
// board's ~8 GiB GTT limit. BindView maps a guest view's chunks into the GPU VM of DrmFd() at the
// view's own address; Lookup resolves a backing offset to (chunk handle, offset-in-chunk) for the
// buffer list of a submission.
//
// This class performs no command submission. userptr buffers cannot be dma-buf exported, so
// presentable surfaces still need dedicated real-GEM allocations elsewhere.
class GemBacking {
public:
	// GEM chunk granularity. 256 MiB divides the 13.5 GiB backing (13824 MiB) evenly and stays
	// well under the board's per-buffer and total GTT limits.
	static constexpr uint64_t kChunkSize = 256ull * 1024 * 1024;

	// Creates the backing for `size` bytes: opens the device, creates the memfd and its contiguous
	// alias. Returns nullptr on failure. No GEM buffers are registered yet.
	static GemBacking* Create(uint64_t size, const char* device = "/dev/dri/renderD128") {
		if (size == 0) {
			return nullptr;
		}
		int drm_fd = ::open(device, O_RDWR | O_CLOEXEC);
		if (drm_fd < 0) {
			std::fprintf(stderr, "GemBacking: open %s failed: %s\n", device, std::strerror(errno));
			return nullptr;
		}
		int mem_fd = static_cast<int>(::syscall(SYS_memfd_create, "KytyDirectMemoryGem", 0));
		if (mem_fd < 0 || ::ftruncate(mem_fd, static_cast<off_t>(size)) != 0) {
			std::fprintf(stderr, "GemBacking: memfd setup failed: %s\n", std::strerror(errno));
			if (mem_fd >= 0) {
				::close(mem_fd);
			}
			::close(drm_fd);
			return nullptr;
		}
		void* alias = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, 0);
		if (alias == MAP_FAILED) {
			std::fprintf(stderr, "GemBacking: alias of %llu MiB failed: %s\n",
			             static_cast<unsigned long long>(size / (1024 * 1024)), std::strerror(errno));
			::close(mem_fd);
			::close(drm_fd);
			return nullptr;
		}
		auto* self       = new GemBacking();
		self->m_drm_fd   = drm_fd;
		self->m_mem_fd   = mem_fd;
		self->m_alias    = static_cast<uint8_t*>(alias);
		self->m_size     = size;
		const auto count = (size + kChunkSize - 1) / kChunkSize;
		self->m_chunks.resize(count);
		for (uint64_t i = 0; i < count; i++) {
			const uint64_t start   = i * kChunkSize;
			self->m_chunks[i].size = (size - start < kChunkSize) ? size - start : kChunkSize;
		}
		return self;
	}

	~GemBacking() {
		for (const auto& chunk: m_chunks) {
			if (chunk.created) {
				GemClose(chunk.handle);
			}
		}
		if (m_alias != nullptr) {
			::munmap(m_alias, m_size);
		}
		if (m_mem_fd >= 0) {
			::close(m_mem_fd);
		}
		if (m_drm_fd >= 0) {
			::close(m_drm_fd);
		}
	}

	GemBacking(const GemBacking&)            = delete;
	GemBacking& operator=(const GemBacking&) = delete;

	[[nodiscard]] uint8_t* AliasBase() const { return m_alias; }
	[[nodiscard]] uint64_t Size() const { return m_size; }

	// Registers every GEM chunk overlapping [offset, offset + size) as a userptr buffer, so the
	// range is GPU-mappable later. Called before a guest range is mapped.
	bool EnsureRange(uint64_t offset, uint64_t size) {
		if (size == 0) {
			return true;
		}
		if (offset >= m_size || size > m_size - offset) {
			return false;
		}
		const uint64_t first = offset / kChunkSize;
		const uint64_t last  = (offset + size - 1) / kChunkSize;
		for (uint64_t i = first; i <= last; i++) {
			if (!EnsureChunk(i)) {
				return false;
			}
		}
		return true;
	}

	// Maps the backing bytes [offset, offset + size) at guest address `vaddr` with protection
	// `prot` (PROT_* flags), after ensuring the GEM chunks they belong to exist. The CPU mapping is
	// a single memfd view (the memfd spans every chunk), so no chunk-crossing split is needed here.
	// On success the region is mapped at exactly `vaddr`.
	bool MapView(uint64_t vaddr, uint64_t size, uint64_t offset, int prot) {
		if (size == 0 || !EnsureRange(offset, size)) {
			return false;
		}
		void* const dst = reinterpret_cast<void*>(vaddr);
		void* const got = ::mmap(dst, size, prot, MAP_SHARED | MAP_FIXED, m_mem_fd,
		                         static_cast<off_t>(offset));
		return got != MAP_FAILED && got == dst;
	}

	// Maps backing bytes [offset, offset + size) into this device's GPU VM at `vaddr`, the same
	// address the CPU view uses, so guest pointers are valid on the GPU unchanged. A range that
	// crosses chunks becomes one GPU mapping per chunk. REPLACE drops whatever was mapped there
	// before, so remapping a guest range needs no separate unbind. All guest memory is mapped
	// readable, writable and executable (shaders live in guest memory).
	bool BindView(uint64_t vaddr, uint64_t size, uint64_t offset) {
		if (size == 0 || !EnsureRange(offset, size)) {
			return false;
		}
		uint64_t done = 0;
		while (done < size) {
			const uint64_t at    = offset + done;
			const uint64_t index = at / kChunkSize;
			const uint64_t inner = at - index * kChunkSize;
			const uint64_t part  = std::min(size - done, m_chunks[index].size - inner);
			if (VaOp(AMDGPU_VA_OP_REPLACE, m_chunks[index].handle, vaddr + done, inner, part) != 0) {
				return false;
			}
			done += part;
		}
		return true;
	}

	// Removes every GPU mapping in [vaddr, vaddr + size), whichever chunk it came from.
	bool UnbindView(uint64_t vaddr, uint64_t size) {
		return size == 0 || VaOp(AMDGPU_VA_OP_CLEAR, 0, vaddr, 0, size) == 0;
	}

	// The DRM file whose GPU VM holds the bindings; command submission must use the same file.
	[[nodiscard]] int DrmFd() const { return m_drm_fd; }

	// Every GEM chunk created so far, for the buffer list of a submission: a command buffer that
	// touches guest memory must name the chunks backing it as resident. Chunks are created lazily,
	// so this grows as guest ranges are mapped.
	[[nodiscard]] std::vector<uint32_t> CreatedHandles() const {
		std::vector<uint32_t> handles;
		handles.reserve(m_chunks.size());
		for (const auto& chunk: m_chunks) {
			if (chunk.created) {
				handles.push_back(chunk.handle);
			}
		}
		return handles;
	}

	// Looks up the GEM buffer backing `offset`, for a later GPU-VM mapping step. Returns false if
	// `offset` is out of range or its chunk has not been created yet.
	bool Lookup(uint64_t offset, uint32_t* handle, uint64_t* offset_in_chunk) const {
		if (offset >= m_size) {
			return false;
		}
		const uint64_t index = offset / kChunkSize;
		if (!m_chunks[index].created) {
			return false;
		}
		if (handle != nullptr) {
			*handle = m_chunks[index].handle;
		}
		if (offset_in_chunk != nullptr) {
			*offset_in_chunk = offset - index * kChunkSize;
		}
		return true;
	}

private:
	struct Chunk {
		uint32_t handle  = 0;
		uint64_t size    = 0;
		bool     created = false;
	};

	GemBacking() = default;

	int Ioctl(unsigned long request, void* arg) const {
		int r = -1;
		do {
			r = ::ioctl(m_drm_fd, request, arg);
		} while (r == -1 && (errno == EINTR || errno == EAGAIN));
		return r == -1 ? -errno : 0;
	}

	int VaOp(uint32_t op, uint32_t handle, uint64_t vaddr, uint64_t offset_in_bo, uint64_t size) const {
		struct drm_amdgpu_gem_va v;
		std::memset(&v, 0, sizeof(v));
		v.handle       = handle;
		v.operation    = op;
		v.flags        = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
		v.va_address   = vaddr;
		v.offset_in_bo = offset_in_bo;
		v.map_size     = size;
		const int r    = Ioctl(DRM_IOCTL_AMDGPU_GEM_VA, &v);
		if (r != 0) {
			std::fprintf(stderr, "GemBacking: GEM_VA op %u at 0x%llx (+0x%llx) failed: %s\n", op,
			             static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
			             std::strerror(-r));
		}
		return r;
	}

	void GemClose(uint32_t handle) const {
		struct drm_gem_close c;
		std::memset(&c, 0, sizeof(c));
		c.handle = handle;
		Ioctl(DRM_IOCTL_GEM_CLOSE, &c);
	}

	bool EnsureChunk(uint64_t index) {
		Chunk& chunk = m_chunks[index];
		if (chunk.created) {
			return true;
		}

		// Register the chunk's slice of the shared memfd alias as a GEM buffer. REGISTER tracks the
		// CPU range via mmu-notifier without pinning; the later GPU-VM bind validates the pages.
		struct drm_amdgpu_gem_userptr u;
		std::memset(&u, 0, sizeof(u));
		u.addr  = reinterpret_cast<uintptr_t>(m_alias + index * kChunkSize);
		u.size  = chunk.size;
		u.flags = AMDGPU_GEM_USERPTR_REGISTER;
		if (Ioctl(DRM_IOCTL_AMDGPU_GEM_USERPTR, &u) != 0) {
			std::fprintf(stderr, "GemBacking: GEM_USERPTR chunk %llu (%llu MiB) failed: %s\n",
			             static_cast<unsigned long long>(index),
			             static_cast<unsigned long long>(chunk.size / (1024 * 1024)),
			             std::strerror(errno));
			return false;
		}
		chunk.handle  = u.handle;
		chunk.created = true;
		return true;
	}

	int                m_drm_fd = -1;
	int                m_mem_fd = -1;
	uint8_t*           m_alias  = nullptr;
	uint64_t           m_size   = 0;
	std::vector<Chunk> m_chunks;
};

} // namespace Libs::Graphics::NativeGpu

#endif // KYTY_GEM_BACKING_SUPPORTED

#endif // EMULATOR_SRC_GRAPHICS_NATIVE_GPU_GEMBACKING_H_
