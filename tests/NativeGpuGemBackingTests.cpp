#include "graphics/native_gpu/gemBacking.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if !KYTY_GEM_BACKING_SUPPORTED

int main() {
	std::puts("NativeGpuGemBackingTests: skipped (platform without amdgpu GEM backing)");
	return 0;
}

#else

#include <cerrno>
#include <ctime>
#include <initializer_list>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

using Libs::Graphics::NativeGpu::GemBacking;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuGemBackingTests: failed: %s\n", message);
		std::abort();
	}
}

// A guest-style placeholder the backing can replace with MAP_FIXED, mirroring the reserved
// address space the emulator hands to MapView.
void* ReserveGuest(uint64_t size) {
	void* p = ::mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
	Check(p != MAP_FAILED, "guest placeholder reservation failed");
	return p;
}


int Ioctl(int fd, unsigned long request, void* arg) {
	int r = -1;
	do {
		r = ::ioctl(fd, request, arg);
	} while (r == -1 && (errno == EINTR || errno == EAGAIN));
	return r == -1 ? -errno : 0;
}

// A write-combined GTT buffer mapped for the CPU and at `gpu_va` in the backing's GPU VM.
struct Buffer {
	uint32_t  handle = 0;
	uint32_t* cpu    = nullptr;
	uint64_t  va     = 0;
};

Buffer CreateBuffer(int fd, uint64_t size, uint64_t gpu_va) {
	Buffer b;
	union drm_amdgpu_gem_create c;
	std::memset(&c, 0, sizeof(c));
	c.in.bo_size      = size;
	c.in.alignment    = 4096;
	c.in.domains      = AMDGPU_GEM_DOMAIN_GTT;
	c.in.domain_flags = AMDGPU_GEM_CREATE_CPU_GTT_USWC;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &c) == 0, "GEM_CREATE of a command buffer failed");
	b.handle = c.out.handle;
	union drm_amdgpu_gem_mmap m;
	std::memset(&m, 0, sizeof(m));
	m.in.handle = b.handle;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &m) == 0, "GEM_MMAP failed");
	void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(m.out.addr_ptr));
	Check(p != MAP_FAILED, "mmap of a command buffer failed");
	b.cpu = static_cast<uint32_t*>(p);
	struct drm_amdgpu_gem_va v;
	std::memset(&v, 0, sizeof(v));
	v.handle     = b.handle;
	v.operation  = AMDGPU_VA_OP_MAP;
	v.flags      = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE | AMDGPU_VM_PAGE_EXECUTABLE;
	v.va_address = gpu_va;
	v.map_size   = size;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_GEM_VA, &v) == 0, "GEM_VA of a command buffer failed");
	b.va = gpu_va;
	return b;
}

uint32_t Pkt3(uint32_t op, uint32_t body_dw) {
	return (3u << 30) | ((body_dw - 1u) << 16) | (op << 8) | (1u << 1);
}

// store.s from tools/bc250-native-probe for gfx1013 wave64: s[0:1] = destination, s2 = base,
// s3 = workgroup id; lane i of group g writes base + g * 64 + i to destination + (g * 64 + i) * 4.
constexpr uint32_t kStoreShader[] = {0x34020082u, 0x8f048803u, 0x4a020204u, 0x8f058603u, 0x80050205u,
                                     0x4a040005u, 0xdc708000u, 0x00000201u, 0xbf810000u};

// Runs kStoreShader over [dst_va, dst_va + groups * 256) on the compute ring, with `extra` GEM
// handles resident, and returns once the job signaled without a context reset.
void RunStore(int fd, uint32_t ctx, const Buffer& ib, const Buffer& code, uint64_t dst_va, uint32_t base,
              uint32_t groups, const uint32_t* extra, uint32_t extra_count) {
	uint32_t n   = 0;
	auto     set = [&](uint32_t reg, std::initializer_list<uint32_t> values) {
		ib.cpu[n++] = Pkt3(0x76u, static_cast<uint32_t>(values.size()) + 1); // SET_SH_REG
		ib.cpu[n++] = (reg - 0xb000u) / 4;
		for (auto v: values) {
			ib.cpu[n++] = v;
		}
	};
	set(0xb830u, {static_cast<uint32_t>(code.va >> 8), static_cast<uint32_t>(code.va >> 40)}); // PGM_LO/HI
	set(0xb848u, {1u | (0xc0u << 12) | (1u << 21) | (1u << 30), (3u << 1) | (1u << 7)});        // RSRC1/2
	set(0xb8a0u, {0});                                                                           // RSRC3
	set(0xb854u, {0});                                                                           // RESOURCE_LIMITS
	set(0xb860u, {0});                                                                           // TMPRING_SIZE
	set(0xb858u, {0xffffffffu, 0xffffffffu});                                                    // THREAD_MGMT_SE0/1
	set(0xb864u, {0xffffffffu, 0xffffffffu});                                                    // THREAD_MGMT_SE2/3
	set(0xb810u, {0, 0, 0});                                                                     // START_X/Y/Z
	set(0xb81cu, {64, 1, 1});                                                                    // NUM_THREAD_X/Y/Z
	set(0xb900u, {static_cast<uint32_t>(dst_va), static_cast<uint32_t>(dst_va >> 32), base});   // USER_DATA_0..2
	ib.cpu[n++] = Pkt3(0x15u, 4); // DISPATCH_DIRECT
	ib.cpu[n++] = groups;
	ib.cpu[n++] = 1;
	ib.cpu[n++] = 1;
	ib.cpu[n++] = 1u | 4u | (1u << 6); // COMPUTE_SHADER_EN | FORCE_START_AT_000 | ORDER_MODE
	while ((n & 7u) != 0) {
		ib.cpu[n++] = 0xffff1000u;
	}
	__sync_synchronize();

	struct drm_amdgpu_bo_list_entry entries[8];
	uint32_t                        count = 0;
	entries[count++]                      = {ib.handle, 0};
	entries[count++]                      = {code.handle, 0};
	for (uint32_t i = 0; i < extra_count; i++) {
		entries[count++] = {extra[i], 0};
	}
	struct drm_amdgpu_bo_list_in list;
	std::memset(&list, 0, sizeof(list));
	list.operation    = ~0u;
	list.list_handle  = ~0u;
	list.bo_number    = count;
	list.bo_info_size = sizeof(entries[0]);
	list.bo_info_ptr  = reinterpret_cast<uintptr_t>(entries);
	struct drm_amdgpu_cs_chunk_ib ib_info;
	std::memset(&ib_info, 0, sizeof(ib_info));
	ib_info.va_start = ib.va;
	ib_info.ib_bytes = n * 4;
	ib_info.ip_type  = AMDGPU_HW_IP_COMPUTE;
	struct drm_amdgpu_cs_chunk chunks[2];
	chunks[0]               = {AMDGPU_CHUNK_ID_IB, sizeof(ib_info) / 4, reinterpret_cast<uintptr_t>(&ib_info)};
	chunks[1]               = {AMDGPU_CHUNK_ID_BO_HANDLES, sizeof(list) / 4, reinterpret_cast<uintptr_t>(&list)};
	uint64_t           ptrs[2] = {reinterpret_cast<uintptr_t>(&chunks[0]), reinterpret_cast<uintptr_t>(&chunks[1])};
	union drm_amdgpu_cs cs;
	std::memset(&cs, 0, sizeof(cs));
	cs.in.ctx_id     = ctx;
	cs.in.num_chunks = 2;
	cs.in.chunks     = reinterpret_cast<uintptr_t>(ptrs);
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CS, &cs) == 0, "command submission failed");

	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	union drm_amdgpu_wait_cs w;
	std::memset(&w, 0, sizeof(w));
	w.in.handle  = cs.out.handle;
	w.in.timeout = static_cast<uint64_t>(ts.tv_sec + 5) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
	w.in.ip_type = AMDGPU_HW_IP_COMPUTE;
	w.in.ctx_id  = ctx;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_WAIT_CS, &w) == 0 && w.out.status == 0, "GPU job did not signal");

	union drm_amdgpu_ctx q;
	std::memset(&q, 0, sizeof(q));
	q.in.op     = AMDGPU_CTX_OP_QUERY_STATE2;
	q.in.ctx_id = ctx;
	Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &q) == 0 && q.out.state.flags == 0, "GPU context was reset");
}

} // namespace

int main() {
	const char* device = "/dev/dri/renderD128";
	int         probe  = ::open(device, O_RDWR | O_CLOEXEC);
	if (probe < 0) {
		std::printf("NativeGpuGemBackingTests: skipped (%s unavailable: %s)\n", device,
		            std::strerror(errno));
		return 0;
	}
	::close(probe);

	const uint64_t chunk = GemBacking::kChunkSize;
	// 2.5 chunks: two full chunks plus a half-size tail chunk, so the last chunk is partial.
	const uint64_t size = chunk * 2 + chunk / 2;

	GemBacking* gem = GemBacking::Create(size);
	Check(gem != nullptr, "Create returned null on a board with the device present");
	Check(gem->AliasBase() != nullptr, "alias base is null");
	Check(gem->Size() == size, "size mismatch");

	auto* alias = gem->AliasBase();

	// Case 1: a view wholly inside chunk 0. Write through the view, read through the alias.
	{
		const uint64_t off = 0x40000;  // 256 KiB in
		const uint64_t len = 0x20000;  // 128 KiB
		auto*          view = static_cast<uint32_t*>(ReserveGuest(len));
		Check(gem->MapView(reinterpret_cast<uint64_t>(view), len, off, PROT_READ | PROT_WRITE),
		      "MapView of in-chunk range failed");
		for (uint32_t i = 0; i < len / 4; i++) {
			view[i] = 0xa5000000u + i;
		}
		const auto* a = reinterpret_cast<const uint32_t*>(alias + off);
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(a[i] == 0xa5000000u + i, "alias did not see view write (in-chunk)");
		}
	}

	// Case 2: a view straddling the chunk 0 / chunk 1 boundary. The CPU view is one contiguous
	// memfd mapping, but the range spans two GEM chunks, so both must register and resolve
	// separately. Check coherency both directions and the chunk lookup across the seam.
	{
		const uint64_t len = 0x40000;        // 256 KiB
		const uint64_t off = chunk - len / 2; // half in chunk 0, half in chunk 1
		auto*          view = static_cast<uint32_t*>(ReserveGuest(len));
		Check(gem->MapView(reinterpret_cast<uint64_t>(view), len, off, PROT_READ | PROT_WRITE),
		      "MapView across a chunk boundary failed");

		for (uint32_t i = 0; i < len / 4; i++) {
			view[i] = 0x5a000000u + i;
		}
		const auto* a = reinterpret_cast<const uint32_t*>(alias + off);
		for (uint32_t i = 0; i < len / 4; i++) {
			Check(a[i] == 0x5a000000u + i, "alias did not see view write (crossing)");
		}

		// Alias write must be visible through the view: same GEM pages, two mappings.
		reinterpret_cast<uint32_t*>(alias + off)[0] = 0xdeadbeefu;
		Check(view[0] == 0xdeadbeefu, "view did not see alias write");

		uint32_t low_handle  = 0;
		uint32_t high_handle = 0;
		uint64_t low_inner   = ~0ull;
		uint64_t high_inner  = ~0ull;
		Check(gem->Lookup(off, &low_handle, &low_inner), "lookup of low side failed");
		Check(gem->Lookup(chunk, &high_handle, &high_inner), "lookup of high side failed");
		Check(low_handle != 0 && high_handle != 0, "lookup returned a null handle");
		Check(low_handle != high_handle, "straddling offsets must resolve to different chunks");
		Check(low_inner == off, "chunk 0 inner offset wrong");
		Check(high_inner == 0, "chunk 1 inner offset must be 0 at the chunk start");
	}

	// Case 3: the half-size tail chunk was never touched, so it is not created and does not resolve.
	{
		uint32_t handle = 0;
		uint64_t inner  = 0;
		Check(!gem->Lookup(chunk * 2 + 0x1000, &handle, &inner),
		      "an untouched chunk must not resolve (lazy allocation)");
	}


	// Case 4: the GPU writes through a guest view at the view's own address. The view straddles the
	// chunk 0 / chunk 1 seam, so BindView must create one GPU mapping per chunk; the result is read
	// back through both the view and the alias. Then unbind, rebind and write again.
	{
		const uint64_t len    = 0x40000;        // 256 KiB = 1024 groups of 64 dwords
		const uint64_t off    = chunk - len / 2; // half in chunk 0, half in chunk 1
		const uint32_t groups = static_cast<uint32_t>(len / 256);
		auto*          view   = static_cast<uint32_t*>(ReserveGuest(len));
		const uint64_t va     = reinterpret_cast<uint64_t>(view);
		Check(gem->MapView(va, len, off, PROT_READ | PROT_WRITE), "MapView for the GPU case failed");
		Check(gem->BindView(va, len, off), "BindView across a chunk boundary failed");
		std::memset(view, 0, len);

		const int fd = gem->DrmFd();
		union drm_amdgpu_ctx c;
		std::memset(&c, 0, sizeof(c));
		c.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
		Check(Ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &c) == 0, "context allocation failed");
		const uint32_t ctx = c.out.alloc.ctx_id;

		// Command and code buffers far below any host address the guest views use.
		const Buffer ib   = CreateBuffer(fd, 64 * 1024, 0x0080000000ull);
		const Buffer code = CreateBuffer(fd, 64 * 1024, 0x0081000000ull);
		std::memcpy(code.cpu, kStoreShader, sizeof(kStoreShader));

		uint32_t chunks[2] = {};
		Check(gem->Lookup(off, &chunks[0], nullptr) && gem->Lookup(off + len - 1, &chunks[1], nullptr),
		      "chunk lookup for the GPU case failed");

		for (const uint32_t base: {0x11000000u, 0x22000000u}) {
			RunStore(fd, ctx, ib, code, va, base, groups, chunks, 2);
			const auto* a = reinterpret_cast<const uint32_t*>(alias + off);
			for (uint32_t i = 0; i < len / 4; i++) {
				Check(view[i] == base + i, "GPU write through the guest VA not visible in the view");
				Check(a[i] == base + i, "GPU write through the guest VA not visible in the alias");
			}
			// Second round goes through a fresh mapping: clear the range, then bind it again.
			Check(gem->UnbindView(va, len), "UnbindView failed");
			Check(gem->BindView(va, len, off), "BindView after UnbindView failed");
		}
		std::printf("NativeGpuGemBackingTests: GPU wrote %llu KiB through guest VA 0x%llx across chunks\n",
		            static_cast<unsigned long long>(len / 1024), static_cast<unsigned long long>(va));
	}

	delete gem;
	std::puts("NativeGpuGemBackingTests: all cases passed");
	return 0;
}

#endif // KYTY_GEM_BACKING_SUPPORTED
