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
#include <fcntl.h>
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

	delete gem;
	std::puts("NativeGpuGemBackingTests: all cases passed");
	return 0;
}

#endif // KYTY_GEM_BACKING_SUPPORTED
