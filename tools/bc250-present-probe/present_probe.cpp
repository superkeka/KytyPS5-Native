// bc250-present-probe — prototype of the BC-250 native-GPU present path.
//
// Goal: take a colour surface that the guest rendered natively into its own (tiled) memory and
// get it to a presentable linear image through Vulkan, without a CPU copy of the guest pages.
//
//  1. A known linear image is tiled on the CPU with Kyty's exact tile layout (linear and
//     SW_64KB_R_X 32bpp) to produce the "guest surface" — the ground truth.
//  2. That surface is made readable by Vulkan with no copy via VK_EXT_external_memory_host:
//     the host pages are imported as VkDeviceMemory and bound to a storage VkBuffer. This is
//     exactly how the guest's memfd/userptr pages would be handed to Vulkan on the native path.
//  3. A compute shader detiles/converts the surface into a tightly-packed linear image.
//  4. The result is read back and checked bit-exactly against the CPU reference, and the GPU
//     time of the detile dispatch is reported for a 1920x1080 frame.
//
// No swapchain here: the detiled image lands in a storage buffer that in Kyty would be the
// source of a blit into the swapchain image (see README). Standalone; does not touch Kyty.

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

#include "shaders.h" // generated: detile_rt_spv[], copy_linear_spv[]

namespace {

#define VKCHECK(expr)                                                                      \
	do {                                                                                   \
		VkResult res_ = (expr);                                                            \
		if (res_ != VK_SUCCESS) {                                                          \
			std::fprintf(stderr, "%s:%d: %s failed: VkResult %d\n", __FILE__, __LINE__,    \
			             #expr, static_cast<int>(res_));                                   \
			std::exit(1);                                                                  \
		}                                                                                  \
	} while (0)

uint32_t AlignUp(uint32_t v, uint32_t a) { return (v + a - 1u) / a * a; }
uint64_t AlignUp64(uint64_t v, uint64_t a) { return (v + a - 1u) / a * a; }

// --- CPU reference tiler -------------------------------------------------------------------
// rt_offset is the within-block byte offset for a 32bpp SW_64KB_R_X surface. Copied verbatim
// from Gen5RenderTargetOffsetInBlock<uint32_t> in src/graphics/guest_gpu/tile.cpp (the
// sizeof(T)==4 branch) and from the ELEMENT_BYTES==4 case of gpu_tiler_render_target.comp.
uint32_t RtOffset(uint32_t x, uint32_t y) {
	uint32_t o = 0;
	o ^= (y << 4u) & 0x0070u;
	o ^= (y << 5u) & 0x0f00u;
	o ^= (y << 9u) & 0x1000u;
	o ^= (y << 8u) & 0x4000u;
	o ^= (x << 2u) & 0x000cu;
	o ^= (x << 5u) & 0x0380u;
	o ^= (x << 4u) & 0x0400u;
	o ^= (x << 6u) & 0x0800u;
	o ^= (x << 9u) & 0xa000u;
	return o;
}

enum class Mode { Linear, RenderTarget };

struct Surface {
	Mode     mode;
	uint32_t width;
	uint32_t height;
	uint32_t pitch_words;    // linear: row pitch in 32-bit words (>= width)
	uint32_t blocks_per_row; // render target: AlignUp(width,128)/128
	uint64_t tiled_bytes;    // size of the tiled guest surface
};

Surface MakeSurface(Mode mode, uint32_t w, uint32_t h) {
	Surface s{};
	s.mode   = mode;
	s.width  = w;
	s.height = h;
	if (mode == Mode::Linear) {
		// Pitch deliberately padded past width*4 so the test proves pitch removal.
		s.pitch_words = AlignUp(w, 64) + 16;
		s.tiled_bytes = static_cast<uint64_t>(s.pitch_words) * h * 4u;
	} else {
		s.blocks_per_row       = AlignUp(w, 128) / 128u;
		const uint32_t rows     = AlignUp(h, 128) / 128u;
		s.tiled_bytes          = static_cast<uint64_t>(s.blocks_per_row) * rows * 65536ull;
	}
	return s;
}

// Byte offset of texel (x,y) in the tiled guest surface.
uint64_t TiledByte(const Surface& s, uint32_t x, uint32_t y) {
	if (s.mode == Mode::Linear) {
		return (static_cast<uint64_t>(y) * s.pitch_words + x) * 4u;
	}
	const uint64_t block_index = static_cast<uint64_t>(y >> 7) * s.blocks_per_row + (x >> 7);
	return block_index * 65536ull + RtOffset(x, y);
}

// Deterministic full-range 32-bit pattern for the linear ground-truth image.
uint32_t Pattern(uint32_t x, uint32_t y) {
	return (x * 2654435761u) ^ (y * 40503u) ^ 0x9e3779b9u ^ ((x ^ y) << 3);
}

uint32_t FindMemoryType(const VkPhysicalDeviceMemoryProperties& mp, uint32_t bits,
                        VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
			return i;
		}
	}
	return UINT32_MAX;
}

struct Vk {
	VkInstance                       instance = VK_NULL_HANDLE;
	VkPhysicalDevice                 phys     = VK_NULL_HANDLE;
	VkDevice                         dev      = VK_NULL_HANDLE;
	VkQueue                          queue    = VK_NULL_HANDLE;
	uint32_t                         qfam     = 0;
	VkPhysicalDeviceMemoryProperties mem{};
	VkPhysicalDeviceProperties       props{};
	uint64_t                         host_align = 4096;
	float                            ts_period  = 1.0f;
	VkCommandPool                    cmd_pool = VK_NULL_HANDLE;
	VkDescriptorPool                 desc_pool = VK_NULL_HANDLE;
	VkDescriptorSetLayout            set_layout = VK_NULL_HANDLE;
	VkPipelineLayout                 pipe_layout = VK_NULL_HANDLE;
	VkPipeline                       pipe_rt = VK_NULL_HANDLE;
	VkPipeline                       pipe_linear = VK_NULL_HANDLE;
	PFN_vkGetMemoryHostPointerPropertiesEXT get_host_ptr_props = nullptr;
};

VkPipeline MakePipeline(const Vk& v, const uint32_t* spv, size_t spv_bytes) {
	VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	smci.codeSize = spv_bytes;
	smci.pCode    = spv;
	VkShaderModule module = VK_NULL_HANDLE;
	VKCHECK(vkCreateShaderModule(v.dev, &smci, nullptr, &module));

	VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	stage.module = module;
	stage.pName  = "main";
	VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cpci.stage  = stage;
	cpci.layout = v.pipe_layout;
	VkPipeline pipe = VK_NULL_HANDLE;
	VKCHECK(vkCreateComputePipelines(v.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));
	vkDestroyShaderModule(v.dev, module, nullptr);
	return pipe;
}

Vk InitVulkan() {
	Vk v{};
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "bc250-present-probe";
	app.apiVersion       = VK_API_VERSION_1_1;
	VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &app;
	VKCHECK(vkCreateInstance(&ici, nullptr, &v.instance));

	uint32_t n = 0;
	VKCHECK(vkEnumeratePhysicalDevices(v.instance, &n, nullptr));
	if (n == 0) {
		std::fprintf(stderr, "no Vulkan physical device\n");
		std::exit(1);
	}
	std::vector<VkPhysicalDevice> devs(n);
	VKCHECK(vkEnumeratePhysicalDevices(v.instance, &n, devs.data()));
	v.phys = devs[0]; // the BC-250 is the only device on the board

	vkGetPhysicalDeviceProperties(v.phys, &v.props);
	vkGetPhysicalDeviceMemoryProperties(v.phys, &v.mem);
	v.ts_period = v.props.limits.timestampPeriod;

	// Require VK_EXT_external_memory_host.
	uint32_t ext_n = 0;
	vkEnumerateDeviceExtensionProperties(v.phys, nullptr, &ext_n, nullptr);
	std::vector<VkExtensionProperties> exts(ext_n);
	vkEnumerateDeviceExtensionProperties(v.phys, nullptr, &ext_n, exts.data());
	bool has_host = false;
	for (const auto& e : exts) {
		if (std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
			has_host = true;
		}
	}
	if (!has_host) {
		std::fprintf(stderr, "VK_EXT_external_memory_host not supported\n");
		std::exit(1);
	}

	VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_props{
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
	VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	p2.pNext = &host_props;
	vkGetPhysicalDeviceProperties2(v.phys, &p2);
	v.host_align = host_props.minImportedHostPointerAlignment;

	uint32_t qn = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(v.phys, &qn, nullptr);
	std::vector<VkQueueFamilyProperties> qprops(qn);
	vkGetPhysicalDeviceQueueFamilyProperties(v.phys, &qn, qprops.data());
	v.qfam = UINT32_MAX;
	for (uint32_t i = 0; i < qn; ++i) {
		if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			v.qfam = i;
			break;
		}
	}
	if (v.qfam == UINT32_MAX) {
		std::fprintf(stderr, "no compute queue\n");
		std::exit(1);
	}

	const float prio = 1.0f;
	VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = v.qfam;
	qci.queueCount       = 1;
	qci.pQueuePriorities = &prio;
	const char* dext[]   = {VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.queueCreateInfoCount    = 1;
	dci.pQueueCreateInfos       = &qci;
	dci.enabledExtensionCount   = 1;
	dci.ppEnabledExtensionNames = dext;
	VKCHECK(vkCreateDevice(v.phys, &dci, nullptr, &v.dev));
	vkGetDeviceQueue(v.dev, v.qfam, 0, &v.queue);

	v.get_host_ptr_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
	    vkGetDeviceProcAddr(v.dev, "vkGetMemoryHostPointerPropertiesEXT"));
	if (v.get_host_ptr_props == nullptr) {
		std::fprintf(stderr, "vkGetMemoryHostPointerPropertiesEXT missing\n");
		std::exit(1);
	}

	VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pci.queueFamilyIndex = v.qfam;
	pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VKCHECK(vkCreateCommandPool(v.dev, &pci, nullptr, &v.cmd_pool));

	VkDescriptorSetLayoutBinding binds[2]{};
	for (uint32_t i = 0; i < 2; ++i) {
		binds[i].binding         = i;
		binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		binds[i].descriptorCount = 1;
		binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	VkDescriptorSetLayoutCreateInfo slci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	slci.bindingCount = 2;
	slci.pBindings    = binds;
	VKCHECK(vkCreateDescriptorSetLayout(v.dev, &slci, nullptr, &v.set_layout));

	VkPushConstantRange pcr{};
	pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pcr.size       = 16;
	VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	plci.setLayoutCount         = 1;
	plci.pSetLayouts            = &v.set_layout;
	plci.pushConstantRangeCount = 1;
	plci.pPushConstantRanges    = &pcr;
	VKCHECK(vkCreatePipelineLayout(v.dev, &plci, nullptr, &v.pipe_layout));

	VkDescriptorPoolSize psize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 16};
	VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dpci.maxSets       = 16;
	dpci.poolSizeCount = 1;
	dpci.pPoolSizes    = &psize;
	dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	VKCHECK(vkCreateDescriptorPool(v.dev, &dpci, nullptr, &v.desc_pool));

	v.pipe_rt     = MakePipeline(v, detile_rt_spv, sizeof(detile_rt_spv));
	v.pipe_linear = MakePipeline(v, copy_linear_spv, sizeof(copy_linear_spv));
	return v;
}

struct PlainBuffer {
	VkBuffer       buf = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
};

PlainBuffer MakeBuffer(const Vk& v, VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props) {
	PlainBuffer b{};
	VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size        = size;
	bci.usage       = usage;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VKCHECK(vkCreateBuffer(v.dev, &bci, nullptr, &b.buf));
	VkMemoryRequirements req{};
	vkGetBufferMemoryRequirements(v.dev, b.buf, &req);
	const uint32_t type = FindMemoryType(v.mem, req.memoryTypeBits, props);
	if (type == UINT32_MAX) {
		std::fprintf(stderr, "no memory type for plain buffer\n");
		std::exit(1);
	}
	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.allocationSize  = req.size;
	mai.memoryTypeIndex = type;
	VKCHECK(vkAllocateMemory(v.dev, &mai, nullptr, &b.mem));
	VKCHECK(vkBindBufferMemory(v.dev, b.buf, b.mem, 0));
	return b;
}

// One case: tile the ground truth on the CPU, import the pages, detile on the GPU, verify.
bool RunCase(const Vk& v, const char* name, const Surface& s, double* gpu_ms_out,
             bool quiet = false) {
	const uint32_t texels      = s.width * s.height;
	const uint64_t linear_size = static_cast<uint64_t>(texels) * 4u;

	// Ground-truth linear image.
	std::vector<uint32_t> reference(texels);
	for (uint32_t y = 0; y < s.height; ++y) {
		for (uint32_t x = 0; x < s.width; ++x) {
			reference[y * s.width + x] = Pattern(x, y);
		}
	}

	// Host pages that stand in for the guest's memfd/userptr surface. mmap gives page alignment,
	// matching minImportedHostPointerAlignment; the mapped size is rounded up to that alignment.
	const uint64_t mapped = AlignUp64(s.tiled_bytes, v.host_align);
	void* host = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (host == MAP_FAILED) {
		std::fprintf(stderr, "mmap failed\n");
		std::exit(1);
	}
	std::memset(host, 0, mapped);
	auto* host_words = static_cast<uint32_t*>(host);
	for (uint32_t y = 0; y < s.height; ++y) {
		for (uint32_t x = 0; x < s.width; ++x) {
			host_words[TiledByte(s, x, y) >> 2] = reference[y * s.width + x];
		}
	}

	// Import the host pages with no copy (VK_EXT_external_memory_host).
	VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
	VKCHECK(v.get_host_ptr_props(v.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
	                             host, &hp));
	uint32_t import_type = FindMemoryType(v.mem, hp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	if (import_type == UINT32_MAX) {
		import_type = FindMemoryType(v.mem, hp.memoryTypeBits, 0);
	}
	if (import_type == UINT32_MAX) {
		std::fprintf(stderr, "no importable memory type for host pointer\n");
		std::exit(1);
	}

	VkExternalMemoryBufferCreateInfo ext_buf{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
	ext_buf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.pNext       = &ext_buf;
	bci.size        = s.tiled_bytes;
	bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer guest_buf = VK_NULL_HANDLE;
	VKCHECK(vkCreateBuffer(v.dev, &bci, nullptr, &guest_buf));

	VkImportMemoryHostPointerInfoEXT import{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
	import.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	import.pHostPointer = host;
	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.pNext           = &import;
	mai.allocationSize  = mapped;
	mai.memoryTypeIndex = import_type;
	VkDeviceMemory guest_mem = VK_NULL_HANDLE;
	VKCHECK(vkAllocateMemory(v.dev, &mai, nullptr, &guest_mem));
	VKCHECK(vkBindBufferMemory(v.dev, guest_buf, guest_mem, 0));

	// Output (device-local) and readback (host-visible) buffers.
	PlainBuffer out = MakeBuffer(v, linear_size,
	                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
	                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	PlainBuffer readback = MakeBuffer(v, linear_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
	                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

	// Descriptor set: binding 0 = guest surface, binding 1 = output.
	VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	dsai.descriptorPool     = v.desc_pool;
	dsai.descriptorSetCount = 1;
	dsai.pSetLayouts        = &v.set_layout;
	VkDescriptorSet set = VK_NULL_HANDLE;
	VKCHECK(vkAllocateDescriptorSets(v.dev, &dsai, &set));
	VkDescriptorBufferInfo in_info{guest_buf, 0, s.tiled_bytes};
	VkDescriptorBufferInfo out_info{out.buf, 0, linear_size};
	VkWriteDescriptorSet writes[2]{};
	for (uint32_t i = 0; i < 2; ++i) {
		writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[i].dstSet          = set;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	}
	writes[0].pBufferInfo = &in_info;
	writes[1].pBufferInfo = &out_info;
	vkUpdateDescriptorSets(v.dev, 2, writes, 0, nullptr);

	// Timestamp query pool.
	VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
	qpci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
	qpci.queryCount = 2;
	VkQueryPool qpool = VK_NULL_HANDLE;
	VKCHECK(vkCreateQueryPool(v.dev, &qpci, nullptr, &qpool));

	VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cbai.commandPool        = v.cmd_pool;
	cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbai.commandBufferCount = 1;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VKCHECK(vkAllocateCommandBuffers(v.dev, &cbai, &cmd));

	const uint32_t   pc[4] = {s.width, s.height,
	                          s.mode == Mode::Linear ? s.pitch_words : s.blocks_per_row, 0};
	const VkPipeline pipe  = s.mode == Mode::Linear ? v.pipe_linear : v.pipe_rt;
	const uint32_t   groups = (texels + 63u) / 64u;

	VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VKCHECK(vkBeginCommandBuffer(cmd, &begin));
	vkCmdResetQueryPool(cmd, qpool, 0, 2);
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, v.pipe_layout, 0, 1, &set, 0,
	                        nullptr);
	vkCmdPushConstants(cmd, v.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
	vkCmdDispatch(cmd, groups, 1, 1);
	vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, 1);

	VkBufferMemoryBarrier bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
	bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
	bar.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
	bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	bar.buffer              = out.buf;
	bar.size                = VK_WHOLE_SIZE;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &bar, 0, nullptr);
	VkBufferCopy copy{0, 0, linear_size};
	vkCmdCopyBuffer(cmd, out.buf, readback.buf, 1, &copy);
	VKCHECK(vkEndCommandBuffer(cmd));

	VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence fence = VK_NULL_HANDLE;
	VKCHECK(vkCreateFence(v.dev, &fci, nullptr, &fence));
	VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &cmd;
	VKCHECK(vkQueueSubmit(v.queue, 1, &submit, fence));
	VKCHECK(vkWaitForFences(v.dev, 1, &fence, VK_TRUE, UINT64_MAX));

	uint64_t ts[2] = {0, 0};
	VKCHECK(vkGetQueryPoolResults(v.dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
	                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
	const double gpu_ms = (ts[1] - ts[0]) * static_cast<double>(v.ts_period) / 1.0e6;
	if (gpu_ms_out != nullptr) {
		*gpu_ms_out = gpu_ms;
	}

	// Verify bit-exactly.
	void* mapped_rb = nullptr;
	VKCHECK(vkMapMemory(v.dev, readback.mem, 0, linear_size, 0, &mapped_rb));
	const auto* got      = static_cast<const uint32_t*>(mapped_rb);
	uint64_t    mismatch = 0;
	uint32_t    first_x = 0, first_y = 0, first_got = 0, first_exp = 0;
	for (uint32_t y = 0; y < s.height && mismatch == 0; ++y) {
		for (uint32_t x = 0; x < s.width; ++x) {
			const uint32_t exp = reference[y * s.width + x];
			const uint32_t g   = got[y * s.width + x];
			if (g != exp) {
				mismatch  = 1;
				first_x   = x;
				first_y   = y;
				first_got = g;
				first_exp = exp;
				break;
			}
		}
	}
	vkUnmapMemory(v.dev, readback.mem);

	const bool ok = mismatch == 0;
	if (!quiet) {
		std::printf("  %-22s %4ux%-4u  %-4s  tiled=%7.2f MiB  gpu=%7.3f ms  %s\n", name, s.width,
		            s.height, s.mode == Mode::Linear ? "lin" : "r_x",
		            s.tiled_bytes / (1024.0 * 1024.0), gpu_ms, ok ? "PASS" : "FAIL");
	}
	if (!ok) {
		std::printf("    first mismatch at (%u,%u): got %08x expected %08x\n", first_x, first_y,
		            first_got, first_exp);
	}

	vkDestroyFence(v.dev, fence, nullptr);
	vkDestroyQueryPool(v.dev, qpool, nullptr);
	vkFreeCommandBuffers(v.dev, v.cmd_pool, 1, &cmd);
	vkFreeDescriptorSets(v.dev, v.desc_pool, 1, &set);
	vkDestroyBuffer(v.dev, out.buf, nullptr);
	vkFreeMemory(v.dev, out.mem, nullptr);
	vkDestroyBuffer(v.dev, readback.buf, nullptr);
	vkFreeMemory(v.dev, readback.mem, nullptr);
	vkDestroyBuffer(v.dev, guest_buf, nullptr);
	vkFreeMemory(v.dev, guest_mem, nullptr); // unimports; does not free the host pages
	munmap(host, mapped);
	return ok;
}

} // namespace

int main(int argc, char** argv) {
	Vk v = InitVulkan();
	std::printf("device: %s  (timestampPeriod=%.3f ns, minImportedHostPointerAlignment=%llu)\n",
	            v.props.deviceName, static_cast<double>(v.ts_period),
	            static_cast<unsigned long long>(v.host_align));

	// Optional CLI: present_probe <linear|rt> <w> <h> runs one custom case.
	if (argc == 4) {
		const Mode mode = std::strcmp(argv[1], "linear") == 0 ? Mode::Linear : Mode::RenderTarget;
		const auto w    = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
		const auto h    = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
		const bool ok   = RunCase(v, "custom", MakeSurface(mode, w, h), nullptr);
		return ok ? 0 : 1;
	}

	bool all_ok = true;
	std::printf("correctness:\n");
	// Odd sizes to exercise partial edge blocks and the pitch padding.
	all_ok &= RunCase(v, "linear small", MakeSurface(Mode::Linear, 17, 13), nullptr);
	all_ok &= RunCase(v, "linear odd", MakeSurface(Mode::Linear, 257, 129), nullptr);
	all_ok &= RunCase(v, "r_x small", MakeSurface(Mode::RenderTarget, 64, 64), nullptr);
	all_ok &= RunCase(v, "r_x one block", MakeSurface(Mode::RenderTarget, 128, 128), nullptr);
	all_ok &= RunCase(v, "r_x odd", MakeSurface(Mode::RenderTarget, 300, 200), nullptr);
	all_ok &= RunCase(v, "r_x 1080p", MakeSurface(Mode::RenderTarget, 1920, 1080), nullptr);
	all_ok &= RunCase(v, "linear 1080p", MakeSurface(Mode::Linear, 1920, 1080), nullptr);

	std::printf("timing (1920x1080, best of 20):\n");
	for (const Mode m : {Mode::RenderTarget, Mode::Linear}) {
		double best = 1e30;
		for (int i = 0; i < 20; ++i) {
			double ms = 0;
			all_ok &= RunCase(v, "frame", MakeSurface(m, 1920, 1080), &ms, /*quiet=*/true);
			if (ms < best) {
				best = ms;
			}
		}
		std::printf("  %-4s detile 1920x1080 best GPU time: %.3f ms\n",
		            m == Mode::Linear ? "lin" : "r_x", best);
	}

	std::printf("%s\n", all_ok ? "ALL PASS" : "FAILURES PRESENT");
	return all_ok ? 0 : 1;
}
