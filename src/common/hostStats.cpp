#include "common/hostStats.h"

#include "common/config.h"
#include "common/timer.h"

#include <algorithm>
#include <array>
#include <thread>
#include <unordered_map>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// clang-format off
#include <psapi.h>
// clang-format on
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <cstring>
#include <libproc.h>
#include <mach/mach.h>
#include <notify.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace Common {

namespace {

// Cumulative counters; rates come from the difference between two samples.
struct Counters {
	struct Thread {
		uint64_t    cpu_ns;
		std::string name;
	};
	std::optional<uint64_t>              process_cpu_ns;
	std::optional<uint64_t>              system_busy;
	std::optional<uint64_t>              system_total;
	std::optional<uint64_t>              context_switches;
	std::optional<uint64_t>              page_faults;
	std::optional<uint64_t>              page_ins;
	std::optional<uint64_t>              disk_read;
	std::optional<uint64_t>              disk_write;
	std::unordered_map<uint64_t, Thread> threads;
	uint64_t                             wall_ns = 0;
};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

uint64_t FileTimeTo100Ns(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
}

struct PlatformState {};

void WantGpuLoad(PlatformState& /*state*/, bool /*wanted*/) {}

void SamplePlatform(PlatformState& /*state*/, HostStats& stats, Counters& now) {
	const auto process = GetCurrentProcess();
	FILETIME   creation {};
	FILETIME   exit {};
	FILETIME   kernel {};
	FILETIME   user {};
	if (GetProcessTimes(process, &creation, &exit, &kernel, &user) != 0) {
		now.process_cpu_ns = (FileTimeTo100Ns(kernel) + FileTimeTo100Ns(user)) * 100u;
	}
	FILETIME idle {};
	if (GetSystemTimes(&idle, &kernel, &user) != 0) {
		// Kernel time includes idle time.
		now.system_total = FileTimeTo100Ns(kernel) + FileTimeTo100Ns(user);
		now.system_busy  = *now.system_total - FileTimeTo100Ns(idle);
	}

	PROCESS_MEMORY_COUNTERS memory {};
	if (K32GetProcessMemoryInfo(process, &memory, sizeof(memory)) != 0) {
		stats.process_memory   = memory.WorkingSetSize;
		stats.process_resident = memory.WorkingSetSize;
		now.page_faults        = memory.PageFaultCount;
	}
	IO_COUNTERS io {};
	if (GetProcessIoCounters(process, &io) != 0) {
		now.disk_read  = io.ReadTransferCount;
		now.disk_write = io.WriteTransferCount;
	}

	MEMORYSTATUSEX system {};
	system.dwLength = sizeof(system);
	if (GlobalMemoryStatusEx(&system) != 0) {
		stats.system_memory_total = system.ullTotalPhys;
		stats.system_memory_used  = system.ullTotalPhys - system.ullAvailPhys;
		stats.system_free         = system.ullAvailPhys;
		stats.memory_pressure     = system.dwMemoryLoad >= 90   ? MemoryPressure::Critical
		                            : system.dwMemoryLoad >= 80 ? MemoryPressure::Warning
		                                                        : MemoryPressure::Normal;
	}
}

#else

void SampleRusage(Counters& now) {
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) != 0) {
		return;
	}
	const auto to_ns = [](const timeval& time) {
		return static_cast<uint64_t>(time.tv_sec) * 1000000000u +
		       static_cast<uint64_t>(time.tv_usec) * 1000u;
	};
	now.process_cpu_ns   = to_ns(usage.ru_utime) + to_ns(usage.ru_stime);
	now.context_switches = static_cast<uint64_t>(usage.ru_nvcsw + usage.ru_nivcsw);
	now.page_faults      = static_cast<uint64_t>(usage.ru_minflt + usage.ru_majflt);
	now.page_ins         = static_cast<uint64_t>(usage.ru_majflt);
}

#if defined(__APPLE__)

template <typename T>
std::optional<T> SysctlValue(const char* name) {
	T      value {};
	size_t size = sizeof(value);
	if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 || size != sizeof(value)) {
		return std::nullopt;
	}
	return value;
}

struct PlatformState {
	mach_port_t             host          = mach_host_self();
	int                     thermal_token = -1;
	bool                    thermal_ready = false;
	std::string             gpu_model;
	std::optional<uint32_t> gpu_cores;

	~PlatformState() {
		if (thermal_ready) {
			notify_cancel(thermal_token);
		}
		mach_port_deallocate(mach_task_self(), host);
	}
};

std::optional<int64_t> DictionaryNumber(CFDictionaryRef dictionary, CFStringRef key) {
	const auto* value  = CFDictionaryGetValue(dictionary, key);
	int64_t     number = 0;
	if (value == nullptr || CFGetTypeID(value) != CFNumberGetTypeID() ||
	    !CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberSInt64Type, &number) ||
	    number < 0) {
		return std::nullopt;
	}
	return number;
}

std::string RegistryString(io_object_t service, CFStringRef key) {
	std::string result;
	auto*       value = IORegistryEntryCreateCFProperty(service, key, kCFAllocatorDefault, 0);
	if (value == nullptr) {
		return result;
	}
	if (CFGetTypeID(value) == CFStringGetTypeID()) {
		char buffer[128] {};
		if (CFStringGetCString(static_cast<CFStringRef>(value), buffer, sizeof(buffer),
		                       kCFStringEncodingUTF8)) {
			result = buffer;
		}
	} else if (CFGetTypeID(value) == CFDataGetTypeID()) {
		// Intel Macs store the model as NUL-terminated data.
		const auto* data   = static_cast<CFDataRef>(value);
		const auto* bytes  = reinterpret_cast<const char*>(CFDataGetBytePtr(data));
		const auto  length = static_cast<size_t>(CFDataGetLength(data));
		result.assign(bytes, strnlen(bytes, length));
	}
	CFRelease(value);
	return result;
}

// IOAccelerator exposes whole-device statistics; there is no per-process figure.
void SampleGpu(PlatformState& state, HostStats& stats) {
	io_iterator_t iterator = 0;
	if (IOServiceGetMatchingServices(MACH_PORT_NULL, IOServiceMatching("IOAccelerator"),
	                                 &iterator) != KERN_SUCCESS) {
		return;
	}
	const auto take_max = [](std::optional<double>& target, std::optional<int64_t> value) {
		if (value) {
			target = std::max(target.value_or(0.0), std::min(100.0, static_cast<double>(*value)));
		}
	};
	const auto add = [](std::optional<uint64_t>& target, std::optional<int64_t> value) {
		if (value) {
			target = target.value_or(0) + static_cast<uint64_t>(*value);
		}
	};
	for (io_object_t service = IOIteratorNext(iterator); service != 0;
	     service             = IOIteratorNext(iterator)) {
		if (state.gpu_model.empty()) {
			state.gpu_model = RegistryString(service, CFSTR("model"));
			auto*   cores   = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"),
			                                                  kCFAllocatorDefault, 0);
			int64_t count   = 0;
			if (cores != nullptr) {
				if (CFGetTypeID(cores) == CFNumberGetTypeID() &&
				    CFNumberGetValue(static_cast<CFNumberRef>(cores), kCFNumberSInt64Type,
				                     &count) &&
				    count > 0) {
					state.gpu_cores = static_cast<uint32_t>(count);
				}
				CFRelease(cores);
			}
		}
		auto* statistics = IORegistryEntryCreateCFProperty(service, CFSTR("PerformanceStatistics"),
		                                                   kCFAllocatorDefault, 0);
		if (statistics != nullptr) {
			if (CFGetTypeID(statistics) == CFDictionaryGetTypeID()) {
				const auto* values = static_cast<CFDictionaryRef>(statistics);
				take_max(stats.gpu_percent,
				         DictionaryNumber(values, CFSTR("Device Utilization %")));
				take_max(stats.gpu_renderer_percent,
				         DictionaryNumber(values, CFSTR("Renderer Utilization %")));
				take_max(stats.gpu_tiler_percent,
				         DictionaryNumber(values, CFSTR("Tiler Utilization %")));
				add(stats.gpu_memory_in_use,
				    DictionaryNumber(values, CFSTR("In use system memory")));
				add(stats.gpu_memory_allocated,
				    DictionaryNumber(values, CFSTR("Alloc system memory")));
				add(stats.gpu_recoveries, DictionaryNumber(values, CFSTR("recoveryCount")));
			}
			CFRelease(statistics);
		}
		IOObjectRelease(service);
	}
	IOObjectRelease(iterator);
	stats.gpu_model = state.gpu_model;
	stats.gpu_cores = state.gpu_cores;
}

void SampleThreads(HostStats& stats, Counters& now) {
	thread_act_array_t     threads = nullptr;
	mach_msg_type_number_t count   = 0;
	const auto             task    = mach_task_self();
	if (task_threads(task, &threads, &count) != KERN_SUCCESS) {
		return;
	}
	stats.thread_count = count;
	for (mach_msg_type_number_t i = 0; i < count; i++) {
		thread_identifier_info_data_t identifier {};
		mach_msg_type_number_t        size = THREAD_IDENTIFIER_INFO_COUNT;
		thread_basic_info_data_t      basic {};
		mach_msg_type_number_t        basic_size = THREAD_BASIC_INFO_COUNT;
		if (thread_info(threads[i], THREAD_IDENTIFIER_INFO,
		                reinterpret_cast<thread_info_t>(&identifier), &size) == KERN_SUCCESS &&
		    thread_info(threads[i], THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&basic),
		                &basic_size) == KERN_SUCCESS) {
			const auto to_ns = [](const time_value_t& time) {
				return static_cast<uint64_t>(time.seconds) * 1000000000u +
				       static_cast<uint64_t>(time.microseconds) * 1000u;
			};
			auto& thread  = now.threads[identifier.thread_id];
			thread.cpu_ns = to_ns(basic.user_time) + to_ns(basic.system_time);
			thread_extended_info_data_t extended {};
			mach_msg_type_number_t      extended_size = THREAD_EXTENDED_INFO_COUNT;
			if (thread_info(threads[i], THREAD_EXTENDED_INFO,
			                reinterpret_cast<thread_info_t>(&extended),
			                &extended_size) == KERN_SUCCESS) {
				thread.name = std::string(extended.pth_name,
				                          strnlen(extended.pth_name, sizeof(extended.pth_name)));
			}
		}
		mach_port_deallocate(task, threads[i]);
	}
	vm_deallocate(task, reinterpret_cast<vm_address_t>(threads), count * sizeof(thread_act_t));
}

void WantGpuLoad(PlatformState& /*state*/, bool /*wanted*/) {}

void SamplePlatform(PlatformState& state, HostStats& stats, Counters& now) {
	SampleRusage(now);

	host_cpu_load_info_data_t load {};
	mach_msg_type_number_t    load_count = HOST_CPU_LOAD_INFO_COUNT;
	if (host_statistics(state.host, HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&load),
	                    &load_count) == KERN_SUCCESS) {
		const uint64_t busy = static_cast<uint64_t>(load.cpu_ticks[CPU_STATE_USER]) +
		                      load.cpu_ticks[CPU_STATE_SYSTEM] + load.cpu_ticks[CPU_STATE_NICE];
		now.system_busy     = busy;
		now.system_total    = busy + load.cpu_ticks[CPU_STATE_IDLE];
	}

	// Physical footprint is what Activity Monitor shows in its Memory column.
	task_vm_info_data_t    vm_info {};
	mach_msg_type_number_t vm_info_count = TASK_VM_INFO_COUNT;
	if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info),
	              &vm_info_count) == KERN_SUCCESS) {
		stats.process_memory      = vm_info.phys_footprint;
		stats.process_resident    = vm_info.resident_size;
		stats.process_compressed  = vm_info.compressed;
		stats.process_file_backed = vm_info.external;
		stats.process_virtual     = vm_info.virtual_size;
	}
	rusage_info_v2 usage {};
	if (proc_pid_rusage(getpid(), RUSAGE_INFO_V2, reinterpret_cast<rusage_info_t*>(&usage)) == 0) {
		now.disk_read  = usage.ri_diskio_bytesread;
		now.disk_write = usage.ri_diskio_byteswritten;
	}

	stats.system_memory_total   = SysctlValue<uint64_t>("hw.memsize");
	vm_size_t              page = 0;
	vm_statistics64_data_t vm {};
	mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
	if (host_page_size(state.host, &page) == KERN_SUCCESS &&
	    host_statistics64(state.host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
	                      &count) == KERN_SUCCESS) {
		// Activity Monitor's "Memory Used": app memory + wired + compressed.
		const uint64_t app       = vm.internal_page_count > vm.purgeable_count
		                               ? vm.internal_page_count - vm.purgeable_count
		                               : 0;
		stats.system_memory_used = (app + vm.wire_count + vm.compressor_page_count) * page;
		stats.system_wired       = static_cast<uint64_t>(vm.wire_count) * page;
		stats.system_compressed  = static_cast<uint64_t>(vm.compressor_page_count) * page;
		stats.system_cached =
		    (static_cast<uint64_t>(vm.external_page_count) + vm.purgeable_count) * page;
		stats.system_free = static_cast<uint64_t>(vm.free_count) * page;
	}

	if (const auto swap = SysctlValue<xsw_usage>("vm.swapusage")) {
		stats.swap_used  = swap->xsu_used;
		stats.swap_total = swap->xsu_total;
	}

	// 1 = normal, 2 = warning, 4 = critical, as reported to dispatch memory-pressure sources.
	if (const auto level = SysctlValue<int>("kern.memorystatus_vm_pressure_level")) {
		stats.memory_pressure = *level >= 4   ? MemoryPressure::Critical
		                        : *level >= 2 ? MemoryPressure::Warning
		                                      : MemoryPressure::Normal;
	}

	// 0 nominal, 1 moderate, 2 heavy, 3 trapping, 4 sleeping.
	if (!state.thermal_ready) {
		state.thermal_ready = notify_register_check("com.apple.system.thermalpressurelevel",
		                                            &state.thermal_token) == NOTIFY_STATUS_OK;
	}
	uint64_t thermal = 0;
	if (state.thermal_ready &&
	    notify_get_state(state.thermal_token, &thermal) == NOTIFY_STATUS_OK) {
		stats.thermal_pressure = thermal >= 3   ? ThermalPressure::Critical
		                         : thermal == 2 ? ThermalPressure::Heavy
		                         : thermal == 1 ? ThermalPressure::Moderate
		                                        : ThermalPressure::Nominal;
	}

	SampleThreads(stats, now);
	SampleGpu(state, stats);
}

#else

using ProcValues = std::unordered_map<std::string, uint64_t>;

// Reads "<key>: <number> [kB]" lines; kB values are returned in bytes.
ProcValues ReadProcValues(const char* path) {
	ProcValues    values;
	std::ifstream file(path);
	std::string   line;
	while (std::getline(file, line)) {
		const auto colon = line.find(':');
		if (colon == std::string::npos) {
			continue;
		}
		std::istringstream stream(line.substr(colon + 1));
		uint64_t           number = 0;
		std::string        unit;
		if (stream >> number) {
			stream >> unit;
			values[line.substr(0, colon)] = unit == "kB" ? number * 1024u : number;
		}
	}
	return values;
}

std::optional<uint64_t> Find(const ProcValues& values, const char* key) {
	const auto it = values.find(key);
	return it != values.end() ? std::optional(it->second) : std::nullopt;
}

// The GPU's busy share from GRBM_STATUS (GUI_ACTIVE), as radeontop reads it: the kernel reports no load on every
// GPU (the BC-250's does not). A thread of its own samples it, and only while it is wanted.
struct GpuBusy {
	struct Query { // drm_amdgpu_info with its read_mmr_reg member
		uint64_t return_pointer;
		uint32_t return_size;
		uint32_t query;
		uint32_t dword_offset;
		uint32_t count;
		uint32_t instance;
		uint32_t flags;
	};

	std::atomic<uint64_t> busy {0};
	std::atomic<uint64_t> total {0};
	std::atomic<bool>     wanted {false};
	std::jthread          thread;
	bool                  tried = false;

	static bool ReadStatus(int fd, uint32_t* value) {
		Query query {reinterpret_cast<uintptr_t>(value), 4, 0x15 /* AMDGPU_INFO_READ_MMR_REG */, 0x2004 /* GRBM_STATUS */, 1,
		             0xffffffffu, 0};
		return ioctl(fd, _IOW('d', 0x45, Query), &query) == 0;
	}

	static int Open() {
		for (int i = 128; i < 192; i++) {
			std::string   vendor;
			std::ifstream vendor_file("/sys/class/drm/renderD" + std::to_string(i) + "/device/vendor");
			vendor_file >> vendor;
			if (vendor != "0x1002") {
				continue;
			}
			const int fd = ::open(("/dev/dri/renderD" + std::to_string(i)).c_str(), O_RDWR | O_CLOEXEC);
			if (fd < 0) {
				continue;
			}
			uint32_t value = 0;
			if (ReadStatus(fd, &value)) {
				return fd;
			}
			::close(fd);
		}
		return -1;
	}

	void Want(bool on) {
		wanted.store(on, std::memory_order_relaxed);
		if (!on || tried) {
			return;
		}
		tried        = true;
		const int fd = Open();
		if (fd < 0) {
			return;
		}
		thread = std::jthread([this, fd](const std::stop_token& stop) {
			while (!stop.stop_requested()) {
				if (wanted.load(std::memory_order_relaxed)) {
					uint32_t value = 0;
					if (ReadStatus(fd, &value)) {
						total.fetch_add(1, std::memory_order_relaxed);
						if (((value >> 31u) & 1u) != 0) {
							busy.fetch_add(1, std::memory_order_relaxed);
						}
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
				} else {
					std::this_thread::sleep_for(std::chrono::milliseconds(200));
				}
			}
			::close(fd);
		});
	}

	// The share since the last call, 200 samples a second; nothing until there are enough.
	std::optional<double> Take() {
		const uint64_t samples = total.exchange(0, std::memory_order_relaxed);
		const uint64_t active  = busy.exchange(0, std::memory_order_relaxed);
		return samples >= 20 ? std::optional(100.0 * static_cast<double>(active) / static_cast<double>(samples)) : std::nullopt;
	}
};

struct PlatformState {
	std::vector<std::array<uint64_t, 2>> cores; // Each core's total and busy ticks at the last sample.
	GpuBusy                              gpu;
};

void WantGpuLoad(PlatformState& state, bool wanted) {
	state.gpu.Want(wanted);
}

void SampleThreads(HostStats& stats, Counters& now) {
	static const double ns_per_tick = 1e9 / static_cast<double>(std::max(sysconf(_SC_CLK_TCK), 1L));
	std::error_code     error;
	uint32_t            count = 0;
	for (std::filesystem::directory_iterator it("/proc/self/task", error), end; !error && it != end;
	     it.increment(error)) {
		count++;
		std::ifstream stat_file(it->path() / "stat");
		std::string   stat;
		uint64_t      id = 0;
		if (!(stat_file >> id)) {
			continue; // The thread may have exited during enumeration.
		}
		std::getline(stat_file, stat);
		const auto close = stat.rfind(')');
		if (close == std::string::npos) {
			continue;
		}
		// Fields after the command name start at field 3 (state); utime and stime are 14 and 15.
		std::istringstream fields(stat.substr(close + 2));
		std::string        field;
		for (int index = 3; index < 14 && fields >> field; index++) {
		}
		uint64_t user   = 0;
		uint64_t system = 0;
		if (!(fields >> user >> system)) {
			continue;
		}
		auto& thread  = now.threads[id];
		thread.cpu_ns = static_cast<uint64_t>(static_cast<double>(user + system) * ns_per_tick);
		std::ifstream name_file(it->path() / "comm");
		std::getline(name_file, thread.name);
	}
	if (!error) {
		stats.thread_count = count;
	}
}

void SamplePlatform(PlatformState& state, HostStats& stats, Counters& now) {
	SampleRusage(now);

	std::ifstream cpu_file("/proc/stat");
	std::string   label;
	if (cpu_file >> label && label == "cpu") {
		uint64_t total = 0;
		uint64_t idle  = 0;
		uint64_t value = 0;
		for (int index = 0; index < 8 && cpu_file >> value; index++) {
			total += value;
			if (index == 3 || index == 4) { // idle, iowait
				idle += value;
			}
		}
		now.system_total = total;
		now.system_busy  = total - idle;
	}

	// Each logical core's share since the previous sample, from the "cpuN" lines.
	{
		std::ifstream cores("/proc/stat");
		std::string   core_line;
		size_t        index = 0;
		while (std::getline(cores, core_line)) {
			if (core_line.size() < 4 || core_line.rfind("cpu", 0) != 0 || core_line[3] < '0' || core_line[3] > '9') {
				continue;
			}
			std::istringstream fields(core_line);
			std::string        name;
			fields >> name;
			uint64_t total = 0;
			uint64_t idle  = 0;
			uint64_t value = 0;
			for (int i = 0; i < 8 && fields >> value; i++) {
				total += value;
				if (i == 3 || i == 4) {
					idle += value;
				}
			}
			const uint64_t busy = total - idle;
			if (index < state.cores.size() && total > state.cores[index][0] && busy >= state.cores[index][1]) {
				stats.core_percent.push_back(std::clamp(static_cast<double>(busy - state.cores[index][1]) /
				                                            static_cast<double>(total - state.cores[index][0]) * 100.0,
				                                        0.0, 100.0));
			}
			if (index >= state.cores.size()) {
				state.cores.resize(index + 1);
			}
			state.cores[index] = {total, busy};
			index++;
		}
	}

	// Temperatures, clock and power from the hwmon nodes of the CPU (k10temp) and the GPU (amdgpu).
	{
		std::error_code error;
		for (std::filesystem::directory_iterator it("/sys/class/hwmon", error), end; !error && it != end;
		     it.increment(error)) {
			std::string   name;
			std::ifstream name_file(it->path() / "name");
			name_file >> name;
			const auto number = [&](const char* file, double divisor) -> std::optional<double> {
				std::ifstream node(it->path() / file);
				double        value = 0.0;
				if (node >> value) {
					return value / divisor;
				}
				return std::nullopt;
			};
			if (name == "k10temp") {
				stats.cpu_temp_c = number("temp1_input", 1000.0);
			} else if (name == "amdgpu") {
				stats.gpu_temp_c   = number("temp1_input", 1000.0);
				stats.gpu_power_w  = number("power1_average", 1e6);
				stats.gpu_clock_mhz = number("freq1_input", 1e6);
			}
		}
		stats.gpu_percent = state.gpu.Take();
	}

	const auto status         = ReadProcValues("/proc/self/status");
	stats.process_memory      = Find(status, "VmRSS");
	stats.process_resident    = Find(status, "VmRSS");
	stats.process_file_backed = Find(status, "RssFile");
	stats.process_compressed  = Find(status, "VmSwap");
	stats.process_virtual     = Find(status, "VmSize");

	const auto io  = ReadProcValues("/proc/self/io");
	now.disk_read  = Find(io, "read_bytes");
	now.disk_write = Find(io, "write_bytes");

	const auto memory    = ReadProcValues("/proc/meminfo");
	const auto total     = Find(memory, "MemTotal");
	const auto available = Find(memory, "MemAvailable");
	if (total && available && *available <= *total) {
		stats.system_memory_total = total;
		stats.system_memory_used  = *total - *available;
	}
	stats.system_free = Find(memory, "MemFree");
	if (const auto cached = Find(memory, "Cached")) {
		stats.system_cached = *cached + Find(memory, "Buffers").value_or(0);
	}
	const auto swap_total = Find(memory, "SwapTotal");
	const auto swap_free  = Find(memory, "SwapFree");
	if (swap_total && swap_free && *swap_free <= *swap_total) {
		stats.swap_total = swap_total;
		stats.swap_used  = *swap_total - *swap_free;
	}

	// Pressure stall information: share of the last 10 s some task waited on memory.
	std::ifstream pressure("/proc/pressure/memory");
	std::string   kind;
	std::string   average;
	double        stalled = 0.0;
	if (pressure >> kind >> average && kind == "some" && average.rfind("avg10=", 0) == 0 &&
	    std::istringstream(average.substr(6)) >> stalled) {
		stats.memory_stall_percent = stalled;
		stats.memory_pressure = stalled >= 20.0  ? MemoryPressure::Critical
		                        : stalled >= 5.0 ? MemoryPressure::Warning
		                                         : MemoryPressure::Normal;
	} else if (total && available && *total != 0) {
		const double free_share = static_cast<double>(*available) / static_cast<double>(*total);
		stats.memory_pressure   = free_share < 0.10   ? MemoryPressure::Critical
		                          : free_share < 0.25 ? MemoryPressure::Warning
		                                              : MemoryPressure::Normal;
	}

	SampleThreads(stats, now);
}

#endif
#endif

constexpr size_t BUSIEST_THREADS = 6;

} // namespace

struct HostStatsSampler::Impl {
	PlatformState state;
	Counters      last;
	uint32_t      logical_cores = std::max(std::thread::hardware_concurrency(), 1u);
	bool          has_last      = false;
};

HostStatsSampler::HostStatsSampler(): m_impl(std::make_unique<Impl>()) {}

HostStatsSampler::~HostStatsSampler() = default;

void HostStatsSampler::SetGpuLoadWanted(bool wanted) {
	WantGpuLoad(m_impl->state, wanted);
}

HostStats HostStatsSampler::Sample() {
	HostStats stats;
	stats.logical_cores = m_impl->logical_cores;

	Counters now;
	now.wall_ns = Common::SteadyNs();
	SamplePlatform(m_impl->state, stats, now);

	const auto& last    = m_impl->last;
	const auto  elapsed = static_cast<double>(now.wall_ns - last.wall_ns) / 1e9;
	if (m_impl->has_last && elapsed > 0.0) {
		const auto rate = [&](std::optional<uint64_t> current,
		                      std::optional<uint64_t> previous) -> std::optional<double> {
			if (!current || !previous || *current < *previous) {
				return std::nullopt;
			}
			return static_cast<double>(*current - *previous) / elapsed;
		};
		if (const auto cpu = rate(now.process_cpu_ns, last.process_cpu_ns)) {
			stats.process_cpu_percent =
			    std::clamp(*cpu / 1e9 * 100.0 / stats.logical_cores, 0.0, 100.0);
		}
		if (now.system_total && last.system_total && now.system_busy && last.system_busy &&
		    *now.system_total > *last.system_total && *now.system_busy >= *last.system_busy) {
			stats.system_cpu_percent =
			    std::clamp(static_cast<double>(*now.system_busy - *last.system_busy) /
			                   static_cast<double>(*now.system_total - *last.system_total) * 100.0,
			               0.0, 100.0);
		}
		stats.context_switches_per_s = rate(now.context_switches, last.context_switches);
		stats.page_faults_per_s      = rate(now.page_faults, last.page_faults);
		stats.page_ins_per_s         = rate(now.page_ins, last.page_ins);
		stats.disk_read_per_s        = rate(now.disk_read, last.disk_read);
		stats.disk_write_per_s       = rate(now.disk_write, last.disk_write);

		for (const auto& [id, thread]: now.threads) {
			// A thread missing from the last sample started during this interval.
			const auto     previous = last.threads.find(id);
			const uint64_t before   = previous != last.threads.end() ? previous->second.cpu_ns : 0;
			if (thread.cpu_ns < before) {
				continue;
			}
			const double percent =
			    static_cast<double>(thread.cpu_ns - before) / 1e9 / elapsed * 100.0;
			stats.busiest_threads.push_back(
			    {thread.name.empty() ? "thread " + std::to_string(id) : thread.name, percent});
		}
		std::sort(stats.busiest_threads.begin(), stats.busiest_threads.end(),
		          [](const auto& a, const auto& b) { return a.cpu_percent > b.cpu_percent; });
		if (stats.busiest_threads.size() > BUSIEST_THREADS) {
			stats.busiest_threads.resize(BUSIEST_THREADS);
		}
	}
	m_impl->last     = std::move(now);
	m_impl->has_last = true;
	return stats;
}

} // namespace Common
