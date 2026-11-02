#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "libs/agc.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>

// Kyty's command processor is the native backend's safety net for refused buffers, and libSceAgc's reset defaults
// (AgcDcbResetQueue ops 2, 4 and 0x800: SET_SH_REG_INDIRECT and SET_UCONFIG_REG_INDIRECT over the SH and UCONFIG
// tables of every register defaults version) are the first thing such a buffer often holds. Every register of those
// tables must either have a handler or fall to the CP's reset-table skip, which works only below SH_NUM / UC_NUM (an
// offset past them ends the process before the skip). The registers left to the skip are printed, so a handler added
// for one shows up as a shorter list.

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AgcResetTableCpTests: failed: %s\n", text);
		g_failures++;
	}
}

void PrintSet(const char* what, const std::set<uint32_t>& offsets) {
	std::printf("%s (%zu):", what, offsets.size());
	for (uint32_t o: offsets) {
		std::printf(" 0x%03" PRIx32, o);
	}
	std::printf("\n");
}

void TestResetTablesCovered() {
	GraphicsInitJmpTablesShIndirect();
	GraphicsInitJmpTablesUcIndirect();

	// The first pair of the sorted SH defaults, the register a refused buffer used to die on.
	Check(g_hw_sh_indirect_func[Pm4::SPI_SHADER_PGM_RSRC4_PS] != nullptr, "SPI_SHADER_PGM_RSRC4_PS has a handler");

	std::set<uint32_t> sh_skipped;
	std::set<uint32_t> uc_skipped;
	for (uint32_t ver = 0; ver <= 13; ver++) {
		const auto defaults = Gen5::AgcStartStateDefaults(ver);
		Check(!defaults.sh.empty(), "every version has SH defaults");
		for (const auto& [offset, value]: defaults.sh) {
			(void)value;
			if (offset == Pm4::SH_NOP) {
				continue;
			}
			if (offset >= Pm4::SH_NUM) {
				std::fprintf(stderr, "AgcResetTableCpTests: version %" PRIu32 " SH offset 0x%" PRIx32 " past SH_NUM\n", ver,
				             offset);
				Check(false, "SH default inside the CP's SH range");
				continue;
			}
			if (g_hw_sh_indirect_func[offset] == nullptr) {
				sh_skipped.insert(offset);
			}
		}
		for (const auto& [offset, value]: defaults.uconfig) {
			(void)value;
			if (offset == Pm4::UC_NOP) {
				continue;
			}
			if (offset >= Pm4::UC_NUM) {
				std::fprintf(stderr, "AgcResetTableCpTests: version %" PRIu32 " UCONFIG offset 0x%" PRIx32 " past UC_NUM\n",
				             ver, offset);
				Check(false, "UCONFIG default inside the CP's UCONFIG range");
				continue;
			}
			if (g_hw_uc_indirect_func[offset] == nullptr) {
				uc_skipped.insert(offset);
			}
		}
	}
	Check(sh_skipped.count(Pm4::SPI_SHADER_PGM_RSRC4_PS) == 0, "RSRC4_PS not left to the skip");
	PrintSet("SH defaults left to the reset-table skip", sh_skipped);
	PrintSet("UCONFIG defaults left to the reset-table skip", uc_skipped);
}

} // namespace

int main() {
	TestResetTablesCovered();
	if (g_failures != 0) {
		return 1;
	}
	std::puts("AgcResetTableCpTests: all passed");
	return 0;
}
