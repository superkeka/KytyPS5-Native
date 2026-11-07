#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "libs/agc.h"

#include <array>
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

// The size queries and patches: the values libSceAgc gives, and packets the writers fill to exactly the size asked for.
void TestSizesAndPatches() {
	namespace G = Libs::Graphics::Gen5;
	Check(G::AgcDcbAtomicGdsGetSize() == 44 && G::AgcAcbAtomicGdsGetSize() == 44 && G::AgcDcbPrimeUtcl2GetSize() == 20 && G::AgcAcbPrimeUtcl2GetSize() == 20,
	      "fixed sizes");
	Check(G::AgcDcbQueueEndOfShaderActionGetSize() == 32 && G::AgcAcbQueueEndOfShaderActionGetSize() == 32 && G::AgcDcbSetBaseDispatchIndirectArgsGetSize() == 16 &&
	          G::AgcDcbSetBaseDrawIndirectArgsGetSize() == 16 && G::AgcDcbSetBoolPredicationEnableGetSize() == 16 && G::AgcDcbSetZPassPredicationEnableGetSize() == 16 &&
	          G::AgcDcbSetIndexIndirectArgsGetSize() == 16 && G::AgcDcbSetShRegisterDirectGetSize() == 12 && G::AgcDcbSetUcRegisterDirectGetSize() == 12 &&
	          G::AgcDcbSetUcRegistersIndirectGetSize() == 20 && G::AgcDcbStallCommandBufferParserGetSize() == 8,
	      "fixed sizes");
	Check(G::AgcCbSetShRegistersDirectGetSize(3) == 36 && G::AgcCbSetUcRegistersDirectGetSize(3) == 36 && G::AgcCbSetUcRegisterRangeDirectGetSize(3) == 20,
	      "sizes by count");

	const auto bad = static_cast<int>(0x8a6c000cu);
	std::array<uint32_t, 14> branch {};
	branch.fill(0xffffffffu);
	branch[0] = KYTY_PM4(14, Pm4::IT_INDIRECT_BUFFER, 0u);
	Check(G::AgcBranchPatchSetCompareAddress(branch.data(), 0x1234567890ull) == 0 && branch[2] == 0x34567897u && branch[3] == 0x12u, "branch compare address");
	Check(G::AgcBranchPatchSetThenTargetAddress(branch.data(), 0x1234567890ull, 5) == 0 && branch[8] == 0x34567893u && branch[9] == 0x12u && branch[10] == 0xfff00005u,
	      "branch then target");
	Check(G::AgcBranchPatchSetElseTargetAddress(branch.data(), 0x1234567890ull, 6) == 0 && branch[11] == 0x34567893u && branch[12] == 0x12u && branch[13] == 0xfff00006u,
	      "branch else target");
	Check(G::AgcBranchPatchSetElseTarget(branch.data(), 2, 0x1234567890ull, 7) == 0 && branch[13] == 0xeff00007u, "branch else target with its policy");
	branch[0] = KYTY_PM4(14, Pm4::IT_NOP, 0u);
	Check(G::AgcBranchPatchSetCompareAddress(branch.data(), 0) == bad && G::AgcBranchPatchSetElseTarget(branch.data(), 0, 0, 0) == bad, "a branch patch checks the packet");

	std::array<uint32_t, 8> release {};
	release[0] = KYTY_PM4(8, Pm4::IT_RELEASE_MEM, 0u);
	release[1] = 0xffffffffu;
	Check(G::AgcQueueEndOfPipeActionPatchGcrCntl(release.data(), 0xabc) == 0 && release[1] == 0xffabcfffu, "release GCR");
	Check(G::AgcQueueEndOfPipeActionPatchType(release.data(), 0x14) == 0 && release[1] == 0xffabc5d4u, "release type: EVENT_INDEX 5");
	Check(G::AgcQueueEndOfPipeActionPatchType(release.data(), 0x2f) == 0 && release[1] == 0xffabc6efu, "release type: EVENT_INDEX 6 from CS_DONE on");

	std::array<uint32_t, 8> write {};
	write[0] = KYTY_PM4(8, Pm4::IT_WRITE_DATA, 0u);
	Check(G::AgcWriteDataPatchCachePolicy(write.data(), 3) == 0 && write[1] == 3u << 25u, "write policy");
	Check(G::AgcWriteDataPatchDstSel(write.data(), 5) == 0 && write[1] == ((3u << 25u) | (5u << 8u)), "write DST_SEL");
	Check(G::AgcWriteDataPatchDst(write.data(), 5) == 0 && write[1] == ((3u << 25u) | (1u << 30u) | (2u << 8u)), "write destination as the writer packs it");
	Check(G::AgcWriteDataPatchAddress(write.data(), 0x1234567890ull) == 0 && write[2] == 0x34567890u && write[3] == 0x12u, "write address");

	std::array<uint32_t, 4> acquire {};
	acquire[0] = KYTY_PM4(4, Pm4::IT_ACQUIRE_MEM, 0u);
	acquire[1] = 0x12345678u;
	Check(G::AgcAcquireMemSetEngine(acquire.data(), 1) == 0 && acquire[1] == 0x92345678u, "acquire engine");

	const uint32_t* payload = nullptr;
	Check(G::AgcGetDataPacketPayloadStart(&payload, write.data(), 0) == 0 && payload == write.data() + 1, "payload after the header");
	Check(G::AgcGetDataPacketPayloadStart(&payload, write.data(), 1) == 0 && payload == write.data() + 2, "payload after the offset");
}

} // namespace

int main() {
	TestResetTablesCovered();
	TestSizesAndPatches();
	if (g_failures != 0) {
		return 1;
	}
	std::puts("AgcResetTableCpTests: all passed");
	return 0;
}
