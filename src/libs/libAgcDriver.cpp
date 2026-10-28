#include "common/abi.h"
#include "graphics/shader/shader.h" // ShaderRegister, passed by value to wrapped exports
#include "libs/agc.h"
#include "libs/agcDiff.h"
#include "libs/libs.h"
#include "loader/symbolDatabase.h"

namespace Libs {

namespace LibGen5 {

LIB_VERSION("Graphics5", 1, "Graphics5", 1, 1);

namespace Gen5 = Graphics::Gen5;

// Registers the HLE export, or with KYTY_AGC_REFERENCE a wrapper that also runs the real library's
// export and compares (libs/agcDiff.h).
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define AGC_FUNC(n, f)                                                                             \
	if (Gen5::AgcDiff::Requested()) {                                                              \
		auto* wrapped = &Gen5::AgcDiff::Wrap<&f, __LINE__>::Call;                                  \
		Gen5::AgcDiff::Register(reinterpret_cast<const void*>(wrapped), n, #f);                    \
		LIB_ADD(n, wrapped, Loader::SymbolType::Func);                                             \
	} else {                                                                                       \
		LIB_FUNC(n, f);                                                                            \
	}

LIB_DEFINE(InitAgcDriver_1) {
	PRINT_NAME_ENABLE(true);

	AGC_FUNC("23LRUSvYu1M", Gen5::AgcInit);
	AGC_FUNC("2JtWUUiYBXs", Gen5::AgcGetRegisterDefaults2);
	AGC_FUNC("wRbq6ZjNop4", Gen5::AgcGetRegisterDefaults2Internal);
	AGC_FUNC("f3dg2CSgRKY", Gen5::AgcCreateShader);
	AGC_FUNC("dolOmWH+huQ", Gen5::AgcGetFusedShaderSize);
	AGC_FUNC("fd5Bp5tGTgo", Gen5::AgcFuseShaderHalves_0200);
	AGC_FUNC("nApJjpKNBl4", Gen5::AgcFuseShaderHalves);
	AGC_FUNC("vcmNN+AAXnY", Gen5::AgcSetCxRegIndirectPatchSetAddress);
	AGC_FUNC("Qrj4c+61z4A", Gen5::AgcSetShRegIndirectPatchSetAddress);
	AGC_FUNC("6lNcCp+fxi4", Gen5::AgcSetUcRegIndirectPatchSetAddress);
	AGC_FUNC("whb1RL7K4Ss", Gen5::AgcSetCxRegIndirectPatchSetNumRegisters);
	AGC_FUNC("nCUgItdN2ms", Gen5::AgcSetShRegIndirectPatchSetNumRegisters);
	AGC_FUNC("fRG-JOH5+sI", Gen5::AgcSetUcRegIndirectPatchSetNumRegisters);
	AGC_FUNC("d-6uF9sZDIU", Gen5::AgcSetCxRegIndirectPatchAddRegisters);
	AGC_FUNC("z2duB-hHQSM", Gen5::AgcSetShRegIndirectPatchAddRegisters);
	AGC_FUNC("vRoArM9zaIk", Gen5::AgcSetUcRegIndirectPatchAddRegisters);
	AGC_FUNC("D9sr1xGUriE", Gen5::AgcCreatePrimState);
	AGC_FUNC("Y3ymLfZ1384", Gen5::AgcUpdatePrimState);
	AGC_FUNC("NKIzURsgV7I", Gen5::AgcGetGsOversubscription);
	AGC_FUNC("HV4j+E0MBHE", Gen5::AgcCreateInterpolantMapping);
	AGC_FUNC("V++UgBtQhn0", Gen5::AgcGetDataPacketPayloadAddress);
	AGC_FUNC("s+VGAMDQ0AQ", Gen5::AgcGetDataPacketPayloadRange);
	AGC_FUNC("fPSCdQxgpSw", Gen5::AgcWriteDataPatchSetAddressOrOffset);
	AGC_FUNC("2BS4EtAaF28", Gen5::AgcJumpPatchSetTarget);
	AGC_FUNC("h9z6+0hEydk", Gen5::AgcSuspendPoint);
	AGC_FUNC("qj7QZpgr9Uw", Gen5::AgcDcbContextStateOp);
	AGC_FUNC("H6vHS5cidSA", Gen5::AgcDcbContextStateOpGetSize);
	AGC_FUNC("BfBDZGbti7A", Gen5::AgcGetIsTrinityMode);
	AGC_FUNC("dbOlWdppb4o", Gen5::AgcCreateInterpolantMapping2);

	AGC_FUNC("F0ZXt5q0ZTA", Gen5::AgcDriverGetDefaultOwner);
	AGC_FUNC("F0Y42t-3e18", Gen5::AgcDriverInitResourceRegistration);
	AGC_FUNC("AOLcoIkQDgM", Gen5::AgcDriverQueryResourceRegistrationUserMemoryRequirements);
	AGC_FUNC("uJziRsODk1c", Gen5::AgcDriverGetResourceRegistrationMaxNameLength);
	AGC_FUNC("X-Nm5KLREeg", Gen5::AgcDriverRegisterOwner);
	AGC_FUNC("W5z4eZrjEas", Gen5::AgcDriverRegisterResource);
	AGC_FUNC("ZLJk9r2+2Aw", Gen5::AgcDriverUnregisterOwnerAndResources);
	AGC_FUNC("pWLG7WOpVcw", Gen5::AgcDriverUnregisterResource);
	AGC_FUNC("3AyTaWcF-H8", Gen5::AgcDriverRegisterWorkloadStream);

	AGC_FUNC("LtTouSCZjHM", Gen5::AgcCbNop);
	AGC_FUNC("t7PlZ9nt5Lc", Gen5::AgcCbNopGetSize);
	AGC_FUNC("k3GhuSNmBLU", Gen5::AgcCbDispatch);
	AGC_FUNC("Abendgtz+3o", Gen5::AgcCbDispatchGetSize);
	AGC_FUNC("w1KFAHVqpaU", Gen5::AgcCbBranch);
	AGC_FUNC("uZW-mqsxkrM", Gen5::AgcCbBranchGetSize);
	AGC_FUNC("7Wa3aeJgeVU", Gen5::AgcBranchPatchSetThenTarget);
	AGC_FUNC("n2fD4A+pb+g", Gen5::AgcCbSetShRegisterRangeDirect);
	AGC_FUNC("bxGoVxpdSPQ", Gen5::AgcCbSetShRegisterRangeDirectGetSize);
	AGC_FUNC("UZbQjYAwwXM", Gen5::AgcCbSetShRegistersDirect);
	AGC_FUNC("03RZmELWWzw", Gen5::AgcCbSetUcRegistersDirect);
	AGC_FUNC("wr23dPKyWc0", Gen5::AgcCbReleaseMem);
	AGC_FUNC("hL7C0IRpWZI", Gen5::AgcCbQueueEndOfPipeActionGetSize);
	AGC_FUNC("T6xuVw0KUJo", Gen5::AgcDebugRaiseException);
	AGC_FUNC("JrtiDtKeS38", Gen5::AgcAcbResetQueue);
	AGC_FUNC("cFazmnXpJOE", Gen5::AgcAcbEventWrite);
	AGC_FUNC("KT-hTp-Ch14", Gen5::AgcAcbAcquireMem);
	AGC_FUNC("ewobAQeMo5k", Gen5::AgcAcbAcquireMemGetSize);
	AGC_FUNC("qyM2bxYFPAk", Gen5::AgcAcbCondExec);
	AGC_FUNC("ozKzBP4aki4", Gen5::AgcAcbCondExecGetSize);
	AGC_FUNC("e1DFTg+Sd8U", Gen5::AgcAcbJump);
	AGC_FUNC("b-oySn+G2tE", Gen5::AgcAcbJumpGetSize);
	AGC_FUNC("DwICrVxerkY", Gen5::AgcAcbRewind);
	AGC_FUNC("htn36gPnBk4", Gen5::AgcAcbWaitRegMem);
	AGC_FUNC("q4VuU-QsLOE", Gen5::AgcAcbMemSemaphore);
	AGC_FUNC("idlaArvdXEs", Gen5::AgcAcbWaitOnAddressGetSize);
	AGC_FUNC("-RnpfpxIhec", Gen5::AgcAcbDmaData);
	AGC_FUNC("qzMN2XKGA4k", Gen5::AgcAcbCopyData);
	AGC_FUNC("CbQh3DKMSno", Gen5::AgcAcbCopyDataGetSize);
	AGC_FUNC("j3EtxFkSIhQ", Gen5::AgcAcbDispatchIndirect);
	AGC_FUNC("eZ4+17OQz4Q", Gen5::AgcAcbWriteData);
	AGC_FUNC("xAeBOa0A3kk", Gen5::AgcAcbSetMarker);
	AGC_FUNC("cpCILPya5Zk", Gen5::AgcAcbPushMarker);
	AGC_FUNC("6mFxkVqdmbQ", Gen5::AgcAcbPopMarker);
	AGC_FUNC("TRO721eVt4g", Gen5::AgcDcbResetQueue);
	AGC_FUNC("MWiElSNE8j8", Gen5::AgcDcbWaitUntilSafeForRendering);
	AGC_FUNC("LFSPFmGc9Hg", Gen5::AgcDcbSetWorkloadsActive);
	AGC_FUNC("hEK26Wdny6s", Gen5::AgcDcbSetWorkloadComplete);
	AGC_FUNC("QhCbS4X9Rl8", Gen5::AgcDcbSetMarker);
	AGC_FUNC("pFLArOT53+w", Gen5::AgcDcbSetShRegisterDirect);
	AGC_FUNC("LHFXRrlTPD8", Gen5::AgcDcbSetCxRegisterDirect);
	AGC_FUNC("1DeUNpRIDDA", Gen5::AgcDcbSetCxRegisterDirectGetSize);
	AGC_FUNC("w4-d0n60hdo", Gen5::AgcDcbSetUcRegisterDirect);
	AGC_FUNC("ZvwO9euwYzc", Gen5::AgcDcbSetCxRegistersIndirect);
	AGC_FUNC("GBCh3zCihoU", Gen5::AgcDcbSetCxRegistersIndirectGetSize);
	AGC_FUNC("-HOOCn0JY48", Gen5::AgcDcbSetShRegistersIndirect);
	AGC_FUNC("nNlUtdDDvZ0", Gen5::AgcDcbSetShRegistersIndirectGetSize);
	AGC_FUNC("hvUfkUIQcOE", Gen5::AgcDcbSetUcRegistersIndirect);
	AGC_FUNC("GIIW2J37e70", Gen5::AgcDcbSetIndexSize);
	AGC_FUNC("l4fM9K-Lyks", Gen5::AgcDcbSetIndexBuffer);
	AGC_FUNC("8N2tmT3jmC8", Gen5::AgcDcbSetIndexCount);
	AGC_FUNC("tSBxhAPyytQ", Gen5::AgcDcbSetNumInstances);
	AGC_FUNC("6DFuRKT4C9w", Gen5::AgcDcbSetNumInstancesGetSize);
	AGC_FUNC("q88lQ+GP5Yk", Gen5::AgcDcbDrawIndex);
	AGC_FUNC("6ee9Hd3EWXQ", Gen5::AgcDcbDrawIndexGetSize);
	AGC_FUNC("Ikfdt-rIqCE", Gen5::AgcJumpPatchSetTarget_0300);
	AGC_FUNC("Rlx+bykm0r0", Gen5::AgcDcbDrawIndexMultiInstanced);
	AGC_FUNC("mR9j7+SfM34", Gen5::AgcDcbDrawIndexMultiInstancedGetSize);
	AGC_FUNC("Yw0jKSqop+E", Gen5::AgcDcbDrawIndexAuto);
	AGC_FUNC("WrdP9Zxx3lQ", Gen5::AgcDcbDrawIndexAutoGetSize);
	AGC_FUNC("B+aG9DUnTKA", Gen5::AgcDcbDrawIndexOffset);
	AGC_FUNC("qMlfB1ZhMDc", Gen5::AgcDcbDrawIndexOffsetGetSize);
	AGC_FUNC("RmaJwLtc8rY", Gen5::AgcDcbSetBaseIndirectArgs);
	AGC_FUNC("1q1titRBL6o", Gen5::AgcDcbDrawIndirect);
	AGC_FUNC("cxPZ4Wgvdj8", Gen5::AgcDcbDrawIndirectGetSize);
	AGC_FUNC("kUlvghKs-mA", Gen5::AgcDcbDrawIndirectMulti);
	AGC_FUNC("t1vNu082-jM", Gen5::AgcDcbDrawIndexIndirect);
	AGC_FUNC("ypVBz4uPKcQ", Gen5::AgcDcbDrawIndexIndirectMulti);
	AGC_FUNC("CtB+A9-VxO0", Gen5::AgcDcbDispatchIndirect);
	AGC_FUNC("w8HVkEeXPv8", Gen5::AgcDcbDispatchIndirectGetSize);
	AGC_FUNC("aJf+j5yntiU", Gen5::AgcDcbEventWrite);
	AGC_FUNC("C4l9fB17t8w", Gen5::AgcDcbEventWriteGetSize);
	AGC_FUNC("57labkp+rSQ", Gen5::AgcDcbAcquireMem);
	AGC_FUNC("-vnlTPPXPrw", Gen5::AgcDcbAcquireMemGetSize);
	AGC_FUNC("1rZSWUv1IRc", Gen5::AgcDcbCopyData);
	AGC_FUNC("b5u0Jzm8TF8", Gen5::AgcDcbCopyDataGetSize); // SpongeBob PPSA26893
	AGC_FUNC("WmAc2MEj6Io", Gen5::AgcDcbDmaData);
	AGC_FUNC("xSAR0LTcRKM", Gen5::AgcDcbJump);
	AGC_FUNC("VEGu4dixjUg", Gen5::AgcDcbJumpGetSize);
	AGC_FUNC("QIXCsbipds0", Gen5::AgcDcbRewindGetSize);
	AGC_FUNC("zfcxg-ewMK8", Gen5::AgcDcbRewind);
	AGC_FUNC("BIPexNBSGog", Gen5::AgcDcbCondExec);
	AGC_FUNC("ou16V5hh5sg", Gen5::AgcDcbCondExecGetSize);
	AGC_FUNC("bbFueFP+J4k", Gen5::AgcDcbSetPredication);
	AGC_FUNC("-KRzWekV120", Gen5::AgcDcbSetIndexSize_0600);
	AGC_FUNC("IxYiarKlXxM", Gen5::AgcDmaDataPatchSetDstAddressOrOffset);
	AGC_FUNC("cdDRpqcFGbU", Gen5::AgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate);
	AGC_FUNC("Lkf86B98qPc", Gen5::AgcGetPacketSize);
	AGC_FUNC("w6Dj1VJt5qY", Gen5::AgcSetPacketPredication);
	AGC_FUNC("n8vgpaQg6dA", Gen5::AgcSetRangePredication);
	AGC_FUNC("K2mciNVxUCE", Gen5::AgcSetNop);
	AGC_FUNC("i1jyy49AjXU", Gen5::AgcDcbWriteData);
	AGC_FUNC("p9tI+yTvx68", Gen5::AgcDcbWriteDataGetSize);
	AGC_FUNC("vuSXe69VILM", Gen5::AgcDcbGetLodStats);
	AGC_FUNC("VmW0Tdpy420", Gen5::AgcDcbWaitRegMem);
	AGC_FUNC("43WJ08sSugE", Gen5::AgcDcbWaitOnAddressGetSize);
	AGC_FUNC("+kSrjIVxKFE", Gen5::AgcDcbPushMarker);
	AGC_FUNC("H7uZqCoNuWk", Gen5::AgcDcbPopMarker);
	AGC_FUNC("u2T2DiA5hRI", Gen5::AgcDcbStallCommandBufferParser);
	AGC_FUNC("3KDcnM3lrcU", Gen5::AgcWaitRegMemPatchAddress);
	AGC_FUNC("7nOoijNPvEU", Gen5::AgcWaitRegMemPatchReference);
	AGC_FUNC("0fWWK5uG9rQ", Gen5::AgcQueueEndOfPipeActionPatchAddress);
	AGC_FUNC("MlEw1feXcjg", Gen5::AgcQueueEndOfPipeActionPatchData);
	AGC_FUNC("ORWsxIbk4TE", Gen5::AgcCondExecPatchSetEnd);
	AGC_FUNC("YWTKOju587o", Gen5::AgcCondExecPatchSetCommandAddress);
	AGC_FUNC("k-JpyR2dYAM", Gen5::AgcCondExecPatchSetEnd);            // sceAgcAsyncCondExecPatchSetEnd
	AGC_FUNC("3ZWa3AoyWZQ", Gen5::AgcCondExecPatchSetCommandAddress); // sceAgcAsyncCondExecPatchSetCommandAddress
	AGC_FUNC("ziVA3whp3p4", Gen5::AgcRewindPatchSetRewindState);
	AGC_FUNC("eWaWyFegzgQ", Gen5::AgcRewindPatchSetRewindState);      // sceAgcAsyncRewindPatchSetRewindState
	AGC_FUNC("YUeqkyT7mEQ", Gen5::AgcDcbSetFlip);
}

} // namespace LibGen5

namespace LibGen5Driver {

LIB_VERSION("Graphics5Driver", 1, "Graphics5Driver", 1, 1);

namespace Gen5Driver = Graphics::Gen5Driver;

LIB_DEFINE(InitAgcDriver_1) {
	PRINT_NAME_ENABLE(true);

	LIB_FUNC("UglJIZjGssM", Gen5Driver::AgcDriverSubmitDcb);
	LIB_FUNC("AhGvpITrf4M", Gen5Driver::AgcDriverSubmitDcb);       // sceAgcDriverAgrSubmitDcb
	LIB_FUNC("6UzEidRZwkg", Gen5Driver::AgcDriverSubmitMultiDcbs);
	LIB_FUNC("+T8Xo6LtFJI", Gen5Driver::AgcDriverSubmitMultiDcbs); // sceAgcDriverAgrSubmitMultiDcbs
	LIB_FUNC("b4fpgH5ZXxQ", Gen5Driver::AgcDriverSubmitCommandBuffer);
	LIB_FUNC("Fj7r9EHzF38", Gen5Driver::AgcDriverSubmitMultiCommandBuffers);
	LIB_FUNC("gSRnr79F8tQ", Gen5Driver::AgcDriverSubmitAcb);
	LIB_FUNC("HF3YllT3mXU", Gen5Driver::AgcDriverSubmitMultiAcbs);
	LIB_FUNC("w2rJhmD+dsE", Gen5Driver::AgcDriverAddEqEvent);
	LIB_FUNC("DL2RXaXOy88", Gen5Driver::AgcDriverDeleteEqEvent);
	LIB_FUNC("5CdQTZIQPxM", Gen5Driver::AgcDriverGetEqEventType);
	LIB_FUNC("Zw7uUVPulbw", Gen5Driver::AgcDriverGetEqContextId);
	LIB_FUNC("XlNp7jzGiPo", Gen5Driver::AgcDriverSetTFRing);
	LIB_FUNC("MM4IZSEYytQ", Gen5Driver::AgcDriverSetHsOffchipParam);
	LIB_FUNC("Ddwk4gLT5j0", Gen5Driver::AgcDriverIsCaptureInProgress);
	LIB_FUNC("U9ueyEhSkF4", Gen5Driver::AgcDriverRegisterDefaultOwner);
	LIB_FUNC("Um-jkyDy9rI", Gen5Driver::AgcDriverGetSharedArena);
}

} // namespace LibGen5Driver

LIB_DEFINE(InitAgcDriver_1) {
	LibGen5::InitAgcDriver_1(s);
	LibGen5Driver::InitAgcDriver_1(s);
}

} // namespace Libs
