#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "libs/ajm/aac_decoder.h"
#include "libs/ajm/atrac9_decoder.h"
#include "libs/ajm/decoder.h"
#include "libs/ajm/mp3_decoder.h"
#include "libs/ajm/opus_decoder.h"
#include "libs/ajm/resampler.h"
#include "kernel/longWait.h"
#include "libs/audio.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
#include <pthread.h>
#endif

namespace Libs::Audio::Ajm {

LIB_NAME("Ajm", "Ajm");

struct AjmBatchInfo {
	void*       buffer;
	size_t      offset;
	size_t      size;
	void*       last_good_job;
	const void* last_good_job_ra;
};

struct AjmBatchError {
	int         error_code;
	const void* job_addr;
	uint32_t    cmd_offset;
	const void* job_ra;
};

struct AjmBuffer {
	void*  address;
	size_t size;
};

struct AjmSidebandResult {
	int32_t result;
	int32_t internal_result;
};

struct AjmSidebandStream {
	int32_t  size_consumed;
	int32_t  size_produced;
	uint64_t total_decoded_samples;
};

struct AjmSidebandMFrame {
	uint32_t num_frames;
	uint32_t reserved;
};

struct AjmGetStatisticsResult {
	AjmSidebandResult result;
	float             engine_usage_batch;
	float             engine_usage_interval[3];
	uint32_t          memory[6];
};

enum class AjmCodec : uint32_t {
	DecMp3           = 0,
	DecAt9           = 1,
	DecM4aac         = 2,
	DecUnknownNewABI = 14,
	DecOpus          = 24,
};

constexpr int32_t AJM_ERROR_INVALID_CONTEXT     = static_cast<int32_t>(0x80930002u);
constexpr int32_t AJM_ERROR_INVALID_PARAMETER   = static_cast<int32_t>(0x80930005u);
constexpr int32_t AJM_ERROR_OUT_OF_RESOURCES    = static_cast<int32_t>(0x80930007u);
constexpr int32_t AJM_ERROR_CODEC_NOT_SUPPORTED = static_cast<int32_t>(0x80930008u);
constexpr int32_t AJM_ERROR_JOB_CREATION        = static_cast<int32_t>(0x80930012u);
constexpr size_t  AJM_JOB_CONTROL_SIZE          = 48;
constexpr size_t  AJM_JOB_RUN_SIZE              = 64;
constexpr size_t  AJM_JOB_GET_STATISTICS_SIZE =
    sizeof(uint32_t) + sizeof(float) * 3 + 8 + AJM_JOB_RUN_SIZE;
constexpr size_t AJM_JOB_SET_RESAMPLE_PARAMETERS_SIZE =
    sizeof(float) * 2 + sizeof(uint32_t) + 12 + AJM_JOB_CONTROL_SIZE;
constexpr size_t   AJM_SIDEBAND_RESULT_SIZE           = sizeof(AjmSidebandResult);
constexpr size_t   AJM_SIDEBAND_STREAM_SIZE           = sizeof(AjmSidebandStream);
constexpr size_t   AJM_SIDEBAND_MFRAME_SIZE           = sizeof(AjmSidebandMFrame);
constexpr size_t   AJM_SIDEBAND_FORMAT_SIZE           = sizeof(AjmSidebandFormat);
constexpr size_t   AJM_SIDEBAND_GAPLESS_SIZE          = sizeof(AjmSidebandGaplessDecode);
constexpr size_t   AJM_SIDEBAND_RESAMPLE_INFO_SIZE    = sizeof(AjmSidebandResampleInfo);
constexpr uint64_t AJM_FLAG_RUN_GET_CODEC_INFO        = 1ull << 11u;
constexpr uint64_t AJM_FLAG_RUN_MULTIPLE_FRAMES       = 1ull << 12u;
constexpr uint64_t AJM_FLAG_CONTROL_RESET             = 1ull << 13u;
constexpr uint64_t AJM_FLAG_CONTROL_INITIALIZE        = 1ull << 14u;
constexpr uint64_t AJM_FLAG_CONTROL_RESAMPLE          = 1ull << 15u;
constexpr uint64_t AJM_FLAG_CONTROL_RESAMPLE_EX       = 1ull << 16u;
constexpr uint64_t AJM_FLAG_SIDEBAND_RESAMPLE_INFO    = 1ull << 44u;
constexpr uint64_t AJM_FLAG_SIDEBAND_GAPLESS          = 1ull << 45u;
constexpr uint64_t AJM_FLAG_SIDEBAND_FORMAT           = 1ull << 46u;
constexpr uint64_t AJM_FLAG_SIDEBAND_STREAM           = 1ull << 47u;
constexpr uint64_t AJM_INSTANCE_FLAG_FORMAT_OFFSET    = 7u;
constexpr uint64_t AJM_INSTANCE_FLAG_FORMAT_MASK      = 0x7u;
constexpr uint64_t AJM_INSTANCE_FLAG_MAX_CHANNEL_MASK = 0x7fu;

// An instance id is the codec above a 14-bit index; the index is unique among live instances.
constexpr uint32_t AJM_INSTANCE_INDEX_BITS = 14u;
constexpr uint32_t AJM_INSTANCE_INDEX_MASK = (1u << AJM_INSTANCE_INDEX_BITS) - 1u;
// Sideband inputs are copied when a job is built; codec parameters are a few words.
constexpr size_t AJM_MAX_SIDEBAND_INPUT_SIZE = 256;

static std::atomic_uint32_t g_ajm_next_batch {1};

static uint32_t AjmGetFlagChannelCount(uint64_t flags) {
	return static_cast<uint32_t>(flags & AJM_INSTANCE_FLAG_MAX_CHANNEL_MASK);
}

static AjmSampleEncoding AjmGetFlagSampleEncoding(uint64_t flags) {
	const auto encoding = static_cast<uint32_t>((flags >> AJM_INSTANCE_FLAG_FORMAT_OFFSET) &
	                                            AJM_INSTANCE_FLAG_FORMAT_MASK);
	switch (encoding) {
		case static_cast<uint32_t>(AjmSampleEncoding::S16): return AjmSampleEncoding::S16;
		case static_cast<uint32_t>(AjmSampleEncoding::S32): return AjmSampleEncoding::S32;
		case static_cast<uint32_t>(AjmSampleEncoding::Float): return AjmSampleEncoding::Float;
		default: EXIT("unsupported AJM PCM sample encoding %u\n", encoding);
	}
	return AjmSampleEncoding::S16;
}

static const char* AjmCodecName(uint32_t codec) {
	switch (codec) {
		case static_cast<uint32_t>(AjmCodec::DecMp3): return "MP3 decoder";
		case static_cast<uint32_t>(AjmCodec::DecAt9): return "ATRAC9 decoder";
		case static_cast<uint32_t>(AjmCodec::DecM4aac): return "MPEG4-AAC decoder";
		case 3: return "CELP8 decoder";
		case 4: return "CELP8 encoder";
		case 12: return "CELP16 decoder";
		case 13: return "CELP16 encoder";
		case 22: return "HE-VAG decoder";
		case 23: return "LPCM decoder";
		case static_cast<uint32_t>(AjmCodec::DecOpus): return "Opus decoder";
		default: return "unknown";
	}
}

static_assert(sizeof(AjmDecAt9ConfigDataInfo) == 20);

int KYTY_SYSV_ABI AjmDecAt9ParseConfigData(const void*              config_data,
                                           AjmDecAt9ConfigDataInfo* config_info) {
	PRINT_NAME();
	if (config_data == nullptr || config_info == nullptr) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	Atrac9CodecInfo codec_info {};
	if (!AjmAt9GetCodecInfo(static_cast<const uint8_t*>(config_data), codec_info)) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	config_info->channels                  = static_cast<uint32_t>(codec_info.channels);
	config_info->sample_rate               = static_cast<uint32_t>(codec_info.samplingRate);
	config_info->frame_samples_per_channel = static_cast<uint32_t>(codec_info.frameSamples);
	config_info->superframe_samples_per_channel =
	    static_cast<uint32_t>(codec_info.frameSamples) *
	    static_cast<uint32_t>(codec_info.framesInSuperframe);
	config_info->superframe_size = static_cast<uint32_t>(codec_info.superframeSize);
	return OK;
}

static_assert(sizeof(AjmDecMp3ParseFrameInfo) == 40);

int KYTY_SYSV_ABI AjmDecMp3ParseFrame(const void* bitstream, size_t size, int parse_ofl,
                                      AjmDecMp3ParseFrameInfo* frame_info) {
	PRINT_NAME();
	LOGF("\t size = %zu, parse_ofl = %d\n", size, parse_ofl);
	if (frame_info == nullptr) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	AjmMp3FrameHeader header {};
	if (!AjmParseMp3Header(static_cast<const uint8_t*>(bitstream), size, &header)) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	*frame_info = {header.frame_size, header.num_channels, header.samples_per_channel,
	               header.bitrate, header.sample_rate};
	return OK;
}

struct AjmInstanceState {
	uint32_t                    context = 0;
	uint32_t                    codec   = 0;
	uint64_t                    flags   = 0;
	AjmGaplessState             gapless;
	std::unique_ptr<AjmDecoder> decoder;
	AjmResampler                resampler;

	// The instance's jobs run in the order their batches were started: a started batch takes a
	// ticket (under the job queue lock) and runs when the earlier tickets have been served.
	uint64_t                next_ticket = 0;
	std::mutex              order_mutex;
	std::condition_variable order_cv;
	uint64_t                serving = 0;
};

using AjmInstancePtr = std::shared_ptr<AjmInstanceState>;

static std::mutex                                   g_ajm_instances_mutex;
static std::unordered_map<uint32_t, AjmInstancePtr> g_ajm_instances;
static std::bitset<AJM_INSTANCE_INDEX_MASK + 1>     g_ajm_used_indices;

static bool AjmCodecIsSupported(uint32_t codec) {
	return codec == static_cast<uint32_t>(AjmCodec::DecMp3) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecAt9) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecM4aac) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecOpus) ||
	       codec == static_cast<uint32_t>(AjmCodec::DecUnknownNewABI);
}

static bool AjmLogCodecSupport(uint32_t codec) {
	const auto supported = AjmCodecIsSupported(codec);
	if (supported) {
		static std::atomic_bool logged {false};
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			std::printf("AJM codec=%u, name=%s\n", codec, AjmCodecName(codec));
		}
	} else {
		std::printf("AJM codec=%u, name=%s\n", codec, AjmCodecName(codec));
		std::printf("AJM unsupported codec=%u (%s)\n", codec, AjmCodecName(codec));
	}
	return supported;
}

static std::unique_ptr<AjmDecoder> AjmCreateDecoder(uint32_t codec, uint64_t flags) {
	if (!AjmCodecIsSupported(codec)) {
		return {};
	}

	auto channels = AjmGetFlagChannelCount(flags);
	if (channels == 0) {
		channels = 2;
	}

	const auto encoding = AjmGetFlagSampleEncoding(flags);
	switch (codec) {
		case static_cast<uint32_t>(AjmCodec::DecMp3):
			return std::make_unique<AjmMp3Decoder>(channels, 48000, encoding, flags);
		case static_cast<uint32_t>(AjmCodec::DecAt9):
			return std::make_unique<AjmAt9Decoder>(channels, 48000, encoding, flags);
		case static_cast<uint32_t>(AjmCodec::DecM4aac):
			return std::make_unique<AjmAacDecoder>(channels, 48000, encoding, flags);
		case static_cast<uint32_t>(AjmCodec::DecOpus):
			return std::make_unique<AjmOpusDecoder>(channels, encoding);
		default: break;
	}

	return {};
}

static uint8_t* AjmReserveJob(AjmBatchInfo* info, size_t job_size) {
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	if (info->offset + job_size > info->size) {
		return nullptr;
	}

	auto* job              = static_cast<uint8_t*>(info->buffer) + info->offset;
	info->last_good_job    = job;
	info->last_good_job_ra = nullptr;
	info->offset += job_size;

	std::memset(job, 0, job_size);

	return job;
}

static int AjmAppendJob(AjmBatchInfo* info, size_t job_size, const char* name) {
	auto* job = AjmReserveJob(info, job_size);
	if (job == nullptr) {
		LOGF("\t %s: not enough batch space, offset=0x%016" PRIx64 ", size=0x%016" PRIx64
		     ", job=0x%016" PRIx64 "\n",
		     name, static_cast<uint64_t>(info->offset), static_cast<uint64_t>(info->size),
		     static_cast<uint64_t>(job_size));
		return AJM_ERROR_JOB_CREATION;
	}

	return OK;
}

static int32_t AjmClampInt32(size_t value) {
	return static_cast<int32_t>(
	    std::min<size_t>(value, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
}

static void AjmWriteResult(void* result, size_t size, const AjmDecodeResult& decode_result = {}) {
	if (result == nullptr || size == 0) {
		return;
	}

	std::memset(result, 0, size);

	if (size >= AJM_SIDEBAND_RESULT_SIZE) {
		auto* sideband            = static_cast<AjmSidebandResult*>(result);
		sideband->result          = decode_result.result;
		sideband->internal_result = decode_result.internal_result;
	}
}

static void AjmWriteStreamResult(void* result, size_t result_size,
                                 const AjmDecodeResult& decode_result, bool multiple_frames) {
	if (result == nullptr) {
		return;
	}

	AjmWriteResult(result, result_size, decode_result);

	if (result_size >= AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE) {
		auto* stream          = reinterpret_cast<AjmSidebandStream*>(static_cast<uint8_t*>(result) +
		                                                             AJM_SIDEBAND_RESULT_SIZE);
		stream->size_consumed = AjmClampInt32(decode_result.input_consumed);
		stream->size_produced = AjmClampInt32(decode_result.output_written);
		stream->total_decoded_samples = decode_result.total_decoded_samples;
	}

	if (multiple_frames && result_size >= AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE +
	                                          AJM_SIDEBAND_MFRAME_SIZE) {
		auto* mframe = reinterpret_cast<AjmSidebandMFrame*>(
		    static_cast<uint8_t*>(result) + AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE);
		mframe->num_frames = decode_result.frames;
	}
}

static void AjmWriteFormat(void* result, size_t result_size, const AjmSidebandFormat& format,
                           const AjmDecodeResult& decode_result = {}) {
	if (result == nullptr || result_size < AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_FORMAT_SIZE) {
		AjmWriteResult(result, result_size, decode_result);
		return;
	}

	AjmWriteResult(result, result_size, decode_result);

	auto* format_output = reinterpret_cast<AjmSidebandFormat*>(static_cast<uint8_t*>(result) +
	                                                           AJM_SIDEBAND_RESULT_SIZE);
	*format_output      = format;
}

static void AjmWriteCodecInfo(AjmDecoder* decoder, void* result, size_t result_size,
                              const AjmDecodeResult& decode_result) {
	if (decoder == nullptr || result == nullptr || result_size <= AJM_SIDEBAND_RESULT_SIZE) {
		return;
	}
	decoder->WriteCodecInfo(static_cast<uint8_t*>(result) + AJM_SIDEBAND_RESULT_SIZE,
	                        result_size - AJM_SIDEBAND_RESULT_SIZE, decode_result);
}

static void AjmWriteSideband(uint64_t flags, void* sideband_output, size_t sideband_output_size,
                             AjmDecoder* decoder, const AjmGaplessState* gapless,
                             const AjmDecodeResult& decode_result) {
	if (sideband_output == nullptr || sideband_output_size == 0) {
		return;
	}

	AjmWriteResult(sideband_output, sideband_output_size, decode_result);

	size_t offset = AJM_SIDEBAND_RESULT_SIZE;

	if ((flags & AJM_FLAG_RUN_GET_CODEC_INFO) != 0 && decoder != nullptr) {
		const auto codec_info_size = decoder->CodecInfoSize();
		if (offset + codec_info_size <= sideband_output_size) {
			decoder->WriteCodecInfo(static_cast<uint8_t*>(sideband_output) + offset,
			                        sideband_output_size - offset, decode_result);
			offset += codec_info_size;
		}
	}

	if ((flags & AJM_FLAG_SIDEBAND_RESAMPLE_INFO) != 0 &&
	    offset + AJM_SIDEBAND_RESAMPLE_INFO_SIZE <= sideband_output_size) {
		auto* info = reinterpret_cast<AjmSidebandResampleInfo*>(
		    static_cast<uint8_t*>(sideband_output) + offset);
		*info = decode_result.resample_info;
		offset += AJM_SIDEBAND_RESAMPLE_INFO_SIZE;
	}

	if ((flags & AJM_FLAG_SIDEBAND_GAPLESS) != 0 &&
	    offset + AJM_SIDEBAND_GAPLESS_SIZE <= sideband_output_size) {
		auto* gapless_output = reinterpret_cast<AjmSidebandGaplessDecode*>(
		    static_cast<uint8_t*>(sideband_output) + offset);
		if (gapless != nullptr) {
			*gapless_output = gapless->current;
		}
		offset += AJM_SIDEBAND_GAPLESS_SIZE;
	}

	if ((flags & AJM_FLAG_SIDEBAND_FORMAT) != 0 &&
	    offset + AJM_SIDEBAND_FORMAT_SIZE <= sideband_output_size) {
		auto* format =
		    reinterpret_cast<AjmSidebandFormat*>(static_cast<uint8_t*>(sideband_output) + offset);
		*format = decode_result.format;
		offset += AJM_SIDEBAND_FORMAT_SIZE;
	}

	if ((flags & AJM_FLAG_SIDEBAND_STREAM) != 0 &&
	    offset + AJM_SIDEBAND_STREAM_SIZE <= sideband_output_size) {
		auto* stream =
		    reinterpret_cast<AjmSidebandStream*>(static_cast<uint8_t*>(sideband_output) + offset);
		stream->size_consumed         = AjmClampInt32(decode_result.input_consumed);
		stream->size_produced         = AjmClampInt32(decode_result.output_written);
		stream->total_decoded_samples = decode_result.total_decoded_samples;
		offset += AJM_SIDEBAND_STREAM_SIZE;
	}

	if ((flags & AJM_FLAG_RUN_MULTIPLE_FRAMES) != 0 &&
	    offset + AJM_SIDEBAND_MFRAME_SIZE <= sideband_output_size) {
		auto* mframe =
		    reinterpret_cast<AjmSidebandMFrame*>(static_cast<uint8_t*>(sideband_output) + offset);
		mframe->num_frames = decode_result.frames;
	}
}

static bool AjmCodecIsValid(uint32_t codec) {
	switch (codec) {
		case static_cast<uint32_t>(AjmCodec::DecMp3):
		case static_cast<uint32_t>(AjmCodec::DecAt9):
		case static_cast<uint32_t>(AjmCodec::DecM4aac):
		case 3:
		case 4:
		case 12:
		case 13:
		case static_cast<uint32_t>(AjmCodec::DecUnknownNewABI):
		case 22:
		case 23:
		case static_cast<uint32_t>(AjmCodec::DecOpus): return true;
		default: return false;
	}
}

// Results a title can stop a voice on. Each kind is logged a few times, with where the stream stood.
static void AjmLogDecodeFailure(uint32_t instance, const AjmDecodeResult& result, const void* input,
                                size_t input_size, size_t output_size) {
	constexpr int32_t BENIGN = AJM_RESULT_PARTIAL_INPUT | AJM_RESULT_NOT_ENOUGH_ROOM;
	if ((result.result & ~BENIGN) == 0) {
		return;
	}
	static std::atomic_uint32_t logged {0};
	if (logged.fetch_add(1) >= 32) {
		return;
	}
	std::string head;
	const auto* bytes = static_cast<const uint8_t*>(input);
	for (size_t i = 0; bytes != nullptr && i < std::min<size_t>(input_size, 16); i++) {
		head += fmt::sprintf("%02x", bytes[i]);
	}
	Log::WriteToConsoleAndLog(fmt::sprintf(
	    "AJM: instance %u: result 0x%08x, internal %d, input %zu consumed %zu, output %zu written %zu, input starts %s\n",
	    instance, static_cast<uint32_t>(result.result), result.internal_result, input_size,
	    static_cast<size_t>(result.input_consumed), output_size, static_cast<size_t>(result.output_written), head));
}

static AjmDecodeResult AjmInvalidInstanceResult() {
	AjmDecodeResult result {};
	result.result = AJM_RESULT_INVALID_PARAMETER;
	return result;
}

static bool AjmIsValid(const AjmInstanceState* state) {
	return state != nullptr && state->decoder != nullptr;
}

static AjmDecodeResult AjmDecodeInstance(uint32_t instance, AjmInstanceState* state, const void* input,
                                         size_t input_size, void* output, size_t output_size,
                                         bool multiple_frames) {
	if (!AjmIsValid(state)) {
		return AjmInvalidInstanceResult();
	}

	// The resampler requires one complete decoded packet per decoder call.
	if (state->codec == static_cast<uint32_t>(AjmCodec::DecOpus) &&
	    (state->flags & AJM_INSTANCE_FLAG_RESAMPLE) != 0) {
		auto resampled = state->resampler.Decode(*state->decoder, state->flags, input, input_size, output,
		                                         output_size, multiple_frames, &state->gapless);
		AjmLogDecodeFailure(instance, resampled, input, input_size, output_size);
		return resampled;
	}
	auto decode_result = state->decoder->Decode(input, input_size, output, output_size, multiple_frames,
	                              &state->gapless);
	AjmLogDecodeFailure(instance, decode_result, input, input_size, output_size);
	return decode_result;
}

static AjmDecodeResult AjmDecodeSplitInstance(uint32_t instance, AjmInstanceState* state,
                                              const std::vector<AjmBuffer>& input_buffers,
                                              const std::vector<AjmBuffer>& output_buffers,
                                              bool                          multiple_frames) {
	// One buffer each way needs no gathering: decoders write only the bytes they report.
	if (input_buffers.size() == 1 && output_buffers.size() == 1 && input_buffers[0].address != nullptr &&
	    output_buffers[0].address != nullptr) {
		return AjmDecodeInstance(instance, state, input_buffers[0].address, input_buffers[0].size,
		                         output_buffers[0].address, output_buffers[0].size, multiple_frames);
	}

	size_t input_size  = 0;
	size_t output_size = 0;
	for (const auto& buffer: input_buffers) {
		input_size += buffer.size;
	}
	for (const auto& buffer: output_buffers) {
		output_size += buffer.size;
	}

	thread_local std::vector<uint8_t> input;
	thread_local std::vector<uint8_t> output;
	input.resize(input_size);
	output.resize(output_size);
	size_t input_offset = 0;
	for (const auto& buffer: input_buffers) {
		if (buffer.address != nullptr && buffer.size != 0) {
			std::memcpy(input.data() + input_offset, buffer.address, buffer.size);
		} else if (buffer.size != 0) {
			std::memset(input.data() + input_offset, 0, buffer.size);
		}
		input_offset += buffer.size;
	}

	auto result = AjmDecodeInstance(instance, state, input.data(), input.size(), output.data(),
	                                output.size(), multiple_frames);

	size_t output_offset = 0;
	for (const auto& buffer: output_buffers) {
		const auto bytes = std::min(
		    buffer.size,
		    (result.output_written > output_offset ? result.output_written - output_offset : 0));
		if (buffer.address != nullptr && bytes != 0) {
			std::memcpy(buffer.address, output.data() + output_offset, bytes);
		}
		output_offset += buffer.size;
	}

	return result;
}

static AjmDecodeResult AjmInitializeInstance(AjmInstanceState* state, const void* codec_parameters,
                                             size_t codec_parameters_size) {
	if (!AjmIsValid(state)) {
		return AjmInvalidInstanceResult();
	}

	state->gapless.Reset();
	return state->decoder->Initialize(codec_parameters, codec_parameters_size);
}

static AjmDecodeResult AjmControlInstance(AjmInstanceState* state, uint64_t flags,
                                          const void* sideband_input, size_t sideband_input_size) {
	if (!AjmIsValid(state)) {
		return AjmInvalidInstanceResult();
	}

	auto* decoder = state->decoder.get();

	AjmDecodeResult result = decoder->MakeResult();
	if ((flags & AJM_FLAG_SIDEBAND_GAPLESS) != 0) {
		if (sideband_input == nullptr || sideband_input_size < AJM_SIDEBAND_GAPLESS_SIZE) {
			result.result = AJM_RESULT_INVALID_PARAMETER;
			return result;
		}
		state->gapless.Set(*static_cast<const AjmSidebandGaplessDecode*>(sideband_input),
		                   (flags & AJM_FLAG_CONTROL_RESET) != 0);
		sideband_input = static_cast<const uint8_t*>(sideband_input) + AJM_SIDEBAND_GAPLESS_SIZE;
		sideband_input_size -= AJM_SIDEBAND_GAPLESS_SIZE;
	}
	for (const auto flag : {AJM_FLAG_CONTROL_RESAMPLE, AJM_FLAG_CONTROL_RESAMPLE_EX}) {
		if ((flags & flag) == 0) {
			continue;
		}
		const bool   extended = flag == AJM_FLAG_CONTROL_RESAMPLE_EX;
		const size_t size     = extended ? 12 : 8;
		if (sideband_input == nullptr || sideband_input_size < size ||
		    (state->flags & AJM_INSTANCE_FLAG_RESAMPLE) == 0) {
			result.result = AJM_RESULT_INVALID_PARAMETER;
			return result;
		}
		float       ratio, change = 0;
		uint32_t    resample_flags;
		const auto* data = static_cast<const uint8_t*>(sideband_input);
		std::memcpy(&ratio, data, sizeof(ratio));
		if (extended) {
			std::memcpy(&change, data + sizeof(ratio), sizeof(change));
		}
		std::memcpy(&resample_flags, data + size - sizeof(resample_flags), sizeof(resample_flags));
		result.result = state->resampler.SetParameters(ratio, change, resample_flags);
		if (result.result != OK) {
			return result;
		}
		sideband_input = data + size;
		sideband_input_size -= size;
	}
	if ((flags & AJM_FLAG_CONTROL_RESET) != 0) {
		state->gapless.Reset();
		decoder->Reset();
		result = decoder->MakeResult();
	}
	if ((flags & AJM_FLAG_CONTROL_INITIALIZE) != 0) {
		state->gapless.Reset();
		result = decoder->Initialize(sideband_input, sideband_input_size);
	}

	result.resample_info = state->resampler.GetInfo();
	return result;
}

static AjmDecodeResult AjmGetGaplessDecode(AjmInstanceState*         state,
                                           AjmSidebandGaplessDecode* gapless_decode) {
	if (!AjmIsValid(state)) {
		return AjmInvalidInstanceResult();
	}

	if (gapless_decode != nullptr) {
		*gapless_decode = state->gapless.current;
	}
	return state->decoder->MakeResult();
}

static AjmSidebandFormat AjmGetInstanceFormat(AjmInstanceState* state, AjmDecodeResult* out_result) {
	if (!AjmIsValid(state)) {
		if (out_result != nullptr) {
			out_result->result = AJM_RESULT_INVALID_PARAMETER;
		}
		return AjmMakeFormat(2, 48000, AjmSampleEncoding::S16);
	}

	if (out_result != nullptr) {
		*out_result               = state->decoder->MakeResult();
		out_result->format        = state->decoder->GetFormat();
		out_result->resample_info = state->resampler.GetInfo();
	}
	return state->decoder->GetFormat();
}

static AjmDecoder* AjmGetDecoder(AjmInstanceState* state) {
	return state != nullptr ? state->decoder.get() : nullptr;
}

static AjmGaplessState* AjmGetGapless(AjmInstanceState* state) {
	return AjmIsValid(state) ? &state->gapless : nullptr;
}

static std::vector<uint8_t> AjmCopySidebandInput(const void* input, size_t size) {
	if (input == nullptr) {
		return {};
	}
	const auto* bytes = static_cast<const uint8_t*>(input);
	return {bytes, bytes + std::min(size, AJM_MAX_SIDEBAND_INPUT_SIZE)};
}

static std::vector<AjmBuffer> AjmCopyBuffers(const AjmBuffer* buffers, size_t count) {
	if (buffers == nullptr) {
		return {};
	}
	return {buffers, buffers + count};
}

// Batches. A job is recorded when it is built and runs once its batch is started, as the
// hardware runs it: on the instance's state, or on nullptr for an instance that does not exist.
// Instances are independent, so a batch runs as one chain of jobs per instance; the chains of a
// batch run in parallel on the AJM workers, each instance's chains in start order, and
// sceAjmBatchWait returns once all chains of the batch are done.
// KYTY_AJM_WORKERS=<n> sets the number of workers (default 4); 0 runs batches when they start,
// on the starting thread.
struct AjmJob {
	uint32_t                               instance = 0;
	size_t                                 end      = 0;
	std::function<void(AjmInstanceState*)> run;
};

struct AjmBatchState {
	std::atomic_uint32_t    pending {0};
	std::mutex              mutex;
	std::condition_variable done_cv;

	void Finish() {
		if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
			std::scoped_lock lock(mutex);
			done_cv.notify_all();
		}
	}

	void Wait() {
		std::unique_lock lock(mutex);
		done_cv.wait(lock, [this] { return pending.load(std::memory_order_acquire) == 0; });
	}
};

struct AjmChain {
	uint32_t                       instance = 0;
	AjmInstancePtr                 state;
	uint64_t                       ticket = 0;
	std::vector<AjmJob>            jobs;
	std::shared_ptr<AjmBatchState> batch;
};

static std::mutex                                                g_ajm_pending_mutex;
static std::unordered_map<const void*, std::vector<AjmJob>>      g_ajm_pending;
static std::mutex                                                g_ajm_batches_mutex;
static std::unordered_map<uint32_t, std::shared_ptr<AjmBatchState>> g_ajm_batches;

// The workers never exit, so their queue is never destroyed.
struct AjmQueue {
	std::mutex              mutex;
	std::condition_variable cv;
	std::deque<AjmChain>    chains;
	int                     worker_count    = -1;
	int                     workers_started = 0;
};

static AjmQueue& AjmGetQueue() {
	static auto* queue = new AjmQueue;
	return *queue;
}

static void AjmRunChain(AjmChain& chain) {
	auto& state = *chain.state;
	{
		std::unique_lock lock(state.order_mutex);
		state.order_cv.wait(lock, [&] { return state.serving == chain.ticket; });
	}
	for (auto& job: chain.jobs) {
		job.run(&state);
	}
	{
		std::scoped_lock lock(state.order_mutex);
		state.serving++;
	}
	state.order_cv.notify_all();
	chain.batch->Finish();
}

static void AjmWorkerMain() {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	pthread_setname_np(pthread_self(), "AjmWorker");
#endif
	// Chains come in bursts; at a low P-state a worker would decode up to twice as slowly.
	Common::Thread::RunAtTopClock("KYTY_UCLAMP");
	auto& queue = AjmGetQueue();
	for (;;) {
		AjmChain chain;
		{
			std::unique_lock lock(queue.mutex);
			queue.cv.wait(lock, [&] { return !queue.chains.empty(); });
			chain = std::move(queue.chains.front());
			queue.chains.pop_front();
		}
		AjmRunChain(chain);
	}
}

static int AjmWorkerCountLocked(AjmQueue& queue) {
	if (queue.worker_count < 0) {
		const char* value  = std::getenv("KYTY_AJM_WORKERS");
		queue.worker_count = value != nullptr ? std::clamp(std::atoi(value), 0, 16) : 4;
	}
	return queue.worker_count;
}

[[maybe_unused]] static void AjmSetWorkerCount(int count) {
	auto&            queue = AjmGetQueue();
	std::scoped_lock lock(queue.mutex);
	queue.worker_count = std::clamp(count, 0, 16);
}

// Tickets are taken and chains queued under one lock, so the queue never holds a chain ahead of an
// earlier chain of the same instance and a worker never waits on a chain queued behind it.
static void AjmSubmit(std::vector<AjmChain>& chains) {
	auto& queue    = AjmGetQueue();
	bool  run_here = false;
	{
		std::scoped_lock lock(queue.mutex);
		for (auto& chain: chains) {
			chain.ticket = chain.state->next_ticket++;
		}
		const int workers = AjmWorkerCountLocked(queue);
		run_here          = workers == 0;
		if (!run_here) {
			for (; queue.workers_started < workers; queue.workers_started++) {
				std::thread(AjmWorkerMain).detach();
			}
			for (auto& chain: chains) {
				queue.chains.push_back(std::move(chain));
			}
		}
	}
	if (run_here) {
		for (auto& chain: chains) {
			AjmRunChain(chain);
		}
	} else {
		queue.cv.notify_all();
	}
}

static int AjmAddJob(AjmBatchInfo* info, size_t job_size, const char* name, uint32_t instance,
                     std::function<void(AjmInstanceState*)> run) {
	const auto start  = info->offset;
	const int  result = AjmAppendJob(info, job_size, name);
	if (result != OK) {
		return result;
	}
	std::scoped_lock lock(g_ajm_pending_mutex);
	auto&            jobs = g_ajm_pending[info->buffer];
	// A job built over earlier ones means the batch was rebuilt without being started.
	if (!jobs.empty() && start < jobs.back().end) {
		jobs.clear();
	}
	jobs.push_back({instance, info->offset, std::move(run)});
	return OK;
}

int KYTY_SYSV_ABI AjmInitialize(int64_t reserved, uint32_t* context) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(context == nullptr);
	if (reserved != 0) {
		LOGF("\t ignoring reserved = 0x%016" PRIx64 "\n", static_cast<uint64_t>(reserved));
	}

	*context = 1;

	return OK;
}

int KYTY_SYSV_ABI AjmFinalize(uint32_t context) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 "\n", context);
	return OK;
}

int KYTY_SYSV_ABI AjmModuleRegister(uint32_t context, uint32_t codec, int64_t reserved) {
	PRINT_NAME();

	if (reserved != 0) {
		LOGF("\t ignoring reserved = 0x%016" PRIx64 "\n", static_cast<uint64_t>(reserved));
	}

	LOGF("\t codec = %u\n", codec);
	LOGF("\t %s\n", AjmCodecName(codec));

	const auto valid     = AjmCodecIsValid(codec);
	const auto supported = valid && AjmLogCodecSupport(codec);
	int        result    = OK;
	if (context != 1) {
		result = AJM_ERROR_INVALID_CONTEXT;
	} else if (!valid) {
		result = AJM_ERROR_INVALID_PARAMETER;
	} else if (!supported) {
		result = AJM_ERROR_CODEC_NOT_SUPPORTED;
	}

	if (!supported) {
		LOGF("\t codec is not implemented\n");
	}

	return result;
}

int KYTY_SYSV_ABI AjmModuleUnregister(uint32_t context, uint32_t codec) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", codec = %" PRIu32 "\n", context, codec);
	return OK;
}

int KYTY_SYSV_ABI AjmInstanceCreate(uint32_t context, uint32_t codec, uint64_t flags,
                                    uint32_t* instance) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(instance == nullptr);
	const auto supported = AjmLogCodecSupport(codec);

	auto state     = std::make_shared<AjmInstanceState>();
	state->context = context;
	state->codec   = codec;
	state->flags   = flags;
	state->decoder = AjmCreateDecoder(codec, flags);

	{
		// The lowest free index: a live instance's id is never handed out again.
		std::scoped_lock lock(g_ajm_instances_mutex);
		uint32_t         index = 1;
		while (index <= AJM_INSTANCE_INDEX_MASK && g_ajm_used_indices.test(index)) {
			index++;
		}
		if (index > AJM_INSTANCE_INDEX_MASK) {
			LOGF("\t no free instance\n");
			return AJM_ERROR_OUT_OF_RESOURCES;
		}
		g_ajm_used_indices.set(index);
		*instance                  = (codec << AJM_INSTANCE_INDEX_BITS) | index;
		g_ajm_instances[*instance] = std::move(state);
	}

	LOGF("\t context  = %" PRIu32 "\n"
	     "\t codec    = %" PRIu32 "\n"
	     "\t flags    = 0x%016" PRIx64 "\n"
	     "\t instance = 0x%08" PRIx32 "\n",
	     context, codec, flags, *instance);
	if (!supported) {
		LOGF("\t unsupported codec instance created without decoder\n");
	}

	return OK;
}

int KYTY_SYSV_ABI AjmInstanceDestroy(uint32_t context, uint32_t instance) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", instance = 0x%08" PRIx32 "\n", context, instance);

	// Batches already started keep the state until their jobs have run.
	AjmInstancePtr state;
	{
		std::scoped_lock lock(g_ajm_instances_mutex);
		if (const auto it = g_ajm_instances.find(instance); it != g_ajm_instances.end()) {
			state = std::move(it->second);
			g_ajm_instances.erase(it);
			g_ajm_used_indices.reset(instance & AJM_INSTANCE_INDEX_MASK);
		}
	}

	return OK;
}

int KYTY_SYSV_ABI AjmMemoryRegister(uint32_t context, void* ptr, size_t pages) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", ptr = 0x%016" PRIx64 ", pages = 0x%016" PRIx64 "\n", context,
	     reinterpret_cast<uint64_t>(ptr), static_cast<uint64_t>(pages));
	return OK;
}

int KYTY_SYSV_ABI AjmMemoryUnregister(uint32_t context, void* ptr) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", ptr = 0x%016" PRIx64 "\n", context,
	     reinterpret_cast<uint64_t>(ptr));
	return OK;
}

int KYTY_SYSV_ABI AjmBatchInitialize(void* buffer, size_t size, AjmBatchInfo* info) {
	PRINT_NAME();

	if (buffer == nullptr || info == nullptr) {
		return AJM_ERROR_INVALID_PARAMETER;
	}

	info->buffer           = buffer;
	info->offset           = 0;
	info->size             = size;
	info->last_good_job    = nullptr;
	info->last_good_job_ra = nullptr;

	std::scoped_lock lock(g_ajm_pending_mutex);
	g_ajm_pending.erase(buffer);

	return OK;
}

int KYTY_SYSV_ABI AjmBatchStart(uint32_t context, const AjmBatchInfo* info, int priority,
                                AjmBatchError* error, uint32_t* batch) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(info == nullptr);
	EXIT_NOT_IMPLEMENTED(batch == nullptr);

	(void)context;
	(void)priority;

	if (error != nullptr) {
		std::memset(error, 0, sizeof(AjmBatchError));
	}

	*batch = g_ajm_next_batch.fetch_add(1, std::memory_order_relaxed);

	std::vector<AjmJob> jobs;
	{
		std::scoped_lock lock(g_ajm_pending_mutex);
		if (const auto it = g_ajm_pending.find(info->buffer); it != g_ajm_pending.end()) {
			jobs = std::move(it->second);
			g_ajm_pending.erase(it);
		}
	}

	auto                  batch_state = std::make_shared<AjmBatchState>();
	std::vector<AjmChain> chains;
	std::vector<AjmJob>   unbound;
	{
		std::scoped_lock lock(g_ajm_instances_mutex);
		for (auto& job: jobs) {
			auto chain = std::find_if(chains.begin(), chains.end(),
			                          [&](const AjmChain& c) { return c.instance == job.instance; });
			if (chain == chains.end()) {
				const auto it = g_ajm_instances.find(job.instance);
				if (it == g_ajm_instances.end()) {
					unbound.push_back(std::move(job));
					continue;
				}
				chain = chains.insert(chains.end(), AjmChain {job.instance, it->second, 0, {}, batch_state});
			}
			chain->jobs.push_back(std::move(job));
		}
	}

	// Jobs without an instance read no instance state.
	for (auto& job: unbound) {
		job.run(nullptr);
	}
	if (chains.empty()) {
		return OK;
	}

	batch_state->pending.store(static_cast<uint32_t>(chains.size()), std::memory_order_relaxed);
	{
		std::scoped_lock lock(g_ajm_batches_mutex);
		// Batches nobody waits for are dropped once done.
		if (g_ajm_batches.size() >= 1024) {
			std::erase_if(g_ajm_batches, [](const auto& entry) {
				return entry.second->pending.load(std::memory_order_acquire) == 0;
			});
		}
		g_ajm_batches[*batch] = batch_state;
	}
	AjmSubmit(chains);

	return OK;
}

// Returns once the batch's jobs have run and their results are written.
int KYTY_SYSV_ABI AjmBatchWait(uint32_t context, uint32_t batch, uint32_t timeout,
                               AjmBatchError* error) {
	(void)context;
	(void)timeout;

	std::shared_ptr<AjmBatchState> state;
	{
		std::scoped_lock lock(g_ajm_batches_mutex);
		if (const auto it = g_ajm_batches.find(batch); it != g_ajm_batches.end()) {
			state = std::move(it->second);
			g_ajm_batches.erase(it);
		}
	}
	if (state != nullptr && state->pending.load(std::memory_order_acquire) != 0) {
		LibKernel::LongWait long_wait("ajm_batch_wait", state.get(), __builtin_return_address(0), batch);
		state->Wait();
	}

	if (error != nullptr) {
		std::memset(error, 0, sizeof(AjmBatchError));
	}

	return OK;
}

int KYTY_SYSV_ABI AjmBatchCancel(uint32_t context, uint32_t batch) {
	PRINT_NAME();
	LOGF("\t context = %" PRIu32 ", batch = %" PRIu32 "\n", context, batch);
	return OK;
}

int KYTY_SYSV_ABI AjmBatchErrorDump(const AjmBatchInfo* info, AjmBatchError* error) {
	PRINT_NAME();
	if (error != nullptr) {
		std::memset(error, 0, sizeof(AjmBatchError));
	}
	LOGF("\t info = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(info));
	return OK;
}

int KYTY_SYSV_ABI AjmBatchJobInitialize(AjmBatchInfo* info, uint32_t instance,
                                        const void* codec_parameters, size_t codec_parameters_size,
                                        void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", codec_parameters = 0x%016" PRIx64 ", size = 0x%016" PRIx64
	     "\n",
	     instance, reinterpret_cast<uint64_t>(codec_parameters),
	     static_cast<uint64_t>(codec_parameters_size));
	return AjmAddJob(info, AJM_JOB_CONTROL_SIZE, "initialize", instance,
	                 [parameters = AjmCopySidebandInput(codec_parameters, codec_parameters_size),
	                  present = codec_parameters != nullptr, result](AjmInstanceState* state) {
		                 const auto decode_result = AjmInitializeInstance(
		                     state, present ? parameters.data() : nullptr, parameters.size());
		                 AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decode_result);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobClearContext(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 "\n", instance);
	return AjmAddJob(info, AJM_JOB_CONTROL_SIZE, "clear-context", instance,
	                 [result](AjmInstanceState* state) {
		                 const auto decode_result =
		                     AjmControlInstance(state, AJM_FLAG_CONTROL_RESET, nullptr, 0);
		                 AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decode_result);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobDecode(AjmBatchInfo* info, uint32_t instance,
                                    const void* bitstream_input, size_t bitstream_input_size,
                                    void* pcm_output, size_t pcm_output_size, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", input = 0x%016" PRIx64 ", input_size = 0x%016" PRIx64
	     ", output = 0x%016" PRIx64 ", output_size = 0x%016" PRIx64 "\n",
	     instance, reinterpret_cast<uint64_t>(bitstream_input),
	     static_cast<uint64_t>(bitstream_input_size), reinterpret_cast<uint64_t>(pcm_output),
	     static_cast<uint64_t>(pcm_output_size));
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "decode", instance, [=](AjmInstanceState* state) {
		const auto decode_result = AjmDecodeInstance(instance, state, bitstream_input,
		                                             bitstream_input_size, pcm_output, pcm_output_size, true);
		AjmWriteStreamResult(
		    result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE + AJM_SIDEBAND_MFRAME_SIZE,
		    decode_result, true);
	});
}

int KYTY_SYSV_ABI AjmBatchJobDecodeSingle(AjmBatchInfo* info, uint32_t instance,
                                          const void* bitstream_input, size_t bitstream_input_size,
                                          void* pcm_output, size_t pcm_output_size, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 "\n", instance);
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "decode-single", instance, [=](AjmInstanceState* state) {
		const auto decode_result = AjmDecodeInstance(instance, state, bitstream_input,
		                                             bitstream_input_size, pcm_output, pcm_output_size, false);
		AjmWriteStreamResult(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE, decode_result,
		                     false);
	});
}

int KYTY_SYSV_ABI AjmBatchJobDecodeSplit(AjmBatchInfo* info, uint32_t instance,
                                         const AjmBuffer* input_buffers, size_t input_buffers_num,
                                         const AjmBuffer* output_buffers, size_t output_buffers_num,
                                         void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", input_buffers = 0x%016" PRIx64
	     ", input_num = 0x%016" PRIx64 ", output_buffers = 0x%016" PRIx64
	     ", output_num = 0x%016" PRIx64 "\n",
	     instance, reinterpret_cast<uint64_t>(input_buffers),
	     static_cast<uint64_t>(input_buffers_num), reinterpret_cast<uint64_t>(output_buffers),
	     static_cast<uint64_t>(output_buffers_num));
	return AjmAddJob(info, 16 * (input_buffers_num + output_buffers_num) + 32, "decode-split", instance,
	                 [instance, inputs = AjmCopyBuffers(input_buffers, input_buffers_num),
	                  outputs = AjmCopyBuffers(output_buffers, output_buffers_num),
	                  result](AjmInstanceState* state) {
		                 const auto decode_result =
		                     AjmDecodeSplitInstance(instance, state, inputs, outputs, true);
		                 AjmWriteStreamResult(result,
		                                      AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE +
		                                          AJM_SIDEBAND_MFRAME_SIZE,
		                                      decode_result, true);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobEncode(AjmBatchInfo* info, uint32_t instance, const void* pcm_input,
                                    size_t pcm_input_size, void* bitstream_output,
                                    size_t bitstream_output_size, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", input = 0x%016" PRIx64 ", output = 0x%016" PRIx64 "\n",
	     instance, reinterpret_cast<uint64_t>(pcm_input),
	     reinterpret_cast<uint64_t>(bitstream_output));
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "encode", instance, [=](AjmInstanceState*) {
		AjmDecodeResult encode_result {};
		encode_result.input_consumed = pcm_input_size;
		encode_result.output_written = bitstream_output_size;
		encode_result.frames = (pcm_input_size != 0 || bitstream_output_size != 0 ? 1u : 0u);
		encode_result.total_decoded_samples = bitstream_output_size;
		if (bitstream_output != nullptr && bitstream_output_size != 0) {
			std::memset(bitstream_output, 0, bitstream_output_size);
		}
		AjmWriteStreamResult(
		    result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_STREAM_SIZE + AJM_SIDEBAND_MFRAME_SIZE,
		    encode_result, true);
	});
}

int KYTY_SYSV_ABI AjmBatchJobGetInfo(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 "\n", instance);
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "get-info", instance, [result](AjmInstanceState* state) {
		AjmDecodeResult decode_result {};
		const auto      format = AjmGetInstanceFormat(state, &decode_result);
		AjmWriteFormat(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_FORMAT_SIZE, format,
		               decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobGetCodecInfo(AjmBatchInfo* info, uint32_t instance, void* result,
                                          size_t result_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", result_size = 0x%016" PRIx64 "\n", instance,
	     static_cast<uint64_t>(result_size));
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "get-codec-info", instance,
	                 [result, result_size](AjmInstanceState* state) {
		                 AjmDecodeResult decode_result {};
		                 (void)AjmGetInstanceFormat(state, &decode_result);
		                 AjmWriteResult(result, result_size, decode_result);
		                 AjmWriteCodecInfo(AjmIsValid(state) ? state->decoder.get() : nullptr, result,
		                                   result_size, decode_result);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobSetGaplessDecode(AjmBatchInfo* info, uint32_t instance,
                                              const void* gapless_decode, int reset, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	LOGF("\t instance = 0x%08" PRIx32 ", gapless_decode = 0x%016" PRIx64 ", reset = %d\n", instance,
	     reinterpret_cast<uint64_t>(gapless_decode), reset);
	const auto flags = AJM_FLAG_SIDEBAND_GAPLESS | (reset != 0 ? AJM_FLAG_CONTROL_RESET : 0);
	return AjmAddJob(info, AJM_JOB_CONTROL_SIZE, "set-gapless", instance,
	                 [flags, result,
	                  gapless = AjmCopySidebandInput(gapless_decode, sizeof(AjmSidebandGaplessDecode)),
	                  present = gapless_decode != nullptr](AjmInstanceState* state) {
		                 const auto decode_result = AjmControlInstance(
		                     state, flags, present ? gapless.data() : nullptr, gapless.size());
		                 AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decode_result);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobGetGaplessDecode(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 "\n", instance);
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "get-gapless", instance, [result](AjmInstanceState* state) {
		AjmSidebandGaplessDecode gapless_decode {};
		const auto               decode_result = AjmGetGaplessDecode(state, &gapless_decode);
		AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_GAPLESS_SIZE, decode_result);
		if (result != nullptr) {
			auto* gapless = reinterpret_cast<AjmSidebandGaplessDecode*>(static_cast<uint8_t*>(result) +
			                                                            AJM_SIDEBAND_RESULT_SIZE);
			*gapless      = gapless_decode;
		}
	});
}

int KYTY_SYSV_ABI AjmBatchJobSetResampleParameters(AjmBatchInfo* info, uint32_t instance,
                                                   float ratio, uint32_t flags, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	LOGF("\t instance = 0x%08" PRIx32 ", ratio = %f, flags = 0x%08" PRIx32 "\n", instance,
	     static_cast<double>(ratio), flags);
	return AjmAddJob(info, AJM_JOB_SET_RESAMPLE_PARAMETERS_SIZE, "set-resample", instance,
	                 [ratio, flags, result](AjmInstanceState* state) {
		                 const struct {
			                 float    ratio;
			                 uint32_t flags;
		                 } params {ratio, flags};
		                 const auto decoded =
		                     AjmControlInstance(state, AJM_FLAG_CONTROL_RESAMPLE, &params, sizeof(params));
		                 AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decoded);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobSetResampleParametersEx(AjmBatchInfo* info, uint32_t instance,
                                                     float ratio_start,
                                                     float ratio_change_per_sample, uint32_t flags,
                                                     void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	LOGF("\t instance = 0x%08" PRIx32 ", ratio_start = %f, ratio_change = %f, flags = 0x%08" PRIx32
	     "\n",
	     instance, static_cast<double>(ratio_start), static_cast<double>(ratio_change_per_sample),
	     flags);
	return AjmAddJob(info, AJM_JOB_SET_RESAMPLE_PARAMETERS_SIZE, "set-resample-ex", instance,
	                 [ratio_start, ratio_change_per_sample, flags, result](AjmInstanceState* state) {
		                 const struct {
			                 float    ratio;
			                 float    change;
			                 uint32_t flags;
		                 } params {ratio_start, ratio_change_per_sample, flags};
		                 const auto decoded = AjmControlInstance(state, AJM_FLAG_CONTROL_RESAMPLE_EX,
		                                                         &params, sizeof(params));
		                 AjmWriteResult(result, AJM_SIDEBAND_RESULT_SIZE, decoded);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobGetResampleInfo(AjmBatchInfo* info, uint32_t instance, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 "\n", instance);
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "get-resample-info", instance,
	                 [result](AjmInstanceState* state) {
		                 AjmDecodeResult decode_result {};
		                 (void)AjmGetInstanceFormat(state, &decode_result);
		                 AjmWriteResult(result,
		                                AJM_SIDEBAND_RESULT_SIZE + AJM_SIDEBAND_RESAMPLE_INFO_SIZE,
		                                decode_result);
		                 if (result != nullptr) {
			                 auto* resample = reinterpret_cast<AjmSidebandResampleInfo*>(
			                     static_cast<uint8_t*>(result) + AJM_SIDEBAND_RESULT_SIZE);
			                 *resample = decode_result.resample_info;
		                 }
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobGetStatistics(AjmBatchInfo* info, float interval, void* result) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);
	(void)interval;
	return AjmAddJob(info, AJM_JOB_GET_STATISTICS_SIZE, "get-statistics", 0,
	                 [result](AjmInstanceState*) { AjmWriteResult(result, sizeof(AjmGetStatisticsResult)); });
}

int KYTY_SYSV_ABI AjmBatchJobControl(AjmBatchInfo* info, uint32_t instance, uint64_t flags,
                                     const void* sideband_input, size_t sideband_input_size,
                                     void* sideband_output, size_t sideband_output_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", flags = 0x%016" PRIx64 ", input = 0x%016" PRIx64
	     ", input_size = 0x%016" PRIx64 ", output = 0x%016" PRIx64 ", output_size = 0x%016" PRIx64
	     "\n",
	     instance, flags, reinterpret_cast<uint64_t>(sideband_input),
	     static_cast<uint64_t>(sideband_input_size), reinterpret_cast<uint64_t>(sideband_output),
	     static_cast<uint64_t>(sideband_output_size));
	return AjmAddJob(info, AJM_JOB_CONTROL_SIZE, "control", instance,
	                 [flags, input = AjmCopySidebandInput(sideband_input, sideband_input_size),
	                  present = sideband_input != nullptr, sideband_output,
	                  sideband_output_size](AjmInstanceState* state) {
		                 const auto decode_result = AjmControlInstance(
		                     state, flags, present ? input.data() : nullptr, input.size());
		                 AjmWriteSideband(flags, sideband_output, sideband_output_size,
		                                  AjmGetDecoder(state), AjmGetGapless(state), decode_result);
	                 });
}

int KYTY_SYSV_ABI AjmBatchJobRun(AjmBatchInfo* info, uint32_t instance, uint64_t flags,
                                 const void* data_input, size_t data_input_size, void* data_output,
                                 size_t data_output_size, void* sideband_output,
                                 size_t sideband_output_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	LOGF("\t instance = 0x%08" PRIx32 ", flags = 0x%016" PRIx64 ", input = 0x%016" PRIx64
	     ", input_size = 0x%016" PRIx64 ", output = 0x%016" PRIx64 ", output_size = 0x%016" PRIx64
	     "\n",
	     instance, flags, reinterpret_cast<uint64_t>(data_input),
	     static_cast<uint64_t>(data_input_size), reinterpret_cast<uint64_t>(data_output),
	     static_cast<uint64_t>(data_output_size));
	return AjmAddJob(info, AJM_JOB_RUN_SIZE, "run", instance, [=](AjmInstanceState* state) {
		const auto decode_result =
		    AjmDecodeInstance(instance, state, data_input, data_input_size, data_output,
		                      data_output_size, (flags & AJM_FLAG_RUN_MULTIPLE_FRAMES) != 0);
		AjmWriteSideband(flags, sideband_output, sideband_output_size, AjmGetDecoder(state),
		                 AjmGetGapless(state), decode_result);
	});
}

int KYTY_SYSV_ABI AjmBatchJobRunSplit(AjmBatchInfo* info, uint32_t instance, uint64_t flags,
                                      const AjmBuffer* input_buffers, size_t input_buffers_num,
                                      const AjmBuffer* output_buffers, size_t output_buffers_num,
                                      void* sideband_output, size_t sideband_output_size) {
	PRINT_NAME();
	EXIT_NOT_IMPLEMENTED(info == nullptr);

	return AjmAddJob(info, 16 * (input_buffers_num + output_buffers_num) + 32, "run-split", instance,
	                 [instance, flags, inputs = AjmCopyBuffers(input_buffers, input_buffers_num),
	                  outputs = AjmCopyBuffers(output_buffers, output_buffers_num), sideband_output,
	                  sideband_output_size](AjmInstanceState* state) {
		                 const auto decode_result = AjmDecodeSplitInstance(
		                     instance, state, inputs, outputs, (flags & AJM_FLAG_RUN_MULTIPLE_FRAMES) != 0);
		                 AjmWriteSideband(flags, sideband_output, sideband_output_size,
		                                  AjmGetDecoder(state), AjmGetGapless(state), decode_result);
	                 });
}

const char* KYTY_SYSV_ABI AjmStrError(int error) {
	PRINT_NAME();
	LOGF("\t error = %d\n", error);
	return "AJM";
}

} // namespace Libs::Audio::Ajm
