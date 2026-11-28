#include "libs/ajm.cpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <opus.h>

#include "AjmMp3Fixtures.inc"

using namespace Libs::Audio::Ajm;

namespace {

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AjmDecodeTests: %s\n", message);
		std::abort();
	}
}

using Clock = std::chrono::steady_clock;

double Seconds(Clock::time_point start) {
	return std::chrono::duration<double>(Clock::now() - start).count();
}

constexpr uint32_t CODEC_MP3  = 0;
constexpr uint32_t CODEC_OPUS = 24;

struct DecodeResult {
	AjmSidebandResult result;
	AjmSidebandStream stream;
	AjmSidebandMFrame frames;
};

struct Batch {
	std::vector<uint8_t> data = std::vector<uint8_t>(4096);
	AjmBatchInfo         info {};
	uint32_t             id = 0;

	Batch() { Reset(); }

	void Reset() { Check(AjmBatchInitialize(data.data(), data.size(), &info) == OK, "initialize batch"); }

	void Start() { Check(AjmBatchStart(1, &info, 0, nullptr, &id) == OK, "start batch"); }
	void Wait() const { Check(AjmBatchWait(1, id, 0xffffffffu, nullptr) == OK, "wait for batch"); }
	void Run() {
		Start();
		Wait();
	}
};

uint32_t CreateInstance(uint32_t codec, uint32_t channels, AjmSampleEncoding encoding) {
	uint32_t instance = 0;
	Check(AjmInstanceCreate(1, codec, channels | (uint64_t(encoding) << 7u), &instance) == OK,
	      "create instance");
	if (codec == CODEC_OPUS) {
		AjmDecOpusInitializeParameters params {channels, 48000, 0};
		AjmSidebandResult              result {};
		Batch                          batch;
		AjmBatchJobInitialize(&batch.info, instance, &params, sizeof(params), &result);
		batch.Run();
		Check(result.result == OK, "initialize Opus instance");
	}
	return instance;
}

// Opus packets as AJM takes them: a 16-bit little-endian size before each packet.
std::vector<uint8_t> EncodeOpus(uint32_t channels, int application, int bitrate, int frame_samples,
                                int packets, uint32_t seed) {
	int   error   = 0;
	auto* encoder = opus_encoder_create(48000, static_cast<int>(channels), application, &error);
	Check(encoder != nullptr && error == OPUS_OK, "create test encoder");
	opus_encoder_ctl(encoder, OPUS_SET_BITRATE(bitrate));
	std::vector<uint8_t> stream;
	std::vector<int16_t> pcm(static_cast<size_t>(frame_samples) * channels);
	std::vector<uint8_t> packet(4000);
	for (int p = 0; p < packets; ++p) {
		for (int i = 0; i < frame_samples; ++i) {
			const double t = (p * frame_samples + i) / 48000.0;
			for (uint32_t c = 0; c < channels; ++c) {
				seed = seed * 1664525u + 1013904223u;
				pcm[i * channels + c] =
				    static_cast<int16_t>(9000 * std::sin(t * 2 * M_PI * (180 + 70 * c + 40 * std::sin(t * 3))) +
				                         static_cast<int>((seed >> 16u) % 3000u) - 1500);
			}
		}
		const int size = opus_encode(encoder, pcm.data(), frame_samples, packet.data(),
		                             static_cast<opus_int32>(packet.size()));
		Check(size > 0, "encode test packet");
		stream.push_back(static_cast<uint8_t>(size));
		stream.push_back(static_cast<uint8_t>(size >> 8));
		stream.insert(stream.end(), packet.begin(), packet.begin() + size);
	}
	opus_encoder_destroy(encoder);
	return stream;
}

// The conversion as the decoders did it before they kept their converter: a sample converter set up
// for each frame.
AVFrame* ConvertPerFrame(const AVFrame* frame, AjmSampleEncoding encoding) {
	AVFrame* converted     = av_frame_alloc();
	converted->format      = AjmSampleEncodingToAvFormat(encoding);
	converted->sample_rate = frame->sample_rate;
	converted->nb_samples  = frame->nb_samples;
	Check(av_channel_layout_copy(&converted->ch_layout, &frame->ch_layout) >= 0 &&
	          av_frame_get_buffer(converted, 0) >= 0,
	      "reference frame");
	SwrContext* swr = nullptr;
	Check(swr_alloc_set_opts2(&swr, &converted->ch_layout,
	                          static_cast<AVSampleFormat>(converted->format), converted->sample_rate,
	                          &frame->ch_layout, static_cast<AVSampleFormat>(frame->format),
	                          frame->sample_rate, 0, nullptr) >= 0 &&
	          swr_init(swr) >= 0 && swr_convert_frame(swr, converted, frame) >= 0,
	      "reference conversion");
	swr_free(&swr);
	return converted;
}

void AppendFrame(std::vector<uint8_t>& pcm, const AVFrame* frame, AjmSampleEncoding encoding) {
	AVFrame*   converted = ConvertPerFrame(frame, encoding);
	const auto bytes     = static_cast<size_t>(converted->nb_samples) *
	                   converted->ch_layout.nb_channels * AjmBytesPerSample(encoding);
	pcm.insert(pcm.end(), converted->data[0], converted->data[0] + bytes);
	av_frame_free(&converted);
}

struct ReferenceOpus {
	AVCodecContext* context = nullptr;

	explicit ReferenceOpus(uint32_t channels) {
		const auto* codec = avcodec_find_decoder(AV_CODEC_ID_OPUS);
		context           = avcodec_alloc_context3(codec);
		av_channel_layout_default(&context->ch_layout, static_cast<int>(channels));
		context->sample_rate = 48000;
		Check(avcodec_open2(context, codec, nullptr) >= 0, "open reference Opus decoder");
	}
	~ReferenceOpus() { avcodec_free_context(&context); }

	void Decode(const std::vector<uint8_t>& stream, size_t begin, size_t end, AjmSampleEncoding encoding,
	            std::vector<uint8_t>& pcm) {
		while (begin < end) {
			const size_t size = stream[begin] | (size_t {stream[begin + 1]} << 8u);
			AVPacket*    packet = av_packet_alloc();
			AVFrame*     frame  = av_frame_alloc();
			Check(av_new_packet(packet, static_cast<int>(size)) >= 0, "reference packet");
			std::memcpy(packet->data, stream.data() + begin + 2, size);
			Check(avcodec_send_packet(context, packet) >= 0 && avcodec_receive_frame(context, frame) >= 0,
			      "reference Opus decode");
			av_channel_layout_default(&frame->ch_layout, frame->ch_layout.nb_channels);
			AppendFrame(pcm, frame, encoding);
			av_packet_free(&packet);
			av_frame_free(&frame);
			begin += size + 2;
		}
	}
};

// The offset just past the first `packets` packets from `begin`.
size_t SkipPackets(const std::vector<uint8_t>& stream, size_t begin, int packets) {
	for (int i = 0; i < packets && begin < stream.size(); ++i) {
		begin += 2 + (stream[begin] | (size_t {stream[begin + 1]} << 8u));
	}
	return begin;
}

// Decodes [begin, end) with jobs of `per_job` packets, appending the PCM.
void AjmDecodeOpus(uint32_t instance, const std::vector<uint8_t>& stream, size_t begin, size_t end,
                   int per_job, std::vector<uint8_t>& pcm) {
	std::vector<uint8_t> output(2880 * 8 * 4 * static_cast<size_t>(per_job));
	Batch                batch;
	while (begin < end) {
		const size_t job_end = std::min(end, SkipPackets(stream, begin, per_job));
		DecodeResult result {};
		batch.Reset();
		AjmBatchJobDecode(&batch.info, instance, stream.data() + begin, job_end - begin, output.data(),
		                  output.size(), &result);
		batch.Run();
		Check((result.result.result & ~AJM_RESULT_PARTIAL_INPUT) == 0 &&
		          result.stream.size_consumed == static_cast<int>(job_end - begin),
		      "AJM Opus decode");
		pcm.insert(pcm.end(), output.begin(), output.begin() + result.stream.size_produced);
		begin = job_end;
	}
}

void OpusMatchesPerFrameConversion(uint32_t channels, AjmSampleEncoding encoding, int application,
                                   int bitrate, int frame_samples) {
	const auto stream = EncodeOpus(channels, application, bitrate, frame_samples, 150, 7 + channels);
	const auto middle = SkipPackets(stream, 0, 70);

	std::vector<uint8_t> expected;
	ReferenceOpus        reference(channels);
	reference.Decode(stream, 0, middle, encoding, expected);
	avcodec_flush_buffers(reference.context);
	reference.Decode(stream, middle, stream.size(), encoding, expected);

	const auto           instance = CreateInstance(CODEC_OPUS, channels, encoding);
	std::vector<uint8_t> actual;
	AjmDecodeOpus(instance, stream, 0, middle, 3, actual);
	AjmSidebandResult cleared {};
	Batch             clear;
	AjmBatchJobClearContext(&clear.info, instance, &cleared);
	clear.Run();
	Check(cleared.result == OK, "clear context");
	AjmDecodeOpus(instance, stream, middle, stream.size(), 5, actual);
	AjmInstanceDestroy(1, instance);

	Check(!expected.empty() && actual == expected,
	      "Opus PCM matches per-frame conversion byte for byte, across a context reset");
}

std::vector<uint8_t> ReferenceMp3(const uint8_t* data, size_t size, AjmSampleEncoding encoding) {
	const auto* codec   = avcodec_find_decoder(AV_CODEC_ID_MP3);
	auto*       context = avcodec_alloc_context3(codec);
	auto*       parser  = av_parser_init(codec->id);
	Check(avcodec_open2(context, codec, nullptr) >= 0, "open reference MP3 decoder");
	std::vector<uint8_t> pcm;
	size_t               offset = 0;
	while (offset < size) {
		uint8_t*  packet_data = nullptr;
		int       packet_size = 0;
		const int consumed = av_parser_parse2(parser, context, &packet_data, &packet_size, data + offset,
		                                      static_cast<int>(size - offset), AV_NOPTS_VALUE,
		                                      AV_NOPTS_VALUE, 0);
		Check(consumed >= 0, "reference MP3 parse");
		offset += static_cast<size_t>(consumed);
		if (packet_size > 0) {
			AVPacket* packet = av_packet_alloc();
			Check(av_new_packet(packet, packet_size) >= 0, "reference packet");
			std::memcpy(packet->data, packet_data, static_cast<size_t>(packet_size));
			Check(avcodec_send_packet(context, packet) >= 0, "reference MP3 decode");
			av_packet_free(&packet);
			for (;;) {
				AVFrame* frame = av_frame_alloc();
				if (avcodec_receive_frame(context, frame) < 0) {
					av_frame_free(&frame);
					break;
				}
				AppendFrame(pcm, frame, encoding);
				av_frame_free(&frame);
			}
		}
		if (consumed == 0) {
			break;
		}
	}
	av_parser_close(parser);
	avcodec_free_context(&context);
	return pcm;
}

void Mp3MatchesPerFrameConversion(const uint8_t* data, size_t size, uint32_t channels,
                                  AjmSampleEncoding encoding) {
	const auto expected = ReferenceMp3(data, size, encoding);
	Check(expected.size() > 4096, "reference MP3 PCM");

	const auto           instance = CreateInstance(CODEC_MP3, channels, encoding);
	std::vector<uint8_t> actual(expected.size() + 4096, 0x5a);
	DecodeResult         result {};
	Batch                batch;
	AjmBatchJobDecode(&batch.info, instance, data, size, actual.data(), actual.size(), &result);
	batch.Run();
	Check(result.stream.size_produced == static_cast<int>(expected.size()) &&
	          std::equal(expected.begin(), expected.end(), actual.begin()),
	      "MP3 PCM matches per-frame conversion byte for byte");

	// Split buffers on a fresh decoder give the same PCM.
	AjmSidebandResult cleared {};
	Batch             clear;
	AjmBatchJobClearContext(&clear.info, instance, &cleared);
	clear.Run();
	std::vector<uint8_t> split(expected.size());
	const size_t         third = size / 3;
	AjmBuffer            inputs[] {{const_cast<uint8_t*>(data), third},
	                               {const_cast<uint8_t*>(data) + third, third},
	                               {const_cast<uint8_t*>(data) + 2 * third, size - 2 * third}};
	AjmBuffer            outputs[] {{split.data(), 1000}, {split.data() + 1000, split.size() - 1000}};
	DecodeResult         split_result {};
	Batch                split_batch;
	AjmBatchJobDecodeSplit(&split_batch.info, instance, inputs, 3, outputs, 2, &split_result);
	split_batch.Run();
	Check(split_result.stream.size_produced == static_cast<int>(expected.size()) && split == expected,
	      "split MP3 decode after a reset matches");
	AjmInstanceDestroy(1, instance);
}

void JobsRunWhenTheBatchStarts() {
	const auto   stream   = EncodeOpus(2, OPUS_APPLICATION_AUDIO, 64000, 960, 2, 3);
	const auto   instance = CreateInstance(CODEC_OPUS, 2, AjmSampleEncoding::S16);
	DecodeResult result;
	std::memset(&result, 0xa5, sizeof(result));
	std::vector<uint8_t> pcm(8192, 0x11);
	Batch                batch;
	AjmBatchJobDecode(&batch.info, instance, stream.data(), stream.size(), pcm.data(), pcm.size(), &result);
	Check(result.result.result == static_cast<int32_t>(0xa5a5a5a5u) && pcm[0] == 0x11,
	      "a built job writes nothing before its batch starts");
	batch.Run();
	Check((result.result.result & ~AJM_RESULT_PARTIAL_INPUT) == 0 &&
	          result.stream.size_produced == 2 * 960 * 4 &&
	          result.frames.num_frames == 2,
	      "the started batch decodes");
	AjmInstanceDestroy(1, instance);
}

struct Voice {
	uint32_t             codec;
	uint32_t             channels;
	AjmSampleEncoding    encoding;
	std::vector<uint8_t> stream;
	uint32_t             instance = 0;
	size_t               offset   = 0;
	std::vector<uint8_t> pcm;
};

std::vector<Voice> MakeVoices(int packets, bool mp3) {
	std::vector<Voice> voices;
	for (uint32_t i = 0; i < 6; ++i) {
		const uint32_t channels = 1 + i % 2;
		const auto     encoding = static_cast<AjmSampleEncoding>(i % 3);
		voices.push_back({CODEC_OPUS, channels, encoding,
		                  EncodeOpus(channels, i % 2 == 0 ? OPUS_APPLICATION_VOIP : OPUS_APPLICATION_AUDIO,
		                             24000 + 8000 * i, i == 4 ? 2880 : 960, packets, 100 + i)});
	}
	if (!mp3) {
		return voices;
	}
	voices.push_back({CODEC_MP3, 2, AjmSampleEncoding::S16,
	                  std::vector<uint8_t>(std::begin(kAjmMp3Stereo48k), std::end(kAjmMp3Stereo48k))});
	voices.push_back({CODEC_MP3, 1, AjmSampleEncoding::Float,
	                  std::vector<uint8_t>(std::begin(kAjmMp3Mono24k), std::end(kAjmMp3Mono24k))});
	return voices;
}

// Every batch has one decode job per voice; `in_flight` batches are started before the oldest is
// waited for, so an instance has jobs in several started batches at once.
std::vector<std::vector<uint8_t>> DecodeVoices(int workers, int in_flight, double* seconds,
                                               int packets = 60, bool mp3 = true) {
	AjmSetWorkerCount(workers);
	auto voices = MakeVoices(packets, mp3);
	for (auto& voice: voices) {
		voice.instance = CreateInstance(voice.codec, voice.channels, voice.encoding);
	}
	struct Job {
		size_t               voice;
		DecodeResult         result {};
		std::vector<uint8_t> output = std::vector<uint8_t>(6 * 2880 * 2 * 4);
	};
	struct Started {
		std::unique_ptr<Batch> batch;
		std::vector<Job>       jobs;
	};
	std::deque<Started> started;
	double              busy = 0; // in sceAjmBatchStart and sceAjmBatchWait
	const auto          finish = [&] {
        auto& oldest = started.front();
        const auto wait = Clock::now();
        oldest.batch->Wait();
        busy += Seconds(wait);
        for (auto& job: oldest.jobs) {
            Check((job.result.result.result & ~AJM_RESULT_PARTIAL_INPUT) == 0, "voice decode");
            auto& pcm = voices[job.voice].pcm;
            pcm.insert(pcm.end(), job.output.begin(), job.output.begin() + job.result.stream.size_produced);
        }
        started.pop_front();
	};
	for (bool more = true; more;) {
		more = false;
		Started next {std::make_unique<Batch>(), {}};
		next.jobs.reserve(voices.size());
		for (size_t v = 0; v < voices.size(); ++v) {
			auto& voice = voices[v];
			if (voice.offset >= voice.stream.size()) {
				continue;
			}
			// Opus: two packets a job; MP3: 600 bytes a job, whatever the frame boundaries.
			const size_t end = voice.codec == CODEC_OPUS ? SkipPackets(voice.stream, voice.offset, 2)
			                                             : std::min(voice.stream.size(), voice.offset + 600);
			auto& job = next.jobs.emplace_back();
			job.voice = v;
			AjmBatchJobDecode(&next.batch->info, voice.instance, voice.stream.data() + voice.offset,
			                  end - voice.offset, job.output.data(), job.output.size(), &job.result);
			voice.offset = end;
			more         = true;
		}
		const auto start = Clock::now();
		next.batch->Start();
		busy += Seconds(start);
		started.push_back(std::move(next));
		if (static_cast<int>(started.size()) >= in_flight) {
			finish();
		}
	}
	while (!started.empty()) {
		finish();
	}
	if (seconds != nullptr) {
		*seconds = busy;
	}
	std::vector<std::vector<uint8_t>> pcm;
	for (auto& voice: voices) {
		AjmInstanceDestroy(1, voice.instance);
		pcm.push_back(std::move(voice.pcm));
	}
	return pcm;
}

void ParallelBatchesMatchSequential() {
	const auto expected = DecodeVoices(0, 1, nullptr);
	for (const auto& pcm: expected) {
		Check(pcm.size() > 1000, "every voice decodes");
	}
	for (int workers: {1, 4}) {
		for (int in_flight: {1, 3}) {
			Check(DecodeVoices(workers, in_flight, nullptr) == expected,
			      "parallel batches give the PCM of sequential decoding");
		}
	}
	AjmSetWorkerCount(0);
}

void InstanceIds() {
	const auto first = CreateInstance(CODEC_OPUS, 2, AjmSampleEncoding::S16);
	Check((first >> 14u) == CODEC_OPUS && (first & 0x3fffu) != 0, "instance id holds codec and index");
	uint32_t reused = 0;
	for (int i = 0; i < 20000; ++i) {
		uint32_t id = 0;
		Check(AjmInstanceCreate(1, CODEC_OPUS, 2, &id) == OK, "create instance");
		Check(id != first && (id & 0x3fffu) != 0, "a live instance's id is never handed out");
		Check(reused == 0 || id == reused, "a destroyed instance's index is reused");
		reused = id;
		AjmInstanceDestroy(1, id);
	}

	// The live instance still decodes after the index space has been cycled.
	const auto           stream = EncodeOpus(2, OPUS_APPLICATION_AUDIO, 64000, 960, 3, 5);
	std::vector<uint8_t> pcm;
	AjmDecodeOpus(first, stream, 0, stream.size(), 3, pcm);
	Check(pcm.size() == 3 * 960 * 4, "live instance decodes");

	std::vector<uint32_t> all;
	for (;;) {
		uint32_t   id     = 0;
		const auto result = AjmInstanceCreate(1, CODEC_OPUS, 2, &id);
		if (result != OK) {
			Check(result == AJM_ERROR_OUT_OF_RESOURCES && all.size() == 0x3fffu - 1,
			      "the index space holds 16383 instances");
			break;
		}
		all.push_back(id);
	}
	for (const auto id: all) {
		AjmInstanceDestroy(1, id);
	}
	AjmInstanceDestroy(1, first);
}

// Decode cost per Opus packet: per-frame conversion as before against the AJM path.
void BenchOpus(const char* name, int application, int bitrate) {
	const auto stream = EncodeOpus(2, application, bitrate, 960, 3000, 11);
	const auto encoding = AjmSampleEncoding::S16;
	double     reference_time = 1e9, ajm_time = 1e9;
	for (int round = 0; round < 3; ++round) {
		std::vector<uint8_t> pcm;
		auto                 start = Clock::now();
		ReferenceOpus        reference(2);
		reference.Decode(stream, 0, stream.size(), encoding, pcm);
		reference_time = std::min(reference_time, Seconds(start));

		std::vector<uint8_t> actual;
		const auto           instance = CreateInstance(CODEC_OPUS, 2, encoding);
		start                         = Clock::now();
		AjmDecodeOpus(instance, stream, 0, stream.size(), 4, actual);
		ajm_time = std::min(ajm_time, Seconds(start));
		AjmInstanceDestroy(1, instance);
		Check(actual == pcm, "bench PCM matches");
	}
	std::printf("AjmDecodeTests bench: %s stereo: per-frame conversion %.2f us/packet, AJM %.2f us/packet\n",
	            name, reference_time * 1e6 / 3000, ajm_time * 1e6 / 3000);
}

void BenchMp3() {
	const auto* data           = kAjmMp3Stereo48k;
	const auto  size           = sizeof(kAjmMp3Stereo48k);
	constexpr int rounds        = 300;
	auto        start          = Clock::now();
	size_t      bytes          = 0;
	for (int i = 0; i < rounds; ++i) {
		bytes += ReferenceMp3(data, size, AjmSampleEncoding::S16).size();
	}
	const double reference_time = Seconds(start);
	const auto   instance       = CreateInstance(CODEC_MP3, 2, AjmSampleEncoding::S16);
	std::vector<uint8_t> output(bytes / rounds + 4096);
	start = Clock::now();
	for (int i = 0; i < rounds; ++i) {
		AjmSidebandResult cleared {};
		DecodeResult      result {};
		Batch             batch;
		AjmBatchJobClearContext(&batch.info, instance, &cleared);
		AjmBatchJobDecode(&batch.info, instance, data, size, output.data(), output.size(), &result);
		batch.Run();
	}
	const double ajm_time = Seconds(start);
	AjmInstanceDestroy(1, instance);
	const double frames = static_cast<double>(bytes / rounds) / (1152 * 4) * rounds;
	std::printf("AjmDecodeTests bench: MP3 48 kHz stereo: per-frame conversion %.2f us/frame, AJM %.2f us/frame "
	            "(decoder setup included in both)\n",
	            reference_time * 1e6 / frames, ajm_time * 1e6 / frames);
}

void BenchParallel() {
	double parallel = 0;
	DecodeVoices(0, 1, &parallel, 300, false);
	const auto batches = 1500 / 2;
	// Workers once started keep running, so only the count of the largest run is measured.
	for (int workers: {0, 4}) {
		DecodeVoices(workers, 1, &parallel, 1500, false);
		std::printf("AjmDecodeTests bench: 6 Opus voices, a batch holds a 2-packet job per voice, started and "
		            "waited at once: workers %d, %.1f us/batch in start and wait\n",
		            workers, parallel * 1e6 / batches);
	}
	AjmSetWorkerCount(0);
}

} // namespace

int main(int argc, char** argv) {
	AjmSetWorkerCount(0);
	for (uint32_t channels: {1u, 2u}) {
		for (auto encoding: {AjmSampleEncoding::S16, AjmSampleEncoding::S32, AjmSampleEncoding::Float}) {
			OpusMatchesPerFrameConversion(channels, encoding, OPUS_APPLICATION_VOIP, 16000, 960);
			OpusMatchesPerFrameConversion(channels, encoding, OPUS_APPLICATION_AUDIO, 96000, 960);
			OpusMatchesPerFrameConversion(channels, encoding, OPUS_APPLICATION_AUDIO, 48000, 2880);
			OpusMatchesPerFrameConversion(channels, encoding, OPUS_APPLICATION_RESTRICTED_LOWDELAY, 64000,
			                              240);
		}
	}
	for (auto encoding: {AjmSampleEncoding::S16, AjmSampleEncoding::S32, AjmSampleEncoding::Float}) {
		Mp3MatchesPerFrameConversion(kAjmMp3Stereo48k, sizeof(kAjmMp3Stereo48k), 2, encoding);
		Mp3MatchesPerFrameConversion(kAjmMp3Mono24k, sizeof(kAjmMp3Mono24k), 1, encoding);
	}
	JobsRunWhenTheBatchStarts();
	ParallelBatchesMatchSequential();
	InstanceIds();
	if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {
		// Guest threads run at the top clock too.
		Common::Thread::RunAtTopClock("KYTY_UCLAMP");
		BenchOpus("Opus SILK 24 kbps", OPUS_APPLICATION_VOIP, 24000);
		BenchOpus("Opus CELT 64 kbps", OPUS_APPLICATION_AUDIO, 64000);
		BenchMp3();
		BenchParallel();
	}
	std::puts("AjmDecodeTests: passed");
}
