#include "graphics/presentation/exitVideo.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include <SDL3/SDL.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace Libs::Graphics {

namespace {

struct Decoder {
	AVCodecContext* context = nullptr;
	int             stream  = -1;

	bool Open(AVFormatContext* format, AVMediaType type) {
		const AVCodec* codec = nullptr;
		stream               = av_find_best_stream(format, type, -1, -1, &codec, 0);
		if (stream < 0 || codec == nullptr) {
			return false;
		}
		context = avcodec_alloc_context3(codec);
		avcodec_parameters_to_context(context, format->streams[stream]->codecpar);
		context->thread_count = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
		return avcodec_open2(context, codec, nullptr) == 0;
	}
	~Decoder() { avcodec_free_context(&context); }
};

} // namespace

// Plays `path` full screen with its sound and returns when it ends or a button or key is pressed.
// Frames that come too late are dropped; the sound sets the pace.
int PlayExitVideo(const char* path) {
	std::FILE* probe = path != nullptr ? std::fopen(path, "rb") : nullptr;
	if (probe == nullptr) {
		return 0;
	}
	std::fclose(probe);
	av_log_set_level(AV_LOG_ERROR);
	AVFormatContext* format = nullptr;
	if (avformat_open_input(&format, path, nullptr, nullptr) != 0 || avformat_find_stream_info(format, nullptr) < 0) {
		std::fprintf(stderr, "exit video: cannot open %s\n", path);
		avformat_close_input(&format);
		return 1;
	}
	Decoder video;
	Decoder audio;
	if (!video.Open(format, AVMEDIA_TYPE_VIDEO)) {
		std::fprintf(stderr, "exit video: no video stream in %s\n", path);
		avformat_close_input(&format);
		return 1;
	}
	const bool has_audio = audio.Open(format, AVMEDIA_TYPE_AUDIO);

	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)) {
		std::fprintf(stderr, "exit video: SDL_Init failed: %s\n", SDL_GetError());
		avformat_close_input(&format);
		return 1;
	}
	SDL_Window*   window   = SDL_CreateWindow("KytyPS5-Native video", 1280, 720, SDL_WINDOW_FULLSCREEN);
	// Vulkan, as the game: under gamescope with Steam unloaded an OpenGL window stays black.
	SDL_Renderer* renderer = window != nullptr ? SDL_CreateRenderer(window, "vulkan") : nullptr;
	if (renderer == nullptr && window != nullptr) {
		renderer = SDL_CreateRenderer(window, nullptr);
	}
	SDL_HideCursor();
	SDL_Texture* texture = renderer != nullptr ? SDL_CreateTexture(renderer, SDL_PIXELFORMAT_IYUV, SDL_TEXTUREACCESS_STREAMING,
	                                                               video.context->width, video.context->height)
	                                           : nullptr;
	SDL_AudioStream* sound = nullptr;
	SwrContext*      resample = nullptr;
	if (has_audio) {
		SDL_AudioSpec spec {SDL_AUDIO_S16, 2, audio.context->sample_rate};
		sound = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, nullptr, nullptr);
		AVChannelLayout stereo = AV_CHANNEL_LAYOUT_STEREO;
		swr_alloc_set_opts2(&resample, &stereo, AV_SAMPLE_FMT_S16, audio.context->sample_rate, &audio.context->ch_layout,
		                    audio.context->sample_fmt, audio.context->sample_rate, 0, nullptr);
		if (resample == nullptr || swr_init(resample) < 0 || sound == nullptr) {
			swr_free(&resample);
			SDL_DestroyAudioStream(sound);
			sound = nullptr;
		} else {
			SDL_ResumeAudioStreamDevice(sound);
		}
	}
	SwsContext*    to_yuv    = nullptr;
	AVFrame*       frame     = av_frame_alloc();
	AVFrame*       yuv       = av_frame_alloc();
	AVPacket*      packet    = av_packet_alloc();
	const double   time_base = av_q2d(format->streams[video.stream]->time_base);
	uint64_t       start     = 0; // the clock starts with the first frame shown
	uint64_t       last_show = 0;
	int            shown     = 0;
	int            dropped   = 0;
	bool           quit      = texture == nullptr;
	yuv->format              = AV_PIX_FMT_YUV420P;
	yuv->width               = video.context->width;
	yuv->height              = video.context->height;
	av_frame_get_buffer(yuv, 32);

	auto show = [&](AVFrame* f) {
		const double pts = (f->best_effort_timestamp == AV_NOPTS_VALUE ? 0 : f->best_effort_timestamp) * time_base;
		if (start == 0) {
			start = SDL_GetTicksNS() - static_cast<uint64_t>(pts * 1e9);
		}
		const double now = static_cast<double>(SDL_GetTicksNS() - start) * 1e-9;
		// A late frame is dropped, unless nothing was shown for 100 ms: a slow decoder still shows frames.
		if (pts + 0.050 < now && SDL_GetTicksNS() - last_show < 100'000'000) {
			dropped++;
			return;
		}
		if (pts > now) {
			SDL_DelayNS(static_cast<uint64_t>((pts - now) * 1e9));
		}
		last_show = SDL_GetTicksNS();
		shown++;
		AVFrame* src = f;
		if (f->format != AV_PIX_FMT_YUV420P) {
			to_yuv = sws_getCachedContext(to_yuv, f->width, f->height, static_cast<AVPixelFormat>(f->format), yuv->width,
			                              yuv->height, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr, nullptr);
			sws_scale(to_yuv, f->data, f->linesize, 0, f->height, yuv->data, yuv->linesize);
			src = yuv;
		}
		SDL_UpdateYUVTexture(texture, nullptr, src->data[0], src->linesize[0], src->data[1], src->linesize[1], src->data[2],
		                     src->linesize[2]);
		SDL_RenderClear(renderer);
		SDL_RenderTexture(renderer, texture, nullptr, nullptr);
		SDL_RenderPresent(renderer);
	};
	auto play_sound = [&](AVFrame* f) {
		uint8_t*  out     = nullptr;
		const int samples = swr_get_out_samples(resample, f->nb_samples);
		if (av_samples_alloc(&out, nullptr, 2, samples, AV_SAMPLE_FMT_S16, 0) < 0) {
			return;
		}
		const int got = swr_convert(resample, &out, samples, const_cast<const uint8_t**>(f->extended_data), f->nb_samples);
		if (got > 0) {
			SDL_PutAudioStreamData(sound, out, got * 2 * 2);
		}
		av_freep(&out);
	};

	while (!quit && av_read_frame(format, packet) >= 0) {
		for (SDL_Event event; SDL_PollEvent(&event);) {
			quit = quit || event.type == SDL_EVENT_QUIT || event.type == SDL_EVENT_KEY_DOWN ||
			       event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
			if (event.type == SDL_EVENT_GAMEPAD_ADDED) {
				SDL_OpenGamepad(event.gdevice.which);
			}
		}
		Decoder* decoder = packet->stream_index == video.stream ? &video
		                   : (sound != nullptr && packet->stream_index == audio.stream) ? &audio
		                                                                                 : nullptr;
		if (decoder != nullptr && avcodec_send_packet(decoder->context, packet) == 0) {
			while (avcodec_receive_frame(decoder->context, frame) == 0) {
				if (decoder == &video) {
					show(frame);
				} else {
					play_sound(frame);
				}
			}
		}
		av_packet_unref(packet);
	}
	// What is still queued to play.
	while (!quit && sound != nullptr && SDL_GetAudioStreamQueued(sound) > 0) {
		SDL_Delay(10);
	}

	std::fprintf(stderr, "exit video: %s, %dx%d, %d frames shown, %d dropped, renderer %s\n", path, video.context->width,
	             video.context->height, shown, dropped, renderer != nullptr ? SDL_GetRendererName(renderer) : "none");

	av_packet_free(&packet);
	av_frame_free(&yuv);
	av_frame_free(&frame);
	sws_freeContext(to_yuv);
	swr_free(&resample);
	SDL_DestroyAudioStream(sound);
	SDL_DestroyTexture(texture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	avformat_close_input(&format);
	return 0;
}

void MarkUserExit() {
	if (const char* marker = std::getenv("KYTY_EXIT_MARKER"); marker != nullptr && *marker != '\0') {
		if (std::FILE* file = std::fopen(marker, "w"); file != nullptr) {
			std::fclose(file);
		}
	}
}

} // namespace Libs::Graphics
