#pragma once

#include "common/assert.h"
#include "common/common.h"
#include "libs/ajm/decoder.h"

#include <cstdint>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

namespace Libs::Audio::Ajm {

inline AVSampleFormat AjmSampleEncodingToAvFormat(AjmSampleEncoding encoding) {
	switch (encoding) {
		case AjmSampleEncoding::S16: return AV_SAMPLE_FMT_S16;
		case AjmSampleEncoding::S32: return AV_SAMPLE_FMT_S32;
		case AjmSampleEncoding::Float: return AV_SAMPLE_FMT_FLT;
		default: EXIT("unsupported AJM PCM sample encoding %u\n", static_cast<uint32_t>(encoding));
	}
	return AV_SAMPLE_FMT_S16;
}

// Converts decoded frames to the instance's PCM encoding. A decoder keeps one converter: a conversion
// that changes neither rate nor layout carries no state from frame to frame, so the converter and its
// output frame are set up again only when the decoded format changes.
class AjmFfmpegConverter {
public:
	AjmFfmpegConverter() = default;
	~AjmFfmpegConverter() {
		swr_free(&m_swr);
		av_frame_free(&m_out);
		av_channel_layout_uninit(&m_in_layout);
	}

	KYTY_CLASS_NO_COPY(AjmFfmpegConverter);

	// The frame stays valid until the next call.
	const AVFrame* Convert(const AVFrame* frame, AjmSampleEncoding encoding, AjmDecodeResult* result) {
		if (frame == nullptr || result == nullptr) {
			return nullptr;
		}

		const auto format = AjmSampleEncodingToAvFormat(encoding);
		if (m_out == nullptr) {
			m_out = av_frame_alloc();
		}
		if (m_out == nullptr) {
			result->result = AJM_RESULT_CODEC_ERROR | AJM_RESULT_FATAL;
			return nullptr;
		}
		const bool same_input = m_swr != nullptr && m_in_format == frame->format &&
		                        m_in_rate == frame->sample_rate && m_out_format == format &&
		                        av_channel_layout_compare(&m_in_layout, &frame->ch_layout) == 0;
		if (!same_input || m_out_samples != frame->nb_samples) {
			av_frame_unref(m_out);
			m_out_samples       = 0;
			m_out->format      = format;
			m_out->sample_rate = frame->sample_rate;
			m_out->nb_samples  = frame->nb_samples;
			if (av_channel_layout_copy(&m_out->ch_layout, &frame->ch_layout) < 0 ||
			    av_frame_get_buffer(m_out, 0) < 0) {
				av_frame_unref(m_out);
				result->result = AJM_RESULT_CODEC_ERROR | AJM_RESULT_FATAL;
				return nullptr;
			}
			m_out_samples = frame->nb_samples;
		}
		if (!same_input) {
			swr_free(&m_swr);
			av_channel_layout_uninit(&m_in_layout);
			if (swr_alloc_set_opts2(&m_swr, &m_out->ch_layout, format, m_out->sample_rate,
			                        &frame->ch_layout, static_cast<AVSampleFormat>(frame->format),
			                        frame->sample_rate, 0, nullptr) < 0 ||
			    swr_init(m_swr) < 0 || av_channel_layout_copy(&m_in_layout, &frame->ch_layout) < 0) {
				swr_free(&m_swr);
				result->result = AJM_RESULT_CODEC_ERROR | AJM_RESULT_INVALID_DATA;
				return nullptr;
			}
			m_in_format  = frame->format;
			m_in_rate    = frame->sample_rate;
			m_out_format = format;
		}
		m_out->nb_samples = m_out_samples;
		if (swr_convert_frame(m_swr, m_out, frame) < 0) {
			swr_free(&m_swr);
			result->result = AJM_RESULT_CODEC_ERROR | AJM_RESULT_INVALID_DATA;
			return nullptr;
		}
		return m_out;
	}

private:
	SwrContext*      m_swr         = nullptr;
	AVFrame*         m_out         = nullptr;
	AVChannelLayout  m_in_layout {};
	int              m_in_format   = AV_SAMPLE_FMT_NONE;
	int              m_in_rate     = 0;
	AVSampleFormat   m_out_format  = AV_SAMPLE_FMT_NONE;
	int              m_out_samples = 0;
};

} // namespace Libs::Audio::Ajm
