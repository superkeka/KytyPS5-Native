#include "libs/libJpegDec.cpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Loader {
void SymbolDatabase::Add(const SymbolResolve&, uint64_t, const std::string&) {}
namespace Timer {
double GetTimeMs() {
	return 0.0;
}
} // namespace Timer
} // namespace Loader

namespace {
using namespace Libs::JpegDec;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "JpegDecTests:%d: %s\n", __LINE__, #condition);                   \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

// A 16x8 JPEG: the left half red, the right half blue.
std::vector<uint8_t> MakeJpeg(uint32_t width = 16, uint32_t height = 8) {
	std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			auto* pixel = &rgb[(static_cast<size_t>(y) * width + x) * 3];
			pixel[0]    = x < width / 2 ? 255 : 0;
			pixel[1]    = 0;
			pixel[2]    = x < width / 2 ? 0 : 255;
		}
	}
	std::vector<uint8_t> jpeg;
	CHECK(stbi_write_jpg_to_func(
	          [](void* context, void* data, int size) {
		          auto* out   = static_cast<std::vector<uint8_t>*>(context);
		          auto* bytes = static_cast<const uint8_t*>(data);
		          out->insert(out->end(), bytes, bytes + size);
	          },
	          &jpeg, static_cast<int>(width), static_cast<int>(height), 3, rgb.data(), 100) != 0);
	return jpeg;
}

bool Near(uint8_t value, uint8_t expected) {
	return value + 24 >= expected && value <= expected + 24;
}

struct Decoder {
	std::array<uint8_t, 0x48> memory {};
	void*                     handle = nullptr;

	Decoder() {
		const JpegDecCreateParam create {sizeof(JpegDecCreateParam), 1, 1920};
		const auto               size = JpegDecQueryMemorySize(&create);
		CHECK(size > 0 && static_cast<size_t>(size) + 1 <= memory.size());
		// Memory that is not 8-byte aligned still gives an aligned handle.
		CHECK(JpegDecCreate(&create, memory.data() + 1, static_cast<uint32_t>(size), &handle) == 0);
		CHECK((reinterpret_cast<uintptr_t>(handle) & 7u) == 0);
	}

	~Decoder() {
		CHECK(JpegDecDelete(handle) == 0);
		CHECK(JpegDecDelete(handle) == JPEG_DEC_ERROR_INVALID_HANDLE);
	}
};

void TestParseHeader() {
	const auto        jpeg = MakeJpeg();
	JpegDecParseParam parse {jpeg.data(), static_cast<uint32_t>(jpeg.size()), 0, 1};
	JpegDecImageInfo  info {};
	CHECK(JpegDecParseHeader(&parse, &info) == 0);
	CHECK(info.image_width == 16 && info.image_height == 8);
	CHECK(info.output_image_width == 16 && info.output_image_height == 8);
	CHECK(info.color_space == JPEG_DEC_COLOR_SPACE_RGB && info.component_number == 3);
	CHECK(info.component_id[0] == 1 && info.component_id[1] == 2 && info.component_id[2] == 3);
	CHECK(info.suitable_csc_attribute == JPEG_DEC_CSC_ORDINARY);

	parse.down_scale = 4;
	CHECK(JpegDecParseHeader(&parse, &info) == 0);
	CHECK(info.output_image_width == 4 && info.output_image_height == 2);

	const uint8_t not_jpeg[] {0x89, 'P', 'N', 'G', 0, 0, 0, 0};
	parse = {not_jpeg, sizeof(not_jpeg), 0, 1};
	CHECK(JpegDecParseHeader(&parse, &info) == JPEG_DEC_ERROR_INVALID_DATA);
	CHECK(JpegDecParseHeader(&parse, nullptr) == JPEG_DEC_ERROR_INVALID_ADDR);
}

void TestDecode() {
	const auto jpeg = MakeJpeg();
	for (const uint16_t format: {JPEG_DEC_PIXEL_FORMAT_R8G8B8A8, JPEG_DEC_PIXEL_FORMAT_B8G8R8A8}) {
		Decoder              decoder;
		constexpr uint32_t   pitch = 16 * 4 + 32;
		std::vector<uint8_t> image(pitch * 8, 0xcd);
		JpegDecDecodeParam   param {};
		param.jpeg_mem_addr  = jpeg.data();
		param.image_mem_addr = image.data();
		param.jpeg_mem_size  = static_cast<uint32_t>(jpeg.size());
		param.image_mem_size = static_cast<uint32_t>(image.size());
		param.down_scale     = 1;
		param.pixel_format   = format;
		param.alpha_value    = 200;
		param.image_pitch    = pitch;
		JpegDecImageInfo info {};
		CHECK(JpegDecDecode(decoder.handle, &param, &info) == ((16 << 16) | 8));
		CHECK(info.output_image_width == 16 && info.output_image_height == 8);
		const uint8_t red_at  = format == JPEG_DEC_PIXEL_FORMAT_R8G8B8A8 ? 0 : 2;
		const uint8_t blue_at = 2 - red_at;
		for (uint32_t y = 0; y < 8; y++) {
			const auto* left  = &image[y * pitch + 1 * 4];
			const auto* right = &image[y * pitch + 14 * 4];
			CHECK(Near(left[red_at], 255) && Near(left[1], 0) && Near(left[blue_at], 0));
			CHECK(Near(right[red_at], 0) && Near(right[1], 0) && Near(right[blue_at], 255));
			CHECK(left[3] == 200 && right[3] == 200);
			CHECK(image[y * pitch + 16 * 4] == 0xcd); // the padding is left alone
		}

		param.image_mem_size = pitch * 7;
		CHECK(JpegDecDecode(decoder.handle, &param, &info) == JPEG_DEC_ERROR_INVALID_SIZE);
		param.image_mem_size = static_cast<uint32_t>(image.size());
		param.image_pitch    = 16 * 4 - 4;
		CHECK(JpegDecDecode(decoder.handle, &param, &info) == JPEG_DEC_ERROR_INVALID_PARAM);
		param.image_pitch  = 0;
		param.pixel_format = 7;
		CHECK(JpegDecDecode(decoder.handle, &param, &info) == JPEG_DEC_ERROR_INVALID_PARAM);
		param.pixel_format  = format;
		param.jpeg_mem_size = 10;
		CHECK(JpegDecDecode(decoder.handle, &param, &info) == JPEG_DEC_ERROR_INVALID_DATA);
	}
	JpegDecDecodeParam param {};
	CHECK(JpegDecDecode(nullptr, &param, nullptr) == JPEG_DEC_ERROR_INVALID_HANDLE);
}

void TestDownScale() {
	const auto           jpeg = MakeJpeg();
	Decoder              decoder;
	std::vector<uint8_t> image(8 * 4 * 4, 0);
	JpegDecDecodeParam   param {};
	param.jpeg_mem_addr  = jpeg.data();
	param.image_mem_addr = image.data();
	param.jpeg_mem_size  = static_cast<uint32_t>(jpeg.size());
	param.image_mem_size = static_cast<uint32_t>(image.size());
	param.down_scale     = 2;
	param.alpha_value    = 0xffff;
	JpegDecImageInfo info {};
	CHECK(JpegDecDecode(decoder.handle, &param, &info) == ((8 << 16) | 4));
	CHECK(info.image_width == 16 && info.output_image_width == 8 && info.output_image_height == 4);
	CHECK(Near(image[0], 255) && Near(image[2], 0) && image[3] == 255);
	CHECK(Near(image[7 * 4], 0) && Near(image[7 * 4 + 2], 255));
}

} // namespace

int main() {
	TestParseHeader();
	TestDecode();
	TestDownScale();
	std::printf("JpegDecTests: all cases passed\n");
	return 0;
}
