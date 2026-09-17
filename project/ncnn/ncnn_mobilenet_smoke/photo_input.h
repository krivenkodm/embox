#ifndef PROJECT_NCNN_PHOTO_INPUT_H_
#define PROJECT_NCNN_PHOTO_INPUT_H_

#include <stddef.h>
#include <mat.h>

namespace mobilenet_photo {
constexpr int kSourceWidth = 295;
constexpr int kSourceHeight = 231;
constexpr int kInputSize = 224;
constexpr size_t kRgbBytes = kSourceWidth * kSourceHeight * 3;
constexpr size_t kInputBytes = kInputSize * kInputSize * 3 * sizeof(float);

/* Decode only the fixed fixture dimensions. The caller verifies JPEG CRC. */
unsigned char *decode_rgb(const unsigned char *jpeg, size_t size);
void free_rgb(unsigned char *rgb);

/* Match the existing project classifier: RGB bilinear resize to a square,
 * then scale to [0,1]. Mean/std normalization is already in the NCNN graph. */
inline ncnn::Mat prepare_input(const unsigned char *rgb) {
	ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB,
			kSourceWidth, kSourceHeight, kInputSize, kInputSize);
	if (!input.empty()) {
		const float norm[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
		input.substract_mean_normalize(nullptr, norm);
	}
	return input;
}
} // namespace mobilenet_photo
#endif
