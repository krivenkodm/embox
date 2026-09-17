#include <limits.h>
#include "photo_input.h"

/* Keep this decoder private; another template may use the general image module.
 * Scalar JPEG decoding is identical on the native reference and Cortex-M7. */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STBIDEF static __attribute__((unused))
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_SIMD
#define STBI_NO_THREAD_LOCALS
#define STBI_MAX_DIMENSIONS 512
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"
#include "../common/image/stb_image.h"
#pragma GCC diagnostic pop

namespace mobilenet_photo {
unsigned char *decode_rgb(const unsigned char *jpeg, size_t size) {
	int width = 0, height = 0, channels = 0;
	if (!jpeg || size > INT_MAX || !stbi_info_from_memory(jpeg, (int)size,
			&width, &height, &channels) || width != kSourceWidth || height != kSourceHeight) {
		return nullptr;
	}
	unsigned char *rgb = stbi_load_from_memory(jpeg, (int)size, &width, &height, &channels, 3);
	if (width != kSourceWidth || height != kSourceHeight) {
		stbi_image_free(rgb);
		return nullptr;
	}
	return rgb;
}
void free_rgb(unsigned char *rgb) {
	stbi_image_free(rgb);
}
} // namespace mobilenet_photo
