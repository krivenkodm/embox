#ifndef PROJECT_NCNN_MOBILENET_INPUT_H_
#define PROJECT_NCNN_MOBILENET_INPUT_H_

namespace mobilenet_fixture {
constexpr int kInputSize = 96;
constexpr int kInputChannels = 3;
constexpr int kOutputSize = 1000;
constexpr int kInputBlob = 0;
constexpr int kOutputBlob = 152;

/* Synthetic RGB values in [0, 1]; mean/std normalization is inside the graph. */
inline float input_value(int channel, int y, int x) {
	return static_cast<float>((3 * x + 5 * y + 17 * channel) % 256) / 255.0f;
}
} // namespace mobilenet_fixture

#endif
