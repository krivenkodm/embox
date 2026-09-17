#ifndef PROJECT_NCNN_CONV_FIXTURE_H_
#define PROJECT_NCNN_CONV_FIXTURE_H_

#include <stdint.h>

namespace ncnn_conv_fixture {

/*
 * Binary NCNN graph (unpacked FP32, no padding):
 * Input(5x5x2) -> Convolution(3x3, stride 1, 2 outputs, bias)
 *             -> ReLU -> MaxPool(2x2, stride 1) -> Output(2x2x2).
 * NCNN layer indexes: Input=16, Convolution=6, ReLU=26, Pooling=21.
 * Pooling pad_mode=1 selects VALID, with no implicit border extension.
 */
alignas(4) constexpr int32_t kNetworkParam[] = {
	7767517, 4, 4,
	16, 0, 1, 0, 0, 5, 1, 5, 2, 2, -233,
	6, 1, 1, 0, 1, 0, 2, 1, 3, 3, 1, 4, 0, 5, 1, 6, 36, 9, 0, -233,
	26, 1, 1, 1, 2, -233,
	21, 1, 1, 2, 3, 0, 0, 1, 2, 2, 1, 3, 0, 5, 1, -233,
};

struct alignas(4) ConvModel {
	uint32_t fp32_tag;
	float weights[36];
	float biases[2];
};

constexpr ConvModel kNetworkModel = {
	0,
	{
		/* Output 0: sum channel 0, plus a diagonal filter on channel 1. */
		1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 0, 0, 0, -1, 0, 0, 0, 1,
		/* Output 1: negative sum channel 0, plus a cross on channel 1. */
		-1, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 1, 0, 1, 0, 1, 0, 1, 0,
	},
	{0.5f, -0.5f},
};

static_assert(sizeof(ConvModel) == 39 * sizeof(uint32_t),
		"NCNN model must contain one FP32 tag, 36 weights and two biases");

constexpr uintptr_t kQspiParamAddress = 0x90ff0000u;
constexpr unsigned kQspiModelOffset = 256;
constexpr unsigned kQspiImageSize = 512;
static_assert(sizeof(kNetworkParam) <= kQspiModelOffset, "param page overflow");
static_assert(kQspiModelOffset + sizeof(kNetworkModel) <= kQspiImageSize,
		"model page overflow");

} // namespace ncnn_conv_fixture

#endif
