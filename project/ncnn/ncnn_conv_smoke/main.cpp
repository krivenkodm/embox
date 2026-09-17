#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fixture.h"

#include <mat.h>
#include <net.h>

extern "C" {
#include <mem/heap/mspace_malloc.h>

int ncnn_stm32f746_qspi_map(void);
}

namespace {

using namespace ncnn_conv_fixture;

constexpr uintptr_t kSdramStart = 0x60040000u; /* LCD occupies the first 256 KiB. */
constexpr uintptr_t kSdramEnd = 0x60800000u;
constexpr int kChannels = 2;

/*
 * Channel 0 is the row-major ramp -12..12; channel 1 is a -2/1 checkerboard.
 * At each valid convolution center, let x be channel 0 and c be channel 1:
 *   conv[0] = 9*x + c + 0.5
 *   conv[1] = -9*x + 4*(-1-c) - 0.5
 * These analytic values give the reference tensors below, before and after
 * ReLU. The pool reference is the maximum of each overlapping 2x2 window.
 */
constexpr float kExpectedConv[] = {
	-55.5f, -43.5f, -37.5f,
	 -7.5f,  -1.5f,  10.5f,
	 34.5f,  46.5f,  52.5f,
	 57.5f,  36.5f,  39.5f,
	  0.5f,   3.5f, -17.5f,
	-32.5f, -53.5f, -50.5f,
};

constexpr float kExpectedRelu[] = {
	0, 0, 0,
	0, 0, 10.5f,
	34.5f, 46.5f, 52.5f,
	57.5f, 36.5f, 39.5f,
	0.5f, 3.5f, 0,
	0, 0, 0,
};

constexpr float kExpectedPool[] = {
	0, 10.5f, 46.5f, 52.5f,
	57.5f, 39.5f, 3.5f, 3.5f,
};

bool in_sdram(const ncnn::Mat &tensor) {
	const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data);
	return address >= kSdramStart && address < kSdramEnd
			&& tensor.total() * tensor.elemsize <= kSdramEnd - address;
}

bool check_tensor(const char *stage, const ncnn::Mat &tensor,
		int width, int height, const float *expected) {
	if (tensor.empty() || tensor.dims != 3 || tensor.w != width
			|| tensor.h != height || tensor.c != kChannels
			|| tensor.elempack != 1 || tensor.elemsize != sizeof(float)) {
		printf("ncnn_conv_smoke: FAIL %s shape dims=%d w=%d h=%d c=%d pack=%d\n",
				stage, tensor.dims, tensor.w, tensor.h, tensor.c, tensor.elempack);
		return false;
	}
	if (!in_sdram(tensor)) {
		printf("ncnn_conv_smoke: FAIL %s buffer is not in SDRAM\n", stage);
		return false;
	}

	for (int q = 0; q < kChannels; ++q) {
		const float *values = tensor.channel(q);
		for (int i = 0; i < width * height; ++i) {
			const float delta = values[i] - expected[q * width * height + i];
			/* Negating the interval check also rejects NaN and infinity. */
			if (!(delta >= -0.00001f && delta <= 0.00001f)) {
				printf("ncnn_conv_smoke: FAIL %s channel=%d element=%d\n",
						stage, q, i);
				return false;
			}
		}
	}

	printf("ncnn_conv_smoke: %s PASS %dx%dx%d data=0x%08lx\n",
			stage, width, height, kChannels, (unsigned long)tensor.data);
	return true;
}

int run_inference(const unsigned char *param, const unsigned char *model) {
	ncnn::Net net;
	net.opt.num_threads = 1;
	/* Keep intermediate blobs to verify every layer separately. */
	net.opt.lightmode = false;
	net.opt.use_packing_layout = false;
	net.opt.use_fp16_packed = false;
	net.opt.use_fp16_storage = false;
	net.opt.use_fp16_arithmetic = false;
	net.opt.use_bf16_storage = false;
	net.opt.use_int8_inference = false;

	const int param_bytes = net.load_param(param);
	if (param_bytes != static_cast<int>(sizeof(kNetworkParam))) {
		printf("ncnn_conv_smoke: FAIL param load (%d/%lu bytes)\n",
				param_bytes, (unsigned long)sizeof(kNetworkParam));
		return -2;
	}
	const int model_bytes = net.load_model(model);
	if (model_bytes != static_cast<int>(sizeof(kNetworkModel))) {
		printf("ncnn_conv_smoke: FAIL model load (%d/%lu bytes)\n",
				model_bytes, (unsigned long)sizeof(kNetworkModel));
		return -3;
	}

	ncnn::Mat input(5, 5, kChannels);
	if (input.empty() || !in_sdram(input)) {
		printf("ncnn_conv_smoke: FAIL input allocation in SDRAM\n");
		return -4;
	}
	float *ramp = input.channel(0);
	float *checker = input.channel(1);
	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 5; ++x) {
			ramp[y * 5 + x] = static_cast<float>(y * 5 + x - 12);
			checker[y * 5 + x] = (x + y) % 2 == 0 ? -2.0f : 1.0f;
		}
	}

	ncnn::Extractor extractor = net.create_extractor();
	if (extractor.input(0, input) != 0) {
		printf("ncnn_conv_smoke: FAIL setting input\n");
		return -5;
	}

	const char *stages[] = {"convolution", "relu", "pooling"};
	const float *references[] = {kExpectedConv, kExpectedRelu, kExpectedPool};
	for (int stage = 0; stage < 3; ++stage) {
		ncnn::Mat output;
		const int result = extractor.extract(stage + 1, output);
		if (result != 0) {
			printf("ncnn_conv_smoke: FAIL %s extract=%d\n", stages[stage], result);
			return -6;
		}
		const int side = stage == 2 ? 2 : 3;
		if (!check_tensor(stages[stage], output, side, side, references[stage])) {
			return -7;
		}
		if (stage == 2) {
			for (int q = 0; q < kChannels; ++q) {
				const float *values = output.channel(q);
				printf("ncnn_conv_smoke: output[%d]=[", q);
				for (int i = 0; i < 4; ++i) {
					const int milli = static_cast<int>(values[i] * 1000.0f + 0.5f);
					printf("%s%d.%03d", i == 0 ? "" : ", ", milli / 1000, milli % 1000);
				}
				printf("]\n");
			}
		}
	}

	return 0;
}

} // namespace

int main(int argc, char **argv) {
	const bool use_qspi = argc == 2 && strcmp(argv[1], "qspi") == 0;
	if (argc != 1 && !use_qspi) {
		printf("Usage: ncnn_conv_smoke [qspi]\n");
		return -1;
	}
	const unsigned char *param = reinterpret_cast<const unsigned char *>(kNetworkParam);
	const unsigned char *model = reinterpret_cast<const unsigned char *>(&kNetworkModel);
	if (use_qspi) {
		if (ncnn_stm32f746_qspi_map() != 0) {
			printf("ncnn_conv_smoke: FAIL QSPI initialization\n");
			return -9;
		}
		param = reinterpret_cast<const unsigned char *>(kQspiParamAddress);
		model = param + kQspiModelOffset;
		/* The pointer-based NCNN loader has no length bound. Only admit this
		 * exact fixed fixture before parsing; blank/corrupt flash is rejected. */
		if (memcmp(param, kNetworkParam, sizeof(kNetworkParam)) != 0
				|| memcmp(model, &kNetworkModel, sizeof(kNetworkModel)) != 0) {
			printf("ncnn_conv_smoke: FAIL QSPI fixture missing or mismatched\n");
			return -10;
		}
		printf("ncnn_conv_smoke: QSPI fixture verified param=0x%08lx model=0x%08lx\n",
				(unsigned long)param, (unsigned long)model);
	}
	printf("ncnn_conv_smoke: FP32 5x5x2 -> conv 3x3x2 -> ReLU -> max pool 2x2x2\n");

	heap_type_t previous_heap;
	if (mspace_set_heap(HEAP_EXTERN_MEM, &previous_heap) != 0) {
		printf("ncnn_conv_smoke: FAIL external heap unavailable\n");
		return -1;
	}
	const int result = run_inference(param, model);
	/* Destroy NCNN objects before restoring the caller's heap. */
	if (mspace_set_heap(previous_heap, nullptr) != 0) {
		printf("ncnn_conv_smoke: FAIL restoring heap\n");
		return result == 0 ? -8 : result;
	}
	if (result == 0) {
		printf("ncnn_conv_smoke: PASS convolution -> ReLU -> pooling in external SDRAM\n");
		if (use_qspi) {
			printf("ncnn_conv_smoke: PASS model loaded from QSPI\n");
		}
	}
	return result;
}
