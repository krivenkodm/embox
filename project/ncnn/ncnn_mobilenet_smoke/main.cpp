#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <net.h>

#include "bounded_reader.h"
#include "input.h"
#include "manifest.h"
#include "reference.h"

extern "C" {
#include <lib/crypt/crc32.h>
#include <mem/heap/mspace_malloc.h>
int ncnn_stm32f746_qspi_map(void);
}

namespace {
using namespace mobilenet_fixture;
constexpr uintptr_t kSdramStart = 0x60000000u;
constexpr uintptr_t kSdramEnd = 0x60800000u;
static_assert(kImageHeader[2] <= 0x00af0000u, "model overlaps convolution fixture");
static_assert(kImageHeader[3] == 64 && kImageHeader[5] % 16 == 0, "bad image alignment");
static_assert(kImageHeader[3] + kImageHeader[4] <= kImageHeader[5], "param overlap");
static_assert(kImageHeader[5] + kImageHeader[6] == kImageHeader[2], "bad model extent");

bool in_sdram(const ncnn::Mat &tensor) {
	const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data);
	return address >= kSdramStart && address < kSdramEnd
			&& tensor.total() * tensor.elemsize <= kSdramEnd - address;
}

uint32_t checksum(const unsigned char *data, size_t size) {
	return static_cast<uint32_t>(count_crc32(const_cast<unsigned char *>(data),
			const_cast<unsigned char *>(data + size)));
}

int run_inference(const unsigned char *param, const unsigned char *model) {
	ncnn::Net net;
	net.opt.num_threads = 1;
	net.opt.lightmode = true;
	net.opt.use_packing_layout = false;
	net.opt.use_fp16_packed = false;
	net.opt.use_fp16_storage = false;
	net.opt.use_fp16_arithmetic = false;
	net.opt.use_bf16_storage = false;
	net.opt.use_int8_inference = false;
	BoundedReader parameters(param, kImageHeader[4]);
	BoundedReader weights(model, kImageHeader[6]);
	if (net.load_param_bin(parameters) != 0 || !parameters.complete()) {
		printf("ncnn_mobilenet_smoke: FAIL parameter load\n");
		return -1;
	}
	if (net.load_model(weights) != 0 || !weights.complete()
			|| weights.referenced() != kImageHeader[6] - kImageHeader[14]) {
		printf("ncnn_mobilenet_smoke: FAIL bounded FP32 model load\n");
		return -2;
	}
	printf("ncnn_mobilenet_smoke: FP32 weights referenced from QSPI: %lu bytes\n",
			(unsigned long)weights.referenced());

	ncnn::Mat input(kInputSize, kInputSize, kInputChannels);
	if (input.empty() || !in_sdram(input)) {
		printf("ncnn_mobilenet_smoke: FAIL input allocation in SDRAM\n");
		return -3;
	}
	for (int q = 0; q < kInputChannels; ++q) {
		float *channel = input.channel(q);
		for (int y = 0; y < kInputSize; ++y) {
			for (int x = 0; x < kInputSize; ++x) {
				channel[y * kInputSize + x] = input_value(q, y, x);
			}
		}
	}
	ncnn::Extractor ex = net.create_extractor();
	if (ex.input(kInputBlob, input) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL setting input\n");
		return -4;
	}
	printf("ncnn_mobilenet_smoke: inference 96x96x3, 138 layers, input=0x%08lx\n",
			(unsigned long)input.data);
	ncnn::Mat output;
	struct timespec start, end;
	const bool timed = clock_gettime(CLOCK_MONOTONIC, &start) == 0;
	const int result = ex.extract(kOutputBlob, output);
	if (timed && clock_gettime(CLOCK_MONOTONIC, &end) == 0) {
		const int64_t ms = (int64_t)(end.tv_sec - start.tv_sec) * 1000
				+ (end.tv_nsec - start.tv_nsec) / 1000000;
		printf("ncnn_mobilenet_smoke: extract time=%ld ms\n", (long)ms);
	}
	if (result != 0 || output.empty() || output.dims != 1
			|| output.w != kOutputSize || output.elempack != 1
			|| output.elemsize != sizeof(float) || !in_sdram(output)) {
		printf("ncnn_mobilenet_smoke: FAIL extract=%d or output shape/location\n", result);
		return -5;
	}
	const float *values = output;
	float max_error = 0;
	int top[5] = {-1, -1, -1, -1, -1};
	for (int i = 0; i < kOutputSize; ++i) {
		const float expected_abs = kReference[i] < 0 ? -kReference[i] : kReference[i];
		const float tolerance = 0.002f + 0.0002f * expected_abs;
		const float delta = values[i] - kReference[i];
		/* This interval test rejects NaN and infinity too. */
		if (!(delta >= -tolerance && delta <= tolerance)) {
			printf("ncnn_mobilenet_smoke: FAIL logit[%d] actual=%f expected=%f\n",
					i, (double)values[i], (double)kReference[i]);
			return -6;
		}
		const float absolute = delta < 0 ? -delta : delta;
		if (absolute > max_error) {
			max_error = absolute;
		}
		for (int rank = 0; rank < 5; ++rank) {
			if (top[rank] == -1 || values[i] > values[top[rank]]) {
				for (int j = 4; j > rank; --j) {
					top[j] = top[j - 1];
				}
				top[rank] = i;
				break;
			}
		}
	}
	printf("ncnn_mobilenet_smoke: 1000 logits verified, max_abs_error=%f output=0x%08lx\n",
			(double)max_error, (unsigned long)output.data);
	for (int rank = 0; rank < 5; ++rank) {
		printf("ncnn_mobilenet_smoke: top%d class=%d logit=%f\n",
				rank + 1, top[rank], (double)values[top[rank]]);
	}
	return 0;
}
} // namespace

int main(int argc, char **argv) {
	(void)argv;
	if (argc != 1) {
		printf("Usage: ncnn_mobilenet_smoke\n");
		return -1;
	}
	if (ncnn_stm32f746_qspi_map() != 0) {
		printf("ncnn_mobilenet_smoke: FAIL QSPI initialization\n");
		return -2;
	}
	const unsigned char *base = reinterpret_cast<const unsigned char *>(kQspiBase);
	if (memcmp(base, kImageHeader, sizeof(kImageHeader)) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL QSPI image missing or incompatible\n");
		return -3;
	}
	const unsigned char *param = base + kImageHeader[3];
	const unsigned char *model = base + kImageHeader[5];
	printf("ncnn_mobilenet_smoke: checking QSPI image at 0x%08lx\n", (unsigned long)base);
	if (checksum(param, kImageHeader[4]) != kImageHeader[7]
			|| checksum(model, kImageHeader[6]) != kImageHeader[8]) {
		printf("ncnn_mobilenet_smoke: FAIL QSPI checksum\n");
		return -4;
	}
	printf("ncnn_mobilenet_smoke: QSPI checksums verified\n");
	heap_type_t previous_heap;
	if (mspace_set_heap(HEAP_EXTERN_MEM, &previous_heap) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL external heap unavailable\n");
		return -5;
	}
	const int result = run_inference(param, model);
	/* All NCNN objects are destroyed before returning to the caller's heap. */
	if (mspace_set_heap(previous_heap, nullptr) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL restoring heap\n");
		return -6;
	}
	if (result == 0) {
		printf("ncnn_mobilenet_smoke: PASS MobileNetV3-Small FP32 96x96 from QSPI\n");
	}
	return result;
}
