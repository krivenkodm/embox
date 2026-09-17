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
#include "timing.h"

extern "C" {
#include "sdram_profile.h"
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

int run_inference(const unsigned char *param, const unsigned char *model, RunTimings &times) {
	CleanupTimer cleanup(times.cleanup);
	Timer step;
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
	times.param = step.milliseconds();
	step.reset();
	if (net.load_model(weights) != 0 || !weights.complete()
			|| weights.referenced() != kImageHeader[6] - kImageHeader[14]) {
		printf("ncnn_mobilenet_smoke: FAIL bounded FP32 model load\n");
		return -2;
	}
	times.model = step.milliseconds();
	printf("ncnn_mobilenet_smoke: FP32 weights referenced from QSPI: %lu bytes\n",
			(unsigned long)weights.referenced());

	step.reset();
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
	times.input = step.milliseconds();
	printf("ncnn_mobilenet_smoke: inference 96x96x3, 138 layers, input=0x%08lx\n",
			(unsigned long)input.data);
	ncnn::Mat output;
	step.reset();
	const int result = ex.extract(kOutputBlob, output);
	times.extract = step.milliseconds();
	printf("ncnn_mobilenet_smoke: extract time=%ld ms\n", times.extract);
	step.reset();
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
	times.verify = step.milliseconds();
	printf("ncnn_mobilenet_smoke: 1000 logits verified, max_abs_error=%f output=0x%08lx\n",
			(double)max_error, (unsigned long)output.data);
	for (int rank = 0; rank < 5; ++rank) {
		printf("ncnn_mobilenet_smoke: top%d class=%d logit=%f\n",
				rank + 1, top[rank], (double)values[top[rank]]);
	}
	cleanup.start();
	return 0;
}
} // namespace

static int run_command() {
	Timer total, step;
	RunTimings times;
	if (ncnn_stm32f746_qspi_map() != 0) {
		printf("ncnn_mobilenet_smoke: FAIL QSPI initialization\n");
		return -2;
	}
	const long qspi_ms = step.milliseconds();
	const unsigned char *base = reinterpret_cast<const unsigned char *>(kQspiBase);
	if (memcmp(base, kImageHeader, sizeof(kImageHeader)) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL QSPI image missing or incompatible\n");
		return -3;
	}
	const unsigned char *param = base + kImageHeader[3];
	const unsigned char *model = base + kImageHeader[5];
	printf("ncnn_mobilenet_smoke: checking QSPI image at 0x%08lx\n", (unsigned long)base);
	step.reset();
	if (checksum(param, kImageHeader[4]) != kImageHeader[7]
			|| checksum(model, kImageHeader[6]) != kImageHeader[8]) {
		printf("ncnn_mobilenet_smoke: FAIL QSPI checksum\n");
		return -4;
	}
	const long crc_ms = step.milliseconds();
	printf("ncnn_mobilenet_smoke: QSPI checksums verified\n");
	heap_type_t previous_heap;
	if (mspace_set_heap(HEAP_EXTERN_MEM, &previous_heap) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL external heap unavailable\n");
		return -5;
	}
	const int result = run_inference(param, model, times);
	/* All NCNN objects are destroyed before returning to the caller's heap. */
	if (mspace_set_heap(previous_heap, nullptr) != 0) {
		printf("ncnn_mobilenet_smoke: FAIL restoring heap\n");
		return -6;
	}
	if (result == 0) {
		const long total_ms = total.milliseconds();
		printf("ncnn_mobilenet_smoke: timing_ms qspi=%ld crc=%ld param=%ld model=%ld input=%ld extract=%ld verify=%ld cleanup=%ld total=%ld\n",
				qspi_ms, crc_ms, times.param, times.model, times.input, times.extract,
				times.verify, times.cleanup, total_ms);
		if (qspi_ms < 0 || crc_ms < 0 || times.param < 0 || times.model < 0
				|| times.input < 0 || times.extract < 0 || times.verify < 0
				|| times.cleanup < 0 || total_ms < 0) {
			printf("ncnn_mobilenet_smoke: FAIL monotonic clock\n");
			return -7;
		}
	}
	return result;
}

int main(int argc, char **argv) {
	(void)argv;
	if (argc != 1) {
		printf("Usage: ncnn_mobilenet_smoke\n");
		return -1;
	}
	if (ncnn_sdram_profile_selftest() != 0 || ncnn_sdram_profile_begin() != 0) {
		printf("ncnn_mobilenet_smoke: FAIL SDRAM profiling hooks\n");
		return -8;
	}
	const int result = run_command();
	struct ncnn_sdram_stats memory;
	ncnn_sdram_profile_end(&memory);
	printf("ncnn_mobilenet_smoke: SDRAM bytes capacity=%lu page=%lu baseline=%lu peak=%lu final=%lu min_free=%lu failed_allocs=%lu other_heap_allocs=%lu\n",
			(unsigned long)memory.capacity, (unsigned long)memory.page_size,
			(unsigned long)memory.baseline, (unsigned long)memory.peak,
			(unsigned long)memory.final_used, (unsigned long)(memory.capacity - memory.peak),
			(unsigned long)memory.failed_allocations, (unsigned long)memory.other_heap_allocations);
	if (memory.final_used != memory.baseline || memory.failed_allocations != 0) {
		printf("ncnn_mobilenet_smoke: FAIL SDRAM release/allocation\n");
		return -9;
	}
	if (result == 0) {
		printf("ncnn_mobilenet_smoke: PASS MobileNetV3-Small FP32 96x96 from QSPI\n");
	}
	return result;
}
