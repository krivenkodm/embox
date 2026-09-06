#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <mat.h>
#include <net.h>

extern "C" {
#include <mem/heap/mspace_malloc.h>
}

namespace {

constexpr uintptr_t kSdramStart = 0x60000000u;
constexpr uintptr_t kSdramEnd = 0x60800000u;

/*
 * Binary NCNN graph:
 *   input[3] -> InnerProduct(2 outputs, bias, ReLU) -> output[2]
 *
 * Layer type indexes come from NCNN's stable layer_type_enum.h:
 * InnerProduct=15, Input=16.
 */
alignas(4) constexpr int32_t kNetworkParam[] = {
	7767517, 2, 2,
	16, 0, 1, 0, 0, 3, -233,
	15, 1, 1, 0, 1, 0, 2, 1, 1, 2, 6, 9, 1, -233,
};

struct alignas(4) DenseModel {
	uint32_t fp32_tag;
	float weights[6];
	float biases[2];
};

constexpr DenseModel kNetworkModel = {
	0,
	{1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f},
	{0.5f, -0.5f},
};

int run_inference(void) {
	ncnn::Net net;
	net.opt.num_threads = 1;
	net.opt.use_packing_layout = false;

	const int param_bytes = net.load_param(
			reinterpret_cast<const unsigned char *>(kNetworkParam));
	if (param_bytes != static_cast<int>(sizeof(kNetworkParam))) {
		printf("ncnn_inference_smoke: FAIL param load (%d/%lu bytes)\n",
				param_bytes, (unsigned long)sizeof(kNetworkParam));
		return -2;
	}

	const int model_bytes = net.load_model(
			reinterpret_cast<const unsigned char *>(&kNetworkModel));
	if (model_bytes != static_cast<int>(sizeof(kNetworkModel))) {
		printf("ncnn_inference_smoke: FAIL model load (%d/%lu bytes)\n",
				model_bytes, (unsigned long)sizeof(kNetworkModel));
		return -3;
	}

	ncnn::Mat input(3);
	if (input.empty()) {
		printf("ncnn_inference_smoke: FAIL input allocation\n");
		return -4;
	}

	float *input_data = input;
	input_data[0] = 1.0f;
	input_data[1] = 2.0f;
	input_data[2] = -1.0f;

	ncnn::Extractor extractor = net.create_extractor();
	if (extractor.input(0, input) != 0) {
		printf("ncnn_inference_smoke: FAIL setting input\n");
		return -5;
	}

	ncnn::Mat output;
	const int extract_result = extractor.extract(1, output);
	printf("ncnn_inference_smoke: extract=%d dims=%d w=%d pack=%d allocated=%lu\n",
			extract_result, output.dims, output.w, output.elempack,
			(unsigned long)output.total());
	if (extract_result != 0 || output.dims != 1 || output.w != 2
			|| output.elempack != 1) {
		printf("ncnn_inference_smoke: FAIL inference\n");
		return -6;
	}

	const float *output_data = output;
	const int output0_milli = static_cast<int>(output_data[0] * 1000.0f);
	const int output1_milli = static_cast<int>(output_data[1] * 1000.0f);
	const uintptr_t output_address = reinterpret_cast<uintptr_t>(output.data);
	const bool output_in_sdram = output_address >= kSdramStart
			&& output_address < kSdramEnd;

	printf("ncnn_inference_smoke: input=[1.000, 2.000, -1.000]\n");
	printf("ncnn_inference_smoke: output=[%d.%03d, %d.%03d]\n",
			output0_milli / 1000, output0_milli % 1000,
			output1_milli / 1000, output1_milli % 1000);
	printf("ncnn_inference_smoke: output_data=0x%08lx\n",
			(unsigned long)output_address);

	if (output0_milli != 2500 || output1_milli != 0) {
		printf("ncnn_inference_smoke: FAIL unexpected output\n");
		return -7;
	}
	if (!output_in_sdram) {
		printf("ncnn_inference_smoke: FAIL output is not in SDRAM\n");
		return -8;
	}

	printf("ncnn_inference_smoke: PASS dense inference in external SDRAM\n");
	return 0;
}

} // namespace

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;

	printf("ncnn_inference_smoke: dense 3->2 with ReLU\n");

	heap_type_t previous_heap;
	if (mspace_set_heap(HEAP_EXTERN_MEM, &previous_heap) != 0) {
		printf("ncnn_inference_smoke: FAIL external heap unavailable\n");
		return -1;
	}

	const int result = run_inference();
	if (mspace_set_heap(previous_heap, nullptr) != 0 && result == 0) {
		printf("ncnn_inference_smoke: FAIL restoring heap\n");
		return -9;
	}

	return result;
}
