#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <net.h>

#include "../ncnn_mobilenet_smoke/bounded_reader.h"
#include "../ncnn_mobilenet_smoke/photo_input.h"
#include "../models/mobilenetv3_small/imagenet1000_labels.h"

using Bytes = std::vector<unsigned char>;
Bytes read_file(const std::string &path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("open " + path);
	return Bytes(std::istreambuf_iterator<char>(f), {});
}
void write_file(const std::string &path, const void *data, size_t size) {
	std::ofstream f(path, std::ios::binary);
	if (!f.write(static_cast<const char *>(data), size)) throw std::runtime_error("write " + path);
}
std::vector<float> infer(const Bytes &param, const Bytes &model, const ncnn::Mat &input) {
	ncnn::Net net;
	net.opt.num_threads = 1;
	net.opt.lightmode = true;
	net.opt.use_packing_layout = false;
	net.opt.use_fp16_packed = false;
	net.opt.use_fp16_storage = false;
	net.opt.use_fp16_arithmetic = false;
	net.opt.use_bf16_storage = false;
	net.opt.use_int8_inference = false;
	BoundedReader parameters(param.data(), param.size()), weights(model.data(), model.size());
	if (net.load_param_bin(parameters) || !parameters.complete()
			|| net.load_model(weights) || !weights.complete()) throw std::runtime_error("bounded model load");
	ncnn::Extractor ex = net.create_extractor();
	ncnn::Mat out;
	if (ex.input(0, input) || ex.extract(152, out) || out.dims != 1 || out.w != 1000
			|| out.elemsize != 4 || out.elempack != 1) throw std::runtime_error("extract/shape");
	const float *values = out;
	return std::vector<float>(values, values + 1000);
}
int main(int argc, char **argv) {
	if (argc != 4) return 1;
	try {
		static_assert(sizeof(float) == 4, "32-bit floats required");
		const uint32_t endian = 1;
		if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
			throw std::runtime_error("little-endian host required");
		const std::string model_dir = argv[1], out_dir = argv[3];
		Bytes jpeg = read_file(argv[2]);
		unsigned char *rgb = mobilenet_photo::decode_rgb(jpeg.data(), jpeg.size());
		if (!rgb) throw std::runtime_error("JPEG decode/dimensions");
		write_file(out_dir + "/decoded.rgb", rgb, mobilenet_photo::kRgbBytes);
		ncnn::Mat input = mobilenet_photo::prepare_input(rgb);
		mobilenet_photo::free_rgb(rgb);
		if (input.empty() || input.total() * input.elemsize != mobilenet_photo::kInputBytes)
			throw std::runtime_error("input shape/packing");
		write_file(out_dir + "/input-fp32.bin", input.data, mobilenet_photo::kInputBytes);
		Bytes param = read_file(model_dir + "/model.param.bin");
		auto original = infer(param, read_file(model_dir + "/model.bin"), input);
		auto reference = infer(param, read_file(model_dir + "/model-fp32.bin"), input);
		for (int i = 0; i < 1000; ++i) {
			if (!std::isfinite(original[i]) || !std::isfinite(reference[i]) || original[i] != reference[i])
				throw std::runtime_error("FP32-storage conversion changed photo inference");
		}
		write_file(out_dir + "/reference-fp32.bin", reference.data(), reference.size() * 4);
		std::vector<int> ranks(1000);
		for (int i = 0; i < 1000; ++i) ranks[i] = i;
		std::sort(ranks.begin(), ranks.end(), [&](int a, int b) { return reference[a] > reference[b]; });
		for (int i = 0; i < 5; ++i) printf("top%d class=%d logit=%.9g label=%s\n",
				i + 1, ranks[i], reference[ranks[i]], mobilenetv3_small_label_name(ranks[i]));
		printf("PASS photo original/FP32-storage: 1000 logits identical\n");
	} catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
