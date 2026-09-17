#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "net.h"
#include "datareader.h"
#include "layer.h"
#include "modelbin.h"

#include "../ncnn_mobilenet_smoke/input.h"
#include "../ncnn_mobilenet_smoke/bounded_reader.h"

using namespace mobilenet_fixture;
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
void options(ncnn::Net &net) {
    net.opt.num_threads = 1;
    net.opt.lightmode = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_int8_inference = false;
}
class ExportFP32 : public ncnn::ModelBin {
    ncnn::ModelBinFromDataReader source;
public:
    mutable Bytes output;
    mutable unsigned tagged = 0, raw = 0;
    explicit ExportFP32(const ncnn::DataReader &dr): source(dr) {}
    ncnn::Mat load(int w, int type) const override {
        if (type != 0 && type != 1) throw std::runtime_error("unsupported weight type");
        ncnn::Mat m = source.load(w, type);
        if (m.empty() || m.elemsize != 4 || m.w != w) throw std::runtime_error("weight load failed");
        if (type == 0) { output.insert(output.end(), 4, 0); tagged++; } else raw++;
        auto ptr = static_cast<const unsigned char *>(m.data);
        output.insert(output.end(), ptr, ptr + size_t(w) * 4);
        return m;
    }
};
std::vector<float> infer(const Bytes &param, const Bytes &model) {
    ncnn::Net net; options(net);
    BoundedReader parameters(param.data(), param.size()), weights(model.data(), model.size());
    if (net.load_param_bin(parameters) != 0 || !parameters.complete()
        || net.load_model(weights) != 0 || !weights.complete())
        throw std::runtime_error("model not consumed exactly");
    ncnn::Mat input(kInputSize, kInputSize, kInputChannels);
    for (int q = 0; q < kInputChannels; q++) {
        float *channel = input.channel(q);
        for (int y = 0; y < kInputSize; y++) for (int x = 0; x < kInputSize; x++)
            channel[y * kInputSize + x] = input_value(q, y, x);
    }
    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat out;
    if (ex.input(kInputBlob, input) != 0 || ex.extract(kOutputBlob, out) != 0 || out.dims != 1 || out.w != kOutputSize || out.elempack != 1 || out.elemsize != 4)
        throw std::runtime_error("inference/shape failed");
    const float *values = out;
    return std::vector<float>(values, values + kOutputSize);
}
int main(int argc, char **argv) {
    if (argc != 2) return 1;
    try {
        const uint32_t endian = 1;
        if (*reinterpret_cast<const uint8_t *>(&endian) != 1) throw std::runtime_error("little-endian host required");
        std::string dir(argv[1]);
        Bytes param = read_file(dir + "/model.param.bin"), original = read_file(dir + "/model.bin");
        ncnn::Net net; options(net);
        if (net.load_param(param.data()) != int(param.size())) throw std::runtime_error("param load failed");
        const unsigned char *ptr = original.data();
        ncnn::DataReaderFromMemory dr(ptr);
        ExportFP32 exporter(dr);
        for (auto *layer : net.layers()) {
            if (layer->load_model(exporter) != 0) throw std::runtime_error("layer load failed");
        }
        if (ptr != original.data() + original.size()) throw std::runtime_error("source weights not consumed exactly");
        write_file(dir + "/model-fp32.bin", exporter.output.data(), exporter.output.size());
        auto reference = infer(param, original);
        auto converted = infer(param, exporter.output);
        float worst = 0.f;
        for (size_t i = 0; i < reference.size(); i++) {
            if (!std::isfinite(reference[i]) || !std::isfinite(converted[i])) throw std::runtime_error("nonfinite output");
            worst = std::max(worst, std::fabs(reference[i] - converted[i]));
        }
        if (worst != 0.f) throw std::runtime_error("FP32 storage conversion changed inference");
        write_file(dir + "/reference-fp32.bin", reference.data(), reference.size() * 4);
        std::vector<int> ranks(kOutputSize);
        for (int i = 0; i < kOutputSize; i++) ranks[i] = i;
        std::sort(ranks.begin(), ranks.end(), [&](int a, int b) { return reference[a] > reference[b]; });
        FILE *stats = fopen((dir + "/conversion.json").c_str(), "w");
        if (!stats) throw std::runtime_error("conversion metadata");
        fprintf(stats, "{\"tagged_blobs\":%u,\"raw_blobs\":%u,\"layers\":%zu}\n",exporter.tagged,exporter.raw,net.layers().size());
        if (fclose(stats) != 0) throw std::runtime_error("metadata close");
        printf("layers=%zu original=%zu FP32=%zu tagged=%u raw=%u; 1000 logits identical\n",net.layers().size(),original.size(),exporter.output.size(),exporter.tagged,exporter.raw);
        for (int i = 0; i < 5; i++) printf("top%d class=%d logit=%.9g\n",i+1,ranks[i],reference[ranks[i]]);
        FILE *header = fopen((dir + "/reference.h").c_str(), "w");
        if (!header) throw std::runtime_error("reference header");
        fprintf(header,"// Generated by prepare_mobilenet_qspi.cpp; see the template README.\n#ifndef PROJECT_NCNN_MOBILENET_REFERENCE_H_\n#define PROJECT_NCNN_MOBILENET_REFERENCE_H_\nnamespace mobilenet_fixture {\nconstexpr float kReference[1000] = {\n");
        for (size_t i = 0; i < reference.size(); i++) fprintf(header,"%.9e%s%s",double(reference[i]),"f,",i%5==4?"\n":" ");
        fprintf(header,"};\n} // namespace mobilenet_fixture\n#endif\n");
        if (fclose(header) != 0) throw std::runtime_error("reference close");
    } catch (const std::exception &e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
