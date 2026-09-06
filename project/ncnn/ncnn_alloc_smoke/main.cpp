#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <mat.h>

extern "C" {
#include <mem/heap/mspace_malloc.h>
}

namespace {

constexpr uintptr_t kSdramStart = 0x60000000u;
constexpr uintptr_t kSdramEnd = 0x60800000u;
constexpr int kWidth = 256;
constexpr int kHeight = 256;
constexpr int kChannels = 3;

} // namespace

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;

	heap_type_t previous_heap;
	if (mspace_set_heap(HEAP_EXTERN_MEM, &previous_heap) != 0) {
		printf("ncnn_alloc_smoke: FAIL external heap unavailable\n");
		return -1;
	}

	int result = 0;
	{
		ncnn::Mat tensor(kWidth, kHeight, kChannels, sizeof(float));
		if (tensor.empty()) {
			printf("ncnn_alloc_smoke: FAIL Mat allocation\n");
			result = -2;
		} else {
			const size_t bytes = tensor.total() * tensor.elemsize;
			unsigned char *data = static_cast<unsigned char *>(tensor.data);
			uint32_t checksum = 0;

			for (size_t i = 0; i < bytes; ++i) {
				data[i] = static_cast<unsigned char>((i * 13u + 7u) & 0xffu);
				checksum += data[i];
			}

			const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data);
			const bool in_sdram = address >= kSdramStart
					&& address < kSdramEnd
					&& bytes <= kSdramEnd - address;

			printf("ncnn_alloc_smoke: Mat %dx%dx%d, %lu bytes\n",
					kWidth, kHeight, kChannels,
					(unsigned long)bytes);
			printf("ncnn_alloc_smoke: data=0x%08lx checksum=%lu\n",
					(unsigned long)address, (unsigned long)checksum);

			if (!in_sdram) {
				printf("ncnn_alloc_smoke: FAIL buffer is not in SDRAM\n");
				result = -3;
			} else {
				printf("ncnn_alloc_smoke: PASS external SDRAM\n");
			}
		}
	}

	if (mspace_set_heap(previous_heap, nullptr) != 0 && result == 0) {
		printf("ncnn_alloc_smoke: FAIL restoring heap\n");
		return -4;
	}

	return result;
}
