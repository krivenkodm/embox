/* Host tool: export the exact firmware fixture to a separate QSPI image. */
#include <stdio.h>
#include <string.h>

#include "../ncnn_conv_smoke/fixture.h"

int main(int argc, char **argv) {
	using namespace ncnn_conv_fixture;
	const uint32_t endian = 1;
	if (argc != 2 || *reinterpret_cast<const unsigned char *>(&endian) != 1) {
		fprintf(stderr, "Usage: export_conv_qspi OUTPUT.bin (little-endian host)\n");
		return 1;
	}
	unsigned char image[kQspiImageSize];
	memset(image, 0xff, sizeof(image));
	memcpy(image, kNetworkParam, sizeof(kNetworkParam));
	memcpy(image + kQspiModelOffset, &kNetworkModel, sizeof(kNetworkModel));
	FILE *file = fopen(argv[1], "wb");
	if (!file) {
		perror(argv[1]);
		return 1;
	}
	const bool written = fwrite(image, 1, sizeof(image), file) == sizeof(image);
	const int closed = fclose(file);
	if (!written || closed != 0) {
		fprintf(stderr, "Failed to write QSPI image\n");
		return 1;
	}
	printf("%u bytes: param=%lu, model=%lu at +%u; flash at 0x%08lx\n",
			kQspiImageSize, (unsigned long)sizeof(kNetworkParam),
			(unsigned long)sizeof(kNetworkModel), kQspiModelOffset,
			(unsigned long)kQspiParamAddress);
	return 0;
}
