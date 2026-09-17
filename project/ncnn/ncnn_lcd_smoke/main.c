#include <stdio.h>
#include <string.h>
#include "lcd.h"

int main(int argc, char **argv) {
	if (argc > 2 || (argc == 2 && strcmp(argv[1], "colors") && strcmp(argv[1], "ready"))) {
		printf("Usage: ncnn_lcd_smoke [colors|ready]\n");
		return -1;
	}
	const int result = ncnn_lcd_test(argc == 2 && !strcmp(argv[1], "colors"));
	if (!result) printf("ncnn_lcd_smoke: PASS display buffer and LTDC\n");
	return result;
}
