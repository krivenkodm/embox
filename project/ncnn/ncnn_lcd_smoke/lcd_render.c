/* CPU-only RGB565 renderer: shared with the native bounds/preview test. */
#include <stddef.h>
#include <stdio.h>
#include <drivers/video/font.h>
#include "lcd_render.h"

#define RGB(r,g,b) ((((r) & 0xf8) << 8) | (((g) & 0xfc) << 3) | ((b) >> 3))
#define BACK RGB(12, 20, 30)
#define PANEL RGB(23, 36, 49)
#define WHITE RGB(240, 246, 250)
#define MUTED RGB(160, 180, 196)
#define GREEN RGB(72, 220, 160)

static void rect(uint16_t *f, int x, int y, int w, int h, uint16_t color) {
	for (int j = y; j < y + h && j < NCNN_LCD_HEIGHT; ++j) {
		for (int i = x; i < x + w && i < NCNN_LCD_WIDTH; ++i) {
			if (i >= 0 && j >= 0) f[j * NCNN_LCD_WIDTH + i] = color;
		}
	}
}

static void text(uint16_t *f, int x, int y, const char *s, int scale, uint16_t color) {
	while (*s && x + 8 * scale <= NCNN_LCD_WIDTH) {
		const unsigned char *glyph = (const unsigned char *)font_vga_8x8.data
				+ (unsigned char)*s++ * 8;
		for (int row = 0; row < 8; ++row) {
			for (int col = 0; col < 8; ++col) {
				if (glyph[row] & (0x80u >> col))
					rect(f, x + col * scale, y + row * scale, scale, scale, color);
			}
		}
		x += 8 * scale;
	}
}

static void base(uint16_t *f) {
	rect(f, 0, 0, 480, 272, BACK);
	text(f, 16, 12, "NCNN", 2, GREEN);
	text(f, 104, 16, "MobileNetV3-Small", 1, WHITE);
	rect(f, 16, 38, 448, 2, PANEL);
	text(f, 16, 252, "STM32F746  /  FP32  /  QSPI", 1, MUTED);
}

void ncnn_lcd_render_ready(uint16_t *f) {
	base(f);
	text(f, 16, 72, "DISPLAY READY", 2, GREEN);
	text(f, 16, 112, "Run from the UART shell:", 1, MUTED);
	text(f, 16, 144, "ncnn_mobilenet_smoke quad photo", 1, WHITE);
	text(f, 16, 200, "Photo inference runs on this board.", 1, MUTED);
}

void ncnn_lcd_render_colors(uint16_t *f) {
	const uint16_t colors[] = {0xf800, 0x07e0, 0x001f, 0xffe0, 0x07ff, 0xf81f, 0, 0xffff};
	for (int i = 0; i < 8; ++i) rect(f, i * 60, 0, 60, 272, colors[i]);
}

void ncnn_lcd_render_photo(uint16_t *f, const unsigned char *rgb, int width, int height) {
	base(f);
	if (!rgb || width <= 0 || width > 512 || height <= 0 || height > 512) return;
	/* Preserve the original photo's aspect ratio for display only. Inference
	 * still uses the separately checked NCNN 224x224 bilinear square input. */
	int w = 224, h = height * 224 / width;
	if (h > 184) { h = 184; w = width * 184 / height; }
	if (h == 0 || w == 0) return;
	const int x = 16 + (224 - w) / 2, y = 52 + (184 - h) / 2;
	rect(f, 12, 48, 232, 192, PANEL);
	for (int j = 0; j < h; ++j) {
		for (int i = 0; i < w; ++i) {
			const unsigned char *p = rgb + ((j * height / h) * width + i * width / w) * 3;
			f[(y+j)*480 + x+i] = RGB(p[0], p[1], p[2]);
		}
	}
	text(f, 260, 56, "CAT2.JPG", 1, MUTED);
	text(f, 260, 88, "RUNNING...", 2, WHITE);
	text(f, 260, 120, "224 x 224 input", 1, MUTED);
	text(f, 260, 144, "Please wait", 1, MUTED);
}

void ncnn_lcd_render_result(uint16_t *f, const char *label, long milliseconds) {
	char timing[32];
	rect(f, 252, 76, 228, 164, BACK);
	/* The known fixture's top label fits; other labels use small text. */
	size_t length = 0;
	while (label[length]) ++length;
	text(f, 260, 88, label, length <= 13 ? 2 : 1, WHITE);
	snprintf(timing, sizeof(timing), "%ld.%03ld s inference", milliseconds / 1000, milliseconds % 1000);
	text(f, 260, 128, timing, 1, MUTED);
	text(f, 260, 164, "1000 outputs checked", 1, MUTED);
	text(f, 260, 200, "PASS", 2, GREEN);
}

void ncnn_lcd_render_error(uint16_t *f) {
	rect(f, 252, 76, 228, 164, BACK);
	text(f, 260, 88, "FAILED", 2, RGB(255, 108, 108));
	text(f, 260, 128, "See UART log", 1, MUTED);
}
