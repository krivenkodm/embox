#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <embox/unit.h>
#include <lib/crypt/crc32.h>
#include <mem/page.h>
#include "stm32746g_discovery_lcd.h"
#include "lcd.h"
#include "lcd_render.h"

#define FRAME_BASE 0x60000000u
#define RESERVED_BYTES 0x40000u
#define HEAP_START (FRAME_BASE + RESERVED_BYTES)
#define HEAP_END 0x60800000u

extern LTDC_HandleTypeDef hLtdcHandler;
extern struct page_allocator *__heap_fixed_pgallocator;
static uint16_t *const frame = (uint16_t *)FRAME_BASE;
static uint32_t expected_crc;
static int ready;

static uint32_t frame_crc(void) {
	return count_crc32((unsigned char *)frame, (unsigned char *)frame + NCNN_LCD_BYTES);
}

static void presented(void) {
	__DSB(); /* The template maps only this reserved framebuffer uncached. */
	expected_crc = frame_crc();
}

static int guard_ok(void) {
	const unsigned char *guard = (const unsigned char *)FRAME_BASE + NCNN_LCD_BYTES;
	for (unsigned i = 0; i < RESERVED_BYTES - NCNN_LCD_BYTES; ++i) {
		if (guard[i] != 0xa5) return 0;
	}
	return 1;
}

int ncnn_lcd_check(void) {
	const uint32_t errors = LTDC->ISR & (LTDC_ISR_FUIF | LTDC_ISR_TERRIF);
	if (!ready || !guard_ok() || frame_crc() != expected_crc || errors
			|| SystemCoreClock != 216000000u || !(LTDC->GCR & LTDC_GCR_LTDCEN)
			|| LTDC_Layer1->CFBAR != FRAME_BASE || LTDC_Layer1->PFCR != LTDC_PIXEL_FORMAT_RGB565) {
		printf("ncnn_lcd: FAIL state/frame/guard errors=0x%lx clock=%lu\n",
				(unsigned long)errors, (unsigned long)SystemCoreClock);
		return -1;
	}
	printf("ncnn_lcd: frame CRC=%08lx guard OK LTDC errors=0 clock=%lu\n",
			(unsigned long)expected_crc, (unsigned long)SystemCoreClock);
	return 0;
}

static int lcd_init(void) {
	LCD_LayerCfgTypeDef layer = {0};
	/* Validate the heap boundary before touching the reserved frame. */
	if (!__heap_fixed_pgallocator || (uintptr_t)__heap_fixed_pgallocator < HEAP_START
			|| (uintptr_t)__heap_fixed_pgallocator >= HEAP_END) return -1;
	if (BSP_LCD_Init() != LCD_OK || HAL_LTDC_GetState(&hLtdcHandler) != HAL_LTDC_STATE_READY) return -1;
	/* Static dashboard: poll latched errors, no LTDC interrupt handler needed. */
	__HAL_LTDC_DISABLE_IT(&hLtdcHandler, LTDC_IT_TE | LTDC_IT_FU | LTDC_IT_LI | LTDC_IT_RR);
	layer.WindowX1 = NCNN_LCD_WIDTH;
	layer.WindowY1 = NCNN_LCD_HEIGHT;
	layer.PixelFormat = LTDC_PIXEL_FORMAT_RGB565;
	layer.FBStartAdress = FRAME_BASE;
	layer.Alpha = 255;
	layer.BlendingFactor1 = LTDC_BLENDING_FACTOR1_PAxCA;
	layer.BlendingFactor2 = LTDC_BLENDING_FACTOR2_PAxCA;
	layer.ImageWidth = NCNN_LCD_WIDTH;
	layer.ImageHeight = NCNN_LCD_HEIGHT;
	memset((unsigned char *)frame + NCNN_LCD_BYTES, 0xa5, RESERVED_BYTES - NCNN_LCD_BYTES);
	ncnn_lcd_render_ready(frame);
	presented();
	if (HAL_LTDC_ConfigLayer(&hLtdcHandler, &layer, 0) != HAL_OK) return -1;
	BSP_LCD_SetLayerVisible(1, DISABLE);
	BSP_LCD_DisplayOn();
	__HAL_LTDC_CLEAR_FLAG(&hLtdcHandler, LTDC_FLAG_TE | LTDC_FLAG_FU);
	ready = 1;
	printf("ncnn_lcd: RGB565 480x272 frame=0x%08lx bytes=%u reserved=%u heap=0x%08lx\n",
			(unsigned long)FRAME_BASE, NCNN_LCD_BYTES, RESERVED_BYTES, (unsigned long)HEAP_START);
	return ncnn_lcd_check();
}
EMBOX_UNIT_INIT(lcd_init);

int ncnn_lcd_show_photo(const unsigned char *rgb, int width, int height) {
	if (!ready || !rgb || width != 295 || height != 231 || !guard_ok() || frame_crc() != expected_crc) return -1;
	ncnn_lcd_render_photo(frame, rgb, width, height);
	presented();
	return ncnn_lcd_check();
}

int ncnn_lcd_show_result(const char *label, long milliseconds) {
	if (ncnn_lcd_check() || !label || milliseconds < 0) return -1;
	ncnn_lcd_render_result(frame, label, milliseconds);
	presented();
	return ncnn_lcd_check();
}

void ncnn_lcd_show_error(void) {
	if (ready) { ncnn_lcd_render_error(frame); presented(); }
}

int ncnn_lcd_test(int colors) {
	if (ncnn_lcd_check()) return -1;
	if (colors) ncnn_lcd_render_colors(frame);
	else ncnn_lcd_render_ready(frame);
	presented();
	return ncnn_lcd_check();
}
