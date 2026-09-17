#ifndef PROJECT_NCNN_LCD_RENDER_H_
#define PROJECT_NCNN_LCD_RENDER_H_
#include <stdint.h>

#define NCNN_LCD_WIDTH 480
#define NCNN_LCD_HEIGHT 272
#define NCNN_LCD_BYTES (NCNN_LCD_WIDTH * NCNN_LCD_HEIGHT * 2)

void ncnn_lcd_render_ready(uint16_t *frame);
void ncnn_lcd_render_colors(uint16_t *frame);
void ncnn_lcd_render_photo(uint16_t *frame, const unsigned char *rgb, int width, int height);
void ncnn_lcd_render_result(uint16_t *frame, const char *label, long milliseconds);
void ncnn_lcd_render_error(uint16_t *frame);
#endif
