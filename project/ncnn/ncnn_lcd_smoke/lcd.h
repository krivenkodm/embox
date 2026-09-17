#ifndef PROJECT_NCNN_LCD_H_
#define PROJECT_NCNN_LCD_H_
/* BSP state lives in a static module, never in the App dependency closure. */
int ncnn_lcd_show_photo(const unsigned char *rgb, int width, int height);
int ncnn_lcd_show_result(const char *label, long milliseconds);
void ncnn_lcd_show_error(void);
int ncnn_lcd_check(void);
int ncnn_lcd_test(int colors);
#endif
