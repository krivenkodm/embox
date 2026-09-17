#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "../ncnn_lcd_smoke/lcd_render.h"
static void check(const uint16_t *mem) {
 for (int i=0;i<16;i++) assert(mem[i]==0xdead && mem[16+480*272+i]==0xdead);
}
int main(int argc,char **argv) {
 if(argc < 3 || argc > 4) { fprintf(stderr,"Usage: %s decoded.rgb output.rgb565 [inference_ms]\n",argv[0]); return 1; }
 uint16_t *mem=malloc((480*272+32)*2);assert(mem);
 for(int i=0;i<480*272+32;i++)mem[i]=0xdead;
 uint16_t *f=mem+16;
 ncnn_lcd_render_ready(f);check(mem);
 ncnn_lcd_render_colors(f);check(mem);
 const uint16_t colors[]={0xf800,0x07e0,0x001f,0xffe0,0x07ff,0xf81f,0,0xffff};
 for(int y=0;y<272;y++)for(int x=0;x<480;x++)assert(f[y*480+x]==colors[x/60]);
 unsigned char *rgb=malloc(295*231*3);assert(rgb);
 FILE *in=fopen(argv[1],"rb");assert(in);
 assert(fread(rgb,1,295*231*3,in)==295*231*3);fclose(in);
 ncnn_lcd_render_photo(f,NULL,0,0);check(mem);
 ncnn_lcd_render_photo(f,rgb,295,231);check(mem);
 ncnn_lcd_render_error(f);check(mem);
 ncnn_lcd_render_result(f,"An intentionally long label to exercise clipping at the screen border",14815);check(mem);
 ncnn_lcd_render_photo(f,rgb,295,231);
 ncnn_lcd_render_result(f,"Egyptian cat",argc>3?atol(argv[3]):14815);check(mem);
 FILE *raw=fopen(argv[2],"wb");assert(raw);assert(fwrite(f,2,480*272,raw)==480*272);fclose(raw);
 free(rgb);free(mem);puts("PASS ASan/UBSan renderer, exact 8 color bars, bounds guards, photo, result, error, long label");
}
