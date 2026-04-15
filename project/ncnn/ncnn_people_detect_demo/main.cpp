#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "platform.h"

#include "net.h"
#include "mat.h"
#include "datareader.h"

#include "mobile0_35x_model/mobile0_35xFPNdw.id.h"
#include "mobile0_35x_model/mobile0_35xFPNdw.mem.h"

#include "dfl_decode.h"
#include "nms.h"
#include "image_io.h"

static inline int iround(float x) { return (int)(x + (x >= 0.f ? 0.5f : -0.5f)); }
static inline float fmaxf2(float a, float b){ return a > b ? a : b; }
static inline float fminf2(float a, float b){ return a < b ? a : b; }

static int ieq(const char* a, const char* b)
{
    while (*a && *b)
    {
        char ca = (char)tolower((unsigned char)*a);
        char cb = (char)tolower((unsigned char)*b);
        if (ca != cb) return 0;
        ++a; ++b;
    }
    return *a == 0 && *b == 0;
}

static void pick_pixel_type_and_order(const char* PIX, int& pixel_type, bool& net_expects_bgr)
{
    // load_image_rgb() возвращает RGB в памяти.
    // PIX говорит, какой порядок ожидает сеть: RGB или BGR.
    if (PIX && ieq(PIX, "BGR"))
    {
        pixel_type = ncnn::Mat::PIXEL_RGB2BGR; // конвертируем RGB -> BGR
        net_expects_bgr = true;
        return;
    }

    // по умолчанию RGB
    pixel_type = ncnn::Mat::PIXEL_RGB;
    net_expects_bgr = false;
}


static inline void clip_box(Det& d, int W, int H)
{
    float x1 = d.x;
    float y1 = d.y;
    float x2 = d.x + d.w;
    float y2 = d.y + d.h;

    if (x1 < 0.f) x1 = 0.f;
    if (y1 < 0.f) y1 = 0.f;
    if (x2 > (float)(W - 1)) x2 = (float)(W - 1);
    if (y2 > (float)(H - 1)) y2 = (float)(H - 1);

    d.x = x1; d.y = y1;
    d.w = x2 - x1; if (d.w < 0.f) d.w = 0.f;
    d.h = y2 - y1; if (d.h < 0.f) d.h = 0.f;
}

static void sort_dets_by_score_desc(std::vector<Det>& dets)
{
    // insertion sort: simplestl-friendly
    for (size_t i = 1; i < dets.size(); ++i)
    {
        Det key = dets[i];
        float s = key.score;
        size_t j = i;
        while (j > 0 && dets[j - 1].score < s)
        {
            dets[j] = dets[j - 1];
            --j;
        }
        dets[j] = key;
    }
}

static void draw_rect_rgb(unsigned char* rgb, int W, int H, int x, int y, int w, int h, int t=3)
{
#define PIX(xx,yy) (rgb + ((yy)*(W) + (xx))*3)
    int x0 = x < 0 ? 0 : x;
    int y0 = y < 0 ? 0 : y;
    int x1 = x + w - 1; if (x1 >= W) x1 = W - 1;
    int y1 = y + h - 1; if (y1 >= H) y1 = H - 1;
    if (x0 > x1 || y0 > y1) return;

    for (int k = 0; k < t; ++k)
    {
        for (int xx = x0; xx <= x1; ++xx) { unsigned char* p = PIX(xx, y0 + k); p[0]=255; p[1]=0; p[2]=0; }
        for (int xx = x0; xx <= x1; ++xx) { unsigned char* p = PIX(xx, y1 - k); p[0]=255; p[1]=0; p[2]=0; }
        for (int yy = y0; yy <= y1; ++yy) { unsigned char* p = PIX(x0 + k, yy); p[0]=255; p[1]=0; p[2]=0; }
        for (int yy = y0; yy <= y1; ++yy) { unsigned char* p = PIX(x1 - k, yy); p[0]=255; p[1]=0; p[2]=0; }
    }
#undef PIX
}

struct LetterboxInfo { int nw, nh, pad_x, pad_y; float scale; };

static std::vector<unsigned char>
make_letterbox_rgb(const unsigned char* src_rgb, int w, int h, int S, LetterboxInfo& L)
{
    float rw = (float)S / (float)w;
    float rh = (float)S / (float)h;
    float r  = (rw < rh) ? rw : rh;

    L.nw    = iround(w * r);
    L.nh    = iround(h * r);
    L.pad_x = (S - L.nw) / 2;
    L.pad_y = (S - L.nh) / 2;
    L.scale = r;

    ncnn::Mat tmp = ncnn::Mat::from_pixels_resize(src_rgb, ncnn::Mat::PIXEL_RGB, w, h, L.nw, L.nh);

    std::vector<unsigned char> resized;
    resized.resize((size_t)L.nw * L.nh * 3);
    tmp.to_pixels(resized.data(), ncnn::Mat::PIXEL_RGB);

    std::vector<unsigned char> lb;
    lb.resize((size_t)S * S * 3);
    for (size_t i = 0; i < lb.size(); ++i) lb[i] = (unsigned char)114;

    for (int y = 0; y < L.nh; ++y)
    {
        unsigned char*       dst = lb.data() + ((y + L.pad_y) * S + L.pad_x) * 3;
        const unsigned char* src = resized.data() + (size_t)y * L.nw * 3;
        memcpy(dst, src, (size_t)L.nw * 3);
    }
    return lb;
}

static void pick_pixel_type_like_mac(const char* PIX, int& pixel_type)
{
    pixel_type = ncnn::Mat::PIXEL_RGB; // default: keep RGB

    if (!PIX) return;

    // как в mac-версии: "RGB" означает: вход RGB, сеть ожидает BGR => конвертим
    if (ieq(PIX, "RGB"))
        pixel_type = ncnn::Mat::PIXEL_RGB2BGR;
    else if (ieq(PIX, "BGR"))
        pixel_type = ncnn::Mat::PIXEL_BGR; // вход уже BGR
    // любое другое (включая "RBG") => PIXEL_RGB (никакой конверсии)
}


static void log_mat(const char* tag, const ncnn::Mat& m)
{
    printf("[%s] Mat: w=%d h=%d c=%d dims=%d elemsize=%zu elempack=%d cstep=%zu data=%p\n",
           tag, m.w, m.h, m.c, m.dims, (size_t)m.elemsize, m.elempack, (size_t)m.cstep, m.data);
}

int main(int argc, char** argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char* img  = (argc > 1) ? argv[1] : "/data/photos/person.ppm";
    const int   S    = (argc > 2) ? atoi(argv[2]) : 320;
    const float CONF = (argc > 3) ? (float)atof(argv[3]) : 0.20f;
    const int   PREP = (argc > 4) ? atoi(argv[4]) : 0;     // 0=caffe mean, 1=/255, 2=(-1..1)
    const char* PIX  = (argc > 5) ? argv[5] : "RGB";       // RGB or BGR
    const float IOU  = (argc > 6) ? (float)atof(argv[6]) : 0.35f;
    const int   TOPK = (argc > 7) ? atoi(argv[7]) : 10;
    const int   GEOM = (argc > 8) ? atoi(argv[8]) : 0;     // 0=resize, 1=letterbox

    printf("[1] img=%s S=%d conf=%.3f preproc=%d pix=%s iou=%.2f topk=%d geom=%d\n",
           img, S, (double)CONF, PREP, PIX, (double)IOU, TOPK, GEOM);

    // ---- load image ----
    std::vector<unsigned char> rgb;
    int w0 = 0, h0 = 0;
    if (!load_image_rgb(img, rgb, w0, h0))
    {
        printf("[E] load_image_rgb failed\n");
        return -2;
    }
    printf("[1] ok: %dx%d bytes=%zu\n", w0, h0, rgb.size());

    // ---- load net ----
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    net.opt.lightmode = true;
    net.opt.use_winograd_convolution = false;
    net.opt.use_sgemm_convolution = false;
    net.opt.use_packing_layout = false;

    const unsigned char* pparam = mobile0_35xFPNdw_param_bin;
    const unsigned char* pmodel = mobile0_35xFPNdw_bin;

    ncnn::DataReaderFromMemory pr(pparam);
    ncnn::DataReaderFromMemory mr(pmodel);

    int r1 = net.load_param_bin(pr);
    int r2 = (r1 == 0) ? net.load_model(mr) : -999;
    printf("[2] load: param=%d model=%d\n", r1, r2);
    if (r1 != 0 || r2 != 0) return -3;

    // ---- preprocess ----
    ncnn::Mat in;
    LetterboxInfo LI;
    LI.nw = LI.nh = LI.pad_x = LI.pad_y = 0;
    LI.scale = 1.f;

    // выбираем один раз: и для resize, и для letterbox
    int pixel_type = ncnn::Mat::PIXEL_RGB;
    pick_pixel_type_like_mac(PIX, pixel_type);

    if (GEOM == 0)
        in = ncnn::Mat::from_pixels_resize(rgb.data(), pixel_type, w0, h0, S, S);
    else {
        std::vector<unsigned char> lb = make_letterbox_rgb(rgb.data(), w0, h0, S, LI);
        in = ncnn::Mat::from_pixels(lb.data(), pixel_type, S, S);
    }

    if (PREP == 0) {
        const float mean_vals[3] = {102.9801f, 115.9465f, 122.7717f}; // BGR mean (как на mac)
        in.substract_mean_normalize(mean_vals, 0);
    }


    // ---- inference ----
    using namespace mobile0_35xFPNdw_param_id;

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);

    int ri = ex.input(BLOB_0, in);

    ncnn::Mat out_a, out_b;
    int ra = (ri == 0) ? ex.extract(BLOB_545, out_a) : -999;
    int rb = (ri == 0) ? ex.extract(BLOB_546, out_b) : -999;

    printf("[4] rc: input=%d a=%d b=%d\n", ri, ra, rb);
    log_mat("out_a", out_a);
    log_mat("out_b", out_b);
    if (ri != 0 || ra != 0 || rb != 0) return -6;


    // detect which is bbox
    ncnn::Mat bbox, score;
    if (out_a.dims == 2 && out_a.w == 4) { bbox = out_a; score = out_b; }
    else if (out_b.dims == 2 && out_b.w == 4) { bbox = out_b; score = out_a; }
    else {
        printf("[E] can't find bbox tensor (dims==2 && w==4)\n");
        return -7;
    }
    printf("24415 row %f\n", score.row(24425)[0]);

    printf("[4] bbox:  w=%d h=%d\n", bbox.w, bbox.h);
    printf("[4] score: w=%d h=%d\n", score.w, score.h);

    // ---- decode ----
    std::vector<Det> dets;

    int added = person_fpn_anchor_decode(score, bbox, S, S, CONF, dets);
    printf("[5] decoded added=%d dets=%zu\n", added, dets.size());
    if (dets.empty())
    {
        printf("[5] no dets.\n");
        return 0;
    }

    // clip, sort
    for (size_t i = 0; i < dets.size(); ++i) clip_box(dets[i], S, S);
    sort_dets_by_score_desc(dets);

    // NMS
    std::vector<int> keep;
    keep.resize(dets.size());

    int k = nms(dets.data(), (int)dets.size(), IOU, keep.data(), (int)keep.size());
    printf("[5] nms keep=%d\n", k);

    // map to original coords
    auto map_to_orig = [&](Det d)->Det
    {
        if (GEOM == 0)
        {
            float sx = (float)w0 / (float)S;
            float sy = (float)h0 / (float)S;
            d.x *= sx; d.w *= sx;
            d.y *= sy; d.h *= sy;
            clip_box(d, w0, h0);
            return d;
        }
        else
        {
            float x0r = (d.x - (float)LI.pad_x) / LI.scale;
            float y0r = (d.y - (float)LI.pad_y) / LI.scale;
            float x1r = (d.x + d.w - (float)LI.pad_x) / LI.scale;
            float y1r = (d.y + d.h - (float)LI.pad_y) / LI.scale;

            d.x = x0r; d.y = y0r;
            d.w = x1r - x0r;
            d.h = y1r - y0r;
            clip_box(d, w0, h0);
            return d;
        }
    };

    // draw TOPK
    int drawn = 0;
    for (int i = 0; i < k && drawn < (TOPK > 0 ? TOPK : 1); ++i)
    {
        Det d = map_to_orig(dets[keep[i]]);
        printf("  person conf=%.3f x=%.1f y=%.1f w=%.1f h=%.1f\n",
               (double)d.score, (double)d.x, (double)d.y, (double)d.w, (double)d.h);

        drawn++;
    }
    return 0;
}
