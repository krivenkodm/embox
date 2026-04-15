#include "dfl_decode.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

// ---------------- helpers ----------------
static inline float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

static inline float round_pos(float x)
{
    // embox-friendly "round" for positive values
    return (float)((int)(x + 0.5f));
}

// ---------------- anchors ----------------
//
// Output Mat: (num_anchors, 4) row = [aw, ah, acx, acy]
//
static ncnn::Mat generate_anchors_fpn(int image_w, int image_h)
{
    static const float ASPECT_RATIO[4] = {0.8f, 1.5f, 2.5f, 3.5f};
    static const int   ANCHOR_SIZE[3]  = {32, 64, 128};
    static const int   STRIDE[3]       = {8, 16, 32};

    const float OCTAVE = 2.0f;
    const int   SCALE_PER_OCTAVE = 3;

    int num_anchors = 0;
    for (int li = 0; li < 3; ++li)
    {
        int s = STRIDE[li];
        int gw = (image_w + s - 1) / s;
        int gh = (image_h + s - 1) / s;
        num_anchors += gw * gh * 4 * SCALE_PER_OCTAVE; // 12 per cell
    }

    ncnn::Mat anchors;
    anchors.create(4, num_anchors);

    int anchor_count = 0;

    for (int li = 0; li < 3; ++li)
    {
        int stride      = STRIDE[li];
        int anchor_size = ANCHOR_SIZE[li];

        float w0 = (float)stride;
        float h0 = (float)stride;
        float x_ctr = 0.5f * (w0 - 1.0f);
        float y_ctr = 0.5f * (h0 - 1.0f);
        float size  = w0 * h0;

        float per_aw[12];
        float per_ah[12];
        float per_cx[12];
        float per_cy[12];
        int per_n = 0;

        for (int r = 0; r < 4; ++r)
        {
            float ar = ASPECT_RATIO[r];

            float size_ratio = size / ar;
            float ws = (float)sqrt((double)size_ratio);
            ws = round_pos(ws);
            float hs = ws * ar;
            hs = round_pos(hs);

            for (int oct = 0; oct < SCALE_PER_OCTAVE; ++oct)
            {
                float e = (float)oct / (float)SCALE_PER_OCTAVE; // как на mac
                float area   = (float)anchor_size * (float)pow((double)OCTAVE, (double)e);
                float scales = area / (float)stride;

                per_aw[per_n] = ws * scales;
                per_ah[per_n] = hs * scales;
                per_cx[per_n] = x_ctr + 0.5f;
                per_cy[per_n] = y_ctr + 0.5f;
                ++per_n;
            }
        }

        for (int y = 0; y < image_h; y += stride)
        {
            for (int x = 0; x < image_w; x += stride)
            {
                for (int k = 0; k < per_n; ++k)
                {
                    float* out = anchors.row(anchor_count++);
                    out[0] = per_aw[k];
                    out[1] = per_ah[k];
                    out[2] = (float)x + per_cx[k];
                    out[3] = (float)y + per_cy[k];
                }
            }
        }
    }

    return anchors;
}

// ---------------- decode ----------------
int person_fpn_anchor_decode(const ncnn::Mat& score_blob,
                            const ncnn::Mat& bbox_blob,
                            int in_w, int in_h,
                            float prob_thr,
                            std::vector<Det>& dets)
{
    // --- validate ---
    if (score_blob.empty() || bbox_blob.empty()) return 0;
    if (score_blob.dims != 2 || bbox_blob.dims != 2) return 0;
    if (score_blob.w < 1 || bbox_blob.w != 4) return 0;
    if (score_blob.h != bbox_blob.h) return 0;

    // --- anchors cache for this input size ---
    static int last_w = 0, last_h = 0;
    static ncnn::Mat anchors;
    if (in_w != last_w || in_h != last_h || anchors.empty())
    {
        anchors = generate_anchors_fpn(in_w, in_h);
        last_w = in_w;
        last_h = in_h;
    }

    const int N = score_blob.h;
    if (anchors.h != N) return 0;

    // --- decide if score already probability in [0..1] ---
    bool score_is_prob = false;
    {
        const int M = std::min(N, 512);
        int in01 = 0;
        for (int i = 0; i < M; ++i) {
            float v = score_blob.row(i)[0];
            if (v >= 0.f && v <= 1.f) ++in01;
        }
        score_is_prob = (in01 > (int)(0.9f * M));
    }

    int before = (int)dets.size();
    printf("%d %d", before, N);

    for (int i = 0; i < N; ++i)
    {
        float s = score_blob.row(i)[0];
        float prob = score_is_prob ? s : sigmoidf(s);

        if (i == 24425) {
            printf("%f %f", s, prob);
        }

        if (prob < prob_thr) continue;

        const float* p = bbox_blob.row(i);   // dx dy dw dh
        const float* a = anchors.row(i);     // aw ah acx acy

        // Retina-style decode params (как в оригинале)
        float dx = p[0] * 0.1f;
        float dy = p[1] * 0.1f;
        float dw = p[2] * 0.2f;
        float dh = p[3] * 0.2f;

        float cx = a[2] + a[0] * dx;
        float cy = a[3] + a[1] * dy;
        float w  = a[0] * expf(dw);
        float h  = a[1] * expf(dh);

        Det d;
        d.x = cx - 0.5f * w;
        d.y = cy - 0.5f * h;
        d.w = w;
        d.h = h;
        d.score = prob;   // уже prob
        d.cls = 0;        // person
        dets.push_back(d);
    }

    return (int)dets.size() - before;
}
