#pragma once

#include "platform.h"  // дает simplestl и std::vector (один шаблонный параметр)
#include "mat.h"       // ncnn::Mat

struct Det
{
    float x, y, w, h;
    float score;
    int   cls;
};

int person_fpn_anchor_decode(const ncnn::Mat& score_blob,
                            const ncnn::Mat& bbox_blob,
                            int in_w, int in_h,
                            float prob_thr,
                            std::vector<Det>& dets);
