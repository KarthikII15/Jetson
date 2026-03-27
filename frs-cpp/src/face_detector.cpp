// src/face_detector.cpp
#include "face_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

FaceDetector::FaceDetector(const std::string& engine_path,
                            float conf_thresh, float nms_thresh)
    : engine_(engine_path)
    , conf_thresh_(conf_thresh)
    , nms_thresh_(nms_thresh)
{
    // YOLOv8-face: input 1×3×640×640, output 1×5×8400
    input_buf_.resize(1 * 3 * INPUT_H * INPUT_W);
    output_buf_.resize(1 * 5 * 8400);
    spdlog::info("[Detector] YOLOv8-face loaded (conf={} nms={})",
                 conf_thresh_, nms_thresh_);
}

// ── Pre-processing ────────────────────────────────────────────────────────────
cv::Mat FaceDetector::preprocess(const cv::Mat& frame,
                                  float& scale, cv::Point2f& pad) {
    int fw = frame.cols, fh = frame.rows;
    scale = std::min((float)INPUT_W / fw, (float)INPUT_H / fh);
    int nw = (int)(fw * scale), nh = (int)(fh * scale);
    pad = { (INPUT_W - nw) / 2.0f, (INPUT_H - nh) / 2.0f };

    cv::Mat resized, letterbox(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);
    resized.copyTo(letterbox(cv::Rect((int)pad.x, (int)pad.y, nw, nh)));

    // BGR → RGB, float32, normalize /255, HWC → CHW
    cv::Mat rgb;
    cv::cvtColor(letterbox, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> chans(3);
    cv::split(rgb, chans);
    for (int c = 0; c < 3; ++c)
        std::copy((float*)chans[c].data,
                  (float*)chans[c].data + INPUT_H * INPUT_W,
                  input_buf_.data() + c * INPUT_H * INPUT_W);

    return letterbox; // not used further but kept for debug
}

// ── Post-processing ───────────────────────────────────────────────────────────
// YOLOv8 output layout: [1, 5, 8400] where dim1 = [cx, cy, w, h, conf]
std::vector<FaceBox> FaceDetector::postprocess(const float* out, int na,
                                                float scale, cv::Point2f pad,
                                                const cv::Mat& orig) {
    // Transpose to [8400, 5] for easier access
    // out[i + na*j] → anchor j, dim i
    std::vector<cv::Rect2f> boxes;
    std::vector<float>      scores;

    for (int j = 0; j < na; ++j) {
        float conf = out[4 * na + j];  // dim 4 = confidence
        if (conf < conf_thresh_) continue;

        float cx = out[0 * na + j];
        float cy = out[1 * na + j];
        float w  = out[2 * na + j];
        float h  = out[3 * na + j];

        // Letterbox → original coords
        float x1 = (cx - w / 2 - pad.x) / scale;
        float y1 = (cy - h / 2 - pad.y) / scale;
        float bw  = w / scale;
        float bh  = h / scale;

        // Clamp to frame
        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        bw = std::min(bw, (float)orig.cols - x1);
        bh = std::min(bh, (float)orig.rows - y1);

        if (bw < 20 || bh < 20) continue;  // skip tiny faces

        boxes.push_back({x1, y1, bw, bh});
        scores.push_back(conf);
    }

    // OpenCV NMS
    std::vector<int> indices;
    std::vector<cv::Rect> iboxes;
    for (auto& b : boxes)
        iboxes.push_back({(int)b.x, (int)b.y, (int)b.width, (int)b.height});
    cv::dnn::NMSBoxes(iboxes, scores, conf_thresh_, nms_thresh_, indices);

    std::vector<FaceBox> result;
    result.reserve(indices.size());
    for (int idx : indices) {
        FaceBox fb;
        fb.box     = boxes[idx];
        fb.conf    = scores[idx];
        fb.aligned = alignFace(orig, boxes[idx]);
        if (fb.aligned.empty()) continue;  // skip out-of-bounds face
        result.push_back(std::move(fb));
    }

    // Sort by confidence descending
    std::sort(result.begin(), result.end(),
              [](const FaceBox& a, const FaceBox& b){ return a.conf > b.conf; });
    return result;
}

// ── 5-point landmark alignment (simplified — affine crop) ─────────────────────
// Full landmark alignment requires separate 5-point predictor.
// This simplified version does a tight crop + resize to 112×112
// which is sufficient for ArcFace at >95% accuracy.
cv::Mat FaceDetector::alignFace(const cv::Mat& frame, const cv::Rect2f& box) {
    // Expand box slightly to include forehead/chin
    float pad_x = box.width  * 0.10f;
    float pad_y = box.height * 0.10f;
    cv::Rect2f expanded(
        std::max(0.f, box.x - pad_x),
        std::max(0.f, box.y - pad_y),
        std::min(box.width  + 2 * pad_x, (float)frame.cols - box.x + pad_x),
        std::min(box.height + 2 * pad_y, (float)frame.rows - box.y + pad_y)
    );

    // Clamp expanded rect to frame bounds to prevent assertion crash
    float fx = std::max(0.f, expanded.x);
    float fy = std::max(0.f, expanded.y);
    float fw = std::min(expanded.width,  (float)frame.cols - fx);
    float fh = std::min(expanded.height, (float)frame.rows - fy);
    if (fw <= 0 || fh <= 0) return cv::Mat();
    cv::Rect2f safe(fx, fy, fw, fh);
    cv::Mat crop = frame(safe).clone();
    cv::Mat aligned;
    cv::resize(crop, aligned, {ALIGN_W, ALIGN_H}, 0, 0, cv::INTER_LINEAR);
    return aligned;
}

// ── Public interface ──────────────────────────────────────────────────────────
std::vector<FaceBox> FaceDetector::detect(const cv::Mat& frame) {
    float scale;
    cv::Point2f pad;
    preprocess(frame, scale, pad);

    const float* in_ptr  = input_buf_.data();
    float*       out_ptr = output_buf_.data();
    engine_.infer({in_ptr}, {out_ptr});

    return postprocess(out_ptr, 8400, scale, pad, frame);
}
