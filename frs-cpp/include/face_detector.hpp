#pragma once
// include/face_detector.hpp
// YOLOv8n-face TensorRT FP16 inference
// Input:  1×3×640×640  float32 (normalized, BGR→RGB)
// Output: 1×5×8400     float32 [cx,cy,w,h,conf] per anchor

#include "trt_engine.hpp"
#include <opencv2/core.hpp>
#include <vector>

struct FaceBox {
    cv::Rect2f box;       // pixel coords in original frame
    float      conf;
    cv::Mat    aligned;   // 112×112 aligned face crop (for ArcFace)
};

class FaceDetector {
public:
    explicit FaceDetector(const std::string& engine_path,
                          float conf_thresh = 0.45f,
                          float nms_thresh  = 0.45f);

    // Detect faces in frame, return sorted by confidence descending
    std::vector<FaceBox> detect(const cv::Mat& bgr_frame);

    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int ALIGN_W = 112;
    static constexpr int ALIGN_H = 112;

private:
    TRTEngine engine_;
    float     conf_thresh_;
    float     nms_thresh_;

    // Pre-process: letterbox resize + normalize
    cv::Mat  preprocess(const cv::Mat& frame, float& scale, cv::Point2f& pad);

    // Post-process: decode + NMS
    std::vector<FaceBox> postprocess(const float* output, int num_anchors,
                                      float scale, cv::Point2f pad,
                                      const cv::Mat& orig_frame);

    // 5-point landmark alignment → 112×112 crop
    static cv::Mat alignFace(const cv::Mat& frame, const cv::Rect2f& box);

    std::vector<float> input_buf_;   // reused across frames
    std::vector<float> output_buf_;
};
