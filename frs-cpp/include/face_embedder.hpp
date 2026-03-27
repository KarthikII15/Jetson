#pragma once
// include/face_embedder.hpp
// ArcFace R50 TensorRT FP16 — 512-d L2-normalized embedding
// Input:  1×3×112×112  float32 (normalized: (x-0.5)/0.5 per channel)
// Output: 1×512        float32

#include "trt_engine.hpp"
#include <opencv2/core.hpp>
#include <array>
#include <vector>

using Embedding = std::array<float, 512>;

class FaceEmbedder {
public:
    explicit FaceEmbedder(const std::string& engine_path);

    // Compute L2-normalized 512-d embedding from 112×112 aligned face crop
    Embedding embed(const cv::Mat& aligned_bgr);

    // Cosine similarity between two embeddings [-1, 1]
    static float cosine(const Embedding& a, const Embedding& b);

    // L2 distance (lower = more similar)
    static float l2(const Embedding& a, const Embedding& b);

    static constexpr int INPUT_W = 112;
    static constexpr int INPUT_H = 112;
    static constexpr int EMB_DIM = 512;

private:
    TRTEngine           engine_;
    std::vector<float>  input_buf_;
    std::vector<float>  output_buf_;

    void preprocess(const cv::Mat& aligned, float* dst);
    static void l2Normalize(float* vec, int n);
};
