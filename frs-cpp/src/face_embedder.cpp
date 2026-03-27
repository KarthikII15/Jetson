// src/face_embedder.cpp
#include "face_embedder.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

FaceEmbedder::FaceEmbedder(const std::string& engine_path)
    : engine_(engine_path)
{
    input_buf_.resize(1 * 3 * INPUT_H * INPUT_W);
    output_buf_.resize(EMB_DIM);
    spdlog::info("[Embedder] ArcFace R50 loaded");
}

// ── Pre-processing ────────────────────────────────────────────────────────────
// ArcFace normalization: (pixel/255 - 0.5) / 0.5  =  pixel/127.5 - 1.0
// Applied per channel, then HWC → CHW
void FaceEmbedder::preprocess(const cv::Mat& aligned, float* dst) {
    cv::Mat rgb;
    cv::cvtColor(aligned, rgb, cv::COLOR_BGR2RGB);

    // Convert to float, normalize
    cv::Mat flt;
    rgb.convertTo(flt, CV_32F, 1.0 / 127.5, -1.0);

    std::vector<cv::Mat> chans(3);
    cv::split(flt, chans);
    for (int c = 0; c < 3; ++c)
        std::copy((float*)chans[c].data,
                  (float*)chans[c].data + INPUT_H * INPUT_W,
                  dst + c * INPUT_H * INPUT_W);
}

// ── L2 normalization ──────────────────────────────────────────────────────────
void FaceEmbedder::l2Normalize(float* vec, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; ++i) norm += vec[i] * vec[i];
    norm = std::sqrt(norm) + 1e-10f;
    for (int i = 0; i < n; ++i) vec[i] /= norm;
}

// ── Public interface ──────────────────────────────────────────────────────────
Embedding FaceEmbedder::embed(const cv::Mat& aligned) {
    preprocess(aligned, input_buf_.data());

    const float* in  = input_buf_.data();
    float*       out = output_buf_.data();
    engine_.infer({in}, {out});

    l2Normalize(out, EMB_DIM);

    Embedding emb;
    std::copy(out, out + EMB_DIM, emb.begin());
    return emb;
}

// ── Similarity metrics ────────────────────────────────────────────────────────
float FaceEmbedder::cosine(const Embedding& a, const Embedding& b) {
    // Both embeddings are already L2-normalized, so dot product = cosine similarity
    float dot = 0.0f;
    for (int i = 0; i < EMB_DIM; ++i) dot += a[i] * b[i];
    return dot;  // range [-1, 1], higher = more similar
}

float FaceEmbedder::l2(const Embedding& a, const Embedding& b) {
    float sum = 0.0f;
    for (int i = 0; i < EMB_DIM; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);  // range [0, 2], lower = more similar
}
