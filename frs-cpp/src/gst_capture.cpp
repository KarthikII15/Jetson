// src/gst_capture.cpp
#include "gst_capture.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

// ── GStreamer pipeline strings ────────────────────────────────────────────────

std::string CaptureThread::buildGstPipeline(bool hw) const {
    // Prama/Hikvision cameras use H.264 over RTSP/TCP
    // Jetson Orin: nvv4l2decoder decodes H.264 entirely on GPU (NVDEC)
    // nvvidconv: NV12 → BGRx in GPU memory, videoconvert: BGRx → BGR
    if (hw) {
        return
            "rtspsrc location=" + cfg_.rtsp_url + " latency=200 protocols=tcp "
            "! rtph264depay ! h264parse "
            "! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx,width=" + std::to_string(cfg_.width) +
            ",height=" + std::to_string(cfg_.height) + " "
            "! videoconvert ! video/x-raw,format=BGR "
            "! appsink drop=true max-buffers=2 sync=false";
    }
    // Software fallback (avdec_h264) — works without Jetson GPU
    return
        "rtspsrc location=" + cfg_.rtsp_url + " latency=200 protocols=tcp "
        "! rtph264depay ! h264parse ! avdec_h264 "
        "! videoconvert ! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=2 sync=false";
}

cv::VideoCapture CaptureThread::openCapture() {
    if (cfg_.hw_decode) {
        auto hw_pipe = buildGstPipeline(true);
        cv::VideoCapture cap(hw_pipe, cv::CAP_GSTREAMER);
        if (cap.isOpened()) {
            spdlog::info("[{}] HW decoder (nvv4l2decoder) opened", cfg_.id);
            return cap;
        }
        spdlog::warn("[{}] HW decoder failed, falling back to SW", cfg_.id);
    }

    auto sw_pipe = buildGstPipeline(false);
    cv::VideoCapture cap(sw_pipe, cv::CAP_GSTREAMER);
    if (cap.isOpened()) {
        spdlog::info("[{}] SW decoder (avdec_h264) opened", cfg_.id);
        return cap;
    }

    // Last resort: plain RTSP (OpenCV internal)
    spdlog::warn("[{}] GStreamer unavailable, using plain RTSP", cfg_.id);
    return cv::VideoCapture(cfg_.rtsp_url);
}

// ── Capture loop ──────────────────────────────────────────────────────────────
void CaptureThread::captureLoop() {
    using namespace std::chrono;
    const auto interval = microseconds(1'000'000 / cfg_.fps_target);

    auto cap = openCapture();
    auto last_frame = steady_clock::now();

    while (running_.load()) {
        if (!cap.isOpened()) {
            spdlog::warn("[{}] Stream lost — reconnecting in 5s", cfg_.id);
            std::this_thread::sleep_for(seconds(5));
            cap.release();
            cap = openCapture();
            reconnects_.fetch_add(1);
            continue;
        }

        // Rate-limit to fps_target
        auto now = steady_clock::now();
        if (now - last_frame < interval) {
            std::this_thread::sleep_for(milliseconds(5));
            continue;
        }

        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            spdlog::warn("[{}] Empty frame — reconnecting in 2s", cfg_.id);
            std::this_thread::sleep_for(seconds(2));
            cap.release();
            cap = openCapture();
            reconnects_.fetch_add(1);
            continue;
        }

        last_frame = steady_clock::now();

        // Deliver frame to inference pipeline (by move — zero copy)
        cb_(cfg_.id, cfg_.device_code, std::move(frame));
    }

    cap.release();
    spdlog::info("[{}] Capture thread stopped", cfg_.id);
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────
CaptureThread::CaptureThread(const CameraConfig& cfg, FrameCallback cb)
    : cfg_(cfg), cb_(std::move(cb)) {}

CaptureThread::~CaptureThread() { stop(); }

void CaptureThread::start() {
    running_.store(true);
    thread_ = std::thread([this]{ captureLoop(); });
    spdlog::info("[{}] Capture thread started ({}fps)", cfg_.id, cfg_.fps_target);
}

void CaptureThread::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
}
