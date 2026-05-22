#pragma once
// include/gst_capture.hpp
// Hardware-accelerated RTSP capture using GStreamer nvv4l2decoder on Jetson Orin.
// One CaptureThread per camera, pushes frames into a lock-free ring buffer.

#include <opencv2/videoio.hpp>
#include <atomic>
#include <functional>
#include <string>
#include <thread>

struct CameraConfig {
    std::string id;
    std::string name;
    std::string rtsp_url;
    std::string device_code;
    std::string gate_name;
    std::string direction;
    int         branch_id;
    std::string branch_name;
    int   fps_target = 5;
    int   width      = 1280;
    int   height     = 720;
    bool  hw_decode  = true;
    bool  enabled    = true;
};

// Callback type: called with each frame ready for inference
using FrameCallback = std::function<void(const std::string& cam_id,
                                          const std::string& device_code,
                                          cv::Mat frame)>;

class CaptureThread {
public:
    CaptureThread(const CameraConfig& cfg, FrameCallback cb);
    ~CaptureThread();

    void start();
    void stop();
    bool isRunning() const { return running_.load(); }
    const std::string& cameraId() const { return cfg_.id; }

    // Reconnect attempts since last successful frame
    int reconnects() const { return reconnects_.load(); }

private:
    CameraConfig   cfg_;
    FrameCallback  cb_;
    std::thread    thread_;
    std::atomic<bool> running_{false};
    std::atomic<int>  reconnects_{0};

    cv::VideoCapture openCapture();
    std::string buildGstPipeline(bool hw) const;
    void captureLoop();
};
