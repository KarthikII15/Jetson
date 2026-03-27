#pragma once
// include/runner.hpp
// Main FRS runner — manages capture threads, inference pipeline, cooldown cache

#include "gst_capture.hpp"
#include "face_detector.hpp"
#include "face_embedder.hpp"
#include "http_client.hpp"

#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <deque>
#include <queue>
#include <condition_variable>
#include <thread>
#include <atomic>

struct DirectionConfig {
    bool        enabled       = false;
    std::string entry_dir     = "increasing"; // "increasing" or "decreasing"
    float       y_threshold   = 45.0f;        // min px movement to count
    int         window_size   = 6;            // frames in sliding window
    double      track_ttl     = 30.0;         // seconds before track expires
    double      cooldown_sec  = 300.0;        // 5 min per employee+direction
};

struct FaceTrack {
    std::string          track_id;
    std::string          employee_id;     // set after recognition
    std::string          full_name;
    std::deque<float>    y_history;       // centroid Y over last N frames
    double               last_seen;       // epoch timestamp
    bool                 direction_fired; // true after direction committed
    std::string          committed_dir;   // "entry" or "exit"
};

struct Config {
    std::string det_engine;       // path to YOLOv8-face .engine
    std::string emb_engine;       // path to ArcFace R50 .engine
    std::string backend_url;      // http://VM_IP:8080
    std::string token_path;       // /opt/frs/device_token.txt
    float       conf_thresh = 0.45f;
    float       nms_thresh  = 0.45f;
    float       match_thresh = 0.55f;  // cosine similarity threshold
    double      cooldown_sec = 10.0;   // min seconds between marks per camera
    int         inference_threads = 2; // parallel TRT inference workers
    int         queue_depth = 64;
    DirectionConfig dir;
};

struct FrameTask {
    std::string cam_id;
    std::string device_code;
    cv::Mat     frame;
};

class FRSRunner {
public:
    FRSRunner(const Config& cfg, const std::vector<CameraConfig>& cameras);
    ~FRSRunner();

    void start();
    void stop();

    // Stats for /health endpoint
    struct Stats {
        uint64_t frames_processed;
        uint64_t faces_detected;
        uint64_t matches;
        uint64_t unknowns;
        int      queue_depth;
        int      active_cameras;
    };
    Stats stats() const;

private:
    Config                               cfg_;
    std::vector<CameraConfig>            cameras_;
    std::vector<std::unique_ptr<CaptureThread>> captures_;

    // Inference pipeline
    std::unique_ptr<FaceDetector>        detector_;
    std::unique_ptr<FaceEmbedder>        embedder_;
    std::unique_ptr<HttpClient>          http_;

    // Frame queue
    std::queue<FrameTask>                queue_;
    std::mutex                           queue_mtx_;
    std::condition_variable              queue_cv_;
    std::atomic<bool>                    running_{false};
    std::vector<std::thread>             workers_;

    // Direction config
    DirectionConfig                         dir_cfg_;

    // Per-camera cooldown
    std::unordered_map<std::string, double> last_sent_;
    std::mutex                              cooldown_mtx_;

    // Face tracks — keyed by track_id
    std::unordered_map<std::string, FaceTrack> tracks_;
    std::mutex                                  tracks_mtx_;
    std::atomic<uint32_t>                       next_track_id_{0};

    // Direction cooldown — keyed as employee_id+direction+date
    std::unordered_map<std::string, double>     dir_cooldown_;

    // Stats
    std::atomic<uint64_t> stat_frames_{0};
    std::atomic<uint64_t> stat_faces_{0};
    std::atomic<uint64_t> stat_matches_{0};
    std::atomic<uint64_t> stat_unknowns_{0};

    void enqueueFrame(const std::string& cam_id, const std::string& dev_code, cv::Mat frame);
    void inferenceWorker();
    bool checkCooldown(const std::string& cam_id);
    std::string nowIso8601();
    std::string assignTrack(const std::string& cam_id, float cx, float cy);
    std::string computeDirection(const std::deque<float>& y_history);
    void        purgeStaleTracksLocked(double now);
    bool        checkDirectionCooldown(const std::string& emp_id,
                                       const std::string& direction,
                                       const std::string& date);
};
