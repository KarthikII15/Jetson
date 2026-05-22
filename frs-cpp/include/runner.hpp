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
    std::string entry_dir     = "increasing"; 
    float       x_threshold   = 30.0f;        
    float       y_threshold   = 80.0f;        
    int         window_size   = 4;            
    double      track_ttl     = 10.0;         
    double      cooldown_sec  = 30.0;         
    float       line_x        = 960.0f;       
    std::string mode          = "slope";      
    std::string axis          = "Y";          
    float       track_match_dist = 150.0f;     
};

struct FaceTrack {
    std::string          track_id;
    std::string          employee_id;     // set after recognition
    std::string          full_name;
    std::deque<float>    x_history;       // centroid X over last N frames
    float                last_x     = -1.0f; // previous X for line_cross mode
    double               last_seen  = 0.0;
    bool                 direction_fired = false;
    std::string          committed_dir;   // "entry" or "exit"
};

struct Config {
    std::string det_engine;       
    std::string emb_engine;       
    std::string backend_url;      
    std::string backend_endpoint;  
    std::string token_path;        
    int         tenant_id = 1;
    int         customer_id = 1;
    int         site_id = 1;
    float       conf_thresh = 0.45f;
    float       nms_thresh  = 0.45f;
    float       match_thresh = 0.55f;  
    double      cooldown_sec = 10.0;   
    int         inference_threads = 2; 
    int         queue_depth = 64;
    int         det_dla_core = -1;     
    int         emb_dla_core = -1;
    int         metrics_port = 5000;   
    DirectionConfig dir;

    std::string device_id = "jetson-orin-01";
    std::string default_gate = "Main Gate";
    int         default_branch_id = 1;

    // Keycloak auto-refresh credentials
    std::string keycloak_url;
    std::string keycloak_client_id  = "attendance-frontend";
    std::string keycloak_username;
    std::string keycloak_password;
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

    // Hot-reload — called by EnrollServer on /config/reload or polling thread
    void reloadCameras(const std::vector<CameraConfig>& new_cameras);
    void updateThresholds(float match_thresh, float cooldown_sec, float conf_thresh);
    void updateToken(const std::string& new_token);

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
    std::mutex                           captures_mtx_;

    // Live-reloadable thresholds (atomic so inference workers read without locking)
    std::atomic<float>  live_match_thresh_{0.38f};
    std::atomic<float>  live_cooldown_sec_{30.0f};
    std::atomic<float>  live_conf_thresh_{0.35f};

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
    std::string computeDirection(const std::deque<float>& x_history);
    std::string computeDirectionLineCross(float prev_x, float curr_x);
    void        purgeStaleTracksLocked(double now);
    bool        checkDirectionCooldown(const std::string& emp_id,
                                       const std::string& direction,
                                       const std::string& date);
};
