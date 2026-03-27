// src/runner.cpp
#include "runner.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>


// ── Direction & Tracking ─────────────────────────────────────────────────────

std::string FRSRunner::assignTrack(const std::string& cam_id, float cx, float cy) {
    std::lock_guard<std::mutex> lock(tracks_mtx_);
    double now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // Purge stale tracks first
    purgeStaleTracksLocked(now);

    // Find closest existing track within 120px
    std::string best_id;
    float best_dist = 150.0f;
    for (auto& [tid, track] : tracks_) {
        if (track.track_id.find(cam_id) == std::string::npos) continue;
        if (track.y_history.empty()) continue;
        float last_y = track.y_history.back();
        float dist = std::abs(last_y - cy);
        if (dist < best_dist) {
            best_dist = dist;
            best_id = tid;
        }
    }

    if (!best_id.empty()) {
        // Update existing track
        auto& t = tracks_[best_id];
        // Reset direction if track was stale (person left and came back)
        double gap = now - t.last_seen;
        if (gap > 5.0 && t.direction_fired) {
            t.direction_fired = false;
            t.committed_dir   = "";
            t.y_history.clear();
            spdlog::info("[Track] {} reset after {:.1f}s gap", best_id, gap);
        }
        // Only add Y if meaningfully different from last value (avoids duplicates)
        if (t.y_history.empty() || std::abs(cy - t.y_history.back()) > 5.0f) {
            t.y_history.push_back(cy);
            if ((int)t.y_history.size() > dir_cfg_.window_size)
                t.y_history.pop_front();
        }
        t.last_seen = now;
        return best_id;
    }

    // Create new track
    spdlog::info("[Track] New track created (no match within 150px)");
    std::string tid = cam_id + "_trk" + std::to_string(next_track_id_++);
    FaceTrack track;
    track.track_id      = tid;
    track.last_seen     = now;
    track.direction_fired = false;
    track.y_history.push_back(cy);
    tracks_[tid] = std::move(track);
    return tid;
}

std::string FRSRunner::computeDirection(const std::deque<float>& y_history) {
    if ((int)y_history.size() < dir_cfg_.window_size) return "unknown";

    // Net delta from first to last
    float net_delta = y_history.back() - y_history.front();

    // Linear regression slope for robustness
    int n = y_history.size();
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    for (int i = 0; i < n; i++) {
        sum_x  += i;
        sum_y  += y_history[i];
        sum_xy += i * y_history[i];
        sum_xx += i * i;
    }
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 1e-6f);
    float total_movement = std::abs(net_delta);

    spdlog::debug("[Direction] slope={:.1f} net_delta={:.1f} total_movement={:.1f} threshold={}",
        slope, net_delta, total_movement, dir_cfg_.y_threshold);

    // Must have minimum movement to count
    if (total_movement < dir_cfg_.y_threshold) return "stationary";

    bool increasing = net_delta > 0;
    if (dir_cfg_.entry_dir == "increasing")
        return increasing ? "entry" : "exit";
    else
        return increasing ? "exit" : "entry";
}

void FRSRunner::purgeStaleTracksLocked(double now) {
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (now - it->second.last_seen > dir_cfg_.track_ttl)
            it = tracks_.erase(it);
        else
            ++it;
    }
}

bool FRSRunner::checkDirectionCooldown(const std::string& emp_id,
                                        const std::string& direction,
                                        const std::string& date) {
    std::string key = emp_id + "_" + direction + "_" + date;
    std::lock_guard<std::mutex> lock(tracks_mtx_);
    double now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    auto it = dir_cooldown_.find(key);
    if (it != dir_cooldown_.end() && now - it->second < dir_cfg_.cooldown_sec)
        return false; // still in cooldown
    dir_cooldown_[key] = now;
    return true; // allowed
}

// ── Constructor ───────────────────────────────────────────────────────────────
FRSRunner::FRSRunner(const Config& cfg, const std::vector<CameraConfig>& cameras)
    : cfg_(cfg), cameras_(cameras), dir_cfg_(cfg.dir)
{
    spdlog::info("[Direction] enabled={} entry_dir={} y_threshold={} window={}", 
        dir_cfg_.enabled, dir_cfg_.entry_dir, dir_cfg_.y_threshold, dir_cfg_.window_size);
    detector_ = std::make_unique<FaceDetector>(cfg_.det_engine,
                                                cfg_.conf_thresh,
                                                cfg_.nms_thresh);
    embedder_ = std::make_unique<FaceEmbedder>(cfg_.emb_engine);
    http_     = std::make_unique<HttpClient>(cfg_.backend_url, cfg_.token_path);

    spdlog::info("[Runner] Initialized with {} camera(s)", cameras_.size());
}

FRSRunner::~FRSRunner() { stop(); }

// ── Lifecycle ─────────────────────────────────────────────────────────────────
void FRSRunner::start() {
    spdlog::info("[Direction] enabled={} entry_dir={} y_threshold={:.0f} window={}",
        dir_cfg_.enabled, dir_cfg_.entry_dir, dir_cfg_.y_threshold, dir_cfg_.window_size);
    running_.store(true);

    // Start inference worker threads (share detector+embedder — TRT is thread-safe for infer)
    for (int i = 0; i < cfg_.inference_threads; ++i) {
        workers_.emplace_back([this]{ inferenceWorker(); });
        spdlog::info("[Runner] Inference worker {} started", i);
    }

    // Start one capture thread per camera
    for (const auto& cam : cameras_) {
        auto cap = std::make_unique<CaptureThread>(cam,
            [this](const std::string& id, const std::string& code, cv::Mat f){
                enqueueFrame(id, code, std::move(f));
            });
        cap->start();
        captures_.push_back(std::move(cap));
    }

    spdlog::info("[Runner] All threads started — FRS is live");
}

void FRSRunner::stop() {
    running_.store(false);
    queue_cv_.notify_all();

    for (auto& cap : captures_) cap->stop();
    captures_.clear();

    for (auto& w : workers_) if (w.joinable()) w.join();
    workers_.clear();

    spdlog::info("[Runner] Stopped");
}

// ── Frame queue ───────────────────────────────────────────────────────────────
void FRSRunner::enqueueFrame(const std::string& cam_id,
                               const std::string& dev_code,
                               cv::Mat frame) {
    std::unique_lock<std::mutex> lock(queue_mtx_);
    if ((int)queue_.size() >= cfg_.queue_depth) {
        // Drop oldest frame under load — attendance latency is acceptable
        queue_.pop();
    }
    queue_.push({cam_id, dev_code, std::move(frame)});
    queue_cv_.notify_one();
}

// ── Cooldown cache ────────────────────────────────────────────────────────────
bool FRSRunner::checkCooldown(const std::string& cam_id) {
    using namespace std::chrono;
    auto now = duration_cast<duration<double>>(
        steady_clock::now().time_since_epoch()).count();

    std::lock_guard<std::mutex> lock(cooldown_mtx_);
    auto it = last_sent_.find(cam_id);
    if (it != last_sent_.end() && (now - it->second) < cfg_.cooldown_sec)
        return false;  // still in cooldown

    last_sent_[cam_id] = now;
    return true;
}

// ── ISO8601 timestamp ─────────────────────────────────────────────────────────
std::string FRSRunner::nowIso8601() {
    auto now  = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms   = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;

    std::ostringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%S")
       << "." << std::setfill('0') << std::setw(3) << ms << "Z";
    return ss.str();
}

// ── Main inference worker ─────────────────────────────────────────────────────
void FRSRunner::inferenceWorker() {
    while (running_.load()) {
        FrameTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this]{
                return !queue_.empty() || !running_.load();
            });
            if (!running_.load() && queue_.empty()) break;
            task = std::move(queue_.front());
            queue_.pop();
        }

        stat_frames_.fetch_add(1);

        // ── Step 1: Face detection (YOLOv8 TRT) ──────────────────────────────
        auto faces = detector_->detect(task.frame);
        if (faces.empty()) continue;

        stat_faces_.fetch_add(faces.size());

        // ── Track Y-history update (every frame, before rate limit) ──────
        if (dir_cfg_.enabled) {
            for (const auto& face : faces) {
                float cx2 = face.box.x + face.box.width  / 2.0f;
                float cy2 = face.box.y + face.box.height / 2.0f;
                assignTrack(task.cam_id, cx2, cy2);
            }
        }

        // ── Step 2: Cooldown check (per camera) ──────────────────────────────
        // Check before embedding to save GPU time
        // Rate limit to 1 FPS per camera to allow continuous multi-face scanning
        {
            double now_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            std::lock_guard<std::mutex> lock(cooldown_mtx_);
            std::string fps_key = "fps_" + task.cam_id;
            auto it = last_sent_.find(fps_key);
            if (it != last_sent_.end() && (now_sec - it->second) < 0.067) continue; // 15 FPS recognition
            last_sent_[fps_key] = now_sec;
        }

        // ── Step 3 & 4: Process ALL detected faces ────────────────────────────
        for (const auto& face : faces) {
            if (face.aligned.empty()) continue;

            // ── Track assignment ─────────────────────────────────────
            float cx = face.box.x + face.box.width  / 2.0f;
            float cy = face.box.y + face.box.height / 2.0f;
            std::string track_id = assignTrack(task.cam_id, cx, cy);


            auto embedding = embedder_->embed(face.aligned);

            RecognitionPayload payload;
            payload.device_id   = task.cam_id;
            payload.device_code = task.device_code;
            payload.confidence  = face.conf;
            payload.timestamp   = nowIso8601();
            payload.track_id    = track_id;
            payload.embedding   = embedding;

            auto result = http_->recognize(payload);

            if (result.matched) {
                // Per-employee cooldown key so multiple people can match same frame
                std::string emp_key = task.cam_id + ":" + result.employee_id;
                {
                    std::lock_guard<std::mutex> lock(cooldown_mtx_);
                    double now = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    auto it  = last_sent_.find(emp_key);
                    if (it != last_sent_.end()) {
                        if ((now - it->second) < cfg_.cooldown_sec) continue;
                    }
                    last_sent_[emp_key] = now;
                }
                stat_matches_.fetch_add(1);
                spdlog::info("[{}] ✅ {} ({}) sim={:.3f}",
                    task.cam_id, result.full_name,
                    result.employee_code, result.similarity);

                // ── Direction detection ──────────────────────────────
                if (dir_cfg_.enabled) {
                    std::lock_guard<std::mutex> tlock(tracks_mtx_);
                    auto it = tracks_.find(track_id);
                    if (it != tracks_.end()) {
                        auto& trk = it->second;
                        trk.employee_id = result.employee_id;
                        trk.full_name   = result.full_name;
                        // Log Y history for debugging
                        std::string y_hist_str;
                        for (auto yv : trk.y_history) y_hist_str += std::to_string((int)yv) + " ";
                        spdlog::info("[Track] {} size={} fired={} Y=[{}]",
                            track_id, trk.y_history.size(), trk.direction_fired, y_hist_str);
                        if (!trk.direction_fired) {
                            std::string dir = computeDirection(trk.y_history);
                            spdlog::info("[Track] {} computed_dir={}", track_id, dir);
                            if (dir == "entry" || dir == "exit") {
                                std::string date = payload.timestamp.substr(0, 10);
                                tlock.~lock_guard();
                                if (checkDirectionCooldown(result.employee_id, dir, date)) {
                                    spdlog::info("[{}] 🧭 {} → {} (track: {})",
                                        task.cam_id, result.full_name, dir, track_id);
                                    payload.direction = dir;
                                    std::lock_guard<std::mutex> tlock2(tracks_mtx_);
                                    auto it2 = tracks_.find(track_id);
                                    if (it2 != tracks_.end()) {
                                        it2->second.direction_fired = true;
                                        it2->second.committed_dir   = dir;
                                    }
                                    // POST direction update to backend
                                    std::string dir_body =
                                        std::string("{") +
                                        "\"employeeId\":\"" + result.employee_id + "\"," +
                                        "\"direction\":\"" + dir + "\"," +
                                        "\"trackId\":\"" + track_id + "\"," +
                                        "\"deviceId\":\"" + task.cam_id + "\"," +
                                        "\"timestamp\":\"" + payload.timestamp + "\"" +
                                        "}";
                                    auto [dr_body, dr_code] = http_->post("/api/attendance/direction", dir_body);
                                    spdlog::info("[{}] Direction POST {} → HTTP {}", task.cam_id, dir, dr_code);
                                }
                            }
                        }
                    }
                }


                // Save proof frame as JPEG with bounding box annotation
                try {
                    std::string fname = "/opt/frs/photos/" + result.employee_id +
                        "_" + payload.timestamp.substr(0,19) + ".jpg";
                    for (auto& ch : fname) if (ch == ':') ch = '-';

                    // Draw bounding box and name on a copy of the frame
                    cv::Mat annotated = task.frame.clone();

                    // Box is in original frame coordinates
                    int x1 = std::max(0, (int)face.box.x);
                    int y1 = std::max(0, (int)face.box.y);
                    int x2 = std::min(task.frame.cols-1, (int)(face.box.x + face.box.width));
                    int y2 = std::min(task.frame.rows-1, (int)(face.box.y + face.box.height));

                    // Green box
                    cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2),
                                  cv::Scalar(0, 255, 0), 2);

                    // Name label background
                    std::string label = result.full_name + " (" +
                        std::to_string((int)(result.similarity * 100)) + "%)";
                    int baseline = 0;
                    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                    int ly = std::max(y1 - 5, ts.height + 5);
                    cv::rectangle(annotated,
                        cv::Point(x1, ly - ts.height - 5),
                        cv::Point(x1 + ts.width + 4, ly + baseline),
                        cv::Scalar(0, 200, 0), cv::FILLED);
                    cv::putText(annotated, label, cv::Point(x1 + 2, ly - 3),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

                    // Timestamp bottom-left
                    std::string ts_str = payload.timestamp.substr(0, 19);
                    cv::putText(annotated, ts_str, cv::Point(8, task.frame.rows - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

                    cv::imwrite(fname, annotated);
                    // Update attendance record with photo URL
                    std::string photo_url = "/api/attendance/photos/" +
                        fname.substr(fname.rfind('/') + 1);
                    // Type: checkin for first match of day, checkout for subsequent
                    std::string frame_type = "checkout";
                    {
                        std::string day_key = "day_" + result.employee_id + "_" + payload.timestamp.substr(0,10);
                        std::lock_guard<std::mutex> lock(cooldown_mtx_);
                        if (last_sent_.find(day_key) == last_sent_.end()) {
                            frame_type = "checkin";
                            last_sent_[day_key] = 0.0;
                        }
                    }
                    std::string hb_body =
                        std::string("{") +
                        "\"frameUrl\":\"" + photo_url + "\"," +
                        "\"employeeId\":\"" + result.employee_id + "\"," +
                        "\"date\":\"" + payload.timestamp.substr(0,10) + "\"," +
                        "\"type\":\"" + frame_type + "\"" +
                        "}";
                    http_->post("/api/attendance/frame", hb_body);
                } catch (...) {}
            } else if (result.http_code == 404) {
                stat_unknowns_.fetch_add(1);
                spdlog::debug("[{}] ⚠  Unknown face (conf={:.2f})", task.cam_id, face.conf);
            } else if (result.http_code == 401) {
                spdlog::warn("[{}] 401 Unauthorized — refreshing token", task.cam_id);
                http_->refreshToken();
                break;
            } else {
                spdlog::warn("[{}] Recognize HTTP {}", task.cam_id, result.http_code);
                stat_unknowns_.fetch_add(1);
            }
        }
}
}

// ── Stats ─────────────────────────────────────────────────────────────────────
FRSRunner::Stats FRSRunner::stats() const {
    std::unique_lock<std::mutex> lock(
        const_cast<std::mutex&>(queue_mtx_));
    return {
        stat_frames_.load(),
        stat_faces_.load(),
        stat_matches_.load(),
        stat_unknowns_.load(),
        (int)queue_.size(),
        (int)captures_.size()
    };
}
