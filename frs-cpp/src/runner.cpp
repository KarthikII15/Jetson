// src/runner.cpp
#include "runner.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ── Direction & Tracking ─────────────────────────────────────────────────────

std::string FRSRunner::assignTrack(const std::string& cam_id, float cx, float cy) {
    std::lock_guard<std::mutex> lock(tracks_mtx_);
    double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    purgeStaleTracksLocked(now);
    float target_pos = (cfg_.dir.axis == "Y" || cfg_.dir.axis == "y") ? cy : cx;
    std::string best_id;
    float best_dist = dir_cfg_.track_match_dist;
    for (auto& [tid, track] : tracks_) {
        if (track.track_id.find(cam_id) == std::string::npos) continue;
        if (track.x_history.empty()) continue;
        if (std::abs(track.x_history.back() - target_pos) < best_dist) {
            best_dist = std::abs(track.x_history.back() - target_pos);
            best_id = tid;
        }
    }
    if (!best_id.empty()) {
        auto& t = tracks_[best_id];
        if (now - t.last_seen > 5.0 && t.direction_fired) {
            t.direction_fired = false;
            t.x_history.clear();
        }
        t.x_history.push_back(target_pos);
        if (t.x_history.size() > dir_cfg_.window_size) t.x_history.pop_front();
        t.last_seen = now;
        return best_id;
    }
    std::string tid = cam_id + "_trk" + std::to_string(next_track_id_++);
    FaceTrack track; track.track_id = tid;
    track.x_history.push_back(target_pos); track.last_seen = now;
    tracks_[tid] = std::move(track);
    return tid;
}

std::string FRSRunner::computeDirection(const std::deque<float>& history) {
    if (history.size() < (size_t)dir_cfg_.window_size) return "unknown";
    float net_delta = history.back() - history.front();
    float threshold = (cfg_.dir.axis == "Y" || cfg_.dir.axis == "y") ? dir_cfg_.y_threshold : dir_cfg_.x_threshold;
    if (std::abs(net_delta) < threshold) return "stationary";
    return (net_delta > 0) == (dir_cfg_.entry_dir == "increasing") ? "entry" : "exit";
}

std::string FRSRunner::computeDirectionLineCross(float prev_x, float curr_x) {
    const float dead = 5.0f;
    float lo = dir_cfg_.line_x - dead;
    float hi = dir_cfg_.line_x + dead;
    if (prev_x < lo && curr_x >= hi)
        return dir_cfg_.entry_dir == "increasing" ? "entry" : "exit";
    if (prev_x > hi && curr_x <= lo)
        return dir_cfg_.entry_dir == "increasing" ? "exit" : "entry";
    return "unknown";
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
        return false;
    dir_cooldown_[key] = now;
    return true;
}

// ── Constructor ───────────────────────────────────────────────────────────────
FRSRunner::FRSRunner(const Config& cfg, const std::vector<CameraConfig>& cameras)
    : cfg_(cfg), cameras_(cameras), dir_cfg_(cfg.dir)
{
    live_match_thresh_.store(cfg_.match_thresh);
    live_cooldown_sec_.store(static_cast<float>(cfg_.cooldown_sec));
    live_conf_thresh_.store(cfg_.conf_thresh);

    detector_ = std::make_unique<FaceDetector>(cfg_.det_engine, cfg_.conf_thresh, cfg_.nms_thresh, cfg_.det_dla_core);
    embedder_ = std::make_unique<FaceEmbedder>(cfg_.emb_engine, cfg_.emb_dla_core);
    http_     = std::make_unique<HttpClient>(cfg_.backend_url, cfg_.token_path,
                                             cfg_.tenant_id, cfg_.customer_id, cfg_.site_id,
                                             cfg_.keycloak_url, cfg_.keycloak_client_id,
                                             cfg_.keycloak_username, cfg_.keycloak_password);
    spdlog::info("[Runner] Initialized with {} camera(s)", cameras_.size());
}

// ── Hot-reload ────────────────────────────────────────────────────────────────

void FRSRunner::reloadCameras(const std::vector<CameraConfig>& new_cameras) {
    std::lock_guard<std::mutex> lock(captures_mtx_);

    // Stop captures for cameras that are removed or disabled
    for (auto it = captures_.begin(); it != captures_.end(); ) {
        const std::string cid = (*it)->cameraId();
        bool keep = false;
        for (const auto& nc : new_cameras)
            if (nc.id == cid && nc.enabled) { keep = true; break; }
        if (!keep) {
            spdlog::info("[Runner] Removing camera: {}", cid);
            (*it)->stop();
            it = captures_.erase(it);
            cameras_.erase(std::remove_if(cameras_.begin(), cameras_.end(),
                [&cid](const CameraConfig& c){ return c.id == cid; }), cameras_.end());
        } else {
            ++it;
        }
    }

    // Add new cameras / restart cameras with changed RTSP URL
    for (const auto& nc : new_cameras) {
        if (!nc.enabled) continue;
        auto existing = std::find_if(cameras_.begin(), cameras_.end(),
            [&nc](const CameraConfig& c){ return c.id == nc.id; });

        if (existing == cameras_.end()) {
            spdlog::info("[Runner] Adding camera: {}", nc.id);
            cameras_.push_back(nc);
            auto cap = std::make_unique<CaptureThread>(nc,
                [this](const std::string& id, const std::string& code, cv::Mat f){
                    enqueueFrame(id, code, std::move(f));
                });
            cap->start();
            captures_.push_back(std::move(cap));
        } else if (existing->rtsp_url != nc.rtsp_url) {
            spdlog::info("[Runner] RTSP changed for {}, restarting capture", nc.id);
            auto old = std::find_if(captures_.begin(), captures_.end(),
                [&nc](const auto& c){ return c->cameraId() == nc.id; });
            if (old != captures_.end()) {
                (*old)->stop();
                captures_.erase(old);
            }
            *existing = nc;
            auto cap = std::make_unique<CaptureThread>(nc,
                [this](const std::string& id, const std::string& code, cv::Mat f){
                    enqueueFrame(id, code, std::move(f));
                });
            cap->start();
            captures_.push_back(std::move(cap));
        }
    }
    spdlog::info("[Runner] Camera reload complete: {} active", captures_.size());
}

void FRSRunner::updateThresholds(float match_thresh, float cooldown_sec, float conf_thresh) {
    if (match_thresh  > 0) live_match_thresh_.store(match_thresh);
    if (cooldown_sec  > 0) live_cooldown_sec_.store(cooldown_sec);
    if (conf_thresh   > 0) live_conf_thresh_.store(conf_thresh);
    spdlog::info("[Runner] Thresholds updated: match={:.2f} cooldown={:.0f}s conf={:.2f}",
                 live_match_thresh_.load(), live_cooldown_sec_.load(), live_conf_thresh_.load());
}

void FRSRunner::updateToken(const std::string& new_token) {
    std::ofstream f(cfg_.token_path, std::ios::trunc);
    if (f.good()) f << new_token;
    else spdlog::error("[Runner] Failed to write token to {}", cfg_.token_path);
    http_->refreshToken();
    spdlog::info("[Runner] Token updated from backend push");
}

FRSRunner::~FRSRunner() { stop(); }

// ── Lifecycle ─────────────────────────────────────────────────────────────────
void FRSRunner::start() {
    running_.store(true);
    for (int i = 0; i < cfg_.inference_threads; ++i) {
        workers_.emplace_back([this]{ inferenceWorker(); });
        spdlog::info("[Runner] Inference worker {} started", i);
    }
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
    {
        std::lock_guard<std::mutex> lock(captures_mtx_);
        for (auto& cap : captures_) cap->stop();
        captures_.clear();
    }
    for (auto& w : workers_) if (w.joinable()) w.join();
    workers_.clear();
    spdlog::info("[Runner] Stopped");
}

// ── Frame queue ───────────────────────────────────────────────────────────────
void FRSRunner::enqueueFrame(const std::string& cam_id, const std::string& dev_code, cv::Mat frame) {
    std::unique_lock<std::mutex> lock(queue_mtx_);
    if ((int)queue_.size() >= cfg_.queue_depth) {
        queue_.pop();
    }
    queue_.push({cam_id, dev_code, std::move(frame)});
    queue_cv_.notify_one();
}

bool FRSRunner::checkCooldown(const std::string& cam_id) {
    double now = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(cooldown_mtx_);
    auto it = last_sent_.find(cam_id);
    if (it != last_sent_.end() && (now - it->second) < live_cooldown_sec_.load())
        return false;
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
            queue_cv_.wait(lock, [this]{ return !queue_.empty() || !running_.load(); });
            if (!running_.load() && queue_.empty()) break;
            task = std::move(queue_.front());
            queue_.pop();
        }

        stat_frames_.fetch_add(1);

        auto faces = detector_->detect(task.frame);
        if (faces.empty()) continue;
        stat_faces_.fetch_add(faces.size());

        // Track X-history update (for all detected faces)
        std::vector<std::string> face_track_ids;
        face_track_ids.reserve(faces.size());
        for (const auto& face : faces) {
            float cx = face.box.x + face.box.width  / 2.0f;
            float cy = face.box.y + face.box.height / 2.0f;
            face_track_ids.push_back(assignTrack(task.cam_id, cx, cy));
        }

        // Step 2: Rate Limiting (Camera Level)
        {
            double now_sec = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            std::lock_guard<std::mutex> lock(cooldown_mtx_);
            std::string fps_key = "fps_" + task.cam_id;
            auto it = last_sent_.find(fps_key);
            if (it != last_sent_.end() && (now_sec - it->second) < 0.067) continue; 
            last_sent_[fps_key] = now_sec;
        }

        // Step 3: Recognition & Accumulation
        struct MatchInfo {
            FaceBox           face;
            std::string       track_id;
            RecognitionPayload payload;
            RecognitionResult result;
        };
        std::vector<MatchInfo> frame_matches;

        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face     = faces[i];
            const auto& track_id = face_track_ids[i];
            if (face.aligned.empty()) continue;

            auto embedding = embedder_->embed(face.aligned);
            RecognitionPayload payload;
            payload.device_id   = task.cam_id;
            payload.device_code = task.device_code;
            payload.confidence  = face.conf;
            payload.timestamp   = nowIso8601();
            payload.track_id    = track_id;
            payload.embedding   = embedding;

            // ── Full-Frame Context (User Request) ─────────────────────────────
            int target_w = 960, target_h = 540;
            int orig_w = task.frame.cols, orig_h = task.frame.rows;
            
            cv::Mat display_frame;
            cv::resize(task.frame, display_frame, cv::Size(target_w, target_h));
            
            // Scale bounding box coordinates
            cv::Rect2f b = face.box;
            payload.scaled_box = {
                static_cast<int>(b.x * target_w / orig_w),
                static_cast<int>(b.y * target_h / orig_h),
                static_cast<int>(b.width * target_w / orig_w),
                static_cast<int>(b.height * target_h / orig_h)
            };
            
            std::vector<uchar> buf;
            cv::imencode(".jpg", display_frame, buf, {cv::IMWRITE_JPEG_QUALITY, 80});
            payload.face_crop_b64 = HttpClient::base64_encode(buf.data(), buf.size());

            auto result_obj = http_->recognize(cfg_.backend_endpoint, payload);

            if (result_obj.matched && result_obj.similarity >= live_match_thresh_.load()) {
                std::string emp_key = task.cam_id + ":" + result_obj.employee_id;
                {
                    std::lock_guard<std::mutex> lock(cooldown_mtx_);
                    double now = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    auto it = last_sent_.find(emp_key);
                    if (it != last_sent_.end() && (now - it->second) < live_cooldown_sec_.load()) {
                        double remaining = live_cooldown_sec_.load() - (now - it->second);
                        spdlog::info("[{}] ⏳ {} on cooldown ({}s remaining)", 
                                     task.cam_id, result_obj.full_name, (int)remaining);
                        continue;  // Skip this match but continue processing other faces
                    }
                }
                frame_matches.push_back({face, track_id, std::move(payload), std::move(result_obj)});
            } else {
                if (result_obj.matched) {
                    spdlog::info("[{}] ⚠️ Match below threshold ({}): sim={:.3f} threshold={:.2f}",
                                  task.cam_id, result_obj.full_name, result_obj.similarity, live_match_thresh_.load());
                }
                if (result_obj.http_code == 404) {
                    stat_unknowns_.fetch_add(1);
                } else if (result_obj.http_code == 401) {
                    // ⚠️ CRITICAL FIX: DO NOT break out of loop - we must process ALL people in frame!
                    // Token refresh will happen after all faces are processed.
                    spdlog::warn("[{}] Face {}/{}: 401 Unauthorized - token refresh needed", 
                                 task.cam_id, (i+1), faces.size());
                    http_->refreshToken();
                    stat_unknowns_.fetch_add(1);
                } else {
                    stat_unknowns_.fetch_add(1);
                }
            }
        }

        // ─── Multi-Person Summary ────────────────────────────────────────────
        spdlog::info("[{}] 📊 Frame Summary: Detected {} faces | Recognized {} | Unknowns {}", 
                     task.cam_id, faces.size(), frame_matches.size(), 
                     (int)(stat_unknowns_.load() % faces.size()));

        // Step 4: Consolidated Output
        if (!frame_matches.empty()) {
            cv::Mat annotated = task.frame.clone();
            std::string shared_timestamp = frame_matches[0].payload.timestamp;
            std::string shared_filename = "multi_" + task.cam_id + "_" + shared_timestamp;
            for (auto& ch : shared_filename) if (ch == ':' || ch == '.') ch = '-';
            shared_filename += ".jpg";
            std::string full_path = "/opt/frs/photos/" + shared_filename;
            std::string photo_url = "/api/attendance/photos/" + shared_filename;

            for (size_t idx = 0; idx < frame_matches.size(); ++idx) {
                auto& mi = frame_matches[idx];
                stat_matches_.fetch_add(1);
                spdlog::info("[{}] ✅ Person {}/{}: {} ({}) sim={:.3f}", 
                             task.cam_id, (idx+1), frame_matches.size(), mi.result.full_name, 
                             mi.result.employee_code, mi.result.similarity);

                if (dir_cfg_.enabled) {
                    std::unique_lock<std::mutex> tlock(tracks_mtx_);
                    auto it = tracks_.find(mi.track_id);
                    if (it != tracks_.end()) {
                        auto& trk = it->second;
                        trk.employee_id = mi.result.employee_id;
                        trk.full_name   = mi.result.full_name;
                        if (!trk.direction_fired) {
                            std::string dir;
                            float cx = mi.face.box.x + mi.face.box.width / 2.0f;
                            if (dir_cfg_.mode == "line_cross") dir = computeDirectionLineCross(trk.last_x, cx);
                            else dir = computeDirection(trk.x_history);
                            
                            if (dir == "entry" || dir == "exit") {
                                std::string date = mi.payload.timestamp.substr(0, 10);
                                tlock.unlock();
                                if (checkDirectionCooldown(mi.result.employee_id, dir, date)) {
                                    spdlog::info("[{}] 🧭 {} → {} (track: {})", task.cam_id, mi.result.full_name, dir, mi.track_id);
                                    mi.payload.direction = dir;
                                    {
                                        std::lock_guard<std::mutex> tlock2(tracks_mtx_);
                                        auto it2 = tracks_.find(mi.track_id);
                                        if (it2 != tracks_.end()) {
                                            it2->second.direction_fired = true;
                                            it2->second.committed_dir   = dir;
                                        }
                                    }
                                    json dir_body;
                                    dir_body["employeeId"] = mi.result.employee_id;
                                    dir_body["direction"]  = dir;
                                    dir_body["trackId"]    = mi.track_id;
                                    dir_body["deviceId"]   = task.cam_id;
                                    dir_body["timestamp"]  = mi.payload.timestamp;
                                    http_->post("/api/attendance/direction", dir_body.dump());
                                }
                                tlock.lock();
                            }
                        }
                    }
                }

                int x1 = std::max(0, (int)mi.face.box.x);
                int y1 = std::max(0, (int)mi.face.box.y);
                int x2 = std::min(task.frame.cols-1, (int)(mi.face.box.x + mi.face.box.width));
                int y2 = std::min(task.frame.rows-1, (int)(mi.face.box.y + mi.face.box.height));
                cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                std::string label = mi.result.full_name + " (" + std::to_string((int)(mi.result.similarity * 100)) + "%)";
                int baseline = 0;
                cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
                int ly = std::max(y1 - 5, ts.height + 5);
                cv::rectangle(annotated, cv::Point(x1, ly - ts.height - 5), cv::Point(x1 + ts.width + 4, ly + baseline), cv::Scalar(0, 200, 0), cv::FILLED);
                cv::putText(annotated, label, cv::Point(x1 + 2, ly - 3), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
            }

            cv::putText(annotated, shared_timestamp.substr(0, 19), cv::Point(8, task.frame.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            try {
                cv::imwrite(full_path, annotated);
                double now_ts = std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                for (auto& mi : frame_matches) {
                    {
                        std::lock_guard<std::mutex> lock(cooldown_mtx_);
                        last_sent_[task.cam_id + ":" + mi.result.employee_id] = now_ts;
                    }
                    std::string day_key = "day_" + mi.result.employee_id + "_" + mi.payload.timestamp.substr(0,10);
                    std::string frame_type = "checkout";
                    {
                        std::lock_guard<std::mutex> lock(cooldown_mtx_);
                        if (last_sent_.find(day_key) == last_sent_.end()) {
                            frame_type = "checkin";
                            last_sent_[day_key] = 0.0;
                        }
                    }
                    json att_body;
                    att_body["frameUrl"]   = photo_url;
                    att_body["employeeId"] = mi.result.employee_id;
                    att_body["date"]       = mi.payload.timestamp.substr(0, 10);
                    att_body["type"]       = frame_type;
                    http_->post("/api/attendance/frame", att_body.dump());
                }
            } catch (const std::exception& e) {
                spdlog::error("[Runner] Consolidated output error: {}", e.what());
            }
        }
    }
}

FRSRunner::Stats FRSRunner::stats() const {
    std::unique_lock<std::mutex> lock(const_cast<std::mutex&>(queue_mtx_));
    return {
        stat_frames_.load(),
        stat_faces_.load(),
        stat_matches_.load(),
        stat_unknowns_.load(),
        (int)queue_.size(),
        (int)captures_.size()
    };
}
