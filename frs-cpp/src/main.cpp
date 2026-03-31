#include "runner.hpp"
#include "http_client.hpp"
#include "enroll_server.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fstream>
#include <csignal>
#include <atomic>
#include <thread>
#include <iostream>

using json = nlohmann::json;
static std::atomic<bool> g_shutdown{false};
void sigHandler(int) { g_shutdown.store(true); }

Config loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("Cannot open config: " + path);
    auto j = json::parse(f);
    Config cfg;
    cfg.det_engine        = j["models"]["face_detection"]["engine_path"];
    cfg.emb_engine        = j["models"]["face_embedding"]["engine_path"];
    cfg.backend_url       = j["backend"]["url"];
    cfg.token_path        = j["backend"]["token_path"];
    cfg.conf_thresh       = j.value("conf_threshold",    0.45f);
    cfg.nms_thresh        = j.value("nms_threshold",     0.45f);
    cfg.match_thresh      = j.value("match_threshold",   0.55f);
    cfg.cooldown_sec      = j.value("cooldown_seconds",  10.0);
    cfg.inference_threads = j.value("inference_threads", 2);
    cfg.queue_depth       = j.value("queue_depth",       64);

    // Direction config
    if (j.contains("direction")) {
        auto& d = j["direction"];
        cfg.dir.enabled      = d.value("enabled",           false);
        cfg.dir.entry_dir    = d.value("entry_direction",    "increasing");
        cfg.dir.x_threshold  = d.value("x_threshold",        30.0f);
        cfg.dir.window_size  = d.value("tracking_window",    4);
        cfg.dir.track_ttl    = d.value("track_ttl_seconds",  10.0);
        cfg.dir.cooldown_sec = d.value("cooldown_seconds",   30.0);
        cfg.dir.line_x       = d.value("line_x",             960.0f);
        cfg.dir.mode         = d.value("mode",               "slope");
        cfg.dir.track_match_dist = d.value("track_match_dist", 150.0f);
    }
    return cfg;
}

std::vector<CameraConfig> loadCameras(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("Cannot open cameras: " + path);
    auto j = json::parse(f);
    std::vector<CameraConfig> cams;
    for (const auto& c : j["cameras"]) {
        if (!c.value("enabled", true)) continue;
        CameraConfig cam;
        cam.id          = c["id"];
        cam.rtsp_url    = c["rtsp_url"];
        cam.device_code = c.value("device_code", cam.id);
        cam.fps_target  = c.value("fps_target",  5);
        cam.width       = c.value("width",        1280);
        cam.height      = c.value("height",       720);
        cam.hw_decode   = c.value("hw_decode",    true);
        cams.push_back(cam);
    }
    return cams;
}

int main(int argc, char** argv) {
    // stdout logging — visible in journalctl
    auto logger = spdlog::stdout_color_mt("frs");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    std::string config_path  = "/opt/frs/config.json";
    std::string cameras_path = "/opt/frs/cameras.json";
    int         sidecar_port = 5000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config"  && i+1 < argc) config_path  = argv[++i];
        if (arg == "--cameras" && i+1 < argc) cameras_path = argv[++i];
        if (arg == "--port"    && i+1 < argc) sidecar_port = std::stoi(argv[++i]);
    }

    spdlog::info("FRS2 C++ Runner starting");
    spdlog::info("  Config:  {}", config_path);
    spdlog::info("  Cameras: {}", cameras_path);
    spdlog::info("  Port:    {}", sidecar_port);

    std::signal(SIGINT,  sigHandler);
    std::signal(SIGTERM, sigHandler);

    try {
        auto cfg     = loadConfig(config_path);
        auto cameras = loadCameras(cameras_path);

        if (cameras.empty()) { spdlog::error("No cameras configured"); return 1; }

        spdlog::info("Loaded {} camera(s)", cameras.size());
        for (const auto& c : cameras)
            spdlog::info("  {} → {}...", c.id, c.rtsp_url.substr(0, 35));

        spdlog::info("Loading TensorRT engines...");
        spdlog::info("  Detection:  {}", cfg.det_engine);
        spdlog::info("  Embedding:  {}", cfg.emb_engine);

        FRSRunner runner(cfg, cameras);

        spdlog::info("Starting inference runner...");
        runner.start();

        EnrollServer enroll(sidecar_port, runner, cfg, cameras);
        std::thread enroll_thread([&enroll]{ enroll.run(); });

        spdlog::info("All systems running. PID={}", getpid());
        spdlog::info("Enroll server listening on :{}", sidecar_port);

        while (!g_shutdown.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            auto s = runner.stats();
            spdlog::info("STATS | frames={} faces={} matches={} unknowns={} queue={}/{}",
                s.frames_processed, s.faces_detected,
                s.matches, s.unknowns,
                s.queue_depth, cfg.queue_depth);

            // Heartbeat — update device status + scan count on backend
            try {
                HttpClient hb_http(cfg.backend_url, cfg.token_path);
                std::string hb_body = std::string("{\"stats\":{\"frames_processed\":") +
                    std::to_string(s.frames_processed) + "}}";
                hb_http.post("/api/cameras/entrance-cam-01/heartbeat", hb_body);
                // Jetson device heartbeat
                std::string jetson_body = std::string("{\"stats\":{\"frames_processed\":") +
                    std::to_string(s.frames_processed) +
                    ",\"faces_detected\":" + std::to_string(s.faces_detected) +
                    ",\"matches\":" + std::to_string(s.matches) + "}}";
                hb_http.post("/api/cameras/jetson-orin-01/heartbeat", jetson_body);
            } catch (...) {}
        }

        spdlog::info("Shutting down...");
        enroll.stop();
        if (enroll_thread.joinable()) enroll_thread.join();
        runner.stop();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }

    spdlog::info("FRS2 runner stopped cleanly");
    return 0;
}
