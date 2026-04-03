#include "runner.hpp"
#include <sys/statvfs.h>
#include <cstring>
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

                // ── Read system telemetry ──────────────────────────────
                // CPU usage via /proc/loadavg (1-min load avg as %)
                float cpu_pct = 0.0f;
                { FILE* f = fopen("/proc/loadavg","r");
                  if (f) { float load1; fscanf(f,"%f",&load1);
                    // normalize by CPU count (8 cores on Jetson Orin NX)
                    cpu_pct = std::min(100.0f, load1 / 8.0f * 100.0f); fclose(f); } }

                // Memory
                long mem_total=0, mem_free=0, mem_available=0;
                { FILE* f = fopen("/proc/meminfo","r"); char key[64]; long val;
                  if (f) { while(fscanf(f,"%s %ld kB",key,&val)==2) {
                    if(strcmp(key,"MemTotal:")==0) mem_total=val;
                    else if(strcmp(key,"MemAvailable:")==0) mem_available=val; }
                    mem_free=mem_available; fclose(f); } }
                long mem_used_mb = (mem_total - mem_available) / 1024;
                long mem_total_mb = mem_total / 1024;

                // Temperature
                float temp_c = 0.0f;
                { FILE* f = fopen("/sys/class/thermal/thermal_zone0/temp","r");
                  if (f) { int t; fscanf(f,"%d",&t); temp_c=(float)t/1000.0f; fclose(f); } }

                // GPU usage via sysfs (non-blocking)
                float gpu_pct = 0.0f;
                { FILE* f = fopen("/sys/class/devfreq/17000000.ga10b/cur_freq","r");
                  if (!f) f = fopen("/sys/class/devfreq/57000000.gpu/cur_freq","r");
                  if (!f) f = fopen("/sys/kernel/debug/bpmp/debug/clk/gpc0clk/rate","r");
                  if (f) { long cur=0,max=1; fscanf(f,"%ld",&cur); fclose(f);
                    FILE* fm = fopen("/sys/class/devfreq/17000000.ga10b/max_freq","r");
                    if (!fm) fm = fopen("/sys/class/devfreq/57000000.gpu/max_freq","r");
                    if (fm) { fscanf(fm,"%ld",&max); fclose(fm); }
                    if (max>0) gpu_pct = (float)cur/(float)max*100.0f; } }

                // Uptime
                long uptime_sec = 0;
                { FILE* f = fopen("/proc/uptime","r");
                  if (f) { double up; fscanf(f,"%lf",&up); uptime_sec=(long)up; fclose(f); } }

                // Disk usage
                float disk_used_gb = 0.0f;
                { struct statvfs st; if(statvfs("/",&st)==0) {
                    unsigned long total = st.f_blocks * st.f_frsize;
                    unsigned long free_ = st.f_bfree  * st.f_frsize;
                    disk_used_gb = (float)(total-free_)/(1024.0f*1024.0f*1024.0f); } }

                // ── Jetson heartbeat with telemetry ───────────────────
                char jetson_body[1024];
                snprintf(jetson_body, sizeof(jetson_body),
                    "{\"status\":\"online\","
                    "\"cpu_percent\":%.1f,"
                    "\"memory_used_mb\":%ld,"
                    "\"memory_total_mb\":%ld,"
                    "\"gpu_percent\":%.1f,"
                    "\"temperature_c\":%.1f,"
                    "\"disk_used_gb\":%.2f,"
                    "\"uptime_seconds\":%ld,"
                    "\"cameras\":[{"
                    "\"cam_id\":\"entrance-cam-01\","
                    "\"status\":\"online\","
                    "\"accuracy\":%.1f,"
                    "\"total_scans\":%d,"
                    "\"error_rate\":0.0"
                    "}]}",
                    cpu_pct, mem_used_mb, mem_total_mb,
                    gpu_pct, temp_c, disk_used_gb, uptime_sec,
                    s.faces_detected > 0 ? (float)s.matches/(float)(s.matches+s.unknowns)*100.0f : 0.0f,
                    s.frames_processed
                );
                spdlog::info("[Heartbeat] cpu={:.1f} gpu={:.1f} temp={:.1f} mem={}/{}", 
                    cpu_pct, gpu_pct, temp_c, mem_used_mb, mem_total_mb);
                spdlog::info("[Heartbeat] body={}", std::string(jetson_body).substr(0,200));
                hb_http.post("/api/devices/nug-boxes/jetson-orin-01/heartbeat", std::string(jetson_body));
            } catch (const std::exception& e) {
                spdlog::error("[Heartbeat] failed: {}", e.what());
            } catch (...) {
                spdlog::error("[Heartbeat] unknown error");
            }
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
