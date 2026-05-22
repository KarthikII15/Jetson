#include "runner.hpp"
#include <sys/statvfs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
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

// Returns the primary outgoing interface IP without making a real network call
static std::string getLocalIP() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) return "unknown";
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(80);
    inet_pton(AF_INET, "8.8.8.8", &addr.sin_addr);
    char ip[INET_ADDRSTRLEN] = "unknown";
    if (connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) == 0) {
        struct sockaddr_in local{};
        socklen_t len = sizeof(local);
        if (getsockname(sock, reinterpret_cast<struct sockaddr*>(&local), &len) == 0)
            inet_ntop(AF_INET, &local.sin_addr, ip, sizeof(ip));
    }
    close(sock);
    return ip;
}

// Parse config response and hot-reload runner
static void applyConfigResponse(const std::string& body, FRSRunner& runner,
                                 std::atomic<bool>& shutdown) {
    try {
        auto j = json::parse(body);

        if (j.contains("device")) {
            auto& d = j["device"];
            std::string status = d.value("status", "active");
            if (status == "deactivated") {
                spdlog::warn("[ConfigPoll] Device deactivated by backend — shutting down");
                shutdown.store(true);
                return;
            }
            float match    = d.value("match_threshold", 0.0f);
            float cooldown = d.value("cooldown_seconds", 0.0f);
            float conf     = d.value("conf_threshold",   0.0f);
            runner.updateThresholds(match, cooldown, conf);
        }

        if (j.contains("cameras") && j["cameras"].is_array()) {
            std::vector<CameraConfig> new_cams;
            for (const auto& c : j["cameras"]) {
                CameraConfig cam;
                cam.id          = c.value("id",          c.value("camera_id", ""));
                cam.name        = c.value("name",        cam.id);
                cam.rtsp_url    = c.value("rtsp_url",    "");
                cam.device_code = c.value("device_code", cam.id);
                cam.gate_name   = c.value("gate_name",   "");
                cam.direction   = c.value("direction",   "entry");
                cam.branch_id   = c.value("branch_id",   1);
                cam.fps_target  = c.value("fps_target",  5);
                cam.hw_decode   = c.value("hw_decode",   true);
                cam.enabled     = c.value("enabled",     true);
                if (!cam.rtsp_url.empty()) new_cams.push_back(cam);
            }
            if (!new_cams.empty()) runner.reloadCameras(new_cams);
        }

        if (j.contains("token") && j["token"].is_string()) {
            runner.updateToken(j["token"].get<std::string>());
        }

    } catch (const std::exception& e) {
        spdlog::error("[ConfigPoll] Failed to parse config response: {}", e.what());
    }
}

Config loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("Cannot open config: " + path);
    auto j = json::parse(f);
    Config cfg;

    if (j.contains("model_config")) {
        auto& mc = j["model_config"];
        cfg.det_engine   = mc["face_detection"]["model_path"];
        cfg.det_dla_core = mc["face_detection"].value("dla_core", 0);
        cfg.conf_thresh  = mc["face_detection"].value("confidence_threshold", 0.45f);
        cfg.emb_engine   = mc["face_embedding"]["model_path"];
        cfg.emb_dla_core = mc["face_embedding"].value("dla_core", 1);
    } else if (j.contains("models")) {
        auto& m = j["models"];
        cfg.det_engine   = m["face_detection"]["engine_path"];
        cfg.emb_engine   = m["face_embedding"]["engine_path"];
        cfg.det_dla_core = m["face_detection"].value("dla_core", -1);
        cfg.emb_dla_core = m["face_embedding"].value("dla_core", -1);
        cfg.conf_thresh  = j.value("conf_threshold", 0.45f);
    }

    if (j.contains("inference")) cfg.inference_threads = j["inference"].value("num_threads", 2);
    else cfg.inference_threads = j.value("inference_threads", 2);
    
    if (j.contains("matching")) {
        cfg.match_thresh = j["matching"].value("threshold", 0.65f);
        cfg.cooldown_sec = j["matching"].value("cooldown_seconds", 5.0);
    } else {
        cfg.match_thresh = j.value("match_threshold", 0.55f);
        cfg.cooldown_sec = j.value("cooldown_seconds", 10.0);
    }

    auto& b = j["backend"];
    cfg.backend_url      = b["url"];
    cfg.backend_endpoint = b.value("endpoint", "/api/face/recognize");
    cfg.tenant_id        = b.value("tenant_id", 1);
    cfg.customer_id      = b.value("customer_id", 1);
    cfg.site_id          = b.value("site_id", 1);
    cfg.device_id        = b.value("device_id", "jetson-orin-01");
    
    // Support both root 'token_path' and nested 'auth.token_file'
    if (b.contains("auth") && b["auth"].contains("token_file")) {
        cfg.token_path = b["auth"]["token_file"];
    } else {
        cfg.token_path = b.value("token_path", "/opt/frs/device_token.txt");
    }

    // Keycloak auto-refresh
    if (b.contains("keycloak")) {
        auto& kc = b["keycloak"];
        cfg.keycloak_url       = kc.value("url", "");
        cfg.keycloak_client_id = kc.value("client_id", "attendance-frontend");
        cfg.keycloak_username  = kc.value("username", "");
        cfg.keycloak_password  = kc.value("password", "");
    }

    if (j.contains("camera")) {
        cfg.default_gate      = j["camera"].value("default_gate", "Main Gate");
        cfg.default_branch_id = j["camera"].value("default_branch_id", 1);
    }

    if (j.contains("direction")) {
        auto& d = j["direction"];
        cfg.dir.enabled      = d.value("enabled", true);
        cfg.dir.mode         = d.value("mode", "slope");
        cfg.dir.axis         = d.value("axis", "Y");
        cfg.dir.entry_dir    = d.value("entry_direction", "increasing");
        cfg.dir.track_ttl    = d.value("tracking_ttl_seconds", d.value("track_ttl_seconds", 10.0));
        if (d.contains("threshold") && d["threshold"].is_object()) {
            cfg.dir.y_threshold = d["threshold"].value("y", 80.0f);
            cfg.dir.x_threshold = d["threshold"].value("x", 50.0f);
        } else cfg.dir.x_threshold = d.value("x_threshold", 30.0f);
    }

    if (j.contains("metrics")) cfg.metrics_port = j["metrics"].value("port", 5000);
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
        if (c.contains("camera_id")) cam.id = c["camera_id"];
        else cam.id = c["id"];
        cam.name        = c.value("name", cam.id);
        cam.rtsp_url    = c["rtsp_url"];
        cam.device_code = c.value("device_code", c.value("camera_id", cam.id));
        cam.gate_name   = c.value("gate_name", "");
        cam.direction   = c.value("direction", "entry");
        cam.branch_id   = c.value("branch_id", 1);
        cam.branch_name = c.value("branch_name", "");
        cam.fps_target  = c.value("fps_target", 5);
        cam.hw_decode   = c.value("hw_decode", true);
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

        if (cfg.metrics_port != 5000 && sidecar_port == 5000) sidecar_port = cfg.metrics_port;
        FRSRunner runner(cfg, cameras);

        spdlog::info("Starting inference runner...");
        runner.start();

        EnrollServer enroll(sidecar_port, runner, cfg, cameras, g_shutdown);
        std::thread enroll_thread([&enroll]{ enroll.run(); });

        // Config polling thread — syncs device config + cameras from backend every 60s
        std::thread poll_thread([&]() {
            // Give the runner 30s to fully initialise before first poll
            for (int i = 0; i < 30 && !g_shutdown.load(); ++i)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            while (!g_shutdown.load()) {
                try {
                    HttpClient poll_http(cfg.backend_url, cfg.token_path,
                                         cfg.tenant_id, cfg.customer_id, cfg.site_id);
                    auto [body, code] = poll_http.get(
                        "/api/device-management/devices/" + cfg.device_id + "/config");
                    if (code == 200)
                        applyConfigResponse(body, runner, g_shutdown);
                    else
                        spdlog::warn("[ConfigPoll] HTTP {} from config endpoint", code);
                } catch (const std::exception& e) {
                    spdlog::error("[ConfigPoll] error: {}", e.what());
                }
                for (int i = 0; i < 60 && !g_shutdown.load(); ++i)
                    std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        });

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
                HttpClient hb_http(cfg.backend_url, cfg.token_path,
                                   cfg.tenant_id, cfg.customer_id, cfg.site_id,
                                   cfg.keycloak_url, cfg.keycloak_client_id,
                                   cfg.keycloak_username, cfg.keycloak_password);
                std::string hb_body = std::string("{\"stats\":{\"frames_processed\":") +
                    std::to_string(s.frames_processed) + "}}";
                hb_http.post("/api/cameras/entrance-cam-01/heartbeat", hb_body);

                // ── Read system telemetry ──────────────────────────────
                // CPU: 1-min load avg normalised by core count
                float cpu_pct = 0.0f;
                float load1 = 0.0f, load5 = 0.0f, load15 = 0.0f;
                { FILE* f = fopen("/proc/loadavg","r");
                  if (f) { fscanf(f,"%f %f %f",&load1,&load5,&load15);
                    cpu_pct = std::min(100.0f, load1 / 8.0f * 100.0f); fclose(f); } }

                // Memory
                long mem_total=0, mem_available=0;
                { FILE* f = fopen("/proc/meminfo","r"); char key[64]; long val;
                  if (f) { while(fscanf(f,"%s %ld kB",key,&val)==2) {
                    if(strcmp(key,"MemTotal:")==0) mem_total=val;
                    else if(strcmp(key,"MemAvailable:")==0) mem_available=val; }
                    fclose(f); } }
                long mem_used_mb  = (mem_total - mem_available) / 1024;
                long mem_total_mb = mem_total / 1024;
                float mem_pct = mem_total > 0 ? (float)(mem_total - mem_available) / (float)mem_total * 100.0f : 0.0f;

                // Temperature (thermal_zone0 = CPU/SoC on Jetson)
                float temp_c = 0.0f;
                { FILE* f = fopen("/sys/class/thermal/thermal_zone0/temp","r");
                  if (f) { int t; fscanf(f,"%d",&t); temp_c=(float)t/1000.0f; fclose(f); } }

                // GPU frequency → utilisation % (Jetson Orin: 17000000.gpu)
                float gpu_pct = 0.0f;
                { const char* cur_paths[] = {
                      "/sys/class/devfreq/17000000.gpu/cur_freq",
                      "/sys/class/devfreq/17000000.ga10b/cur_freq",
                      "/sys/class/devfreq/57000000.gpu/cur_freq", nullptr };
                  const char* max_paths[] = {
                      "/sys/class/devfreq/17000000.gpu/max_freq",
                      "/sys/class/devfreq/17000000.ga10b/max_freq",
                      "/sys/class/devfreq/57000000.gpu/max_freq", nullptr };
                  for (int pi = 0; cur_paths[pi]; ++pi) {
                    FILE* fc = fopen(cur_paths[pi],"r");
                    if (!fc) continue;
                    long cur=0, mx=1; fscanf(fc,"%ld",&cur); fclose(fc);
                    FILE* fm = fopen(max_paths[pi],"r");
                    if (fm) { fscanf(fm,"%ld",&mx); fclose(fm); }
                    if (mx > 0) gpu_pct = (float)cur / (float)mx * 100.0f;
                    break; } }

                // Uptime
                long uptime_sec = 0;
                { FILE* f = fopen("/proc/uptime","r");
                  if (f) { double up; fscanf(f,"%lf",&up); uptime_sec=(long)up; fclose(f); } }

                // Disk usage
                float disk_used_gb=0.0f, disk_total_gb=0.0f, disk_free_gb=0.0f, disk_pct=0.0f;
                { struct statvfs st; if(statvfs("/",&st)==0) {
                    unsigned long total = st.f_blocks * st.f_frsize;
                    unsigned long free_ = st.f_bfree  * st.f_frsize;
                    disk_total_gb = (float)total /(1024.0f*1024.0f*1024.0f);
                    disk_free_gb  = (float)free_ /(1024.0f*1024.0f*1024.0f);
                    disk_used_gb  = disk_total_gb - disk_free_gb;
                    disk_pct = total > 0 ? (float)(total-free_)/(float)total*100.0f : 0.0f; } }

                // Recognition rate
                int   total_scans     = s.faces_detected;
                int   total_matches   = s.matches;
                float recognition_rate = total_scans > 0
                    ? (float)total_matches / (float)total_scans * 100.0f : 0.0f;

                // ── Jetson heartbeat with telemetry ───────────────────
                std::string local_ip = getLocalIP();
                char jetson_body[2560];
                snprintf(jetson_body, sizeof(jetson_body),
                    "{"
                    "\"status\":\"online\","
                    "\"local_ip\":\"%s\","
                    "\"sidecar_port\":%d,"
                    "\"cpu_percent\":%.1f,"
                    "\"load_avg\":{\"1m\":%.2f,\"5m\":%.2f,\"15m\":%.2f},"
                    "\"memory_used_mb\":%ld,"
                    "\"memory_total_mb\":%ld,"
                    "\"memory_percent\":%.1f,"
                    "\"gpu_percent\":%.1f,"
                    "\"temperature_c\":%.1f,"
                    "\"disk_used_gb\":%.2f,"
                    "\"disk_total_gb\":%.2f,"
                    "\"disk_free_gb\":%.2f,"
                    "\"disk_percent\":%.1f,"
                    "\"uptime_seconds\":%ld,"
                    "\"recognition\":{\"rate\":%.1f,\"matches\":%lu,\"total_scans\":%lu,\"frames\":%d},"
                    "\"cameras\":[{"
                    "\"cam_id\":\"%s\","
                    "\"status\":\"online\","
                    "\"fps\":5,"
                    "\"accuracy\":%.1f,"
                    "\"total_scans\":%d"
                    "}]}",
                    local_ip.c_str(),
                    sidecar_port,
                    cpu_pct,
                    load1, load5, load15,
                    mem_used_mb, mem_total_mb, mem_pct,
                    gpu_pct,
                    temp_c,
                    disk_used_gb, disk_total_gb, disk_free_gb, disk_pct,
                    uptime_sec,
                    recognition_rate, total_matches, total_scans, s.frames_processed,
                    cameras[0].id.c_str(),
                    recognition_rate, total_scans
                );
                spdlog::info("[Heartbeat] cpu={:.1f}% gpu={:.1f}% temp={:.1f}°C mem={}/{}MB ({:.1f}%) disk={:.1f}% recog={:.1f}%",
                    cpu_pct, gpu_pct, temp_c, mem_used_mb, mem_total_mb, mem_pct, disk_pct, recognition_rate);
                hb_http.post("/api/device-management/devices/" + cfg.device_id + "/heartbeat", std::string(jetson_body));
            } catch (const std::exception& e) {
                spdlog::error("[Heartbeat] failed: {}", e.what());
            } catch (...) {
                spdlog::error("[Heartbeat] unknown error");
            }
        }

        spdlog::info("Shutting down...");
        enroll.stop();
        if (enroll_thread.joinable()) enroll_thread.join();
        if (poll_thread.joinable()) poll_thread.join();
        runner.stop();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }

    spdlog::info("FRS2 runner stopped cleanly");
    return 0;
}
