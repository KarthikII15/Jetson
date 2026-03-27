#include <curl/curl.h>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
// src/enroll_server.cpp
#include "enroll_server.hpp"
#include "face_detector.hpp"
#include "face_embedder.hpp"
#include "http_client.hpp"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <thread>

using json = nlohmann::json;

EnrollServer::EnrollServer(int port, FRSRunner& runner,
                             const Config& cfg,
                             const std::vector<CameraConfig>& cameras)
    : port_(port), runner_(runner), cfg_(cfg), cameras_(cameras) {}

// ── HTTP helpers ──────────────────────────────────────────────────────────────
std::string EnrollServer::respond(int code, const std::string& ct,
                                   const std::string& body) {
    std::ostringstream ss;
    ss << "HTTP/1.1 " << code << " ";
    switch (code) {
        case 200: ss << "OK"; break;
        case 201: ss << "Created"; break;
        case 400: ss << "Bad Request"; break;
        case 404: ss << "Not Found"; break;
        case 500: ss << "Internal Server Error"; break;
        default:  ss << "OK";
    }
    ss << "\r\nContent-Type: " << ct
       << "\r\nContent-Length: " << body.size()
       << "\r\nAccess-Control-Allow-Origin: *"
       << "\r\nConnection: close\r\n\r\n"
       << body;
    return ss.str();
}

std::string EnrollServer::respondJson(int code, const std::string& body) {
    return respond(code, "application/json", body);
}

// ── Request handlers ──────────────────────────────────────────────────────────
std::string EnrollServer::handleHealth() {
    auto s = runner_.stats();
    json j;
    j["status"]            = "ok";
    j["frames_processed"]  = s.frames_processed;
    j["faces_detected"]    = s.faces_detected;
    j["matches"]           = s.matches;
    j["unknowns"]          = s.unknowns;
    j["queue_depth"]       = s.queue_depth;
    j["active_cameras"]    = s.active_cameras;
    j["inference_engine"]  = "tensorrt_fp16";
    j["models"]["detection"] = cfg_.det_engine;
    j["models"]["embedding"] = cfg_.emb_engine;

    json cam_arr = json::array();
    for (const auto& c : cameras_) {
        json cam;
        cam["id"]      = c.id;
        cam["fps"]     = c.fps_target;
        cam["hw"]      = c.hw_decode;
        cam_arr.push_back(cam);
    }
    j["cameras"] = cam_arr;
    return respondJson(200, j.dump(2));
}

std::string EnrollServer::handleEnroll(const std::string& body) {
    try {
        auto j = json::parse(body);
        std::string emp_id = j["employee_id"].get<std::string>();
        std::string cam_id = j.value("cam_id", cameras_.empty() ? "" : cameras_[0].id);

        const CameraConfig* cam = nullptr;
        for (const auto& cc : cameras_)
            if (cc.id == cam_id) { cam = &cc; break; }
        if (!cam) return respondJson(404, R"({"error":"camera not found"})");

        spdlog::info("[Enroll] Enrolling employee {} from camera {}", emp_id, cam_id);

        std::string rtsp = cam->rtsp_url;
        auto pos = rtsp.rfind("/102");
        if (pos != std::string::npos) rtsp.replace(pos, 4, "/101");

        // Capture single frame via OpenCV VideoCapture using unique pipeline name
        // Uses a short-lived pipeline separate from the main runner
        spdlog::info("[Enroll] Capturing enrollment frame from RTSP...");

        std::string snap_file = "/tmp/frs_enroll_" + emp_id + ".jpg";

        // Build GStreamer pipeline with multifilesink to write one JPEG
        std::string gst_snap =
            "rtspsrc location=" + cam->rtsp_url +
            " latency=300 protocols=tcp !"
            " rtph264depay ! h264parse ! nvv4l2decoder !"
            " nvvidconv ! video/x-raw,format=BGRx !"
            " videoconvert ! video/x-raw,format=BGR !"
            " jpegenc !"
            " multifilesink location=" + snap_file + " max-files=1 post-messages=true";

        // Launch as background process, wait 6s then kill
        std::string cmd = "GST_DEBUG=0 gst-launch-1.0 -e " + gst_snap +
                          " > /dev/null 2>&1 & echo $! > /tmp/frs_gst_pid.txt";
        system(cmd.c_str());
        sleep(6);
        system("kill $(cat /tmp/frs_gst_pid.txt 2>/dev/null) 2>/dev/null; sleep 1");

        // Read JPEG file into cv::Mat
        cv::Mat frame = cv::imread(snap_file);
        // Clean up temp file
        system(("rm -f " + snap_file).c_str());

        if (frame.empty()) {
            return respondJson(503, R"({"error":"Failed to capture frame from camera"})");
        }

        spdlog::info("[Enroll] Frame captured ({}x{})", frame.cols, frame.rows);

        FaceDetector detector(cfg_.det_engine, 0.40f, 0.40f);
        auto faces = detector.detect(frame);

        if (faces.empty())
            return respondJson(422, R"({"error":"No face detected. Employee must face the camera."})");
        if ((int)faces.size() > 1) {
            json err;
            err["error"] = "Multiple faces (" + std::to_string(faces.size()) + "). Only one person should be visible.";
            return respondJson(422, err.dump());
        }

        FaceEmbedder embedder(cfg_.emb_engine);
        if (faces[0].aligned.empty()) {
            return respondJson(422, R"({"error":"Face detected but alignment failed. Try again with better lighting."})");
        }
        auto emb = embedder.embed(faces[0].aligned);

        json payload;
        payload["embedding"] = json::array();
        for (float v : emb) payload["embedding"].push_back(v);
        payload["confidence"] = faces[0].conf;
        payload["source"]     = "cpp-enroll-server";

        HttpClient http(cfg_.backend_url, cfg_.token_path);
        auto [resp_body, code] = http.post(
            "/api/employees/" + emp_id + "/enroll-face-direct",
            payload.dump());

        if (code == 201) {
            json resp;
            resp["success"]     = true;
            resp["employee_id"] = emp_id;
            resp["confidence"]  = faces[0].conf;
            resp["message"]     = "Enrolled";
            spdlog::info("[Enroll] Employee {} enrolled (conf={:.2f})", emp_id, faces[0].conf);
            return respondJson(201, resp.dump());
        }
        json err; err["error"] = "Backend HTTP " + std::to_string(code); err["detail"] = resp_body.substr(0,300);
        return respondJson(500, err.dump());

    } catch (const std::exception& e) {
        json err; err["error"] = e.what();
        return respondJson(500, err.dump());
    }
}

std::string EnrollServer::handleEnrollImage(const std::string& body, const std::string& content_type) {
    try {
        // Extract JPEG bytes from multipart or raw body
        std::vector<uchar> img_bytes;

        if (content_type.find("multipart") != std::string::npos) {
            // Extract boundary
            auto b_pos = content_type.find("boundary=");
            if (b_pos == std::string::npos)
                return respondJson(400, R"({"error":"No boundary in multipart"})");
            std::string boundary = "--" + content_type.substr(b_pos + 9);
            // Find image data after headers
            auto data_start = body.find("\r\n\r\n");
            if (data_start == std::string::npos)
                return respondJson(400, R"({"error":"No image data found"})");
            data_start += 4;
            auto data_end = body.rfind("\r\n" + boundary);
            if (data_end == std::string::npos) data_end = body.size();
            std::string img_data = body.substr(data_start, data_end - data_start);
            img_bytes.assign(img_data.begin(), img_data.end());
        } else {
            // Raw JPEG bytes
            img_bytes.assign(body.begin(), body.end());
        }

        if (img_bytes.empty())
            return respondJson(400, R"({"error":"Empty image data"})");

        // Decode image
        cv::Mat frame = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
        if (frame.empty())
            return respondJson(400, R"({"error":"Could not decode image"})");

        // Detect faces
        FaceDetector detector(cfg_.det_engine, cfg_.conf_thresh, cfg_.nms_thresh);
        auto faces = detector.detect(frame);
        if (faces.empty())
            return respondJson(422, R"({"error":"No face detected in image"})");
        if (faces.size() > 1)
            return respondJson(422, R"({"error":"Multiple faces detected. Use single-face photo"})");

        // Extract embedding
        FaceEmbedder embedder(cfg_.emb_engine);
        auto embedding = embedder.embed(faces[0].aligned);
        if (embedding.empty())
            return respondJson(500, R"({"error":"Embedding extraction failed"})");

        // Build response
        json resp;
        resp["success"]    = true;
        resp["confidence"] = faces[0].conf;
        resp["faceCount"]  = 1;
        resp["embedding"]  = embedding;
        return respondJson(200, resp.dump());

    } catch (const std::exception& e) {
        json err; err["error"] = e.what();
        return respondJson(500, err.dump());
    }
}

std::string EnrollServer::handleRecognizeOnce(const std::string& body) {
    // Triggered by admin UI "Test Recognition" button
    // { "cam_id": "entrance-cam-01" }
    try {
        auto j = json::parse(body);
        std::string cam_id = j.value("cam_id", cameras_.empty() ? "" : cameras_[0].id);

        json resp;
        resp["status"] = "triggered";
        resp["cam_id"] = cam_id;
        resp["note"]   = "Next face detected on this camera will be logged";
        return respondJson(200, resp.dump());
    } catch (const std::exception& e) {
        json err; err["error"] = e.what();
        return respondJson(400, err.dump());
    }
}

// ── Request parser ────────────────────────────────────────────────────────────
EnrollServer::HttpRequest EnrollServer::parseRequest(const std::string& raw) {
    HttpRequest req;
    std::istringstream ss(raw);
    std::string line;

    // First line: METHOD PATH HTTP/1.1
    if (std::getline(ss, line)) {
        std::istringstream ls(line);
        ls >> req.method >> req.path;
    }

    // Skip headers until blank line
    int content_length = 0;
    while (std::getline(ss, line) && line != "\r") {
        if (line.find("Content-Length:") != std::string::npos) {
            content_length = std::stoi(line.substr(16));
        }
        if (line.find("Content-Type:") != std::string::npos) {
            req.content_type = line.substr(14);
            if (!req.content_type.empty() && req.content_type.back() == '\r')
                req.content_type.pop_back();
        }
    }

    // Body
    if (content_length > 0) {
        req.body.resize(content_length);
        ss.read(req.body.data(), content_length);
    }

    return req;
}

// ── Client handler ────────────────────────────────────────────────────────────
void EnrollServer::handleClient(int fd) {
    // Read full request with loop (needed for large image uploads)
    std::vector<char> buf;
    buf.reserve(5 * 1024 * 1024);
    char chunk[65536];
    ssize_t n;
    // Read until we have complete HTTP request
    std::string raw_data;
    raw_data.reserve(5 * 1024 * 1024);
    
    // First read to get headers
    n = recv(fd, chunk, sizeof(chunk), 0);
    if (n <= 0) { close(fd); return; }
    raw_data.append(chunk, n);
    
    // Parse content-length from headers
    size_t content_length = 0;
    auto cl_pos = raw_data.find("Content-Length:");
    if (cl_pos == std::string::npos) cl_pos = raw_data.find("content-length:");
    if (cl_pos != std::string::npos) {
        content_length = std::stoul(raw_data.substr(cl_pos + 16, 20));
    }
    
    // Find header end
    auto header_end = raw_data.find("\r\n\r\n");
    size_t header_size = (header_end != std::string::npos) ? header_end + 4 : raw_data.size();
    size_t body_received = raw_data.size() - header_size;
    
    // Keep reading until we have the full body
    while (content_length > 0 && body_received < content_length) {
        n = recv(fd, chunk, std::min(sizeof(chunk), content_length - body_received), 0);
        if (n <= 0) break;
        raw_data.append(chunk, n);
        body_received += n;
    }

    auto req = parseRequest(raw_data);
    std::string response;

    if (req.method == "GET" && req.path == "/health") {
        response = handleHealth();
    } else if (req.method == "POST" && req.path == "/enroll-image") {
        response = handleEnrollImage(req.body, req.content_type);
    } else if (req.method == "POST" && req.path == "/enroll") {
        response = handleEnroll(req.body);
    } else if (req.method == "POST" && req.path == "/recognize/once") {
        response = handleRecognizeOnce(req.body);
    } else if (req.method == "OPTIONS") {
        response = respond(200, "text/plain", "");
    } else if (req.method == "GET" && req.path.substr(0, 8) == "/photos/") {
        std::string fname = "/opt/frs/photos/" + req.path.substr(8);
        std::ifstream img(fname, std::ios::binary);
        if (img) {
            std::string data((std::istreambuf_iterator<char>(img)), {});
            response = "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\n"
                       "Content-Length: " + std::to_string(data.size()) +
                       "\r\nAccess-Control-Allow-Origin: *\r\n\r\n" + data;
        } else {
            response = respondJson(404, R"({"error":"photo not found"})");
        }
    } else {
        response = respondJson(404, R"({"error":"not found"})");
    }

    send(fd, response.c_str(), response.size(), 0);
    close(fd);
}

// ── Server loop ───────────────────────────────────────────────────────────────
void EnrollServer::run() {
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(port_);

    if (bind(server_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
        spdlog::error("[EnrollServer] bind failed on port {}", port_);
        return;
    }
    listen(server_fd_, 10);
    spdlog::info("[EnrollServer] Listening on :{}", port_);

    while (running_.load()) {
        fd_set fds; FD_ZERO(&fds); FD_SET(server_fd_, &fds);
        timeval tv{1, 0};  // 1s timeout for clean shutdown
        if (select(server_fd_+1, &fds, nullptr, nullptr, &tv) <= 0) continue;

        int client = accept(server_fd_, nullptr, nullptr);
        if (client < 0) continue;

        // Handle in background thread (non-blocking server)
        std::thread([this, client]{ handleClient(client); }).detach();
    }

    close(server_fd_);
}

void EnrollServer::stop() {
    running_.store(false);
}
