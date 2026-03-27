#pragma once
// include/enroll_server.hpp
// Lightweight HTTP server (port 5000) for:
//   GET  /health         — liveness probe + stats
//   POST /enroll         — enroll employee face from camera snapshot
//   POST /recognize/once — one-shot recognition for admin UI testing

#include "runner.hpp"
#include <atomic>
#include <string>

class EnrollServer {
public:
    EnrollServer(int port, FRSRunner& runner,
                 const Config& cfg,
                 const std::vector<CameraConfig>& cameras);

    void run();   // blocking
    void stop();

private:
    int                              port_;
    FRSRunner&                       runner_;
    Config                           cfg_;
    std::vector<CameraConfig>        cameras_;
    std::atomic<bool>                running_{true};
    int                              server_fd_ = -1;

    // Request handlers
    std::string handleHealth();
    std::string handleEnroll(const std::string& body);
    std::string handleEnrollImage(const std::string& body, const std::string& content_type);
    std::string handleRecognizeOnce(const std::string& body);

    // HTTP helpers
    std::string respond(int code, const std::string& content_type,
                         const std::string& body);
    std::string respondJson(int code, const std::string& json_body);

    // Parse simple HTTP request
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string body;
        std::string content_type;
    };
    HttpRequest parseRequest(const std::string& raw);

    void handleClient(int client_fd);
};
