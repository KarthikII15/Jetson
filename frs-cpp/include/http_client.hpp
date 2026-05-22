#pragma once
#include <string>
#include <vector>
#include <array>
#include <mutex>
#include <chrono>

using Embedding = std::array<float, 512>;

struct RecognitionPayload {
    std::string device_id;
    std::string device_code;
    float       confidence;
    std::string timestamp;
    Embedding   embedding;
    std::vector<int> scaled_box; // [x, y, w, h] scaled to context size
    std::string direction;   // "entry", "exit", or "" (unknown)
    std::string track_id;    // tracker ID for this face
    std::string face_crop_b64; // Base64 encoded face crop
};

struct RecognitionResult {
    bool        matched      = false;
    int         http_code    = 0;
    std::string employee_id;
    std::string full_name;
    std::string employee_code;
    float       similarity   = 0.0f;
};

class HttpClient {
public:
    HttpClient(const std::string& backend_url, const std::string& token_path,
               int tenant_id = 1, int customer_id = 1, int site_id = 1,
               const std::string& keycloak_url = "",
               const std::string& keycloak_client_id = "attendance-frontend",
               const std::string& keycloak_username = "",
               const std::string& keycloak_password = "");

    RecognitionResult recognize(const std::string& endpoint, const RecognitionPayload& payload);
    std::string payloadToJson(const RecognitionPayload& payload);
    void        refreshToken();
    bool markAttendance(const std::string& employee_id,
                        const std::string& device_code,
                        const std::string& timestamp,
                        const std::string& status = "present");

    // Public so EnrollServer can POST/GET arbitrary endpoints
    std::pair<std::string, int> post(const std::string& path, const std::string& body);
    std::pair<std::string, int> get(const std::string& path);

    static std::string base64_encode(const unsigned char* buf, unsigned int bufLen);

private:
    std::string backend_url_;
    std::string token_path_;
    std::string token_;
    int         tenant_id_;
    int         customer_id_;
    int         site_id_;

    std::string keycloak_url_;
    std::string keycloak_client_id_;
    std::string keycloak_username_;
    std::string keycloak_password_;

    std::mutex  token_mutex_;
    std::chrono::steady_clock::time_point token_expiry_;

    std::string readToken();
    bool        fetchTokenFromKeycloak();   // must be called with token_mutex_ held
    bool        ensureValidToken();         // thread-safe, locks token_mutex_
    std::string embeddingToJson(const Embedding& emb);
};
