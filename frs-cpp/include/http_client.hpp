#pragma once
#include <string>
#include <vector>
#include <array>

using Embedding = std::array<float, 512>;

struct RecognitionPayload {
    std::string device_id;
    std::string device_code;
    float       confidence;
    std::string timestamp;
    Embedding   embedding;
    std::string direction;   // "entry", "exit", or "" (unknown)
    std::string track_id;    // tracker ID for this face
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
    HttpClient(const std::string& backend_url, const std::string& token_path);

    RecognitionResult recognize(const RecognitionPayload& payload);
    void        refreshToken();
    bool markAttendance(const std::string& employee_id,
                        const std::string& device_code,
                        const std::string& timestamp,
                        const std::string& status = "present");

    // Public so EnrollServer can POST arbitrary endpoints
    std::pair<std::string, int> post(const std::string& path, const std::string& body);

private:
    std::string backend_url_;
    std::string token_path_;
    std::string token_;

    std::string readToken();

    std::string embeddingToJson(const Embedding& emb);
};
