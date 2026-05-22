// src/http_client.cpp
#include "http_client.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <spdlog/spdlog.h>

using json = nlohmann::json;

// ── CURL write callback ───────────────────────────────────────────────────────
static size_t writeCallback(char* ptr, size_t size, size_t nmemb, std::string* s) {
    s->append(ptr, size * nmemb);
    return size * nmemb;
}

// ── Constructor ───────────────────────────────────────────────────────────────
HttpClient::HttpClient(const std::string& backend_url, const std::string& token_path)
    : backend_url_(backend_url), token_path_(token_path)
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
    refreshToken();
}

// ── Token management ──────────────────────────────────────────────────────────
std::string HttpClient::readToken() {
    std::ifstream f(token_path_);
    if (!f.good()) return "";
    std::string tok;
    std::getline(f, tok);
    // Trim whitespace
    while (!tok.empty() && (tok.back() == '\n' || tok.back() == '\r' || tok.back() == ' '))
        tok.pop_back();
    return tok;
}

void HttpClient::refreshToken() {
    token_ = readToken();
    if (token_.empty())
        spdlog::warn("[HTTP] Token file empty or missing: {}", token_path_);
}

// ── Embedding → JSON array ────────────────────────────────────────────────────
std::string HttpClient::embeddingToJson(const Embedding& emb) {
    // Build compact JSON array: [0.123,0.456,...] (no spaces)
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < emb.size(); ++i) {
        if (i) ss << ",";
        ss << std::fixed << std::setprecision(6) << emb[i];
    }
    ss << "]";
    return ss.str();
}

// ── Low-level POST ────────────────────────────────────────────────────────────
std::pair<std::string, int> HttpClient::post(const std::string& path,
                                              const std::string& body) {
    CURL* curl = curl_easy_init();
    if (!curl) return {"", -1};

    std::string url = backend_url_ + path;
    std::string response;
    long http_code = 0;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_hdr = "Authorization: Bearer " + token_;
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, "x-tenant-id: 1");
    headers = curl_slist_append(headers, "x-customer-id: 1");
    headers = curl_slist_append(headers, "x-site-id: 1");

    curl_easy_setopt(curl, CURLOPT_URL,            url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,      headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,      body.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE,   (long)body.size());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,   writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,       &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,         8L);
    curl_easy_setopt(curl, CURLOPT_FRESH_CONNECT,   1L);
    curl_easy_setopt(curl, CURLOPT_FORBID_REUSE,    1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,  4L);

    CURLcode res = curl_easy_perform(curl);
    if (res == CURLE_OK)
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    else
        spdlog::error("[HTTP] curl error: {}", curl_easy_strerror(res));

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return {response, (int)http_code};
}

// ── Face recognition ──────────────────────────────────────────────────────────
RecognitionResult HttpClient::recognize(const RecognitionPayload& payload) {
    // Re-read token in case it was refreshed on disk by cron
    refreshToken();

    json body;
    body["embedding"]   = json::parse(embeddingToJson(payload.embedding));
    body["confidence"]  = payload.confidence;
    body["deviceId"]    = payload.device_code;
    body["timestamp"]   = payload.timestamp;
    body["direction"]   = payload.direction;
    body["trackId"]     = payload.track_id;

    auto [resp_body, code] = post("/api/face/recognize", body.dump());

    RecognitionResult result;
    result.http_code = code;
    result.matched   = false;

    if (code == 200) {
        try {
            auto j = json::parse(resp_body);
            auto& r = j["result"];
            result.matched       = true;
            result.employee_id   = r.value("employeeId",   "");
            result.full_name     = r.value("fullName",     "");
            result.employee_code = r.value("employeeCode", "");
            result.similarity    = r.value("similarity",   0.0f);
        } catch (...) {
            spdlog::error("[HTTP] JSON parse error: {}", resp_body.substr(0, 200));
        }
    } else if (code == 404) {
        // Face not enrolled — not an error, just unknown visitor
        spdlog::debug("[{}] Face not enrolled", payload.device_id);
    } else if (code == 401) {
        spdlog::warn("[HTTP] Token expired — will refresh on next frame");
        token_.clear();
    } else {
        spdlog::error("[HTTP] recognize HTTP {}: {}", code, resp_body.substr(0, 200));
    }

    return result;
}

// ── Direct attendance mark ────────────────────────────────────────────────────
bool HttpClient::markAttendance(const std::string& employee_id,
                                 const std::string& device_code,
                                 const std::string& timestamp,
                                 const std::string& status) {
    json body;
    body["employeeId"] = employee_id;
    body["deviceId"]   = device_code;
    body["timestamp"]  = timestamp;
    body["status"]     = status;

    auto [resp_body, code] = post("/api/attendance/mark", body.dump());
    return code == 200 || code == 201;
}
