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

// ── Base64 Helper ────────────────────────────────────────────────────────────
std::string HttpClient::base64_encode(const unsigned char* buf, unsigned int bufLen) {
    static const char* base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string ret;
    int i = 0, j = 0;
    unsigned char char_array_3[3], char_array_4[4];
    while (bufLen--) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for (i = 0; i < 4; i++) ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
        for (j = i; j < 3; j++) char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (j = 0; j < i + 1; j++) ret += base64_chars[char_array_4[j]];
        while (i++ < 3) ret += '=';
    }
    return ret;
}

// ── Constructor ───────────────────────────────────────────────────────────────
HttpClient::HttpClient(const std::string& backend_url, const std::string& token_path,
                       int tenant_id, int customer_id, int site_id,
                       const std::string& keycloak_url,
                       const std::string& keycloak_client_id,
                       const std::string& keycloak_username,
                       const std::string& keycloak_password)
    : backend_url_(backend_url), token_path_(token_path),
      tenant_id_(tenant_id), customer_id_(customer_id), site_id_(site_id),
      keycloak_url_(keycloak_url), keycloak_client_id_(keycloak_client_id),
      keycloak_username_(keycloak_username), keycloak_password_(keycloak_password),
      token_expiry_(std::chrono::steady_clock::now())  // expired → triggers fetch on first use
{
    curl_global_init(CURL_GLOBAL_DEFAULT);
    ensureValidToken();
}

// ── Token management ──────────────────────────────────────────────────────────
std::string HttpClient::readToken() {
    std::ifstream f(token_path_);
    if (!f.good()) return "";
    std::string tok;
    std::getline(f, tok);
    while (!tok.empty() && (tok.back() == '\n' || tok.back() == '\r' || tok.back() == ' '))
        tok.pop_back();
    return tok;
}

// Called with token_mutex_ already held.
bool HttpClient::fetchTokenFromKeycloak() {
    if (keycloak_url_.empty() || keycloak_username_.empty() || keycloak_password_.empty()) {
        spdlog::warn("[HTTP] Keycloak credentials not configured — cannot auto-refresh token");
        return false;
    }

    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string response;
    std::string post_fields =
        "grant_type=password"
        "&client_id=" + keycloak_client_id_ +
        "&username="  + keycloak_username_ +
        "&password="  + keycloak_password_;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");

    curl_easy_setopt(curl, CURLOPT_URL,           keycloak_url_.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS,     post_fields.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    if (res == CURLE_OK)
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK || http_code != 200) {
        spdlog::error("[HTTP] Keycloak token fetch failed: curl={} http={}", curl_easy_strerror(res), http_code);
        return false;
    }

    try {
        auto j = json::parse(response);
        std::string new_token = j["access_token"];
        int expires_in        = j.value("expires_in", 1800);

        // Expire 60s early to avoid using a token right at its boundary
        token_expiry_ = std::chrono::steady_clock::now()
                      + std::chrono::seconds(expires_in - 60);

        std::ofstream f(token_path_, std::ios::trunc);
        if (f.good()) f << new_token;

        token_ = new_token;
        spdlog::info("[HTTP] Token refreshed from Keycloak (valid for {}s)", expires_in - 60);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("[HTTP] Failed to parse Keycloak response: {}", e.what());
        return false;
    }
}

// Thread-safe: ensures token_ is valid before a request.
// Returns false if no valid token could be obtained.
bool HttpClient::ensureValidToken() {
    std::lock_guard<std::mutex> lock(token_mutex_);

    // Token still valid — use it
    if (!token_.empty() &&
        std::chrono::steady_clock::now() < token_expiry_)
        return true;

    // Try reading a valid token written by another process (e.g. manual update)
    std::string file_tok = readToken();
    if (!file_tok.empty() && file_tok != token_) {
        token_        = file_tok;
        token_expiry_ = std::chrono::steady_clock::now() + std::chrono::seconds(1740);
        return true;
    }

    return fetchTokenFromKeycloak();
}

void HttpClient::refreshToken() {
    ensureValidToken();
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
    headers = curl_slist_append(headers, ("x-tenant-id: " + std::to_string(tenant_id_)).c_str());
    headers = curl_slist_append(headers, ("x-customer-id: " + std::to_string(customer_id_)).c_str());
    headers = curl_slist_append(headers, ("x-site-id: " + std::to_string(site_id_)).c_str());

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

// ── Low-level GET ─────────────────────────────────────────────────────────────
std::pair<std::string, int> HttpClient::get(const std::string& path) {
    ensureValidToken();
    CURL* curl = curl_easy_init();
    if (!curl) return {"", -1};

    std::string url = backend_url_ + path;
    std::string response;
    long http_code = 0;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    std::string auth_hdr = "Authorization: Bearer " + token_;
    headers = curl_slist_append(headers, auth_hdr.c_str());
    headers = curl_slist_append(headers, ("x-tenant-id: " + std::to_string(tenant_id_)).c_str());
    headers = curl_slist_append(headers, ("x-customer-id: " + std::to_string(customer_id_)).c_str());
    headers = curl_slist_append(headers, ("x-site-id: " + std::to_string(site_id_)).c_str());

    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER,     headers);
    curl_easy_setopt(curl, CURLOPT_HTTPGET,        1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,  writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,      &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        8L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 4L);

    CURLcode res = curl_easy_perform(curl);
    if (res == CURLE_OK)
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    else
        spdlog::error("[HTTP] GET curl error: {}", curl_easy_strerror(res));

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return {response, (int)http_code};
}

// ── Face recognition ──────────────────────────────────────────────────────────
std::string HttpClient::payloadToJson(const RecognitionPayload& payload) {
    json body;
    body["embedding"]   = json::parse(embeddingToJson(payload.embedding));
    body["confidence"]  = payload.confidence;
    body["deviceId"]    = payload.device_code;
    body["timestamp"]   = payload.timestamp;
    body["direction"]   = payload.direction;
    body["trackId"]     = payload.track_id;
    body["frame"]       = payload.face_crop_b64; 
    
    if (!payload.scaled_box.empty()) {
        body["box"] = payload.scaled_box;
    } else {
        body["box"] = nullptr;
    }
    
    return body.dump();
}

static RecognitionResult parseRecognitionResponse(const std::string& body, int code) {
    RecognitionResult result;
    result.http_code = code;
    if (code != 200) return result;
    try {
        auto j = json::parse(body);
        if (j.value("recognized", false)) {
            auto& r          = j["result"];
            result.matched       = true;
            result.employee_id   = r.value("employeeId",   "");
            result.full_name     = r.value("fullName",     "");
            result.employee_code = r.value("employeeCode", "");
            result.similarity    = r.value("similarity",   0.0f);
        }
    } catch (const std::exception& e) {
        spdlog::error("[HTTP] JSON parse error: {} | Body: {}", e.what(), body.substr(0, 200));
    }
    return result;
}

RecognitionResult HttpClient::recognize(const std::string& endpoint, const RecognitionPayload& payload) {
    ensureValidToken();
    auto [resp_body, code] = post(endpoint, payloadToJson(payload));

    if (code == 200 || code == 404) {
        if (code == 404)
            spdlog::info("[{}] Backend 404 (not enrolled): {}", payload.device_id, resp_body.substr(0, 150));
        else
            spdlog::info("[{}] Backend 200: {}", payload.device_id, resp_body.substr(0, 150));
        return parseRecognitionResponse(resp_body, code);
    }

    if (code == 401) {
        // Force a fresh token fetch (one thread at a time) then retry once
        spdlog::warn("[HTTP] 401 on recognize — forcing token refresh");
        {
            std::lock_guard<std::mutex> lock(token_mutex_);
            token_.clear();
            fetchTokenFromKeycloak();
        }
        auto [retry_body, retry_code] = post(endpoint, payloadToJson(payload));
        if (retry_code != 200 && retry_code != 404)
            spdlog::error("[HTTP] recognize still {} after token refresh", retry_code);
        return parseRecognitionResponse(retry_body, retry_code);
    }

    spdlog::error("[HTTP] recognize HTTP {} for {}: {}", code, payload.device_id, resp_body.substr(0, 200));
    return RecognitionResult{};
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
