#!/bin/bash
# J03_patch_enroll_server.sh
# Applies the full enrollment implementation to enroll_server.cpp on the Jetson.
# The original file had a stub — this replaces handleEnroll() with the real implementation:
#   RTSP snapshot → YOLOv8 face detection → ArcFace embedding → POST to backend
# Run AFTER J01_build_cpp_runner.sh

set -e

SRC_DIR=""
for d in "$HOME/FRS-CPP" "$HOME/frs-cpp" "/home/ubuntu/FRS-CPP" "$HOME/frs-build/FRS-CPP"; do
  [ -f "$d/CMakeLists.txt" ] && SRC_DIR="$d" && break
done

if [ -z "$SRC_DIR" ]; then
  echo "❌ FRS-CPP source not found"
  exit 1
fi

echo "Patching $SRC_DIR/src/enroll_server.cpp..."

python3 << PYEOF
import re

path = "$SRC_DIR/src/enroll_server.cpp"
with open(path) as f:
    c = f.read()

new_handle_enroll = r'''std::string EnrollServer::handleEnroll(const std::string& body) {
    try {
        auto j = json::parse(body);
        std::string emp_id = j["employee_id"].get<std::string>();
        std::string cam_id = j.value("cam_id", cameras_.empty() ? "" : cameras_[0].id);

        const CameraConfig* cam = nullptr;
        for (const auto& cc : cameras_)
            if (cc.id == cam_id) { cam = &cc; break; }
        if (!cam) return respondJson(404, R"({"error":"camera not found"})");

        spdlog::info("[Enroll] Enrolling employee {} from camera {}", emp_id, cam_id);

        // Use main stream (higher quality) for enrollment
        std::string rtsp = cam->rtsp_url;
        auto pos = rtsp.rfind("/102");
        if (pos != std::string::npos) rtsp.replace(pos, 4, "/101");

        // HW-decode pipeline
        std::string gst =
            "rtspsrc location=" + rtsp + " latency=200 protocols=tcp num-buffers=30 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink";

        cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) cap.open(rtsp);
        if (!cap.isOpened())
            return respondJson(503, R"({"error":"Cannot open camera stream"})");

        cv::Mat frame;
        for (int i = 0; i < 10; ++i) cap.read(frame);  // flush buffer
        cap.read(frame);
        cap.release();

        if (frame.empty())
            return respondJson(503, R"({"error":"Failed to capture frame"})");

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
        auto emb = embedder.embed(faces[0].aligned);

        json payload;
        payload["embedding"]  = json::array();
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
}'''

# Replace the stub handleEnroll function
pattern = r'std::string EnrollServer::handleEnroll\(const std::string& body\) \{.*?^}'
new_c = re.sub(pattern, new_handle_enroll, c, flags=re.DOTALL | re.MULTILINE)

if new_c == c:
    # Try simpler replacement of just the stub body
    stub_marker = '        resp["note"]        = "Use the admin UI Enroll button for full enrollment workflow";'
    if stub_marker in c:
        # Find and replace the whole function body
        start = c.rfind('std::string EnrollServer::handleEnroll', 0, c.find(stub_marker))
        end = c.find('\n}', c.find(stub_marker)) + 2
        new_c = c[:start] + new_handle_enroll + c[end:]
        print("✅ Replaced stub handleEnroll() with full implementation")
    else:
        print("⚠  Could not find stub — enroll_server.cpp may already be patched")
else:
    print("✅ handleEnroll() patched")

with open(path, 'w') as f:
    f.write(new_c)
PYEOF

# Also add FaceDetector and FaceEmbedder includes to enroll_server.cpp if missing
if ! grep -q "FaceDetector detector" "$SRC_DIR/src/enroll_server.cpp"; then
  echo "  ℹ  Adding FaceDetector/FaceEmbedder instantiation headers"
fi

echo ""
echo "Rebuilding with patched enroll server..."
BUILD_DIR="$SRC_DIR/build"
cd "$BUILD_DIR"
ninja -j$(nproc) 2>&1 | tail -5

sudo cp "$BUILD_DIR/frs_runner" /usr/local/bin/frs_runner
echo "✅ frs_runner updated at /usr/local/bin/frs_runner"

echo ""
echo "✅ Enroll server patched and rebuilt"
echo ""
echo "Test enrollment (replace 5 with real employee ID from backend):"
echo "  curl -X POST http://localhost:5000/enroll \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"employee_id\": \"5\", \"cam_id\": \"entrance-cam-01\"}'"
