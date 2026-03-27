#!/bin/bash
# vm/fix_enrollment_routes.sh
# Adds a direct-embedding enrollment endpoint to the backend so:
#   1. Admin can upload a photo via the HR UI (uses EdgeAI C++ sidecar on Jetson)
#   2. The C++ runner's EnrollServer can POST a pre-computed 512-d embedding directly
#   3. Enrollment works even when the Jetson sidecar is not reachable (offline enrollment)
#
# Run on VM: bash ~/FRS_/FRS--Java-Verison/vm/fix_enrollment_routes.sh

set -e
PROJECT="$HOME/FRS_/FRS--Java-Verison"
cd "$PROJECT"

echo "=================================================="
echo " Patching enrollment routes"
echo "=================================================="

# ── 1. Add /api/employees/:id/enroll-face-direct route ───────────────────────
# This accepts a pre-computed embedding — no EdgeAI sidecar call needed.
# Used by:
#   - C++ runner's EnrollServer POST /enroll
#   - Admin UI when you want to enroll from camera snapshot

python3 << 'PYEOF'
import os, re

path = os.path.expanduser("~/FRS_/FRS--Java-Verison/backend/src/routes/employeeRoutes.js")
with open(path) as f:
    c = f.read()

# Check if direct route already exists
if 'enroll-face-direct' in c:
    print("✅ Direct enrollment route already present")
else:
    # Append before the export line
    new_route = '''
// ── POST /api/employees/:employeeId/enroll-face-direct
// Accepts a pre-computed 512-d ArcFace embedding from the C++ runner.
// No EdgeAI sidecar call needed — embedding was computed on Jetson hardware.
// Body: { "embedding": [0.12, -0.34, ...] (512 floats), "confidence": 0.92 }
router.post(
  "/:employeeId/enroll-face-direct",
  requirePermission("users.write"),
  asyncHandler(async (req, res) => {
    const { employeeId } = req.params;
    const { embedding, confidence, source } = req.body;

    if (!Array.isArray(embedding) || embedding.length !== 512) {
      return res.status(400).json({
        message: "embedding must be a 512-element float array (ArcFace output)",
      });
    }

    // Validate it looks like an L2-normalized vector
    const norm = Math.sqrt(embedding.reduce((s, v) => s + v*v, 0));
    if (norm < 0.9 || norm > 1.1) {
      return res.status(422).json({
        message: `Embedding must be L2-normalized (norm=${norm.toFixed(3)}). ` +
                 "Apply L2 normalization before sending.",
      });
    }

    // Store in pgvector
    const vectorStr = `[${embedding.join(",")}]`;
    await pool.query(
      `INSERT INTO employee_face_embeddings
         (employee_id, embedding, quality_score, is_primary, enrolled_by, model_version)
       VALUES ($1, $2::vector, $3, TRUE, $4, $5)
       ON CONFLICT DO NOTHING`,
      [
        employeeId,
        vectorStr,
        confidence || null,
        req.auth?.user?.id || "cpp-runner",
        "arcface-r50-fp16",
      ]
    );

    // Sync to SQLite FaceDB fallback
    await faceDB.addFace(embedding, {
      id:         `emp_${employeeId}_${Date.now()}`,
      employeeId: String(employeeId),
      source:     source || "cpp-runner",
      enrolledAt: new Date().toISOString(),
    });

    return res.status(201).json({
      success:    true,
      employeeId,
      confidence: confidence || null,
      source:     source || "cpp-runner",
      message:    "Face enrolled successfully via direct embedding. Employee will be recognised immediately.",
    });
  })
);

// ── GET /api/employees/:employeeId/enroll-face-direct
// Quick enrollment status check (same as /enroll-face but separate path for C++ runner)
router.get(
  "/:employeeId/enroll-face-direct",
  requirePermission("users.read"),
  asyncHandler(async (req, res) => {
    const { rows } = await pool.query(
      `SELECT count(*)::int as c FROM employee_face_embeddings WHERE employee_id = $1`,
      [req.params.employeeId]
    );
    return res.json({ enrolled: rows[0].c > 0, count: rows[0].c });
  })
);

'''
    # Insert before the export line
    c = c.replace('export { router as employeeRoutes };',
                  new_route + 'export { router as employeeRoutes };')
    print("✅ Direct enrollment route added")

with open(path, 'w') as f:
    f.write(c)
PYEOF

# ── 2. Fix existing /enroll-face to return helpful error when sidecar is offline ─
python3 << 'PYEOF'
import os

path = os.path.expanduser("~/FRS_/FRS--Java-Verison/backend/src/routes/employeeRoutes.js")
with open(path) as f:
    c = f.read()

# Improve the 503 error message to tell user about direct embedding alternative
old = '''    } catch (e) {
      return res.status(503).json({
        message: "EdgeAI sidecar is unavailable: " + e.message +
          ". Ensure the Python sidecar is running on port 5000.",
      });
    }'''

new = '''    } catch (e) {
      return res.status(503).json({
        message: "EdgeAI sidecar is unavailable: " + e.message,
        hint: "The C++ enrollment server on the Jetson (port 5000) handles this. " +
              "Alternatively use POST /api/employees/:id/enroll-face-direct with a pre-computed embedding.",
        sidecarUrl: process.env.EDGE_AI_URL || "http://172.18.3.202:5000",
      });
    }'''

if old in c:
    c = c.replace(old, new)
    print("✅ 503 error message improved")
else:
    print("ℹ  503 message already updated or pattern not found")

with open(path, 'w') as f:
    f.write(c)
PYEOF

# ── 3. Also update the C++ EnrollServer to POST to the correct endpoint ──────
# The existing enroll_server.cpp handleEnroll() is a stub — patch it properly
# We'll create a patch file to copy to the Jetson

cat > /tmp/enroll_server_patch.cpp << 'CPPEOF'
// PATCH: Replace handleEnroll() stub in enroll_server.cpp
// This version:
//   1. Opens a GStreamer RTSP snapshot from the camera
//   2. Runs YOLO detection + ArcFace embedding
//   3. POSTs the embedding to /api/employees/:id/enroll-face-direct

std::string EnrollServer::handleEnroll(const std::string& body) {
    try {
        auto j = json::parse(body);
        std::string emp_id = j["employee_id"].get<std::string>();
        std::string cam_id = j.value("cam_id", cameras_.empty() ? "" : cameras_[0].id);

        // Find camera
        const CameraConfig* cam = nullptr;
        for (const auto& c : cameras_)
            if (c.id == cam_id) { cam = &c; break; }
        if (!cam) return respondJson(404, R"({"error":"camera not found"})");

        spdlog::info("[Enroll] Starting enrollment for employee {} from camera {}", emp_id, cam_id);

        // ── Capture single frame from camera via OpenCV/GStreamer ────────────
        // Use the main RTSP stream for highest quality enrollment photo
        std::string rtsp_main = cam->rtsp_url;
        // Replace /102 (sub-stream) with /101 (main stream) if present
        auto pos = rtsp_main.rfind("/102");
        if (pos != std::string::npos) rtsp_main.replace(pos, 4, "/101");

        spdlog::info("[Enroll] Connecting to {} for enrollment snapshot", rtsp_main.substr(0, 40) + "...");

        // Build GStreamer HW pipeline
        std::string gst_pipeline =
            "rtspsrc location=" + rtsp_main + " latency=200 protocols=tcp num-buffers=1 ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! "
            "nvvidconv ! video/x-raw,format=BGRx,width=1280,height=720 ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink";

        cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);
        if (!cap.isOpened()) {
            // Fallback: plain RTSP
            spdlog::warn("[Enroll] HW decoder failed, trying plain RTSP");
            cap.open(rtsp_main);
        }

        if (!cap.isOpened()) {
            return respondJson(503, R"({"error":"Cannot connect to camera RTSP stream"})");
        }

        cv::Mat frame;
        // Read a few frames to get past initial buffering
        for (int i = 0; i < 5; ++i) cap.read(frame);
        cap.release();

        if (frame.empty()) {
            return respondJson(503, R"({"error":"Failed to capture frame from camera"})");
        }

        spdlog::info("[Enroll] Frame captured ({}x{}), running detection...", frame.cols, frame.rows);

        // ── Run face detection ───────────────────────────────────────────────
        // Access runner's detector through a shared reference
        // We need FaceDetector + FaceEmbedder — use a fresh instance
        FaceDetector detector(cfg_.det_engine, 0.4f, 0.4f);
        auto faces = detector.detect(frame);

        if (faces.empty()) {
            return respondJson(422, R"({"error":"No face detected in camera frame. Ensure the employee is facing the camera."})");
        }
        if (faces.size() > 1) {
            json err;
            err["error"] = "Multiple faces detected (" + std::to_string(faces.size()) +
                           "). Only one person should be in front of the camera.";
            return respondJson(422, err.dump());
        }

        float conf = faces[0].conf;
        spdlog::info("[Enroll] Face detected (conf={:.2f}), computing embedding...", conf);

        // ── Compute ArcFace embedding ────────────────────────────────────────
        FaceEmbedder embedder(cfg_.emb_engine);
        auto emb = embedder.embed(faces[0].aligned);

        // ── POST to backend /api/employees/:id/enroll-face-direct ────────────
        json payload;
        payload["embedding"]  = json::array();
        for (float v : emb) payload["embedding"].push_back(v);
        payload["confidence"] = conf;
        payload["source"]     = "cpp-enroll-server";

        HttpClient http(cfg_.backend_url, cfg_.token_path);
        auto [resp_body, code] = http.post(
            "/api/employees/" + emp_id + "/enroll-face-direct",
            payload.dump()
        );

        if (code == 201) {
            json resp;
            resp["success"]     = true;
            resp["employee_id"] = emp_id;
            resp["confidence"]  = conf;
            resp["message"]     = "Face enrolled successfully";
            spdlog::info("[Enroll] ✅ Employee {} enrolled (conf={:.2f})", emp_id, conf);
            return respondJson(201, resp.dump());
        } else {
            spdlog::error("[Enroll] Backend returned HTTP {}: {}", code, resp_body.substr(0, 200));
            json err;
            err["error"]     = "Backend enrollment failed";
            err["http_code"] = code;
            err["detail"]    = resp_body.substr(0, 500);
            return respondJson(500, err.dump());
        }

    } catch (const std::exception& e) {
        spdlog::error("[Enroll] Exception: {}", e.what());
        json err; err["error"] = e.what();
        return respondJson(500, err.dump());
    }
}
CPPEOF

echo "  ✅ EnrollServer patch written to /tmp/enroll_server_patch.cpp"
echo "     Apply on Jetson: see J03_patch_enroll_server.sh"

# ── 4. Rebuild backend ────────────────────────────────────────────────────────
echo ""
echo "Rebuilding backend..."
docker compose build backend 2>&1 | tail -3
docker compose up -d backend
sleep 8

# Verify new route exists
TOKEN=$(curl -s -X POST \
  "http://172.20.100.222:9090/realms/attendance/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=attendance-frontend&username=admin@company.com&password=admin123&grant_type=password" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null)

CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "http://172.20.100.222:8080/api/employees/1/enroll-face-direct" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"embedding":[], "confidence": 0.9}' \
  --max-time 5)

if [ "$CODE" = "400" ]; then
  echo "  ✅ /api/employees/:id/enroll-face-direct is live (returns 400 for empty embedding — correct)"
elif [ "$CODE" = "201" ]; then
  echo "  ✅ Route is live"
else
  echo "  ⚠  Route returned HTTP $CODE — check backend logs"
fi

echo ""
echo "=================================================="
echo " ✅ Enrollment routes patched"
echo "=================================================="
echo ""
echo "NEXT: On Jetson, run J03_patch_enroll_server.sh"
echo "      Then enroll faces via HR Dashboard"
