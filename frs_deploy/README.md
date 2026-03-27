# FRS2 — Face Enrollment + C++ Runner Deployment Guide

## IPs
| Device      | IP              |
|-------------|-----------------|
| VM (backend)| 172.20.100.222  |
| Jetson Orin | 172.18.3.202    |
| Camera      | 172.18.3.201    |

Camera credentials: `admin / Mli@Frs!2026`

---

## Step 1 — VM: Patch enrollment routes (run once)

```bash
ssh administrator@172.20.100.222
cd ~/FRS_/FRS--Java-Verison

# Copy the fix script here and run it
bash vm/fix_enrollment_routes.sh
```

This adds `POST /api/employees/:id/enroll-face-direct` which accepts a
pre-computed 512-d embedding from the C++ runner. No Python sidecar needed.

## Step 2 — VM: Update enrollment UI

```bash
bash vm/fix_face_enrollment_ui.sh
```

This updates the `FaceEnrollButton` to show:
- **Enroll from Camera** button (uses Jetson C++ runner, primary method)
- **Jetson Online/Offline** indicator
- Photo upload fallback
- Clear error messages

## Step 3 — Copy FRS-CPP to Jetson

```bash
# From wherever you have FRS-CPP:
scp -r /path/to/FRS-CPP ubuntu@172.18.3.202:~/FRS-CPP
```

Also copy the deployment scripts:
```bash
scp jetson/J01_build_cpp_runner.sh ubuntu@172.18.3.202:~/
scp jetson/J02_test_connections.sh ubuntu@172.18.3.202:~/
scp jetson/J03_patch_enroll_server.sh ubuntu@172.18.3.202:~/
```

## Step 4 — Jetson: Build the C++ runner

```bash
ssh ubuntu@172.18.3.202
bash ~/J01_build_cpp_runner.sh
```

This will:
- Install build dependencies (cmake, libcurl, nlohmann-json, spdlog)
- Write `/opt/frs/config.json` (DLA0 detection engine + GPU embedding engine)
- Write `/opt/frs/cameras.json` (with real camera password)
- Build `frs_runner` binary with ninja
- Install as `/usr/local/bin/frs_runner`
- Install `frs-runner.service` systemd unit

## Step 5 — Jetson: Patch the enroll server

```bash
bash ~/J03_patch_enroll_server.sh
```

Replaces the stub `handleEnroll()` with the real implementation:
RTSP snapshot → YOLOv8 detection → ArcFace embedding → POST to backend.

## Step 6 — Jetson: Test all connections

```bash
bash ~/J02_test_connections.sh
```

Verifies:
- Camera ping + RTSP + ISAPI snapshot
- Backend health
- Keycloak token fetch
- Token accepted by backend

All ✅ required before proceeding.

## Step 7 — Jetson: Get initial token + start runner

```bash
# Get token
bash /opt/frs/token_manager.sh

# Install token refresh cron (every 25 min)
(crontab -l 2>/dev/null; echo "*/25 * * * * /opt/frs/token_manager.sh >> /var/log/frs_token.log 2>&1") | crontab -

# Start runner
sudo systemctl start frs-runner
sudo systemctl enable frs-runner

# Watch logs
sudo journalctl -u frs-runner -f
```

Expected startup output:
```
[Runner] Initialized with 1 camera(s)
[Runner] Inference worker 0 started
[Runner] Inference worker 1 started
[entrance-cam-01] Opened with HW decoder (nvv4l2decoder)
[EnrollServer] Listening on :5000
[Runner] All threads started — FRS is live
```

## Step 8 — Enroll employee faces

### Option A: One at a time via HR Dashboard
1. Open http://172.20.100.222:5173
2. Login → HR Dashboard → Employee Management
3. Click any employee row → expand the Face Enrollment section
4. Click **Enroll from Camera** (Jetson Online indicator must be green)
5. Ask employee to stand in front of the camera
6. Wait ~5-10 seconds for enrollment to complete

### Option B: Bulk enrollment (all employees at once)
```bash
# On VM
bash vm/bulk_enroll_employees.sh
```
Walks you through each un-enrolled employee interactively.

## Step 9 — Verify recognition is working

Walk in front of the camera. Within 10 seconds you should see in runner logs:
```
[entrance-cam-01] ✅ Sarah Johnson (EMP001) sim=0.847
```

And in the HR Dashboard → Attendance History, a new record appears.

---

## Troubleshooting

### frs-runner won't start
```bash
sudo journalctl -u frs-runner -n 50 --no-pager
```

Common issues:
- `Cannot open config` — check `/opt/frs/config.json` exists
- `Cannot open engine` — verify engine paths in config.json match actual files in `/opt/frs-models/trt/`
- `Token file empty` — run `bash /opt/frs/token_manager.sh` first

### Enrollment fails: "No face detected"
- Ensure the employee is 0.5-1.5m from camera
- Check lighting — face should be evenly lit from the front
- Verify camera is reachable: `curl -s http://admin:Mli@Frs!2026@172.18.3.201:80/ISAPI/System/deviceInfo`

### Recognition not marking attendance
- Check face_embedded count: `curl http://172.18.3.202:5000/health | python3 -m json.tool`
- Verify pgvector has embeddings: 
  ```bash
  docker exec attendance-postgres psql -U postgres -d attendance_intelligence \
    -c "SELECT count(*) FROM employee_face_embeddings;"
  ```
- Lower match threshold if similarity is consistently 0.50-0.55 (edit `/opt/frs/config.json`: `"match_threshold": 0.50`)

### Camera sub-stream vs main stream
- Attendance: uses channel 102 (sub-stream, 720p, 5fps) — set in cameras.json
- Enrollment: uses channel 101 (main stream, full res) — enroll server switches automatically

---

## File Reference

| File | Purpose |
|------|---------|
| `/opt/frs/config.json` | Runner config (model paths, thresholds, backend URL) |
| `/opt/frs/cameras.json` | Camera list with RTSP URLs |
| `/opt/frs/device_token.txt` | Keycloak JWT (refreshed by cron every 25 min) |
| `/opt/frs/token_manager.sh` | Token refresh script |
| `/opt/frs-models/trt/yolov8n-face-dla0-fp16.engine` | YOLOv8 face detector (DLA0) |
| `/opt/frs-models/trt/arcface-r50-fp16.engine` | ArcFace R50 embedder (GPU) |
| `/usr/local/bin/frs_runner` | Compiled C++ binary |
