#!/bin/bash
# J02_test_connections.sh
# Tests camera RTSP, backend reachability, and Keycloak token fetch
# Run BEFORE starting frs_runner to verify everything is wired up.

BACKEND_IP="172.20.100.222"
CAM_IP="172.18.3.201"
CAM_PASS="Mli@Frs!2026"
KEYCLOAK_URL="http://$BACKEND_IP:9090"
BACKEND_URL="http://$BACKEND_IP:8080"

PASS=0; FAIL=0
ok()  { echo "  ✅ $1"; ((PASS++)); }
err() { echo "  ❌ $1"; ((FAIL++)); }
warn(){ echo "  ⚠  $1"; }

echo "=================================================="
echo " FRS2 Connection Tests"
echo " $(date)"
echo "=================================================="

# ── 1. Network reachability ───────────────────────────────────────────────────
echo ""
echo "[1/5] Network reachability..."

ping -c 2 -W 2 $CAM_IP &>/dev/null \
  && ok "Camera $CAM_IP reachable" \
  || err "Camera $CAM_IP UNREACHABLE — check network/subnet routing"

ping -c 2 -W 2 $BACKEND_IP &>/dev/null \
  && ok "Backend VM $BACKEND_IP reachable" \
  || err "Backend VM $BACKEND_IP UNREACHABLE — check routing between subnets"

# ── 2. Camera RTSP test ───────────────────────────────────────────────────────
echo ""
echo "[2/5] Camera RTSP stream..."

RTSP_SUB="rtsp://admin:$CAM_PASS@$CAM_IP:554/Streaming/Channels/102"
RTSP_MAIN="rtsp://admin:$CAM_PASS@$CAM_IP:554/Streaming/Channels/101"

# Test sub-stream (used for live attendance)
if timeout 8 ffprobe -v quiet -rtsp_transport tcp \
    -i "$RTSP_SUB" -show_streams -select_streams v 2>/dev/null | grep -q "codec_name"; then
  ok "Sub-stream (ch1/102) — RTSP accessible"
else
  # Try curl RTSP OPTIONS
  HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 5 "rtsp://admin:$CAM_PASS@$CAM_IP:554/Streaming/Channels/102" 2>/dev/null || echo "000")
  warn "ffprobe test inconclusive (ffprobe may not be installed) — trying alternate test"
  
  # Test ISAPI HTTP instead
  ISAPI_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time 5 "http://admin:$CAM_PASS@$CAM_IP:80/ISAPI/System/deviceInfo" 2>/dev/null)
  [ "$ISAPI_CODE" = "200" ] \
    && ok "Camera ISAPI reachable (HTTP 200) — RTSP should work" \
    || err "Camera ISAPI returned HTTP $ISAPI_CODE — wrong password or camera off?"
fi

# ── 3. Camera snapshot test ───────────────────────────────────────────────────
echo ""
echo "[3/5] Camera ISAPI snapshot..."

SNAP_URL="http://admin:$CAM_PASS@$CAM_IP:80/ISAPI/Streaming/channels/101/picture"
SNAP_FILE="/tmp/frs_snap_test.jpg"

SNAP_CODE=$(curl -s -o "$SNAP_FILE" -w "%{http_code}" --max-time 10 "$SNAP_URL" 2>/dev/null)
if [ "$SNAP_CODE" = "200" ] && [ -s "$SNAP_FILE" ]; then
  SIZE=$(wc -c < "$SNAP_FILE")
  ok "Snapshot captured (${SIZE} bytes) → $SNAP_FILE"
  echo "     Preview: scp ubuntu@172.18.3.202:$SNAP_FILE ./test_snap.jpg"
else
  err "Snapshot failed (HTTP $SNAP_CODE) — check camera IP and credentials"
fi

# ── 4. Backend health check ───────────────────────────────────────────────────
echo ""
echo "[4/5] Backend API health..."

HEALTH=$(curl -s --max-time 5 "$BACKEND_URL/api/health" 2>/dev/null)
if echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='UP' else 1)" 2>/dev/null; then
  ok "Backend health UP — $BACKEND_URL"
  DB_STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('services',{}).get('database','?'))" 2>/dev/null)
  ok "Database: $DB_STATUS"
else
  err "Backend health check failed — is the Docker stack running on the VM?"
  echo "     On VM: cd ~/FRS_/FRS--Java-Verison && docker compose ps"
fi

# ── 5. Keycloak token fetch ───────────────────────────────────────────────────
echo ""
echo "[5/5] Keycloak auth token..."

TOKEN_RESP=$(curl -s --max-time 10 -X POST \
  "$KEYCLOAK_URL/realms/attendance/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=attendance-frontend&username=admin@company.com&password=admin123&grant_type=password" \
  2>/dev/null)

TOKEN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)

if [ -n "$TOKEN" ] && [ "$TOKEN" != "None" ] && [ ${#TOKEN} -gt 50 ]; then
  ok "Keycloak token obtained (${#TOKEN} chars)"

  # Write token to file
  mkdir -p /opt/frs
  echo "$TOKEN" > /opt/frs/device_token.txt
  chmod 600 /opt/frs/device_token.txt
  ok "Token written to /opt/frs/device_token.txt"

  # Verify token works against backend
  BOOT=$(curl -s --max-time 5 "$BACKEND_URL/api/auth/bootstrap" \
    -H "Authorization: Bearer $TOKEN" 2>/dev/null)
  if echo "$BOOT" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('user') else 1)" 2>/dev/null; then
    EMAIL=$(echo "$BOOT" | python3 -c "import sys,json; print(json.load(sys.stdin)['user']['email'])" 2>/dev/null)
    ok "Token valid against backend — user: $EMAIL"
    TENANT_ID=$(echo "$BOOT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['memberships'][0]['scope']['tenantId'])" 2>/dev/null)
    ok "Tenant ID: $TENANT_ID"
  else
    err "Token not accepted by backend — check backend logs"
  fi
else
  err "Keycloak token fetch failed"
  echo "     Keycloak response: $(echo $TOKEN_RESP | head -c 200)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Results: ✅ $PASS passed   ❌ $FAIL failed"
echo "=================================================="

if [ $FAIL -gt 0 ]; then
  echo ""
  echo "Fix failures before starting frs-runner."
  exit 1
else
  echo ""
  echo "All checks passed! Run:"
  echo "  sudo systemctl start frs-runner"
  echo "  sudo journalctl -u frs-runner -f"
fi
