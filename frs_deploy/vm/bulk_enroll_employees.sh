#!/bin/bash
# vm/bulk_enroll_employees.sh
# Enrolls all active employees who don't yet have a face enrolled.
# For each un-enrolled employee:
#   1. Gets employee ID from backend
#   2. Calls Jetson C++ enroll server: POST /enroll {employee_id, cam_id}
#   3. Polls until enrolled or failed
#
# Usage: bash bulk_enroll_employees.sh
# Operator workflow: walk each employee in front of camera, script does the rest.

BACKEND_URL="http://172.20.100.222:8080"
KEYCLOAK_URL="http://172.20.100.222:9090"
JETSON_SIDECAR="http://172.18.3.202:5000"
CAM_ID="entrance-cam-01"
PAUSE_BETWEEN=8  # seconds between enrollments — time for next person to step up

echo "=================================================="
echo " FRS2 Bulk Face Enrollment"
echo " Backend: $BACKEND_URL"
echo " Jetson:  $JETSON_SIDECAR"
echo " Camera:  $CAM_ID"
echo "=================================================="

# ── Get auth token ────────────────────────────────────────────────────────────
echo ""
echo "Getting auth token..."
TOKEN_RESP=$(curl -s -X POST \
  "$KEYCLOAK_URL/realms/attendance/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=attendance-frontend&username=admin@company.com&password=admin123&grant_type=password")

TOKEN=$(echo "$TOKEN_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])" 2>/dev/null)
if [ -z "$TOKEN" ] || [ "$TOKEN" = "None" ]; then
  echo "❌ Could not get auth token"
  exit 1
fi
echo "  ✅ Token obtained"

# Get tenant ID
TENANT_ID=$(curl -s "$BACKEND_URL/api/auth/bootstrap" \
  -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['memberships'][0]['scope']['tenantId'])" 2>/dev/null)
echo "  ✅ Tenant ID: $TENANT_ID"

# ── Check Jetson sidecar ──────────────────────────────────────────────────────
echo ""
echo "Checking Jetson sidecar..."
HEALTH=$(curl -s --max-time 5 "$JETSON_SIDECAR/health" 2>/dev/null)
if ! echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('status')=='ok' else 1)" 2>/dev/null; then
  echo "❌ Jetson sidecar not reachable at $JETSON_SIDECAR"
  echo "   On Jetson: sudo systemctl start frs-runner"
  echo "   Then:      curl http://172.18.3.202:5000/health"
  exit 1
fi
echo "  ✅ Jetson sidecar online"
echo "  $(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"frames={d.get('frames_processed',0)} cameras={d.get('active_cameras',0)}\")" 2>/dev/null)"

# ── Get list of employees without face enrollment ─────────────────────────────
echo ""
echo "Fetching employee list..."
EMP_RESP=$(curl -s "$BACKEND_URL/api/employees" \
  -H "Authorization: Bearer $TOKEN" \
  -H "x-tenant-id: $TENANT_ID" \
  --max-time 10)

# Parse: get all active employees
EMPLOYEES=$(echo "$EMP_RESP" | python3 << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
rows = d.get('data', d if isinstance(d, list) else [])
for e in rows:
    emp_id = e.get('pk_employee_id') or e.get('id')
    name   = e.get('full_name') or e.get('name', '?')
    code   = e.get('employee_code', '')
    dept   = e.get('department_name', '')
    enrolled = e.get('face_enrolled', False)
    status = e.get('status', 'active')
    if status == 'active':
        print(f"{emp_id}\t{code}\t{name}\t{dept}\t{'enrolled' if enrolled else 'not_enrolled'}")
PYEOF
)

if [ -z "$EMPLOYEES" ]; then
  echo "❌ No active employees found"
  exit 1
fi

TOTAL=$(echo "$EMPLOYEES" | wc -l)
NOT_ENROLLED=$(echo "$EMPLOYEES" | grep "not_enrolled" | wc -l)
ALREADY=$(echo "$EMPLOYEES" | grep -v "not_enrolled" | wc -l)

echo "  Total active employees: $TOTAL"
echo "  Already enrolled:       $ALREADY"
echo "  Need enrollment:        $NOT_ENROLLED"

if [ "$NOT_ENROLLED" -eq 0 ]; then
  echo ""
  echo "✅ All employees already enrolled!"
  exit 0
fi

echo ""
echo "Employees to enroll:"
echo "$EMPLOYEES" | grep "not_enrolled" | awk -F'\t' '{printf "  [%s] %s — %s (%s)\n", $1, $2, $3, $4}'

echo ""
read -p "Start enrollment? Each employee should face camera $CAM_ID. Press Enter to begin or Ctrl+C to cancel..."

# ── Enroll loop ───────────────────────────────────────────────────────────────
SUCCESS=0; FAIL=0

while IFS=$'\t' read -r EMP_ID EMP_CODE EMP_NAME DEPT STATUS; do
  [ "$STATUS" = "enrolled" ] && continue

  echo ""
  echo "────────────────────────────────────────"
  echo "  Next: $EMP_NAME ($EMP_CODE)"
  echo "  Dept: $DEPT"
  echo ""
  echo "  ▶ Ask $EMP_NAME to stand in front of the camera"
  read -p "  Press Enter when ready (or 's' to skip): " CONFIRM
  
  [ "$CONFIRM" = "s" ] && echo "  Skipped." && continue

  echo "  Capturing and enrolling..."
  
  RESULT=$(curl -s --max-time 35 -X POST "$JETSON_SIDECAR/enroll" \
    -H "Content-Type: application/json" \
    -d "{\"employee_id\": \"$EMP_ID\", \"cam_id\": \"$CAM_ID\"}" \
    2>/dev/null)
  
  SUCCESS_FLAG=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('success','false'))" 2>/dev/null)
  
  if [ "$SUCCESS_FLAG" = "True" ] || [ "$SUCCESS_FLAG" = "true" ]; then
    CONF=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); c=d.get('confidence',0); print(f'{c*100:.0f}%' if c else 'Good')" 2>/dev/null)
    echo "  ✅ $EMP_NAME enrolled (quality: $CONF)"
    ((SUCCESS++))
  else
    ERR=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error',d.get('message','Unknown error')))" 2>/dev/null)
    echo "  ❌ Failed: $ERR"
    echo "     Tip: Ensure $EMP_NAME is facing the camera directly, well-lit, no glasses"
    ((FAIL++))
    
    read -p "  Retry? (Enter=yes, n=skip): " RETRY
    if [ "$RETRY" != "n" ]; then
      RESULT=$(curl -s --max-time 35 -X POST "$JETSON_SIDECAR/enroll" \
        -H "Content-Type: application/json" \
        -d "{\"employee_id\": \"$EMP_ID\", \"cam_id\": \"$CAM_ID\"}" \
        2>/dev/null)
      SUCCESS_FLAG=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('success','false'))" 2>/dev/null)
      if [ "$SUCCESS_FLAG" = "True" ] || [ "$SUCCESS_FLAG" = "true" ]; then
        echo "  ✅ $EMP_NAME enrolled on retry"
        ((FAIL--)); ((SUCCESS++))
      else
        echo "  ❌ Retry failed — skipping $EMP_NAME"
      fi
    fi
  fi
  
  # Small pause between enrollments
  if [ $PAUSE_BETWEEN -gt 0 ] && [ "$CONFIRM" != "s" ]; then
    echo "  Waiting ${PAUSE_BETWEEN}s before next enrollment..."
    sleep $PAUSE_BETWEEN
  fi

done <<< "$EMPLOYEES"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Enrollment Complete"
echo " ✅ Enrolled: $SUCCESS"
echo " ❌ Failed:   $FAIL"
echo "=================================================="
echo ""
echo "Verify in dashboard: http://172.20.100.222:5173"
echo "  HR Dashboard → Employee Management → face_enrolled column"
echo ""
if [ $FAIL -gt 0 ]; then
  echo "For failed employees, try:"
  echo "  1. Better lighting (face lit from front)"
  echo "  2. Remove glasses"
  echo "  3. Stand 0.5-1m from camera"
  echo "  4. Face directly toward camera (< 15° angle)"
fi
