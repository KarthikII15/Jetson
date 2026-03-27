#!/bin/bash
# token_manager.sh — Get device JWT from Keycloak and keep it refreshed
# Run once manually, then cron every 25 minutes:
#   */25 * * * * /opt/frs/token_manager.sh >> /var/log/frs_token.log 2>&1

set -e

KEYCLOAK_URL="http://172.20.100.222:9090"
REALM="attendance"
CLIENT_ID="attendance-frontend"
USERNAME="admin@company.com"
PASSWORD="admin123"
TOKEN_FILE="/opt/frs/device_token.txt"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] [token_manager]"

echo "$LOG_PREFIX Refreshing device token..."

RESPONSE=$(curl -s -X POST \
  "$KEYCLOAK_URL/realms/$REALM/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=$CLIENT_ID&username=$USERNAME&password=$PASSWORD&grant_type=password" \
  --max-time 15)

TOKEN=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)

if [ -z "$TOKEN" ] || [ "$TOKEN" = "None" ]; then
  echo "$LOG_PREFIX ERROR: Failed to get token"
  echo "$LOG_PREFIX Response: $(echo $RESPONSE | head -c 200)"
  exit 1
fi

mkdir -p "$(dirname $TOKEN_FILE)"
echo "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

EXPIRY=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('expires_in','?'))" 2>/dev/null)
echo "$LOG_PREFIX Token refreshed (expires in ${EXPIRY}s)"
echo "$LOG_PREFIX Written to: $TOKEN_FILE"
