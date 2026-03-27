#!/bin/bash
# /opt/frs/token_manager.sh
# Fetches a JWT from Keycloak and writes it to /opt/frs/device_token.txt
# The C++ runner reads this file on every recognition request.
#
# Install cron: crontab -e
#   */25 * * * * /opt/frs/token_manager.sh >> /var/log/frs_token.log 2>&1

KEYCLOAK_URL="http://172.20.100.222:9090"
REALM="attendance"
CLIENT_ID="attendance-frontend"
USERNAME="admin@company.com"
PASSWORD="admin123"
TOKEN_FILE="/opt/frs/device_token.txt"
LOG="[$(date '+%Y-%m-%d %H:%M:%S')] [token_manager]"

echo "$LOG Fetching device token from Keycloak..."

RESPONSE=$(curl -s -X POST \
  "$KEYCLOAK_URL/realms/$REALM/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=$CLIENT_ID&username=$USERNAME&password=$PASSWORD&grant_type=password" \
  --max-time 15 2>/dev/null)

TOKEN=$(echo "$RESPONSE" | python3 -c \
  "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null)

if [ -z "$TOKEN" ] || [ "$TOKEN" = "None" ] || [ ${#TOKEN} -lt 50 ]; then
  echo "$LOG ERROR: Token fetch failed"
  echo "$LOG Response snippet: $(echo $RESPONSE | cut -c1-200)"
  # Don't exit 1 — let cron retry in 25 min; existing token may still work
  exit 0
fi

mkdir -p "$(dirname $TOKEN_FILE)"
echo "$TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"

EXPIRY=$(echo "$RESPONSE" | python3 -c \
  "import sys,json; print(json.load(sys.stdin).get('expires_in','?'))" 2>/dev/null)
echo "$LOG Token refreshed — expires in ${EXPIRY}s, written to $TOKEN_FILE"
