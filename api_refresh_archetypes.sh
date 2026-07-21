#!/usr/bin/env bash
# Rebuild AE + worldwide parquets from Snowflake, then full train (archetypes).
#
# 1. POST /v1/data/refresh_model   — Snowflake → parquets → S3
# 2. POST /v1/data/refresh_data    — CSV pull + train_artifacts (csv_only=False)
#
# Heavy: allow 1–2+ hours total. Run on EC2 with Snowflake creds and enough RAM.
set -euo pipefail
API_URL="${API_URL:-http://127.0.0.1:8000}"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT_MODEL="${MAX_WAIT_MODEL:-7200}"
MAX_WAIT_DATA="${MAX_WAIT_DATA:-14400}"
API_KEY="${API_KEY:-${TIDE_API_KEY:-}}"
HDR=()
[[ -n "$API_KEY" ]] && HDR=(-H "X-API-Key: ${API_KEY}")
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

poll_job() {
  local JOB_ID="$1" LABEL="$2" MAX_WAIT="$3"
  local START LAST_STEP=""
  START=$(date +%s)
  while :; do
    (( $(date +%s) - START > MAX_WAIT )) && {
      log "ERROR timeout ${MAX_WAIT}s $LABEL job=$JOB_ID"
      return 2
    }
    JOB_JSON=$(curl -sS "${HDR[@]}" "$API_URL/v1/jobs/$JOB_ID")
    read -r STATUS STEP <<<"$(printf '%s' "$JOB_JSON" | python3 -c 'import sys,json
d=json.load(sys.stdin); s=d.get("steps") or []
last=s[-1] if s else {}
step=last.get("step","") if isinstance(last,dict) else str(last)
print(d.get("status",""), step)')"
    [[ "$STEP" != "$LAST_STEP" && -n "$STEP" ]] && { log "$LABEL step: $STEP"; LAST_STEP="$STEP"; }
    case "$STATUS" in
      succeeded|completed)
        log "$LABEL DONE in $(( $(date +%s) - START ))s"
        return 0
        ;;
      failed)
        log "$LABEL FAILED: $JOB_JSON"
        return 3
        ;;
      running|pending|queued) sleep "$POLL_SEC" ;;
      *) log "$LABEL unknown status=$STATUS"; sleep "$POLL_SEC" ;;
    esac
  done
}

find_active_job_id() {
  local NAME="$1"
  curl -sS "${HDR[@]}" "$API_URL/v1/jobs?limit=50" | python3 -c 'import sys,json
name=sys.argv[1]
active={"pending","running","queued"}
for j in json.load(sys.stdin):
    if j.get("name")==name and j.get("status") in active:
        print(j.get("id",""))
        break
' "$NAME"
}

dispatch_and_poll() {
  local ENDPOINT="$1" LABEL="$2" MAX_WAIT="$3"
  log "POST $API_URL$ENDPOINT"
  RESP=$(curl -sS -w "\n%{http_code}" -X POST "${HDR[@]}" "$API_URL$ENDPOINT")
  HTTP_CODE=$(printf '%s' "$RESP" | tail -n1)
  BODY=$(printf '%s' "$RESP" | sed '$d')
  JOB_ID=$(printf '%s' "$BODY" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("job_id",""))' 2>/dev/null || true)
  if [[ -z "$JOB_ID" && "$HTTP_CODE" == "409" ]]; then
    JOB_ID=$(find_active_job_id "$LABEL")
    if [[ -n "$JOB_ID" ]]; then
      log "$LABEL already running — polling existing job_id=$JOB_ID"
    fi
  fi
  [[ -n "$JOB_ID" ]] || { log "ERROR $LABEL no job_id (http=$HTTP_CODE): $BODY"; return 1; }
  [[ "$HTTP_CODE" == "409" ]] || log "$LABEL accepted job_id=$JOB_ID"
  poll_job "$JOB_ID" "$LABEL" "$MAX_WAIT"
}

dispatch_and_poll "/v1/data/refresh_model" "refresh_model" "$MAX_WAIT_MODEL"
dispatch_and_poll "/v1/data/refresh_data" "refresh_data" "$MAX_WAIT_DATA"
log "all stages complete"
