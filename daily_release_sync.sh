#!/usr/bin/env bash
# DAILY (02:00 PT): backfill releases, then backfill streaming roster.
#
# 1. POST /v1/releases/backfill — pulls new mrelg_ids from Snowflake, inserts
#    into SQLite, refreshes per-release historical metrics, pushes DB to S3.
# 2. POST /v1/revenue/backfill_streaming_roster — incremental roster update
#    (last 30 days by default). Pushes DB to S3 when new rows are inserted.
set -euo pipefail
API_URL="${API_URL:-http://127.0.0.1:8000}"
POLL_SEC="${POLL_SEC:-20}"
MAX_WAIT="${MAX_WAIT:-1800}"
HDR=()
[[ -n "${API_KEY:-}" ]] && HDR=(-H "X-API-Key: ${API_KEY}")
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

poll_job() {
  local JOB_ID="$1" LABEL="$2"
  local START LAST_STEP=""
  START=$(date +%s)
  while :; do
    (( $(date +%s) - START > MAX_WAIT )) && { log "ERROR timeout $LABEL job=$JOB_ID"; return 2; }
    JOB_JSON=$(curl -sS "${HDR[@]}" "$API_URL/v1/jobs/$JOB_ID")
    read -r STATUS STEP <<<"$(printf '%s' "$JOB_JSON" | python3 -c 'import sys,json
d=json.load(sys.stdin); s=d.get("steps") or []
print(d.get("status",""), s[-1] if s else "")')"
    [[ "$STEP" != "$LAST_STEP" && -n "$STEP" ]] && { log "$LABEL step: $STEP"; LAST_STEP="$STEP"; }
    case "$STATUS" in
      succeeded|completed) log "$LABEL DONE in $(( $(date +%s) - START ))s"; return 0 ;;
      failed) log "$LABEL FAILED: $JOB_JSON"; return 3 ;;
      running|pending|queued) sleep "$POLL_SEC" ;;
      *) log "$LABEL unknown status=$STATUS"; sleep "$POLL_SEC" ;;
    esac
  done
}

dispatch_and_poll() {
  local ENDPOINT="$1" LABEL="$2"
  log "POST $API_URL$ENDPOINT"
  RESP=$(curl -sS -X POST "${HDR[@]}" "$API_URL$ENDPOINT")
  JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("job_id",""))')
  [[ -n "$JOB_ID" ]] || { log "ERROR $LABEL no job_id: $RESP"; return 1; }
  log "$LABEL accepted job_id=$JOB_ID"
  poll_job "$JOB_ID" "$LABEL"
}

# --- Stage 1: release backfill ---
dispatch_and_poll "/v1/releases/backfill" "releases"

# --- Stage 2: streaming roster backfill ---
dispatch_and_poll "/v1/revenue/backfill_streaming_roster" "streaming_roster"
