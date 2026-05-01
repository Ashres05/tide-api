#!/usr/bin/env bash
# DAILY (02:00 PT): hit /v1/releases/backfill and poll. Endpoint pulls new
# mrelg_ids from Snowflake, inserts them into SQLite, refreshes per-release
# historical metrics, and pushes the SQLite db back to S3 when inserts > 0
# (handled inside model_handler.backfill_releases).
set -euo pipefail
API_URL="${API_URL:-http://127.0.0.1:8000}"
POLL_SEC="${POLL_SEC:-20}"
MAX_WAIT="${MAX_WAIT:-1800}"
HDR=()
[[ -n "${API_KEY:-}" ]] && HDR=(-H "X-API-Key: ${API_KEY}")
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "POST $API_URL/v1/releases/backfill"
RESP=$(curl -sS -X POST "${HDR[@]}" "$API_URL/v1/releases/backfill")
JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("job_id",""))')
[[ -n "$JOB_ID" ]] || { log "ERROR no job_id: $RESP"; exit 1; }
log "accepted job_id=$JOB_ID"

START=$(date +%s); LAST_STEP=""
while :; do
  (( $(date +%s) - START > MAX_WAIT )) && { log "ERROR timeout job=$JOB_ID"; exit 2; }
  JOB_JSON=$(curl -sS "${HDR[@]}" "$API_URL/v1/jobs/$JOB_ID")
  read -r STATUS STEP <<<"$(printf '%s' "$JOB_JSON" | python3 -c 'import sys,json
d=json.load(sys.stdin); s=d.get("steps") or []
print(d.get("status",""), s[-1] if s else "")')"
  [[ "$STEP" != "$LAST_STEP" && -n "$STEP" ]] && { log "step: $STEP"; LAST_STEP="$STEP"; }
  case "$STATUS" in
    completed) log "DONE in $(( $(date +%s) - START ))s"; exit 0 ;;
    failed) log "FAILED: $JOB_JSON"; exit 3 ;;
    running|pending|queued) sleep "$POLL_SEC" ;;
    *) log "unknown status=$STATUS"; sleep "$POLL_SEC" ;;
  esac
done
