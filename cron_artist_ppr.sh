#!/usr/bin/env bash
# NIGHTLY artist PPR: pull latest rate for roster ∪ expected, patch boot.json.
#
# Suggested crontab (06:00 UTC, after ARTIST_PPR_DAILY_V2 usually lands):
#   0 6 * * * /home/ubuntu/tide-data-pipeline/tide-api/cron_artist_ppr.sh >> /home/ubuntu/tide-data-pipeline/tide-api/cron_artist_ppr.log 2>&1
#
set -euo pipefail
API_URL="${API_URL:-http://127.0.0.1:8000}"
POLL_SEC="${POLL_SEC:-15}"
MAX_WAIT="${MAX_WAIT:-900}"
HDR=()
[[ -n "${API_KEY:-}" ]] && HDR=(-H "X-API-Key: ${API_KEY}")
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

if [[ -z "${JOB_ID:-}" ]]; then
  log "POST $API_URL/v1/data/refresh_artist_ppr"
  RESP=$(curl -sS -X POST "${HDR[@]}" "$API_URL/v1/data/refresh_artist_ppr")
  JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("job_id",""))')
  [[ -n "$JOB_ID" ]] || { log "ERROR no job_id: $RESP"; exit 1; }
  log "accepted job_id=$JOB_ID"
else
  log "resume poll job_id=$JOB_ID MAX_WAIT=${MAX_WAIT}s"
fi

START=$(date +%s); LAST_STEP=""
while :; do
  (( $(date +%s) - START > MAX_WAIT )) && { log "ERROR timeout after ${MAX_WAIT}s job=$JOB_ID"; exit 2; }
  JOB_JSON=$(curl -sS "${HDR[@]}" "$API_URL/v1/jobs/$JOB_ID")
  read -r STATUS STEP <<<"$(printf '%s' "$JOB_JSON" | python3 -c 'import sys,json
d=json.load(sys.stdin); s=d.get("steps") or []
last=s[-1] if s else {}
step=last.get("step","") if isinstance(last,dict) else str(last)
print(d.get("status",""), step)')"
  [[ "$STEP" != "$LAST_STEP" && -n "$STEP" ]] && { log "step: $STEP"; LAST_STEP="$STEP"; }
  case "$STATUS" in
    succeeded|completed)
      log "DONE in $(( $(date +%s) - START ))s"
      printf '%s' "$JOB_JSON" | python3 -c 'import sys,json
d=json.load(sys.stdin)
r=d.get("result") or {}
print("PPR artists=%s rows=%s ok=%s" % (r.get("artists"), r.get("rows"), r.get("ok")))
boot=r.get("boot") or {}
print("BOOT patched=%s ppr_count=%s s3=%s" % (boot.get("patched"), boot.get("ppr_count"), boot.get("s3_uri")))
'
      exit 0
      ;;
    failed|error)
      log "ERROR job failed: $JOB_JSON"
      exit 3
      ;;
  esac
  sleep "$POLL_SEC"
done
