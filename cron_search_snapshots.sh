#!/usr/bin/env bash
# DAILY search-snapshot delta (no day-2 streams). Upsert recent albums/EPs/singles
# and rebuild local artist/distributor typeahead.
#
# Crontab (07:00 UTC, skip Monday inside this script):
#   0 7 * * * /home/ubuntu/tide-data-pipeline/tide-api/cron_search_snapshots.sh >> /home/ubuntu/tide-data-pipeline/tide-api/cron_search_snapshots.log 2>&1
#
# Monday skip: weekly sqlite job writes the same DB at 15:00 UTC.
set -euo pipefail
ROOT="/home/ubuntu/tide-data-pipeline/tide-api"
cd "$ROOT"
set -a
[ -f .env ] && source .env
set +a

API_URL="${API_URL:-http://127.0.0.1:8000}"
POLL_SEC="${POLL_SEC:-15}"
MAX_WAIT="${MAX_WAIT:-3600}"
LOCK="${ROOT}/.search_snapshots.lock"
HDR=()
[[ -n "${API_KEY:-${TIDE_API_KEY:-}}" ]] && HDR=(-H "X-API-Key: ${API_KEY:-$TIDE_API_KEY}")
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

if [[ "$(date -u +%u)" == "1" ]]; then
  log "skip Monday (weekly sqlite job owns this DB)"
  exit 0
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  log "skip: another search-snapshot job holds $LOCK"
  exit 0
fi

if [[ -z "${JOB_ID:-}" ]]; then
  log "POST $API_URL/v1/data/refresh_search_snapshots"
  RESP=$(curl -sS -X POST "${HDR[@]}" "$API_URL/v1/data/refresh_search_snapshots")
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
print("SEARCH_DELTA min_street=%s albums=%s singles=%s artists=%s distributors=%s" % (
    r.get("min_street_date"), r.get("albums"), r.get("singles"),
    r.get("artists"), r.get("distributors"),
))
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
