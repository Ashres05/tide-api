#!/usr/bin/env bash
# WEEKLY (Mon 14:00 UTC): hit /v1/data/refresh_weekly and poll until done.
# Endpoint pulls weekly inputs from S3, runs CSV-only refresh+train, runs
# release backfill, and pushes outputs back to S3. Heavy parquets are skipped
# by default — call /v1/data/refresh_model when those need rebuilding.
# Quarterly-share CSVs (bi_sandbox) are best-effort: if Snowflake denies that
# database, core CSVs, training, backfill, and S3 sync still complete.
set -euo pipefail
API_URL="${API_URL:-http://127.0.0.1:8000}"
POLL_SEC="${POLL_SEC:-30}"
MAX_WAIT="${MAX_WAIT:-4800}"
HDR=()
[[ -n "${API_KEY:-}" ]] && HDR=(-H "X-API-Key: ${API_KEY}")
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

log "POST $API_URL/v1/data/refresh_weekly"
RESP=$(curl -sS -X POST "${HDR[@]}" "$API_URL/v1/data/refresh_weekly")
JOB_ID=$(printf '%s' "$RESP" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("job_id",""))')
[[ -n "$JOB_ID" ]] || { log "ERROR no job_id: $RESP"; exit 1; }
log "accepted job_id=$JOB_ID"

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
rd=(d.get("result") or {}).get("stages",{}).get("refresh_data",{}) or {}
for st in rd.get("csv_stages") or []:
    name=st.get("csv","?")
    added=st.get("rows_added",0)
    max_wk=st.get("max_week")
    anchor=st.get("anchor_week")
    extra=[]
    if anchor: extra.append(f"anchor={anchor}")
    if max_wk: extra.append(f"max_week={max_wk}")
    if st.get("skipped"): extra.append("skipped")
    suffix=(", " + ", ".join(extra)) if extra else ""
    print(f"CSV_REFRESH {name}: +{added} rows{suffix}")
errs=rd.get("optional_csv_errors") or []
for e in errs:
    print("WARN optional_csv_skipped:", e.get("csv"), ":", (e.get("error") or "")[:200])' || true
      exit 0
      ;;
    failed) log "FAILED: $JOB_JSON"; exit 3 ;;
    running|pending|queued) sleep "$POLL_SEC" ;;
    *) log "unknown status=$STATUS"; sleep "$POLL_SEC" ;;
  esac
done
