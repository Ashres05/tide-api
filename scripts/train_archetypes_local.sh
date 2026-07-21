#!/usr/bin/env bash
# Train parquets + archetype artifacts on a local/dev machine (not EC2 API).
#
# Why CLI instead of POST /v1/data/refresh_model on a small EC2:
#   - No uvicorn memory overhead (OOM risk on 4GB instances)
#   - No 6GB root volume pressure from API + pandas in one process
#   - Job IDs are in-memory; SSH drops don't matter
#
# Prerequisites:
#   - Python venv with requirements.txt installed
#   - secrets/amg_research.env (Snowflake keypair)
#   - ~10 GB free disk, 8+ GB RAM recommended
#   - AWS creds with s3:PutObject to parquetgarage (for push)
#
# Env:
#   TIDE_ARTIFACTS_S3_URI=s3://parquetgarage
#   TIDE_ARTIFACTS_S3_PUSH=1
#   PULL_INPUTS_FROM_S3=1   optional: pull csv baseline before train
#
# After this finishes on your laptop, on EC2:
#   aws s3 sync ... OR restart API (startup pulls archetypes) OR:
#   curl -X POST http://127.0.0.1:8000/v1/data/reload_artifacts -H "X-API-Key: ..."
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

if [[ ! -f secrets/amg_research.env ]]; then
  log "ERROR: missing secrets/amg_research.env (see README.md)"
  exit 1
fi

if [[ "${PULL_INPUTS_FROM_S3:-0}" == "1" ]]; then
  log "Pulling csv + parquets baseline from S3 (optional)"
  "$PY" -c "
from api.s3_pull import sync_artifacts_from_s3_if_configured, SCOPE_CSVS, SCOPE_PARQUETS
sync_artifacts_from_s3_if_configured(scopes={SCOPE_CSVS, SCOPE_PARQUETS})
"
fi

log "Stage 1/3: Snowflake → model/data/*.parquet (full history; slow)"
"$PY" -c "
from train_model import update_parquet_metrics
from snowflake_conn import get_snowflake_connection
with get_snowflake_connection() as sf:
    update_parquet_metrics(sf)
print('parquets written under model/data/')
"

log "Stage 2/3: CSV incremental pull + train_artifacts_main(csv_only=False)"
"$PY" -c "
from train_model import refresh_data
import json
print(json.dumps(refresh_data(csv_only=False), indent=2))
"

if [[ "${SKIP_S3_PUSH:-0}" != "1" ]]; then
  log "Stage 3/3: push parquets + artifacts_75k + archetypes + csvs to S3"
  export TIDE_ARTIFACTS_S3_PUSH="${TIDE_ARTIFACTS_S3_PUSH:-1}"
  "$PY" -c "
from api.s3_pull import (
    sync_artifacts_to_s3_if_configured,
    SCOPE_PARQUETS,
    SCOPE_ARTIFACTS_75K,
    SCOPE_ARCHETYPES,
    SCOPE_CSVS,
)
sync_artifacts_to_s3_if_configured(
    scopes={SCOPE_PARQUETS, SCOPE_ARTIFACTS_75K, SCOPE_ARCHETYPES, SCOPE_CSVS}
)
"
else
  log "SKIP_S3_PUSH=1 — artifacts only on local disk"
fi

log "DONE"
log "Verify: ls -lh model/data/*compressed*.parquet model/archetypes_artifacts/streams/archetype_params.json"
