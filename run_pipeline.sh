#!/usr/bin/env bash
#
# Tide data pipeline: SQLite refresh → model training → S3 upload.
#
# Usage (from repo root on EC2):
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
#
# Optional overrides (export before running):
#   PROJECT_DIR=/home/ubuntu/tide-api     # default: directory containing this script
#   VENV_DIR=/home/ubuntu/tide-api/venv   # default: $PROJECT_DIR/venv
#   ENV_FILE=/home/ubuntu/tide-api/.env   # default: $PROJECT_DIR/.env
#   S3_BUCKET=s3://parquetgarage            # default below
#   S3_PREFIX=tide-pipeline               # default below; dated subfolder is appended
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/pipeline.log}"

S3_BUCKET="${S3_BUCKET:-s3://parquetgarage}"
# ISO-ish UTC timestamp for versioned uploads
S3_RUN_PREFIX="${S3_PREFIX:-tide-pipeline}/$(date -u +%Y%m%dT%H%M%SZ)"

log() {
  echo "$*" | tee -a "$LOG_FILE"
}

log "=== Starting pipeline: $(date -u +"%Y-%m-%d %H:%M:%S UTC") ==="

cd "$PROJECT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  log "ERROR: venv not found at $VENV_DIR"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  log "Loaded environment from $ENV_FILE"
else
  log "WARNING: $ENV_FILE not found; continuing with current environment"
fi

PIPELINE_OK=1

log "Running sqlite_handler.py..."
if python sqlite_handler.py >>"$LOG_FILE" 2>&1; then
  log "SQLite update SUCCESS. Running train_model.py..."
  if python train_model.py >>"$LOG_FILE" 2>&1; then
    log "Model training SUCCESS."
    PIPELINE_OK=0
  else
    log "ERROR: train_model.py failed."
  fi
else
  log "ERROR: sqlite_handler.py failed. Skipping model training."
fi

if [[ "$PIPELINE_OK" -ne 0 ]]; then
  log "=== Pipeline finished with errors: $(date -u +"%Y-%m-%d %H:%M:%S UTC") ==="
  exit 1
fi

log "Uploading artifacts to ${S3_BUCKET}/${S3_RUN_PREFIX}/ ..."
DEST="${S3_BUCKET%/}/${S3_RUN_PREFIX#/}"

# Prefer per-directory sync (reliable; avoids fragile --include globs).
aws s3 sync "$PROJECT_DIR/model/data" \
  "${DEST}/model/data" \
  --only-show-errors >>"$LOG_FILE" 2>&1

aws s3 sync "$PROJECT_DIR/model/artifacts_75k" \
  "${DEST}/model/artifacts_75k" \
  --only-show-errors >>"$LOG_FILE" 2>&1

if [[ -d "$PROJECT_DIR/model/archetypes_artifacts" ]]; then
  aws s3 sync "$PROJECT_DIR/model/archetypes_artifacts" \
    "${DEST}/model/archetypes_artifacts" \
    --only-show-errors >>"$LOG_FILE" 2>&1
fi

if [[ -f "$PROJECT_DIR/marketshare_data.db" ]]; then
  aws s3 cp "$PROJECT_DIR/marketshare_data.db" \
    "${DEST}/marketshare_data.db" \
    --only-show-errors >>"$LOG_FILE" 2>&1
fi

log "S3 upload complete."
log "=== Pipeline complete: $(date -u +"%Y-%m-%d %H:%M:%S UTC") ==="
