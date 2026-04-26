#!/usr/bin/env bash
#
# tide-api: optional pull from S3 → local refresh_data() → sync back to the same S3 prefix.
#
# Uses one stable S3 root (no per-run timestamp folder). Default: bucket root
#   s3://parquetgarage/model/data
#   s3://parquetgarage/model/artifacts_75k
#   …
# Optional S3_KEY_PREFIX (or legacy S3_PREFIX) nests under that root, e.g. tide-api/.
#
# Mimics a local run like:
#   cd <repo-root> && source venv/bin/activate
#   python -c "from train_model import refresh_data; refresh_data()"
#
# Env (optional):
#   PROJECT_DIR              — repo root (default: directory of this script)
#   VENV_DIR                 — venv (default: $PROJECT_DIR/venv)
#   ENV_FILE                 — sourced with set -a if present (default: $PROJECT_DIR/.env)
#   S3_BUCKET                — default s3://parquetgarage (stable root = bucket root)
#   S3_KEY_PREFIX / S3_PREFIX — optional subfolder (default empty). Example: tide-api → …/tide-api/model/…
#   S3_DEST_URI              — if set, overrides bucket+prefix (full URI to stable root)
#   S3_PULL_BEFORE_REFRESH   — default 1: aws s3 sync canonical model/data down before Python
#                               (set 0 to skip if local CSVs are always source of truth)
#   S3_PULL_ARTIFACTS        — default 0: if 1, also sync model/artifacts_75k from S3 before run (large)
#   LOG_FILE                 — default $PROJECT_DIR/api_refresh_s3.log
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"
S3_BUCKET="${S3_BUCKET:-s3://parquetgarage}"
# Default: parquetgarage bucket root. Set S3_KEY_PREFIX or S3_PREFIX to nest (e.g. tide-api).
S3_KEY_PREFIX="${S3_KEY_PREFIX:-${S3_PREFIX:-}}"
_s3pfx="${S3_KEY_PREFIX#\/}"
_s3pfx="${_s3pfx%/}"
if [[ -n "${S3_DEST_URI:-}" ]]; then
  DEST="${S3_DEST_URI%/}"
elif [[ -n "$_s3pfx" ]]; then
  DEST="${S3_BUCKET%/}/${_s3pfx}"
else
  DEST="${S3_BUCKET%/}"
fi
S3_PULL_BEFORE_REFRESH="${S3_PULL_BEFORE_REFRESH:-1}"
S3_PULL_ARTIFACTS="${S3_PULL_ARTIFACTS:-0}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/api_refresh_s3.log}"

log() { echo "$*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_DIR"

log "=== refresh_data + S3 (stable prefix): $(date -u +"%Y-%m-%d %H:%M:%S UTC") ==="
log "PROJECT_DIR=$PROJECT_DIR"
log "DEST=$DEST (same path every run; no dated subfolder)"

if ! command -v aws >/dev/null 2>&1; then
  log "ERROR: aws CLI not found on PATH."
  exit 1
fi

log "Checking AWS credentials..."
aws sts get-caller-identity | tee -a "$LOG_FILE"

PY=python3
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
  PY=python
  log "Activated venv: $VENV_DIR"
else
  log "WARNING: venv not found at $VENV_DIR; using python3 on PATH"
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  log "Sourced environment file: $ENV_FILE"
fi

if [[ "$S3_PULL_BEFORE_REFRESH" == "1" || "$S3_PULL_BEFORE_REFRESH" == "true" ]]; then
  mkdir -p "$PROJECT_DIR/model/data"
  log "Pulling ${DEST}/model/data → local (canonical CSVs / parquets for max-date incremental)..."
  aws s3 sync "${DEST}/model/data" "$PROJECT_DIR/model/data" --only-show-errors | tee -a "$LOG_FILE"
fi

if [[ "$S3_PULL_ARTIFACTS" == "1" || "$S3_PULL_ARTIFACTS" == "true" ]]; then
  mkdir -p "$PROJECT_DIR/model/artifacts_75k"
  log "Pulling ${DEST}/model/artifacts_75k → local..."
  aws s3 sync "${DEST}/model/artifacts_75k" "$PROJECT_DIR/model/artifacts_75k" --only-show-errors | tee -a "$LOG_FILE"
fi

log "Running train_model.refresh_data()..."
"$PY" -c "from train_model import refresh_data; refresh_data()" 2>&1 | tee -a "$LOG_FILE"

log "Syncing to S3 (in-place under $DEST)..."
aws s3 sync "$PROJECT_DIR/model/data" "${DEST}/model/data" --only-show-errors | tee -a "$LOG_FILE"
aws s3 sync "$PROJECT_DIR/model/artifacts_75k" "${DEST}/model/artifacts_75k" --only-show-errors | tee -a "$LOG_FILE"

if [[ -d "$PROJECT_DIR/model/archetypes_artifacts" ]]; then
  aws s3 sync "$PROJECT_DIR/model/archetypes_artifacts" "${DEST}/model/archetypes_artifacts" --only-show-errors | tee -a "$LOG_FILE"
fi

if [[ -f "$PROJECT_DIR/marketshare_data.db" ]]; then
  aws s3 cp "$PROJECT_DIR/marketshare_data.db" "${DEST}/marketshare_data.db" --only-show-errors | tee -a "$LOG_FILE"
fi

log "Done. Canonical prefix: $DEST"
log "=== Finished: $(date -u +"%Y-%m-%d %H:%M:%S UTC") ==="
