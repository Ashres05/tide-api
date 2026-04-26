#!/usr/bin/env bash
#
# Daily job (cron): keep API-backed release data aligned with Snowflake.
#
# What this does vs the two SQL files:
#   S3_DOWNLOAD=0|1            — BEFORE sqlite_handler: pull DB + model/* from S3 (default 0).
#                                Use 1 on EC2 when artifacts are built elsewhere (no refresh_data).
#                                Same paths as upload: $DEST/model/data, artifacts_75k, archetypes, DB.
#
#   query_release_backfill.sql
#     Used by POST /v1/releases/backfill (model_handler.backfill_releases).
#     This script runs that logic once per day so new 75k+ releases get rows
#     in SQLite and metrics can be joined — no need to "update" the .sql file;
#     Snowflake data moves forward; backfill applies the current query text.
#
#   query_release_global_streaming.sql
#     Used at request time by GET /v1/releases/{id}/global_streaming for
#     observed weekly history (Snowflake, per mrelg_id).
#     The decay *model* still needs local files under
#       model/archetypes_artifacts/worldwide_streams/
#     built from model/data/worldwide_streams_compressed.parquet (Snowflake →
#     parquet via train_model.update_parquet_metrics, then archetype train).
#     Optional: set REFRESH_WORLDWIDE_STREAMS=1 below to refresh that parquet
#     and retrain worldwide_streams archetypes before S3 upload (heavier).
#
# Env (optional):
#   VENV_DIR=/home/ubuntu/venv
#   S3_UPLOAD=0|1              — upload model/* + DB after sync (default 1)
#   REFRESH_WORLDWIDE_STREAMS=0|1 — refresh worldwide_streams parquet + archetypes (default 0; slower)
#
#   sqlite_handler incremental (ON by default when tables already have rows):
#     TIDE_SQLITE_FULL_REFRESH=1           — full Snowflake pulls for weekly/ytd/metrics
#     TIDE_MARKETSHARE_LOOKBACK_DAYS=120   — rolling window for weekly+ytd (default 120 days)
#     TIDE_RELEASE_METRICS_LOOKBACK_DAYS=560 — rolling window for release metrics (default 560 days)
#
#   backfill_releases incremental (ON by default when EXPECTED_RELEASES already has rows):
#     TIDE_BACKFILL_FULL=1                 — full Snowflake album scan (cold start / audit)
#     TIDE_BACKFILL_LOOKBACK_DAYS=90       — only look at albums released in last N days (default 90)
#
# Run from EC2 after deploy (adjust paths):
#   chmod +x daily_release_sync.sh
#   crontab -e
#
# Every day at 2:00 AM Pacific, while the instance stays on UTC:
#   CRON_TZ=America/Los_Angeles
#   0 2 * * * /home/ubuntu/tide-data-pipeline/tide-api/daily_release_sync.sh
#
# CRON_TZ makes the five-field schedule use that zone (PST/PDT handled automatically).
# If your cron rejects CRON_TZ, upgrade cron or use a systemd timer with
# OnCalendar= in America/Los_Angeles.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/daily_release_sync.log}"
S3_BUCKET="${S3_BUCKET:-s3://parquetgarage}"
S3_KEY_PREFIX="${S3_KEY_PREFIX:-${S3_PREFIX:-tide-api}}"
S3_UPLOAD="${S3_UPLOAD:-1}"
S3_DOWNLOAD="${S3_DOWNLOAD:-0}"

# 1 = pull latest worldwide_streams parquet from Snowflake and retrain
#     model/archetypes_artifacts/worldwide_streams (then upload with S3_UPLOAD).
REFRESH_WORLDWIDE_STREAMS="${REFRESH_WORLDWIDE_STREAMS:-0}"

_s3pfx="${S3_KEY_PREFIX#\/}"
_s3pfx="${_s3pfx%/}"
if [[ -n "${S3_DEST_URI:-}" ]]; then
  DEST="${S3_DEST_URI%/}"
else
  DEST="${S3_BUCKET%/}/${_s3pfx}"
fi

log() { echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") $*" | tee -a "$LOG_FILE"; }

cd "$PROJECT_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  log "ERROR: venv not found: $VENV_DIR"
  exit 1
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
else
  log "WARNING: missing $ENV_FILE"
fi

log "=== daily_release_sync start ==="

if [[ "$S3_DOWNLOAD" == "1" || "$S3_DOWNLOAD" == "true" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    log "ERROR: aws CLI not found on PATH (required for S3_DOWNLOAD)."
    exit 1
  fi
  log "S3_DOWNLOAD=1: syncing from $DEST into repo (model/* + marketshare_data.db)..."
  mkdir -p "$PROJECT_DIR/model/data" "$PROJECT_DIR/model/artifacts_75k" "$PROJECT_DIR/model/archetypes_artifacts"
  aws s3 sync "${DEST}/model/data" "$PROJECT_DIR/model/data" --only-show-errors | tee -a "$LOG_FILE" || true
  aws s3 sync "${DEST}/model/artifacts_75k" "$PROJECT_DIR/model/artifacts_75k" --only-show-errors | tee -a "$LOG_FILE" || true
  aws s3 sync "${DEST}/model/archetypes_artifacts" "$PROJECT_DIR/model/archetypes_artifacts" --only-show-errors | tee -a "$LOG_FILE" || true
  if aws s3 cp "${DEST}/marketshare_data.db" "$PROJECT_DIR/marketshare_data.db" --only-show-errors >>"$LOG_FILE" 2>&1; then
    :
  else
    log "WARNING: could not download ${DEST}/marketshare_data.db (keep existing local DB or run sqlite_handler only)."
  fi
  log "S3_DOWNLOAD: finished (restart uvicorn so the API reloads forecast artifacts from disk)."
fi

log "Running sqlite_handler.py (Snowflake → SQLite marketshare / metrics tables)..."
python sqlite_handler.py >>"$LOG_FILE" 2>&1

log "Running backfill_releases() (incremental: albums in last TIDE_BACKFILL_LOOKBACK_DAYS days only)..."
# run_sqlite_refresh=False: sqlite_handler already ran above; skip the redundant second pull.
python - <<'PY' >>"$LOG_FILE" 2>&1
import json
from model_handler import backfill_releases

print(json.dumps(backfill_releases(run_sqlite_refresh=False), indent=2))
PY

if [[ "$REFRESH_WORLDWIDE_STREAMS" == "1" || "$REFRESH_WORLDWIDE_STREAMS" == "true" ]]; then
  log "REFRESH_WORLDWIDE_STREAMS=1: Snowflake → worldwide_streams_compressed.parquet + archetype train..."
  python - <<'PY' >>"$LOG_FILE" 2>&1
import argparse
import os
from pathlib import Path

from model.all_data_archetypes_simulator_ae import train as train_archetype_model
from snowflake_conn import get_snowflake_connection
from train_model import update_parquet_metrics

repo = Path.cwd()
data_dir = repo / "model" / "data"
worldwide_env = os.environ.get("TIDE_WORLDWIDE_STREAMS_PARQUET", "").strip()
worldwide_parquet = (
    Path(worldwide_env).expanduser().resolve()
    if worldwide_env
    else (data_dir / "worldwide_streams_compressed.parquet")
)
out_dir = repo / "model" / "archetypes_artifacts" / "worldwide_streams"

with get_snowflake_connection() as sf:
    update_parquet_metrics(sf)

if not worldwide_parquet.is_file():
    raise SystemExit(f"Missing parquet after update: {worldwide_parquet}")

out_dir.mkdir(parents=True, exist_ok=True)
train_archetype_model(
    argparse.Namespace(
        parquet_path=str(worldwide_parquet),
        out_dir=str(out_dir),
        metric="worldwide_streams",
        horizon_weeks=78,
        n_clusters=4,
        random_state=42,
        kmeans_batch_size=2048,
        max_tracks_for_features=None,
        sanity_artist=None,
        sanity_peak_volume=None,
        sanity_peak_week=None,
        sanity_genre=None,
    )
)
print(f"OK: worldwide_streams artifacts -> {out_dir}")
PY
fi

if [[ "$S3_UPLOAD" == "1" || "$S3_UPLOAD" == "true" ]]; then
  if ! command -v aws >/dev/null 2>&1; then
    log "ERROR: aws CLI not found on PATH (required for S3 upload)."
    exit 1
  fi

  log "Uploading refreshed artifacts to $DEST ..."
  if [[ -d "$PROJECT_DIR/model/data" ]]; then
    aws s3 sync "$PROJECT_DIR/model/data" "${DEST}/model/data" --only-show-errors | tee -a "$LOG_FILE"
  fi
  if [[ -d "$PROJECT_DIR/model/artifacts_75k" ]]; then
    aws s3 sync "$PROJECT_DIR/model/artifacts_75k" "${DEST}/model/artifacts_75k" --only-show-errors | tee -a "$LOG_FILE"
  fi
  if [[ -d "$PROJECT_DIR/model/archetypes_artifacts" ]]; then
    aws s3 sync "$PROJECT_DIR/model/archetypes_artifacts" "${DEST}/model/archetypes_artifacts" --only-show-errors | tee -a "$LOG_FILE"
  fi
  if [[ -f "$PROJECT_DIR/marketshare_data.db" ]]; then
    if [[ ! -s "$PROJECT_DIR/marketshare_data.db" ]]; then
      log "ERROR: marketshare_data.db is empty; refusing S3 upload (would clobber good remote DB)."
      exit 1
    fi
    aws s3 cp "$PROJECT_DIR/marketshare_data.db" "${DEST}/marketshare_data.db" --only-show-errors | tee -a "$LOG_FILE"
  fi
fi

log "=== daily_release_sync complete ==="
