#!/usr/bin/env bash
# Sync album artwork for streaming roster mrelg_ids.
#
# 1. Reads STREAMING_ROSTER_2026 from local marketshare_data.db
# 2. Queries Snowflake (Apple Feed) for artwork URLs
# 3. Downloads images → s3://parquetgarage/album_art/{MRELG_ID}.jpg
#    (skips mrelg_ids that already have art in S3)
#
# Crontab (daily at 05:00 UTC, after daily_release_sync finishes):
#   0 5 * * * /home/ubuntu/tide-data-pipeline/tide-api/cron_roster_artwork.sh >> /home/ubuntu/tide-data-pipeline/tide-api/cron_roster_artwork.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"

log() { echo "$LOG_PREFIX $*"; }

# Load Snowflake creds
for f in secrets/amg_research.env .env; do
    [[ -f "$f" ]] && set -a && source "$f" && set +a
done

PY="${TIDE_PYTHON:-/home/ubuntu/venv/bin/python3}"

log "Starting roster artwork sync"
"$PY" scripts/fetch_roster_artwork.py 2>&1
log "Finished roster artwork sync (exit $?)"
