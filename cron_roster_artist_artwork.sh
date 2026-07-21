#!/usr/bin/env bash
# Sync main-artist profile images for streaming roster MRELG IDs.
#
# 1. Reads STREAMING_ROSTER_2026 from local marketshare_data.db
# 2. Resolves each MRELG's main ARTIST_ID through Snowflake
# 3. Joins CURRENT_DEV.DATA.ARTIST_METADATA for PROFILE_IMAGE
# 4. Downloads images -> s3://parquetgarage/artist_art/{LUMINATE_ARTIST_ID}.jpeg
#    (skips artist IDs that already have art in S3)
#
# Suggested crontab (daily at 12:30 UTC, after cron_roster_artwork.sh):
#   30 12 * * * /home/ubuntu/tide-data-pipeline/tide-api/cron_roster_artist_artwork.sh >> /home/ubuntu/tide-data-pipeline/tide-api/cron_roster_artist_artwork.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_PREFIX="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"
log() { echo "$LOG_PREFIX $*"; }

for f in secrets/amg_research.env .env; do
    [[ -f "$f" ]] && set -a && source "$f" && set +a
done

PY="${TIDE_PYTHON:-/home/ubuntu/venv/bin/python3}"

log "Starting roster artist artwork sync"
"$PY" scripts/fetch_roster_artist_artwork.py 2>&1
log "Finished roster artist artwork sync (exit $?)"
