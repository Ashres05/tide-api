#!/bin/bash
set -e
cd /home/ubuntu/tide-data-pipeline/tide-api
# systemd LimitNOFILE=65536; also set here so non-systemd runs match.
ulimit -n 65536 2>/dev/null || true
# Snowflake secrets (PEM / Meltwater keys) live in secrets/amg_research.env
# and are loaded by snowflake_conn.get_snowflake_connection() via dotenv.
# Do not `source` that file here: multiline PEM and unquoted special chars
# make bash treat key fragments as commands (systemd then gets exit 127).
set -a
[ -f .env ] && source .env
set +a
exec /home/ubuntu/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
