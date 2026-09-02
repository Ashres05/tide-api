#!/bin/bash
set -e
cd /home/ubuntu/tide-data-pipeline/tide-api
# systemd LimitNOFILE=65536; also set here so non-systemd runs match.
ulimit -n 65536 2>/dev/null || true
set -a
[ -f secrets/amg_research.env ] && source secrets/amg_research.env
[ -f .env ] && source .env
set +a
exec /home/ubuntu/venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
