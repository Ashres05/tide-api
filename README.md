## Tide Marketshare API

### How to Run:
1. Clone this repo.
2. Create a directory named `secrets/` in the repo root.
3. Place the Snowflake secrets file (`amg_research.env`) inside `secrets/`.
4. (Prod only) export `TIDE_API_KEY=<a long random string>`. If unset, admin
   endpoints are open and a warning is logged on first call.
5. Launch:
   ```bash
   python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
   ```
6. Open the interactive docs: http://127.0.0.1:8000/docs

### Weekly Refresh (cron)

All admin endpoints are now asynchronous: they return `202 Accepted` with a
`job_id` and the work runs in a background thread. This replaces the old
synchronous pattern where the HTTP call blocked for several minutes and was
frequently killed by the browser/proxy before training finished.

The weekly cron should hit a single endpoint:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/data/refresh_weekly \
     -H "X-API-Key: $TIDE_API_KEY"
# -> {"job_id": "...", "name": "refresh_weekly", "status": "pending", "status_url": "/v1/jobs/..."}
```

Poll the job:

```bash
curl -s http://127.0.0.1:8000/v1/jobs/<job_id>
# -> {"id": "...", "status": "running" | "succeeded" | "failed", ...}
```

`refresh_weekly` runs, in order:

1. (optional) parquet refresh — only with `force_refresh_parquets=true`
2. `refresh_data` (CSV-only) — pulls CSVs and retrains LGBM / Prophet / spike /
   `df_full`
3. `backfill_releases` — new Luminate releases into SQLite + metrics
4. (optional) streaming roster backfill — off by default
5. weekly stream prewarm + **`export_boot_json_snapshot`** — publishes the
   **full** `boot.json` (streaming actuals, marketshare, expected releases with
   **Actual + Forecast** album units). Boot export failure fails the job.

Crontab (this host): `0 15 * * 1 …/api_refresh_s3.sh` → Monday 15:00 UTC.

On success the in-process forecast engine cache is invalidated so subsequent
`/v1/marketshare/*` and `/v1/releases/*/weekly` calls see the fresh models.

### Admin endpoints (all `POST`, all gated by `X-API-Key`)

| Endpoint | Purpose |
| --- | --- |
| `/v1/data/refresh_weekly` | Full weekly pipeline (preferred) |
| `/v1/data/refresh_data` | CSV refresh + model retrain only |
| `/v1/data/refresh_model` | Parquet refresh only |
| `/v1/releases/backfill` | Pull new releases into SQLite |

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/jobs/{job_id}` | Status of a specific job |
| `GET /v1/jobs?limit=N` | Recent jobs (default 20, max 200) |

A 409 response on any admin POST means a job with that name is already
running — the cron should treat this as "already in progress" and move on.

### Read endpoints (unchanged, synchronous)

`/v1/releases`, `/v1/releases/{id}`, `/v1/releases/{id}/weekly`,
`/v1/marketshare/actuals`, `/v1/marketshare/weekly`,
`/v1/releases/{id}/global_streaming`.

### Notes

- Admin endpoints previously used `GET`; they are now `POST`. If you had a
  `GET /v1/data/refresh_data` in the cron, switch to `POST`.
- The job manager is in-process. Scaling the API to multiple workers or
  boxes will require an external queue (Redis/Celery) — a Phase 3 concern.
