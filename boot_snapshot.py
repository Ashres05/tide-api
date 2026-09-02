"""
Build and export boot.json — one static startup payload for the frontend.

=============================================================================
UI CONTRACT (cold start vs live mutations)
=============================================================================

Cold start (almost instantaneous — one fetch):
  GET /v1/boot.json   OR   s3://parquetgarage/boot.json
  Use:
    streaming.releases[]     — roster board + weekly_streams (SQLite actuals only)
    marketshare.actuals      — same shape as GET /v1/marketshare/actuals
    marketshare.weekly       — same shape as GET /v1/marketshare/weekly (full year)
    releases.items[]         — expected/backfill drops + weekly Actual + Forecast
                              (same shape as GET /v1/releases/{id}/weekly)

Live interactions (unchanged endpoints — do NOT wait on boot rebuild):
  POST   /v1/releases              — insert a drop
  PUT    /v1/releases/{id}         — edit a drop
  DELETE /v1/releases/{id}         — remove a drop
  GET    /v1/releases/{id}         — full editor payload
  GET    /v1/releases/{id}/weekly  — live Actual + Forecast (optional; boot already has it)
  GET    /v1/marketshare/weekly    — re-simulate after insert/edit (cache cleared server-side)
  GET    /v1/revenue/streaming_roster  — live roster if needed (debug; prefer boot)

After insert/edit/delete the FE should:
  1. Patch local state from the mutation response (boot-shaped release item,
     including baked Actual+Forecast weekly when available).
  2. Refetch GET /v1/marketshare/weekly (and actuals if shown) — live, not boot.
  3. Treat boot.json as stale until the next weekly/daily export — do not block UX on it.

Boot snapshot sources:
  streaming.weekly_streams  ← MARKETSHARE_WEEKLY_GLOBAL_STREAMS (actuals only)
  releases.weekly           ← get_release_forecasts (Actual + Forecast album units)
  marketshare.*             ← get_marketshare_actuals / get_marketshare_forecasts
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from sqlite_handler import sqlite_connect

logger = logging.getLogger(__name__)

BOOT_VERSION = 2
BOOT_JSON_NAME = "boot.json"
BOOT_LOCAL_REL = Path("model") / "data" / BOOT_JSON_NAME

# Embedded so FE/devtools can discover the split without separate docs.
UI_CONTRACT: Dict[str, Any] = {
    "cold_start": {
        "endpoint": "GET /v1/boot.json",
        "s3": "s3://parquetgarage/boot.json",
        "use": [
            "streaming.releases (+ weekly_streams actuals)",
            "marketshare.actuals",
            "marketshare.weekly",
            "releases.items (+ weekly Actual + Forecast album units)",
        ],
    },
    "live_mutations": {
        "create": "POST /v1/releases",
        "update": "PUT /v1/releases/{id}",
        "delete": "DELETE /v1/releases/{id}",
        "get": "GET /v1/releases/{id}",
        "note": (
            "Mutations clear the server marketshare cache. After create/update/delete, "
            "patch local releases from the response and refetch GET /v1/marketshare/weekly."
        ),
    },
    "release_weekly": {
        "boot": "releases.items[].weekly (Actual + Forecast, same shape as live)",
        "live": "GET /v1/releases/{id}/weekly",
        "note": (
            "Expected-release album-unit curves are baked at export time. "
            "Live weekly remains available after mutations or if boot is stale."
        ),
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def boot_json_local_path() -> Path:
    return _repo_root() / BOOT_LOCAL_REL


def _database_name() -> str:
    from sqlite_handler import DATABASE_NAME

    return DATABASE_NAME


def _records_from_df(df: pd.DataFrame) -> list:
    """Match model_handler.df_to_json rounding, return Python list of dicts."""
    import model_handler

    raw = model_handler.df_to_json(df)
    return json.loads(raw)


def _num_int(v: Any) -> int:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0
        return int(round(float(v)))
    except (TypeError, ValueError):
        return 0


def _num_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        if key in row.keys():
            return row[key]
    except Exception:
        pass
    return default


def _weekly_rows_from_metrics(release_id: int) -> List[dict]:
    """SQLite Actual-only fallback when get_release_forecasts is empty/unavailable."""
    weekly: List[dict] = []
    sql = (
        "SELECT WEEK_ENDING_DATE, STREAMING_EQUIVALENT, PRODUCT_SALES, SONG_SALE_EQUIVALENT "
        "FROM MARKETSHARE_RELEASE_METRICS WHERE RELEASE_ID = ? "
        "ORDER BY date(WEEK_ENDING_DATE) ASC"
    )
    with sqlite_connect() as conn:
        for week, streams, sales, songs in conn.execute(sql, (int(release_id),)):
            week_s = str(week).split(" ")[0][:10] if week is not None else ""
            se, ps, ss = _num_int(streams), _num_int(sales), _num_int(songs)
            weekly.append(
                {
                    "Week Ending Date": week_s,
                    "data_type": "Actual",
                    "streaming_equivalent": se,
                    "product_sales": ps,
                    "song_sale_equivalent": ss,
                    "total": se + ps + ss,
                }
            )
    return weekly


def _weekly_from_release_forecasts(release_id: int) -> List[dict]:
    """
    Same payload as GET /v1/releases/{id}/weekly — Actual + Forecast album units.

    Falls back to SQLite Actual-only metrics if the forecast path returns empty
    or raises (e.g. missing peak / engine not loaded).
    """
    import model_handler

    try:
        df = model_handler.get_release_forecasts(int(release_id))
    except Exception:
        logger.exception(
            "boot releases: get_release_forecasts failed for id=%s; using Actual-only",
            release_id,
        )
        return _weekly_rows_from_metrics(release_id)

    if df is None or df.empty:
        return _weekly_rows_from_metrics(release_id)

    # Match live GET /v1/releases/{id}/weekly JSON rounding/shape.
    return _records_from_df(df)


def release_boot_item_from_row(
    row: sqlite3.Row,
    *,
    weekly: Optional[List[dict]] = None,
) -> dict:
    """
    One EXPECTED_RELEASES row → boot ``releases.items[]`` shape.

    Enough for cold-start lists/charts; editor may still call GET /v1/releases/{id}.
    """
    rid = _row_get(row, "RELEASE_ID")
    mrelg = str(_row_get(row, "MRELG_ID") or "").strip()
    product_type = str(_row_get(row, "PRODUCT_TYPE") or "").strip()
    cluster_raw = _row_get(row, "CLUSTER")
    cluster: Optional[int]
    try:
        cluster = int(cluster_raw) if cluster_raw is not None else None
    except (TypeError, ValueError):
        cluster = None

    return {
        "id": int(rid) if rid is not None else None,
        "album": str(_row_get(row, "TITLE") or "").strip(),
        "artist": str(_row_get(row, "ARTIST") or "").strip(),
        "label_name": str(_row_get(row, "LABEL_NAME") or "").strip(),
        "release_date": (
            str(_row_get(row, "RELEASE_DATE")).split(" ")[0][:10]
            if _row_get(row, "RELEASE_DATE") is not None
            else ""
        ),
        "mrelg_id": mrelg,
        "product_type": product_type,
        "genre": str(_row_get(row, "GENRE") or "").strip(),
        "scenario": str(_row_get(row, "SCENARIO") or "").strip() or "Base",
        # Persisted street-week AE (chart week of RELEASE_DATE; stub weeks already skipped).
        "fw_vol": _num_float(_row_get(row, "EXPECTED_ALBUM_EQUIVALENT")),
        "fw_streams": _num_float(_row_get(row, "FW_STREAMS")),
        "fw_songs": _num_float(_row_get(row, "FW_SONGS")),
        "fw_sales": _num_float(_row_get(row, "FW_SALES")),
        "fy_vol": _num_float(_row_get(row, "FY_VOL")),
        "avg_historical_w1_product_ratio": _num_float(
            _row_get(row, "AVG_HISTORICAL_W1_PRODUCT_RATIO"), 0.3
        ),
        "product_ratio_coefficient": _num_float(
            _row_get(row, "PRODUCT_RATIO_COEFFICIENT"), 0.3
        ),
        "cluster": cluster,
        "weekly": list(weekly or []),
    }


def release_boot_item_by_id(release_id: int) -> dict:
    """Load one release as a boot item (for create/update API responses)."""
    import model_handler
    from snowflake_conn import load_sql
    from sqlite_handler import ensure_expected_releases_fw_columns

    model_handler._verify_id(release_id)
    query = load_sql(model_handler.RELEASE_GET_QUERY)
    with sqlite_connect() as conn:
        ensure_expected_releases_fw_columns(conn)
        conn.row_factory = sqlite3.Row
        row = conn.execute(query, (int(release_id),)).fetchone()
        if row is None:
            raise ValueError(f"No release found with id = {release_id}.")

    weekly = _weekly_from_release_forecasts(int(release_id))
    return release_boot_item_from_row(row, weekly=weekly)


def build_streaming_section() -> dict:
    """Roster rows + SQLite weekly worldwide streams (no Snowflake / no forecast)."""
    import model_handler

    roster = model_handler.get_streaming_roster_2026()
    mrelg_ids = [
        str(r.get("MRELG_ID") or "").strip()
        for r in roster
        if str(r.get("MRELG_ID") or "").strip()
    ]

    by_mrelg: Dict[str, List[dict]] = {mid: [] for mid in mrelg_ids}
    if mrelg_ids:
        placeholders = ",".join("?" * len(mrelg_ids))
        sql = (
            "SELECT MRELG_ID, WEEK_ENDING_DATE, GLOBAL_STREAMS "
            "FROM MARKETSHARE_WEEKLY_GLOBAL_STREAMS "
            f"WHERE MRELG_ID IN ({placeholders}) "
            "ORDER BY MRELG_ID, date(WEEK_ENDING_DATE) ASC"
        )
        with sqlite_connect() as conn:
            cur = conn.execute(sql, mrelg_ids)
            for mid, week, streams in cur.fetchall():
                key = str(mid or "").strip()
                if key not in by_mrelg:
                    continue
                week_s = str(week).split(" ")[0][:10] if week is not None else ""
                by_mrelg[key].append(
                    {
                        "week_ending_date": week_s,
                        "global_streams": _num_int(streams),
                    }
                )

    releases: List[dict] = []
    for row in roster:
        mid = str(row.get("MRELG_ID") or "").strip()
        if not mid:
            continue
        releases.append(
            {
                "MRELG_ID": mid,
                "PRODUCT_TYPE": row.get("PRODUCT_TYPE"),
                "TITLE": row.get("TITLE"),
                "ARTIST": row.get("ARTIST"),
                "LUMINATE_ARTIST_ID": row.get("LUMINATE_ARTIST_ID"),
                "LABEL_NAME": row.get("LABEL_NAME"),
                "PARENT_GROUP": row.get("PARENT_GROUP"),
                "RELEASE_DATE": (
                    str(row.get("RELEASE_DATE") or "").split(" ")[0][:10]
                    if row.get("RELEASE_DATE") is not None
                    else ""
                ),
                "forecast_route": row.get("forecast_route")
                or model_handler.streaming_forecast_route(row.get("PRODUCT_TYPE")),
                "weekly_streams": by_mrelg.get(mid, []),
            }
        )

    return {"count": len(releases), "releases": releases}


def build_marketshare_section() -> dict:
    """YTD actuals + full-year weekly (actuals stitched + forecast) from existing APIs."""
    import model_handler

    actuals = _records_from_df(model_handler.get_marketshare_actuals())
    weekly = _records_from_df(model_handler.get_marketshare_forecasts())
    return {"actuals": actuals, "weekly": weekly}


def build_releases_section() -> dict:
    """
    EXPECTED_RELEASES (backfill + manual drops) + weekly Actual + Forecast.

    Uses the same path as GET /v1/releases/{id}/weekly so boot curves match live.
    """
    import model_handler

    # Warm the forecast engine once before the per-release loop.
    try:
        model_handler.get_engine()
    except Exception:
        logger.exception(
            "boot releases: forecast engine failed to load; Actual-only fallbacks may apply"
        )

    rows = model_handler._get_all_release_rows()
    items: List[dict] = []
    forecast_ok = 0
    forecast_fallback = 0

    for i, row in enumerate(rows):
        rid = _row_get(row, "RELEASE_ID")
        if rid is None:
            continue
        rid_i = int(rid)
        weekly = _weekly_from_release_forecasts(rid_i)
        has_forecast = any(str(w.get("data_type") or "") == "Forecast" for w in weekly)
        if has_forecast:
            forecast_ok += 1
        else:
            forecast_fallback += 1
        items.append(release_boot_item_from_row(row, weekly=weekly))
        if (i + 1) % 25 == 0:
            logger.info(
                "boot releases: progress %d/%d (with_forecast=%d fallback=%d)",
                i + 1,
                len(rows),
                forecast_ok,
                forecast_fallback,
            )

    logger.info(
        "boot releases: done count=%d with_forecast=%d actual_only_fallback=%d",
        len(items),
        forecast_ok,
        forecast_fallback,
    )
    return {
        "count": len(items),
        "items": items,
        "with_forecast": forecast_ok,
        "actual_only_fallback": forecast_fallback,
    }


def build_boot_payload() -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": BOOT_VERSION,
        "ui": UI_CONTRACT,
        "streaming": build_streaming_section(),
        "marketshare": build_marketshare_section(),
        "releases": build_releases_section(),
    }


def export_boot_json_snapshot() -> dict:
    """
    Build boot.json, write model/data/boot.json, upload to S3 root boot.json.

    Returns a summary dict for job / refresh_weekly stage reporting.
    """
    import gc
    import os
    from api.s3_pull import upload_boot_json

    # Reclaim leaked sqlite FDs from prewarm (`with sqlite3.connect` does not
    # close on 3.14) before we open query files / write boot.json.
    gc.collect()
    try:
        nfd = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        nfd = -1
    logger.info("export_boot_json_snapshot: open_fds=%s after gc", nfd)

    try:
        from sqlite_handler import wal_checkpoint

        wal_checkpoint("TRUNCATE")
    except Exception:
        logger.warning("export_boot_json_snapshot: wal_checkpoint failed", exc_info=True)

    t0 = datetime.now(timezone.utc)
    payload = build_boot_payload()
    body = json.dumps(payload, separators=(",", ":"), default=str)
    body_bytes = body.encode("utf-8")

    local_path = boot_json_local_path()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp then rename so ENOSPC cannot truncate the live file
    # the frontend is already serving.
    tmp_path = local_path.with_name(local_path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(body_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, local_path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    s3_uri = upload_boot_json(body_bytes)

    streaming = payload.get("streaming") or {}
    releases = payload.get("releases") or {}
    ms = payload.get("marketshare") or {}
    summary = {
        "ok": True,
        "path": str(local_path),
        "s3_uri": s3_uri,
        "bytes": len(body_bytes),
        "generated_at": payload.get("generated_at"),
        "streaming_count": streaming.get("count", 0),
        "releases_count": releases.get("count", 0),
        "releases_with_forecast": releases.get("with_forecast", 0),
        "releases_actual_only_fallback": releases.get("actual_only_fallback", 0),
        "marketshare_actuals": len(ms.get("actuals") or []),
        "marketshare_weekly": len(ms.get("weekly") or []),
        "elapsed_sec": round((datetime.now(timezone.utc) - t0).total_seconds(), 2),
    }
    logger.info("export_boot_json_snapshot: %s", summary)
    return summary


def load_boot_json_bytes() -> Optional[bytes]:
    """Return local boot.json bytes if present."""
    path = boot_json_local_path()
    if not path.is_file():
        return None
    return path.read_bytes()
