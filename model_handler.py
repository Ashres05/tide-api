from __future__ import annotations
import contextlib
import json
import math
import numbers
import os
import sqlite3
import logging
import time
import pandas as pd

from datetime import datetime, date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from sqlite_handler import (
    DATABASE_NAME,
    ensure_expected_releases_fw_columns,
    ensure_marketshare_search_summary_columns,
    refresh_marketshare_search_summary,
)
from search_text import normalize_search_text
import marketshare_from_csv
import album_art
from model.marketshare_75k_simulation import DISTRIBUTIONS, NUM_WEEKS
from snowflake_conn import load_sql
from model.forecast_engine_server import ForecastEngine
from snowflake_conn import get_snowflake_connection, Snowflake
from train_model import refresh_data, update_parquet_metrics
from sqlite_handler import update_sqlite_main
from model.worldwide_streams_api import simulate_one_worldwide_streams
from api.s3_pull import (
    sync_artifacts_from_s3_if_configured,
    sync_artifacts_to_s3_if_configured,
    sync_db_from_s3,
    sync_db_to_s3,
    sync_full_inputs_from_s3,
    sync_full_outputs_to_s3,
    sync_parquets_to_s3,
    sync_weekly_inputs_from_s3,
    sync_weekly_outputs_to_s3,
)

# Set up logging.
logger = logging.getLogger(__name__)

# TODO: Upload API to EC2 instance.
# TODO: Make delete_release() function delete all release data from SQLite.

# Default training output: model/train_marketshare_artifacts.py writes here (not repo-root artifacts_75k).
ARTIFACTS_DIR = Path(__file__).resolve().parent / "model" / "artifacts_75k"

# Archetype decay artifact dirs — written by train_model.py via all_data_archetypes_simulator_ae.train().
_ARCHETYPES_BASE = Path(__file__).resolve().parent / "model" / "archetypes_artifacts"
ARCHETYPES_STREAMS_DIR = _ARCHETYPES_BASE / "streams"
ARCHETYPES_SALES_DIR   = _ARCHETYPES_BASE / "sales"
ARCHETYPES_SONGS_DIR   = _ARCHETYPES_BASE / "songs"
ARCHETYPES_WORLDWIDE_STREAMS_DIR   = _ARCHETYPES_BASE / "worldwide_streams"


# SQLite table name for observed per-release metrics (populated by sqlite_handler.py)
MARKETSHARE_RELEASE_METRICS_TABLE = "MARKETSHARE_RELEASE_METRICS"

# Query names for the database.
RELEASE_CREATE_QUERY = "release_create.sql"
RELEASE_UPDATE_QUERY = "release_update.sql"
RELEASE_DELETE_QUERY = "release_delete.sql"
RELEASE_GET_QUERY = "release_get.sql"
RELEASE_GET_ALL_QUERY = "release_get_all.sql"
MARKETSHARE_ACTUALS_QUERY = "select_marketshare_actuals.sql"
MARKETSHARE_WEEKLY_ACTUALS_QUERY = "select_marketshare_weekly_actuals.sql"
MRELG_METADATA_QUERY = "query_mrelg_id.sql"
RELEASE_BACKFILL_QUERY = "query_release_backfill.sql"
GLOBAL_STREAMING_QUERY = "query_release_global_streaming.sql"

_RELEASE_FIELD_KEYS = frozenset(
    {
        "mrelg_id",
        "name",
        "artist",
        "label_name",
        "release_date",
        "genre",
        "scenario",
        "known_vols",
        "fw_vol",
        "fw_streams",
        "fw_songs",
        "fw_sales",
        "fy_vol",
        "avg_historical_w1_product_ratio",
        "product_ratio_coefficient",
        "cluster",
    }
)

_REQUIRED_NONEMPTY_STR = (
    "name",
    "artist",
    "label_name",
    "release_date",
    "genre",
    "scenario",
)

_REAL_NUMERIC_FIELDS = (
    "fw_vol",
    "fw_streams",
    "fw_songs",
    "fw_sales",
    "fy_vol",
    "avg_historical_w1_product_ratio",
    "product_ratio_coefficient",
)

_ALLOWED_SCENARIOS = frozenset[str]({"Bear", "Base", "Bull"})

GLOBAL_FORECAST_ENGINE = None
GLOBAL_WORLDWIDE_ARTIFACTS = None


# ---------------------------------------------------------------------------
# Performance instrumentation
# ---------------------------------------------------------------------------
# Lightweight timing helper. Logs INFO with a stable structured prefix so the
# timings are easy to grep in api_refresh / FastAPI logs:
#
#     PERF endpoint=search_global_streaming phase=db_fetch ms=412.1 rows=403211
#
# Each top-level endpoint creates a span dict, then writes a final aggregated
# line at the end summarising every phase + total wall time. Phase timers also
# log per-phase so we still see partial progress on slow requests.

_PERF_LOG_PREFIX = "PERF"


def _now() -> float:
    return time.perf_counter()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


@contextlib.contextmanager
def _perf_phase(
    span: Dict[str, Any], phase: str, *, endpoint: str, **extra: Any
) -> Iterator[Dict[str, Any]]:
    """
    Time a single phase of an endpoint and append it to ``span``.

    ``extra`` is mutable inside the with-block (the dict is yielded so callers
    can record observed counts), and is logged once on exit alongside the
    phase duration.
    """
    t0 = _now()
    info: Dict[str, Any] = dict(extra)
    try:
        yield info
    finally:
        duration_ms = _ms(t0)
        record = {"phase": phase, "ms": duration_ms, **info}
        span.setdefault("phases", []).append(record)
        kv = " ".join(f"{k}={v}" for k, v in record.items())
        logger.info("%s endpoint=%s %s", _PERF_LOG_PREFIX, endpoint, kv)


def _perf_summary(span: Dict[str, Any], *, endpoint: str, t_start: float, **extra: Any) -> None:
    """Emit the final aggregated PERF log for an endpoint span."""
    total_ms = _ms(t_start)
    span["total_ms"] = total_ms
    span.update(extra)
    parts = [f"total_ms={total_ms}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(
        "%s endpoint=%s phase=__summary__ %s",
        _PERF_LOG_PREFIX,
        endpoint,
        " ".join(parts),
    )


# ---------------------------------------------------------------------------
# Forecast response cache
# ---------------------------------------------------------------------------
# Bounded TTL cache keyed by (mrelg_id, daily-data-cutoff). The Snowflake
# query for global streaming filters by ``CURRENT_DATE() - 2 days`` so the
# answer is stable for a given (mrelg_id, today) pair. We keep entries for
# ``_FORECAST_CACHE_TTL_S`` and bound the cache size to ``_FORECAST_CACHE_MAX``
# so unbounded growth doesn't pin memory in long-running API workers.
_FORECAST_CACHE_TTL_S = 6 * 60 * 60  # 6 hours
_FORECAST_CACHE_MAX = 256


import threading as _threading  # noqa: E402  (kept local to forecast cache)

_FORECAST_CACHE: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}
_FORECAST_CACHE_LOCK = _threading.Lock()


def _forecast_cache_key(mrelg_id: str) -> Tuple[str, str]:
    # Bind to the calendar date so refreshed Snowflake data is picked up the
    # next day even if the worker has not been restarted; the TTL still
    # guards against same-day invalidation if the daily snapshot changes.
    return (mrelg_id, date.today().isoformat())


def _forecast_cache_lookup(mrelg_id: str) -> Optional[pd.DataFrame]:
    key = _forecast_cache_key(mrelg_id)
    with _FORECAST_CACHE_LOCK:
        entry = _FORECAST_CACHE.get(key)
        if entry is None:
            return None
        ts, df = entry
        if (time.time() - ts) > _FORECAST_CACHE_TTL_S:
            _FORECAST_CACHE.pop(key, None)
            return None
    return df


def _forecast_cache_store(mrelg_id: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    key = _forecast_cache_key(mrelg_id)
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE[key] = (time.time(), df.copy())
        # Cheap LRU-ish eviction: drop the oldest entries beyond the cap.
        if len(_FORECAST_CACHE) > _FORECAST_CACHE_MAX:
            evict = sorted(_FORECAST_CACHE.items(), key=lambda kv: kv[1][0])
            for k, _ in evict[: len(_FORECAST_CACHE) - _FORECAST_CACHE_MAX]:
                _FORECAST_CACHE.pop(k, None)


def forecast_cache_clear() -> None:
    """Drop all cached forecast responses (used after a data refresh)."""
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE.clear()


def _cap_weekly_series(values: List[float] | None, max_weeks: int) -> List[float]:
    """Keep only the first ``max_weeks`` points (chronological SQLite order)."""
    if not values:
        return []
    if len(values) <= max_weeks:
        return [float(x) for x in values]
    return [float(x) for x in values[:max_weeks]]


def _sanitize_release_for_simulation(release_map: dict) -> dict:
    """
    Normalize per-release inputs before ``ForecastEngine.simulate``:
    cap known weekly series to the same horizon as the archetype decay window
    so ``fit_backfill_forecast`` never sees more actuals than ``end_week``.
    """
    for key in ("known_vols", "known_streams", "known_sales", "known_songs"):
        if key in release_map and release_map[key]:
            release_map[key] = _cap_weekly_series(release_map[key], NUM_WEEKS)
    # Keep known_week_dates aligned with known_vols after the cap so the
    # frontend can still identify the in-progress week by date.
    dates = release_map.get("known_week_dates")
    if isinstance(dates, list) and dates:
        if len(dates) > NUM_WEEKS:
            release_map["known_week_dates"] = dates[:NUM_WEEKS]
    return release_map


def _sanitize_forecast_engine_artifacts(engine: ForecastEngine) -> None:
    """
    In-memory cleanup of parquet-backed frames after load. Keeps files on disk
    unchanged while fixing NaNs / dtypes that would break ``run_archetype_scenario``.
    """
    df = engine.df_full
    if df is None or getattr(df, "empty", True):
        return

    wk_col = "Week Ending Date"
    if wk_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[wk_col]):
        df[wk_col] = pd.to_datetime(df[wk_col], errors="coerce")

    if "Predicted_Baseline_Share" in df.columns:
        if "Owner" in df.columns:
            bad = df["Predicted_Baseline_Share"].isna()
            if bad.any():
                logger.warning(
                    "df_full: repairing %d NaN Predicted_Baseline_Share cells (per-Owner ffill/bfill)",
                    int(bad.sum()),
                )
                df["Predicted_Baseline_Share"] = (
                    df.groupby("Owner", sort=False)["Predicted_Baseline_Share"]
                    .transform(lambda s: s.ffill().bfill())
                )
        df["Predicted_Baseline_Share"] = pd.to_numeric(
            df["Predicted_Baseline_Share"], errors="coerce"
        ).fillna(0.0)

    if "Final_Unified_Share" in df.columns and "Predicted_Baseline_Share" in df.columns:
        mask = df["Final_Unified_Share"].isna()
        if mask.any():
            df.loc[mask, "Final_Unified_Share"] = df.loc[mask, "Predicted_Baseline_Share"]

    for col in ("big_release_flag", "Incremental_Share"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    act = engine.actuals_2026
    if act is None or getattr(act, "empty", True):
        return
    if wk_col in act.columns and not pd.api.types.is_datetime64_any_dtype(act[wk_col]):
        act[wk_col] = pd.to_datetime(act[wk_col], errors="coerce")
    if "AE_Share" in act.columns:
        act["AE_Share"] = pd.to_numeric(act["AE_Share"], errors="coerce").fillna(0.0)


def get_engine():
    """Instantiates the engine once, and returns it for all future calls."""
    global GLOBAL_FORECAST_ENGINE
    if GLOBAL_FORECAST_ENGINE is None:
        GLOBAL_FORECAST_ENGINE = ForecastEngine(
            artifacts_dir=ARTIFACTS_DIR,
            streams_dir=ARCHETYPES_STREAMS_DIR,
            sales_dir=ARCHETYPES_SALES_DIR,
            songs_dir=ARCHETYPES_SONGS_DIR,
        )
        _sanitize_forecast_engine_artifacts(GLOBAL_FORECAST_ENGINE)
    return GLOBAL_FORECAST_ENGINE


def get_worldwide_artifacts():
    """Loads worldwide-streams archetype artifacts once and returns them."""
    global GLOBAL_WORLDWIDE_ARTIFACTS
    if GLOBAL_WORLDWIDE_ARTIFACTS is None:
        from model.all_data_archetypes_simulator_ae import load_artifacts
        if not ARCHETYPES_WORLDWIDE_STREAMS_DIR.exists():
            raise FileNotFoundError(
                f"Worldwide streams artifacts not found at {ARCHETYPES_WORLDWIDE_STREAMS_DIR}. "
                "Run training with --metric worldwide_streams first."
            )
        GLOBAL_WORLDWIDE_ARTIFACTS = load_artifacts(str(ARCHETYPES_WORLDWIDE_STREAMS_DIR))
    return GLOBAL_WORLDWIDE_ARTIFACTS


def create_release(
    *,
    mrelg_id: str | None = None,
    name: str, 
    artist: str, 
    label_name: str, 
    release_date: str, 
    genre: str, 
    scenario: str, 
    fw_vol: float, 
    fw_streams: float = 0.0,
    fw_songs: float = 0.0,
    fw_sales: float = 0.0,
    fy_vol: float = 0.0, 
    known_vols: list[float] | None = None,
    avg_historical_w1_product_ratio: float = 0.3, 
    product_ratio_coefficient: float = 0.3,
    cluster: int = 0,
    **_kwargs,
) -> int:
    """
    Creates a new release in the database.
    Returns the release ID.
    """
    if known_vols is None:
        known_vols = []
    _verify_release_fields(locals())

    params = (
        mrelg_id,
        name,
        artist,
        label_name,
        release_date,
        genre,
        fw_vol,
        fw_streams,
        fw_songs,
        fw_sales,
        scenario,
        fy_vol,
        avg_historical_w1_product_ratio,
        product_ratio_coefficient,
        cluster,
    )
    try:
        query = load_sql(RELEASE_CREATE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            cursor = conn.cursor()
            id = cursor.execute(query, params).fetchone()[0]
            conn.commit()
        return id
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error creating release: {e}") from e


def _create_backfilled_release(
    *,
    mrelg_id: str,
    label_name: str,
    scenario: str = "Base",
    fw_vol: float = 100000.0,  # Default; irrelevant once known_vols are loaded.
    _sf: "Snowflake | None" = None,
) -> int:
    """
    Creates a new release in the database from a mrelg_id.
    Returns the release ID.

    Pass _sf to reuse an existing Snowflake connection (avoids one connect per release).
    """
    import contextlib

    @contextlib.contextmanager
    def _maybe_conn():
        if _sf is not None:
            yield _sf
        else:
            with get_snowflake_connection() as sf2:
                yield sf2

    with _maybe_conn() as sf:
        mrelg_metadata = _verify_mrelg_id(mrelg_id, sf)

    name = mrelg_metadata["TITLE"].iloc[0]
    artist = mrelg_metadata["DISPLAY_ARTIST"].iloc[0]
    release_date = _validate_date(mrelg_metadata["RELEASE_DATE"].iloc[0])
    genre = mrelg_metadata["GENRE"].iloc[0]

    # Temporary adjustment to genres to ensure they are in the distribution.
    # TODO: Remove this once the right genres are in query_mrelg_id.sql
    genre_aliases = {
        "Alt. Rock": "Rock",
        "R&B": "R&B/Hip-Hop",
    }
    genre = genre_aliases.get(genre, genre)
    if genre not in DISTRIBUTIONS["Genre"]:
        genre = "Pop"

    metadata_cols = [name, artist, release_date, genre]

    for col in metadata_cols:
        if col is None:
            raise ValueError(
                f"MRELG ID {mrelg_id} has no {col} in the metadata. Try create_release() instead."
            )

    return create_release(
        mrelg_id=mrelg_id,
        name=name,
        artist=artist,
        label_name=label_name,
        release_date=release_date,
        genre=genre,
        scenario=scenario,
        fw_vol=fw_vol,
        fy_vol=0.0,
        avg_historical_w1_product_ratio=0.3,
        product_ratio_coefficient=0.3,
        cluster=0,
    )


def backfill_releases(
    *,
    run_sqlite_refresh: bool = True,
) -> dict:
    """
    Backfills releases from Snowflake into the local SQLite database.

    Returns a summary dict:
        {"inserted": int, "skipped": int, "errors": [...]}

    Incremental mode (default when EXPECTED_RELEASES already has rows):
      Only queries Snowflake for albums whose release_date falls within the last
      TIDE_BACKFILL_LOOKBACK_DAYS (default 90 days). Albums older than that are
      already in SQLite.  Set TIDE_BACKFILL_FULL=1 to force a complete scan.

    run_sqlite_refresh:
      When True (default) and new releases were inserted, update_sqlite_main() is
      called to pull their metrics into SQLite.  Pass False from the daily script
      (sqlite_handler already ran immediately before) to skip the second refresh.
    """
    import os as _os

    # Backfill only mutates marketshare_data.db; pull only that scope from S3.
    sync_db_from_s3()

    existing_mrelg_ids = {
        (row["MRELG_ID"] or "").strip()
        for row in _get_all_release_rows()
        if "MRELG_ID" in row.keys() and row["MRELG_ID"]
    }

    full_refresh = _os.environ.get("TIDE_BACKFILL_FULL", "").strip().lower() in (
        "1", "true", "yes"
    )
    lookback_days = int(_os.environ.get("TIDE_BACKFILL_LOOKBACK_DAYS", "90"))

    if existing_mrelg_ids and not full_refresh:
        date_filter = (
            f"AND mrelg.release_date >= DATEADD(day, -{int(lookback_days)}, CURRENT_DATE())"
        )
        logger.info(
            "backfill_releases: incremental — querying albums released in last %d days "
            "(TIDE_BACKFILL_FULL=1 for full scan)",
            lookback_days,
        )
    else:
        date_filter = ""
        reason = "TIDE_BACKFILL_FULL=1" if full_refresh else "EXPECTED_RELEASES is empty"
        logger.info("backfill_releases: full Snowflake scan (%s)", reason)

    query = load_sql(RELEASE_BACKFILL_QUERY).replace("{RELEASE_DATE_FILTER}", date_filter)

    logger.info("backfill_releases: running Snowflake query_release_backfill...")
    # Single Snowflake connection reused for the candidate list AND per-release metadata.
    with get_snowflake_connection() as sf:
        df = sf.query(query)
        logger.info("backfill_releases: Snowflake returned %d candidate rows", len(df))

        if df.empty:
            return {"inserted": 0, "skipped": 0, "errors": []}

        pending = [
            (mid, lbl)
            for mid, lbl in df.itertuples(index=False, name=None)
            if (mid or "").strip() not in existing_mrelg_ids
        ]
        skipped = len(df) - len(pending)
        logger.info(
            "backfill_releases: %d new / %d already in SQLite",
            len(pending),
            skipped,
        )

        inserted = 0
        errors: list[dict] = []

        for mrelg_id, label_name in pending:
            try:
                logger.info(
                    "backfill_releases: inserting mrelg_id=%s label=%s",
                    mrelg_id,
                    label_name,
                )
                _create_backfilled_release(mrelg_id=mrelg_id, label_name=label_name, _sf=sf)
                inserted += 1
                existing_mrelg_ids.add(mrelg_id)
            except Exception as e:
                errors.append({"mrelg_id": mrelg_id, "error": str(e)})

    # Always refresh per-release weekly metrics from Snowflake, even when 0
    # new releases were inserted. The previous `inserted > 0` gate caused a
    # silent staleness bug: when refresh_weekly's daily/weekly cadence found
    # no new mrelg_ids, MARKETSHARE_RELEASE_METRICS / MARKETSHARE_WEEKLY /
    # MARKETSHARE_YTD all stayed pinned at whatever date they were when
    # someone last triggered an insert. The /v1/releases/{id}/weekly endpoint
    # then returned multi-week-stale "AE YTD" sums (e.g. OCTANE showing 837K
    # instead of 894K because the 2026-04-23 row was missing).
    #
    # update_sqlite_main is idempotent and incremental — its own internal
    # max-date watermark keeps the Snowflake roundtrip narrow even when no
    # new releases were inserted, so the cost of always running it is small
    # (~30s typical) compared to the staleness it prevents.
    if run_sqlite_refresh:
        logger.info(
            "backfill_releases: running update_sqlite_main() (inserted=%d, refresh per-release metrics)",
            inserted,
        )
        try:
            from sqlite_handler import update_sqlite_main
            update_sqlite_main()
        except Exception as e:
            errors.append(
                {
                    "stage": "update_sqlite_main",
                    "error": str(e)
                    + "\nWARNING: weekly metric data may be stale.",
                }
            )
    else:
        logger.info(
            "backfill_releases: skipping update_sqlite_main (run_sqlite_refresh=False)"
        )

    # Push DB to S3 whenever we mutated SQLite (either by insert OR by
    # update_sqlite_main refreshing the per-release metrics tables).
    if inserted > 0 or run_sqlite_refresh:
        sync_db_to_s3()

    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def update_release(
    *,
    id: int,
    mrelg_id: str,
    name: str,
    artist: str, 
    label_name: str, 
    release_date: str, 
    genre: str, 
    scenario: str, 
    fw_vol: float = 0.0, 
    fw_streams: float = 0.0,
    fw_songs: float = 0.0,
    fw_sales: float = 0.0,
    fy_vol: float = 0.0, 
    known_vols: list[float] | None = None,
    avg_historical_w1_product_ratio: float = 0.3, 
    product_ratio_coefficient: float = 0.3,
    cluster: int = 0,
    **_kwargs,
) -> None:
    """Updates a release in the database."""

    _verify_id(id)
    if known_vols is None:
        known_vols = []
    _verify_release_fields(locals())

    params = (
        mrelg_id,
        name,
        artist,
        label_name,
        release_date,
        genre,
        fw_vol,
        fw_streams,
        fw_songs,
        fw_sales,
        scenario,
        fy_vol,
        avg_historical_w1_product_ratio,
        product_ratio_coefficient,
        cluster,
        id,
    )
    try:
        query = load_sql(RELEASE_UPDATE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error updating release: {e}") from e


def delete_release(id: int) -> None:
    """
    Deletes a release from the database.
    """
    _verify_id(id)
    try:
        query = load_sql(RELEASE_DELETE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            conn.commit()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error deleting release: {e}")


def get_release(id: int) -> dict:
    """
    Loads one release by primary key and returns a dict shaped for ForecastEngine.simulate().
    """
    _verify_id(id)
    query = load_sql(RELEASE_GET_QUERY)
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"No release found with id = {id}.")
            return _sqlite_row_to_release_map(row)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error getting release: {e}") from e


def get_all_releases() -> List[dict]:
    """
    Gets all releases from the database.
    Returns a list of dictionaries of the release data.
    """
    try:
        rows = [{k: row[k] for k in row.keys()} for row in _get_all_release_rows()]
        out: List[Dict[str, Any]] = []
        for r in rows:
            rid = r.get("RELEASE_ID")
            if rid is None:
                continue
            out.append(
                {
                    "id": int(rid),
                    "album": (r.get("TITLE") or "").strip(),
                    "artist": (r.get("ARTIST") or "").strip(),
                }
            )
        return out
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error getting all releases: {e}")


def get_all_releases_series_json() -> str:
    """
    Returns a JSON string of the compact releases series from `get_all_releases()`.

    Shape: [{"id": <int>, "album": <str>, "artist": <str>}, ...]
    """
    return json.dumps(get_all_releases())


# ---------------------------------------------------------------------------
# Search: artist+title -> ranked MRELG candidates
#
# Powered by the local MARKETSHARE_SEARCH_SUMMARY table (rebuilt every
# refresh_data call). Ranks rows with a weighted blend of fuzzy text match
# and log-scaled daily streams so popular releases bubble up *without*
# drowning out close text matches on smaller releases.
# ---------------------------------------------------------------------------

_SEARCH_TEXT_WEIGHT = 0.75
_SEARCH_STREAM_WEIGHT = 0.25
_SEARCH_DEFAULT_LIMIT = 20
_SEARCH_TEXT_FLOOR = 0.30  # rows with text similarity below this are dropped

# Two-stage retrieval tuning: SQL prefilter shrinks the candidate pool that
# the Python fuzzy scorer iterates over. Tokens shorter than MIN_TOKEN_LEN
# generate too many false positives in LIKE scans, so we drop them. The
# candidate pool is capped by CANDIDATE_LIMIT (taking the most-streamed
# matches first) so even worst-case queries stay below ~10k Python iterations.
_SEARCH_MIN_TOKEN_LEN = 2
_SEARCH_CANDIDATE_LIMIT = 8000
_SEARCH_FALLBACK_TOPK = 2000  # used when no usable tokens exist (e.g. all 1-char)


def _normalize_search_text(value: Any) -> str:
    """Backwards-compatible alias for the shared normalizer."""
    return normalize_search_text(value)


def _text_similarity(a: str, b: str) -> float:
    """
    Hybrid text similarity in [0, 1]:
    - SequenceMatcher.ratio for overall ordering / typo tolerance.
    - Token overlap (Jaccard) so partial matches like "untitled unmastered"
      vs "untitled unmastered." score high even when punctuation/order vary.
    """
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if a_tokens and b_tokens:
        jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    else:
        jaccard = 0.0

    return max(seq, jaccard)


def _search_tokens(*texts: str) -> List[str]:
    """Tokenize already-normalized search inputs and drop noise."""
    tokens: List[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for tok in text.split():
            if len(tok) < _SEARCH_MIN_TOKEN_LEN:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            tokens.append(tok)
    return tokens


def _fetch_search_candidates(
    *,
    artist_norm: str,
    title_norm: str,
    span: Dict[str, Any],
) -> List[sqlite3.Row]:
    """
    Two-stage retrieval: SQL prefilter on indexed normalized text columns to
    shrink the pool we feed to the Python fuzzy scorer. Falls back to the
    most-streamed releases when the query has no usable tokens (e.g. all
    1-char tokens) so we still return *something* and the result quality
    matches the legacy behaviour for those edge cases.

    Always selects ARTIST_SEARCH/TITLE_SEARCH so the caller can avoid
    re-normalizing every row at request time. Older databases that predate
    the columns are detected once and a graceful fallback path is used.
    """
    tokens = _search_tokens(artist_norm, title_norm)

    fetch_sql = (
        "SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, GENRE, "
        "DAILY_GLOBAL_STREAMS, ARTIST_SEARCH, TITLE_SEARCH "
        "FROM MARKETSHARE_SEARCH_SUMMARY"
    )

    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            normalized_columns_present = _has_normalized_search_columns(cur)

            if not normalized_columns_present:
                # Legacy schema: fall back to the original full-table fetch so
                # the function still works after a deploy that hasn't run the
                # refresh job yet.
                with _perf_phase(
                    span,
                    "db_fetch",
                    endpoint="search_global_streaming",
                    mode="full_scan_legacy",
                ) as info:
                    cur.execute(
                        "SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, "
                        "GENRE, DAILY_GLOBAL_STREAMS FROM MARKETSHARE_SEARCH_SUMMARY"
                    )
                    rows = cur.fetchall()
                    info["rows"] = len(rows)
                return rows

            if tokens:
                like_clauses: List[str] = []
                params: List[Any] = []
                for tok in tokens:
                    like_clauses.append("ARTIST_SEARCH LIKE ?")
                    params.append(f"%{tok}%")
                    like_clauses.append("TITLE_SEARCH LIKE ?")
                    params.append(f"%{tok}%")

                sql = (
                    f"{fetch_sql} WHERE ({' OR '.join(like_clauses)}) "
                    "ORDER BY DAILY_GLOBAL_STREAMS DESC LIMIT ?"
                )
                params.append(_SEARCH_CANDIDATE_LIMIT)
                with _perf_phase(
                    span,
                    "db_fetch",
                    endpoint="search_global_streaming",
                    mode="prefilter",
                ) as info:
                    info["tokens"] = len(tokens)
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    info["rows"] = len(rows)
                return rows

            # No usable tokens: return top-K by streams so we don't fall back
            # to a full scan but the Python scorer still has *something* to
            # rank. This matches what the user typically wants (popular
            # releases) when the query is too short to filter on.
            with _perf_phase(
                span,
                "db_fetch",
                endpoint="search_global_streaming",
                mode="topk_no_tokens",
            ) as info:
                cur.execute(
                    f"{fetch_sql} ORDER BY DAILY_GLOBAL_STREAMS DESC LIMIT ?",
                    (_SEARCH_FALLBACK_TOPK,),
                )
                rows = cur.fetchall()
                info["rows"] = len(rows)
            return rows
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error querying MARKETSHARE_SEARCH_SUMMARY: {e}") from e


_HAS_NORMALIZED_SEARCH_COLUMNS: Optional[bool] = None


def _has_normalized_search_columns(cur: sqlite3.Cursor) -> bool:
    """
    Cache whether the local SQLite schema has the persisted search columns
    AND whether they have been populated. The column check runs once per
    process; the value is only flipped to ``True`` after a refresh has
    written non-NULL values, so the migration window between an online
    schema-only ALTER and the next refresh still falls back to the legacy
    full-scan path (rather than running a LIKE prefilter against all-NULL
    columns and returning empty results).
    """
    global _HAS_NORMALIZED_SEARCH_COLUMNS
    if _HAS_NORMALIZED_SEARCH_COLUMNS is True:
        return True
    cur.execute("PRAGMA table_info(MARKETSHARE_SEARCH_SUMMARY)")
    cols = {row[1] for row in cur.fetchall()}
    if "ARTIST_SEARCH" not in cols or "TITLE_SEARCH" not in cols:
        _HAS_NORMALIZED_SEARCH_COLUMNS = False
        return False
    cur.execute(
        "SELECT 1 FROM MARKETSHARE_SEARCH_SUMMARY "
        "WHERE ARTIST_SEARCH IS NOT NULL OR TITLE_SEARCH IS NOT NULL LIMIT 1"
    )
    populated = cur.fetchone() is not None
    if populated:
        _HAS_NORMALIZED_SEARCH_COLUMNS = True
    return populated


def search_releases_by_artist_title(
    artist: str,
    title: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Search the MARKETSHARE_SEARCH_SUMMARY table for the best-matching
    Luminate release groups given a free-text artist and album title.

    Ranking: combined_score = TEXT_WEIGHT * text_score + STREAM_WEIGHT *
    popularity_score, where text_score is a fuzzy match on artist+title
    (50/50 average) and popularity_score is log1p(daily_global_streams)
    normalized to [0, 1] across the candidate pool. The streaming weight is
    intentionally bounded so popular releases bubble up only when text
    relevance is comparable; smaller releases with stronger text matches
    still surface near the top.

    Returns up to `limit` results sorted by combined_score (desc), each with
    metadata + component scores so the front end can debug / display reasons.
    """
    if not isinstance(artist, str):
        artist = "" if artist is None else str(artist)
    if not isinstance(title, str):
        title = "" if title is None else str(title)
    artist = artist.strip()
    title = title.strip()
    if not artist and not title:
        raise ValueError("At least one of `artist` or `title` is required.")
    if limit is None:
        limit = _SEARCH_DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError) as e:
        raise ValueError("limit must be a positive integer.") from e
    if limit < 1:
        raise ValueError("limit must be a positive integer.")
    limit = min(limit, 100)

    span: Dict[str, Any] = {}
    t_start = _now()

    artist_norm = _normalize_search_text(artist)
    title_norm = _normalize_search_text(title)

    rows = _fetch_search_candidates(
        artist_norm=artist_norm,
        title_norm=title_norm,
        span=span,
    )

    if not rows:
        logger.info("search: candidate set is empty; returning no matches")
        _perf_summary(
            span,
            endpoint="search_global_streaming",
            t_start=t_start,
            results=0,
            candidates=0,
        )
        return []

    candidates: List[Dict[str, Any]] = []
    with _perf_phase(span, "score", endpoint="search_global_streaming") as info:
        info["pool"] = len(rows)
        for row in rows:
            # Refresh writes pre-normalized columns; legacy rows are normalized
            # on the fly so a stale DB still works.
            row_artist_norm = (
                row["ARTIST_SEARCH"] if "ARTIST_SEARCH" in row.keys() and row["ARTIST_SEARCH"]
                else _normalize_search_text(row["ARTIST"])
            )
            row_title_norm = (
                row["TITLE_SEARCH"] if "TITLE_SEARCH" in row.keys() and row["TITLE_SEARCH"]
                else _normalize_search_text(row["TITLE"])
            )

            if artist_norm and title_norm:
                artist_score = _text_similarity(artist_norm, row_artist_norm)
                title_score = _text_similarity(title_norm, row_title_norm)
                text_score = 0.5 * artist_score + 0.5 * title_score
            elif artist_norm:
                artist_score = _text_similarity(artist_norm, row_artist_norm)
                title_score = 0.0
                text_score = artist_score
            else:
                artist_score = 0.0
                title_score = _text_similarity(title_norm, row_title_norm)
                text_score = title_score

            if text_score < _SEARCH_TEXT_FLOOR:
                continue

            try:
                daily_streams = float(row["DAILY_GLOBAL_STREAMS"] or 0)
            except (TypeError, ValueError):
                daily_streams = 0.0

            candidates.append(
                {
                    "mrelg_id": row["MRELG_ID"],
                    "title": row["TITLE"],
                    "artist": row["ARTIST"],
                    "label_name": row["LABEL_NAME"],
                    "release_date": row["RELEASE_DATE"],
                    "genre": row["GENRE"],
                    "daily_global_streams": int(daily_streams),
                    "artist_score": round(float(artist_score), 4),
                    "title_score": round(float(title_score), 4),
                    "text_score": round(float(text_score), 4),
                    "_raw_streams": daily_streams,
                }
            )
        info["matched"] = len(candidates)

    if not candidates:
        _perf_summary(
            span,
            endpoint="search_global_streaming",
            t_start=t_start,
            results=0,
            candidates=len(rows),
        )
        return []

    with _perf_phase(span, "rank", endpoint="search_global_streaming") as info:
        max_log_streams = max(math.log1p(c["_raw_streams"]) for c in candidates)
        if max_log_streams <= 0:
            max_log_streams = 1.0  # avoid divide-by-zero when nothing has streams

        for c in candidates:
            stream_score = math.log1p(c["_raw_streams"]) / max_log_streams
            c["stream_score"] = round(float(stream_score), 4)
            c["combined_score"] = round(
                float(_SEARCH_TEXT_WEIGHT * c["text_score"] + _SEARCH_STREAM_WEIGHT * stream_score),
                4,
            )
            c.pop("_raw_streams", None)

        candidates.sort(
            key=lambda x: (x["combined_score"], x["text_score"], x["daily_global_streams"]),
            reverse=True,
        )
        info["matched"] = len(candidates)

    results = candidates[:limit]
    _perf_summary(
        span,
        endpoint="search_global_streaming",
        t_start=t_start,
        results=len(results),
        candidates=len(rows),
    )
    return results


def search_releases_by_artist_title_json(
    artist: str,
    title: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
) -> str:
    """JSON-serialized form of search_releases_by_artist_title for the API layer."""
    return json.dumps(search_releases_by_artist_title(artist, title, limit=limit))


def get_known_vols(
    release_id: int,
) -> List[float]:
    """
    Pull known weekly actuals (ALBUM_EQUIVALENT) for a release from SQLite, ordered by week end.
    """
    _, vols = get_known_vols_with_dates(release_id)
    return vols


def get_known_vols_with_dates(
    release_id: int,
) -> tuple[List[str], List[float]]:
    """
    Same as get_known_vols, but also returns the parallel WEEK_ENDING_DATE list
    so callers can identify in-progress (partial) weeks. Both lists are aligned
    by index and sorted ascending by week end.
    """
    _verify_id(release_id)

    # TODO: Replace with query_known_vols_sqlite.sql, not a priority.
    sql = (
        f"SELECT WEEK_ENDING_DATE, ALBUM_EQUIVALENT "
        f"FROM {MARKETSHARE_RELEASE_METRICS_TABLE} "
        f"WHERE RELEASE_ID = ? "
        f"ORDER BY date(WEEK_ENDING_DATE) ASC;"
    )

    with sqlite3.connect(DATABASE_NAME) as conn:
        df = pd.read_sql_query(sql, conn, params=(int(release_id),))
    if df.empty:
        return [], []
    vals = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    vals = vals.replace([float("inf"), float("-inf")], pd.NA)
    df = df.assign(_v=vals).dropna(subset=["_v"])
    dates = [str(x) for x in df["WEEK_ENDING_DATE"].to_list()]
    vols = [float(x) for x in df["_v"].to_list()]
    return dates, vols


def get_known_component_vols(
    release_id: int,
) -> Dict[str, List[float]]:
    """
    Pull per-component weekly actuals for a release from SQLite, ordered by week end.
    Returns {"streams": [...], "sales": [...], "songs": [...]}.
    """
    _verify_id(release_id)

    sql = (
        f"SELECT WEEK_ENDING_DATE, "
        f"       STREAMING_EQUIVALENT, PRODUCT_SALES, SONG_SALE_EQUIVALENT "
        f"FROM {MARKETSHARE_RELEASE_METRICS_TABLE} "
        f"WHERE RELEASE_ID = ? "
        f"ORDER BY date(WEEK_ENDING_DATE) ASC;"
    )

    with sqlite3.connect(DATABASE_NAME) as conn:
        df = pd.read_sql_query(sql, conn, params=(int(release_id),))

    result: Dict[str, List[float]] = {"streams": [], "sales": [], "songs": []}
    if df.empty:
        return result

    for col, key in [
        ("STREAMING_EQUIVALENT", "streams"),
        ("PRODUCT_SALES", "sales"),
        ("SONG_SALE_EQUIVALENT", "songs"),
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            vals = vals.replace([float("inf"), float("-inf")], 0.0)
            result[key] = [float(x) for x in vals.to_list()]

    return result


def get_marketshare_actuals() -> pd.DataFrame:
    """
    Year-to-date observed marketshare timeline derived directly from
    Current_Data.csv (the same file the LGBM/Prophet trainer ingests). The
    YEAR column in Current_Data is Luminate's authoritative chart-year, so
    boundary weeks straddling Jan 1 are bucketed correctly without any
    ``startsWith('YYYY')`` heuristics.

    Returns rows tagged Data_Type='Actual' in the unified_ytd schema. Reads
    are O(1) after the first hit (in-memory cache invalidated by
    reload_artifacts() at the end of every refresh_weekly).
    """
    return marketshare_from_csv.ytd_actuals_for_year()


def _weekly_marketshare_observed() -> pd.DataFrame:
    """
    Per-week observed marketshare for the current Luminate chart-year, sourced
    directly from Current_Data.csv. Used to overlay engine forecasts so the
    "actual" portion of the YTD chart matches the same CSV the trainer ingests
    (no SQLite-vs-CSV cadence drift). Shares are percent points.
    """
    return marketshare_from_csv.weekly_actuals_for_year()


def _apply_weekly_actuals_to_unified_ytd(unified_ytd: pd.DataFrame) -> pd.DataFrame:
    """
    Replace weekly Active_Share / Total_Market for weeks present in
    Current_Data.csv so /v1/marketshare/weekly stitches actuals (past) +
    engine forecast (future). The forecast doesn't "restart from week 1"
    because the weeks before the first forecast week are explicit actuals
    drawn from the same CSV the trainer used.

    Recomputes cumulative YTD columns the same way as run_archetype_scenario.
    """
    if unified_ytd.empty:
        return unified_ytd
    obs = _weekly_marketshare_observed()
    if obs.empty:
        return unified_ytd

    df = unified_ytd.copy()
    df["_wk"] = pd.to_datetime(df["Week Ending Date"], errors="coerce")
    obs = obs.copy()
    obs["_wk"] = pd.to_datetime(obs["Week Ending Date"], errors="coerce")
    patch = obs[["Owner", "_wk", "Active_Share", "Total_Market_AE_Volume"]].rename(
        columns={
            "Active_Share": "_sqlite_share",
            "Total_Market_AE_Volume": "_sqlite_tot",
        }
    )
    merged = df.merge(patch, on=["Owner", "_wk"], how="left")
    hit = merged["_sqlite_share"].notna() & merged["_sqlite_tot"].notna()
    merged.loc[hit, "Active_Share"] = merged.loc[hit, "_sqlite_share"]
    merged.loc[hit, "Total_Market_AE_Volume"] = merged.loc[hit, "_sqlite_tot"]
    merged.loc[hit, "Data_Type"] = "Actual"
    merged = merged.drop(columns=["_sqlite_share", "_sqlite_tot"], errors="ignore")

    merged = merged.sort_values(by=["Owner", "_wk"]).reset_index(drop=True)
    merged["Active_Share"] = pd.to_numeric(merged["Active_Share"], errors="coerce").fillna(0.0)
    merged["Total_Market_AE_Volume"] = pd.to_numeric(
        merged["Total_Market_AE_Volume"], errors="coerce"
    ).fillna(0.0)
    merged["Weighted_Numerator"] = merged["Active_Share"] * merged["Total_Market_AE_Volume"]
    merged["Cum_Numerator"] = merged.groupby("Owner", sort=False)["Weighted_Numerator"].cumsum()
    merged["Cum_Denominator"] = merged.groupby("Owner", sort=False)["Total_Market_AE_Volume"].cumsum()
    den = merged["Cum_Denominator"].replace(0, float("nan"))
    merged["Unified_YTD_Share"] = (merged["Cum_Numerator"] / den).round(4)

    eng = get_engine()
    es = float(getattr(eng, "e_score_default", 0.82) or 0.82)
    fc_mask = merged["Data_Type"].astype(str) == "Forecast"
    merged.loc[~fc_mask, "YTD_Share_Upper"] = merged.loc[~fc_mask, "Unified_YTD_Share"]
    merged.loc[~fc_mask, "YTD_Share_Lower"] = merged.loc[~fc_mask, "Unified_YTD_Share"]
    merged.loc[fc_mask, "YTD_Share_Upper"] = merged.loc[fc_mask, "Unified_YTD_Share"] + es
    merged.loc[fc_mask, "YTD_Share_Lower"] = merged.loc[fc_mask, "Unified_YTD_Share"] - es

    wk_str = merged["_wk"].dt.strftime("%Y-%m-%d")
    merged["Week Ending Date"] = wk_str.where(
        merged["_wk"].notna(), merged["Week Ending Date"].astype(str)
    )
    merged = merged.drop(columns=["_wk"], errors="ignore")
    return merged


def get_marketshare_forecasts(week_ending_date: str | None = None) -> pd.DataFrame:
    """
    Takes in a week ending date and returns the marketshare forecasts for that week.
    The week ending date must be in the format YYYY-MM-DD if provided.
    If no week ending date is provided, all marketshare forecasts are returned.
    """
    # Verify the week ending date (if provided).
    if week_ending_date is not None:
        _validate_date(week_ending_date)

    # Get all releases and verify the parquet file.
    releases = [_sqlite_row_to_release_map(row) for row in _get_all_release_rows()]
    df_full = ARTIFACTS_DIR / "df_full.parquet"
    _verify_parquet_file(df_full)

    # Simulate the releases and return the marketshare forecasts for the week ending date.
    forecasts = get_engine().simulate(releases)
    unified_ytd = pd.DataFrame(forecasts["unified_ytd"])
    if unified_ytd.empty:
        return unified_ytd
    unified_ytd = _apply_weekly_actuals_to_unified_ytd(unified_ytd)
    if week_ending_date is None:
        return unified_ytd
    else:
        mask = unified_ytd["Week Ending Date"].astype(str) == week_ending_date.strip()
        return unified_ytd.loc[mask]


def get_release_forecasts(id: int, week_ending_date: str | None = None) -> pd.DataFrame:
    """
    Takes in a release ID and returns the marketshare forecasts for that release as a pandas DataFrame.
    The week ending date must be in the format YYYY-MM-DD if provided.
    If no week ending date is provided, all weekly forecasts are returned.
    """
    # Verify the week ending date (if provided) and ID.
    _verify_id(id)
    if week_ending_date is not None:
        _validate_date(week_ending_date)

    # Get the release and verify the parquet file.
    release = get_release(id)
    df_full = ARTIFACTS_DIR / "df_full.parquet"
    _verify_parquet_file(df_full)

    # Simulate the release and return the forecasts for the week ending date.
    forecasts = get_engine().simulate([release])
    weekly_injections = pd.DataFrame(forecasts["weekly_injections"])
    if weekly_injections.empty:
        return weekly_injections
    if week_ending_date is None:
        return weekly_injections
    else:
        mask = weekly_injections["Week Ending Date"].astype(str) == week_ending_date
        return weekly_injections.loc[mask]


def train_model() -> None:
    """
    Trains the model.

    Also rebuilds the MARKETSHARE_SEARCH_SUMMARY SQLite table so the search
    endpoint always reflects the latest daily Snowflake snapshot. The table
    is fully overwritten (delete + insert) because daily-stream snapshots
    are not additive across runs.
    """
    # Full refresh trains every scope, so pull all canonical inputs from S3
    # (csv + parquets + artifacts_75k + archetypes + db) before training.
    sync_full_inputs_from_s3()
    refresh_data()
    reload_artifacts()
    try:
        refresh_marketshare_search_summary()
    except Exception:
        logger.exception(
            "train_model: search summary refresh failed; search results may be stale"
        )
    forecast_cache_clear()
    # Persist refreshed CSV + parquets + artifacts + db for future incremental runs.
    sync_full_outputs_to_s3()


def refresh_model() -> None:
    """
    Refreshes the model parquets from Snowflake and pushes them to S3.
    """
    with get_snowflake_connection() as sf:
        update_parquet_metrics(sf)
    reload_artifacts()
    # Parquets are the only thing this path writes; push just that scope.
    sync_parquets_to_s3()


_PARQUET_DIR = Path(__file__).resolve().parent / "model" / "data"
# Some environments still carry the legacy worldwide parquet name
# `streams_worldwide_compressed.parquet`. Treat either as satisfying the
# requirement so refresh_weekly does not trigger an unnecessary full rebuild.
_REQUIRED_PARQUET_ALTERNATIVES = (
    (_PARQUET_DIR / "streams_product_songs_ae_compressed.parquet",),
    (
        _PARQUET_DIR / "worldwide_streams_compressed.parquet",
        _PARQUET_DIR / "streams_worldwide_compressed.parquet",
    ),
)


def _resolve_missing_required_parquets() -> list[str]:
    missing: list[str] = []
    for alternatives in _REQUIRED_PARQUET_ALTERNATIVES:
        if not any(p.is_file() for p in alternatives):
            missing.append(str(alternatives[0]))
    return missing


def _resolved_required_parquet_names() -> list[str]:
    out: list[str] = []
    for alternatives in _REQUIRED_PARQUET_ALTERNATIVES:
        for p in alternatives:
            if p.is_file():
                out.append(p.name)
                break
    return out


def refresh_weekly(force_refresh_parquets: bool = False) -> dict:
    """
    Single weekly orchestration designed around the canonical S3 pattern:
      - Pull only weekly inputs (db + csvs + artifacts_75k) to local disk.
      - Train CSV-only (skips heavy parquet KMeans/archetype decay).
      - Push weekly outputs (db + csvs + artifacts_75k) back to S3.

    Heavy steps (parquet rebuild, full backfill) are intentionally separated
    so the weekly path stays fast, low-memory, and reliable. Use the dedicated
    endpoints when those need to run:
      - /v1/data/refresh_model  — rebuild AE + worldwide parquets
      - /v1/data/refresh_data   — full retrain (parquets + archetypes + csvs)
      - /v1/releases/backfill   — full release backfill

    Stage order:
      1. (optional) refresh_parquets — only when force_refresh_parquets=True.
         Pulls the parquets scope from S3 first so the rebuild starts from
         the canonical state; pushes them back when done.
      2. refresh_data(csv_only=True) — pulls 3 CSVs incrementally from
         Snowflake and retrains LGBM / Prophet / spike / df_full from CSVs
         only. Skips AE parquet KMeans/DNA and all archetype decay.
      3. backfill_releases — inserts new mrelg_ids and refreshes per-release
         historical metrics. Skipped when TIDE_WEEKLY_SKIP_BACKFILL=1.

    Returns a per-stage summary. Stage 1 failures are non-fatal (archetype
    training falls back to whatever parquets are already on disk); stages 2
    and 3 re-raise so the job is marked failed.

    Progress: each stage calls api.jobs.set_step() so the caller can
    diagnose which phase is slow by polling GET /v1/jobs/{id}.steps. The
    import is local to avoid a hard dependency on the api package when this
    module is imported outside FastAPI (e.g. by a script). set_step() is a
    no-op when not running under a JobManager.
    """
    from api.jobs import set_step

    summary: Dict[str, Any] = {"stages": {}}

    skip_backfill = os.environ.get("TIDE_WEEKLY_SKIP_BACKFILL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    # Pull only the weekly inputs (db + csvs + artifacts_75k). Skip the heavy
    # parquets/archetypes scopes — those are not needed for CSV-only training.
    set_step("sync_from_s3:start")
    sync_weekly_inputs_from_s3()
    set_step("sync_from_s3:done")

    # Stage 1: parquet rebuild (opt-in only). Pull parquet scope first so the
    # rebuild has the latest canonical state, then push the new ones back.
    if force_refresh_parquets:
        set_step("refresh_parquets:start")
        t0 = _now()
        logger.info("refresh_weekly: force_refresh_parquets=True; rebuilding parquets")
        try:
            sync_artifacts_from_s3_if_configured(scopes={"parquets"})
            with get_snowflake_connection() as sf:
                update_parquet_metrics(sf)
            sync_parquets_to_s3()
            summary["stages"]["refresh_parquets"] = {
                "ok": True, "skipped": False, "elapsed_sec": _elapsed(t0),
            }
        except Exception as e:
            logger.exception("refresh_weekly: refresh_parquets failed")
            summary["stages"]["refresh_parquets"] = {
                "ok": False,
                "skipped": False,
                "error": str(e),
                "elapsed_sec": _elapsed(t0),
            }
    else:
        logger.info(
            "refresh_weekly: skipping parquet refresh (CSV-only weekly path). "
            "Call /refresh_model or pass force_refresh_parquets=True to rebuild."
        )
        summary["stages"]["refresh_parquets"] = {
            "ok": True,
            "skipped": True,
            "reason": "weekly path is CSV-only by design",
        }

    # Stage 2: CSV-only training.
    set_step("refresh_data:start")
    t0 = _now()
    try:
        refresh_data(csv_only=True)
        summary["stages"]["refresh_data"] = {"ok": True, "elapsed_sec": _elapsed(t0)}
    except Exception as e:
        logger.exception("refresh_weekly: refresh_data failed")
        summary["stages"]["refresh_data"] = {
            "ok": False, "error": str(e), "elapsed_sec": _elapsed(t0),
        }
        set_step("reload_artifacts")
        reload_artifacts()
        raise

    # Stage 3: backfill (skippable for fast weekly runs).
    if skip_backfill:
        logger.info(
            "refresh_weekly: skipping backfill_releases (TIDE_WEEKLY_SKIP_BACKFILL=1). "
            "Run POST /v1/releases/backfill separately when needed."
        )
        summary["stages"]["backfill_releases"] = {
            "ok": True,
            "skipped": True,
            "reason": "TIDE_WEEKLY_SKIP_BACKFILL=1",
        }
    else:
        set_step("backfill_releases:start")
        t0 = _now()
        try:
            backfill_result = backfill_releases()
            summary["stages"]["backfill_releases"] = {
                "ok": True, "elapsed_sec": _elapsed(t0), **backfill_result,
            }
        except Exception as e:
            logger.exception("refresh_weekly: backfill_releases failed")
            summary["stages"]["backfill_releases"] = {
                "ok": False, "error": str(e), "elapsed_sec": _elapsed(t0),
            }
            set_step("reload_artifacts")
            reload_artifacts()
            raise

    set_step("reload_artifacts")
    reload_artifacts()
    # Push only what weekly mutates: db + csvs + artifacts_75k.
    set_step("sync_to_s3")
    sync_weekly_outputs_to_s3()
    set_step("done")
    return summary


def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def reload_artifacts() -> None:
    """
    Clear cached engine/artifacts so the next forecast request loads the
    freshly written parquets/pkls from disk. Call after any data refresh.

    Note: this only invalidates in-process caches. If the API is scaled out
    to multiple workers (gunicorn -w N, multiple EC2 instances) each worker
    has its own globals and will need an external reload signal. That's a
    Phase 3 concern once artifacts move to S3.
    """
    global GLOBAL_FORECAST_ENGINE, GLOBAL_WORLDWIDE_ARTIFACTS
    GLOBAL_FORECAST_ENGINE = None
    GLOBAL_WORLDWIDE_ARTIFACTS = None
    forecast_cache_clear()
    marketshare_from_csv.clear_cache()
    album_art.clear_cache()


def df_to_json(
    df: pd.DataFrame
) -> str:
    """
    Returns JSON string from a pandas DataFrame.
    """
    if df is None or df.empty:
        return "[]"

    out = df.copy()

    # Ensure date columns serialize as YYYY-MM-DD.
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")

    numeric_cols = list(out.select_dtypes(include=["number"]).columns)
    share_cols = [c for c in numeric_cols if "share" in str(c).lower() or "percent" in str(c).lower()]
    other_numeric_cols = [c for c in numeric_cols if c not in share_cols]

    # Round share columns to 2 decimal places and other numeric columns to the nearest integer.
    if share_cols:
        out[share_cols] = out[share_cols].round(2)

    if other_numeric_cols:
        rounded = out[other_numeric_cols].round(0)
        out[other_numeric_cols] = rounded.astype("Int64")

    return out.to_json(orient="records")


def _get_all_release_rows() -> List[sqlite3.Row]:
    query = load_sql(RELEASE_GET_ALL_QUERY)
    with sqlite3.connect(DATABASE_NAME) as conn:
        ensure_expected_releases_fw_columns(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        return cur.fetchall()


def _sqlite_row_to_release_map(row: sqlite3.Row) -> dict:
    """Map EXPECTED_RELEASES columns to keys expected by run_archetype_scenario."""
    title = (row["TITLE"] or "").strip() or None
    artist = (row["ARTIST"] or "").strip() or None
    mrelg_id = (row["MRELG_ID"] or "").strip() if "MRELG_ID" in row.keys() else ""
    rid = row["RELEASE_ID"] if "RELEASE_ID" in row.keys() else None

    known_vols: List[float] = []
    known_week_dates: List[str] = []
    component_vols: Dict[str, List[float]] = {"streams": [], "sales": [], "songs": []}
    if mrelg_id and rid is not None:
        try:
            known_week_dates, known_vols = get_known_vols_with_dates(int(rid))
        except Exception:
            known_week_dates, known_vols = [], []
        try:
            component_vols = get_known_component_vols(int(rid))
        except Exception:
            component_vols = {"streams": [], "sales": [], "songs": []}

    rd = row["RELEASE_DATE"]
    date_str = rd if isinstance(rd, str) else str(rd)

    expected_fw_vol = float(row["EXPECTED_ALBUM_EQUIVALENT"] or 0)
    if mrelg_id:
        has_nonzero_known = bool(known_vols) and any(float(x) > 0 for x in known_vols)
        if (not has_nonzero_known) and expected_fw_vol <= 0:
            raise ValueError(
                f"Release {rid} has mrelg_id={mrelg_id!r} but no backfilled metrics yet "
                "Ensure known_vols is populated."
            )
        fw_vol = float(max(known_vols)) if has_nonzero_known else expected_fw_vol
    else:
        fw_vol = expected_fw_vol

    def _sql_float(col: str, default: float = 0.0) -> float:
        if col not in row.keys():
            return default
        v = row[col]
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    release_map: dict = {
        "mrelg_id": mrelg_id or None,
        "name": artist or title or "Unknown",
        "artist": artist or "",
        "title": title or "",
        "label": row["LABEL_NAME"],
        "date": date_str,
        "genre": row["GENRE"],
        "cluster": int(row["CLUSTER"] or 0),
        "fw_vol": fw_vol,
        "fw_streams": _sql_float("FW_STREAMS"),
        "fw_songs": _sql_float("FW_SONGS"),
        "fw_sales": _sql_float("FW_SALES"),
        "scenario": row["SCENARIO"],
        "known_vols": known_vols,
        "known_week_dates": known_week_dates,
        "fy_vol": float(row["FY_VOL"] or 0),
        "avg_historical_w1_product_ratio": float(row["AVG_HISTORICAL_W1_PRODUCT_RATIO"] or 0),
        "product_ratio_coefficient": float(row["PRODUCT_RATIO_COEFFICIENT"] or 0),
    }

    if component_vols["streams"]:
        release_map["known_streams"] = component_vols["streams"]
    if component_vols["sales"]:
        release_map["known_sales"] = component_vols["sales"]
    if component_vols["songs"]:
        release_map["known_songs"] = component_vols["songs"]

    return _sanitize_release_for_simulation(release_map)


def _is_real_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_integral(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _verify_release_fields(inputs: dict) -> None:
    """
    Validate release create/update parameters from a dict (e.g. locals()).
    Raises ValueError if validation fails.
    """
    data = {k: inputs[k] for k in _RELEASE_FIELD_KEYS if k in inputs}
    data.setdefault("known_vols", [])
    for _fw in ("fw_streams", "fw_songs", "fw_sales"):
        data.setdefault(_fw, 0.0)

    # Verify required fields are not empty.
    for field in _REQUIRED_NONEMPTY_STR:
        val = data.get(field)
        if val is None or not isinstance(val, str) or not val.strip():
            raise ValueError(f"{field} is required.")

    # Verify numeric fields are real numbers.
    for field in _REAL_NUMERIC_FIELDS:
        val = data[field]
        if not _is_real_number(val):
            raise ValueError(f"{field} must be a number.")
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"{field} must be a finite number.")

    # Verify cluster is a non-negative integer.
    cluster = data.get("cluster", 0)
    if not _is_integral(cluster) or int(cluster) < 0:
        raise ValueError("cluster must be a non-negative integer.")

    # Verify known volumes is a list of real numbers.
    known_vols = data.get("known_vols", [])
    if known_vols is None:
        known_vols = []
    if not isinstance(known_vols, (list, tuple)):
        raise ValueError("Known volumes must be a list.")
    for i, x in enumerate(known_vols):
        if not _is_real_number(x):
            raise ValueError(f"Value {x} at index {i} is not a number.")
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            raise ValueError(f"Value {x} at index {i} is not a finite number.")

    # Verify total first-week signal when known_vols is empty (combined AE and/or component AE).
    fw_vol = float(data["fw_vol"])
    comp_w1 = float(data["fw_streams"]) + float(data["fw_songs"]) + float(data["fw_sales"])
    if len(known_vols) == 0 and fw_vol <= 0 and comp_w1 <= 0:
        raise ValueError(
            "Expected weekly volume must be positive when known volumes is empty, "
            "unless first-week streams / song / product AE components sum to a positive total."
        )

    # Verify genre is in the distribution.
    genre = data["genre"]
    if genre not in DISTRIBUTIONS["Genre"]:
        raise ValueError(f"Genre {genre!r} not found in distribution: {DISTRIBUTIONS['Genre']}.")

    # Verify label is in the distribution.
    label_name = data["label_name"]
    if label_name not in DISTRIBUTIONS["Label"]:
        raise ValueError(f"Label {label_name!r} not found in distribution: {DISTRIBUTIONS['Label']}.")

    try:
        datetime.strptime(data["release_date"].strip(), "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"release_date {data['release_date']!r} is not a valid YYYY-MM-DD date."
        ) from e

    scenario = data["scenario"].strip()
    if scenario not in _ALLOWED_SCENARIOS:
        raise ValueError(
            f"Scenario {scenario!r} must be one of {sorted(_ALLOWED_SCENARIOS)}."
        )


def _verify_id(id: int | None) -> None:
    """Verifies that the id is a positive integer and is not None."""
    if id is None:
        raise ValueError("ID is required.")
    if not _is_integral(id) or int(id) < 0:
        raise ValueError("ID must be a positive integer.")


def _validate_date(date_value: str | None) -> str:
    """Validates that the date is a valid YYYY-MM-DD date and returns it as a string."""
    try:
        if date_value is None:
            raise ValueError(f"Date is required. Got {date_value!r}.")
        if pd.isna(date_value):
            raise ValueError(f"Date is required. Got {date_value!r}.")
    except TypeError:
        raise TypeError(f"Date {date_value!r} must be a string.")
    if isinstance(date_value, datetime):
        s = date_value.date().isoformat()
    elif isinstance(date_value, date):
        s = date_value.isoformat()
    else:
        s = str(date_value).strip()
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Date {date_value!r} must be in the format YYYY-MM-DD.") from e
    return s


def _verify_parquet_file(parquet_file: Path) -> None:
    """Verifies that the parquet file exists."""
    if not parquet_file.is_file():
        raise FileNotFoundError(
            f"Forecast artifacts not found at {parquet_file}. "
            "Run `python train_model.py` (or `model/train_marketshare_artifacts.py`) to generate them."
        )
    return parquet_file


def _verify_mrelg_id(mrelg_id: str, _sf: Snowflake) -> pd.DataFrame:
    """Verifies that the mrelg_id is a valid mrelg_id."""
    query = load_sql(MRELG_METADATA_QUERY)
    mrelg_metadata = _sf.query(query.format(MRELG_ID=f"'{mrelg_id}'"))
    if mrelg_metadata.empty:
        raise ValueError(f"Invalid mrelg_id: {mrelg_id}")
    return mrelg_metadata


def get_global_streaming_forecast(id: int) -> pd.DataFrame:
    """
    Returns a DataFrame of worldwide streaming forecasts for a single release.

    Historical observed weeks fetched from Snowflake are passed as
    known_worldwide_streams to anchor the archetype decay curve via
    fit_backfill_forecast.  When no history exists yet (future release),
    fw_streams (or fw_vol) is used as the cold-start peak volume.

    Columns: release_id, mrelg_id, artist, title, week, week_ending_date,
             data_type, pred_worldwide_streams, cumulative_worldwide_streams

    data_type is "Actual" for observed weeks and "Forecast" for model-predicted weeks.

    Deprecated: prefer get_global_streaming_forecast_by_mrelg(mrelg_id), which
    is what the new front end uses after the search endpoint resolves an MRELG.
    Kept for backward compatibility while the UI is migrated.
    """
    _verify_id(id)
    release = get_release(id)
    mrelg_id = (release.get("mrelg_id") or "").strip()
    if not mrelg_id:
        raise ValueError(
            f"Release {id} has no mrelg_id; worldwide streaming forecast requires "
            "a Luminate release group ID."
        )
    release_date = _validate_date(release.get("date"))

    fw_peak = float(release.get("fw_streams") or 0.0) or float(release.get("fw_vol") or 0.0)

    df = _build_global_streaming_forecast(
        mrelg_id=mrelg_id,
        release_date=release_date,
        artist=release.get("artist") or release.get("name") or "",
        title=release.get("title") or release.get("name") or "",
        genre=release.get("genre"),
        fw_streams_peak=fw_peak,
    )
    df.insert(0, "release_id", id)
    return df


def get_global_streaming_forecast_by_mrelg(mrelg_id: str) -> pd.DataFrame:
    """
    Returns a DataFrame of worldwide streaming forecasts for a given MRELG ID,
    independent of any local SQLite release record.

    Metadata (artist, title, release_date, genre) is resolved from the local
    MARKETSHARE_SEARCH_SUMMARY table when available and falls back to a direct
    Snowflake lookup so previously unseen Luminate releases can still be
    forecast on demand.

    Output schema mirrors get_global_streaming_forecast (minus release_id):
    mrelg_id, artist, title, week, data_type, week_ending_date,
    pred_worldwide_streams, cumulative_worldwide_streams.
    """
    if not isinstance(mrelg_id, str) or not mrelg_id.strip():
        raise ValueError("mrelg_id is required.")
    mrelg_id = mrelg_id.strip()

    span: Dict[str, Any] = {}
    t_start = _now()

    cached = _forecast_cache_lookup(mrelg_id)
    if cached is not None:
        _perf_summary(
            span,
            endpoint="global_streaming_by_mrelg",
            t_start=t_start,
            mrelg_id=mrelg_id,
            cache="hit",
            rows=len(cached),
        )
        return cached.copy()

    with _perf_phase(span, "metadata_lookup_local", endpoint="global_streaming_by_mrelg") as info:
        local_meta = _resolve_mrelg_metadata_local(mrelg_id)
        info["hit"] = bool(local_meta)

    # Single Snowflake session per request: re-used for the (rare) metadata
    # fallback AND the always-required global streams pull. This eliminates
    # the previous double-connect on the metadata-miss path and removes one
    # auth roundtrip on the hot path even when metadata is in SQLite.
    with get_snowflake_connection() as sf:
        if local_meta is not None:
            metadata = local_meta
        else:
            with _perf_phase(
                span, "metadata_lookup_snowflake", endpoint="global_streaming_by_mrelg"
            ):
                metadata = _resolve_mrelg_metadata_snowflake(mrelg_id, sf)

        release_date = _validate_date(metadata.get("release_date"))
        df = _build_global_streaming_forecast(
            mrelg_id=mrelg_id,
            release_date=release_date,
            artist=metadata.get("artist") or "",
            title=metadata.get("title") or "",
            genre=metadata.get("genre"),
            fw_streams_peak=0.0,
            span=span,
            sf=sf,
        )

    _forecast_cache_store(mrelg_id, df)

    _perf_summary(
        span,
        endpoint="global_streaming_by_mrelg",
        t_start=t_start,
        mrelg_id=mrelg_id,
        cache="miss",
        rows=len(df),
    )
    return df


def get_daily_global_streams_by_mrelg(mrelg_id: str) -> pd.DataFrame:
    """
    Live Revenue board — daily worldwide streams since release for a single
    MRELG release group. Reads from the cached SQLite table
    MARKETSHARE_DAILY_GLOBAL_STREAMS; lazily refreshes from Snowflake when
    the cache is empty or older than DAILY_STREAMS_STALE_DAYS.

    Returns columns: report_date (str YYYY-MM-DD), global_streams (float).

    Intentionally isolated from the model: no callers in the simulator,
    forecast engine, or training pipeline depend on this — a regression
    here only affects the Live Revenue surface.
    """
    from sqlite_handler import get_daily_global_streams_for_mrelg

    if not isinstance(mrelg_id, str) or not mrelg_id.strip():
        raise ValueError("mrelg_id is required.")
    mrelg_id = mrelg_id.strip()

    # Need release_date to bound the Snowflake query when the cache is cold.
    # Try local metadata first; fall back to Snowflake only if no cached row.
    meta = _resolve_mrelg_metadata_local(mrelg_id)
    release_date: Optional[str] = None
    if meta:
        rd = meta.get("release_date")
        release_date = str(rd).split(" ")[0] if rd else None
    if not release_date:
        try:
            with get_snowflake_connection() as sf:
                snowflake_meta = _resolve_mrelg_metadata_snowflake(mrelg_id, sf)
                release_date = (snowflake_meta.get("release_date") or "").split(" ")[0] or None
        except Exception as e:
            logger.warning(
                "get_daily_global_streams_by_mrelg: metadata lookup failed for %s: %s",
                mrelg_id,
                e,
            )

    df = get_daily_global_streams_for_mrelg(mrelg_id, release_date=release_date)
    df = df.rename(columns=str.lower) if df is not None else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["report_date", "global_streams"])
    df["report_date"] = df["report_date"].astype(str)
    df["global_streams"] = pd.to_numeric(df["global_streams"], errors="coerce").fillna(0.0)
    return df[["report_date", "global_streams"]]


def _resolve_mrelg_metadata(mrelg_id: str) -> Dict[str, Any]:
    """
    Look up MRELG metadata. Prefer the local MARKETSHARE_SEARCH_SUMMARY table
    (populated daily) for speed, and fall back to a direct Snowflake query if
    the row is not present locally.

    Kept for backward compatibility with any caller that does not have its
    own Snowflake session; new code paths should call
    ``_resolve_mrelg_metadata_local`` first and pass an existing Snowflake
    connection into ``_resolve_mrelg_metadata_snowflake`` to avoid opening a
    second connection.
    """
    local = _resolve_mrelg_metadata_local(mrelg_id)
    if local is not None:
        return local
    with get_snowflake_connection() as _sf:
        return _resolve_mrelg_metadata_snowflake(mrelg_id, _sf)


def _resolve_mrelg_metadata_local(mrelg_id: str) -> Optional[Dict[str, Any]]:
    """
    Try to resolve metadata from the daily SQLite snapshot. Returns ``None``
    when the row is not present so the caller can decide whether to fall
    back to Snowflake (and reuse an existing session if it has one).
    """
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, GENRE "
                "FROM MARKETSHARE_SEARCH_SUMMARY WHERE MRELG_ID = ? LIMIT 1",
                (mrelg_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "mrelg_id": row["MRELG_ID"],
                "title": row["TITLE"],
                "artist": row["ARTIST"],
                "label_name": row["LABEL_NAME"],
                "release_date": row["RELEASE_DATE"],
                "genre": row["GENRE"],
            }
    except sqlite3.Error as e:
        logger.warning("mrelg metadata: SQLite lookup failed (%s); will fall back to Snowflake", e)
        return None


def _resolve_mrelg_metadata_snowflake(mrelg_id: str, sf: Snowflake) -> Dict[str, Any]:
    """Resolve metadata from Snowflake using an already-open session."""
    df = _verify_mrelg_id(mrelg_id, sf).rename(columns=str.upper)
    row = df.iloc[0]
    return {
        "mrelg_id": str(row.get("MRELG_ID") or mrelg_id),
        "title": (row.get("TITLE") or ""),
        "artist": (row.get("DISPLAY_ARTIST") or row.get("ARTIST") or ""),
        "label_name": None,
        "release_date": str(row.get("RELEASE_DATE") or "").split(" ")[0],
        "genre": row.get("GENRE"),
    }


def _build_global_streaming_forecast(
    mrelg_id: str,
    release_date: str,
    artist: str,
    title: str,
    genre: Any,
    fw_streams_peak: float,
    span: Optional[Dict[str, Any]] = None,
    sf: Optional[Snowflake] = None,
) -> pd.DataFrame:
    """
    Shared backbone for both the legacy release_id-driven and the new MRELG-driven
    global streaming forecast endpoints. Pulls observed weekly streams from
    Snowflake, runs the worldwide-streams archetype simulation, and returns
    the actual-plus-forecast frame.

    ``sf`` lets callers pass an already-open Snowflake session so we don't pay
    the connect/auth cost more than once per request. ``span`` enables phase-
    level timing logs without polluting the production code path with
    bookkeeping when omitted (single-shot scripts).
    """
    @contextlib.contextmanager
    def _phase(name: str, **extra: Any):
        if span is not None:
            with _perf_phase(span, name, endpoint="global_streaming_by_mrelg", **extra) as info:
                yield info
        else:
            yield {}

    @contextlib.contextmanager
    def _sf_session():
        if sf is not None:
            yield sf
        else:
            with get_snowflake_connection() as _sf:
                yield _sf

    with _phase("snowflake_streams_query") as info:
        with _sf_session() as session:
            hist_df = _get_known_vols_global_streaming(mrelg_id, release_date, session)
        info["rows"] = len(hist_df)

    if hist_df.empty:
        raise ValueError(f"No historical observed weeks found for mrelg_id: {mrelg_id}")

    artifacts = get_worldwide_artifacts()
    horizon_weeks = int(artifacts.horizon_weeks)

    known: List[float] = []
    stream_col = next(
        (c for c in hist_df.columns if "stream" in c.lower()),
        hist_df.columns[-1],
    )
    series = pd.to_numeric(hist_df[stream_col], errors="coerce").fillna(0.0)
    known = series.tolist()
    if len(known) > horizon_weeks:
        logger.info(
            "global_streaming: truncating observed weeks for mrelg_id=%s from %d to %d",
            mrelg_id,
            len(known),
            horizon_weeks,
        )
        # fit_backfill_forecast requires len(actuals) <= end_week <= horizon.
        # Keep the earliest weeks (week 1..horizon) for a valid backfill fit.
        known = known[:horizon_weeks]
        hist_df = hist_df.iloc[:horizon_weeks].copy()

    if not any(x > 0 for x in known) and fw_streams_peak <= 0:
        raise ValueError(
            f"mrelg_id {mrelg_id} has no observed worldwide stream history and no "
            "first-week peak available; cannot produce a forecast."
        )

    release_dict: Dict[str, Any] = {
        "artist": artist or "",
        "name": title or "",
        "genre": genre,
        "date": release_date,
        "known_worldwide_streams": known,
        "fw_worldwide_streams": float(fw_streams_peak or 0.0),
    }

    with _phase("simulation") as info:
        result = simulate_one_worldwide_streams(release_dict, artifacts, end_week=horizon_weeks)
        info["weeks"] = horizon_weeks
        info["known"] = len(known)

    n_known = len(known)
    df = pd.DataFrame(result["weekly"])  # week, pred_worldwide_streams, cumulative_worldwide_streams

    if not hist_df.empty:
        date_col = next(c for c in hist_df.columns if "date" in c.lower())
        hist_dates = pd.to_datetime(hist_df[date_col]).reset_index(drop=True)
        last_known_date = hist_dates.iloc[-1]

        def _week_to_date(week: int) -> str:
            idx = week - 1
            if idx < len(hist_dates):
                return hist_dates.iloc[idx].strftime("%Y-%m-%d")
            return (last_known_date + pd.Timedelta(weeks=(week - n_known))).strftime("%Y-%m-%d")
    else:
        rel_dt = pd.to_datetime(release_date)

        def _week_to_date(week: int) -> str:  # type: ignore[misc]
            return (rel_dt + pd.Timedelta(weeks=week)).strftime("%Y-%m-%d")

    df["week_ending_date"] = df["week"].apply(_week_to_date)
    df["data_type"] = df["week"].apply(lambda w: "Actual" if w <= n_known else "Forecast")

    df.insert(0, "mrelg_id", mrelg_id)
    df.insert(1, "artist", artist or "")
    df.insert(2, "title", title or "")
    week_pos = df.columns.get_loc("week")
    for col in ("data_type", "week_ending_date"):
        df.insert(week_pos + 1, col, df.pop(col))

    return df


def _get_known_vols_global_streaming(mrelg_id: str, release_date: str, _sf: Snowflake) -> pd.DataFrame:
    sql = (
        load_sql(GLOBAL_STREAMING_QUERY)
        .replace("{MRELG_ID}", f"'{mrelg_id}'")
        .replace("{RELEASE_DATE}", f"'{release_date}'")
    )
    df = _sf.query(sql)
    if df.empty:
        raise ValueError(
            f"No global streaming data found for mrelg_id: {mrelg_id} "
            f"on/after release_date: {release_date}"
        )
    return df
