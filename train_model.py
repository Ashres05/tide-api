from model.train_marketshare_artifacts import train_artifacts_main as train_model
from snowflake_conn import Snowflake, get_snowflake_connection, load_sql
from pathlib import Path
from typing import Callable
import datetime
import logging
import time
import pandas as pd


def _set_step(step: str) -> None:
    """
    Thin wrapper around api.jobs.set_step that tolerates running outside of a
    FastAPI context (direct CLI, tests). Imported lazily so this module has no
    hard dep on the api package.
    """
    try:
        from api.jobs import set_step
    except Exception:
        return
    set_step(step)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "model" / "data"

CURRENT_DATA_QUERY = "query_model_current_data.sql"
A_LIST_75K_QUERY = "query_model_a_list_75k.sql"
BIG_RELEASE_FLAG_75K_QUERY = "query_model_big_release_flag.sql"

MODEL_PARQUET_METRICS_QUERY = "query_model_parquet_metrics.sql"
MODEL_PARQUET_METRICS_STREAMING_QUERY = "query_model_parquet_metrics_streaming.sql"

# Phase 2 design notes
# --------------------
# The three CSV queries (Current_Data, alist_75k, bigreleaseflag_75k) now accept
# a {MIN_WEEK_END_DATE} placeholder and emit only weeks >= that anchor (with a
# -2 day upper guard so the in-progress week is never persisted). The Python
# layer appends the result onto the existing CSV and de-dupes on a row-level
# primary key so re-runs within the overlap window are idempotent.
#
# The anchor is the max WEEK_END_DATE already present in alist_75k.csv. Using
# a single canonical anchor for all three keeps the three files in lockstep —
# even if Current_Data happens to publish a week earlier than alist, we never
# persist a Current_Data row that wouldn't also be covered by a future alist
# refresh.
#
# Cold start (no alist_75k.csv yet): anchor defaults to '2018-01-01', which
# reproduces the original behavior of the un-parameterized queries.
#
# Parquet queries (query_model_parquet_metrics*.sql) are intentionally NOT
# incremental in this phase: they back the archetype KMeans models, which we
# retrain infrequently, and are invoked via /refresh_model rather than the
# weekly cron.

# Cold-start anchor when alist_75k.csv does not yet exist.
_COLD_START_MIN_WEEK = "2018-01-01"


def refresh_data(*, csv_only: bool = False) -> None:
    """
    Incrementally refresh the three weekly CSVs from Snowflake, then train artifacts.

    ``csv_only=False`` (default): full train including AE parquet KMeans/DNA and
    archetype decay (heavy; use from ``/v1/data/refresh_data`` or ad-hoc runs).

    ``csv_only=True``: CSV pull + LGBM/Prophet/spike/df_full only; skips parquet
    reads and archetype retrains. Used by ``refresh_weekly`` to avoid OOM.
    """
    _set_step("refresh_data:pull_csvs")
    _refresh_data_directory()
    _set_step("refresh_data:train_artifacts")
    train_model(csv_only=csv_only)


def update_parquet_metrics(sf: Snowflake) -> None:
    """
    Updates: streams_product_songs_ae_compressed.parquet

    Not incremental — archetype decay KMeans is retrained on the full history
    and these parquets are only rebuilt via /refresh_model, not the weekly
    /refresh_weekly path.
    """
    _set_step("refresh_parquets:ae_query")
    df = sf.query(load_sql(MODEL_PARQUET_METRICS_QUERY))
    _set_step("refresh_parquets:worldwide_query")
    df_streaming = sf.query(load_sql(MODEL_PARQUET_METRICS_STREAMING_QUERY))

    _set_step("refresh_parquets:write")
    if not df.empty:
        logger.info("train_model.py: Updated streams_product_songs_ae_compressed.parquet")
        df.to_parquet(DATA_DIR / "streams_product_songs_ae_compressed.parquet", index=False)

    if not df_streaming.empty:
        logger.info("train_model.py: Updated worldwide_streams_compressed.parquet")
        df_streaming.to_parquet(DATA_DIR / "worldwide_streams_compressed.parquet", index=False)


def _refresh_data_directory() -> None:
    """
    Incrementally refresh the three weekly CSVs. Sequential (not ThreadPooled)
    because `snowflake.connector` cursors serialize work on a single socket —
    the old pool gave us contention without actual parallelism. A future pass
    can give each query its own connection if we ever observe Snowflake-side
    queuing as the bottleneck.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    min_week = _get_min_week_end_date()
    logger.info("train_model.py: Refreshing CSV directory (min_week_end_date=%s)", min_week)

    with get_snowflake_connection() as sf:
        for name, updater in (
            ("Current_Data.csv", _update_current_data),
            ("alist_75k.csv", _update_a_list_75k),
            ("bigreleaseflag_75k.csv", _update_big_release_flag_75k),
        ):
            _set_step(f"refresh_data:csv:{name}")
            _run_stage(name, lambda sf=sf, updater=updater: updater(sf, min_week))


def _run_stage(name: str, fn: Callable[[], int]) -> None:
    t0 = time.perf_counter()
    added = fn()
    elapsed = time.perf_counter() - t0
    logger.info("train_model.py: %s +%d rows (%.1fs)", name, added, elapsed)


def _get_min_week_end_date() -> str:
    """
    Canonical incremental anchor: the max WEEK_END_DATE currently in
    alist_75k.csv. Defaults to '2018-01-01' so a cold run (no CSV yet)
    reproduces the original full-history pull.

    A corrupt or schema-drifted CSV also falls back to cold start rather than
    raising — safer to over-pull once than to skip weeks silently.
    """
    alist = DATA_DIR / "alist_75k.csv"
    if not alist.exists():
        logger.info("alist_75k.csv not found; cold start from %s", _COLD_START_MIN_WEEK)
        return _COLD_START_MIN_WEEK
    try:
        df = pd.read_csv(alist, usecols=["WEEK_END_DATE"])
    except (ValueError, KeyError) as e:
        logger.warning("alist_75k.csv missing WEEK_END_DATE column (%s); cold start", e)
        return _COLD_START_MIN_WEEK
    if df.empty:
        return _COLD_START_MIN_WEEK
    max_wk = pd.to_datetime(df["WEEK_END_DATE"], errors="coerce").max()
    if pd.isna(max_wk):
        logger.warning("alist_75k.csv WEEK_END_DATE column is all NaT; cold start")
        return _COLD_START_MIN_WEEK
    return max_wk.strftime("%Y-%m-%d")


def _normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Snowflake's connector returns DATE columns as ``datetime.date`` objects,
    but the persisted CSVs store them as ISO ``YYYY-MM-DD`` strings (because
    pandas re-reads them as strings on the next run). Mixing the two in a
    single column after ``pd.concat`` breaks ``sort_values`` with
    ``TypeError: '<' not supported between instances of 'datetime.date' and 'str'``.

    Normalize every date-ish column to a string so concat + sort + dedupe all
    operate on a single, comparable dtype.
    """
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            df[col] = s.dt.strftime("%Y-%m-%d")
            continue
        if s.dtype == object:
            non_null = s.dropna()
            if not non_null.empty and all(
                isinstance(v, datetime.date) and not isinstance(v, datetime.datetime)
                for v in non_null
            ):
                df[col] = s.map(
                    lambda v: v.strftime("%Y-%m-%d") if isinstance(v, datetime.date) else v
                )
    return df


def _run_incremental_query(sf: Snowflake, query_name: str, min_week: str) -> pd.DataFrame:
    """Render {MIN_WEEK_END_DATE} into a SQL template and execute."""
    sql = load_sql(query_name).replace("{MIN_WEEK_END_DATE}", min_week)
    df = sf.query(sql)
    df = _normalize_date_columns(df)
    return df


def _append_and_write_csv(
    path: Path,
    new_df: pd.DataFrame,
    dedupe_subset: list[str],
    sort_by: list[str],
) -> int:
    """
    Merge new rows into an existing CSV and rewrite the file.

    - No-op (returns 0) when `new_df` is empty. An empty incremental pull is
      a normal outcome mid-week or right after a refresh — don't panic.
    - De-dupes on `dedupe_subset` preferring the newly-fetched row so a late-
      arriving Snowflake correction overwrites the stale row on the next run.
    - Cold-start case: writes `new_df` (dedup'd) directly.

    Returns the net row delta (rows added beyond what was already on disk).
    """
    if new_df is None or new_df.empty:
        return 0

    if path.exists():
        existing = pd.read_csv(path)
        before = len(existing)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        before = 0
        merged = new_df.copy()

    merged = merged.drop_duplicates(subset=dedupe_subset, keep="last")
    if sort_by:
        merged = merged.sort_values(sort_by).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return len(merged) - before


def _update_current_data(sf: Snowflake, min_week: str) -> int:
    """
    Current_Data rows are one-per-(week, label). De-dupe on that composite.
    """
    df = _run_incremental_query(sf, CURRENT_DATA_QUERY, min_week)
    return _append_and_write_csv(
        DATA_DIR / "Current_Data.csv",
        df,
        dedupe_subset=["WEEK_ENDING_DATE", "LABEL_NAME"],
        sort_by=["WEEK_ENDING_DATE"],
    )


def _update_a_list_75k(sf: Snowflake, min_week: str) -> int:
    """alist_75k rows are one-per-week. De-dupe on WEEK_END_DATE alone."""
    df = _run_incremental_query(sf, A_LIST_75K_QUERY, min_week)
    return _append_and_write_csv(
        DATA_DIR / "alist_75k.csv",
        df,
        dedupe_subset=["WEEK_END_DATE"],
        sort_by=["WEEK_END_DATE"],
    )


def _update_big_release_flag_75k(sf: Snowflake, min_week: str) -> int:
    """bigreleaseflag rows are one-per-week. De-dupe on WEEK_END_DATE alone."""
    df = _run_incremental_query(sf, BIG_RELEASE_FLAG_75K_QUERY, min_week)
    return _append_and_write_csv(
        DATA_DIR / "bigreleaseflag_75k.csv",
        df,
        dedupe_subset=["WEEK_END_DATE"],
        sort_by=["WEEK_END_DATE"],
    )
