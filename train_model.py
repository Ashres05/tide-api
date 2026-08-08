from model.train_marketshare_artifacts import train_artifacts_main as train_model
from snowflake_conn import Snowflake, get_snowflake_connection, load_sql
from pathlib import Path
from typing import Any, Callable
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

CURRENT_DATA_ALL_QUERY = "query_model_current_data_all.sql"
BIGRELEASE_ALIST_75K_ALL_QUERY = "query_model_bigrelease_alist_75k_all.sql"
YTD_FISCAL_REVENUE_BY_LABEL_QUERY = "query_ytd_fiscal_revenue_by_label.sql"
RELEASES_BY_Q_AMG_LABELS_QUERY = "query_releases_by_q_amg_labels.sql"
# monthly, not quarterly now but name left for ease
QUARTERLY_SHARE_AND_QTD_QUERY = "query_monthly_share_and_qtd.sql"
QUARTERLY_SHARE_AND_QTD_MAX_P_DAY_QUERY = "query_quarterly_share_and_qtd_max_p_day.sql"
# monthly, not quarterly now but name left for ease
QUARTERLY_SHARE_LEVEL3_QUERY = "query_monthly_share_level3.sql"

MODEL_PARQUET_METRICS_QUERY = "query_model_parquet_metrics.sql"
MODEL_PARQUET_METRICS_STREAMING_QUERY = "query_model_parquet_metrics_streaming.sql"

# Phase 2 design notes
# --------------------
# The weekly CSV queries (current_data_all, bigrelease_alist_75k_all) accept
# a {MIN_WEEK_END_DATE} placeholder and emit only weeks >= that anchor (with an
# upper guard so the in-progress week is never persisted). The Python layer
# appends the result onto the existing CSV and de-dupes on a row-level primary
# key so re-runs within the overlap window are idempotent.
#
# Each CSV computes its own anchor from its own max week column. This avoids
# a stale/corrupt file pinning the other to an old lower bound.
#
# Cold start (CSV missing): anchor defaults to '2018-01-01', which reproduces
# the original full-history behavior.
#
# Parquet queries (query_model_parquet_metrics*.sql) are intentionally NOT
# incremental in this phase: they back the archetype KMeans models, which we
# retrain infrequently, and are invoked via /refresh_model rather than the
# weekly cron.

# Cold-start anchor when a weekly model CSV does not yet exist.
_COLD_START_MIN_WEEK = "2018-01-01"

# Quarterly-share CSVs read bi_sandbox. When that database is unavailable the
# weekly cron should still refresh core model CSVs, train artifacts, backfill,
# and push to S3.
_BI_SANDBOX_CSV_STAGES = (
    "monthly_share_and_qtd.csv",
    "monthly_share_labels.csv",
)


def refresh_data(*, csv_only: bool = False) -> dict[str, Any]:
    """
    Incrementally refresh weekly model CSVs from Snowflake, then train artifacts.

    ``csv_only=False`` (default): full train including AE parquet KMeans/DNA and
    archetype decay (heavy; use from ``/v1/data/refresh_data`` or ad-hoc runs).

    ``csv_only=True``: CSV pull + LGBM/Prophet/spike/df_full only; skips parquet
    reads and archetype retrains. Used by ``refresh_weekly`` to avoid OOM.

    Returns ``{"csv_stages": [...], "optional_csv_errors": [...]}`` for cron
    logging and ``GET /v1/jobs/{id}`` diagnostics.
    """
    _set_step("refresh_data:pull_csvs")
    csv_stages, optional_csv_errors = _refresh_data_directory()
    _set_step("refresh_data:train_artifacts")
    train_model(csv_only=csv_only)
    return {"csv_stages": csv_stages, "optional_csv_errors": optional_csv_errors}


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


def _refresh_data_directory() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Incrementally refresh weekly CSVs from Snowflake. Sequential because
    snowflake.connector cursors serialize work on a single socket.

    Core model CSVs (current_data_all, bigrelease_alist_75k_all) and
    releases_by_q_amg_labels are required — failures abort refresh_data.
    Monthly-share CSVs depend on bi_sandbox and are best-effort.
    """
    optional_errors: list[dict[str, str]] = []
    csv_stages: list[dict[str, Any]] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current_min_week = _get_min_week_end_date(
        DATA_DIR / "current_data_all.csv",
        week_col="WEEK_ENDING_DATE",
    )
    bigrelease_alist_min_week = _get_min_week_end_date(
        DATA_DIR / "bigrelease_alist_75k_all.csv",
        week_col="WEEK_END_DATE",
    )
    logger.info(
        "train_model.py: Refreshing CSV directory "
        "(current_data_all min=%s, bigrelease_alist_75k_all min=%s)",
        current_min_week,
        bigrelease_alist_min_week,
    )

    with get_snowflake_connection() as sf:
        for name, updater, min_week, week_col in (
            (
                "current_data_all.csv",
                _update_current_data_all,
                current_min_week,
                "WEEK_ENDING_DATE",
            ),
            (
                "bigrelease_alist_75k_all.csv",
                _update_bigrelease_alist_75k_all,
                bigrelease_alist_min_week,
                "WEEK_END_DATE",
            ),
        ):
            csv_stages.append(
                _run_stage(
                    name,
                    lambda sf=sf, updater=updater, min_week=min_week: updater(sf, min_week),
                    csv_path=DATA_DIR / name,
                    week_col=week_col,
                    anchor_week=min_week,
                )
            )
        bi_sandbox_updaters = {
            "monthly_share_and_qtd.csv": _update_quarterly_share_and_qtd,
            "monthly_share_labels.csv": _update_quarterly_share_level3,
        }
        for name in _BI_SANDBOX_CSV_STAGES:
            updater = bi_sandbox_updaters[name]
            csv_stages.append(
                _run_stage_optional(
                    name,
                    lambda sf=sf, updater=updater: updater(sf),
                    optional_errors,
                )
            )
        csv_stages.append(
            _run_stage(
                "releases_by_q_amg_labels.csv",
                lambda sf=sf: _update_releases_by_q_amg_labels(sf),
                csv_path=DATA_DIR / "releases_by_q_amg_labels.csv",
            )
        )
    return csv_stages, optional_errors


def _csv_max_week(path: Path, week_col: str | None) -> str | None:
    """Latest week in a persisted CSV, for cron/job logging."""
    if not week_col or not path.exists():
        return None
    try:
        return _get_min_week_end_date(path, week_col=week_col)
    except Exception:
        return None


def _run_stage(
    name: str,
    fn: Callable[[], int],
    *,
    csv_path: Path | None = None,
    week_col: str | None = None,
    anchor_week: str | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    added = fn()
    elapsed = time.perf_counter() - t0
    max_week = _csv_max_week(csv_path, week_col) if csv_path else None
    stage = {
        "csv": name,
        "rows_added": int(added),
        "max_week": max_week,
        "anchor_week": anchor_week,
        "elapsed_sec": round(elapsed, 2),
    }
    step_msg = f"refresh_data:csv:{name}:+{added}_rows"
    if max_week:
        step_msg += f",max_week={max_week}"
    _set_step(step_msg)
    logger.info(
        "train_model.py: %s +%d rows (%.1fs)%s",
        name,
        added,
        elapsed,
        f", max_week={max_week}" if max_week else "",
    )
    return stage


def _run_stage_optional(
    name: str,
    fn: Callable[[], int],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Run a bi_sandbox CSV stage; record and continue when Snowflake denies access."""
    try:
        return _run_stage(name, fn)
    except Exception as e:
        logger.exception(
            "train_model.py: optional BI_SANDBOX stage %s failed; continuing weekly refresh",
            name,
        )
        _set_step(f"refresh_data:csv:{name}:skipped")
        errors.append({"csv": name, "error": str(e)})
        return {"csv": name, "rows_added": 0, "max_week": None, "skipped": True, "error": str(e)}


def _get_min_week_end_date(path: Path, *, week_col: str) -> str:
    """
    Incremental anchor for a single CSV: max value currently in `week_col`.
    Defaults to '2018-01-01' so a cold run (no CSV yet)
    reproduces the original full-history pull.

    A corrupt or schema-drifted CSV also falls back to cold start rather than
    raising — safer to over-pull once than to skip weeks silently.
    """
    if not path.exists():
        logger.info("%s not found; cold start from %s", path.name, _COLD_START_MIN_WEEK)
        return _COLD_START_MIN_WEEK
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        if week_col not in df.columns:
            raise KeyError(week_col)
        df = df[[week_col]]
    except (ValueError, KeyError) as e:
        logger.warning("%s missing %s column (%s); cold start", path.name, week_col, e)
        return _COLD_START_MIN_WEEK
    if df.empty:
        return _COLD_START_MIN_WEEK
    max_wk = pd.to_datetime(df[week_col], errors="coerce").max()
    if pd.isna(max_wk):
        logger.warning("%s %s column is all NaT; cold start", path.name, week_col)
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


def _run_incremental_first_sale_query(
    sf: Snowflake, query_name: str, min_first_sale_date: str
) -> pd.DataFrame:
    """Render {MIN_FIRST_SALE_DATE} into a SQL template and execute."""
    sql = load_sql(query_name).replace("{MIN_FIRST_SALE_DATE}", min_first_sale_date)
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
        existing.columns = [str(c).strip().lstrip("\ufeff") for c in existing.columns]
        before = len(existing)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        before = 0
        merged = new_df.copy()

    merged.columns = [str(c).strip().lstrip("\ufeff") for c in merged.columns]
    merged = merged.drop_duplicates(subset=dedupe_subset, keep="last")
    if sort_by:
        merged = merged.sort_values(sort_by).reset_index(drop=True)
    merged.to_csv(path, index=False)
    return len(merged) - before


def _update_current_data_all(sf: Snowflake, min_week: str) -> int:
    """
    current_data_all rows are one-per-(week, label). De-dupe on that composite.
    """
    df = _run_incremental_query(sf, CURRENT_DATA_ALL_QUERY, min_week)
    df.columns = [str(c).strip().lstrip("\ufeff").upper() for c in df.columns]
    return _append_and_write_csv(
        DATA_DIR / "current_data_all.csv",
        df,
        dedupe_subset=["WEEK_ENDING_DATE", "LABEL_NAME"],
        sort_by=["WEEK_ENDING_DATE", "LABEL_NAME"],
    )


def _update_bigrelease_alist_75k_all(sf: Snowflake, min_week: str) -> int:
    """Combined A-list scrub + big-release flags; one row per WEEK_END_DATE."""
    df = _run_incremental_query(sf, BIGRELEASE_ALIST_75K_ALL_QUERY, min_week)
    df.columns = [str(c).strip().lstrip("\ufeff").upper() for c in df.columns]
    return _append_and_write_csv(
        DATA_DIR / "bigrelease_alist_75k_all.csv",
        df,
        dedupe_subset=["WEEK_END_DATE"],
        sort_by=["WEEK_END_DATE"],
    )


def _update_ytd_fiscal_revenue_by_label(sf: Snowflake, min_week: str) -> int:
    """
    Weekly proxy revenue by distributor label (worldwide on-demand streams * 0.004).
    One row per (week_end_date, level_1, level_2, level_3).
    """
    df = _run_incremental_query(sf, YTD_FISCAL_REVENUE_BY_LABEL_QUERY, min_week)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return _append_and_write_csv(
        DATA_DIR / "ytd_fiscal_revenue_by_label.csv",
        df,
        dedupe_subset=[
            "week_end_date",
            "level_1_distributor",
            "level_2_distributor",
            "level_3_distributor",
        ],
        sort_by=[
            "week_end_date",
            "level_1_distributor",
            "level_2_distributor",
            "level_3_distributor",
        ],
    )


def _snowflake_max_p_day_for_quarterly_share(sf: Snowflake) -> datetime.date | None:
    import quarterly_share_from_csv

    df = sf.query(load_sql(QUARTERLY_SHARE_AND_QTD_MAX_P_DAY_QUERY))
    if df.empty:
        return None
    col = "max_p_day" if "max_p_day" in df.columns else df.columns[0]
    return quarterly_share_from_csv.coerce_snowflake_max_p_day(df.iloc[0][col])


def _update_quarterly_share_and_qtd(sf: Snowflake) -> int:
    """Rewrite monthly_share_and_qtd.csv when Snowflake max(p_day) advances."""
    import quarterly_share_from_csv

    csv_path = DATA_DIR / "monthly_share_and_qtd.csv"
    local_max = quarterly_share_from_csv.max_p_day_from_csv(csv_path)
    snowflake_max = _snowflake_max_p_day_for_quarterly_share(sf)
    if snowflake_max is None:
        logger.warning("monthly_share_and_qtd: Snowflake max(p_day) unavailable; skipping")
        return 0
    if snowflake_max <= local_max:
        logger.info(
            "monthly_share_and_qtd: up to date (snowflake max=%s local max=%s)",
            snowflake_max,
            local_max,
        )
        return 0
    logger.info(
        "monthly_share_and_qtd: refreshing (snowflake max=%s > local max=%s)",
        snowflake_max,
        local_max,
    )
    df = sf.query(load_sql(QUARTERLY_SHARE_AND_QTD_QUERY))
    df = _normalize_date_columns(df)
    if "p_day" in df.columns:
        df = df.drop(columns=["p_day"])
    df.columns = [str(c).strip().upper() for c in df.columns]
    out = quarterly_share_from_csv.normalize_quarterly_share_dataframe(df)
    before = 0
    if csv_path.exists():
        try:
            before = len(pd.read_csv(csv_path))
        except Exception:
            before = 0
    out.to_csv(csv_path, index=False)
    quarterly_share_from_csv.clear_cache()
    return len(out) - before


def _update_quarterly_share_level3(sf: Snowflake) -> int:
    """Rewrite monthly_share_labels.csv when Snowflake max(p_day) advances."""
    import quarterly_share_level3_from_csv as lvl3_csv

    csv_path = DATA_DIR / "monthly_share_labels.csv"
    local_max = lvl3_csv.max_p_day_from_level3_csv(csv_path)
    snowflake_max = _snowflake_max_p_day_for_quarterly_share(sf)
    if snowflake_max is None:
        logger.warning("monthly_share_labels: Snowflake max(p_day) unavailable; skipping")
        return 0
    if snowflake_max <= local_max:
        logger.info(
            "monthly_share_labels: up to date (snowflake max=%s local max=%s)",
            snowflake_max,
            local_max,
        )
        return 0
    logger.info(
        "monthly_share_labels: refreshing (snowflake max=%s > local max=%s)",
        snowflake_max,
        local_max,
    )
    df = sf.query(load_sql(QUARTERLY_SHARE_LEVEL3_QUERY))
    df = _normalize_date_columns(df)
    if "p_day" in df.columns:
        df = df.drop(columns=["p_day"])
    df.columns = [str(c).strip().upper() for c in df.columns]
    out = lvl3_csv.normalize_quarterly_share_level3_dataframe(df)
    before = 0
    if csv_path.exists():
        try:
            before = len(pd.read_csv(csv_path))
        except Exception:
            before = 0
    out.to_csv(csv_path, index=False)
    lvl3_csv.clear_cache()
    return len(out) - before


def _update_releases_by_q_amg_labels(sf: Snowflake) -> int:
    """
    Append releases with ``first_sale_date`` greater than the CSV max (cold start
    anchor 2023-08-31 so the first pull includes 2023-09-01+). Pushed to S3 with
    other model/data CSVs on refresh_weekly.
    """
    import releases_by_q_amg_labels_from_csv as rel_csv

    csv_path = DATA_DIR / "releases_by_q_amg_labels.csv"
    min_fsd = rel_csv.max_first_sale_date_from_csv(csv_path)
    df = _run_incremental_first_sale_query(sf, RELEASES_BY_Q_AMG_LABELS_QUERY, min_fsd)
    df.columns = [str(c).strip().upper() for c in df.columns]
    added = _append_and_write_csv(
        csv_path,
        rel_csv.normalize_releases_dataframe(df),
        dedupe_subset=["MRELG_ID"],
        sort_by=["DISPLAY_ARTIST", "TITLE", "FIRST_SALE_DATE"],
    )
    if added:
        rel_csv.clear_cache()
    return added
