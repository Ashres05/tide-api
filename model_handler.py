from __future__ import annotations
import json
import math
import numbers
import sqlite3
import logging
import pandas as pd

from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List
from sqlite_handler import DATABASE_NAME, ensure_expected_releases_fw_columns
from model.marketshare_75k_simulation import DISTRIBUTIONS, NUM_WEEKS
from snowflake_conn import load_sql
from model.forecast_engine_server import ForecastEngine
from snowflake_conn import get_snowflake_connection, Snowflake
from train_model import refresh_data, update_parquet_metrics
from sqlite_handler import update_sqlite_main
from model.worldwide_streams_api import simulate_one_worldwide_streams

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
    fw_vol: float = 100000.0 # Default to 100,000 units for backfilled releases, this doesn't matter as backfilled releases will have known volumes.
) -> int:
    """
    Creates a new release in the database from a mrelg_id.
    Returns the release ID.
    """
    with get_snowflake_connection() as sf:
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
            raise ValueError(f"MRELG ID {mrelg_id} has no {col} in the metadata. Try create_release() instead.")

    
    return create_release(mrelg_id=mrelg_id, 
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
        cluster=0)


def backfill_releases() -> dict:
    """
    Backfills releases from Snowflake into the local SQLite database.

    Returns a summary dict with counts and any per-release errors:
        {"inserted": int, "skipped": int, "errors": [{"mrelg_id": str, "error": str}, ...]}
    """
    query = load_sql(RELEASE_BACKFILL_QUERY)
    
    # Get the releases from Snowflake.
    with get_snowflake_connection() as sf:
        df = sf.query(query)

    if df.empty:
        return {"inserted": 0, "skipped": 0, "errors": []}

    # Get the existing mrelg_ids from the SQLite database.
    existing_mrelg_ids = {
        (row["MRELG_ID"] or "").strip()
        for row in _get_all_release_rows()
        if "MRELG_ID" in row.keys() and row["MRELG_ID"]
    }

    # Keep track of the number of inserted, skipped, and errors.
    inserted = 0
    skipped = 0
    errors: list[dict] = []

    for mrelg_id, label_name in df.itertuples(index=False, name=None):
        if mrelg_id in existing_mrelg_ids:
            # Skip if the release already exists in the database.
            skipped += 1
            continue

        try:
            # Create the release in the database.
            _create_backfilled_release(mrelg_id=mrelg_id, label_name=label_name)
            inserted += 1
            existing_mrelg_ids.add(mrelg_id)
        except Exception as e:
            errors.append({"mrelg_id": mrelg_id, "error": str(e)})
    try:
        # Refresh SQLite data for backfilled releases.
        if inserted > 0:
            update_sqlite_main()
    except Exception as e:
        errors.append({"stage": "update_sqlite_main", "error": str(e) + "\nWARNING: This means official weekly equivalent data for some releases may not be available at the moment."})

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


def get_known_vols(
    release_id: int,
) -> List[float]:
    """
    Pull known weekly actuals (ALBUM_EQUIVALENT) for a release from SQLite, ordered by week end.
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
        return []
    vals = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    vals = vals.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return [float(x) for x in vals.to_list()]


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
    Returns the year-to-date observed marketshare timeline from MARKETSHARE_YTD,
    reshaped to match the unified_ytd schema. Rows tagged Data_Type='Actual'.
    """
    query = load_sql(MARKETSHARE_ACTUALS_QUERY)
    with sqlite3.connect(DATABASE_NAME) as conn:
        df = pd.read_sql_query(query, conn)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Week Ending Date", "Owner", "Total_Market_AE_Volume",
                "Active_Share", "Data_Type", "Unified_YTD_Share",
                "YTD_Share_Upper", "YTD_Share_Lower",
            ]
        )
    out = pd.DataFrame(
        {
            "Week Ending Date": df["WEEK_ENDING_DATE"].astype(str),
            "Owner": df["LABEL_NAME"].astype(str),
            "Total_Market_AE_Volume": pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce").fillna(0),
            "Active_Share": pd.to_numeric(df["ALBUM_EQUIVALENT_SHARE"], errors="coerce").fillna(0),
            "Data_Type": "Actual",
            "Unified_YTD_Share": pd.to_numeric(df["ALBUM_EQUIVALENT_SHARE"], errors="coerce").fillna(0),
        }
    )
    out["YTD_Share_Upper"] = out["Unified_YTD_Share"]
    out["YTD_Share_Lower"] = out["Unified_YTD_Share"]
    return out


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
    """
    refresh_data()
    reload_artifacts()


def refresh_model() -> None:
    """
    Refreshes the model.
    """
    with get_snowflake_connection() as sf:
        update_parquet_metrics(sf)
    reload_artifacts()


_PARQUET_DIR = Path(__file__).resolve().parent / "model" / "data"
_REQUIRED_PARQUETS = (
    _PARQUET_DIR / "streams_product_songs_ae_compressed.parquet",
    _PARQUET_DIR / "worldwide_streams_compressed.parquet",
)


def refresh_weekly(force_refresh_parquets: bool = False) -> dict:
    """
    Single weekly orchestration: (optionally) refresh parquets, refresh CSV
    data and retrain, then backfill releases. This is what the weekly cron
    should call — it replaces the prior pattern of invoking /refresh_model,
    /refresh_data, and /releases/backfill separately (which made ordering
    easy to get wrong).

    Stage order:
      1. update_parquet_metrics — AE + worldwide_streams parquets feed the
         archetype KMeans step that train_artifacts_main runs. SKIPPED when
         both parquet files already exist on disk, because the two queries
         that back this stage scan Luminate from 2018 to present and take
         multiple minutes each. The parquets change infrequently; force a
         refresh by passing force_refresh_parquets=True or by calling the
         dedicated /v1/data/refresh_model endpoint.
      2. refresh_data — pulls the three CSVs (incremental since Phase 2) and
         retrains LGBM / Prophet / Ridge and (if parquets exist) archetype
         decay models.
      3. backfill_releases — inserts any new mrelg_ids into SQLite and
         refreshes per-release historical metrics used by /weekly forecasts.

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

    set_step("refresh_parquets:start")
    t0 = _now()
    missing = [p for p in _REQUIRED_PARQUETS if not p.is_file()]
    if not force_refresh_parquets and not missing:
        existing = [p.name for p in _REQUIRED_PARQUETS]
        logger.info(
            "refresh_weekly: skipping parquet refresh (found %s). "
            "Call refresh_weekly(force_refresh_parquets=True) or /refresh_model "
            "to rebuild.",
            existing,
        )
        set_step("refresh_parquets:skipped")
        summary["stages"]["refresh_parquets"] = {
            "ok": True,
            "skipped": True,
            "reason": "parquets already exist on disk",
            "paths": existing,
            "elapsed_sec": _elapsed(t0),
        }
    else:
        if force_refresh_parquets:
            logger.info("refresh_weekly: force_refresh_parquets=True; rebuilding parquets")
        else:
            logger.info(
                "refresh_weekly: parquet(s) missing, rebuilding: %s",
                [str(p) for p in missing],
            )
        try:
            with get_snowflake_connection() as sf:
                update_parquet_metrics(sf)
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

    set_step("refresh_data:start")
    t0 = _now()
    try:
        refresh_data()
        summary["stages"]["refresh_data"] = {"ok": True, "elapsed_sec": _elapsed(t0)}
    except Exception as e:
        logger.exception("refresh_weekly: refresh_data failed")
        summary["stages"]["refresh_data"] = {
            "ok": False, "error": str(e), "elapsed_sec": _elapsed(t0),
        }
        set_step("reload_artifacts")
        reload_artifacts()
        raise

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
    set_step("done")
    return summary


def _now() -> float:
    import time as _t
    return _t.perf_counter()


def _elapsed(t0: float) -> float:
    import time as _t
    return round(_t.perf_counter() - t0, 2)


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
    component_vols: Dict[str, List[float]] = {"streams": [], "sales": [], "songs": []}
    if mrelg_id and rid is not None:
        try:
            known_vols = get_known_vols(int(rid))
        except Exception:
            known_vols = []
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
    """
    # Verify parameters.
    _verify_id(id)
    release = get_release(id)
    mrelg_id = (release.get("mrelg_id") or "").strip()
    if not mrelg_id:
        raise ValueError(
            f"Release {id} has no mrelg_id; worldwide streaming forecast requires "
            "a Luminate release group ID."
        )

    # Get historical observed weeks from Snowflake.
    with get_snowflake_connection() as _sf:
        hist_df = _get_known_vols_global_streaming(mrelg_id, _sf)
    if hist_df.empty:
        raise ValueError(f"No historical observed weeks found for mrelg_id: {mrelg_id}")

    # Extract ordered weekly raw stream counts; empty list = cold-start mode.
    known: List[float] = []
    if not hist_df.empty:
        stream_col = next(
            (c for c in hist_df.columns if "stream" in c.lower()),
            hist_df.columns[-1],
        )
        series = pd.to_numeric(hist_df[stream_col], errors="coerce").fillna(0.0)
        known = series.tolist()

    # Cold-start peak: prefer fw_streams, fall back to fw_vol.
    fw_peak = float(release.get("fw_streams") or 0.0) or float(release.get("fw_vol") or 0.0)
    if not any(x > 0 for x in known) and fw_peak <= 0:
        raise ValueError(
            f"Release {id} ({mrelg_id}) has no observed worldwide stream history "
            "and no fw_streams / fw_vol peak set. "
            "Provide at least one observed week or set fw_streams > 0."
        )

    artifacts = get_worldwide_artifacts()
    release_dict: Dict[str, Any] = {
        "artist": release.get("artist") or release.get("name") or "",
        "name": release.get("name") or "",
        "genre": release.get("genre"),
        "date": release.get("date"),
        "known_worldwide_streams": known,
        "fw_worldwide_streams": fw_peak,
    }

    result = simulate_one_worldwide_streams(release_dict, artifacts, end_week=int(artifacts.horizon_weeks))

    n_known = len(known)
    df = pd.DataFrame(result["weekly"])  # week, pred_worldwide_streams, cumulative_worldwide_streams

    # --- week_ending_date -------------------------------------------------------
    # Observed weeks: use real dates from hist_df (already ordered by week_end_date).
    # Forecast weeks: extrapolate 7 days per week past the last known date.
    # Cold-start (no hist_df): derive entirely from release date.
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
        release_date = pd.to_datetime(release.get("date"))

        def _week_to_date(week: int) -> str:  # type: ignore[misc]
            return (release_date + pd.Timedelta(weeks=week)).strftime("%Y-%m-%d")

    df["week_ending_date"] = df["week"].apply(_week_to_date)

    # --- data_type --------------------------------------------------------------
    df["data_type"] = df["week"].apply(lambda w: "Actual" if w <= n_known else "Forecast")

    # --- final column order -----------------------------------------------------
    df.insert(0, "release_id", id)
    df.insert(1, "mrelg_id", mrelg_id)
    df.insert(2, "artist", release.get("artist") or "")
    df.insert(3, "title", release.get("title") or release.get("name") or "")
    # Place week_ending_date and data_type immediately after week
    week_pos = df.columns.get_loc("week")
    for col in ("data_type", "week_ending_date"):
        df.insert(week_pos + 1, col, df.pop(col))

    return df


def _get_known_vols_global_streaming(mrelg_id: str, _sf: Snowflake) -> pd.DataFrame:
    df = _sf.query(load_sql(GLOBAL_STREAMING_QUERY).replace("{MRELG_ID}", f"'{mrelg_id}'"))
    if df.empty:
        raise ValueError(f"No global streaming data found for mrelg_id: {mrelg_id}")
    return df
