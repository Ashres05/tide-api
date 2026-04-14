# TODO: Create CRUD endpoints for the Tide API.
# TODO: Upload API to EC2 instance.
from __future__ import annotations

import json
import math
import numbers
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from sqlite_handler import DATABASE_NAME
from model.marketshare_75k_simulation import DISTRIBUTIONS
from snowflake_conn import load_sql
from model.forecast_engine_server import ForecastEngine

# Default training output: model/train_marketshare_artifacts.py writes here (not repo-root artifacts_75k).
ARTIFACTS_DIR = Path(__file__).resolve().parent / "model" / "artifacts_75k"

# SQLite table name for observed per-release metrics (populated by sqlite_handler.py)
MARKETSHARE_RELEASE_METRICS_TABLE = "MARKETSHARE_RELEASE_METRICS"

# Query names for the database.
RELEASE_CREATE_QUERY = "release_create.sql"
RELEASE_UPDATE_QUERY = "release_update.sql"
RELEASE_DELETE_QUERY = "release_delete.sql"
RELEASE_GET_QUERY = "release_get.sql"
RELEASE_GET_ALL_QUERY = "release_get_all.sql"

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
    "fy_vol",
    "avg_historical_w1_product_ratio",
    "product_ratio_coefficient",
)

_ALLOWED_SCENARIOS = frozenset[str]({"Bear", "Base", "Bull"})


def create_release(
    *,
    mrelg_id: str | None = None,
    name: str, 
    artist: str, 
    label_name: str, 
    release_date: str, 
    genre: str, 
    scenario: str, 
    known_vols: List[float] = [], 
    fw_vol: float = 0.0, 
    fy_vol: float = 0.0, 
    avg_historical_w1_product_ratio: float = 0.3, 
    product_ratio_coefficient: float = 0.3,
    cluster: int = 0,
    **_kwargs,
) -> int:
    """
    Creates a new release in the database.
    Returns the release ID.
    """
    _verify_release_fields(locals())

    params = (
        mrelg_id,
        name,
        artist,
        label_name,
        release_date,
        genre,
        fw_vol,
        scenario,
        fy_vol,
        avg_historical_w1_product_ratio,
        product_ratio_coefficient,
        cluster,
    )
    try:
        query = load_sql(RELEASE_CREATE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            id = cursor.execute(query, params).fetchone()[0]
            conn.commit()
        return id
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error creating release: {e}") from e

# TODO: Adjust the usage of known_volumes_json within the database.
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
    fy_vol: float = 0.0, 
    avg_historical_w1_product_ratio: float = 0.3, 
    product_ratio_coefficient: float = 0.3,
    cluster: int = 0,
    **_kwargs,
) -> None:
    """Updates a release in the database."""

    _verify_id(id)
    _verify_release_fields(locals())

    params = (
        mrelg_id,
        name,
        artist,
        label_name,
        release_date,
        genre,
        fw_vol,
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
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"No release found with id={id}.")
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


def get_known_vols_from_sqlite(
    release_id: int,
) -> List[float]:
    """
    Pull known weekly actuals (ALBUM_EQUIVALENT) for a release from SQLite, ordered by week end.
    """
    _verify_id(release_id)

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


def get_marketshare_forecasts(week_ending_date: str | None = None) -> pd.DataFrame:
    """
    Takes in a week ending date and returns the marketshare forecasts for that week.
    The week ending date must be in the format YYYY-MM-DD if provided.
    If no week ending date is provided, all marketshare forecasts are returned.
    """
    # Verify the week ending date (if provided).
    if week_ending_date is not None:
        _verify_week_ending_date(week_ending_date)

    # Get all releases and verify the parquet file.
    releases = [_sqlite_row_to_release_map(row) for row in _get_all_release_rows()]
    df_full = ARTIFACTS_DIR / "df_full.parquet"
    _verify_parquet_file(df_full)

    # Simulate the releases and return the marketshare forecasts for the week ending date.
    forecasts = ForecastEngine(artifacts_dir=ARTIFACTS_DIR).simulate(releases)
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
        _verify_week_ending_date(week_ending_date)

    # Get the release and verify the parquet file.
    release = get_release(id)
    df_full = ARTIFACTS_DIR / "df_full.parquet"
    _verify_parquet_file(df_full)

    # Simulate the release and return the forecasts for the week ending date.
    forecasts = ForecastEngine(artifacts_dir=ARTIFACTS_DIR).simulate([release])
    weekly_injections = pd.DataFrame(forecasts["weekly_injections"])
    if weekly_injections.empty:
        return weekly_injections
    if week_ending_date is None:
        return weekly_injections
    else:
        mask = weekly_injections["Week Ending Date"].astype(str) == week_ending_date.strip()
        return weekly_injections.loc[mask]


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
    if mrelg_id and rid is not None:
        try:
            known_vols = get_known_vols_from_sqlite(int(rid))
        except Exception:
            known_vols = []

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
        # Prefer observed actuals when available; otherwise use expected.
        fw_vol = float(known_vols[0]) if has_nonzero_known else expected_fw_vol
    else:
        fw_vol = expected_fw_vol

    return {
        "name": artist or title or "Unknown",
        "artist": artist or "",
        "title": title or "",
        "label": row["LABEL_NAME"],
        "date": date_str,
        "genre": row["GENRE"],
        "cluster": int(row["CLUSTER"] or 0),
        "fw_vol": fw_vol,
        "scenario": row["SCENARIO"],
        "known_vols": known_vols,
        "fy_vol": float(row["FY_VOL"] or 0),
        "avg_historical_w1_product_ratio": float(row["AVG_HISTORICAL_W1_PRODUCT_RATIO"] or 0),
        "product_ratio_coefficient": float(row["PRODUCT_RATIO_COEFFICIENT"] or 0),
    }


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
        raise ValueError("known_vols must be a list.")
    for i, x in enumerate(known_vols):
        if not _is_real_number(x):
            raise ValueError(f"known_vols[{i}] must be a number.")
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            raise ValueError(f"known_vols[{i}] must be a finite number.")

    # Verify fw_vol is a positive number when known_vols is empty.
    fw_vol = float(data["fw_vol"])
    if len(known_vols) == 0 and fw_vol <= 0:
        raise ValueError("fw_vol must be positive when known_vols is empty.")

    # Verify genre is in the distribution.
    genre = data["genre"]
    if genre not in DISTRIBUTIONS["Genre"]:
        raise ValueError(f"Genre {genre!r} not found in distribution.")

    # Verify label is in the distribution.
    label_name = data["label_name"]
    if label_name not in DISTRIBUTIONS["Label"]:
        raise ValueError(f"Label {label_name!r} not found in distribution.")

    try:
        datetime.strptime(data["release_date"].strip(), "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"release_date {data['release_date']!r} is not a valid YYYY-MM-DD date."
        ) from e

    scenario = data["scenario"].strip()
    if scenario not in _ALLOWED_SCENARIOS:
        raise ValueError(
            f"scenario must be one of {sorted(_ALLOWED_SCENARIOS)}; got {scenario!r}."
        )


def _verify_id(id: int) -> None:
    """Verifies that the id is a positive integer and is not None."""
    if id is None:
        raise ValueError("id is required.")
    if not _is_integral(id) or int(id) < 0:
        raise ValueError("id must be a positive integer.")


def _verify_week_ending_date(week_ending_date: str) -> None:
    """Verifies that the week ending date is a valid YYYY-MM-DD date."""
    try:
        datetime.strptime(week_ending_date.strip(), "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"week_ending_date {week_ending_date!r} is not a valid YYYY-MM-DD date."
        ) from e


def _verify_parquet_file(parquet_file: Path) -> None:
    """Verifies that the parquet file exists."""
    if not parquet_file.is_file():
        raise FileNotFoundError(
            f"Forecast artifacts not found at {parquet_file}. "
            "Run `python train_model.py` (or `model/train_marketshare_artifacts.py`) to generate them."
        )
    return parquet_file
