"""
Compute marketshare actuals (per-week and YTD) directly from current_data_all.csv.

Single source of truth for what the API exposes as "actuals":
  - /v1/marketshare/actuals          → ytd_actuals_for_year()
  - /v1/marketshare/weekly stitching → weekly_actuals_for_year()

Replaces the legacy SQLite chain (MARKETSHARE_YTD / MARKETSHARE_WEEKLY tables
populated by sqlite_handler.update_sqlite_main). The legacy chain refreshed
on a different cadence than the CSVs, which let the YTD endpoint drift behind
alist data. By reading current_data_all.csv directly:

  * The same DataFrame the model trains on is what the API serves.
  * The Luminate chart-year boundary (the YEAR column) is honored — no need
    to infer year from week_ending_date string prefix.
  * No drift possible between actuals and the CSV the next refresh produces.

Caching: the CSV is small, so the parse cost is under 50ms cold. We still
cache parsed output keyed by (file mtime, year) so hot requests are O(1).
reload_artifacts() in model_handler clears this cache after every refresh.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model.marketshare_labels import (
    CURRENT_DATA_ALL_CSV,
    MARKET_ANCHOR_LABEL,
    TARGET_LABELS,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent / "model" / "data" / CURRENT_DATA_ALL_CSV
)


@dataclass(frozen=True)
class _CacheKey:
    mtime_ns: int
    size: int
    year: Optional[int]


_cache_lock = threading.Lock()
_weekly_cache: dict[_CacheKey, pd.DataFrame] = {}
_ytd_cache: dict[_CacheKey, pd.DataFrame] = {}


def clear_cache() -> None:
    """Drop memoized DataFrames. Called from reload_artifacts()."""
    with _cache_lock:
        _weekly_cache.clear()
        _ytd_cache.clear()


def _file_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _normalize_year_column(current: pd.DataFrame) -> pd.DataFrame:
    """Normalize Year/YEAR/year column to nullable Int64 ``Year``."""
    for cand in ("YEAR", "Year", "year"):
        if cand in current.columns:
            out = current.rename(columns={cand: "Year"}) if cand != "Year" else current.copy()
            out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
            return out
    raise KeyError(
        f"{CURRENT_DATA_ALL_CSV} has no Year/YEAR column — cannot derive Luminate chart-year."
    )


def _load_current_data(csv_path: Path) -> pd.DataFrame:
    """
    Load + filter current_data_all.csv to TARGET_LABELS, RELEASE_AGE='Current',
    US rows when present, with usable share + volume. Returns a DataFrame with
    normalized columns ready for aggregation.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"{CURRENT_DATA_ALL_CSV} not found at {csv_path}")
    df = pd.read_csv(csv_path)
    df = _normalize_year_column(df)

    if "COUNTRY_CODE" in df.columns:
        df = df.loc[df["COUNTRY_CODE"].astype(str).str.upper() == "US"].copy()

    # The training pipeline only uses RELEASE_AGE=Current rows. Mirror that
    # so the API serves the same view of the world.
    if "RELEASE_AGE" in df.columns:
        ra = df["RELEASE_AGE"].astype(str).str.strip().str.casefold()
        df = df.loc[ra == "current"].copy()

    df = df.loc[df["LABEL_NAME"].isin(TARGET_LABELS)].copy()

    df["ALBUM_EQUIVALENT"] = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    df["ALBUM_EQUIVALENT_SHARE"] = pd.to_numeric(df["ALBUM_EQUIVALENT_SHARE"], errors="coerce")

    # Some Snowflake exports return share as 0–1, others as 0–100. Normalize
    # to percent points (0–100) using the median as the heuristic. The
    # legacy recompute_ytd_share_from_current_data uses the inverse rule
    # (divide by 100 if median > 1) because its working units were 0–1
    # fractions; we keep percent points to match the API's existing schema.
    share_med = df["ALBUM_EQUIVALENT_SHARE"].dropna().median()
    if pd.notna(share_med) and share_med <= 1.0:
        df["ALBUM_EQUIVALENT_SHARE"] = df["ALBUM_EQUIVALENT_SHARE"] * 100.0

    df = df.loc[
        df["Year"].notna()
        & df["ALBUM_EQUIVALENT"].notna()
        & df["ALBUM_EQUIVALENT_SHARE"].notna()
        & (df["ALBUM_EQUIVALENT_SHARE"] > 0)
    ].copy()

    if df.empty:
        return df

    df["WEEK_ENDING_DATE"] = pd.to_datetime(df["WEEK_ENDING_DATE"], errors="coerce")
    df = df.loc[df["WEEK_ENDING_DATE"].notna()].copy()

    # Drop Owner×Week dups (keeps last) — same as training load_weekly_labels.
    dup_n = int(df.duplicated(subset=["LABEL_NAME", "WEEK_ENDING_DATE"]).sum())
    if dup_n:
        logger.warning(
            "%s: dropping %d duplicate LABEL_NAME×WEEK rows (keeps last)",
            csv_path.name,
            dup_n,
        )
        df = df.drop_duplicates(subset=["LABEL_NAME", "WEEK_ENDING_DATE"], keep="last")

    # Market total inverted from MARKET_ANCHOR_LABEL only (matches training).
    anchor = df.loc[df["LABEL_NAME"] == MARKET_ANCHOR_LABEL].copy()
    anchor["__total_market_est"] = np.where(
        anchor["ALBUM_EQUIVALENT_SHARE"] > 0,
        (anchor["ALBUM_EQUIVALENT"] / anchor["ALBUM_EQUIVALENT_SHARE"]) * 100.0,
        np.nan,
    )
    market_ref = (
        anchor[["WEEK_ENDING_DATE", "__total_market_est"]]
        .drop_duplicates(subset=["WEEK_ENDING_DATE"], keep="last")
    )
    df = df.drop(columns=["__total_market_est"], errors="ignore").merge(
        market_ref, on="WEEK_ENDING_DATE", how="left"
    )

    df["Year"] = df["Year"].astype(int)
    return df


def _resolve_year(df: pd.DataFrame, year: Optional[int]) -> int:
    """Pick the requested chart-year, defaulting to the latest in the data."""
    if year is not None:
        return int(year)
    if df.empty:
        # Fall back to calendar year if the file is empty for some reason —
        # caller will get an empty DataFrame either way.
        return pd.Timestamp.today().year
    return int(df["Year"].max())


def _aggregate_for_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Per-week, per-Owner aggregates for one chart-year. Returns columns:
      Week Ending Date (datetime), Owner, Total_Market_AE_Volume,
      Active_Share (per-week %, label vs market for that week),
      Cum_Numerator, Cum_Denominator,
      Unified_YTD_Share (cumulative %, percent points).
    """
    yr = df.loc[df["Year"] == year].copy()
    if yr.empty:
        return pd.DataFrame(
            columns=[
                "Week Ending Date",
                "Owner",
                "Total_Market_AE_Volume",
                "Active_Share",
                "Cum_Numerator",
                "Cum_Denominator",
                "Unified_YTD_Share",
            ]
        )

    weekly_market = (
        yr.groupby("WEEK_ENDING_DATE", as_index=False)["__total_market_est"]
        .first()
        .rename(columns={"__total_market_est": "Total_Market_AE_Volume"})
    )

    label_weekly = (
        yr.groupby(["WEEK_ENDING_DATE", "LABEL_NAME"], as_index=False)["ALBUM_EQUIVALENT"]
        .sum()
        .rename(columns={"LABEL_NAME": "Owner", "ALBUM_EQUIVALENT": "Owner_AE_Volume"})
    )

    out = label_weekly.merge(weekly_market, on="WEEK_ENDING_DATE", how="inner")

    # Per-week active share (label volume / market volume for that week).
    market = out["Total_Market_AE_Volume"].replace(0, np.nan)
    out["Active_Share"] = (out["Owner_AE_Volume"] / market) * 100.0

    # YTD cumulatives per Owner. ORDER MATTERS — sort by date BEFORE cumsum,
    # otherwise YTD is computed on whatever pandas merge order returned.
    out = out.sort_values(["Owner", "WEEK_ENDING_DATE"]).reset_index(drop=True)
    out["Cum_Numerator"] = out.groupby("Owner", sort=False)["Owner_AE_Volume"].cumsum()
    out["Cum_Denominator"] = out.groupby("Owner", sort=False)["Total_Market_AE_Volume"].cumsum()
    den = out["Cum_Denominator"].replace(0, np.nan)
    out["Unified_YTD_Share"] = (out["Cum_Numerator"] / den) * 100.0

    out = out.rename(columns={"WEEK_ENDING_DATE": "Week Ending Date"})
    out = out.drop(columns=["Owner_AE_Volume"])
    out["Active_Share"] = out["Active_Share"].fillna(0.0)
    out["Unified_YTD_Share"] = out["Unified_YTD_Share"].fillna(0.0)
    return out


def _key(csv_path: Path, year: Optional[int]) -> _CacheKey:
    mtime_ns, size = _file_signature(csv_path)
    return _CacheKey(mtime_ns=mtime_ns, size=size, year=year)


def weekly_actuals_for_year(
    forecast_year: Optional[int] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """
    Per-week-per-Owner observed marketshare for the given Luminate chart-year
    (or the latest year present in the CSV if not specified).

    Shape mirrors model_handler._sqlite_weekly_marketshare_observed:
      Week Ending Date (datetime64), Owner (str),
      Total_Market_AE_Volume (float), Active_Share (float, percent points).

    Used as the "observed" overlay when stitching engine forecasts.
    """
    key = _key(csv_path, forecast_year)
    with _cache_lock:
        cached = _weekly_cache.get(key)
        if cached is not None:
            return cached.copy()

    raw = _load_current_data(csv_path)
    yr = _resolve_year(raw, forecast_year)
    agg = _aggregate_for_year(raw, yr)
    out = agg[["Week Ending Date", "Owner", "Total_Market_AE_Volume", "Active_Share"]].copy()
    with _cache_lock:
        _weekly_cache[key] = out
    return out.copy()


def ytd_actuals_for_year(
    forecast_year: Optional[int] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> pd.DataFrame:
    """
    YTD timeline for /v1/marketshare/actuals. Returns the unified_ytd schema
    with Data_Type='Actual' and YTD bands collapsed onto Unified_YTD_Share
    (no uncertainty for observed weeks — bands only widen on forecast rows
    in get_marketshare_forecasts, never here).
    """
    key = _key(csv_path, forecast_year)
    with _cache_lock:
        cached = _ytd_cache.get(key)
        if cached is not None:
            return cached.copy()

    raw = _load_current_data(csv_path)
    yr = _resolve_year(raw, forecast_year)
    agg = _aggregate_for_year(raw, yr)
    if agg.empty:
        empty = pd.DataFrame(
            columns=[
                "Week Ending Date",
                "Owner",
                "Total_Market_AE_Volume",
                "Active_Share",
                "Data_Type",
                "Unified_YTD_Share",
                "YTD_Share_Upper",
                "YTD_Share_Lower",
            ]
        )
        with _cache_lock:
            _ytd_cache[key] = empty
        return empty.copy()

    out = pd.DataFrame(
        {
            "Week Ending Date": agg["Week Ending Date"].dt.strftime("%Y-%m-%d"),
            "Owner": agg["Owner"].astype(str),
            "Total_Market_AE_Volume": agg["Total_Market_AE_Volume"].astype(float).fillna(0.0),
            "Active_Share": agg["Active_Share"].astype(float).fillna(0.0),
            "Data_Type": "Actual",
            "Unified_YTD_Share": agg["Unified_YTD_Share"].astype(float).fillna(0.0),
        }
    )
    out["YTD_Share_Upper"] = out["Unified_YTD_Share"]
    out["YTD_Share_Lower"] = out["Unified_YTD_Share"]
    with _cache_lock:
        _ytd_cache[key] = out
    return out.copy()


def latest_actual_week(
    forecast_year: Optional[int] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> Optional[pd.Timestamp]:
    """Most recent week_ending_date with observed actuals in the chart-year."""
    df = weekly_actuals_for_year(forecast_year, csv_path)
    if df.empty:
        return None
    return df["Week Ending Date"].max()
