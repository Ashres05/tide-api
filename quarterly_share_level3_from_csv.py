"""
Read fiscal-quarter level-3 profit-center share metrics from CSV.

Source file (canonical S3 path):
  s3://parquetgarage/model/data/quarterly_share_level3.csv

Refreshed from Snowflake when ``train_model._update_quarterly_share_level3()``
detects ``max(P_DAY)`` in the source is newer than the local CSV anchor.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from quarterly_share_from_csv import _COLD_START_P_DAY, parse_p_day_from_row

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent / "model" / "data" / "quarterly_share_level3.csv"
)

_CANONICAL_COLUMNS = (
    "FISCAL_YEAR",
    "FISCAL_QUARTER",
    "PROFIT_CENTER_LABEL",
    "LABEL_STREAMS",
    "TOTAL_UNIVERSE_STREAMS",
    "LABEL_SHARE",
    "SHARE_CHANGE_BPS",
    "LABEL_STREAM_GROWTH_YOY",
    "MARKET_GROWTH_YOY",
)

_DEDUPE_COLS = ("FISCAL_YEAR", "FISCAL_QUARTER", "PROFIT_CENTER_LABEL")

_PDAY_COLS = ("FISCAL_YEAR", "FISCAL_QUARTER")


@dataclass(frozen=True)
class _CacheKey:
    mtime_ns: int
    size: int


_cache_lock = threading.Lock()
_frame_cache: dict[_CacheKey, pd.DataFrame] = {}


def clear_cache() -> None:
    with _cache_lock:
        _frame_cache.clear()


def _file_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def max_p_day_from_level3_csv(csv_path: Path = DEFAULT_CSV_PATH) -> date:
    """Latest quarter-end implied by rows on disk (max over parsed ``FISCAL_QUARTER`` ends)."""
    if not csv_path.is_file():
        logger.info("%s not found; cold start p_day=%s", csv_path.name, _COLD_START_P_DAY)
        return _COLD_START_P_DAY
    try:
        df = pd.read_csv(csv_path, usecols=list(_PDAY_COLS))
    except (ValueError, KeyError) as e:
        logger.warning("%s missing fiscal columns (%s); cold start p_day", csv_path.name, e)
        return _COLD_START_P_DAY
    if df.empty:
        return _COLD_START_P_DAY
    parsed: list[date] = []
    for row in df.itertuples(index=False):
        p = parse_p_day_from_row(row[0], row[1])
        if p is not None:
            parsed.append(p)
    if not parsed:
        return _COLD_START_P_DAY
    return max(parsed)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for want in _CANONICAL_COLUMNS:
        key = want.lower()
        if key in col_map and col_map[key] != want:
            rename[col_map[key]] = want
    out = df.rename(columns=rename)
    missing = [c for c in _CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(
            f"quarterly_share_level3.csv missing columns {missing}; got {list(df.columns)}"
        )
    out = out[list(_CANONICAL_COLUMNS)].copy()
    out["FISCAL_YEAR"] = pd.to_numeric(out["FISCAL_YEAR"], errors="coerce").astype("Int64")
    out["FISCAL_QUARTER"] = out["FISCAL_QUARTER"].astype(str).str.strip()
    out["PROFIT_CENTER_LABEL"] = out["PROFIT_CENTER_LABEL"].astype(str).str.strip()
    for col in ("LABEL_STREAMS", "TOTAL_UNIVERSE_STREAMS"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in (
        "LABEL_SHARE",
        "SHARE_CHANGE_BPS",
        "LABEL_STREAM_GROWTH_YOY",
        "MARKET_GROWTH_YOY",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(
        subset=[
            "FISCAL_YEAR",
            "FISCAL_QUARTER",
            "PROFIT_CENTER_LABEL",
            "LABEL_STREAMS",
            "TOTAL_UNIVERSE_STREAMS",
        ]
    )
    out = out.drop_duplicates(subset=list(_DEDUPE_COLS), keep="last")
    out["_p_day"] = [
        parse_p_day_from_row(fy, fq) for fy, fq in zip(out["FISCAL_YEAR"], out["FISCAL_QUARTER"])
    ]
    out = out.sort_values(
        ["PROFIT_CENTER_LABEL", "_p_day", "FISCAL_YEAR", "FISCAL_QUARTER"],
        na_position="first",
    )
    out = out.drop(columns=["_p_day"]).reset_index(drop=True)
    return out


def normalize_quarterly_share_level3_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_columns(df)


def load_quarterly_share_level3_df(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"quarterly_share_level3.csv not found at {csv_path}")
    key = _CacheKey(*_file_signature(csv_path))
    with _cache_lock:
        cached = _frame_cache.get(key)
        if cached is not None:
            return cached.copy()
    df = _normalize_columns(pd.read_csv(csv_path))
    with _cache_lock:
        _frame_cache.clear()
        _frame_cache[key] = df
    return df.copy()


def query_quarterly_share_level3(
    *,
    fiscal_year: Optional[int] = None,
    profit_center_label: Optional[str] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[dict[str, Any]]:
    df = load_quarterly_share_level3_df(csv_path)
    if fiscal_year is not None:
        df = df.loc[df["FISCAL_YEAR"] == int(fiscal_year)]
    if profit_center_label:
        v = str(profit_center_label).strip().casefold()
        df = df.loc[df["PROFIT_CENTER_LABEL"].str.casefold() == v]
    records = df.to_dict(orient="records")
    for row in records:
        fy = row.get("FISCAL_YEAR")
        if pd.notna(fy):
            row["FISCAL_YEAR"] = int(fy)
        for col in ("LABEL_STREAMS", "TOTAL_UNIVERSE_STREAMS"):
            v = row.get(col)
            if pd.notna(v):
                row[col] = int(v)
    return records
