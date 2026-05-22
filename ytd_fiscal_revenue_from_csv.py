"""
Read weekly proxy fiscal revenue by distributor label from CSV.

Source file (canonical S3 path):
  s3://parquetgarage/model/data/ytd_fiscal_revenue_by_label.csv

Refreshed incrementally by train_model._update_ytd_fiscal_revenue_by_label().
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent / "model" / "data" / "ytd_fiscal_revenue_by_label.csv"
)

_DEDUPE_COLS = (
    "week_end_date",
    "level_1_distributor",
    "level_2_distributor",
    "level_3_distributor",
)


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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Snowflake/CSV column variants to the canonical lowercase names."""
    col_map = {str(c).strip().lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for want in _DEDUPE_COLS + ("proxy_revenue",):
        if want in col_map and col_map[want] != want:
            rename[col_map[want]] = want
    out = df.rename(columns=rename)
    missing = [c for c in _DEDUPE_COLS + ("proxy_revenue",) if c not in out.columns]
    if missing:
        raise KeyError(
            f"ytd_fiscal_revenue_by_label.csv missing columns {missing}; got {list(df.columns)}"
        )
    out["week_end_date"] = pd.to_datetime(out["week_end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["proxy_revenue"] = pd.to_numeric(out["proxy_revenue"], errors="coerce")
    for col in ("level_1_distributor", "level_2_distributor", "level_3_distributor"):
        out[col] = out[col].astype(str).str.strip()
    out = out.dropna(subset=["week_end_date", "proxy_revenue"])
    out = out.drop_duplicates(subset=list(_DEDUPE_COLS), keep="last")
    return out.sort_values(list(_DEDUPE_COLS)).reset_index(drop=True)


def load_ytd_fiscal_revenue_df(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"ytd_fiscal_revenue_by_label.csv not found at {csv_path}")
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


def query_ytd_fiscal_revenue(
    *,
    week_end_date: Optional[str] = None,
    level_1_distributor: Optional[str] = None,
    level_2_distributor: Optional[str] = None,
    level_3_distributor: Optional[str] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[dict[str, Any]]:
    """
    Return rows from the CSV with optional filters (case-insensitive string match).
    """
    df = load_ytd_fiscal_revenue_df(csv_path)
    if week_end_date:
        wk = str(week_end_date).strip()[:10]
        df = df.loc[df["week_end_date"] == wk]
    if level_1_distributor:
        v = str(level_1_distributor).strip().casefold()
        df = df.loc[df["level_1_distributor"].str.casefold() == v]
    if level_2_distributor:
        v = str(level_2_distributor).strip().casefold()
        df = df.loc[df["level_2_distributor"].str.casefold() == v]
    if level_3_distributor:
        v = str(level_3_distributor).strip().casefold()
        df = df.loc[df["level_3_distributor"].str.casefold() == v]
    return df.to_dict(orient="records")
