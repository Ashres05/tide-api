"""
Read AMG level-3 release catalog from CSV (immutable ``first_sale_date`` per row).

Source file (canonical S3 path):
  s3://parquetgarage/model/data/releases_by_q_amg_labels.csv

Refreshed incrementally by ``train_model._update_releases_by_q_amg_labels()`` when
new rows appear in Snowflake with ``first_sale_date`` greater than the CSV max.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from quarterly_share_from_csv import fiscal_quarter_date_range, level_3_distributor_for_profit_center

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent / "model" / "data" / "releases_by_q_amg_labels.csv"
)

_COLD_START_MIN_FIRST_SALE = "2023-08-31"

_CANONICAL_COLUMNS = (
    "MRELG_ID",
    "DISPLAY_ARTIST",
    "TITLE",
    "LEVEL_3_DISTRIBUTOR",
    "FIRST_SALE_DATE",
)

_DEDUPE_COLS = ("MRELG_ID",)


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


def _resolve_level_3_distributor(
    *,
    level_3_distributor: Optional[str] = None,
    profit_center_label: Optional[str] = None,
) -> Optional[str]:
    if level_3_distributor:
        return str(level_3_distributor).strip()
    if profit_center_label:
        return level_3_distributor_for_profit_center(profit_center_label)
    return None


def max_first_sale_date_from_csv(csv_path: Path = DEFAULT_CSV_PATH) -> str:
    if not csv_path.is_file():
        logger.info(
            "%s not found; cold start min first_sale_date=%s",
            csv_path.name,
            _COLD_START_MIN_FIRST_SALE,
        )
        return _COLD_START_MIN_FIRST_SALE
    try:
        df = pd.read_csv(csv_path, usecols=["FIRST_SALE_DATE"])
    except (ValueError, KeyError) as e:
        logger.warning("%s missing FIRST_SALE_DATE (%s); cold start", csv_path.name, e)
        return _COLD_START_MIN_FIRST_SALE
    if df.empty:
        return _COLD_START_MIN_FIRST_SALE
    mx = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce").max()
    if pd.isna(mx):
        return _COLD_START_MIN_FIRST_SALE
    return mx.strftime("%Y-%m-%d")


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
            f"releases_by_q_amg_labels.csv missing columns {missing}; got {list(df.columns)}"
        )
    out = out[list(_CANONICAL_COLUMNS)].copy()
    out["MRELG_ID"] = out["MRELG_ID"].astype(str).str.strip()
    out["DISPLAY_ARTIST"] = out["DISPLAY_ARTIST"].astype(str).str.strip()
    out["TITLE"] = out["TITLE"].astype(str).str.strip()
    out["LEVEL_3_DISTRIBUTOR"] = out["LEVEL_3_DISTRIBUTOR"].astype(str).str.strip()
    out["FIRST_SALE_DATE"] = pd.to_datetime(out["FIRST_SALE_DATE"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    out = out.dropna(subset=["MRELG_ID", "FIRST_SALE_DATE"])
    out = out.drop_duplicates(subset=list(_DEDUPE_COLS), keep="last")
    return out.sort_values(["DISPLAY_ARTIST", "TITLE", "FIRST_SALE_DATE"]).reset_index(drop=True)


def normalize_releases_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_columns(df)


def load_releases_by_q_amg_labels_df(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"releases_by_q_amg_labels.csv not found at {csv_path}")
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


def _filter_frame(
    df: pd.DataFrame,
    *,
    level_3_distributor: Optional[str] = None,
    first_sale_date_from: Optional[str] = None,
    first_sale_date_to: Optional[str] = None,
) -> pd.DataFrame:
    out = df
    if level_3_distributor:
        v = str(level_3_distributor).strip().casefold()
        out = out.loc[out["LEVEL_3_DISTRIBUTOR"].str.casefold() == v]
    if first_sale_date_from:
        lo = str(first_sale_date_from).strip()[:10]
        out = out.loc[out["FIRST_SALE_DATE"] >= lo]
    if first_sale_date_to:
        hi = str(first_sale_date_to).strip()[:10]
        out = out.loc[out["FIRST_SALE_DATE"] <= hi]
    return out


def query_releases_by_q_amg_labels(
    *,
    level_3_distributor: Optional[str] = None,
    profit_center_label: Optional[str] = None,
    first_sale_date_from: Optional[str] = None,
    first_sale_date_to: Optional[str] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[dict[str, Any]]:
    label = _resolve_level_3_distributor(
        level_3_distributor=level_3_distributor,
        profit_center_label=profit_center_label,
    )
    df = load_releases_by_q_amg_labels_df(csv_path)
    return _filter_frame(
        df,
        level_3_distributor=label,
        first_sale_date_from=first_sale_date_from,
        first_sale_date_to=first_sale_date_to,
    ).to_dict(orient="records")


def query_releases_by_q_amg_labels_comparison(
    *,
    level_3_distributor: Optional[str] = None,
    profit_center_label: Optional[str] = None,
    baseline_fiscal_year: int,
    baseline_fiscal_quarter: str,
    comparison_fiscal_year: int,
    comparison_fiscal_quarter: str,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> dict[str, Any]:
    label = _resolve_level_3_distributor(
        level_3_distributor=level_3_distributor,
        profit_center_label=profit_center_label,
    )
    if not label:
        raise ValueError("level_3_distributor or profit_center_label is required")
    base_w = fiscal_quarter_date_range(baseline_fiscal_year, baseline_fiscal_quarter)
    cmp_w = fiscal_quarter_date_range(comparison_fiscal_year, comparison_fiscal_quarter)
    df = load_releases_by_q_amg_labels_df(csv_path)
    return {
        "baseline": _filter_frame(
            df,
            level_3_distributor=label,
            first_sale_date_from=base_w.start_date.isoformat(),
            first_sale_date_to=base_w.end_date.isoformat(),
        ).to_dict(orient="records"),
        "comparison": _filter_frame(
            df,
            level_3_distributor=label,
            first_sale_date_from=cmp_w.start_date.isoformat(),
            first_sale_date_to=cmp_w.end_date.isoformat(),
        ).to_dict(orient="records"),
        "baseline_window": {
            "fiscal_year": base_w.fiscal_year,
            "fiscal_quarter": base_w.fiscal_quarter,
            "start_date": base_w.start_date.isoformat(),
            "end_date": base_w.end_date.isoformat(),
        },
        "comparison_window": {
            "fiscal_year": cmp_w.fiscal_year,
            "fiscal_quarter": cmp_w.fiscal_quarter,
            "start_date": cmp_w.start_date.isoformat(),
            "end_date": cmp_w.end_date.isoformat(),
        },
        "level_3_distributor": label,
    }
