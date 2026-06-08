"""
Read fiscal-quarter AMG market share / QTD metrics from CSV.

Source file (canonical S3 path):
  s3://parquetgarage/model/data/quarterly_share_and_qtd.csv

Refreshed from Snowflake table
``bi_sandbox.aidan_ow.luminate_market_share_revenue_pre_total`` when
``train_model._update_quarterly_share_and_qtd()`` detects ``max(P_DAY)`` in
the source is newer than the local CSV anchor (parsed from ``FISCAL_QUARTER``
end dates — e.g. ``Q3 (Mar-May) - (END May 28)`` with ``FISCAL_YEAR=2026``).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent / "model" / "data" / "quarterly_share_and_qtd.csv"
)

_COLD_START_P_DAY = date(1900, 1, 1)

_CANONICAL_COLUMNS = (
    "FISCAL_YEAR",
    "FISCAL_QUARTER",
    "AMG_STREAMS",
    "TOTAL_UNIVERSE_STREAMS",
    "AMG_SHARE",
    "SHARE_GROWTH_YOY",
    "MARKET_GROWTH_YOY",
    "TOTAL_GROWTH_YOY",
)

_DEDUPE_COLS = ("FISCAL_YEAR", "FISCAL_QUARTER")

_END_DATE_RE = re.compile(r"\(END\s+([A-Za-z]+)\s+(\d+)\)", re.I)
_QUARTER_NUM_RE = re.compile(r"\bQ(\d)\b", re.I)
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_QUARTER_END_FALLBACK = {1: (11, 30), 2: (2, 28), 3: (5, 31), 4: (8, 31)}

# Share CSV ``profit_center_label`` → ``marketshare_map_icpns.level_3_distributor``
PROFIT_CENTER_TO_LEVEL3_DISTRIBUTOR: dict[str, str] = {
    "10K Holdings Project": "10K Projects",
    "300 Entertainment": "300 Entertainment",
    "Atlantic": "Atlantic Records",
    "Rhino": "Rhino",
}


@dataclass(frozen=True)
class FiscalQuarterWindow:
    fiscal_year: int
    fiscal_quarter: str
    start_date: date
    end_date: date


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


def _quarter_end_calendar_year(fiscal_year: int, month: int) -> int:
    fy = int(fiscal_year)
    if month in (9, 10, 11, 12):
        return fy - 1
    return fy


def parse_p_day_from_row(fiscal_year: Any, fiscal_quarter: Any) -> Optional[date]:
    """Parse quarter-end anchor from ``FISCAL_YEAR`` + ``FISCAL_QUARTER`` (CSV row label)."""
    try:
        fy = int(fiscal_year)
    except (TypeError, ValueError):
        return None
    fq = str(fiscal_quarter or "").strip()
    if not fq:
        return None
    m = _END_DATE_RE.search(fq)
    if m:
        mon_key = m.group(1).strip().lower()[:3]
        month = _MONTHS.get(mon_key)
        if month is None:
            return None
        try:
            day = int(m.group(2))
            return date(fy, month, day)
        except ValueError:
            return None
    qm = _QUARTER_NUM_RE.search(fq)
    if not qm:
        return None
    q = int(qm.group(1))
    fb = _QUARTER_END_FALLBACK.get(q)
    if fb is None:
        return None
    month, day = fb
    try:
        return date(fy, month, day)
    except ValueError:
        return None


def fiscal_quarter_start(fiscal_year: int, fiscal_quarter: str) -> Optional[date]:
    try:
        fy = int(fiscal_year)
    except (TypeError, ValueError):
        return None
    qm = _QUARTER_NUM_RE.search(str(fiscal_quarter or ""))
    if not qm:
        return None
    q = int(qm.group(1))
    if q == 1:
        return date(fy - 1, 9, 1)
    if q == 2:
        return date(fy - 1, 12, 1)
    if q == 3:
        return date(fy, 3, 1)
    if q == 4:
        return date(fy, 6, 1)
    return None


def fiscal_quarter_end(fiscal_year: int, fiscal_quarter: str) -> Optional[date]:
    try:
        fy = int(fiscal_year)
    except (TypeError, ValueError):
        return None
    fq = str(fiscal_quarter or "").strip()
    if not fq:
        return None
    m = _END_DATE_RE.search(fq)
    if m:
        mon_key = m.group(1).strip().lower()[:3]
        month = _MONTHS.get(mon_key)
        if month is None:
            return None
        try:
            day = int(m.group(2))
            cy = _quarter_end_calendar_year(fy, month)
            return date(cy, month, day)
        except ValueError:
            return None
    qm = _QUARTER_NUM_RE.search(fq)
    if not qm:
        return None
    q = int(qm.group(1))
    fb = _QUARTER_END_FALLBACK.get(q)
    if fb is None:
        return None
    month, day = fb
    cy = _quarter_end_calendar_year(fy, month)
    try:
        return date(cy, month, day)
    except ValueError:
        return None


def fiscal_quarter_date_range(
    fiscal_year: int,
    fiscal_quarter: str,
) -> FiscalQuarterWindow:
    start = fiscal_quarter_start(fiscal_year, fiscal_quarter)
    end = fiscal_quarter_end(fiscal_year, fiscal_quarter)
    if start is None or end is None:
        raise ValueError(
            f"Could not resolve fiscal quarter window for fiscal_year={fiscal_year!r} "
            f"fiscal_quarter={fiscal_quarter!r}"
        )
    if end < start:
        raise ValueError(f"Invalid fiscal quarter window: start={start} end={end}")
    return FiscalQuarterWindow(
        fiscal_year=int(fiscal_year),
        fiscal_quarter=str(fiscal_quarter).strip(),
        start_date=start,
        end_date=end,
    )


def level_3_distributor_for_profit_center(profit_center_label: str) -> str:
    key = str(profit_center_label or "").strip()
    if not key:
        raise ValueError("profit_center_label is required")
    return PROFIT_CENTER_TO_LEVEL3_DISTRIBUTOR.get(key, key)


def max_p_day_from_csv(csv_path: Path = DEFAULT_CSV_PATH) -> date:
    if not csv_path.is_file():
        logger.info("%s not found; cold start p_day=%s", csv_path.name, _COLD_START_P_DAY)
        return _COLD_START_P_DAY
    try:
        df = pd.read_csv(csv_path, usecols=list(_DEDUPE_COLS))
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
            f"quarterly_share_and_qtd.csv missing columns {missing}; got {list(df.columns)}"
        )
    out = out[list(_CANONICAL_COLUMNS)].copy()
    out["FISCAL_YEAR"] = pd.to_numeric(out["FISCAL_YEAR"], errors="coerce").astype("Int64")
    out["FISCAL_QUARTER"] = out["FISCAL_QUARTER"].astype(str).str.strip()
    for col in ("AMG_STREAMS", "TOTAL_UNIVERSE_STREAMS"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("AMG_SHARE", "SHARE_GROWTH_YOY", "MARKET_GROWTH_YOY", "TOTAL_GROWTH_YOY"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["FISCAL_YEAR", "FISCAL_QUARTER", "AMG_STREAMS", "TOTAL_UNIVERSE_STREAMS"])
    out = out.drop_duplicates(subset=list(_DEDUPE_COLS), keep="last")
    out["_p_day"] = [
        parse_p_day_from_row(fy, fq) for fy, fq in zip(out["FISCAL_YEAR"], out["FISCAL_QUARTER"])
    ]
    out = out.sort_values(["_p_day", "FISCAL_YEAR", "FISCAL_QUARTER"], na_position="first")
    out = out.drop(columns=["_p_day"]).reset_index(drop=True)
    return out


def normalize_quarterly_share_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_columns(df)


def load_quarterly_share_and_qtd_df(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"quarterly_share_and_qtd.csv not found at {csv_path}")
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


def query_quarterly_share_and_qtd(
    *,
    fiscal_year: Optional[int] = None,
    csv_path: Path = DEFAULT_CSV_PATH,
) -> list[dict[str, Any]]:
    df = load_quarterly_share_and_qtd_df(csv_path)
    if fiscal_year is not None:
        df = df.loc[df["FISCAL_YEAR"] == int(fiscal_year)]
    records = df.to_dict(orient="records")
    for row in records:
        fy = row.get("FISCAL_YEAR")
        if pd.notna(fy):
            row["FISCAL_YEAR"] = int(fy)
        for col in ("AMG_STREAMS", "TOTAL_UNIVERSE_STREAMS"):
            v = row.get(col)
            if pd.notna(v):
                row[col] = int(v)
    return records


def coerce_snowflake_max_p_day(value: Any) -> Optional[date]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()
