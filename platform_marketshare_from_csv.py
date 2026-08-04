"""
Read Apple / Spotify label marketshare metrics from CSV.

Canonical S3 paths under ``s3://parquetgarage/model/data/``:
  - labels_apple_marketshare.csv
  - labels_spotify_marketshare.csv
  - amg_full_apple_marketshare.csv
  - amg_full_spotify_marketshare.csv

Served read-only for the frontend (same pattern as monthly share CSVs).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "model" / "data"

_APPLE_COLUMNS = (
    "PERIOD_START_DATE",
    "LABEL_BUCKET",
    "LABEL_APPLE_STREAMS",
    "TOTAL_US_MARKET_STREAMS",
    "US_APPLE_MARKETSHARE",
    "ESTIMATED_APPLE_REVENUE",
    "SHARE_CHANGE_BPS",
    "SHARE_GROWTH_YOY",
    "MARKET_GROWTH_YOY",
    "TOTAL_GROWTH_YOY",
)

_SPOTIFY_COLUMNS = (
    "PERIOD_START_DATE",
    "LABEL_BUCKET",
    "LABEL_SPOTIFY_STREAMS",
    "TOTAL_US_MARKET_STREAMS",
    "US_SPOTIFY_MARKETSHARE",
    "ESTIMATED_SPOTIFY_REVENUE",
    "SHARE_CHANGE_BPS",
    "SHARE_GROWTH_YOY",
    "MARKET_GROWTH_YOY",
    "TOTAL_GROWTH_YOY",
)

_INT_COLS_APPLE = ("LABEL_APPLE_STREAMS", "TOTAL_US_MARKET_STREAMS")
_INT_COLS_SPOTIFY = ("LABEL_SPOTIFY_STREAMS", "TOTAL_US_MARKET_STREAMS")
_FLOAT_COLS_APPLE = (
    "US_APPLE_MARKETSHARE",
    "ESTIMATED_APPLE_REVENUE",
    "SHARE_CHANGE_BPS",
    "SHARE_GROWTH_YOY",
    "MARKET_GROWTH_YOY",
    "TOTAL_GROWTH_YOY",
)
_FLOAT_COLS_SPOTIFY = (
    "US_SPOTIFY_MARKETSHARE",
    "ESTIMATED_SPOTIFY_REVENUE",
    "SHARE_CHANGE_BPS",
    "SHARE_GROWTH_YOY",
    "MARKET_GROWTH_YOY",
    "TOTAL_GROWTH_YOY",
)


@dataclass(frozen=True)
class _Dataset:
    name: str
    csv_path: Path
    columns: tuple[str, ...]
    int_cols: tuple[str, ...]
    float_cols: tuple[str, ...]


DATASETS: dict[str, _Dataset] = {
    "labels_apple_marketshare": _Dataset(
        name="labels_apple_marketshare",
        csv_path=_DATA_DIR / "labels_apple_marketshare.csv",
        columns=_APPLE_COLUMNS,
        int_cols=_INT_COLS_APPLE,
        float_cols=_FLOAT_COLS_APPLE,
    ),
    "labels_spotify_marketshare": _Dataset(
        name="labels_spotify_marketshare",
        csv_path=_DATA_DIR / "labels_spotify_marketshare.csv",
        columns=_SPOTIFY_COLUMNS,
        int_cols=_INT_COLS_SPOTIFY,
        float_cols=_FLOAT_COLS_SPOTIFY,
    ),
    "amg_full_apple_marketshare": _Dataset(
        name="amg_full_apple_marketshare",
        csv_path=_DATA_DIR / "amg_full_apple_marketshare.csv",
        columns=_APPLE_COLUMNS,
        int_cols=_INT_COLS_APPLE,
        float_cols=_FLOAT_COLS_APPLE,
    ),
    "amg_full_spotify_marketshare": _Dataset(
        name="amg_full_spotify_marketshare",
        csv_path=_DATA_DIR / "amg_full_spotify_marketshare.csv",
        columns=_SPOTIFY_COLUMNS,
        int_cols=_INT_COLS_SPOTIFY,
        float_cols=_FLOAT_COLS_SPOTIFY,
    ),
}


@dataclass(frozen=True)
class _CacheKey:
    dataset: str
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


def _strip_bom_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: str(c).lstrip("\ufeff").strip() for c in df.columns}
    return df.rename(columns=rename)


def _normalize_columns(df: pd.DataFrame, dataset: _Dataset) -> pd.DataFrame:
    out = _strip_bom_columns(df)
    col_map = {str(c).strip().lower(): c for c in out.columns}
    rename: dict[str, str] = {}
    for want in dataset.columns:
        key = want.lower()
        if key in col_map and col_map[key] != want:
            rename[col_map[key]] = want
    out = out.rename(columns=rename)
    missing = [c for c in dataset.columns if c not in out.columns]
    if missing:
        raise KeyError(
            f"{dataset.name}.csv missing columns {missing}; got {list(df.columns)}"
        )
    out = out[list(dataset.columns)].copy()
    out["PERIOD_START_DATE"] = pd.to_datetime(
        out["PERIOD_START_DATE"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    out["LABEL_BUCKET"] = out["LABEL_BUCKET"].astype(str).str.strip()
    for col in dataset.int_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in dataset.float_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["PERIOD_START_DATE", "LABEL_BUCKET", *dataset.int_cols])
    out = out.drop_duplicates(
        subset=["PERIOD_START_DATE", "LABEL_BUCKET"], keep="last"
    )
    return out.sort_values(["PERIOD_START_DATE", "LABEL_BUCKET"]).reset_index(drop=True)


def load_platform_marketshare_df(dataset_key: str) -> pd.DataFrame:
    if dataset_key not in DATASETS:
        raise KeyError(f"Unknown platform marketshare dataset: {dataset_key!r}")
    dataset = DATASETS[dataset_key]
    csv_path = dataset.csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"{dataset.name}.csv not found at {csv_path}")
    key = _CacheKey(dataset_key, *_file_signature(csv_path))
    with _cache_lock:
        cached = _frame_cache.get(key)
        if cached is not None:
            return cached.copy()
    df = _normalize_columns(pd.read_csv(csv_path), dataset)
    with _cache_lock:
        # Drop stale entries for this dataset only.
        stale = [k for k in _frame_cache if k.dataset == dataset_key]
        for k in stale:
            del _frame_cache[k]
        _frame_cache[key] = df
    return df.copy()


def query_platform_marketshare(
    dataset_key: str,
    *,
    period_start_date: Optional[str] = None,
    year: Optional[int] = None,
    label_bucket: Optional[str] = None,
) -> list[dict[str, Any]]:
    dataset = DATASETS[dataset_key]
    df = load_platform_marketshare_df(dataset_key)
    if period_start_date:
        wk = str(period_start_date).strip()[:10]
        df = df.loc[df["PERIOD_START_DATE"] == wk]
    if year is not None:
        y = int(year)
        df = df.loc[df["PERIOD_START_DATE"].str.slice(0, 4) == str(y)]
    if label_bucket:
        v = str(label_bucket).strip().casefold()
        df = df.loc[df["LABEL_BUCKET"].str.casefold() == v]
    records = df.to_dict(orient="records")
    for row in records:
        for col in dataset.int_cols:
            v = row.get(col)
            if pd.notna(v):
                row[col] = int(v)
        for col in dataset.float_cols:
            v = row.get(col)
            if v is not None and pd.isna(v):
                row[col] = None
    return records
