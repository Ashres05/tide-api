from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "MRELG_ID",
    "TITLE",
    "GENRES",
    "RELEASE_DATE",
    "FIRST_SALE_DATE",
    "WEEK_END_DATE",
    "WEEKS_SINCE_RELEASE",
    "WORLDWIDE_STREAMS",
    "LAG1W_STREAMS",       
    "LAG4W_AVG_STREAMS",   
    "LAG12W_AVG_STREAMS",  
]

# Training / decode contract for catalog decay head (see ``write_artifacts`` bundle keys).
CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1 = "retained_multiplier_lag1"
CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52 = "rel_residual_baseline52"
CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52 = "hybrid_spike_gate_baseline52"

BASELINE52_WINDOW = 52
BASELINE52_MIN_PERIODS = 8
BASELINE52_EPS = 1e-6
REL_RESIDUAL_CLIP_LOW = -0.98
REL_RESIDUAL_CLIP_HIGH = 5.0

SPIKE_GATE_BASELINE_RATIO_THR = 1.5


def baseline52_median_from_history_stream(
    series: list[float],
    *,
    window: int = BASELINE52_WINDOW,
    min_periods: int = BASELINE52_MIN_PERIODS,
) -> float:
    """
    Median of up to the last ``window`` completed weekly stream levels.

    ``series`` should end at the most recent known week (lag1). Matches training
    ``shift(1).rolling(52).median()`` at the forecast boundary.
    """
    if not series:
        return float("nan")
    arr = np.asarray(series[-int(window) :], dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0.0)]
    if arr.size < int(min_periods):
        full = np.asarray(series, dtype=float)
        full = full[np.isfinite(full) & (full >= 0.0)]
        if full.size == 0:
            return float("nan")
        m = float(np.median(full))
        return m if math.isfinite(m) else float("nan")
    m = float(np.median(arr))
    return m if math.isfinite(m) else float("nan")


def volatility_cv_last4_from_history(series: list[float]) -> float:
    """Rolling CV (std/mean) of the last up-to-4 completed weekly levels."""
    if not series:
        return 0.0
    tail = series[-4:] if len(series) >= 4 else list(series)
    if len(tail) < 2:
        return 0.0
    a = np.asarray(tail, dtype=float)
    a = a[np.isfinite(a) & (a >= 0.0)]
    if a.size < 2:
        return 0.0
    mu = float(np.mean(a))
    if mu <= 1e-9:
        return 0.0
    s = float(np.std(a, ddof=0))
    cv = s / mu
    return cv if math.isfinite(cv) else 0.0


def hybrid_inference_denominator_and_features(
    series: list[float],
) -> tuple[float, float, float, float]:
    """
    From history ending at lag1: return
    ``(Volatility_Context, Baseline_Ratio, Spike_State, decode_denom)``.
    Spike is 1.0 iff anomaly (lag1 vs 4w/12w) OR Baseline_Ratio > ``SPIKE_GATE_BASELINE_RATIO_THR``.
    ``decode_denom`` is lag1 if spike else 52w median (fallback lag1 if baseline missing).
    """
    lag1 = float(series[-1]) if series else 0.0
    b = baseline52_median_from_history_stream(series)
    b_ok = math.isfinite(b) and b >= BASELINE52_EPS
    lag4 = float(np.mean(series[-4:])) if len(series) >= 1 else lag1
    lag12 = float(np.mean(series[-12:])) if len(series) >= 1 else lag1
    short = (lag1 / lag4) if np.isfinite(lag4) and lag4 > 0 and np.isfinite(lag1) else 1.0
    long = (lag1 / lag12) if np.isfinite(lag12) and lag12 > 0 and np.isfinite(lag1) else 1.0
    anomaly = 1 if (short > 1.5 and long > 2.0) else 0
    vol = volatility_cv_last4_from_history(series)
    br = lag1 / max(b, BASELINE52_EPS) if b_ok else 1.0
    if not math.isfinite(br):
        br = 1.0
    spike = 1.0 if (anomaly == 1 or br > SPIKE_GATE_BASELINE_RATIO_THR) else 0.0
    if spike >= 0.5:
        d = max(lag1, BASELINE52_EPS)
    else:
        d = max(b, BASELINE52_EPS) if b_ok else max(lag1, BASELINE52_EPS)
    return vol, br, spike, d


def _panel_baseline52_median_worldwide(d: pd.DataFrame) -> pd.Series:
    """Prior 52w median of WORLDWIDE_STREAMS per row (same as training residual baseline)."""
    tmp = d[["MRELG_ID", "WEEK_END_DATE", "WORLDWIDE_STREAMS"]].copy()
    tmp["__row"] = np.arange(len(tmp), dtype=np.int64)
    tmp["__m"] = tmp["MRELG_ID"].astype(str)
    tmp["__w"] = pd.to_datetime(tmp["WEEK_END_DATE"], errors="coerce")
    st = tmp.sort_values(["__m", "__w"])
    yv = pd.to_numeric(st["WORLDWIDE_STREAMS"], errors="coerce")
    base_roll = yv.groupby(st["__m"], observed=False).transform(
        lambda s: s.shift(1).rolling(BASELINE52_WINDOW, min_periods=BASELINE52_MIN_PERIODS).median()
    )
    st = st.assign(_b52=base_roll.values)
    st = st.sort_values("__row")
    return pd.Series(st["_b52"].values, index=d.index, dtype="float64")


def _panel_volatility_cv_last4_worldwide(d: pd.DataFrame) -> pd.Series:
    """CV of prior 4 completed weekly WORLDWIDE_STREAMS (shifted block), per row."""
    tmp = d[["MRELG_ID", "WEEK_END_DATE", "WORLDWIDE_STREAMS"]].copy()
    tmp["__row"] = np.arange(len(tmp), dtype=np.int64)
    tmp["__m"] = tmp["MRELG_ID"].astype(str)
    tmp["__w"] = pd.to_datetime(tmp["WEEK_END_DATE"], errors="coerce")
    st = tmp.sort_values(["__m", "__w"])
    st["_y"] = pd.to_numeric(st["WORLDWIDE_STREAMS"], errors="coerce")
    st["_y1"] = st.groupby("__m", observed=False)["_y"].shift(1)
    st["_y2"] = st.groupby("__m", observed=False)["_y"].shift(2)
    st["_y3"] = st.groupby("__m", observed=False)["_y"].shift(3)
    st["_y4"] = st.groupby("__m", observed=False)["_y"].shift(4)
    mat = st[["_y4", "_y3", "_y2", "_y1"]]
    std = mat.std(axis=1, ddof=0)
    mu = mat.mean(axis=1).clip(lower=1e-9)
    cv = (std / mu).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    st = st.assign(_vol=cv.values)
    st = st.sort_values("__row")
    return pd.Series(st["_vol"].values, index=d.index, dtype="float64")


@dataclass
class CatalogDecayArtifacts:
    model: Any
    feature_columns: list[str]
    top_genres: list[str]
    top_artists: list[str]
    as_of_date: pd.Timestamp
    catalog_min_weeks: int
    artist_history_default_log_median: float
    # ``log1p``: model was fit on ``np.log1p(WORLDWIDE_STREAMS)``; serve with ``expm1``.
    # ``none`` + ``target_is_multiplier``: model predicts retained multiplier; serve as m * lag1.
    # ``none`` + not multiplier: legacy raw weekly level target.
    target_transform: str = "none"
    target_is_multiplier: bool = True
    # ``retained_multiplier_lag1`` | ``rel_residual_baseline52`` | ``hybrid_spike_gate_baseline52``.
    catalog_decay_target: str = CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: c.upper() for c in df.columns}
    return df.rename(columns=rename)


def _apply_low_memory_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Numeric downcast
    for col in (
        "WEEKS_SINCE_RELEASE", 
        "WORLDWIDE_STREAMS",
        "LAG1W_STREAMS",
        "LAG4W_AVG_STREAMS",
        "LAG12W_AVG_STREAMS",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")
    # Category downcast for repeated strings
    for col in ("MRELG_ID", "TITLE", "DISPLAY_ARTIST", "PARSED_MAIN_GENRE"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def load_catalog_streams(
    path: Path,
    *,
    low_memory: bool = False,
    min_weeks_since_release: int | None = None,
    memory_efficient_load: bool = False,
) -> pd.DataFrame:
    required_with_artist = REQUIRED_COLUMNS + ["DISPLAY_ARTIST"]
    # Prefer scan-time predicate/column pushdown to avoid loading full parquet.
    df: pd.DataFrame
    try:
        import importlib

        ds = importlib.import_module("pyarrow.dataset")

        dataset = ds.dataset(str(path), format="parquet")
        available = set(dataset.schema.names)
        columns = [c for c in required_with_artist if c in available]
        if "DISPLAY_ARTIST" not in columns and "ARTIST" in available:
            columns.append("ARTIST")

        filt = ds.field("WORLDWIDE_STREAMS") > 0
        if min_weeks_since_release is not None:
            filt = filt & (ds.field("WEEKS_SINCE_RELEASE") >= int(min_weeks_since_release))

        table = dataset.to_table(columns=columns, filter=filt)
        # self_destruct frees Arrow buffers while building the pandas frame (lower peak RAM).
        if memory_efficient_load:
            try:
                df = table.to_pandas(types_mapper=None, self_destruct=True)
            except TypeError:
                df = table.to_pandas(types_mapper=None)
        else:
            df = table.to_pandas(types_mapper=None)
    except Exception:
        # Fallback when pyarrow dataset scan is unavailable.
        columns = REQUIRED_COLUMNS + ["DISPLAY_ARTIST", "ARTIST"]
        try:
            df = pd.read_parquet(path, columns=columns)
        except Exception:
            df = pd.read_parquet(path)

    df = _normalize_columns(df)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"catalog parquet missing required columns: {missing}")
    if "DISPLAY_ARTIST" not in df.columns:
        if "ARTIST" in df.columns:
            df["DISPLAY_ARTIST"] = df["ARTIST"]
        else:
            df["DISPLAY_ARTIST"] = "unknown"
    df["PARSED_MAIN_GENRE"] = df["GENRES"].map(extract_main_genre)

    for date_col in ["RELEASE_DATE", "FIRST_SALE_DATE", "WEEK_END_DATE"]:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["WEEKS_SINCE_RELEASE"] = pd.to_numeric(df["WEEKS_SINCE_RELEASE"], errors="coerce")
    df["WORLDWIDE_STREAMS"] = pd.to_numeric(df["WORLDWIDE_STREAMS"], errors="coerce")

    df = df.dropna(
        subset=[
            "MRELG_ID",
            "TITLE",
            "WEEK_END_DATE",
            "WEEKS_SINCE_RELEASE",
            "WORLDWIDE_STREAMS",
        ]
    ).copy()
    floor = int(min_weeks_since_release) if min_weeks_since_release is not None else 0
    df = df[df["WEEKS_SINCE_RELEASE"] >= floor].copy()
    df = df[df["WORLDWIDE_STREAMS"] > 0].copy()

    # Keep one row per project/week if upstream duplicates exist.
    df = (
        df.sort_values(["MRELG_ID", "WEEK_END_DATE"])
        .drop_duplicates(subset=["MRELG_ID", "WEEK_END_DATE"], keep="last")
        .reset_index(drop=True)
    )
    if low_memory:
        df = _apply_low_memory_dtypes(df)
    return df


def _normalize_genre_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = str(v).strip().lower()
    if not s:
        return "unknown"
    # GENRES can be pipe/comma separated; use first token as primary.
    for sep in ("|", ",", ";", "/"):
        if sep in s:
            s = s.split(sep, 1)[0].strip()
            break
    return s or "unknown"


def extract_main_genre(genre_data: Any) -> str:
    if pd.isna(genre_data) or not genre_data:
        return "Unknown"

    try:
        if isinstance(genre_data, str):
            parsed_data = json.loads(genre_data)
        else:
            parsed_data = genre_data
    except Exception:
        return "Unknown"

    if not isinstance(parsed_data, list):
        return "Unknown"

    for provider in parsed_data:
        if isinstance(provider, dict) and provider.get("CLIENT_DOMAIN") == "Luminate":
            return provider.get("MAIN_GENRE", "Unknown")

    for provider in parsed_data:
        if isinstance(provider, dict) and provider.get("CLIENT_DOMAIN") == "Billboard":
            return provider.get("MAIN_GENRE", "Unknown")

    return "Unknown"


def _normalize_artist_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    s = str(v).strip().lower()
    return s if s else "unknown"


def _safe_feature_token(v: str) -> str:
    """
    LightGBM feature names cannot contain certain JSON-special characters.
    Normalize dynamic tokens (genres/artists) to a safe subset.
    """
    s = str(v).strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _fit_top_genres(df: pd.DataFrame, top_n: int = 12) -> list[str]:
    g = df["PARSED_MAIN_GENRE"].map(_normalize_genre_value)
    return g.value_counts().head(top_n).index.tolist()


def _fit_top_artists(df: pd.DataFrame, top_n: int = 60) -> list[str]:
    a = df["DISPLAY_ARTIST"].map(_normalize_artist_value)
    return a.value_counts().head(top_n).index.tolist()


def _build_artist_history_profiles(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """
    Build artist-level history priors (weighted by artist catalog footprint).
    Returns:
      - artist -> weighted log median streams
      - global default weighted log median
    """
    tmp = df.copy()
    tmp["__artist"] = tmp["DISPLAY_ARTIST"].map(_normalize_artist_value)
    tmp["__log_streams"] = np.log1p(pd.to_numeric(tmp["WORLDWIDE_STREAMS"], errors="coerce")).fillna(0.0)
    g = tmp.groupby("__artist").agg(
        log_median=("__log_streams", "median"),
        weeks=("__log_streams", "size"),
    ).reset_index()
    if g.empty:
        return {}, 0.0
    # Weight toward artists with deeper history.
    g["hist_weight"] = np.sqrt(g["weeks"].astype(float))
    global_default = float(np.average(g["log_median"], weights=g["hist_weight"]))
    return dict(zip(g["__artist"], g["log_median"])), global_default


def _weighted_sample_train_df(train_df: pd.DataFrame, max_train_rows: int) -> pd.DataFrame:
    if max_train_rows <= 0 or len(train_df) <= max_train_rows:
        return train_df
    streams = pd.to_numeric(train_df["WORLDWIDE_STREAMS"], errors="coerce").fillna(0.0).astype(float)
    # Flatter high-tier weighting so 80k-200k tracks are not drowned by mega-hits.
    w = np.log1p(np.maximum(streams.to_numpy(), 0.0))
    finite = np.isfinite(w)
    w = np.where(finite, w, 0.0)
    positive_mask = w > 0
    positive_count = int(np.sum(positive_mask))
    if positive_count == 0:
        return train_df.sample(n=max_train_rows, random_state=42)

    if positive_count >= max_train_rows:
        # Weighted sample only from positive-weight rows. Avoid materializing
        # ``train_df.loc[positive_mask].copy()`` (~all rows) — that duplicates RAM
        # and commonly triggers OOM before we downsample to max_train_rows.
        pos_indices = np.flatnonzero(positive_mask)
        w_pos = w[positive_mask]
        probs = w_pos / w_pos.sum()
        rng = np.random.default_rng(42)
        try:
            rel_pick = rng.choice(
                len(pos_indices),
                size=max_train_rows,
                replace=False,
                p=probs,
            )
        except ValueError:
            rel_pick = rng.choice(len(pos_indices), size=max_train_rows, replace=False)
        chosen_rows = pos_indices[rel_pick]
        return train_df.iloc[chosen_rows].copy().reset_index(drop=True)

    # Not enough strictly-positive weights to satisfy replace=False.
    # Take all positive rows, then fill remainder uniformly from the rest.
    pos_df = train_df.loc[positive_mask].copy()
    remainder = max_train_rows - len(pos_df)
    if remainder <= 0:
        return pos_df.sample(n=max_train_rows, random_state=42).reset_index(drop=True)
    rest_df = train_df.loc[~positive_mask].copy()
    if rest_df.empty:
        # All rows were positive but count mismatch due to numerical edge.
        return pos_df.sample(n=max_train_rows, replace=True, random_state=42).reset_index(drop=True)
    fill = rest_df.sample(
        n=min(remainder, len(rest_df)),
        replace=False,
        random_state=42,
    )
    out = pd.concat([pos_df, fill], ignore_index=True)
    if len(out) < max_train_rows:
        # Final top-up (rare): sample from all rows without weights.
        top_up = train_df.sample(
            n=max_train_rows - len(out),
            replace=False,
            random_state=42,
        )
        out = pd.concat([out, top_up], ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _build_feature_frame(
    df: pd.DataFrame,
    *,
    top_genres: list[str],
    top_artists: list[str],
    lag1_streams: pd.Series | np.ndarray | None = None,
    lag4_avg_streams: pd.Series | np.ndarray | None = None,
    lag12_avg_streams: pd.Series | np.ndarray | None = None,
    short_momentum: pd.Series | np.ndarray | None = None,
    long_momentum: pd.Series | np.ndarray | None = None,
    anomaly_flag: pd.Series | np.ndarray | None = None,
    weeks_since_peak: pd.Series | np.ndarray | None = None,
    artist_history_log_median: dict[str, float] | None = None,
    artist_history_default_log_median: float = 0.0,
) -> pd.DataFrame:
    cols: dict[str, Any] = {}
    if "PARSED_MAIN_GENRE" not in df.columns:
        df = df.copy()
        df["PARSED_MAIN_GENRE"] = df["GENRES"].map(extract_main_genre)
    w = df["WEEKS_SINCE_RELEASE"].astype(float).clip(lower=0.0)

    # Age effects (explicitly excluding static log/sqrt age curves).
    cols["age_capped_260"] = np.minimum(w, 260.0)
    cols["age_tail_260_plus"] = np.maximum(w - 260.0, 0.0)
    cols["age_weeks"] = w
    cols["age_weeks_sq"] = np.square(np.minimum(w, 520.0))

    # Seasonality from week-ending date.
    week_of_year = df["WEEK_END_DATE"].dt.isocalendar().week.astype(int)
    theta = 2.0 * np.pi * week_of_year / 52.0
    cols["woy_sin_1"] = np.sin(theta)
    cols["woy_cos_1"] = np.cos(theta)
    cols["woy_sin_2"] = np.sin(2.0 * theta)
    cols["woy_cos_2"] = np.cos(2.0 * theta)
    cols["is_q4"] = (df["WEEK_END_DATE"].dt.quarter == 4).astype(float)
    cols["is_q1"] = (df["WEEK_END_DATE"].dt.quarter == 1).astype(float)

    # Release-era effect.
    rel_year = df["FIRST_SALE_DATE"].dt.year.fillna(df["RELEASE_DATE"].dt.year).fillna(2010)
    cols["release_year_centered"] = rel_year.astype(float) - 2015.0

    genre = df["PARSED_MAIN_GENRE"].map(_normalize_genre_value)
    for g in top_genres:
        cols[f"genre__{_safe_feature_token(g)}"] = (genre == g).astype(float)
    cols["genre__other"] = (~genre.isin(top_genres)).astype(float)

    artist = df["DISPLAY_ARTIST"].map(_normalize_artist_value)
    for a in top_artists:
        cols[f"artist__{_safe_feature_token(a)}"] = (artist == a).astype(float)
    cols["artist__other"] = (~artist.isin(top_artists)).astype(float)
    if artist_history_log_median:
        cols["artist_hist_log_median"] = artist.map(
            lambda a: float(artist_history_log_median.get(a, artist_history_default_log_median))
        )
    else:
        cols["artist_hist_log_median"] = float(artist_history_default_log_median)
    cols["artist_hist_x_age"] = cols["artist_hist_log_median"] * cols["age_weeks"]

    # Keep absolute lag features in the model even with multiplier target.
    cols["Lag1W_Streams"] = (
        pd.to_numeric(lag1_streams, errors="coerce") if lag1_streams is not None else np.nan
    )
    cols["Lag4W_Avg_Streams"] = (
        pd.to_numeric(lag4_avg_streams, errors="coerce") if lag4_avg_streams is not None else np.nan
    )
    cols["Lag12W_Avg_Streams"] = (
        pd.to_numeric(lag12_avg_streams, errors="coerce")
        if lag12_avg_streams is not None
        else np.nan
    )
    cols["Short_Momentum"] = (
        pd.to_numeric(short_momentum, errors="coerce") if short_momentum is not None else np.nan
    )
    cols["Long_Momentum"] = (
        pd.to_numeric(long_momentum, errors="coerce") if long_momentum is not None else np.nan
    )
    if anomaly_flag is not None:
        # Forecast path passes Python lists / ndarrays; pd.to_numeric yields ndarray (no .fillna).
        a = np.asarray(pd.to_numeric(anomaly_flag, errors="coerce"), dtype=float)
        cols["Anomaly_Flag"] = np.where(np.isfinite(a), a, 0.0)
    else:
        cols["Anomaly_Flag"] = 0.0
    cols["Weeks_Since_Peak"] = (
        pd.to_numeric(weeks_since_peak, errors="coerce")
        if weeks_since_peak is not None
        else np.nan
    )

    for name, default in (
        ("Volatility_Context", 0.0),
        ("Baseline_Ratio", 1.0),
        ("Spike_State", 0.0),
    ):
        if name in df.columns:
            cols[name] = pd.to_numeric(df[name], errors="coerce").fillna(float(default)).astype(float)
        else:
            cols[name] = float(default)

    return pd.DataFrame(cols, index=df.index).copy()


def _safe_ratio(numer: pd.Series, denom: pd.Series, *, fallback: float = 1.0) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce").astype(float)
    d = pd.to_numeric(denom, errors="coerce").astype(float)
    out = pd.Series(float(fallback), index=n.index, dtype="float64")
    mask = np.isfinite(n.to_numpy()) & np.isfinite(d.to_numpy()) & (d.to_numpy() > 0.0)
    if mask.any():
        out.loc[mask] = n.loc[mask] / d.loc[mask]
    return out


def _compute_weeks_since_peak_by_track(df: pd.DataFrame, anomaly_col: str) -> pd.Series:
    """
    Stateful per-track counter:
    - 0 when anomaly flag is 1
    - otherwise increments weekly from prior value
    """
    if df.empty:
        return pd.Series(dtype="float64")
    work = df[["MRELG_ID", "WEEK_END_DATE", anomaly_col]].copy()
    work["MRELG_ID"] = work["MRELG_ID"].astype(str)
    work["WEEK_END_DATE"] = pd.to_datetime(work["WEEK_END_DATE"], errors="coerce")
    work = work.sort_values(["MRELG_ID", "WEEK_END_DATE"]).copy()
    out = pd.Series(index=work.index, dtype="float64")
    for _, idx in work.groupby("MRELG_ID", sort=False).groups.items():
        prev = 0
        for i in idx:
            flag = int(work.at[i, anomaly_col])
            if flag == 1:
                cur = 0
            else:
                cur = prev + 1
            out.at[i] = float(cur)
            prev = cur
    return out.reindex(df.index).astype(float)


def prepare_df_model(
    df: pd.DataFrame,
    *,
    top_genres: list[str],
    top_artists: list[str],
    artist_history_log_median: dict[str, float] | None = None,
    artist_history_default_log_median: float = 0.0,
    catalog_decay_target: str = CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
) -> pd.DataFrame:
    """
    Build training frame. Lag features are now pre-calculated in the parquet.

    ``rel_residual_baseline52``: target is ``(y - baseline) / max(baseline, eps)``
    with baseline = rolling median of **prior** ``BASELINE52_WINDOW`` observed
    weeks (shifted rolling, ``min_periods=BASELINE52_MIN_PERIODS``). Decode at
    serve time: ``y_hat = baseline * (1 + r_hat)``.

    ``hybrid_spike_gate_baseline52``: target ``y / d`` with ``d`` = 52w median when
    stable else ``lag1``; adds ``Volatility_Context``, ``Baseline_Ratio``, ``Spike_State``.
    Decode ``y_hat = clip(r_hat,0,5) * d`` with row-specific ``d`` at AR time.
    """
    d = df.copy()
    lag1 = pd.to_numeric(d["LAG1W_STREAMS"], errors="coerce").astype(float)
    lag4 = pd.to_numeric(d["LAG4W_AVG_STREAMS"], errors="coerce").astype(float)
    lag12 = pd.to_numeric(d["LAG12W_AVG_STREAMS"], errors="coerce").astype(float)
    y_level = pd.to_numeric(d["WORLDWIDE_STREAMS"], errors="coerce").astype(float)

    short_momentum = _safe_ratio(lag1, lag4, fallback=1.0)
    long_momentum = _safe_ratio(lag1, lag12, fallback=1.0)
    anomaly_flag = ((short_momentum > 1.5) & (long_momentum > 2.0)).astype(int)
    weeks_since_peak = _compute_weeks_since_peak_by_track(
        pd.DataFrame(
            {
                "MRELG_ID": d["MRELG_ID"].astype(str),
                "WEEK_END_DATE": pd.to_datetime(d["WEEK_END_DATE"], errors="coerce"),
                "anomaly_flag": anomaly_flag,
            }
        ),
        "anomaly_flag",
    )
    target_multiplier = _safe_ratio(y_level, lag1, fallback=1.0).clip(lower=0.0, upper=5.0)

    baseline_ser: pd.Series | None = None
    if catalog_decay_target in (
        CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52,
        CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52,
    ):
        baseline_ser = _panel_baseline52_median_worldwide(d)

    targ: pd.Series | None = None
    baseline_col: np.ndarray | None = None

    if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
        if baseline_ser is None:
            raise ValueError("internal: baseline_ser required for rel_residual_baseline52")
        b = pd.to_numeric(baseline_ser, errors="coerce").astype(float)
        b = pd.Series(b.values, index=d.index, dtype="float64")
        targ = ((y_level - b) / np.maximum(b, BASELINE52_EPS)).clip(
            REL_RESIDUAL_CLIP_LOW, REL_RESIDUAL_CLIP_HIGH
        )
        baseline_col = b.to_numpy(dtype=float)
    elif catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        if baseline_ser is None:
            raise ValueError("internal: baseline_ser required for hybrid_spike_gate_baseline52")
        vol = _panel_volatility_cv_last4_worldwide(d)
        b = pd.to_numeric(baseline_ser, errors="coerce").astype(float)
        br = lag1 / np.maximum(b, BASELINE52_EPS)
        br = br.where(np.isfinite(br), 1.0)
        spike = ((anomaly_flag.astype(float) > 0.5) | (br > SPIKE_GATE_BASELINE_RATIO_THR)).astype(float)
        b_np = b.to_numpy(dtype=float)
        lag_np = lag1.to_numpy(dtype=float)
        sp_np = spike.to_numpy(dtype=float)
        d_np = np.where(
            sp_np > 0.5,
            lag_np,
            np.where(np.isfinite(b_np) & (b_np >= BASELINE52_EPS), b_np, lag_np),
        )
        targ = pd.Series(
            np.clip(y_level.to_numpy(dtype=float) / np.maximum(d_np, BASELINE52_EPS), 0.0, 5.0),
            index=d.index,
            dtype="float64",
        )
        d["Volatility_Context"] = vol
        d["Baseline_Ratio"] = br
        d["Spike_State"] = spike
        baseline_col = b.to_numpy(dtype=float)
    else:
        targ = None
        baseline_col = None

    base = _build_feature_frame(
        d,
        top_genres=top_genres,
        top_artists=top_artists,
        lag1_streams=lag1,
        lag4_avg_streams=lag4,
        lag12_avg_streams=lag12,
        short_momentum=short_momentum,
        long_momentum=long_momentum,
        anomaly_flag=anomaly_flag,
        weeks_since_peak=weeks_since_peak,
        artist_history_log_median=artist_history_log_median,
        artist_history_default_log_median=artist_history_default_log_median,
    )

    if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
        extras = pd.DataFrame(
            {
                "target_worldwide_streams": pd.to_numeric(d["WORLDWIDE_STREAMS"], errors="coerce"),
                "target_rel_residual_baseline52": targ.astype(float),
                "BASELINE_52W_MEDIAN": baseline_col,
                "MRELG_ID": d["MRELG_ID"].astype(str).values,
                "WEEK_END_DATE": pd.to_datetime(d["WEEK_END_DATE"]).values,
                "WEEKS_SINCE_RELEASE": pd.to_numeric(d["WEEKS_SINCE_RELEASE"], errors="coerce").values,
            },
            index=base.index,
        )
    elif catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        extras = pd.DataFrame(
            {
                "target_worldwide_streams": pd.to_numeric(d["WORLDWIDE_STREAMS"], errors="coerce"),
                "target_hybrid_volgate": targ.astype(float),
                "BASELINE_52W_MEDIAN": baseline_col,
                "MRELG_ID": d["MRELG_ID"].astype(str).values,
                "WEEK_END_DATE": pd.to_datetime(d["WEEK_END_DATE"]).values,
                "WEEKS_SINCE_RELEASE": pd.to_numeric(d["WEEKS_SINCE_RELEASE"], errors="coerce").values,
            },
            index=base.index,
        )
    else:
        extras = pd.DataFrame(
            {
                "target_worldwide_streams": pd.to_numeric(d["WORLDWIDE_STREAMS"], errors="coerce"),
                "target_multiplier": target_multiplier,
                "MRELG_ID": d["MRELG_ID"].astype(str).values,
                "WEEK_END_DATE": pd.to_datetime(d["WEEK_END_DATE"]).values,
                "WEEKS_SINCE_RELEASE": pd.to_numeric(d["WEEKS_SINCE_RELEASE"], errors="coerce").values,
            },
            index=base.index,
        )

    # Single concat avoids DataFrame fragmentation from repeated column inserts.
    return pd.concat([base, extras], axis=1, copy=False).copy()


def fit_catalog_decay_model(
    df: pd.DataFrame,
    *,
    catalog_min_weeks: int = 52,
    artifact_catalog_min_weeks: int | None = None,
    top_genres_n: int = 12,
    top_artists_n: int = 1200,
    ridge_alpha: float = 2.0,
    max_train_rows: int = 0,
    target_transform: str = "none",
    memory_efficient_fit: bool = False,
    lgbm_n_jobs: int = 1,
    catalog_decay_target: str = CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
) -> CatalogDecayArtifacts:
    if catalog_decay_target not in (
        CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
        CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52,
        CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52,
    ):
        raise ValueError(
            f"unknown catalog_decay_target={catalog_decay_target!r}; "
            f"expected one of {CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1!r}, "
            f"{CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52!r}, "
            f"{CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52!r}"
        )
    if float(catalog_min_weeks) > 0:
        train_df = df[df["WEEKS_SINCE_RELEASE"] >= float(catalog_min_weeks)].copy()
    else:
        # Caller already restricted to catalog tail (e.g. released full panel).
        train_df = df
    if train_df.empty:
        raise ValueError("No training rows after catalog_min_weeks filter.")
    cmw_for_artifacts = (
        int(artifact_catalog_min_weeks)
        if artifact_catalog_min_weeks is not None
        else int(catalog_min_weeks)
    )
    if max_train_rows and max_train_rows > 0:
        before = len(train_df)
        train_df = _weighted_sample_train_df(train_df, int(max_train_rows))
        logger.info(
            "catalog_decay: sampled training rows %d -> %d (max_train_rows=%d)",
            before,
            len(train_df),
            int(max_train_rows),
        )

    top_genres = _fit_top_genres(train_df, top_n=top_genres_n)
    top_artists = _fit_top_artists(train_df, top_n=top_artists_n)
    artist_hist_map, artist_hist_default = _build_artist_history_profiles(train_df)
    df_model = prepare_df_model(
        train_df,
        top_genres=top_genres,
        top_artists=top_artists,
        artist_history_log_median=artist_hist_map,
        artist_history_default_log_median=artist_hist_default,
        catalog_decay_target=catalog_decay_target,
    )
    del train_df
    gc.collect()
    if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
        drop_subset = [
            "Lag1W_Streams",
            "Lag4W_Avg_Streams",
            "Lag12W_Avg_Streams",
            "Short_Momentum",
            "Long_Momentum",
            "Anomaly_Flag",
            "Weeks_Since_Peak",
            "target_rel_residual_baseline52",
            "BASELINE_52W_MEDIAN",
        ]
    elif catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        drop_subset = [
            "Lag1W_Streams",
            "Lag4W_Avg_Streams",
            "Lag12W_Avg_Streams",
            "Short_Momentum",
            "Long_Momentum",
            "Anomaly_Flag",
            "Weeks_Since_Peak",
            "Volatility_Context",
            "Baseline_Ratio",
            "Spike_State",
            "target_hybrid_volgate",
            "BASELINE_52W_MEDIAN",
        ]
    else:
        drop_subset = [
            "Lag1W_Streams",
            "Lag4W_Avg_Streams",
            "Lag12W_Avg_Streams",
            "Short_Momentum",
            "Long_Momentum",
            "Anomaly_Flag",
            "Weeks_Since_Peak",
            "target_multiplier",
        ]
    df_model = df_model.dropna(subset=drop_subset).copy()
    if df_model.empty:
        raise ValueError("No rows left after lag feature construction.")

    feature_cols = [
        c for c in df_model.columns
        if c
        not in (
            "target_worldwide_streams",
            "target_multiplier",
            "target_rel_residual_baseline52",
            "target_hybrid_volgate",
            "BASELINE_52W_MEDIAN",
            "MRELG_ID",
            "WEEK_END_DATE",
            "WEEKS_SINCE_RELEASE",
        )
    ]
    as_of = pd.to_datetime(df_model["WEEK_END_DATE"].max())
    if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
        y = df_model["target_rel_residual_baseline52"].astype(float).to_numpy(dtype=np.float32)
    elif catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        y = df_model["target_hybrid_volgate"].astype(float).to_numpy(dtype=np.float32)
    else:
        y = (
            df_model["target_multiplier"]
            .astype(float)
            .clip(lower=0.0, upper=5.0)
            .to_numpy(dtype=np.float32)
        )
    sample_weight: np.ndarray | None = None
    if catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        volw = df_model["target_worldwide_streams"].astype(float).clip(lower=0.0)
        sample_weight = np.log1p(volw.to_numpy(dtype=np.float32))
    x_block = df_model[feature_cols]
    del df_model
    gc.collect()
    # Contiguous float32 feature matrix only — avoids keeping a second float64 DataFrame + LightGBM copy.
    x = np.ascontiguousarray(x_block.to_numpy(dtype=np.float32, copy=True))
    del x_block
    gc.collect()
    tt = "none"
    if str(target_transform or "none").strip().lower() != "none":
        logger.warning(
            "catalog_decay: ignoring target_transform=%r for this head (using 'none' metadata only)",
            target_transform,
        )
    if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
        logger.info(
            "catalog_decay: fitting LightGBM on rel-residual vs prior-%d-w median baseline",
            BASELINE52_WINDOW,
        )
    elif catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        logger.info(
            "catalog_decay: fitting LightGBM on hybrid spike-gate target (stable=y/b52, volatile=y/lag1) "
            "with sample_weight=log1p(volume)"
        )
    else:
        logger.info("catalog_decay: fitting LightGBM on retained multiplier target")

    if memory_efficient_fit:
        logger.info(
            "catalog_decay: memory_efficient_fit — LightGBM max_bin=127 force_col_wise n_jobs=1 "
            "(slower, lower peak RAM)"
        )

    lgbm_kw: dict[str, Any] = dict(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=float(max(ridge_alpha, 0.0)),
        random_state=42,
        n_jobs=int(lgbm_n_jobs),
    )
    if memory_efficient_fit:
        lgbm_kw.update(
            max_bin=127,
            force_col_wise=True,
            n_jobs=1,
        )

    model = LGBMRegressor(**lgbm_kw)
    if sample_weight is not None:
        model.fit(x, y, sample_weight=sample_weight)
    else:
        model.fit(x, y)

    del x, y
    gc.collect()
    is_mult = catalog_decay_target != CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52
    return CatalogDecayArtifacts(
        model=model,
        feature_columns=list(feature_cols),
        top_genres=top_genres,
        top_artists=top_artists,
        as_of_date=as_of,
        catalog_min_weeks=cmw_for_artifacts,
        artist_history_default_log_median=artist_hist_default,
        target_transform=tt,
        target_is_multiplier=is_mult,
        catalog_decay_target=str(catalog_decay_target),
    )


def _build_future_grid(
    latest_rows: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    if end_date <= as_of_date:
        return pd.DataFrame(columns=latest_rows.columns)

    # Keep weekday aligned to data cadence.
    freq = "W-THU"
    if as_of_date.weekday() == 6:
        freq = "W-SUN"
    elif as_of_date.weekday() == 2:
        freq = "W-WED"

    future_dates = pd.date_range(
        start=as_of_date + pd.Timedelta(days=7),
        end=end_date,
        freq=freq,
    )
    if len(future_dates) == 0:
        return pd.DataFrame(columns=latest_rows.columns)

    rows = []
    for _, r in latest_rows.iterrows():
        base_age = float(r["WEEKS_SINCE_RELEASE"])
        for i, dt in enumerate(future_dates, start=1):
            rows.append(
                {
                    "MRELG_ID": r["MRELG_ID"],
                    "TITLE": r["TITLE"],
                    "DISPLAY_ARTIST": r.get("DISPLAY_ARTIST", "unknown"),
                    "GENRES": r["GENRES"],
                    "RELEASE_DATE": r["RELEASE_DATE"],
                    "FIRST_SALE_DATE": r["FIRST_SALE_DATE"],
                    "WEEK_END_DATE": dt,
                    "WEEKS_SINCE_RELEASE": base_age + i,
                }
            )
    return pd.DataFrame(rows)


def forecast_catalog_projects(
    df: pd.DataFrame,
    artifacts: CatalogDecayArtifacts,
    *,
    forecast_end_date: pd.Timestamp,
    min_history_rows: int = 8,
) -> pd.DataFrame:
    as_of = artifacts.as_of_date
    hist = df[df["WEEK_END_DATE"] <= as_of].copy()
    hist = hist.sort_values(["MRELG_ID", "WEEK_END_DATE"])
    artist_hist_map, artist_hist_default = _build_artist_history_profiles(hist)

    latest = hist.groupby("MRELG_ID", as_index=False, observed=False).tail(1).copy()
    latest["is_catalog"] = latest["WEEKS_SINCE_RELEASE"] >= artifacts.catalog_min_weeks
    latest = latest[latest["is_catalog"]].copy()
    if latest.empty:
        return pd.DataFrame()

    hist_counts = hist.groupby("MRELG_ID", observed=False).size().reset_index(name="hist_rows")
    latest = latest.merge(hist_counts, on="MRELG_ID", how="left")
    latest = latest[latest["hist_rows"] >= int(min_history_rows)].copy()
    if latest.empty:
        return pd.DataFrame()

    future = _build_future_grid(latest, as_of_date=as_of, end_date=forecast_end_date)
    if future.empty:
        return pd.DataFrame()
        
    history_by_mrelg: dict[str, list[float]] = {}
    for mid, chunk in hist.groupby("MRELG_ID", sort=False, observed=False):
        vals = pd.to_numeric(chunk["WORLDWIDE_STREAMS"], errors="coerce").dropna().astype(float).tolist()
        history_by_mrelg[str(mid)] = vals
    weeks_since_peak_state: dict[str, int] = {}
    for mid, series in history_by_mrelg.items():
        w = 0
        for i in range(len(series)):
            lag1 = float(series[i - 1]) if i >= 1 else float(series[i])
            lag4 = float(np.mean(series[max(0, i - 4):i])) if i >= 1 else lag1
            lag12 = float(np.mean(series[max(0, i - 12):i])) if i >= 1 else lag1
            short = lag1 / lag4 if np.isfinite(lag4) and lag4 > 0 else 1.0
            long = lag1 / lag12 if np.isfinite(lag12) and lag12 > 0 else 1.0
            anomaly = 1 if (short > 1.5 and long > 2.0) else 0
            w = 0 if anomaly == 1 else (w + 1)
        weeks_since_peak_state[mid] = int(w)

    # --- NEW VECTORIZED INFERENCE LOOP ---
    
    # 1. Group by Date instead of Track
    future_dates = sorted(future["WEEK_END_DATE"].unique())
    rows_out = []

    # 2. Loop through time chronologically
    for dt in future_dates:
        # Get all track rows for this specific week
        chunk = future[future["WEEK_END_DATE"] == dt].copy()
        
        # Ensure PARSED_MAIN_GENRE is present for the feature builder
        if "PARSED_MAIN_GENRE" not in chunk.columns:
            chunk["PARSED_MAIN_GENRE"] = chunk["GENRES"].apply(extract_main_genre)
        
        # Vectorized lag calculation: lookup latest history for all tracks at once
        lag1_list, lag4_list, lag12_list = [], [], []
        baseline52_list: list[float] = []
        short_momentum_list, long_momentum_list, anomaly_flag_list, weeks_since_peak_list = [], [], [], []
        for mid in chunk["MRELG_ID"]:
            series = history_by_mrelg.get(str(mid), [])
            bl = baseline52_median_from_history_stream(series)
            baseline52_list.append(float(bl) if math.isfinite(bl) else float("nan"))
            lag1 = float(series[-1]) if len(series) >= 1 else np.nan
            lag4 = float(np.mean(series[-4:])) if len(series) >= 1 else np.nan
            lag12 = float(np.mean(series[-12:])) if len(series) >= 1 else np.nan
            short = (lag1 / lag4) if np.isfinite(lag4) and lag4 > 0 and np.isfinite(lag1) else 1.0
            long = (lag1 / lag12) if np.isfinite(lag12) and lag12 > 0 and np.isfinite(lag1) else 1.0
            anomaly = 1 if (short > 1.5 and long > 2.0) else 0
            current_wsp = int(weeks_since_peak_state.get(str(mid), 0))
            next_wsp = 0 if anomaly == 1 else current_wsp + 1
            lag1_list.append(lag1)
            lag4_list.append(lag4)
            lag12_list.append(lag12)
            short_momentum_list.append(short)
            long_momentum_list.append(long)
            anomaly_flag_list.append(anomaly)
            weeks_since_peak_list.append(float(next_wsp))
            weeks_since_peak_state[str(mid)] = next_wsp

        ct = str(getattr(artifacts, "catalog_decay_target", CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1) or "")
        if ct == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
            vol_list, br_list, sp_list = [], [], []
            for mid in chunk["MRELG_ID"]:
                series = history_by_mrelg.get(str(mid), [])
                vol, br, spk, _d = hybrid_inference_denominator_and_features(series)
                vol_list.append(vol)
                br_list.append(br)
                sp_list.append(spk)
            chunk["Volatility_Context"] = vol_list
            chunk["Baseline_Ratio"] = br_list
            chunk["Spike_State"] = sp_list

        # 3. Build the feature matrix for ALL tracks in this week at once
        base = _build_feature_frame(
            chunk,
            top_genres=artifacts.top_genres,
            top_artists=artifacts.top_artists,
            lag1_streams=lag1_list,
            lag4_avg_streams=lag4_list,
            lag12_avg_streams=lag12_list,
            short_momentum=short_momentum_list,
            long_momentum=long_momentum_list,
            anomaly_flag=anomaly_flag_list,
            weeks_since_peak=weeks_since_peak_list,
            artist_history_log_median=artist_hist_map,
            artist_history_default_log_median=artist_hist_default,
        )

        x = base.reindex(columns=artifacts.feature_columns).fillna(0.0)

        raw_head = np.asarray(artifacts.model.predict(x), dtype=float)
        lag1_arr = np.nan_to_num(np.asarray(lag1_list, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if ct == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
            # r_hat clipped like training; streams = baseline * (1 + r).
            r = np.nan_to_num(raw_head, nan=0.0, posinf=REL_RESIDUAL_CLIP_HIGH, neginf=REL_RESIDUAL_CLIP_LOW)
            r = np.clip(r, REL_RESIDUAL_CLIP_LOW, REL_RESIDUAL_CLIP_HIGH)
            b = np.asarray(baseline52_list, dtype=float)
            b = np.where(np.isfinite(b) & (b >= BASELINE52_EPS), b, np.maximum(lag1_arr, BASELINE52_EPS))
            preds = np.maximum(0.0, b * (1.0 + r))
            predicted_multiplier = r  # legacy column name: raw model head, not m=y/lag1
        elif ct == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
            mhat = np.clip(np.nan_to_num(raw_head, nan=0.0), 0.0, 5.0)
            d_list: list[float] = []
            for mid in chunk["MRELG_ID"]:
                series = history_by_mrelg.get(str(mid), [])
                _v, _br, _s, d_i = hybrid_inference_denominator_and_features(series)
                d_list.append(max(float(d_i), BASELINE52_EPS))
            d_arr = np.asarray(d_list, dtype=float)
            preds = np.maximum(0.0, mhat * d_arr)
            predicted_multiplier = mhat
        else:
            predicted_multiplier = np.clip(raw_head, 0.0, 5.0)
            preds = np.maximum(0.0, predicted_multiplier * lag1_arr)
        
        # 5. Append predictions back to history so next week's lags are correct
        for i, mid in enumerate(chunk["MRELG_ID"]):
            history_by_mrelg[str(mid)].append(preds[i])
            
        # Collect results
        chunk["predicted_multiplier"] = predicted_multiplier
        chunk["predicted_worldwide_streams"] = preds
        chunk["as_of_date"] = artifacts.as_of_date
        rows_out.append(chunk)

    out = pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(["MRELG_ID", "WEEK_END_DATE"]).reset_index(drop=True)
        
    return out


def write_artifacts(
    out_dir: Path,
    artifacts: CatalogDecayArtifacts,
    forecast_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": artifacts.model,
        "feature_columns": artifacts.feature_columns,
        "top_genres": artifacts.top_genres,
        "top_artists": artifacts.top_artists,
        "as_of_date": str(artifacts.as_of_date.date()),
        "catalog_min_weeks": artifacts.catalog_min_weeks,
        "artist_history_default_log_median": artifacts.artist_history_default_log_median,
        "target_transform": getattr(artifacts, "target_transform", "none") or "none",
        "target_is_multiplier": bool(getattr(artifacts, "target_is_multiplier", True)),
        "catalog_decay_target": str(
            getattr(artifacts, "catalog_decay_target", CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1)
        ),
    }
    joblib.dump(model_bundle, out_dir / "catalog_decay_model.joblib")
    with open(out_dir / "catalog_decay_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    feature_importance = getattr(artifacts.model, "feature_importances_", None)
    if feature_importance is not None:
        coef = pd.DataFrame(
            {"feature": artifacts.feature_columns, "feature_importance": feature_importance}
        ).sort_values("feature_importance", ascending=False)
    else:
        coef = pd.DataFrame({"feature": artifacts.feature_columns})
    coef.to_csv(out_dir / "catalog_decay_coefficients.csv", index=False)

    forecast_df.to_parquet(out_dir / "catalog_decay_forecasts.parquet", index=False)
    if len(forecast_df) <= 200_000:
        forecast_df.to_csv(out_dir / "catalog_decay_forecasts.csv", index=False)

    meta = {
        "as_of_date": str(artifacts.as_of_date.date()),
        "catalog_min_weeks": artifacts.catalog_min_weeks,
        "top_genres": artifacts.top_genres,
        "top_artists": artifacts.top_artists,
        "forecast_rows": int(len(forecast_df)),
        "target_transform": getattr(artifacts, "target_transform", "none") or "none",
        "target_is_multiplier": bool(getattr(artifacts, "target_is_multiplier", True)),
        "catalog_decay_target": str(
            getattr(artifacts, "catalog_decay_target", CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1)
        ),
    }
    with open(out_dir / "catalog_decay_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_and_forecast_catalog_decay(
    *,
    input_parquet: Path,
    output_dir: Path,
    forecast_end_date: pd.Timestamp | None = None,
    catalog_min_weeks: int = 52,
    top_genres_n: int = 12,
    top_artists_n: int = 60,
    ridge_alpha: float = 2.0,
    min_history_rows: int = 8,
    max_train_rows: int = 0,
    low_memory: bool = False,
    target_transform: str = "none",
    memory_efficient_fit: bool = False,
    lgbm_n_jobs: int = 1,
    skip_forecast: bool = False,
    catalog_decay_target: str = CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
) -> pd.DataFrame:
    df = load_catalog_streams(
        input_parquet,
        low_memory=low_memory,
        min_weeks_since_release=max(0, int(catalog_min_weeks) - 2),
        memory_efficient_load=memory_efficient_fit,
    )
    n_loaded_rows = len(df)
    if forecast_end_date is None:
        as_of = pd.to_datetime(df["WEEK_END_DATE"].max())
        forecast_end_date = _default_end_of_year(as_of)
    if skip_forecast:
        # Peak RAM: drop the full loaded panel once we have the catalog-tail pool for fit.
        fit_pool = df.loc[df["WEEKS_SINCE_RELEASE"] >= float(catalog_min_weeks)].copy()
        del df
        gc.collect()
        artifacts = fit_catalog_decay_model(
            fit_pool,
            catalog_min_weeks=0,
            artifact_catalog_min_weeks=int(catalog_min_weeks),
            top_genres_n=top_genres_n,
            top_artists_n=top_artists_n,
            ridge_alpha=ridge_alpha,
            max_train_rows=max_train_rows,
            target_transform=target_transform,
            memory_efficient_fit=memory_efficient_fit,
            lgbm_n_jobs=lgbm_n_jobs,
            catalog_decay_target=catalog_decay_target,
        )
        del fit_pool
        gc.collect()
    else:
        artifacts = fit_catalog_decay_model(
            df,
            catalog_min_weeks=catalog_min_weeks,
            top_genres_n=top_genres_n,
            top_artists_n=top_artists_n,
            ridge_alpha=ridge_alpha,
            max_train_rows=max_train_rows,
            target_transform=target_transform,
            memory_efficient_fit=memory_efficient_fit,
            lgbm_n_jobs=lgbm_n_jobs,
            catalog_decay_target=catalog_decay_target,
        )
    if skip_forecast:
        gc.collect()
        forecast_df = pd.DataFrame()
        logger.info("catalog_decay: skip_forecast=True — not building future grid or weekly chunks")
    else:
        forecast_df = forecast_catalog_projects(
            df,
            artifacts,
            forecast_end_date=forecast_end_date,
            min_history_rows=min_history_rows,
        )
    write_artifacts(output_dir, artifacts, forecast_df)
    if skip_forecast:
        logger.info(
            "catalog_decay: loaded %d rows, forecast skipped (wrote %d forecast rows)",
            n_loaded_rows,
            len(forecast_df),
        )
    else:
        logger.info(
            "catalog_decay: trained on %d rows, forecasted %d rows through %s",
            n_loaded_rows,
            len(forecast_df),
            pd.to_datetime(forecast_end_date).date(),
        )
    return forecast_df


def _default_end_of_year(as_of_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=as_of_date.year, month=12, day=31)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a catalog decay model from catalog_streams.parquet and forecast "
            "catalog project streams through year-end."
        )
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        default=_repo_root() / "model" / "data" / "catalog_streams.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "model" / "catalog_decay_artifacts",
    )
    parser.add_argument("--forecast-end-date", type=str, default=None)
    parser.add_argument(
        "--catalog-min-weeks",
        type=int,
        default=52,
        help="Train only on rows with WEEKS_SINCE_RELEASE >= this (catalog tail).",
    )
    parser.add_argument("--top-genres-n", type=int, default=12)
    parser.add_argument("--top-artists-n", type=int, default=1200)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=2.0,
        help="Mapped to LightGBM reg_alpha (L1 regularization).",
    )
    parser.add_argument("--min-history-rows", type=int, default=8)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Cap rows used for model fit (0 = no cap). Uses weighted sampling.",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Downcast dtypes and categories to reduce memory footprint.",
    )
    parser.add_argument(
        "--memory-efficient-fit",
        action="store_true",
        help=(
            "Lower peak RAM at the cost of speed: Arrow self_destruct on parquet→pandas, "
            "float32 X/y before LightGBM, drop wide frames early, max_bin=127, "
            "force_col_wise=True, n_jobs=1."
        ),
    )
    parser.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=1,
        help=(
            "LightGBM n_jobs (thread count). Default 1 avoids multi-worker RAM spikes on small hosts; "
            "use -1 for all CPUs on large-memory machines."
        ),
    )
    parser.add_argument(
        "--target-transform",
        choices=("none",),
        default="none",
        help="Retained multiplier target uses only 'none'.",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help=(
            "Train and write model artifacts only; skip forecast_catalog_projects. "
            "Avoids the large future grid and per-week DataFrame peak RAM."
        ),
    )
    parser.add_argument(
        "--catalog-decay-target",
        choices=(
            CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
            CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52,
            CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52,
        ),
        default=CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
        help=(
            "Training head: retained y/lag1 (default); rel residual vs 52w median; "
            "or hybrid spike-gate (stable=y/b52, volatile=y/lag1) with log1p(volume) sample weights."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    end_date = pd.to_datetime(args.forecast_end_date) if args.forecast_end_date else None

    _ = train_and_forecast_catalog_decay(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
        forecast_end_date=end_date,
        catalog_min_weeks=args.catalog_min_weeks,
        top_genres_n=args.top_genres_n,
        top_artists_n=args.top_artists_n,
        ridge_alpha=args.ridge_alpha,
        min_history_rows=args.min_history_rows,
        max_train_rows=args.max_train_rows,
        low_memory=args.low_memory,
        target_transform=args.target_transform,
        memory_efficient_fit=args.memory_efficient_fit,
        lgbm_n_jobs=args.lgbm_n_jobs,
        skip_forecast=args.skip_forecast,
        catalog_decay_target=args.catalog_decay_target,
    )


if __name__ == "__main__":
    main()

