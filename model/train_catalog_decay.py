from __future__ import annotations

import argparse
import json
import logging
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


@dataclass
class CatalogDecayArtifacts:
    model: Any
    feature_columns: list[str]
    top_genres: list[str]
    top_artists: list[str]
    as_of_date: pd.Timestamp
    catalog_min_weeks: int
    artist_history_default_log_median: float


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
    # Weight by sqrt(streams) so large projects are represented but not dominant.
    w = np.sqrt(np.maximum(streams.to_numpy(), 0.0))
    finite = np.isfinite(w)
    w = np.where(finite, w, 0.0)
    positive_mask = w > 0
    positive_count = int(np.sum(positive_mask))
    if positive_count == 0:
        return train_df.sample(n=max_train_rows, random_state=42)

    if positive_count >= max_train_rows:
        # Weighted sample only from positive-weight rows.
        sub = train_df.loc[positive_mask].copy()
        probs = w[positive_mask]
        probs = probs / probs.sum()
        try:
            sampled = sub.sample(
                n=max_train_rows,
                replace=False,
                weights=probs,
                random_state=42,
            )
        except ValueError:
            # Pandas can still reject sparse/degenerate weight vectors in some
            # edge cases. Fall back to uniform sample from positive rows.
            sampled = sub.sample(
                n=max_train_rows,
                replace=False,
                random_state=42,
            )
        return sampled.reset_index(drop=True)

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

    return pd.DataFrame(cols, index=df.index).copy()


def prepare_df_model(
    df: pd.DataFrame,
    *,
    top_genres: list[str],
    top_artists: list[str],
    artist_history_log_median: dict[str, float] | None = None,
    artist_history_default_log_median: float = 0.0,
) -> pd.DataFrame:
    """
    Build training frame. Lag features are now pre-calculated in the parquet.
    """
    d = df.copy()

    base = _build_feature_frame(
        d,
        top_genres=top_genres,
        top_artists=top_artists,
        artist_history_log_median=artist_history_log_median,
        artist_history_default_log_median=artist_history_default_log_median,
    )

    extras = pd.DataFrame(
        {
            "Lag1W_Streams": pd.to_numeric(d["LAG1W_STREAMS"], errors="coerce"),
            "Lag4W_Avg_Streams": pd.to_numeric(d["LAG4W_AVG_STREAMS"], errors="coerce"),
            "Lag12W_Avg_Streams": pd.to_numeric(d["LAG12W_AVG_STREAMS"], errors="coerce"),
            "target_worldwide_streams": pd.to_numeric(d["WORLDWIDE_STREAMS"], errors="coerce"),
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
    catalog_min_weeks: int = 78,
    top_genres_n: int = 12,
    top_artists_n: int = 1200,
    ridge_alpha: float = 2.0,
    max_train_rows: int = 0,
) -> CatalogDecayArtifacts:
    train_df = df[df["WEEKS_SINCE_RELEASE"] >= float(catalog_min_weeks)].copy()
    if train_df.empty:
        raise ValueError("No training rows after catalog_min_weeks filter.")
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
    )
    df_model = df_model.dropna(
        subset=["Lag1W_Streams", "Lag4W_Avg_Streams", "Lag12W_Avg_Streams", "target_worldwide_streams"]
    ).copy()
    if df_model.empty:
        raise ValueError("No rows left after lag feature construction.")

    feature_cols = [
        c for c in df_model.columns
        if c
        not in ("target_worldwide_streams", "MRELG_ID", "WEEK_END_DATE", "WEEKS_SINCE_RELEASE")
    ]
    x = df_model[feature_cols]
    y = df_model["target_worldwide_streams"].astype(float)

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=float(max(ridge_alpha, 0.0)),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)

    as_of = pd.to_datetime(df["WEEK_END_DATE"].max())
    return CatalogDecayArtifacts(
        model=model,
        feature_columns=list(feature_cols),
        top_genres=top_genres,
        top_artists=top_artists,
        as_of_date=as_of,
        catalog_min_weeks=catalog_min_weeks,
        artist_history_default_log_median=artist_hist_default,
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

    latest = hist.groupby("MRELG_ID", as_index=False).tail(1).copy()
    latest["is_catalog"] = latest["WEEKS_SINCE_RELEASE"] >= artifacts.catalog_min_weeks
    latest = latest[latest["is_catalog"]].copy()
    if latest.empty:
        return pd.DataFrame()

    hist_counts = hist.groupby("MRELG_ID").size().reset_index(name="hist_rows")
    latest = latest.merge(hist_counts, on="MRELG_ID", how="left")
    latest = latest[latest["hist_rows"] >= int(min_history_rows)].copy()
    if latest.empty:
        return pd.DataFrame()

    future = _build_future_grid(latest, as_of_date=as_of, end_date=forecast_end_date)
    if future.empty:
        return pd.DataFrame()
        
    history_by_mrelg: dict[str, list[float]] = {}
    for mid, chunk in hist.groupby("MRELG_ID", sort=False):
        vals = pd.to_numeric(chunk["WORLDWIDE_STREAMS"], errors="coerce").dropna().astype(float).tolist()
        history_by_mrelg[str(mid)] = vals

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
        for mid in chunk["MRELG_ID"]:
            series = history_by_mrelg.get(str(mid), [])
            lag1_list.append(series[-1] if len(series) >= 1 else np.nan)
            lag4_list.append(np.mean(series[-4:]) if len(series) >= 1 else np.nan)
            lag12_list.append(np.mean(series[-12:]) if len(series) >= 1 else np.nan)
            
        # 3. Build the feature matrix for ALL tracks in this week at once
        base = _build_feature_frame(
            chunk, 
            top_genres=artifacts.top_genres,
            top_artists=artifacts.top_artists,
            artist_history_log_median=artist_hist_map,
            artist_history_default_log_median=artist_hist_default,
        )
        base["Lag1W_Streams"] = lag1_list
        base["Lag4W_Avg_Streams"] = lag4_list
        base["Lag12W_Avg_Streams"] = lag12_list
        
        x = base.reindex(columns=artifacts.feature_columns).fillna(0.0)
        
        # 4. Predict ALL tracks for this week in a single fast call
        preds = np.maximum(0.0, artifacts.model.predict(x))
        
        # 5. Append predictions back to history so next week's lags are correct
        for i, mid in enumerate(chunk["MRELG_ID"]):
            history_by_mrelg[str(mid)].append(preds[i])
            
        # Collect results
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
    }
    with open(out_dir / "catalog_decay_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train_and_forecast_catalog_decay(
    *,
    input_parquet: Path,
    output_dir: Path,
    forecast_end_date: pd.Timestamp | None = None,
    catalog_min_weeks: int = 78,
    top_genres_n: int = 12,
    top_artists_n: int = 60,
    ridge_alpha: float = 2.0,
    min_history_rows: int = 8,
    max_train_rows: int = 0,
    low_memory: bool = False,
) -> pd.DataFrame:
    df = load_catalog_streams(
        input_parquet,
        low_memory=low_memory,
        min_weeks_since_release=max(0, int(catalog_min_weeks) - 2),
    )
    if forecast_end_date is None:
        as_of = pd.to_datetime(df["WEEK_END_DATE"].max())
        forecast_end_date = _default_end_of_year(as_of)
    artifacts = fit_catalog_decay_model(
        df,
        catalog_min_weeks=catalog_min_weeks,
        top_genres_n=top_genres_n,
        top_artists_n=top_artists_n,
        ridge_alpha=ridge_alpha,
        max_train_rows=max_train_rows,
    )
    forecast_df = forecast_catalog_projects(
        df,
        artifacts,
        forecast_end_date=forecast_end_date,
        min_history_rows=min_history_rows,
    )
    write_artifacts(output_dir, artifacts, forecast_df)
    logger.info(
        "catalog_decay: trained on %d rows, forecasted %d rows through %s",
        len(df),
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
    parser.add_argument("--catalog-min-weeks", type=int, default=78)
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
    )


if __name__ == "__main__":
    main()

