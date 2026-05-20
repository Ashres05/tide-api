import argparse
import re
import json
import os
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def gamma_norm(t: np.ndarray, a: float, t_peak: float) -> np.ndarray:
    """
    Gamma-like normalized shape where f(t_peak) = 1.

    f(t) = (t / t_peak)^a * exp(-a * (t/t_peak - 1))

    Works for t>=1, a>0, t_peak>=1.
    """
    t = np.asarray(t, dtype=float)
    t_peak = float(t_peak)
    # Numerical stability for extreme values
    ratio = np.clip(t / max(t_peak, 1e-9), 1e-12, None)
    a = float(a)
    return (ratio**a) * np.exp(-a * (ratio - 1.0))


# Bear/Base/Bull shape scaling relative to Base per trained archetype cluster (0..3).
# Used as a multiplier on each cluster's gamma_norm contribution; Base => ratio 1.0.
ARCHETYPE_SCENARIO_MULTIPLIERS: Dict[int, Dict[str, float]] = {
    0: {"Bear": 3.35, "Base": 6.20, "Bull": 7.72},
    1: {"Bear": 2.68, "Base": 4.34, "Bull": 5.14},
    2: {"Bear": 3.20, "Base": 5.68, "Bull": 6.73},
    3: {"Bear": 6.74, "Base": 10.85, "Bull": 13.14},
}


def normalize_archetype_scenario_label(scenario: Optional[str]) -> str:
    if scenario is None or not str(scenario).strip():
        return "Base"
    key = str(scenario).strip().lower()
    if key == "bear":
        return "Bear"
    if key == "bull":
        return "Bull"
    if key == "base":
        return "Base"
    return "Base"


def archetype_scenario_shape_ratio(cluster_id: int, scenario: Optional[str]) -> float:
    """Scale normalized decay shape vs Base for this cluster (1.0 when scenario is Base)."""
    scen = normalize_archetype_scenario_label(scenario)
    row = ARCHETYPE_SCENARIO_MULTIPLIERS.get(int(cluster_id))
    if not row:
        return 1.0
    base = float(row.get("Base", 0.0))
    if base <= 0:
        return 1.0
    return float(row.get(scen, base)) / base


# Cluster ids used to derive a scenario-wide scalar when the caller has no
# specific cluster pinned. Matches the keys of ARCHETYPE_SCENARIO_MULTIPLIERS.
_SCENARIO_FALLBACK_CLUSTERS: Tuple[int, ...] = (0, 1, 2, 3)


def _ratio_from_learned_table(
    table: Dict[str, Dict[str, float]],
    cluster_key: str,
    label: str,
) -> Optional[float]:
    """Bear/Base or Bull/Base ratio for a cluster from a multipliers table.

    Returns None when the cluster entry is missing, malformed, or has a
    non-positive Base percentile.
    """
    row = table.get(cluster_key)
    if not isinstance(row, dict):
        return None
    try:
        base = float(row.get("Base", 0.0))
        scen = float(row.get(label, base))
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(base) and base > 0):
        return None
    if not np.isfinite(scen):
        return None
    return float(scen / base)


def resolve_scenario_multiplier(
    scenario: Optional[str],
    cluster_id: Optional[int] = None,
    scenario_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Translate a user scenario label into a single scalar multiplier.

    The returned value is intended to be applied as a *post-fit* shock on the
    future portion of a decay curve relative to the asymptotic floor — see the
    ``scenario_multiplier`` argument on ``fit_backfill_forecast`` and the
    upstream wrappers. The fit itself must always run with ``scenario="Base"``
    so the floor never shifts between scenarios; this helper exists purely to
    answer "by how much should I shock the future weeks above the floor".

    Resolution order:

      1. If ``scenario_multipliers`` is provided (e.g. learned natively from a
         specific metric's parquet during training, currently the
         worldwide_streams artifact's ``scenario_multipliers.json``), use it.
         With a known cluster_id, return ``scenario_pct / base_pct`` for that
         cluster. Without a cluster_id, average the per-cluster ratios across
         every cluster in the table.
      2. Otherwise fall back to the hardcoded global
         ``ARCHETYPE_SCENARIO_MULTIPLIERS`` via
         ``archetype_scenario_shape_ratio`` — preserving behavior for older
         artifacts that don't ship a learned table.

    Returns 1.0 for Base, or any time inputs are degenerate.
    """
    label = normalize_archetype_scenario_label(scenario)
    if label == "Base":
        return 1.0

    if scenario_multipliers:
        if cluster_id is not None:
            try:
                ratio = _ratio_from_learned_table(
                    scenario_multipliers, str(int(cluster_id)), label
                )
            except (TypeError, ValueError):
                ratio = None
            if ratio is not None:
                return ratio
        learned_ratios = [
            r
            for r in (
                _ratio_from_learned_table(scenario_multipliers, c_key, label)
                for c_key in scenario_multipliers.keys()
            )
            if r is not None
        ]
        if learned_ratios:
            return float(np.mean(learned_ratios))

    if cluster_id is not None:
        try:
            return float(archetype_scenario_shape_ratio(int(cluster_id), label))
        except (TypeError, ValueError):
            pass
    ratios = [archetype_scenario_shape_ratio(c, label) for c in _SCENARIO_FALLBACK_CLUSTERS]
    return float(np.mean(ratios))


def extract_main_genre(genre_val: Any) -> str:
    """
    Extract the "MAIN_GENRE" from the GENRES JSON blob.
    Priority: Luminate -> Billboard -> first available.
    """
    if genre_val is None or (isinstance(genre_val, float) and np.isnan(genre_val)):
        return "Unknown"

    try:
        data = json.loads(genre_val) if isinstance(genre_val, str) else genre_val
        if not data:
            return "Unknown"

        for entry in data:
            if entry.get("CLIENT_DOMAIN") == "Luminate":
                return entry.get("MAIN_GENRE", "Unknown")

        for entry in data:
            if entry.get("CLIENT_DOMAIN") == "Billboard":
                return entry.get("MAIN_GENRE", "Unknown")

        first = data[0] if isinstance(data, list) and data else {}
        return first.get("MAIN_GENRE", "Unknown")
    except Exception:
        return "Unknown"


#def compute_week_index(df: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
def compute_week_index(df: pd.DataFrame, horizon_weeks: int, metric_col: str) -> pd.DataFrame: # str = "WEEKLY_STREAMS"
    #df = df.copy()
    #df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"])
    #df["WEEKLY_EQUIVALENTS"] = pd.to_numeric(df["WEEKLY_EQUIVALENTS"], errors="coerce").fillna(0.0)
    df = df.copy()
    df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"])
    # Use dynamic metric column
    df["TARGET_METRIC"] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0.0)

    # "Exact diff" -> round(days/7) approach from the original notebook.
    first_dates = df.groupby("MRELG_ID")["WEEK_END_DATE"].transform("min")
    weeks_since_release = ((df["WEEK_END_DATE"] - first_dates).dt.days / 7).round().astype(int)
    df["WEEKS_SINCE_RELEASE"] = weeks_since_release

    # Align each release so its timeline starts at week=1.
    min_week = df.groupby("MRELG_ID")["WEEKS_SINCE_RELEASE"].transform("min")
    df["week"] = (df["WEEKS_SINCE_RELEASE"] - min_week) + 1

    df = df[(df["week"] >= 1) & (df["week"] <= horizon_weeks)].copy()
    df["week"] = df["week"].astype(int)
    return df


def safe_linreg_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Fit a simple linear model y ~ m*x + b and return slope m.
    Expects x and y to be 1D arrays, len(x)>=3.
    """
    if len(x) < 3:
        return float("nan")
    m, _b = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(m)


def extract_track_features(g: pd.DataFrame, horizon_weeks: int) -> Optional[Dict[str, Any]]:
    """
    Compute missing-week-safe shape features from observed weeks only.

    Key idea: we never create a full Week_1..Week_horizon vector with implicit zeros.
    Instead, we derive features from whatever weeks exist in the data for that release.
    """
    g = g.sort_values("week")
    w = g["week"].to_numpy(dtype=int)
    #y = g["WEEKLY_STREAMS"].to_numpy(dtype=float)
    y = g["TARGET_METRIC"].to_numpy(dtype=float)

    if len(w) == 0:
        return None

    peak_idx = int(np.argmax(y))
    peak_y = float(y[peak_idx])
    if not np.isfinite(peak_y) or peak_y <= 0:
        return None

    y_norm = y / peak_y

    peak_week_obs = float(w[peak_idx])
    last_week_obs = float(w.max())
    # Normalized value at week=1 if observed; else earliest observation.
    if np.any(w == 1):
        y_week1_norm = float(y_norm[w == 1][0])
    else:
        y_week1_norm = float(y_norm[0])

    y_last_norm = float(y_norm[w == w.max()][0]) if np.any(w == w.max()) else float(y_norm[-1])
    tail_volume_obs = float(y[w == w.max()][0]) if np.any(w == w.max()) else float(y[-1])

    # Half-life: first post-peak week where y_norm <= 0.5 (observed only).
    post_mask = w >= int(peak_week_obs)
    w_post = w[post_mask]
    y_post = y_norm[post_mask]

    half_life_week = float("nan")
    if len(w_post) >= 2:
        idx = np.where(y_post <= 0.5)[0]
        if len(idx) > 0:
            half_life_week = float(w_post[idx[0]])
        else:
            half_life_week = float(w_post.max())

    # Decay log slope on post-peak points with y_norm>0.
    decay_log_slope = float("nan")
    if len(w_post) >= 3 and np.any(y_post > 0):
        mask2 = y_post > 0
        x = w_post[mask2].astype(float)
        yy = np.log(np.clip(y_post[mask2].astype(float), 1e-12, None))
        if len(x) >= 3:
            decay_log_slope = safe_linreg_slope(x, yy)

    # Normalized AUC over observed time only:
    # scale by (t_last) so tracks with shorter histories don't become "all zeros".
    auc_norm_time = float(np.trapezoid(y_norm, w) / max(float(w.max()), 1.0))

    return {
        "peak_volume_obs": peak_y,
        "peak_week_obs": peak_week_obs,
        "y_week1_norm": y_week1_norm,
        "y_last_norm": y_last_norm,
        "half_life_week": half_life_week,
        "decay_log_slope": decay_log_slope,
        "auc_norm_time": auc_norm_time,
        "tail_volume_obs": tail_volume_obs,
    }


def build_feature_table(
    df: pd.DataFrame,
    horizon_weeks: int,
    max_tracks: Optional[int],
    random_state: int,
) -> pd.DataFrame:
    track_ids = df["MRELG_ID"].unique()
    if max_tracks is not None:
        rng = np.random.default_rng(random_state)
        if len(track_ids) > max_tracks:
            track_ids = rng.choice(track_ids, size=max_tracks, replace=False)

    # Track-level metadata: 1 row per release.
    meta_cols = ["TITLE", "DISPLAY_ARTIST", "GENRES", "FIRST_SALE_DATE"]
    track_meta = (
        df.groupby("MRELG_ID", sort=False)[meta_cols]
        .agg("first")
        .reset_index()
    )

    records = []
    # Looping is slower but keeps memory usage reasonable.
    df_min = df[["MRELG_ID", "week", "TARGET_METRIC"]].copy()
    subset = df_min[df_min["MRELG_ID"].isin(track_ids)]

    for i, (mrelg_id, g) in enumerate(subset.groupby("MRELG_ID", sort=False)):
        if i % 5000 == 0 and i > 0:
            print(f"  features: processed {i} tracks...")
        feats = extract_track_features(g, horizon_weeks=horizon_weeks)
        if feats is None:
            continue
        feats["MRELG_ID"] = mrelg_id
        records.append(feats)

    features = pd.DataFrame.from_records(records)
    if features.empty or "MRELG_ID" not in features.columns:
        raise ValueError(
            "No releases produced valid feature rows for archetype clustering. "
            f"Checked {len(track_ids):,} releases; 0 had a positive peak on TARGET_METRIC. "
            "Common for singles panels where PRODUCT_SALES or SONG_SALE_EQUIVALENT are all zero — "
            "train only --metric streaming_equivalent, or verify the parquet column has non-zero values."
        )
    features = features.merge(track_meta, on="MRELG_ID", how="left")

    features["main_genre"] = features["GENRES"].apply(extract_main_genre)
    features["FIRST_SALE_DATE"] = pd.to_datetime(features["FIRST_SALE_DATE"], errors="coerce")
    features["release_month"] = features["FIRST_SALE_DATE"].dt.month
    features["release_quarter"] = features["FIRST_SALE_DATE"].dt.quarter

    return features


def fit_archetype_clusters(
    features: pd.DataFrame,
    n_clusters: int,
    random_state: int,
    batch_size: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], StandardScaler, SimpleImputer, MiniBatchKMeans]:
    feature_cols = [
        "peak_week_obs",
        "y_week1_norm",
        "y_last_norm",
        "half_life_week",
        "decay_log_slope",
        "auc_norm_time",
    ]

    X = features[feature_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init="auto",
    )
    km.fit(X_scaled)

    out = features.copy()
    out["Archetype_Cluster"] = km.labels_

    model_info: Dict[str, Any] = {
        "feature_cols": feature_cols,
        "n_clusters": n_clusters,
        "random_state": random_state,
    }
    return out, model_info, scaler, imputer, km


def fit_archetype_curves(
    df: pd.DataFrame,
    features_with_clusters: pd.DataFrame,
    horizon_weeks: int,
    n_clusters: int,
) -> Dict[str, Any]:
    """
    Fit one parametric normalized curve per cluster using only observed week-rows.

    We:
      - normalize each release by its observed peak
      - compute the median normalized curve per cluster per observed week
      - fit gamma_norm(t; a, t_peak) to those observed-median points
    """
    cluster_map = (
        features_with_clusters.set_index("MRELG_ID")["Archetype_Cluster"].to_dict()
    )
    df2 = df.copy()
    df2["Archetype_Cluster"] = df2["MRELG_ID"].map(cluster_map)
    df2 = df2.dropna(subset=["Archetype_Cluster"]).copy()
    df2["Archetype_Cluster"] = df2["Archetype_Cluster"].astype(int)

    # Normalize each release by its observed peak in the data window.
    peak_per_track = df2.groupby("MRELG_ID")["TARGET_METRIC"].transform("max")
    peak_per_track = peak_per_track.replace(0, np.nan)
    df2["y_norm"] = df2["TARGET_METRIC"] / peak_per_track

    # Median aggregation reduces outlier skew in weekly stream levels.
    cluster_week_median = (
        df2.groupby(["Archetype_Cluster", "week"], sort=False)["y_norm"]
        .median()
        .reset_index(name="y_median_norm")
    )

    params: Dict[str, Any] = {}

    # Fit each cluster.
    t_grid = np.arange(1, horizon_weeks + 1)

    for c in range(n_clusters):
        sub = cluster_week_median[cluster_week_median["Archetype_Cluster"] == c].copy()
        sub = sub.dropna(subset=["y_median_norm"])

        if len(sub) < 5:
            continue

        t_fit = sub["week"].to_numpy(dtype=float)
        y_fit = sub["y_median_norm"].to_numpy(dtype=float)

        # Scale so the curve peak is comparable.
        y_max = float(np.nanmax(y_fit))
        if not np.isfinite(y_max) or y_max <= 0:
            continue
        y_fit_scaled = y_fit / y_max

        # Initial guesses
        peak_idx = int(np.nanargmax(y_fit_scaled))
        t_peak0 = float(t_fit[peak_idx])
        a0 = 2.0

        # Bounds: a>0, t_peak in [1, horizon]
        try:
            popt, _pcov = curve_fit(
                f=gamma_norm,
                xdata=t_fit,
                ydata=y_fit_scaled,
                p0=[a0, t_peak0],
                bounds=([0.2, 1.0], [20.0, float(horizon_weeks)]),
                maxfev=20000,
            )
            a_hat, t_peak_hat = popt
        except Exception as e:
            print(f"  curve_fit failed for cluster {c}: {repr(e)}")
            a_hat, t_peak_hat = a0, t_peak0

        params[str(c)] = {
            "a": float(a_hat),
            "t_peak": float(t_peak_hat),
        }

    if len(params) != n_clusters:
        print(f"  Warning: only fitted {len(params)}/{n_clusters} clusters.")
    return params


def compute_artist_alignment(
    features_with_clusters: pd.DataFrame,
    archetype_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
      - artist_stats (median peak volume, median peak week)
      - artist_cluster_probs (mixture distribution over clusters per artist)
    """
    # Artist stats
    artist_stats = (
        features_with_clusters.groupby("DISPLAY_ARTIST", sort=False)
        .agg(
            n_releases=("MRELG_ID", "count"),
            median_peak_volume=("peak_volume_obs", "median"),
            median_peak_week=("peak_week_obs", "median"),
            median_tail_volume=("tail_volume_obs", "median"),
        )
        .reset_index()
    )

    # Artist cluster mixture
    artist_cluster_probs = (
        features_with_clusters.groupby(["DISPLAY_ARTIST", "Archetype_Cluster"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    artist_cluster_probs["prob"] = (
        artist_cluster_probs.groupby("DISPLAY_ARTIST")["count"]
        .transform(lambda x: x / x.sum())
    )

    # Keep only clusters we actually fitted
    fitted_clusters = set(map(int, archetype_params.keys()))
    artist_cluster_probs = artist_cluster_probs[artist_cluster_probs["Archetype_Cluster"].isin(fitted_clusters)]

    return artist_stats, artist_cluster_probs


def compute_artist_genre_alignment(
    features_with_clusters: pd.DataFrame,
    archetype_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute artist + main_genre + cluster mixture distribution.
    """
    fitted_clusters = set(map(int, archetype_params.keys()))
    f = features_with_clusters[features_with_clusters["Archetype_Cluster"].isin(fitted_clusters)].copy()

    artist_genre_cluster_probs = (
        f.groupby(["DISPLAY_ARTIST", "main_genre", "Archetype_Cluster"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )

    artist_genre_cluster_probs["prob"] = (
        artist_genre_cluster_probs.groupby(["DISPLAY_ARTIST", "main_genre"])["count"]
        .transform(lambda x: x / x.sum())
    )

    return artist_genre_cluster_probs


# 18 months ≈ 78 weekly buckets (matches training horizon default).
LIFECYCLE_WEEKS_18MO = 78


def parse_drop_date(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None or not str(s).strip():
        return None
    ts = pd.to_datetime(str(s).strip(), errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid --drop-date: {s!r} (use YYYY-MM-DD)")
    return ts.normalize()


def weeks_from_drop_to_end_of_year(drop_date: pd.Timestamp, horizon_cap: int) -> int:
    """Weeks from drop date through Dec 31 of the same calendar year (inclusive spans)."""
    y = int(drop_date.year)
    year_end = pd.Timestamp(year=y, month=12, day=31)
    if drop_date > year_end:
        return 1
    days = (year_end - drop_date).days + 1
    w = int(np.ceil(days / 7.0))
    return int(max(1, min(horizon_cap, w)))


def resolve_output_weeks(
    *,
    drop_date: Optional[pd.Timestamp],
    forecast_target: str,
    end_week_manual: int,
    horizon_cap: int,
) -> Tuple[int, Dict[str, Any]]:
    """
    forecast_target: 'manual' | 'end-of-year' | 'lifecycle'
    """
    meta: Dict[str, Any] = {"forecast_target": forecast_target}
    if forecast_target == "lifecycle":
        w = min(horizon_cap, LIFECYCLE_WEEKS_18MO)
        meta["end_week_computed"] = w
        meta["note"] = "18-month lifecycle (78 weeks, capped by model horizon)"
        return w, meta
    if forecast_target == "end-of-year":
        if drop_date is None or pd.isna(drop_date):
            raise ValueError("--drop-date is required when --forecast-target is end-of-year")
        w = weeks_from_drop_to_end_of_year(drop_date, horizon_cap)
        meta["end_week_computed"] = w
        meta["year_end"] = f"{int(drop_date.year)}-12-31"
        meta["note"] = f"Weeks from drop through Dec 31 {int(drop_date.year)}"
        return w, meta
    w = int(max(1, min(end_week_manual, horizon_cap)))
    meta["end_week_computed"] = w
    return w, meta


def add_week_ending_column(df: pd.DataFrame, drop_date: pd.Timestamp) -> pd.DataFrame:
    """Week k ends drop_date + 7*k days (week 1 = first week since drop)."""
    out = df.copy()
    wk = out["week"].astype(int).to_numpy()
    out["week_ending"] = (drop_date + pd.to_timedelta(wk * 7, unit="d")).strftime("%Y-%m-%d")
    return out


def cli_resolve_end_week(args: argparse.Namespace, horizon_cap: int) -> Tuple[int, Optional[pd.Timestamp], Dict[str, Any]]:
    dd = parse_drop_date(getattr(args, "drop_date", None))
    ft = str(getattr(args, "forecast_target", "manual") or "manual")
    ew_manual = int(getattr(args, "end_week", horizon_cap) or horizon_cap)
    end_week, meta = resolve_output_weeks(
        drop_date=dd, forecast_target=ft, end_week_manual=ew_manual, horizon_cap=horizon_cap
    )
    return end_week, dd, meta


def slice_forecast_output(
    pred: pd.DataFrame,
    summary: Dict[str, Any],
    end_week: int,
    drop_date: Optional[pd.Timestamp],
    forecast_meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    end_week = int(min(end_week, len(pred)))
    pred2 = pred.iloc[:end_week].copy()
    pred2["cumulative_pred_streams"] = np.cumsum(pred2["pred_weekly_streams"].to_numpy(dtype=float))
    if drop_date is not None and not pd.isna(drop_date):
        pred2 = add_week_ending_column(pred2, drop_date)
    summary = {**summary, **forecast_meta}
    summary["output_weeks"] = end_week
    if drop_date is not None and not pd.isna(drop_date):
        summary["drop_date"] = str(drop_date.date())
    summary["total_lifecycle_pred_streams"] = float(pred2["cumulative_pred_streams"].iloc[-1])
    return pred2, summary


def compute_scenario_multipliers(
    df: pd.DataFrame,
    features_with_clusters: pd.DataFrame,
    *,
    mature_weeks: int = 52,
    bear_pct: float = 0.20,
    base_pct: float = 0.50,
    bull_pct: float = 0.80,
) -> Dict[str, Dict[str, float]]:
    """Compute empirical Bear/Base/Bull multipliers per Archetype_Cluster.

    For every release that has been observed for at least ``mature_weeks`` weeks,
    we compute its empirical lifecycle multiplier as

        empirical_multiplier = sum(TARGET_METRIC over weeks 1..mature_weeks)
                               / peak_volume_obs

    i.e. the cumulative volume in the first ``mature_weeks`` expressed in units
    of the release's observed peak. Per ``Archetype_Cluster`` we then take the
    {bear_pct, base_pct, bull_pct} percentiles (defaults 20/50/80) of that
    distribution. The resulting nested dict is metric-specific (e.g. it will
    differ between worldwide_streams and product_sales artifacts) and is
    consumed downstream as a post-fit shock anchored to the Base percentile
    (so the asymptotic floor is never disturbed by scenario choice).

    Returns a dict shaped like::

        {
            "0": {"Bear": ..., "Base": ..., "Bull": ...},
            "1": {"Bear": ..., "Base": ..., "Bull": ...},
            ...
        }

    Clusters with insufficient mature releases (<2 samples) are omitted; callers
    should fall back to a global table when that happens.
    """
    if "week" not in df.columns or "TARGET_METRIC" not in df.columns:
        raise ValueError(
            "compute_scenario_multipliers requires 'week' and 'TARGET_METRIC' "
            "columns on df (run compute_week_index first)."
        )

    needed = {"MRELG_ID", "Archetype_Cluster", "peak_volume_obs"}
    missing = needed - set(features_with_clusters.columns)
    if missing:
        raise ValueError(
            f"compute_scenario_multipliers requires columns on features_with_clusters: "
            f"{sorted(missing)} are missing."
        )

    max_week_per_release = (
        df.groupby("MRELG_ID", sort=False)["week"].max().rename("max_observed_week")
    )
    mature_ids = max_week_per_release.index[max_week_per_release >= int(mature_weeks)]
    if len(mature_ids) == 0:
        return {}

    df_mature = df[df["MRELG_ID"].isin(mature_ids) & (df["week"] <= int(mature_weeks))]
    sum_to_horizon = (
        df_mature.groupby("MRELG_ID", sort=False)["TARGET_METRIC"]
        .sum()
        .rename("sum_to_horizon")
    )

    feats = features_with_clusters[
        ["MRELG_ID", "Archetype_Cluster", "peak_volume_obs"]
    ].copy()
    feats = feats.dropna(subset=["Archetype_Cluster", "peak_volume_obs"])
    feats["Archetype_Cluster"] = feats["Archetype_Cluster"].astype(int)

    merged = feats.merge(sum_to_horizon, on="MRELG_ID", how="inner")
    merged = merged[
        np.isfinite(merged["peak_volume_obs"]) & (merged["peak_volume_obs"] > 0)
    ]
    if merged.empty:
        return {}

    merged["empirical_multiplier"] = (
        merged["sum_to_horizon"].astype(float) / merged["peak_volume_obs"].astype(float)
    )
    merged = merged[np.isfinite(merged["empirical_multiplier"])]
    if merged.empty:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for cluster_id, sub in merged.groupby("Archetype_Cluster", sort=True):
        vals = sub["empirical_multiplier"].to_numpy(dtype=float)
        if vals.size < 2:
            continue
        bear_v, base_v, bull_v = np.quantile(vals, [bear_pct, base_pct, bull_pct])
        out[str(int(cluster_id))] = {
            "Bear": float(bear_v),
            "Base": float(base_v),
            "Bull": float(bull_v),
        }
    return out


@dataclass
class SimulatorArtifacts:
    horizon_weeks: int
    archetype_params: Dict[str, Any]
    artist_stats: pd.DataFrame
    artist_cluster_probs: pd.DataFrame
    artist_genre_cluster_probs: Optional[pd.DataFrame] = None
    artist_release_history: Optional[pd.DataFrame] = None
    # Empirical Bear/Base/Bull multipliers per Archetype_Cluster, learned from the
    # training parquet for this metric (e.g. worldwide_streams). Keyed by
    # str(cluster_id), inner keys "Bear" / "Base" / "Bull". Populated by train()
    # via compute_scenario_multipliers and persisted as scenario_multipliers.json.
    # When None (older artifacts), callers fall back to the hardcoded global
    # ARCHETYPE_SCENARIO_MULTIPLIERS table.
    scenario_multipliers: Optional[Dict[str, Dict[str, float]]] = None


def resolve_peak_match_column(peak_match_on: str, hist: pd.DataFrame) -> str:
    """
    Which per-release peak to use when selecting a similar-peak subset for archetype mixing.

    - trained_metric: peak_volume_obs from the metric used to train this artifact (e.g. streams AE).
    - album_equivalents: peak_album_equiv_obs when present (from TOTAL_ALBUM_EQUIVALENTS in training parquet).
    """
    mode = (peak_match_on or "trained_metric").strip().lower()
    if mode == "album_equivalents":
        if "peak_album_equiv_obs" in hist.columns and bool(hist["peak_album_equiv_obs"].notna().any()):
            return "peak_album_equiv_obs"
    return "peak_volume_obs"


def simulate_future_drop(
    *,
    artist: str,
    peak_volume: Optional[float],
    peak_week: Optional[float],
    genre: Optional[str],
    artifacts: SimulatorArtifacts,
    stream_floor: Optional[float] = None,
    peak_sim_log_radius: float = 0.35,
    peak_sim_min_subset_releases: int = 10,
    peak_sim_spread_threshold_log_std: float = 0.25,
    peak_sim_min_artist_releases: int = 20,
    peak_match_on: str = "trained_metric",
    peak_match_target: Optional[float] = None,
    scenario: str = "Base",
    archetype_cluster_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    peak_match_target: optional volume used only for similar-peak subset selection (log-radius).
    When None, uses peak_volume. Set explicitly when peak_match_on=album_equivalents so the
    target is on the same scale as peak_album_equiv_obs (e.g. sum of expected weekly AE peaks).
    """

    # Resolve Canonical Artist Name
    artist_canon = artist
    if artifacts.artist_stats is not None:
        match = artifacts.artist_stats[
            artifacts.artist_stats["DISPLAY_ARTIST"].str.lower() == artist.lower()
        ]
        if not match.empty:
            artist_canon = match["DISPLAY_ARTIST"].iloc[0]
    artist = artist_canon

    # Defaults for amplitude & timing
    if peak_volume is None:
        if artifacts.artist_stats is not None and "median_peak_volume" in artifacts.artist_stats.columns:
            stats_row = artifacts.artist_stats[artifacts.artist_stats["DISPLAY_ARTIST"] == artist]
            if not stats_row.empty:
                peak_volume = float(stats_row["median_peak_volume"].iloc[0])
        if peak_volume is None:
            peak_volume = 1000.0

    if peak_week is None:
        if artifacts.artist_stats is not None and "median_peak_week" in artifacts.artist_stats.columns:
            stats_row = artifacts.artist_stats[artifacts.artist_stats["DISPLAY_ARTIST"] == artist]
            if not stats_row.empty:
                peak_week = float(stats_row["median_peak_week"].iloc[0])
        if peak_week is None:
            peak_week = 1.0

    horizon = int(artifacts.horizon_weeks)
    t = np.arange(1, horizon + 1, dtype=float)

    # Calculate Mixture Probabilities
    probs = pd.DataFrame()

    if artifacts.artist_release_history is not None and not artifacts.artist_release_history.empty:
        hist = artifacts.artist_release_history
        hist = hist[hist["DISPLAY_ARTIST"] == artist].copy()
        if genre is not None and "main_genre" in hist.columns:
            hist = hist[hist["main_genre"] == genre].copy()

        if not hist.empty:
            pcol = resolve_peak_match_column(peak_match_on, hist)
            pk = pd.to_numeric(hist[pcol], errors="coerce")
            hist_m = hist.loc[(pk.notna()) & (pk > 0)].copy()
            if len(hist_m) >= 2:
                peak_obs = np.clip(pk.loc[hist_m.index].astype(float).to_numpy(), 1e-9, None)
                log_peaks = np.log10(peak_obs)
                log_std = float(np.std(log_peaks))
                n_hist = len(hist_m)

                if n_hist >= peak_sim_min_artist_releases and log_std > peak_sim_spread_threshold_log_std:
                    target_for_subset = peak_match_target if peak_match_target is not None else peak_volume
                    target_peak = float(max(target_for_subset, 1e-9))
                    target_log = float(np.log10(target_peak))
                    radius = float(peak_sim_log_radius)
                    subset = pd.DataFrame()
                    sub_hist = hist_m

                    for _ in range(8):
                        pv = pd.to_numeric(sub_hist[pcol], errors="coerce").astype(float).to_numpy()
                        subset = sub_hist[
                            np.abs(np.log10(np.clip(pv, 1e-9, None)) - target_log) <= radius
                        ].copy()
                        if len(subset) >= peak_sim_min_subset_releases or radius > 3.0:
                            break
                        radius *= 1.7

                    if not subset.empty and len(subset) >= 2:
                        counts = subset["Archetype_Cluster"].value_counts().sort_index()
                        probs = (counts / counts.sum()).reset_index()
                        probs.columns = ["Archetype_Cluster", "prob"]

    # Fallback to precomputed mixtures (fast path)
    if probs.empty:
        if genre is not None and artifacts.artist_genre_cluster_probs is not None:
            ag = artifacts.artist_genre_cluster_probs
            sub = ag[(ag["DISPLAY_ARTIST"] == artist) & (ag["main_genre"] == genre)].copy()
            if not sub.empty:
                probs = sub[["Archetype_Cluster", "prob"]].copy()

        if probs.empty and artifacts.artist_cluster_probs is not None:
            probs = artifacts.artist_cluster_probs[
                artifacts.artist_cluster_probs["DISPLAY_ARTIST"] == artist
            ][["Archetype_Cluster", "prob"]].copy()

    if archetype_cluster_id is not None:
        c_ov = int(archetype_cluster_id)
        if str(c_ov) not in artifacts.archetype_params:
            raise ValueError(
                f"archetype_cluster_id={c_ov} is not present in trained archetype parameters."
            )
        probs = pd.DataFrame({"Archetype_Cluster": [c_ov], "prob": [1.0]})

    if probs.empty:
        raise ValueError(f"No fitted archetype mixture available for artist={artist}, genre={genre}")

    probs = probs.dropna(subset=["Archetype_Cluster", "prob"]).copy()
    probs["Archetype_Cluster"] = probs["Archetype_Cluster"].astype(int)
    probs["prob"] = probs["prob"].astype(float)
    probs_sum = float(probs["prob"].sum()) if len(probs) else 0.0
    
    if probs_sum > 0:
        probs["prob"] = probs["prob"] / probs_sum

    # Generate the normalized curve
    scen_lbl = normalize_archetype_scenario_label(scenario)
    y_norm = np.zeros_like(t, dtype=float)
    for _, row in probs.iterrows():
        c = int(row["Archetype_Cluster"])
        p = float(row["prob"])
        par = artifacts.archetype_params.get(str(c))
        if par is None:
            continue
        a = float(par["a"])
        r = archetype_scenario_shape_ratio(c, scen_lbl)
        y_norm += p * r * gamma_norm(t, a=a, t_peak=peak_week)

    # ==========================================
    # --- UNIFIED ASYMPTOTIC FLOOR LOGIC ---
    # ==========================================
    final_floor = 0.0
    
    # 1. Check for manual override first!
    if stream_floor is not None:
        final_floor = float(stream_floor)
    # 2. If blank, calculate dynamic Top-3
    else:
        history_df = getattr(artifacts, "artist_release_history", getattr(artifacts, "release_history", None))
        if history_df is not None:
            hist_floor = history_df[history_df["DISPLAY_ARTIST"] == artist].copy()
            if not hist_floor.empty and "tail_volume_obs" in hist_floor.columns and "peak_volume_obs" in hist_floor.columns:
                hist_floor = hist_floor.dropna(subset=["tail_volume_obs", "peak_volume_obs"])
                
            if not hist_floor.empty:
                hist_floor["peak_diff"] = (hist_floor["peak_volume_obs"] - peak_volume).abs()
                top3 = hist_floor.sort_values("peak_diff").head(3).copy()
                top3["retention_ratio"] = top3["tail_volume_obs"] / top3["peak_volume_obs"].replace(0, 1e-9)
                median_ratio = float(top3["retention_ratio"].median())
                final_floor = float(peak_volume * median_ratio)
                
            elif "median_tail_volume" in artifacts.artist_stats.columns and "median_peak_volume" in artifacts.artist_stats.columns:
                stats_row = artifacts.artist_stats[artifacts.artist_stats["DISPLAY_ARTIST"] == artist]
                if not stats_row.empty and pd.notna(stats_row["median_tail_volume"].iloc[0]):
                    median_tail = float(stats_row["median_tail_volume"].iloc[0])
                    median_peak = float(stats_row["median_peak_volume"].iloc[0])
                    median_ratio = median_tail / max(median_peak, 1e-9)
                    final_floor = float(peak_volume * median_ratio)

    # 3. Apply final math
    if peak_volume > final_floor:
        amplitude = float(peak_volume - final_floor)
    else:
        amplitude = 0.0

    y_streams = (amplitude * y_norm) + final_floor
    y_streams = np.clip(y_streams, 0, None)

    # Format Output
    weeks = np.arange(1, horizon + 1, dtype=int)
    out = pd.DataFrame({"week": weeks, "pred_weekly_streams": y_streams})
    out["cumulative_pred_streams"] = np.cumsum(out["pred_weekly_streams"].to_numpy(dtype=float))

    total_pred = float(out["cumulative_pred_streams"].iloc[-1])
    summary = {
        "artist": artist,
        "genre_used": genre if genre is not None else "ALL",
        "peak_volume": peak_volume,
        "peak_week": peak_week,
        "peak_match_on": peak_match_on,
        "peak_match_target": peak_match_target if peak_match_target is not None else peak_volume,
        "total_lifecycle_pred_streams": total_pred,
        "cluster_probs": probs.sort_values("prob", ascending=False)[["Archetype_Cluster", "prob"]].to_dict(orient="records"),
        "scenario_applied": scen_lbl,
        "archetype_cluster_override": archetype_cluster_id,
    }
    
    return out, summary


def simulate_future_drop_average(
    *,
    artists: List[str],
    peak_volume: Optional[float],
    peak_week: Optional[float],
    genre: Optional[str],
    artifacts: SimulatorArtifacts,
    stream_floor: Optional[float] = None,
    peak_sim_log_radius: float = 0.35,
    peak_sim_min_subset_releases: int = 10,
    peak_sim_spread_threshold_log_std: float = 0.25,
    peak_sim_min_artist_releases: int = 20,
    peak_match_on: str = "trained_metric",
    peak_match_target: Optional[float] = None,
    scenario: str = "Base",
    archetype_cluster_id: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Average simulated curves over multiple DISPLAY_ARTIST names."""
    if not artists:
        raise ValueError("artists list is empty")

    preds: List[np.ndarray] = []
    summaries: List[Dict[str, Any]] = []
    used: List[str] = []
    skipped: List[Dict[str, Any]] = []
    not_in_training: List[str] = []
    resolved: List[Dict[str, Any]] = []
    pred0: Optional[pd.DataFrame] = None

    available_lower = set(artifacts.artist_stats["DISPLAY_ARTIST"].astype(str).str.lower().unique().tolist())

    for a in artists:
        ai = str(a).strip()
        if ai.lower() not in available_lower:
            not_in_training.append(ai)
        try:
            pred_i, summ_i = simulate_future_drop(
                artist=a,
                peak_volume=peak_volume,
                peak_week=peak_week,
                genre=genre,
                artifacts=artifacts,
                stream_floor=stream_floor,
                peak_sim_log_radius=peak_sim_log_radius,
                peak_sim_min_subset_releases=peak_sim_min_subset_releases,
                peak_sim_spread_threshold_log_std=peak_sim_spread_threshold_log_std,
                peak_sim_min_artist_releases=peak_sim_min_artist_releases,
                peak_match_on=peak_match_on,
                peak_match_target=peak_match_target,
                scenario=scenario,
                archetype_cluster_id=archetype_cluster_id,
            )
        except Exception as e:
            skipped.append({"artist": a, "reason": str(e)[:400]})
            resolved.append({"input": ai, "resolved": [], "status": "skipped", "reason": str(e)[:250]})
            continue

        if pred0 is None:
            pred0 = pred_i.copy()
        preds.append(pred_i["pred_weekly_streams"].to_numpy(dtype=float))
        summaries.append(summ_i)
        used.append(a)
        # Summaries may either contain:
        # - "artist" (canonical training name) for exact/single resolution
        # - "artists_used" (list of canonical names) when we averaged during fuzzy mapping
        if isinstance(summ_i, dict) and "artists_used" in summ_i and isinstance(summ_i.get("artists_used"), list):
            res_list = [str(x) for x in summ_i["artists_used"]]
        else:
            res_list = [str(summ_i.get("artist", ai))]
        resolved.append({"input": ai, "resolved": res_list, "status": "used"})

    if not preds:
        raise RuntimeError(
            "All provided --artist-candidates failed to simulate. "
            f"Skipped: {[d.get('artist') for d in skipped[:5]]}"
        )

    avg_streams = np.mean(np.stack(preds, axis=0), axis=0)
    out = pred0.copy() if pred0 is not None else pd.DataFrame()
    out["pred_weekly_streams"] = avg_streams
    out["cumulative_pred_streams"] = np.cumsum(avg_streams)

    # Average cluster probabilities across candidates (renormalize).
    cluster_map: Dict[int, List[float]] = {}
    for summ in summaries:
        for d in summ.get("cluster_probs", []) or []:
            c = int(d.get("Archetype_Cluster"))
            p = float(d.get("prob"))
            cluster_map.setdefault(c, []).append(p)

    cluster_probs_avg: List[Dict[str, Any]] = []
    for c, plist in cluster_map.items():
        cluster_probs_avg.append({"Archetype_Cluster": c, "prob": float(np.mean(plist))})
    cluster_probs_avg = sorted(cluster_probs_avg, key=lambda d: float(d["prob"]), reverse=True)
    prob_sum = float(sum(d["prob"] for d in cluster_probs_avg))
    if prob_sum > 0:
        for d in cluster_probs_avg:
            d["prob"] = float(d["prob"] / prob_sum)

    peak_volume_used = float(peak_volume) if peak_volume is not None else float(np.max(avg_streams))
    peak_week_used = float(peak_week) if peak_week is not None else float(np.mean([s.get("peak_week", np.nan) for s in summaries]))

    summary: Dict[str, Any] = {
        "artist": artists[0],
        "artist_candidates_input": artists,
        "artists_used": used,
        "artist_candidates_used": used,
        "artist_candidates_skipped": skipped,
        "artist_candidates_not_in_training": not_in_training,
        "artist_candidates_resolved": resolved,
        "genre_used": genre if genre is not None else "ALL",
        "peak_volume": peak_volume_used,
        "peak_week": peak_week_used,
        "total_lifecycle_pred_streams": float(out["cumulative_pred_streams"].iloc[-1]),
        "cluster_probs": cluster_probs_avg,
    }
    return out, summary


def _parse_actuals_list(actuals_s: str) -> np.ndarray:
    """
    Parse a comma/space separated list of numeric weekly streams.
    Example: "120, 130, 90" -> array([120,130,90])
    """
    s = (actuals_s or "").strip()
    if not s:
        raise ValueError("actuals list is empty")
    parts = [p for p in re.split(r"[,\s]+", s) if p]
    vals = [float(p) for p in parts]
    if len(vals) < 1:
        raise ValueError("Need at least 1 observed week value in actuals list.")
    return np.asarray(vals, dtype=float)


def _resolve_artist_for_artifacts(artist: str, artifacts: SimulatorArtifacts) -> str:
    """Case-insensitive remap of the user-provided artist to the canonical stored name."""
    artist_in = artist.strip()
    available_artists = artifacts.artist_stats["DISPLAY_ARTIST"].astype(str)
    available_lower = available_artists.str.lower()
    needle_lower = artist_in.lower()

    if needle_lower in set(available_lower):
        canon_match = available_artists[available_lower == needle_lower].iloc[0]
        return str(canon_match)

    # Fallback suggestions (keep it short for usability)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", needle_lower) if tok]
    cand_mask = np.ones(len(available_artists), dtype=bool)
    for tok in tokens[:4]:
        cand_mask &= available_lower.str.contains(tok, na=False)
    candidates = available_artists[cand_mask].unique().tolist()
    close = difflib.get_close_matches(artist_in, available_artists.unique().tolist(), n=10, cutoff=0.55)
    suggestions = (candidates[:10] + close[:10])[:15]
    suggestions = [s for s in suggestions if s and s != artist_in]
    if suggestions:
        best = str(suggestions[0])
        print(f'Fuzzy-artist mapping: "{artist_in}" -> "{best}"')
        return best
    raise ValueError(
        f"Unknown artist: {artist_in}. (Not present in training artifacts.)\n"
        f"Top similar names in artifacts: {suggestions if suggestions else 'N/A'}"
    )


def fit_backfill_forecast(
    *,
    artist: str,
    genre: Optional[str],
    actuals_weekly_streams: np.ndarray,
    artifacts: SimulatorArtifacts,
    end_week: Optional[int] = None,
    peak_week_search: Optional[Tuple[int, int]] = None,
    fit_logspace: bool = True,
    peak_week_margin: int = 3,
    stream_floor: Optional[float] = None,
    mixture_fit_weight: float = 0.5,
    scenario: str = "Base",
    archetype_cluster_id: Optional[int] = None,
    scenario_multiplier: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backfill + forecast:
      - Weeks 1..K: use provided actuals
      - Weeks K+1..end_week: forecast using the fitted archetype mixture decay model

    We fit the artist/genre mixture by estimating:
      - peak_week (t_peak) on a discrete grid
      - peak_volume (amplitude) by least squares scaling

    Forecasting best practice: fitting and scenario shocks are strictly isolated.
    The fit always reflects the Base trajectory and asymptotic floor. After the
    boundary-aligned curve is built, ``scenario_multiplier`` scales the volume
    *above the floor* for unobserved (future) weeks only — actuals and floor
    are untouched.
    """
    horizon = int(artifacts.horizon_weeks)
    if end_week is None:
        end_week = horizon
    end_week = int(min(int(end_week), horizon))

    k = int(len(actuals_weekly_streams))
    if k > end_week:
        raise ValueError("actuals length cannot exceed end_week.")

    # Artist matching: if the artist isn't in training artifacts, fall back to
    # averaging over up to 3 similar DISPLAY_ARTIST candidates.
    artist_in = artist.strip()
    available_artists = artifacts.artist_stats["DISPLAY_ARTIST"].astype(str)
    available_lower = available_artists.str.lower()
    needle_lower = artist_in.lower()

    if needle_lower in set(available_lower):
        canon_match = available_artists[available_lower == needle_lower].iloc[0]
        artist_candidates = [str(canon_match)]
    else:
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", needle_lower) if tok]
        cand_mask = np.ones(len(available_artists), dtype=bool)
        for tok in tokens[:4]:
            cand_mask &= available_lower.str.contains(tok, na=False)

        candidates = available_artists[cand_mask].unique().tolist()
        close = difflib.get_close_matches(artist_in, available_artists.unique().tolist(), n=10, cutoff=0.55)
        suggestions = (candidates[:10] + close[:10])[:15]
        suggestions = [s for s in suggestions if s and s != artist_in]

        if genre is not None and artifacts.artist_release_history is not None and not artifacts.artist_release_history.empty:
            g = str(genre).strip()
            if g:
                hist = artifacts.artist_release_history
                hist_g = hist[hist["main_genre"] == g]
                if not hist_g.empty:
                    allowed = set(hist_g["DISPLAY_ARTIST"].astype(str).unique().tolist())
                    suggestions = [s for s in suggestions if s in allowed]

        if not suggestions:
            raise ValueError(
                f"Unknown artist: {artist_in}. (Not present in training artifacts.)\n"
                f"Top similar names in artifacts: {suggestions if suggestions else 'N/A'}"
            )

        artist_candidates = [str(s) for s in suggestions[:3]]

    if len(artist_candidates) > 1:
        preds = []
        summaries = []
        for cand in artist_candidates:
            pred_c, summ_c = fit_backfill_forecast(
                artist=cand,
                genre=genre,
                actuals_weekly_streams=actuals_weekly_streams,
                artifacts=artifacts,
                end_week=end_week,
                peak_week_search=peak_week_search,
                fit_logspace=fit_logspace,
                peak_week_margin=peak_week_margin,
                stream_floor=stream_floor,
                mixture_fit_weight=mixture_fit_weight,
                scenario=scenario,
                archetype_cluster_id=archetype_cluster_id,
                scenario_multiplier=scenario_multiplier,
            )
            preds.append(pred_c["pred_weekly_streams"].to_numpy(dtype=float))
            summaries.append(summ_c)

        avg_streams = np.mean(np.stack(preds, axis=0), axis=0)
        out = pred_c.copy()
        out["pred_weekly_streams"] = avg_streams
        out["cumulative_pred_streams"] = np.cumsum(avg_streams)

        # Average a few useful summary fields.
        summary = {
            "artist": artist_in,
            "artists_used": artist_candidates,
            "genre_used": genre if genre is not None else "ALL",
            "observed_weeks": k,
            "fit_peak_week": float(np.mean([s.get("fit_peak_week") for s in summaries if "fit_peak_week" in s])),
            "fit_peak_volume": float(np.mean([s.get("fit_peak_volume") for s in summaries if "fit_peak_volume" in s])),
            "fit_sse": float(np.mean([s.get("fit_sse") for s in summaries if "fit_sse" in s])),
            "total_lifecycle_pred_streams": float(out["cumulative_pred_streams"].iloc[-1]),
            "scenario_applied": normalize_archetype_scenario_label(scenario),
            "archetype_cluster_override": archetype_cluster_id,
            "scenario_multiplier": float(scenario_multiplier),
        }
        return out, summary

    artist_canon = artist_candidates[0]

    # Mixture distribution p(cluster)
    probs_df = None
    if genre is not None and artifacts.artist_genre_cluster_probs is not None:
        ag = artifacts.artist_genre_cluster_probs
        sub = ag[(ag["DISPLAY_ARTIST"] == artist_canon) & (ag["main_genre"] == genre)].copy()
        if not sub.empty:
            probs_df = sub[["Archetype_Cluster", "prob"]].copy()

    if probs_df is None or probs_df.empty:
        probs_df = artifacts.artist_cluster_probs[
            artifacts.artist_cluster_probs["DISPLAY_ARTIST"] == artist_canon
        ][["Archetype_Cluster", "prob"]].copy()

    probs_df = probs_df.dropna(subset=["Archetype_Cluster", "prob"]).copy()
    if probs_df.empty:
        raise ValueError(f"No archetype mixture found for artist={artist_canon}, genre={genre}")

    probs_df["Archetype_Cluster"] = probs_df["Archetype_Cluster"].astype(int)

    if archetype_cluster_id is not None:
        c_ov = int(archetype_cluster_id)
        if str(c_ov) not in artifacts.archetype_params:
            raise ValueError(
                f"archetype_cluster_id={c_ov} is not in trained archetype parameters."
            )
        probs_df = pd.DataFrame({"Archetype_Cluster": [c_ov], "prob": [1.0]})

    # Pull 'a' for each cluster we will use in gamma_norm.
    cluster_a: Dict[int, float] = {}
    for _, row in probs_df.iterrows():
        c = int(row["Archetype_Cluster"])
        par = artifacts.archetype_params.get(str(c))
        if par is None:
            continue
        cluster_a[c] = float(par["a"])

    probs_df = probs_df[probs_df["Archetype_Cluster"].isin(cluster_a.keys())].copy()
    if probs_df.empty:
        raise ValueError("No overlapping clusters found between artist mixture and fitted archetype parameters.")

    # Normalize probabilities after potential dropping
    probs_df["prob"] = probs_df["prob"].astype(float)
    probs_sum = float(probs_df["prob"].sum())
    probs_df["prob"] = probs_df["prob"] / probs_sum

    t_obs = np.arange(1, k + 1, dtype=float)
    y_obs = np.clip(actuals_weekly_streams.astype(float), 0, None)

    max_y = float(np.max(y_obs)) if len(y_obs) else 0.0
    if not np.isfinite(max_y) or max_y <= 0:
        raise ValueError(
            "Backfill actuals must contain at least one positive weekly value (finite and > 0)."
        )
    y_norm_target = y_obs / max_y

    observed_peak_week = int(np.argmax(y_obs)) + 1  # week index in 1..K

    if peak_week_search is None:
        lo = 1
        hi = end_week
    else:
        lo, hi = peak_week_search
        lo = max(1, int(lo))
        hi = min(end_week, int(hi))

    # Crucial constraints:
    # 1) peak_week cannot be before the observed peak week (prevents week-1 snap).
    # 2) for young projects, avoid peaks far in the future (prevents rising forecasts).
    # 3) if the observed maximum happens *before* the last observed week, treat that
    #    as the peak and do not allow a later fitted peak (prevents “peak-at-end” behavior).
    lo = max(lo, observed_peak_week)
    if observed_peak_week < k:
        hi = min(hi, observed_peak_week)
    else:
        hi = min(hi, observed_peak_week + max(0, int(peak_week_margin)))
    if hi < lo:
        # Fall back to a minimal feasible range.
        hi = lo

    probs_df = probs_df.sort_values("Archetype_Cluster").reset_index(drop=True)
    cluster_ids = [int(c) for c in probs_df["Archetype_Cluster"].tolist()]
    a_list = [float(cluster_a[c]) for c in cluster_ids]
    p_prior_vec = probs_df["prob"].astype(float).to_numpy(dtype=float)
    C = len(cluster_ids)

    from scipy.optimize import nnls

    log_y = np.log1p(np.clip(y_obs, 0, None)) if fit_logspace else None

    best: Optional[Dict[str, Any]] = None
    best_p: Optional[np.ndarray] = None

    # Grid search over peak week. For each candidate:
    #  - fit the mixture weights p across clusters to match the *normalized shape*
    #  - estimate peak_volume via least squares scaling
    #  - score error (log-space by default) on the original scale
    scen_fit = normalize_archetype_scenario_label(scenario)
    for t_peak in range(lo, hi + 1):
        # G[t, j] = gamma_norm(week_t; a_j, t_peak)
        G = np.zeros((k, C), dtype=float)
        for j, a in enumerate(a_list):
            c_id = cluster_ids[j]
            rj = archetype_scenario_shape_ratio(c_id, scen_fit)
            G[:, j] = gamma_norm(t_obs, a=a, t_peak=float(t_peak)) * rj

        # Fit non-negative mixture weights to match the normalized target curve.
        # NNLS can hit iteration limits for some candidate peak-week settings,
        # so treat those candidates as invalid instead of crashing.
        try:
            p_raw, _rnorm = nnls(G, y_norm_target, maxiter=20000)
        except RuntimeError:
            continue
        p_sum = float(np.sum(p_raw))
        if p_sum <= 0:
            continue
        p_fit = p_raw / p_sum  # enforce mixture weights sum to 1
        # Regularize mixture weights towards the artist's prior mixture.
        # This reduces "hard archetype lock" and lets the fit blend archetypes.
        w = float(np.clip(mixture_fit_weight, 0.0, 1.0))
        p = (1.0 - w) * p_prior_vec + w * p_fit

        y_norm_pred_obs = G @ p

        denom = float(np.sum(y_norm_pred_obs * y_norm_pred_obs))
        if denom <= 0:
            continue

        peak_volume_hat = float(np.sum(y_obs * y_norm_pred_obs) / denom)
        peak_volume_hat = max(0.0, peak_volume_hat)

        y_pred_obs = peak_volume_hat * y_norm_pred_obs
        if fit_logspace:
            sse = float(np.sum((log_y - np.log1p(np.clip(y_pred_obs, 0, None))) ** 2))
        else:
            sse = float(np.sum((y_obs - y_pred_obs) ** 2))

        if best is None or sse < best["sse"]:
            best = {"t_peak": float(t_peak), "peak_volume": peak_volume_hat, "sse": sse}
            best_p = p

    if best is None or best_p is None:
        #raise RuntimeError("Backfill fit failed to find a valid peak_week candidate.")
        best_p = p_prior_vec
        best = {"t_peak": float(observed_peak_week), "sse": float("inf")}
        
        # Analytically calculate the scaling factor (amplitude) for the default curve
        G_fallback = np.zeros((k, C), dtype=float)
        for j, a in enumerate(a_list):
            c_id = cluster_ids[j]
            rj = archetype_scenario_shape_ratio(c_id, scen_fit)
            G_fallback[:, j] = gamma_norm(t_obs, a=a, t_peak=float(observed_peak_week)) * rj
            
        y_norm_fallback = G_fallback @ best_p
        denom = float(np.sum(y_norm_fallback * y_norm_fallback))
        
        if denom > 0:
            best["peak_volume"] = float(np.sum(y_obs * y_norm_fallback) / denom)
        else:
            best["peak_volume"] = float(np.max(y_obs))

    # Build full normalized curve up to end_week using fitted t_peak and fitted mixture weights.
    t_full = np.arange(1, end_week + 1, dtype=float)
    G_full = np.zeros((len(t_full), C), dtype=float)
    for j, a in enumerate(a_list):
        c_id = cluster_ids[j]
        rj = archetype_scenario_shape_ratio(c_id, scen_fit)
        G_full[:, j] = gamma_norm(t_full, a=a, t_peak=float(best["t_peak"])) * rj

    # best_p is the mixture weights used for the observed fit (already regularized).
    #y_norm_full = G_full @ best_p
    #y_pred_full = np.clip(best["peak_volume"] * y_norm_full, 0, None)
    y_norm_full = G_full @ best_p
    fitted_peak = float(best["peak_volume"])
    
    if stream_floor is None:
        dynamic_floor = 0.0
    
    if stream_floor is not None:
        dynamic_floor = float(stream_floor)
    else:
        history_df = getattr(artifacts, "artist_release_history", getattr(artifacts, "release_history", None))
        
        if history_df is not None:
            hist = history_df[history_df["DISPLAY_ARTIST"] == artist_canon].copy()
            if not hist.empty and "tail_volume_obs" in hist.columns and "peak_volume_obs" in hist.columns:
                hist = hist.dropna(subset=["tail_volume_obs", "peak_volume_obs"])
                
            if not hist.empty:
                hist["peak_diff"] = (hist["peak_volume_obs"] - fitted_peak).abs()
                top3 = hist.sort_values("peak_diff").head(3).copy()
                top3["retention_ratio"] = top3["tail_volume_obs"] / top3["peak_volume_obs"].replace(0, 1e-9)
                median_ratio = float(top3["retention_ratio"].median())
                dynamic_floor = float(fitted_peak * median_ratio)
                
            elif "median_tail_volume" in artifacts.artist_stats.columns and "median_peak_volume" in artifacts.artist_stats.columns:
                stats_row = artifacts.artist_stats[artifacts.artist_stats["DISPLAY_ARTIST"] == artist_canon]
                if not stats_row.empty and pd.notna(stats_row["median_tail_volume"].iloc[0]):
                    median_tail = float(stats_row["median_tail_volume"].iloc[0])
                    median_peak = float(stats_row["median_peak_volume"].iloc[0])
                    median_ratio = median_tail / max(median_peak, 1e-9)
                    dynamic_floor = float(fitted_peak * median_ratio)

    # --- APPLY ASYMPTOTIC FLOOR MATH ---
    if fitted_peak > dynamic_floor:
        amplitude = float(fitted_peak - dynamic_floor)
    else:
        amplitude = 0.0
        
    y_pred_full = (amplitude * y_norm_full) + dynamic_floor
    y_pred_full = np.clip(y_pred_full, 0, None)

    # Boundary alignment:
    # We overwrite weeks 1..K with actuals in the output, but forecasts (K+1..)
    # depend on the fitted curve. If the fitted curve doesn't match the last
    # observed week value, the forecast can jump unnaturally.
    #
    # IMPORTANT: do not multiply the whole series by scale when dynamic_floor > 0
    # (including a user hard-coded floor). That would shrink the floor to
    # dynamic_floor * scale and allow values *below* the intended minimum.
    # Instead, rescale only the portion above the floor (affine about the floor).
    best_peak_volume_scaled = float(best["peak_volume"])
    if k >= 1:
        pred_k = float(y_pred_full[k - 1])
        y_k = float(y_obs[-1])
        F = float(dynamic_floor)
        above = pred_k - F
        if above > 1e-12 and np.isfinite(y_k):
            scale_k = (y_k - F) / above
            # Negative scale would invert the decay above the floor; skip in that edge case.
            if np.isfinite(scale_k) and scale_k >= 0.0:
                y_pred_full = F + scale_k * (y_pred_full - F)
                y_pred_full = np.clip(y_pred_full, F, None)
                best_peak_volume_scaled = float(F + scale_k * (best["peak_volume"] - F))
        elif pred_k > 1e-12 and np.isfinite(y_k) and F <= 1e-12:
            # Pure multiplicative alignment when floor is ~0 (original behavior).
            scale = float(y_k / pred_k)
            y_pred_full = np.clip(y_pred_full * scale, 0, None)
            best_peak_volume_scaled = float(best["peak_volume"] * scale)

    # --- APPLY SCENARIO SHOCK (Future Weeks Only) ---
    # The fit above is deliberately scenario-free so that the trajectory and
    # asymptotic floor reflect the Base reality of the actuals. Bear/Bull
    # scenarios are layered in here as a multiplicative shock on the volume
    # *above the floor* for unobserved weeks K+1..end_week. Observed weeks
    # (and the floor itself) are left untouched, preserving boundary
    # continuity at week K and the empirical retention floor.
    if scenario_multiplier != 1.0:
        future_above_floor = y_pred_full[k:] - dynamic_floor
        future_above_floor = np.clip(future_above_floor, 0, None)
        y_pred_full[k:] = dynamic_floor + (future_above_floor * scenario_multiplier)

    # Backfill actuals for observed weeks
    y_out = y_pred_full.copy()
    y_out[:k] = y_obs

    out = pd.DataFrame({"week": np.arange(1, end_week + 1, dtype=int), "pred_weekly_streams": y_out})
    out["cumulative_pred_streams"] = np.cumsum(out["pred_weekly_streams"].to_numpy(dtype=float))

    summary = {
        "artist": artist_canon,
        "genre_used": genre if genre is not None else "ALL",
        "observed_weeks": k,
        "fit_peak_week": best["t_peak"],
        "fit_peak_volume": best_peak_volume_scaled,
        "fit_sse": best["sse"],
        "total_lifecycle_pred_streams": float(out["cumulative_pred_streams"].iloc[-1]),
        "scenario_applied": scen_fit,
        "archetype_cluster_override": archetype_cluster_id,
        "scenario_multiplier": float(scenario_multiplier),
    }
    return out, summary


def fit_backfill_forecast_average(
    *,
    artists: List[str],
    genre: Optional[str],
    actuals_weekly_streams: np.ndarray,
    artifacts: SimulatorArtifacts,
    end_week: Optional[int] = None,
    peak_week_search: Optional[Tuple[int, int]] = None,
    fit_logspace: bool = True,
    peak_week_margin: int = 3,
    stream_floor: Optional[float] = None,
    mixture_fit_weight: float = 0.5,
    scenario_multiplier: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Average backfill+forecast curves over multiple DISPLAY_ARTIST names.

    ``scenario_multiplier`` is forwarded to each per-artist fit and applied as a
    post-fit Bear/Bull shock to the future portion of the curve relative to the
    asymptotic floor. The fit itself is always scenario-free.
    """
    if not artists:
        raise ValueError("artists list is empty")

    preds: List[np.ndarray] = []
    summaries: List[Dict[str, Any]] = []
    used: List[str] = []
    skipped: List[Dict[str, Any]] = []
    not_in_training: List[str] = []
    resolved: List[Dict[str, Any]] = []
    pred0: Optional[pd.DataFrame] = None

    available_lower = set(artifacts.artist_stats["DISPLAY_ARTIST"].astype(str).str.lower().unique().tolist())

    for a in artists:
        ai = str(a).strip()
        if ai.lower() not in available_lower:
            not_in_training.append(ai)
        try:
            pred_i, summ_i = fit_backfill_forecast(
                artist=a,
                genre=genre,
                actuals_weekly_streams=actuals_weekly_streams,
                artifacts=artifacts,
                end_week=end_week,
                peak_week_search=peak_week_search,
                stream_floor=stream_floor,
                fit_logspace=fit_logspace,
                peak_week_margin=peak_week_margin,
                mixture_fit_weight=mixture_fit_weight,
                scenario_multiplier=scenario_multiplier,
            )
        except Exception as e:
            skipped.append({"artist": a, "reason": str(e)[:400]})
            resolved.append({"input": ai, "resolved": [], "status": "skipped", "reason": str(e)[:250]})
            continue

        if pred0 is None:
            pred0 = pred_i.copy()
        preds.append(pred_i["pred_weekly_streams"].to_numpy(dtype=float))
        summaries.append(summ_i)
        used.append(a)
        if isinstance(summ_i, dict) and "artists_used" in summ_i and isinstance(summ_i.get("artists_used"), list):
            res_list = [str(x) for x in summ_i["artists_used"]]
        else:
            res_list = [str(summ_i.get("artist", a))]
        resolved.append({"input": ai, "resolved": res_list, "status": "used"})

    if not preds:
        raise RuntimeError(
            "All provided --artist-candidates failed to backfill/fit. "
            f"Skipped: {[d.get('artist') for d in skipped[:5]]}"
        )

    avg_streams = np.mean(np.stack(preds, axis=0), axis=0)
    out = pred0.copy() if pred0 is not None else pd.DataFrame()
    out["pred_weekly_streams"] = avg_streams
    out["cumulative_pred_streams"] = np.cumsum(avg_streams)

    summary: Dict[str, Any] = {
        "artist": artists[0],
        "artist_candidates_input": artists,
        "artists_used": used,
        "artist_candidates_used": used,
        "artist_candidates_skipped": skipped,
        "artist_candidates_not_in_training": not_in_training,
        "artist_candidates_resolved": resolved,
        "genre_used": genre if genre is not None else "ALL",
        "observed_weeks": len(actuals_weekly_streams),
        "fit_peak_week": float(np.mean([s.get("fit_peak_week", np.nan) for s in summaries])),
        "fit_peak_volume": float(np.mean([s.get("fit_peak_volume", np.nan) for s in summaries])),
        "fit_sse": float(np.mean([s.get("fit_sse", np.nan) for s in summaries])),
        "total_lifecycle_pred_streams": float(out["cumulative_pred_streams"].iloc[-1]),
        "scenario_multiplier": float(scenario_multiplier),
    }
    return out, summary


def backfill(args: argparse.Namespace) -> None:
    sales_artifacts = load_artifacts(args.sales_dir)
    streams_artifacts = load_artifacts(args.streams_dir)
    songs_artifacts = load_artifacts(args.songs_dir)

    horizon = int(streams_artifacts.horizon_weeks)
    end_week_cap, drop_d, fmeta = cli_resolve_end_week(args, horizon)
    end_week = int(min(int(end_week_cap), horizon))

    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    sales_floor = getattr(args, "sales_floor", None)
    songs_floor = getattr(args, "songs_floor", None)

    if args.total_actuals:
        total_actuals = _parse_actuals_list(args.total_actuals)
        K = len(total_actuals)

        pm_on = getattr(args, "peak_match_on", "trained_metric")
        pm_tgt = getattr(args, "peak_match_target", None)
        if pm_on == "album_equivalents" and pm_tgt is None:
            sp = getattr(args, "sales_peak", None)
            st = getattr(args, "streams_peak", None)
            ss = getattr(args, "songs_peak", None)
            if sp is not None and st is not None and ss is not None:
                pm_tgt = float(sp + st + ss)

        # Expected peaks (optional): scale each channel's baseline so simple-mode total-AE
        # splitting reflects user intent; None keeps median_peak_* from artifacts per channel.
        pv_sales = getattr(args, "sales_peak", None)
        pv_streams = getattr(args, "streams_peak", None)
        pv_songs = getattr(args, "songs_peak", None)
        pw_streams = float(getattr(args, "streams_peak_week", None) or 1.0)

        # 1. Generate default shape baselines to calculate historical proportions
        ps_r = float(getattr(args, "peak_sim_log_radius", 0.35))
        ps_min = int(getattr(args, "peak_sim_min_subset_releases", 10))
        ps_std = float(getattr(args, "peak_sim_spread_threshold_log_std", 0.25))
        ps_art = int(getattr(args, "peak_sim_min_artist_releases", 20))

        if len(artists) == 1:
            base_sal, _ = simulate_future_drop(
                artist=artists[0], peak_volume=pv_sales, peak_week=1.0, genre=args.genre, artifacts=sales_artifacts,
                stream_floor=sales_floor if sales_floor is not None else 0.0,
                peak_match_on=pm_on, peak_match_target=pm_tgt,
                peak_sim_log_radius=ps_r,
                peak_sim_min_subset_releases=ps_min,
                peak_sim_spread_threshold_log_std=ps_std,
                peak_sim_min_artist_releases=ps_art,
            )
            base_str, _ = simulate_future_drop(
                artist=artists[0], peak_volume=pv_streams, peak_week=pw_streams, genre=args.genre, artifacts=streams_artifacts,
                stream_floor=args.streams_floor,
                peak_match_on=pm_on, peak_match_target=pm_tgt,
                peak_sim_log_radius=ps_r,
                peak_sim_min_subset_releases=ps_min,
                peak_sim_spread_threshold_log_std=ps_std,
                peak_sim_min_artist_releases=ps_art,
            )
            base_sng, _ = simulate_future_drop(
                artist=artists[0], peak_volume=pv_songs, peak_week=1.0, genre=args.genre, artifacts=songs_artifacts,
                stream_floor=songs_floor if songs_floor is not None else 0.0,
                peak_match_on=pm_on, peak_match_target=pm_tgt,
                peak_sim_log_radius=ps_r,
                peak_sim_min_subset_releases=ps_min,
                peak_sim_spread_threshold_log_std=ps_std,
                peak_sim_min_artist_releases=ps_art,
            )
        else:
            base_sal, _ = simulate_future_drop_average(
                artists=artists, peak_volume=pv_sales, peak_week=1.0, genre=args.genre, artifacts=sales_artifacts,
                stream_floor=sales_floor if sales_floor is not None else 0.0,
                peak_match_on=pm_on, peak_match_target=pm_tgt,
                peak_sim_log_radius=ps_r,
                peak_sim_min_subset_releases=ps_min,
                peak_sim_spread_threshold_log_std=ps_std,
                peak_sim_min_artist_releases=ps_art,
            )
            base_str, _ = simulate_future_drop_average(
                artists=artists, peak_volume=pv_streams, peak_week=pw_streams, genre=args.genre, artifacts=streams_artifacts,
                stream_floor=args.streams_floor,
                peak_match_on=pm_on, peak_match_target=pm_tgt,
                peak_sim_log_radius=ps_r,
                peak_sim_min_subset_releases=ps_min,
                peak_sim_spread_threshold_log_std=ps_std,
                peak_sim_min_artist_releases=ps_art,
            )
            base_sng, _ = simulate_future_drop_average(
                artists=artists, peak_volume=pv_songs, peak_week=1.0, genre=args.genre, artifacts=songs_artifacts,
                stream_floor=songs_floor if songs_floor is not None else 0.0,
                peak_match_on=pm_on, peak_match_target=pm_tgt,
                peak_sim_log_radius=ps_r,
                peak_sim_min_subset_releases=ps_min,
                peak_sim_spread_threshold_log_std=ps_std,
                peak_sim_min_artist_releases=ps_art,
            )

        # 2. Extract just the first K weeks of the baselines
        y_sal = base_sal["pred_weekly_streams"].to_numpy(dtype=float)[:K]
        y_str = base_str["pred_weekly_streams"].to_numpy(dtype=float)[:K]
        y_sng = base_sng["pred_weekly_streams"].to_numpy(dtype=float)[:K]

        # 3. Calculate weekly proportions (add 1e-9 to prevent division by zero)
        y_tot = y_sal + y_str + y_sng + 1e-9

        # 4. Slice the user's Total array into three proportionally accurate arrays!
        sales_actuals = total_actuals * (y_sal / y_tot)
        streams_actuals = total_actuals * (y_str / y_tot)
        songs_actuals = total_actuals * (y_sng / y_tot)

    else:
        sales_actuals = _parse_actuals_list(args.sales_actuals)
        streams_actuals = _parse_actuals_list(args.streams_actuals)
        songs_actuals = _parse_actuals_list(args.songs_actuals)

    k = len(sales_actuals)
    if not (len(streams_actuals) == len(songs_actuals) == k):
        raise ValueError("Sales, streams, and songs actuals must all have the same length (K weeks).")
    if k > end_week:
        raise ValueError(
            f"You have {k} weeks of actuals but forecast horizon is only {end_week} weeks "
            f"({fmeta.get('note', '')})."
        )

    # --- 1. FIT SALES ---
    if len(artists) == 1:
        pred_sales, summ_sales = fit_backfill_forecast(
            artist=artists[0], genre=args.genre, actuals_weekly_streams=sales_actuals,
            artifacts=sales_artifacts, end_week=end_week, peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
            stream_floor=sales_floor if sales_floor is not None else 0.0,
        )
    else:
        pred_sales, summ_sales = fit_backfill_forecast_average(
            artists=artists, genre=args.genre, actuals_weekly_streams=sales_actuals,
            artifacts=sales_artifacts, end_week=end_week, peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
            stream_floor=sales_floor if sales_floor is not None else 0.0,
        )

    # --- 2. FIT STREAMS ---
    if len(artists) == 1:
        pred_streams, summ_streams = fit_backfill_forecast(
            artist=artists[0], genre=args.genre, actuals_weekly_streams=streams_actuals,
            artifacts=streams_artifacts, end_week=end_week, peak_week_margin=args.peak_week_margin,
            stream_floor=args.streams_floor,
            mixture_fit_weight=args.mixture_fit_weight,
        )
    else:
        pred_streams, summ_streams = fit_backfill_forecast_average(
            artists=artists, genre=args.genre, actuals_weekly_streams=streams_actuals,
            artifacts=streams_artifacts, end_week=end_week, peak_week_margin=args.peak_week_margin,
            stream_floor=args.streams_floor,
            mixture_fit_weight=args.mixture_fit_weight,
        )
    
    # --- 3. fit song sales ---
    if len(artists) == 1:
        pred_songs, summ_songs = fit_backfill_forecast(
            artist=artists[0], genre=args.genre, actuals_weekly_streams=songs_actuals,
            artifacts=songs_artifacts, end_week=end_week, peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
            stream_floor=songs_floor if songs_floor is not None else 0.0,
        )
    else:
        pred_songs, summ_songs = fit_backfill_forecast_average(
            artists=artists, genre=args.genre, actuals_weekly_streams=songs_actuals,
            artifacts=songs_artifacts, end_week=end_week, peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
            stream_floor=songs_floor if songs_floor is not None else 0.0,
        )

    # --- 3. COMBINE ---
    pred_combined = pd.DataFrame({"week": pred_sales["week"]})
    pred_combined["pred_sales"] = pred_sales["pred_weekly_streams"]
    pred_combined["pred_streams"] = pred_streams["pred_weekly_streams"]
    pred_combined["pred_songs"] = pred_songs["pred_weekly_streams"]
    pred_combined["pred_weekly_equivalents"] = pred_combined["pred_sales"] + pred_combined["pred_streams"] + pred_combined["pred_songs"]
    pred_combined["pred_weekly_streams"] = pred_combined["pred_weekly_equivalents"] # Ghost column for slicing function

    summ_combined = {
        "artist": summ_streams.get("artist", artists[0]),
        "genre_used": args.genre if args.genre is not None else "ALL",
        "observed_weeks": k,
        "fit_sales_peak": summ_sales.get("fit_peak_volume"),
        "fit_streams_peak": summ_streams.get("fit_peak_volume"),
    }

    pred_final, summ_final = slice_forecast_output(pred_combined, summ_combined, end_week, drop_d, fmeta)
    
    # Cleanup display columns
    pred_final["cumulative_pred_equivalents"] = pred_final["cumulative_pred_streams"]
    pred_final = pred_final.drop(columns=["pred_weekly_streams", "cumulative_pred_streams"], errors="ignore")
    summ_final["total_lifecycle_pred_equivalents"] = summ_final.pop("total_lifecycle_pred_streams", 0.0)

    print(json.dumps(summ_final, ensure_ascii=False, indent=2))
    if getattr(args, "out_csv", None) is not None:
        pred_final.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")
    else:
        print(pred_final.to_string(index=False))

def plot_backfill(args: argparse.Namespace) -> None:
    # We route plot_backfill through the standard plot command for now 
    # since we combined the UI rendering logic there. 
    # The true plotting is handled by the frontend catching the CSV output.
    pass


def chi_squared_feature_alignment(
    features_with_clusters: pd.DataFrame,
    feature_col: str,
    target_col: str = "Archetype_Cluster",
) -> Tuple[float, float, float]:
    """
    Returns (p_value, chi2_stat, cramer_v).
    """
    # Import lazily to keep base dependencies small.
    from scipy.stats import chi2_contingency

    table = pd.crosstab(features_with_clusters[feature_col], features_with_clusters[target_col])
    chi2, p_value, _dof, _expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    min_dim = min(table.shape) - 1
    if min_dim <= 0 or n <= 0:
        return p_value, chi2, 0.0
    cramer_v = float(np.sqrt(chi2 / (n * min_dim)))
    return float(p_value), float(chi2), cramer_v


def save_artifacts(
    out_dir: str,
    horizon_weeks: int,
    archetype_params: Dict[str, Any],
    artist_stats: pd.DataFrame,
    artist_cluster_probs: pd.DataFrame,
    artist_genre_cluster_probs: Optional[pd.DataFrame] = None,
    artist_release_history: Optional[pd.DataFrame] = None,
    scenario_multipliers: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "archetype_params.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_weeks": horizon_weeks, "archetype_params": archetype_params}, f, ensure_ascii=False, indent=2)

    artist_stats.to_parquet(os.path.join(out_dir, "artist_stats.parquet"), index=False)
    artist_cluster_probs.to_parquet(os.path.join(out_dir, "artist_cluster_probs.parquet"), index=False)

    if artist_genre_cluster_probs is not None:
        artist_genre_cluster_probs.to_parquet(os.path.join(out_dir, "artist_genre_cluster_probs.parquet"), index=False)

    if artist_release_history is not None:
        artist_release_history.to_parquet(os.path.join(out_dir, "artist_release_history.parquet"), index=False)

    if scenario_multipliers:
        # Per-cluster empirical Bear/Base/Bull multipliers for this metric.
        # Consumed by upstream callers as a post-fit shock; absence is fine and
        # falls back to the hardcoded global table.
        with open(
            os.path.join(out_dir, "scenario_multipliers.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(scenario_multipliers, f, ensure_ascii=False, indent=2)


def load_artifacts(out_dir: str) -> SimulatorArtifacts:
    with open(os.path.join(out_dir, "archetype_params.json"), "r", encoding="utf-8") as f:
        payload = json.load(f)

    horizon_weeks = int(payload["horizon_weeks"])
    archetype_params = payload["archetype_params"]

    artist_stats = pd.read_parquet(os.path.join(out_dir, "artist_stats.parquet"))
    artist_cluster_probs = pd.read_parquet(os.path.join(out_dir, "artist_cluster_probs.parquet"))

    genre_path = os.path.join(out_dir, "artist_genre_cluster_probs.parquet")
    if os.path.exists(genre_path):
        artist_genre_cluster_probs = pd.read_parquet(genre_path)
    else:
        artist_genre_cluster_probs = None

    release_history_path = os.path.join(out_dir, "artist_release_history.parquet")
    if os.path.exists(release_history_path):
        artist_release_history = pd.read_parquet(release_history_path)
    else:
        artist_release_history = None

    scenario_path = os.path.join(out_dir, "scenario_multipliers.json")
    scenario_multipliers: Optional[Dict[str, Dict[str, float]]] = None
    if os.path.exists(scenario_path):
        try:
            with open(scenario_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                # Coerce to the expected nested shape with str cluster keys and
                # float values, ignoring anything malformed.
                scenario_multipliers = {}
                for c_key, mults in raw.items():
                    if not isinstance(mults, dict):
                        continue
                    coerced: Dict[str, float] = {}
                    for k in ("Bear", "Base", "Bull"):
                        if k in mults:
                            try:
                                coerced[k] = float(mults[k])
                            except (TypeError, ValueError):
                                pass
                    if coerced:
                        scenario_multipliers[str(c_key)] = coerced
                if not scenario_multipliers:
                    scenario_multipliers = None
        except (OSError, json.JSONDecodeError):
            scenario_multipliers = None

    return SimulatorArtifacts(
        horizon_weeks=horizon_weeks,
        archetype_params=archetype_params,
        artist_stats=artist_stats,
        artist_cluster_probs=artist_cluster_probs,
        artist_genre_cluster_probs=artist_genre_cluster_probs,
        artist_release_history=artist_release_history,
        scenario_multipliers=scenario_multipliers,
    )


#def train(args: argparse.Namespace) -> None:
   # df = pd.read_parquet(args.parquet_path, engine="fastparquet")
    #required_cols = {"MRELG_ID", "DISPLAY_ARTIST", "GENRES", "FIRST_SALE_DATE", "WEEK_END_DATE", "TARGET_METRIC", "TITLE"} # target_metric instead of worldwide_streams
    #missing = required_cols - set(df.columns)
    #if missing:
     #   raise ValueError(f"Parquet missing required columns: {sorted(missing)}")

    #df = compute_week_index(df, horizon_weeks=args.horizon_weeks)
    #df = compute_week_index(df, horizon_weeks=args.horizon_weeks, metric_col=args.metric)
def train(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.parquet_path, engine="fastparquet")
    df.columns = [str(c).upper() for c in df.columns]
    target_metric_col = args.metric.upper()  # convert to all uppercase from cli
    # Raw worldwide weekly *counts* (not streaming_equivalent): parquet column WORLDWIDE_STREAMS
    # when present; older extracts may use WEEKLY_STREAMS or STREAMING.
    if target_metric_col == "WORLDWIDE_STREAMS":
        for cand in ("WORLDWIDE_STREAMS", "WEEKLY_STREAMS", "STREAMING"):
            if cand in df.columns:
                target_metric_col = cand
                break
    required_cols = {"MRELG_ID", "DISPLAY_ARTIST", "GENRES", "FIRST_SALE_DATE", "WEEK_END_DATE", target_metric_col, "TITLE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {sorted(missing)}\nAvailable columns: {list(df.columns)}")
    df = compute_week_index(df, horizon_weeks=args.horizon_weeks, metric_col=target_metric_col)

    # Releases need at least one week with TARGET_METRIC > 0 (see extract_track_features).
    pos_by_release = (
        df.groupby("MRELG_ID", sort=False)["TARGET_METRIC"]
        .max()
        .gt(0)
        .sum()
    )
    n_releases = df["MRELG_ID"].nunique()
    if pos_by_release == 0:
        raise ValueError(
            f"Metric column {target_metric_col!r} has no positive values in "
            f"{args.parquet_path} (after week indexing). Skip this metric or fix the extract."
        )
    if pos_by_release < n_releases * 0.05:
        print(
            f"  Warning: only {pos_by_release:,}/{n_releases:,} releases have "
            f"peak {target_metric_col} > 0; clustering may be sparse."
        )

    print("Building feature table (missing-week-safe)...")
    features = build_feature_table(
        df,
        horizon_weeks=args.horizon_weeks,
        max_tracks=args.max_tracks_for_features,
        random_state=args.random_state,
    )

    print(f"Feature rows: {len(features):,} (unique releases: {features['MRELG_ID'].nunique():,})")

    print("Clustering into archetypes...")
    features_with_clusters, _model_info, _scaler, _imputer, _km = fit_archetype_clusters(
        features=features,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        batch_size=args.kmeans_batch_size,
    )

    total_col = "TOTAL_ALBUM_EQUIVALENTS"
    if total_col in df.columns:
        ap = (
            df.groupby("MRELG_ID", sort=False)[total_col]
            .max()
            .reset_index()
            .rename(columns={total_col: "peak_album_equiv_obs"})
        )
        features_with_clusters = features_with_clusters.merge(ap, on="MRELG_ID", how="left")
    else:
        features_with_clusters["peak_album_equiv_obs"] = np.nan

    # Empirical Bear/Base/Bull multipliers per cluster, computed natively from
    # this metric's parquet. Scoped intentionally: this is opt-in via the
    # ``compute_scenario_multipliers`` arg and is currently only enabled by
    # train_marketshare_artifacts for the worldwide_streams metric. The AE
    # panel metrics (streams / sales / songs) deliberately keep using the
    # hardcoded global ARCHETYPE_SCENARIO_MULTIPLIERS table — those scenario
    # numbers come from offline calibration against album-equivalents and
    # we don't want to silently replace them.
    if bool(getattr(args, "compute_scenario_multipliers", False)):
        print(
            f"Computing empirical scenario multipliers for metric={args.metric} "
            "(52-week, p20/p50/p80)..."
        )
        scenario_multipliers = compute_scenario_multipliers(
            df=df,
            features_with_clusters=features_with_clusters,
            mature_weeks=52,
            bear_pct=0.20,
            base_pct=0.50,
            bull_pct=0.80,
        )
        if scenario_multipliers:
            for cid in sorted(scenario_multipliers.keys()):
                mults = scenario_multipliers[cid]
                print(
                    f"  cluster {cid}: Bear={mults['Bear']:.3f}, "
                    f"Base={mults['Base']:.3f}, Bull={mults['Bull']:.3f}"
                )
        else:
            print(
                "  no clusters had >=2 mature (>=52wk) releases — "
                "scenario_multipliers.json will not be written; downstream will "
                "fall back to global ARCHETYPE_SCENARIO_MULTIPLIERS."
            )
    else:
        scenario_multipliers = None

    print("Fitting archetype curve functions...")
    archetype_params = fit_archetype_curves(
        df=df,
        features_with_clusters=features_with_clusters,
        horizon_weeks=args.horizon_weeks,
        n_clusters=args.n_clusters,
    )

    print("Computing artist alignment tables...")
    artist_stats, artist_cluster_probs = compute_artist_alignment(
        features_with_clusters=features_with_clusters,
        archetype_params=archetype_params,
    )
    artist_genre_cluster_probs = compute_artist_genre_alignment(
        features_with_clusters=features_with_clusters,
        archetype_params=archetype_params,
    )

    # Save per-release artist history so simulation can match on similar peak sizes.
    # This is what enables "use similar peak_volume past releases" logic at inference-time.
    required_cols = {
        "MRELG_ID",
        "DISPLAY_ARTIST",
        "peak_volume_obs",
        "peak_week_obs",
        "Archetype_Cluster",
        "main_genre",
        "FIRST_SALE_DATE",
        "tail_volume_obs",
    }
    missing_cols = required_cols - set(features_with_clusters.columns)
    if missing_cols:
        raise ValueError(f"features_with_clusters missing columns needed for release history: {sorted(missing_cols)}")

    hist_cols = [
        "MRELG_ID",
        "DISPLAY_ARTIST",
        "peak_volume_obs",
        "peak_week_obs",
        "Archetype_Cluster",
        "main_genre",
        "FIRST_SALE_DATE",
        "tail_volume_obs",
    ]
    if "peak_album_equiv_obs" in features_with_clusters.columns:
        hist_cols.append("peak_album_equiv_obs")

    artist_release_history = features_with_clusters[hist_cols].copy()

    # Example alignment diagnostics
    try:
        for col in ["main_genre", "release_month"]:
            p_value, chi2, cramer_v = chi_squared_feature_alignment(
                features_with_clusters=features_with_clusters,
                feature_col=col,
                target_col="Archetype_Cluster",
            )
            print(f"Alignment {col} vs Archetype_Cluster: p={p_value:.3e}, CramerV={cramer_v:.4f}")
    except Exception as e:
        print(f"Alignment diagnostics skipped due to error: {repr(e)}")

    print("Saving artifacts...")
    save_artifacts(
        out_dir=args.out_dir,
        horizon_weeks=args.horizon_weeks,
        archetype_params=archetype_params,
        artist_stats=artist_stats,
        artist_cluster_probs=artist_cluster_probs,
        artist_genre_cluster_probs=artist_genre_cluster_probs,
        artist_release_history=artist_release_history,
        scenario_multipliers=scenario_multipliers,
    )

    # Quick sanity sim
    if args.sanity_artist is not None:
        artifacts = load_artifacts(args.out_dir)
        pred, summary = simulate_future_drop(
            artist=args.sanity_artist,
            peak_volume=args.sanity_peak_volume,
            peak_week=args.sanity_peak_week,
            genre=args.sanity_genre,
            artifacts=artifacts,
        )
        print("Sanity simulation summary:", summary)
        print(pred.head(10).to_string(index=False))


def simulate(args: argparse.Namespace) -> None:
    # load both from training
    sales_artifacts = load_artifacts(args.sales_dir)
    streams_artifacts = load_artifacts(args.streams_dir)
    songs_artifacts = load_artifacts(args.songs_dir)
    
    # We can use streams_artifacts to get the horizon since they share the same timeline
    h = int(streams_artifacts.horizon_weeks)
    end_week, drop_d, fmeta = cli_resolve_end_week(args, h)
    
    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    pm_on = getattr(args, "peak_match_on", "trained_metric")
    pm_tgt = getattr(args, "peak_match_target", None)
    if pm_tgt is None and pm_on == "album_equivalents":
        sp, st, ss = getattr(args, "sales_peak", None), getattr(args, "streams_peak", None), getattr(args, "songs_peak", None)
        if sp is not None and st is not None and ss is not None:
            pm_tgt = float(sp + st + ss)

    sale_floor = getattr(args, "sales_floor", None)
    song_floor = getattr(args, "songs_floor", None)

    # sale simulation
    # forced peak_week=1.0 for sales because physical/digital drops almost always peak on release.
    if len(artists) == 1:
        pred_sales, summ_sales = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.sales_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=sales_artifacts,
            stream_floor=sale_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )
    else:
        pred_sales, summ_sales = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.sales_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=sales_artifacts,
            stream_floor=sale_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )

    # stream simulation
    if len(artists) == 1:
        pred_streams, summ_streams = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.streams_peak,
            peak_week=args.streams_peak_week,
            genre=args.genre,
            artifacts=streams_artifacts,
            stream_floor=args.streams_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )
    else:
        pred_streams, summ_streams = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.streams_peak,
            peak_week=args.streams_peak_week,
            genre=args.genre,
            artifacts=streams_artifacts,
            stream_floor=args.streams_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )

    # --- 3. SIMULATE SONG SALES ---
    if len(artists) == 1:
        pred_songs, summ_songs = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.songs_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=songs_artifacts,
            stream_floor=song_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )
    else:
        pred_songs, summ_songs = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.songs_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=songs_artifacts,
            stream_floor=song_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )


    # total album equivalents from streams and sales
    pred_combined = pd.DataFrame()
    pred_combined["week"] = pred_sales["week"]
    
    # pred same as before
    pred_combined["pred_sales"] = pred_sales["pred_weekly_streams"] 
    pred_combined["pred_streams"] = pred_streams["pred_weekly_streams"]
    pred_combined["pred_songs"] = pred_songs["pred_weekly_streams"]
    
    # total weekly
    pred_combined["pred_weekly_equivalents"] = pred_combined["pred_sales"] + pred_combined["pred_streams"] + pred_combined["pred_songs"]
    
    # satisfies old slice logic
    pred_combined["pred_weekly_streams"] = pred_combined["pred_weekly_equivalents"]
    
    # combined summary block
    summ_combined = {
        "artist": summ_streams.get("artist", artists[0]),
        "genre_used": args.genre if args.genre is not None else "ALL",
        "sales_peak_used": args.sales_peak,
        "streams_peak_used": args.streams_peak,
        "streams_peak_week_used": args.streams_peak_week,
        "cluster_probs_sales": summ_sales.get("cluster_probs", []),
        "cluster_probs_streams": summ_streams.get("cluster_probs", [])
    }

    # slice to requested end week
    pred_final, summ_final = slice_forecast_output(pred_combined, summ_combined, end_week, drop_d, fmeta)
    
    # clean
    pred_final["cumulative_pred_equivalents"] = pred_final["cumulative_pred_streams"]
    pred_final = pred_final.drop(columns=["pred_weekly_streams", "cumulative_pred_streams"], errors="ignore")
    
    # reaname
    summ_final["total_lifecycle_pred_equivalents"] = summ_final.pop("total_lifecycle_pred_streams", 0.0)

    #output
    print(json.dumps(summ_final, ensure_ascii=False, indent=2))
    if getattr(args, "out_csv", None) is not None:
        pred_final.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")
    else:
        print(pred_final.to_string(index=False))


def plot_archetypes_and_simulation(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt

    # 1. Load both sets of artifacts
    sales_artifacts = load_artifacts(args.sales_dir)
    streams_artifacts = load_artifacts(args.streams_dir)
    songs_artifacts = load_artifacts(args.songs_dir)

    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    pm_on = getattr(args, "peak_match_on", "trained_metric")
    pm_tgt = getattr(args, "peak_match_target", None)
    if pm_tgt is None and pm_on == "album_equivalents":
        sp, st, ss = getattr(args, "sales_peak", None), getattr(args, "streams_peak", None), getattr(args, "songs_peak", None)
        if sp is not None and st is not None and ss is not None:
            pm_tgt = float(sp + st + ss)

    sale_floor = getattr(args, "sales_floor", None)
    song_floor = getattr(args, "songs_floor", None)

    # --- 2. SIMULATE SALES ---
    if len(artists) == 1:
        pred_sales, summ_sales = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.sales_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=sales_artifacts,
            stream_floor=sale_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )
    else:
        pred_sales, summ_sales = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.sales_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=sales_artifacts,
            stream_floor=sale_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )

    # --- 3. SIMULATE STREAMS ---
    if len(artists) == 1:
        pred_streams, summ_streams = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.streams_peak,
            peak_week=args.streams_peak_week,
            genre=args.genre,
            artifacts=streams_artifacts,
            stream_floor=args.streams_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )
    else:
        pred_streams, summ_streams = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.streams_peak,
            peak_week=args.streams_peak_week,
            genre=args.genre,
            artifacts=streams_artifacts,
            stream_floor=args.streams_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )

    if len(artists) == 1:
        pred_songs, summ_songs = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.songs_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=songs_artifacts,
            stream_floor=song_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )
    else:
        pred_songs, summ_songs = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.songs_peak,
            peak_week=1.0,
            genre=args.genre,
            artifacts=songs_artifacts,
            stream_floor=song_floor,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
            peak_match_on=pm_on,
            peak_match_target=pm_tgt,
        )


    # --- 4. PREPARE DATA FOR PLOTTING ---
    h = int(streams_artifacts.horizon_weeks)
    end_week, _drop_d, _fmeta = cli_resolve_end_week(args, h)
    end_week = int(min(end_week, h))
    t = np.arange(1, end_week + 1, dtype=float)

    # Extract the simulated values up to the end week
    y_sales = pred_sales["pred_weekly_streams"].to_numpy(dtype=float)[:end_week]
    y_streams = pred_streams["pred_weekly_streams"].to_numpy(dtype=float)[:end_week]
    y_songs = pred_songs["pred_weekly_streams"].to_numpy(dtype=float)[:end_week]
    y_total = y_sales + y_streams + y_songs

    # --- 5. RENDER PLOT ---
    plt.figure(figsize=(12, 6))

    # Plot components (dashed lines)
    plt.plot(t, y_sales, color="#d62728", linewidth=2, linestyle="--", alpha=0.8, label="Predicted Product Sales")
    plt.plot(t, y_streams, color="#1f77b4", linewidth=2, linestyle="--", alpha=0.8, label="Predicted Streams (AEU)")
    plt.plot(t, y_songs, color="#2ca02c", linewidth=2, linestyle="--", alpha=0.8, label="Song Sales (AEU)")
    # Plot Total (solid black line)
    plt.plot(t, y_total, color="black", linewidth=3, alpha=0.9, label="Total Album Equivalents")

    plt.title(f"Simulated Two-Component Lifecycle: {args.artist}")
    plt.xlabel("Week since release")
    plt.ylabel("Album Equivalent Units")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.out_plot is not None:
        plt.savefig(args.out_plot, dpi=150)
        print(f"Wrote plot: {args.out_plot}")
    else:
        plt.show()

    # Consolidate a quick summary for the plot logs
    summ_combined = {
        "artist": summ_streams.get("artist", artists[0]),
        "genre_used": args.genre if args.genre is not None else "ALL",
        "sales_peak_used": args.sales_peak,
        "streams_peak_used": args.streams_peak,
        "total_lifecycle_pred_equivalents": float(np.sum(y_total))
    }
    print(json.dumps(summ_combined, ensure_ascii=False, indent=2))

from pathlib import Path
AE_COMPRESSED_PARQUET_PATH = Path(__file__).resolve().parent / "data" / "streams_product_songs_ae_compressed.parquet"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument(
        "--parquet-path",
        type=str,
        default=AE_COMPRESSED_PARQUET_PATH,
        help="Training panel (e.g. streams_product_songs_ae_compressed.parquet with PRODUCT_SALES / STREAMING_EQUIVALENT / SONG_SALE_EQUIVALENT / TOTAL_ALBUM_EQUIVALENTS).",
    )
    train_p.add_argument("--out-dir", type=str, default="archetypes_artifacts")
    train_p.add_argument("--horizon-weeks", type=int, default=78)
    train_p.add_argument("--n-clusters", type=int, default=4)
    train_p.add_argument("--random-state", type=int, default=42)
    train_p.add_argument("--kmeans-batch-size", type=int, default=2048)
    train_p.add_argument("--max-tracks-for-features", type=int, default=None)
    train_p.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=[
            "product_sales",
            "streaming_equivalent",
            "song_sale_equivalent",
            "worldwide_streams",
        ],
        help=(
            "Which column to train on (headers uppercased after load). "
            "streaming_equivalent / product_sales / song_sale_equivalent = AE panel parquet only. "
            "worldwide_streams = raw worldwide weekly counts (WORLDWIDE_STREAMS column, or legacy WEEKLY_STREAMS / STREAMING); not equivalents."
        ),
    )

    train_p.add_argument("--sanity-artist", type=str, default=None, help="If set, run a single simulation after training.")
    train_p.add_argument("--sanity-peak-volume", type=float, default=None)
    train_p.add_argument("--sanity-peak-week", type=float, default=None)
    train_p.add_argument("--sanity-genre", type=str, default=None)
    train_p.add_argument(
        "--compute-scenario-multipliers",
        action="store_true",
        default=False,
        help=(
            "Compute and persist empirical Bear/Base/Bull multipliers per cluster as "
            "scenario_multipliers.json in --out-dir. Currently intended for "
            "--metric worldwide_streams only; AE panel metrics (streaming_equivalent / "
            "product_sales / song_sale_equivalent) deliberately keep using the hardcoded "
            "global ARCHETYPE_SCENARIO_MULTIPLIERS table and should leave this off."
        ),
    )

    sim_p = sub.add_parser("simulate")
    sim_p.add_argument("--out-dir", type=str, default="archetypes_artifacts")
    sim_p.add_argument("--artist", type=str, required=True)
    sim_p.add_argument(
        "--artist-candidates",
        type=str,
        default=None,
        help="Comma-separated DISPLAY_ARTIST names to average (up to 3). If set, overrides automatic matching.",
    )
    sim_p.add_argument("--peak-volume", type=float, default=None)
    sim_p.add_argument("--peak-week", type=float, default=None)
    sim_p.add_argument("--genre", type=str, default=None)
    sim_p.add_argument("--out-csv", type=str, default=None)
    sim_p.add_argument("--sales-dir", type=str, default="archetypes_artifacts/sales") # new
    sim_p.add_argument("--streams-dir", type=str, default="archetypes_artifacts/streams") # new
    sim_p.add_argument("--sales-peak", type=float, default=None) # new 
    sim_p.add_argument("--streams-peak", type=float, default=None) # new
    sim_p.add_argument("--streams-floor", type=float, default=None)
    sim_p.add_argument("--sales-floor", type=float, default=None, help="Weekly product-sales floor override; omit for dynamic tail from history.")
    sim_p.add_argument("--songs-floor", type=float, default=None, help="Weekly song-sales AE floor override; omit for dynamic tail from history.")
    sim_p.add_argument("--songs-dir", type=str, default="archetypes_artifacts/songs")
    sim_p.add_argument("--songs-peak", type=float, default=None)

    sim_p.add_argument("--streams-peak-week", type=float, default=1.0)
    sim_p.add_argument(
        "--peak-match-on",
        type=str,
        choices=["trained_metric", "album_equivalents"],
        default="trained_metric",
        help="Similar-release peak filter: trained metric column, or total album peak (needs peak_album_equiv_obs from training).",
    )
    sim_p.add_argument(
        "--peak-match-target",
        type=float,
        default=None,
        help="Optional scalar for similar-peak matching (e.g. expected total AE). Default: sum of sales+streams+songs peaks when --peak-match-on album_equivalents.",
    )
    sim_p.add_argument("--peak-sim-log-radius", type=float, default=0.35, help="Log10 peak-volume radius for similarity selection.")
    sim_p.add_argument("--peak-sim-min-subset-releases", type=int, default=10, help="Minimum # releases in similar-peak subset.")
    sim_p.add_argument("--peak-sim-spread-threshold-log-std", type=float, default=0.25, help="If artist's log-peak std exceeds this, use similar-peak subset.")
    sim_p.add_argument("--peak-sim-min-artist-releases", type=int, default=20, help="Minimum # releases required to attempt similar-peak filtering.")
    sim_p.add_argument("--drop-date", type=str, default=None, help="Release date YYYY-MM-DD; adds week_ending to CSV when set.")
    sim_p.add_argument(
        "--forecast-target",
        type=str,
        choices=["manual", "end-of-year", "lifecycle"],
        default="manual",
        help="manual: use --end-week. end-of-year: through Dec 31 of drop year (needs --drop-date). lifecycle: 18 months (~78 wks).",
    )
    sim_p.add_argument("--end-week", type=int, default=78, help="Output weeks 1..N when forecast-target is manual (capped by model horizon).")

    plot_p = sub.add_parser("plot")
    plot_p.add_argument("--sales-dir", type=str, default="archetypes_artifacts/sales")
    plot_p.add_argument("--streams-dir", type=str, default="archetypes_artifacts/streams")
    plot_p.add_argument("--songs-dir", type=str, default="archetypes_artifacts/songs")
    plot_p.add_argument("--songs-peak", type=float, default=None)
    
    plot_p.add_argument("--artist", type=str, required=True)
    plot_p.add_argument(
        "--artist-candidates",
        type=str,
        default=None,
        help="Comma-separated DISPLAY_ARTIST names to average (up to 3). If set, overrides automatic matching.",
    )
    
    # Add the dual peaks
    plot_p.add_argument("--sales-peak", type=float, default=None)
    plot_p.add_argument("--streams-peak", type=float, default=None)
    plot_p.add_argument("--streams-peak-week", type=float, default=1.0)
    plot_p.add_argument("--streams-floor", type=float, default=None)
    plot_p.add_argument("--sales-floor", type=float, default=None)
    plot_p.add_argument("--songs-floor", type=float, default=None)
    plot_p.add_argument(
        "--peak-match-on",
        type=str,
        choices=["trained_metric", "album_equivalents"],
        default="trained_metric",
    )
    plot_p.add_argument("--peak-match-target", type=float, default=None)
    plot_p.add_argument("--genre", type=str, default=None)
    plot_p.add_argument("--out-plot", type=str, default=None, help="If set, save plot to this path (e.g. plot.png).")
    
    # The rest remain the same
    plot_p.add_argument("--peak-sim-log-radius", type=float, default=0.35)
    plot_p.add_argument("--peak-sim-min-subset-releases", type=int, default=10)
    plot_p.add_argument("--peak-sim-spread-threshold-log-std", type=float, default=0.25)
    plot_p.add_argument("--peak-sim-min-artist-releases", type=int, default=20)
    plot_p.add_argument("--drop-date", type=str, default=None)
    plot_p.add_argument("--forecast-target", type=str, choices=["manual", "end-of-year", "lifecycle"], default="manual")
    plot_p.add_argument("--end-week", type=int, default=78)

    backfill_p = sub.add_parser("backfill")
    backfill_p.add_argument("--sales-dir", type=str, default="archetypes_artifacts/sales")
    backfill_p.add_argument("--streams-dir", type=str, default="archetypes_artifacts/streams")
    backfill_p.add_argument("--songs-dir", type=str, default="archetypes_artifacts/songs")
    backfill_p.add_argument("--songs-actuals", type=str, required=False, help="Observed weekly SONG SALES equivalents")
    backfill_p.add_argument("--artist", type=str, required=True)
    backfill_p.add_argument("--total-actuals", type=str, required=False, help="Observed TOTAL AEU actuals")
    backfill_p.add_argument(
        "--artist-candidates",
        type=str,
        default=None,
        help="Comma-separated DISPLAY_ARTIST names to average (up to 3). If set, overrides automatic matching.",
    )
    # Replaced single --actuals with dual actuals
    backfill_p.add_argument("--sales-actuals", type=str, required=False, help="Observed weekly SALES streams for weeks 1..K.")
    backfill_p.add_argument("--streams-actuals", type=str, required=False, help="Observed weekly STREAMS streams for weeks 1..K.")
    backfill_p.add_argument("--genre", type=str, default=None)
    backfill_p.add_argument("--end-week", type=int, default=78)
    backfill_p.add_argument("--drop-date", type=str, default=None)
    backfill_p.add_argument("--forecast-target", type=str, choices=["manual", "end-of-year", "lifecycle"], default="manual")
    backfill_p.add_argument("--peak-week-margin", type=int, default=3)
    backfill_p.add_argument("--mixture-fit-weight", type=float, default=0.5)
    backfill_p.add_argument("--out-csv", type=str, default=None)
    backfill_p.add_argument("--streams-floor", type=float, default=None)
    backfill_p.add_argument("--sales-floor", type=float, default=None)
    backfill_p.add_argument("--songs-floor", type=float, default=None)
    backfill_p.add_argument("--sales-peak", type=float, default=None, help="Used with --total-actuals for peak-match and proportion baselines.")
    backfill_p.add_argument("--streams-peak", type=float, default=None)
    backfill_p.add_argument("--streams-peak-week", type=float, default=1.0, help="Streaming peak week for baselines when splitting --total-actuals.")
    backfill_p.add_argument("--songs-peak", type=float, default=None)
    backfill_p.add_argument(
        "--peak-match-on",
        type=str,
        choices=["trained_metric", "album_equivalents"],
        default="trained_metric",
    )
    backfill_p.add_argument("--peak-match-target", type=float, default=None)
    backfill_p.add_argument("--peak-sim-log-radius", type=float, default=0.35, help="Log10 peak-volume radius (total-AEU split baselines).")
    backfill_p.add_argument("--peak-sim-min-subset-releases", type=int, default=10)
    backfill_p.add_argument("--peak-sim-spread-threshold-log-std", type=float, default=0.25)
    backfill_p.add_argument("--peak-sim-min-artist-releases", type=int, default=20)
    plot_backfill_p = sub.add_parser("plot-backfill")
    plot_backfill_p.add_argument("--out-dir", type=str, default="archetypes_artifacts")
    plot_backfill_p.add_argument("--artist", type=str, required=True)
    plot_backfill_p.add_argument(
        "--artist-candidates",
        type=str,
        default=None,
        help="Comma-separated DISPLAY_ARTIST names to average (up to 3). If set, overrides automatic matching.",
    )
    plot_backfill_p.add_argument("--sales-dir", type=str, default="archetypes_artifacts/sales")
    plot_backfill_p.add_argument("--streams-dir", type=str, default="archetypes_artifacts/streams")
    plot_backfill_p.add_argument("--songs-dir", type=str, default="archetypes_artifacts/songs")
    plot_backfill_p.add_argument("--songs-actuals", type=str, required=True, help="Observed weekly SONG SALES equivalents")
    plot_backfill_p.add_argument("--sales-actuals", type=str, required=True)
    plot_backfill_p.add_argument("--streams-actuals", type=str, required=True)
    plot_backfill_p.add_argument("--genre", type=str, default=None)
    plot_backfill_p.add_argument("--end-week", type=int, default=78)
    plot_backfill_p.add_argument("--drop-date", type=str, default=None)
    plot_backfill_p.add_argument("--forecast-target", type=str, choices=["manual", "end-of-year", "lifecycle"], default="manual")
    plot_backfill_p.add_argument("--peak-week-margin", type=int, default=3)
    plot_backfill_p.add_argument("--mixture-fit-weight", type=float, default=0.5)
    plot_backfill_p.add_argument("--out-plot", type=str, default=None)
    plot_backfill_p.add_argument("--streams-floor", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "simulate":
        simulate(args)
    elif args.cmd == "plot":
        plot_archetypes_and_simulation(args)
    elif args.cmd == "backfill":
        backfill(args)
    elif args.cmd == "plot-backfill":
        plot_backfill(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

