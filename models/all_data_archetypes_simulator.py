"""
Stream archetype clustering, curve fitting, and simulation helpers.

API / batch refresh: call ``main()`` to rebuild the artifact parquets and JSON consumed by the
forecast API (no command-line interface). Input: ``ARCHETYPES_INPUT_PARQUET``; output:
``ARCHETYPES_ARTIFACTS_DIR`` (see ``save_artifacts`` for filenames). Paths use ``REPO_ROOT``.
"""

import re
import json
import os
import difflib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .marketshare_75k_simulation import MODELS_DIR, REPO_ROOT

# --- Input parquet lives alongside this module; artifacts go under project root ---
ARCHETYPES_INPUT_PARQUET = MODELS_DIR / "all_releases_18_25_compressed.parquet"
ARCHETYPES_ARTIFACTS_DIR = REPO_ROOT / "archetypes_artifacts"

TRAIN_HORIZON_WEEKS = 78
TRAIN_N_CLUSTERS = 4
TRAIN_RANDOM_STATE = 42
TRAIN_KMEANS_BATCH_SIZE = 2048
TRAIN_MAX_TRACKS_FOR_FEATURES: Optional[int] = None


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


def compute_week_index(df: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    df = df.copy()
    df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"])
    df["WEEKLY_STREAMS"] = pd.to_numeric(df["WEEKLY_STREAMS"], errors="coerce").fillna(0.0)

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
    y = g["WEEKLY_STREAMS"].to_numpy(dtype=float)

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
    df_min = df[["MRELG_ID", "week", "WEEKLY_STREAMS"]].copy()
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
    peak_per_track = df2.groupby("MRELG_ID")["WEEKLY_STREAMS"].transform("max")
    peak_per_track = peak_per_track.replace(0, np.nan)
    df2["y_norm"] = df2["WEEKLY_STREAMS"] / peak_per_track

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


def cli_resolve_end_week(
    horizon_cap: int,
    *,
    drop_date: Optional[str] = None,
    forecast_target: str = "manual",
    end_week: Optional[int] = None,
) -> Tuple[int, Optional[pd.Timestamp], Dict[str, Any]]:
    dd = parse_drop_date(drop_date)
    ft = str(forecast_target or "manual")
    ew_manual = int(end_week if end_week is not None else horizon_cap)
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


@dataclass
class SimulatorArtifacts:
    horizon_weeks: int
    archetype_params: Dict[str, Any]
    artist_stats: pd.DataFrame
    artist_cluster_probs: pd.DataFrame
    artist_genre_cluster_probs: Optional[pd.DataFrame] = None
    artist_release_history: Optional[pd.DataFrame] = None


def simulate_future_drop(
    *,
    artist: str,
    peak_volume: Optional[float],
    peak_week: Optional[float],
    genre: Optional[str],
    artifacts: SimulatorArtifacts,
    peak_sim_log_radius: float = 0.35,
    peak_sim_min_subset_releases: int = 10,
    peak_sim_spread_threshold_log_std: float = 0.25,
    peak_sim_min_artist_releases: int = 20,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulate week-by-week streams for `horizon_weeks`.

    Shape:
      - mixture over archetypes by artist history
      - gamma_norm curve uses the *input/estimated peak_week* for all archetypes
      - amplitude is peak_volume
    """
    horizon = artifacts.horizon_weeks
    t = np.arange(1, horizon + 1, dtype=float)

    # Artist matching: training artifacts may store names with different casing
    # (e.g., "BRUNO MARS") or as collab strings (e.g., "Lady Gaga, Bruno Mars").
    artist_in = artist.strip()
    available_artists = artifacts.artist_stats["DISPLAY_ARTIST"].astype(str)
    available_lower = available_artists.str.lower()
    needle_lower = artist_in.lower()

    if needle_lower not in set(available_lower):
        # Try token substring match (quick heuristic for "Bruno Mars" -> find "BRUNO MARS")
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", needle_lower) if tok]
        cand_mask = np.ones(len(available_artists), dtype=bool)
        for tok in tokens[:4]:  # avoid extremely broad matches
            cand_mask &= available_lower.str.contains(tok, na=False)

        candidates = available_artists[cand_mask].unique().tolist()
        # Close matches (edit distance) as a fallback
        close = difflib.get_close_matches(artist_in, available_artists.unique().tolist(), n=10, cutoff=0.55)
        suggestions = (candidates[:10] + close[:10])[:15]
        suggestions = [s for s in suggestions if s and s != artist_in]

        # If we can’t find the artist verbatim, fuzzy-map to the closest trained DISPLAY_ARTIST.
        # Best-effort constraining:
        #   - constrain by genre (main_genre) if genre provided
        #   - constrain by release year window (2018-2025) if FIRST_SALE_DATE is available
        if artifacts.artist_release_history is not None and not artifacts.artist_release_history.empty:
            hist = artifacts.artist_release_history

            # Year filtering (best-effort).
            allowed_year = None
            if "FIRST_SALE_DATE" in hist.columns:
                h = hist.copy()
                h["FIRST_SALE_DATE"] = pd.to_datetime(h["FIRST_SALE_DATE"], errors="coerce")
                h = h[h["FIRST_SALE_DATE"].notna()].copy()
                if not h.empty:
                    h = h[h["FIRST_SALE_DATE"].dt.year.between(2018, 2025)].copy()
                    if not h.empty:
                        allowed_year = set(h["DISPLAY_ARTIST"].astype(str).unique().tolist())

            # Optional genre filtering.
            if genre is not None and str(genre).strip() and "main_genre" in hist.columns:
                g = str(genre).strip()
                hist = hist[hist["main_genre"] == g].copy()

            # If we have any year constraint info, apply it.
            if allowed_year is not None:
                suggestions_filtered = [s for s in suggestions if s in allowed_year]
                if suggestions_filtered:
                    suggestions = suggestions_filtered

            # If genre narrowed the hist, also apply it as an allowlist.
            if genre is not None and str(genre).strip() and "DISPLAY_ARTIST" in hist.columns:
                allowed_genre = set(hist["DISPLAY_ARTIST"].astype(str).unique().tolist())
                suggestions_filtered = [s for s in suggestions if s in allowed_genre]
                if suggestions_filtered:
                    suggestions = suggestions_filtered

        if not suggestions:
            raise ValueError(
                f"Unknown artist: {artist_in}. (Not present in training artifacts.)\n"
                f"Top similar names in artifacts: {suggestions if suggestions else 'N/A'}\n"
                "Tip: provide --artist-candidates with canonical DISPLAY_ARTIST names from training "
                "(or leave it blank and provide --genre). "
                "The simulator also prefers similar artists with releases between 2018 and 2025 "
                "when release-date metadata is available."
            )

        artist_candidates = [str(s) for s in suggestions[:3]]
        if len(artist_candidates) == 1:
            needle_lower = artist_candidates[0].lower()
        else:
            print(
                f'Fuzzy-artist mapping (averaging): "{artist_in}" -> {artist_candidates}'
            )
            preds = []
            summaries = []
            for cand in artist_candidates:
                pred_i, summ_i = simulate_future_drop(
                    artist=cand,
                    peak_volume=peak_volume,
                    peak_week=peak_week,
                    genre=genre,
                    artifacts=artifacts,
                    peak_sim_log_radius=peak_sim_log_radius,
                    peak_sim_min_subset_releases=peak_sim_min_subset_releases,
                    peak_sim_spread_threshold_log_std=peak_sim_spread_threshold_log_std,
                    peak_sim_min_artist_releases=peak_sim_min_artist_releases,
                )
                preds.append(pred_i["pred_weekly_streams"].to_numpy(dtype=float))
                summaries.append(summ_i)

            avg_streams = np.mean(np.stack(preds, axis=0), axis=0)
            out = pred_i.copy()
            out["pred_weekly_streams"] = avg_streams
            out["cumulative_pred_streams"] = np.cumsum(avg_streams)

            # Average cluster-probabilities across candidates (renormalize).
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

            peak_volume_used = float(peak_volume) if peak_volume is not None else float(np.mean([s.get("peak_volume", np.nan) for s in summaries]))
            peak_week_used = float(peak_week) if peak_week is not None else float(np.mean([s.get("peak_week", np.nan) for s in summaries]))
            summary = {
                "artist": artist_in,
                "artists_used": artist_candidates,
                "genre_used": genre if genre is not None else "ALL",
                "peak_volume": float(peak_volume_used),
                "peak_week": float(peak_week_used),
                "total_lifecycle_pred_streams": float(out["cumulative_pred_streams"].iloc[-1]),
                "cluster_probs": cluster_probs_avg,
            }
            return out, summary

    # If case differs, remap to the canonical stored name.
    canon_match = available_artists[available_lower == needle_lower].iloc[0]
    artist = str(canon_match)

    # Amplitude & timing
    if peak_volume is None:
        peak_volume = float(
            artifacts.artist_stats.loc[
                artifacts.artist_stats["DISPLAY_ARTIST"] == artist, "median_peak_volume"
            ].iloc[0]
        )
    if peak_week is None:
        peak_week = float(
            artifacts.artist_stats.loc[
                artifacts.artist_stats["DISPLAY_ARTIST"] == artist, "median_peak_week"
            ].iloc[0]
        )

    peak_week = float(np.clip(peak_week, 1.0, horizon))

    # Mixture distribution:
    # - If artist's historical peak volumes vary a lot, blend archetypes using ONLY releases
    #   whose observed peak_volume is "similar" (log distance) to the target peak_volume.
    # - Otherwise, use the full in-window archetype mixture for the artist (optionally filtered by genre).
    probs = pd.DataFrame()

    if artifacts.artist_release_history is not None and not artifacts.artist_release_history.empty:
        hist = artifacts.artist_release_history
        hist = hist[hist["DISPLAY_ARTIST"] == artist].copy()
        if genre is not None and "main_genre" in hist.columns:
            hist = hist[hist["main_genre"] == genre].copy()

        if not hist.empty:
            n_hist = len(hist)
            peak_obs = hist["peak_volume_obs"].astype(float).to_numpy()
            peak_obs = np.clip(peak_obs, 1e-9, None)
            log_peaks = np.log10(peak_obs)
            log_std = float(np.std(log_peaks))

            if n_hist >= peak_sim_min_artist_releases and log_std > peak_sim_spread_threshold_log_std:
                # Similarity selection in log space around target peak_volume
                target_peak = float(max(peak_volume, 1e-9))
                target_log = float(np.log10(target_peak))

                radius = float(peak_sim_log_radius)
                subset = pd.DataFrame()
                for _ in range(8):  # expand a few times if subset is too small
                    subset = hist[np.abs(np.log10(np.clip(hist["peak_volume_obs"].astype(float).to_numpy(), 1e-9, None)) - target_log) <= radius].copy()
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

        if probs.empty:
            probs = artifacts.artist_cluster_probs[
                artifacts.artist_cluster_probs["DISPLAY_ARTIST"] == artist
            ][["Archetype_Cluster", "prob"]].copy()

    if probs.empty:
        raise ValueError(f"No fitted archetype mixture available for artist={artist}, genre={genre}")

    probs = probs.dropna(subset=["Archetype_Cluster", "prob"]).copy()
    probs["Archetype_Cluster"] = probs["Archetype_Cluster"].astype(int)
    probs["prob"] = probs["prob"].astype(float)
    probs_sum = float(probs["prob"].sum()) if len(probs) else 0.0
    if probs_sum > 0:
        probs["prob"] = probs["prob"] / probs_sum

    y_norm = np.zeros_like(t, dtype=float)
    for _, row in probs.iterrows():
        c = int(row["Archetype_Cluster"])
        p = float(row["prob"])
        par = artifacts.archetype_params.get(str(c))
        if par is None:
            continue
        a = float(par["a"])
        y_norm += p * gamma_norm(t, a=a, t_peak=peak_week)

    y_streams = peak_volume * y_norm
    y_streams = np.clip(y_streams, 0, None)

    weeks = np.arange(1, horizon + 1, dtype=int)
    out = pd.DataFrame({"week": weeks, "pred_weekly_streams": y_streams})
    out["cumulative_pred_streams"] = np.cumsum(out["pred_weekly_streams"].to_numpy(dtype=float))

    total_pred = float(out["cumulative_pred_streams"].iloc[-1])
    summary = {
        "artist": artist,
        "genre_used": genre if genre is not None else "ALL",
        "peak_volume": peak_volume,
        "peak_week": peak_week,
        "total_lifecycle_pred_streams": total_pred,
        "cluster_probs": probs.sort_values("prob", ascending=False)[["Archetype_Cluster", "prob"]].to_dict(orient="records"),
    }
    return out, summary


def simulate_future_drop_average(
    *,
    artists: List[str],
    peak_volume: Optional[float],
    peak_week: Optional[float],
    genre: Optional[str],
    artifacts: SimulatorArtifacts,
    peak_sim_log_radius: float = 0.35,
    peak_sim_min_subset_releases: int = 10,
    peak_sim_spread_threshold_log_std: float = 0.25,
    peak_sim_min_artist_releases: int = 20,
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
                peak_sim_log_radius=peak_sim_log_radius,
                peak_sim_min_subset_releases=peak_sim_min_subset_releases,
                peak_sim_spread_threshold_log_std=peak_sim_spread_threshold_log_std,
                peak_sim_min_artist_releases=peak_sim_min_artist_releases,
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
    mixture_fit_weight: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Backfill + forecast:
      - Weeks 1..K: use provided actuals
      - Weeks K+1..end_week: forecast using the fitted archetype mixture decay model

    We fit the artist/genre mixture by estimating:
      - peak_week (t_peak) on a discrete grid
      - peak_volume (amplitude) by least squares scaling
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
                mixture_fit_weight=mixture_fit_weight,
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
        raise ValueError("Backfill actuals must contain at least one positive weekly streams value.")
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
    for t_peak in range(lo, hi + 1):
        # G[t, j] = gamma_norm(week_t; a_j, t_peak)
        G = np.zeros((k, C), dtype=float)
        for j, a in enumerate(a_list):
            G[:, j] = gamma_norm(t_obs, a=a, t_peak=float(t_peak))

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
        raise RuntimeError("Backfill fit failed to find a valid peak_week candidate.")

    # Build full normalized curve up to end_week using fitted t_peak and fitted mixture weights.
    t_full = np.arange(1, end_week + 1, dtype=float)
    G_full = np.zeros((len(t_full), C), dtype=float)
    for j, a in enumerate(a_list):
        G_full[:, j] = gamma_norm(t_full, a=a, t_peak=float(best["t_peak"]))

    # best_p is the mixture weights used for the observed fit (already regularized).
    y_norm_full = G_full @ best_p
    y_pred_full = np.clip(best["peak_volume"] * y_norm_full, 0, None)

    # Boundary alignment:
    # We overwrite weeks 1..K with actuals in the output, but forecasts (K+1..)
    # depend on the fitted curve. If the fitted curve doesn't match the last
    # observed week value, the forecast can jump unnaturally.
    # We fix that by rescaling the entire fitted curve so week K matches actual week K.
    if k >= 1 and y_pred_full[k - 1] > 0:
        scale = float(y_obs[-1] / y_pred_full[k - 1])
        y_pred_full = np.clip(y_pred_full * scale, 0, None)
        best_peak_volume_scaled = float(best["peak_volume"] * scale)
    else:
        best_peak_volume_scaled = float(best["peak_volume"])

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
    mixture_fit_weight: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Average backfill+forecast curves over multiple DISPLAY_ARTIST names."""
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
                fit_logspace=fit_logspace,
                peak_week_margin=peak_week_margin,
                mixture_fit_weight=mixture_fit_weight,
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
    }
    return out, summary


def backfill(args: SimpleNamespace) -> None:
    artifacts = load_artifacts(args.out_dir)
    actuals = _parse_actuals_list(args.actuals)
    h = int(artifacts.horizon_weeks)
    end_week, drop_d, fmeta = cli_resolve_end_week(
        h,
        drop_date=getattr(args, "drop_date", None),
        forecast_target=str(getattr(args, "forecast_target", "manual") or "manual"),
        end_week=getattr(args, "end_week", None),
    )
    k = len(actuals)
    if k > end_week:
        raise ValueError(
            f"You have {k} weeks of actuals but forecast horizon is only {end_week} weeks "
            f"({fmeta.get('note', '')}). Use a longer horizon (e.g. lifecycle) or fewer actuals."
        )
    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    if len(artists) == 1:
        pred_df, summary = fit_backfill_forecast(
            artist=artists[0],
            genre=args.genre,
            actuals_weekly_streams=actuals,
            artifacts=artifacts,
            end_week=end_week,
            peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
        )
    else:
        pred_df, summary = fit_backfill_forecast_average(
            artists=artists,
            genre=args.genre,
            actuals_weekly_streams=actuals,
            artifacts=artifacts,
            end_week=end_week,
            peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
        )
    pred_df, summary = slice_forecast_output(pred_df, summary, end_week, drop_d, fmeta)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_csv is not None:
        pred_df.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")
    else:
        print(pred_df.to_string(index=False))


def plot_backfill(args: SimpleNamespace) -> None:
    import matplotlib.pyplot as plt

    artifacts = load_artifacts(args.out_dir)
    actuals = _parse_actuals_list(args.actuals)
    h = int(artifacts.horizon_weeks)
    end_week, _dd, _fm = cli_resolve_end_week(
        h,
        drop_date=getattr(args, "drop_date", None),
        forecast_target=str(getattr(args, "forecast_target", "manual") or "manual"),
        end_week=getattr(args, "end_week", None),
    )
    k = len(actuals)
    if k > end_week:
        raise ValueError(
            f"You have {k} weeks of actuals but forecast horizon is only {end_week} weeks. "
            "Adjust --forecast-target / --drop-date or shorten actuals."
        )
    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    if len(artists) == 1:
        pred_df, summary = fit_backfill_forecast(
            artist=artists[0],
            genre=args.genre,
            actuals_weekly_streams=actuals,
            artifacts=artifacts,
            end_week=end_week,
            peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
        )
    else:
        pred_df, summary = fit_backfill_forecast_average(
            artists=artists,
            genre=args.genre,
            actuals_weekly_streams=actuals,
            artifacts=artifacts,
            end_week=end_week,
            peak_week_margin=args.peak_week_margin,
            mixture_fit_weight=args.mixture_fit_weight,
        )
    t = np.arange(1, end_week + 1, dtype=float)
    # Normalization for visualization:
    # In backfill mode we overwrite weeks 1..K with actuals, so the fitted
    # `peak_volume` is not guaranteed to equal the max of the plotted series.
    # Normalize by the peak of the plotted series so y-axis is interpretable.
    pred_peak = float(pred_df["pred_weekly_streams"].max())
    if pred_peak > 0:
        y_norm_sim = pred_df["pred_weekly_streams"].to_numpy(dtype=float) / pred_peak
        y_obs_norm = np.asarray(actuals, dtype=float) / pred_peak
    else:
        y_norm_sim = pred_df["pred_weekly_streams"].to_numpy(dtype=float)
        y_obs_norm = np.asarray(actuals, dtype=float)

    plt.figure(figsize=(12, 6))

    # Plot fitted archetype shapes (normalized reference)
    for c_str, par in sorted(artifacts.archetype_params.items(), key=lambda kv: int(kv[0])):
        a = float(par["a"])
        t_peak = float(par["t_peak"])
        y_curve = gamma_norm(t, a=a, t_peak=t_peak)
        plt.plot(t, y_curve, linewidth=2, alpha=0.25)

    # Plot fitted artist curve + overlay observed points
    plt.plot(t, y_norm_sim, color="black", linewidth=3, alpha=0.9, label="Fit curve (normalized)")
    k = len(actuals)
    plt.scatter(np.arange(1, k + 1), y_obs_norm, color="red", s=30, alpha=0.9, label="Observed actuals (weeks 1..K)")

    plt.title(f"Backfill fit: {summary['artist']} (K={k} observed weeks)")
    plt.xlabel("Week since release")
    plt.ylabel("Normalized streams (peak of plotted series = 1)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    if args.out_plot is not None:
        plt.savefig(args.out_plot, dpi=150)
        print(f"Wrote plot: {args.out_plot}")
    else:
        plt.show()

    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
) -> None:
    """
    Persist files under ``out_dir`` for ``load_artifacts`` / API use:

    - ``archetype_params.json``
    - ``artist_stats.parquet``, ``artist_cluster_probs.parquet``
    - ``artist_genre_cluster_probs.parquet``, ``artist_release_history.parquet`` (when present)
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "archetype_params.json"), "w", encoding="utf-8") as f:
        json.dump({"horizon_weeks": horizon_weeks, "archetype_params": archetype_params}, f, ensure_ascii=False, indent=2)

    artist_stats.to_parquet(os.path.join(out_dir, "artist_stats.parquet"), index=False)
    artist_cluster_probs.to_parquet(os.path.join(out_dir, "artist_cluster_probs.parquet"), index=False)

    if artist_genre_cluster_probs is not None:
        artist_genre_cluster_probs.to_parquet(os.path.join(out_dir, "artist_genre_cluster_probs.parquet"), index=False)

    if artist_release_history is not None:
        artist_release_history.to_parquet(os.path.join(out_dir, "artist_release_history.parquet"), index=False)


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

    return SimulatorArtifacts(
        horizon_weeks=horizon_weeks,
        archetype_params=archetype_params,
        artist_stats=artist_stats,
        artist_cluster_probs=artist_cluster_probs,
        artist_genre_cluster_probs=artist_genre_cluster_probs,
        artist_release_history=artist_release_history,
    )


def train(args: SimpleNamespace) -> None:
    df = pd.read_parquet(args.parquet_path)
    required_cols = {"MRELG_ID", "DISPLAY_ARTIST", "GENRES", "FIRST_SALE_DATE", "WEEK_END_DATE", "WEEKLY_STREAMS", "TITLE"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Parquet missing required columns: {sorted(missing)}")

    df = compute_week_index(df, horizon_weeks=args.horizon_weeks)

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
    }
    missing_cols = required_cols - set(features_with_clusters.columns)
    if missing_cols:
        raise ValueError(f"features_with_clusters missing columns needed for release history: {sorted(missing_cols)}")

    artist_release_history = features_with_clusters[
        [
            "MRELG_ID",
            "DISPLAY_ARTIST",
            "peak_volume_obs",
            "peak_week_obs",
            "Archetype_Cluster",
            "main_genre",
            "FIRST_SALE_DATE",
        ]
    ].copy()

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
    )


def simulate(args: SimpleNamespace) -> None:
    artifacts = load_artifacts(args.out_dir)
    h = int(artifacts.horizon_weeks)
    end_week, drop_d, fmeta = cli_resolve_end_week(
        h,
        drop_date=getattr(args, "drop_date", None),
        forecast_target=str(getattr(args, "forecast_target", "manual") or "manual"),
        end_week=getattr(args, "end_week", None),
    )
    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    if len(artists) == 1:
        pred, summary = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.peak_volume,
            peak_week=args.peak_week,
            genre=args.genre,
            artifacts=artifacts,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
        )
    else:
        pred, summary = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.peak_volume,
            peak_week=args.peak_week,
            genre=args.genre,
            artifacts=artifacts,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
        )
    pred, summary = slice_forecast_output(pred, summary, end_week, drop_d, fmeta)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_csv is not None:
        pred.to_csv(args.out_csv, index=False)
        print(f"Wrote {args.out_csv}")
    else:
        print(pred.to_string(index=False))


def plot_archetypes_and_simulation(args: SimpleNamespace) -> None:
    """
    Plot:
      1) fitted archetype curve shapes (normalized)
      2) the simulated curve shape for the chosen artist (normalized)
    """
    import matplotlib.pyplot as plt

    artifacts = load_artifacts(args.out_dir)
    artists = [args.artist]
    if getattr(args, "artist_candidates", None):
        artists = [a.strip() for a in str(args.artist_candidates).split(",") if a.strip()][:3]
        if not artists:
            artists = [args.artist]

    if len(artists) == 1:
        pred, summary = simulate_future_drop(
            artist=artists[0],
            peak_volume=args.peak_volume,
            peak_week=args.peak_week,
            genre=args.genre,
            artifacts=artifacts,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
        )
    else:
        pred, summary = simulate_future_drop_average(
            artists=artists,
            peak_volume=args.peak_volume,
            peak_week=args.peak_week,
            genre=args.genre,
            artifacts=artifacts,
            peak_sim_log_radius=args.peak_sim_log_radius,
            peak_sim_min_subset_releases=args.peak_sim_min_subset_releases,
            peak_sim_spread_threshold_log_std=args.peak_sim_spread_threshold_log_std,
            peak_sim_min_artist_releases=args.peak_sim_min_artist_releases,
        )

    h = int(artifacts.horizon_weeks)
    end_week, _drop_d, _fmeta = cli_resolve_end_week(
        h,
        drop_date=getattr(args, "drop_date", None),
        forecast_target=str(getattr(args, "forecast_target", "manual") or "manual"),
        end_week=getattr(args, "end_week", None),
    )
    end_week = int(min(end_week, h))
    t = np.arange(1, end_week + 1, dtype=float)
    pred = pred.iloc[:end_week].copy()

    # Normalize the simulated curve by peak_volume so it can be compared to archetype shapes.
    peak_volume_used = float(pred["pred_weekly_streams"].max())
    if peak_volume_used <= 0:
        y_sim_norm = pred["pred_weekly_streams"].to_numpy(dtype=float)
    else:
        y_sim_norm = pred["pred_weekly_streams"].to_numpy(dtype=float) / peak_volume_used

    plt.figure(figsize=(12, 6))

    # Fitted archetype curves (use each archetype's fitted t_peak)
    for c_str, par in sorted(artifacts.archetype_params.items(), key=lambda kv: int(kv[0])):
        a = float(par["a"])
        t_peak = float(par["t_peak"])
        y_curve = gamma_norm(t, a=a, t_peak=t_peak)
        plt.plot(t, y_curve, linewidth=2, alpha=0.8, label=f"Archetype {int(c_str)}")

    # Artist simulated curve (shape based on artist peak_week + blended archetypes)
    plt.plot(t, y_sim_norm, color="black", linewidth=3, alpha=0.9, label="Simulated artist curve")

    plt.title(f"Fitted archetype curves + simulated curve: {args.artist}")
    plt.xlabel("Week since release")
    plt.ylabel("Normalized streams (peak = 1)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()

    if args.out_plot is not None:
        plt.savefig(args.out_plot, dpi=150)
        print(f"Wrote plot: {args.out_plot}")
    else:
        plt.show()

    # Also print the total lifecycle predicted streams for convenience.
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _api_training_config() -> SimpleNamespace:
    """Fixed training job matching module-level ``TRAIN_*`` constants (no CLI)."""
    return SimpleNamespace(
        parquet_path=str(ARCHETYPES_INPUT_PARQUET),
        out_dir=str(ARCHETYPES_ARTIFACTS_DIR),
        horizon_weeks=TRAIN_HORIZON_WEEKS,
        n_clusters=TRAIN_N_CLUSTERS,
        random_state=TRAIN_RANDOM_STATE,
        kmeans_batch_size=TRAIN_KMEANS_BATCH_SIZE,
        max_tracks_for_features=TRAIN_MAX_TRACKS_FOR_FEATURES,
    )


def main() -> None:
    """
    Recompute and overwrite archetype API artifacts: parquets + ``archetype_params.json`` under
    ``ARCHETYPES_ARTIFACTS_DIR``, reading from ``ARCHETYPES_INPUT_PARQUET``. No argv / CLI.
    """
    train(_api_training_config())


if __name__ == "__main__":
    main()

