"""
RIAA historical training panel — feature engineering.

Reads ``TIDE_HISTORICAL_TRAIN.parquet`` from S3 (or local path), engineers weekly
and release-level features, and writes three artifacts for downstream archetype
training and inference.

Outputs (default under ``model/data/``):
  - riaa_train_panel.parquet       — weekly rows + engineered columns
  - riaa_release_features.parquet  — one row per riaa_album_id
  - riaa_artist_profiles.json      — median w1_product_ratio by artist

Before clustering, albums are dropped unless max(cumulative_aeu) >= 5000 and
max(week) >= 12 (override via CLI). Re-run feature engineering before retraining.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_S3_URI = "s3://parquetgarage/model/data/TIDE_HISTORICAL_TRAIN.parquet"

PANEL_OUTPUT = "riaa_train_panel.parquet"
RELEASE_FEATURES_OUTPUT = "riaa_release_features.parquet"
ARTIST_PROFILES_OUTPUT = "riaa_artist_profiles.json"

HORIZON_WEEKS_DEFAULT = 78
DEEP_CATALOG_MIN_WEEK = 52
TAIL_WEEKS_MAX = 24
TAIL_WEEKS_MIN = 12
FLOOR_RATE_MIN = 0.002
FLOOR_RATE_MAX = 0.15
N_CLUSTERS_DEFAULT = 4

# Training eligibility (applied before release features / archetype train).
MIN_CUMULATIVE_AEU_TRAIN_DEFAULT = 5000.0
MIN_OBSERVED_WEEKS_TRAIN_DEFAULT = 12

# Source parquet columns (snake_case after normalization).
COL_ALBUM_ID = "riaa_album_id"
COL_WEEK_END = "week_end_date"
COL_RELEASE = "release_date"
COL_TOTAL = "total_album_equivalent"
COL_SALES = "product_sales"
COL_STREAMS = "streaming_equivalent"
COL_SONGS = "song_sale_equivalent"

METRIC_COLS = (COL_SALES, COL_STREAMS, COL_SONGS, COL_TOTAL)


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        for c in out.columns
    ]
    return out


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri.strip())
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Expected s3://bucket/key parquet URI, got {uri!r}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not key:
        raise ValueError(f"S3 URI missing object key: {uri!r}")
    return bucket, key


def load_riaa_historical_parquet(
    source: str,
    *,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load the RIAA historical panel from ``s3://...`` or a local file path.
    """
    src = (source or DEFAULT_S3_URI).strip()
    if src.lower().startswith("s3://"):
        bucket, key = _parse_s3_uri(src)
        try:
            import boto3
        except ImportError as e:
            raise ImportError("boto3 is required to read parquet from S3") from e
        logger.info("Loading parquet from s3://%s/%s", bucket, key)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        buf = io.BytesIO(body)
        try:
            df = pd.read_parquet(buf, columns=columns)
        except Exception:
            buf.seek(0)
            df = pd.read_parquet(buf, engine="fastparquet", columns=columns)
    else:
        path = Path(src).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Parquet not found: {path}")
        logger.info("Loading parquet from %s", path)
        df = pd.read_parquet(path, columns=columns)

    df = _normalize_column_names(df)
    required = {
        COL_ALBUM_ID,
        "title",
        "artist",
        "genre",
        COL_RELEASE,
        COL_WEEK_END,
        COL_SALES,
        COL_STREAMS,
        COL_SONGS,
        COL_TOTAL,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Historical parquet missing columns: {sorted(missing)}. "
            f"Available: {list(df.columns)}"
        )
    return df


def _coerce_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[COL_WEEK_END] = pd.to_datetime(out[COL_WEEK_END], errors="coerce")
    out[COL_RELEASE] = pd.to_datetime(out[COL_RELEASE], errors="coerce")
    for c in METRIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0)
    out[COL_ALBUM_ID] = out[COL_ALBUM_ID].astype(str).str.strip()
    return out.dropna(subset=[COL_ALBUM_ID, COL_WEEK_END])


def compute_week_index_riaa(
    df: pd.DataFrame,
    horizon_weeks: int,
    *,
    release_id_col: str = COL_ALBUM_ID,
) -> pd.DataFrame:
    """
    4.1 — ``weeks_since_release`` and aligned ``week`` (1..horizon) per album.
    """
    out = df.copy()
    first_dates = out.groupby(release_id_col)[COL_WEEK_END].transform("min")
    weeks_since = ((out[COL_WEEK_END] - first_dates).dt.days / 7).round().astype(int)
    out["weeks_since_release"] = weeks_since
    min_week = out.groupby(release_id_col)["weeks_since_release"].transform("min")
    out["week"] = (out["weeks_since_release"] - min_week) + 1
    out = out[(out["week"] >= 1) & (out["week"] <= int(horizon_weeks))].copy()
    out["week"] = out["week"].astype(int)
    return out


def add_cumulative_aeu(
    df: pd.DataFrame,
    *,
    release_id_col: str = COL_ALBUM_ID,
) -> pd.DataFrame:
    """4.2 — Running sum of ``total_album_equivalent`` per album (chronological)."""
    out = df.sort_values([release_id_col, COL_WEEK_END]).copy()
    out["cumulative_aeu"] = (
        out.groupby(release_id_col, sort=False)[COL_TOTAL].cumsum().astype(float)
    )
    return out


def filter_training_eligible_albums(
    panel: pd.DataFrame,
    *,
    min_cumulative_aeu: float = MIN_CUMULATIVE_AEU_TRAIN_DEFAULT,
    min_observed_weeks: int = MIN_OBSERVED_WEEKS_TRAIN_DEFAULT,
    release_id_col: str = COL_ALBUM_ID,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Drop albums that fail training gates before clustering / archetype fit.

    Keeps albums where:
      - max(cumulative_aeu) >= min_cumulative_aeu
      - max(week) >= min_observed_weeks (at least that many indexed weeks)
    """
    min_cumulative_aeu = float(min_cumulative_aeu)
    min_observed_weeks = int(min_observed_weeks)

    stats: Dict[str, int] = {
        "albums_before_filter": int(panel[release_id_col].nunique()),
        "rows_before_filter": int(len(panel)),
    }

    agg = panel.groupby(release_id_col, sort=False).agg(
        max_cumulative_aeu=("cumulative_aeu", "max"),
        max_week=("week", "max"),
        n_weeks=("week", "count"),
    )
    keep_ids = agg.index[
        (agg["max_cumulative_aeu"] >= min_cumulative_aeu)
        & (agg["max_week"] >= min_observed_weeks)
    ]
    dropped = agg.index.difference(keep_ids)

    low_cum = int((agg["max_cumulative_aeu"] < min_cumulative_aeu).sum())
    short_hist = int((agg["max_week"] < min_observed_weeks).sum())
    both = int(
        (
            (agg["max_cumulative_aeu"] < min_cumulative_aeu)
            & (agg["max_week"] < min_observed_weeks)
        ).sum()
    )

    out = panel[panel[release_id_col].isin(keep_ids)].copy()
    stats.update(
        {
            "albums_after_filter": int(out[release_id_col].nunique()),
            "rows_after_filter": int(len(out)),
            "albums_dropped": int(len(dropped)),
            "dropped_low_cumulative_aeu": low_cum,
            "dropped_short_history": short_hist,
            "dropped_both_reasons": both,
            "min_cumulative_aeu": min_cumulative_aeu,
            "min_observed_weeks": min_observed_weeks,
        }
    )
    logger.info(
        "Training eligibility filter: kept %d / %d albums (cum>=%s, weeks>=%d); "
        "dropped %d (low_cum=%d, short=%d, both=%d)",
        stats["albums_after_filter"],
        stats["albums_before_filter"],
        min_cumulative_aeu,
        min_observed_weeks,
        stats["albums_dropped"],
        low_cum,
        short_hist,
        both,
    )
    if out.empty:
        raise ValueError(
            "No albums remain after training eligibility filter; relax "
            f"min_cumulative_aeu={min_cumulative_aeu} or min_observed_weeks={min_observed_weeks}."
        )
    return out, stats


def _extract_shape_features(g: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Shape features on total AEU for provisional clustering (mirrors archetype FE)."""
    g = g.sort_values("week")
    w = g["week"].to_numpy(dtype=int)
    y = g[COL_TOTAL].to_numpy(dtype=float)
    if len(w) == 0:
        return None
    peak_idx = int(np.argmax(y))
    peak_y = float(y[peak_idx])
    if not np.isfinite(peak_y) or peak_y <= 0:
        return None
    y_norm = y / peak_y
    peak_week_obs = float(w[peak_idx])
    if np.any(w == 1):
        y_week1_norm = float(y_norm[w == 1][0])
    else:
        y_week1_norm = float(y_norm[0])
    y_last_norm = float(y_norm[w == w.max()][0]) if np.any(w == w.max()) else float(y_norm[-1])
    post_mask = w >= int(peak_week_obs)
    w_post = w[post_mask]
    y_post = y_norm[post_mask]
    half_life_week = float("nan")
    if len(w_post) >= 2:
        idx = np.where(y_post <= 0.5)[0]
        half_life_week = float(w_post[idx[0]]) if len(idx) else float(w_post.max())
    decay_log_slope = float("nan")
    if len(w_post) >= 3 and np.any(y_post > 0):
        mask = y_post > 0
        x = w_post[mask].astype(float)
        yy = np.log(np.clip(y_post[mask].astype(float), 1e-12, None))
        if len(x) >= 3:
            coeffs = np.polyfit(x, yy, 1)
            decay_log_slope = float(coeffs[0])
    auc_norm_time = float(np.trapezoid(y_norm, w) / max(float(w.max()), 1.0))
    return {
        "peak_volume_obs": peak_y,
        "peak_week_obs": peak_week_obs,
        "y_week1_norm": y_week1_norm,
        "y_last_norm": y_last_norm,
        "half_life_week": half_life_week,
        "decay_log_slope": decay_log_slope,
        "auc_norm_time": auc_norm_time,
        "max_week_obs": float(w.max()),
        "n_weeks_obs": float(len(w)),
    }


def build_release_features(
    panel: pd.DataFrame,
    *,
    n_clusters: int = N_CLUSTERS_DEFAULT,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    4.3–4.4 — Per-album peaks, week-1 product mix, shape cluster, catalog floor.
    """
    meta_cols = ["title", "artist", "genre", COL_RELEASE]
    meta = (
        panel.groupby(COL_ALBUM_ID, sort=False)[meta_cols]
        .agg("first")
        .reset_index()
    )

    records: List[Dict[str, Any]] = []
    for album_id, g in panel.groupby(COL_ALBUM_ID, sort=False):
        feats = _extract_shape_features(g)
        if feats is None:
            continue
        row: Dict[str, Any] = {COL_ALBUM_ID: album_id, **feats}
        w1 = g[g["week"] == 1]
        if not w1.empty:
            t1 = float(w1[COL_TOTAL].iloc[0])
            s1 = float(w1[COL_SALES].iloc[0])
            st1 = float(w1[COL_STREAMS].iloc[0])
            sg1 = float(w1[COL_SONGS].iloc[0])
            if t1 > 0:
                row["w1_product_ratio"] = float(np.clip(s1 / t1, 0.0, 1.0))
                row["w1_stream_ratio"] = float(np.clip(st1 / t1, 0.0, 1.0))
                row["w1_song_ratio"] = float(np.clip(sg1 / t1, 0.0, 1.0))
            else:
                row["w1_product_ratio"] = np.nan
                row["w1_stream_ratio"] = np.nan
                row["w1_song_ratio"] = np.nan
        else:
            row["w1_product_ratio"] = np.nan
            row["w1_stream_ratio"] = np.nan
            row["w1_song_ratio"] = np.nan
        records.append(row)

    if not records:
        raise ValueError("No releases with positive total_album_equivalent peak.")

    releases = pd.DataFrame.from_records(records).merge(meta, on=COL_ALBUM_ID, how="left")
    releases = _assign_archetype_clusters(releases, n_clusters=n_clusters, random_state=random_state)
    releases = _assign_catalog_floor_rate(releases, panel)
    return releases


def _assign_archetype_clusters(
    releases: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int,
) -> pd.DataFrame:
    """Provisional clusters for young-release floor lookup (refined at full train)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.impute import SimpleImputer

    feature_cols = [
        "peak_week_obs",
        "y_week1_norm",
        "y_last_norm",
        "half_life_week",
        "decay_log_slope",
        "auc_norm_time",
    ]
    X = releases[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    km = MiniBatchKMeans(
        n_clusters=int(n_clusters),
        random_state=int(random_state),
        batch_size=min(1024, max(256, len(releases) // 10)),
        n_init="auto",
    )
    out = releases.copy()
    out["archetype_cluster"] = km.fit_predict(X_imp).astype(int)
    return out


def _deep_catalog_self_floor(g: pd.DataFrame, peak_volume: float) -> Optional[float]:
    """
    Deep Catalog Rule: album observed past week 52 — median of last 12–24 weeks / peak.
    """
    if float(g["week"].max()) <= DEEP_CATALOG_MIN_WEEK:
        return None
    tail = g.sort_values("week").tail(TAIL_WEEKS_MAX)
    if len(tail) < TAIL_WEEKS_MIN:
        return None
    med = float(tail[COL_TOTAL].median())
    if med <= 0 or peak_volume <= 0:
        return None
    rate = med / peak_volume
    return float(np.clip(rate, FLOOR_RATE_MIN, FLOOR_RATE_MAX))


def _assign_catalog_floor_rate(
    releases: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    catalog_floor_rate:

    - Deep catalog (max observed week > 52): median(last 12–24 weeks) / peak_volume_obs.
    - Young release (max week <= 52): cluster median of deep-catalog self floors only.
    - Fallback: global median of deep-catalog self floors when cluster has no deep members.
    """
    out = releases.copy()
    self_floors: List[float] = []
    self_floor_by_album: Dict[str, float] = {}
    floor_source: List[str] = []

    panel_by_album = {k: v for k, v in panel.groupby(COL_ALBUM_ID, sort=False)}

    for _, row in out.iterrows():
        album_id = str(row[COL_ALBUM_ID])
        peak = float(row["peak_volume_obs"])
        g = panel_by_album.get(album_id)
        if g is None or g.empty:
            self_floor_by_album[album_id] = np.nan  # type: ignore[assignment]
            continue
        sf = _deep_catalog_self_floor(g, peak)
        self_floor_by_album[album_id] = sf if sf is not None else np.nan  # type: ignore[assignment]
        if sf is not None:
            self_floors.append(sf)

    global_median = float(np.median(self_floors)) if self_floors else float(FLOOR_RATE_MIN)
    global_median = float(np.clip(global_median, FLOOR_RATE_MIN, FLOOR_RATE_MAX))

    deep = out[out["max_week_obs"] > DEEP_CATALOG_MIN_WEEK].copy()
    cluster_medians: Dict[int, float] = {}
    if not deep.empty and self_floors:
        for cluster_id, chunk in deep.groupby("archetype_cluster", sort=False):
            vals = [
                self_floor_by_album[str(a)]
                for a in chunk[COL_ALBUM_ID]
                if str(a) in self_floor_by_album
                and np.isfinite(self_floor_by_album[str(a)])
            ]
            if vals:
                cluster_medians[int(cluster_id)] = float(
                    np.clip(np.median(vals), FLOOR_RATE_MIN, FLOOR_RATE_MAX)
                )

    catalog_rates: List[float] = []
    for _, row in out.iterrows():
        album_id = str(row[COL_ALBUM_ID])
        is_young = float(row["max_week_obs"]) <= DEEP_CATALOG_MIN_WEEK
        sf = self_floor_by_album.get(album_id)

        if not is_young and sf is not None and np.isfinite(sf):
            catalog_rates.append(float(sf))
            floor_source.append("deep_catalog")
            continue

        if is_young:
            cid = int(row["archetype_cluster"])
            rate = cluster_medians.get(cid, global_median)
            catalog_rates.append(rate)
            floor_source.append(
                "cluster_median" if cid in cluster_medians else "global_median"
            )
            continue

        # Mature but insufficient tail weeks for self floor.
        cid = int(row["archetype_cluster"])
        rate = cluster_medians.get(cid, global_median)
        catalog_rates.append(rate)
        floor_source.append(
            "cluster_median_insufficient_tail"
            if cid in cluster_medians
            else "global_median_insufficient_tail"
        )

    out["catalog_floor_rate"] = catalog_rates
    out["floor_source"] = floor_source
    return out


def build_artist_profiles(releases: pd.DataFrame) -> Dict[str, float]:
    """Median week-1 product sales share by artist (for inference defaults)."""
    sub = releases.dropna(subset=["w1_product_ratio", "artist"]).copy()
    if sub.empty:
        return {}
    agg = (
        sub.groupby("artist", sort=False)["w1_product_ratio"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return {str(k): float(v) for k, v in agg.items()}


def join_release_features_to_panel(
    panel: pd.DataFrame,
    releases: pd.DataFrame,
) -> pd.DataFrame:
    """Attach release-level features to each weekly row."""
    join_cols = [
        COL_ALBUM_ID,
        "peak_volume_obs",
        "peak_week_obs",
        "w1_product_ratio",
        "w1_stream_ratio",
        "w1_song_ratio",
        "catalog_floor_rate",
        "floor_source",
        "archetype_cluster",
        "max_week_obs",
    ]
    feat = releases[[c for c in join_cols if c in releases.columns]].copy()
    return panel.merge(feat, on=COL_ALBUM_ID, how="inner")


def run_feature_engineering(
    source: str = DEFAULT_S3_URI,
    *,
    out_dir: Path,
    horizon_weeks: int = HORIZON_WEEKS_DEFAULT,
    n_clusters: int = N_CLUSTERS_DEFAULT,
    random_state: int = 42,
    min_cumulative_aeu: float = MIN_CUMULATIVE_AEU_TRAIN_DEFAULT,
    min_observed_weeks: int = MIN_OBSERVED_WEEKS_TRAIN_DEFAULT,
    apply_train_filter: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: load → engineer → write three artifacts.

    Returns paths and summary counts.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_riaa_historical_parquet(source)
    df = _coerce_metrics(raw)
    n_albums_in = df[COL_ALBUM_ID].nunique()
    n_rows_in = len(df)

    panel = compute_week_index_riaa(df, horizon_weeks)
    panel = add_cumulative_aeu(panel)

    filter_stats: Dict[str, int] = {}
    if apply_train_filter:
        panel, filter_stats = filter_training_eligible_albums(
            panel,
            min_cumulative_aeu=min_cumulative_aeu,
            min_observed_weeks=min_observed_weeks,
        )

    releases = build_release_features(
        panel, n_clusters=n_clusters, random_state=random_state
    )
    profiles = build_artist_profiles(releases)
    panel = join_release_features_to_panel(panel, releases)

    panel_path = out_dir / PANEL_OUTPUT
    releases_path = out_dir / RELEASE_FEATURES_OUTPUT
    profiles_path = out_dir / ARTIST_PROFILES_OUTPUT

    panel.to_parquet(panel_path, index=False)
    releases.to_parquet(releases_path, index=False)
    profiles_path.write_text(json.dumps(profiles, indent=2, sort_keys=True), encoding="utf-8")

    young = int((releases["max_week_obs"] <= DEEP_CATALOG_MIN_WEEK).sum())
    deep_self = int((releases["floor_source"] == "deep_catalog").sum())

    summary = {
        "source": source,
        "out_dir": str(out_dir),
        "panel_path": str(panel_path),
        "release_features_path": str(releases_path),
        "artist_profiles_path": str(profiles_path),
        "rows_in": n_rows_in,
        "albums_in": int(n_albums_in),
        "rows_panel": int(len(panel)),
        "albums_panel": int(panel[COL_ALBUM_ID].nunique()),
        "albums_release_features": int(len(releases)),
        "artists_profiled": len(profiles),
        "young_releases_cluster_floor": young,
        "deep_catalog_self_floor": deep_self,
        "horizon_weeks": int(horizon_weeks),
        "apply_train_filter": apply_train_filter,
        "min_cumulative_aeu": float(min_cumulative_aeu),
        "min_observed_weeks": int(min_observed_weeks),
        **filter_stats,
    }
    logger.info("RIAA feature engineering complete: %s", summary)
    return summary


def genre_to_simulator_genres(val) -> str:
    """``extract_main_genre`` expects JSON; RIAA panel often has plain genre strings."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "[]"
    s = str(val).strip()
    if not s:
        return "[]"
    try:
        parsed = json.loads(s)
        if parsed:
            return s
    except (json.JSONDecodeError, TypeError):
        pass
    return json.dumps([{"CLIENT_DOMAIN": "Luminate", "MAIN_GENRE": s}])


def prepare_simulator_train_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: rename columns to ``all_data_archetypes_simulator_ae.train()`` schema.
    """
    out = panel.copy()
    out = out.rename(
        columns={
            COL_ALBUM_ID: "MRELG_ID",
            "artist": "DISPLAY_ARTIST",
            "genre": "GENRES",
            COL_RELEASE: "FIRST_SALE_DATE",
            COL_WEEK_END: "WEEK_END_DATE",
            COL_SALES: "PRODUCT_SALES",
            COL_STREAMS: "STREAMING_EQUIVALENT",
            COL_SONGS: "SONG_SALE_EQUIVALENT",
            COL_TOTAL: "TOTAL_ALBUM_EQUIVALENTS",
            "title": "TITLE",
        }
    )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="RIAA historical feature engineering.")
    p.add_argument(
        "--source",
        type=str,
        default=os.environ.get("TIDE_RIAA_HISTORICAL_S3_URI", DEFAULT_S3_URI),
        help=f"S3 URI or local parquet (default: {DEFAULT_S3_URI})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory for the three output artifacts",
    )
    p.add_argument("--horizon-weeks", type=int, default=HORIZON_WEEKS_DEFAULT)
    p.add_argument("--n-clusters", type=int, default=N_CLUSTERS_DEFAULT)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--min-cumulative-aeu",
        type=float,
        default=MIN_CUMULATIVE_AEU_TRAIN_DEFAULT,
        help="Drop albums whose max cumulative_aeu is below this (default 5000).",
    )
    p.add_argument(
        "--min-observed-weeks",
        type=int,
        default=MIN_OBSERVED_WEEKS_TRAIN_DEFAULT,
        help="Drop albums with fewer than this many indexed weeks (default 12).",
    )
    p.add_argument(
        "--no-train-filter",
        action="store_true",
        help="Keep all albums (disable cumulative / week-count gates).",
    )
    args = p.parse_args()

    summary = run_feature_engineering(
        args.source,
        out_dir=args.out_dir,
        horizon_weeks=args.horizon_weeks,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        min_cumulative_aeu=args.min_cumulative_aeu,
        min_observed_weeks=args.min_observed_weeks,
        apply_train_filter=not args.no_train_filter,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
