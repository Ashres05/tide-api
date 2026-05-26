from __future__ import annotations
import contextlib
import io
import json
import math
import pickle
import numbers
import os
import sqlite3
import logging
import time
import pandas as pd
import joblib

from datetime import datetime, date, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from sqlite_handler import (
    DATABASE_NAME,
    ensure_expected_releases_fw_columns,
    ensure_streaming_roster_2026_table,
    refresh_marketshare_search_summary,
    refresh_marketshare_search_summary_singles,
)
from search_text import normalize_search_text
import marketshare_from_csv
import album_art
from model.marketshare_75k_simulation import DISTRIBUTIONS, NUM_WEEKS
from model.train_catalog_decay import (
    BASELINE52_EPS,
    CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52,
    CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52,
    CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
    REL_RESIDUAL_CLIP_HIGH,
    REL_RESIDUAL_CLIP_LOW,
    _build_feature_frame,
    _normalize_artist_value,
    baseline52_median_from_history_stream,
    extract_main_genre,
    hybrid_inference_denominator_and_features,
)
from snowflake_conn import load_sql
from model.forecast_engine_server import ForecastEngine
from snowflake_conn import get_snowflake_connection, Snowflake
from train_model import refresh_data, update_parquet_metrics
from sqlite_handler import update_sqlite_main
from model.worldwide_streams_api import simulate_one_worldwide_streams
from api.s3_pull import (
    sync_artifacts_from_s3_if_configured,
    sync_artifacts_to_s3_if_configured,
    sync_db_from_s3,
    sync_db_to_s3,
    sync_full_inputs_from_s3,
    sync_full_outputs_to_s3,
    sync_parquets_to_s3,
    sync_weekly_inputs_from_s3,
    sync_weekly_outputs_to_s3,
)

# Set up logging.
logger = logging.getLogger(__name__)

# TODO: Upload API to EC2 instance.
# TODO: Make delete_release() function delete all release data from SQLite.

# Default training output: model/train_marketshare_artifacts.py writes here (not repo-root artifacts_75k).
ARTIFACTS_DIR = Path(__file__).resolve().parent / "model" / "artifacts_75k"

# Archetype decay artifact dirs — written by train_model.py via all_data_archetypes_simulator_ae.train().
_ARCHETYPES_BASE = Path(__file__).resolve().parent / "model" / "archetypes_artifacts"
ARCHETYPES_STREAMS_DIR = _ARCHETYPES_BASE / "streams"
ARCHETYPES_SALES_DIR   = _ARCHETYPES_BASE / "sales"
ARCHETYPES_SONGS_DIR   = _ARCHETYPES_BASE / "songs"
ARCHETYPES_SINGLES_BASE = _ARCHETYPES_BASE / "singles"
ARCHETYPES_SINGLES_STREAMS_DIR = ARCHETYPES_SINGLES_BASE / "streams"
ARCHETYPES_SINGLES_SALES_DIR = ARCHETYPES_SINGLES_BASE / "sales"
ARCHETYPES_SINGLES_SONGS_DIR = ARCHETYPES_SINGLES_BASE / "songs"
ARCHETYPES_WORLDWIDE_STREAMS_DIR = _ARCHETYPES_BASE / "worldwide_streams"
ARCHETYPES_WORLDWIDE_STREAMS_SINGLES_DIR = _ARCHETYPES_BASE / "worldwide_streams_singles"

# Release calendar keys that route decay to ARCHETYPES_SINGLES_* (see release_is_single).
RELEASE_PRODUCT_ALBUM = "album"
RELEASE_PRODUCT_SINGLE = "single"

# Forecast cache product discriminator (album vs single archetype bundles).
_WORLDWIDE_STREAMING_PRODUCT_ALBUM = "album"
_WORLDWIDE_STREAMING_PRODUCT_SINGLE = "single"


# SQLite table name for observed per-release metrics (populated by sqlite_handler.py)
MARKETSHARE_RELEASE_METRICS_TABLE = "MARKETSHARE_RELEASE_METRICS"
MARKETSHARE_SEARCH_SUMMARY_TABLE = "MARKETSHARE_SEARCH_SUMMARY"
MARKETSHARE_SEARCH_SUMMARY_SINGLES_TABLE = "MARKETSHARE_SEARCH_SUMMARY_SINGLES"

# Query names for the database.
RELEASE_CREATE_QUERY = "release_create.sql"
RELEASE_UPDATE_QUERY = "release_update.sql"
RELEASE_DELETE_QUERY = "release_delete.sql"
RELEASE_GET_QUERY = "release_get.sql"
RELEASE_GET_ALL_QUERY = "release_get_all.sql"
MARKETSHARE_ACTUALS_QUERY = "select_marketshare_actuals.sql"
MARKETSHARE_WEEKLY_ACTUALS_QUERY = "select_marketshare_weekly_actuals.sql"
MRELG_METADATA_QUERY = "query_mrelg_id.sql"
RELEASE_BACKFILL_QUERY = "query_release_backfill.sql"
STREAMING_ROSTER_YTD_QUERY = "query_streaming_roster_ytd.sql"
GLOBAL_STREAMING_QUERY = "query_release_global_streaming.sql"
CREATE_STREAMING_ROSTER_2026_TABLE = "create_streaming_roster_2026_table.sql"
INSERT_STREAMING_ROSTER_2026 = "insert_streaming_roster_2026.sql"
STREAMING_ROSTER_2026_LIST_QUERY = "streaming_roster_2026_list.sql"

_RELEASE_FIELD_KEYS = frozenset(
    {
        "mrelg_id",
        "name",
        "artist",
        "label_name",
        "release_date",
        "genre",
        "scenario",
        "known_vols",
        "fw_vol",
        "fw_streams",
        "fw_songs",
        "fw_sales",
        "fy_vol",
        "avg_historical_w1_product_ratio",
        "product_ratio_coefficient",
        "cluster",
    }
)

_REQUIRED_NONEMPTY_STR = (
    "name",
    "artist",
    "label_name",
    "release_date",
    "genre",
    "scenario",
)

_REAL_NUMERIC_FIELDS = (
    "fw_vol",
    "fw_streams",
    "fw_songs",
    "fw_sales",
    "fy_vol",
    "avg_historical_w1_product_ratio",
    "product_ratio_coefficient",
)

_ALLOWED_SCENARIOS = frozenset[str]({"Bear", "Base", "Bull"})

GLOBAL_FORECAST_ENGINE = None
GLOBAL_WORLDWIDE_ARTIFACTS = None
GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS = None
GLOBAL_1M_FORECASTER = None

# Default: catalog decay forecaster artifact in S3 (override with TIDE_CATALOG_DECAY_MODEL_S3_URI).
_CATALOG_DECAY_MODEL_S3_URI_DEFAULT = (
    #"s3://parquetgarage/model/catalog_decay_artifacts_slow/catalog_decay_model.pkl"
    #"s3://parquetgarage/model/decay_artifacts_multiplier/catalog_decay_model.pkl"
    #"s3://parquetgarage/model/decay_artifacts_multiplier_baseline/catalog_decay_model.pkl"
    #"s3://parquetgarage/model/decay_artifacts_hybrid/catalog_decay_model.pkl"
    "s3://parquetgarage/model/decay_artifacts_hybrid_baseline/catalog_decay_model.pkl"
)
# used to be _80k


def _catalog_eoy_ar_trailing_weeks_excluded() -> int:
    """
    Optional unconditional strip: drop this many trailing Snowflake weeks from the
    AR lag seed regardless of volume. Default ``0`` so we rely on
    ``_catalog_eoy_partial_last_week_ratio``; set ``TIDE_CATALOG_EOY_AR_EXCLUDE_TRAILING_WEEKS=1``
    to always omit the latest week (legacy behavior).
    """
    try:
        n = int(os.environ.get("TIDE_CATALOG_EOY_AR_EXCLUDE_TRAILING_WEEKS", "0").strip())
    except ValueError:
        n = 0
    return max(0, n)


def _catalog_eoy_partial_last_week_ratio() -> float:
    """
    If the last weekly global-stream total is below this fraction of the prior
    week, treat the last row as a partial / unsettled chart week and omit it
    from the AR ``rolling`` seed. By default the same trailing week(s) are also
    omitted from ``Actual`` rows in ``get_eoy_search_forecast`` so the last
    printed actual matches the lag anchor for the first forecast (see
    ``TIDE_CATALOG_EOY_INCLUDE_TRAILING_PARTIAL_ACTUALS``).

    Default ``0.75``. Set ``TIDE_CATALOG_EOY_PARTIAL_LAST_WEEK_RATIO=0`` to disable.
    Large real WoW drops can also trip this — tune (e.g. ``0.65``) if needed.
    """
    try:
        r = float(os.environ.get("TIDE_CATALOG_EOY_PARTIAL_LAST_WEEK_RATIO", "0.75").strip())
    except ValueError:
        r = 0.75
    if r <= 0:
        return 0.0
    return float(min(r, 0.9999))


def _catalog_eoy_ar_seed_trailing_trim(series: List[float]) -> int:
    """Weeks to drop from the end of ``series`` when building the AR lag seed only."""
    if len(series) < 2:
        return 0
    exclude = _catalog_eoy_ar_trailing_weeks_excluded()
    trim = max(0, min(exclude, len(series) - 1))
    ratio = _catalog_eoy_partial_last_week_ratio()
    if ratio > 0:
        prev_wk = float(series[-2])
        last_wk = float(series[-1])
        if prev_wk > 0 and last_wk < ratio * prev_wk:
            trim = max(trim, 1)
    return min(trim, len(series) - 1)


def _catalog_eoy_include_trailing_partial_actuals() -> bool:
    """
    If true, emit Snowflake trailing week(s) as ``Actual`` even when they are
    excluded from the AR seed (legacy chart: can show a low partial week next to
    a forecast anchored on full-week lags).
    """
    v = os.environ.get("TIDE_CATALOG_EOY_INCLUDE_TRAILING_PARTIAL_ACTUALS", "0").strip().lower()
    return v in ("1", "true", "yes")


def _catalog_decay_blend_alpha_and_mode() -> tuple[float, str]:
    """
    Optional convex blend after decoding the catalog-decay head:
    ``pred = alpha * model_level + (1 - alpha) * baseline_level``.

    - ``TIDE_CATALOG_DECAY_BLEND_ALPHA``: weight on the model in ``[0, 1]``.
      Default ``1.0`` (no blend). Example ``0.65`` keeps 65% model, 35% baseline.
    - ``TIDE_CATALOG_DECAY_BLEND_BASELINE``: ``lag1`` (repeat last week, default)
      or ``b52`` (prior 52-week median level from rolling history, matches training baseline).
    """
    try:
        alpha = float(os.environ.get("TIDE_CATALOG_DECAY_BLEND_ALPHA", "1.0").strip() or 1.0)
    except ValueError:
        alpha = 1.0
    alpha = max(0.0, min(1.0, alpha))
    mode = (os.environ.get("TIDE_CATALOG_DECAY_BLEND_BASELINE", "lag1") or "lag1").strip().lower()
    if mode not in ("lag1", "b52"):
        mode = "lag1"
    return alpha, mode


def _catalog_decay_simple_baseline_level(
    mode: str,
    *,
    lag1w: float,
    baseline_52w: float,
) -> float:
    """Simple non-learned level reference for blending (same units as streams/week)."""
    lf = float(lag1w)
    lag1 = max(0.0, lf if math.isfinite(lf) else 0.0)
    b = float(baseline_52w) if math.isfinite(baseline_52w) else float("nan")
    if mode == "b52" and math.isfinite(b) and b >= BASELINE52_EPS:
        return b
    return max(lag1, BASELINE52_EPS)


def _apply_catalog_decay_baseline_blend(
    model_level: float,
    *,
    lag1w: float,
    baseline_52w: float,
) -> float:
    """Blend decoded model weekly level with a simple baseline (see env in ``_catalog_decay_blend_alpha_and_mode``)."""
    alpha, mode = _catalog_decay_blend_alpha_and_mode()
    if alpha >= 1.0 - 1e-15:
        return float(model_level) if math.isfinite(float(model_level)) else 0.0
    base = _catalog_decay_simple_baseline_level(
        mode, lag1w=lag1w, baseline_52w=baseline_52w
    )
    mp = float(model_level) if math.isfinite(float(model_level)) else 0.0
    return max(0.0, alpha * mp + (1.0 - alpha) * base)


def _parse_s3_uri_to_bucket_key(uri: str) -> tuple[str, str]:
    u = uri.strip()
    low = u.lower()
    if not low.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri!r}")
    rest = u[5:]
    parts = rest.split("/", 1)
    bucket = parts[0].strip()
    if not bucket:
        raise ValueError(f"Invalid S3 URI (empty bucket): {uri!r}")
    key = parts[1].lstrip("/") if len(parts) > 1 else ""
    if not key:
        raise ValueError(f"Invalid S3 URI (empty key): {uri!r}")
    return bucket, key


# Optional 2025 catalog revenue by MRELG (CSV in S3). Loaded once per process.
_CATALOG_REVENUE_2025_BY_MRELG: Optional[Dict[str, float]] = None
_CATALOG_REVENUE_2025_LOAD_FAILED: bool = False


def _catalog_revenue_2025_csv_s3_uri() -> str:
    return os.environ.get(
        "TIDE_CATALOG_REVENUE_2025_CSV_S3_URI",
        "s3://parquetgarage/model/data/2025_revenue_catalog.csv",
    ).strip()


def _norm_csv_header(name: Any) -> str:
    return str(name).strip().upper().replace(" ", "_")


def _normalize_mrelg_id_key(raw: Any) -> str:
    """
    Canonical MRELG id string for CSV/API joins.

    Pandas often reads numeric MRELG_ID cells as floats (``12345.0``), which
    would not match SQLite/API string ``"12345"``. Strip Excel quirks and
    normalize integers to a stable decimal string.
    """
    if raw is None:
        return ""
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    if isinstance(raw, bool):
        return ""
    if isinstance(raw, numbers.Integral):
        return str(int(raw))
    if isinstance(raw, numbers.Real):
        rf = float(raw)
        if not math.isfinite(rf):
            return ""
        if rf == int(rf):
            return str(int(rf))
        s = str(rf).strip()
    else:
        s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    # Excel-style ="id" or 'id'
    if s.startswith("="):
        s = s[1:].strip().strip('"').strip("'")
    try:
        f = float(s)
        if math.isfinite(f) and f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s


def _load_catalog_revenue_2025_csv_map() -> Dict[str, float]:
    """
    Lazy-load a map of MRELG_ID -> 2025 revenue from the configured S3 CSV.
    Returns an empty dict if the file is missing or unreadable.

    (Named distinctly from ``get_catalog_revenue_2025_by_mrelg(mrelg_id)``, which
    reads a single id from SQLite for the Live Revenue API route.)
    """
    global _CATALOG_REVENUE_2025_BY_MRELG, _CATALOG_REVENUE_2025_LOAD_FAILED
    if _CATALOG_REVENUE_2025_LOAD_FAILED:
        return {}
    if _CATALOG_REVENUE_2025_BY_MRELG is not None:
        return _CATALOG_REVENUE_2025_BY_MRELG

    uri = _catalog_revenue_2025_csv_s3_uri()
    if not uri or not uri.lower().startswith("s3://"):
        logger.info("catalog_revenue_2025: no s3:// URI configured; skipping")
        _CATALOG_REVENUE_2025_BY_MRELG = {}
        return {}

    try:
        import boto3

        bucket, key = _parse_s3_uri_to_bucket_key(uri)
        raw = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as e:
        logger.warning("catalog_revenue_2025: could not load %s: %s", uri, e)
        _CATALOG_REVENUE_2025_LOAD_FAILED = True
        _CATALOG_REVENUE_2025_BY_MRELG = {}
        return {}

    if df.empty:
        _CATALOG_REVENUE_2025_BY_MRELG = {}
        return {}

    col_lookup = {_norm_csv_header(c): c for c in df.columns}
    # Prefer exact ``MRELG_ID`` (case/spacing-insensitive) to match Luminate / SQLite.
    mrelg_col = col_lookup.get("MRELG_ID")
    if not mrelg_col:
        mrelg_candidates = ("MRELG", "MRELGID", "MRELGIDS", "RELEASE_GROUP_ID")
        mrelg_col = next((col_lookup[c] for c in mrelg_candidates if c in col_lookup), None)
    rev_candidates = (
        "2025_REVENUE",
        "REVENUE_2025",
        "CATALOG_REVENUE_2025",
        "TOTAL_REVENUE_2025",
        "Y2025_REVENUE",
        "REVENUE",
        "TOTAL_REVENUE",
    )
    rev_col = next((col_lookup[c] for c in rev_candidates if c in col_lookup), None)
    if not mrelg_col or not rev_col:
        logger.warning(
            "catalog_revenue_2025: CSV missing expected columns (mrelg=%s revenue=%s) headers=%s",
            mrelg_col,
            rev_col,
            list(df.columns),
        )
        _CATALOG_REVENUE_2025_BY_MRELG = {}
        return {}

    out: Dict[str, float] = {}
    for _, row in df.iterrows():
        raw_id = row[mrelg_col] if mrelg_col in row.index else None
        mid = _normalize_mrelg_id_key(raw_id)
        if not mid:
            continue
        try:
            val = float(row[rev_col])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(val):
            continue
        out[mid] = val

    sample_keys = list(out.keys())[:3]
    logger.info(
        "catalog_revenue_2025: %d rows, MRELG column=%r revenue column=%r sample_mrelg_keys=%s uri=%s",
        len(out),
        mrelg_col,
        rev_col,
        sample_keys,
        uri,
    )
    _CATALOG_REVENUE_2025_BY_MRELG = out
    return out


def catalog_revenue_2025_for_mrelg(mrelg_id: Optional[str]) -> Optional[float]:
    """Return 2025 catalog revenue for ``mrelg_id``, or ``None`` if unknown."""
    key = _normalize_mrelg_id_key(mrelg_id)
    if not key:
        return None
    m = _load_catalog_revenue_2025_csv_map()
    if not m:
        return None
    v = m.get(key)
    if v is None and key.isdigit():
        # Rare: map built with int str but API sent zero-padded or spaced
        v = m.get(str(int(key)))
    return float(v) if v is not None and math.isfinite(float(v)) else None


def get_1m_forecaster():
    """
    Instantiate the catalog-decay (1M-scale) search forecaster once and return it.

    Loads ``catalog_decay_model.pkl`` from S3 (IAM role or env credentials).
    Override location with ``TIDE_CATALOG_DECAY_MODEL_S3_URI``.
    """
    global GLOBAL_1M_FORECASTER
    if GLOBAL_1M_FORECASTER is None:
        uri = os.environ.get(
            "TIDE_CATALOG_DECAY_MODEL_S3_URI", _CATALOG_DECAY_MODEL_S3_URI_DEFAULT
        ).strip()
        bucket, key = _parse_s3_uri_to_bucket_key(uri)
        logger.info("Loading catalog decay forecaster from s3://%s/%s", bucket, key)
        try:
            import boto3
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required to load the catalog decay model from S3"
            ) from e
        client = boto3.client("s3")
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
            raw = resp["Body"].read()
        except Exception as e:
            raise FileNotFoundError(
                f"Catalog decay model not found or unreadable at s3://{bucket}/{key}: {e}"
            ) from e
        buf = io.BytesIO(raw)
        try:
            buf.seek(0)
            GLOBAL_1M_FORECASTER = pickle.load(buf)
        except Exception:
            buf.seek(0)
            GLOBAL_1M_FORECASTER = joblib.load(buf)
        logger.info("Catalog decay forecaster loaded into memory.")
    return GLOBAL_1M_FORECASTER


# Column order matches ``catalog_streams_pruned_80k.parquet`` / train_catalog_decay.REQUIRED_COLUMNS + DISPLAY_ARTIST.
_CATALOG_STREAMS_PARQUET_OUTPUT_COLS: Tuple[str, ...] = (
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
    "DISPLAY_ARTIST",
    # Not in training parquet; added for API consumers (matches global_streaming data_type).
    "DATA_TYPE",
)


def _unpack_catalog_decay_bundle(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict) or "model" not in raw or "feature_columns" not in raw:
        raise TypeError(
            "catalog_decay artifact must be a dict with 'model' and 'feature_columns' "
            "(as produced by model.train_catalog_decay.write_artifacts)."
        )
    return raw


def _metadata_genre_to_parquet_genres(genre: Any) -> str:
    """Shape SQLite/Snowflake genre into GENRES JSON like catalog_streams parquet."""
    if genre is None or (isinstance(genre, float) and pd.isna(genre)):
        return json.dumps([{"CLIENT_DOMAIN": "Luminate", "MAIN_GENRE": "Unknown"}])
    s = str(genre).strip()
    if s.startswith("["):
        return s
    return json.dumps([{"CLIENT_DOMAIN": "Luminate", "MAIN_GENRE": s or "Unknown"}])


def _mean_tail(seq: List[float], n: int) -> float:
    if not seq:
        return 0.0
    tail = seq[-n:] if len(seq) >= n else seq
    return float(sum(tail) / len(tail))


def _catalog_decay_level_from_raw_pred(raw: float, target_transform: str) -> float:
    """Invert training target for **level** heads (not retained-multiplier models)."""
    tt = (target_transform or "none").strip().lower()
    if tt != "log1p":
        return max(0.0, float(raw))
    r = float(raw)
    if not math.isfinite(r):
        return 0.0
    r = max(-50.0, min(50.0, r))
    return max(0.0, math.expm1(r))


def _catalog_eoy_use_release_artist_hist() -> bool:
    v = os.environ.get("TIDE_CATALOG_EOY_USE_RELEASE_ARTIST_HIST", "1").strip().lower()
    return v not in ("0", "false", "no")


def _artist_history_log_map_for_inference(
    display_artist: str,
    weekly_streams: List[float],
    *,
    default_log_median: float,
) -> Optional[Dict[str, float]]:
    """
    Approximate training's per-artist ``artist_history_log_median`` with the log1p
    median of **this release's** observed weekly stream levels (same artist key
    normalization as ``train_catalog_decay``). Reduces train/serve skew vs always
    using the bundle's global default for every row.
    """
    if not _catalog_eoy_use_release_artist_hist():
        return None
    key = _normalize_artist_value(display_artist)
    vals: List[float] = []
    for x in weekly_streams:
        try:
            xf = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(xf) and xf > 0:
            vals.append(xf)
    if not vals:
        return {key: float(default_log_median)}
    med = float(pd.Series(vals).median())
    return {key: float(math.log1p(med))}


def _predict_catalog_decay_step(
    *,
    model: Any,
    feature_columns: List[str],
    top_genres: List[str],
    top_artists: List[str],
    artist_history_log_median: Optional[Dict[str, float]],
    artist_history_default_log_median: float,
    mrelg_id: str,
    title: str,
    display_artist: str,
    genres_json: str,
    release_dt: pd.Timestamp,
    first_sale_dt: pd.Timestamp,
    week_end: pd.Timestamp,
    weeks_since_release: float,
    lag1w: float,
    lag4w: float,
    lag12w: float,
    target_transform: str = "none",
    target_is_multiplier: bool = False,
    catalog_decay_target: str = CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
    baseline_52w_median: Optional[float] = None,
    hybrid_volatility_context: float = 0.0,
    hybrid_baseline_ratio: float = 1.0,
    hybrid_spike_state: float = 0.0,
    hybrid_decode_denom: Optional[float] = None,
) -> float:
    is_legacy = float(weeks_since_release) > 156.0
    is_whale = float(lag1w) > 5000000.0  # Tracks doing >5M streams/week

    chunk = pd.DataFrame(
        [
            {
                "MRELG_ID": mrelg_id,
                "TITLE": title,
                "DISPLAY_ARTIST": display_artist,
                "GENRES": genres_json,
                "RELEASE_DATE": release_dt,
                "FIRST_SALE_DATE": first_sale_dt,
                "WEEK_END_DATE": week_end,
                "WEEKS_SINCE_RELEASE": float(weeks_since_release),
            }
        ]
    )
    chunk["PARSED_MAIN_GENRE"] = chunk["GENRES"].map(extract_main_genre)
    ct0 = str(catalog_decay_target or CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1).strip()
    if ct0 == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        chunk["Volatility_Context"] = [float(hybrid_volatility_context)]
        chunk["Baseline_Ratio"] = [float(hybrid_baseline_ratio)]
        chunk["Spike_State"] = [float(hybrid_spike_state)]
    base = _build_feature_frame(
        chunk,
        top_genres=top_genres,
        top_artists=top_artists,
        artist_history_log_median=artist_history_log_median,
        artist_history_default_log_median=float(artist_history_default_log_median),
    )
    base["Lag1W_Streams"] = [float(lag1w)]
    base["Lag4W_Avg_Streams"] = [float(lag4w)]
    base["Lag12W_Avg_Streams"] = [float(lag12w)]
    x = base.reindex(columns=list(feature_columns)).fillna(0.0)
    raw = float(model.predict(x)[0])
    ct = ct0
    # Stable legacy catalog OR High-Volume "Whales": floor multiplier-like raw preds before decode to
    # slow AR "death spiral" from tree bias slightly below 1.0.
    if (
        (is_legacy or is_whale)
        and float(hybrid_volatility_context) < 0.15
        and ct != CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52
        and (
            ct == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52
            or (ct == CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1 and target_is_multiplier)
        )
        and math.isfinite(raw)
    ):
        raw = max(raw, 0.99)
    if ct == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52:
        r = raw if math.isfinite(raw) else 0.0
        r = max(REL_RESIDUAL_CLIP_LOW, min(REL_RESIDUAL_CLIP_HIGH, r))
        b = float(baseline_52w_median) if baseline_52w_median is not None else float("nan")
        lag1 = float(lag1w)
        if not math.isfinite(lag1) or lag1 < 0.0:
            lag1 = 0.0
        if not math.isfinite(b) or b < BASELINE52_EPS:
            b = max(lag1, BASELINE52_EPS)
        return max(0.0, b * (1.0 + r))
    if ct == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
        m = raw if math.isfinite(raw) else 0.0
        m = max(0.0, min(5.0, m))
        dd = float(hybrid_decode_denom) if hybrid_decode_denom is not None else float("nan")
        lag1 = float(lag1w)
        if not math.isfinite(lag1) or lag1 < 0.0:
            lag1 = 0.0
        if not math.isfinite(dd) or dd < BASELINE52_EPS:
            dd = max(lag1, BASELINE52_EPS)

        # ANTI-GRAVITY FIX: If it's a floored legacy/whale track, and the baseline is higher than lag1,
        # anchor the denominator to current reality to prevent phantom inflation.
        if (is_legacy or is_whale) and dd > lag1 and hybrid_spike_state < 0.5:
            dd = max(lag1, BASELINE52_EPS)

        return max(0.0, m * dd)
    if target_is_multiplier:
        # Matches ``train_catalog_decay.forecast_catalog_projects``: clip m, then m * lag1.
        m = raw if math.isfinite(raw) else 0.0
        m = max(0.0, min(5.0, m))
        lag1 = float(lag1w)
        if not math.isfinite(lag1) or lag1 < 0.0:
            lag1 = 0.0
        return max(0.0, m * lag1)
    return _catalog_decay_level_from_raw_pred(raw, target_transform)


def get_eoy_search_forecast(
    mrelg_id: str,
    target_year: int = 2026,
    scenario: str = "Base",
) -> pd.DataFrame:
    """
    Given an MRELG from search, load weekly worldwide streams history from Snowflake,
    then autoregress with the catalog-decay LightGBM bundle to ``target_year``-12-31.

    ``scenario`` ("Base" / "Bear" / "Bull") shocks the worldwide_streams archetype
    that powers the first 78 forecast weeks (the bridge weeks between the last
    actual and the AR handoff). The catalog-decay AR continuation past week 78
    has no learned scenario_multipliers itself, but inherits the shock through
    its ``lag1w / lag4w / lag12w`` features because ``rolling`` is seeded with
    the already-shocked bridge weeks before AR starts. By week ~77 the
    archetype has converged to the dynamic floor regardless of scenario, so
    the AR tail past the handoff is similar across Bear / Base / Bull — the
    user-visible scenario impact lives in weeks ~1–50 of the curve.

    Output columns match ``catalog_streams_pruned_80k.parquet`` (see
    ``_CATALOG_STREAMS_PARQUET_OUTPUT_COLS``). ``WORLDWIDE_STREAMS`` holds observed
    values for past weeks in the year and model predictions for future weeks.
    Lag columns are the trailing 1 / 4 / 12-week stream moments used for that row
    (prior week only for actuals; updated each AR step for forecasts).

    Forecast AR ``rolling`` omits trailing week(s) when either
    ``TIDE_CATALOG_EOY_AR_EXCLUDE_TRAILING_WEEKS`` forces it or the last week is
    much lower than the prior (partial week — see
    ``TIDE_CATALOG_EOY_PARTIAL_LAST_WEEK_RATIO``). Those same indices are omitted
    from ``Actual`` rows by default so the last actual volume aligns with
    ``LAG1W_STREAMS`` on the first forecast. Set
    ``TIDE_CATALOG_EOY_INCLUDE_TRAILING_PARTIAL_ACTUALS=1`` to show them again.

    Per-artist ``artist_hist_log_median`` for the model is inferred from this
    release's observed weekly streams (see ``_artist_history_log_map_for_inference``).
    Disable with ``TIDE_CATALOG_EOY_USE_RELEASE_ARTIST_HIST=0`` to restore the
    global-default-only path.

    Optional blend of the decoded model level with a simple baseline (tames
    over-decay from small / weak fits): set ``TIDE_CATALOG_DECAY_BLEND_ALPHA`` to
    a value in ``(0, 1)`` and optionally ``TIDE_CATALOG_DECAY_BLEND_BASELINE`` to
    ``lag1`` or ``b52``; see ``_catalog_decay_blend_alpha_and_mode``.
    """
    if not str(mrelg_id or "").strip():
        raise ValueError("mrelg_id is required.")
    mrelg_id = str(mrelg_id).strip()
    scenario = _normalize_scenario_label(scenario)

    # Cache hit short-circuits the entire pipeline (Snowflake history pull,
    # worldwide_streams archetype simulation, and the per-week catalog-decay
    # AR loop) so repeated Bear/Bull/Base toggles on the same release within
    # the TTL window return immediately. Keyed on (mrelg_id, target_year,
    # scenario, today) — see _SEARCH_FORECAST_CACHE.
    cached = _search_forecast_cache_lookup(mrelg_id, target_year, scenario)
    if cached is not None:
        return cached.copy()

    bundle = _unpack_catalog_decay_bundle(get_1m_forecaster())
    model = bundle["model"]
    feature_columns: List[str] = list(bundle["feature_columns"])
    top_genres: List[str] = list(bundle["top_genres"])
    top_artists: List[str] = list(bundle["top_artists"])
    artist_hist_default = float(bundle.get("artist_history_default_log_median", 0.0))
    target_transform = str(bundle.get("target_transform") or "none").strip().lower()
    # Retained-multiplier models (current ``train_catalog_decay``) decode as m * lag1;
    # log1p-level bundles use ``expm1`` only. Bundles omitting the flag are treated as
    # multiplier unless ``target_transform`` is log1p; set ``target_is_multiplier`` false
    # in the artifact dict for a legacy raw-level ``none`` model.
    if target_transform == "log1p":
        target_is_multiplier = False
    elif "target_is_multiplier" in bundle:
        target_is_multiplier = bool(bundle["target_is_multiplier"])
    else:
        target_is_multiplier = True

    catalog_decay_target = str(
        bundle.get("catalog_decay_target") or CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1
    ).strip()

    with get_snowflake_connection() as sf:
        meta = _resolve_mrelg_metadata_local(mrelg_id)
        if meta is None:
            meta = _resolve_mrelg_metadata_snowflake(mrelg_id, sf)
        release_date = _validate_date(meta.get("release_date"))
        artist = (meta.get("artist") or "").strip()
        title = (meta.get("title") or "").strip()
        genres_json = _metadata_genre_to_parquet_genres(meta.get("genre"))
        hist_df = _get_known_vols_global_streaming(mrelg_id, release_date, sf)
        # The first 78 forecast weeks come from the worldwide_streams archetype,
        # which is the only piece that owns scenario_multipliers. Pass the
        # scenario through so Bear / Bull shock those bridge weeks; the
        # catalog-decay AR (weeks 79..EOY) then inherits that shock through
        # its lag1w / lag4w / lag12w features without needing its own
        # multiplier table. The shock peaks early (weeks 5–50) and tapers
        # toward the dynamic floor by week 77, so the AR tail naturally
        # re-converges across scenarios — visible scenario impact lives in
        # the user-facing portion of the curve.
        #
        # ``hist_df`` is forwarded so _build_global_streaming_forecast doesn't
        # re-issue the same Snowflake query we just ran above — that was the
        # second-largest fixed cost on every search request before.
        stream_decay_df = _build_global_streaming_forecast(
            mrelg_id=mrelg_id,
            release_date=release_date,
            artist=artist,
            title=title,
            genre=meta.get("genre"),
            fw_streams_peak=0.0,
            sf=sf,
            scenario=scenario,
            hist_df=hist_df,
        )

    if hist_df.empty:
        raise ValueError(f"No historical streaming weeks for mrelg_id: {mrelg_id}")

    stream_col = next(
        (c for c in hist_df.columns if "stream" in c.lower()),
        hist_df.columns[-1],
    )
    date_col = next(c for c in hist_df.columns if "date" in c.lower())
    hist_df = hist_df.sort_values(date_col).reset_index(drop=True)
    series = pd.to_numeric(hist_df[stream_col], errors="coerce").fillna(0.0).astype(float).tolist()
    if not any(v > 0 for v in series):
        raise ValueError(f"No positive stream history for mrelg_id: {mrelg_id}")

    # Match training: per-artist log-median prior. Training uses corpus-wide artist
    # profiles; at serve time we approximate with this release's weekly levels.
    artist_hist_map = _artist_history_log_map_for_inference(
        artist,
        series,
        default_log_median=artist_hist_default,
    )

    week_ends = pd.to_datetime(hist_df[date_col]).dt.normalize()
    release_dt = pd.to_datetime(release_date).normalize()
    first_sale_dt = release_dt

    trim_tail = _catalog_eoy_ar_seed_trailing_trim(series)
    omit_trailing_actuals = trim_tail > 0 and not _catalog_eoy_include_trailing_partial_actuals()
    trim_from_i = len(series) - trim_tail  # drop actuals for indices >= this (same tail as AR seed)

    # Search/live uses the worldwide-streams archetype for the first 78 weeks.
    # At/after week 78, we hand off to catalog decay. Bridge weeks (archetype
    # output for weeks that fall after the last Snowflake observation but
    # still inside the 78-week archetype horizon) are emitted as
    # DATA_TYPE="Forecast" — they're model output, not real observations.
    # The chart can still render them continuously by concatenating Actual +
    # Forecast in chronological order; only the label changed (commits prior
    # to this used "Actual" for chart-styling reasons but that mislabeled
    # forecast weeks as actuals on the live-streaming popup).
    MODEL_HORIZON_WEEKS = 78.0
    stream_decay_by_week_end: Dict[pd.Timestamp, float] = {}
    if not stream_decay_df.empty:
        for _, r in stream_decay_df.iterrows():
            try:
                we = pd.to_datetime(r.get("week_ending_date")).normalize()
            except Exception:
                continue
            if int(we.year) != int(target_year):
                continue
            wsr = max(0.0, (we - release_dt).days / 7.0)
            if wsr < MODEL_HORIZON_WEEKS:
                v = float(r.get("pred_worldwide_streams") or 0.0)
                stream_decay_by_week_end[we] = max(0.0, v)

    rows_out: List[Dict[str, Any]] = []
    emitted_weeks: set[str] = set()

    # Actuals: weeks in ``target_year`` only (parquet-aligned rows).
    for i, we in enumerate(week_ends):
        if int(we.year) != int(target_year):
            continue
        if omit_trailing_actuals and i >= trim_from_i:
            # Skip weeks excluded from AR seed so last actual matches forecast lags.
            continue
        prefix = series[:i]
        lag1w = float(prefix[-1]) if prefix else 0.0
        lag4w = _mean_tail(prefix, 4)
        lag12w = _mean_tail(prefix, 12)
        wsr = max(0.0, (we - release_dt).days / 7.0)
        stream_bridge = stream_decay_by_week_end.get(we)
        observed_or_bridged = float(stream_bridge) if stream_bridge is not None else float(series[i])
        rows_out.append(
            {
                "MRELG_ID": mrelg_id,
                "TITLE": title,
                "GENRES": genres_json,
                "RELEASE_DATE": release_dt.strftime("%Y-%m-%d"),
                "FIRST_SALE_DATE": first_sale_dt.strftime("%Y-%m-%d"),
                "WEEK_END_DATE": we.strftime("%Y-%m-%d"),
                "WEEKS_SINCE_RELEASE": float(wsr),
                "WORLDWIDE_STREAMS": observed_or_bridged,
                "LAG1W_STREAMS": lag1w,
                "LAG4W_AVG_STREAMS": lag4w,
                "LAG12W_AVG_STREAMS": lag12w,
                "DISPLAY_ARTIST": artist,
                "DATA_TYPE": "Actual",
            }
        )
        emitted_weeks.add(we.strftime("%Y-%m-%d"))

    # AR seed for catalog-decay. We also append any stream-decay bridge weeks
    # beyond observed history (still <78w) so catalog picks up from that level.
    #
    # These ``bridge_future`` rows are forecasts produced by the
    # worldwide_streams archetype, NOT real Snowflake observations, so we now
    # label them DATA_TYPE="Forecast". (Previously they were emitted as
    # "Actual" to keep the chart line continuous, but that hid them from any
    # consumer that counts DATA_TYPE labels — e.g. the live-streaming popup
    # which was reporting "Nw actuals + 0w forecast" for releases where the
    # 78-week archetype window covered the rest of target_year.) The values
    # in WORLDWIDE_STREAMS are unchanged, so revenue totals and the chart
    # itself are unaffected; only the label flips.
    last_known_date = week_ends.iloc[-1]
    eoy = pd.Timestamp(year=int(target_year), month=12, day=31)
    ar_seed = series[: len(series) - trim_tail] if trim_tail else list(series)
    rolling = list(ar_seed)
    rolling_last_date = week_ends.iloc[len(ar_seed) - 1] if ar_seed else last_known_date

    bridge_future = sorted(
        d
        for d in stream_decay_by_week_end.keys()
        if d > rolling_last_date and d <= eoy
    )
    for d in bridge_future:
        lag1w = float(rolling[-1]) if rolling else 0.0
        lag4w = _mean_tail(rolling, 4)
        lag12w = _mean_tail(rolling, 12)
        wsr = max(0.0, (d - release_dt).days / 7.0)
        v = float(stream_decay_by_week_end[d])
        week_iso = d.strftime("%Y-%m-%d")
        if week_iso not in emitted_weeks:
            rows_out.append(
                {
                    "MRELG_ID": mrelg_id,
                    "TITLE": title,
                    "GENRES": genres_json,
                    "RELEASE_DATE": release_dt.strftime("%Y-%m-%d"),
                    "FIRST_SALE_DATE": first_sale_dt.strftime("%Y-%m-%d"),
                    "WEEK_END_DATE": week_iso,
                    "WEEKS_SINCE_RELEASE": float(wsr),
                    "WORLDWIDE_STREAMS": v,
                    "LAG1W_STREAMS": lag1w,
                    "LAG4W_AVG_STREAMS": lag4w,
                    "LAG12W_AVG_STREAMS": lag12w,
                    "DISPLAY_ARTIST": artist,
                    "DATA_TYPE": "Forecast",
                }
            )
            emitted_weeks.add(week_iso)
        rolling.append(v)
        rolling_last_date = d

    # Forecast: catalog-decay from the week after the stream-decay window/seed.
    current = rolling_last_date + pd.Timedelta(days=7)
    while current <= eoy:
        wsr = max(0.0, (current.normalize() - release_dt).days / 7.0)
        week_iso = current.strftime("%Y-%m-%d")
        if wsr < MODEL_HORIZON_WEEKS and current.normalize() in stream_decay_by_week_end:
            # Defensive path: if a pre-78w week wasn't emitted in the bridge
            # loop above, emit it here. These are also archetype-bridge
            # forecasts (same provenance as bridge_future), so we keep the
            # label consistent at "Forecast".
            lag1w = float(rolling[-1]) if rolling else 0.0
            lag4w = _mean_tail(rolling, 4)
            lag12w = _mean_tail(rolling, 12)
            v = float(stream_decay_by_week_end[current.normalize()])
            if week_iso not in emitted_weeks:
                rows_out.append(
                    {
                        "MRELG_ID": mrelg_id,
                        "TITLE": title,
                        "GENRES": genres_json,
                        "RELEASE_DATE": release_dt.strftime("%Y-%m-%d"),
                        "FIRST_SALE_DATE": first_sale_dt.strftime("%Y-%m-%d"),
                        "WEEK_END_DATE": week_iso,
                        "WEEKS_SINCE_RELEASE": float(wsr),
                        "WORLDWIDE_STREAMS": v,
                        "LAG1W_STREAMS": lag1w,
                        "LAG4W_AVG_STREAMS": lag4w,
                        "LAG12W_AVG_STREAMS": lag12w,
                        "DISPLAY_ARTIST": artist,
                        "DATA_TYPE": "Forecast",
                    }
                )
                emitted_weeks.add(week_iso)
            rolling.append(v)
            current = current + pd.Timedelta(days=7)
            continue

        lag1w = float(rolling[-1]) if rolling else 0.0
        lag4w = _mean_tail(rolling, 4)
        lag12w = _mean_tail(rolling, 12)
        bl52 = baseline52_median_from_history_stream(rolling)
        if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52 and not math.isfinite(
            bl52
        ):
            bl52 = float(rolling[-1]) if rolling else 0.0
        h_vol = h_br = h_sp = 0.0
        h_dd: Optional[float] = None
        if catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
            h_vol, h_br, h_sp, h_dd = hybrid_inference_denominator_and_features(rolling)
        pred = _predict_catalog_decay_step(
            model=model,
            feature_columns=feature_columns,
            top_genres=top_genres,
            top_artists=top_artists,
            artist_history_log_median=artist_hist_map,
            artist_history_default_log_median=artist_hist_default,
            mrelg_id=mrelg_id,
            title=title,
            display_artist=artist,
            genres_json=genres_json,
            release_dt=release_dt,
            first_sale_dt=first_sale_dt,
            week_end=current.normalize(),
            weeks_since_release=wsr,
            lag1w=lag1w,
            lag4w=lag4w,
            lag12w=lag12w,
            target_transform=target_transform,
            target_is_multiplier=target_is_multiplier,
            catalog_decay_target=catalog_decay_target,
            baseline_52w_median=bl52,
            hybrid_volatility_context=h_vol,
            hybrid_baseline_ratio=h_br,
            hybrid_spike_state=h_sp,
            hybrid_decode_denom=h_dd,
        )
        pred = _apply_catalog_decay_baseline_blend(
            float(pred), lag1w=lag1w, baseline_52w=bl52
        )
        rows_out.append(
            {
                "MRELG_ID": mrelg_id,
                "TITLE": title,
                "GENRES": genres_json,
                "RELEASE_DATE": release_dt.strftime("%Y-%m-%d"),
                "FIRST_SALE_DATE": first_sale_dt.strftime("%Y-%m-%d"),
                "WEEK_END_DATE": current.strftime("%Y-%m-%d"),
                "WEEKS_SINCE_RELEASE": float(wsr),
                "WORLDWIDE_STREAMS": float(pred),
                "LAG1W_STREAMS": lag1w,
                "LAG4W_AVG_STREAMS": lag4w,
                "LAG12W_AVG_STREAMS": lag12w,
                "DISPLAY_ARTIST": artist,
                "DATA_TYPE": "Forecast",
            }
        )
        emitted_weeks.add(week_iso)
        rolling.append(pred)
        current = current + pd.Timedelta(days=7)

    out = pd.DataFrame(rows_out)
    if out.empty:
        return out
    out = out.reindex(columns=list(_CATALOG_STREAMS_PARQUET_OUTPUT_COLS))
    _search_forecast_cache_store(mrelg_id, target_year, out, scenario)
    return out


# ---------------------------------------------------------------------------
# Performance instrumentation
# ---------------------------------------------------------------------------
# Lightweight timing helper. Logs INFO with a stable structured prefix so the
# timings are easy to grep in api_refresh / FastAPI logs:
#
#     PERF endpoint=search_global_streaming phase=db_fetch ms=412.1 rows=403211
#
# Each top-level endpoint creates a span dict, then writes a final aggregated
# line at the end summarising every phase + total wall time. Phase timers also
# log per-phase so we still see partial progress on slow requests.

_PERF_LOG_PREFIX = "PERF"


def _now() -> float:
    return time.perf_counter()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


@contextlib.contextmanager
def _perf_phase(
    span: Dict[str, Any], phase: str, *, endpoint: str, **extra: Any
) -> Iterator[Dict[str, Any]]:
    """
    Time a single phase of an endpoint and append it to ``span``.

    ``extra`` is mutable inside the with-block (the dict is yielded so callers
    can record observed counts), and is logged once on exit alongside the
    phase duration.
    """
    t0 = _now()
    info: Dict[str, Any] = dict(extra)
    try:
        yield info
    finally:
        duration_ms = _ms(t0)
        record = {"phase": phase, "ms": duration_ms, **info}
        span.setdefault("phases", []).append(record)
        kv = " ".join(f"{k}={v}" for k, v in record.items())
        logger.info("%s endpoint=%s %s", _PERF_LOG_PREFIX, endpoint, kv)


def _perf_summary(span: Dict[str, Any], *, endpoint: str, t_start: float, **extra: Any) -> None:
    """Emit the final aggregated PERF log for an endpoint span."""
    total_ms = _ms(t_start)
    span["total_ms"] = total_ms
    span.update(extra)
    parts = [f"total_ms={total_ms}"]
    for k, v in extra.items():
        parts.append(f"{k}={v}")
    logger.info(
        "%s endpoint=%s phase=__summary__ %s",
        _PERF_LOG_PREFIX,
        endpoint,
        " ".join(parts),
    )


# ---------------------------------------------------------------------------
# Forecast response cache
# ---------------------------------------------------------------------------
# Bounded TTL cache keyed by (mrelg_id, daily-data-cutoff). The Snowflake
# query for global streaming filters by ``CURRENT_DATE() - 2 days`` so the
# answer is stable for a given (mrelg_id, today) pair. We keep entries for
# ``_FORECAST_CACHE_TTL_S`` and bound the cache size to ``_FORECAST_CACHE_MAX``
# so unbounded growth doesn't pin memory in long-running API workers.
_FORECAST_CACHE_TTL_S = 6 * 60 * 60  # 6 hours
_FORECAST_CACHE_MAX = 256


import threading as _threading  # noqa: E402  (kept local to forecast cache)

_FORECAST_CACHE: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
_FORECAST_CACHE_LOCK = _threading.Lock()
# Search forecast cache (catalog-EOY view powering /v1/forecast/search). The
# search path is materially more expensive than the 78-week revenue path
# because it adds a per-week catalog-decay AR loop on top of the worldwide
# archetype, and on every call without this cache the request paid for two
# Snowflake history queries + a full archetype simulate + ~52 LightGBM
# inference steps. Keyed on (mrelg_id, target_year, scenario, today) so that
# Bear/Bull toggles on the same release day-of return immediately and
# different target years stay isolated.
_SEARCH_FORECAST_CACHE: Dict[Tuple[str, int, str, str], Tuple[float, pd.DataFrame]] = {}
_SEARCH_FORECAST_CACHE_LOCK = _threading.Lock()
_MARKETSHARE_YTD_CACHE: Optional[pd.DataFrame] = None
_MARKETSHARE_YTD_CACHE_LOCK = _threading.Lock()


def _normalize_scenario_label(scenario: Optional[str]) -> str:
    """Coerce arbitrary user input to canonical Bear / Base / Bull (defaulting to Base).

    Mirrors ``normalize_archetype_scenario_label`` in the simulator without
    pulling that import into model_handler's hot path. Used for the forecast
    cache key so Bear/Bull don't collide with Base in cached responses.
    """
    if scenario is None:
        return "Base"
    s = str(scenario).strip().lower()
    if s == "bear":
        return "Bear"
    if s == "bull":
        return "Bull"
    return "Base"


def _forecast_cache_key(
    mrelg_id: str,
    scenario: str = "Base",
    *,
    product: str = _WORLDWIDE_STREAMING_PRODUCT_ALBUM,
) -> Tuple[str, str, str, str]:
    # Bind to the calendar date so refreshed Snowflake data is picked up the
    # next day even if the worker has not been restarted; the TTL still
    # guards against same-day invalidation if the daily snapshot changes.
    # Scenario is part of the key because Bear / Base / Bull return materially
    # different forecast frames for the same release on the same day.
    # Product separates album vs singles archetype bundles when both endpoints
    # share the same mrelg_id key space (defensive; Luminate MRELGs are unique).
    return (
        product,
        mrelg_id,
        _normalize_scenario_label(scenario),
        date.today().isoformat(),
    )


def _forecast_cache_lookup(
    mrelg_id: str,
    scenario: str = "Base",
    *,
    product: str = _WORLDWIDE_STREAMING_PRODUCT_ALBUM,
) -> Optional[pd.DataFrame]:
    key = _forecast_cache_key(mrelg_id, scenario, product=product)
    with _FORECAST_CACHE_LOCK:
        entry = _FORECAST_CACHE.get(key)
        if entry is None:
            return None
        ts, df = entry
        if (time.time() - ts) > _FORECAST_CACHE_TTL_S:
            _FORECAST_CACHE.pop(key, None)
            return None
    return df


def _forecast_cache_store(
    mrelg_id: str,
    df: pd.DataFrame,
    scenario: str = "Base",
    *,
    product: str = _WORLDWIDE_STREAMING_PRODUCT_ALBUM,
) -> None:
    if df is None or df.empty:
        return
    key = _forecast_cache_key(mrelg_id, scenario, product=product)
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE[key] = (time.time(), df.copy())
        # Cheap LRU-ish eviction: drop the oldest entries beyond the cap.
        if len(_FORECAST_CACHE) > _FORECAST_CACHE_MAX:
            evict = sorted(_FORECAST_CACHE.items(), key=lambda kv: kv[1][0])
            for k, _ in evict[: len(_FORECAST_CACHE) - _FORECAST_CACHE_MAX]:
                _FORECAST_CACHE.pop(k, None)


def _search_forecast_cache_key(
    mrelg_id: str, target_year: int, scenario: str = "Base"
) -> Tuple[str, int, str, str]:
    return (
        mrelg_id,
        int(target_year),
        _normalize_scenario_label(scenario),
        date.today().isoformat(),
    )


def _search_forecast_cache_lookup(
    mrelg_id: str, target_year: int, scenario: str = "Base"
) -> Optional[pd.DataFrame]:
    key = _search_forecast_cache_key(mrelg_id, target_year, scenario)
    with _SEARCH_FORECAST_CACHE_LOCK:
        entry = _SEARCH_FORECAST_CACHE.get(key)
        if entry is None:
            return None
        ts, df = entry
        if (time.time() - ts) > _FORECAST_CACHE_TTL_S:
            _SEARCH_FORECAST_CACHE.pop(key, None)
            return None
    return df


def _search_forecast_cache_store(
    mrelg_id: str, target_year: int, df: pd.DataFrame, scenario: str = "Base"
) -> None:
    if df is None or df.empty:
        return
    key = _search_forecast_cache_key(mrelg_id, target_year, scenario)
    with _SEARCH_FORECAST_CACHE_LOCK:
        _SEARCH_FORECAST_CACHE[key] = (time.time(), df.copy())
        if len(_SEARCH_FORECAST_CACHE) > _FORECAST_CACHE_MAX:
            evict = sorted(_SEARCH_FORECAST_CACHE.items(), key=lambda kv: kv[1][0])
            for k, _ in evict[: len(_SEARCH_FORECAST_CACHE) - _FORECAST_CACHE_MAX]:
                _SEARCH_FORECAST_CACHE.pop(k, None)


def forecast_cache_clear() -> None:
    """Drop all cached forecast responses (used after a data refresh)."""
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE.clear()
    with _SEARCH_FORECAST_CACHE_LOCK:
        _SEARCH_FORECAST_CACHE.clear()


def marketshare_cache_clear() -> None:
    """Drop cached marketshare unified_ytd frame (explicit invalidation only)."""
    global _MARKETSHARE_YTD_CACHE
    with _MARKETSHARE_YTD_CACHE_LOCK:
        _MARKETSHARE_YTD_CACHE = None


def _cap_weekly_series(values: List[float] | None, max_weeks: int) -> List[float]:
    """Keep only the first ``max_weeks`` points (chronological SQLite order)."""
    if not values:
        return []
    if len(values) <= max_weeks:
        return [float(x) for x in values]
    return [float(x) for x in values[:max_weeks]]


_PARTIAL_WEEK_LAG_DAYS = 3


def _strip_partial_known_week(release_map: dict, lag_days: int = _PARTIAL_WEEK_LAG_DAYS) -> dict:
    """
    For simulation inputs only, treat the trailing in-progress known week as
    partial and let the model forecast that slot.

    This prevents a mid-week dip where WEEK_ENDING_DATE has only a few days of
    observed volume and would otherwise override the model at the boundary.
    """
    known_vols = release_map.get("known_vols")
    known_dates = release_map.get("known_week_dates")
    if not isinstance(known_vols, list) or not known_vols:
        return release_map
    if not isinstance(known_dates, list) or len(known_dates) != len(known_vols):
        return release_map

    try:
        last_date = datetime.strptime(str(known_dates[-1]), "%Y-%m-%d").date()
    except Exception:
        return release_map

    cutoff = datetime.utcnow().date() - timedelta(days=lag_days)
    if last_date < cutoff:
        return release_map

    # Drop the trailing partial point from all aligned known series.
    release_map["known_vols"] = known_vols[:-1]
    release_map["known_week_dates"] = known_dates[:-1]
    for key in ("known_streams", "known_sales", "known_songs"):
        vals = release_map.get(key)
        if isinstance(vals, list) and len(vals) >= len(known_vols):
            release_map[key] = vals[:-1]
    return release_map


def _sanitize_release_for_simulation(release_map: dict) -> dict:
    """
    Normalize per-release inputs before ``ForecastEngine.simulate``:
    cap known weekly series to the same horizon as the archetype decay window
    so ``fit_backfill_forecast`` never sees more actuals than ``end_week``.
    """
    for key in ("known_vols", "known_streams", "known_sales", "known_songs"):
        if key in release_map and release_map[key]:
            release_map[key] = _cap_weekly_series(release_map[key], NUM_WEEKS)
    # Keep known_week_dates aligned with known_vols after the cap so the
    # frontend can still identify the in-progress week by date.
    dates = release_map.get("known_week_dates")
    if isinstance(dates, list) and dates:
        if len(dates) > NUM_WEEKS:
            release_map["known_week_dates"] = dates[:NUM_WEEKS]
    return _strip_partial_known_week(release_map)


def _sanitize_forecast_engine_artifacts(engine: ForecastEngine) -> None:
    """
    In-memory cleanup of parquet-backed frames after load. Keeps files on disk
    unchanged while fixing NaNs / dtypes that would break ``run_archetype_scenario``.
    """
    df = engine.df_full
    if df is None or getattr(df, "empty", True):
        return

    wk_col = "Week Ending Date"
    if wk_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[wk_col]):
        df[wk_col] = pd.to_datetime(df[wk_col], errors="coerce")

    if "Predicted_Baseline_Share" in df.columns:
        if "Owner" in df.columns:
            bad = df["Predicted_Baseline_Share"].isna()
            if bad.any():
                logger.warning(
                    "df_full: repairing %d NaN Predicted_Baseline_Share cells (per-Owner ffill/bfill)",
                    int(bad.sum()),
                )
                df["Predicted_Baseline_Share"] = (
                    df.groupby("Owner", sort=False)["Predicted_Baseline_Share"]
                    .transform(lambda s: s.ffill().bfill())
                )
        df["Predicted_Baseline_Share"] = pd.to_numeric(
            df["Predicted_Baseline_Share"], errors="coerce"
        ).fillna(0.0)

    if "Final_Unified_Share" in df.columns and "Predicted_Baseline_Share" in df.columns:
        mask = df["Final_Unified_Share"].isna()
        if mask.any():
            df.loc[mask, "Final_Unified_Share"] = df.loc[mask, "Predicted_Baseline_Share"]

    for col in ("big_release_flag", "Incremental_Share"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    act = engine.actuals_2026
    if act is None or getattr(act, "empty", True):
        return
    if wk_col in act.columns and not pd.api.types.is_datetime64_any_dtype(act[wk_col]):
        act[wk_col] = pd.to_datetime(act[wk_col], errors="coerce")
    if "AE_Share" in act.columns:
        act["AE_Share"] = pd.to_numeric(act["AE_Share"], errors="coerce").fillna(0.0)


def _archetypes_singles_decay_dir(metric: str) -> Path:
    """Resolve singles decay subdir (streams / sales / songs) with optional env override."""
    override = os.environ.get("TIDE_SINGLES_DECAY_ARTIFACTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve() / metric
    return ARCHETYPES_SINGLES_BASE / metric


def _forecast_engine_singles_decay_paths() -> dict[str, Optional[Path]]:
    """Paths passed into ForecastEngine when singles bundles exist on disk."""
    streams = _archetypes_singles_decay_dir("streams")
    songs = _archetypes_singles_decay_dir("songs")
    sales = _archetypes_singles_decay_dir("sales")
    out: dict[str, Optional[Path]] = {
        "singles_streams_dir": streams if (streams / "archetype_params.json").is_file() else None,
        "singles_songs_dir": songs if (songs / "archetype_params.json").is_file() else None,
        "singles_sales_dir": sales if (sales / "archetype_params.json").is_file() else None,
    }
    if out["singles_streams_dir"] is None:
        logger.info(
            "Singles decay artifacts not found under %s (sync "
            "s3://parquetgarage/model/archetypes_artifacts/singles/).",
            ARCHETYPES_SINGLES_BASE,
        )
    return out


def get_engine() -> ForecastEngine:
    """
    Lazily build the shared ForecastEngine (album + optional singles decay bundles).

    Singles routing is per release via ``product_type`` / ``release_type`` on the
    calendar row (see ``release_is_single``). EXPECTED_RELEASES will gain
    ``PRODUCT_TYPE`` in a later migration; until then rows default to album decay.
    """
    global GLOBAL_FORECAST_ENGINE
    if GLOBAL_FORECAST_ENGINE is None:
        GLOBAL_FORECAST_ENGINE = ForecastEngine(
            artifacts_dir=ARTIFACTS_DIR,
            streams_dir=ARCHETYPES_STREAMS_DIR,
            sales_dir=ARCHETYPES_SALES_DIR,
            songs_dir=ARCHETYPES_SONGS_DIR,
            **_forecast_engine_singles_decay_paths(),
        )
        _sanitize_forecast_engine_artifacts(GLOBAL_FORECAST_ENGINE)
    return GLOBAL_FORECAST_ENGINE


def _worldwide_streams_artifacts_dir(*, singles: bool) -> Path:
    """Resolve on-disk archetype bundle directory (album or singles)."""
    if singles:
        override = os.environ.get("TIDE_WORLDWIDE_STREAMS_SINGLES_ARTIFACTS_DIR", "").strip()
        if override:
            return Path(override).expanduser().resolve()
        return ARCHETYPES_WORLDWIDE_STREAMS_SINGLES_DIR
    override = os.environ.get("TIDE_WORLDWIDE_STREAMS_ARTIFACTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return ARCHETYPES_WORLDWIDE_STREAMS_DIR


def get_worldwide_artifacts():
    """Loads album worldwide-streams archetype artifacts once and returns them."""
    global GLOBAL_WORLDWIDE_ARTIFACTS
    if GLOBAL_WORLDWIDE_ARTIFACTS is None:
        from model.all_data_archetypes_simulator_ae import load_artifacts

        artifact_dir = _worldwide_streams_artifacts_dir(singles=False)
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Worldwide streams artifacts not found at {artifact_dir}. "
                "Run training with --metric worldwide_streams first."
            )
        GLOBAL_WORLDWIDE_ARTIFACTS = load_artifacts(str(artifact_dir))
    return GLOBAL_WORLDWIDE_ARTIFACTS


def get_worldwide_singles_artifacts():
    """Loads singles worldwide-streams archetype artifacts once and returns them."""
    global GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS
    if GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS is None:
        from model.all_data_archetypes_simulator_ae import load_artifacts

        artifact_dir = _worldwide_streams_artifacts_dir(singles=True)
        if not artifact_dir.exists():
            raise FileNotFoundError(
                f"Worldwide streams (singles) artifacts not found at {artifact_dir}. "
                "Sync from s3://parquetgarage/model/worldwide_streams_singles/ or set "
                "TIDE_WORLDWIDE_STREAMS_SINGLES_ARTIFACTS_DIR."
            )
        GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS = load_artifacts(str(artifact_dir))
    return GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS


def create_release(
    *,
    mrelg_id: str | None = None,
    name: str, 
    artist: str, 
    label_name: str, 
    release_date: str, 
    genre: str, 
    scenario: str, 
    fw_vol: float, 
    product_type: str | None = None,
    fw_streams: float = 0.0,
    fw_songs: float = 0.0,
    fw_sales: float = 0.0,
    fy_vol: float = 0.0, 
    known_vols: list[float] | None = None,
    avg_historical_w1_product_ratio: float = 0.3, 
    product_ratio_coefficient: float = 0.3,
    cluster: int | None = None,
    **_kwargs,
) -> int:
    """
    Creates a new release in the database.
    Returns the release ID.
    """
    if known_vols is None:
        known_vols = []
    _verify_release_fields(locals())

    params = (
        mrelg_id,
        name,
        artist,
        label_name,
        release_date,
        genre,
        product_type,
        fw_vol,
        fw_streams,
        fw_songs,
        fw_sales,
        scenario,
        fy_vol,
        avg_historical_w1_product_ratio,
        product_ratio_coefficient,
        cluster,
    )
    try:
        query = load_sql(RELEASE_CREATE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            cursor = conn.cursor()
            id = cursor.execute(query, params).fetchone()[0]
            conn.commit()
        marketshare_cache_clear()
        return id
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error creating release: {e}") from e


def _create_backfilled_release(
    *,
    mrelg_id: str,
    label_name: str,
    product_type: str | None = None,
    scenario: str = "Base",
    fw_vol: float = 100000.0,  # Default; irrelevant once known_vols are loaded.
    _sf: "Snowflake | None" = None,
) -> int:
    """
    Creates a new release in the database from a mrelg_id.
    Returns the release ID.

    Pass _sf to reuse an existing Snowflake connection (avoids one connect per release).
    """
    import contextlib

    @contextlib.contextmanager
    def _maybe_conn():
        if _sf is not None:
            yield _sf
        else:
            with get_snowflake_connection() as sf2:
                yield sf2

    with _maybe_conn() as sf:
        mrelg_metadata = _verify_mrelg_id(mrelg_id, sf)

    name = mrelg_metadata["TITLE"].iloc[0]
    artist = mrelg_metadata["DISPLAY_ARTIST"].iloc[0]
    release_date = _validate_date(mrelg_metadata["RELEASE_DATE"].iloc[0])
    genre = mrelg_metadata["GENRE"].iloc[0]

    # Temporary adjustment to genres to ensure they are in the distribution.
    # TODO: Remove this once the right genres are in query_mrelg_id.sql
    genre_aliases = {
        "Alt. Rock": "Rock",
        "R&B": "R&B/Hip-Hop",
    }
    genre = genre_aliases.get(genre, genre)
    if genre not in DISTRIBUTIONS["Genre"]:
        genre = "Pop"

    metadata_cols = [name, artist, release_date, genre]

    for col in metadata_cols:
        if col is None:
            raise ValueError(
                f"MRELG ID {mrelg_id} has no {col} in the metadata. Try create_release() instead."
            )

    rid = create_release(
        mrelg_id=mrelg_id,
        name=name,
        artist=artist,
        label_name=label_name,
        release_date=release_date,
        genre=genre,
        product_type=product_type,
        scenario=scenario,
        fw_vol=fw_vol,
        fy_vol=0.0,
        avg_historical_w1_product_ratio=0.3,
        product_ratio_coefficient=0.3,
        cluster=None,
    )
    marketshare_cache_clear()
    return rid


def _backfill_candidates_from_df(df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    """
    Normalize Snowflake backfill rows to (mrelg_id, release_type, label_name).

    Supports query_release_backfill.sql with or without release_type (legacy 2-col).
    """
    norm = {str(c).strip().lower(): c for c in df.columns}
    for required in ("mrelg_id", "label_name"):
        if required not in norm:
            raise ValueError(
                f"backfill query must return mrelg_id and label_name; got {list(df.columns)}"
            )
    has_type = "release_type" in norm
    out: List[Tuple[str, str, str]] = []
    for row in df.itertuples(index=False, name=None):
        if has_type and len(row) >= 3:
            mid, rtype, lbl = row[0], row[1], row[2]
        elif len(row) >= 2:
            mid, lbl = row[0], row[-1]
            rtype = row[1] if has_type and len(row) == 3 else ""
        else:
            continue
        out.append((str(mid or "").strip(), str(rtype or "").strip(), str(lbl or "").strip()))
    return out


def backfill_releases(
    *,
    run_sqlite_refresh: bool = True,
) -> dict:
    """
    Backfills releases from Snowflake into the local SQLite database.

    Returns a summary dict:
        {"inserted": int, "skipped": int, "errors": [...]}

    Incremental mode (default when EXPECTED_RELEASES already has rows):
      Only queries Snowflake for releases whose release_date falls within the last
      TIDE_BACKFILL_LOOKBACK_DAYS (default 90 days). Older rows are already in SQLite.
      Set TIDE_BACKFILL_FULL=1 to force a complete scan. Candidates include albums
      and singles (see query_release_backfill.sql); PRODUCT_TYPE is stored on insert.

    run_sqlite_refresh:
      When True (default) and new releases were inserted, update_sqlite_main() is
      called to pull their metrics into SQLite.  Pass False from the daily script
      (sqlite_handler already ran immediately before) to skip the second refresh.
    """
    import os as _os

    # Backfill only mutates marketshare_data.db; pull only that scope from S3.
    sync_db_from_s3()

    existing_mrelg_ids = {
        (row["MRELG_ID"] or "").strip()
        for row in _get_all_release_rows()
        if "MRELG_ID" in row.keys() and row["MRELG_ID"]
    }

    full_refresh = _os.environ.get("TIDE_BACKFILL_FULL", "").strip().lower() in (
        "1", "true", "yes"
    )
    lookback_days = int(_os.environ.get("TIDE_BACKFILL_LOOKBACK_DAYS", "90"))

    if existing_mrelg_ids and not full_refresh:
        date_filter = (
            f"AND mrelg.release_date >= DATEADD(day, -{int(lookback_days)}, CURRENT_DATE())"
        )
        logger.info(
            "backfill_releases: incremental — querying releases released in last %d days "
            "(TIDE_BACKFILL_FULL=1 for full scan)",
            lookback_days,
        )
    else:
        date_filter = ""
        reason = "TIDE_BACKFILL_FULL=1" if full_refresh else "EXPECTED_RELEASES is empty"
        logger.info("backfill_releases: full Snowflake scan (%s)", reason)

    query = load_sql(RELEASE_BACKFILL_QUERY).replace("{RELEASE_DATE_FILTER}", date_filter)

    logger.info("backfill_releases: running Snowflake query_release_backfill...")
    # Single Snowflake connection reused for the candidate list AND per-release metadata.
    with get_snowflake_connection() as sf:
        df = sf.query(query)
        logger.info("backfill_releases: Snowflake returned %d candidate rows", len(df))

        if df.empty:
            return {"inserted": 0, "skipped": 0, "errors": []}

        pending = [
            (mid, rtype, lbl)
            for mid, rtype, lbl in _backfill_candidates_from_df(df)
            if mid and mid not in existing_mrelg_ids
        ]
        skipped = len(df) - len(pending)
        logger.info(
            "backfill_releases: %d new / %d already in SQLite",
            len(pending),
            skipped,
        )

        inserted = 0
        errors: list[dict] = []

        for mrelg_id, release_type, label_name in pending:
            try:
                logger.info(
                    "backfill_releases: inserting mrelg_id=%s product_type=%s label=%s",
                    mrelg_id,
                    release_type,
                    label_name,
                )
                _create_backfilled_release(
                    mrelg_id=mrelg_id,
                    label_name=label_name,
                    product_type=release_type or None,
                    _sf=sf,
                )
                inserted += 1
                existing_mrelg_ids.add(mrelg_id)
            except Exception as e:
                errors.append({"mrelg_id": mrelg_id, "error": str(e)})

    # Always refresh per-release weekly metrics from Snowflake, even when 0
    # new releases were inserted. The previous `inserted > 0` gate caused a
    # silent staleness bug: when refresh_weekly's daily/weekly cadence found
    # no new mrelg_ids, MARKETSHARE_RELEASE_METRICS / MARKETSHARE_WEEKLY /
    # MARKETSHARE_YTD all stayed pinned at whatever date they were when
    # someone last triggered an insert. The /v1/releases/{id}/weekly endpoint
    # then returned multi-week-stale "AE YTD" sums (e.g. OCTANE showing 837K
    # instead of 894K because the 2026-04-23 row was missing).
    #
    # update_sqlite_main is idempotent and incremental — its own internal
    # max-date watermark keeps the Snowflake roundtrip narrow even when no
    # new releases were inserted, so the cost of always running it is small
    # (~30s typical) compared to the staleness it prevents.
    if run_sqlite_refresh:
        logger.info(
            "backfill_releases: running update_sqlite_main() (inserted=%d, refresh per-release metrics)",
            inserted,
        )
        try:
            from sqlite_handler import update_sqlite_main
            update_sqlite_main()
        except Exception as e:
            errors.append(
                {
                    "stage": "update_sqlite_main",
                    "error": str(e)
                    + "\nWARNING: weekly metric data may be stale.",
                }
            )
    else:
        logger.info(
            "backfill_releases: skipping update_sqlite_main (run_sqlite_refresh=False)"
        )

    # Push DB to S3 whenever we mutated SQLite (either by insert OR by
    # update_sqlite_main refreshing the per-release metrics tables).
    if inserted > 0 or run_sqlite_refresh:
        sync_db_to_s3()

    return {"inserted": inserted, "skipped": skipped, "errors": errors}


def streaming_forecast_route(product_type: str | None) -> str:
    """
    Which worldwide-streaming API bundle to use for a roster PRODUCT_TYPE.

    - ``singles`` -> global_streaming_singles_by_mrelg (Single only)
    - ``album``   -> global_streaming_by_mrelg (Album, EP, and anything else)
    """
    pt = str(product_type or "").strip().lower()
    if pt in ("single", "singles"):
        return "singles"
    return "album"


def _max_streaming_roster_release_date() -> str | None:
    """Latest RELEASE_DATE in STREAMING_ROSTER_2026, or None if empty/null."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        ensure_streaming_roster_2026_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT MAX(RELEASE_DATE) FROM STREAMING_ROSTER_2026")
        row = cur.fetchone()
        val = (row[0] if row else None) or None
        if val:
            return str(val).split(" ")[0][:10]
        return None


def _streaming_roster_mrelg_ids() -> set[str]:
    with sqlite3.connect(DATABASE_NAME) as conn:
        ensure_streaming_roster_2026_table(conn)
        cur = conn.cursor()
        cur.execute("SELECT MRELG_ID FROM STREAMING_ROSTER_2026")
        return {
            (row[0] or "").strip()
            for row in cur.fetchall()
            if row[0] and str(row[0]).strip()
        }


def _streaming_roster_rows_from_df(df: pd.DataFrame) -> List[tuple]:
    """Snowflake roster rows -> tuples for INSERT_STREAMING_ROSTER_2026."""
    norm = {str(c).strip().lower(): c for c in df.columns}
    if "mrelg_id" not in norm:
        raise ValueError(
            f"streaming roster query must return mrelg_id; got {list(df.columns)}"
        )

    out: List[tuple] = []
    cols = [str(c).strip().lower() for c in df.columns]
    for row in df.itertuples(index=False, name=None):
        d = dict(zip(cols, row))
        mid = str(d.get("mrelg_id") or "").strip()
        if not mid:
            continue
        rtype = str(d.get("release_type") or d.get("product_type") or "").strip()
        title = str(d.get("title") or "").strip()
        artist = str(d.get("artist") or "").strip()
        label = str(d.get("label_name") or d.get("label_group") or "").strip()
        parent_group = str(d.get("parent_group") or "").strip() or None
        rd = d.get("release_date")
        if rd is None or (isinstance(rd, float) and pd.isna(rd)):
            release_date = ""
        elif hasattr(rd, "strftime"):
            release_date = rd.strftime("%Y-%m-%d")
        else:
            release_date = str(rd).strip()[:10]
        out.append((mid, rtype or None, title, artist, label, parent_group, release_date))
    return out


def backfill_streaming_roster() -> dict:
    """
    Populate STREAMING_ROSTER_2026 from Snowflake (YTD streaming revenue board).

    Returns {"inserted": int, "upserted": int, "skipped": int, "errors": [...]}.

    Full scan (first run or TIDE_STREAMING_ROSTER_FULL=1):
      release_date >= start of current calendar year.

    Incremental (default when the table already has rows):
      release_date >= (max RELEASE_DATE in roster) minus TIDE_STREAMING_ROSTER_OVERLAP_DAYS
      (default 7). Only queries Snowflake for releases newer than what we already have.

    Does not run update_sqlite_main() or touch EXPECTED_RELEASES / marketshare tables.
    """
    import os as _os

    sync_db_from_s3()

    existing = _streaming_roster_mrelg_ids()
    full_refresh = _os.environ.get("TIDE_STREAMING_ROSTER_FULL", "").strip().lower() in (
        "1", "true", "yes"
    )
    overlap_days = int(_os.environ.get("TIDE_STREAMING_ROSTER_OVERLAP_DAYS", "7"))

    if existing and not full_refresh:
        max_release_date = _max_streaming_roster_release_date()
        if max_release_date:
            date_filter = (
                f"AND mrelg.release_date >= DATEADD(day, -{overlap_days + 1}, '{max_release_date}'::DATE)"
            )
            logger.info(
                "backfill_streaming_roster: incremental — release_date >= %s minus %d+1 day overlap "
                "(TIDE_STREAMING_ROSTER_FULL=1 for full YTD scan)",
                max_release_date,
                overlap_days,
            )
        else:
            date_filter = (
                "AND mrelg.release_date >= DATE_TRUNC('year', CURRENT_DATE())"
            )
            logger.info(
                "backfill_streaming_roster: no RELEASE_DATE in roster, falling back to YTD scan"
            )
    else:
        date_filter = (
            "AND mrelg.release_date >= DATE_TRUNC('year', CURRENT_DATE())"
        )
        reason = (
            "TIDE_STREAMING_ROSTER_FULL=1"
            if full_refresh
            else "STREAMING_ROSTER_2026 is empty"
        )
        logger.info("backfill_streaming_roster: full YTD Snowflake scan (%s)", reason)

    query = load_sql(STREAMING_ROSTER_YTD_QUERY).replace(
        "{RELEASE_DATE_FILTER}", date_filter
    )
    logger.info("backfill_streaming_roster: running %s...", STREAMING_ROSTER_YTD_QUERY)

    with get_snowflake_connection() as sf:
        df = sf.query(query)
    logger.info("backfill_streaming_roster: Snowflake returned %d rows", len(df))

    if df.empty:
        return {"inserted": 0, "upserted": 0, "skipped": 0, "errors": []}

    rows = _streaming_roster_rows_from_df(df)
    new_rows = [r for r in rows if r[0] not in existing]
    skipped = len(rows) - len(new_rows)

    insert_sql = load_sql(INSERT_STREAMING_ROSTER_2026)
    batch_size = 500
    errors: list[dict] = []
    upserted = 0

    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_streaming_roster_2026_table(conn)
            cur = conn.cursor()
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                try:
                    cur.executemany(insert_sql, chunk)
                    upserted += len(chunk)
                except sqlite3.Error as e:
                    errors.append({"batch_start": i, "error": str(e)})
            conn.commit()
    except sqlite3.Error as e:
        errors.append({"stage": "sqlite", "error": str(e)})

    inserted = len(new_rows)
    logger.info(
        "backfill_streaming_roster: upserted=%d new_mrelg_ids=%d skipped_existing=%d",
        upserted,
        inserted,
        skipped,
    )

    if upserted > 0 and not errors:
        sync_db_to_s3()

    return {
        "inserted": inserted,
        "upserted": upserted,
        "skipped": skipped,
        "errors": errors,
    }


def get_streaming_roster_2026() -> List[dict]:
    """All rows in STREAMING_ROSTER_2026 for the streaming revenue board."""
    with sqlite3.connect(DATABASE_NAME) as conn:
        ensure_streaming_roster_2026_table(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(load_sql(STREAMING_ROSTER_2026_LIST_QUERY))
        out: List[dict] = []
        for row in cur.fetchall():
            d = dict(row)
            d["forecast_route"] = streaming_forecast_route(d.get("PRODUCT_TYPE"))
            out.append(d)
        return out


def get_streaming_roster_2026_json() -> str:
    return json.dumps(get_streaming_roster_2026(), default=str)


def get_ytd_fiscal_revenue_by_label(
    *,
    week_end_date: str | None = None,
    level_1_distributor: str | None = None,
    level_2_distributor: str | None = None,
    level_3_distributor: str | None = None,
) -> list[dict[str, Any]]:
    """Weekly proxy revenue by distributor label from ytd_fiscal_revenue_by_label.csv."""
    import ytd_fiscal_revenue_from_csv

    return ytd_fiscal_revenue_from_csv.query_ytd_fiscal_revenue(
        week_end_date=week_end_date,
        level_1_distributor=level_1_distributor,
        level_2_distributor=level_2_distributor,
        level_3_distributor=level_3_distributor,
    )


def get_ytd_fiscal_revenue_by_label_json(
    *,
    week_end_date: str | None = None,
    level_1_distributor: str | None = None,
    level_2_distributor: str | None = None,
    level_3_distributor: str | None = None,
) -> str:
    return json.dumps(
        get_ytd_fiscal_revenue_by_label(
            week_end_date=week_end_date,
            level_1_distributor=level_1_distributor,
            level_2_distributor=level_2_distributor,
            level_3_distributor=level_3_distributor,
        ),
        default=str,
    )


def update_release(
    *,
    id: int,
    mrelg_id: str,
    name: str,
    artist: str, 
    label_name: str, 
    release_date: str, 
    genre: str, 
    scenario: str, 
    fw_vol: float = 0.0, 
    fw_streams: float = 0.0,
    fw_songs: float = 0.0,
    fw_sales: float = 0.0,
    fy_vol: float = 0.0, 
    known_vols: list[float] | None = None,
    avg_historical_w1_product_ratio: float = 0.3, 
    product_ratio_coefficient: float = 0.3,
    cluster: int | None = None,
    **_kwargs,
) -> None:
    """Updates a release in the database."""

    _verify_id(id)
    if known_vols is None:
        known_vols = []
    _verify_release_fields(locals())

    params = (
        mrelg_id,
        name,
        artist,
        label_name,
        release_date,
        genre,
        fw_vol,
        fw_streams,
        fw_songs,
        fw_sales,
        scenario,
        fy_vol,
        avg_historical_w1_product_ratio,
        product_ratio_coefficient,
        cluster,
        id,
    )
    try:
        query = load_sql(RELEASE_UPDATE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
        marketshare_cache_clear()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error updating release: {e}") from e


def delete_release(id: int) -> None:
    """
    Deletes a release from the database.
    """
    _verify_id(id)
    try:
        query = load_sql(RELEASE_DELETE_QUERY)
        with sqlite3.connect(DATABASE_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            conn.commit()
        marketshare_cache_clear()
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error deleting release: {e}")


def get_release(id: int) -> dict:
    """
    Loads one release by primary key and returns a dict shaped for ForecastEngine.simulate().
    """
    _verify_id(id)
    query = load_sql(RELEASE_GET_QUERY)
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, (id,))
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"No release found with id = {id}.")
            return _sqlite_row_to_release_map(row)
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error getting release: {e}") from e


def get_all_releases() -> List[dict]:
    """
    Gets all releases from the database.
    Returns a list of dictionaries of the release data.
    """
    try:
        rows = [{k: row[k] for k in row.keys()} for row in _get_all_release_rows()]
        out: List[Dict[str, Any]] = []
        for r in rows:
            rid = r.get("RELEASE_ID")
            if rid is None:
                continue
            out.append(
                {
                    "id": int(rid),
                    "album": (r.get("TITLE") or "").strip(),
                    "artist": (r.get("ARTIST") or "").strip(),
                }
            )
        return out
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error getting all releases: {e}")


def get_all_releases_series_json() -> str:
    """
    Returns a JSON string of the compact releases series from `get_all_releases()`.

    Shape: [{"id": <int>, "album": <str>, "artist": <str>}, ...]
    """
    return json.dumps(get_all_releases())


# ---------------------------------------------------------------------------
# Search: artist+title -> ranked MRELG candidates
#
# Powered by the local MARKETSHARE_SEARCH_SUMMARY table (rebuilt every
# refresh_data call). Ranks rows with a weighted blend of fuzzy text match
# and log-scaled daily streams so popular releases bubble up *without*
# drowning out close text matches on smaller releases.
# ---------------------------------------------------------------------------

_SEARCH_TEXT_WEIGHT = 0.75
_SEARCH_STREAM_WEIGHT = 0.25
_SEARCH_DEFAULT_LIMIT = 20
_SEARCH_TEXT_FLOOR = 0.30  # rows with text similarity below this are dropped

# Two-stage retrieval tuning: SQL prefilter shrinks the candidate pool that
# the Python fuzzy scorer iterates over. Tokens shorter than MIN_TOKEN_LEN
# generate too many false positives in LIKE scans, so we drop them. The
# candidate pool is capped by CANDIDATE_LIMIT (taking the most-streamed
# matches first) so even worst-case queries stay below ~10k Python iterations.
_SEARCH_MIN_TOKEN_LEN = 2
_SEARCH_CANDIDATE_LIMIT = 8000
_SEARCH_FALLBACK_TOPK = 2000  # used when no usable tokens exist (e.g. all 1-char)


def _normalize_search_text(value: Any) -> str:
    """Backwards-compatible alias for the shared normalizer."""
    return normalize_search_text(value)


def _text_similarity(a: str, b: str) -> float:
    """
    Hybrid text similarity in [0, 1]:
    - SequenceMatcher.ratio for overall ordering / typo tolerance.
    - Token overlap (Jaccard) so partial matches like "untitled unmastered"
      vs "untitled unmastered." score high even when punctuation/order vary.
    """
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()

    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if a_tokens and b_tokens:
        jaccard = len(a_tokens & b_tokens) / len(a_tokens | b_tokens)
    else:
        jaccard = 0.0

    return max(seq, jaccard)


def _search_tokens(*texts: str) -> List[str]:
    """Tokenize already-normalized search inputs and drop noise."""
    tokens: List[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for tok in text.split():
            if len(tok) < _SEARCH_MIN_TOKEN_LEN:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            tokens.append(tok)
    return tokens


def _fetch_search_candidates(
    *,
    artist_norm: str,
    title_norm: str,
    span: Dict[str, Any],
    search_table: str = MARKETSHARE_SEARCH_SUMMARY_TABLE,
    endpoint: str = "search_global_streaming",
) -> List[sqlite3.Row]:
    """
    Two-stage retrieval: SQL prefilter on indexed normalized text columns to
    shrink the pool we feed to the Python fuzzy scorer. Falls back to the
    most-streamed releases when the query has no usable tokens (e.g. all
    1-char tokens) so we still return *something* and the result quality
    matches the legacy behaviour for those edge cases.

    Always selects ARTIST_SEARCH/TITLE_SEARCH so the caller can avoid
    re-normalizing every row at request time. Older databases that predate
    the columns are detected once and a graceful fallback path is used.
    """
    tokens = _search_tokens(artist_norm, title_norm)

    fetch_sql = (
        "SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, GENRE, "
        "DAILY_GLOBAL_STREAMS, ARTIST_SEARCH, TITLE_SEARCH "
        f"FROM {search_table}"
    )

    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            normalized_columns_present = _has_normalized_search_columns(
                cur, search_table
            )

            if not normalized_columns_present:
                # Legacy schema: fall back to the original full-table fetch so
                # the function still works after a deploy that hasn't run the
                # refresh job yet.
                with _perf_phase(
                    span,
                    "db_fetch",
                    endpoint=endpoint,
                    mode="full_scan_legacy",
                ) as info:
                    cur.execute(
                        f"SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, "
                        f"GENRE, DAILY_GLOBAL_STREAMS FROM {search_table}"
                    )
                    rows = cur.fetchall()
                    info["rows"] = len(rows)
                return rows

            if tokens:
                like_clauses: List[str] = []
                params: List[Any] = []
                for tok in tokens:
                    like_clauses.append("ARTIST_SEARCH LIKE ?")
                    params.append(f"%{tok}%")
                    like_clauses.append("TITLE_SEARCH LIKE ?")
                    params.append(f"%{tok}%")

                sql = (
                    f"{fetch_sql} WHERE ({' OR '.join(like_clauses)}) "
                    "ORDER BY DAILY_GLOBAL_STREAMS DESC LIMIT ?"
                )
                params.append(_SEARCH_CANDIDATE_LIMIT)
                with _perf_phase(
                    span,
                    "db_fetch",
                    endpoint=endpoint,
                    mode="prefilter",
                ) as info:
                    info["tokens"] = len(tokens)
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    info["rows"] = len(rows)
                return rows

            # No usable tokens: return top-K by streams so we don't fall back
            # to a full scan but the Python scorer still has *something* to
            # rank. This matches what the user typically wants (popular
            # releases) when the query is too short to filter on.
            with _perf_phase(
                span,
                "db_fetch",
                endpoint=endpoint,
                mode="topk_no_tokens",
            ) as info:
                cur.execute(
                    f"{fetch_sql} ORDER BY DAILY_GLOBAL_STREAMS DESC LIMIT ?",
                    (_SEARCH_FALLBACK_TOPK,),
                )
                rows = cur.fetchall()
                info["rows"] = len(rows)
            return rows
    except sqlite3.Error as e:
        raise sqlite3.Error(f"Error querying {search_table}: {e}") from e


_HAS_NORMALIZED_SEARCH_COLUMNS: Optional[bool] = None
_HAS_NORMALIZED_SEARCH_COLUMNS_BY_TABLE: Dict[str, bool] = {}


def _has_normalized_search_columns(
    cur: sqlite3.Cursor,
    table: str = MARKETSHARE_SEARCH_SUMMARY_TABLE,
) -> bool:
    """
    Cache whether the local SQLite schema has the persisted search columns
    AND whether they have been populated. The column check runs once per
    process; the value is only flipped to ``True`` after a refresh has
    written non-NULL values, so the migration window between an online
    schema-only ALTER and the next refresh still falls back to the legacy
    full-scan path (rather than running a LIKE prefilter against all-NULL
    columns and returning empty results).
    """
    global _HAS_NORMALIZED_SEARCH_COLUMNS
    cached = _HAS_NORMALIZED_SEARCH_COLUMNS_BY_TABLE.get(table)
    if cached is True:
        return True
    if table == MARKETSHARE_SEARCH_SUMMARY_TABLE and _HAS_NORMALIZED_SEARCH_COLUMNS is True:
        return True
    cur.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if "ARTIST_SEARCH" not in cols or "TITLE_SEARCH" not in cols:
        _HAS_NORMALIZED_SEARCH_COLUMNS_BY_TABLE[table] = False
        if table == MARKETSHARE_SEARCH_SUMMARY_TABLE:
            _HAS_NORMALIZED_SEARCH_COLUMNS = False
        return False
    cur.execute(
        f"SELECT 1 FROM {table} "
        "WHERE ARTIST_SEARCH IS NOT NULL OR TITLE_SEARCH IS NOT NULL LIMIT 1"
    )
    populated = cur.fetchone() is not None
    _HAS_NORMALIZED_SEARCH_COLUMNS_BY_TABLE[table] = populated
    if populated and table == MARKETSHARE_SEARCH_SUMMARY_TABLE:
        _HAS_NORMALIZED_SEARCH_COLUMNS = True
    return populated


def search_releases_by_artist_title(
    artist: str,
    title: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
    *,
    search_table: str = MARKETSHARE_SEARCH_SUMMARY_TABLE,
    endpoint: str = "search_global_streaming",
) -> List[Dict[str, Any]]:
    """
    Search a MARKETSHARE_SEARCH_SUMMARY* table for the best-matching
    Luminate release groups given a free-text artist and title.

    Ranking: combined_score = TEXT_WEIGHT * text_score + STREAM_WEIGHT *
    popularity_score, where text_score is a fuzzy match on artist+title
    (50/50 average) and popularity_score is log1p(daily_global_streams)
    normalized to [0, 1] across the candidate pool. The streaming weight is
    intentionally bounded so popular releases bubble up only when text
    relevance is comparable; smaller releases with stronger text matches
    still surface near the top.

    Returns up to `limit` results sorted by combined_score (desc), each with
    metadata + component scores so the front end can debug / display reasons.
    """
    if not isinstance(artist, str):
        artist = "" if artist is None else str(artist)
    if not isinstance(title, str):
        title = "" if title is None else str(title)
    artist = artist.strip()
    title = title.strip()
    if not artist and not title:
        raise ValueError("At least one of `artist` or `title` is required.")
    if limit is None:
        limit = _SEARCH_DEFAULT_LIMIT
    try:
        limit = int(limit)
    except (TypeError, ValueError) as e:
        raise ValueError("limit must be a positive integer.") from e
    if limit < 1:
        raise ValueError("limit must be a positive integer.")
    limit = min(limit, 100)

    span: Dict[str, Any] = {}
    t_start = _now()

    artist_norm = _normalize_search_text(artist)
    title_norm = _normalize_search_text(title)

    rows = _fetch_search_candidates(
        artist_norm=artist_norm,
        title_norm=title_norm,
        span=span,
        search_table=search_table,
        endpoint=endpoint,
    )

    if not rows:
        logger.info("search: candidate set is empty; returning no matches")
        _perf_summary(
            span,
            endpoint=endpoint,
            t_start=t_start,
            results=0,
            candidates=0,
        )
        return []

    candidates: List[Dict[str, Any]] = []
    with _perf_phase(span, "score", endpoint=endpoint) as info:
        info["pool"] = len(rows)
        for row in rows:
            # Refresh writes pre-normalized columns; legacy rows are normalized
            # on the fly so a stale DB still works.
            row_artist_norm = (
                row["ARTIST_SEARCH"] if "ARTIST_SEARCH" in row.keys() and row["ARTIST_SEARCH"]
                else _normalize_search_text(row["ARTIST"])
            )
            row_title_norm = (
                row["TITLE_SEARCH"] if "TITLE_SEARCH" in row.keys() and row["TITLE_SEARCH"]
                else _normalize_search_text(row["TITLE"])
            )

            if artist_norm and title_norm:
                artist_score = _text_similarity(artist_norm, row_artist_norm)
                title_score = _text_similarity(title_norm, row_title_norm)
                text_score = 0.5 * artist_score + 0.5 * title_score
            elif artist_norm:
                artist_score = _text_similarity(artist_norm, row_artist_norm)
                title_score = 0.0
                text_score = artist_score
            else:
                artist_score = 0.0
                title_score = _text_similarity(title_norm, row_title_norm)
                text_score = title_score

            if text_score < _SEARCH_TEXT_FLOOR:
                continue

            try:
                daily_streams = float(row["DAILY_GLOBAL_STREAMS"] or 0)
            except (TypeError, ValueError):
                daily_streams = 0.0

            candidates.append(
                {
                    "mrelg_id": row["MRELG_ID"],
                    "title": row["TITLE"],
                    "artist": row["ARTIST"],
                    "label_name": row["LABEL_NAME"],
                    "release_date": row["RELEASE_DATE"],
                    "genre": row["GENRE"],
                    "daily_global_streams": int(daily_streams),
                    "artist_score": round(float(artist_score), 4),
                    "title_score": round(float(title_score), 4),
                    "text_score": round(float(text_score), 4),
                    "_raw_streams": daily_streams,
                }
            )
        info["matched"] = len(candidates)

    if not candidates:
        _perf_summary(
            span,
            endpoint=endpoint,
            t_start=t_start,
            results=0,
            candidates=len(rows),
        )
        return []

    with _perf_phase(span, "rank", endpoint=endpoint) as info:
        max_log_streams = max(math.log1p(c["_raw_streams"]) for c in candidates)
        if max_log_streams <= 0:
            max_log_streams = 1.0  # avoid divide-by-zero when nothing has streams

        for c in candidates:
            stream_score = math.log1p(c["_raw_streams"]) / max_log_streams
            c["stream_score"] = round(float(stream_score), 4)
            c["combined_score"] = round(
                float(_SEARCH_TEXT_WEIGHT * c["text_score"] + _SEARCH_STREAM_WEIGHT * stream_score),
                4,
            )
            c.pop("_raw_streams", None)

        candidates.sort(
            key=lambda x: (x["combined_score"], x["text_score"], x["daily_global_streams"]),
            reverse=True,
        )
        info["matched"] = len(candidates)

    results = candidates[:limit]
    for row in results:
        mid = row.get("mrelg_id")
        row["catalog_revenue_2025"] = catalog_revenue_2025_for_mrelg(
            str(mid) if mid is not None else None
        )

    _perf_summary(
        span,
        endpoint=endpoint,
        t_start=t_start,
        results=len(results),
        candidates=len(rows),
    )
    return results


def search_releases_by_artist_title_singles(
    artist: str,
    title: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
) -> List[Dict[str, Any]]:
    """Search MARKETSHARE_SEARCH_SUMMARY_SINGLES for singles release groups."""
    return search_releases_by_artist_title(
        artist,
        title,
        limit=limit,
        search_table=MARKETSHARE_SEARCH_SUMMARY_SINGLES_TABLE,
        endpoint="search_global_streaming_singles",
    )


def search_releases_by_artist_title_json(
    artist: str,
    title: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
) -> str:
    """JSON-serialized form of search_releases_by_artist_title for the API layer."""
    return json.dumps(search_releases_by_artist_title(artist, title, limit=limit))


def search_releases_by_artist_title_singles_json(
    artist: str,
    title: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
) -> str:
    """JSON-serialized singles search for the API layer."""
    return json.dumps(
        search_releases_by_artist_title_singles(artist, title, limit=limit)
    )


def get_known_vols(
    release_id: int,
) -> List[float]:
    """
    Pull known weekly actuals (ALBUM_EQUIVALENT) for a release from SQLite, ordered by week end.
    """
    _, vols = get_known_vols_with_dates(release_id)
    return vols


def get_known_vols_with_dates(
    release_id: int,
) -> tuple[List[str], List[float]]:
    """
    Same as get_known_vols, but also returns the parallel WEEK_ENDING_DATE list
    so callers can identify in-progress (partial) weeks. Both lists are aligned
    by index and sorted ascending by week end.
    """
    _verify_id(release_id)

    # TODO: Replace with query_known_vols_sqlite.sql, not a priority.
    sql = (
        f"SELECT WEEK_ENDING_DATE, ALBUM_EQUIVALENT "
        f"FROM {MARKETSHARE_RELEASE_METRICS_TABLE} "
        f"WHERE RELEASE_ID = ? "
        f"ORDER BY date(WEEK_ENDING_DATE) ASC;"
    )

    with sqlite3.connect(DATABASE_NAME) as conn:
        df = pd.read_sql_query(sql, conn, params=(int(release_id),))
    if df.empty:
        return [], []
    vals = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    vals = vals.replace([float("inf"), float("-inf")], pd.NA)
    df = df.assign(_v=vals).dropna(subset=["_v"])
    dates = [str(x) for x in df["WEEK_ENDING_DATE"].to_list()]
    vols = [float(x) for x in df["_v"].to_list()]
    return dates, vols


def get_known_component_vols(
    release_id: int,
) -> Dict[str, List[float]]:
    """
    Pull per-component weekly actuals for a release from SQLite, ordered by week end.
    Returns {"streams": [...], "sales": [...], "songs": [...]}.
    """
    _verify_id(release_id)

    sql = (
        f"SELECT WEEK_ENDING_DATE, "
        f"       STREAMING_EQUIVALENT, PRODUCT_SALES, SONG_SALE_EQUIVALENT "
        f"FROM {MARKETSHARE_RELEASE_METRICS_TABLE} "
        f"WHERE RELEASE_ID = ? "
        f"ORDER BY date(WEEK_ENDING_DATE) ASC;"
    )

    with sqlite3.connect(DATABASE_NAME) as conn:
        df = pd.read_sql_query(sql, conn, params=(int(release_id),))

    result: Dict[str, List[float]] = {"streams": [], "sales": [], "songs": []}
    if df.empty:
        return result

    for col, key in [
        ("STREAMING_EQUIVALENT", "streams"),
        ("PRODUCT_SALES", "sales"),
        ("SONG_SALE_EQUIVALENT", "songs"),
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            vals = vals.replace([float("inf"), float("-inf")], 0.0)
            result[key] = [float(x) for x in vals.to_list()]

    return result


def get_marketshare_actuals() -> pd.DataFrame:
    """
    Year-to-date observed marketshare timeline derived directly from
    Current_Data.csv (the same file the LGBM/Prophet trainer ingests). The
    YEAR column in Current_Data is Luminate's authoritative chart-year, so
    boundary weeks straddling Jan 1 are bucketed correctly without any
    ``startsWith('YYYY')`` heuristics.

    Returns rows tagged Data_Type='Actual' in the unified_ytd schema. Reads
    are O(1) after the first hit (in-memory cache invalidated by
    reload_artifacts() at the end of every refresh_weekly).
    """
    return marketshare_from_csv.ytd_actuals_for_year()


def _weekly_marketshare_observed() -> pd.DataFrame:
    """
    Per-week observed marketshare for the current Luminate chart-year, sourced
    directly from Current_Data.csv. Used to overlay engine forecasts so the
    "actual" portion of the YTD chart matches the same CSV the trainer ingests
    (no SQLite-vs-CSV cadence drift). Shares are percent points.
    """
    return marketshare_from_csv.weekly_actuals_for_year()


def _apply_weekly_actuals_to_unified_ytd(unified_ytd: pd.DataFrame) -> pd.DataFrame:
    """
    Replace weekly Active_Share / Total_Market for weeks present in
    Current_Data.csv so /v1/marketshare/weekly stitches actuals (past) +
    engine forecast (future). The forecast doesn't "restart from week 1"
    because the weeks before the first forecast week are explicit actuals
    drawn from the same CSV the trainer used.

    Recomputes cumulative YTD columns the same way as run_archetype_scenario.
    """
    if unified_ytd.empty:
        return unified_ytd
    obs = _weekly_marketshare_observed()
    if obs.empty:
        return unified_ytd

    df = unified_ytd.copy()
    df["_wk"] = pd.to_datetime(df["Week Ending Date"], errors="coerce")
    obs = obs.copy()
    obs["_wk"] = pd.to_datetime(obs["Week Ending Date"], errors="coerce")
    patch = obs[["Owner", "_wk", "Active_Share", "Total_Market_AE_Volume"]].rename(
        columns={
            "Active_Share": "_sqlite_share",
            "Total_Market_AE_Volume": "_sqlite_tot",
        }
    )
    merged = df.merge(patch, on=["Owner", "_wk"], how="left")
    hit = merged["_sqlite_share"].notna() & merged["_sqlite_tot"].notna()
    merged.loc[hit, "Active_Share"] = merged.loc[hit, "_sqlite_share"]
    merged.loc[hit, "Total_Market_AE_Volume"] = merged.loc[hit, "_sqlite_tot"]
    merged.loc[hit, "Data_Type"] = "Actual"
    merged = merged.drop(columns=["_sqlite_share", "_sqlite_tot"], errors="ignore")

    merged = merged.sort_values(by=["Owner", "_wk"]).reset_index(drop=True)
    merged["Active_Share"] = pd.to_numeric(merged["Active_Share"], errors="coerce").fillna(0.0)
    merged["Total_Market_AE_Volume"] = pd.to_numeric(
        merged["Total_Market_AE_Volume"], errors="coerce"
    ).fillna(0.0)
    merged["Weighted_Numerator"] = merged["Active_Share"] * merged["Total_Market_AE_Volume"]
    merged["Cum_Numerator"] = merged.groupby("Owner", sort=False)["Weighted_Numerator"].cumsum()
    merged["Cum_Denominator"] = merged.groupby("Owner", sort=False)["Total_Market_AE_Volume"].cumsum()
    den = merged["Cum_Denominator"].replace(0, float("nan"))
    merged["Unified_YTD_Share"] = (merged["Cum_Numerator"] / den).round(4)

    eng = get_engine()
    es = float(getattr(eng, "e_score_default", 0.82) or 0.82)
    fc_mask = merged["Data_Type"].astype(str) == "Forecast"
    merged.loc[~fc_mask, "YTD_Share_Upper"] = merged.loc[~fc_mask, "Unified_YTD_Share"]
    merged.loc[~fc_mask, "YTD_Share_Lower"] = merged.loc[~fc_mask, "Unified_YTD_Share"]
    merged.loc[fc_mask, "YTD_Share_Upper"] = merged.loc[fc_mask, "Unified_YTD_Share"] + es
    merged.loc[fc_mask, "YTD_Share_Lower"] = merged.loc[fc_mask, "Unified_YTD_Share"] - es

    wk_str = merged["_wk"].dt.strftime("%Y-%m-%d")
    merged["Week Ending Date"] = wk_str.where(
        merged["_wk"].notna(), merged["Week Ending Date"].astype(str)
    )
    merged = merged.drop(columns=["_wk"], errors="ignore")
    return merged


def get_marketshare_forecasts(week_ending_date: str | None = None) -> pd.DataFrame:
    """
    Takes in a week ending date and returns the marketshare forecasts for that week.
    The week ending date must be in the format YYYY-MM-DD if provided.
    If no week ending date is provided, all marketshare forecasts are returned.
    """
    global _MARKETSHARE_YTD_CACHE

    # Verify the week ending date (if provided).
    if week_ending_date is not None:
        _validate_date(week_ending_date)

    with _MARKETSHARE_YTD_CACHE_LOCK:
        cached = None if _MARKETSHARE_YTD_CACHE is None else _MARKETSHARE_YTD_CACHE.copy()
    if cached is not None:
        if week_ending_date is None:
            return cached
        mask = cached["Week Ending Date"].astype(str) == week_ending_date.strip()
        return cached.loc[mask].copy()

    # Get all releases and verify the parquet file.
    releases = [_sqlite_row_to_release_map(row) for row in _get_all_release_rows()]
    df_full = ARTIFACTS_DIR / "df_full.parquet"
    _verify_parquet_file(df_full)

    # Simulate the releases and return the marketshare forecasts for the week ending date.
    forecasts = get_engine().simulate(releases)
    unified_ytd = pd.DataFrame(forecasts["unified_ytd"])
    if unified_ytd.empty:
        return unified_ytd
    unified_ytd = _apply_weekly_actuals_to_unified_ytd(unified_ytd)
    with _MARKETSHARE_YTD_CACHE_LOCK:
        _MARKETSHARE_YTD_CACHE = unified_ytd.copy()
    if week_ending_date is None:
        return unified_ytd.copy()
    else:
        mask = unified_ytd["Week Ending Date"].astype(str) == week_ending_date.strip()
        return unified_ytd.loc[mask].copy()


def get_release_forecasts(id: int, week_ending_date: str | None = None) -> pd.DataFrame:
    """
    Weekly marketshare injections for one EXPECTED_RELEASES row.

    Decay routing: when the release map has ``product_type`` / ``release_type``
    set to ``single`` (or ``singles``), ForecastEngine uses
    ``archetypes_artifacts/singles/{streams,songs}``; otherwise album bundles.
    Sales channel stays zero for singles. Until singles are backfilled into
    EXPECTED_RELEASES, rows without those fields use album decay.
    """
    # Verify the week ending date (if provided) and ID.
    _verify_id(id)
    if week_ending_date is not None:
        _validate_date(week_ending_date)

    # Get the release and verify the parquet file.
    release = get_release(id)
    df_full = ARTIFACTS_DIR / "df_full.parquet"
    _verify_parquet_file(df_full)

    # Simulate the release and return the forecasts for the week ending date.
    forecasts = get_engine().simulate([release])
    weekly_injections = pd.DataFrame(forecasts["weekly_injections"])
    if weekly_injections.empty:
        return weekly_injections
    if week_ending_date is None:
        return weekly_injections
    else:
        mask = weekly_injections["Week Ending Date"].astype(str) == week_ending_date
        return weekly_injections.loc[mask]


def train_model() -> None:
    """
    Trains the model.

    Also rebuilds the MARKETSHARE_SEARCH_SUMMARY SQLite table so the search
    endpoint always reflects the latest daily Snowflake snapshot. The table
    is fully overwritten (delete + insert) because daily-stream snapshots
    are not additive across runs.
    """
    # Full refresh trains every scope, so pull all canonical inputs from S3
    # (csv + parquets + artifacts_75k + archetypes + db) before training.
    sync_full_inputs_from_s3()
    refresh_data()
    reload_artifacts()
    try:
        refresh_marketshare_search_summary()
    except Exception:
        logger.exception(
            "train_model: search summary refresh failed; search results may be stale"
        )
    try:
        refresh_marketshare_search_summary_singles()
    except Exception:
        logger.exception(
            "train_model: singles search summary refresh failed; search results may be stale"
        )
    forecast_cache_clear()
    # Persist refreshed CSV + parquets + artifacts + db for future incremental runs.
    sync_full_outputs_to_s3()


def refresh_model() -> None:
    """
    Refreshes the model parquets from Snowflake and pushes them to S3.
    """
    with get_snowflake_connection() as sf:
        update_parquet_metrics(sf)
    reload_artifacts()
    # Parquets are the only thing this path writes; push just that scope.
    sync_parquets_to_s3()


_PARQUET_DIR = Path(__file__).resolve().parent / "model" / "data"
# Some environments still carry the legacy worldwide parquet name
# `streams_worldwide_compressed.parquet`. Treat either as satisfying the
# requirement so refresh_weekly does not trigger an unnecessary full rebuild.
_REQUIRED_PARQUET_ALTERNATIVES = (
    (_PARQUET_DIR / "streams_product_songs_ae_compressed.parquet",),
    (
        _PARQUET_DIR / "worldwide_streams_compressed.parquet",
        _PARQUET_DIR / "streams_worldwide_compressed.parquet",
    ),
)


def _resolve_missing_required_parquets() -> list[str]:
    missing: list[str] = []
    for alternatives in _REQUIRED_PARQUET_ALTERNATIVES:
        if not any(p.is_file() for p in alternatives):
            missing.append(str(alternatives[0]))
    return missing


def _resolved_required_parquet_names() -> list[str]:
    out: list[str] = []
    for alternatives in _REQUIRED_PARQUET_ALTERNATIVES:
        for p in alternatives:
            if p.is_file():
                out.append(p.name)
                break
    return out


def refresh_weekly(force_refresh_parquets: bool = False) -> dict:
    """
    Single weekly orchestration designed around the canonical S3 pattern:
      - Pull only weekly inputs (db + csvs + artifacts_75k) to local disk.
      - Train CSV-only (skips heavy parquet KMeans/archetype decay).
      - Push weekly outputs (db + csvs + artifacts_75k) back to S3.

    Heavy steps (parquet rebuild, full backfill) are intentionally separated
    so the weekly path stays fast, low-memory, and reliable. Use the dedicated
    endpoints when those need to run:
      - /v1/data/refresh_model  — rebuild AE + worldwide parquets
      - /v1/data/refresh_data   — full retrain (parquets + archetypes + csvs)
      - /v1/releases/backfill   — full release backfill

    Stage order:
      1. (optional) refresh_parquets — only when force_refresh_parquets=True.
         Pulls the parquets scope from S3 first so the rebuild starts from
         the canonical state; pushes them back when done.
      2. refresh_data(csv_only=True) — pulls 3 CSVs incrementally from
         Snowflake and retrains LGBM / Prophet / spike / df_full from CSVs
         only. Skips AE parquet KMeans/DNA and all archetype decay.
      3. backfill_releases — inserts new mrelg_ids and refreshes per-release
         historical metrics. Skipped when TIDE_WEEKLY_SKIP_BACKFILL=1.

    Returns a per-stage summary. Stage 1 failures are non-fatal (archetype
    training falls back to whatever parquets are already on disk); stages 2
    and 3 re-raise so the job is marked failed.

    Progress: each stage calls api.jobs.set_step() so the caller can
    diagnose which phase is slow by polling GET /v1/jobs/{id}.steps. The
    import is local to avoid a hard dependency on the api package when this
    module is imported outside FastAPI (e.g. by a script). set_step() is a
    no-op when not running under a JobManager.
    """
    from api.jobs import set_step

    summary: Dict[str, Any] = {"stages": {}}

    skip_backfill = os.environ.get("TIDE_WEEKLY_SKIP_BACKFILL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    skip_streaming_roster = os.environ.get(
        "TIDE_WEEKLY_SKIP_STREAMING_ROSTER_BACKFILL", "1"
    ).strip().lower() in ("1", "true", "yes", "on")

    # Pull only the weekly inputs (db + csvs + artifacts_75k). Skip the heavy
    # parquets/archetypes scopes — those are not needed for CSV-only training.
    set_step("sync_from_s3:start")
    sync_weekly_inputs_from_s3()
    set_step("sync_from_s3:done")

    # Stage 1: parquet rebuild (opt-in only). Pull parquet scope first so the
    # rebuild has the latest canonical state, then push the new ones back.
    if force_refresh_parquets:
        set_step("refresh_parquets:start")
        t0 = _now()
        logger.info("refresh_weekly: force_refresh_parquets=True; rebuilding parquets")
        try:
            sync_artifacts_from_s3_if_configured(scopes={"parquets"})
            with get_snowflake_connection() as sf:
                update_parquet_metrics(sf)
            sync_parquets_to_s3()
            summary["stages"]["refresh_parquets"] = {
                "ok": True, "skipped": False, "elapsed_sec": _elapsed(t0),
            }
        except Exception as e:
            logger.exception("refresh_weekly: refresh_parquets failed")
            summary["stages"]["refresh_parquets"] = {
                "ok": False,
                "skipped": False,
                "error": str(e),
                "elapsed_sec": _elapsed(t0),
            }
    else:
        logger.info(
            "refresh_weekly: skipping parquet refresh (CSV-only weekly path). "
            "Call /refresh_model or pass force_refresh_parquets=True to rebuild."
        )
        summary["stages"]["refresh_parquets"] = {
            "ok": True,
            "skipped": True,
            "reason": "weekly path is CSV-only by design",
        }

    # Stage 2: CSV-only training.
    set_step("refresh_data:start")
    t0 = _now()
    try:
        refresh_data(csv_only=True)
        summary["stages"]["refresh_data"] = {"ok": True, "elapsed_sec": _elapsed(t0)}
    except Exception as e:
        logger.exception("refresh_weekly: refresh_data failed")
        summary["stages"]["refresh_data"] = {
            "ok": False, "error": str(e), "elapsed_sec": _elapsed(t0),
        }
        set_step("reload_artifacts")
        reload_artifacts()
        raise

    # Stage 3: backfill (skippable for fast weekly runs).
    if skip_backfill:
        logger.info(
            "refresh_weekly: skipping backfill_releases (TIDE_WEEKLY_SKIP_BACKFILL=1). "
            "Run POST /v1/releases/backfill separately when needed."
        )
        summary["stages"]["backfill_releases"] = {
            "ok": True,
            "skipped": True,
            "reason": "TIDE_WEEKLY_SKIP_BACKFILL=1",
        }
    else:
        set_step("backfill_releases:start")
        t0 = _now()
        try:
            backfill_result = backfill_releases()
            summary["stages"]["backfill_releases"] = {
                "ok": True, "elapsed_sec": _elapsed(t0), **backfill_result,
            }
        except Exception as e:
            logger.exception("refresh_weekly: backfill_releases failed")
            summary["stages"]["backfill_releases"] = {
                "ok": False, "error": str(e), "elapsed_sec": _elapsed(t0),
            }
            set_step("reload_artifacts")
            reload_artifacts()
            raise

    if skip_streaming_roster:
        logger.info(
            "refresh_weekly: skipping backfill_streaming_roster "
            "(TIDE_WEEKLY_SKIP_STREAMING_ROSTER_BACKFILL=1). "
            "Run POST /v1/revenue/backfill_streaming_roster for YTD or incremental roster."
        )
        summary["stages"]["backfill_streaming_roster"] = {
            "ok": True,
            "skipped": True,
            "reason": "TIDE_WEEKLY_SKIP_STREAMING_ROSTER_BACKFILL=1",
        }
    else:
        set_step("backfill_streaming_roster:start")
        t0 = _now()
        try:
            streaming_result = backfill_streaming_roster()
            summary["stages"]["backfill_streaming_roster"] = {
                "ok": True,
                "elapsed_sec": _elapsed(t0),
                **streaming_result,
            }
        except Exception as e:
            logger.exception("refresh_weekly: backfill_streaming_roster failed")
            summary["stages"]["backfill_streaming_roster"] = {
                "ok": False,
                "error": str(e),
                "elapsed_sec": _elapsed(t0),
            }

    set_step("reload_artifacts")
    reload_artifacts()
    # Push only what weekly mutates: db + csvs + artifacts_75k.
    set_step("sync_to_s3")
    sync_weekly_outputs_to_s3()
    set_step("done")
    return summary


def _elapsed(t0: float) -> float:
    return round(time.perf_counter() - t0, 2)


def reload_artifacts() -> None:
    """
    Clear cached engine/artifacts so the next forecast request loads the
    freshly written parquets/pkls from disk. Call after any data refresh.

    Note: this only invalidates in-process caches. If the API is scaled out
    to multiple workers (gunicorn -w N, multiple EC2 instances) each worker
    has its own globals and will need an external reload signal. That's a
    Phase 3 concern once artifacts move to S3.
    """
    global GLOBAL_FORECAST_ENGINE, GLOBAL_WORLDWIDE_ARTIFACTS, GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS
    GLOBAL_FORECAST_ENGINE = None
    GLOBAL_WORLDWIDE_ARTIFACTS = None
    GLOBAL_WORLDWIDE_SINGLES_ARTIFACTS = None
    forecast_cache_clear()
    marketshare_cache_clear()
    marketshare_from_csv.clear_cache()
    import ytd_fiscal_revenue_from_csv

    ytd_fiscal_revenue_from_csv.clear_cache()
    album_art.clear_cache()


def df_to_json(
    df: pd.DataFrame
) -> str:
    """
    Returns JSON string from a pandas DataFrame.
    """
    if df is None or df.empty:
        return "[]"

    out = df.copy()

    # Ensure date columns serialize as YYYY-MM-DD.
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")

    numeric_cols = list(out.select_dtypes(include=["number"]).columns)
    share_cols = [c for c in numeric_cols if "share" in str(c).lower() or "percent" in str(c).lower()]
    other_numeric_cols = [c for c in numeric_cols if c not in share_cols]

    # Round share columns to 2 decimal places and other numeric columns to the nearest integer.
    if share_cols:
        out[share_cols] = out[share_cols].round(2)

    if other_numeric_cols:
        rounded = out[other_numeric_cols].round(0)
        out[other_numeric_cols] = rounded.astype("Int64")

    return out.to_json(orient="records")


def _get_all_release_rows() -> List[sqlite3.Row]:
    query = load_sql(RELEASE_GET_ALL_QUERY)
    with sqlite3.connect(DATABASE_NAME) as conn:
        ensure_expected_releases_fw_columns(conn)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        return cur.fetchall()


def _sqlite_row_to_release_map(row: sqlite3.Row) -> dict:
    """Map EXPECTED_RELEASES columns to keys expected by run_archetype_scenario."""
    title = (row["TITLE"] or "").strip() or None
    artist = (row["ARTIST"] or "").strip() or None
    mrelg_id = (row["MRELG_ID"] or "").strip() if "MRELG_ID" in row.keys() else ""
    rid = row["RELEASE_ID"] if "RELEASE_ID" in row.keys() else None

    known_vols: List[float] = []
    known_week_dates: List[str] = []
    component_vols: Dict[str, List[float]] = {"streams": [], "sales": [], "songs": []}
    if mrelg_id and rid is not None:
        try:
            known_week_dates, known_vols = get_known_vols_with_dates(int(rid))
        except Exception:
            known_week_dates, known_vols = [], []
        try:
            component_vols = get_known_component_vols(int(rid))
        except Exception:
            component_vols = {"streams": [], "sales": [], "songs": []}

    rd = row["RELEASE_DATE"]
    date_str = rd if isinstance(rd, str) else str(rd)

    expected_fw_vol = float(row["EXPECTED_ALBUM_EQUIVALENT"] or 0)
    if mrelg_id:
        has_nonzero_known = bool(known_vols) and any(float(x) > 0 for x in known_vols)
        if (not has_nonzero_known) and expected_fw_vol <= 0:
            raise ValueError(
                f"Release {rid} has mrelg_id={mrelg_id!r} but no backfilled metrics yet "
                "Ensure known_vols is populated."
            )
        fw_vol = float(max(known_vols)) if has_nonzero_known else expected_fw_vol
    else:
        fw_vol = expected_fw_vol

    def _sql_float(col: str, default: float = 0.0) -> float:
        if col not in row.keys():
            return default
        v = row[col]
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    release_map: dict = {
        "mrelg_id": mrelg_id or None,
        "name": artist or title or "Unknown",
        "artist": artist or "",
        "title": title or "",
        "label": row["LABEL_NAME"],
        "date": date_str,
        "genre": row["GENRE"],
        "fw_vol": fw_vol,
        "fw_streams": _sql_float("FW_STREAMS"),
        "fw_songs": _sql_float("FW_SONGS"),
        "fw_sales": _sql_float("FW_SALES"),
        "scenario": row["SCENARIO"],
        "known_vols": known_vols,
        "known_week_dates": known_week_dates,
        "fy_vol": float(row["FY_VOL"] or 0),
        "avg_historical_w1_product_ratio": float(row["AVG_HISTORICAL_W1_PRODUCT_RATIO"] or 0),
        "product_ratio_coefficient": float(row["PRODUCT_RATIO_COEFFICIENT"] or 0),
    }

    _cl = row["CLUSTER"] if "CLUSTER" in row.keys() else None
    release_map["cluster"] = int(_cl) if _cl is not None else None

    if component_vols["streams"]:
        release_map["known_streams"] = component_vols["streams"]
    if component_vols["sales"]:
        release_map["known_sales"] = component_vols["sales"]
    if component_vols["songs"]:
        release_map["known_songs"] = component_vols["songs"]

    if "PRODUCT_TYPE" in row.keys() and row["PRODUCT_TYPE"]:
        pt = str(row["PRODUCT_TYPE"]).strip()
        release_map["product_type"] = pt
        release_map["release_type"] = pt
    elif "RELEASE_TYPE" in row.keys() and row["RELEASE_TYPE"]:
        release_map["release_type"] = str(row["RELEASE_TYPE"]).strip()

    return _sanitize_release_for_simulation(release_map)


def _is_real_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_integral(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _verify_release_fields(inputs: dict) -> None:
    """
    Validate release create/update parameters from a dict (e.g. locals()).
    Raises ValueError if validation fails.
    """
    data = {k: inputs[k] for k in _RELEASE_FIELD_KEYS if k in inputs}
    data.setdefault("known_vols", [])
    for _fw in ("fw_streams", "fw_songs", "fw_sales"):
        data.setdefault(_fw, 0.0)

    # Verify required fields are not empty.
    for field in _REQUIRED_NONEMPTY_STR:
        val = data.get(field)
        if val is None or not isinstance(val, str) or not val.strip():
            raise ValueError(f"{field} is required.")

    # Verify numeric fields are real numbers.
    for field in _REAL_NUMERIC_FIELDS:
        val = data[field]
        if not _is_real_number(val):
            raise ValueError(f"{field} must be a number.")
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"{field} must be a finite number.")

    # Optional manual archetype cluster 0..3; None / omitted => computed mixture.
    cluster = data.get("cluster", None)
    if cluster is not None:
        if not _is_integral(cluster):
            raise ValueError("cluster must be null (auto) or an integer 0-3.")
        ci = int(cluster)
        if ci < 0 or ci > 3:
            raise ValueError("cluster must be between 0 and 3 inclusive when set.")

    # Verify known volumes is a list of real numbers.
    known_vols = data.get("known_vols", [])
    if known_vols is None:
        known_vols = []
    if not isinstance(known_vols, (list, tuple)):
        raise ValueError("Known volumes must be a list.")
    for i, x in enumerate(known_vols):
        if not _is_real_number(x):
            raise ValueError(f"Value {x} at index {i} is not a number.")
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            raise ValueError(f"Value {x} at index {i} is not a finite number.")

    # Verify total first-week signal when known_vols is empty (combined AE and/or component AE).
    fw_vol = float(data["fw_vol"])
    comp_w1 = float(data["fw_streams"]) + float(data["fw_songs"]) + float(data["fw_sales"])
    if len(known_vols) == 0 and fw_vol <= 0 and comp_w1 <= 0:
        raise ValueError(
            "Expected weekly volume must be positive when known volumes is empty, "
            "unless first-week streams / song / product AE components sum to a positive total."
        )

    # Verify genre is in the distribution.
    genre = data["genre"]
    if genre not in DISTRIBUTIONS["Genre"]:
        raise ValueError(f"Genre {genre!r} not found in distribution: {DISTRIBUTIONS['Genre']}.")

    # Verify label is in the distribution.
    label_name = data["label_name"]
    if label_name not in DISTRIBUTIONS["Label"]:
        raise ValueError(f"Label {label_name!r} not found in distribution: {DISTRIBUTIONS['Label']}.")

    try:
        datetime.strptime(data["release_date"].strip(), "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"release_date {data['release_date']!r} is not a valid YYYY-MM-DD date."
        ) from e

    scenario = data["scenario"].strip()
    if scenario not in _ALLOWED_SCENARIOS:
        raise ValueError(
            f"Scenario {scenario!r} must be one of {sorted(_ALLOWED_SCENARIOS)}."
        )


def _verify_id(id: int | None) -> None:
    """Verifies that the id is a positive integer and is not None."""
    if id is None:
        raise ValueError("ID is required.")
    if not _is_integral(id) or int(id) < 0:
        raise ValueError("ID must be a positive integer.")


def _validate_date(date_value: str | None) -> str:
    """Validates that the date is a valid YYYY-MM-DD date and returns it as a string."""
    try:
        if date_value is None:
            raise ValueError(f"Date is required. Got {date_value!r}.")
        if pd.isna(date_value):
            raise ValueError(f"Date is required. Got {date_value!r}.")
    except TypeError:
        raise TypeError(f"Date {date_value!r} must be a string.")
    if isinstance(date_value, datetime):
        s = date_value.date().isoformat()
    elif isinstance(date_value, date):
        s = date_value.isoformat()
    else:
        s = str(date_value).strip()
        # Snowflake/SQLite often return "YYYY-MM-DD HH:MM:SS[.fff]"; global_streaming SQL only needs the day.
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            try:
                datetime.strptime(s[:10], "%Y-%m-%d")
                s = s[:10]
            except ValueError:
                pass
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Date {date_value!r} must be in the format YYYY-MM-DD.") from e
    return s


def _verify_parquet_file(parquet_file: Path) -> None:
    """Verifies that the parquet file exists."""
    if not parquet_file.is_file():
        raise FileNotFoundError(
            f"Forecast artifacts not found at {parquet_file}. "
            "Run `python train_model.py` (or `model/train_marketshare_artifacts.py`) to generate them."
        )
    return parquet_file


def _verify_mrelg_id(mrelg_id: str, _sf: Snowflake) -> pd.DataFrame:
    """Verifies that the mrelg_id is a valid mrelg_id."""
    query = load_sql(MRELG_METADATA_QUERY)
    mrelg_metadata = _sf.query(query.format(MRELG_ID=f"'{mrelg_id}'"))
    if mrelg_metadata.empty:
        raise ValueError(f"Invalid mrelg_id: {mrelg_id}")
    return mrelg_metadata


def get_global_streaming_forecast(id: int, scenario: str = "Base") -> pd.DataFrame:
    """
    Returns a DataFrame of worldwide streaming forecasts for a single release.

    Historical observed weeks fetched from Snowflake are passed as
    known_worldwide_streams to anchor the archetype decay curve via
    fit_backfill_forecast.  When no history exists yet (future release),
    fw_streams (or fw_vol) is used as the cold-start peak volume.

    Columns: release_id, mrelg_id, artist, title, week, week_ending_date,
             data_type, pred_worldwide_streams, cumulative_worldwide_streams

    data_type is "Actual" for observed weeks and "Forecast" for model-predicted weeks.

    ``scenario`` ("Base" / "Bear" / "Bull") is forwarded to
    ``_build_global_streaming_forecast`` for the post-fit shock; defaults to
    Base so legacy callers remain unaffected. Stored release.scenario is
    intentionally not used here — the caller passes scenario explicitly so a
    single release can be charted under multiple scenarios without mutating
    the SQLite row.

    Deprecated: prefer get_global_streaming_forecast_by_mrelg(mrelg_id), which
    is what the new front end uses after the search endpoint resolves an MRELG.
    Kept for backward compatibility while the UI is migrated.
    """
    _verify_id(id)
    release = get_release(id)
    mrelg_id = (release.get("mrelg_id") or "").strip()
    if not mrelg_id:
        raise ValueError(
            f"Release {id} has no mrelg_id; worldwide streaming forecast requires "
            "a Luminate release group ID."
        )
    release_date = _validate_date(release.get("date"))

    fw_peak = float(release.get("fw_streams") or 0.0) or float(release.get("fw_vol") or 0.0)

    df = _build_global_streaming_forecast(
        mrelg_id=mrelg_id,
        release_date=release_date,
        artist=release.get("artist") or release.get("name") or "",
        title=release.get("title") or release.get("name") or "",
        genre=release.get("genre"),
        fw_streams_peak=fw_peak,
        scenario=scenario,
    )
    df.insert(0, "release_id", id)
    return df


def _sqlite_fw_peak_for_mrelg(mrelg_id: str) -> float:
    """
    First-week stream / AE volume from EXPECTED_RELEASES for this MRELG.
    Used as ``fw_streams_peak`` when the MRELG-driven path would otherwise pass 0,
    so sparse or all-zero Snowflake weeks can still cold-start the decay fit.
    """
    key = (mrelg_id or "").strip()
    if not key:
        return 0.0
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_expected_releases_fw_columns(conn)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT FW_STREAMS, EXPECTED_ALBUM_EQUIVALENT
                FROM EXPECTED_RELEASES
                WHERE MRELG_ID IS NOT NULL AND UPPER(TRIM(MRELG_ID)) = UPPER(?)
                LIMIT 1
                """,
                (key,),
            )
            row = cur.fetchone()
            if row is None:
                return 0.0
            fs = float(row["FW_STREAMS"] or 0) if row["FW_STREAMS"] is not None else 0.0
            ae = (
                float(row["EXPECTED_ALBUM_EQUIVALENT"] or 0)
                if row["EXPECTED_ALBUM_EQUIVALENT"] is not None
                else 0.0
            )
            return float(max(fs, ae, 0.0))
    except sqlite3.Error as e:
        logger.warning("sqlite fw_peak lookup failed for mrelg_id=%s: %s", key, e)
        return 0.0
    except (TypeError, ValueError) as e:
        logger.warning("sqlite fw_peak parse failed for mrelg_id=%s: %s", key, e)
        return 0.0


def get_global_streaming_forecast_by_mrelg(
    mrelg_id: str,
    scenario: str = "Base",
) -> pd.DataFrame:
    """
    Album (release-group) worldwide streaming forecast for a Luminate MRELG ID.
    Uses archetypes_artifacts/worldwide_streams.
    """
    return _get_global_streaming_forecast_by_mrelg(
        mrelg_id,
        scenario=scenario,
        singles=False,
        endpoint="global_streaming_by_mrelg",
        cache_product=_WORLDWIDE_STREAMING_PRODUCT_ALBUM,
    )


def get_global_streaming_forecast_singles_by_mrelg(
    mrelg_id: str,
    scenario: str = "Base",
) -> pd.DataFrame:
    """
    Singles worldwide streaming forecast for a Luminate MRELG ID.
    Uses archetypes_artifacts/worldwide_streams_singles (same simulator as albums).
    """
    return _get_global_streaming_forecast_by_mrelg(
        mrelg_id,
        scenario=scenario,
        singles=True,
        endpoint="global_streaming_singles_by_mrelg",
        cache_product=_WORLDWIDE_STREAMING_PRODUCT_SINGLE,
    )


def _get_global_streaming_forecast_by_mrelg(
    mrelg_id: str,
    *,
    scenario: str,
    singles: bool,
    endpoint: str,
    cache_product: str,
) -> pd.DataFrame:
    """
    Shared MRELG global-streaming path for album and singles archetype bundles.

    Metadata is resolved from MARKETSHARE_SEARCH_SUMMARY when available and
    falls back to Snowflake. Forecast cache key includes product, mrelg_id,
    scenario, and calendar date.
    """
    if not isinstance(mrelg_id, str) or not mrelg_id.strip():
        raise ValueError("mrelg_id is required.")
    mrelg_id = mrelg_id.strip()
    scenario = _normalize_scenario_label(scenario)

    span: Dict[str, Any] = {}
    t_start = _now()

    cached = _forecast_cache_lookup(mrelg_id, scenario, product=cache_product)
    if cached is not None:
        _perf_summary(
            span,
            endpoint=endpoint,
            t_start=t_start,
            mrelg_id=mrelg_id,
            cache="hit",
            rows=len(cached),
        )
        return cached.copy()

    search_table = (
        MARKETSHARE_SEARCH_SUMMARY_SINGLES_TABLE
        if singles
        else MARKETSHARE_SEARCH_SUMMARY_TABLE
    )
    with _perf_phase(span, "metadata_lookup_local", endpoint=endpoint) as info:
        local_meta = _resolve_mrelg_metadata_local(mrelg_id, search_table=search_table)
        if local_meta is None:
            local_meta = _resolve_mrelg_metadata_from_streaming_roster(mrelg_id)
        info["hit"] = bool(local_meta)

    with get_snowflake_connection() as sf:
        if local_meta is not None:
            metadata = local_meta
        else:
            with _perf_phase(span, "metadata_lookup_snowflake", endpoint=endpoint):
                metadata = _resolve_mrelg_metadata_snowflake(mrelg_id, sf)

        release_date = _validate_date(metadata.get("release_date"))
        fw_peak = _sqlite_fw_peak_for_mrelg(mrelg_id)
        df = _build_global_streaming_forecast(
            mrelg_id=mrelg_id,
            release_date=release_date,
            artist=metadata.get("artist") or "",
            title=metadata.get("title") or "",
            genre=metadata.get("genre"),
            fw_streams_peak=fw_peak,
            span=span,
            sf=sf,
            scenario=scenario,
            singles=singles,
        )

    _forecast_cache_store(mrelg_id, df, scenario, product=cache_product)

    _perf_summary(
        span,
        endpoint=endpoint,
        t_start=t_start,
        mrelg_id=mrelg_id,
        cache="miss",
        rows=len(df),
    )
    return df



def get_daily_global_streams_by_mrelg(mrelg_id: str) -> pd.DataFrame:
    """
    Live Revenue board — daily worldwide streams since release for a single
    MRELG release group. Reads from the cached SQLite table
    MARKETSHARE_DAILY_GLOBAL_STREAMS; lazily refreshes from Snowflake when
    the cache is empty or older than DAILY_STREAMS_STALE_DAYS.

    Returns columns: report_date (str YYYY-MM-DD), global_streams (float).

    Intentionally isolated from the model: no callers in the simulator,
    forecast engine, or training pipeline depend on this — a regression
    here only affects the Live Revenue surface.
    """
    from sqlite_handler import get_daily_global_streams_for_mrelg

    if not isinstance(mrelg_id, str) or not mrelg_id.strip():
        raise ValueError("mrelg_id is required.")
    mrelg_id = mrelg_id.strip()

    # Need release_date to bound the Snowflake query when the cache is cold.
    meta = _resolve_mrelg_metadata_local(mrelg_id) or _resolve_mrelg_metadata_from_streaming_roster(mrelg_id)
    release_date: Optional[str] = None
    if meta:
        rd = meta.get("release_date")
        release_date = str(rd).split(" ")[0] if rd else None
    if not release_date:
        try:
            with get_snowflake_connection() as sf:
                snowflake_meta = _resolve_mrelg_metadata_snowflake(mrelg_id, sf)
                release_date = (snowflake_meta.get("release_date") or "").split(" ")[0] or None
        except Exception as e:
            logger.warning(
                "get_daily_global_streams_by_mrelg: metadata lookup failed for %s: %s",
                mrelg_id,
                e,
            )

    df = get_daily_global_streams_for_mrelg(mrelg_id, release_date=release_date)
    df = df.rename(columns=str.lower) if df is not None else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["report_date", "global_streams"])
    df["report_date"] = df["report_date"].astype(str)
    df["global_streams"] = pd.to_numeric(df["global_streams"], errors="coerce").fillna(0.0)
    return df[["report_date", "global_streams"]]


def get_catalog_revenue_2025_by_mrelg(mrelg_id: str) -> Dict[str, Any]:
    """
    Live Revenue board — 2025 catalog revenue for a single MRELG release
    group, sourced from the local MARKETSHARE_REVENUE_2025 table (loaded
    one-shot from s3://parquetgarage/model/data/2025_revenue_catalog.csv).
    Returns {"catalog_revenue_2025": float | None}; null when the MRELG
    isn't in the file (frontend treats null as "not in file").
    """
    from sqlite_handler import get_catalog_revenue_2025_for_mrelg

    if not isinstance(mrelg_id, str) or not mrelg_id.strip():
        raise ValueError("mrelg_id is required.")
    value = get_catalog_revenue_2025_for_mrelg(mrelg_id.strip())
    return {"catalog_revenue_2025": value}


def _resolve_mrelg_metadata(mrelg_id: str) -> Dict[str, Any]:
    """
    Look up MRELG metadata. Prefer the local MARKETSHARE_SEARCH_SUMMARY table
    (populated daily) for speed, and fall back to a direct Snowflake query if
    the row is not present locally.

    Kept for backward compatibility with any caller that does not have its
    own Snowflake session; new code paths should call
    ``_resolve_mrelg_metadata_local`` first and pass an existing Snowflake
    connection into ``_resolve_mrelg_metadata_snowflake`` to avoid opening a
    second connection.
    """
    local = _resolve_mrelg_metadata_local(mrelg_id)
    if local is not None:
        return local
    with get_snowflake_connection() as _sf:
        return _resolve_mrelg_metadata_snowflake(mrelg_id, _sf)


def _resolve_mrelg_metadata_local(
    mrelg_id: str,
    *,
    search_table: str = MARKETSHARE_SEARCH_SUMMARY_TABLE,
) -> Optional[Dict[str, Any]]:
    """
    Try to resolve metadata from the daily SQLite snapshot. Returns ``None``
    when the row is not present so the caller can decide whether to fall
    back to Snowflake (and reuse an existing session if it has one).
    """
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                f"SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, RELEASE_DATE, GENRE "
                f"FROM {search_table} WHERE MRELG_ID = ? LIMIT 1",
                (mrelg_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "mrelg_id": row["MRELG_ID"],
                "title": row["TITLE"],
                "artist": row["ARTIST"],
                "label_name": row["LABEL_NAME"],
                "release_date": row["RELEASE_DATE"],
                "genre": row["GENRE"],
            }
    except sqlite3.Error as e:
        logger.warning("mrelg metadata: SQLite lookup failed (%s); will fall back to Snowflake", e)
        return None


def _resolve_mrelg_metadata_snowflake(mrelg_id: str, sf: Snowflake) -> Dict[str, Any]:
    """Resolve metadata from Snowflake using an already-open session."""
    df = _verify_mrelg_id(mrelg_id, sf).rename(columns=str.upper)
    row = df.iloc[0]
    return {
        "mrelg_id": str(row.get("MRELG_ID") or mrelg_id),
        "title": (row.get("TITLE") or ""),
        "artist": (row.get("DISPLAY_ARTIST") or row.get("ARTIST") or ""),
        "label_name": None,
        "release_date": str(row.get("RELEASE_DATE") or "").split(" ")[0],
        "genre": row.get("GENRE"),
    }


def _build_global_streaming_forecast(
    mrelg_id: str,
    release_date: str,
    artist: str,
    title: str,
    genre: Any,
    fw_streams_peak: float,
    span: Optional[Dict[str, Any]] = None,
    sf: Optional[Snowflake] = None,
    scenario: str = "Base",
    hist_df: Optional[pd.DataFrame] = None,
    singles: bool = False,
) -> pd.DataFrame:
    """
    Shared backbone for both the legacy release_id-driven and the new MRELG-driven
    global streaming forecast endpoints. Pulls observed weekly streams from
    Snowflake, runs the worldwide-streams archetype simulation, and returns
    the actual-plus-forecast frame.

    ``singles=True`` loads archetypes from ``worldwide_streams_singles`` instead
    of ``worldwide_streams``; simulation code is unchanged.

    ``sf`` lets callers pass an already-open Snowflake session so we don't pay
    the connect/auth cost more than once per request. ``hist_df`` lets callers
    that have already pulled the same observed-weekly-streams frame skip the
    duplicate Snowflake query entirely — the search/EOY path needs the same
    data to seed its catalog-decay AR loop, so reusing it cuts the search
    request's Snowflake cost roughly in half. ``span`` enables phase-level
    timing logs without polluting the production code path with bookkeeping
    when omitted (single-shot scripts).

    ``scenario`` ("Base" / "Bear" / "Bull") is forwarded into the simulation's
    release_dict so the worldwide_streams engine can apply the corresponding
    post-fit shock — see ``simulate_one_worldwide_streams``. Defaults to "Base"
    so all existing callers and tests are unaffected.
    """
    @contextlib.contextmanager
    def _phase(name: str, **extra: Any):
        if span is not None:
            with _perf_phase(span, name, endpoint="global_streaming_by_mrelg", **extra) as info:
                yield info
        else:
            yield {}

    @contextlib.contextmanager
    def _sf_session():
        if sf is not None:
            yield sf
        else:
            with get_snowflake_connection() as _sf:
                yield _sf

    with _phase("weekly_streams_cache") as info:
        with _sf_session() as session:
            try:
                hist_df = _get_known_vols_global_streaming(mrelg_id, release_date, session)
            except ValueError as err:
                msg = str(err)
                if fw_streams_peak > 0 and "No global streaming data found" in msg:
                    logger.info(
                        "global_streaming: %s; using cold-start (fw_streams_peak=%s) for mrelg_id=%s",
                        msg,
                        fw_streams_peak,
                        mrelg_id,
                    )
                    hist_df = pd.DataFrame()
                else:
                    raise
        info["rows"] = len(hist_df)

    artifacts = (
        get_worldwide_singles_artifacts()
        if singles
        else get_worldwide_artifacts()
    )
    horizon_weeks = int(artifacts.horizon_weeks)

    known: List[float] = []
    if hist_df.empty:
        if fw_streams_peak <= 0:
            raise ValueError(
                f"No global streaming data in Snowflake for mrelg_id: {mrelg_id} "
                f"on/after release_date {release_date!r}, and no first-week peak "
                "(set FW_STREAMS / EXPECTED_ALBUM_EQUIVALENT on EXPECTED_RELEASES, or fix Snowflake access)."
            )
    else:
        stream_col = next(
            (c for c in hist_df.columns if "stream" in c.lower()),
            hist_df.columns[-1],
        )
        series = pd.to_numeric(hist_df[stream_col], errors="coerce").fillna(0.0)
        known = series.tolist()
    if len(known) > horizon_weeks:
        logger.info(
            "global_streaming: truncating observed weeks for mrelg_id=%s from %d to %d",
            mrelg_id,
            len(known),
            horizon_weeks,
        )
        # fit_backfill_forecast requires len(actuals) <= end_week <= horizon.
        # Keep the earliest weeks (week 1..horizon) for a valid backfill fit.
        known = known[:horizon_weeks]
        hist_df = hist_df.iloc[:horizon_weeks].copy()

    if not any(x > 0 for x in known) and fw_streams_peak <= 0:
        raise ValueError(
            f"mrelg_id {mrelg_id} has no observed worldwide stream history and no "
            "first-week peak available; cannot produce a forecast."
        )

    release_dict: Dict[str, Any] = {
        "artist": artist or "",
        "name": title or "",
        "genre": genre,
        "date": release_date,
        "known_worldwide_streams": known,
        "fw_worldwide_streams": float(fw_streams_peak or 0.0),
        "scenario": _normalize_scenario_label(scenario),
    }

    with _phase("simulation") as info:
        result = simulate_one_worldwide_streams(release_dict, artifacts, end_week=horizon_weeks)
        info["weeks"] = horizon_weeks
        info["known"] = len(known)

    n_known = len(known)
    df = pd.DataFrame(result["weekly"])  # week, pred_worldwide_streams, cumulative_worldwide_streams

    if not hist_df.empty:
        date_col = next(c for c in hist_df.columns if "date" in c.lower())
        hist_dates = pd.to_datetime(hist_df[date_col]).reset_index(drop=True)
        last_known_date = hist_dates.iloc[-1]

        def _week_to_date(week: int) -> str:
            idx = week - 1
            if idx < len(hist_dates):
                return hist_dates.iloc[idx].strftime("%Y-%m-%d")
            return (last_known_date + pd.Timedelta(weeks=(week - n_known))).strftime("%Y-%m-%d")
    else:
        rel_dt = pd.to_datetime(release_date)

        def _week_to_date(week: int) -> str:  # type: ignore[misc]
            return (rel_dt + pd.Timedelta(weeks=week)).strftime("%Y-%m-%d")

    df["week_ending_date"] = df["week"].apply(_week_to_date)
    df["data_type"] = df["week"].apply(lambda w: "Actual" if w <= n_known else "Forecast")

    df.insert(0, "mrelg_id", mrelg_id)
    df.insert(1, "artist", artist or "")
    df.insert(2, "title", title or "")
    week_pos = df.columns.get_loc("week")
    for col in ("data_type", "week_ending_date"):
        df.insert(week_pos + 1, col, df.pop(col))

    return df


def _get_known_vols_global_streaming(
    mrelg_id: str,
    release_date: str,
    _sf: Snowflake,
    *,
    refresh_if_stale: bool = True,
) -> pd.DataFrame:
    """Weekly worldwide streams for archetype fit — SQLite cache, incremental Snowflake."""
    from sqlite_handler import get_weekly_global_streams_for_mrelg

    df = get_weekly_global_streams_for_mrelg(
        mrelg_id,
        release_date=release_date,
        refresh_if_stale=refresh_if_stale,
        sf_conn=_sf,
    )
    if df.empty:
        raise ValueError(
            f"No global streaming data found for mrelg_id: {mrelg_id} "
            f"on/after release_date: {release_date}"
        )
    return df


def _resolve_mrelg_metadata_from_streaming_roster(mrelg_id: str) -> Optional[Dict[str, Any]]:
    """Metadata from STREAMING_ROSTER_2026 when search tables lack the row."""
    try:
        with sqlite3.connect(DATABASE_NAME) as conn:
            ensure_streaming_roster_2026_table(conn)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT MRELG_ID, TITLE, ARTIST, LABEL_NAME, PARENT_GROUP, RELEASE_DATE, PRODUCT_TYPE "
                "FROM STREAMING_ROSTER_2026 WHERE MRELG_ID = ? LIMIT 1",
                (mrelg_id.strip(),),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "mrelg_id": row["MRELG_ID"],
                "title": row["TITLE"],
                "artist": row["ARTIST"],
                "label_name": row["LABEL_NAME"],
                "parent_group": row["PARENT_GROUP"],
                "release_date": row["RELEASE_DATE"],
                "genre": None,
                "product_type": row["PRODUCT_TYPE"],
            }
    except sqlite3.Error as e:
        logger.warning("streaming roster metadata lookup failed for %s: %s", mrelg_id, e)
        return None


def prewarm_streaming_roster_caches(
    *,
    daily: bool = True,
    weekly: bool = True,
    only_stale: bool = True,
    limit: int | None = None,
) -> dict:
    """
    Batch-fill daily + weekly worldwide stream caches for STREAMING_ROSTER_2026.

    Uses one Snowflake session and incremental pulls per MRELG. Intended after
    roster backfill or on a nightly cron (30-day roster incremental + prewarm stale).

    Env:
      TIDE_PREWARM_STREAMING_ROSTER_LIMIT — cap rows processed (testing)
    """
    import os as _os

    from sqlite_handler import (
        _daily_streams_is_fresh,
        _weekly_streams_is_fresh,
        refresh_daily_global_streams_for_mrelg,
        refresh_weekly_global_streams_for_mrelg,
    )

    if limit is None:
        lim_env = _os.environ.get("TIDE_PREWARM_STREAMING_ROSTER_LIMIT", "").strip()
        limit = int(lim_env) if lim_env.isdigit() else None

    roster = get_streaming_roster_2026()
    if limit is not None:
        roster = roster[: int(limit)]

    stats = {
        "roster_rows": len(roster),
        "daily_refreshed": 0,
        "daily_skipped": 0,
        "weekly_refreshed": 0,
        "weekly_skipped": 0,
        "errors": [],
    }
    if not roster:
        return stats

    with get_snowflake_connection() as sf:
        for i, row in enumerate(roster):
            mrelg_id = (row.get("MRELG_ID") or "").strip()
            if not mrelg_id:
                continue
            rd = row.get("RELEASE_DATE")
            release_date = str(rd).split(" ")[0][:10] if rd else ""
            if not release_date:
                stats["errors"].append({"mrelg_id": mrelg_id, "error": "missing RELEASE_DATE"})
                continue

            try:
                with sqlite3.connect(DATABASE_NAME) as conn:
                    cur = conn.cursor()
                    if daily:
                        if only_stale and _daily_streams_is_fresh(cur, mrelg_id):
                            stats["daily_skipped"] += 1
                        else:
                            refresh_daily_global_streams_for_mrelg(
                                mrelg_id, release_date, sf_conn=sf
                            )
                            stats["daily_refreshed"] += 1
                    if weekly:
                        if only_stale and _weekly_streams_is_fresh(cur, mrelg_id):
                            stats["weekly_skipped"] += 1
                        else:
                            refresh_weekly_global_streams_for_mrelg(
                                mrelg_id, release_date, sf_conn=sf
                            )
                            stats["weekly_refreshed"] += 1
            except Exception as e:
                stats["errors"].append({"mrelg_id": mrelg_id, "error": str(e)})
                logger.exception(
                    "prewarm_streaming_roster: failed mrelg_id=%s (%d/%d)",
                    mrelg_id,
                    i + 1,
                    len(roster),
                )

            if (i + 1) % 100 == 0:
                logger.info(
                    "prewarm_streaming_roster: progress %d/%d daily_refreshed=%d weekly_refreshed=%d",
                    i + 1,
                    len(roster),
                    stats["daily_refreshed"],
                    stats["weekly_refreshed"],
                )

    if stats["daily_refreshed"] or stats["weekly_refreshed"]:
        sync_db_to_s3()

    logger.info("prewarm_streaming_roster_caches: %s", stats)
    return stats
