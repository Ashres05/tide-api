#!/usr/bin/env python3
"""
Rolling autoregressive backtest for a catalog-decay bundle vs parquet actuals.

Uses the same one-step predictor as the search EOY API
(``model_handler._predict_catalog_decay_step``): multiplier × lag1 when the
bundle says so, no Snowflake.

Example::

    cd /path/to/tide-api
    python model/backtest_catalog_decay_ar.py \\
      --bundle model/catalog_decay_multiplier_artifacts/catalog_decay_model.joblib \\
      --parquet model/data/samples/catalog_streams_pruned_80k_lifecycle_1m.parquet \\
      --horizon 8 --n-tracks 200 --seed 42

Optional blend with a simple baseline (same env vars as ``get_eoy_search_forecast``)::

    TIDE_CATALOG_DECAY_BLEND_ALPHA=0.65 TIDE_CATALOG_DECAY_BLEND_BASELINE=b52 \\
      python model/backtest_catalog_decay_ar.py ...
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import model_handler as mh  # noqa: E402
from model.train_catalog_decay import (  # noqa: E402
    CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52,
    CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52,
    CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1,
    baseline52_median_from_history_stream,
    hybrid_inference_denominator_and_features,
)

logger = logging.getLogger(__name__)


def _norm_upper(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.upper() for c in df.columns})


def _bundle_target_flags(bundle: dict) -> tuple[str, bool]:
    tt = str(bundle.get("target_transform") or "none").strip().lower()
    if tt == "log1p":
        return tt, False
    if "target_is_multiplier" in bundle:
        return tt, bool(bundle["target_is_multiplier"])
    return tt, True


def _ar_forecast_h(
    bundle: dict,
    *,
    mrelg_id: str,
    hist_df: pd.DataFrame,
    horizon: int,
) -> list[tuple[pd.Timestamp, float]]:
    """
    History rows: sorted ascending by WEEK_END_DATE, all <= as_of (caller ensures).
    Returns list of (forecast_week_end, pred_streams) length ``horizon``.
    """
    if hist_df.empty or horizon < 1:
        return []

    hist_df = hist_df.sort_values("WEEK_END_DATE").reset_index(drop=True)
    series = pd.to_numeric(hist_df["WORLDWIDE_STREAMS"], errors="coerce").fillna(0.0).astype(float).tolist()
    if not any(v > 0 for v in series):
        return []

    last = hist_df.iloc[-1]
    last_known = pd.Timestamp(hist_df["WEEK_END_DATE"].iloc[-1]).normalize()
    title = str(last.get("TITLE", "") or "")
    display_artist = str(last.get("DISPLAY_ARTIST", last.get("ARTIST", "")) or "unknown")
    genres = last.get("GENRES")
    genres_json = str(genres) if genres is not None and not (isinstance(genres, float) and np.isnan(genres)) else "{}"
    _release = pd.to_datetime(last.get("RELEASE_DATE"), errors="coerce")
    if pd.isna(_release):
        release_dt = last_known
    else:
        release_dt = pd.Timestamp(_release).normalize()
    _first_sale = pd.to_datetime(last.get("FIRST_SALE_DATE"), errors="coerce")
    if pd.isna(_first_sale):
        first_sale_dt = release_dt
    else:
        first_sale_dt = pd.Timestamp(_first_sale).normalize()

    default_hist = float(bundle.get("artist_history_default_log_median", 0.0))
    artist_hist_map = mh._artist_history_log_map_for_inference(
        display_artist,
        series,
        default_log_median=default_hist,
    )

    model = bundle["model"]
    feature_columns: list[str] = list(bundle["feature_columns"])
    top_genres: list[str] = list(bundle["top_genres"])
    top_artists: list[str] = list(bundle["top_artists"])
    target_transform, target_is_multiplier = _bundle_target_flags(bundle)
    catalog_decay_target = str(
        bundle.get("catalog_decay_target") or CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1
    ).strip()

    rolling = list(series)
    current = last_known + pd.Timedelta(days=7)
    preds: list[tuple[pd.Timestamp, float]] = []

    for _ in range(horizon):
        lag1w = float(rolling[-1]) if rolling else 0.0
        lag4w = mh._mean_tail(rolling, 4)
        lag12w = mh._mean_tail(rolling, 12)
        wsr = max(0.0, (current.normalize() - release_dt).days / 7.0)
        bl52 = baseline52_median_from_history_stream(rolling)
        if catalog_decay_target == CATALOG_DECAY_TARGET_REL_RESIDUAL_BASELINE52 and not math.isfinite(
            bl52
        ):
            bl52 = float(rolling[-1]) if rolling else 0.0
        h_vol = h_br = h_sp = 0.0
        h_dd = None
        if catalog_decay_target == CATALOG_DECAY_TARGET_HYBRID_SPIKE_GATE_BASELINE52:
            h_vol, h_br, h_sp, h_dd = hybrid_inference_denominator_and_features(rolling)
        pred = mh._predict_catalog_decay_step(
            model=model,
            feature_columns=feature_columns,
            top_genres=top_genres,
            top_artists=top_artists,
            artist_history_log_median=artist_hist_map,
            artist_history_default_log_median=default_hist,
            mrelg_id=str(mrelg_id),
            title=title,
            display_artist=display_artist,
            genres_json=genres_json,
            release_dt=release_dt,
            first_sale_dt=first_sale_dt,
            week_end=current.normalize(),
            weeks_since_release=float(wsr),
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
        pred = mh._apply_catalog_decay_baseline_blend(
            float(pred), lag1w=lag1w, baseline_52w=bl52
        )
        preds.append((current.normalize(), float(pred)))
        rolling.append(float(pred))
        current = current + pd.Timedelta(days=7)

    return preds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True, help="catalog_decay_model.joblib or .pkl")
    p.add_argument("--parquet", type=Path, required=True, help="Catalog streams parquet (panel)")
    p.add_argument("--horizon", type=int, default=8, help="Weeks of AR to score after as_of")
    p.add_argument("--n-tracks", type=int, default=200, help="Random MRELGs to evaluate")
    p.add_argument("--min-history", type=int, default=8, help="Min weekly rows on or before as_of")
    p.add_argument("--as-of", type=str, default=None, help="ISO as-of week end (default: max week - buffer)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=None, help="Optional row-level actual vs pred CSV")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    buf = args.bundle.read_bytes()
    try:
        import io

        import joblib

        bio = io.BytesIO(buf)
        bundle = joblib.load(bio)
    except Exception:
        import pickle

        bundle = pickle.loads(buf)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise SystemExit("Bundle must be a dict with 'model' (same as write_artifacts).")

    tt, tim = _bundle_target_flags(bundle)
    cdt = str(bundle.get("catalog_decay_target") or CATALOG_DECAY_TARGET_RETAINED_MULT_LAG1)
    logger.info(
        "Loaded bundle: catalog_decay_target=%s target_transform=%s target_is_multiplier=%s n_features=%d",
        cdt,
        tt,
        tim,
        len(bundle.get("feature_columns") or []),
    )

    df = _norm_upper(pd.read_parquet(args.parquet))
    for col in ("WEEK_END_DATE", "RELEASE_DATE", "FIRST_SALE_DATE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "MRELG_ID" not in df.columns or "WORLDWIDE_STREAMS" not in df.columns:
        raise SystemExit("Parquet must include MRELG_ID and WORLDWIDE_STREAMS")

    df["MRELG_ID"] = df["MRELG_ID"].astype(str)
    max_week = pd.Timestamp(df["WEEK_END_DATE"].max()).normalize()
    if args.as_of:
        as_of = pd.Timestamp(args.as_of).normalize()
    else:
        as_of = max_week - pd.DateOffset(weeks=int(args.horizon) + 8)
    logger.info("as_of=%s max_week=%s horizon=%d", as_of.date(), max_week.date(), args.horizon)

    rng = random.Random(int(args.seed))
    all_ids = df["MRELG_ID"].unique().tolist()
    rng.shuffle(all_ids)

    rows: list[dict] = []
    used = 0
    for mid in all_ids:
        if used >= int(args.n_tracks):
            break
        sub = df[df["MRELG_ID"] == mid].sort_values("WEEK_END_DATE")
        hist = sub[sub["WEEK_END_DATE"] <= as_of]
        if len(hist) < int(args.min_history):
            continue
        last_hist_week = pd.Timestamp(hist["WEEK_END_DATE"].max()).normalize()
        # Score weeks strictly after the history tail (matches AR starting last_hist + 7d).
        future = sub[sub["WEEK_END_DATE"] > last_hist_week].sort_values("WEEK_END_DATE")
        if len(future) < int(args.horizon):
            continue

        preds = _ar_forecast_h(bundle, mrelg_id=mid, hist_df=hist, horizon=int(args.horizon))
        if len(preds) != int(args.horizon):
            continue

        fut_slice = future.head(int(args.horizon)).reset_index(drop=True)
        for i in range(int(args.horizon)):
            we = pd.Timestamp(fut_slice.loc[i, "WEEK_END_DATE"]).normalize()
            act = float(pd.to_numeric(fut_slice.loc[i, "WORLDWIDE_STREAMS"], errors="coerce") or 0.0)
            pred_week, pr = preds[i]
            # Allow small calendar drift vs parquet cadence
            if abs((we - pred_week).days) > 4:
                logger.debug("week mismatch mrelg=%s row=%d parquet=%s ar=%s", mid, i, we, pred_week)
            rows.append(
                {
                    "MRELG_ID": mid,
                    "week_end": str(we.date()),
                    "actual": act,
                    "pred": pr,
                    "abs_err": abs(act - pr),
                    "ape": abs(act - pr) / max(act, 1.0),
                }
            )
        used += 1

    if not rows:
        raise SystemExit("No scored rows — relax min-history, horizon, or n-tracks.")

    out = pd.DataFrame(rows)
    mae = float(out["abs_err"].mean())
    wape = float(out["abs_err"].sum() / max(out["actual"].sum(), 1.0))
    bias = float((out["pred"] - out["actual"]).mean())
    med_ape = float(out["ape"].median())
    tiny = float((out["pred"] < 1.0).mean())
    logger.info(
        "tracks=%d rows=%d MAE=%.2f WAPE=%.4f bias(pred-act)=%.2f median_APE=%.4f frac_pred_lt_1=%.4f",
        used,
        len(out),
        mae,
        wape,
        bias,
        med_ape,
        tiny,
    )
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        logger.info("Wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
