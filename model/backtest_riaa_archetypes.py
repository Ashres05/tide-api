#!/usr/bin/env python3
"""
Rolling hold-out backtest for RIAA archetype artifacts (sales / streams / songs).

For each album, fit ``fit_backfill_forecast`` on the first K observed weeks, then
score the next ``horizon`` weeks against held-out actuals. Mirrors the inference
path used at serve time (Base scenario, per-channel floors from ``catalog_floor_rate``).

Example::

    python -m model.backtest_riaa_archetypes \\
      --panel model/data/riaa_train_panel.parquet \\
      --sales-dir model/archetypes_artifacts/riaa/sales \\
      --streams-dir model/archetypes_artifacts/riaa/streams \\
      --songs-dir model/archetypes_artifacts/riaa/songs \\
      --holdout-weeks 8 --n-albums 300 --out-csv model/data/riaa_backtest_rows.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.all_data_archetypes_simulator_ae import (  # noqa: E402
    SimulatorArtifacts,
    fit_backfill_forecast,
    load_artifacts,
)
from model.riaa_feature_engineering import (  # noqa: E402
    COL_ALBUM_ID,
    COL_SALES,
    COL_SONGS,
    COL_STREAMS,
    COL_TOTAL,
    genre_to_simulator_genres,
)

logger = logging.getLogger(__name__)

METRIC_SPECS = (
    ("product_sales", COL_SALES, "sales"),
    ("streaming_equivalent", COL_STREAMS, "streams"),
    ("song_sale_equivalent", COL_SONGS, "songs"),
)


def _fit_channel(
    *,
    artist: str,
    genre: Optional[str],
    known: np.ndarray,
    artifacts: SimulatorArtifacts,
    end_week: int,
    stream_floor: Optional[float],
) -> np.ndarray:
    """Weekly predictions for weeks 1..end_week (actuals overwritten in output)."""
    if len(known) == 0 or not np.any(np.isfinite(known) & (known > 0)):
        return np.zeros(end_week, dtype=float)
    pred_df, _ = fit_backfill_forecast(
        artist=artist,
        genre=genre,
        actuals_weekly_streams=np.asarray(known, dtype=float),
        artifacts=artifacts,
        end_week=end_week,
        stream_floor=stream_floor,
        scenario="Base",
        scenario_multiplier=1.0,
    )
    return pred_df["pred_weekly_streams"].to_numpy(dtype=float)[:end_week]


def _channel_floor(known: np.ndarray, floor_rate: float) -> float:
    peak = float(np.max(known)) if len(known) else 0.0
    if peak <= 0 or not np.isfinite(floor_rate):
        return 0.0
    return float(peak * floor_rate)


def backtest_one_album(
    g: pd.DataFrame,
    *,
    holdout_weeks: int,
    min_train_weeks: int,
    artifacts: Dict[str, SimulatorArtifacts],
) -> Optional[List[Dict[str, Any]]]:
    """
    Hold out the last ``holdout_weeks`` indexed weeks; train on weeks 1..K.
    """
    g = g.sort_values("week")
    w = g["week"].to_numpy(dtype=int)
    if len(w) < min_train_weeks + holdout_weeks:
        return None

    k = int(len(w) - holdout_weeks)
    if k < min_train_weeks:
        return None

    artist = str(g["artist"].iloc[0] or "").strip()
    if not artist:
        return None
    genre_raw = g["genre"].iloc[0]
    genre_blob = genre_to_simulator_genres(genre_raw)
    try:
        import json as _json

        genre = _json.loads(genre_blob)[0].get("MAIN_GENRE")
    except Exception:
        genre = str(genre_raw) if genre_raw is not None else None

    floor_rate = float(g["catalog_floor_rate"].iloc[0]) if "catalog_floor_rate" in g.columns else 0.0
    end_week = int(w[-1])

    rows: List[Dict[str, Any]] = []
    channel_preds: Dict[str, np.ndarray] = {}

    for _metric_name, col, key in METRIC_SPECS:
        y = pd.to_numeric(g[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        known = y[:k]
        floor = _channel_floor(known, floor_rate)
        try:
            full_pred = _fit_channel(
                artist=artist,
                genre=genre,
                known=known,
                artifacts=artifacts[key],
                end_week=end_week,
                stream_floor=floor if floor > 0 else None,
            )
        except Exception as e:
            logger.debug("skip %s %s: %s", g[COL_ALBUM_ID].iloc[0], key, e)
            return None
        channel_preds[key] = full_pred

    total_pred = (
        channel_preds["sales"] + channel_preds["streams"] + channel_preds["songs"]
    )
    y_total = pd.to_numeric(g[COL_TOTAL], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    album_id = str(g[COL_ALBUM_ID].iloc[0])
    for idx in range(k, len(w)):
        week = int(w[idx])
        actual_total = float(y_total[idx])
        pred_total = float(total_pred[week - 1])
        rows.append(
            {
                COL_ALBUM_ID: album_id,
                "artist": artist,
                "week": week,
                "actual_total_aeu": actual_total,
                "pred_total_aeu": pred_total,
                "actual_sales": float(g[COL_SALES].iloc[idx]),
                "pred_sales": float(channel_preds["sales"][week - 1]),
                "actual_streams": float(g[COL_STREAMS].iloc[idx]),
                "pred_streams": float(channel_preds["streams"][week - 1]),
                "actual_songs": float(g[COL_SONGS].iloc[idx]),
                "pred_songs": float(channel_preds["songs"][week - 1]),
                "train_weeks": k,
                "holdout_weeks": holdout_weeks,
            }
        )
    return rows


def summarize_scores(df: pd.DataFrame, *, label: str) -> Dict[str, float]:
    act = df[f"actual_{label}"].to_numpy(dtype=float)
    pred = df[f"pred_{label}"].to_numpy(dtype=float)
    err = pred - act
    abs_err = np.abs(err)
    return {
        "label": label,
        "rows": int(len(df)),
        "mae": float(abs_err.mean()),
        "wape": float(abs_err.sum() / max(act.sum(), 1.0)),
        "bias": float(err.mean()),
        "median_ape": float(np.median(abs_err / np.maximum(act, 1.0))),
        "rmse": float(np.sqrt((err**2).mean())),
    }


def certification_week_error(
    panel: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    threshold: float,
    holdout_weeks: int,
) -> pd.DataFrame:
    """
    Compare week index when cumulative AEU crosses ``threshold`` (actual vs forecast).
    Uses full history for actual cert week; forecast extends holdout preds with last train cum.
    """
    records = []
    for album_id, g in panel.groupby(COL_ALBUM_ID, sort=False):
        g = g.sort_values("week")
        cum_act = g[COL_TOTAL].cumsum()
        act_hit = g.loc[cum_act >= threshold, "week"]
        if act_hit.empty:
            continue
        actual_cert_week = int(act_hit.iloc[0])

        sub = scored[scored[COL_ALBUM_ID] == album_id].sort_values("week")
        if sub.empty:
            continue
        k = int(sub["train_weeks"].iloc[0])
        cum_train = float(g.loc[g["week"] <= k, COL_TOTAL].sum())
        cum = cum_train
        pred_cert_week = None
        for _, r in sub.iterrows():
            cum += float(r["pred_total_aeu"])
            if cum >= threshold:
                pred_cert_week = int(r["week"])
                break
        if pred_cert_week is None:
            continue
        records.append(
            {
                COL_ALBUM_ID: album_id,
                "actual_cert_week": actual_cert_week,
                "pred_cert_week": pred_cert_week,
                "cert_week_err": pred_cert_week - actual_cert_week,
                "threshold": threshold,
            }
        )
    return pd.DataFrame.from_records(records)


def run_backtest(
    *,
    panel_path: Path,
    sales_dir: Path,
    streams_dir: Path,
    songs_dir: Path,
    holdout_weeks: int,
    min_train_weeks: int,
    n_albums: int,
    seed: int,
    out_csv: Optional[Path],
    cert_threshold: Optional[float],
) -> Dict[str, Any]:
    panel = pd.read_parquet(panel_path)
    artifacts = {
        "sales": load_artifacts(str(sales_dir)),
        "streams": load_artifacts(str(streams_dir)),
        "songs": load_artifacts(str(songs_dir)),
    }
    horizon = int(artifacts["streams"].horizon_weeks)

    ids = panel[COL_ALBUM_ID].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)

    all_rows: List[Dict[str, Any]] = []
    used = 0
    for album_id in ids:
        if used >= n_albums:
            break
        g = panel[panel[COL_ALBUM_ID] == album_id]
        chunk = backtest_one_album(
            g,
            holdout_weeks=holdout_weeks,
            min_train_weeks=min_train_weeks,
            artifacts=artifacts,
        )
        if not chunk:
            continue
        all_rows.extend(chunk)
        used += 1

    if not all_rows:
        raise ValueError(
            "No scored rows — relax --min-train-weeks, --holdout-weeks, or --n-albums."
        )

    scored = pd.DataFrame(all_rows)
    summaries = [
        summarize_scores(scored, label="total_aeu"),
        summarize_scores(scored, label="sales"),
        summarize_scores(scored, label="streams"),
        summarize_scores(scored, label="songs"),
    ]

    result: Dict[str, Any] = {
        "panel_path": str(panel_path),
        "albums_scored": used,
        "rows_scored": len(scored),
        "holdout_weeks": holdout_weeks,
        "min_train_weeks": min_train_weeks,
        "horizon_weeks": horizon,
        "metrics": summaries,
    }

    if cert_threshold is not None and cert_threshold > 0:
        cert_df = certification_week_error(
            panel, scored, threshold=float(cert_threshold), holdout_weeks=holdout_weeks
        )
        if not cert_df.empty:
            result["certification"] = {
                "threshold": float(cert_threshold),
                "albums": int(len(cert_df)),
                "mae_weeks": float(cert_df["cert_week_err"].abs().mean()),
                "median_week_err": float(cert_df["cert_week_err"].median()),
                "within_4_weeks_pct": float((cert_df["cert_week_err"].abs() <= 4).mean()),
            }
        else:
            result["certification"] = {"threshold": float(cert_threshold), "albums": 0}

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_csv, index=False)
        result["out_csv"] = str(out_csv)

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model_dir = Path(__file__).resolve().parent
    data_dir = model_dir / "data"
    artifacts_base = model_dir / "archetypes_artifacts" / "riaa"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", type=Path, default=data_dir / "riaa_train_panel.parquet")
    p.add_argument("--sales-dir", type=Path, default=artifacts_base / "sales")
    p.add_argument("--streams-dir", type=Path, default=artifacts_base / "streams")
    p.add_argument("--songs-dir", type=Path, default=artifacts_base / "songs")
    p.add_argument("--holdout-weeks", type=int, default=8, help="Trailing weeks to score")
    p.add_argument("--min-train-weeks", type=int, default=12, help="Min weeks before holdout for fit")
    p.add_argument("--n-albums", type=int, default=300, help="Random albums to evaluate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument(
        "--cert-threshold",
        type=float,
        default=500_000.0,
        help="If >0, also score certification week error vs this cumulative AEU (0=skip).",
    )
    args = p.parse_args()

    result = run_backtest(
        panel_path=args.panel,
        sales_dir=args.sales_dir,
        streams_dir=args.streams_dir,
        songs_dir=args.songs_dir,
        holdout_weeks=args.holdout_weeks,
        min_train_weeks=args.min_train_weeks,
        n_albums=args.n_albums,
        seed=args.seed,
        out_csv=args.out_csv,
        cert_threshold=args.cert_threshold if args.cert_threshold > 0 else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
