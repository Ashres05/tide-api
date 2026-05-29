"""
Train RIAA archetype bundles (sales / streams / songs) from ``riaa_train_panel.parquet``.

Writes simulator-ready parquet and runs ``all_data_archetypes_simulator_ae train`` three times.

  python -m model.train_riaa_archetypes
  python -m model.train_riaa_archetypes --panel-path model/data/riaa_train_panel.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from model.riaa_feature_engineering import (
    PANEL_OUTPUT,
    genre_to_simulator_genres,
    prepare_simulator_train_frame,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent
DATA_DIR = MODEL_DIR / "data"
ARTIFACTS_BASE = MODEL_DIR / "archetypes_artifacts" / "riaa"
SIMULATOR_TRAIN_PARQUET = "riaa_simulator_train_panel.parquet"

METRICS = (
    ("product_sales", "sales"),
    ("streaming_equivalent", "streams"),
    ("song_sale_equivalent", "songs"),
)

TRAIN_COLUMNS = (
    "MRELG_ID",
    "TITLE",
    "DISPLAY_ARTIST",
    "GENRES",
    "FIRST_SALE_DATE",
    "WEEK_END_DATE",
    "PRODUCT_SALES",
    "STREAMING_EQUIVALENT",
    "SONG_SALE_EQUIVALENT",
    "TOTAL_ALBUM_EQUIVALENTS",
)


def build_simulator_train_parquet(
    panel_path: Path,
    out_path: Path,
) -> Path:
    """Map RIAA panel → archetype simulator schema and persist."""
    logger.info("Reading %s", panel_path)
    panel = pd.read_parquet(panel_path)
    train_df = prepare_simulator_train_frame(panel)
    train_df["GENRES"] = train_df["GENRES"].map(genre_to_simulator_genres)
    keep = [c for c in TRAIN_COLUMNS if c in train_df.columns]
    train_df = train_df[keep].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_path, index=False)
    logger.info(
        "Wrote simulator train panel: %s (%d rows, %d releases)",
        out_path,
        len(train_df),
        train_df["MRELG_ID"].nunique(),
    )
    return out_path


def run_metric_train(
    *,
    parquet_path: Path,
    out_dir: Path,
    metric: str,
    horizon_weeks: int,
    n_clusters: int,
    random_state: int,
    dry_run: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "model.all_data_archetypes_simulator_ae",
        "train",
        "--parquet-path",
        str(parquet_path),
        "--out-dir",
        str(out_dir),
        "--metric",
        metric,
        "--horizon-weeks",
        str(horizon_weeks),
        "--n-clusters",
        str(n_clusters),
        "--random-state",
        str(random_state),
    ]
    logger.info("Running: %s", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(MODEL_DIR.parent))


def train_all(
    *,
    panel_path: Path,
    data_dir: Path,
    artifacts_base: Path,
    horizon_weeks: int,
    n_clusters: int,
    random_state: int,
    skip_prepare: bool,
    dry_run: bool,
) -> dict:
    sim_parquet = data_dir / SIMULATOR_TRAIN_PARQUET
    if not skip_prepare:
        build_simulator_train_parquet(panel_path, sim_parquet)
    elif not sim_parquet.is_file():
        raise FileNotFoundError(
            f"--skip-prepare set but {sim_parquet} is missing; run without --skip-prepare first."
        )

    results = {"simulator_parquet": str(sim_parquet), "artifacts": {}}
    for metric, subdir in METRICS:
        out_dir = artifacts_base / subdir
        run_metric_train(
            parquet_path=sim_parquet,
            out_dir=out_dir,
            metric=metric,
            horizon_weeks=horizon_weeks,
            n_clusters=n_clusters,
            random_state=random_state,
            dry_run=dry_run,
        )
        results["artifacts"][subdir] = str(out_dir)
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Train RIAA sales/streams/songs archetype artifacts.")
    p.add_argument(
        "--panel-path",
        type=Path,
        default=DATA_DIR / PANEL_OUTPUT,
        help="Feature-engineered weekly panel from riaa_feature_engineering",
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--artifacts-base", type=Path, default=ARTIFACTS_BASE)
    p.add_argument("--horizon-weeks", type=int, default=78)
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Reuse existing riaa_simulator_train_panel.parquet",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = p.parse_args()

    if not args.panel_path.is_file():
        raise FileNotFoundError(
            f"Panel not found: {args.panel_path}. "
            "Run: python -m model.riaa_feature_engineering"
        )

    summary = train_all(
        panel_path=args.panel_path,
        data_dir=args.data_dir,
        artifacts_base=args.artifacts_base,
        horizon_weeks=args.horizon_weeks,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        skip_prepare=args.skip_prepare,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
