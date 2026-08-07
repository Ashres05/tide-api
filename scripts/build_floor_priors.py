#!/usr/bin/env python3
"""
Build floor_priors.json from artist_release_history.parquet for AE and WW
archetype artifact dirs.

Eligibility for the prior table:
  - peak_week_obs <= 26 (early primary peak)
  - finite peak/tail, peak > 0
  - retention clipped out of extreme incompleteness only at aggregate time

Writes floor_priors.json next to each history parquet so load_artifacts() picks it up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

AE_BINS: List[Tuple[str, float, float]] = [
    ("<1k", 0.0, 1_000.0),
    ("1-5k", 1_000.0, 5_000.0),
    ("5-20k", 5_000.0, 20_000.0),
    ("20-50k", 20_000.0, 50_000.0),
    ("50-100k", 50_000.0, 100_000.0),
    ("100-300k", 100_000.0, 300_000.0),
    ("300k+", 300_000.0, float("inf")),
]

WW_BINS: List[Tuple[str, float, float]] = [
    ("<1M", 0.0, 1_000_000.0),
    ("1-5M", 1_000_000.0, 5_000_000.0),
    ("5-20M", 5_000_000.0, 20_000_000.0),
    ("20-50M", 20_000_000.0, 50_000_000.0),
    ("50-100M", 50_000_000.0, 100_000_000.0),
    ("100-300M", 100_000_000.0, 300_000_000.0),
    ("300M+", 300_000_000.0, float("inf")),
]

DEFAULT_DIRS = [
    ("ae", REPO / "model" / "archetypes_artifacts" / "streams"),
    ("ae", REPO / "model" / "archetypes_artifacts" / "singles" / "streams"),
    ("ww", REPO / "model" / "archetypes_artifacts" / "worldwide_streams"),
    ("ww", REPO / "model" / "archetypes_artifacts" / "worldwide_streams_singles"),
]

MATURE_MAX_PEAK_WEEK = 26.0
MIN_GENRE_N = 50
MAX_RET_FOR_PRIOR = 0.80  # drop near-flat incomplete rows from prior construction


def _bin_name(peak: float, bins: List[Tuple[str, float, float]]) -> Optional[str]:
    for name, lo, hi in bins:
        if peak >= lo and peak < hi:
            return name
    # last open-ended bin uses hi=inf; still catch equality on lo of last
    if bins and peak >= bins[-1][1]:
        return bins[-1][0]
    return None


def _agg(rets: pd.Series) -> Optional[Dict[str, float]]:
    r = pd.to_numeric(rets, errors="coerce").dropna()
    r = r[(r > 0) & (r <= MAX_RET_FOR_PRIOR)]
    if r.empty:
        return None
    return {
        "median": float(r.median()),
        "p10": float(r.quantile(0.10)),
        "p90": float(r.quantile(0.90)),
        "n": int(len(r)),
    }


def build_priors(history: pd.DataFrame, kind: str) -> Dict[str, Any]:
    bins = AE_BINS if kind == "ae" else WW_BINS
    h = history.dropna(subset=["peak_volume_obs", "tail_volume_obs"]).copy()
    h["peak"] = pd.to_numeric(h["peak_volume_obs"], errors="coerce")
    h["tail"] = pd.to_numeric(h["tail_volume_obs"], errors="coerce")
    h["pw"] = pd.to_numeric(h.get("peak_week_obs"), errors="coerce")
    h = h.dropna(subset=["peak", "tail"])
    h = h[h["peak"] > 0].copy()
    h["ret"] = h["tail"] / h["peak"]
    # Early primary peak only for prior table.
    h = h[h["pw"].notna() & (h["pw"] <= MATURE_MAX_PEAK_WEEK)].copy()
    h["bin"] = h["peak"].apply(lambda p: _bin_name(float(p), bins))
    h = h[h["bin"].notna()].copy()

    global_priors: Dict[str, Dict[str, float]] = {}
    for name, _lo, _hi in bins:
        agg = _agg(h.loc[h["bin"] == name, "ret"])
        if agg is not None:
            global_priors[name] = agg

    genre_priors: Dict[str, Dict[str, Dict[str, float]]] = {}
    if "main_genre" in h.columns:
        h["genre"] = h["main_genre"].astype(str).fillna("Unknown")
        for genre, gdf in h.groupby("genre"):
            g_name = str(genre).strip() or "Unknown"
            if g_name.casefold() == "unknown":
                continue
            by_bin: Dict[str, Dict[str, float]] = {}
            for name, _lo, _hi in bins:
                agg = _agg(gdf.loc[gdf["bin"] == name, "ret"])
                if agg is not None and agg["n"] >= MIN_GENRE_N:
                    by_bin[name] = agg
            if by_bin:
                genre_priors[g_name] = by_bin

    return {
        "kind": kind,
        "mature_max_peak_week": MATURE_MAX_PEAK_WEEK,
        "min_genre_n": MIN_GENRE_N,
        "max_ret_for_prior": MAX_RET_FOR_PRIOR,
        "bins": [{"name": n, "lo": lo, "hi": None if np.isinf(hi) else hi} for n, lo, hi in bins],
        "global": global_priors,
        "genre": genre_priors,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dir",
        action="append",
        default=[],
        help="Artifact dir containing artist_release_history.parquet (repeatable).",
    )
    p.add_argument(
        "--kind",
        choices=("ae", "ww", "auto"),
        default="auto",
        help="Binning scheme; auto picks from --dir defaults or median peak.",
    )
    args = p.parse_args()

    targets: List[Tuple[str, Path]] = []
    if args.dir:
        for d in args.dir:
            path = Path(d)
            kind = args.kind
            if kind == "auto":
                # Heuristic: worldwide in path => ww
                kind = "ww" if "worldwide" in str(path).lower() else "ae"
            targets.append((kind, path))
    else:
        targets = [(k, d) for k, d in DEFAULT_DIRS if (d / "artist_release_history.parquet").is_file()]

    if not targets:
        print("No artifact dirs with artist_release_history.parquet found.", file=sys.stderr)
        return 1

    for kind, out_dir in targets:
        hist_path = out_dir / "artist_release_history.parquet"
        if not hist_path.is_file():
            print(f"skip missing {hist_path}")
            continue
        hist = pd.read_parquet(hist_path)
        if args.kind != "auto" and args.dir:
            kind = args.kind
        priors = build_priors(hist, kind)
        out_path = out_dir / "floor_priors.json"
        out_path.write_text(json.dumps(priors, indent=2), encoding="utf-8")
        n_global = len(priors["global"])
        n_genre = sum(len(v) for v in priors["genre"].values())
        print(
            f"wrote {out_path} kind={kind} global_bins={n_global} "
            f"genre_bin_cells={n_genre} eligible_rows="
            f"{sum(int(v['n']) for v in priors['global'].values())}"
        )
        # Show a few key bins
        for key in ("100-300k", "50-100k", "100-300M", "50-100M"):
            if key in priors["global"]:
                g = priors["global"][key]
                print(f"  {key}: median={g['median']:.4f} p10={g['p10']:.4f} p90={g['p90']:.4f} n={g['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
