"""
Worldwide raw weekly stream-count simulation using archetypes_artifacts/worldwide_streams.

Separate from AE marketshare simulation (streams_equivalent + sales + songs). Outputs use
pred_worldwide_streams naming in JSON; internally reuses pred_weekly_streams from the decay engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .all_data_archetypes_simulator_ae import (
    SimulatorArtifacts,
    fit_backfill_forecast,
    simulate_future_drop,
)


def _weekly_table(pred_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in pred_df.iterrows():
        rows.append(
            {
                "week": int(row["week"]),
                "pred_worldwide_streams": float(row["pred_weekly_streams"]),
                "cumulative_worldwide_streams": float(row["cumulative_pred_streams"]),
            }
        )
    return rows


def _public_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in summary.items() if k != "total_lifecycle_pred_streams"}
    if "total_lifecycle_pred_streams" in summary:
        out["total_lifecycle_pred_worldwide_streams"] = float(summary["total_lifecycle_pred_streams"])
    return out


def simulate_one_worldwide_streams(
    release: Dict[str, Any],
    artifacts: SimulatorArtifacts,
    end_week: int,
) -> Dict[str, Any]:
    """
    release keys:
      - artist (required) and optional name (display title)
      - genre (optional but recommended; used for archetype mixture)
      - date — echoed back only (drop date for UI; not passed to decay engine)
      - fw_worldwide_streams — peak weekly count when no known_worldwide_streams
      - known_worldwide_streams — optional observed early weeks (raw counts)
      - peak_week — optional, default 1.0 for cold start
      - stream_floor — optional override for tail floor
    """
    artist = str(release.get("artist") or release.get("name") or "").strip()
    if not artist:
        raise ValueError("artist (or name) is required for worldwide_streams simulation.")

    genre = release.get("genre")
    if genre is not None:
        genre = str(genre).strip() or None

    known = release.get("known_worldwide_streams")
    if known is None:
        known = []
    if not isinstance(known, (list, tuple)):
        raise ValueError("known_worldwide_streams must be a list of numbers.")
    known_arr = np.array([float(x) for x in known], dtype=float)
    has_known_weeks = len(known_arr) > 0
    has_positive_known = bool(np.any(np.isfinite(known_arr) & (known_arr > 0)))

    fw_raw = release.get("fw_worldwide_streams")
    fw = float(fw_raw) if fw_raw is not None else 0.0
    stream_floor = release.get("stream_floor")
    if stream_floor is not None and stream_floor != "":
        stream_floor = float(stream_floor)
    else:
        stream_floor = None

    peak_week = release.get("peak_week")
    if peak_week is not None and peak_week != "":
        peak_week = float(peak_week)
    else:
        peak_week = None

    horizon = int(artifacts.horizon_weeks)
    end_week = int(max(1, min(int(end_week), horizon)))

    # Backfill requires at least one strictly positive observed week; otherwise the decay engine errors.
    if has_positive_known:
        pred_df, summary = fit_backfill_forecast(
            artist=artist,
            genre=genre,
            actuals_weekly_streams=known_arr,
            artifacts=artifacts,
            end_week=end_week,
            stream_floor=stream_floor,
        )
    elif has_known_weeks and not has_positive_known:
        if fw <= 0:
            raise ValueError(
                "known_worldwide_streams has no positive values; backfill cannot run. "
                "Omit known_worldwide_streams and set fw_worldwide_streams > 0 for a cold-start curve, "
                "or provide at least one week with count > 0."
            )
        pred_df, summary = simulate_future_drop(
            artist=artist,
            peak_volume=fw,
            peak_week=peak_week if peak_week is not None else 1.0,
            genre=genre,
            artifacts=artifacts,
            stream_floor=stream_floor,
        )
    else:
        if fw <= 0:
            raise ValueError("fw_worldwide_streams must be > 0 when known_worldwide_streams is empty.")
        pred_df, summary = simulate_future_drop(
            artist=artist,
            peak_volume=fw,
            peak_week=peak_week if peak_week is not None else 1.0,
            genre=genre,
            artifacts=artifacts,
            stream_floor=stream_floor,
        )

    pred_df = pred_df.iloc[:end_week].copy()
    weekly = _weekly_table(pred_df)
    return {
        "input": {
            "artist": artist,
            "name": release.get("name"),
            "genre": genre,
            "date": release.get("date"),
            "fw_worldwide_streams": fw,
            "known_worldwide_streams": [float(x) for x in known_arr.tolist()],
        },
        "summary": _public_summary(summary),
        "weekly": weekly,
    }


def simulate_worldwide_streams_calendar(
    releases: List[Dict[str, Any]],
    artifacts: SimulatorArtifacts,
    end_week: int = 78,
) -> Dict[str, Any]:
    """Run simulate_one_worldwide_streams for each release dict."""
    horizon = int(artifacts.horizon_weeks)
    end_week = int(max(1, min(int(end_week), horizon)))
    out_releases = [simulate_one_worldwide_streams(r, artifacts, end_week) for r in releases]
    return {"horizon_weeks": horizon, "end_week": end_week, "releases": out_releases}
