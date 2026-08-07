#!/usr/bin/env python3
"""
Scorecard for dynamic stream floor.

Sources:
  --source roster   STREAMING_ROSTER_2026 + MARKETSHARE_WEEKLY_GLOBAL_STREAMS
                    + worldwide_streams artifacts
  --source backfill EXPECTED_RELEASES + MARKETSHARE_RELEASE_METRICS.STREAMING_EQUIVALENT
                    + AE streams archetypes (marketshare / backfill path)

Metrics:
  - boundary jump: y_hat[K+1] / y[K]  (want ~<=1, almost never >>1)
  - week-40+ MAPE: hold out weeks >=40, fit on earlier actuals, score held-out
  - late holdout MAPE: if n>=16, hold out last 4 (proxy)
  - named pain-case spot checks
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.all_data_archetypes_simulator_ae import (  # noqa: E402
    estimate_dynamic_stream_floor,
    fit_backfill_forecast,
    load_artifacts,
)

DB = REPO / "marketshare_data.db"
WW_DIR = REPO / "model" / "archetypes_artifacts" / "worldwide_streams"
WW_SINGLES_DIR = REPO / "model" / "archetypes_artifacts" / "worldwide_streams_singles"
AE_STREAMS_DIR = REPO / "model" / "archetypes_artifacts" / "streams"
AE_SINGLES_STREAMS_DIR = REPO / "model" / "archetypes_artifacts" / "singles" / "streams"

PAIN_SUBSTR = ("KEEM", "TOLIVER", "HARRY STYLES", "ELLA LANGLEY")
N_SAMPLE = 50
MIN_WEEKS_JUMP = 4
MIN_WEEKS_LATE_HOLDOUT = 16
HOLDOUT_TAIL = 4
HORIZON = 78


def _legacy_floor(artist: str, peak: float, artifacts) -> float:
    """Pre-MVP floor: top-3 by |peak| with no scale/mature/live cap."""
    hist = getattr(artifacts, "artist_release_history", None)
    if hist is None or not artist:
        return 0.0
    h = hist[hist["DISPLAY_ARTIST"].astype(str) == str(artist)].copy()
    if h.empty or "tail_volume_obs" not in h.columns:
        return 0.0
    h = h.dropna(subset=["tail_volume_obs", "peak_volume_obs"])
    h = h[pd.to_numeric(h["peak_volume_obs"], errors="coerce") > 0]
    if h.empty:
        return 0.0
    h = h.copy()
    h["peak_diff"] = (h["peak_volume_obs"].astype(float) - float(peak)).abs()
    top3 = h.sort_values("peak_diff").head(3)
    ret = (
        top3["tail_volume_obs"].astype(float)
        / top3["peak_volume_obs"].astype(float).replace(0, 1e-9)
    ).median()
    if not np.isfinite(ret) or ret <= 0:
        return 0.0
    return float(peak * ret)


def _load_roster_sample(conn: sqlite3.Connection) -> pd.DataFrame:
    roster = pd.read_sql_query(
        """
        SELECT r.MRELG_ID AS id_key, r.MRELG_ID, r.TITLE, r.ARTIST, r.LABEL_NAME,
               r.PRODUCT_TYPE, r.RELEASE_DATE,
               COUNT(w.WEEK_ENDING_DATE) AS n_weeks
        FROM STREAMING_ROSTER_2026 r
        JOIN MARKETSHARE_WEEKLY_GLOBAL_STREAMS w ON w.MRELG_ID = r.MRELG_ID
        GROUP BY r.MRELG_ID, r.TITLE, r.ARTIST, r.LABEL_NAME, r.PRODUCT_TYPE, r.RELEASE_DATE
        HAVING COUNT(w.WEEK_ENDING_DATE) >= ?
        """,
        conn,
        params=(MIN_WEEKS_JUMP,),
    )
    roster["RELEASE_DATE"] = pd.to_datetime(roster["RELEASE_DATE"], errors="coerce")
    albums = roster[roster["PRODUCT_TYPE"].astype(str).str.lower() == "album"].copy()
    albums = albums.sort_values("RELEASE_DATE", ascending=False).head(N_SAMPLE)
    pain = roster[
        roster["ARTIST"].astype(str).str.upper().apply(lambda a: any(p in a for p in PAIN_SUBSTR))
    ]
    return (
        pd.concat([albums, pain], ignore_index=True)
        .drop_duplicates(subset=["id_key"])
        .sort_values("RELEASE_DATE", ascending=False)
        .reset_index(drop=True)
    )


def _load_backfill_sample(conn: sqlite3.Connection) -> pd.DataFrame:
    releases = pd.read_sql_query(
        """
        SELECT e.RELEASE_ID AS id_key, e.RELEASE_ID, e.MRELG_ID, e.TITLE, e.ARTIST,
               e.LABEL_NAME, COALESCE(e.PRODUCT_TYPE, 'Album') AS PRODUCT_TYPE,
               e.RELEASE_DATE, e.GENRE,
               COUNT(m.WEEK_ENDING_DATE) AS n_weeks
        FROM EXPECTED_RELEASES e
        JOIN MARKETSHARE_RELEASE_METRICS m ON m.RELEASE_ID = e.RELEASE_ID
        GROUP BY e.RELEASE_ID, e.MRELG_ID, e.TITLE, e.ARTIST, e.LABEL_NAME,
                 e.PRODUCT_TYPE, e.RELEASE_DATE, e.GENRE
        HAVING COUNT(m.WEEK_ENDING_DATE) >= ?
        """,
        conn,
        params=(MIN_WEEKS_JUMP,),
    )
    releases["RELEASE_DATE"] = pd.to_datetime(releases["RELEASE_DATE"], errors="coerce")
    recent = releases.sort_values("RELEASE_DATE", ascending=False).head(N_SAMPLE)
    pain = releases[
        releases["ARTIST"].astype(str).str.upper().apply(lambda a: any(p in a for p in PAIN_SUBSTR))
    ]
    return (
        pd.concat([recent, pain], ignore_index=True)
        .drop_duplicates(subset=["id_key"])
        .sort_values("RELEASE_DATE", ascending=False)
        .reset_index(drop=True)
    )


def _series_roster(conn: sqlite3.Connection, mrelg_id: str) -> np.ndarray:
    df = pd.read_sql_query(
        """
        SELECT WEEK_ENDING_DATE, GLOBAL_STREAMS AS y
        FROM MARKETSHARE_WEEKLY_GLOBAL_STREAMS
        WHERE MRELG_ID = ?
        ORDER BY WEEK_ENDING_DATE
        """,
        conn,
        params=(mrelg_id,),
    )
    return pd.to_numeric(df["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _series_backfill(conn: sqlite3.Connection, release_id: int) -> np.ndarray:
    df = pd.read_sql_query(
        """
        SELECT WEEK_ENDING_DATE, STREAMING_EQUIVALENT AS y
        FROM MARKETSHARE_RELEASE_METRICS
        WHERE RELEASE_ID = ?
        ORDER BY WEEK_ENDING_DATE
        """,
        conn,
        params=(int(release_id),),
    )
    return pd.to_numeric(df["y"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-9)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])))


def _eval_one(row: pd.Series, y: np.ndarray, artifacts, genre: str | None) -> dict:
    artist = str(row["ARTIST"] or "")
    title = str(row["TITLE"] or "")
    k = int(len(y))
    peak_obs = float(np.max(y)) if k else 0.0
    y_k = float(y[-1]) if k else 0.0

    new_floor = estimate_dynamic_stream_floor(
        artist=artist,
        peak_volume=peak_obs,
        artifacts=artifacts,
        last_actual=y_k,
    )
    legacy = _legacy_floor(artist, peak_obs, artifacts)

    jump = float("nan")
    y_hat_next = float("nan")
    err = None
    try:
        pred, _summ = fit_backfill_forecast(
            artist=artist,
            genre=genre,
            actuals_weekly_streams=y,
            artifacts=artifacts,
            end_week=min(HORIZON, max(k + 8, k + 1)),
            scenario="Base",
        )
        yhat = pred["pred_weekly_streams"].to_numpy(dtype=float)
        if k < len(yhat) and y_k > 0:
            y_hat_next = float(yhat[k])
            jump = y_hat_next / y_k
    except Exception as e:
        err = str(e)

    mape_40 = float("nan")
    mape_late4 = float("nan")
    if k >= 44:
        train = y[:40]
        hold = y[40:]
        try:
            pred_h, _ = fit_backfill_forecast(
                artist=artist,
                genre=genre,
                actuals_weekly_streams=train,
                artifacts=artifacts,
                end_week=k,
                scenario="Base",
            )
            yhat_h = pred_h["pred_weekly_streams"].to_numpy(dtype=float)
            mape_40 = _mape(hold, yhat_h[40:k])
        except Exception:
            pass
    if k >= MIN_WEEKS_LATE_HOLDOUT:
        cut = k - HOLDOUT_TAIL
        train = y[:cut]
        hold = y[cut:]
        try:
            pred_h, _ = fit_backfill_forecast(
                artist=artist,
                genre=genre,
                actuals_weekly_streams=train,
                artifacts=artifacts,
                end_week=k,
                scenario="Base",
            )
            yhat_h = pred_h["pred_weekly_streams"].to_numpy(dtype=float)
            mape_late4 = _mape(hold, yhat_h[cut:k])
        except Exception:
            pass

    return {
        "id_key": row["id_key"],
        "mrelg_id": row.get("MRELG_ID"),
        "title": title,
        "artist": artist,
        "label": row.get("LABEL_NAME"),
        "product_type": row["PRODUCT_TYPE"],
        "release_date": str(row["RELEASE_DATE"].date()) if pd.notna(row["RELEASE_DATE"]) else None,
        "n_weeks": k,
        "peak_obs": peak_obs,
        "y_k": y_k,
        "legacy_floor": legacy,
        "new_floor": new_floor,
        "y_hat_k1": y_hat_next,
        "boundary_jump": jump,
        "mape_week40_plus": mape_40,
        "mape_late4_holdout": mape_late4,
        "is_pain_case": any(p in artist.upper() for p in PAIN_SUBSTR),
        "error": err,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=("roster", "backfill"),
        default="backfill",
        help="roster=streaming roster WW streams; backfill=EXPECTED_RELEASES AE streams",
    )
    args = p.parse_args()

    if not DB.is_file():
        print(f"missing db {DB}", file=sys.stderr)
        return 1

    if args.source == "roster":
        arts_album = load_artifacts(str(WW_DIR))
        arts_singles = (
            load_artifacts(str(WW_SINGLES_DIR)) if WW_SINGLES_DIR.is_dir() else arts_album
        )
        out_csv = REPO / "model" / "artifacts_75k" / "floor_scorecard_roster.csv"
        series_fn = lambda conn, row: _series_roster(conn, row["MRELG_ID"])
        load_sample = _load_roster_sample
        metric_name = "GLOBAL_STREAMS"
    else:
        arts_album = load_artifacts(str(AE_STREAMS_DIR))
        arts_singles = (
            load_artifacts(str(AE_SINGLES_STREAMS_DIR))
            if AE_SINGLES_STREAMS_DIR.is_dir()
            else arts_album
        )
        out_csv = REPO / "model" / "artifacts_75k" / "floor_scorecard_backfill.csv"
        series_fn = lambda conn, row: _series_backfill(conn, int(row["RELEASE_ID"]))
        load_sample = _load_backfill_sample
        metric_name = "STREAMING_EQUIVALENT"

    with sqlite3.connect(DB) as conn:
        sample = load_sample(conn)
        n_pain = int(
            sample["ARTIST"]
            .astype(str)
            .str.upper()
            .apply(lambda a: any(p in a for p in PAIN_SUBSTR))
            .sum()
        )
        print(
            f"Source={args.source} metric={metric_name} "
            f"Evaluating {len(sample)} titles ({n_pain} pain-case rows)..."
        )
        rows = []
        for _, row in sample.iterrows():
            y = series_fn(conn, row)
            while len(y) > 1 and y[-1] <= 0:
                y = y[:-1]
            if len(y) < MIN_WEEKS_JUMP or float(np.max(y)) <= 0:
                continue
            pt = str(row["PRODUCT_TYPE"] or "").lower()
            arts = arts_singles if pt == "single" else arts_album
            genre = row["GENRE"] if "GENRE" in row.index else None
            if genre is not None and (pd.isna(genre) or not str(genre).strip()):
                genre = None
            rows.append(_eval_one(row, y, arts, genre=genre if genre is None else str(genre)))
            if len(rows) % 10 == 0:
                print(f"  ...{len(rows)} done")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    jumps = df["boundary_jump"].dropna()
    print("\n=== Boundary jump (y_hat[K+1] / y[K]) ===")
    if jumps.empty:
        print("n=0")
    else:
        print(
            f"n={len(jumps)}  median={jumps.median():.3f}  p90={jumps.quantile(0.9):.3f}  "
            f"max={jumps.max():.3f}  share>1.05={(jumps > 1.05).mean():.1%}  "
            f"share>1.5={(jumps > 1.5).mean():.1%}  share>2={(jumps > 2).mean():.1%}"
        )

    m40 = df["mape_week40_plus"].dropna()
    print("\n=== Week-40+ MAPE (holdout) ===")
    if m40.empty:
        print("n=0 — not enough titles with 40+ observed weeks")
    else:
        print(f"n={len(m40)}  median={m40.median():.3f}  mean={m40.mean():.3f}")

    m4 = df["mape_late4_holdout"].dropna()
    print("\n=== Late-4 holdout MAPE (proxy, n_weeks>=16) ===")
    if m4.empty:
        print("n=0")
    else:
        print(f"n={len(m4)}  median={m4.median():.3f}  mean={m4.mean():.3f}")

    print("\n=== Pain-case spot check ===")
    pain = df[df["is_pain_case"]].sort_values(["artist", "product_type", "release_date"])
    cols = [
        "artist",
        "title",
        "product_type",
        "n_weeks",
        "y_k",
        "legacy_floor",
        "new_floor",
        "y_hat_k1",
        "boundary_jump",
        "mape_week40_plus",
        "mape_late4_holdout",
    ]
    if pain.empty:
        print("(none found in EXPECTED_RELEASES / sample)")
    else:
        with pd.option_context(
            "display.max_rows",
            50,
            "display.width",
            220,
            "display.float_format",
            lambda x: f"{x:,.3f}",
        ):
            print(pain[cols].to_string(index=False))

    print("\n=== Worst boundary jumps (top 10) ===")
    worst = (
        df.dropna(subset=["boundary_jump"]).sort_values("boundary_jump", ascending=False).head(10)
    )
    with pd.option_context("display.width", 220, "display.float_format", lambda x: f"{x:,.3f}"):
        print(
            worst[
                ["artist", "title", "n_weeks", "y_k", "new_floor", "boundary_jump", "mape_week40_plus"]
            ].to_string(index=False)
        )

    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
