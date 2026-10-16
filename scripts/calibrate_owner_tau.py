#!/usr/bin/env python3
"""Per-owner τL grid on the Phase 1 floor. Shared τΔ=25. No Snowflake."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from change_attribution import (  # noqa: E402
    AMG,
    CURRENT_DATA_ALL_CSV,
    MRELG_FLOOR_CSV,
    PANEL,
    TARGET_LABELS,
    apply_gate,
    build_week_market,
    classify_driver,
    prep_mrelg,
)

GRID = (75, 50, 40, 25)


def owner_decomp(panel: pd.DataFrame, market_weeks: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["WEEK_ENDING_DATE"] = pd.to_datetime(df["WEEK_ENDING_DATE"])
    df["AE"] = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    share = pd.to_numeric(df["ALBUM_EQUIVALENT_SHARE"], errors="coerce")
    df["SHARE"] = np.where(share > 1, share / 100.0, share)
    df["LABEL_NAME"] = df["LABEL_NAME"].astype(str).str.strip()
    mkt = market_weeks[
        ["WEEK_ENDING_DATE", "MARKET_AE", "D_MARKET_AE", "D_PANEL_AE", "D_OTHER_AE"]
    ]
    rows = []
    for lab in TARGET_LABELS:
        sub = df[df["LABEL_NAME"] == lab][
            ["WEEK_ENDING_DATE", "AE", "SHARE"]
        ].drop_duplicates("WEEK_ENDING_DATE")
        sub = sub.merge(mkt, on="WEEK_ENDING_DATE", how="inner")
        sub = sub.sort_values("WEEK_ENDING_DATE")
        sub["OWNER_LABEL"] = lab
        sub["PREV_AE"] = sub["AE"].shift(1)
        sub["PREV_SHARE"] = sub["SHARE"].shift(1)
        sub["D_AE"] = sub["AE"] - sub["PREV_AE"]
        sub["D_SHARE"] = sub["SHARE"] - sub["PREV_SHARE"]
        sub["D_SHARE_BPS"] = sub["D_SHARE"] * 10000.0
        sub["OWN_EFFECT_BPS"] = 10000.0 * sub["D_AE"] / sub["MARKET_AE"]
        sub["MARKET_EFFECT_BPS"] = (
            -10000.0 * sub["PREV_SHARE"] * sub["D_MARKET_AE"] / sub["MARKET_AE"]
        )
        fake = sub.rename(
            columns={
                "AE": "AMG_AE",
                "PREV_AE": "PREV_AMG_AE",
                "D_AE": "D_AMG_AE",
                "SHARE": "AMG_SHARE",
            }
        )
        fake["DRIVER"] = fake.apply(classify_driver, axis=1)
        sub["DRIVER"] = fake["DRIVER"]
        rows.append(sub.dropna(subset=["D_SHARE"]))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    panel = pd.read_csv(CURRENT_DATA_ALL_CSV)
    weeks = build_week_market(panel)
    prepped = prep_mrelg(pd.read_csv(MRELG_FLOOR_CSV))
    fw = pd.to_datetime(prepped["WEEK_END_DATE"]).dropna().sort_values().unique()
    if len(fw) > 1:
        fw = fw[1:]
    weeks = weeks[weeks["WEEK_ENDING_DATE"].isin(fw)].copy()
    prepped = prepped[prepped["WEEK_END_DATE"].isin(fw)].copy()
    owners = owner_decomp(panel, weeks)

    print("τΔ=25  GRID τL", GRID)
    print(
        f"{'owner':<22} {'τL':>4} {'held':>5} {'hN':>4} "
        f"{'med_own_anc':>12} {'p90_own_anc':>12} {'medN':>6} lock?"
    )
    locks: dict[str, int] = {}
    for lab in TARGET_LABELS:
        held_w = owners[(owners["OWNER_LABEL"] == lab) & (owners["DRIVER"] == "held")]
        held_dates = set(held_w["WEEK_ENDING_DATE"])
        n_held = len(held_dates)
        picked = None
        for tl in GRID:
            mapping = {lab: tl}
            if lab != AMG:
                mapping[AMG] = 40
            gated = apply_gate(prepped, weeks, tau_level_by_owner=mapping)
            own = gated[gated["OWNER_LABEL"] == lab]
            n_anc = (
                own[own["GATE"].isin(["anchor", "both"])]
                .groupby("WEEK_END_DATE")
                .size()
            )
            n_anc = n_anc.reindex(weeks["WEEK_ENDING_DATE"]).fillna(0)
            held_ok = 0
            max_title = []
            for wk in held_dates:
                sub = own[own["WEEK_END_DATE"] == wk]
                anc = sub[sub["GATE"].isin(["anchor", "both"])]
                if len(anc) >= 1:
                    held_ok += 1
                mx = float(sub["AE_SHARE_BPS"].max()) if len(sub) else 0.0
                max_title.append(mx)
            held_score = held_ok / n_held if n_held else 1.0
            n_week = gated.groupby("WEEK_END_DATE").size()
            med_n = float(n_week.median()) if len(n_week) else 0
            flag = ""
            if held_score >= 1.0 and picked is None:
                picked = tl
                flag = " <-- lock (coarsest that clears all held)"
            print(
                f"{lab:<22} {tl:4d} {held_score:5.2f} {n_held:4d} "
                f"{float(n_anc.median()):12.1f} {float(n_anc.quantile(0.9)):12.1f} "
                f"{med_n:6.1f}{flag}"
            )
        locks[lab] = picked if picked is not None else 25
        if n_held:
            print(
                f"  held weeks: "
                + ", ".join(
                    f"{pd.Timestamp(r.WEEK_ENDING_DATE).date()} "
                    f"d_share={r.D_SHARE_BPS:+.0f}"
                    for r in held_w.itertuples()
                )
            )
    print("\nLOCKED TAU_LEVEL_BY_OWNER =")
    print(locks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
