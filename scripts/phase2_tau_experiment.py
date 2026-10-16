#!/usr/bin/env python3
"""
Phase 2 — τ experiment on the Phase 1 floor (no Snowflake rescan).

Replay dual-gate over (τΔ, τL, τL,AMG[, τΔ,AMG]) and score:
  rows/week, coverage_d_market, outgrown/held labeled weeks,
  Other residual, title-set stability.

Lock rule (roadmap): coarsest τ that clears outgrown + held, p90 ≲ 40,
coverage ≳ 70% on large-move weeks. Prefer higher τ on ties.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
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
    apply_gate,
    build_week_market,
    classify_driver,
    prep_mrelg,
)

MAX_ROWS_GUARD = 50
P90_TARGET = 40
COVERAGE_TARGET = 0.70
LARGE_MOVE_AE = 50_000  # |d_market_ae| "large week"

TAU_DELTA_GRID = (10, 15, 25, 40)
TAU_LEVEL_GRID = (50, 75, 100)
TAU_LEVEL_AMG_GRID = (25, 40, 50, 75, 100)
TAU_DELTA_AMG_GRID = (None, 10)  # None = same as τΔ


@dataclass
class Cell:
    tau_delta: int
    tau_level: int
    tau_level_amg: int
    tau_delta_amg: int
    median_n: float
    p90_n: float
    max_n: int
    frac_over_50: float
    coverage_large: float
    coverage_all: float
    residual_other_frac: float
    stability: float
    outgrown_clear: float
    held_clear: float
    other_up_clear: float
    labeled_score: float
    n_weeks: int
    viable: bool
    notes: str


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def pick_labeled_weeks(weeks: pd.DataFrame) -> pd.DataFrame:
    w = weeks.dropna(subset=["D_SHARE", "D_AMG_AE", "D_MARKET_AE"]).copy()
    w["DRIVER"] = w.apply(classify_driver, axis=1)
    picked: list[dict] = []

    def take(mask, n, reason):
        sub = w[mask].copy()
        if sub.empty:
            return
        sub = sub.reindex(sub["D_SHARE_BPS"].abs().sort_values(ascending=False).index)
        for _, row in sub.head(n).iterrows():
            picked.append({**row.to_dict(), "WHY_LABELED": reason})

    take(w["DRIVER"].eq("outgrown"), 4, "outgrown")
    take(w["DRIVER"].eq("held"), 3, "held")
    take(w["DRIVER"].eq("amg_up"), 2, "amg_up")
    take(w["DRIVER"].eq("amg_down"), 2, "amg_down")
    take(w["DRIVER"].eq("other_up"), 2, "other_up")
    take(w["DRIVER"].eq("market_up"), 1, "market_up")
    take(pd.Series(True, index=w.index), 2, "largest_|d_share|")
    quiet = w.reindex(w["D_SHARE_BPS"].abs().sort_values().index).head(2)
    for _, row in quiet.iterrows():
        picked.append({**row.to_dict(), "WHY_LABELED": "quiet"})

    out = pd.DataFrame(picked).drop_duplicates("WEEK_ENDING_DATE")
    return out.sort_values("WEEK_ENDING_DATE")


def score_labeled(gated: pd.DataFrame, labeled: pd.DataFrame) -> dict[str, float]:
    out_s, held_s, other_s, all_s = [], [], [], []
    per_week: list[dict] = []
    for _, w in labeled.iterrows():
        wk = w["WEEK_ENDING_DATE"]
        rows = gated[gated["WEEK_END_DATE"] == wk]
        driver = w["DRIVER"]
        n_amg_bal = int(((rows["IS_AMG"]) & (rows["GATE"].isin(["ballast", "both"]))).sum())
        n_amg_any = int(rows["IS_AMG"].sum()) if not rows.empty else 0
        n_non_amg_mover = int(
            ((~rows["IS_AMG"]) & (rows["GATE"].isin(["mover", "both"]))).sum()
        ) if not rows.empty else 0
        n_other_mover = int(
            ((~rows["IN_PANEL"]) & (rows["GATE"].isin(["mover", "both"]))).sum()
        ) if not rows.empty else 0
        n = int(len(rows))
        if driver == "outgrown":
            score = 2 if n_non_amg_mover >= 1 else (1 if n >= 1 else 0)
            out_s.append(score)
        elif driver == "held":
            score = 2 if n_amg_bal >= 1 else (1 if n_amg_any >= 1 else 0)
            held_s.append(score)
        elif driver == "other_up":
            score = 2 if n_other_mover >= 1 else (1 if n_non_amg_mover >= 1 else 0)
            other_s.append(score)
        else:
            score = 2 if n >= 1 else 0
        all_s.append(score)
        per_week.append(
            {
                "week": _iso(wk),
                "driver": driver,
                "why": w.get("WHY_LABELED"),
                "score": score,
                "n": n,
                "n_amg_ballast": n_amg_bal,
                "n_amg": n_amg_any,
                "n_non_amg_mover": n_non_amg_mover,
                "n_other_mover": n_other_mover,
                "d_share_bps": float(w["D_SHARE_BPS"]),
            }
        )

    def avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "outgrown_clear": avg(out_s),
        "held_clear": avg(held_s),
        "other_up_clear": avg(other_s),
        "labeled_score": avg(all_s),
        "per_week": per_week,
    }


def week_n(gated: pd.DataFrame, weeks: pd.DataFrame) -> pd.Series:
    n = gated.groupby("WEEK_END_DATE").size()
    return n.reindex(weeks["WEEK_ENDING_DATE"]).fillna(0)


def coverage_d_market(gated: pd.DataFrame, weeks: pd.DataFrame, *, large_only: bool) -> float:
    sums = gated.groupby("WEEK_END_DATE")["DELTA_AE"].sum()
    rows = []
    for _, w in weeks.dropna(subset=["D_MARKET_AE"]).iterrows():
        dm = w["D_MARKET_AE"]
        if pd.isna(dm) or abs(dm) < 1:
            continue
        if large_only and abs(dm) < LARGE_MOVE_AE:
            continue
        listed = float(sums.get(w["WEEK_ENDING_DATE"], 0.0))
        rows.append(min(1.0, abs(listed) / abs(dm)))
    return float(np.mean(rows)) if rows else 0.0


def residual_other(gated: pd.DataFrame, weeks: pd.DataFrame) -> float:
    """Median |d_other_ae − Σ gated Other ΔAE| / |d_other_ae| on weeks with |d_other| large."""
    other = gated[~gated["IN_PANEL"]].groupby("WEEK_END_DATE")["DELTA_AE"].sum()
    fracs = []
    for _, w in weeks.dropna(subset=["D_OTHER_AE"]).iterrows():
        d = w["D_OTHER_AE"]
        if pd.isna(d) or abs(d) < LARGE_MOVE_AE:
            continue
        listed = float(other.get(w["WEEK_ENDING_DATE"], 0.0))
        fracs.append(abs(d - listed) / abs(d))
    return float(np.median(fracs)) if fracs else 0.0


def stability(gated: pd.DataFrame, weeks: pd.DataFrame) -> float:
    """
    Mean Jaccard of MRELG sets on consecutive weeks where driver is unchanged
    and |Δshare| < 15 bps (the story didn't change).
    """
    w = weeks.dropna(subset=["D_SHARE", "DRIVER"]).sort_values("WEEK_ENDING_DATE")
    ids = {
        pd.Timestamp(wk).normalize(): set(g["MRELG_ID"].astype(str))
        for wk, g in gated.groupby("WEEK_END_DATE")
    }
    js_all, js_quiet = [], []
    prev = None
    for _, row in w.iterrows():
        k = pd.Timestamp(row["WEEK_ENDING_DATE"]).normalize()
        if prev is not None:
            a = ids.get(k, set())
            b = ids.get(pd.Timestamp(prev["WEEK_ENDING_DATE"]).normalize(), set())
            union = a | b
            if union:
                j = len(a & b) / len(union)
                js_all.append(j)
                same_driver = row["DRIVER"] == prev["DRIVER"]
                quiet = abs(row["D_SHARE_BPS"]) < 15 and abs(prev["D_SHARE_BPS"]) < 15
                if same_driver and quiet:
                    js_quiet.append(j)
        prev = row
    # Quiet same-driver pairs are rare in this window; report consecutive Jaccard.
    return float(np.mean(js_all)) if js_all else 0.0


def run_cell(
    prepped: pd.DataFrame,
    weeks: pd.DataFrame,
    labeled: pd.DataFrame,
    td: int,
    tl: int,
    tl_amg: int,
    td_amg: int | None,
) -> tuple[Cell, pd.DataFrame, dict]:
    td_amg_v = td if td_amg is None else td_amg
    gated = apply_gate(
        prepped,
        weeks,
        tau_delta_bps=td,
        tau_level_bps=tl,
        tau_level_amg_bps=tl_amg,
        tau_delta_amg_bps=td_amg_v,
    )
    n = week_n(gated, weeks)
    lab = score_labeled(gated, labeled)
    cov_l = coverage_d_market(gated, weeks, large_only=True)
    cov_a = coverage_d_market(gated, weeks, large_only=False)
    p90 = float(n.quantile(0.9)) if len(n) else 0.0
    med = float(n.median()) if len(n) else 0.0
    notes = []
    if p90 > MAX_ROWS_GUARD:
        notes.append("p90>50")
    elif p90 > P90_TARGET:
        notes.append("p90>40")
    if cov_l < COVERAGE_TARGET:
        notes.append("cov<70%")
    if lab["outgrown_clear"] < 2.0:
        notes.append("outgrown miss")
    if lab["held_clear"] < 2.0:
        notes.append("held miss")
    viable = (
        lab["outgrown_clear"] >= 2.0
        and lab["held_clear"] >= 2.0
        and p90 <= P90_TARGET
        and cov_l >= COVERAGE_TARGET
    )
    cell = Cell(
        tau_delta=td,
        tau_level=tl,
        tau_level_amg=tl_amg,
        tau_delta_amg=td_amg_v,
        median_n=med,
        p90_n=p90,
        max_n=int(n.max()) if len(n) else 0,
        frac_over_50=float((n > MAX_ROWS_GUARD).mean()) if len(n) else 0.0,
        coverage_large=cov_l,
        coverage_all=cov_a,
        residual_other_frac=residual_other(gated, weeks),
        stability=stability(gated, weeks),
        outgrown_clear=lab["outgrown_clear"],
        held_clear=lab["held_clear"],
        other_up_clear=lab["other_up_clear"],
        labeled_score=lab["labeled_score"],
        n_weeks=int(len(n)),
        viable=viable,
        notes="; ".join(notes),
    )
    return cell, gated, lab


def amg_ballast_sizes(prepped: pd.DataFrame, weeks: pd.DataFrame) -> dict:
    """Distribution of AMG title AE-share (bps) on held weeks — why τL,AMG exists."""
    held = weeks[weeks["DRIVER"].eq("held")]
    if held.empty:
        return {}
    df = prepped.merge(
        weeks.rename(columns={"WEEK_ENDING_DATE": "WEEK_END_DATE"})[
            ["WEEK_END_DATE", "MARKET_AE", "DRIVER"]
        ],
        on="WEEK_END_DATE",
        how="inner",
    )
    amg = df[(df["IS_AMG"]) & (df["DRIVER"] == "held") & (df["MARKET_AE"] > 0)].copy()
    amg["AE_SHARE_BPS"] = 10000.0 * amg["AE"] / amg["MARKET_AE"]
    top = (
        amg.sort_values(["WEEK_END_DATE", "AE_SHARE_BPS"], ascending=[True, False])
        .groupby("WEEK_END_DATE")
        .head(3)
    )
    return {
        "held_weeks": [_iso(x) for x in held["WEEK_ENDING_DATE"]],
        "amg_title_bps_p50": float(amg["AE_SHARE_BPS"].median()) if len(amg) else None,
        "amg_title_bps_p90": float(amg["AE_SHARE_BPS"].quantile(0.9)) if len(amg) else None,
        "max_amg_bps_by_week": {
            _iso(wk): float(g["AE_SHARE_BPS"].max())
            for wk, g in amg.groupby("WEEK_END_DATE")
        },
        "top3_by_week": [
            {
                "week": _iso(r["WEEK_END_DATE"]),
                "title": str(r["TITLE"])[:40],
                "artist": str(r["ARTIST"])[:28],
                "ae": float(r["AE"]),
                "bps": float(r["AE_SHARE_BPS"]),
                "l3": str(r["LEVEL_3_DISTRIBUTOR"]),
            }
            for _, r in top.iterrows()
        ],
    }


def title_snapshot(gated: pd.DataFrame, week, n: int = 8) -> list[dict]:
    rows = gated[gated["WEEK_END_DATE"] == week].head(n)
    out = []
    for _, r in rows.iterrows():
        tag = "AMG" if r["IS_AMG"] else ("PANEL" if r["IN_PANEL"] else "OTHER")
        out.append(
            {
                "gate": r["GATE"],
                "tag": tag,
                "d_bps": float(r["D_MARKET_BPS"]),
                "lvl_bps": float(r["AE_SHARE_BPS"]),
                "ae": float(r["AE"]),
                "artist": str(r["ARTIST"])[:28],
                "title": str(r["TITLE"])[:36],
                "owner": str(r["OWNER_LABEL"])[:28],
                "l3": str(r["LEVEL_3_DISTRIBUTOR"] or "")[:20],
                "is_new": bool(r["IS_NEW"]),
            }
        )
    return out


def fmt_ae(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if a >= 1_000:
        return f"{v/1_000:.1f}k"
    return f"{v:.0f}"


def print_week_card(week: pd.Series, titles: pd.DataFrame) -> None:
    wk = _iso(week["WEEK_ENDING_DATE"])
    print("\n" + "=" * 78)
    print(f"WEEK ENDING {wk}   driver={week['DRIVER']}   labeled_as={week.get('WHY_LABELED')}")
    print("-" * 78)
    print(
        f"AMG share  {week['PREV_AMG_SHARE']*100:.2f}% → {week['AMG_SHARE']*100:.2f}%  "
        f"({week['D_SHARE_BPS']:+.0f} bps)"
    )
    print(
        f"AMG AE     {fmt_ae(week['PREV_AMG_AE'])} → {fmt_ae(week['AMG_AE'])}  "
        f"({week['D_AMG_AE']:+,.0f})"
    )
    print(
        f"Market AE  {fmt_ae(week['PREV_MARKET_AE'])} → {fmt_ae(week['MARKET_AE'])}  "
        f"({week['D_MARKET_AE']:+,.0f})"
    )
    print(
        f"Decomp     own {week['OWN_EFFECT_BPS']:+.0f} bps   "
        f"market {week['MARKET_EFFECT_BPS']:+.0f} bps"
    )
    if titles.empty:
        print("No titles cleared the dual gate.")
        return
    print(f"{'gate':<7} {'tag':<5} {'Δbps':>6} {'lvl':>5}  {'AE':>8}  artist / title  [owner]")
    for _, r in titles.iterrows():
        tag = "AMG" if r["IS_AMG"] else ("PANEL" if r["IN_PANEL"] else "OTHER")
        new = " NEW" if r["IS_NEW"] else ""
        l3 = str(r["LEVEL_3_DISTRIBUTOR"] or "").strip()
        owner = r["OWNER_LABEL"]
        if l3 and l3 not in ("None", "nan", owner):
            owner = f"{owner}/{l3}"
        print(
            f"  {r['GATE']:<7} {tag:<5} {r['D_MARKET_BPS']:+6.0f} {r['AE_SHARE_BPS']:5.0f}  "
            f"{fmt_ae(r['AE']):>8}  {str(r['ARTIST'])[:18]:<18} {str(r['TITLE'])[:26]:<26}  "
            f"[{owner}]{new}"
        )


def main() -> int:
    panel = pd.read_csv(CURRENT_DATA_ALL_CSV)
    floor = pd.read_csv(MRELG_FLOOR_CSV)
    weeks = build_week_market(panel)
    weeks["DRIVER"] = weeks.apply(classify_driver, axis=1)
    prepped = prep_mrelg(floor)
    floor_weeks = pd.to_datetime(prepped["WEEK_END_DATE"]).dropna().sort_values().unique()
    if len(floor_weeks) > 1:
        floor_weeks = floor_weeks[1:]
    weeks = weeks[weeks["WEEK_ENDING_DATE"].isin(floor_weeks)].copy()
    labeled = pick_labeled_weeks(weeks)

    print("========== PHASE 2 τ EXPERIMENT ==========")
    print(f"Floor: {MRELG_FLOOR_CSV}")
    print(
        f"Weeks with lag: {len(weeks)}  "
        f"({_iso(weeks['WEEK_ENDING_DATE'].min())} → {_iso(weeks['WEEK_ENDING_DATE'].max())})"
    )
    print("Drivers:", weeks["DRIVER"].value_counts().to_dict())
    print("\nLabeled weeks:")
    for _, r in labeled.iterrows():
        print(
            f"  {_iso(r['WEEK_ENDING_DATE'])}  {r['DRIVER']:<10}  "
            f"d_share={r['D_SHARE_BPS']:+6.0f}bps  ({r['WHY_LABELED']})"
        )

    ballast = amg_ballast_sizes(prepped, weeks)
    print("\n========== AMG BALLAST ON HELD WEEKS ==========")
    print("held weeks:", ballast.get("held_weeks"))
    print("max AMG title bps by week:", ballast.get("max_amg_bps_by_week"))

    # Pass 1: (τΔ, τL) with τL,AMG = τL (Phase 0 equivalent)
    print("\n========== GRID A — τL,AMG = τL (no AMG exception) ==========")
    print(
        f"{'τΔ':>4} {'τL':>4} {'medN':>6} {'p90N':>6} {'max':>4} "
        f"{'covL':>5} {'resid':>5} {'stab':>5} {'outg':>5} {'held':>5} {'oth':>5} viable"
    )
    grid_a: list[Cell] = []
    for td in TAU_DELTA_GRID:
        for tl in TAU_LEVEL_GRID:
            cell, _, _ = run_cell(prepped, weeks, labeled, td, tl, tl, None)
            grid_a.append(cell)
            print(
                f"{td:4d} {tl:4d} {cell.median_n:6.1f} {cell.p90_n:6.1f} {cell.max_n:4d} "
                f"{cell.coverage_large:5.0%} {cell.residual_other_frac:5.0%} "
                f"{cell.stability:5.0%} {cell.outgrown_clear:5.1f} {cell.held_clear:5.1f} "
                f"{cell.other_up_clear:5.1f} {'Y' if cell.viable else 'n'} {cell.notes}"
            )

    # Pass 2: vary τL,AMG around the production candidate neighborhood
    print("\n========== GRID B — τL,AMG variants (τΔ × τL × τL,AMG) ==========")
    print(
        f"{'τΔ':>4} {'τL':>4} {'τLα':>4} {'medN':>6} {'p90N':>6} "
        f"{'covL':>5} {'held':>5} {'outg':>5} viable"
    )
    grid_b: list[Cell] = []
    for td in TAU_DELTA_GRID:
        for tl in TAU_LEVEL_GRID:
            for tl_a in TAU_LEVEL_AMG_GRID:
                if tl_a > tl:
                    continue  # AMG ballast should be looser, not tighter
                cell, _, _ = run_cell(prepped, weeks, labeled, td, tl, tl_a, None)
                grid_b.append(cell)
                mark = " <-- 25/75/40" if (td, tl, tl_a) == (25, 75, 40) else ""
                print(
                    f"{td:4d} {tl:4d} {tl_a:4d} {cell.median_n:6.1f} {cell.p90_n:6.1f} "
                    f"{cell.coverage_large:5.0%} {cell.held_clear:5.1f} {cell.outgrown_clear:5.1f} "
                    f"{'Y' if cell.viable else 'n'} {cell.notes}{mark}"
                )

    # Pass 3: looser AMG movers on the locked neighborhood
    print("\n========== GRID C — τΔ,AMG looser movers (locked neighborhood) ==========")
    grid_c: list[Cell] = []
    neighborhood = [(25, 75, 40), (15, 75, 40), (25, 50, 40), (40, 75, 40), (25, 75, 25)]
    for td, tl, tl_a in neighborhood:
        for td_a in TAU_DELTA_AMG_GRID:
            cell, _, _ = run_cell(prepped, weeks, labeled, td, tl, tl_a, td_a)
            grid_c.append(cell)
            print(
                f"τΔ={td} τL={tl} τL,AMG={tl_a} τΔ,AMG={cell.tau_delta_amg}  "
                f"med={cell.median_n:.0f} p90={cell.p90_n:.0f} covL={cell.coverage_large:.0%} "
                f"held={cell.held_clear:.1f} outg={cell.outgrown_clear:.1f} "
                f"resid={cell.residual_other_frac:.0%} stab={cell.stability:.0%} "
                f"{'VIABLE' if cell.viable else cell.notes}"
            )

    def rank_key(c: Cell):
        return (
            int(c.viable),
            c.outgrown_clear + c.held_clear,
            c.labeled_score,
            -c.p90_n,
            c.tau_delta,
            c.tau_level,
            c.tau_level_amg,
        )

    viable_b = [c for c in grid_b if c.viable]
    if viable_b:
        best = sorted(viable_b, key=lambda c: (c.tau_delta, c.tau_level, c.tau_level_amg), reverse=True)[0]
        reason = "coarsest viable (outgrown+held, p90≤40, covL≥70%)"
    else:
        best = sorted(grid_b, key=rank_key, reverse=True)[0]
        reason = "best labeled score; no cell fully viable"
    locked = next(c for c in grid_b if (c.tau_delta, c.tau_level, c.tau_level_amg) == (25, 75, 40))

    print("\n========== LOCK RECOMMENDATION ==========")
    print(
        f"Experiment coarsest viable: τΔ={best.tau_delta} τL={best.tau_level} "
        f"τL,AMG={best.tau_level_amg}  ({reason})"
    )
    print(
        f"Phase 1 locked 25/75/40: viable={locked.viable} medN={locked.median_n:.0f} "
        f"p90N={locked.p90_n:.0f} covL={locked.coverage_large:.0%} "
        f"held={locked.held_clear:.1f} outg={locked.outgrown_clear:.1f} "
        f"resid={locked.residual_other_frac:.0%} stab={locked.stability:.0%} {locked.notes}"
    )

    # Cards at Phase 1 lock
    print("\n========== WEEK CARDS AT PHASE 1 LOCK (25 / 75 / 40) ==========")
    _, gated_lock, lab_lock = run_cell(prepped, weeks, labeled, 25, 75, 40, None)
    labeled_cards = []
    for _, w in labeled.iterrows():
        titles = gated_lock[gated_lock["WEEK_END_DATE"] == w["WEEK_ENDING_DATE"]]
        print_week_card(w, titles)
        labeled_cards.append(
            {
                "week": _iso(w["WEEK_ENDING_DATE"]),
                "driver": w["DRIVER"],
                "why": w["WHY_LABELED"],
                "d_share_bps": float(w["D_SHARE_BPS"]),
                "d_amg_ae": float(w["D_AMG_AE"]),
                "d_market_ae": float(w["D_MARKET_AE"]),
                "d_other_ae": float(w["D_OTHER_AE"]),
                "amg_share": float(w["AMG_SHARE"]),
                "prev_amg_share": float(w["PREV_AMG_SHARE"]),
                "own_effect_bps": float(w["OWN_EFFECT_BPS"]),
                "market_effect_bps": float(w["MARKET_EFFECT_BPS"]),
                "n": int(len(titles)),
                "titles": title_snapshot(gated_lock, w["WEEK_ENDING_DATE"], n=10),
            }
        )

    # Coverage by |d_market| bin at lock
    sums = gated_lock.groupby("WEEK_END_DATE")["DELTA_AE"].sum()
    cov_rows = []
    for _, w in weeks.dropna(subset=["D_MARKET_AE"]).iterrows():
        dm = w["D_MARKET_AE"]
        if pd.isna(dm) or abs(dm) < 1:
            continue
        listed = float(sums.get(w["WEEK_ENDING_DATE"], 0.0))
        cov_rows.append(
            {
                "week": _iso(w["WEEK_ENDING_DATE"]),
                "driver": w["DRIVER"],
                "d_market_ae": float(dm),
                "d_share_bps": float(w["D_SHARE_BPS"]),
                "n": int((gated_lock["WEEK_END_DATE"] == w["WEEK_ENDING_DATE"]).sum()),
                "coverage": min(1.0, abs(listed) / abs(dm)),
            }
        )

    n_by_week = week_n(gated_lock, weeks)
    payload = {
        "window": {
            "start": _iso(weeks["WEEK_ENDING_DATE"].min()),
            "end": _iso(weeks["WEEK_ENDING_DATE"].max()),
            "n_weeks": int(len(weeks)),
            "drivers": weeks["DRIVER"].value_counts().to_dict(),
        },
        "labeled": [
            {
                "week": _iso(r["WEEK_ENDING_DATE"]),
                "driver": r["DRIVER"],
                "why": r["WHY_LABELED"],
                "d_share_bps": float(r["D_SHARE_BPS"]),
            }
            for _, r in labeled.iterrows()
        ],
        "ballast": ballast,
        "grid_a": [asdict(c) for c in grid_a],
        "grid_b": [asdict(c) for c in grid_b],
        "grid_c": [asdict(c) for c in grid_c],
        "locked_25_75_40": asdict(locked),
        "experiment_best": asdict(best),
        "labeled_scores_at_lock": lab_lock,
        "labeled_cards": labeled_cards,
        "coverage_weeks_at_lock": cov_rows,
        "n_by_week_at_lock": {
            _iso(k): int(v) for k, v in n_by_week.items() if pd.notna(k)
        },
        "lock_n": {
            "median": float(n_by_week.median()),
            "p90": float(n_by_week.quantile(0.9)),
            "min": int(n_by_week.min()),
            "max": int(n_by_week.max()),
        },
    }
    out_path = Path("/tmp/attribution_phase2.json")
    out_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
