#!/usr/bin/env python3
"""Phase 0 + τ grid for AMG weekly share attribution (dual gate, all owners)."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.marketshare_labels import MARKET_ANCHOR_LABEL, TARGET_LABELS  # noqa: E402
from snowflake_conn import get_snowflake_connection, load_sql  # noqa: E402

CACHE_DIR = Path("/tmp/attribution_phase0")
WEEKS = 40
MIN_AE = 2500
MAX_ROWS_GUARD = 50
PANEL = set(TARGET_LABELS)
AMG = MARKET_ANCHOR_LABEL

TAU_DELTA_GRID = (10, 15, 25, 40)
TAU_LEVEL_GRID = (50, 75, 100)


UNIVERSE_LABELS = {
    "Total Universe",
    "Total Universe (Excl Under Review)",
}
CURRENT_AGE_MONTHS = 18


def _share_to_frac(series: pd.Series) -> pd.Series:
    """marketshare_weekly ALBUM_EQUIVALENT_SHARE is percent (8.36 = 8.36%)."""
    s = pd.to_numeric(series, errors="coerce")
    out = s.copy()
    bps = s > 100
    pct = (s > 1) & ~bps
    out = np.where(bps, s / 10000.0, np.where(pct, s / 100.0, s))
    return pd.Series(out, index=s.index)


def _week_start(min_weeks: int) -> str:
    return (date.today() - timedelta(days=7 * (min_weeks + 2))).isoformat()


def _load_sql(name: str, **repl) -> str:
    sql = load_sql(name)
    for k, v in repl.items():
        sql = sql.replace("{" + k + "}", str(v))
    return sql


def fetch_snowflake() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = CACHE_DIR / "panel.parquet"
    mrelg_path = CACHE_DIR / "mrelg.parquet"
    confirm_path = CACHE_DIR / "confirm.parquet"
    icpn_path = CACHE_DIR / "icpn.parquet"
    min_week = _week_start(WEEKS)

    with get_snowflake_connection() as sf:
        if not confirm_path.is_file():
            print("Snowflake: phase0 confirm…", flush=True)
            confirm = sf.query(load_sql("query_attribution_phase0_confirm.sql"))
            confirm.to_parquet(confirm_path, index=False)
        else:
            confirm = pd.read_parquet(confirm_path)

        if not icpn_path.is_file():
            print("Snowflake: ICPN CASE…", flush=True)
            icpn = sf.query(load_sql("query_attribution_icpn_case.sql"))
            icpn.to_parquet(icpn_path, index=False)
        else:
            icpn = pd.read_parquet(icpn_path)

        if not panel_path.is_file():
            print(f"Snowflake: panel weeks >= {min_week}…", flush=True)
            panel = sf.query(
                _load_sql("query_attribution_panel_weekly.sql", MIN_WEEK_END_DATE=min_week)
            )
            panel.to_parquet(panel_path, index=False)
        else:
            panel = pd.read_parquet(panel_path)

        if not mrelg_path.is_file():
            print(
                f"Snowflake: MRELG weekly AE >= {MIN_AE}, weeks >= {min_week}…",
                flush=True,
            )
            mrelg = sf.query(
                _load_sql(
                    "query_attribution_mrelg_weekly.sql",
                    MIN_WEEK_END_DATE=min_week,
                    MIN_AE=MIN_AE,
                )
            )
            mrelg.to_parquet(mrelg_path, index=False)
        else:
            mrelg = pd.read_parquet(mrelg_path)

    return confirm, icpn, panel, mrelg


def build_week_market(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["WEEK_ENDING_DATE"] = pd.to_datetime(df["WEEK_ENDING_DATE"], errors="coerce")
    df["ALBUM_EQUIVALENT"] = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    df["ALBUM_EQUIVALENT_SHARE"] = _share_to_frac(df["ALBUM_EQUIVALENT_SHARE"])
    df["LABEL_NAME"] = df["LABEL_NAME"].astype(str).str.strip()
    df = df[~df["LABEL_NAME"].isin(UNIVERSE_LABELS)].copy()

    amg = df[df["LABEL_NAME"] == AMG].copy()
    amg["MARKET_AE"] = np.where(
        amg["ALBUM_EQUIVALENT_SHARE"] > 0,
        amg["ALBUM_EQUIVALENT"] / amg["ALBUM_EQUIVALENT_SHARE"],
        np.nan,
    )
    market = amg[
        ["WEEK_ENDING_DATE", "ALBUM_EQUIVALENT", "ALBUM_EQUIVALENT_SHARE", "MARKET_AE"]
    ].rename(
        columns={
            "ALBUM_EQUIVALENT": "AMG_AE",
            "ALBUM_EQUIVALENT_SHARE": "AMG_SHARE",
        }
    )
    market = market.sort_values("WEEK_ENDING_DATE").drop_duplicates(
        "WEEK_ENDING_DATE", keep="last"
    )

    panel_sum = (
        df[df["LABEL_NAME"].isin(PANEL)]
        .groupby("WEEK_ENDING_DATE", as_index=False)["ALBUM_EQUIVALENT"]
        .sum()
        .rename(columns={"ALBUM_EQUIVALENT": "PANEL_AE"})
    )
    all_sum = (
        df.groupby("WEEK_ENDING_DATE", as_index=False)["ALBUM_EQUIVALENT"]
        .sum()
        .rename(columns={"ALBUM_EQUIVALENT": "WEEKLY_TABLE_AE"})
    )
    out = market.merge(panel_sum, on="WEEK_ENDING_DATE", how="left")
    out = out.merge(all_sum, on="WEEK_ENDING_DATE", how="left")
    out["OTHER_AE"] = out["MARKET_AE"] - out["PANEL_AE"]
    out = out.sort_values("WEEK_ENDING_DATE").reset_index(drop=True)
    out["PREV_AMG_AE"] = out["AMG_AE"].shift(1)
    out["PREV_MARKET_AE"] = out["MARKET_AE"].shift(1)
    out["PREV_AMG_SHARE"] = out["AMG_SHARE"].shift(1)
    out["PREV_PANEL_AE"] = out["PANEL_AE"].shift(1)
    out["PREV_OTHER_AE"] = out["OTHER_AE"].shift(1)
    out["D_AMG_AE"] = out["AMG_AE"] - out["PREV_AMG_AE"]
    out["D_MARKET_AE"] = out["MARKET_AE"] - out["PREV_MARKET_AE"]
    out["D_PANEL_AE"] = out["PANEL_AE"] - out["PREV_PANEL_AE"]
    out["D_OTHER_AE"] = out["OTHER_AE"] - out["PREV_OTHER_AE"]
    out["D_SHARE"] = out["AMG_SHARE"] - out["PREV_AMG_SHARE"]
    out["D_SHARE_BPS"] = out["D_SHARE"] * 10000
    out["OWN_EFFECT_BPS"] = 10000 * out["D_AMG_AE"] / out["MARKET_AE"]
    out["MARKET_EFFECT_BPS"] = (
        -10000 * out["PREV_AMG_SHARE"] * out["D_MARKET_AE"] / out["MARKET_AE"]
    )
    return out


def classify_driver(row: pd.Series) -> str:
    d_share = row["D_SHARE"]
    d_amg = row["D_AMG_AE"]
    d_mkt = row["D_MARKET_AE"]
    d_other = row["D_OTHER_AE"]
    d_panel = row["D_PANEL_AE"]
    if pd.isna(d_share) or pd.isna(d_amg) or pd.isna(d_mkt):
        return "na"
    if d_amg > 0 and d_share < 0:
        return "outgrown"
    amg_pct = abs(d_amg) / max(abs(row["PREV_AMG_AE"] or 0), 1.0)
    if amg_pct < 0.02 and d_mkt < 0 and d_share > 0:
        return "held"
    if d_share < 0 and pd.notna(d_other) and d_other > max(float(d_panel or 0) - float(d_amg or 0), 0):
        return "other_up"
    if d_share > 0 and d_amg > 0 and (row["OWN_EFFECT_BPS"] or 0) >= abs(row["MARKET_EFFECT_BPS"] or 0):
        return "amg_up"
    if d_share < 0 and d_amg < 0 and abs(row["OWN_EFFECT_BPS"] or 0) >= abs(row["MARKET_EFFECT_BPS"] or 0):
        return "amg_down"
    if d_share < 0 and d_mkt > 0:
        return "market_up"
    if d_share > 0 and d_mkt < 0:
        return "market_down"
    return "mix"


def prep_mrelg(mrelg: pd.DataFrame) -> pd.DataFrame:
    df = mrelg.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"], errors="coerce")
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")
    df["AE"] = pd.to_numeric(df["AE"], errors="coerce")
    df["OWNER_LABEL"] = df["OWNER_LABEL"].fillna("Unknown").astype(str).str.strip()
    df["TITLE"] = df["TITLE"].fillna("").astype(str)
    df["ARTIST"] = df["ARTIST"].fillna("").astype(str)
    df["LEVEL_3_DISTRIBUTOR"] = df["LEVEL_3_DISTRIBUTOR"].fillna("").astype(str)
    df["IS_AMG"] = df["OWNER_LABEL"].eq(AMG)
    df["IN_PANEL"] = df["OWNER_LABEL"].isin(PANEL)
    prev = df[["WEEK_END_DATE", "MRELG_ID", "AE"]].rename(
        columns={"WEEK_END_DATE": "PREV_WEEK", "AE": "PREV_AE"}
    )
    df["PREV_WEEK"] = df["WEEK_END_DATE"] - pd.Timedelta(days=7)
    df = df.merge(prev, on=["PREV_WEEK", "MRELG_ID"], how="left")
    df["IS_NEW"] = (
        df["FIRST_SALE_DATE"].notna()
        & (df["FIRST_SALE_DATE"] > df["WEEK_END_DATE"] - pd.Timedelta(days=7))
        & (df["FIRST_SALE_DATE"] <= df["WEEK_END_DATE"])
    )
    # Home-page panel is release_age=Current (~18 months). Catalog AE does not
    # belong in the Current pie; keep it out of the dual gate.
    current_cut = df["WEEK_END_DATE"] - pd.DateOffset(months=CURRENT_AGE_MONTHS)
    df = df[df["FIRST_SALE_DATE"].isna() | (df["FIRST_SALE_DATE"] >= current_cut)].copy()
    df["DELTA_AE"] = np.where(
        df["IS_NEW"] | df["PREV_AE"].isna(),
        df["AE"],
        df["AE"] - df["PREV_AE"],
    )
    return df


def apply_gate(
    mrelg: pd.DataFrame,
    weeks: pd.DataFrame,
    tau_delta_bps: float,
    tau_level_bps: float,
    tau_delta_amg_bps: float | None = None,
) -> pd.DataFrame:
    tau_d = tau_delta_bps / 10000.0
    tau_l = tau_level_bps / 10000.0
    tau_da = (tau_delta_amg_bps if tau_delta_amg_bps is not None else tau_delta_bps) / 10000.0
    w = weeks.rename(columns={"WEEK_ENDING_DATE": "WEEK_END_DATE"})
    df = mrelg.merge(
        w[["WEEK_END_DATE", "MARKET_AE"]],
        on="WEEK_END_DATE",
        how="inner",
    )
    df = df[df["MARKET_AE"] > 0].copy()
    df["AE_SHARE"] = df["AE"] / df["MARKET_AE"]
    df["D_MARKET_SHARE"] = df["DELTA_AE"] / df["MARKET_AE"]
    mover = df["D_MARKET_SHARE"].abs() >= np.where(df["IS_AMG"], tau_da, tau_d)
    ballast = df["AE_SHARE"] >= tau_l
    df["GATE"] = np.where(
        mover & ballast,
        "both",
        np.where(mover, "mover", np.where(ballast, "ballast", "")),
    )
    df = df[df["GATE"] != ""].copy()
    df["AE_SHARE_BPS"] = df["AE_SHARE"] * 10000
    df["D_MARKET_BPS"] = df["D_MARKET_SHARE"] * 10000
    df["SORT_KEY"] = np.maximum(df["AE_SHARE_BPS"].abs(), df["D_MARKET_BPS"].abs())
    df = df.sort_values(["WEEK_END_DATE", "SORT_KEY"], ascending=[True, False])
    df["RANK"] = df.groupby("WEEK_END_DATE").cumcount() + 1
    df["CAPPED"] = df["RANK"] > MAX_ROWS_GUARD
    return df


@dataclass
class TauScore:
    tau_delta: int
    tau_level: int
    median_n: float
    p90_n: float
    frac_over_cap: float
    coverage_d_market: float
    outgrown_clear: float
    held_clear: float
    labeled_score: float
    n_weeks: int


def week_n(gated: pd.DataFrame, weeks: pd.DataFrame) -> pd.Series:
    n = gated[~gated["CAPPED"]].groupby("WEEK_END_DATE").size()
    return n.reindex(weeks["WEEK_ENDING_DATE"]).fillna(0)


def coverage_d_market(gated: pd.DataFrame, weeks: pd.DataFrame) -> float:
    g = gated[~gated["CAPPED"]]
    sums = g.groupby("WEEK_END_DATE")["DELTA_AE"].sum()
    rows = []
    for _, w in weeks.dropna(subset=["D_MARKET_AE"]).iterrows():
        dm = w["D_MARKET_AE"]
        if pd.isna(dm) or abs(dm) < 1:
            continue
        listed = float(sums.get(w["WEEK_ENDING_DATE"], 0.0))
        rows.append(min(1.0, abs(listed) / abs(dm)))
    return float(np.mean(rows)) if rows else 0.0


def pick_labeled_weeks(weeks: pd.DataFrame) -> pd.DataFrame:
    w = weeks.dropna(subset=["D_SHARE", "D_AMG_AE", "D_MARKET_AE"]).copy()
    w["DRIVER"] = w.apply(classify_driver, axis=1)
    picked = []

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
    take(pd.Series(True, index=w.index), 3, "largest_|d_share|")
    quiet = w.reindex(w["D_SHARE_BPS"].abs().sort_values().index).head(2)
    for _, row in quiet.iterrows():
        picked.append({**row.to_dict(), "WHY_LABELED": "quiet"})

    out = pd.DataFrame(picked)
    out = out.drop_duplicates("WEEK_ENDING_DATE")
    return out.sort_values("WEEK_ENDING_DATE")


def score_labeled(gated: pd.DataFrame, labeled: pd.DataFrame) -> tuple[float, float, float]:
    g = gated[~gated["CAPPED"]]
    out_scores = []
    held_scores = []
    all_scores = []
    for _, w in labeled.iterrows():
        wk = w["WEEK_ENDING_DATE"]
        rows = g[g["WEEK_END_DATE"] == wk]
        driver = w["DRIVER"]
        score = 0
        if driver == "outgrown":
            non_amg = rows[(~rows["IS_AMG"]) & (rows["GATE"].isin(["mover", "both"]))]
            score = 2 if len(non_amg) >= 1 else 0
            out_scores.append(score)
        elif driver == "held":
            bal = rows[rows["IS_AMG"] & (rows["GATE"].isin(["ballast", "both"]))]
            score = 2 if len(bal) >= 1 else 0
            held_scores.append(score)
        else:
            score = 2 if len(rows) >= 1 else 0
        all_scores.append(score)
    def avg(xs):
        return float(np.mean(xs)) if xs else 0.0
    return avg(out_scores), avg(held_scores), avg(all_scores)


def fmt_ae(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    a = abs(v)
    if a >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if a >= 1_000:
        return f"{v/1_000:.1f}k"
    return f"{v:.0f}"


def fmt_pct(frac: float) -> str:
    if pd.isna(frac):
        return "n/a"
    return f"{frac * 100:.2f}%"


def print_confirm(confirm: pd.DataFrame, icpn: pd.DataFrame, weeks: pd.DataFrame, panel: pd.DataFrame) -> None:
    print("\n========== PHASE 0 CONFIRMATIONS ==========")
    p = panel.copy()
    p.columns = [str(x).strip().upper() for x in p.columns]
    p["WEEK_ENDING_DATE"] = pd.to_datetime(p["WEEK_ENDING_DATE"], errors="coerce")
    p["ALBUM_EQUIVALENT"] = pd.to_numeric(p["ALBUM_EQUIVALENT"], errors="coerce")
    p["ALBUM_EQUIVALENT_SHARE"] = _share_to_frac(p["ALBUM_EQUIVALENT_SHARE"])
    p["LABEL_NAME"] = p["LABEL_NAME"].astype(str).str.strip()
    latest = p["WEEK_ENDING_DATE"].max()
    week_rows = p[p["WEEK_ENDING_DATE"] == latest]
    n_labels = p.loc[
        p["WEEK_ENDING_DATE"] >= latest - pd.Timedelta(days=56), "LABEL_NAME"
    ].nunique()
    print(f"Distinct labels in marketshare_weekly (last 8w Current US): {n_labels}")
    print(f"Latest weekly table week: {latest.date()}")
    print(f"Labels in that week (table): {week_rows['LABEL_NAME'].nunique()}")
    print("  label                         AE          share     implied market")
    show = week_rows[~week_rows["LABEL_NAME"].isin(UNIVERSE_LABELS)]
    show = show.sort_values("ALBUM_EQUIVALENT", ascending=False).head(25)
    for _, r in show.iterrows():
        share = r["ALBUM_EQUIVALENT_SHARE"]
        implied = r["ALBUM_EQUIVALENT"] / share if share else math.nan
        marker = " ← PANEL" if str(r["LABEL_NAME"]).strip() in PANEL else ""
        print(
            f"  {str(r['LABEL_NAME'])[:28]:<28} {fmt_ae(r['ALBUM_EQUIVALENT']):>8}  "
            f"{fmt_pct(share):>8}  {fmt_ae(implied):>10}{marker}"
        )
    univ = week_rows[week_rows["LABEL_NAME"].isin(UNIVERSE_LABELS)]
    if not univ.empty:
        print("\nUniverse rollup rows (denominator, not an owner):")
        for _, r in univ.iterrows():
            print(f"  {r['LABEL_NAME']}: {fmt_ae(r['ALBUM_EQUIVALENT'])}")
    last = weeks.dropna(subset=["MARKET_AE"]).iloc[-1]
    print(f"\nAMG-inverted market (latest): {fmt_ae(last['MARKET_AE'])}")
    print(
        f"Sum of eight TARGET_LABELS AE: {fmt_ae(last['PANEL_AE'])} "
        f"({last['PANEL_AE']/last['MARKET_AE']*100:.1f}% of inverted market)"
    )
    print(
        f"Other (inverted − eight): {fmt_ae(last['OTHER_AE'])} "
        f"({last['OTHER_AE']/last['MARKET_AE']*100:.1f}%)"
    )
    print(
        "AMG row is a parent of Atlantic Records + 10K + 300 in this table; "
        "do not add those three on top of AMG. Share is % of Total Universe Current."
    )

    print("\nICPN owner CASE (Atlantic / 10K / 300):")
    ic = icpn.copy()
    ic.columns = [str(x).strip().upper() for x in ic.columns]
    for _, r in ic.iterrows():
        print(
            f"  owner={str(r.get('OWNER_LABEL','')):<22} "
            f"l2={str(r.get('LEVEL_2_DISTRIBUTOR','')):<24} "
            f"l3={str(r.get('LEVEL_3_DISTRIBUTOR','')):<24} n={r.get('N_ICPNS')}"
        )

    usable = weeks.dropna(subset=["D_SHARE"]).copy()
    print(f"\nHistory weeks with lag: {len(usable)}  ({usable['WEEK_ENDING_DATE'].min().date()} → {usable['WEEK_ENDING_DATE'].max().date()})")
    print(f"Median other_ae / market: {(usable['OTHER_AE']/usable['MARKET_AE']).median()*100:.1f}%")


def print_week_card(week: pd.Series, titles: pd.DataFrame) -> None:
    wk = pd.Timestamp(week["WEEK_ENDING_DATE"]).date()
    driver = week["DRIVER"]
    why = week.get("WHY_LABELED", "")
    print("\n" + "=" * 78)
    print(f"WEEK ENDING {wk}   driver={driver}   labeled_as={why}")
    print("-" * 78)
    print(
        f"AMG share  {fmt_pct(week['PREV_AMG_SHARE'])} → {fmt_pct(week['AMG_SHARE'])}  "
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
        f"Panel AE   {fmt_ae(week['PANEL_AE'])}   Other AE {fmt_ae(week['OTHER_AE'])}  "
        f"(Δ other {week['D_OTHER_AE']:+,.0f})"
    )
    print(
        f"Decomp     own {week['OWN_EFFECT_BPS']:+.0f} bps   "
        f"market {week['MARKET_EFFECT_BPS']:+.0f} bps"
    )
    if driver == "outgrown":
        print("READ: AMG volume is UP but share is DOWN — the pie grew faster (name non-AMG movers).")
    elif driver == "held":
        print("READ: AMG volume is ~flat, market faded — share UP because ballast held (name AMG ballast).")

    show = titles[~titles["CAPPED"]].copy()
    if show.empty:
        print("\nNo titles cleared the dual gate this week.")
        return

    def line(r):
        tag = "AMG" if r["IS_AMG"] else ("PANEL" if r["IN_PANEL"] else "OTHER")
        l3 = str(r["LEVEL_3_DISTRIBUTOR"] or "").strip()
        owner = r["OWNER_LABEL"]
        if l3 and l3 not in ("None", owner):
            owner = f"{owner} / {l3}"
        new = " NEW" if r["IS_NEW"] else ""
        return (
            f"  {r['GATE']:<7} {tag:<5} {r['D_MARKET_BPS']:+6.0f}Δbps  "
            f"{r['AE_SHARE_BPS']:5.0f} lvl  {fmt_ae(r['AE']):>8}  "
            f"{r['ARTIST'][:22]:<22}  {r['TITLE'][:28]:<28}  [{owner}]{new}"
        )

    movers = show[show["GATE"].isin(["mover", "both"])]
    ballast = show[show["GATE"].isin(["ballast", "both"])]
    print("\nMOVERS (|ΔAE|/market ≥ τ_Δ) — titles that changed the pie:")
    if movers.empty:
        print("  (none)")
    else:
        for _, r in movers.iterrows():
            print(line(r))
    print("\nBALLAST (AE/market ≥ τ_L) — titles large enough that holding still matters:")
    if ballast.empty:
        print("  (none)")
    else:
        for _, r in ballast.iterrows():
            print(line(r))


def main() -> int:
    confirm, icpn, panel, mrelg = fetch_snowflake()
    weeks = build_week_market(panel)
    weeks["DRIVER"] = weeks.apply(classify_driver, axis=1)
    mrelg_p = prep_mrelg(mrelg)

    print_confirm(confirm, icpn, weeks, panel)

    usable = weeks.dropna(subset=["D_SHARE", "MARKET_AE"]).copy()
    labeled = pick_labeled_weeks(usable)
    print("\n========== LABELED WEEKS FOR τ SEARCH ==========")
    for _, r in labeled.iterrows():
        print(
            f"  {pd.Timestamp(r['WEEK_ENDING_DATE']).date()}  {r['DRIVER']:<10}  "
            f"d_share={r['D_SHARE_BPS']:+6.0f}bps  d_amg={r['D_AMG_AE']:+.0f}  "
            f"d_mkt={r['D_MARKET_AE']:+.0f}  ({r['WHY_LABELED']})"
        )

    print("\n========== τ GRID (dual gate, all owners) ==========")
    print(
        f"{'τΔ':>4} {'τL':>4} {'medN':>6} {'p90N':>6} {'>50':>6} "
        f"{'covΔM':>6} {'outg':>5} {'held':>5} {'lab':>5}"
    )
    scores: list[TauScore] = []
    gated_by: dict[tuple[int, int], pd.DataFrame] = {}
    for td in TAU_DELTA_GRID:
        for tl in TAU_LEVEL_GRID:
            gated = apply_gate(mrelg_p, usable, td, tl)
            gated_by[(td, tl)] = gated
            n = week_n(gated, usable)
            out_s, held_s, lab_s = score_labeled(gated, labeled)
            sc = TauScore(
                tau_delta=td,
                tau_level=tl,
                median_n=float(n.median()) if len(n) else 0,
                p90_n=float(n.quantile(0.9)) if len(n) else 0,
                frac_over_cap=float((n > MAX_ROWS_GUARD).mean()) if len(n) else 0,
                coverage_d_market=coverage_d_market(gated, usable),
                outgrown_clear=out_s,
                held_clear=held_s,
                labeled_score=lab_s,
                n_weeks=int(len(n)),
            )
            scores.append(sc)
            print(
                f"{td:4d} {tl:4d} {sc.median_n:6.1f} {sc.p90_n:6.1f} {sc.frac_over_cap:6.0%} "
                f"{sc.coverage_d_market:6.0%} {sc.outgrown_clear:5.1f} {sc.held_clear:5.1f} "
                f"{sc.labeled_score:5.1f}"
            )

    # Coarsest τ that still clears outgrown + held (score 2.0), else best labeled score
    # with p90 rows under guard. Prefer higher τ on ties.
    def ok(s: TauScore) -> bool:
        return s.outgrown_clear >= 2.0 and s.held_clear >= 2.0 and s.p90_n <= MAX_ROWS_GUARD + 10

    viable = [s for s in scores if ok(s)]
    if viable:
        best = sorted(viable, key=lambda s: (s.tau_delta, s.tau_level), reverse=True)[0]
        reason = "coarsest τ that fully clears outgrown + held with p90 rows in range"
    else:
        best = sorted(
            scores,
            key=lambda s: (
                s.outgrown_clear + s.held_clear,
                s.labeled_score,
                -s.p90_n,
                s.tau_delta,
                s.tau_level,
            ),
            reverse=True,
        )[0]
        reason = "best labeled-week score (no pair fully cleared both outgrown and held at p90 cap)"

    print(f"\nLOCKED τ_Δ={best.tau_delta} bps  τ_L={best.tau_level} bps")
    print(f"Reason: {reason}")
    print(
        f"medN={best.median_n:.0f} p90N={best.p90_n:.0f} covΔM={best.coverage_d_market:.0%} "
        f"outgrown={best.outgrown_clear:.1f} held={best.held_clear:.1f}"
    )

    gated = gated_by[(best.tau_delta, best.tau_level)]
    print("\n========== WEEK CARDS (human, no LLM) ==========")
    print(
        f"Gates: mover |ΔAE|/market ≥ {best.tau_delta} bps; "
        f"ballast AE/market ≥ {best.tau_level} bps. All owners. "
        f"Δbps = 10000×ΔAE/market (contribution to the pie)."
    )
    for _, w in labeled.iterrows():
        titles = gated[gated["WEEK_END_DATE"] == w["WEEK_ENDING_DATE"]]
        print_week_card(w, titles)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
