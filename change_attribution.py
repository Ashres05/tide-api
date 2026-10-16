"""
AMG weekly share attribution (Phase 3 serving).

Locked dual-gate τ (bps of inverted Current-US market AE):
  τ_Δ      = 25  mover: |ΔAE| / market (shared)
  τ_L      = per owner (see TAU_LEVEL_BY_OWNER); default 75
  τ_L,AMG  = 40  (AMG entry in that map)

Facts (rewritten after each floor pull):
  model/data/attribution_mrelg_weekly.csv  — raw MRELG×week AE (all owners, ≥ MIN_AE)
  model/data/amg_week_market.csv           — 1 row / week (AMG + market)
  model/data/amg_week_owners.csv           — 8 rows / week (panel decomp + coverage)
  model/data/amg_week_mrelgs.csv           — dual-gate titles
  model/data/amg_week_labels.csv           — owner rollup

Market AE is AMG-inverted from current_data_all (universe Current share).
Do not sum TARGET_LABELS as the pie. AMG is the parent of Atlantic Records /
10K / 300 in marketshare_weekly — do not add those three on top of AMG.

Serving: GET /v1/marketshare/attribution — driver-aware title order, ui.lead
flags, coverage_same_sign + coverage_mover_same_sign (clipped at 1).
Facets filter mrelgs/labels only; week decomp is always full-market.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from model.marketshare_labels import MARKET_ANCHOR_LABEL, TARGET_LABELS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "model" / "data"
CURRENT_DATA_ALL_CSV = DATA_DIR / "current_data_all.csv"
MRELG_FLOOR_CSV = DATA_DIR / "attribution_mrelg_weekly.csv"
WEEK_MARKET_CSV = DATA_DIR / "amg_week_market.csv"
WEEK_OWNERS_CSV = DATA_DIR / "amg_week_owners.csv"
WEEK_MRELGS_CSV = DATA_DIR / "amg_week_mrelgs.csv"
WEEK_LABELS_CSV = DATA_DIR / "amg_week_labels.csv"

MRELG_QUERY = "query_attribution_mrelg_weekly.sql"

TAU_DELTA_BPS = 25
TAU_LEVEL_BPS = 75
TAU_LEVEL_AMG_BPS = 40  # kept as AMG alias of TAU_LEVEL_BY_OWNER
MIN_AE = 2500
CURRENT_AGE_MONTHS = 18
OVERLAP_DAYS = 7

# Per-owner anchor τL (bps of inverted market). τΔ is shared. Default 75.
TAU_LEVEL_BY_OWNER: dict[str, int] = {
    "Warner Records": 25,
    "Atlantic Music Group": 40,
    "IGA": 50,
    "CMG": 25,
    "REPUBLIC Collective": 75,
    "THE ORCHARD": 25,
    "RCA Records": 40,
    "Columbia Records": 75,
}

PANEL = frozenset(TARGET_LABELS)
AMG = MARKET_ANCHOR_LABEL
UNIVERSE_LABELS = frozenset(
    {
        "Total Universe",
        "Total Universe (Excl Under Review)",
    }
)

CONFIG = {
    "tau_delta_bps": TAU_DELTA_BPS,
    "tau_level_bps": TAU_LEVEL_BPS,
    "tau_level_amg_bps": TAU_LEVEL_AMG_BPS,
    "tau_level_by_owner": dict(TAU_LEVEL_BY_OWNER),
    "min_ae": MIN_AE,
    "current_age_months": CURRENT_AGE_MONTHS,
}

# Frontend copy + sort. Facets never change these strings.
DRIVER_UI: dict[str, dict[str, str]] = {
    "outgrown": {
        "headline": "AMG volume is up; share is down because the pie grew faster",
        "lead_rule": "non_amg_movers",
        "sort": "non-AMG movers that grew the pie first by |d_market_bps|, then AMG movers, then anchors",
    },
    "held": {
        "headline": "AMG held while the market faded — share up on anchors",
        "lead_rule": "amg_anchor_then_market_shrink",
        "sort": "AMG anchors first by ae_share_bps, then titles with negative d_market_bps, then rest",
    },
    "other_up": {
        "headline": "Share fell because Other (outside the eight) grew the market",
        "lead_rule": "pie_growers",
        "sort": "movers that grew the pie first by |d_market_bps| (tag=other marks non-eight)",
    },
    "amg_up": {
        "headline": "AMG share rose on AMG title gains",
        "lead_rule": "amg_movers",
        "sort": "AMG movers first by |d_market_bps|, then other movers, then anchors",
    },
    "amg_down": {
        "headline": "AMG share fell on AMG title losses",
        "lead_rule": "amg_movers",
        "sort": "AMG movers first by |d_market_bps|, then other movers, then anchors",
    },
    "market_up": {
        "headline": "Market expanded faster than AMG",
        "lead_rule": "non_amg_movers",
        "sort": "non-AMG movers that grew the pie first by |d_market_bps|",
    },
    "market_down": {
        "headline": "Market contracted; AMG share moved with the fade",
        "lead_rule": "market_shrink",
        "sort": "negative d_market_bps first, then rest by |d_market_bps|",
    },
    "mix": {
        "headline": "Mixed own vs market effects",
        "lead_rule": "abs_d_market_bps",
        "sort": "|d_market_bps| descending",
    },
    "na": {
        "headline": "Not enough lag to classify this week",
        "lead_rule": "abs_d_market_bps",
        "sort": "|d_market_bps| descending",
    },
}


def tau_level_for(owner_label: str) -> int:
    return int(TAU_LEVEL_BY_OWNER.get(str(owner_label), TAU_LEVEL_BPS))


def headline_for(driver: str, owner_label: str) -> str:
    name = owner_label
    if driver == "outgrown":
        return f"{name} volume is up; share is down because the pie grew faster"
    if driver == "held":
        return f"{name} held while the market faded — share up on anchors"
    if driver == "other_up":
        return f"{name} share fell because Other (outside the eight) grew the market"
    if driver in ("amg_up", "own_up"):
        return f"{name} share rose on {name} title gains"
    if driver in ("amg_down", "own_down"):
        return f"{name} share fell on {name} title losses"
    if driver == "market_up":
        return f"Market expanded faster than {name}"
    if driver == "market_down":
        return f"Market contracted; {name} share moved with the fade"
    if driver == "na":
        return "Not enough lag to classify this week"
    return f"Mixed own vs market effects for {name}"


def _owner_driver_token(driver: str, owner_label: str) -> str:
    if owner_label == AMG:
        return driver
    if driver == "amg_up":
        return "own_up"
    if driver == "amg_down":
        return "own_down"
    return driver


@dataclass(frozen=True)
class _CacheKey:
    mtime_ns: int
    size: int


_cache_lock = threading.Lock()
_week_cache: dict[_CacheKey, pd.DataFrame] = {}
_owner_cache: dict[_CacheKey, pd.DataFrame] = {}
_mrelg_cache: dict[_CacheKey, pd.DataFrame] = {}
_label_cache: dict[_CacheKey, pd.DataFrame] = {}


def clear_cache() -> None:
    with _cache_lock:
        _week_cache.clear()
        _owner_cache.clear()
        _mrelg_cache.clear()
        _label_cache.clear()


def cold_start_min_week() -> str:
    return (date.today() - timedelta(weeks=80)).strftime("%Y-%m-%d")


def _file_sig(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_mtime_ns, st.st_size


def _share_to_frac(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    bps = s > 100
    pct = (s > 1) & ~bps
    return pd.Series(
        np.where(bps, s / 10000.0, np.where(pct, s / 100.0, s)),
        index=s.index,
    )


def _iso(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


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
    out = market.merge(panel_sum, on="WEEK_ENDING_DATE", how="left")
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
    out["D_SHARE_BPS"] = out["D_SHARE"] * 10000.0
    out["OWN_EFFECT_BPS"] = 10000.0 * out["D_AMG_AE"] / out["MARKET_AE"]
    out["MARKET_EFFECT_BPS"] = (
        -10000.0 * out["PREV_AMG_SHARE"] * out["D_MARKET_AE"] / out["MARKET_AE"]
    )
    out["DRIVER"] = out.apply(classify_driver, axis=1)
    return out


def classify_driver(row: pd.Series) -> str:
    d_share = row.get("D_SHARE")
    d_amg = row.get("D_AMG_AE")
    d_mkt = row.get("D_MARKET_AE")
    d_other = row.get("D_OTHER_AE")
    d_panel = row.get("D_PANEL_AE")
    if pd.isna(d_share) or pd.isna(d_amg) or pd.isna(d_mkt):
        return "na"
    if d_amg > 0 and d_share < 0:
        return "outgrown"
    prev_amg = row.get("PREV_AMG_AE") or 0.0
    amg_pct = abs(d_amg) / max(abs(prev_amg), 1.0)
    if amg_pct < 0.02 and d_mkt < 0 and d_share > 0:
        return "held"
    panel_ex_amg = float(d_panel or 0) - float(d_amg or 0)
    if d_share < 0 and pd.notna(d_other) and d_other > max(panel_ex_amg, 0):
        return "other_up"
    own = row.get("OWN_EFFECT_BPS") or 0.0
    mkt_fx = row.get("MARKET_EFFECT_BPS") or 0.0
    if d_share > 0 and d_amg > 0 and own >= abs(mkt_fx):
        return "amg_up"
    if d_share < 0 and d_amg < 0 and abs(own) >= abs(mkt_fx):
        return "amg_down"
    if d_share < 0 and d_mkt > 0:
        return "market_up"
    if d_share > 0 and d_mkt < 0:
        return "market_down"
    return "mix"


def build_owner_weeks(panel: pd.DataFrame, market_weeks: pd.DataFrame) -> pd.DataFrame:
    """One row per TARGET_LABELS owner per week. Same inverted MARKET_AE as AMG."""
    df = panel.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["WEEK_ENDING_DATE"] = pd.to_datetime(df["WEEK_ENDING_DATE"], errors="coerce")
    df["AE"] = pd.to_numeric(df["ALBUM_EQUIVALENT"], errors="coerce")
    df["SHARE"] = _share_to_frac(df["ALBUM_EQUIVALENT_SHARE"])
    df["LABEL_NAME"] = df["LABEL_NAME"].astype(str).str.strip()
    df = df[df["LABEL_NAME"].isin(PANEL)].copy()
    mkt = market_weeks[
        [
            "WEEK_ENDING_DATE",
            "MARKET_AE",
            "D_MARKET_AE",
            "D_PANEL_AE",
            "D_OTHER_AE",
            "PANEL_AE",
            "OTHER_AE",
        ]
    ]
    frames = []
    for lab in TARGET_LABELS:
        sub = (
            df[df["LABEL_NAME"] == lab][["WEEK_ENDING_DATE", "AE", "SHARE"]]
            .drop_duplicates("WEEK_ENDING_DATE", keep="last")
            .sort_values("WEEK_ENDING_DATE")
        )
        sub = sub.merge(mkt, on="WEEK_ENDING_DATE", how="inner")
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
        sub["DRIVER"] = fake.apply(classify_driver, axis=1)
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prep_mrelg(mrelg: pd.DataFrame) -> pd.DataFrame:
    df = mrelg.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["WEEK_END_DATE"] = pd.to_datetime(df["WEEK_END_DATE"], errors="coerce")
    df["FIRST_SALE_DATE"] = pd.to_datetime(df["FIRST_SALE_DATE"], errors="coerce")
    df["AE"] = pd.to_numeric(df["AE"], errors="coerce")
    df["OWNER_LABEL"] = df["OWNER_LABEL"].fillna("Unknown").astype(str).str.strip()
    df["TITLE"] = df["TITLE"].fillna("").astype(str)
    df["ARTIST"] = df["ARTIST"].fillna("").astype(str)
    df["LEVEL_2_DISTRIBUTOR"] = df.get(
        "LEVEL_2_DISTRIBUTOR", pd.Series("", index=df.index)
    ).fillna("").astype(str)
    df["LEVEL_3_DISTRIBUTOR"] = df.get(
        "LEVEL_3_DISTRIBUTOR", pd.Series("", index=df.index)
    ).fillna("").astype(str)
    df["MRELG_ID"] = df["MRELG_ID"].astype(str)
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
    *,
    tau_delta_bps: int | None = None,
    tau_level_bps: int | None = None,
    tau_level_amg_bps: int | None = None,
    tau_delta_amg_bps: int | None = None,
    tau_level_by_owner: dict[str, int] | None = None,
) -> pd.DataFrame:
    tau_d = (TAU_DELTA_BPS if tau_delta_bps is None else tau_delta_bps) / 10000.0
    default_l = TAU_LEVEL_BPS if tau_level_bps is None else tau_level_bps
    tau_d_amg = (tau_d if tau_delta_amg_bps is None else tau_delta_amg_bps / 10000.0)
    w = weeks.rename(columns={"WEEK_ENDING_DATE": "WEEK_END_DATE"})
    df = mrelg.merge(w[["WEEK_END_DATE", "MARKET_AE"]], on="WEEK_END_DATE", how="inner")
    df = df[df["MARKET_AE"] > 0].copy()
    df["AE_SHARE"] = df["AE"] / df["MARKET_AE"]
    df["D_MARKET_SHARE"] = df["DELTA_AE"] / df["MARKET_AE"]
    mapping = TAU_LEVEL_BY_OWNER if tau_level_by_owner is None else tau_level_by_owner
    owner_l = df["OWNER_LABEL"].map(mapping).fillna(default_l).astype(float) / 10000.0
    if tau_level_amg_bps is not None:
        owner_l = np.where(df["IS_AMG"], tau_level_amg_bps / 10000.0, owner_l)
    mover = df["D_MARKET_SHARE"].abs() >= np.where(df["IS_AMG"], tau_d_amg, tau_d)
    anchor = df["AE_SHARE"] >= owner_l
    df["GATE"] = np.where(
        mover & anchor,
        "both",
        np.where(mover, "mover", np.where(anchor, "anchor", "")),
    )
    df = df[df["GATE"] != ""].copy()
    df["AE_SHARE_BPS"] = df["AE_SHARE"] * 10000.0
    df["D_MARKET_BPS"] = df["D_MARKET_SHARE"] * 10000.0
    df["SORT_KEY"] = np.maximum(df["AE_SHARE_BPS"].abs(), df["D_MARKET_BPS"].abs())
    return df.sort_values(["WEEK_END_DATE", "SORT_KEY"], ascending=[True, False])


def _coverage_same_sign(
    gated: pd.DataFrame,
    weeks: pd.DataFrame,
    *,
    movers_only: bool,
) -> pd.Series:
    """
    Fraction of |Δ market AE| named by gated titles that moved with the market.
    Clipped at 1. Empty / tiny-move weeks are NaN.
    """
    denom = weeks["D_MARKET_AE"].abs()
    empty = pd.Series(np.nan, index=weeks.index)
    if gated.empty:
        return empty
    g = gated.copy()
    g["WEEK_END_DATE"] = pd.to_datetime(g["WEEK_END_DATE"])
    if movers_only:
        g = g[g["GATE"].isin(["mover", "both"])]
    if g.empty:
        return pd.Series(np.where(denom > 1, 0.0, np.nan), index=weeks.index)
    dm = weeks.set_index("WEEK_ENDING_DATE")["D_MARKET_AE"]
    g = g.join(dm.rename("D_MARKET_AE"), on="WEEK_END_DATE")
    with np.errstate(invalid="ignore"):
        mask = (
            g["D_MARKET_AE"].notna()
            & (g["D_MARKET_AE"].abs() > 1)
            & (np.sign(g["DELTA_AE"]) == np.sign(g["D_MARKET_AE"]))
            & (g["DELTA_AE"] != 0)
        )
    same_sum = g.loc[mask].groupby("WEEK_END_DATE")["DELTA_AE"].sum()
    mapped = weeks["WEEK_ENDING_DATE"].map(same_sum).fillna(0.0)
    return pd.Series(
        np.where(denom > 1, np.minimum(1.0, mapped.abs() / denom), np.nan),
        index=weeks.index,
    )


def _attach_coverage(weeks: pd.DataFrame, gated: pd.DataFrame) -> pd.DataFrame:
    out = weeks.copy()
    if gated.empty:
        out["N_MOVERS"] = 0
        out["N_ANCHORS"] = 0
        out["N_TITLES"] = 0
        out["COVERAGE_D_MARKET"] = np.nan
        out["COVERAGE_SAME_SIGN"] = np.nan
        out["COVERAGE_MOVER_SAME_SIGN"] = np.nan
        out["COVERAGE_AMG"] = np.nan
        return out
    g = gated.copy()
    g["WEEK_END_DATE"] = pd.to_datetime(g["WEEK_END_DATE"])
    n_m = g[g["GATE"].isin(["mover", "both"])].groupby("WEEK_END_DATE").size()
    n_b = g[g["GATE"].isin(["anchor", "both"])].groupby("WEEK_END_DATE").size()
    n_t = g.groupby("WEEK_END_DATE").size()
    d_sum = g.groupby("WEEK_END_DATE")["DELTA_AE"].sum()
    amg_ae = g[g["IS_AMG"]].groupby("WEEK_END_DATE")["AE"].sum()
    out["N_MOVERS"] = out["WEEK_ENDING_DATE"].map(n_m).fillna(0).astype(int)
    out["N_ANCHORS"] = out["WEEK_ENDING_DATE"].map(n_b).fillna(0).astype(int)
    out["N_TITLES"] = out["WEEK_ENDING_DATE"].map(n_t).fillna(0).astype(int)
    listed = out["WEEK_ENDING_DATE"].map(d_sum)
    out["COVERAGE_D_MARKET"] = np.where(
        out["D_MARKET_AE"].abs() > 1,
        listed.abs() / out["D_MARKET_AE"].abs(),
        np.nan,
    )
    out["COVERAGE_SAME_SIGN"] = _coverage_same_sign(g, out, movers_only=False)
    out["COVERAGE_MOVER_SAME_SIGN"] = _coverage_same_sign(g, out, movers_only=True)
    out["COVERAGE_AMG"] = np.where(
        out["AMG_AE"] > 0,
        out["WEEK_ENDING_DATE"].map(amg_ae).fillna(0) / out["AMG_AE"],
        np.nan,
    )
    return out


def _week_market_out(weeks: pd.DataFrame) -> pd.DataFrame:
    df = weeks.dropna(subset=["D_SHARE"]).copy()
    df["week_end_date"] = _iso(df["WEEK_ENDING_DATE"])
    cols = {
        "AMG_AE": "amg_ae",
        "AMG_SHARE": "amg_share",
        "MARKET_AE": "market_ae",
        "PANEL_AE": "panel_ae",
        "OTHER_AE": "other_ae",
        "D_AMG_AE": "d_amg_ae",
        "D_MARKET_AE": "d_market_ae",
        "D_PANEL_AE": "d_panel_ae",
        "D_OTHER_AE": "d_other_ae",
        "D_SHARE": "d_share",
        "D_SHARE_BPS": "d_share_bps",
        "OWN_EFFECT_BPS": "own_effect_bps",
        "MARKET_EFFECT_BPS": "market_effect_bps",
        "DRIVER": "driver",
        "N_MOVERS": "n_movers",
        "N_ANCHORS": "n_anchors",
        "N_TITLES": "n_titles",
        "COVERAGE_D_MARKET": "coverage_d_market",
        "COVERAGE_SAME_SIGN": "coverage_same_sign",
        "COVERAGE_MOVER_SAME_SIGN": "coverage_mover_same_sign",
        "COVERAGE_AMG": "coverage_amg",
    }
    out = df.rename(columns=cols)
    out["tau_delta_bps"] = TAU_DELTA_BPS
    out["tau_level_bps"] = TAU_LEVEL_BPS
    out["tau_level_amg_bps"] = TAU_LEVEL_AMG_BPS
    keep = ["week_end_date", *cols.values(), "tau_delta_bps", "tau_level_bps", "tau_level_amg_bps"]
    return out[keep].sort_values("week_end_date").reset_index(drop=True)


def _owners_out(owner_weeks: pd.DataFrame, gated: pd.DataFrame) -> pd.DataFrame:
    df = owner_weeks.dropna(subset=["D_SHARE"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "week_end_date",
                "owner_label",
                "ae",
                "share",
                "d_ae",
                "d_share",
                "d_share_bps",
                "own_effect_bps",
                "market_effect_bps",
                "driver",
                "coverage_level",
                "coverage_delta",
                "n_titles",
                "n_movers",
                "n_anchors",
                "tau_level_bps",
            ]
        )
    df["week_end_date"] = _iso(df["WEEK_ENDING_DATE"])
    if gated.empty:
        g = pd.DataFrame(
            columns=["WEEK_END_DATE", "OWNER_LABEL", "AE", "DELTA_AE", "GATE"]
        )
    else:
        g = gated.copy()
        g["WEEK_END_DATE"] = pd.to_datetime(g["WEEK_END_DATE"])
        g["week_end_date"] = _iso(g["WEEK_END_DATE"])
    stats = []
    for _, row in df.iterrows():
        wk = row["week_end_date"]
        lab = row["OWNER_LABEL"]
        sub = (
            g[(g["week_end_date"] == wk) & (g["OWNER_LABEL"] == lab)]
            if not g.empty and "week_end_date" in g.columns
            else pd.DataFrame()
        )
        ae = float(row["AE"]) if pd.notna(row["AE"]) else 0.0
        d_ae = float(row["D_AE"]) if pd.notna(row["D_AE"]) else 0.0
        listed_ae = float(sub["AE"].sum()) if len(sub) else 0.0
        listed_d = float(sub["DELTA_AE"].sum()) if len(sub) else 0.0
        n_m = int(sub["GATE"].isin(["mover", "both"]).sum()) if len(sub) else 0
        n_a = int(sub["GATE"].isin(["anchor", "both"]).sum()) if len(sub) else 0
        cov_l = (listed_ae / ae) if ae > 0 else np.nan
        cov_d = (
            min(1.0, abs(listed_d) / abs(d_ae)) if abs(d_ae) > 1 else np.nan
        )
        token = _owner_driver_token(str(row["DRIVER"]), lab)
        stats.append(
            {
                "week_end_date": wk,
                "owner_label": lab,
                "ae": row["AE"],
                "share": row["SHARE"],
                "d_ae": row["D_AE"],
                "d_share": row["D_SHARE"],
                "d_share_bps": row["D_SHARE_BPS"],
                "own_effect_bps": row["OWN_EFFECT_BPS"],
                "market_effect_bps": row["MARKET_EFFECT_BPS"],
                "driver": token,
                "coverage_level": cov_l,
                "coverage_delta": cov_d,
                "n_titles": int(len(sub)),
                "n_movers": n_m,
                "n_anchors": n_a,
                "tau_level_bps": tau_level_for(lab),
            }
        )
    return pd.DataFrame(stats).sort_values(
        ["week_end_date", "share"], ascending=[True, False]
    ).reset_index(drop=True)


def _mrelgs_out(gated: pd.DataFrame) -> pd.DataFrame:
    if gated.empty:
        return pd.DataFrame(
            columns=[
                "week_end_date",
                "mrelg_id",
                "title",
                "artist",
                "owner_label",
                "level_2_distributor",
                "level_3_distributor",
                "in_panel",
                "is_amg",
                "ae",
                "prev_ae",
                "delta_ae",
                "is_new",
                "first_sale_date",
                "ae_share_bps",
                "d_market_bps",
                "gate",
            ]
        )
    out = pd.DataFrame(
        {
            "week_end_date": _iso(gated["WEEK_END_DATE"]),
            "mrelg_id": gated["MRELG_ID"],
            "title": gated["TITLE"],
            "artist": gated["ARTIST"],
            "owner_label": gated["OWNER_LABEL"],
            "level_2_distributor": gated["LEVEL_2_DISTRIBUTOR"],
            "level_3_distributor": gated["LEVEL_3_DISTRIBUTOR"],
            "in_panel": gated["IN_PANEL"].astype(bool),
            "is_amg": gated["IS_AMG"].astype(bool),
            "ae": gated["AE"],
            "prev_ae": gated["PREV_AE"],
            "delta_ae": gated["DELTA_AE"],
            "is_new": gated["IS_NEW"].astype(bool),
            "first_sale_date": _iso(gated["FIRST_SALE_DATE"]),
            "ae_share_bps": gated["AE_SHARE_BPS"],
            "d_market_bps": gated["D_MARKET_BPS"],
            "gate": gated["GATE"],
        }
    )
    return out.reset_index(drop=True)


def _labels_out(gated: pd.DataFrame) -> pd.DataFrame:
    if gated.empty:
        return pd.DataFrame(
            columns=[
                "week_end_date",
                "owner_label",
                "in_panel",
                "is_amg",
                "ae",
                "delta_ae",
                "d_market_bps",
                "n_titles",
            ]
        )
    g = gated.copy()
    g["week_end_date"] = _iso(g["WEEK_END_DATE"])
    grouped = g.groupby(
        ["week_end_date", "OWNER_LABEL", "IN_PANEL", "IS_AMG"],
        as_index=False,
    ).agg(
        ae=("AE", "sum"),
        delta_ae=("DELTA_AE", "sum"),
        d_market_bps=("D_MARKET_BPS", "sum"),
        n_titles=("MRELG_ID", "size"),
    )
    return grouped.rename(
        columns={
            "OWNER_LABEL": "owner_label",
            "IN_PANEL": "in_panel",
            "IS_AMG": "is_amg",
        }
    ).sort_values(["week_end_date", "d_market_bps"], ascending=[True, False])


def rebuild_gated_tables(
    *,
    panel_path: Path = CURRENT_DATA_ALL_CSV,
    floor_path: Path = MRELG_FLOOR_CSV,
) -> dict[str, int]:
    panel = _load_csv(panel_path)
    floor = _load_csv(floor_path)
    if panel.empty:
        raise FileNotFoundError(f"missing {panel_path}")
    if floor.empty:
        raise FileNotFoundError(f"missing {floor_path}")

    weeks = build_week_market(panel)
    prepped = prep_mrelg(floor)
    floor_weeks = pd.to_datetime(prepped["WEEK_END_DATE"]).dropna().sort_values().unique()
    # First floor week has no prior MRELG row, so ΔAE is the full weekly AE.
    if len(floor_weeks) > 1:
        floor_weeks = floor_weeks[1:]
    weeks = weeks[weeks["WEEK_ENDING_DATE"].isin(floor_weeks)].copy()
    gated = apply_gate(prepped, weeks)
    weeks = _attach_coverage(weeks, gated)
    owner_weeks = build_owner_weeks(panel, weeks)
    week_out = _week_market_out(weeks)
    owner_out = _owners_out(owner_weeks, gated)
    mrelg_out = _mrelgs_out(gated)
    label_out = _labels_out(gated)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    week_out.to_csv(WEEK_MARKET_CSV, index=False)
    owner_out.to_csv(WEEK_OWNERS_CSV, index=False)
    mrelg_out.to_csv(WEEK_MRELGS_CSV, index=False)
    label_out.to_csv(WEEK_LABELS_CSV, index=False)
    clear_cache()
    logger.info(
        "change_attribution: wrote weeks=%d owners=%d titles=%d labels=%d τΔ=%s",
        len(week_out),
        len(owner_out),
        len(mrelg_out),
        len(label_out),
        TAU_DELTA_BPS,
    )
    return {
        "weeks": int(len(week_out)),
        "owners": int(len(owner_out)),
        "mrelgs": int(len(mrelg_out)),
        "labels": int(len(label_out)),
    }


def pull_mrelg_floor(sf, min_week: str) -> pd.DataFrame:
    from snowflake_conn import load_sql

    sql = (
        load_sql(MRELG_QUERY)
        .replace("{MIN_WEEK_END_DATE}", min_week)
        .replace("{MIN_AE}", str(MIN_AE))
    )
    df = sf.query(sql)
    df.columns = [str(c).strip().lstrip("\ufeff").upper() for c in df.columns]
    for col in ("WEEK_END_DATE", "FIRST_SALE_DATE"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _append_csv(path: Path, new_df: pd.DataFrame, dedupe: list[str], sort_by: list[str]) -> int:
    if new_df is None or new_df.empty:
        return 0
    if path.exists():
        existing = pd.read_csv(path)
        existing.columns = [str(c).strip().lstrip("\ufeff") for c in existing.columns]
        before = len(existing)
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        before = 0
        merged = new_df.copy()
    merged.columns = [str(c).strip().lstrip("\ufeff") for c in merged.columns]
    merged = merged.drop_duplicates(subset=dedupe, keep="last")
    merged = merged.sort_values(sort_by).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return len(merged) - before


def refresh_change_attribution(sf, min_week: str) -> int:
    """
    Incremental floor pull + full gated rebuild.
    Returns net rows added to attribution_mrelg_weekly.csv.
    """
    new_df = pull_mrelg_floor(sf, min_week)
    added = _append_csv(
        MRELG_FLOOR_CSV,
        new_df,
        ["WEEK_END_DATE", "MRELG_ID"],
        ["WEEK_END_DATE", "AE"],
    )
    rebuild_gated_tables()
    return added


def _cached_csv(path: Path, bucket: dict[_CacheKey, pd.DataFrame]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path.name} not found; run refresh_data / change_attribution")
    key = _CacheKey(*_file_sig(path))
    with _cache_lock:
        hit = bucket.get(key)
        if hit is not None:
            return hit.copy()
    df = pd.read_csv(path)
    with _cache_lock:
        bucket[key] = df
    return df.copy()


def load_week_market() -> pd.DataFrame:
    return _cached_csv(WEEK_MARKET_CSV, _week_cache)


def load_week_owners() -> pd.DataFrame:
    return _cached_csv(WEEK_OWNERS_CSV, _owner_cache)


def load_week_mrelgs() -> pd.DataFrame:
    return _cached_csv(WEEK_MRELGS_CSV, _mrelg_cache)


def load_week_labels() -> pd.DataFrame:
    return _cached_csv(WEEK_LABELS_CSV, _label_cache)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            continue
        if out[col].dtype == object:
            continue
        if pd.api.types.is_integer_dtype(out[col]):
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].where(out[col].notna(), None)
    return out.replace({np.nan: None}).to_dict(orient="records")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def _title_tag(is_amg: pd.Series, in_panel: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(is_amg, "amg", np.where(in_panel, "panel", "other")),
        index=is_amg.index,
    )


def annotate_titles_for_driver(
    titles: pd.DataFrame,
    driver: str,
    subject_owner: str = AMG,
) -> pd.DataFrame:
    """Add tag / lead / section and apply the Phase 3 UI sort. Does not drop rows."""
    df = titles.copy()
    if df.empty:
        for col, val in (("tag", "other"), ("lead", False), ("section", "also")):
            df[col] = val
        return df
    df["is_amg"] = _as_bool(df["is_amg"])
    df["in_panel"] = _as_bool(df["in_panel"])
    df["is_new"] = _as_bool(df["is_new"]) if "is_new" in df.columns else False
    df["tag"] = _title_tag(df["is_amg"], df["in_panel"])
    df["d_market_bps"] = pd.to_numeric(df["d_market_bps"], errors="coerce")
    df["ae_share_bps"] = pd.to_numeric(df["ae_share_bps"], errors="coerce")
    is_subject = df["owner_label"].astype(str) == str(subject_owner)
    mover = df["gate"].isin(["mover", "both"])
    anchor = df["gate"].isin(["anchor", "both"])
    abs_d = df["d_market_bps"].abs()

    if driver in ("outgrown", "market_up"):
        grew = mover & (df["d_market_bps"] > 0)
        pri = np.where(~is_subject & grew, 0, np.where(is_subject & mover, 1, 2))
        lead = pri == 0
        sec = -abs_d
    elif driver == "held":
        pri = np.where(is_subject & anchor, 0, np.where(df["d_market_bps"] < 0, 1, 2))
        lead = pri <= 1
        sec = np.where(pri == 0, -df["ae_share_bps"], -abs_d)
    elif driver == "other_up":
        grew = mover & (df["d_market_bps"] > 0)
        pri = np.where(grew, 0, 1)
        lead = pri == 0
        sec = -abs_d
    elif driver in ("amg_up", "amg_down", "own_up", "own_down"):
        pri = np.where(is_subject & mover, 0, np.where(mover, 1, 2))
        lead = pri == 0
        sec = -abs_d
    elif driver == "market_down":
        pri = np.where(df["d_market_bps"] < 0, 0, 1)
        lead = pri == 0
        sec = -abs_d
    else:
        pri = np.zeros(len(df), dtype=int)
        lead = np.ones(len(df), dtype=bool)
        sec = -abs_d

    df["lead"] = lead.astype(bool)
    df["section"] = np.where(df["lead"], "lead", "also")
    df["_pri"] = pri
    df["_sec"] = sec
    df = df.sort_values(["_pri", "_sec"], kind="mergesort").drop(columns=["_pri", "_sec"])
    return df.reset_index(drop=True)


def _apply_title_facets(
    titles: pd.DataFrame,
    *,
    owner_label: str | None = None,
    in_panel: bool | None = None,
    is_amg: bool | None = None,
    gate: str | None = None,
    tag: str | None = None,
) -> pd.DataFrame:
    df = titles
    if df.empty:
        return df
    if owner_label is not None and "owner_label" in df.columns:
        df = df[df["owner_label"].astype(str) == owner_label]
    if in_panel is not None and "in_panel" in df.columns:
        df = df[_as_bool(df["in_panel"]) == bool(in_panel)]
    if is_amg is not None and "is_amg" in df.columns:
        df = df[_as_bool(df["is_amg"]) == bool(is_amg)]
    if gate is not None and "gate" in df.columns:
        df = df[df["gate"].astype(str) == gate]
    if tag is not None and "tag" in df.columns:
        df = df[df["tag"].astype(str) == tag]
    return df


def query_week_pack(
    week_end_date: str | None = None,
    *,
    owner_label: str | None = None,
    in_panel: bool | None = None,
    is_amg: bool | None = None,
    gate: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    weeks = load_week_market()
    empty_ui = {
        "headline": None,
        "lead_rule": None,
        "sort": None,
        "facet_note": (
            "owner_label, in_panel, is_amg, gate, tag filter mrelgs/labels only; "
            "week decomp is always the full Current-US inverted market"
        ),
    }
    if weeks.empty:
        return {
            "config": CONFIG,
            "weeks": [],
            "week": None,
            "ui": empty_ui,
            "mrelgs": [],
            "labels": [],
            "owners": [],
        }
    weeks["week_end_date"] = weeks["week_end_date"].astype(str)
    available = sorted(weeks["week_end_date"].unique().tolist())
    want = (week_end_date or available[-1]).strip()
    week_rows = weeks[weeks["week_end_date"] == want]
    if week_rows.empty:
        raise ValueError(f"no attribution week {want}; have {available[0]} … {available[-1]}")
    week = _records(week_rows)[0]
    driver = str(week.get("driver") or "mix")
    copy = DRIVER_UI.get(driver, DRIVER_UI["mix"])
    ui = {
        "driver": driver,
        "headline": copy["headline"],
        "lead_rule": copy["lead_rule"],
        "sort": copy["sort"],
        "facet_note": empty_ui["facet_note"],
    }

    mrelgs = load_week_mrelgs()
    labels = load_week_labels()
    mrelgs["week_end_date"] = mrelgs["week_end_date"].astype(str)
    labels["week_end_date"] = labels["week_end_date"].astype(str)
    titles = annotate_titles_for_driver(
        mrelgs[mrelgs["week_end_date"] == want].copy(),
        driver,
        subject_owner=AMG,
    )
    titles = _apply_title_facets(
        titles,
        owner_label=owner_label,
        in_panel=in_panel,
        is_amg=is_amg,
        gate=gate,
        tag=tag,
    )
    lab = labels[labels["week_end_date"] == want].copy()
    if not lab.empty:
        lab["is_amg"] = _as_bool(lab["is_amg"])
        lab["in_panel"] = _as_bool(lab["in_panel"])
        lab["tag"] = _title_tag(lab["is_amg"], lab["in_panel"])
        lab["d_market_bps"] = pd.to_numeric(lab["d_market_bps"], errors="coerce")
        lab["_pri"] = np.where(lab["is_amg"], 0, 1)
        lab["_sec"] = -lab["d_market_bps"].abs()
        lab = lab.sort_values(["_pri", "_sec"], kind="mergesort").drop(columns=["_pri", "_sec"])
        lab = _apply_title_facets(
            lab,
            owner_label=owner_label,
            in_panel=in_panel,
            is_amg=is_amg,
            tag=tag,
        )
    owner_payload: list[dict[str, Any]] = []
    try:
        owners_df = load_week_owners()
    except FileNotFoundError:
        owners_df = pd.DataFrame()
    if not owners_df.empty:
        owners_df["week_end_date"] = owners_df["week_end_date"].astype(str)
        own_rows = owners_df[owners_df["week_end_date"] == want].copy()
        base = mrelgs[mrelgs["week_end_date"] == want].copy()
        for rec in _records(own_rows):
            lab_name = str(rec.get("owner_label") or "")
            drv = str(rec.get("driver") or "mix")
            rec["headline"] = headline_for(drv, lab_name)
            rec["tag"] = "amg" if lab_name == AMG else "panel"
            annotated = annotate_titles_for_driver(
                base.copy(), drv, subject_owner=lab_name
            )
            rec["lead_mrelg_ids"] = [
                str(x) for x in annotated.loc[annotated["lead"], "mrelg_id"].tolist()
            ]
            owner_payload.append(rec)
    return {
        "config": CONFIG,
        "weeks": available,
        "week": week,
        "ui": ui,
        "mrelgs": _records(titles),
        "labels": _records(lab),
        "owners": owner_payload,
    }
