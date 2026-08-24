#!/usr/bin/env python3
"""
CLI for 75k marketshare decay and full scenario simulation.

Examples (from repo root, with venv active):

  # Artist decay only — weekly AE curve + 2026 CY sum
  python scripts/simulate_marketshare.py decay \\
    --artist "Olivia Rodrigo" \\
    --title "you seem pretty sad for a girl so in love" \\
    --label "Interscope/Geffen/A&M" \\
    --genre Pop \\
    --date 2026-06-12 \\
    --fw-streams 210000 \\
    --fw-sales 250000

  # Full marketshare simulation (weekly injections + YTD share)
  python scripts/simulate_marketshare.py marketshare \\
    --artist "Olivia Rodrigo" \\
    --title "you seem pretty sad for a girl so in love" \\
    --label "Interscope/Geffen/A&M" \\
    --genre Pop \\
    --date 2026-06-12 \\
    --fw-streams 210000 \\
    --fw-sales 250000 \\
    --volume-threshold 75000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.all_data_archetypes_simulator_ae import load_artifacts
from model.forecast_engine_server import ForecastEngine
from model.marketshare_75k_simulation import NUM_WEEKS, generate_archetype_decay_curve

DEFAULT_ARTIFACTS = ROOT / "model" / "artifacts_75k"
DEFAULT_STREAMS = ROOT / "model" / "archetypes_artifacts" / "streams"
DEFAULT_SALES = ROOT / "model" / "archetypes_artifacts" / "sales"
DEFAULT_SONGS = ROOT / "model" / "archetypes_artifacts" / "songs"
DEFAULT_EOY = "2026-12-31"
LABEL_OWNERS = ("Atlantic Music Group", "Interscope/Geffen/A&M")


def _build_release(args: argparse.Namespace) -> dict:
    title = args.title or args.artist
    release = {
        "name": title,
        "artist": args.artist,
        "title": title,
        "label": args.label,
        "genre": args.genre,
        "date": args.date,
        "scenario": args.scenario,
        "fw_streams": float(args.fw_streams),
        "fw_sales": float(args.fw_sales),
        "fw_songs": float(args.fw_songs),
    }
    if args.cluster is not None:
        release["cluster"] = int(args.cluster)
    if args.fy_vol:
        release["fy_vol"] = float(args.fy_vol)
    comp_w1 = release["fw_streams"] + release["fw_sales"] + release["fw_songs"]
    if comp_w1 > 0:
        release["fw_vol"] = comp_w1
    elif args.fw_vol:
        release["fw_vol"] = float(args.fw_vol)
    return release


def _load_decay_triplet(streams_dir: Path, sales_dir: Path, songs_dir: Path):
    return (
        load_artifacts(str(streams_dir)),
        load_artifacts(str(sales_dir)),
        load_artifacts(str(songs_dir)),
    )


def _cy_weeks(drop_date: str, eoy: str) -> int:
    drop = pd.to_datetime(drop_date)
    end = pd.to_datetime(eoy)
    return min(max(0, (end - drop).days // 7 + 1), NUM_WEEKS)


def _print_decay_summary(release: dict, curve: list[float], eoy: str, *, weekly: bool) -> None:
    w1 = float(release["fw_streams"]) + float(release["fw_sales"]) + float(release["fw_songs"])
    weeks = _cy_weeks(release["date"], eoy)
    cy_total = sum(curve[:weeks])
    w1_curve = curve[0] if curve else 0.0

    print(f"Artist:     {release['artist']}")
    print(f"Release:    {release['name']}")
    print(f"Drop date:  {release['date']}")
    print(f"Scenario:   {release['scenario']}")
    print(f"W1 inputs:  streams={release['fw_streams']:,.0f}  sales={release['fw_sales']:,.0f}  songs={release['fw_songs']:,.0f}")
    print(f"W1 total:   {w1:,.0f} AE")
    print(f"Weeks in CY ({eoy[:4]}): {weeks}")
    print(f"CY AE sum:  {cy_total:,.0f}")
    if w1_curve > 0 and len(curve) > 1:
        print(f"W2 model:   {curve[1]:,.0f} AE  (W2/W1 = {curve[1] / w1_curve:.2%})")

    if weekly:
        print("\nWeek  AE")
        print("----  ----------")
        for i, v in enumerate(curve[:weeks], start=1):
            print(f"{i:4d}  {v:,.0f}")


def _weekly_ytd_table(ytd: pd.DataFrame, *, from_drop: str | None = None) -> pd.DataFrame:
    """Pivot unified YTD share for Atlantic and Interscope by week."""
    if ytd.empty:
        return pd.DataFrame()
    sub = ytd[ytd["Owner"].astype(str).isin(LABEL_OWNERS)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["Week Ending Date"] = pd.to_datetime(sub["Week Ending Date"])
    sub = sub.sort_values(["Week Ending Date", "Owner"])
    if from_drop:
        sub = sub[sub["Week Ending Date"] >= pd.to_datetime(from_drop)]

    wide = sub.pivot_table(
        index=["Week Ending Date", "Data_Type"],
        columns="Owner",
        values="Unified_YTD_Share",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide["Week Ending Date"] = wide["Week Ending Date"].dt.strftime("%Y-%m-%d")
    return wide


def _print_weekly_ytd_marketshare(ytd: pd.DataFrame, *, from_drop: str | None = None) -> None:
    wide = _weekly_ytd_table(ytd, from_drop=from_drop)
    if wide.empty:
        return
    amg_col = "Atlantic Music Group"
    int_col = "Interscope/Geffen/A&M"
    print("\n=== Weekly YTD marketshare (Atlantic vs Interscope) ===")
    hdr = f"{'Week ending':<12} {'Type':<9}"
    hdr += f" {'Atlantic YTD%':>14}" if amg_col in wide.columns else ""
    hdr += f" {'Interscope YTD%':>16}" if int_col in wide.columns else ""
    print(hdr)
    print("-" * len(hdr))
    for _, row in wide.iterrows():
        line = f"{row['Week Ending Date']:<12} {str(row['Data_Type']):<9}"
        if amg_col in wide.columns:
            v = row.get(amg_col)
            line += f" {float(v):>14.2f}" if pd.notna(v) else f" {'—':>14}"
        if int_col in wide.columns:
            v = row.get(int_col)
            line += f" {float(v):>16.2f}" if pd.notna(v) else f" {'—':>16}"
        print(line)


def cmd_decay(args: argparse.Namespace) -> int:
    release = _build_release(args)
    streams, sales, songs = _load_decay_triplet(
        Path(args.streams_dir), Path(args.sales_dir), Path(args.songs_dir)
    )
    curve = generate_archetype_decay_curve(release, streams, sales, songs)
    if not any(curve):
        print("ERROR: decay curve is all zeros — check artist/genre and artifact paths.", file=sys.stderr)
        return 1
    _print_decay_summary(release, curve, args.eoy, weekly=args.weekly)
    if args.json_out:
        weeks = _cy_weeks(release["date"], args.eoy)
        payload = {
            "release": release,
            "weekly_ae": curve[:weeks],
            "cy_ae_sum": sum(curve[:weeks]),
            "weeks_in_cy": weeks,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


def cmd_marketshare(args: argparse.Namespace) -> int:
    release = _build_release(args)
    engine = ForecastEngine(
        Path(args.artifacts_dir),
        Path(args.streams_dir),
        Path(args.sales_dir),
        Path(args.songs_dir),
    )
    out = engine.simulate(
        [release],
        e_score=args.e_score,
        volume_threshold=args.volume_threshold,
    )

    streams, sales, songs = _load_decay_triplet(
        Path(args.streams_dir), Path(args.sales_dir), Path(args.songs_dir)
    )
    curve = generate_archetype_decay_curve(release, streams, sales, songs)
    if not any(curve):
        print("ERROR: decay curve is all zeros — check artist/genre and artifact paths.", file=sys.stderr)
        return 1

    print("=== Artist decay ===")
    _print_decay_summary(release, curve, args.eoy, weekly=args.weekly)

    inj = pd.DataFrame(out.get("weekly_injections") or [])
    ytd = pd.DataFrame(out.get("unified_ytd") or [])
    label = release["name"]

    if not inj.empty and label in inj.columns:
        inj_col = pd.to_numeric(inj[label], errors="coerce").fillna(0.0)
        cy_inj = inj_col.sum()
        print(f"\n=== Marketshare injections ({label}) ===")
        print(f"CY injected AE sum (tracker): {cy_inj:,.0f}")
        if args.weekly and label in inj.columns:
            print("\nWeek ending       Injected AE")
            print("-------------     -----------")
            for _, row in inj.iterrows():
                v = float(row.get(label) or 0)
                if v > 0 or args.weekly_all:
                    print(f"{row['Week Ending Date']}  {v:,.0f}")

    if not ytd.empty:
        owner = release["label"]
        owner_key = owner
        sub = ytd[ytd["Owner"].astype(str) == owner_key].copy()
        if not sub.empty:
            act = sub[sub["Data_Type"].astype(str) == "Actual"]
            fc = sub[sub["Data_Type"].astype(str) == "Forecast"]
            print(f"\n=== Marketshare ({owner_key}) ===")
            if not act.empty:
                last_act = act.iloc[-1]
                print(
                    f"Last actual week:     {last_act['Week Ending Date']}  "
                    f"YTD={float(last_act['Unified_YTD_Share']):.2f}%  "
                    f"weekly={float(last_act['Active_Share']):.2f}%"
                )
            drop_ts = pd.to_datetime(release["date"])
            drop_rows = sub[pd.to_datetime(sub["Week Ending Date"]) >= drop_ts]
            if not drop_rows.empty:
                at_drop = drop_rows.iloc[0]
                print(
                    f"At/after drop week:   {at_drop['Week Ending Date']}  "
                    f"YTD={float(at_drop['Unified_YTD_Share']):.2f}%  "
                    f"weekly={float(at_drop['Active_Share']):.2f}%"
                )
            if not fc.empty:
                last = fc.iloc[-1]
                print(
                    f"EOY forecast week:    {last['Week Ending Date']}  "
                    f"YTD={float(last['Unified_YTD_Share']):.2f}%  "
                    f"weekly={float(last['Active_Share']):.2f}%"
                )
            print("(Use YTD % for cumulative share; weekly % is that week only.)")

    from_drop = release["date"] if args.ytd_from_drop else None
    if not args.no_weekly_ytd:
        _print_weekly_ytd_marketshare(ytd, from_drop=from_drop)

    if args.ytd_csv:
        wide = _weekly_ytd_table(ytd, from_drop=from_drop)
        if not wide.empty:
            wide.to_csv(args.ytd_csv, index=False)
            print(f"\nWrote weekly YTD table to {args.ytd_csv}")

    if args.json_out:
        payload = dict(out)
        wide = _weekly_ytd_table(ytd, from_drop=from_drop)
        if not wide.empty:
            payload["weekly_ytd_marketshare"] = wide.to_dict(orient="records")
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


def _add_release_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--artist", required=True, help="Artist name (used for archetype decay lookup)")
    p.add_argument("--title", default=None, help="Album/release title (display label; defaults to --artist)")
    p.add_argument("--label", required=True, help='e.g. "Atlantic Music Group" or "Interscope/Geffen/A&M"')
    p.add_argument("--genre", required=True, help="Genre (must match training taxonomy, e.g. Pop)")
    p.add_argument("--date", required=True, help="Release/drop date YYYY-MM-DD")
    p.add_argument("--scenario", default="Base", choices=["Base", "Bear", "Bull"])
    p.add_argument("--fw-streams", type=float, default=0.0, help="Week-1 streaming AE")
    p.add_argument("--fw-sales", type=float, default=0.0, help="Week-1 product AE")
    p.add_argument("--fw-songs", type=float, default=0.0, help="Week-1 song-sale AE")
    p.add_argument(
        "--fw-vol",
        type=float,
        default=0.0,
        help="Week-1 total AE (only if streams/sales/songs not set)",
    )
    p.add_argument("--fy-vol", type=float, default=0.0, help="Optional FY volume hint")
    p.add_argument("--cluster", type=int, default=None, help="Optional archetype cluster 0-3")
    p.add_argument("--eoy", default=DEFAULT_EOY, help="Calendar-year end date for CY sum (default 2026-12-31)")
    p.add_argument("--weekly", action="store_true", help="Print week-by-week AE")
    p.add_argument("--json-out", default=None, help="Write JSON results to this path")
    p.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS))
    p.add_argument("--streams-dir", default=str(DEFAULT_STREAMS))
    p.add_argument("--sales-dir", default=str(DEFAULT_SALES))
    p.add_argument("--songs-dir", default=str(DEFAULT_SONGS))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="75k marketshare: artist decay curve and/or full scenario simulation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_decay = sub.add_parser("decay", help="Archetype decay only (no market dilution)")
    _add_release_args(p_decay)
    p_decay.set_defaults(func=cmd_decay)

    p_ms = sub.add_parser("marketshare", help="Full marketshare sim + decay summary")
    _add_release_args(p_ms)
    p_ms.add_argument("--e-score", type=float, default=0.82, help="Conformal band width on forecast YTD")
    p_ms.add_argument(
        "--volume-threshold",
        type=float,
        default=20_000,
        help="Catalog floor AE subtracted from 75k-book weekly volume "
        "(inject max(0, weekly_AE - floor); default 20000)",
    )
    p_ms.add_argument(
        "--weekly-all",
        action="store_true",
        help="With --weekly, print all injection weeks (including zeros)",
    )
    p_ms.add_argument(
        "--no-weekly-ytd",
        action="store_true",
        help="Skip the weekly Atlantic vs Interscope YTD table",
    )
    p_ms.add_argument(
        "--ytd-from-drop",
        action="store_true",
        help="Only print YTD weeks on/after the release drop date",
    )
    p_ms.add_argument(
        "--ytd-csv",
        default=None,
        help="Write weekly YTD table (Atlantic + Interscope) to this CSV path",
    )
    p_ms.set_defaults(func=cmd_marketshare)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
