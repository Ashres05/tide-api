#!/usr/bin/env python3
"""
Streamlit test UI for the 75k archetypal simulator.

  streamlit run streamlit_marketshare_75k.py

Requires: streamlit, matplotlib, pandas (same env as training). Train artifacts first:

  python train_marketshare_artifacts.py --artifacts-dir ./artifacts_75k

Visual style follows `75k_parlay.ipynb` (YTD cumulative share, conformal band, label colors).

Primary input is a **fill-in table** (artist, label, dates, volumes). Optional **Advanced JSON** for pasted notebooks.

Calendar text (advanced) accepts **JSON** or **Python literals** (`ast.literal_eval`).
"""

from __future__ import annotations

import ast
import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Reuse engine loader + simulation (no HTTP server)
from forecast_engine_server import ForecastEngine

_DEFAULT_CALENDAR = """[
  {"name": "Don Toliver", "label": "Atlantic Music Group", "genre": "Rap", "known_streams": [138000, 100000, 81000, 70000, 68000, 62000, 58000, 57000, 53000, 45000], "known_sales": [31000, 1400, 900, 1350, 586, 500, 400, 300, 2300, 2800], "known_songs": [170, 113, 89, 92, 95, 85, 76, 76, 71, 58], "date": "2026-01-20"},
  {"name": "Forrest Frank", "label": "Atlantic Music Group", "genre": "Christian", "fw_vol": 30000, "scenario": "Base", "date": "2026-07-03"},
  {"name": "A Boogie wit a hoodie", "label": "Atlantic Music Group", "genre": "Rap", "fw_vol": 15000, "scenario": "Base", "date": "2026-04-24"},
  {"name": "Gunna", "label": "Atlantic Music Group", "genre": "Rap", "fw_vol": 75000, "scenario": "Base", "date": "2026-07-03"},
  {"name": "Kehlani", "label": "Atlantic Music Group", "genre": "R&B/Hip-Hop", "fw_vol": 50000, "scenario": "Base", "date": "2026-08-14"},
  {"name": "Pooh Shiesty", "label": "Atlantic Music Group", "genre": "Rap", "fw_vol": 35000, "scenario": "Base", "date": "2026-07-24"},
  {"name": "YFN Lucci", "label": "Atlantic Music Group", "genre": "Rap", "fw_vol": 25000, "scenario": "Base", "date": "2026-05-08"},
  {"name": "Alex Warren", "label": "Atlantic Music Group", "genre": "Pop", "fw_vol": 80000, "scenario": "Base", "date": "2026-04-17"},
  {"name": "Coldplay", "label": "Atlantic Music Group", "genre": "Rock", "fw_vol": 70000, "scenario": "Base", "date": "2026-06-12"},
  {"name": "ROSÉ", "label": "Atlantic Music Group", "genre": "K-Pop", "fw_vol": 60000, "scenario": "Base", "date": "2026-08-07"},
  {"name": "Bailey Zimmerman", "label": "Atlantic Music Group", "genre": "Country", "fw_vol": 32000, "scenario": "Base", "date": "2026-03-27"},
  {"name": "BTS", "label": "Interscope/Geffen/A&M", "genre": "K-Pop", "fw_vol": 636000, "scenario": "Base", "known_streams": [104000, 71000, 47000], "known_sales": [535000, 114000, 57000], "known_songs": [15000, 7000, 1400], "date": "2026-03-20"},
  {"name": "Olivia Rodrigo", "label": "Interscope/Geffen/A&M", "genre": "Pop", "fw_vol": 700000, "fy_vol": 2000000, "scenario": "Base", "date": "2026-04-24"}
]"""

# Lightweight hypothetical for quick edits / small screens
_MINIMAL_CALENDAR = """[
  {"name": "Atlantic release", "label": "Atlantic Music Group", "genre": "Pop", "fw_vol": 150000, "scenario": "Base", "date": "2026-07-01"},
  {"name": "Interscope release", "label": "Interscope/Geffen/A&M", "genre": "Pop", "fw_vol": 200000, "scenario": "Base", "date": "2026-09-01"}
]"""


def parse_calendar_text(text: str) -> list:
    """
    Accept strict JSON (double quotes) or Python literals (single-quoted dicts / lists),
    matching how calendars are often copied from notebooks.
    """
    text = text.strip()
    if not text:
        raise ValueError("Calendar is empty.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as je:
        try:
            data = ast.literal_eval(text)
        except (ValueError, SyntaxError) as pe:
            raise ValueError(
                "Could not parse calendar. Use JSON (double quotes) or a Python list of dicts "
                f"(single quotes OK). JSON error: {je}; literal error: {pe}"
            ) from pe
    if not isinstance(data, list):
        raise ValueError("Calendar must be a list of release objects.")
    return data


LABEL_OPTIONS = ["Atlantic Music Group", "Interscope/Geffen/A&M", "Other"]
SCENARIO_OPTIONS = ["Base", "Bear", "Bull"]

_EDITOR_COLUMNS = [
    "artist_name",
    "label",
    "genre",
    "drop_date",
    "scenario",
    "cluster",
    "fw_vol",
    "fy_vol",
    "known_weekly_vols",
    "known_streams", 
    "known_sales",   
    "known_songs",
]


def releases_list_to_editor_df(releases: list) -> pd.DataFrame:
    """Turn API-style release dicts into the editable table (for templates + JSON import)."""
    rows: List[dict] = []
    for r in releases:
        kv = r.get("known_vols") or []
        d_raw = r.get("date")
        if d_raw:
            try:
                dt = pd.to_datetime(d_raw)
                dval: Optional[date] = dt.date()
            except Exception:
                dval = None
        else:
            dval = None
        cl = r.get("cluster")
        if cl is None or cl == "Auto":
            cluster_str = ""
        else:
            cluster_str = str(int(cl)) if str(cl).replace(".", "").isdigit() else str(cl)
        fw_disp: Optional[float]
        if kv:
            fw_disp = float(kv[0])
        elif r.get("fw_vol") is not None:
            fw_disp = float(r["fw_vol"])
        else:
            fw_disp = None
        fy_raw = r.get("fy_vol")
        rows.append(
            {
                "artist_name": r.get("name") or "",
                "label": r.get("label") or "Atlantic Music Group",
                "genre": r.get("genre") or "",
                "drop_date": dval,
                "scenario": r.get("scenario") or "Base",
                "cluster": cluster_str,
                "fw_vol": fw_disp,
                "fy_vol": float(fy_raw) if fy_raw is not None else None,
                "known_weekly_vols": ", ".join(str(int(x)) for x in kv) if kv else "",
                "known_streams": ", ".join(str(int(x)) for x in r.get("known_streams", [])) if r.get("known_streams") else "",
                "known_sales": ", ".join(str(int(x)) for x in r.get("known_sales", [])) if r.get("known_sales") else "",
                "known_songs": ", ".join(str(int(x)) for x in r.get("known_songs", [])) if r.get("known_songs") else "",
            }
        )
    return pd.DataFrame(rows, columns=_EDITOR_COLUMNS)


def _default_editor_df() -> pd.DataFrame:
    return releases_list_to_editor_df(parse_calendar_text(_DEFAULT_CALENDAR))


def _minimal_editor_df() -> pd.DataFrame:
    return releases_list_to_editor_df(parse_calendar_text(_MINIMAL_CALENDAR))


def _blank_editor_df(num_rows: int = 1) -> pd.DataFrame:
    """Empty rows with sensible defaults so the grid stays usable."""
    placeholder = date(2026, 6, 1)
    rows = []
    for _ in range(max(1, int(num_rows))):
        rows.append(
            {
                "artist_name": "",
                "label": "Atlantic Music Group",
                "genre": "",
                "drop_date": placeholder,
                "scenario": "Base",
                "cluster": "",
                "fw_vol": None,
                "fy_vol": None,
                "known_weekly_vols": "",
                "known_streams": "",
                "known_sales": "",
                "known_songs": "",
            }
        )
    return pd.DataFrame(rows, columns=_EDITOR_COLUMNS)


def _parse_known_vols_cell(s: Any) -> List[float]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    text = str(s).strip()
    if not text:
        return []
    out: List[float] = []
    for part in text.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _row_to_release(row: pd.Series) -> Optional[dict]:
    name = str(row.get("artist_name", "") or "").strip()
    if not name:
        return None
    label = str(row.get("label") or "Atlantic Music Group").strip()
    genre = str(row.get("genre", "") or "").strip()
    scen = str(row.get("scenario") or "Base").strip()
    if scen not in SCENARIO_OPTIONS:
        scen = "Base"

    d = row.get("drop_date")
    if d is None or (isinstance(d, float) and pd.isna(d)):
        raise ValueError(f"Release '{name}': drop date is required.")
    if hasattr(d, "strftime"):
        date_str = d.strftime("%Y-%m-%d")
    else:
        date_str = str(pd.to_datetime(d).date())

    out: dict[str, Any] = {
        "name": name,
        "label": label,
        "genre": genre,
        "scenario": scen,
        "date": date_str,
    }

    # fix to not require fw vol if other knowns are given
    kv = _parse_known_vols_cell(row.get("known_weekly_vols"))
    if kv: out["known_vols"] = kv

    k_str = _parse_known_vols_cell(row.get("known_streams"))
    if k_str: out["known_streams"] = k_str

    k_sal = _parse_known_vols_cell(row.get("known_sales"))
    if k_sal: out["known_sales"] = k_sal

    k_son = _parse_known_vols_cell(row.get("known_songs"))
    if k_son: out["known_songs"] = k_son

    # 2. Check what was provided
    fw = row.get("fw_vol")
    has_fw = fw is not None and not (isinstance(fw, float) and pd.isna(fw))
    has_granular = bool(k_str or k_sal or k_son)
    
    # 3. Crash only if absolutely everything is blank
    if not has_fw and not kv and not has_granular:
        raise ValueError(f"Release '{name}': enter first-week volume (fw_vol), known weekly volumes, or granular actuals.")

    # 4. Safely set or dynamically calculate the total First Week Volume
    if has_fw:
        out["fw_vol"] = float(fw)
    else:
        # If fw_vol was left blank, mathematically infer it from the arrays!
        if kv:
            first_vol = float(kv[0])
        else:
            str_vol = float(k_str[0]) if k_str else 0.0
            sal_vol = float(k_sal[0]) if k_sal else 0.0
            son_vol = float(k_son[0]) if k_son else 0.0
            first_vol = str_vol + sal_vol + son_vol
        
        out["fw_vol"] = float(first_vol)

    fy = row.get("fy_vol")
    if fy is not None and not (isinstance(fy, float) and pd.isna(fy)):
        try:
            out["fy_vol"] = float(fy)
        except (TypeError, ValueError):
            pass

    cs = row.get("cluster")
    if cs is not None and not (isinstance(cs, float) and pd.isna(cs)):
        cst = str(cs).strip()
        if cst and cst.lower() != "auto":
            try:
                out["cluster"] = int(float(cst))
            except ValueError:
                pass

    return out


def editor_df_to_releases(df: pd.DataFrame) -> list:
    """Convert the fill-in table to `release_calendar` dicts for the engine."""
    if df is None or df.empty:
        raise ValueError("Add at least one release row.")
    releases: list = []
    errors: list[str] = []
    for j, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            r = _row_to_release(row)
            if r is not None:
                releases.append(r)
        except ValueError as e:
            errors.append(f"Row {j}: {e}")
    if errors:
        raise ValueError("\n".join(errors))
    if not releases:
        raise ValueError("No valid release rows (need artist name and date at minimum).")
    return releases


def try_export_calendar_json(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    """Returns (json_string, None) or (None, error_message)."""
    try:
        rel = editor_df_to_releases(df)
        return json.dumps(rel, indent=2), None
    except ValueError as e:
        return None, str(e)


_LABEL_COLORS = {
    "Atlantic Music Group": "#DC143C",
    "Interscope/Geffen/A&M": "#4169E1",
}


def _short_owner_label(owner: str) -> str:
    """First segment before '/' or first word — matches notebook-style EOY tags."""
    s = owner.split("/")[0].strip()
    return s.split()[0] if s else owner


def _configure_matplotlib() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    elif "seaborn-whitegrid" in plt.style.available:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams["figure.figsize"] = (14, 6)


def _records_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "Week Ending Date" in df.columns:
        df["Week Ending Date"] = pd.to_datetime(df["Week Ending Date"])
    return df


@st.cache_resource
def _load_engine(artifacts_dir: str, streams_dir: str, sales_dir: str, songs_dir: str) -> ForecastEngine:
    return ForecastEngine(Path(artifacts_dir), Path(streams_dir), Path(sales_dir), Path(songs_dir))


def _plot_ytd_market_share(df: pd.DataFrame, *, show_eoy: bool = True) -> plt.Figure:
    """Notebook-style: banked actuals, dashed forecast, conformal band; optional EOY % callouts."""
    fig, ax = plt.subplots(figsize=(14, 7))
    if df.empty or "Owner" not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    need = {"Week Ending Date", "Owner", "Unified_YTD_Share"}
    if not need.issubset(df.columns):
        ax.text(0.5, 0.5, "Missing YTD columns in result", ha="center", va="center", transform=ax.transAxes)
        return fig

    split = None
    if "Data_Type" in df.columns:
        actual = df[df["Data_Type"] == "Actual"]
        if not actual.empty:
            split = actual["Week Ending Date"].max()

    owners_sorted = sorted(df["Owner"].unique())
    for i, owner in enumerate(owners_sorted):
        sub = df[df["Owner"] == owner].sort_values("Week Ending Date")
        color = _LABEL_COLORS.get(owner, "#333333")
        if "Data_Type" in sub.columns:
            act = sub[sub["Data_Type"] == "Actual"]
            fc = sub[sub["Data_Type"] == "Forecast"]
            if not act.empty:
                ax.plot(
                    act["Week Ending Date"],
                    act["Unified_YTD_Share"],
                    color=color,
                    linewidth=3.2,
                    label=f"{owner} (banked actuals)",
                )
            if not fc.empty:
                if not act.empty:
                    bridge = pd.concat([act.tail(1), fc.head(1)], ignore_index=True)
                    ax.plot(
                        bridge["Week Ending Date"],
                        bridge["Unified_YTD_Share"],
                        color=color,
                        linestyle="--",
                        linewidth=2,
                    )
                ax.plot(
                    fc["Week Ending Date"],
                    fc["Unified_YTD_Share"],
                    color=color,
                    linestyle="--",
                    linewidth=2.5,
                    label=f"{owner} (projected YTD)",
                )
                if {"YTD_Share_Lower", "YTD_Share_Upper"}.issubset(fc.columns):
                    ax.fill_between(
                        fc["Week Ending Date"],
                        fc["YTD_Share_Lower"],
                        fc["YTD_Share_Upper"],
                        color=color,
                        alpha=0.12,
                    )
        else:
            ax.plot(sub["Week Ending Date"], sub["Unified_YTD_Share"], color=color, linewidth=2.5, label=owner)

        if show_eoy and not sub.empty:
            last = sub.iloc[-1]
            eoy_date = last["Week Ending Date"]
            eoy_val = float(last["Unified_YTD_Share"])
            short = _short_owner_label(owner)
            ax.annotate(
                f"{short} EOY: {eoy_val:.2f}%",
                xy=(eoy_date, eoy_val),
                xytext=(12, 14 + i * 26),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color, lw=1.5, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.0, shrinkA=0, shrinkB=4),
            )

    if show_eoy:
        ax.margins(x=0.06)

    if split is not None:
        ax.axvline(split, color="red", linestyle=":", linewidth=2, label=f"Forecast start ({split.date()})")

    ax.set_title("2026 YTD market share (archetypal simulation)", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Week ending")
    ax.set_ylabel("Cumulative YTD share")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _plot_weekly_injections(inj: pd.DataFrame) -> plt.Figure | None:
    if inj.empty:
        return None
    date_col = "Week Ending Date"
    if date_col not in inj.columns:
        return None
    value_cols = [c for c in inj.columns if c != date_col]
    if not value_cols:
        return None
    long_df = inj.melt(id_vars=[date_col], var_name="Release", value_name="Weekly AE volume")
    long_df = long_df[long_df["Weekly AE volume"] > 0]
    if long_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No non-zero weekly injection rows", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(14, 6))
    for release, g in long_df.groupby("Release"):
        g = g.sort_values(date_col)
        ax.plot(g[date_col], g["Weekly AE volume"], marker="o", linewidth=1.5, label=release, alpha=0.85)
    ax.set_title("Injected weekly volume by release (archetype decay)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Week ending")
    ax.set_ylabel("Weekly volume (AE)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="75k Marketshare Simulator", layout="wide")
    _configure_matplotlib()

    st.title("75k archetypal marketshare simulator")
    st.caption("Loads parquet/JSON artifacts from training; runs the same engine as `forecast_engine_server`.")

    with st.sidebar:
        st.header("Artifacts")
        # 1. The original 75k YTD models
        default_art = Path(__file__).resolve().parent / "artifacts_75k"
        art_path = st.text_input("75k Artifacts directory", value=str(default_art))
        streams_dir = st.text_input("Decay Artifacts (Streams)", value=str(Path(__file__).resolve().parent / "archetypes_artifacts" / "streams"))
        sales_dir = st.text_input("Decay Artifacts (Sales)", value=str(Path(__file__).resolve().parent / "archetypes_artifacts" / "sales"))
        songs_dir = st.text_input("Decay Artifacts (Songs)", value=str(Path(__file__).resolve().parent / "archetypes_artifacts" / "songs"))
        # 2. The NEW dynamic decay models (Streams)
        default_streams = Path(__file__).resolve().parent / "archetypes_artifacts" / "streams"
        streams_dir = st.text_input("Decay Artifacts directory (Streams)", value=str(default_streams))
        st.header("Simulation parameters")
        e_score = st.number_input("E-score (conformal band on forecast)", value=0.71, min_value=0.0, max_value=5.0, step=0.01)
        vol_threshold = st.number_input("Injection threshold (AE / week to dilute market)", value=75000.0, min_value=0.0, step=1000.0)
        enrich = st.checkbox("Auto-enrich product ratios (artist history)", value=True)
        enrich_w2 = st.checkbox(
            "Adjust week-2+ tail using full65 W2/W1 (median per artist)",
            value=True,
            help="Requires `artist_w2_retention.json` from training with full65+.csv.",
        )
        st.header("Global Peak Tuning")
        st.caption("Controls the historical subset used for all releases in the table.")
        st.header("Global Peak Tuning (Decay Engine)")
        peak_sim_log_radius = st.number_input("Log10 radius", min_value=0.0, max_value=2.0, value=0.15, step=0.05)
        peak_sim_min_subset_releases = st.number_input("Min subset releases", min_value=1, value=5, step=1)
        peak_sim_spread_threshold_log_std = st.number_input("If log-std exceeds", min_value=0.0, value=0.15, step=0.05)
        peak_sim_min_artist_releases = st.number_input("Min artist releases to filter", min_value=1, value=20, step=1)
        show_eoy = st.checkbox("Show EOY % on YTD chart", value=True)

    st.subheader("Release calendar")
    input_mode = st.radio(
        "How do you want to enter releases?",
        ["Fill-in table (recommended)", "Advanced: JSON or Python"],
        horizontal=True,
        key="calendar_input_mode",
    )

    edited_df: Optional[pd.DataFrame] = None

    if input_mode.startswith("Fill"):
        if "cal_df" not in st.session_state:
            st.session_state.cal_df = _default_editor_df()

        with st.container(border=True):
            st.markdown("##### Start from a template")
            st.caption(
                "Replace the whole table with a starter, then edit freely. Use **+** at the bottom of the grid "
                "to add releases or remove rows you don’t need — works for small hypotheticals or large calendars."
            )
            pc1, pc2 = st.columns([2.2, 1])
            with pc1:
                preset = st.selectbox(
                    "Template",
                    [
                        "Full example (12 releases)",
                        "Minimal — 2 hypothetical drops",
                        "Blank — 1 empty row",
                        "Blank — 3 empty rows",
                        "Blank — 8 empty rows",
                    ],
                    label_visibility="collapsed",
                    key="calendar_preset_select",
                )
            with pc2:
                load_tpl = st.button("Load template", use_container_width=True, type="secondary")

            if load_tpl:
                if preset.startswith("Full"):
                    st.session_state.cal_df = _default_editor_df()
                elif preset.startswith("Minimal"):
                    st.session_state.cal_df = _minimal_editor_df()
                elif "1 empty" in preset:
                    st.session_state.cal_df = _blank_editor_df(1)
                elif "3 empty" in preset:
                    st.session_state.cal_df = _blank_editor_df(3)
                else:
                    st.session_state.cal_df = _blank_editor_df(8)
                st.rerun()

            with st.expander("Import / export JSON (different hypothetical or share with teammates)"):
                st.caption(
                    "**Import** overwrites the table. **Export** needs every row to have artist, date, and volume."
                )
                i1, i2 = st.columns(2)
                with i1:
                    up = st.file_uploader("Upload a `.json` array", type=["json"], key="cal_json_upload")
                    if st.button("Import file → table", key="btn_import_file"):
                        if up is None:
                            st.warning("Choose a file first.")
                        else:
                            try:
                                raw = up.getvalue().decode("utf-8")
                                st.session_state.cal_df = releases_list_to_editor_df(parse_calendar_text(raw))
                                st.success("Imported.")
                                st.rerun()
                            except Exception as ex:
                                st.error(str(ex))
                with i2:
                    paste = st.text_area("Or paste JSON / Python here", height=100, key="paste_json_cal", placeholder="[ {...}, ... ]")
                    if st.button("Import paste → table", key="btn_import_paste"):
                        try:
                            st.session_state.cal_df = releases_list_to_editor_df(parse_calendar_text(paste))
                            st.success("Imported.")
                            st.rerun()
                        except Exception as ex:
                            st.error(str(ex))

                exp_payload, exp_err = try_export_calendar_json(st.session_state.cal_df)
                if exp_payload:
                    st.download_button(
                        "Download current table as JSON",
                        data=exp_payload,
                        file_name="calendar.json",
                        mime="application/json",
                        key="dl_cal_json",
                    )
                else:
                    st.caption(f"Export needs valid rows: {exp_err or ''}")

        n_rows = len(st.session_state.cal_df)
        st.caption(
            f"**{n_rows} row(s)** · **First-week AE** required unless **Known weekly vols** is set. "
            "**Cluster** blank = Auto (genre + label priors)."
        )

        edited_df = st.data_editor(
            st.session_state.cal_df,
            column_config={
                "artist_name": st.column_config.TextColumn(
                    "Artist / release",
                    help="Display name for this drop",
                    width="medium",
                ),
                "label": st.column_config.SelectboxColumn(
                    "Label",
                    options=LABEL_OPTIONS,
                    required=True,
                    width="large",
                ),
                "genre": st.column_config.TextColumn(
                    "Genre",
                    help="Used when cluster is Auto",
                    width="small",
                ),
                "drop_date": st.column_config.DateColumn(
                    "Drop date",
                    format="YYYY-MM-DD",
                    width="small",
                ),
                "scenario": st.column_config.SelectboxColumn(
                    "Scenario",
                    options=SCENARIO_OPTIONS,
                    width="small",
                ),
                "cluster": st.column_config.TextColumn(
                    "Cluster",
                    help="0–3, or leave blank for Auto",
                    width="small",
                ),
                "fw_vol": st.column_config.NumberColumn(
                    "First-week AE",
                    help="Ignored when known weekly vols are set",
                    min_value=0.0,
                    format="%d",
                    width="small",
                ),
                "fy_vol": st.column_config.NumberColumn(
                    "FY target (optional)",
                    min_value=0.0,
                    format="%d",
                    width="small",
                ),
                "known_weekly_vols": st.column_config.TextColumn(
                    "Known weekly vols (optional)",
                    help="Comma-separated: 170000, 101000, 82000, …",
                    width="large",
                ),
                "known_streams": st.column_config.TextColumn(
                    "Known Streams",
                    help="Granular streams: 50000, 45000...",
                    width="medium",
                ),
                "known_sales": st.column_config.TextColumn(
                    "Known Sales",
                    help="Granular sales: 250000, 15000...",
                    width="medium",
                ),
                "known_songs": st.column_config.TextColumn(
                    "Known Songs",
                    help="Granular songs: 2000, 1000...",
                    width="medium",
                ),
            },
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            height=min(700, 120 + n_rows * 42),
        )
        st.session_state.cal_df = edited_df
    else:
        calendar_json = st.text_area(
            "JSON **or** Python literal — `name`, `label`, `genre`, `fw_vol`, `scenario`, `date`; optional `cluster`, `fy_vol`, `known_vols`",
            value=_DEFAULT_CALENDAR,
            height=320,
            key="calendar_json_advanced",
        )

    run = st.button("Run simulation", type="primary")

    if not run:
        st.info(
            "Fill in the table (or use **Advanced**), then click **Run simulation**."
            if input_mode.startswith("Fill")
            else "Paste JSON/Python, then click **Run simulation**."
        )
        return

    try:
        if input_mode.startswith("Fill"):
            calendar = editor_df_to_releases(edited_df if edited_df is not None else st.session_state.cal_df)
        else:
            calendar = parse_calendar_text(calendar_json)
    except ValueError as e:
        st.error(str(e))
        return

    path = Path(art_path).expanduser().resolve()
    if not (path / "df_full.parquet").exists():
        st.error(f"Missing `df_full.parquet` under {path}. Run `train_marketshare_artifacts.py` first.")
        return

    logging.getLogger("marketshare_75k_simulation").setLevel(logging.WARNING)
    for release in calendar:
        release["peak_sim_log_radius"] = peak_sim_log_radius
        release["peak_sim_min_subset_releases"] = peak_sim_min_subset_releases
        release["peak_sim_spread_threshold_log_std"] = peak_sim_spread_threshold_log_std
        release["peak_sim_min_artist_releases"] = peak_sim_min_artist_releases
    try:
        engine = _load_engine(str(path), str(streams_dir), str(sales_dir), str(songs_dir)) 
        
        out = engine.simulate(
            calendar,
            e_score=e_score,
            volume_threshold=vol_threshold,
        )
    except Exception as e:
        st.exception(e)
        return

    df_ytd = _records_to_df(out.get("unified_ytd") or [])
    df_inj = _records_to_df(out.get("weekly_injections") or [])

    tab1, tab2, tab3 = st.tabs(["YTD market share", "Weekly volumes", "EOY snapshot"])

    with tab1:
        fig = _plot_ytd_market_share(df_ytd, show_eoy=show_eoy)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with tab2:
        fig2 = _plot_weekly_injections(df_inj)
        if fig2 is not None:
            st.pyplot(fig2, clear_figure=True)
            plt.close(fig2)
            
        if not df_inj.empty:
            # 1. Create a display copy sorted by date
            display_df = df_inj.sort_values("Week Ending Date").copy()
            
            # 2. Format the dates as strings to allow our text row at the bottom
            display_df["Week Ending Date"] = display_df["Week Ending Date"].dt.strftime("%Y-%m-%d")
            
            # 3. Calculate a "Weekly Grand Total" column across all individual projects
            numeric_cols = display_df.select_dtypes(include=["number"]).columns
            display_df["Grand Total (All Releases)"] = display_df[numeric_cols].sum(axis=1)
            
            # 4. Calculate the CY sum for EACH project AND the new Grand Total column
            totals = display_df.select_dtypes(include=["number"]).sum()
            total_row = pd.DataFrame([totals])
            total_row["Week Ending Date"] = "TOTAL CY VOLUME"
            
            # 5. Append the totals row to the bottom of the table
            display_df = pd.concat([display_df, total_row], ignore_index=True)
            
            st.dataframe(display_df, use_container_width=True, height=400)

    with tab3:
        if df_ytd.empty:
            st.write("No rows.")
        else:
            last_dt = df_ytd["Week Ending Date"].max()
            snap = df_ytd[df_ytd["Week Ending Date"] == last_dt].copy()
            
            st.markdown(f"**Last week in simulation:** `{last_dt.date()}`")
            
            # --- NEW: Extract and display all three EOY Volumes ---
            if "Cum_Numerator" in snap.columns and "Cum_Denominator" in snap.columns:
                st.markdown("### Expected EOY Volumes (AE)")
                
                # Create 3 columns instead of 2 for the metric cards
                m1, m2, m3 = st.columns(3)
                
                amg_row = snap[snap["Owner"] == "Atlantic Music Group"]
                int_row = snap[snap["Owner"] == "Interscope/Geffen/A&M"]
                
                # Extract label volumes
                amg_vol = (amg_row["Cum_Numerator"].values[0] /100) if not amg_row.empty else 0
                int_vol = (int_row["Cum_Numerator"].values[0] /100) if not int_row.empty else 0
                
                # Extract total market volume (Cum_Denominator is identical for both rows, so we safely pull from the first available)
                total_market_vol = snap["Cum_Denominator"].values[0] if not snap.empty else 0
                
                # Display large metric cards with comma formatting
                m1.metric("Atlantic Music Group", f"{int(amg_vol):,}")
                m2.metric("Interscope/Geffen/A&M", f"{int(int_vol):,}")
                m3.metric("Total Market", f"{int(total_market_vol):,}")
                
                st.write("---") 
                
                # Add the cleanly formatted volumes to the dataframe as well
                snap["Expected EOY Volume"] = (snap["Cum_Numerator"] / 100).apply(lambda x: f"{int(x):,}")
                display_cols = [
                    "Owner", "Week Ending Date", "Unified_YTD_Share", 
                    "Expected EOY Volume", "YTD_Share_Lower", "YTD_Share_Upper", "Data_Type"
                ]
            else:
                display_cols = [
                    "Owner", "Week Ending Date", "Unified_YTD_Share", 
                    "YTD_Share_Lower", "YTD_Share_Upper", "Data_Type"
                ]

            # Filter the dataframe to only show the relevant columns
            final_cols = [c for c in display_cols if c in snap.columns]
            st.dataframe(snap[final_cols], use_container_width=True)

            if not df_inj.empty:
                st.markdown("### Individual Project CY Volumes")
                
                # 1. Sum all numeric columns (isolates your simulated releases)
                project_sums = df_inj.select_dtypes(include=["number"]).sum()
                proj_df = project_sums.reset_index()
                proj_df.columns = ["Project / Release", "Expected CY Volume"]
                
                # 2. Extract the Label AND Date for each project from your original input calendar
                label_map = {r.get("name", f"Release_{i+1}"): r.get("label", "Unknown") for i, r in enumerate(calendar)}
                date_map = {r.get("name", f"Release_{i+1}"): r.get("date", "Unknown") for i, r in enumerate(calendar)}
                
                proj_df["Label"] = proj_df["Project / Release"].map(label_map)
                proj_df["Drop Date"] = proj_df["Project / Release"].map(date_map)
                
                # 3. Sort by volume in descending order (BEFORE formatting as text)
                proj_df = proj_df.sort_values(by="Expected CY Volume", ascending=False).reset_index(drop=True)
                
                # 4. Format the numbers with commas
                proj_df["Expected CY Volume"] = proj_df["Expected CY Volume"].apply(lambda x: f"{int(x):,}")
                
                # 5. Reorder the columns cleanly and display!
                proj_df = proj_df[["Project / Release", "Label", "Drop Date", "Expected CY Volume"]]
                st.dataframe(proj_df, use_container_width=True)

    with st.expander("Raw JSON response"):
        st.json({"unified_ytd_rows": len(df_ytd), "weekly_injection_rows": len(df_inj)})


if __name__ == "__main__":
    main()
