"""
75k parlay simulation core: archetypal decay and enrichment.

Uses K-Means cluster shapes, genre/label archetype priors (DISTRIBUTIONS), per-artist
history (artist_dna_lookup), Bear/Base/Bull multipliers (ARCHETYPE_MULTIPLIERS), optional
physical-product adjustment (GLOBAL_PRODUCT_COEF + artist_profile_dict), and release
inputs (fw_vol, fy_vol, scenario, cluster, dates, known_vols).

Extracted from `75k_parlay.ipynb`. Used by the training job and the forecast engine.
"""

from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Static coefficients & tables (also persisted in artifacts) ---

GLOBAL_PRODUCT_COEF = -0.61

# Persisted in training metadata; bounds product-driven adjustment in M52 space (see notebook lineage).
PRODUCT_M52_PENALTY_CAP = 1.0

# When full65+ supplies median W2/W1 per artist, blend toward it; alpha = min(1, n_releases / K).
W2_RETENTION_BLEND_K = 4.0

ARCHETYPE_MULTIPLIERS = {
    0: {"Bear": 3.35, "Base": 6.20, "Bull": 7.72},
    1: {"Bear": 2.68, "Base": 4.34, "Bull": 5.14},
    2: {"Bear": 3.20, "Base": 5.68, "Bull": 6.73},
    3: {"Bear": 6.74, "Base": 10.85, "Bull": 13.14},
}

DISTRIBUTIONS = {
    "Genre": {
        "Christian": [100.0, 0.0, 0.0, 0.0],
        "Country": [70.6, 23.5, 0.0, 5.9],
        "EDM": [66.7, 33.3, 0.0, 0.0],
        "Folk": [33.3, 50.0, 0.0, 16.7],
        "Jazz": [100.0, 0.0, 0.0, 0.0],
        "K-Pop": [81.8, 9.1, 9.1, 0.0],
        "Latin": [0.0, 71.4, 0.0, 28.6],
        "Latin Pop": [100.0, 0.0, 0.0, 0.0],
        "Metal": [100.0, 0.0, 0.0, 0.0],
        "Pop": [79.4, 17.6, 0.0, 2.9],
        "R&B/Hip-Hop": [50.0, 40.9, 0.0, 9.1],
        "Rap": [54.3, 43.1, 0.0, 2.6],
        "Reggaeton": [0.0, 100.0, 0.0, 0.0],
        "Religious": [100.0, 0.0, 0.0, 0.0],
        "Rock": [90.0, 10.0, 0.0, 0.0],
    },
    "Label": {
        "Atlantic Music Group": [35.07, 8.21, 18.66, 38.06],
        "Interscope/Geffen/A&M": [9.86, 14.08, 50.70, 25.35],
    },
}


def get_weighted_archetype_curve(cluster_weights: Dict[Any, float], num_weeks: int = 52) -> np.ndarray:
    t = np.arange(0, num_weeks)
    combined_raw = np.zeros(num_weeks, dtype=float)
    shapes = {
        0: (t + 1) ** -0.5058,
        1: np.exp(-0.2258 * t),
        2: (t + 1) ** -0.6484,
        3: np.exp(-0.0482 * t),
    }
    for c_id, weight in cluster_weights.items():
        c_int = int(c_id) if not isinstance(c_id, str) else c_id
        if c_int in shapes:
            combined_raw += shapes[c_int] * float(weight)
    if combined_raw.sum() == 0:
        return shapes[0]
    return combined_raw


def get_inferred_cluster(release_dict: dict) -> int:
    c = release_dict.get("cluster")
    if c is not None and c != "Auto" and c != "DNA Blend":
        if isinstance(c, int) or (isinstance(c, str) and str(c).isdigit()):
            return int(c)
    genre = release_dict.get("Genre") or release_dict.get("genre")
    label = release_dict.get("Label") or release_dict.get("label")
    genre_dist = np.array(DISTRIBUTIONS["Genre"].get(genre, [25.0, 25.0, 25.0, 25.0]))
    label_dist = np.array(DISTRIBUTIONS["Label"].get(label, [25.0, 25.0, 25.0, 25.0]))
    joint_prob = genre_dist * label_dist
    if joint_prob.sum() == 0:
        return 0
    return int(np.argmax(joint_prob))


def _adjust_tail_weights_for_empirical_w2(
    tw: np.ndarray,
    total_tail_vol: float,
    fw_vol: float,
    release_dict: dict,
) -> np.ndarray:
    """
    Blend the archetype tail split toward historical W2/W1 from full65+ (week 2 vs week 1 AE),
    while keeping total tail mass fixed. See train_marketshare_artifacts.build_artist_w2_retention.
    """
    if release_dict.get("empirical_w2_over_w1") is None:
        return tw
    try:
        r_hist = float(release_dict["empirical_w2_over_w1"])
    except (TypeError, ValueError):
        return tw
    if fw_vol <= 0 or total_tail_vol <= 0:
        return tw
    m_tail = total_tail_vol / float(fw_vol)
    if m_tail <= 1e-12:
        return tw
    tw = np.asarray(tw, dtype=float)
    r_theo = m_tail * tw[0]
    n_raw = release_dict.get("w2_retention_n_releases")
    try:
        n_rel = float(n_raw) if n_raw is not None else 1.0
    except (TypeError, ValueError):
        n_rel = 1.0
    alpha = min(1.0, max(0.0, n_rel / W2_RETENTION_BLEND_K))
    r_target = (1.0 - alpha) * r_theo + alpha * r_hist
    r_target = max(1e-12, min(r_target, m_tail * 0.999))
    tw0_new = r_target / m_tail
    tw0_new = max(1e-12, min(tw0_new, 0.999))
    rest = tw[1:]
    s = float(rest.sum())
    if s <= 1e-15:
        return tw
    out = np.empty_like(tw, dtype=float)
    out[0] = tw0_new
    out[1:] = rest / s * (1.0 - tw0_new)
    return out


def generate_archetype_decay_curve(
    release_dict: dict,
    artist_dna_lookup: Dict[str, Dict[Any, float]],
    num_weeks: int = 52,
) -> List[float]:
    known_vols = release_dict.get("known_vols") or []
    fw_vol = max(known_vols) if known_vols else release_dict.get("fw_vol", 0)

    artist_name = release_dict.get("name", "Unknown")
    fallback_cluster = release_dict.get("cluster", 0)
    if isinstance(fallback_cluster, str) and not fallback_cluster.isdigit():
        fallback_cluster = 0
    else:
        fallback_cluster = int(fallback_cluster) if fallback_cluster is not None else 0

    weights = artist_dna_lookup.get(artist_name, {fallback_cluster: 1.0})

    scenario = release_dict.get("scenario", "Base")
    target_fy_vol = release_dict.get("fy_vol", None)
    product_ratio = float(release_dict.get("avg_historical_w1_product_ratio", 0.0))
    product_coef = float(release_dict.get("product_ratio_coefficient", GLOBAL_PRODUCT_COEF))

    if num_weeks <= 0:
        return []
    if fw_vol == 0:
        return [0] * num_weeks

    if target_fy_vol is not None and target_fy_vol > fw_vol:
        total_tail_vol = float(target_fy_vol) - float(fw_vol)
    else:
        m52_sum = 0.0
        for c_id, weight in weights.items():
            cid = int(c_id) if not isinstance(c_id, str) or str(c_id).isdigit() else 0
            m_val = ARCHETYPE_MULTIPLIERS.get(cid, ARCHETYPE_MULTIPLIERS[0]).get(scenario, 6.20)
            m52_sum += m_val * float(weight)
        dynamic_m52 = max(1.0, m52_sum + (product_ratio * product_coef))
        total_tail_vol = float(fw_vol) * (dynamic_m52 - 1.0)

    raw_curve = get_weighted_archetype_curve(weights, 52)
    tail_raw = raw_curve[1:]
    tw = tail_raw / tail_raw.sum()
    tw = _adjust_tail_weights_for_empirical_w2(tw, total_tail_vol, float(fw_vol), release_dict)
    weekly_tail_volumes = total_tail_vol * tw
    theo_curve = [float(fw_vol)] + weekly_tail_volumes.tolist()

    if known_vols:
        k = len(known_vols)
        if k >= num_weeks:
            return [float(x) for x in known_vols[:num_weeks]]
        last_known = float(known_vols[-1])
        theo_equivalent = theo_curve[k - 1]
        correction_ratio = last_known / theo_equivalent if theo_equivalent > 0 else 1.0
        correction_ratio = min(max(correction_ratio, 0.5), 2.0)
        remaining_theo = theo_curve[k:num_weeks]
        smoothed_remaining = [vol * correction_ratio for vol in remaining_theo]
        return [float(x) for x in known_vols] + smoothed_remaining

    return theo_curve[:num_weeks]


def calibrate_mid_flight_release(
    release_dict: dict,
    anchor_date: pd.Timestamp,
    artist_dna_lookup: Dict[str, Dict[Any, float]],
) -> dict:
    if "todate_vol" not in release_dict or release_dict.get("fw_vol", 0) == 0:
        return release_dict

    drop_date = pd.to_datetime(release_dict["date"])
    if drop_date > anchor_date:
        return release_dict

    weeks_live = (anchor_date - drop_date).days // 7 + 1
    if weeks_live <= 1 or release_dict["todate_vol"] <= release_dict["fw_vol"]:
        return release_dict

    fw_vol = release_dict["fw_vol"]
    todate_vol = release_dict["todate_vol"]
    artist_name = release_dict.get("name", "Unknown")

    if artist_name in artist_dna_lookup:
        weights = artist_dna_lookup[artist_name]
        release_dict["cluster"] = "DNA Blend"
    else:
        assigned_cluster = release_dict.get("cluster")
        if assigned_cluster is None or assigned_cluster == "Auto":
            best_cluster = 0
            min_error = float("inf")
            for c in [0, 1, 2, 3]:
                raw = get_weighted_archetype_curve({c: 1.0}, 52)
                m52 = ARCHETYPE_MULTIPLIERS[c]["Base"]
                theoretical_total_tail = fw_vol * (m52 - 1.0)
                tail_weights = raw[1:] / raw[1:].sum()
                pct_tail_completed = tail_weights[: (weeks_live - 1)].sum()
                theoretical_todate = fw_vol + (theoretical_total_tail * pct_tail_completed)
                error = abs(theoretical_todate - todate_vol)
                if error < min_error:
                    min_error = error
                    best_cluster = c
            assigned_cluster = best_cluster
            release_dict["cluster"] = assigned_cluster
            weights = {assigned_cluster: 1.0}
        else:
            weights = {int(assigned_cluster): 1.0}

    raw_curve = get_weighted_archetype_curve(weights, 52)
    tail_weights = raw_curve[1:] / raw_curve[1:].sum()
    pct_tail_completed = tail_weights[: (weeks_live - 1)].sum()

    if pct_tail_completed > 0:
        actual_tail_to_date = todate_vol - fw_vol
        projected_total_tail = actual_tail_to_date / pct_tail_completed
        release_dict["fy_vol"] = fw_vol + projected_total_tail

    return release_dict


def run_archetype_scenario(
    release_calendar: List[dict],
    df_full: pd.DataFrame,
    actuals_2026: pd.DataFrame,
    artist_dna_lookup: Dict[str, Dict[Any, float]],
    e_score: float = 0.8,
    volume_threshold: float = 20000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    max_hist_date = actuals_2026["Week Ending Date"].max()
    fut_sim = df_full[
        (df_full["Week Ending Date"] > max_hist_date) & (df_full["Week Ending Date"].dt.year == 2026)
    ].copy()
    fut_dates = fut_sim["Week Ending Date"].sort_values().unique()

    fut_sim["Injected_AMG"] = 0.0
    fut_sim["Injected_Int"] = 0.0
    fut_sim["Injected_Oth"] = 0.0

    df_tracker = pd.DataFrame({"Week Ending Date": fut_dates})

    def inject_volume(label_col: str, release_dict: dict, drop_date: Any, release_name: str) -> Optional[pd.DataFrame]:
        if release_dict.get("known_vols"):
            fw_vol = max(release_dict["known_vols"])
        else:
            fw_vol = release_dict.get("fw_vol", 0)
        if pd.isna(drop_date) or not drop_date or fw_vol == 0:
            return None
        req_date = pd.to_datetime(drop_date)
        full_curve = generate_archetype_decay_curve(release_dict, artist_dna_lookup, 52)
        temp_curve_rows = []
        for current_date in fut_dates:
            days_since = (current_date - req_date).days
            if 0 <= days_since < 364:
                week_idx = days_since // 7
                if week_idx < len(full_curve):
                    weekly_vol = full_curve[week_idx]
                    temp_curve_rows.append({"Week Ending Date": current_date, release_name: weekly_vol})
                    if weekly_vol >= volume_threshold:
                        mask = fut_sim["Week Ending Date"] == current_date
                        fut_sim.loc[mask, label_col] += weekly_vol
        if temp_curve_rows:
            return pd.DataFrame(temp_curve_rows)
        return None

    cluster_names = {
        0: "Standard",
        1: "Burnout",
        2: "Supernova",
        3: "Sticky",
        "Auto": "Auto",
        "DNA Blend": "DNA Blend",
    }
    volume_report_data = []
    end_of_year_date = fut_dates.max()
    anchor_date = actuals_2026["Week Ending Date"].max()

    for i, raw_release in enumerate(release_calendar):
        release = calibrate_mid_flight_release(raw_release, anchor_date, artist_dna_lookup)
        if release["label"] == "Atlantic Music Group":
            target_col = "Injected_AMG"
        elif release["label"] == "Interscope/Geffen/A&M":
            target_col = "Injected_Int"
        else:
            target_col = "Injected_Oth"

        cluster_id = release.get("cluster")
        if cluster_id is None or cluster_id == "Auto":
            cluster_id = get_inferred_cluster(release)
            release["cluster"] = cluster_id

        genre_name = release.get("genre", "Unknown Genre")
        cluster_desc = cluster_names.get(cluster_id, str(cluster_id))
        release_name = release.get("name", f"Release_{i+1}")
        logger.info(
            "Injecting: %s | %s | %s | Cluster: %s",
            release_name,
            release["label"],
            genre_name,
            cluster_desc,
        )

        artist_curve = inject_volume(target_col, release, release["date"], release_name)
        full_curve = generate_archetype_decay_curve(release, artist_dna_lookup, 52)
        drop_dt = pd.to_datetime(release["date"])
        if drop_dt <= end_of_year_date:
            weeks_active = (end_of_year_date - drop_dt).days // 7 + 1
            weeks_active = min(max(0, weeks_active), 52)
            cy_total = sum(full_curve[:weeks_active])
        else:
            cy_total = 0
        volume_report_data.append(
            {
                "Artist / Release": release_name,
                "Drop Date": release["date"],
                "Cluster": cluster_desc,
                "2026 CY Volume": cy_total,
            }
        )
        if artist_curve is not None:
            df_tracker = df_tracker.merge(artist_curve, on="Week Ending Date", how="left")

    df_tracker = df_tracker.fillna(0)

    fut_sim["Sim_Total_Market_AE_Volume"] = (
        fut_sim["Total_Market_AE_Volume"]
        + fut_sim["Injected_AMG"]
        + fut_sim["Injected_Int"]
        + fut_sim["Injected_Oth"]
    )
    fut_sim["Base_Num"] = fut_sim["Predicted_Baseline_Share"] * fut_sim["Total_Market_AE_Volume"]
    fut_sim["Sim_AMG_Num"] = np.where(
        fut_sim["Owner"] == "Atlantic Music Group",
        fut_sim["Base_Num"] + (fut_sim["Injected_AMG"] * 100),
        fut_sim["Base_Num"],
    )
    fut_sim["Sim_Int_Num"] = np.where(
        fut_sim["Owner"] == "Interscope/Geffen/A&M",
        fut_sim["Base_Num"] + (fut_sim["Injected_Int"] * 100),
        fut_sim["Base_Num"],
    )
    fut_sim["Active_Share"] = np.where(
        fut_sim["Owner"] == "Atlantic Music Group",
        fut_sim["Sim_AMG_Num"] / fut_sim["Sim_Total_Market_AE_Volume"],
        fut_sim["Sim_Int_Num"] / fut_sim["Sim_Total_Market_AE_Volume"],
    )
    fut_sim = fut_sim.drop(columns=["Total_Market_AE_Volume"]).rename(
        columns={"Sim_Total_Market_AE_Volume": "Total_Market_AE_Volume"}
    )
    fut_sim["Data_Type"] = "Forecast"

    hist_stack = actuals_2026[
        ["Week Ending Date", "Owner", "Total_Market_AE_Volume", "AE_Share"]
    ].copy().rename(columns={"AE_Share": "Active_Share"})
    hist_stack["Data_Type"] = "Actual"
    # Duplicate label×week rows (often from merged CSVs) double-count in cumulative YTD and zig-zag the chart.
    hist_stack = hist_stack.drop_duplicates(
        subset=["Owner", "Week Ending Date", "Data_Type"], keep="last"
    )

    common_cols = ["Week Ending Date", "Owner", "Total_Market_AE_Volume", "Active_Share", "Data_Type"]
    df_unified = pd.concat([hist_stack, fut_sim[common_cols]], ignore_index=True)
    df_unified = df_unified.sort_values(by=["Owner", "Week Ending Date"]).reset_index(drop=True)

    df_unified["Weighted_Numerator"] = df_unified["Active_Share"] * df_unified["Total_Market_AE_Volume"]
    df_unified["Cum_Numerator"] = df_unified.groupby(["Owner"])["Weighted_Numerator"].cumsum()
    df_unified["Cum_Denominator"] = df_unified.groupby(["Owner"])["Total_Market_AE_Volume"].cumsum()
    df_unified["Unified_YTD_Share"] = (df_unified["Cum_Numerator"] / df_unified["Cum_Denominator"]).round(4)
    df_unified["YTD_Share_Upper"] = df_unified["Unified_YTD_Share"]
    df_unified["YTD_Share_Lower"] = df_unified["Unified_YTD_Share"]
    forecast_mask = df_unified["Data_Type"] == "Forecast"
    df_unified.loc[forecast_mask, "YTD_Share_Upper"] = (
        df_unified.loc[forecast_mask, "Unified_YTD_Share"] + e_score
    )
    df_unified.loc[forecast_mask, "YTD_Share_Lower"] = (
        df_unified.loc[forecast_mask, "Unified_YTD_Share"] - e_score
    )

    if release_calendar:
        volume_report = pd.DataFrame(volume_report_data).sort_values(
            by="2026 CY Volume", ascending=False
        ).reset_index(drop=True)
        logger.info("Injected release volume report:\n%s", volume_report.to_string())

    return df_unified, df_tracker


def auto_enrich_w2_retention(
    release_calendar: List[dict],
    w2_dict: Dict[str, Dict[str, Any]],
    match_threshold: float = 0.8,
) -> List[dict]:
    """
    Attach empirical_w2_over_w1 and w2_retention_n_releases from full65+ aggregates when the
    artist name fuzzy-matches keys in w2_dict (same pattern as product-ratio enrichment).
    """
    if not w2_dict:
        return release_calendar
    known = list(w2_dict.keys())
    enriched: List[dict] = []
    for release in release_calendar:
        out = dict(release)
        if out.get("empirical_w2_over_w1") is not None:
            enriched.append(out)
            continue
        target = out.get("artist", out.get("name", "Unknown"))
        matches = difflib.get_close_matches(str(target), known, n=1, cutoff=match_threshold)
        if matches:
            best = matches[0]
            info = w2_dict[best]
            if isinstance(info, dict):
                out["empirical_w2_over_w1"] = float(info["median_w2_over_w1"])
                out["w2_retention_n_releases"] = int(info.get("n_releases", 1))
            else:
                out["empirical_w2_over_w1"] = float(info)
                out["w2_retention_n_releases"] = 1
            logger.debug("W2 retention: matched artist %s -> %s", target, best)
        enriched.append(out)
    return enriched


def auto_enrich_calendar(
    release_calendar: List[dict],
    profile_dict: Dict[str, float],
    global_coef: float,
    match_threshold: float = 0.8,
) -> List[dict]:
    enriched: List[dict] = []
    known_artists = list(profile_dict.keys())
    for release in release_calendar:
        enriched_release = dict(release)
        if "product_ratio_coefficient" not in enriched_release:
            enriched_release["product_ratio_coefficient"] = global_coef
        if "avg_historical_w1_product_ratio" in enriched_release:
            enriched.append(enriched_release)
            continue
        target_artist = enriched_release.get("artist", enriched_release.get("name", "Unknown"))
        matches = difflib.get_close_matches(str(target_artist), known_artists, n=1, cutoff=match_threshold)
        if matches:
            best = matches[0]
            enriched_release["avg_historical_w1_product_ratio"] = profile_dict[best]
            logger.debug("Auto-matched artist %s -> %s", target_artist, best)
        else:
            enriched_release["avg_historical_w1_product_ratio"] = 0.0
        enriched.append(enriched_release)
    return enriched


def dna_lookup_to_jsonable(d: Dict[str, Dict[Any, float]]) -> Dict[str, Dict[str, float]]:
    def _ck(k: Any) -> str:
        if isinstance(k, int):
            return str(k)
        return str(int(float(k)))

    out: Dict[str, Dict[str, float]] = {}
    for artist, clusters in d.items():
        out[str(artist)] = {_ck(k): float(v) for k, v in clusters.items()}
    return out


def dna_lookup_from_jsonable(d: Dict[str, Dict[str, float]]) -> Dict[str, Dict[int, float]]:
    def _cluster_key(k: Any) -> int:
        if isinstance(k, int):
            return k
        s = str(k).strip()
        return int(float(s)) if "." in s else int(s)

    out: Dict[str, Dict[int, float]] = {}
    for artist, clusters in d.items():
        out[str(artist)] = {_cluster_key(k): float(v) for k, v in clusters.items()}
    return out
