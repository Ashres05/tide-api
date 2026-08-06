"""
75k parlay simulation core: archetypal decay and enrichment.

Uses genre/label archetype priors (DISTRIBUTIONS), Bear/Base/Bull shape scaling and
optional manual archetype cluster (see ``simulate_future_drop`` in
``all_data_archetypes_simulator_ae``), trained decay artifacts (streams / sales / songs),
GLOBAL_PRODUCT_COEF for product tail scaling, and release inputs (fw_vol, fy_vol,
scenario, cluster, dates, known_vols). Optional per-release fields (e.g. empirical W2,
product ratio) may be supplied on the release dict when callers have external estimates.

Extracted from `75k_parlay.ipynb`. Used by the training job and the forecast engine.
"""
# TODO: release vs release_dict in inject_volume

from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from .all_data_archetypes_simulator_ae import (
    simulate_future_drop,
    fit_backfill_forecast,
    SimulatorArtifacts,
    resolve_scenario_multiplier,
)
from .marketshare_labels import TARGET_LABELS

logger = logging.getLogger(__name__)

# Per-Owner injection columns on the future marketshare frame.
_INJECT_COL_PREFIX = "Injected__"
_INJECT_OTHER_COL = "Injected_Other"


def _inject_col_for_owner(owner: str) -> str:
    return f"{_INJECT_COL_PREFIX}{owner}"


def _resolve_inject_col(label: str) -> str:
    """Map release label to an injection column (TARGET_LABELS or Other)."""
    if label in TARGET_LABELS:
        return _inject_col_for_owner(label)
    return _INJECT_OTHER_COL


def _all_inject_cols() -> List[str]:
    return [_inject_col_for_owner(o) for o in TARGET_LABELS] + [_INJECT_OTHER_COL]


def _resolve_scenario_multiplier(
    scenario: Optional[str],
    cluster_id: Optional[int] = None,
    artifacts: Optional[SimulatorArtifacts] = None,
) -> float:
    """Thin shim around ``resolve_scenario_multiplier`` for marketshare code.

    Pulls the learned scenario_multipliers table from ``artifacts`` (when
    supplied) and forwards to the source-of-truth helper in
    ``all_data_archetypes_simulator_ae`` so streams, sales, and songs
    channels each consult their own metric's table while sharing a single
    resolution implementation with worldwide_streams.
    """
    learned = getattr(artifacts, "scenario_multipliers", None) if artifacts else None
    return resolve_scenario_multiplier(
        scenario,
        cluster_id=cluster_id,
        scenario_multipliers=learned,
    )

# --- Static coefficients & tables (also persisted in artifacts) ---

GLOBAL_PRODUCT_COEF = -0.61

NUM_WEEKS = 78 # Releases limited to 18 months


def release_is_single(release_dict: dict) -> bool:
    """True when a release calendar row should use singles decay artifacts."""
    pt = str(
        release_dict.get("product_type")
        or release_dict.get("release_type")
        or ""
    ).strip().lower()
    return pt in ("single", "singles")


def normalize_single_release_for_decay(release_dict: dict) -> dict:
    """
    Singles have no product-sales channel; keep sales at zero so the sales
    decay curve does not contribute even when album-style splits exist on the row.
    """
    out = dict(release_dict)
    out["fw_sales"] = 0.0
    out["known_sales"] = []
    return out


def decay_artifacts_for_release(
    release_dict: dict,
    artifacts_streams: SimulatorArtifacts,
    artifacts_sales: SimulatorArtifacts,
    artifacts_songs: SimulatorArtifacts,
    *,
    artifacts_streams_singles: Optional[SimulatorArtifacts] = None,
    artifacts_sales_singles: Optional[SimulatorArtifacts] = None,
    artifacts_songs_singles: Optional[SimulatorArtifacts] = None,
) -> Tuple[dict, SimulatorArtifacts, SimulatorArtifacts, SimulatorArtifacts]:
    """Pick album vs singles decay bundles; normalize singles release inputs."""
    if not release_is_single(release_dict):
        return release_dict, artifacts_streams, artifacts_sales, artifacts_songs
    if artifacts_streams_singles is None:
        logger.warning(
            "Release %s is single but singles streams artifacts are not loaded; "
            "using album streams decay.",
            release_dict.get("name") or release_dict.get("title"),
        )
        return normalize_single_release_for_decay(release_dict), (
            artifacts_streams,
            artifacts_sales,
            artifacts_songs,
        )
    sales = artifacts_sales_singles or artifacts_sales
    songs = artifacts_songs_singles or artifacts_songs
    return (
        normalize_single_release_for_decay(release_dict),
        artifacts_streams_singles,
        sales,
        songs,
    )

# When release_dict supplies empirical_w2_over_w1, blend tail toward it; alpha = min(1, n_releases / K).
W2_RETENTION_BLEND_K = 4.0 # 4 or more releases means we fully trust artist history and note archetype curve

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
        # AMG-like prior (default for most of the 7-label set until recomputed)
        "Atlantic Music Group": [35.07, 8.21, 18.66, 38.06],
        "Warner Records": [35.07, 8.21, 18.66, 38.06],
        "THE ORCHARD": [35.07, 8.21, 18.66, 38.06],
        "RCA Records": [35.07, 8.21, 18.66, 38.06],
        "Columbia Records": [35.07, 8.21, 18.66, 38.06],
        # Former Interscope/Geffen/A&M prior — renamed + copied to Republic
        "Interscope-Capitol": [9.86, 14.08, 50.70, 25.35],
        "REPUBLIC Collective": [9.86, 14.08, 50.70, 25.35],
    },
}


def get_weighted_archetype_curve(cluster_weights: Dict[Any, float], num_weeks: int = NUM_WEEKS) -> np.ndarray:
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


def release_album_title(release_dict: dict) -> str:
    """Canonical album / release-group label for injection keys and reports."""
    title = release_dict.get("title")
    if title is not None and str(title).strip():
        return str(title).strip()
    legacy = release_dict.get("name")
    if legacy is not None and str(legacy).strip():
        return str(legacy).strip()
    return "Unknown"


def decay_artist_name(release_dict: dict) -> str:
    """Artist for archetype decay lookup; never fall back to album title (``name``/``title``)."""
    artist = release_dict.get("artist")
    if artist is not None and str(artist).strip():
        return str(artist).strip()
    return "Unknown"


def release_peak_w1_vol(release_dict: dict) -> float:
    """Peak W1 AE for injection gating (known actuals, components, or fw_vol)."""
    known_vols = release_dict.get("known_vols") or []
    if known_vols:
        return float(max(known_vols))
    comp = (
        float(release_dict.get("fw_streams", 0.0))
        + float(release_dict.get("fw_sales", 0.0))
        + float(release_dict.get("fw_songs", 0.0))
    )
    if comp > 0:
        return comp
    return float(release_dict.get("fw_vol", 0.0))


def _adjust_tail_weights_for_empirical_w2(
    tw: np.ndarray,
    total_tail_vol: float,
    fw_vol: float,
    release_dict: dict,
) -> np.ndarray:
    """
    Blend the archetype tail split toward release_dict empirical W2/W1 when present,
    while keeping total tail mass fixed.
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
    artifacts_streams: SimulatorArtifacts,
    artifacts_sales: SimulatorArtifacts,
    artifacts_songs: SimulatorArtifacts,
    num_weeks: int = NUM_WEEKS,
    *,
    artifacts_streams_singles: Optional[SimulatorArtifacts] = None,
    artifacts_sales_singles: Optional[SimulatorArtifacts] = None,
    artifacts_songs_singles: Optional[SimulatorArtifacts] = None,
) -> List[float]:
    """Weekly total album-equivalent decay (sum of streams + product + song channels)."""
    curves = generate_archetype_decay_component_curves(
        release_dict,
        artifacts_streams,
        artifacts_sales,
        artifacts_songs,
        num_weeks=num_weeks,
        artifacts_streams_singles=artifacts_streams_singles,
        artifacts_sales_singles=artifacts_sales_singles,
        artifacts_songs_singles=artifacts_songs_singles,
    )
    return curves["total"]


def generate_archetype_decay_component_curves(
    release_dict: dict,
    artifacts_streams: SimulatorArtifacts,
    artifacts_sales: SimulatorArtifacts,
    artifacts_songs: SimulatorArtifacts,
    num_weeks: int = NUM_WEEKS,
    *,
    artifacts_streams_singles: Optional[SimulatorArtifacts] = None,
    artifacts_sales_singles: Optional[SimulatorArtifacts] = None,
    artifacts_songs_singles: Optional[SimulatorArtifacts] = None,
) -> Dict[str, List[float]]:
    """
    Weekly archetype decay per US album-equivalent channel plus total.

    Returns keys: ``streaming_equivalent``, ``product_sales``,
    ``song_sale_equivalent``, ``total`` (each length ``num_weeks``).
    """
    release_dict, artifacts_streams, artifacts_sales, artifacts_songs = (
        decay_artifacts_for_release(
            release_dict,
            artifacts_streams,
            artifacts_sales,
            artifacts_songs,
            artifacts_streams_singles=artifacts_streams_singles,
            artifacts_sales_singles=artifacts_sales_singles,
            artifacts_songs_singles=artifacts_songs_singles,
        )
    )
    artist = decay_artist_name(release_dict)
    genre = release_dict.get("genre")
    scenario = release_dict.get("scenario") or "Base"
    cluster_raw = release_dict.get("cluster")
    archetype_cluster_id: Optional[int]
    if cluster_raw is None:
        archetype_cluster_id = None
    else:
        try:
            archetype_cluster_id = int(cluster_raw)
        except (TypeError, ValueError):
            archetype_cluster_id = None

    # Intercept the user's scenario choice here. The fit must always run with
    # scenario="Base" so the basis functions, NNLS mixture, fitted peak_volume,
    # and dynamic_floor never shift between scenarios — Bear/Bull is layered in
    # strictly post-fit. The actual scalar multiplier is resolved *per channel*
    # inside _get_curve so each metric (streams, sales, songs) consults its own
    # learned scenario_multipliers.json (when present).
    fit_scenario = "Base"
    
    known_streams = release_dict.get("known_streams", [])
    known_sales = release_dict.get("known_sales", [])
    known_songs = release_dict.get("known_songs", [])
    
    fw_streams = float(release_dict.get("fw_streams", 0.0))
    fw_sales = float(release_dict.get("fw_sales", 0.0))
    fw_songs = float(release_dict.get("fw_songs", 0.0))
    
    known_vols = release_dict.get("known_vols") or []
    fw_vol = max(known_vols) if known_vols else float(release_dict.get("fw_vol", 0.0))
    prod_ratio = float(release_dict.get("avg_historical_w1_product_ratio", 0.0))
    
    if not known_streams and known_vols:
        known_streams = [v * (1.0 - prod_ratio) for v in known_vols]
        known_sales = [v * prod_ratio for v in known_vols]
        known_songs = [0.0 for _ in known_vols]
        
    # Only derive W1 splits from fw_vol when no explicit component breakdown was given.
    explicit_w1_components = (fw_streams > 0) or (fw_sales > 0) or (fw_songs > 0)
    if not explicit_w1_components and fw_vol > 0:
        fw_streams = fw_vol * (1.0 - prod_ratio)
        fw_sales = fw_vol * prod_ratio
        fw_songs = 0.0

    # Extract global tuning params
    radius_raw = release_dict.get("peak_sim_log_radius")
    radius: Optional[float] = None
    if radius_raw not in (None, ""):
        radius = float(radius_raw)
    min_sub = int(release_dict.get("peak_sim_min_subset_releases", 10))
    log_std = float(release_dict.get("peak_sim_spread_threshold_log_std", 0.25))
    min_art = int(release_dict.get("peak_sim_min_artist_releases", 20))
    stream_floor_override = release_dict.get("stream_floor", None)
    if stream_floor_override is not None:
        stream_floor_override = float(stream_floor_override)

    def _get_curve(known, fw, artifacts, force_floor) -> List[float]:
        sim_tuning_kwargs = {
            "peak_sim_min_subset_releases": min_sub,
            "peak_sim_spread_threshold_log_std": log_std,
            "peak_sim_min_artist_releases": min_art,
        }
        if radius is not None:
            sim_tuning_kwargs["peak_sim_log_radius"] = radius

        # Per-channel scenario shock: resolved against the channel's own
        # artifacts so streams uses streams' learned p20/p50/p80, sales uses
        # sales' learned table, etc. Falls back to the global hardcoded
        # ARCHETYPE_SCENARIO_MULTIPLIERS when scenario_multipliers.json was
        # not present in the artifact dir.
        scenario_multiplier = _resolve_scenario_multiplier(
            scenario, archetype_cluster_id, artifacts=artifacts
        )

        if not known and fw == 0:
            return [0.0] * num_weeks
        if known:
            try:
                # fit_backfill_forecast does not accept peak_sim_* tuning args (simulate_future_drop does).
                # Scenario hygiene: the fit is always Base so dynamic_floor never shifts;
                # Bear/Bull is delivered exclusively via scenario_multiplier post-fit.
                pred_df, _ = fit_backfill_forecast(
                    artist=artist,
                    genre=genre,
                    actuals_weekly_streams=np.array(known, dtype=float),
                    artifacts=artifacts,
                    end_week=num_weeks,
                    stream_floor=force_floor,
                    scenario=fit_scenario,
                    archetype_cluster_id=archetype_cluster_id,
                    scenario_multiplier=scenario_multiplier,
                )
                return pred_df["pred_weekly_streams"].tolist()
            except Exception as e:
                logger.warning(f"Backfill failed for {artist}: {e}. Falling back to day-0 sim.")
                fw = float(known[0])
        try:
            # Same scenario hygiene as above: simulate at Base, then shock the
            # future-only portion of the returned curve relative to its floor.
            pred_df, _ = simulate_future_drop(
                artist=artist, peak_volume=fw, peak_week=1.0, genre=genre,
                artifacts=artifacts, stream_floor=force_floor,
                scenario=fit_scenario,
                archetype_cluster_id=archetype_cluster_id,
                **sim_tuning_kwargs,
            )
            # NB: pandas/numpy can hand back a read-only view from
            # ``to_numpy(dtype=float)`` when no dtype conversion is needed.
            # The post-fit shock writes back into ``curve`` (``curve[k:] =
            # ...``), so force a writable copy here — without it, Bear/Bull
            # raise ``ValueError: assignment destination is read-only`` and the
            # outer ``except`` quietly returns a zero curve, which presents as
            # "the scenario went lower than Base" on the frontend.
            curve = pred_df["pred_weekly_streams"].iloc[:num_weeks].to_numpy(dtype=float, copy=True)
            if scenario_multiplier != 1.0:
                # Anchor the shock to the same floor simulate_future_drop used.
                # When the caller pinned stream_floor, that's authoritative;
                # otherwise the dynamic floor equals the curve's asymptote, so
                # min(curve) is a safe proxy.
                if force_floor is not None:
                    floor = float(force_floor)
                else:
                    floor = float(np.min(curve)) if curve.size else 0.0
                k = len(known)
                if k < curve.size:
                    above = np.clip(curve[k:] - floor, 0.0, None)
                    curve[k:] = floor + above * scenario_multiplier
            return curve.tolist()
        except Exception as e:
            logger.error(f"Simulation failed for {artist}: {e}. Returning zeros.")
            return [0.0] * num_weeks

    # 3. Generate the 3 component curves independently!
    curve_streams = _get_curve(known_streams, fw_streams, artifacts_streams, force_floor=stream_floor_override)
    curve_sales = _get_curve(known_sales, fw_sales, artifacts_sales, force_floor=0.0)
    curve_songs = _get_curve(known_songs, fw_songs, artifacts_songs, force_floor=0.0)

    # 4–5. Boundary-aligned splice on total; keep components consistent so they sum to total.
    combined = [curve_streams[i] + curve_sales[i] + curve_songs[i] for i in range(num_weeks)]
    if known_vols:
        K = min(len(known_vols), num_weeks)
        if K > 0 and K < num_weeks:
            last_actual = float(known_vols[K - 1])
            model_at_boundary = combined[K - 1]
            if abs(model_at_boundary) > 1e-12:
                scale = last_actual / model_at_boundary
                for i in range(K, num_weeks):
                    curve_streams[i] *= scale
                    curve_sales[i] *= scale
                    curve_songs[i] *= scale
        for i in range(K):
            if (
                known_streams
                and known_sales
                and known_songs
                and i < len(known_streams)
                and i < len(known_sales)
                and i < len(known_songs)
            ):
                curve_streams[i] = float(known_streams[i])
                curve_sales[i] = float(known_sales[i])
                curve_songs[i] = float(known_songs[i])
            else:
                target = float(known_vols[i])
                model_tot = curve_streams[i] + curve_sales[i] + curve_songs[i]
                if model_tot > 1e-12:
                    curve_streams[i] = target * curve_streams[i] / model_tot
                    curve_sales[i] = target * curve_sales[i] / model_tot
                    curve_songs[i] = target * curve_songs[i] / model_tot
                else:
                    curve_streams[i] = target
                    curve_sales[i] = 0.0
                    curve_songs[i] = 0.0

    total = [
        float(curve_streams[i] + curve_sales[i] + curve_songs[i])
        for i in range(num_weeks)
    ]
    return {
        "streaming_equivalent": [float(x) for x in curve_streams],
        "product_sales": [float(x) for x in curve_sales],
        "song_sale_equivalent": [float(x) for x in curve_songs],
        "total": total,
    }


def build_release_weekly_component_forecast_df(
    release_dict: dict,
    tracker_dates: Any,
    artifacts_streams: SimulatorArtifacts,
    artifacts_sales: SimulatorArtifacts,
    artifacts_songs: SimulatorArtifacts,
    *,
    artifacts_streams_singles: Optional[SimulatorArtifacts] = None,
    artifacts_sales_singles: Optional[SimulatorArtifacts] = None,
    artifacts_songs_singles: Optional[SimulatorArtifacts] = None,
) -> pd.DataFrame:
    """
    Map per-release component decay curves onto 2026 week-ending dates.

    Returns a long frame with ``Week Ending Date``, ``streaming_equivalent``,
    ``product_sales``, ``song_sale_equivalent``, and ``total``.
    """
    empty_cols = [
        "Week Ending Date",
        "streaming_equivalent",
        "product_sales",
        "song_sale_equivalent",
        "total",
    ]
    if release_peak_w1_vol(release_dict) <= 0:
        return pd.DataFrame(columns=empty_cols)

    drop_raw = release_dict.get("date")
    if drop_raw is None or (isinstance(drop_raw, float) and pd.isna(drop_raw)):
        return pd.DataFrame(columns=empty_cols)
    drop_date = pd.to_datetime(drop_raw)

    curves = generate_archetype_decay_component_curves(
        release_dict,
        artifacts_streams,
        artifacts_sales,
        artifacts_songs,
        NUM_WEEKS,
        artifacts_streams_singles=artifacts_streams_singles,
        artifacts_sales_singles=artifacts_sales_singles,
        artifacts_songs_singles=artifacts_songs_singles,
    )

    rows: List[Dict[str, Any]] = []
    for current_date in tracker_dates:
        current_dt = pd.to_datetime(current_date)
        days_since = (current_dt - drop_date).days
        if days_since < 0 or days_since >= NUM_WEEKS * 7:
            continue
        week_idx = days_since // 7
        if week_idx >= len(curves["total"]):
            continue
        rows.append(
            {
                "Week Ending Date": current_dt.strftime("%Y-%m-%d"),
                "streaming_equivalent": float(curves["streaming_equivalent"][week_idx]),
                "product_sales": float(curves["product_sales"][week_idx]),
                "song_sale_equivalent": float(curves["song_sale_equivalent"][week_idx]),
                "total": float(curves["total"][week_idx]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows)


def run_archetype_scenario(
    release_calendar: List[dict],
    df_full: pd.DataFrame,
    actuals_2026: pd.DataFrame,
    #artifacts: SimulatorArtifacts, 
    artifacts_streams: SimulatorArtifacts, 
    artifacts_sales: SimulatorArtifacts,   
    artifacts_songs: SimulatorArtifacts,
    e_score: float = 0.8,
    volume_threshold: float = 75000,
    *,
    artifacts_streams_singles: Optional[SimulatorArtifacts] = None,
    artifacts_sales_singles: Optional[SimulatorArtifacts] = None,
    artifacts_songs_singles: Optional[SimulatorArtifacts] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    max_hist_date = actuals_2026["Week Ending Date"].max()
    fut_sim = df_full[
        (df_full["Week Ending Date"] > max_hist_date) & (df_full["Week Ending Date"].dt.year == 2026)
    ].copy()
    fut_dates = fut_sim["Week Ending Date"].sort_values().unique()

    # --- NEW: Create a separate timeline for the frontend tracker ---
    tracker_dates = df_full[df_full["Week Ending Date"].dt.year == 2026]["Week Ending Date"].sort_values().unique()

    inject_cols = _all_inject_cols()
    for col in inject_cols:
        fut_sim[col] = 0.0

    # --- UPDATE: Assign the full year to the tracker ---
    df_tracker = pd.DataFrame({"Week Ending Date": tracker_dates})

    def inject_volume(label_col: str, release_dict: dict, drop_date: Any, release_name: str) -> Optional[pd.DataFrame]:
        fw_vol = release_peak_w1_vol(release_dict)
        if pd.isna(drop_date) or not drop_date or fw_vol <= 0:
            return None
            
        req_date = pd.to_datetime(drop_date)
        full_curve = generate_archetype_decay_curve(
            release_dict,
            artifacts_streams,
            artifacts_sales,
            artifacts_songs,
            NUM_WEEKS,
            artifacts_streams_singles=artifacts_streams_singles,
            artifacts_sales_singles=artifacts_sales_singles,
            artifacts_songs_singles=artifacts_songs_singles,
        )

        temp_curve_rows = []
        # --- UPDATE: Loop through the full year instead of just the future ---
        for current_date in tracker_dates:
            days_since = (current_date - req_date).days
            if 0 <= days_since < NUM_WEEKS * 7:
                week_idx = days_since // 7
                if week_idx < len(full_curve):
                    weekly_vol = full_curve[week_idx]
                    temp_curve_rows.append({"Week Ending Date": current_date, release_name: weekly_vol})
                    
                    # --- UPDATE: Only inject into marketshare if the date is in the future! ---
                    if weekly_vol >= volume_threshold and current_date > max_hist_date:
                        mask = fut_sim["Week Ending Date"] == current_date
                        fut_sim.loc[mask, label_col] += weekly_vol
        return pd.DataFrame(temp_curve_rows) if temp_curve_rows else None

    volume_report_data = []
    end_of_year_date = fut_dates.max() if len(fut_dates) else tracker_dates.max()

    for i, release in enumerate(release_calendar):
        lbl = str(release.get("label") or "")
        target_col = _resolve_inject_col(lbl)
        if target_col == _INJECT_OTHER_COL and lbl:
            logger.warning(
                "Release label %r not in TARGET_LABELS; AE injects into market denominator only",
                lbl,
            )

        release_name = release_album_title(release)
        logger.info(f"Injecting: {release_name} | {lbl} | {release.get('genre', 'Unknown Genre')}")

        artist_curve = inject_volume(target_col, release, release.get("date"), release_name)
        full_curve = generate_archetype_decay_curve(
            release, 
            artifacts_streams, 
            artifacts_sales, 
            artifacts_songs, 
            NUM_WEEKS,
            artifacts_streams_singles=artifacts_streams_singles,
            artifacts_sales_singles=artifacts_sales_singles,
            artifacts_songs_singles=artifacts_songs_singles,
        )
        
        drop_dt = pd.to_datetime(release.get("date"))
        cy_total = 0
        if pd.notna(drop_dt) and drop_dt <= end_of_year_date:
            weeks_active = min(max(0, (end_of_year_date - drop_dt).days // 7 + 1), NUM_WEEKS)
            cy_total = sum(full_curve[:weeks_active])
            
        cl_disp = "auto"
        if release.get("cluster") is not None:
            try:
                cl_disp = str(int(release["cluster"]))
            except (TypeError, ValueError):
                cl_disp = "auto"
        volume_report_data.append({
            "Artist / Release": release_name,
            "Drop Date": release.get("date"),
            "Cluster": cl_disp,
            "Scenario": str(release.get("scenario") or "Base"),
            "2026 CY Volume": cy_total,
        })
        if artist_curve is not None:
            df_tracker = df_tracker.merge(artist_curve, on="Week Ending Date", how="left")

    df_tracker = df_tracker.fillna(0)

    # --- Marketshare: N-way Owner injection ---
    # Sim_Total = baseline market + sum of all injected AE (tracked labels + Other)
    # Active_Share[owner] = (Base_Num[owner] + Injected[owner]*100) / Sim_Total
    fut_sim["Sim_Total_Market_AE_Volume"] = fut_sim["Total_Market_AE_Volume"] + fut_sim[inject_cols].sum(axis=1)
    fut_sim["Base_Num"] = fut_sim["Predicted_Baseline_Share"] * fut_sim["Total_Market_AE_Volume"]

    injected_for_owner = np.zeros(len(fut_sim), dtype=float)
    for owner in TARGET_LABELS:
        col = _inject_col_for_owner(owner)
        mask = fut_sim["Owner"].astype(str) == owner
        if mask.any():
            injected_for_owner[mask.to_numpy()] = fut_sim.loc[mask, col].to_numpy(dtype=float)

    denom = fut_sim["Sim_Total_Market_AE_Volume"].replace(0, np.nan)
    fut_sim["Active_Share"] = (fut_sim["Base_Num"] + injected_for_owner * 100.0) / denom
    fut_sim["Active_Share"] = fut_sim["Active_Share"].fillna(0.0)

    fut_sim = fut_sim.drop(columns=["Total_Market_AE_Volume"]).rename(
        columns={"Sim_Total_Market_AE_Volume": "Total_Market_AE_Volume"}
    )
    fut_sim["Data_Type"] = "Forecast"

    hist_stack = actuals_2026[["Week Ending Date", "Owner", "Total_Market_AE_Volume", "AE_Share"]].copy().rename(columns={"AE_Share": "Active_Share"})
    hist_stack["Data_Type"] = "Actual"
    hist_stack = hist_stack.drop_duplicates(subset=["Owner", "Week Ending Date", "Data_Type"], keep="last")

    df_unified = pd.concat([hist_stack, fut_sim[["Week Ending Date", "Owner", "Total_Market_AE_Volume", "Active_Share", "Data_Type"]]], ignore_index=True)
    df_unified = df_unified.sort_values(by=["Owner", "Week Ending Date"]).reset_index(drop=True)

    df_unified["Weighted_Numerator"] = df_unified["Active_Share"] * df_unified["Total_Market_AE_Volume"]
    df_unified["Cum_Numerator"] = df_unified.groupby(["Owner"])["Weighted_Numerator"].cumsum()
    df_unified["Cum_Denominator"] = df_unified.groupby(["Owner"])["Total_Market_AE_Volume"].cumsum()
    df_unified["Unified_YTD_Share"] = (df_unified["Cum_Numerator"] / df_unified["Cum_Denominator"]).round(4)
    df_unified["YTD_Share_Upper"] = df_unified["Unified_YTD_Share"]
    df_unified["YTD_Share_Lower"] = df_unified["Unified_YTD_Share"]
    
    forecast_mask = df_unified["Data_Type"] == "Forecast"
    df_unified.loc[forecast_mask, "YTD_Share_Upper"] = df_unified.loc[forecast_mask, "Unified_YTD_Share"] + e_score
    df_unified.loc[forecast_mask, "YTD_Share_Lower"] = df_unified.loc[forecast_mask, "Unified_YTD_Share"] - e_score

    return df_unified, df_tracker


def auto_enrich_w2_retention(
    release_calendar: List[dict],
    w2_dict: Dict[str, Dict[str, Any]],
    match_threshold: float = 0.8,
) -> List[dict]:
    """
    Attach empirical_w2_over_w1 and w2_retention_n_releases from w2_dict when the
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
        target = out.get("artist") or "Unknown"
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
    match_threshold: float = 0.8,
) -> List[dict]:
    """
    Fuzzy-match artist names to attach avg_historical_w1_product_ratio from profile_dict.

    Does not set product_ratio_coefficient; that comes from per-release overrides or
    cluster_product_coef / GLOBAL_PRODUCT_COEF in generate_archetype_decay_curve.
    """
    enriched: List[dict] = []
    known_artists = list(profile_dict.keys())
    for release in release_calendar:
        enriched_release = dict(release)
        if "avg_historical_w1_product_ratio" in enriched_release:
            enriched.append(enriched_release)
            continue
        target_artist = enriched_release.get("artist") or "Unknown"
        matches = difflib.get_close_matches(str(target_artist), known_artists, n=1, cutoff=match_threshold)
        if matches:
            best = matches[0]
            enriched_release["avg_historical_w1_product_ratio"] = profile_dict[best]
            logger.debug("Auto-matched artist %s -> %s", target_artist, best)
        else:
            enriched_release["avg_historical_w1_product_ratio"] = 0.0
        enriched.append(enriched_release)
    return enriched


def cluster_product_coef_from_jsonable(d: Dict[str, Any]) -> Dict[int, float]:
    """Restore cluster -> coefficient from JSON (string keys from json.dump)."""

    def _cluster_key(k: Any) -> int:
        if isinstance(k, int):
            return k
        s = str(k).strip()
        return int(float(s)) if "." in s else int(s)

    return {_cluster_key(k): float(v) for k, v in d.items()}


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
