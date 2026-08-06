#!/usr/bin/env python3
"""
Weekly/monthly training job for 75k marketshare artifacts.

Inputs (7 labels; see ``marketshare_labels.py``):
  model/data/current_data_all.csv
  model/data/bigrelease_alist_75k_all.csv

Market total AE is inverted from Atlantic Music Group only:
  (ALBUM_EQUIVALENT * 100) / ALBUM_EQUIVALENT_SHARE

A-list scrub is Owner×Week: AE_Volume -= A_LIST_AE[Owner, week].
LGBM stays pooled with Owner as a categorical feature; one Prophet per Owner.

Outputs (default: ./artifacts_75k/):
  production_lgbm_75k.pkl, production_prophet_models_75k.pkl,
  production_spike_engine.pkl,
  df_full.parquet, actuals_2026.parquet,
  artist_profile_dict.json, artist_dna_lookup.json, artist_w2_retention.json,
  cluster_product_coef.json, distributions.json, metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

import lightgbm as lgb

from .marketshare_75k_simulation import (
    DISTRIBUTIONS,
    GLOBAL_PRODUCT_COEF,
    W2_RETENTION_BLEND_K,
    dna_lookup_to_jsonable,
)
from .all_data_archetypes_simulator_ae import train as train_archetype_model
from .marketshare_labels import (
    ALIST_VOL_COLS,
    BIG_RELEASE_COLS,
    BIGRELEASE_ALIST_CSV,
    CURRENT_DATA_ALL_CSV,
    MARKET_ALBUMS_COL,
    MARKET_ANCHOR_LABEL,
    TARGET_LABELS,
    TRAINING_CATEGORIES,
    WEEK_END_COL,
    assert_label_maps_complete,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _normalize_current_data_year_column(current: pd.DataFrame) -> pd.DataFrame:
    """Normalize Year/YEAR/year from Current_Data.csv to nullable int column ``Year``."""
    for cand in ("Year", "YEAR", "year"):
        if cand in current.columns:
            out = current.rename(columns={cand: "Year"}) if cand != "Year" else current.copy()
            out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
            return out
    return current


def forecast_year_row_mask(df: pd.DataFrame, forecast_year: int) -> pd.Series:
    """
    Rows for chart/industry year ``forecast_year`` from Current_Data ``Year``,
    not calendar week-ending date. Rows with null ``Year`` fall back to
    ``Week Ending Date >= {forecast_year}-01-08``. If ``Year`` is absent or all-null,
    use that date threshold only.
    """
    wk = pd.to_datetime(df["Week Ending Date"])
    jan8 = pd.Timestamp(year=forecast_year, month=1, day=8)
    legacy = wk >= jan8
    if "Year" not in df.columns:
        logger.warning(
            "No Year column; using Week Ending Date >= %s for forecast_year=%s",
            jan8.date().isoformat(),
            forecast_year,
        )
        return legacy
    yr = pd.to_numeric(df["Year"], errors="coerce")
    if not yr.notna().any():
        logger.warning(
            "Year column all null; using Week Ending Date >= %s for forecast_year=%s",
            jan8.date().isoformat(),
            forecast_year,
        )
        return legacy
    return (yr.notna() & (yr == forecast_year)) | (yr.isna() & legacy)


def load_weekly_labels(data_dir: Path) -> pd.DataFrame:
    """
    Owner×Week AE panel from ``current_data_all.csv`` for ``TARGET_LABELS``.

    Total market AE is inverted from Atlantic Music Group only:
    ``(ALBUM_EQUIVALENT * 100) / ALBUM_EQUIVALENT_SHARE``.
    """
    assert_label_maps_complete()
    path = data_dir / CURRENT_DATA_ALL_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; sync {CURRENT_DATA_ALL_CSV} from S3.")
    current = pd.read_csv(path)
    current = _normalize_current_data_year_column(current)
    if "COUNTRY_CODE" in current.columns:
        current = current.loc[current["COUNTRY_CODE"].astype(str).str.upper() == "US"].copy()
    if "RELEASE_AGE" in current.columns:
        ra = current["RELEASE_AGE"].astype(str).str.strip()
        n_before = len(current)
        current = current.loc[ra.str.casefold() == "current"].copy()
        n_after = len(current)
        if n_before and n_after < n_before:
            logger.info(
                "%s: RELEASE_AGE=='Current' only (%d rows, dropped %d)",
                CURRENT_DATA_ALL_CSV,
                n_after,
                n_before - n_after,
            )
        if n_after == 0:
            logger.warning(
                "%s: zero rows after RELEASE_AGE=='Current' filter; check values/casing",
                CURRENT_DATA_ALL_CSV,
            )
    weekly = current[current["LABEL_NAME"].isin(TARGET_LABELS)].copy()
    weekly = weekly.rename(
        columns={
            "WEEK_ENDING_DATE": "Week Ending Date",
            "ALBUM_EQUIVALENT": "AE_Volume",
            "ALBUM_EQUIVALENT_SHARE": "AE_Share",
            "LABEL_NAME": "Owner",
        }
    )
    weekly["Week Ending Date"] = pd.to_datetime(weekly["Week Ending Date"], errors="coerce")
    weekly["AE_Volume"] = pd.to_numeric(weekly["AE_Volume"], errors="coerce")
    weekly["AE_Share"] = pd.to_numeric(weekly["AE_Share"], errors="coerce")
    weekly = weekly.dropna(subset=["Week Ending Date", "Owner", "AE_Volume", "AE_Share"])

    amg = weekly[weekly["Owner"] == MARKET_ANCHOR_LABEL].copy()
    amg["True_Total_Market"] = np.where(
        amg["AE_Share"] > 0,
        (amg["AE_Volume"] / amg["AE_Share"]) * 100,
        np.nan,
    )
    market_ref = amg[["Week Ending Date", "True_Total_Market"]].drop_duplicates(
        subset=["Week Ending Date"], keep="last"
    )
    if "Total_Market_AE_Volume" in weekly.columns:
        weekly = weekly.drop(columns=["Total_Market_AE_Volume"])
    weekly = weekly.merge(market_ref, on="Week Ending Date", how="left")
    weekly = weekly.rename(columns={"True_Total_Market": "Total_Market_AE_Volume"})

    dup_n = int(weekly.duplicated(subset=["Owner", "Week Ending Date"]).sum())
    if dup_n:
        logger.warning(
            "%s: dropping %d duplicate Owner×Week rows (keeps last)",
            CURRENT_DATA_ALL_CSV,
            dup_n,
        )
        weekly = weekly.drop_duplicates(
            subset=["Owner", "Week Ending Date"], keep="last"
        ).reset_index(drop=True)
    logger.info(
        "load_weekly_labels: %d rows, %d owners, weeks %s → %s",
        len(weekly),
        weekly["Owner"].nunique(),
        weekly["Week Ending Date"].min().date() if len(weekly) else None,
        weekly["Week Ending Date"].max().date() if len(weekly) else None,
    )
    return weekly


# Back-compat alias for callers that still import the old name.
load_weekly_amg_int = load_weekly_labels


def load_alist_bigrelease_long(data_dir: Path) -> pd.DataFrame:
    """
    Melt ``bigrelease_alist_75k_all.csv`` to long Owner×Week scrub + flag rows.

    Columns: WEEK_END_DATE, Owner, A_LIST_AE, BIG_RELEASE_FLAG, MARKET_ALBUMS.
    """
    assert_label_maps_complete()
    path = data_dir / BIGRELEASE_ALIST_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; sync {BIGRELEASE_ALIST_CSV} from S3.")
    wide = pd.read_csv(path)
    wide.columns = [str(c).strip().upper() for c in wide.columns]
    if WEEK_END_COL not in wide.columns:
        raise ValueError(f"{BIGRELEASE_ALIST_CSV} missing {WEEK_END_COL}")
    wide[WEEK_END_COL] = pd.to_datetime(wide[WEEK_END_COL], errors="coerce")
    wide = wide.dropna(subset=[WEEK_END_COL])
    if MARKET_ALBUMS_COL in wide.columns:
        wide[MARKET_ALBUMS_COL] = pd.to_numeric(wide[MARKET_ALBUMS_COL], errors="coerce").fillna(0.0)
    else:
        wide[MARKET_ALBUMS_COL] = 0.0

    rows: List[pd.DataFrame] = []
    for owner, vol_col in ALIST_VOL_COLS.items():
        flag_col = BIG_RELEASE_COLS[owner]
        if vol_col not in wide.columns:
            raise ValueError(f"{BIGRELEASE_ALIST_CSV} missing alist col {vol_col}")
        if flag_col not in wide.columns:
            raise ValueError(f"{BIGRELEASE_ALIST_CSV} missing flag col {flag_col}")
        chunk = pd.DataFrame(
            {
                "WEEK_END_DATE": wide[WEEK_END_COL],
                "Owner": owner,
                "A_LIST_AE": pd.to_numeric(wide[vol_col], errors="coerce").fillna(0.0),
                "BIG_RELEASE_FLAG": pd.to_numeric(wide[flag_col], errors="coerce").fillna(0.0),
                "MARKET_ALBUMS": wide[MARKET_ALBUMS_COL],
            }
        )
        rows.append(chunk)
    long = pd.concat(rows, ignore_index=True)
    long = long.drop_duplicates(subset=["Owner", "WEEK_END_DATE"], keep="last").reset_index(drop=True)
    logger.info(
        "load_alist_bigrelease_long: %d rows, weeks %s → %s",
        len(long),
        long["WEEK_END_DATE"].min().date() if len(long) else None,
        long["WEEK_END_DATE"].max().date() if len(long) else None,
    )
    return long


def build_wk_minus(weekly_labels: pd.DataFrame, alist_long: pd.DataFrame) -> pd.DataFrame:
    """
    Scrub A-list AE from each Owner's weekly volume:
    ``AE_Volume = AE_Volume - A_LIST_AE[Owner, week]``.
    """
    weekly = weekly_labels.copy()
    weekly["Week Ending Date"] = pd.to_datetime(weekly["Week Ending Date"])
    scrub = alist_long.copy()
    scrub["WEEK_END_DATE"] = pd.to_datetime(scrub["WEEK_END_DATE"])

    # Train only on weeks covered by the combined alist/big-release file.
    if not scrub.empty:
        min_scrub_week = scrub["WEEK_END_DATE"].min()
        before = len(weekly)
        weekly = weekly[weekly["Week Ending Date"] >= min_scrub_week].reset_index(drop=True)
        dropped = before - len(weekly)
        if dropped:
            logger.info(
                "build_wk_minus: dropped %d weeks before alist coverage (%s)",
                dropped,
                min_scrub_week.date().isoformat(),
            )

    wk_minus = weekly.merge(
        scrub[["WEEK_END_DATE", "Owner", "A_LIST_AE", "BIG_RELEASE_FLAG", "MARKET_ALBUMS"]],
        left_on=["Week Ending Date", "Owner"],
        right_on=["WEEK_END_DATE", "Owner"],
        how="left",
    )
    wk_minus["A_LIST_AE"] = wk_minus["A_LIST_AE"].fillna(0.0)
    wk_minus["BIG_RELEASE_FLAG"] = wk_minus["BIG_RELEASE_FLAG"].fillna(0.0)
    wk_minus["MARKET_ALBUMS"] = wk_minus["MARKET_ALBUMS"].fillna(0.0)
    wk_minus["AE_Volume"] = wk_minus["AE_Volume"] - wk_minus["A_LIST_AE"]
    wk_minus["Competitor_AE_Volume"] = wk_minus["Total_Market_AE_Volume"] - wk_minus["AE_Volume"]
    wk_minus["AE_Share"] = np.where(
        wk_minus["Total_Market_AE_Volume"] > 0,
        (wk_minus["AE_Volume"] / wk_minus["Total_Market_AE_Volume"]) * 100,
        0.0,
    )
    return wk_minus


def prepare_df_model(wk_minus: pd.DataFrame) -> pd.DataFrame:
    df_model = wk_minus.copy()
    df_model["Week Ending Date"] = pd.to_datetime(df_model["Week Ending Date"])
    df_model = df_model.sort_values(["Owner", "Week Ending Date"]).reset_index(drop=True)
    df_model["Competitor_AE_Volume"] = df_model["Total_Market_AE_Volume"] - df_model["AE_Volume"]
    df_model["Lag1_AE_Volume"] = df_model.groupby("Owner")["AE_Volume"].shift(1)
    df_model["Lag1_Competitor_Volume"] = df_model.groupby("Owner")["Competitor_AE_Volume"].shift(1)
    df_model["Roll4W_AE_Volume"] = df_model.groupby("Owner")["AE_Volume"].transform(
        lambda x: x.shift(1).rolling(window=4).mean()
    )
    df_model["Roll4W_Competitor_Volume"] = df_model.groupby("Owner")["Competitor_AE_Volume"].transform(
        lambda x: x.shift(1).rolling(window=4).mean()
    )
    df_model = df_model.dropna(subset=["Roll4W_AE_Volume"]).reset_index(drop=True)
    df_model["Owner"] = pd.Categorical(df_model["Owner"], categories=TRAINING_CATEGORIES)
    return df_model


def train_lgbm_prophet(df_model: pd.DataFrame) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
    """Pooled quantile LGBM (Owner categorical) + one Prophet per Owner."""
    features = [
        "Owner",
        "Lag1_AE_Volume",
        "Lag1_Competitor_Volume",
        "Roll4W_AE_Volume",
        "Roll4W_Competitor_Volume",
    ]
    X_full = df_model[features].copy()
    X_full["Owner"] = pd.Categorical(X_full["Owner"], categories=TRAINING_CATEGORIES)
    y_full = df_model["AE_Share"]

    production_lgbm = lgb.LGBMRegressor(
        objective="quantile",
        alpha=0.5,
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
    )
    production_lgbm.fit(X_full, y_full, categorical_feature=["Owner"])

    production_prophet_models: Dict[str, Any] = {}
    for label in TRAINING_CATEGORIES:
        label_df = df_model[df_model["Owner"] == label].copy()
        if label_df.empty:
            logger.warning("train_lgbm_prophet: no rows for Owner=%s; skipping Prophet", label)
            continue
        prophet_train = label_df[["Week Ending Date", "AE_Share"]].rename(
            columns={"Week Ending Date": "ds", "AE_Share": "y"}
        )
        prophet_model = Prophet(
            yearly_seasonality=2,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        prophet_model.fit(prophet_train.dropna())
        production_prophet_models[label] = prophet_model

    return production_lgbm, production_prophet_models

def conformal_e80_2026(
    df_model: pd.DataFrame,
    production_lgbm: lgb.LGBMRegressor,
    production_prophet_models: Dict[str, Any],
    forecast_year: int = 2026,
) -> float:
    backtest_raw = df_model[df_model["Week Ending Date"] >= "2025-11-01"].copy()
    backtest_raw = backtest_raw[backtest_raw["Owner"].isin(TARGET_LABELS)].sort_values(
        ["Owner", "Week Ending Date"]
    )
    
    ensemble_errors: List[pd.Series] = []
    
    for label in TARGET_LABELS:
        if label not in production_prophet_models:
            continue
        df_label = backtest_raw[backtest_raw["Owner"] == label].copy()

        df_label = df_label[forecast_year_row_mask(df_label, forecast_year)].dropna()
        if df_label.empty:
            continue

        lgbm_features = df_label[
            ["Owner", "Lag1_AE_Volume", "Lag1_Competitor_Volume", "Roll4W_AE_Volume", "Roll4W_Competitor_Volume"]
        ].copy()
        lgbm_features["Owner"] = pd.Categorical(lgbm_features["Owner"], categories=TRAINING_CATEGORIES)

        df_label["LGBM_Pred"] = production_lgbm.predict(lgbm_features)

        prophet_input = df_label[["Week Ending Date"]].rename(columns={"Week Ending Date": "ds"})
        df_label["Prophet_Pred"] = production_prophet_models[label].predict(prophet_input)["yhat"].values

        df_label["Ensemble_Share"] = (df_label["LGBM_Pred"] + df_label["Prophet_Pred"]) / 2

        df_label["Actual_Weekly_Share"] = (df_label["AE_Volume"] / df_label["Total_Market_AE_Volume"]) * 100
        df_label["Abs_Error"] = (df_label["Actual_Weekly_Share"] - df_label["Ensemble_Share"]).abs()

        ensemble_errors.append(df_label["Abs_Error"])
        
    if not ensemble_errors:
        return 0.82
        
    all_errors = pd.concat(ensemble_errors, ignore_index=True)
    return float(np.percentile(all_errors, 80))


def forecast_baseline_future(
    df_model: pd.DataFrame,
    weekly_amg_int: pd.DataFrame,
    production_lgbm: lgb.LGBMRegressor,
    production_prophet_models: Dict[str, Any],
    end_of_year: str,
    week_freq: str = "W-THU",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (future_label_df, future_market_volumes) using an autoregressive loop."""
    last_date = pd.to_datetime(df_model["Week Ending Date"].max())
    end_dt = pd.to_datetime(end_of_year)
    remaining_weeks = pd.date_range(start=last_date + pd.Timedelta(days=7), end=end_dt, freq=week_freq)

    # 1. Forecast Total Market First (So we have the denominator)
    market_history = weekly_amg_int[["Week Ending Date", "Total_Market_AE_Volume"]].drop_duplicates().sort_values("Week Ending Date")
    market_prophet_train = market_history.rename(columns={"Week Ending Date": "ds", "Total_Market_AE_Volume": "y"})
    market_model = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False)
    market_model.fit(market_prophet_train.dropna())
    future_market = pd.DataFrame({"ds": remaining_weeks})
    market_forecast = market_model.predict(future_market)
    future_market_volumes = market_forecast[["ds", "yhat"]].rename(
        columns={"ds": "Week Ending Date", "yhat": "Predicted_Total_Market_Volume"}
    )

    # 2. Autoregressive Loop for Labels
    future_label_df = pd.DataFrame()
    for label in TARGET_LABELS:
        if label not in production_prophet_models:
            logger.warning("forecast_baseline_future: skip %s (no Prophet model)", label)
            continue
        label_history = df_model[df_model["Owner"] == label].sort_values("Week Ending Date")
        if label_history.empty or len(label_history) < 4:
            logger.warning("forecast_baseline_future: skip %s (need ≥4 history weeks)", label)
            continue
        # Keep the last 4 weeks of volume in a list to seed the rolling averages
        recent_vols = list(label_history.tail(4)["AE_Volume"].values)
        recent_comp = list(label_history.tail(4)["Competitor_AE_Volume"].values)

        temp_rows = []
        for i, week_dt in enumerate(remaining_weeks):
            # Calculate dynamic lags based on the running memory
            current_lag1_ae = recent_vols[-1]
            current_lag1_comp = recent_comp[-1]
            current_roll4w_ae = sum(recent_vols[-4:]) / 4
            current_roll4w_comp = sum(recent_comp[-4:]) / 4

            # Predict LightGBM
            lgbm_in = pd.DataFrame({
                "Owner": [label],
                "Lag1_AE_Volume": [current_lag1_ae],
                "Lag1_Competitor_Volume": [current_lag1_comp],
                "Roll4W_AE_Volume": [current_roll4w_ae],
                "Roll4W_Competitor_Volume": [current_roll4w_comp]
            })
            lgbm_in["Owner"] = pd.Categorical(lgbm_in["Owner"], categories=TRAINING_CATEGORIES)
            lgbm_pred = float(production_lgbm.predict(lgbm_in)[0])

            # Predict Prophet
            prophet_pred = float(production_prophet_models[label].predict(pd.DataFrame({"ds": [week_dt]}))["yhat"].iloc[0])

            pred_weekly_share = (lgbm_pred + prophet_pred) / 2

            temp_rows.append({
                "Week Ending Date": week_dt,
                "Owner": label,
                "Lag1_AE_Volume": current_lag1_ae,
                "Lag1_Competitor_Volume": current_lag1_comp,
                "Roll4W_AE_Volume": current_roll4w_ae,
                "Roll4W_Competitor_Volume": current_roll4w_comp,
                "LGBM_Share": lgbm_pred,
                "Prophet_Share": prophet_pred,
                "Predicted_Weekly_Share": pred_weekly_share
            })

            # AUTOREGRESSION: Convert predicted share back to volume for next week's input
            current_market_vol = future_market_volumes.iloc[i]["Predicted_Total_Market_Volume"]
            new_ae_vol = (pred_weekly_share / 100) * current_market_vol
            new_comp_vol = current_market_vol - new_ae_vol

            recent_vols.append(new_ae_vol)
            recent_comp.append(new_comp_vol)

        future_label_df = pd.concat([future_label_df, pd.DataFrame(temp_rows)], ignore_index=True)

    return future_label_df, future_market_volumes


def build_ytd_projections(
    df_2026_base: pd.DataFrame,
    ytd_bank: Dict[str, Dict[str, float]],
    final_forecast: pd.DataFrame,
    e_score: float,
) -> pd.DataFrame:
    ytd_projections = pd.DataFrame()
    for label in TARGET_LABELS:
        if label not in ytd_bank:
            continue
        lf = final_forecast[final_forecast["Owner"] == label].sort_values("Week Ending Date").copy()
        if lf.empty:
            continue
        banked_num = ytd_bank[label]["Banked_Numerator"]
        banked_den = ytd_bank[label]["Banked_Denominator"]
        lf["Weekly_Num"] = lf["Predicted_Weekly_Share"] * lf["Predicted_Total_Market_Volume"]
        lf["Proj_Cum_Num"] = banked_num + lf["Weekly_Num"].cumsum()
        lf["Proj_Cum_Den"] = banked_den + lf["Predicted_Total_Market_Volume"].cumsum()
        lf["Projected_YTD_Share"] = lf["Proj_Cum_Num"] / lf["Proj_Cum_Den"]
        lf["Upper_Weekly_Num"] = (lf["Predicted_Weekly_Share"] + e_score) * lf["Predicted_Total_Market_Volume"]
        lf["Lower_Weekly_Num"] = (lf["Predicted_Weekly_Share"] - e_score) * lf["Predicted_Total_Market_Volume"]
        lf["Cum_Num_Upper"] = banked_num + lf["Upper_Weekly_Num"].cumsum()
        lf["Cum_Num_Lower"] = banked_num + lf["Lower_Weekly_Num"].cumsum()
        lf["YTD_Share_Upper"] = lf["Cum_Num_Upper"] / lf["Proj_Cum_Den"]
        lf["YTD_Share_Lower"] = lf["Cum_Num_Lower"] / lf["Proj_Cum_Den"]
        ytd_projections = pd.concat([ytd_projections, lf], ignore_index=True)
    return ytd_projections


def enrich_weekly_for_spike(
    weekly_labels: pd.DataFrame,
    alist_long: pd.DataFrame,
    wk_minus: pd.DataFrame,
    big_release_flag: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach per-Owner A-list incremental volume and big-release flags.

    ``big_release_flag`` is ignored when ``alist_long`` already carries
    ``BIG_RELEASE_FLAG`` (combined ``bigrelease_alist_75k_all.csv``).
    """
    w = weekly_labels.copy()
    w["Week Ending Date"] = pd.to_datetime(w["Week Ending Date"])
    scrub = alist_long.copy()
    scrub["WEEK_END_DATE"] = pd.to_datetime(scrub["WEEK_END_DATE"])
    w = w.merge(
        scrub[["WEEK_END_DATE", "Owner", "A_LIST_AE", "BIG_RELEASE_FLAG"]],
        left_on=["Week Ending Date", "Owner"],
        right_on=["WEEK_END_DATE", "Owner"],
        how="left",
    )
    w["Incremental_Volume"] = w["A_LIST_AE"].fillna(0.0)
    w["big_release_flag"] = w["BIG_RELEASE_FLAG"].fillna(0.0)
    w["Incremental_Share"] = np.where(
        w["Total_Market_AE_Volume"] > 0,
        (w["Incremental_Volume"] / w["Total_Market_AE_Volume"]) * 100,
        0.0,
    )
    scrubbed_vols = wk_minus[["Week Ending Date", "Owner", "AE_Volume"]].rename(
        columns={"AE_Volume": "Baseline_Volume"}
    )
    w = w.merge(scrubbed_vols, on=["Week Ending Date", "Owner"], how="left")
    # Weeks before alist coverage have no wk_minus row: treat as zero A-list scrub.
    w["Baseline_Volume"] = w["Baseline_Volume"].fillna(w["AE_Volume"])
    w["Historical_Baseline_Share"] = np.where(
        w["Total_Market_AE_Volume"] > 0,
        (w["Baseline_Volume"] / w["Total_Market_AE_Volume"]) * 100,
        np.nan,
    )
    w["Predicted_Baseline_Share"] = w["Historical_Baseline_Share"]
    return w


def train_spike_and_df_full(
    weekly_amg_int_enriched: pd.DataFrame,
    ytd_projections: pd.DataFrame,
) -> Tuple[Any, pd.DataFrame, List[str]]:
    df_model_b = weekly_amg_int_enriched.sort_values(["Owner", "Week Ending Date"]).reset_index(drop=True)
    df_model_b["Lag1_Inc_Share"] = df_model_b.groupby("Owner")["Incremental_Share"].shift(1).fillna(0)
    df_model_b["Lag2_Inc_Share"] = df_model_b.groupby("Owner")["Incremental_Share"].shift(2).fillna(0)
    df_model_b["Lag1_Flag"] = df_model_b.groupby("Owner")["big_release_flag"].shift(1).fillna(0)
    df_model_b["Lag2_Flag"] = df_model_b.groupby("Owner")["big_release_flag"].shift(2).fillna(0)
    spike_zone = (
        (df_model_b["big_release_flag"] == 1)
        | (df_model_b["Lag1_Flag"] == 1)
        | (df_model_b["Lag2_Flag"] == 1)
    )
    features = ["Predicted_Baseline_Share", "Incremental_Share", "Lag1_Inc_Share", "Lag2_Inc_Share"]
    df_ready = df_model_b[spike_zone].copy()
    df_ready = df_ready.dropna(subset=features + ["AE_Share"])
    X = df_ready[features]
    y = df_ready["AE_Share"]
    spike_engine = Ridge(alpha=1.0)
    spike_engine.fit(X, y)

    hist_cols = ["Week Ending Date", "Owner", "Predicted_Baseline_Share", "Incremental_Share", "big_release_flag"]
    df_hist = weekly_amg_int_enriched[hist_cols].copy()
    max_hist_date = df_hist["Week Ending Date"].max()
    future_col = "Predicted_Weekly_Share"
    df_future = ytd_projections[ytd_projections["Week Ending Date"] > max_hist_date].copy()
    df_future = df_future.rename(columns={future_col: "Predicted_Baseline_Share"})
    df_future["Incremental_Share"] = 0.0
    df_future["big_release_flag"] = 0.0
    df_full = pd.concat(
        [
            df_hist,
            df_future[
                ["Week Ending Date", "Owner", "Predicted_Baseline_Share", "Incremental_Share", "big_release_flag"]
            ],
        ],
        ignore_index=True,
    )
    df_full["Week Ending Date"] = pd.to_datetime(df_full["Week Ending Date"])
    df_full = df_full.sort_values(by=["Owner", "Week Ending Date"]).reset_index(drop=True)
    df_full["Lag1_Inc_Share"] = df_full.groupby("Owner")["Incremental_Share"].shift(1).fillna(0)
    df_full["Lag2_Inc_Share"] = df_full.groupby("Owner")["Incremental_Share"].shift(2).fillna(0)
    df_full["Lag1_Flag"] = df_full.groupby("Owner")["big_release_flag"].shift(1).fillna(0)
    df_full["Lag2_Flag"] = df_full.groupby("Owner")["big_release_flag"].shift(2).fillna(0)
    df_full["Spike_Engine_Pred"] = spike_engine.predict(df_full[features])
    spike_zone_mask = (
        (df_full["big_release_flag"] == 1) | (df_full["Lag1_Flag"] == 1) | (df_full["Lag2_Flag"] == 1)
    )
    df_full["Final_Unified_Share"] = np.where(
        spike_zone_mask,
        df_full["Spike_Engine_Pred"],
        df_full["Predicted_Baseline_Share"],
    )
    return spike_engine, df_full, features


def attach_total_market_volume(
    df_full: pd.DataFrame,
    weekly_amg_int: pd.DataFrame,
    future_forecast_yhat: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weekly total market volume: history + Prophet future."""
    hist_vol = weekly_amg_int[["Week Ending Date", "Total_Market_AE_Volume"]].drop_duplicates()
    max_hist_date = hist_vol["Week Ending Date"].max()
    fut_vol = future_forecast_yhat.copy()
    fut_vol = fut_vol.rename(columns={"ds": "Week Ending Date", "yhat": "Total_Market_AE_Volume"})
    fut_vol = fut_vol[fut_vol["Week Ending Date"] > max_hist_date]
    master_vol = pd.concat([hist_vol, fut_vol], ignore_index=True)
    master_vol["Week Ending Date"] = pd.to_datetime(master_vol["Week Ending Date"])
    out = df_full.drop(columns=["Total_Market_AE_Volume"], errors="ignore").merge(
        master_vol, on="Week Ending Date", how="left"
    )
    return out


def build_actuals_2026(weekly_amg_int: pd.DataFrame, forecast_year: int = 2026) -> pd.DataFrame:
    actuals_2026 = weekly_amg_int[
        forecast_year_row_mask(weekly_amg_int, forecast_year)
        & weekly_amg_int["Owner"].isin(TARGET_LABELS)
    ].copy()
    actuals_2026 = actuals_2026.sort_values(by=["Owner", "Week Ending Date"]).reset_index(drop=True)
    dup_n = actuals_2026.duplicated(subset=["Owner", "Week Ending Date"]).sum()
    if dup_n:
        logger.warning(
            "actuals_2026: dropping %d duplicate Owner×Week rows (keeps last)",
            int(dup_n),
        )
        actuals_2026 = actuals_2026.drop_duplicates(
            subset=["Owner", "Week Ending Date"], keep="last"
        ).reset_index(drop=True)
    actuals_2026["Weighted_Numerator"] = actuals_2026["AE_Share"] * actuals_2026["Total_Market_AE_Volume"]
    actuals_2026["Cum_Numerator"] = actuals_2026.groupby(["Owner"])["Weighted_Numerator"].cumsum()
    actuals_2026["Cum_Denominator"] = actuals_2026.groupby(["Owner"])["Total_Market_AE_Volume"].cumsum()
    actuals_2026["Calculated_YTD_Share"] = (
        actuals_2026["Cum_Numerator"] / actuals_2026["Cum_Denominator"]
    ).round(2)
    return actuals_2026


def _week_vol(g: pd.DataFrame, w: int) -> float:
    s = g.loc[g["WEEKS_SINCE_RELEASE"] == w, "TOTAL_ALBUM_EQUIVALENTS"]
    return float(s.iloc[0]) if not s.empty else 0.0


def _release_w2_over_w1_ratio(
    g: pd.DataFrame,
    min_w1_share_of_early_peak: float = 0.12,
) -> Optional[float]:
    """Robust W2/W1 ratio from AE parquet release history."""
    v0, v1, v2, v3 = (
        _week_vol(g, 0),
        _week_vol(g, 1),
        _week_vol(g, 2),
        _week_vol(g, 3),
    )
    if v2 <= 0:
        return None
    peak_early = max(v0, v1, v2, v3)
    if peak_early <= 0:
        return None
    if v2 > v1 * 1.005:
        denom = max(v0, v1, v2, v3)
        return float(min(v2 / denom, 1.0)) if denom > 0 else None
    if v1 > 0 and v1 < peak_early * min_w1_share_of_early_peak:
        return None
    equiv_w1 = max(v0, v1) if max(v0, v1) > 0 else v1
    if equiv_w1 <= 0:
        return None
    return float(min(v2 / equiv_w1, 1.0))


def build_artist_w2_retention_from_ae_parquet(ae_parquet: Path) -> Dict[str, Dict[str, float]]:
    cols = ["DISPLAY_ARTIST", "MRELG_ID", "WEEKS_SINCE_RELEASE", "TOTAL_ALBUM_EQUIVALENTS"]
    df = pd.read_parquet(ae_parquet, columns=cols)
    df["TOTAL_ALBUM_EQUIVALENTS"] = pd.to_numeric(df["TOTAL_ALBUM_EQUIVALENTS"], errors="coerce")
    df["WEEKS_SINCE_RELEASE"] = pd.to_numeric(df["WEEKS_SINCE_RELEASE"], errors="coerce")
    df = df.dropna(subset=["WEEKS_SINCE_RELEASE", "TOTAL_ALBUM_EQUIVALENTS"])
    ratios: List[Dict[str, Any]] = []
    for (artist, _mid), g in df.groupby(["DISPLAY_ARTIST", "MRELG_ID"], sort=False):
        r = _release_w2_over_w1_ratio(g)
        if r is not None:
            ratios.append({"DISPLAY_ARTIST": str(artist), "ratio": r})
    if not ratios:
        return {}
    rdf = pd.DataFrame(ratios)
    agg = rdf.groupby("DISPLAY_ARTIST", sort=False).agg(
        median_w2_over_w1=("ratio", "median"),
        n_releases=("ratio", "count"),
    )
    out: Dict[str, Dict[str, float]] = {}
    for artist, row in agg.iterrows():
        out[str(artist)] = {
            "median_w2_over_w1": float(row["median_w2_over_w1"]),
            "n_releases": int(row["n_releases"]),
        }
    return out


def _coerce_ae_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Snowflake/pyarrow parquets often store metrics as decimal.Decimal."""
    out = df.copy()
    for c in (
        "WEEKS_SINCE_RELEASE",
        "PRODUCT_SALES",
        "STREAMING_EQUIVALENT",
        "SONG_SALE_EQUIVALENT",
        "TOTAL_ALBUM_EQUIVALENTS",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _ae_release_pivot_from_parquet(ae_parquet: Path) -> pd.DataFrame:
    cols = [
        "MRELG_ID",
        "TITLE",
        "DISPLAY_ARTIST",
        "GENRES",
        "WEEKS_SINCE_RELEASE",
        "PRODUCT_SALES",
        "TOTAL_ALBUM_EQUIVALENTS",
    ]
    df = pd.read_parquet(ae_parquet, columns=cols)
    df = _coerce_ae_metric_columns(df)
    df = df.dropna(subset=["WEEKS_SINCE_RELEASE"]).copy()
    pivot = df.pivot_table(
        index=["MRELG_ID", "TITLE", "DISPLAY_ARTIST", "GENRES"],
        columns="WEEKS_SINCE_RELEASE",
        values=["TOTAL_ALBUM_EQUIVALENTS", "PRODUCT_SALES"],
        aggfunc="sum",
    ).reset_index()
    pivot.columns = [
        f"{col[0]}_w{int(col[1])}" if col[1] != "" else col[0] for col in pivot.columns.values
    ]
    return pivot


def kmeans_and_dna_from_ae_parquet(
    ae_parquet: Path,
) -> Tuple[Dict[str, Dict[int, float]], Any, Dict[str, int], Dict[str, int]]:
    cols = ["MRELG_ID", "TITLE", "DISPLAY_ARTIST", "WEEKS_SINCE_RELEASE", "TOTAL_ALBUM_EQUIVALENTS"]
    df = pd.read_parquet(ae_parquet, columns=cols)
    df = _coerce_ae_metric_columns(df)
    df["project_label"] = df["DISPLAY_ARTIST"].astype(str).fillna("") + " - " + df["TITLE"].astype(str).fillna("")
    lifespan = df.groupby("project_label")["WEEKS_SINCE_RELEASE"].max()
    mature = lifespan[lifespan >= 52].index
    df = df[df["project_label"].isin(mature)].copy()
    df_52 = df[df["WEEKS_SINCE_RELEASE"] <= 52].copy()
    shape_df = df_52.pivot_table(
        index="project_label",
        columns="WEEKS_SINCE_RELEASE",
        values="TOTAL_ALBUM_EQUIVALENTS",
        aggfunc="sum",
    ).fillna(0)
    peaks = shape_df.max(axis=1)
    normalized = shape_df.div(peaks.replace(0, np.nan), axis=0).replace([np.inf, -np.inf], np.nan).dropna()
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized.to_numpy())
    normalized = normalized.copy()
    normalized["archetype_cluster"] = labels
    historical = normalized[["archetype_cluster"]].reset_index()
    features = df.merge(historical, on="project_label", how="left")
    probs = (
        features.groupby(["DISPLAY_ARTIST", "archetype_cluster"], sort=False)
        .agg(count=("MRELG_ID", "nunique"))
        .reset_index()
    )
    probs["prob"] = probs.groupby("DISPLAY_ARTIST")["count"].transform(lambda x: x / x.sum())
    artist_dna_lookup = probs.groupby("DISPLAY_ARTIST").apply(
        lambda x: dict(zip(x["archetype_cluster"], x["prob"]))
    ).to_dict()
    project_to_cluster = normalized["archetype_cluster"].astype(int).to_dict()
    mrelg_rows = features.dropna(subset=["archetype_cluster"]).drop_duplicates(subset=["MRELG_ID"], keep="first")
    mrelg_to_cluster = {str(r["MRELG_ID"]): int(r["archetype_cluster"]) for _, r in mrelg_rows.iterrows()}
    return artist_dna_lookup, kmeans, project_to_cluster, mrelg_to_cluster


def build_artist_profile_dict_from_ae_parquet(ae_parquet: Path) -> Dict[str, float]:
    pvt = _ae_release_pivot_from_parquet(ae_parquet)
    if "PRODUCT_SALES_w1" not in pvt.columns or "TOTAL_ALBUM_EQUIVALENTS_w1" not in pvt.columns:
        return {}
    den = pd.to_numeric(pvt["TOTAL_ALBUM_EQUIVALENTS_w1"], errors="coerce").replace(0, np.nan)
    num = pd.to_numeric(pvt["PRODUCT_SALES_w1"], errors="coerce")
    pvt["w1_product_ratio"] = num / den
    pvt["w1_product_ratio"] = pvt["w1_product_ratio"].replace([np.inf, -np.inf], np.nan)
    agg = pvt.dropna(subset=["w1_product_ratio"]).groupby("DISPLAY_ARTIST")["w1_product_ratio"].median()
    return {str(k): float(v) for k, v in agg.items()}


def _row_m52_multiplier_from_pivot(row: pd.Series) -> float:
    total = 0.0
    for k in range(0, 52):
        col = f"TOTAL_ALBUM_EQUIVALENTS_w{k}"
        if col in row.index and pd.notna(row[col]):
            total += float(row[col])
    w1_col = "TOTAL_ALBUM_EQUIVALENTS_w1"
    if w1_col not in row.index or pd.isna(row[w1_col]) or float(row[w1_col]) <= 0:
        return float("nan")
    return total / float(row[w1_col])


def fit_cluster_product_coefficients_from_ae_parquet(
    ae_parquet: Path,
    project_to_cluster: Dict[str, int],
    mrelg_to_cluster: Dict[str, int],
    global_fallback: float = GLOBAL_PRODUCT_COEF,
    min_cluster_n: int = 25,
    ridge_alpha: float = 2.0,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    pvt = _ae_release_pivot_from_parquet(ae_parquet)
    if "PRODUCT_SALES_w1" not in pvt.columns or "TOTAL_ALBUM_EQUIVALENTS_w1" not in pvt.columns:
        return {}, {}
    pvt["w1_product_ratio"] = pvt["PRODUCT_SALES_w1"] / pvt["TOTAL_ALBUM_EQUIVALENTS_w1"]
    pvt["w1_product_ratio"] = pvt["w1_product_ratio"].replace([np.inf, -np.inf], np.nan).clip(lower=0.0, upper=1.0)
    pvt["y_target"] = pvt.apply(_row_m52_multiplier_from_pivot, axis=1)
    mid = pvt["MRELG_ID"].astype(str)
    pvt["archetype_cluster"] = mid.map(mrelg_to_cluster)
    miss = pvt["archetype_cluster"].isna()
    if miss.any():
        pvt.loc[miss, "archetype_cluster"] = (
            pvt.loc[miss, "DISPLAY_ARTIST"].astype(str).fillna("")
            + " - "
            + pvt.loc[miss, "TITLE"].astype(str).fillna("")
        ).map(project_to_cluster)
    sub = pvt.dropna(subset=["y_target", "w1_product_ratio", "archetype_cluster"]).copy()
    sub["archetype_cluster"] = sub["archetype_cluster"].astype(int)

    coefs: Dict[int, float] = {}
    meta: Dict[str, Any] = {}
    for c in sorted(sub["archetype_cluster"].unique()):
        if c not in (0, 1, 2, 3):
            continue
        chunk = sub[sub["archetype_cluster"] == c]
        n = len(chunk)
        if n < min_cluster_n:
            coefs[int(c)] = float(global_fallback)
            meta[str(c)] = {"n": n, "slope": None, "fallback": True}
            continue
        X = chunk[["w1_product_ratio"]].to_numpy(dtype=float)
        y = chunk["y_target"].to_numpy(dtype=float)
        model = Ridge(alpha=ridge_alpha)
        model.fit(X, y)
        coefs[int(c)] = float(model.coef_[0])
        meta[str(c)] = {
            "n": n,
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "fallback": False,
        }
    for c in (0, 1, 2, 3):
        coefs.setdefault(c, float(global_fallback))
    return coefs, meta


def main() -> None:
    p = argparse.ArgumentParser(description="Train and export 75k marketshare artifacts.")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "Data")
    p.add_argument("--artifacts-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts_75k")
    p.add_argument("--end-of-year", type=str, default="2026-12-31")
    p.add_argument("--forecast-year", type=int, default=2026)
    args = p.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    art_dir = args.artifacts_dir.expanduser().resolve()
    art_dir.mkdir(parents=True, exist_ok=True)

    weekly_labels = load_weekly_labels(data_dir)
    alist_long = load_alist_bigrelease_long(data_dir)
    wk_minus = build_wk_minus(weekly_labels, alist_long)

    df_model = prepare_df_model(wk_minus)
    production_lgbm, production_prophet_models = train_lgbm_prophet(df_model)
    e80 = conformal_e80_2026(
        df_model, production_lgbm, production_prophet_models, forecast_year=args.forecast_year
    )
    e_score = e80
    logger.info("Conformal 80th percentile E: %.4f (used as E_score for YTD bands)", e_score)

    future_label_df, future_market_volumes = forecast_baseline_future(
        df_model, weekly_labels, production_lgbm, production_prophet_models, args.end_of_year
    )
    final_forecast = pd.merge(future_label_df, future_market_volumes, on="Week Ending Date", how="inner")

    df_2026_base = wk_minus[forecast_year_row_mask(wk_minus, args.forecast_year)].copy()
    df_2026_base = df_2026_base[df_2026_base["Owner"].isin(TARGET_LABELS)].sort_values(
        ["Owner", "Week Ending Date"]
    ).reset_index(drop=True)
    df_2026_base["Baseline_Weekly_Share"] = (
        df_2026_base["AE_Volume"] / df_2026_base["Total_Market_AE_Volume"]
    ) * 100
    df_2026_base["Weighted_Numerator"] = df_2026_base["Baseline_Weekly_Share"] * df_2026_base["Total_Market_AE_Volume"]
    df_2026_base["Cum_Numerator"] = df_2026_base.groupby("Owner")["Weighted_Numerator"].cumsum()
    df_2026_base["Cum_Denominator"] = df_2026_base.groupby("Owner")["Total_Market_AE_Volume"].cumsum()
    df_2026_base["Baseline_YTD_Share"] = df_2026_base["Cum_Numerator"] / df_2026_base["Cum_Denominator"]

    last_actual_date = df_2026_base["Week Ending Date"].max()
    ytd_bank = {}
    for label in TARGET_LABELS:
        rows = df_2026_base[
            (df_2026_base["Owner"] == label) & (df_2026_base["Week Ending Date"] == last_actual_date)
        ]
        if rows.empty:
            logger.warning("YTD bank: no row for %s at %s; skipping", label, last_actual_date.date())
            continue
        latest_row = rows.iloc[0]
        ytd_bank[label] = {
            "Banked_Numerator": float(latest_row["Cum_Numerator"]),
            "Banked_Denominator": float(latest_row["Cum_Denominator"]),
            "Current_YTD_Share": float(latest_row["Baseline_YTD_Share"]),
        }

    ytd_projections = build_ytd_projections(df_2026_base, ytd_bank, final_forecast, e_score=e_score)

    weekly_enriched = enrich_weekly_for_spike(weekly_labels, alist_long, wk_minus)
    spike_engine, df_full, spike_features = train_spike_and_df_full(weekly_enriched, ytd_projections)

    # Market volume for future weeks (Prophet on total market) — reuse series from forecast_baseline_future
    market_history = weekly_labels[["Week Ending Date", "Total_Market_AE_Volume"]].drop_duplicates().sort_values(
        "Week Ending Date"
    )
    market_prophet_train = market_history.rename(columns={"Week Ending Date": "ds", "Total_Market_AE_Volume": "y"})
    market_model = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False)
    market_model.fit(market_prophet_train.dropna())
    last_date = pd.to_datetime(df_model["Week Ending Date"].max())
    remaining_weeks = pd.date_range(
        start=last_date + pd.Timedelta(days=7), end=pd.to_datetime(args.end_of_year), freq="W-THU"
    )
    full_fc = market_model.predict(pd.DataFrame({"ds": remaining_weeks}))
    df_full = attach_total_market_volume(df_full, weekly_labels, full_fc[["ds", "yhat"]])

    actuals_2026 = build_actuals_2026(weekly_labels, forecast_year=args.forecast_year)

    ae_parquet = data_dir / "streams_product_songs_ae_compressed.parquet"
    artist_dna_lookup: Dict[str, Dict[int, float]] = {}
    artist_w2_retention: Dict[str, Dict[str, float]] = {}
    artist_profile_dict: Dict[str, float] = {}
    cluster_product_coef: Dict[int, float] = {}
    cluster_product_meta: Dict[str, Any] = {}
    if ae_parquet.exists():
        try:
            artist_dna_lookup, kmeans_model, project_to_cluster, mrelg_to_cluster = (
                kmeans_and_dna_from_ae_parquet(ae_parquet)
            )
            joblib.dump(kmeans_model, art_dir / "kmeans_archetype_75k.pkl")
            artist_w2_retention = build_artist_w2_retention_from_ae_parquet(ae_parquet)
            artist_profile_dict = build_artist_profile_dict_from_ae_parquet(ae_parquet)
            cluster_product_coef, cluster_product_meta = fit_cluster_product_coefficients_from_ae_parquet(
                ae_parquet,
                project_to_cluster,
                mrelg_to_cluster,
                global_fallback=GLOBAL_PRODUCT_COEF,
            )
        except Exception as e:
            logger.warning("Parquet-derived enrichment failed (%s); using fallback globals.", e)
    if not cluster_product_coef:
        cluster_product_coef = {c: float(GLOBAL_PRODUCT_COEF) for c in (0, 1, 2, 3)}

    joblib.dump(production_lgbm, art_dir / "production_lgbm_75k.pkl")
    joblib.dump(production_prophet_models, art_dir / "production_prophet_models_75k.pkl")
    joblib.dump(spike_engine, art_dir / "production_spike_engine.pkl")

    df_full.to_parquet(art_dir / "df_full.parquet", index=False)
    actuals_2026.to_parquet(art_dir / "actuals_2026.parquet", index=False)

    with open(art_dir / "artist_profile_dict.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in artist_profile_dict.items()}, f, indent=2)
    with open(art_dir / "artist_dna_lookup.json", "w", encoding="utf-8") as f:
        json.dump(dna_lookup_to_jsonable(artist_dna_lookup), f, indent=2)
    with open(art_dir / "artist_w2_retention.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in artist_w2_retention.items()}, f, indent=2)
    with open(art_dir / "distributions.json", "w", encoding="utf-8") as f:
        json.dump(DISTRIBUTIONS, f, indent=2)
    with open(art_dir / "cluster_product_coef.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in sorted(cluster_product_coef.items())}, f, indent=2)
    if cluster_product_meta:
        with open(art_dir / "cluster_product_regression.json", "w", encoding="utf-8") as f:
            json.dump(cluster_product_meta, f, indent=2)

    meta = {
        "GLOBAL_PRODUCT_COEF": GLOBAL_PRODUCT_COEF,
        #"PRODUCT_M52_PENALTY_CAP": PRODUCT_M52_PENALTY_CAP,
        "W2_RETENTION_BLEND_K": W2_RETENTION_BLEND_K,
        "E_score": e_score,
        "e80_conformal": e80,
        "forecast_year": args.forecast_year,
        "target_labels": TARGET_LABELS,
        "spike_features": spike_features,
        "end_of_year": args.end_of_year,
    }
    with open(art_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wrote artifacts to %s", art_dir)


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Could not load %s (%s); using empty dict", path, e)
        return {}


def _is_numberish(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _load_parquet_sidecars_from_artifacts_dir(
    art_dir: Path,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, Dict[str, float]], Dict[str, float], Dict[int, float], Dict[str, Any]]:
    """
    Reload DNA / profile / W2 / cluster coefficients written on a prior full
    train so a csv_only weekly run does not overwrite them with empty defaults.
    """
    artist_dna_raw = _load_json_dict(art_dir / "artist_dna_lookup.json")
    artist_dna_lookup: Dict[str, Dict[int, float]] = {}
    for artist, clusters in artist_dna_raw.items():
        if not isinstance(clusters, dict):
            continue
        inner: Dict[int, float] = {}
        for cid, val in clusters.items():
            try:
                inner[int(cid)] = float(val)
            except (TypeError, ValueError):
                continue
        if inner:
            artist_dna_lookup[str(artist)] = inner

    artist_w2_raw = _load_json_dict(art_dir / "artist_w2_retention.json")
    artist_w2_retention: Dict[str, Dict[str, float]] = {}
    for k, v in artist_w2_raw.items():
        if isinstance(v, dict):
            artist_w2_retention[str(k)] = {str(sk): float(sv) for sk, sv in v.items() if _is_numberish(sv)}

    profile_raw = _load_json_dict(art_dir / "artist_profile_dict.json")
    artist_profile_dict: Dict[str, float] = {}
    for k, v in profile_raw.items():
        try:
            artist_profile_dict[str(k)] = float(v)
        except (TypeError, ValueError):
            continue

    coef_raw = _load_json_dict(art_dir / "cluster_product_coef.json")
    cluster_product_coef: Dict[int, float] = {}
    for k, v in coef_raw.items():
        try:
            cluster_product_coef[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    cluster_product_meta = _load_json_dict(art_dir / "cluster_product_regression.json")

    return (
        artist_dna_lookup,
        artist_w2_retention,
        artist_profile_dict,
        cluster_product_coef,
        cluster_product_meta,
    )


def train_artifacts_main(*, csv_only: bool = False) -> None:
    """
    Train 75k marketshare artifacts (LGBM, Prophet, spike, df_full, etc.).

    When ``csv_only=True`` (used by ``refresh_weekly``): skip reading the AE
    parquet for KMeans/DNA/cluster regression and skip all archetype decay
    retrains (AE panel + worldwide). Existing sidecar JSON under
    ``artifacts_75k/`` is reloaded so weekly runs do not clobber parquet-derived
    files. Full parquet + archetype pipeline: call with ``csv_only=False`` or
    use ``/v1/data/refresh_model`` + a full ``refresh_data`` train.
    """
    # --- Hardcoded Configuration ---
    base_path = Path(__file__).resolve().parent
    data_dir = (base_path / "data").expanduser().resolve()
    art_dir = (base_path / "artifacts_75k").expanduser().resolve()
    end_of_year = "2026-12-31"
    forecast_year = 2026
    # -------------------------------
    
    art_dir.mkdir(parents=True, exist_ok=True)

    weekly_labels = load_weekly_labels(data_dir)
    alist_long = load_alist_bigrelease_long(data_dir)
    wk_minus = build_wk_minus(weekly_labels, alist_long)

    df_model = prepare_df_model(wk_minus)
    production_lgbm, production_prophet_models = train_lgbm_prophet(df_model)
    e80 = conformal_e80_2026(
        df_model, production_lgbm, production_prophet_models, forecast_year=forecast_year
    )
    e_score = e80
    logger.info("Conformal 80th percentile E: %.4f (used as E_score for YTD bands)", e_score)

    future_label_df, future_market_volumes = forecast_baseline_future(
        df_model, weekly_labels, production_lgbm, production_prophet_models, end_of_year
    )
    final_forecast = pd.merge(future_label_df, future_market_volumes, on="Week Ending Date", how="inner")

    df_2026_base = wk_minus[forecast_year_row_mask(wk_minus, forecast_year)].copy()
    df_2026_base = df_2026_base[df_2026_base["Owner"].isin(TARGET_LABELS)].sort_values(
        ["Owner", "Week Ending Date"]
    ).reset_index(drop=True)
    df_2026_base["Baseline_Weekly_Share"] = (
        df_2026_base["AE_Volume"] / df_2026_base["Total_Market_AE_Volume"]
    ) * 100
    df_2026_base["Weighted_Numerator"] = df_2026_base["Baseline_Weekly_Share"] * df_2026_base["Total_Market_AE_Volume"]
    df_2026_base["Cum_Numerator"] = df_2026_base.groupby("Owner")["Weighted_Numerator"].cumsum()
    df_2026_base["Cum_Denominator"] = df_2026_base.groupby("Owner")["Total_Market_AE_Volume"].cumsum()
    df_2026_base["Baseline_YTD_Share"] = df_2026_base["Cum_Numerator"] / df_2026_base["Cum_Denominator"]

    last_actual_date = df_2026_base["Week Ending Date"].max()
    ytd_bank = {}
    for label in TARGET_LABELS:
        rows = df_2026_base[
            (df_2026_base["Owner"] == label) & (df_2026_base["Week Ending Date"] == last_actual_date)
        ]
        if rows.empty:
            logger.warning("YTD bank: no row for %s at %s; skipping", label, last_actual_date.date())
            continue
        latest_row = rows.iloc[0]
        ytd_bank[label] = {
            "Banked_Numerator": float(latest_row["Cum_Numerator"]),
            "Banked_Denominator": float(latest_row["Cum_Denominator"]),
            "Current_YTD_Share": float(latest_row["Baseline_YTD_Share"]),
        }

    ytd_projections = build_ytd_projections(df_2026_base, ytd_bank, final_forecast, e_score=e_score)

    weekly_enriched = enrich_weekly_for_spike(weekly_labels, alist_long, wk_minus)
    spike_engine, df_full, spike_features = train_spike_and_df_full(weekly_enriched, ytd_projections)

    # Market volume for future weeks (Prophet on total market) — reuse series from forecast_baseline_future
    market_history = weekly_labels[["Week Ending Date", "Total_Market_AE_Volume"]].drop_duplicates().sort_values(
        "Week Ending Date"
    )
    market_prophet_train = market_history.rename(columns={"Week Ending Date": "ds", "Total_Market_AE_Volume": "y"})
    market_model = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False)
    market_model.fit(market_prophet_train.dropna())
    last_date = pd.to_datetime(df_model["Week Ending Date"].max())
    remaining_weeks = pd.date_range(
        start=last_date + pd.Timedelta(days=7), end=pd.to_datetime(end_of_year), freq="W-THU"
    )
    full_fc = market_model.predict(pd.DataFrame({"ds": remaining_weeks}))
    df_full = attach_total_market_volume(df_full, weekly_labels, full_fc[["ds", "yhat"]])

    actuals_2026 = build_actuals_2026(weekly_labels, forecast_year=forecast_year)

    ae_parquet = data_dir / "streams_product_songs_ae_compressed.parquet"
    artist_dna_lookup: Dict[str, Dict[int, float]] = {}
    artist_w2_retention: Dict[str, Dict[str, float]] = {}
    artist_profile_dict: Dict[str, float] = {}
    cluster_product_coef: Dict[int, float] = {}
    cluster_product_meta: Dict[str, Any] = {}
    if csv_only:
        logger.info(
            "train_artifacts_main: csv_only=True — skipping AE parquet KMeans/DNA "
            "and archetype decay; reusing prior artifacts_75k sidecars when present"
        )
        (
            artist_dna_lookup,
            artist_w2_retention,
            artist_profile_dict,
            cluster_product_coef,
            cluster_product_meta,
        ) = _load_parquet_sidecars_from_artifacts_dir(art_dir)
        if not cluster_product_coef:
            cluster_product_coef = {c: float(GLOBAL_PRODUCT_COEF) for c in (0, 1, 2, 3)}
    elif ae_parquet.exists():
        try:
            artist_dna_lookup, kmeans_model, project_to_cluster, mrelg_to_cluster = (
                kmeans_and_dna_from_ae_parquet(ae_parquet)
            )
            joblib.dump(kmeans_model, art_dir / "kmeans_archetype_75k.pkl")
            artist_w2_retention = build_artist_w2_retention_from_ae_parquet(ae_parquet)
            artist_profile_dict = build_artist_profile_dict_from_ae_parquet(ae_parquet)
            cluster_product_coef, cluster_product_meta = fit_cluster_product_coefficients_from_ae_parquet(
                ae_parquet,
                project_to_cluster,
                mrelg_to_cluster,
                global_fallback=GLOBAL_PRODUCT_COEF,
            )
        except Exception as e:
            logger.warning("Parquet-derived enrichment failed (%s); using fallback globals.", e)
    if not cluster_product_coef:
        cluster_product_coef = {c: float(GLOBAL_PRODUCT_COEF) for c in (0, 1, 2, 3)}

    joblib.dump(production_lgbm, art_dir / "production_lgbm_75k.pkl")
    joblib.dump(production_prophet_models, art_dir / "production_prophet_models_75k.pkl")
    joblib.dump(spike_engine, art_dir / "production_spike_engine.pkl")

    df_full.to_parquet(art_dir / "df_full.parquet", index=False)
    actuals_2026.to_parquet(art_dir / "actuals_2026.parquet", index=False)

    with open(art_dir / "artist_profile_dict.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in artist_profile_dict.items()}, f, indent=2)
    with open(art_dir / "artist_dna_lookup.json", "w", encoding="utf-8") as f:
        json.dump(dna_lookup_to_jsonable(artist_dna_lookup), f, indent=2)
    with open(art_dir / "artist_w2_retention.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in artist_w2_retention.items()}, f, indent=2)
    with open(art_dir / "distributions.json", "w", encoding="utf-8") as f:
        json.dump(DISTRIBUTIONS, f, indent=2)
    with open(art_dir / "cluster_product_coef.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in sorted(cluster_product_coef.items())}, f, indent=2)
    if cluster_product_meta:
        with open(art_dir / "cluster_product_regression.json", "w", encoding="utf-8") as f:
            json.dump(cluster_product_meta, f, indent=2)

    meta = {
        "GLOBAL_PRODUCT_COEF": GLOBAL_PRODUCT_COEF,
        #"PRODUCT_M52_PENALTY_CAP": PRODUCT_M52_PENALTY_CAP,
        "W2_RETENTION_BLEND_K": W2_RETENTION_BLEND_K,
        "E_score": e_score,
        "e80_conformal": e80,
        "forecast_year": forecast_year,
        "target_labels": TARGET_LABELS,
        "spike_features": spike_features,
        "end_of_year": end_of_year,
    }
    with open(art_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wrote artifacts to %s", art_dir)

    # Archetype decay — memory-heavy; skipped on weekly csv_only runs.
    if csv_only:
        logger.info(
            "train_artifacts_main: csv_only=True — skipping AE-panel and worldwide_streams "
            "archetype decay (use /v1/data/refresh_model or train with csv_only=False to rebuild)"
        )
    else:
        # Archetype decay: two disjoint data sources (do not conflate).
        # (1) AE panel parquet — album-equivalent weekly metrics (streaming equivalents, product sales, song sales).
        # (2) worldwide_streams parquet — raw weekly counts only (separate file, separate artifact dir).
        ae_archetypes_parquet = data_dir / "streams_product_songs_ae_compressed.parquet"
        if ae_archetypes_parquet.exists():
            archetypes_base = base_path / "archetypes_artifacts"
            for metric, subdir in [
                ("streaming_equivalent", "streams"),
                ("product_sales", "sales"),
                ("song_sale_equivalent", "songs"),
            ]:
                out_dir = archetypes_base / subdir
                logger.info(
                    "Archetype decay (AE panel %s): metric=%s → %s",
                    ae_archetypes_parquet.name,
                    metric,
                    out_dir,
                )
                archetype_args = argparse.Namespace(
                    parquet_path=str(ae_archetypes_parquet),
                    out_dir=str(out_dir),
                    metric=metric,
                    horizon_weeks=78,
                    n_clusters=4,
                    random_state=42,
                    kmeans_batch_size=2048,
                    max_tracks_for_features=None,
                    sanity_artist=None,
                    sanity_peak_volume=None,
                    sanity_peak_week=None,
                    sanity_genre=None,
                    # AE panel metrics keep using the hardcoded global
                    # ARCHETYPE_SCENARIO_MULTIPLIERS table; only worldwide_streams
                    # learns its own empirical Bear/Base/Bull table.
                    compute_scenario_multipliers=False,
                )
                train_archetype_model(archetype_args)
            logger.info("Wrote archetype artifacts to %s", archetypes_base)
        else:
            logger.warning(
                "Skipping AE-panel archetype decay — parquet not found: %s", ae_archetypes_parquet
            )

        # worldwide_streams weekly counts (not streaming_equivalent): its own parquet and archetypes_artifacts/worldwide_streams/.
        # Default: model/data/worldwide_streams_compressed.parquet
        # Override: TIDE_WORLDWIDE_STREAMS_PARQUET=/path/to/file.parquet
        worldwide_env = os.environ.get("TIDE_WORLDWIDE_STREAMS_PARQUET", "").strip()
        worldwide_parquet = (
            Path(worldwide_env).expanduser().resolve()
            if worldwide_env
            else (data_dir / "worldwide_streams_compressed.parquet")
        )
        legacy_worldwide = data_dir / "streams_worldwide_compressed.parquet"
        if not worldwide_parquet.is_file() and legacy_worldwide.is_file():
            worldwide_parquet = legacy_worldwide
        if worldwide_parquet.exists():
            archetypes_base = base_path / "archetypes_artifacts"
            out_dir = archetypes_base / "worldwide_streams"
            logger.info(
                "Archetype decay (worldwide_streams %s): metric=worldwide_streams → %s",
                worldwide_parquet.name,
                out_dir,
            )
            archetype_args = argparse.Namespace(
                parquet_path=str(worldwide_parquet),
                out_dir=str(out_dir),
                metric="worldwide_streams",
                horizon_weeks=78,
                n_clusters=4,
                random_state=42,
                kmeans_batch_size=2048,
                max_tracks_for_features=None,
                sanity_artist=None,
                sanity_peak_volume=None,
                sanity_peak_week=None,
                sanity_genre=None,
                # worldwide_streams is the only metric that learns its own
                # Bear/Base/Bull table natively. AE panel metrics continue to
                # rely on the hardcoded ARCHETYPE_SCENARIO_MULTIPLIERS values.
                compute_scenario_multipliers=True,
            )
            train_archetype_model(archetype_args)
            logger.info("Wrote worldwide_streams archetype artifacts to %s", out_dir)
        else:
            logger.info(
                "Skipping worldwide_streams archetype training — parquet not found: %s "
                "(not streaming_equivalent; place worldwide_streams_compressed.parquet under %s or set TIDE_WORLDWIDE_STREAMS_PARQUET)",
                worldwide_parquet,
                data_dir,
            )


if __name__ == "__main__":
    main()

