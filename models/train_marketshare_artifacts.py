#!/usr/bin/env python3
"""
Weekly/monthly training job for 75k marketshare artifacts.

Combines logic from `baselinemarket_75k.ipynb` (LGBM + Prophet + pkls) and
`75k_parlay.ipynb` (market Prophet, baseline YTD, spike Ridge, df_full,
actuals_2026, K-Means / artist DNA, product OLS, exported static tables).

Outputs (under project root, default ``artifacts_75k/``):
  production_lgbm_75k.pkl, production_prophet_models_75k.pkl,
  production_spike_engine.pkl,
  df_full.parquet, actuals_2026.parquet,
  artist_profile_dict.json, artist_dna_lookup.json, artist_w2_retention.json,
  cluster_product_coef.json, cluster_product_regression.json (when full65 + product CSV),
  distributions.json,
  metadata.json (E_score, paths, forecast horizon, etc.)
"""

from __future__ import annotations

import json
import logging
import sqlite3
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
    ARCHETYPE_MULTIPLIERS,
    DISTRIBUTIONS,
    GLOBAL_PRODUCT_COEF,
    PRODUCT_M52_PENALTY_CAP,
    REPO_ROOT,
    W2_RETENTION_BLEND_K,
    dna_lookup_to_jsonable,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_LABELS = ["Atlantic Music Group", "Interscope/Geffen/A&M"]
TRAINING_CATEGORIES = TARGET_LABELS

# Matches queries/create_weekly_marketshare_table.sql (SQLite identifier).
WEEKLY_MARKETSHARE_TABLE = "MARKETSHARE_WEEKLY"

END_OF_YEAR = "2026-12-31"
FORECAST_YEAR = 2026
PRODUCT_REGRESSION_TARGET = "m52_multiplier"
PRODUCT_REGRESSION_MIN_N = 25
PRODUCT_RIDGE_ALPHA = 2.0


def load_weekly_amg_int(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        current = pd.read_sql_query(f"SELECT * FROM {WEEKLY_MARKETSHARE_TABLE}", conn)
    finally:
        conn.close()
    if "RELEASE_AGE" in current.columns:
        ra = current["RELEASE_AGE"].astype(str).str.strip()
        n_before = len(current)
        current = current.loc[ra.str.casefold() == "current"].copy()
        n_after = len(current)
        if n_before and n_after < n_before:
            logger.info(
                "%s: RELEASE_AGE=='Current' only (%d rows, dropped %d)",
                WEEKLY_MARKETSHARE_TABLE,
                n_after,
                n_before - n_after,
            )
        if n_after == 0:
            logger.warning(
                "%s: zero rows after RELEASE_AGE=='Current' filter; check values/casing",
                WEEKLY_MARKETSHARE_TABLE,
            )
    weekly_amg_int = current[
        current["LABEL_NAME"].isin(TARGET_LABELS)
    ].copy()
    weekly_amg_int = weekly_amg_int.rename(
        columns={
            "WEEK_ENDING_DATE": "Week Ending Date",
            "ALBUM_EQUIVALENT": "AE_Volume",
            "ALBUM_EQUIVALENT_SHARE": "AE_Share",
            "LABEL_NAME": "Owner",
        }
    )
    amg_truth = weekly_amg_int[weekly_amg_int["Owner"] == "Atlantic Music Group"].copy()
    amg_truth["True_Total_Market"] = np.where(
        amg_truth["AE_Share"] > 0,
        (amg_truth["AE_Volume"] / amg_truth["AE_Share"]) * 100,
        np.nan,
    )
    market_ref = amg_truth[["Week Ending Date", "True_Total_Market"]].drop_duplicates()
    weekly_amg_int["Week Ending Date"] = pd.to_datetime(weekly_amg_int["Week Ending Date"])
    market_ref["Week Ending Date"] = pd.to_datetime(market_ref["Week Ending Date"])
    if "Total_Market_AE_Volume" in weekly_amg_int.columns:
        weekly_amg_int = weekly_amg_int.drop(columns=["Total_Market_AE_Volume"])
    weekly_amg_int = weekly_amg_int.merge(market_ref, on="Week Ending Date", how="left")
    weekly_amg_int = weekly_amg_int.rename(columns={"True_Total_Market": "Total_Market_AE_Volume"})
    dup_n = weekly_amg_int.duplicated(subset=["Owner", "Week Ending Date"]).sum()
    if dup_n:
        logger.warning(
            "%s: dropping %d duplicate Owner×Week rows (keeps last); duplicates skew YTD cumulatives",
            WEEKLY_MARKETSHARE_TABLE,
            int(dup_n),
        )
        weekly_amg_int = weekly_amg_int.drop_duplicates(
            subset=["Owner", "Week Ending Date"], keep="last"
        ).reset_index(drop=True)
    return weekly_amg_int


def build_wk_minus(weekly_amg_int: pd.DataFrame, a_list_wk: pd.DataFrame) -> pd.DataFrame:
    weekly_amg_int = weekly_amg_int.copy()
    weekly_amg_int["Week Ending Date"] = pd.to_datetime(weekly_amg_int["Week Ending Date"])
    a_list_wk = a_list_wk.copy()
    a_list_wk["WEEK_END_DATE"] = pd.to_datetime(a_list_wk["WEEK_END_DATE"])
    wk_minus = pd.merge(
        weekly_amg_int,
        a_list_wk,
        left_on="Week Ending Date",
        right_on="WEEK_END_DATE",
        how="left",
    )
    wk_minus["Competitor_AE_Volume"] = wk_minus["Total_Market_AE_Volume"] - wk_minus["AE_Volume"]
    for c in ["AMG_ALBUMS", "INTERSCOPE_ALBUMS", "MARKET_ALBUMS"]:
        if c in wk_minus.columns:
            wk_minus[c] = wk_minus[c].fillna(0)
    wk_minus["AE_Volume"] = np.where(
        wk_minus["Owner"] == "Atlantic Music Group",
        wk_minus["AE_Volume"] - wk_minus["AMG_ALBUMS"].fillna(0),
        np.where(
            wk_minus["Owner"] == "Interscope/Geffen/A&M",
            wk_minus["AE_Volume"] - wk_minus["INTERSCOPE_ALBUMS"].fillna(0),
            wk_minus["AE_Volume"],
        ),
    )
    wk_minus["AE_Share"] = np.where(
        wk_minus["Total_Market_AE_Volume"] > 0,
        (wk_minus["AE_Volume"] / wk_minus["Total_Market_AE_Volume"]) * 100,
        0.0
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
    df_model["Owner"] = df_model["Owner"].astype("category")
    return df_model


def train_lgbm_prophet(df_model: pd.DataFrame) -> Tuple[lgb.LGBMRegressor, Dict[str, Any]]:
    features = [
        "Owner",
        "Lag1_AE_Volume",
        "Lag1_Competitor_Volume",
        "Roll4W_AE_Volume",
        "Roll4W_Competitor_Volume",
    ]
    X_full = df_model[features]
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
    for label in df_model["Owner"].unique():
        label_df = df_model[df_model["Owner"] == label].copy()
        
        prophet_train = label_df[["Week Ending Date", "AE_Share"]].rename(
            columns={"Week Ending Date": "ds", "AE_Share": "y"} # BACK TO SHARE
        )
        prophet_model = Prophet(yearly_seasonality=2, weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(prophet_train.dropna())
        production_prophet_models[label] = prophet_model
        
    return production_lgbm, production_prophet_models

def conformal_e90_2026(
    df_model: pd.DataFrame,
    production_lgbm: lgb.LGBMRegressor,
    production_prophet_models: Dict[str, Any],
) -> float:
    backtest_raw = df_model[df_model["Week Ending Date"] >= "2025-11-01"].copy()
    backtest_raw = backtest_raw[backtest_raw["Owner"].isin(TARGET_LABELS)].sort_values(
        ["Owner", "Week Ending Date"]
    )
    
    ensemble_errors: List[pd.Series] = []
    
    for label in TARGET_LABELS:
        df_label = backtest_raw[backtest_raw["Owner"] == label].copy()
                
        df_label = df_label[df_label["Week Ending Date"] >= "2026-01-08"].dropna()
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
    return float(np.percentile(all_errors, 90))


def forecast_baseline_future(
    df_model: pd.DataFrame,
    production_lgbm: lgb.LGBMRegressor,
    production_prophet_models: Dict[str, Any],
    end_of_year: str,
    week_freq: str = "W-THU",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (future_label_df, future_market_volumes)."""
    last_date = pd.to_datetime(df_model["Week Ending Date"].max())
    end_dt = pd.to_datetime(end_of_year)
    remaining_weeks = pd.date_range(start=last_date + pd.Timedelta(days=7), end=end_dt, freq=week_freq)
    future_label_df = pd.DataFrame()
    for label in TARGET_LABELS:
        label_history = df_model[df_model["Owner"] == label].sort_values("Week Ending Date")
        last_4 = label_history.tail(4)
        current_lag1_ae = last_4.iloc[-1]["AE_Volume"]
        current_lag1_comp = last_4.iloc[-1]["Competitor_AE_Volume"]
        current_roll4w_ae = last_4["AE_Volume"].mean()
        current_roll4w_comp = last_4["Competitor_AE_Volume"].mean()
        df_temp = pd.DataFrame({"Week Ending Date": remaining_weeks, "Owner": label})
        df_temp["Owner"] = df_temp["Owner"].astype("category")
        df_temp["Lag1_AE_Volume"] = current_lag1_ae
        df_temp["Lag1_Competitor_Volume"] = current_lag1_comp
        df_temp["Roll4W_AE_Volume"] = current_roll4w_ae
        df_temp["Roll4W_Competitor_Volume"] = current_roll4w_comp
        lgbm_features = df_temp[
            ["Owner", "Lag1_AE_Volume", "Lag1_Competitor_Volume", "Roll4W_AE_Volume", "Roll4W_Competitor_Volume"]
        ]
        df_temp["LGBM_Share"] = production_lgbm.predict(lgbm_features)
        prophet_future = df_temp[["Week Ending Date"]].rename(columns={"Week Ending Date": "ds"})
        df_temp["Prophet_Share"] = production_prophet_models[label].predict(prophet_future)["yhat"].values
        df_temp["Predicted_Weekly_Share"] = (df_temp["LGBM_Share"] + df_temp["Prophet_Share"]) / 2
        future_label_df = pd.concat([future_label_df, df_temp], ignore_index=True)

    wk_for_market = df_model  # use same df for distinct market timeline
    market_history = wk_for_market[["Week Ending Date", "Total_Market_AE_Volume"]].drop_duplicates()
    market_history = market_history.sort_values("Week Ending Date").reset_index(drop=True)
    market_prophet_train = market_history.rename(columns={"Week Ending Date": "ds", "Total_Market_AE_Volume": "y"})
    market_model = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False)
    market_model.fit(market_prophet_train.dropna())
    last_actual_date = market_history["Week Ending Date"].max()
    future_market = pd.DataFrame({"ds": remaining_weeks})
    market_forecast = market_model.predict(future_market)
    future_market_volumes = market_forecast[["ds", "yhat"]].rename(
        columns={"ds": "Week Ending Date", "yhat": "Predicted_Total_Market_Volume"}
    )
    return future_label_df, future_market_volumes


def build_ytd_projections(
    df_2026_base: pd.DataFrame,
    ytd_bank: Dict[str, Dict[str, float]],
    final_forecast: pd.DataFrame,
    e_score: float,
) -> pd.DataFrame:
    ytd_projections = pd.DataFrame()
    for label in TARGET_LABELS:
        lf = final_forecast[final_forecast["Owner"] == label].sort_values("Week Ending Date").copy()
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
    weekly_amg_int: pd.DataFrame,
    a_list_wk: pd.DataFrame,
    big_release_flag: pd.DataFrame,
    wk_minus: pd.DataFrame,
) -> pd.DataFrame:
    w = weekly_amg_int.copy()
    w["Week Ending Date"] = pd.to_datetime(w["Week Ending Date"])
    a_list_wk = a_list_wk.copy()
    a_list_wk["WEEK_END_DATE"] = pd.to_datetime(a_list_wk["WEEK_END_DATE"])
    w = w.merge(
        a_list_wk[["WEEK_END_DATE", "AMG_ALBUMS", "INTERSCOPE_ALBUMS"]],
        left_on="Week Ending Date",
        right_on="WEEK_END_DATE",
        how="left",
    )
    w["Incremental_Volume"] = 0.0
    amg_mask = w["Owner"] == "Atlantic Music Group"
    w.loc[amg_mask, "Incremental_Volume"] = w.loc[amg_mask, "AMG_ALBUMS"].fillna(0)
    int_mask = w["Owner"] == "Interscope/Geffen/A&M"
    w.loc[int_mask, "Incremental_Volume"] = w.loc[int_mask, "INTERSCOPE_ALBUMS"].fillna(0)
    w["Incremental_Share"] = np.where(
        w["Total_Market_AE_Volume"] > 0,
        (w["Incremental_Volume"] / w["Total_Market_AE_Volume"]) * 100,
        0.0,
    )
    big_release_flag = big_release_flag.copy()
    big_release_flag["WEEK_END_DATE"] = pd.to_datetime(big_release_flag["WEEK_END_DATE"])
    w = w.merge(
        big_release_flag[["WEEK_END_DATE", "BIG_RELEASE_ATLANTIC", "BIG_RELEASE_INTERSCOPE"]],
        left_on="Week Ending Date",
        right_on="WEEK_END_DATE",
        how="left",
    )
    w["big_release_flag"] = 0.0
    w.loc[amg_mask, "big_release_flag"] = w.loc[amg_mask, "BIG_RELEASE_ATLANTIC"].fillna(0)
    w.loc[int_mask, "big_release_flag"] = w.loc[int_mask, "BIG_RELEASE_INTERSCOPE"].fillna(0)
    scrubbed_vols = wk_minus[["Week Ending Date", "Owner", "AE_Volume"]].rename(columns={"AE_Volume": "Baseline_Volume"})
    w = w.merge(scrubbed_vols, on=["Week Ending Date", "Owner"], how="left")
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


def build_actuals_2026(weekly_amg_int: pd.DataFrame) -> pd.DataFrame:
    actuals_2026 = weekly_amg_int[
        (weekly_amg_int["Week Ending Date"] >= "2026-01-08") & (weekly_amg_int["Owner"].isin(TARGET_LABELS))
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
    s = g.loc[g["WEEKS_SINCE_RELEASE"] == w, "WEEKLY_EQUIVALENT_QUANTITY"]
    return float(s.iloc[0]) if not s.empty else 0.0


def _release_w2_over_w1_ratio(
    g: pd.DataFrame,
    min_w1_share_of_early_peak: float = 0.12,
) -> Optional[float]:
    """
    Week-2 / first-week AE ratio for one release, robust to partial week 0 or partial week 1.

    - If week 2 > week 1, week 1 is often a partial week: use max(week0..week3) as the
      denominator and cap at 1.0 so we never treat a low week-1 bucket as "weak retention"
      and inflate the artist median.
    - Otherwise (normal week-2 drop): denominator is max(week0, week1) so abbreviated week 0
      does not inflate v2/v1.
    - When week 2 is not higher than week 1, drops releases where week 1 is an implausibly
      small share of the early peak (likely mis-tagged weeks).
    """
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

    # Handle week-2 > week-1 first (partial week 1) before the "weak week 1" exclusion
    if v2 > v1 * 1.005:
        denom = max(v0, v1, v2, v3)
        if denom <= 0:
            return None
        return float(min(v2 / denom, 1.0))

    # Standard: first-week volume = max(week0, week1) to handle abbreviated week 0
    if v1 > 0 and v1 < peak_early * min_w1_share_of_early_peak:
        return None

    equiv_w1 = max(v0, v1) if max(v0, v1) > 0 else v1
    if equiv_w1 <= 0:
        return None

    return float(min(v2 / equiv_w1, 1.0))


def build_artist_w2_retention(full65_path: Path) -> Dict[str, Dict[str, float]]:
    """
    Per DISPLAY_ARTIST: median W2 / first-week AE from full65+.

    Weeks are WEEKS_SINCE_RELEASE indices (0–3 used for robustness). The first-week volume
    is max(0,1) when both exist so abbreviated week 0 does not inflate retention; suspicious
    rows are dropped (see _release_w2_over_w1_ratio).
    """
    df = pd.read_csv(
        full65_path,
        usecols=["DISPLAY_ARTIST", "MRELG_ID", "WEEKS_SINCE_RELEASE", "WEEKLY_EQUIVALENT_QUANTITY"],
    )
    df["WEEKLY_EQUIVALENT_QUANTITY"] = pd.to_numeric(df["WEEKLY_EQUIVALENT_QUANTITY"], errors="coerce")
    df["WEEKS_SINCE_RELEASE"] = pd.to_numeric(df["WEEKS_SINCE_RELEASE"], errors="coerce")
    df = df.dropna(subset=["WEEKS_SINCE_RELEASE", "WEEKLY_EQUIVALENT_QUANTITY"])
    ratios: List[Dict[str, Any]] = []
    for (artist, _mid), g in df.groupby(["DISPLAY_ARTIST", "MRELG_ID"], sort=False):
        r = _release_w2_over_w1_ratio(g)
        if r is None:
            continue
        ratios.append({"DISPLAY_ARTIST": str(artist), "ratio": r})
    if not ratios:
        logger.warning("build_artist_w2_retention: no valid W2/W1 ratios from %s", full65_path)
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
    logger.info("Built artist_w2_retention for %d artists from full65+", len(out))
    return out


def kmeans_and_dna(
    full65_path: Path,
) -> Tuple[Dict[str, Dict[int, float]], Any, Dict[str, int], Dict[str, int]]:
    df_history = pd.read_csv(full65_path)
    df_history["project_label"] = (
        df_history["DISPLAY_ARTIST"].astype(str).fillna("")
        + " - "
        + df_history["TITLE"].astype(str).fillna("")
    )
    project_lifespans = df_history.groupby("project_label")["WEEKS_SINCE_RELEASE"].max()
    mature_projects = project_lifespans[project_lifespans >= 52].index
    df_mature = df_history[df_history["project_label"].isin(mature_projects)].copy()
    df_52w = df_mature[df_mature["WEEKS_SINCE_RELEASE"] <= 52].copy()
    shape_df = df_52w.pivot_table(
        index="project_label",
        columns="WEEKS_SINCE_RELEASE",
        values="WEEKLY_EQUIVALENT_QUANTITY",
        aggfunc="sum",
    ).fillna(0)
    album_peaks = shape_df.max(axis=1)
    normalized_shapes = shape_df.div(album_peaks, axis=0)
    normalized_shapes = normalized_shapes.replace([np.inf, -np.inf], np.nan).dropna()
    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_shapes.to_numpy())
    normalized_shapes = normalized_shapes.copy()
    normalized_shapes["archetype_cluster"] = cluster_labels
    historical_clusters = normalized_shapes[["archetype_cluster"]].reset_index()
    features_with_clusters = df_history.merge(historical_clusters, on="project_label", how="left")

    artist_cluster_probs = (
        features_with_clusters.groupby(["DISPLAY_ARTIST", "archetype_cluster"], sort=False)
        .agg(count=("MRELG_ID", "nunique"))
        .reset_index()
    )
    artist_cluster_probs["prob"] = artist_cluster_probs.groupby("DISPLAY_ARTIST")["count"].transform(
        lambda x: x / x.sum()
    )
    fitted = set(map(int, ARCHETYPE_MULTIPLIERS.keys()))
    artist_cluster_probs = artist_cluster_probs[artist_cluster_probs["archetype_cluster"].isin(fitted)]

    artist_dna_lookup = artist_cluster_probs.groupby("DISPLAY_ARTIST").apply(
        lambda x: dict(zip(x["archetype_cluster"], x["prob"]))
    ).to_dict()
    project_to_cluster = normalized_shapes["archetype_cluster"].astype(int).to_dict()
    mrelg_rows = features_with_clusters.dropna(subset=["archetype_cluster"]).drop_duplicates(
        subset=["MRELG_ID"], keep="first"
    )
    mrelg_to_cluster = {
        str(r["MRELG_ID"]): int(r["archetype_cluster"]) for _, r in mrelg_rows.iterrows()
    }
    return artist_dna_lookup, kmeans, project_to_cluster, mrelg_to_cluster


def _product_release_pivot(product_csv: Path) -> pd.DataFrame:
    """Wide table per MRELG_ID with WEEKLY_ALBUM_EQUIVALENTS_w{k}, PRODUCT_SALES_w{k}."""
    df = pd.read_csv(product_csv)
    df["project_label"] = df["DISPLAY_ARTIST"].astype(str).fillna("") + " - " + df["TITLE"].astype(str).fillna("")
    df_pivot = df.pivot_table(
        index=["MRELG_ID", "project_label", "TITLE", "DISPLAY_ARTIST", "FIRST_SALE_DATE", "GENRES"],
        columns="WEEKS_SINCE_RELEASE",
        values=["WEEKLY_ALBUM_EQUIVALENTS", "PRODUCT_SALES"],
    ).reset_index()
    df_pivot.columns = [
        f"{col[0]}_w{int(col[1])}" if col[1] else col[0] for col in df_pivot.columns.values
    ]
    return df_pivot


def _row_m52_multiplier_from_pivot(row: pd.Series) -> float:
    """Sum of first 52 weekly album-equivalent weeks (w0..w51) / week-1 AE."""
    total = 0.0
    for k in range(0, 52):
        col = f"WEEKLY_ALBUM_EQUIVALENTS_w{k}"
        if col in row.index and pd.notna(row[col]):
            total += float(row[col])
    w1_col = "WEEKLY_ALBUM_EQUIVALENTS_w1"
    if w1_col not in row.index or pd.isna(row[w1_col]) or float(row[w1_col]) <= 0:
        return float("nan")
    return total / float(row[w1_col])


def fit_cluster_product_coefficients(
    product_csv: Path,
    project_to_cluster: Dict[str, int],
    mrelg_to_cluster: Dict[str, int],
    global_fallback: float = GLOBAL_PRODUCT_COEF,
    min_cluster_n: int = 25,
    ridge_alpha: float = 2.0,
    target: str = "m52_multiplier",
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """
    Per-cluster Ridge: target ~ intercept + w1_product_ratio.
    target='m52_multiplier': observed 52-week AE sum / W1 AE (aligned with archetype M52 scale).
    target='w2_retention': W2 AE / W1 AE (different units; coef still used as M52 penalty scale).
    """
    df_pivot = _product_release_pivot(product_csv)
    df_pivot["w1_product_ratio"] = df_pivot["PRODUCT_SALES_w1"] / df_pivot["WEEKLY_ALBUM_EQUIVALENTS_w1"]
    df_pivot["w1_product_ratio"] = df_pivot["w1_product_ratio"].replace([np.inf, -np.inf], np.nan)
    if target == "m52_multiplier":
        df_pivot["y_target"] = df_pivot.apply(_row_m52_multiplier_from_pivot, axis=1)
    elif target == "w2_retention":
        df_pivot["y_target"] = df_pivot["WEEKLY_ALBUM_EQUIVALENTS_w2"] / df_pivot["WEEKLY_ALBUM_EQUIVALENTS_w1"]
        df_pivot["y_target"] = df_pivot["y_target"].replace([np.inf, -np.inf], np.nan)
    else:
        raise ValueError(f"Unknown product regression target: {target}")

    mid = df_pivot["MRELG_ID"].astype(str)
    df_pivot["archetype_cluster"] = mid.map(mrelg_to_cluster)
    miss = df_pivot["archetype_cluster"].isna()
    if miss.any():
        df_pivot.loc[miss, "archetype_cluster"] = df_pivot.loc[miss, "project_label"].map(project_to_cluster)
    sub = df_pivot.dropna(subset=["y_target", "w1_product_ratio", "archetype_cluster"]).copy()
    sub["w1_product_ratio"] = sub["w1_product_ratio"].clip(lower=0.0, upper=1.0)
    sub["archetype_cluster"] = sub["archetype_cluster"].astype(int)

    coefs: Dict[int, float] = {}
    meta_per_cluster: Dict[str, Any] = {}
    for c in sorted(sub["archetype_cluster"].unique()):
        if c not in (0, 1, 2, 3):
            continue
        chunk = sub[sub["archetype_cluster"] == c]
        n = len(chunk)
        if n < min_cluster_n:
            coefs[int(c)] = float(global_fallback)
            meta_per_cluster[str(c)] = {"n": n, "slope": None, "fallback": True}
            logger.warning(
                "Cluster %s: only %d releases for product regression — using GLOBAL_PRODUCT_COEF",
                c,
                n,
            )
            continue
        X = chunk[["w1_product_ratio"]].to_numpy(dtype=float)
        y = chunk["y_target"].to_numpy(dtype=float)
        model = Ridge(alpha=ridge_alpha)
        model.fit(X, y)
        slope = float(model.coef_[0])
        coefs[int(c)] = slope
        meta_per_cluster[str(c)] = {
            "n": n,
            "slope": slope,
            "intercept": float(model.intercept_),
            "fallback": False,
        }

    for c in (0, 1, 2, 3):
        coefs.setdefault(c, float(global_fallback))

    summary = {
        "target": target,
        "ridge_alpha": float(ridge_alpha),
        "min_cluster_n": int(min_cluster_n),
        "global_fallback": global_fallback,
        "per_cluster": meta_per_cluster,
        "n_rows_used": int(len(sub)),
    }
    logger.info("Cluster product coefficients: %s", coefs)
    return coefs, summary


def product_artist_profiles(product_csv: Path) -> Dict[str, float]:
    df_pivot = _product_release_pivot(product_csv)
    df_pivot["w1_product_ratio"] = df_pivot["PRODUCT_SALES_w1"] / df_pivot["WEEKLY_ALBUM_EQUIVALENTS_w1"]
    df_pivot["w2_retention_rate"] = df_pivot["WEEKLY_ALBUM_EQUIVALENTS_w2"] / df_pivot["WEEKLY_ALBUM_EQUIVALENTS_w1"]
    df_pivot.fillna({"w1_product_ratio": 0, "w2_retention_rate": 0}, inplace=True)
    valid_profiles = df_pivot.dropna(subset=["w1_product_ratio"])
    artist_profiles = valid_profiles.groupby("DISPLAY_ARTIST").agg(
        avg_historical_w1_product_ratio=("w1_product_ratio", "mean"),
        recorded_albums=("MRELG_ID", "count"),
    ).reset_index()
    profile_dict = pd.Series(
        artist_profiles["avg_historical_w1_product_ratio"].values,
        index=artist_profiles["DISPLAY_ARTIST"],
    ).to_dict()
    logger.info("Built artist_profile_dict with %d artists", len(profile_dict))
    return profile_dict


def main() -> None:
    root = REPO_ROOT
    data_dir = root / "data"
    art_dir = (root / "artifacts_75k").resolve()
    sqlite_db = root / "marketshare_data.db"
    art_dir.mkdir(parents=True, exist_ok=True)

    a_list_wk = pd.read_csv(data_dir / "alist_75k.csv")
    big_release_flag = pd.read_csv(data_dir / "bigreleaseflag_75k.csv")
    weekly_amg_int = load_weekly_amg_int(sqlite_db.resolve())
    wk_minus = build_wk_minus(weekly_amg_int, a_list_wk)

    df_model = prepare_df_model(wk_minus)
    production_lgbm, production_prophet_models = train_lgbm_prophet(df_model)
    e90 = conformal_e90_2026(df_model, production_lgbm, production_prophet_models)
    e_score = e90
    logger.info("Conformal 90th percentile E: %.4f (used as E_score for YTD bands)", e_score)

    future_label_df, future_market_volumes = forecast_baseline_future(
        df_model, production_lgbm, production_prophet_models, END_OF_YEAR
    )
    final_forecast = pd.merge(future_label_df, future_market_volumes, on="Week Ending Date", how="inner")

    df_2026_base = wk_minus[wk_minus["Week Ending Date"] >= "2026-01-08"].copy()
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
        latest_row = df_2026_base[
            (df_2026_base["Owner"] == label) & (df_2026_base["Week Ending Date"] == last_actual_date)
        ].iloc[0]
        ytd_bank[label] = {
            "Banked_Numerator": float(latest_row["Cum_Numerator"]),
            "Banked_Denominator": float(latest_row["Cum_Denominator"]),
            "Current_YTD_Share": float(latest_row["Baseline_YTD_Share"]),
        }

    ytd_projections = build_ytd_projections(df_2026_base, ytd_bank, final_forecast, e_score=e_score)

    weekly_enriched = enrich_weekly_for_spike(weekly_amg_int, a_list_wk, big_release_flag, wk_minus)
    spike_engine, df_full, spike_features = train_spike_and_df_full(weekly_enriched, ytd_projections)

    # Market volume for future weeks (Prophet on total market) — reuse series from forecast_baseline_future
    market_history = df_model[["Week Ending Date", "Total_Market_AE_Volume"]].drop_duplicates().sort_values(
        "Week Ending Date"
    )
    market_prophet_train = market_history.rename(columns={"Week Ending Date": "ds", "Total_Market_AE_Volume": "y"})
    market_model = Prophet(yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False)
    market_model.fit(market_prophet_train.dropna())
    last_date = pd.to_datetime(df_model["Week Ending Date"].max())
    remaining_weeks = pd.date_range(
        start=last_date + pd.Timedelta(days=7), end=pd.to_datetime(END_OF_YEAR), freq="W-THU"
    )
    full_fc = market_model.predict(pd.DataFrame({"ds": remaining_weeks}))
    df_full = attach_total_market_volume(df_full, weekly_amg_int, full_fc[["ds", "yhat"]])

    actuals_2026 = build_actuals_2026(weekly_amg_int)

    full65 = (data_dir / "full65+.csv").resolve()
    artist_w2_retention: Dict[str, Dict[str, float]] = {}
    project_to_cluster: Dict[str, int] = {}
    mrelg_to_cluster: Dict[str, int] = {}
    if full65.exists():
        artist_dna_lookup, kmeans_model, project_to_cluster, mrelg_to_cluster = kmeans_and_dna(full65)
        joblib.dump(kmeans_model, art_dir / "kmeans_archetype_75k.pkl")
        artist_w2_retention = build_artist_w2_retention(full65)
    else:
        logger.warning("Missing %s — skipping K-Means / DNA (empty artist_dna_lookup)", full65)
        artist_dna_lookup = {}

    product_csv = (data_dir / "product25k+_release_date.csv").resolve()
    cluster_product_coef: Dict[int, float] = {}
    cluster_product_meta: Dict[str, Any] = {}
    if product_csv.exists():
        artist_profile_dict = product_artist_profiles(product_csv)
        if project_to_cluster or mrelg_to_cluster:
            cluster_product_coef, cluster_product_meta = fit_cluster_product_coefficients(
                product_csv,
                project_to_cluster,
                mrelg_to_cluster,
                global_fallback=GLOBAL_PRODUCT_COEF,
                target=PRODUCT_REGRESSION_TARGET,
                min_cluster_n=PRODUCT_REGRESSION_MIN_N,
                ridge_alpha=PRODUCT_RIDGE_ALPHA,
            )
        else:
            logger.warning("No project_to_cluster map — skipping cluster product regression")
    else:
        logger.warning("Missing %s — artist_profile_dict empty", product_csv)
        artist_profile_dict = {}

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
    if not cluster_product_coef:
        cluster_product_coef = {c: float(GLOBAL_PRODUCT_COEF) for c in (0, 1, 2, 3)}
    with open(art_dir / "cluster_product_coef.json", "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in sorted(cluster_product_coef.items())}, f, indent=2)
    if cluster_product_meta:
        with open(art_dir / "cluster_product_regression.json", "w", encoding="utf-8") as f:
            json.dump(cluster_product_meta, f, indent=2)

    meta = {
        "GLOBAL_PRODUCT_COEF": GLOBAL_PRODUCT_COEF,
        "PRODUCT_M52_PENALTY_CAP": PRODUCT_M52_PENALTY_CAP,
        "product_regression_target": PRODUCT_REGRESSION_TARGET,
        "W2_RETENTION_BLEND_K": W2_RETENTION_BLEND_K,
        "E_score": e_score,
        "e90_conformal": e90,
        "forecast_year": FORECAST_YEAR,
        "target_labels": TARGET_LABELS,
        "spike_features": spike_features,
        "end_of_year": END_OF_YEAR,
    }
    with open(art_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Wrote artifacts to %s", art_dir)


if __name__ == "__main__":
    main()
