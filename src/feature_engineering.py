# src/feature_engineering.py
import numpy as np
import pandas as pd

def calculate_country_risk(df: pd.DataFrame, target_series: pd.Series = None) -> pd.DataFrame:
    """
    Computes or applies target encoding for country risk.
    """
    df = df.copy()
    
    if target_series is not None:
        temp_df = pd.DataFrame({"country": df["country"], "class": target_series})
        global_mean = temp_df["class"].mean()
        
        counts = temp_df.groupby("country")["class"].count()
        means = temp_df.groupby("country")["class"].mean()
        smooth_weight = 10
        
        smooth_risk = (counts * means + smooth_weight * global_mean) / (counts + smooth_weight)
        df["country_fraud_risk"] = df["country"].map(smooth_risk).fillna(global_mean)
    else:
        df["country_fraud_risk"] = 0.01 
        
    return df

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek
    return df

def engineer_signup_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["time_since_signup"] = delta.clip(lower=0)  
    same_day = (df["signup_time"].dt.date == df["purchase_time"].dt.date).astype(int)
    df["is_same_day"] = same_day
    return df

def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 1. User-specific structural velocity tracking
    df["user_txn_count"] = df.groupby("user_id")["user_id"].transform("count")
    days = df["time_since_signup"] / 86_400  
    days_safe = days.replace(0, 1)            
    df["user_txn_velocity"] = df["user_txn_count"] / days_safe

    # 2. Platform-wide multi-account 30-minute velocity tracking window
    df["_orig_order_idx"] = np.arange(len(df))
    v_df = df[["_orig_order_idx", "browser", "country", "purchase_time"]].sort_values("purchase_time")
    
    merged = pd.merge_asof(
        v_df,
        v_df,
        on="purchase_time",
        by=["browser", "country"],
        tolerance=pd.Timedelta("30min"),
        allow_exact_matches=True
    )
    
    velocity_counts = (
        merged.groupby("_orig_order_idx_x")
        .size()
        .reindex(df["_orig_order_idx"], fill_value=1)
    )
    
    df["platform_30min_velocity"] = velocity_counts.values
    df.drop(columns=["_orig_order_idx"], inplace=True)
    
    print(f"[feature_engineering] platform_30min_velocity: mean={df['platform_30min_velocity'].mean():.2f} max={df['platform_30min_velocity'].max()}")
    return df

def engineer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["account_age_days"] = df["time_since_signup"] / 86_400
    hours = df["time_since_signup"] / 3600
    hours_safe = hours.replace(0, 1)

    df["transactions_per_hour"] = df["user_txn_count"] / hours_safe
    df["avg_purchase_value"] = df.groupby("user_id")["purchase_value"].transform("mean")
    
    # Enhanced Purchase Deviation by Browser-Country Cohort Groupings
    cohort_avg = df.groupby(["browser", "country"])["purchase_value"].transform("median")
    df["cohort_purchase_deviation"] = df["purchase_value"] / (cohort_avg + 1)
    
    df["purchase_deviation"] = df["purchase_value"] / (df["avg_purchase_value"] + 1)

    df = df.sort_values(["user_id", "purchase_time"])
    df["time_since_prev_txn"] = df.groupby("user_id")["purchase_time"].diff().dt.total_seconds()
    df["time_since_prev_txn"] = df["time_since_prev_txn"].fillna(999999)

    return df

def engineer_all(df: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
    """
    Apply all feature engineering transformations cleanly.
    """
    original_cols = set(df.columns)

    df = engineer_time_features(df)
    df = engineer_signup_features(df)
    df = engineer_velocity_features(df)
    df = engineer_behavior_features(df)
    
    # Inject Target Risk Matrix Maps
    target_series = y_train if y_train is not None else (df["class"] if "class" in df.columns else None)
    df = calculate_country_risk(df, target_series=target_series)

    new_cols = set(df.columns) - original_cols
    print(f"[feature_engineering] Completed! Features added ({len(new_cols)}): {sorted(new_cols)}")
    return df