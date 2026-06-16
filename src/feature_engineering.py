"""
Feature Engineering  (Fraud_Data only)
"""

import numpy as np
import pandas as pd

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df["hour_of_day"] = df["purchase_time"].dt.hour
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    print(f"[feature_engineering] hour_of_day range  : "
          f"[{df['hour_of_day'].min()} – {df['hour_of_day'].max()}]")
    print(f"[feature_engineering] day_of_week range  : "
          f"[{df['day_of_week'].min()} – {df['day_of_week'].max()}]")
    return df

def engineer_signup_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    delta = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["time_since_signup"] = delta.clip(lower=0)  
    same_day = (
        df["signup_time"].dt.date == df["purchase_time"].dt.date
    ).astype(int)
    df["is_same_day"] = same_day

    pct_same_day = same_day.mean() * 100
    print(f"[feature_engineering] time_since_signup  : "
          f"min={df['time_since_signup'].min():.0f}s  "
          f"median={df['time_since_signup'].median():.0f}s  "
          f"max={df['time_since_signup'].max():.0f}s")
    print(f"[feature_engineering] is_same_day        : "
          f"{same_day.sum():,} rows ({pct_same_day:.1f}%)")
    return df

def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    df["user_txn_count"] = df.groupby("user_id")["user_id"].transform("count")
    days = df["time_since_signup"] / 86_400  
    days_safe = days.replace(0, 1)            
    df["user_txn_velocity"] = df["user_txn_count"] / days_safe

    print(f"[feature_engineering] user_txn_count     : "
          f"mean={df['user_txn_count'].mean():.2f}  "
          f"max={df['user_txn_count'].max()}")
    print(f"[feature_engineering] user_txn_velocity  : "
          f"mean={df['user_txn_velocity'].mean():.4f}  "
          f"max={df['user_txn_velocity'].max():.4f}")
    return df

def engineer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df["account_age_days"] = df["time_since_signup"] / 86_400
    hours = df["time_since_signup"] / 3600
    hours_safe = hours.replace(0, 1)

    df["transactions_per_hour"] = (
        df["user_txn_count"] / hours_safe
    )
    df["avg_purchase_value"] = (
        df.groupby("user_id")["purchase_value"]
        .transform("mean")
    )
    df["purchase_deviation"] = (
        df["purchase_value"]
        / (df["avg_purchase_value"] + 1)
    )

    df = df.sort_values(["user_id", "purchase_time"])

    df["time_since_prev_txn"] = (
        df.groupby("user_id")["purchase_time"]
        .diff()
        .dt.total_seconds()
    )

    df["time_since_prev_txn"] = (
        df["time_since_prev_txn"]
        .fillna(999999)
    )

    print(f"[feature_engineering] account_age_days      : mean={df['account_age_days'].mean():.2f}")
    print(f"[feature_engineering] transactions_per_hour : mean={df['transactions_per_hour'].mean():.4f}")
    print(f"[feature_engineering] avg_purchase_value    : mean={df['avg_purchase_value'].mean():.2f}")
    print(f"[feature_engineering] purchase_deviation    : mean={df['purchase_deviation'].mean():.4f}")

    return df

def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps systematically.
    """
    original_cols = set(df.columns)

    df = engineer_time_features(df)
    df = engineer_signup_features(df)
    df = engineer_velocity_features(df)
    df = engineer_behavior_features(df)

    new_cols = set(df.columns) - original_cols

    print(
        f"\n[feature_engineering] New features added "
        f"({len(new_cols)}): {sorted(new_cols)}"
    )

    return df