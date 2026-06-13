"""
src/feature_engineering.py
==========================
Task 1 — Step 4: Feature Engineering  (Fraud_Data only)

New features created:

┌──────────────────────┬──────────────────────────────────────────────────────┐
│ Feature              │ Why it helps detect fraud                            │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ hour_of_day          │ Fraudsters often operate at unusual hours (late       │
│                      │ night / early morning) to avoid real-time oversight.  │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ day_of_week          │ Fraud patterns differ on weekdays vs weekends —       │
│                      │ fewer support staff means slower detection.           │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ time_since_signup    │ A very short gap between signup and purchase           │
│ (seconds)            │ (e.g. < 60 s) is a classic account-abuse signal.     │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ user_txn_count       │ Total transactions by this user in the dataset.       │
│                      │ High velocity is a red flag.                          │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ user_txn_velocity    │ Transactions per day since signup.  Normalises        │
│                      │ txn_count by account age to avoid penalising          │
│                      │ legitimate long-time users.                           │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ is_same_day          │ Binary: purchase happened on the same calendar day   │
│                      │ as signup — often indicates scripted bot behaviour.   │
└──────────────────────┴──────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal signals from purchase_time.

    Requires:
      - purchase_time  : datetime64 column.

    Adds:
      - hour_of_day  (int, 0-23)
      - day_of_week  (int, 0=Monday … 6=Sunday)
    """
    df = df.copy()

    # Hour the purchase was made
    df["hour_of_day"] = df["purchase_time"].dt.hour

    # Day of week (0 = Monday, 6 = Sunday)
    df["day_of_week"] = df["purchase_time"].dt.dayofweek

    print(f"[feature_engineering] hour_of_day range  : "
          f"[{df['hour_of_day'].min()} – {df['hour_of_day'].max()}]")
    print(f"[feature_engineering] day_of_week range  : "
          f"[{df['day_of_week'].min()} – {df['day_of_week'].max()}]")
    return df


def engineer_signup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive features based on the signup → purchase gap.

    Requires:
      - signup_time   : datetime64 column.
      - purchase_time : datetime64 column.

    Adds:
      - time_since_signup (float, seconds) — clipped at 0 to handle any
        data-entry errors where purchase precedes signup.
      - is_same_day (int, 0/1) — 1 if signup and purchase are on the same
        calendar date.
    """
    df = df.copy()

    # Seconds between account creation and purchase
    delta = (df["purchase_time"] - df["signup_time"]).dt.total_seconds()
    df["time_since_signup"] = delta.clip(lower=0)  # no negative durations

    # Same-day flag
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
    """
    Compute per-user transaction frequency and velocity.

    Requires:
      - user_id            : user identifier.
      - time_since_signup  : seconds since signup (from engineer_signup_features).

    Adds:
      - user_txn_count    (int)   — how many rows share this user_id.
      - user_txn_velocity (float) — txn_count / days_since_signup.
        For users with time_since_signup = 0 (same second as signup), velocity
        is set to user_txn_count directly (1 day assumed as denominator).
    """
    df = df.copy()

    # Count of transactions per user across the whole dataset
    df["user_txn_count"] = df.groupby("user_id")["user_id"].transform("count")

    # Transactions per day since signup
    days = df["time_since_signup"] / 86_400   # seconds → days
    days_safe = days.replace(0, 1)            # avoid division by zero
    df["user_txn_velocity"] = df["user_txn_count"] / days_safe

    print(f"[feature_engineering] user_txn_count     : "
          f"mean={df['user_txn_count'].mean():.2f}  "
          f"max={df['user_txn_count'].max()}")
    print(f"[feature_engineering] user_txn_velocity  : "
          f"mean={df['user_txn_velocity'].mean():.4f}  "
          f"max={df['user_txn_velocity'].max():.4f}")
    return df


def engineer_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.

    Order matters:
      engineer_signup_features must run before engineer_velocity_features
      because velocity needs time_since_signup.

    Parameters
    ----------
    df : cleaned + geo-enriched Fraud_Data DataFrame.

    Returns
    -------
    DataFrame with 6 new columns added.
    """
    original_cols = set(df.columns)
    df = engineer_time_features(df)
    df = engineer_signup_features(df)
    df = engineer_velocity_features(df)
    new_cols = set(df.columns) - original_cols
    print(f"\n[feature_engineering] New features added ({len(new_cols)}): "
          f"{sorted(new_cols)}")
    return df