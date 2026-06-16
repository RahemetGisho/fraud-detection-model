"""
src/feature_engineering.py
==========================
Task 1 — Step 4: Feature Engineering  (Fraud_Data only)

Final Feature Set Summary:
---------------------------
The Targets:
 - class: Keep (Target variable Y)

The Categorical Features (Require Encoding in Step 5):
 - source, browser, country, sex

The Behavioral & Temporal Features:
┌───────────────────────┬──────────────────────────────────────────────────────┐
│ Feature               │ Why it helps detect fraud                            │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ hour_of_day           │ Fraudsters often operate at unusual hours to avoid   │
│                       │ real-time oversight.                                 │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ day_of_week           │ Fraud patterns differ on weekdays vs weekends due to │
│                       │ changes in security support coverage.                │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ time_since_signup     │ A very short gap between signup and purchase         │
│ (seconds)             │ (e.g. < 60s) is a classic bot/abuse signal.          │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ is_same_day           │ Binary flag for calendar-day match; catches rapid    │
│                       │ scripted account-to-checkout flows.                  │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ user_txn_count        │ Total transaction frequency footprint across the     │
│                       │ available window. High density implies abuse.        │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ user_txn_velocity     │ Transactions per day since signup. Normalizes raw    │
│                       │ counts to avoid penalizing loyal users.              │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ account_age_days      │ Continuous asset metric tracked in day units.        │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ transactions_per_hour │ Fine-grained velocity snapshot metrics.              │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ avg_purchase_value    │ Tracks historical spending patterns for user profiling│
├───────────────────────┼──────────────────────────────────────────────────────┤
│ purchase_deviation    │ Identifies anomalous price spikes relative to their  │
│                       │ normal behavior.                                     │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ time_since_prev_txn   │ Sequences transaction pacing. Sudden bursts flag bots│
└───────────────────────┴──────────────────────────────────────────────────────┘

CRITICAL IMPLEMENTATION NOTE:
Because this script uses `.transform("count")` and `.transform("mean")` aggregated 
by `user_id`, you MUST execute `engineer_all(df)` SEPARATELY on your Train and 
Test partitions after splitting. Running this on the raw full dataset combined 
will introduce massive Lookahead Data Leakage.
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
    """
    df = df.copy()

    # Count of transactions per user across the dataset
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


def engineer_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional behavioral fraud features.

    Adds:
      - account_age_days
      - transactions_per_hour
      - avg_purchase_value
      - purchase_deviation
      - time_since_prev_txn
    """
    df = df.copy()

    # Account age in days
    df["account_age_days"] = df["time_since_signup"] / 86_400

    # Transactions per hour since signup
    hours = df["time_since_signup"] / 3600
    hours_safe = hours.replace(0, 1)

    df["transactions_per_hour"] = (
        df["user_txn_count"] / hours_safe
    )

    # User average purchase amount
    df["avg_purchase_value"] = (
        df.groupby("user_id")["purchase_value"]
        .transform("mean")
    )

    # Current purchase relative to user's average
    df["purchase_deviation"] = (
        df["purchase_value"]
        / (df["avg_purchase_value"] + 1)
    )

    # Time since previous transaction
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