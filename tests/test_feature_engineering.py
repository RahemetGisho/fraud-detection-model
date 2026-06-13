import numpy as np
import pandas as pd

from src.feature_engineering import (
    engineer_time_features,
    engineer_signup_features,
    engineer_velocity_features,
    engineer_all,
)


# ─────────────────────────────────────────────────────────────
# Base test dataset
# ─────────────────────────────────────────────────────────────

def sample_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2],
        "signup_time": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:00",
            "2024-01-02 12:00:00",
        ]),
        "purchase_time": pd.to_datetime([
            "2024-01-01 01:00:00",  # hour 1
            "2024-01-01 01:30:00",  # hour 1
            "2024-01-03 12:00:00",  # hour 12
        ]),
    })


# ─────────────────────────────────────────────────────────────
# 1. Time features
# ─────────────────────────────────────────────────────────────

def test_engineer_time_features():
    df = sample_df()

    out = engineer_time_features(df)

    assert "hour_of_day" in out.columns
    assert "day_of_week" in out.columns

    # check correctness
    assert out.loc[0, "hour_of_day"] == 1
    assert out.loc[2, "hour_of_day"] == 12

    assert 0 <= out["hour_of_day"].min() <= 23
    assert 0 <= out["day_of_week"].max() <= 6


# ─────────────────────────────────────────────────────────────
# 2. Signup features
# ─────────────────────────────────────────────────────────────

def test_engineer_signup_features():
    df = sample_df()

    out = engineer_signup_features(df)

    assert "time_since_signup" in out.columns
    assert "is_same_day" in out.columns

    # time should be non-negative
    assert (out["time_since_signup"] >= 0).all()

    # same-day check
    assert out.loc[0, "is_same_day"] == 1
    assert out.loc[2, "is_same_day"] == 0


# ─────────────────────────────────────────────────────────────
# 3. Velocity features
# ─────────────────────────────────────────────────────────────

def test_engineer_velocity_features():
    df = sample_df()

    df = engineer_signup_features(df)
    out = engineer_velocity_features(df)

    assert "user_txn_count" in out.columns
    assert "user_txn_velocity" in out.columns

    # user 1 appears twice
    assert out[out["user_id"] == 1]["user_txn_count"].iloc[0] == 2

    # velocity should be positive
    assert (out["user_txn_velocity"] > 0).all()


# ─────────────────────────────────────────────────────────────
# 4. Full pipeline test
# ─────────────────────────────────────────────────────────────

def test_engineer_all_pipeline():
    df = sample_df()

    out = engineer_all(df)

    expected_cols = [
        "hour_of_day",
        "day_of_week",
        "time_since_signup",
        "is_same_day",
        "user_txn_count",
        "user_txn_velocity",
    ]

    for col in expected_cols:
        assert col in out.columns


# ─────────────────────────────────────────────────────────────
# 5. Edge case: zero time handling
# ─────────────────────────────────────────────────────────────

def test_velocity_no_divide_by_zero():
    df = pd.DataFrame({
        "user_id": [1],
        "signup_time": pd.to_datetime(["2024-01-01"]),
        "purchase_time": pd.to_datetime(["2024-01-01"]),
    })

    df = engineer_signup_features(df)
    out = engineer_velocity_features(df)

    assert np.isfinite(out["user_txn_velocity"].iloc[0])