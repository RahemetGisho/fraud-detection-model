import pytest
import numpy as np
import pandas as pd

from src.feature_engineering import (
    engineer_time_features,
    engineer_signup_features,
    engineer_velocity_features,
    engineer_behavior_features,
    engineer_all
)

@pytest.fixture
def base_mock_data():
    """Generates an initial structured mock dataframe for simple feature extraction."""
    data = {
        "user_id": ["USR_A", "USR_A", "USR_B"],
        "signup_time": [
            pd.Timestamp("2026-01-01 10:00:00"),
            pd.Timestamp("2026-01-01 10:00:00"),
            pd.Timestamp("2026-01-01 15:00:00")
        ],
        "purchase_time": [
            pd.Timestamp("2026-01-01 10:00:30"),  
            pd.Timestamp("2026-01-02 11:00:00"),  
            pd.Timestamp("2026-01-01 14:00:00")   
        ],
        "purchase_value": [50.0, 150.0, 100.0]
    }
    return pd.DataFrame(data)

# INDIVIDUAL COMPONENT UNIT TESTS
def test_engineer_time_features(base_mock_data):
    """Verifies hour of day extraction and day index positions."""
    df_out = engineer_time_features(base_mock_data)
    
    assert "hour_of_day" in df_out.columns
    assert "day_of_week" in df_out.columns
    
    assert df_out.loc[0, "hour_of_day"] == 10
    assert df_out.loc[1, "hour_of_day"] == 11
    assert df_out.loc[0, "day_of_week"] == 3  

def test_engineer_signup_features_clipping_and_flags(base_mock_data):
    """Verifies same day classification and time clip safety bounds."""
    df_out = engineer_signup_features(base_mock_data)
    
    assert "time_since_signup" in df_out.columns
    assert "is_same_day" in df_out.columns
    
    assert df_out.loc[0, "time_since_signup"] == 30.0
    assert df_out.loc[0, "is_same_day"] == 1
    assert df_out.loc[1, "is_same_day"] == 0
    
    assert df_out.loc[2, "time_since_signup"] == 0.0

def test_engineer_velocity_features_and_division_safety(base_mock_data):
    """Verifies frequency calculations do not fail on 0-second division windows."""
    df_input = engineer_signup_features(base_mock_data)
    df_out = engineer_velocity_features(df_input)
    
    assert "user_txn_count" in df_out.columns
    assert "user_txn_velocity" in df_out.columns
    
    assert df_out.loc[0, "user_txn_count"] == 2
    assert df_out.loc[2, "user_txn_count"] == 1
    
    assert not np.isinf(df_out.loc[2, "user_txn_velocity"])
    assert df_out.loc[2, "user_txn_velocity"] == 1.0  # (count 1 / days_safe 1)

def test_engineer_behavior_features_and_sequences(base_mock_data):
    """Verifies mean transformations, variance calculations, and sequential intervals."""
    df_input = engineer_signup_features(base_mock_data)
    df_input = engineer_velocity_features(df_input)
    
    df_out = engineer_behavior_features(df_input)
    
    expected_cols = ["account_age_days", "transactions_per_hour", "avg_purchase_value", "purchase_deviation", "time_since_prev_txn"]
    for col in expected_cols:
        assert col in df_out.columns

    user_a_rows = df_out[df_out["user_id"] == "USR_A"].sort_values("purchase_time")
    
    assert (user_a_rows["avg_purchase_value"] == 100.0).all()
    
    assert user_a_rows.iloc[0]["time_since_prev_txn"] == 999999.0
    
    expected_gap = (pd.Timestamp("2026-01-02 11:00:00") - pd.Timestamp("2026-01-01 10:00:30")).total_seconds()
    assert user_a_rows.iloc[1]["time_since_prev_txn"] == expected_gap

# ORCHESTRATOR PIPELINE TESTS
def test_engineer_all_execution_flow(base_mock_data):
    """Ensures every single expected new feature gets returned uniformly through the wrapper."""
    df_out = engineer_all(base_mock_data)
    
    final_expected_features = [
        "hour_of_day", "day_of_week", "time_since_signup", "is_same_day",
        "user_txn_count", "user_txn_velocity", "account_age_days",
        "transactions_per_hour", "avg_purchase_value", "purchase_deviation",
        "time_since_prev_txn"
    ]
    
    for feature in final_expected_features:
        assert feature in df_out.columns