import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_transformation import transform_fraud_data, transform_creditcard, _FRAUD_NUM_SCALE_COLS

@pytest.fixture
def sample_fraud_data():
    data = {
        "class": [0, 1],
        "user_id": ["USR1", "USR2"],
        "device_id": ["DEV1", "DEV2"],
        "signup_time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
        "purchase_time": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03")],
        "ip_address": ["192.168.1.1", "192.168.1.2"],
        "source": ["SEO", "Ads"],
        "browser": ["Chrome", "Safari"],
        "sex": ["M", "F"],
        "country": ["United States", "Canada"],
        # Continuous scale columns
        "purchase_value": [100.0, 200.0],
        "age": [30, 40],
        "hour_of_day": [12, 14],
        "day_of_week": [4, 5],
        "time_since_signup": [3600.0, 7200.0],
        "user_txn_count": [1, 2],
        "user_txn_velocity": [0.5, 1.0],
        "account_age_days": [1.0, 2.0],
        "transactions_per_hour": [0.1, 0.2],
        "avg_purchase_value": [100.0, 150.0],
        "purchase_deviation": [1.0, 1.3],
        "time_since_prev_txn": [999999.0, 500.0]
    }
    return pd.DataFrame(data)

# FRAUD DATASET TESTS

def test_fraud_transform_train_guarantees(sample_fraud_data):
    """Verifies targets are isolated, IDs dropped, and arrays are transformed correctly in train mode."""
    df_out, y_out, scaler, train_cols = transform_fraud_data(sample_fraud_data, fit=True)

    assert "class" not in df_out.columns
    pd.testing.assert_series_equal(y_out, sample_fraud_data["class"].astype(int))

    for dropped_col in ["user_id", "device_id", "signup_time", "purchase_time", "ip_address"]:
        assert dropped_col not in df_out.columns

    assert isinstance(df_out["purchase_value"].dtype, (object, np.floating))
    assert isinstance(scaler, StandardScaler)
    
    assert df_out["source_SEO"].isin([0, 1]).all()
    assert df_out["source_SEO"].dtype == np.int64

def test_fraud_transform_test_alignment_and_oof_protection(sample_fraud_data):
    """Verifies that an unseen categorical class in the test data does not throw off schema boundaries."""
    df_train, _, scaler, train_cols = transform_fraud_data(sample_fraud_data, fit=True)

    test_data = sample_fraud_data.copy()
    test_data.loc[0, "country"] = "UnknownLand"

    df_test, y_test, _, _ = transform_fraud_data(
        test_data, scaler=scaler, fit=False, train_columns=train_cols
    )

    assert df_train.columns.tolist() == df_test.columns.tolist()
    assert "country_UnknownLand" not in df_test.columns

def test_fraud_transform_missing_parameters_raises_error(sample_fraud_data):
    """Verifies that passing fit=False without tracking dependencies raises an explicit runtime ValueError."""
    with pytest.raises(ValueError, match="An instantiated training scaler must be provided"):
        transform_fraud_data(sample_fraud_data, scaler=None, fit=False)

    scaler = StandardScaler()
    scaler.fit(sample_fraud_data[_FRAUD_NUM_SCALE_COLS])  
    
    with pytest.raises(ValueError, match="train_columns must be provided"):
        transform_fraud_data(sample_fraud_data, scaler=scaler, fit=False, train_columns=None)

# CREDIT CARD DATASET TESTS
def test_creditcard_transform_flow():
    """Validates simple structural separation and target isolation behavior for credit card datasets."""
    cc_mock = pd.DataFrame({
        "Class": [0, 1],
        "Time": [0.0, 3600.0],
        "Amount": [50.0, 2500.0],
        "V1": [-1.34, 0.45],
        "V2": [0.25, -0.89]
    })

    df_train, y_train, scaler = transform_creditcard(cc_mock, fit=True)

    assert "Class" not in df_train.columns
    assert "Time" in df_train.columns
    assert isinstance(scaler, StandardScaler)
    
    df_test, y_test, _ = transform_creditcard(cc_mock, scaler=scaler, fit=False)
    assert df_train.shape == df_test.shape