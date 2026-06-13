import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data_transformation import (
    transform_fraud_data,
    transform_creditcard,
)


# ─────────────────────────────────────────────────────────────
# Fraud dataset sample
# ─────────────────────────────────────────────────────────────

def fraud_sample():
    return pd.DataFrame({
        "user_id": [1, 2],
        "device_id": ["d1", "d2"],
        "signup_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "purchase_time": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "ip_address": [123, 456],
        "purchase_value": [100, 200],

        "source": ["SEO", "Ads"],
        "browser": ["Chrome", "Firefox"],
        "sex": ["M", "F"],
        "country": ["A", "B"],

        "hour_of_day": [1, 2],
        "day_of_week": [0, 1],
        "time_since_signup": [3600, 7200],
        "is_same_day": [1, 0],

        "class": [0, 1],
    })


# ─────────────────────────────────────────────────────────────
# Credit card sample
# ─────────────────────────────────────────────────────────────

def credit_sample():
    return pd.DataFrame({
        "Time": [10, 20, 30],
        "Amount": [100, 200, 300],
        "V1": [0.1, 0.2, 0.3],
        "V2": [0.1, 0.2, 0.3],
        "Class": [0, 1, 0],
    })


# ─────────────────────────────────────────────────────────────
# 1. Fraud transformation shape test
# ─────────────────────────────────────────────────────────────

def test_fraud_transform_basic():
    df = fraud_sample()

    X, y, scaler = transform_fraud_data(df)

    assert "class" not in X.columns
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    assert scaler is not None


# ─────────────────────────────────────────────────────────────
# 2. Fraud one-hot encoding test
# ─────────────────────────────────────────────────────────────

def test_fraud_one_hot_encoding():
    df = fraud_sample()

    X, _, _ = transform_fraud_data(df)

    # check that categorical columns were encoded
    assert any(col.startswith("source_") for col in X.columns)
    assert any(col.startswith("browser_") for col in X.columns)
    assert any(col.startswith("sex_") for col in X.columns)
    assert any(col.startswith("country_") for col in X.columns)


# ─────────────────────────────────────────────────────────────
# 3. Fraud scaler consistency test
# ─────────────────────────────────────────────────────────────

def test_fraud_scaler_reuse():
    df = fraud_sample()

    X_train, y_train, scaler = transform_fraud_data(df, fit=True)

    X_test, y_test, _ = transform_fraud_data(df, scaler=scaler, fit=False)

    # scaling should produce same shape
    assert X_train.shape == X_test.shape

    # values should match (deterministic scaling)
    np.testing.assert_array_almost_equal(
        X_train.values,
        X_test.values
    )


# ─────────────────────────────────────────────────────────────
# 4. Credit card transformation test
# ─────────────────────────────────────────────────────────────

def test_creditcard_transform():
    df = credit_sample()

    X, y, scaler = transform_creditcard(df)

    # target removed
    assert "Class" not in X.columns

    # shape consistency
    assert len(X) == len(y)

    # scaler exists
    assert scaler is not None

    # ONLY Time and Amount should be scaled (not V1)
    assert np.isclose(X["Time"].mean(), 0, atol=1e-6)
    assert np.isclose(X["Amount"].mean(), 0, atol=1e-6)

    # PCA features should remain unchanged
    assert np.array_equal(X["V1"].values, df["V1"].values)


# ─────────────────────────────────────────────────────────────
# 5. Missing scaler error test
# ─────────────────────────────────────────────────────────────

def test_missing_scaler_error():
    df = credit_sample()

    with pytest.raises(ValueError):
        transform_creditcard(df, scaler=None, fit=False)