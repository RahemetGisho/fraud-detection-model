"""
tests/test_preprocessing.py
===========================

Unit tests for src.preprocessing
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    clean_fraud_data,
    clean_creditcard
)


# =========================================================
# FRAUD DATA TESTS
# =========================================================

def test_clean_fraud_removes_duplicates():

    df = pd.DataFrame({
        "user_id": [1, 1],
        "signup_time": ["2024-01-01", "2024-01-01"],
        "purchase_time": ["2024-01-02", "2024-01-02"],
        "purchase_value": [100, 100],
        "device_id": ["d1", "d1"],
        "source": ["SEO", "SEO"],
        "browser": ["Chrome", "Chrome"],
        "sex": ["M", "M"],
        "age": [25, 25],
        "ip_address": [12345, 12345],
        "class": [0, 0]
    })

    cleaned = clean_fraud_data(df)

    assert len(cleaned) == 1


def test_clean_fraud_drops_null_target():

    df = pd.DataFrame({
        "user_id": [1, 2],
        "signup_time": ["2024-01-01", "2024-01-01"],
        "purchase_time": ["2024-01-02", "2024-01-02"],
        "purchase_value": [100, 200],
        "device_id": ["d1", "d2"],
        "source": ["SEO", "Ads"],
        "browser": ["Chrome", "Firefox"],
        "sex": ["M", "F"],
        "age": [25, 30],
        "ip_address": [12345, 67890],
        "class": [0, np.nan]
    })

    cleaned = clean_fraud_data(df)

    assert len(cleaned) == 1
    assert cleaned["class"].isna().sum() == 0


def test_clean_fraud_drops_null_ip():

    df = pd.DataFrame({
        "user_id": [1, 2],
        "signup_time": ["2024-01-01", "2024-01-01"],
        "purchase_time": ["2024-01-02", "2024-01-02"],
        "purchase_value": [100, 200],
        "device_id": ["d1", "d2"],
        "source": ["SEO", "Ads"],
        "browser": ["Chrome", "Firefox"],
        "sex": ["M", "F"],
        "age": [25, 30],
        "ip_address": [12345, np.nan],
        "class": [0, 1]
    })

    cleaned = clean_fraud_data(df)

    assert len(cleaned) == 1
    assert cleaned["ip_address"].isna().sum() == 0


def test_age_imputation():

    df = pd.DataFrame({
        "user_id": [1, 2, 3],
        "signup_time": ["2024-01-01"] * 3,
        "purchase_time": ["2024-01-02"] * 3,
        "purchase_value": [100, 200, 300],
        "device_id": ["d1", "d2", "d3"],
        "source": ["SEO", "Ads", "SEO"],
        "browser": ["Chrome", "Firefox", "Chrome"],
        "sex": ["M", "F", "M"],
        "age": [20, np.nan, 40],
        "ip_address": [1, 2, 3],
        "class": [0, 1, 0]
    })

    cleaned = clean_fraud_data(df)

    # This test will fail until preprocessing.py is fixed
    assert cleaned["age"].isna().sum() == 0


def test_fill_unknown_categoricals():

    df = pd.DataFrame({
        "user_id": [1],
        "signup_time": ["2024-01-01"],
        "purchase_time": ["2024-01-02"],
        "purchase_value": [100],
        "device_id": ["d1"],
        "source": [None],
        "browser": [None],
        "sex": [None],
        "age": [25],
        "ip_address": [12345],
        "class": [0]
    })

    cleaned = clean_fraud_data(df)

    assert cleaned.loc[0, "source"] == "Unknown"
    assert cleaned.loc[0, "browser"] == "Unknown"
    assert cleaned.loc[0, "sex"] == "Unknown"


def test_ip_address_converted_to_int64():

    df = pd.DataFrame({
        "user_id": [1],
        "signup_time": ["2024-01-01"],
        "purchase_time": ["2024-01-02"],
        "purchase_value": [100],
        "device_id": ["d1"],
        "source": ["SEO"],
        "browser": ["Chrome"],
        "sex": ["M"],
        "age": [25],
        "ip_address": [12345.0],
        "class": [0]
    })

    cleaned = clean_fraud_data(df)

    assert str(cleaned["ip_address"].dtype) == "int64"


def test_datetime_conversion():

    df = pd.DataFrame({
        "user_id": [1],
        "signup_time": ["2024-01-01"],
        "purchase_time": ["2024-01-02"],
        "purchase_value": [100],
        "device_id": ["d1"],
        "source": ["SEO"],
        "browser": ["Chrome"],
        "sex": ["M"],
        "age": [25],
        "ip_address": [12345],
        "class": [0]
    })

    cleaned = clean_fraud_data(df)

    assert pd.api.types.is_datetime64_any_dtype(
        cleaned["signup_time"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        cleaned["purchase_time"]
    )


def test_class_converted_to_int():

    df = pd.DataFrame({
        "user_id": [1],
        "signup_time": ["2024-01-01"],
        "purchase_time": ["2024-01-02"],
        "purchase_value": [100],
        "device_id": ["d1"],
        "source": ["SEO"],
        "browser": ["Chrome"],
        "sex": ["M"],
        "age": [25],
        "ip_address": [12345],
        "class": [0.0]
    })

    cleaned = clean_fraud_data(df)

    assert str(cleaned["class"].dtype).startswith("int")


# =========================================================
# CREDIT CARD TESTS
# =========================================================

def test_clean_creditcard_removes_duplicates():

    row = {
        "Time": 1,
        **{f"V{i}": 0.1 for i in range(1, 29)},
        "Amount": 100,
        "Class": 0
    }

    df = pd.DataFrame([row, row])

    cleaned = clean_creditcard(df)

    assert len(cleaned) == 1


def test_clean_creditcard_removes_null_rows():

    rows = [
        {
            "Time": 1,
            **{f"V{i}": 0.1 for i in range(1, 29)},
            "Amount": 100,
            "Class": 0
        },
        {
            "Time": np.nan,
            **{f"V{i}": 0.1 for i in range(1, 29)},
            "Amount": 100,
            "Class": 0
        }
    ]

    df = pd.DataFrame(rows)

    cleaned = clean_creditcard(df)

    assert len(cleaned) == 1


def test_creditcard_class_dtype():

    df = pd.DataFrame([{
        "Time": 1,
        **{f"V{i}": 0.1 for i in range(1, 29)},
        "Amount": 100,
        "Class": 1.0
    }])

    cleaned = clean_creditcard(df)

    assert str(cleaned["Class"].dtype).startswith("int")