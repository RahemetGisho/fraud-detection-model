"""
tests/test_data_loader.py
=========================

Unit tests for src.data_loader
"""

import pandas as pd
import pytest

from src.data_loader import (
    _validate_columns,
    _validate_min_rows,
    load_fraud_data,
    load_ip_country,
    load_creditcard,
    FRAUD_REQUIRED_COLS,
    IP_REQUIRED_COLS,
    CC_REQUIRED_COLS,
)


# =========================================================
# INTERNAL VALIDATION HELPERS
# =========================================================

def test_validate_columns_passes():

    df = pd.DataFrame(columns=FRAUD_REQUIRED_COLS)

    _validate_columns(
        df,
        FRAUD_REQUIRED_COLS,
        "Fraud_Data"
    )


def test_validate_columns_raises():

    df = pd.DataFrame(
        columns=["user_id", "signup_time"]
    )

    with pytest.raises(ValueError):
        _validate_columns(
            df,
            FRAUD_REQUIRED_COLS,
            "Fraud_Data"
        )


def test_validate_min_rows_passes():

    df = pd.DataFrame({"a": range(1000)})

    _validate_min_rows(
        df,
        1000,
        "Test"
    )


def test_validate_min_rows_raises():

    df = pd.DataFrame({"a": range(10)})

    with pytest.raises(ValueError):
        _validate_min_rows(
            df,
            1000,
            "Test"
        )


# =========================================================
# FRAUD DATA LOADER
# =========================================================

def test_load_fraud_data_success(tmp_path):

    file_path = tmp_path / "Fraud_Data.csv"

    rows = 1000

    fraud_df = pd.DataFrame({
        "user_id": range(rows),
        "signup_time": ["2024-01-01"] * rows,
        "purchase_time": ["2024-01-02"] * rows,
        "purchase_value": [100] * rows,
        "device_id": ["dev"] * rows,
        "source": ["SEO"] * rows,
        "browser": ["Chrome"] * rows,
        "sex": ["M"] * rows,
        "age": [30] * rows,
        "ip_address": [123456] * rows,
        "class": [0] * rows,
    })

    fraud_df.to_csv(file_path, index=False)

    loaded = load_fraud_data(str(file_path))

    assert loaded.shape[0] == rows

    assert pd.api.types.is_datetime64_any_dtype(
        loaded["signup_time"]
    )

    assert pd.api.types.is_datetime64_any_dtype(
        loaded["purchase_time"]
    )


def test_load_fraud_data_missing_file():

    with pytest.raises(FileNotFoundError):
        load_fraud_data(
            "does_not_exist.csv"
        )


def test_load_fraud_data_missing_column(tmp_path):

    file_path = tmp_path / "Fraud_Data.csv"

    rows = 1000

    df = pd.DataFrame({
        "user_id": range(rows)
    })

    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_fraud_data(str(file_path))


# =========================================================
# IP COUNTRY LOADER
# =========================================================

def test_load_ip_country_success(tmp_path):

    file_path = tmp_path / "IpAddress_to_Country.csv"

    ip_df = pd.DataFrame({
        "lower_bound_ip_address": [1000],
        "upper_bound_ip_address": [2000],
        "country": ["Ethiopia"]
    })

    ip_df.to_csv(file_path, index=False)

    loaded = load_ip_country(str(file_path))

    assert list(loaded.columns) == IP_REQUIRED_COLS

    assert str(
        loaded["lower_bound_ip_address"].dtype
    ) == "int64"

    assert str(
        loaded["upper_bound_ip_address"].dtype
    ) == "int64"


def test_load_ip_country_missing_file():

    with pytest.raises(FileNotFoundError):
        load_ip_country(
            "missing.csv"
        )


# =========================================================
# CREDIT CARD LOADER
# =========================================================

def test_load_creditcard_success(tmp_path):

    file_path = tmp_path / "creditcard.csv"

    rows = 1000

    data = {
        "Time": [1] * rows
    }

    for i in range(1, 29):
        data[f"V{i}"] = [0.1] * rows

    data["Amount"] = [100.0] * rows
    data["Class"] = [0] * rows

    cc_df = pd.DataFrame(data)

    cc_df.to_csv(file_path, index=False)

    loaded = load_creditcard(
        str(file_path)
    )

    assert loaded.shape[0] == rows

    assert all(
        col in loaded.columns
        for col in CC_REQUIRED_COLS
    )


def test_load_creditcard_missing_file():

    with pytest.raises(FileNotFoundError):
        load_creditcard(
            "missing.csv"
        )


def test_load_creditcard_too_few_rows(tmp_path):

    file_path = tmp_path / "creditcard.csv"

    data = {"Time": [1]}

    for i in range(1, 29):
        data[f"V{i}"] = [0.1]

    data["Amount"] = [10]
    data["Class"] = [0]

    pd.DataFrame(data).to_csv(
        file_path,
        index=False
    )

    with pytest.raises(ValueError):
        load_creditcard(
            str(file_path)
        )