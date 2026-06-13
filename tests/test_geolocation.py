import os
import numpy as np
import pandas as pd

from src.geolocation import (
    merge_ip_to_country,
    fraud_rate_by_country,
)


# ─────────────────────────────────────────────────────────────
# Fixtures (small fake datasets)
# ─────────────────────────────────────────────────────────────

def sample_fraud_df():
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "ip_address": [10, 20, 30, 9999],
        "class": [0, 1, 0, 1],
    })


def sample_ip_df():
    return pd.DataFrame({
        "lower_bound_ip_address": [0, 15, 25],
        "upper_bound_ip_address": [14, 24, 100],
        "country": ["A", "B", "C"],
    })


# ─────────────────────────────────────────────────────────────
# 1. Test IP conversion safety
# ─────────────────────────────────────────────────────────────

def test_ip_is_int64():
    fraud = sample_fraud_df()
    ip = sample_ip_df()

    merged = merge_ip_to_country(fraud, ip)

    assert merged["ip_address"].dtype == np.int64


# ─────────────────────────────────────────────────────────────
# 2. Test correct country mapping
# ─────────────────────────────────────────────────────────────

def test_country_mapping_correct():
    fraud = sample_fraud_df()
    ip = sample_ip_df()

    merged = merge_ip_to_country(fraud, ip)

    # expected mapping:
    # 10 → A
    # 20 → B
    # 30 → C
    assert merged.loc[0, "country"] == "A"
    assert merged.loc[1, "country"] == "B"
    assert merged.loc[2, "country"] == "C"


# ─────────────────────────────────────────────────────────────
# 3. Test unknown IP handling
# ─────────────────────────────────────────────────────────────

def test_unknown_ip_handling():
    fraud = sample_fraud_df()
    ip = sample_ip_df()

    merged = merge_ip_to_country(fraud, ip)

    assert "Unknown" in merged["country"].values


# ─────────────────────────────────────────────────────────────
# 4. Test fraud rate calculation logic
# ─────────────────────────────────────────────────────────────

def test_fraud_rate_calculation():
    df = pd.DataFrame({
        "country": ["A", "A", "B", "B", "B"],
        "class": [1, 0, 1, 1, 0]
    })

    result = fraud_rate_by_country(df, min_transactions=1)

    a_rate = result[result["country"] == "A"]["fraud_rate"].values[0]
    b_rate = result[result["country"] == "B"]["fraud_rate"].values[0]

    # A: 1/2 = 0.5
    # B: 2/3 ≈ 0.666
    assert round(a_rate, 2) == 0.50
    assert round(b_rate, 2) == 0.67


# ─────────────────────────────────────────────────────────────
# 5. Test output columns exist
# ─────────────────────────────────────────────────────────────

def test_fraud_rate_output_columns():
    df = pd.DataFrame({
        "country": ["A", "A", "B"],
        "class": [0, 1, 1]
    })

    result = fraud_rate_by_country(df, min_transactions=1)

    assert "fraud_rate" in result.columns
    assert "total" in result.columns
    assert "fraud" in result.columns