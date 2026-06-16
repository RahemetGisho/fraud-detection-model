import pytest
import numpy as np
import pandas as pd

# Assuming your cleaning functions reside in src.preprocessing
from src.preprocessing import clean_fraud_data, clean_creditcard

# FIXTURES (Mock Data Subsets)

@pytest.fixture
def dirty_fraud_data():
    """Generates a dirty dataframe tracking duplicates, missing items, and mismatched types."""
    return pd.DataFrame({
        "user_id": [1, 1, 2, 3, 4, 5],  
        "signup_time": ["2026-01-01 10:00:00"] * 6,
        "purchase_time": ["2026-01-01 10:30:00"] * 6,
        "ip_address": [123456, 123456, 789012, 345678, None, 901234],  
        "class": [0, 0, 1, None, 0, 1],                              
        "age": [30.0, 30.0, 40.0, 35.0, 25.0, None],                  
        "source": ["SEO", "SEO", "Ads", "Direct", "SEO", None],        
        "browser": ["Chrome", "Chrome", "Safari", "IE", "Chrome", "Firefox"],
        "sex": ["M", "M", "F", "M", "F", "F"]
    })

@pytest.fixture
def dirty_creditcard_data():
    """Generates a dirty credit card dataframe tracking duplicate rows and empty cells."""
    return pd.DataFrame({
        "Time": [0.0, 0.0, 1.0, 2.0],
        "V1": [-1.35, -1.35, 0.45, None],  
        "Amount": [50.0, 50.0, 99.0, 10.0],
        "Class": [0.0, 0.0, 1.0, 0.0]       
    })

# FRAUD DATA CLEANING UNIT TESTS

def test_clean_fraud_data_pipeline(dirty_fraud_data):
    """Validates the execution flow of the data engine against the full checklist."""
    df_clean = clean_fraud_data(dirty_fraud_data)

    # 1. Row Deletion Assertions
    assert len(df_clean) == 3

    # 2. Imputation Assertions (Check median behavior)
    assert df_clean.loc[2, "age"] == 35.0  
    assert df_clean["age"].isna().sum() == 0

    # 3. Categorical Missing Fill Assertion
    assert df_clean.loc[2, "source"] == "Unknown"

    # 4. Data Type Structural Conformity Assertions
    assert df_clean["ip_address"].dtype == np.int64
    assert df_clean["class"].dtype == int
    assert pd.api.types.is_datetime64_any_dtype(df_clean["signup_time"])
    assert pd.api.types.is_datetime64_any_dtype(df_clean["purchase_time"])

# CREDIT CARD CLEANING UNIT TESTS

def test_clean_creditcard_pipeline(dirty_creditcard_data):
    """Validates strict row dropping and type conversion for the credit card tracking dataset."""
    df_clean = clean_creditcard(dirty_creditcard_data)
    assert len(df_clean) == 2
    assert df_clean["Class"].dtype == int
    assert df_clean.loc[0, "Class"] == 0
    assert df_clean.loc[1, "Class"] == 1
    