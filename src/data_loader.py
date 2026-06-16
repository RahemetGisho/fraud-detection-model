"""
data_loader.py
Responsible for reading the three raw CSV files from disk and performing
basic structural validation (expected columns, minimum row count).
"""

import os
import pandas as pd

FRAUD_REQUIRED_COLS = [
    "user_id", "signup_time", "purchase_time", "purchase_value",
    "device_id", "source", "browser", "sex", "age", "ip_address", "class",
]

IP_REQUIRED_COLS = [
    "lower_bound_ip_address", "upper_bound_ip_address", "country",
]

CC_REQUIRED_COLS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)

def _validate_columns(df: pd.DataFrame, required: list, name: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

def _validate_min_rows(df: pd.DataFrame, min_rows: int, name: str) -> None:
    """Raise ValueError if the DataFrame has fewer rows than expected."""
    if len(df) < min_rows:
        raise ValueError(
            f"[{name}] Expected at least {min_rows} rows, got {len(df)}."
        )

def load_fraud_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fraud data file not found: {path}")

    df = pd.read_csv(path, parse_dates=["signup_time", "purchase_time"])

    _validate_columns(df, FRAUD_REQUIRED_COLS, "Fraud_Data")
    _validate_min_rows(df, 1000, "Fraud_Data")

    print(f"[load_fraud_data]  Loaded {len(df):,} rows, {df.shape[1]} cols "
          f"from '{path}'")
    return df

def load_ip_country(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"IP-country file not found: {path}")

    df = pd.read_csv(path)

    _validate_columns(df, IP_REQUIRED_COLS, "IpAddress_to_Country")

    # Force int64 on the bound columns — crucial for merge_asof dtype matching
    df["lower_bound_ip_address"] = df["lower_bound_ip_address"].astype("int64")
    df["upper_bound_ip_address"] = df["upper_bound_ip_address"].astype("int64")

    print(f"[load_ip_country]  Loaded {len(df):,} IP ranges "
          f"covering {df['country'].nunique()} countries from '{path}'")
    return df


def load_creditcard(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Credit-card file not found: {path}")

    df = pd.read_csv(path)

    _validate_columns(df, CC_REQUIRED_COLS, "CreditCard")
    _validate_min_rows(df, 1000, "CreditCard")

    print(f"[load_creditcard]  Loaded {len(df):,} rows, {df.shape[1]} cols "
          f"from '{path}'")
    return df