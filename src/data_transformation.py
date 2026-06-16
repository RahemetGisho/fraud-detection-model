"""
src/data_transformation.py
===========================
Task 1 — Step 5: Data Transformation (Encoding & Scaling)

Guarantees:
✔ Correct feature selection: Raw identifiers are cleanly omitted.
✔ No accidental column leaks: Explicit alignment prevents shape mismatches.
✔ Correct categorical encoding: One-Hot Encodes text fields safely.
✔ Safe scaling boundaries: Scales continuous features ONLY (protects OHE features).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────
# CONFIG (Maintains explicit data schema boundaries)
# ─────────────────────────────────────────────────────────────

_FRAUD_DROP_COLS = [
    "user_id",
    "device_id",
    "signup_time",
    "purchase_time",
    "ip_address",
]

_FRAUD_CAT_COLS = ["source", "browser", "sex", "country"]

# Continuous features requiring scaling (Excludes OHE binary flags)
_FRAUD_NUM_SCALE_COLS = [
    "purchase_value", 
    "age", 
    "hour_of_day", 
    "day_of_week", 
    "time_since_signup", 
    "user_txn_count", 
    "user_txn_velocity", 
    "account_age_days", 
    "transactions_per_hour", 
    "avg_purchase_value", 
    "purchase_deviation", 
    "time_since_prev_txn"
]


# ─────────────────────────────────────────────────────────────
# CORE TRANSFORM (FRAUD DATASET)
# ─────────────────────────────────────────────────────────────

def transform_fraud_data(df: pd.DataFrame,
                         scaler: StandardScaler = None,
                         fit: bool = True,
                         train_columns: list = None):
    """
    Transforms data schema safely:
    - Train (fit=True): Fits scaler, performs OHE, captures base feature layout.
    - Test (fit=False): Reuses training scaler and forces column alignment.
    """
    df = df.copy()

    # ── 1. Target Extraction ──────────────────────────────────
    y = df["class"].astype(int)
    df.drop(columns=["class"], inplace=True)

    # ── 2. Structural Pruning (Feature Selection) ──────────────
    df.drop(columns=[c for c in _FRAUD_DROP_COLS if c in df.columns],
            inplace=True,
            errors="ignore")

    # ── 3. Scale Continuous Features Only ─────────────────────
    # Must happen before OHE or target column alignment to isolate true numerical ranges
    if fit:
        scaler = StandardScaler()
        df[_FRAUD_NUM_SCALE_COLS] = scaler.fit_transform(df[_FRAUD_NUM_SCALE_COLS])
    else:
        if scaler is None:
            raise ValueError("An instantiated training scaler must be provided for test transform.")
        df[_FRAUD_NUM_SCALE_COLS] = scaler.transform(df[_FRAUD_NUM_SCALE_COLS])

    # ── 4. Categorical Encoding (OHE) ─────────────────────────
    df = pd.get_dummies(df, columns=_FRAUD_CAT_COLS, drop_first=False)

    # Coerce categorical booleans to numeric flag spaces (0/1)
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # ── 5. Schema Alignment (Prevents Out-of-Vocabulary Leaks) ─
    if fit:
        train_columns = df.columns.tolist()
    else:
        if train_columns is None:
            raise ValueError("train_columns must be provided for test transform alignment.")
        # Drops unseen test categories, adds missing train categories as 0
        df = df.reindex(columns=train_columns, fill_value=0)

    mode = "TRAIN" if fit else "TEST"
    print(f"[data_transformation] Fraud {mode} shape: {df.shape}")

    return df, y, scaler, train_columns


# ─────────────────────────────────────────────────────────────
# CREDIT CARD TRANSFORM
# ─────────────────────────────────────────────────────────────

def transform_creditcard(df: pd.DataFrame,
                         scaler: StandardScaler = None,
                         fit: bool = True):
    """
    Applies dedicated continuous scaling to credit card dataset features.
    """
    df = df.copy()

    y = df["Class"].astype(int)
    df.drop(columns=["Class"], inplace=True)

    scale_cols = ["Time", "Amount"]

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        if scaler is None:
            raise ValueError("An instantiated training scaler must be provided for test transform.")
        df[scale_cols] = scaler.transform(df[scale_cols])

    mode = "TRAIN" if fit else "TEST"
    print(f"[data_transformation] CreditCard {mode} X shape : {df.shape}")

    return df, y, scaler