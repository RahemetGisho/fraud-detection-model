"""
data_transformation.py (Encoding & Scaling)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# CONFIG (Maintains explicit data schema boundaries)

_FRAUD_DROP_COLS = [
    "user_id",
    "device_id",
    "signup_time",
    "purchase_time",
    "ip_address",
    "country",          # DROPPED: Avoids 200+ raw high-cardinality dummy columns
]

# Categorical columns to one-hot encode (country removed safely)
_FRAUD_CAT_COLS = ["source", "browser", "sex"]

# Continuous features requiring scaling (Updated with our 3 engineered metrics)
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
    "time_since_prev_txn",
    "country_fraud_risk",         # INJECTED: Scaled target-encoded risk weights
    "platform_30min_velocity",    # INJECTED: Scaled multi-account velocity window
    "cohort_purchase_deviation"   # INJECTED: Scaled behavioral cohort deviation score
]

# CORE TRANSFORM (FRAUD DATASET)

def transform_fraud_data(df: pd.DataFrame,
                         scaler: StandardScaler = None,
                         fit: bool = True,
                         train_columns: list = None):
    df = df.copy()

    # 1. Target Extraction 
    y = df["class"].astype(int)
    df.drop(columns=["class"], inplace=True)

    # 2. Structural Pruning (Feature Selection) 
    df.drop(columns=[c for c in _FRAUD_DROP_COLS if c in df.columns],
            inplace=True,
            errors="ignore")

    # 3. Scale Continuous Features Only (Including newly engineered columns)
    # Ensure any required column missing from a truncated inference payload fails safely or fills gracefully
    active_scale_cols = [c for c in _FRAUD_NUM_SCALE_COLS if c in df.columns]
    
    if fit:
        scaler = StandardScaler()
        df[active_scale_cols] = scaler.fit_transform(df[active_scale_cols])
    else:
        if scaler is None:
            raise ValueError("An instantiated training scaler must be provided for test transform.")
        df[active_scale_cols] = scaler.transform(df[active_scale_cols])

    # 4. Categorical Encoding (OHE) 
    active_cat_cols = [c for c in _FRAUD_CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=active_cat_cols, drop_first=False)
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # 5. Schema Alignment (Prevents Out-of-Vocabulary Leaks) 
    if fit:
        train_columns = df.columns.tolist()
    else:
        if train_columns is None:
            raise ValueError("train_columns must be provided for test transform alignment.")
        df = df.reindex(columns=train_columns, fill_value=0)

    mode = "TRAIN" if fit else "TEST"
    print(f"[data_transformation] Fraud {mode} shape: {df.shape}")

    return df, y, scaler, train_columns

# CREDIT CARD TRANSFORM

def transform_creditcard(df: pd.DataFrame,
                         scaler: StandardScaler = None,
                         fit: bool = True):
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