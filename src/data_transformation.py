"""
src/data_transformation.py
===========================
Task 1 — Step 5: Data Transformation

Responsibilities:
  1. Drop columns that are identifiers or raw timestamps (not model inputs).
  2. One-hot encode categorical features.
  3. Scale numerical features with StandardScaler.

Why StandardScaler?
  Logistic Regression and distance-based models are sensitive to feature
  magnitude. StandardScaler (μ=0, σ=1) brings all features onto the same
  scale without distorting their distributions.  MinMaxScaler would clip
  outliers into the [0,1] range, but fraud data has extreme outliers
  (e.g. very high purchase values) that carry signal — we want to keep
  their relative magnitude.

The scaler is returned so it can be:
  - Saved and reused at inference time (critical for production).
  - Applied to the test set with the SAME parameters (no data leakage).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Fraud_Data
# ─────────────────────────────────────────────────────────────────────────────

# Columns to drop before modelling — these are IDs, raw timestamps,
# or already encoded as derived features
_FRAUD_DROP_COLS = [
    "user_id",        # identifier — not a predictive feature
    "device_id",      # high-cardinality identifier — too many unique values
    "signup_time",    # replaced by time_since_signup and is_same_day
    "purchase_time",  # replaced by hour_of_day and day_of_week
    "ip_address",     # replaced by country after geolocation merge
]

# Categorical columns to one-hot encode
_FRAUD_CAT_COLS = ["source", "browser", "sex", "country"]


def transform_fraud_data(df: pd.DataFrame,
                          scaler: StandardScaler = None,
                          fit: bool = True):
    """
    Prepare the feature-engineered Fraud_Data for modelling.

    Parameters
    ----------
    df     : DataFrame that has passed through feature_engineering.engineer_all.
    scaler : An existing fitted StandardScaler.  Pass one when transforming
             the test set (fit=False) to avoid data leakage.
    fit    : If True, fit a new scaler on df and return it.
             If False, use the provided scaler — transform only.

    Returns
    -------
    X      : pd.DataFrame — scaled, encoded feature matrix.
    y      : pd.Series    — target (0/1).
    scaler : fitted StandardScaler (None if fit=False and none provided).
    """
    df = df.copy()

    # ── Separate target ────────────────────────────────────────────────────
    y = df["class"].astype(int)
    df.drop(columns=["class"], inplace=True)

    # ── Drop identifier / raw timestamp columns ────────────────────────────
    drop = [c for c in _FRAUD_DROP_COLS if c in df.columns]
    df.drop(columns=drop, inplace=True)

    # ── One-hot encode categoricals ────────────────────────────────────────
    cat_cols = [c for c in _FRAUD_CAT_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    # Ensure all dummy columns are int (some pandas versions return bool)
    bool_cols = df.select_dtypes(include=bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    # ── Scale numerical features ───────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        if scaler is None:
            raise ValueError(
                "fit=False requires a pre-fitted scaler to be passed."
            )
        df[num_cols] = scaler.transform(df[num_cols])

    print(f"[data_transformation] Fraud_Data X shape : {df.shape}")
    print(f"[data_transformation] Target distribution:\n"
          f"  {y.value_counts().to_dict()}")
    return df, y, scaler


# ─────────────────────────────────────────────────────────────────────────────
# CreditCard
# ─────────────────────────────────────────────────────────────────────────────

def transform_creditcard(df: pd.DataFrame,
                          scaler: StandardScaler = None,
                          fit: bool = True):
    """
    Prepare the CreditCard dataset for modelling.

    V1-V28 are already PCA-transformed (zero-mean by construction), so
    only Time and Amount need scaling.

    Parameters
    ----------
    df     : cleaned CreditCard DataFrame.
    scaler : existing fitted StandardScaler (for test set transformation).
    fit    : True → fit new scaler; False → transform only.

    Returns
    -------
    X      : pd.DataFrame.
    y      : pd.Series — target (Class 0/1).
    scaler : fitted StandardScaler on [Time, Amount].
    """
    df = df.copy()

    # Separate target
    y = df["Class"].astype(int)
    df.drop(columns=["Class"], inplace=True)

    # Scale Time and Amount — V1-V28 are already on a comparable PCA scale
    scale_cols = ["Time", "Amount"]

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        if scaler is None:
            raise ValueError(
                "fit=False requires a pre-fitted scaler to be passed."
            )
        df[scale_cols] = scaler.transform(df[scale_cols])

    print(f"[data_transformation] CreditCard X shape : {df.shape}")
    print(f"[data_transformation] Target distribution:\n"
          f"  {y.value_counts().to_dict()}")
    return df, y, scaler