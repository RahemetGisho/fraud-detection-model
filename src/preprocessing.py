"""
src/preprocessing.py
====================
Task 1 — Step 1: Data Cleaning

Responsibilities:
  1. Remove duplicate rows.
  2. Handle missing values (drop or impute with justification).
  3. Correct data types.

This module is intentionally narrow — it only cleans.
Geolocation, feature engineering, and transformation live in separate modules.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Fraud_Data cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the e-commerce fraud dataset.

    Steps performed
    ---------------
    1. Drop exact duplicate rows.
       Justification: duplicate transactions inflate class counts and distort
       velocity / frequency features computed later.

    2. Drop rows where target ('class') is null.
       Justification: a row without a label is useless for supervised learning.

    3. Drop rows where ip_address is null.
       Justification: IP is needed for the geolocation merge; we cannot
       impute a realistic IP address.

    4. Impute missing 'age' with the median age.
       Justification: age is not available at prediction time for every user,
       and the median is a robust central-tendency estimate that does not
       introduce outliers the way a mean would on a right-skewed distribution.

    5. Fill missing categorical columns (source, browser, sex) with 'Unknown'.
       Justification: these are nominal features; labelling them 'Unknown'
       preserves the row without inventing a category.

    6. Convert ip_address from float64 → int64.
       The CSV stores it as a float (because pandas reads mixed integer/null
       columns as float).  Integer form is required for the range-based
       geolocation merge.

    7. Ensure signup_time and purchase_time are datetime64.
       parse_dates in the loader handles this normally; this step is a safety
       guard in case the DataFrame was constructed another way.

    Parameters
    ----------
    df : raw Fraud_Data DataFrame (output of load_fraud_data).

    Returns
    -------
    Cleaned DataFrame, reset index.
    """
    df = df.copy()
    initial_rows = len(df)

    # ── Step 1: Remove duplicates ─────────────────────────────────────────
    df.drop_duplicates(inplace=True)
    print(f"[clean_fraud_data] Duplicates dropped  : "
          f"{initial_rows - len(df):,}")

    # ── Step 2: Drop rows with null target ────────────────────────────────
    before = len(df)
    df.dropna(subset=["class"], inplace=True)
    print(f"[clean_fraud_data] Null-target rows dropped : {before - len(df):,}")

    # ── Step 3: Drop rows with null ip_address ────────────────────────────
    before = len(df)
    df.dropna(subset=["ip_address"], inplace=True)
    print(f"[clean_fraud_data] Null-ip rows dropped     : {before - len(df):,}")

    # ── Step 4: Impute missing age with median ────────────────────────────
    missing_age = df["age"].isna().sum()

    if missing_age > 0:
        median_age = df["age"].median()

        df["age"] = df["age"].fillna(median_age)

        print(
        f"[clean_fraud_data] Age nulls imputed "
        f"(median={median_age:.0f}): {missing_age:,}"
        )
    
     

    # ── Step 5: Fill missing categoricals with 'Unknown' ─────────────────
    for col in ["source", "browser", "sex"]:

        n_null = df[col].isna().sum()

        if n_null > 0:

            df[col] = df[col].fillna("Unknown")

            print(
            f"[clean_fraud_data] "
            f"{col} nulls filled : {n_null:,}"
            )

    # ── Step 6: Convert ip_address to int64 ───────────────────────────────
    df["ip_address"] = df["ip_address"].astype(np.int64)

    # ── Step 7: Ensure datetime types ────────────────────────────────────
    for col in ["signup_time", "purchase_time"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    # ── Step 8: Ensure target is int ─────────────────────────────────────
    df["class"] = df["class"].astype(int)

    df.reset_index(drop=True, inplace=True)
    print(f"[clean_fraud_data] Final shape : {df.shape}  "
          f"(removed {initial_rows - len(df):,} rows total)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CreditCard cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the bank credit-card transaction dataset.

    Steps performed
    ---------------
    1. Drop exact duplicate rows.
       Justification: same as above — duplicates bias model training.

    2. Drop rows with any null value.
       Justification: V1-V28 are PCA-transformed; imputing principal
       components is not meaningful.  The dataset has no missing values
       in practice, so this is a safety check.

    3. Ensure 'Class' is integer.

    Parameters
    ----------
    df : raw CreditCard DataFrame (output of load_creditcard).

    Returns
    -------
    Cleaned DataFrame, reset index.
    """
    df = df.copy()
    initial_rows = len(df)

    # Step 1: Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"[clean_creditcard] Duplicates dropped : "
          f"{initial_rows - len(df):,}")

    # Step 2: Drop rows with any null
    before = len(df)
    df.dropna(inplace=True)
    print(f"[clean_creditcard] Null rows dropped  : {before - len(df):,}")

    # Step 3: Correct target dtype
    df["Class"] = df["Class"].astype(int)

    df.reset_index(drop=True, inplace=True)
    print(f"[clean_creditcard] Final shape : {df.shape}  "
          f"(removed {initial_rows - len(df):,} rows total)")
    return df