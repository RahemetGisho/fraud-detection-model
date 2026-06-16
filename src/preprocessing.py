"""
Data Cleaning
Responsibilities:
  1. Remove duplicate rows.
  2. Handle missing values (drop or impute with justification).
  3. Correct data types.

"""
import numpy as np
import pandas as pd

# Fraud_Data cleaning
def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)
    # 1: Remove duplicates 
    df.drop_duplicates(inplace=True)
    print(f"[clean_fraud_data] Duplicates dropped  : "
          f"{initial_rows - len(df):,}")

    # 2. Drop rows with null target 
    before = len(df)
    df.dropna(subset=["class"], inplace=True)
    print(f"[clean_fraud_data] Null-target rows dropped : {before - len(df):,}")

    # 3.Drop rows with null ip_address 
    before = len(df)
    df.dropna(subset=["ip_address"], inplace=True)
    print(f"[clean_fraud_data] Null-ip rows dropped     : {before - len(df):,}")

    # 4.Impute missing age with median 
    missing_age = df["age"].isna().sum()
    if missing_age > 0:
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)
        print(
        f"[clean_fraud_data] Age nulls imputed "
        f"(median={median_age:.0f}): {missing_age:,}"
        )

    # 5: Fill missing categoricals with 'Unknown' 
    for col in ["source", "browser", "sex"]:
        n_null = df[col].isna().sum()
        if n_null > 0:
            df[col] = df[col].fillna("Unknown")
            print(
            f"[clean_fraud_data] "
            f"{col} nulls filled : {n_null:,}"
            )

    # 6: Convert ip_address to int64 
    df["ip_address"] = df["ip_address"].astype(np.int64)

    # 7: Ensure datetime types 
    for col in ["signup_time", "purchase_time"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    # 8: Ensure target is int 
    df["class"] = df["class"].astype(int)
    df.reset_index(drop=True, inplace=True)
    print(f"[clean_fraud_data] Final shape : {df.shape}  "
          f"(removed {initial_rows - len(df):,} rows total)")
    return df

# CreditCard cleaning
def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    initial_rows = len(df)

    # 1: Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"[clean_creditcard] Duplicates dropped : "
          f"{initial_rows - len(df):,}")

    # 2: Drop rows with any null
    before = len(df)
    df.dropna(inplace=True)
    print(f"[clean_creditcard] Null rows dropped  : {before - len(df):,}")

    # 3: Correct target dtype
    df["Class"] = df["Class"].astype(int)
    df.reset_index(drop=True, inplace=True)
    print(f"[clean_creditcard] Final shape : {df.shape}  "
          f"(removed {initial_rows - len(df):,} rows total)")
    return df