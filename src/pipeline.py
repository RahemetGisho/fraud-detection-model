"""
src/pipeline.py
================
End-to-end ML pipeline with:
- Logging
- Processed data saving
- Reproducible training flow
"""

import os
import logging
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_loader import (
    load_fraud_data,
    load_ip_country,
    load_creditcard,
)

from src.preprocessing import clean_fraud_data, clean_creditcard
from src.geolocation import merge_ip_to_country
from src.feature_engineering import engineer_all
from src.data_transformation import (
    transform_fraud_data,
    transform_creditcard,
)
from src.imbalance_handling import handle_imbalance


# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def log(msg: str):
    print(msg)
    logging.info(msg)


# ─────────────────────────────────────────────
# Helper save function
# ─────────────────────────────────────────────

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────
# FRAUD PIPELINE
# ─────────────────────────────────────────────

def run_fraud_pipeline(
    fraud_path: str,
    ip_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    log("\n========== FRAUD PIPELINE START ==========")

    base = "data/processed/fraud"

    # 1. Load
    fraud_df = load_fraud_data(fraud_path)
    ip_df = load_ip_country(ip_path)
    log(f"Loaded fraud={len(fraud_df)} ip={len(ip_df)}")

    # 2. Clean
    fraud_df = clean_fraud_data(fraud_df)
    save_csv(fraud_df, f"{base}/01_cleaned.csv")
    log(f"After cleaning: {fraud_df.shape}")

    # 3. Geolocation
    fraud_df = merge_ip_to_country(fraud_df, ip_df)
    save_csv(fraud_df, f"{base}/02_geo.csv")

    # 4. Feature engineering
    fraud_df = engineer_all(fraud_df)
    save_csv(fraud_df, f"{base}/03_features.csv")

    # 5. Transform
    X, y, scaler = transform_fraud_data(fraud_df, fit=True)
    joblib.dump(scaler, f"{base}/fraud_scaler.pkl")

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # attach target safely (NO reserved keyword issue)
    train_df = X_train.copy()
    train_df["class"] = y_train.values

    test_df = X_test.copy()
    test_df["class"] = y_test.values

    save_csv(train_df, f"{base}/04_train.csv")
    save_csv(test_df, f"{base}/05_test.csv")

    log(f"Train={len(X_train)} Test={len(X_test)}")

    # 7. Imbalance handling (TRAIN ONLY)
    X_train_bal, y_train_bal = handle_imbalance(
        X_train, y_train, strategy="oversample", random_state=random_state
    )

    # attach target safely (NO reserved keyword usage)
    train_bal_df = X_train_bal.copy()
    train_bal_df["class"] = y_train_bal.values

    save_csv(
        train_bal_df,
        f"{base}/06_train_balanced.csv",
    )

    log("Fraud pipeline completed successfully.")
    log("=====================================\n")

    return {
        "X_train": X_train_bal,
        "X_test": X_test,
        "y_train": y_train_bal,
        "y_test": y_test,
        "scaler": scaler,
    }


# ─────────────────────────────────────────────
# CREDIT CARD PIPELINE
# ─────────────────────────────────────────────

def run_creditcard_pipeline(
    credit_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    log("\n========== CREDITCARD PIPELINE START ==========")

    base = "data/processed/creditcard"

    # 1. Load
    df = load_creditcard(credit_path)
    log(f"Loaded creditcard={len(df)}")

    # 2. Clean
    df = clean_creditcard(df)
    save_csv(df, f"{base}/01_cleaned.csv")

    # 3. Transform
    X, y, scaler = transform_creditcard(df, fit=True)
    joblib.dump(scaler, f"{base}/credit_scaler.pkl")

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    save_csv(X_train.assign(Class=y_train.values), f"{base}/02_train.csv")
    save_csv(X_test.assign(Class=y_test.values), f"{base}/03_test.csv")

    log("Creditcard pipeline completed successfully.")
    log("=====================================\n")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


# ─────────────────────────────────────────────
# CLI ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":
    FRAUD_PATH = "data/raw/Fraud_Data.csv"
    IP_PATH = "data/raw/IpAddress_to_Country.csv"
    CREDIT_PATH = "data/raw/creditcard (3).csv"

    fraud_results = run_fraud_pipeline(FRAUD_PATH, IP_PATH)
    credit_results = run_creditcard_pipeline(CREDIT_PATH)

    log("All pipelines finished successfully.")