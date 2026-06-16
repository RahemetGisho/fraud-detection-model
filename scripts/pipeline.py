"""
Clean ML Pipeline (Production-style correct version)

✔ Split BEFORE feature engineering to block group-by lookahead leakage
✔ Separate train/test feature calculation paths
✔ No leakage in scaler (continuous metrics isolated)
✔ Imbalance handling isolated strictly to training set
✔ Consistent train/test schema mapping via data transformation reindexing
"""

import os
import logging
import joblib

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
# ENVIRONMENT & CONFIG SETUP
# ─────────────────────────────────────────────
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def log(msg):
    print(msg)
    logging.info(msg)


def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────
# FRAUD PIPELINE
# ─────────────────────────────────────────────

def run_fraud_pipeline(fraud_path, ip_path, test_size=0.2, random_state=42):

    log("\n========== FRAUD PIPELINE START ==========")

    base = "data/processed/fraud"

    # 1. LOAD
    fraud_df = load_fraud_data(fraud_path)
    ip_df = load_ip_country(ip_path)

    # 2. CLEAN + GEO-ENRICHMENT
    fraud_df = clean_fraud_data(fraud_df)
    fraud_df = merge_ip_to_country(fraud_df, ip_df)

    # 3. SPLIT (CRITICAL FIX: Split before engineering to safeguard group aggregates)
    train_df, test_df = train_test_split(
        fraud_df,
        test_size=test_size,
        stratify=fraud_df["class"],
        random_state=random_state
    )

    # 4. FEATURE ENGINEERING (Executed inside separate validation boundaries)
    log("[pipeline] Calculating historical behavioral signals on Train partition...")
    train_df = engineer_all(train_df)

    log("[pipeline] Calculating historical behavioral signals on Test partition...")
    test_df = engineer_all(test_df)

    # 5. TRANSFORM (Isolates scaling distributions & aligns OHE categories)
    X_train, y_train, scaler, train_cols = transform_fraud_data(
        train_df, fit=True
    )

    X_test, y_test, _, _ = transform_fraud_data(
        test_df,
        scaler=scaler,
        fit=False,
        train_columns=train_cols
    )

    os.makedirs(base, exist_ok=True)
    joblib.dump(scaler, f"{base}/fraud_scaler.pkl")

    # 6. IMBALANCE HANDLING (TRAIN ONLY)
    X_train_bal, y_train_bal = handle_imbalance(
        X_train,
        y_train,
        strategy="undersample",
        random_state=random_state
    )

    log(f"Train: {len(X_train_bal)} rows | Test: {len(X_test)} rows")

    

    # 7. SAVE FINAL RESAMPLED TRAIN & CLEAN TEST DATASETS
    # BEFORE balancing snapshot

    train_final = X_train.copy()
    train_final["class"] = y_train.values
    save_csv(train_final, f"{base}/train_final.csv")

    # AFTER balancing snapshot
    train_balanced = X_train_bal.copy()
    train_balanced["class"] = y_train_bal.values
    save_csv(train_balanced, f"{base}/train_balanced.csv")

    # TEST (never touched)
    test_final = X_test.copy()
    test_final["class"] = y_test.values
    save_csv(test_final, f"{base}/test_final.csv")
# ─────────────────────────────────────────────
# CREDIT CARD PIPELINE
# ─────────────────────────────────────────────

def run_creditcard_pipeline(credit_path, test_size=0.2, random_state=42):

    log("\n========== CREDITCARD PIPELINE START ==========")
    base = "data/processed/creditcard"

    # 1. LOAD & 2. CLEAN
    df = load_creditcard(credit_path)
    df = clean_creditcard(df)

    # 3. SPLIT
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["Class"],
        random_state=random_state
    )

    # 4. TRANSFORM
    X_train, y_train, scaler = transform_creditcard(train_df, fit=True)

    X_test, y_test, _ = transform_creditcard(
        test_df,
        scaler=scaler,
        fit=False
    )

    os.makedirs(base, exist_ok=True)
    joblib.dump(scaler, f"{base}/scaler.pkl")

    log("Creditcard pipeline completed successfully.")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    run_fraud_pipeline(
        "data/raw/Fraud_Data.csv",
        "data/raw/IpAddress_to_Country.csv"
    )

    run_creditcard_pipeline(
        "data/raw/creditcard (3).csv"
    )

    log("\nALL PIPELINES FINISHED SUCCESSFULLY")