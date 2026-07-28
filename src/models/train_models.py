"""
src/models/train_models.py

Trains a baseline Logistic Regression and a tuned XGBoost classifier for
each dataset defined in ModelConfig, and persists both to disk with joblib.

Prerequisite: scripts/pipeline.py must have been run first, so that the
processed train/test CSVs referenced in ModelConfig actually exist.

Run standalone:
    python src/models/train_models.py
"""

import os
import logging
from typing import Tuple

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.models.config import ModelConfig, DatasetSpec

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def load_train_test(
    train_path: str, test_path: str, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load a processed train/test CSV pair and split features from target."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training file not found: {train_path}. Run scripts/pipeline.py first."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test file not found: {test_path}. Run scripts/pipeline.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    return X_train, y_train, X_test, y_test


def train_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> LogisticRegression:
    """Fit the interpretable baseline model."""
    model = LogisticRegression(
        max_iter=config.logreg.max_iter,
        class_weight=config.logreg.class_weight,
    )
    model.fit(X_train, y_train)
    return model


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """Ratio of negative to positive examples, used to cost-weight XGBoost
    towards the minority (fraud) class."""
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / max(n_pos, 1)


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> XGBClassifier:
    """Fit the main ensemble model with the validated hyperparameters."""
    model = XGBClassifier(
        n_estimators=config.xgb.n_estimators,
        max_depth=config.xgb.max_depth,
        learning_rate=config.xgb.learning_rate,
        subsample=config.xgb.subsample,
        colsample_bytree=config.xgb.colsample_bytree,
        min_child_weight=config.xgb.min_child_weight,
        scale_pos_weight=compute_scale_pos_weight(y_train),
        eval_metric=config.xgb.eval_metric,
        n_jobs=config.xgb.n_jobs,
        random_state=config.xgb.random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_dataset(dataset: DatasetSpec, config: ModelConfig) -> dict:
    """Train and persist both models for a single dataset."""
    logger.info("Training models for dataset: %s", dataset.name)
    X_train, y_train, _, _ = load_train_test(
        dataset.train_path, dataset.test_path, dataset.target_col
    )

    os.makedirs(config.paths.models_dir, exist_ok=True)

    lr_model = train_logistic_regression(X_train, y_train, config)
    lr_path = os.path.join(config.paths.models_dir, f"lr_{dataset.model_prefix}.pkl")
    joblib.dump(lr_model, lr_path)
    logger.info("Saved Logistic Regression -> %s", lr_path)

    xgb_model = train_xgboost(X_train, y_train, config)
    xgb_path = os.path.join(config.paths.models_dir, f"xgb_{dataset.model_prefix}.pkl")
    joblib.dump(xgb_model, xgb_path)
    logger.info("Saved XGBoost -> %s", xgb_path)

    return {"logistic_regression": lr_model, "xgboost": xgb_model}


def train_all(config: ModelConfig = None) -> dict:
    """Train and persist models for every dataset in the config."""
    config = config or ModelConfig()
    return {dataset.name: train_dataset(dataset, config) for dataset in config.datasets}


if __name__ == "__main__":
    train_all()
    logger.info("All models trained and saved.")
