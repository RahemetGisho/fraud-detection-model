"""
src/models/cross_validate.py

Stratified K-fold cross-validation on the training split only, to check
that AUC-PR performance is stable rather than an artifact of one particular
train/test split. Uses the same XGBoost configuration as train_models.py.

Run standalone:
    python src/models/cross_validate.py
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc
from xgboost import XGBClassifier

from src.models.config import ModelConfig, DatasetSpec
from src.models.train_models import load_train_test, compute_scale_pos_weight

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def cross_validate_auc_pr(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> dict:
    """Run stratified K-fold CV and return per-fold and aggregate AUC-PR."""
    skf = StratifiedKFold(
        n_splits=config.eval.cv_folds,
        shuffle=True,
        random_state=config.eval.random_state,
    )

    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train), start=1
    ):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(
            n_estimators=config.xgb.n_estimators,
            max_depth=config.xgb.max_depth,
            learning_rate=config.xgb.learning_rate,
            subsample=config.xgb.subsample,
            colsample_bytree=config.xgb.colsample_bytree,
            min_child_weight=config.xgb.min_child_weight,
            scale_pos_weight=compute_scale_pos_weight(y_fold_train),
            eval_metric=config.xgb.eval_metric,
            n_jobs=config.xgb.n_jobs,
            random_state=config.xgb.random_state,
        )
        model.fit(X_fold_train, y_fold_train)

        proba = model.predict_proba(X_fold_val)[:, 1]
        precision, recall, _ = precision_recall_curve(y_fold_val, proba)
        fold_auc_pr = float(auc(recall, precision))
        fold_scores.append(fold_auc_pr)
        logger.info(
            "Fold %d/%d AUC-PR: %.4f", fold_idx, config.eval.cv_folds, fold_auc_pr
        )

    return {
        "fold_scores": [round(s, 4) for s in fold_scores],
        "mean_auc_pr": round(float(np.mean(fold_scores)), 4),
        "std_auc_pr": round(float(np.std(fold_scores)), 4),
    }


def cross_validate_dataset(dataset: DatasetSpec, config: ModelConfig) -> dict:
    logger.info("Cross-validating dataset: %s", dataset.name)
    X_train, y_train, _, _ = load_train_test(
        dataset.train_path, dataset.test_path, dataset.target_col
    )
    return cross_validate_auc_pr(X_train, y_train, config)


def cross_validate_all(config: ModelConfig = None) -> dict:
    config = config or ModelConfig()
    results = {
        dataset.name: cross_validate_dataset(dataset, config)
        for dataset in config.datasets
    }

    os.makedirs(config.paths.reports_dir, exist_ok=True)
    report_path = os.path.join(config.paths.reports_dir, "cross_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote cross-validation report to %s", report_path)

    return results


if __name__ == "__main__":
    cross_validate_all()
