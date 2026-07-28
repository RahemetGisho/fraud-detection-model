"""
src/models/evaluate_models.py

Loads the persisted models and scores them on the held-out test set using
metrics appropriate for imbalanced classification: AUC-PR, ROC-AUC, a
threshold tuned for best F1, and a confusion matrix. Writes a machine
readable report to reports/evaluation_report.json.

Prerequisite: src/models/train_models.py must have been run first.

Run standalone:
    python src/models/evaluate_models.py
"""

import os
import json
import logging

import numpy as np
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from src.models.config import ModelConfig, DatasetSpec
from src.models.train_models import load_train_test

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def find_best_threshold(y_true, proba, config: ModelConfig) -> tuple:
    """Sweep a threshold grid and return the (threshold, F1) pair that
    maximizes F1 on the given labels/probabilities."""
    best_threshold, best_f1 = 0.5, 0.0
    grid = np.arange(
        config.eval.threshold_grid_start,
        config.eval.threshold_grid_stop,
        config.eval.threshold_grid_step,
    )
    for threshold in grid:
        preds = (proba >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_threshold = score, threshold
    return round(float(best_threshold), 2), round(float(best_f1), 4)


def evaluate_model(model, X_test, y_test, config: ModelConfig, name: str) -> dict:
    """Compute AUC-PR, ROC-AUC, best-threshold F1, and confusion matrix."""
    proba = model.predict_proba(X_test)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_test, proba, config)
    preds = (proba >= best_threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y_test, proba)
    auc_pr = round(float(auc(recall, precision)), 4)
    roc_auc = round(float(roc_auc_score(y_test, proba)), 4)
    cm = confusion_matrix(y_test, preds).tolist()

    result = {
        "model": name,
        "best_threshold": best_threshold,
        "auc_pr": auc_pr,
        "roc_auc": roc_auc,
        "f1": best_f1,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_test, preds, output_dict=True, zero_division=0
        ),
    }
    logger.info(
        "%s -> AUC-PR=%.4f | ROC-AUC=%.4f | F1=%.4f @ threshold=%.2f",
        name,
        auc_pr,
        roc_auc,
        best_f1,
        best_threshold,
    )
    return result


def evaluate_dataset(dataset: DatasetSpec, config: ModelConfig) -> dict:
    """Load both saved models for one dataset and evaluate each on its test set."""
    _, _, X_test, y_test = load_train_test(
        dataset.train_path, dataset.test_path, dataset.target_col
    )

    dataset_results = {}
    for model_type, label in (("lr", "Logistic Regression"), ("xgb", "XGBoost")):
        model_path = os.path.join(
            config.paths.models_dir, f"{model_type}_{dataset.model_prefix}.pkl"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run src/models/train_models.py first."
            )
        model = joblib.load(model_path)
        dataset_results[model_type] = evaluate_model(
            model, X_test, y_test, config, f"{label} - {dataset.name}"
        )
    return dataset_results


def evaluate_all(config: ModelConfig = None) -> dict:
    config = config or ModelConfig()
    results = {
        dataset.name: evaluate_dataset(dataset, config) for dataset in config.datasets
    }

    os.makedirs(config.paths.reports_dir, exist_ok=True)
    report_path = os.path.join(config.paths.reports_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote evaluation report to %s", report_path)

    return results


if __name__ == "__main__":
    evaluate_all()
