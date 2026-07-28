import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.models.config import ModelConfig, DatasetSpec
from src.models.train_models import (
    load_train_test,
    compute_scale_pos_weight,
    train_logistic_regression,
    train_xgboost,
    train_all,
)
from src.models.evaluate_models import find_best_threshold, evaluate_all
from src.models.cross_validate import cross_validate_auc_pr


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Writes a small train/test CSV pair to a temp dir and returns a
    ModelConfig pointed at them, mirroring what scripts/pipeline.py would
    have produced."""
    np.random.seed(42)
    feature_cols = ["f1", "f2", "f3"]

    def make_split(n_rows, n_pos):
        X = pd.DataFrame({c: np.random.randn(n_rows) for c in feature_cols})
        y = np.array([0] * (n_rows - n_pos) + [1] * n_pos)
        np.random.shuffle(y)
        df = X.copy()
        df["class"] = y
        return df

    train_df = make_split(120, 40)
    test_df = make_split(40, 8)

    fraud_dir = tmp_path / "fraud"
    fraud_dir.mkdir()
    train_path = fraud_dir / "train_balanced.csv"
    test_path = fraud_dir / "test_final.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    dataset = DatasetSpec(
        name="fraud",
        train_path=str(train_path),
        test_path=str(test_path),
        target_col="class",
        model_prefix="fraud",
    )
    config = ModelConfig(datasets=[dataset])
    return config, dataset


def test_load_train_test_splits_features_from_target(synthetic_dataset):
    """Verifies the target column is separated correctly and shapes align."""
    config, dataset = synthetic_dataset
    X_train, y_train, X_test, y_test = load_train_test(
        dataset.train_path, dataset.test_path, dataset.target_col
    )
    assert "class" not in X_train.columns
    assert len(X_train) == len(y_train) == 120
    assert len(X_test) == len(y_test) == 40


def test_load_train_test_missing_file_raises():
    """Verifies a clear FileNotFoundError instead of a raw pandas error."""
    with pytest.raises(FileNotFoundError, match="Training file not found"):
        load_train_test("nope_train.csv", "nope_test.csv", "class")


def test_compute_scale_pos_weight():
    """Verifies the negative/positive ratio used to cost-weight XGBoost."""
    y = pd.Series([0] * 90 + [1] * 10)
    assert compute_scale_pos_weight(y) == pytest.approx(9.0)


def test_train_and_evaluate_round_trip(tmp_path, synthetic_dataset, monkeypatch):
    """End-to-end: train both models, persist them, then evaluate_all reads
    them back and produces a report — this is the exact chain that was
    broken in main.py before the fix."""
    config, dataset = synthetic_dataset
    monkeypatch.chdir(tmp_path)

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    config = replace(
        config,
        paths=replace(config.paths, models_dir=str(models_dir), reports_dir=str(reports_dir)),
    )

    train_all(config)
    assert (models_dir / "lr_fraud.pkl").exists()
    assert (models_dir / "xgb_fraud.pkl").exists()

    results = evaluate_all(config)
    assert "fraud" in results
    assert set(results["fraud"].keys()) == {"lr", "xgb"}
    for model_result in results["fraud"].values():
        assert "auc_pr" in model_result
        assert "confusion_matrix" in model_result

    report_path = reports_dir / "evaluation_report.json"
    assert report_path.exists()
    with open(report_path) as f:
        saved = json.load(f)
    assert saved == results


def test_find_best_threshold_picks_highest_f1():
    """A model with clear separation should find a threshold with F1 > 0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.7])
    config = ModelConfig()
    threshold, f1 = find_best_threshold(y_true, proba, config)
    assert 0.0 < threshold < 1.0
    assert f1 == pytest.approx(1.0)


def test_cross_validate_returns_per_fold_scores(synthetic_dataset):
    """Verifies CV returns one AUC-PR per fold plus a mean/std summary."""
    config, dataset = synthetic_dataset
    fast_config = replace(config, eval=replace(config.eval, cv_folds=3))
    X_train, y_train, _, _ = load_train_test(
        dataset.train_path, dataset.test_path, dataset.target_col
    )
    results = cross_validate_auc_pr(X_train, y_train, fast_config)
    assert len(results["fold_scores"]) == 3
    assert "mean_auc_pr" in results
    assert "std_auc_pr" in results
