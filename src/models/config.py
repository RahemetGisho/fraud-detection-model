"""
src/models/config.py

Centralized, typed configuration for model training, cross-validation, and
evaluation. Every hyperparameter and path that was previously a bare literal
scattered across notebooks lives here, named and documented in one place.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to locate and train on one processed dataset."""
    name: str
    train_path: str
    test_path: str
    target_col: str
    model_prefix: str


@dataclass(frozen=True)
class PathConfig:
    models_dir: str = "models"
    reports_dir: str = "reports"


@dataclass(frozen=True)
class XGBConfig:
    """XGBoost hyperparameters, matched to the values validated in
    notebooks/modeling.ipynb. scale_pos_weight is computed per-dataset at
    train time, not fixed here, since it depends on the class balance of
    whatever training split is passed in."""
    n_estimators: int = 600
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    eval_metric: str = "aucpr"
    n_jobs: int = -1
    random_state: int = 42


@dataclass(frozen=True)
class LogRegConfig:
    max_iter: int = 2000
    class_weight: str = "balanced"


@dataclass(frozen=True)
class EvalConfig:
    """Threshold sweep used to pick the operating point that maximizes F1,
    plus cross-validation settings."""
    threshold_grid_start: float = 0.1
    threshold_grid_stop: float = 0.95
    threshold_grid_step: float = 0.05
    cv_folds: int = 5
    random_state: int = 42


def _default_datasets() -> List[DatasetSpec]:
    return [
        DatasetSpec(
            name="fraud",
            # Undersampled at training time only (see scripts/pipeline.py);
            # the test split is never resampled.
            train_path="data/processed/fraud/train_balanced.csv",
            test_path="data/processed/fraud/test_final.csv",
            target_col="class",
            model_prefix="fraud",
        ),
        DatasetSpec(
            name="creditcard",
            # No row-level resampling here: at ~285k rows with <0.2% fraud,
            # undersampling would discard most legitimate transactions and
            # destroy the PCA feature distribution. Class imbalance is
            # instead handled via class_weight / scale_pos_weight below.
            train_path="data/processed/creditcard/train_final.csv",
            test_path="data/processed/creditcard/test_final.csv",
            target_col="Class",
            model_prefix="cc",
        ),
    ]


@dataclass(frozen=True)
class ModelConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    xgb: XGBConfig = field(default_factory=XGBConfig)
    logreg: LogRegConfig = field(default_factory=LogRegConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    datasets: List[DatasetSpec] = field(default_factory=_default_datasets)
