import os

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.explainability import (
    get_confusion_cohorts,
    built_in_feature_importance,
    compute_shap_values,
    top_shap_drivers,
    compare_importance_rankings,
    explain_model,
)


@pytest.fixture
def trained_model_and_data():
    """A small but real XGBoost model with clear signal, so SHAP has
    something meaningful to explain and force plots have at least one
    example of each confusion-matrix outcome."""
    np.random.seed(0)
    n = 300
    X = pd.DataFrame({
        "risky_feature": np.random.randn(n),
        "noise_1": np.random.randn(n),
        "noise_2": np.random.randn(n),
    })
    # Make the target strongly (but not perfectly) driven by one feature,
    # so both correct and incorrect predictions occur.
    y = ((X["risky_feature"] + 0.3 * np.random.randn(n)) > 0.5).astype(int)

    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    return model, X, y


def test_get_confusion_cohorts_partitions_all_rows():
    """Every row should land in exactly one of the four cohorts."""
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])
    cohorts = get_confusion_cohorts(y_true, y_pred)

    assert list(cohorts.true_positive) == [0]
    assert list(cohorts.false_negative) == [1]
    assert list(cohorts.true_negative) == [2]
    assert list(cohorts.false_positive) == [3]


def test_built_in_feature_importance_sorted_descending(trained_model_and_data):
    """Verifies the Gain importance table is a valid ranking over all features."""
    model, X, _ = trained_model_and_data
    importance = built_in_feature_importance(model, X.columns)

    assert set(importance["feature"]) == set(X.columns)
    values = importance["importance_gain"].to_numpy()
    assert (values[:-1] >= values[1:]).all()


def test_top_shap_drivers_surfaces_the_informative_feature(trained_model_and_data):
    """The engineered signal feature should dominate the SHAP ranking."""
    model, X, _ = trained_model_and_data
    _, shap_values = compute_shap_values(model, X)
    drivers = top_shap_drivers(shap_values, X.columns, top_n=3)

    assert drivers.iloc[0]["feature"] == "risky_feature"


def test_compare_importance_rankings_includes_both_ranks(trained_model_and_data):
    """The comparison table should carry a rank from each method for
    features present in both rankings."""
    model, X, _ = trained_model_and_data
    gain = built_in_feature_importance(model, X.columns)
    _, shap_values = compute_shap_values(model, X)
    shap_drivers = top_shap_drivers(shap_values, X.columns, top_n=len(X.columns))

    comparison = compare_importance_rankings(gain, shap_drivers, top_n=len(X.columns))
    assert "gain_rank" in comparison.columns
    assert "shap_rank" in comparison.columns
    assert len(comparison) > 0


def test_explain_model_end_to_end(tmp_path, trained_model_and_data):
    """Full pipeline: importance + SHAP + at least one force plot saved
    per non-empty confusion-matrix cohort (TP/FP/FN)."""
    model, X, y = trained_model_and_data
    result = explain_model(model, X, y, output_dir=str(tmp_path), dataset_label="test")

    assert os.path.exists(os.path.join(str(tmp_path), "test_shap_global_summary.png"))
    assert result["cohort_sizes"]["true_positive"] > 0
    assert len(result["force_plots"]) > 0
    for saved_path in result["force_plots"].values():
        assert os.path.exists(saved_path)
