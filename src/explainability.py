"""
src/explainability.py

Reusable model explainability utilities: built-in feature importance,
global and local SHAP analysis, and confusion-matrix cohort extraction
(true positive / false positive / false negative) for spotlighting
individual predictions.

This replaces the logic that previously only existed inside
notebooks/shap_explainability.ipynb — including hardcoded absolute
Windows paths — with functions the dashboard, reports, and tests can
all import directly.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap

TOP_N_FEATURES_DEFAULT = 10
TOP_N_DRIVERS_DEFAULT = 5


@dataclass(frozen=True)
class ConfusionCohorts:
    """Row indices (positional, into X_test) for each prediction outcome."""
    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    true_negative: np.ndarray


def get_confusion_cohorts(y_true: pd.Series, y_pred: np.ndarray) -> ConfusionCohorts:
    """Split test-set row indices into TP / FP / FN / TN cohorts so a
    specific example of each can be pulled out for a local SHAP force plot."""
    y_true_arr = np.asarray(y_true).flatten()
    y_pred_arr = np.asarray(y_pred).flatten()

    return ConfusionCohorts(
        true_positive=np.where((y_true_arr == 1) & (y_pred_arr == 1))[0],
        false_positive=np.where((y_true_arr == 0) & (y_pred_arr == 1))[0],
        false_negative=np.where((y_true_arr == 1) & (y_pred_arr == 0))[0],
        true_negative=np.where((y_true_arr == 0) & (y_pred_arr == 0))[0],
    )


def built_in_feature_importance(model, feature_names) -> pd.DataFrame:
    """Extract the model's native Gain-based feature importance, sorted
    descending. Works for any sklearn-API tree ensemble that exposes
    `feature_importances_` (XGBoost, LightGBM, RandomForest)."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    return pd.DataFrame({
        "feature": np.asarray(feature_names)[order],
        "importance_gain": importances[order],
    }).reset_index(drop=True)


def compute_shap_values(model, X_test: pd.DataFrame) -> Tuple[shap.TreeExplainer, shap.Explanation]:
    """Compute Shapley values with TreeExplainer, which is exact and fast
    for tree ensembles (unlike KernelExplainer, which is model-agnostic
    but far slower and only approximate)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def top_shap_drivers(shap_values: shap.Explanation, feature_names, top_n: int = TOP_N_DRIVERS_DEFAULT) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value — the global driver
    ranking used for the 'top N drivers of fraud' business narrative."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranked = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return ranked.head(top_n)


def compare_importance_rankings(
    gain_importance: pd.DataFrame, shap_importance: pd.DataFrame, top_n: int = TOP_N_DRIVERS_DEFAULT
) -> pd.DataFrame:
    """Side-by-side rank comparison between built-in Gain importance and
    SHAP mean-|value| importance, to surface features where the two
    disagree (a common, genuinely informative SHAP finding)."""
    gain_rank = {row.feature: i + 1 for i, row in gain_importance.iterrows()}
    shap_rank = {row.feature: i + 1 for i, row in shap_importance.iterrows()}

    features = list(dict.fromkeys(list(gain_rank) + list(shap_rank)))
    rows = []
    for feature in features:
        g_rank = gain_rank.get(feature)
        s_rank = shap_rank.get(feature)
        if g_rank is None and s_rank is None:
            continue
        rows.append({
            "feature": feature,
            "gain_rank": g_rank,
            "shap_rank": s_rank,
        })
    comparison = pd.DataFrame(rows)
    comparison = comparison[
        (comparison["gain_rank"].notna()) | (comparison["shap_rank"].notna())
    ]
    return comparison.sort_values(
        by=["shap_rank", "gain_rank"], na_position="last"
    ).head(top_n * 2).reset_index(drop=True)


def plot_global_summary(shap_values: shap.Explanation, X_test: pd.DataFrame, save_path: str,
                          max_display: int = TOP_N_FEATURES_DEFAULT) -> str:
    """Save the global SHAP summary plot (magnitude + directionality)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.title("Global SHAP Summary: Feature Impact & Direction", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def plot_local_force(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    X_test: pd.DataFrame,
    row_idx: int,
    case_label: str,
    save_path: str,
) -> str:
    """Save a force plot for one specific prediction (e.g. the first
    true positive, false positive, or false negative)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 3))
    shap.plots.force(
        explainer.expected_value,
        shap_values.values[row_idx],
        X_test.iloc[row_idx],
        matplotlib=True,
        show=False,
    )
    plt.title(f"SHAP Force Plot: {case_label} (row {row_idx})", fontsize=11, weight="bold", pad=20)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def explain_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
    dataset_label: str = "model",
) -> Dict[str, object]:
    """
    Full explainability pass for one trained model: built-in importance,
    global SHAP summary, one local force plot per confusion-matrix cohort
    (when that cohort is non-empty), and a Gain-vs-SHAP ranking comparison.

    Returns a dict with the computed tables/paths, suitable for feeding a
    dashboard or a written report — nothing here requires re-running SHAP.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    gain_importance = built_in_feature_importance(model, X_test.columns)

    explainer, shap_values = compute_shap_values(model, X_test)
    shap_drivers = top_shap_drivers(shap_values, X_test.columns)
    comparison = compare_importance_rankings(gain_importance, shap_drivers)

    summary_path = plot_global_summary(
        shap_values, X_test, os.path.join(output_dir, f"{dataset_label}_shap_global_summary.png")
    )

    cohorts = get_confusion_cohorts(y_test, y_pred)
    force_plot_paths = {}
    for case_name, indices in (
        ("true_positive", cohorts.true_positive),
        ("false_positive", cohorts.false_positive),
        ("false_negative", cohorts.false_negative),
    ):
        if len(indices) == 0:
            continue
        row_idx = int(indices[0])
        path = os.path.join(output_dir, f"{dataset_label}_shap_force_{case_name}.png")
        force_plot_paths[case_name] = plot_local_force(
            explainer, shap_values, X_test, row_idx, case_name.replace("_", " ").title(), path
        )

    return {
        "dataset": dataset_label,
        "gain_importance": gain_importance,
        "shap_drivers": shap_drivers,
        "gain_vs_shap_comparison": comparison,
        "global_summary_plot": summary_path,
        "force_plots": force_plot_paths,
        "cohort_sizes": {
            "true_positive": len(cohorts.true_positive),
            "false_positive": len(cohorts.false_positive),
            "false_negative": len(cohorts.false_negative),
            "true_negative": len(cohorts.true_negative),
        },
    }
