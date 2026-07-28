"""
dashboard/app.py

Interactive Streamlit dashboard for the fraud detection models. Lets a
non-technical stakeholder:
  - see model performance metrics for both datasets
  - explore individual predictions and their SHAP explanations
  - see the confusion matrix translated into dollar terms

Run with:
    streamlit run dashboard/app.py

Expects that scripts/pipeline.py, src/models/train_models.py, and
src/models/evaluate_models.py have already been run, so that
models/*.pkl, data/processed/**/*.csv, and reports/evaluation_report.json
exist. The app degrades gracefully (with clear instructions) if any of
those are missing, rather than crashing.
"""

import json
import os
import sys

# streamlit run only puts this file's own folder on sys.path, not the
# project root — insert the root before any project import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import streamlit as st

from src.models.config import ModelConfig
from src.explainability import (
    compute_shap_values,
    plot_global_summary,
    get_confusion_cohorts,
)
from src.business_impact import business_impact_from_confusion_matrix, CostAssumptions

st.set_page_config(page_title="Fraud Detection — Model Insights", layout="wide")

CONFIG = ModelConfig()
EVAL_REPORT_PATH = os.path.join(CONFIG.paths.reports_dir, "evaluation_report.json")


@st.cache_data
def load_evaluation_report():
    if not os.path.exists(EVAL_REPORT_PATH):
        return None
    with open(EVAL_REPORT_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model(dataset_prefix: str, model_type: str):
    path = os.path.join(CONFIG.paths.models_dir, f"{model_type}_{dataset_prefix}.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data
def load_test_data(test_path: str, target_col: str):
    if not os.path.exists(test_path):
        return None, None
    df = pd.read_csv(test_path)
    return df.drop(columns=[target_col]), df[target_col]


def missing_artifacts_notice():
    st.warning(
        "No trained models or evaluation report found yet.\n\n"
        "Run the pipeline first:\n"
        "```bash\npython main.py\n```\n"
        "or step by step:\n"
        "```bash\npython -m scripts.pipeline\n"
        "python -m src.models.train_models\n"
        "python -m src.models.cross_validate\n"
        "python -m src.models.evaluate_models\n```"
    )


st.title("Fraud Detection — Model Insights Dashboard")
st.caption(
    "Explore model performance, individual prediction explanations, "
    "and the business impact of the fraud detection pipeline."
)

report = load_evaluation_report()
if report is None:
    missing_artifacts_notice()
    st.stop()

dataset_names = list(report.keys())
dataset_choice = st.sidebar.selectbox("Dataset", dataset_names)
model_choice = st.sidebar.selectbox(
    "Model",
    ["xgb", "lr"],
    format_func=lambda m: "XGBoost" if m == "xgb" else "Logistic Regression",
)

dataset_spec = next(d for d in CONFIG.datasets if d.name == dataset_choice)
metrics = report[dataset_choice][model_choice]

tab_metrics, tab_predictions, tab_explainability, tab_business = st.tabs(
    ["Performance Metrics", "Prediction Explorer", "Explainability", "Business Impact"]
)

with tab_metrics:
    st.subheader(f"{dataset_choice} — {model_choice.upper()} performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-PR", f"{metrics['auc_pr']:.3f}")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col3.metric("F1 (best threshold)", f"{metrics['f1']:.3f}")
    col4.metric("Decision threshold", f"{metrics['best_threshold']:.2f}")

    st.markdown("**Confusion matrix**")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Legit", "Actual: Fraud"],
        columns=["Predicted: Legit", "Predicted: Fraud"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**All models on this dataset**")
    comparison_rows = []
    for m_type, m_result in report[dataset_choice].items():
        comparison_rows.append(
            {
                "Model": "XGBoost" if m_type == "xgb" else "Logistic Regression",
                "AUC-PR": m_result["auc_pr"],
                "ROC-AUC": m_result["roc_auc"],
                "F1": m_result["f1"],
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

with tab_predictions:
    st.subheader("Explore individual predictions")
    X_test, y_test = load_test_data(dataset_spec.test_path, dataset_spec.target_col)
    model = load_model(dataset_spec.model_prefix, model_choice)

    if X_test is None or model is None:
        missing_artifacts_notice()
    else:
        row_idx = st.slider("Test-set row", 0, len(X_test) - 1, 0)
        row = X_test.iloc[[row_idx]]
        proba = model.predict_proba(row)[0, 1]
        actual = int(y_test.iloc[row_idx])

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted fraud probability", f"{proba:.1%}")
        col2.metric(
            "Model decision",
            "FRAUD" if proba >= metrics["best_threshold"] else "Legitimate",
        )
        col3.metric("Actual label", "Fraud" if actual == 1 else "Legitimate")

        st.markdown("**Feature values for this transaction**")
        st.dataframe(row.T.rename(columns={row_idx: "value"}), use_container_width=True)

with tab_explainability:
    st.subheader("Why does the model decide this way?")
    X_test, y_test = load_test_data(dataset_spec.test_path, dataset_spec.target_col)
    model = load_model(dataset_spec.model_prefix, model_choice)

    if X_test is None or model is None:
        missing_artifacts_notice()
    elif model_choice != "xgb":
        st.info(
            "SHAP explanations are shown for the XGBoost model — switch the Model selector to see them."
        )
    else:
        sample_size = min(200, len(X_test))
        X_sample = X_test.sample(sample_size, random_state=42)
        _, shap_values = compute_shap_values(model, X_sample)

        plot_path = os.path.join(
            CONFIG.paths.reports_dir,
            "plots",
            f"{dataset_choice}_dashboard_shap_summary.png",
        )
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_global_summary(shap_values, X_sample, plot_path)

        st.markdown(f"**Global feature impact** (sampled {sample_size} rows for speed)")
        st.image(plot_path, use_container_width=True)
        st.caption(
            "Each point is one transaction. Position on the x-axis shows whether that "
            "feature pushed the prediction toward fraud (right) or legitimate (left)."
        )

with tab_business:
    st.subheader("Translating detection performance into dollars")
    st.caption("Adjust the cost assumptions to match your own unit economics.")

    col1, col2 = st.columns(2)
    avg_fraud_loss = col1.number_input(
        "Average loss per missed fraud case ($)", value=100.0, min_value=0.0
    )
    avg_investigation_cost = col2.number_input(
        "Average cost per false alarm investigated ($)", value=5.0, min_value=0.0
    )

    impact = business_impact_from_confusion_matrix(
        metrics["confusion_matrix"],
        CostAssumptions(
            avg_fraud_loss_if_missed=avg_fraud_loss,
            avg_investigation_cost_per_false_alarm=avg_investigation_cost,
        ),
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud caught", f"{impact['fraud_caught_rate']:.1%}")
    col2.metric("Cost of missed fraud", f"${impact['cost_of_missed_fraud']:,.0f}")
    col3.metric("Cost of false alarms", f"${impact['cost_of_false_alarms']:,.0f}")

    st.metric(
        "Estimated savings vs. no model at all",
        f"${impact['estimated_savings_vs_no_model']:,.0f}",
        help="Baseline assumes every fraud case would have gone undetected without a model.",
    )
