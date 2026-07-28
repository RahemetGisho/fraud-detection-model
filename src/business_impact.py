"""
src/business_impact.py

Translates a confusion matrix into plain-English financial terms: money
lost to missed fraud, money lost to investigating false alarms, and the
net position versus doing nothing. Kept separate from src/explainability.py
so it can be unit tested without any SHAP/model dependency, and reused by
both the dashboard and the technical report.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CostAssumptions:
    """Editable cost assumptions behind the business-impact estimate.
    Defaults are illustrative placeholders — swap in real unit economics
    (chargeback cost, average investigation labor cost) before publishing
    a number externally."""
    avg_fraud_loss_if_missed: float = 100.0
    avg_investigation_cost_per_false_alarm: float = 5.0


def business_impact_from_confusion_matrix(
    confusion_matrix: list, assumptions: CostAssumptions = CostAssumptions()
) -> Dict[str, float]:
    """
    confusion_matrix must be the standard sklearn 2x2 layout:
        [[TN, FP],
         [FN, TP]]

    Returns money lost to missed fraud (FN), money spent investigating
    false alarms (FP), and the combined cost — all directly comparable to
    the do-nothing baseline of losing every fraud case.
    """
    (tn, fp), (fn, tp) = confusion_matrix

    cost_of_missed_fraud = fn * assumptions.avg_fraud_loss_if_missed
    cost_of_false_alarms = fp * assumptions.avg_investigation_cost_per_false_alarm
    total_fraud_cases = fn + tp
    baseline_cost_if_no_model = total_fraud_cases * assumptions.avg_fraud_loss_if_missed

    total_model_cost = cost_of_missed_fraud + cost_of_false_alarms
    estimated_savings = baseline_cost_if_no_model - total_model_cost

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "cost_of_missed_fraud": cost_of_missed_fraud,
        "cost_of_false_alarms": cost_of_false_alarms,
        "total_model_cost": total_model_cost,
        "baseline_cost_if_no_model": baseline_cost_if_no_model,
        "estimated_savings_vs_no_model": estimated_savings,
        "fraud_caught_rate": (tp / total_fraud_cases) if total_fraud_cases else 0.0,
    }
