import pytest

from src.business_impact import business_impact_from_confusion_matrix, CostAssumptions


def test_perfect_model_has_zero_missed_fraud_cost():
    """No false negatives means no cost attributed to missed fraud."""
    cm = [[90, 0], [0, 10]]
    result = business_impact_from_confusion_matrix(cm, CostAssumptions(avg_fraud_loss_if_missed=100.0))
    assert result["cost_of_missed_fraud"] == 0.0
    assert result["fraud_caught_rate"] == pytest.approx(1.0)


def test_all_fraud_missed_costs_full_baseline():
    """If every fraud case is a false negative, the model provides zero savings."""
    cm = [[90, 0], [10, 0]]
    result = business_impact_from_confusion_matrix(cm, CostAssumptions(avg_fraud_loss_if_missed=100.0))
    assert result["cost_of_missed_fraud"] == 1000.0
    assert result["estimated_savings_vs_no_model"] == 0.0
    assert result["fraud_caught_rate"] == 0.0


def test_false_alarms_cost_the_investigation_rate():
    """False positives should cost exactly fp * investigation cost, independent of fraud loss."""
    cm = [[80, 20], [0, 0]]
    result = business_impact_from_confusion_matrix(
        cm, CostAssumptions(avg_fraud_loss_if_missed=100.0, avg_investigation_cost_per_false_alarm=5.0)
    )
    assert result["cost_of_false_alarms"] == 100.0


def test_savings_reflect_partial_detection():
    """A model that catches half the fraud should show savings roughly
    equal to half the baseline cost, minus its false-alarm overhead."""
    cm = [[95, 5], [5, 5]]
    result = business_impact_from_confusion_matrix(
        cm, CostAssumptions(avg_fraud_loss_if_missed=100.0, avg_investigation_cost_per_false_alarm=5.0)
    )
    assert result["baseline_cost_if_no_model"] == 1000.0
    assert result["cost_of_missed_fraud"] == 500.0
    assert result["cost_of_false_alarms"] == 25.0
    assert result["estimated_savings_vs_no_model"] == pytest.approx(1000.0 - 525.0)


def test_no_fraud_cases_gives_zero_caught_rate_without_error():
    """Edge case: a test set with no positive class shouldn't divide by zero."""
    cm = [[100, 0], [0, 0]]
    result = business_impact_from_confusion_matrix(cm)
    assert result["fraud_caught_rate"] == 0.0
