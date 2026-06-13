import pandas as pd
import numpy as np
import pytest

from src.imbalance_handling import handle_imbalance


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_dataset():
    """
    Creates an intentionally imbalanced dataset:
    8 majority (0), 2 minority (1)
    """
    X = pd.DataFrame({
        "f1": range(10),
        "f2": range(10, 20)
    })
    y = pd.Series([0]*8 + [1]*2)
    return X, y


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_oversample_balances_to_majority():
    X, y = make_dataset()

    X_res, y_res = handle_imbalance(X, y, strategy="oversample")

    assert len(X_res) == len(y_res)

    counts = y_res.value_counts()
    assert counts[0] == counts[1]          # perfectly balanced
    assert counts[0] == 8                  # matches majority size


def test_undersample_balances_to_minority():
    X, y = make_dataset()

    X_res, y_res = handle_imbalance(X, y, strategy="undersample")

    counts = y_res.value_counts()

    assert counts[0] == counts[1]
    assert counts[0] == 2                  # matches minority size


def test_combined_balances_to_half_majority():
    X, y = make_dataset()

    X_res, y_res = handle_imbalance(X, y, strategy="combined")

    counts = y_res.value_counts()

    # majority = 8 → half = 4
    assert counts[0] == 4
    assert counts[1] == 4
    assert len(X_res) == 8


def test_target_not_in_features():
    X, y = make_dataset()

    X_res, y_res = handle_imbalance(X, y)

    assert "__target__" not in X_res.columns
    assert "f1" in X_res.columns
    assert "f2" in X_res.columns


def test_invalid_strategy_raises():
    X, y = make_dataset()

    with pytest.raises(ValueError):
        handle_imbalance(X, y, strategy="SMOTE")