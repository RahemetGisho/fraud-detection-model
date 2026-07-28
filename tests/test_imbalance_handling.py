import pytest
import numpy as np
import pandas as pd

from src.imbalance_handling import handle_imbalance, _print_distribution

@pytest.fixture
def imbalanced_mock_data():
    np.random.seed(42)    
    X = pd.DataFrame({
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randint(0, 100, size=100)
    })
    
    y = pd.Series([0] * 90 + [1] * 10) 
    return X, y

# STRATEGY TESTS
def test_handle_imbalance_undersample(imbalanced_mock_data):
    """Verifies majority class is cut down to match minority class size exactly."""
    X_train, y_train = imbalanced_mock_data
    X_res, y_res = handle_imbalance(X_train, y_train, strategy="undersample", random_state=42)
    
    assert len(X_res) == 20
    assert len(y_res) == 20
    
    assert (y_res == 0).sum() == 10
    assert (y_res == 1).sum() == 10
    
    assert list(X_res.columns) == list(X_train.columns)

def test_handle_imbalance_oversample(imbalanced_mock_data):
    """Verifies minority class is duplicated to match majority class size exactly."""
    X_train, y_train = imbalanced_mock_data
    
    X_res, y_res = handle_imbalance(X_train, y_train, strategy="oversample", random_state=42)
    assert len(X_res) == 180
    assert len(y_res) == 180
    
    assert (y_res == 0).sum() == 90
    assert (y_res == 1).sum() == 90

def test_handle_imbalance_combined(imbalanced_mock_data):
    """Verifies both classes meet in the middle at target = n_majority // 2."""
    X_train, y_train = imbalanced_mock_data
    
    X_res, y_res = handle_imbalance(X_train, y_train, strategy="combined", random_state=42)
    assert len(X_res) == 90
    assert len(y_res) == 90
    
    assert (y_res == 0).sum() == 45
    assert (y_res == 1).sum() == 45

# EDGE CASES & VALIDATIONS
def test_handle_imbalance_invalid_strategy_error(imbalanced_mock_data):
    """Verifies passing an unconfigured option raises an explicit ValueError."""
    X_train, y_train = imbalanced_mock_data
    with pytest.raises(ValueError, match="Unknown strategy 'invalid_strategy'"):
        handle_imbalance(X_train, y_train, strategy="invalid_strategy")

def test_print_distribution_execution(capsys):
    """Verifies distribution text log calculations output cleanly without throwing exceptions."""
    y_test = pd.Series([0, 0, 1])
    
    _print_distribution(y_test, "Test Baseline")
    captured = capsys.readouterr()
    assert "[Test Baseline]" in captured.out
    assert "Class 0" in captured.out
    assert "Class 1" in captured.out