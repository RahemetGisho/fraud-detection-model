"""
src/imbalance_handling.py
=========================
Task 1 — Step 6: Handle Class Imbalance

Strategies available:
  'oversample'   — Random oversampling of the minority (fraud) class.
  'undersample'  — Random undersampling of the majority (legit) class.
  'combined'     — Oversample minority to 50% of majority size, then
                   undersample majority to that same size.
                   Produces a balanced set while keeping it smaller than
                   pure oversampling.

CRITICAL RULE:
  Resampling is applied ONLY to the TRAINING set.
  The test set is NEVER touched — it must reflect the real-world class
  distribution so evaluation metrics are not artificially inflated.

Justification for chosen strategy (oversample):
  ┌─────────────────────┬────────────────────────────────────────────────────┐
  │ Strategy            │ Pros / Cons                                        │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ Oversample          │ + No information loss (all legit rows kept).        │
  │ (chosen)            │ + Safe when minority class is small (<10%).         │
  │                     │ − Can overfit to duplicated minority rows.          │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ Undersample         │ + Fast, smaller training set.                       │
  │                     │ − Throws away real data; risky when legit rows      │
  │                     │   contain important variance.                       │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ SMOTE (synthetic)   │ + Creates new synthetic minority examples.          │
  │                     │ − Requires imbalanced-learn library.                │
  │                     │ − Can generate unrealistic synthetic samples in    │
  │                     │   sparse high-dimensional spaces.                   │
  └─────────────────────┴────────────────────────────────────────────────────┘

  For Fraud_Data (9.4% fraud):  oversample is appropriate.
  For CreditCard (0.17% fraud): combined is recommended to avoid an
  extremely large training set while still giving the model enough fraud
  examples.
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample


def _print_distribution(y: pd.Series, label: str) -> None:
    """Helper — print class counts and percentages."""
    counts = y.value_counts().sort_index()
    pct    = counts / len(y) * 100
    print(f"  [{label}]")
    for cls in counts.index:
        print(f"    Class {int(cls)}: {counts[cls]:>8,}  ({pct[cls]:5.2f}%)")


def handle_imbalance(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     strategy: str = "oversample",
                     random_state: int = 42) -> tuple:
    """
    Rebalance the training set using the chosen strategy.

    Parameters
    ----------
    X_train      : feature matrix (training split only).
    y_train      : target series (training split only).
    strategy     : 'oversample' | 'undersample' | 'combined'.
    random_state : for reproducibility.

    Returns
    -------
    X_resampled, y_resampled : balanced DataFrame and Series.
    """
    valid_strategies = ("oversample", "undersample", "combined")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {valid_strategies}"
        )

    print(f"\n[imbalance_handling] Strategy : {strategy}")
    print(f"[imbalance_handling] Before resampling:")
    _print_distribution(y_train, "train")

    # ── Assemble one DataFrame for easy splitting ─────────────────────────
    combined = X_train.copy()
    combined["__target__"] = y_train.values

    majority = combined[combined["__target__"] == 0]
    minority = combined[combined["__target__"] == 1]

    n_majority = len(majority)
    n_minority = len(minority)

    # ── Apply chosen strategy ─────────────────────────────────────────────
    if strategy == "oversample":
        # Duplicate minority rows until count matches majority
        minority_up = resample(
            minority,
            replace=True,
            n_samples=n_majority,
            random_state=random_state,
        )
        balanced = pd.concat([majority, minority_up])

    elif strategy == "undersample":
        # Remove majority rows until count matches minority
        majority_down = resample(
            majority,
            replace=False,
            n_samples=n_minority,
            random_state=random_state,
        )
        balanced = pd.concat([majority_down, minority])

    elif strategy == "combined":
        # Step 1: oversample minority to half the majority size
        target_n = n_majority // 2
        minority_up = resample(
            minority,
            replace=True,
            n_samples=target_n,
            random_state=random_state,
        )
        # Step 2: undersample majority to the same target size
        majority_down = resample(
            majority,
            replace=False,
            n_samples=target_n,
            random_state=random_state,
        )
        balanced = pd.concat([majority_down, minority_up])

    # Shuffle to avoid any ordering bias
    balanced = balanced.sample(frac=1, random_state=random_state)
    balanced.reset_index(drop=True, inplace=True)

    y_res = balanced["__target__"].astype(int)
    X_res = balanced.drop(columns=["__target__"])

    print(f"[imbalance_handling] After resampling:")
    _print_distribution(y_res, "train resampled")
    print(f"[imbalance_handling] Training set size: "
          f"{len(X_train):,} → {len(X_res):,}")

    return X_res, y_res