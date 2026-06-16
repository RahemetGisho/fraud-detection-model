
# Handle Class Imbalance

import numpy as np
import pandas as pd
from sklearn.utils import resample

def _print_distribution(y: pd.Series, label: str) -> None:
    counts = y.value_counts().sort_index()
    pct = counts / len(y) * 100

    print(f"\n[{label}]")
    for cls in counts.index:
        print(f"  Class {int(cls)}: {counts[cls]:>8,} ({pct[cls]:.2f}%)")

def handle_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "undersample",   
    random_state: int = 42
) -> tuple:
    valid_strategies = ("oversample", "undersample", "combined")

    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from {valid_strategies}"
        )

    print(f"\n[imbalance_handling] Strategy: {strategy}")
    print("[imbalance_handling] Before resampling:")
    _print_distribution(y_train, "train")
    data = X_train.copy()
    data["__target__"] = y_train.values

    majority = data[data["__target__"] == 0]
    minority = data[data["__target__"] == 1]

    n_majority = len(majority)
    n_minority = len(minority)

    # 1. UNDERSAMPLING (RECOMMENDED)
    if strategy == "undersample":

        majority_down = resample(
            majority,
            replace=False,
            n_samples=n_minority,
            random_state=random_state,
        )

        balanced = pd.concat([majority_down, minority])

    # 2. OVERSAMPLING (optional)
    elif strategy == "oversample":

        minority_up = resample(
            minority,
            replace=True,
            n_samples=n_majority,
            random_state=random_state,
        )

        balanced = pd.concat([majority, minority_up])

    # 3. COMBINED (balanced medium size)
    elif strategy == "combined":

        target = n_majority // 2

        minority_up = resample(
            minority,
            replace=True,
            n_samples=target,
            random_state=random_state,
        )

        majority_down = resample(
            majority,
            replace=False,
            n_samples=target,
            random_state=random_state,
        )

        balanced = pd.concat([majority_down, minority_up])
    balanced = balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    y_res = balanced["__target__"].astype(int)
    X_res = balanced.drop(columns=["__target__"])

    print("[imbalance_handling] After resampling:")
    _print_distribution(y_res, "train resampled")

    print(f"\n[imbalance_handling] Size: {len(X_train):,} → {len(X_res):,}")

    return X_res, y_res