"""
feature_cleanup.py
─────────────────────────────────────────────
Defensive cleanup applied right after loading any processed fraud
dataframe, before it touches a model.

Why this exists:
  train_final.csv / test_final.csv (as produced by the current upstream
  feature-engineering step) ship with two problems that silently degrade
  both LR and XGBoost without throwing any error:

  1. Exact duplicate columns (confirmed identical to ~1e-16 float noise):
       - transactions_per_hour  ≡  user_txn_velocity
       - account_age_days       ≡  time_since_signup
     Keeping both wastes model capacity, distorts feature-importance
     rankings, and inflates coefficient variance in Logistic Regression
     when two columns carry literally the same information.

  2. All-zero constant columns (verified nunique == 1 across every row):
       - user_txn_count
       - platform_30min_velocity
       - time_since_prev_txn
     These contribute nothing and only add noise to L1/L2 regularization
     weighting and clutter feature-importance plots.

  None of this is fraud-signal leakage — is_same_day / user_txn_velocity
  are genuine, very strong predictors (93.3% fraud rate when
  is_same_day=1 vs 4.5% otherwise), consistent with the known
  "fast signup-to-purchase" fraud pattern in this data. The fix here is
  purely about removing redundant/dead columns, not removing signal.

  This is applied here — at the modeling-script level — rather than only
  in the upstream feature-engineering script, so that:
    (a) train_models.py and cross_validate.py stay correct even if the
        upstream script regenerating train_final.csv/test_final.csv
        hasn't been patched yet,
    (b) the assumption is explicit and self-documenting at the point
        where it actually matters,
    (c) if the upstream script silently changes the duplicate/constant
        columns in the future, the safety checks below will tell you
        loudly instead of failing silently.

Import this in both train_models.py and cross_validate.py so the two
scripts can never drift apart on what "clean" means.
"""

import warnings

# Known-redundant columns confirmed in train_final.csv as of this writing.
# Drop the duplicate, KEEP the canonical name on each pair.
DUPLICATE_COLUMNS_TO_DROP = [
    "transactions_per_hour",   # ≡ user_txn_velocity (kept)
    "account_age_days",        # ≡ time_since_signup (kept)
]

# Known all-zero / zero-variance columns confirmed in train_final.csv.
CONSTANT_COLUMNS_TO_DROP = [
    "user_txn_count",
    "platform_30min_velocity",
    "time_since_prev_txn",
]

COLUMNS_TO_DROP = DUPLICATE_COLUMNS_TO_DROP + CONSTANT_COLUMNS_TO_DROP


def clean_fraud_features(df, verbose=True):
    """
    Drops known-redundant and known-constant columns from a fraud
    dataframe (train_final.csv / test_final.csv) and re-validates that
    the assumptions which justified the drop still hold. Safe to call on
    any dataframe with or without the target column present.

    If the data changes upstream and a "duplicate" pair is no longer
    actually identical, or a "constant" column now varies, this raises
    instead of silently dropping a column that may now carry signal.
    """
    df = df.copy()
    present_drops = [c for c in COLUMNS_TO_DROP if c in df.columns]

    # ── Re-validate duplicate pairs before dropping ────────────────────────
    pairs = [("transactions_per_hour", "user_txn_velocity"),
             ("account_age_days", "time_since_signup")]
    for dropped, kept in pairs:
        if dropped in df.columns and kept in df.columns:
            max_diff = (df[dropped] - df[kept]).abs().max()
            if max_diff > 1e-6:
                raise ValueError(
                    f"'{dropped}' was expected to be an exact duplicate of "
                    f"'{kept}' (max float diff <= 1e-6) but differs by "
                    f"{max_diff:.6g}. Refusing to drop it automatically — "
                    f"these may no longer be redundant. Inspect before "
                    f"removing from COLUMNS_TO_DROP / treating as duplicate."
                )

    # ── Re-validate constant columns before dropping ───────────────────────
    for col in CONSTANT_COLUMNS_TO_DROP:
        if col in df.columns and df[col].nunique() > 1:
            raise ValueError(
                f"'{col}' was expected to be constant (all-zero) but now "
                f"has {df[col].nunique()} unique values. Refusing to drop "
                f"it automatically — it may now carry real signal. "
                f"Inspect before removing from CONSTANT_COLUMNS_TO_DROP."
            )

    df = df.drop(columns=present_drops, errors="ignore")

    if verbose and present_drops:
        print(f"[feature_cleanup] Dropped {len(present_drops)} redundant/"
              f"constant columns: {present_drops}")
    elif verbose:
        print("[feature_cleanup] No known redundant/constant columns found "
              "to drop (already clean, or upstream schema changed).")

    # ── Catch-all: warn (don't fail) if any other zero-variance column
    #    slipped in that we don't yet know about. Doesn't drop it —
    #    just surfaces it so a human decides.
    feature_cols = [c for c in df.columns if c not in ("class", "Class")]
    zero_var = [c for c in feature_cols if df[c].nunique() <= 1]
    if zero_var:
        warnings.warn(
            f"[feature_cleanup] Found additional zero-variance column(s) "
            f"not in the known cleanup list: {zero_var}. These were left "
            f"in place but contribute no signal — consider adding them to "
            f"CONSTANT_COLUMNS_TO_DROP once confirmed.",
            stacklevel=2,
        )

    return df