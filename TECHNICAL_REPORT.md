# Fraud Detection: From Raw Transactions to Explainable, Production-Ready Models

*A technical report on building and hardening a fraud detection system across two very different transaction streams.*

## 1. The Business Problem

Fraud detection is a cost-balancing act, not an accuracy contest. Missing fraud (a false negative) costs real money; blocking a legitimate customer (a false positive) costs trust. Both datasets here are severely imbalanced — a model that predicts "legitimate" for every transaction would look 90%+ accurate on the e-commerce data and 99.8% accurate on the credit-card data while catching nothing. That's why every metric in this report leans on AUC-PR and F1 rather than raw accuracy.

## 2. Data Analysis & Preprocessing

**E-commerce (`Fraud_Data.csv`):** 151,112 rows, 11 columns, zero missing values, zero duplicate rows. Class balance: 136,961 legitimate (90.6%) vs. 14,151 fraud (9.4%).

**Credit card (`creditcard.csv`):** 284,807 rows, 31 columns (`Time`, 28 PCA-anonymized `V` features, `Amount`, `Class`), zero missing values, but **1,081 exact duplicate rows** — removed, leaving 283,726 rows. Class balance: 283,253 legitimate (99.83%) vs. 473 fraud (0.17%) — over 50x more imbalanced than the e-commerce set, which directly shaped the resampling decision in Section 4.

Both datasets needed correct dtype handling — `signup_time`/`purchase_time` parsed as datetimes for the e-commerce set, since every downstream time-based feature depends on that.

### Bivariate signal: country-level fraud rates (e-commerce)

Fraud rate by country isn't uniform — Canada (11.7%) and the UK (10.6%) run meaningfully above the US (9.6%) and Germany (7.2%) in the top-10-by-volume countries, which is the basis for the `country_fraud_risk` engineered feature (a smoothed target encoding, not a raw one-hot, to avoid overfitting to countries with few observations).

### Bivariate signal: PCA feature correlation (credit card)

Since the credit-card features are anonymized, direct business interpretation isn't possible — but a straightforward correlation-with-target check flags `V17`, `V14`, and `V12` as the strongest linear signals (correlation ≈0.31, 0.29, 0.25 respectively), which foreshadows the SHAP findings in Section 6: `V14` shows up as the top driver by both methods.

## 3. Geolocation Integration

`IpAddress_to_Country.csv` gives IP ranges, not exact IPs, so the join is a range-based lookup (`src/geolocation.py`): each transaction's IP is cast to an integer and matched against the `[lower_bound, upper_bound]` range containing it, rather than an exact-match join. IPs falling outside every known range are labeled "Unknown" rather than dropped — "Unknown" turned out to be the second-highest-volume "country" bucket (21,966 transactions), so dropping unmatched rows would have thrown away meaningful volume.

## 4. Feature Engineering & Resampling — and Why the Two Datasets Are Treated Differently

E-commerce features engineered in `src/feature_engineering.py`: `hour_of_day`, `day_of_week`, `time_since_signup` (clipped at zero to guard against clock skew), per-user transaction count and velocity, a 30-minute cross-account velocity window (catching multiple accounts transacting from the same browser/country in a tight window — a common fraud-ring signal that a single-user velocity feature would miss), and the smoothed country-risk encoding from Section 2.

**Resampling decision, and why it isn't the same for both datasets:**

- **E-commerce**: undersampled the majority class *on the training split only* (`src/imbalance_handling.py`), down to match the minority class. At a 90/10 split with 151k rows, there's still plenty of majority-class signal left after undersampling to ~25k balanced rows.
- **Credit card**: **no row-level resampling**. At 283,726 rows with only 473 fraud cases (0.17%), undersampling to match would throw away over 99% of legitimate transactions and badly distort the PCA feature distributions the model relies on. Instead, class imbalance is handled at the algorithm level: `class_weight="balanced"` for Logistic Regression, and a computed `scale_pos_weight` for XGBoost (`src/models/config.py`, `src/models/train_models.py`).

In both cases, resampling/reweighting touches training data only — test sets are never rebalanced, so evaluation metrics reflect real-world class proportions.

Every hyperparameter and threshold in this pipeline — the 30-minute velocity window, the smoothing weight in the country-risk encoding, the XGBoost tree depth and learning rate — lives in one place, `src/models/config.py`, as named dataclass fields rather than scattered magic numbers.

## 5. Model Building & Comparison

Logistic Regression (interpretable baseline) and XGBoost (main ensemble model) were trained per dataset, evaluated with AUC-PR, ROC-AUC, and F1 at a threshold swept for the best value (`notebooks/modeling.ipynb`, now reproducible via `src/models/train_models.py` + `evaluate_models.py`):

| Dataset | Model | AUC-PR | ROC-AUC | F1 (best threshold) | Confusion Matrix (TN, FP / FN, TP) |
|---|---|---|---|---|---|
| E-commerce | Logistic Regression | 0.623 | 0.768 | 0.690 | 27,393, 0 / 1,339, 1,491 |
| E-commerce | XGBoost | 0.618 | 0.760 | 0.690 | 27,393, 0 / 1,339, 1,491 |
| Credit Card | Logistic Regression | 0.705 | 0.966 | 0.398 | 56,428, 223 / 16, 79 |
| Credit Card | **XGBoost** | **0.825** | **0.979** | **0.874** | 56,648, 3 / 19, 76 |

**Model selection:** XGBoost is the clear winner on credit-card data — more than double the F1 of the linear baseline, with far fewer false alarms (3 vs. 223) at comparable recall. On e-commerce data the two models are statistically close; given that, Logistic Regression remains a defensible choice there specifically *because* it's natively interpretable, while XGBoost is chosen as the primary model overall since it dominates on the harder dataset and SHAP (Section 6) closes the interpretability gap it would otherwise have.

Stratified 5-fold cross-validation (`src/models/cross_validate.py`) is used to confirm this performance is stable across splits rather than an artifact of one particular train/test division, reporting mean and standard deviation of AUC-PR per fold.

## 6. Model Explainability (SHAP)

Built-in Gain importance and SHAP were run on the credit-card XGBoost model (`notebooks/shap_explainability.ipynb`; the same analysis is available for any trained model via `src/explainability.py`).

**Feature importance, by method:**

| Rank | Built-in Gain | SHAP mean \|value\| |
|---|---|---|
| 1 | V14 (0.542) | V14 (2.753) |
| 2 | V4 (0.061) | V4 (1.764) |
| 3 | V12 (0.041) | V12 (1.127) |
| 4 | V8 (0.033) | V10 (0.884) |
| 5 | V17 (0.026) | V11 (0.831) |

**Where the two methods agree and disagree:** both rank `V14` as the dominant feature by a wide margin — it alone accounts for over half the Gain-based importance. They diverge further down the list: Gain ranks `V17` in the top 5 (favoring features that produce clean splits near the root of individual trees), while SHAP ranks `V10` and `V11` higher (reflecting real marginal impact on individual predictions that Gain, a training-time-only statistic, doesn't capture). The practical takeaway: Gain tells you what the model leaned on structurally during training; SHAP tells you what actually moved each prediction, and the two aren't always the same features.

**Cohort sizes on the credit-card test set:** 74 true positives, 7 false positives, 21 false negatives, evaluated with individual SHAP force plots for one representative case of each.

## 7. Business Recommendations

Each recommendation below is tied to a specific SHAP finding from Section 6, not a generic best practice:

1. **Automated step-up authentication for extreme-`V14` transactions.** True-positive force plots consistently show a sharp drop in `V14` as the single strongest structural signal of real compromise. Rather than a hard auto-decline, route these to a secondary verification step (biometric or SMS OTP) — this keeps legitimate edge cases from being outright blocked while still stopping the fraud pattern immediately.
2. **A dynamic override for high-value transactions from established customers.** False-positive cases show legitimate high-`Amount` transactions being flagged largely because of the amount alone, overriding otherwise clean identity signals. For customers with a verified history of high-value spending, apply a dampening weight to the amount feature at inference time rather than treating every large transaction as equally suspicious.
3. **Context-aware decision thresholds during high-risk windows.** False negatives cluster around cases where `V14` and `V12` sit close to typical legitimate ranges — attackers structuring transactions to blend in. During historically higher-risk windows, lowering the decision threshold when secondary indicators (`V4`, `V11`) show even mild elevation catches more of these disguised cases without materially raising the false-alarm rate the rest of the time.

## 8. What's Next

The credit-card model has the deepest SHAP walkthrough here since it's the stronger-performing model; the same `src/explainability.py` module works unchanged on the e-commerce XGBoost model, and running it there — plus calibrating `src/business_impact.py`'s cost assumptions against the client's real chargeback and investigation costs — are the two most valuable next steps for turning this from a strong portfolio piece into an operational tool.
