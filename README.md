# Fraud Detection Model

![CI](https://github.com/RahemetGisho/fraud-detection-model/actions/workflows/ci.yml/badge.svg)

An end-to-end, production-oriented fraud detection system for a FinTech client (Adey Innovations Inc.), covering two transaction streams: e-commerce purchases and bank credit-card transactions.

## Business Problem

Fraud detection lives on a cost trade-off: **missed fraud (false negatives)** costs the business direct financial loss, while **false alarms (false positives)** frustrate legitimate customers and erode trust. Both datasets here are heavily imbalanced — fraud is 9.4% of e-commerce transactions and just 0.17% of credit-card transactions — so a naive model that predicts "not fraud" for everything would score >99% accuracy while catching zero fraud. The real objective is maximizing recall on the minority class without making false alarms unmanageable, and doing it in a way a compliance or risk team can actually explain.

## Solution Overview

- **Data pipeline**: leakage-safe preprocessing — the train/test split happens _before_ any behavioral feature is calculated, so no future information leaks into training rows (`scripts/pipeline.py`).
- **Geolocation enrichment**: range-based IP-to-country lookup joined onto the e-commerce transactions.
- **Feature engineering**: signup-to-purchase timing, hour-of-day/day-of-week, per-user transaction velocity, and a 30-minute cross-account velocity window (`src/feature_engineering.py`).
- **Imbalance handling**: undersampling on the e-commerce training split only; cost-sensitive class weighting (`scale_pos_weight`, `class_weight="balanced"`) for the credit-card model, since undersampling ~285k rows down to ~1k would destroy the PCA feature distribution (`src/imbalance_handling.py`, `src/models/config.py`).
- **Models**: Logistic Regression baseline + tuned XGBoost per dataset, evaluated with AUC-PR, ROC-AUC, and threshold-tuned F1 (`src/models/train_models.py`, `src/models/evaluate_models.py`).
- **Explainability**: SHAP global summary + local force plots for true-positive/false-positive/false-negative cases, compared against built-in Gain importance (`src/explainability.py`).
- **Dashboard**: Streamlit app for exploring metrics, individual predictions, SHAP explanations, and business-impact dollar estimates (`dashboard/app.py`).

## Key Results

From the trained models (`notebooks/modeling.ipynb`):

| Dataset                 | Model               | AUC-PR    | ROC-AUC   | F1 (best threshold) |
| ----------------------- | ------------------- | --------- | --------- | ------------------- |
| E-commerce (Fraud_Data) | Logistic Regression | 0.623     | 0.768     | 0.690               |
| E-commerce (Fraud_Data) | XGBoost             | 0.618     | 0.760     | 0.690               |
| Credit Card             | Logistic Regression | 0.705     | 0.966     | 0.398               |
| Credit Card             | **XGBoost**         | **0.825** | **0.979** | **0.874**           |

- On credit-card transactions, XGBoost catches fraud at **87% F1** vs. 40% for the linear baseline — the clearest case for the extra model complexity.
- On e-commerce transactions, the two models are statistically close (0.618–0.623 AUC-PR); Logistic Regression is a legitimate choice there if interpretability matters more than a marginal AUC-PR gain.
- Full metrics, confusion matrices, and classification reports are reproduced by running the pipeline — see `reports/evaluation_report.json` after training.

## Quick Start

```bash
git clone https://github.com/RahemetGisho/fraud-detection-model.git
cd fraud-detection-model
pip install -r requirements.txt

# Place the three raw CSVs under data/raw/:
#   Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv

python main.py                 # runs pipeline -> train -> cross-validate -> evaluate
streamlit run dashboard/app.py # explore results interactively
pytest -v                      # 52 passing tests
```

## Project Structure

```text
fraud-detection-model/
├── .github/workflows/ci.yml    # CI: pytest + import smoke test on every push
├── dashboard/
│   └── app.py                  # Streamlit dashboard (metrics, predictions, SHAP, $ impact)
├── data/                       # gitignored — raw/ and processed/
├── notebooks/                  # EDA, modeling, SHAP analysis (exploratory record)
├── src/
│   ├── data_loader.py          # load + structural validation for all 3 raw files
│   ├── preprocessing.py        # cleaning
│   ├── geolocation.py          # IP -> country range lookup
│   ├── feature_engineering.py  # velocity, time, behavioral features
│   ├── data_transformation.py  # scaling + one-hot encoding, train/test-safe
│   ├── imbalance_handling.py   # undersample / oversample / combined
│   ├── explainability.py       # SHAP + built-in importance, reusable
│   ├── business_impact.py      # confusion matrix -> dollar impact
│   └── models/
│       ├── config.py           # dataclass-based config, no magic numbers
│       ├── train_models.py
│       ├── cross_validate.py
│       └── evaluate_models.py
├── scripts/pipeline.py         # orchestrates load -> clean -> split -> engineer -> transform -> resample
├── main.py                     # runs the full chain end-to-end
├── tests/                      # 52 unit tests
├── reports/                    # saved plots + JSON evaluation/CV reports
├── requirements.txt
└── README.md
```

## Demo

Run `streamlit run dashboard/app.py` after training to get four tabs: Performance Metrics, Prediction Explorer, Explainability (SHAP), and Business Impact (adjustable cost-per-missed-fraud and cost-per-false-alarm sliders).

## Technical Details

- **Data**: `Fraud_Data.csv` (151,112 rows, 9.4% fraud) + `IpAddress_to_Country.csv` for the e-commerce stream; `creditcard.csv` (284,807 rows, 0.17% fraud, 28 PCA-anonymized features) for the bank stream. No missing values in either raw dataset; the credit-card file had 1,081 exact duplicate rows, removed.
- **Model**: Logistic Regression (`class_weight="balanced"`, `max_iter=2000`) as an interpretable baseline; XGBoost (`n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=<computed per split>`) as the main ensemble model — see `src/models/config.py` for every value with justification in-line.
- **Evaluation**: AUC-PR (primary metric for severe imbalance — ROC-AUC is optimistic here since the negative class dominates), ROC-AUC, F1 at a threshold swept for the best value, confusion matrix, and stratified 5-fold cross-validation for stability.

## Future Improvements

- Wire the fraud (e-commerce) model into the same SHAP force-plot walkthrough currently run in depth for the credit-card model, so both datasets get matched local explainability writeups.
- Replace the illustrative `CostAssumptions` defaults in `src/business_impact.py` with the client's real chargeback and investigation-labor costs.
- Add model monitoring for feature drift, since e-commerce behavioral patterns (velocity, country risk) will shift over time in a way PCA credit-card features may not.

## Author

Rahmet Hussen
`gishorahemeth@gmail.com`
`linkedin.com/in/rahemethussen/`
