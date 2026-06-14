# Fraud Detection Model Project

## Overview

This project is a complete end-to-end fraud detection system covering:

- Data analysis and preprocessing (Task 1)
- Model training and evaluation (Task 2)
- Cross-validation and model comparison
- Feature engineering and imbalance handling
- Production-ready ML pipeline structure

The goal is to build a robust, explainable, and high-performing fraud detection system using both classical ML and ensemble methods.

---

## Project Structure

```text
fraud-detection-model/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   └── creditcard.csv
│   │
│   └── processed/
│       ├── fraud_cleaned.csv
│       ├── fraud_geo.csv
│       ├── fraud_engineered.csv
│       ├── fraud_train.csv
│       ├── fraud_test.csv
│       ├── fraud_train_balanced.csv
│       ├── creditcard_cleaned.csv
│       └── model_ready_features/
│
├── logs/
│   └── pipeline.log
│
├── notebooks/
│   ├── eda-creditcard
│   └── eda-fraud-data.ipynb
│   └── feature-engineering.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── geolocation.py
│   ├── feature_engineering.py
│   ├── data_transformation.py
│   ├── imbalance_handling.py
│   └── models/
│        └── evaluation.py
│        └── cross_validate.py
│        └── train_models.py
│
├── models/
│   ├── logistic_regression/
│   └── xgboost/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_geolocation.py
│   ├── test_feature_engineering.py
│   ├── test_data_transformation.py
│   ├── test_imbalance_handling.py
│   └── test_models.py
│
├── scripts/
│
├── .gitignore
├── requirements.txt
└── README.md

# Work Completed

## Task 1: Data Analysis and Preprocessing

### Data Cleaning
- Removed duplicate records
- Handled missing values
- Fixed incorrect data types
- Standardized target labels

### Exploratory Data Analysis (EDA)
- Distribution analysis of numerical and categorical features
- Fraud vs non-fraud imbalance analysis
- Time-based fraud pattern discovery
- Country-level fraud behavior (geolocation enriched)

### Geolocation Integration
- Converted IP addresses to integer format
- Mapped IP ranges to countries
- Identified high-risk regions

### Feature Engineering
Created behavioral fraud indicators:

- hour_of_day
- day_of_week
- time_since_signup
- transaction_count
- transactions_last_24h
- user_velocity_features
- is_same_day_transaction

### Data Transformation
- One-hot encoding for categorical variables
- Feature scaling using StandardScaler
- Final model-ready datasets created

### Class Imbalance Handling
- Applied resampling ONLY on training data
- Preserved original test distribution for realistic evaluation
- Documented class imbalance ratios

---

## Task 2: Model Training and Evaluation

### Baseline Models
- Logistic Regression trained with `class_weight="balanced"`
- Threshold tuning for optimal F1-score
- Interpretable baseline for benchmarking

### Ensemble Models
- XGBoost classifier for both datasets
- Tuned hyperparameters:
  - max_depth
  - learning_rate
  - subsample
  - colsample_bytree
- Used `scale_pos_weight` instead of SMOTE
- Improved handling of extreme imbalance

### Evaluation Metrics
- ROC-AUC
- AUC-PR (primary metric)
- F1-score
- Confusion matrix
- Precision-Recall curves

### Cross-Validation
- Stratified K-Fold (k=5)
- Evaluated using:
  - F1-score
  - ROC-AUC
  - AUC-PR
- Ensured no data leakage

### Model Comparison
- Logistic Regression vs XGBoost
- Selected best model based on AUC-PR
- XGBoost selected as final model

### Interpretability
- SHAP analysis for feature importance
- Global and local explanations for fraud predictions
```
