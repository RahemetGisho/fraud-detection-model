# Fraud Detection Model Project

## Overview

This project is an end-to-end fraud detection system that covers the full machine learning lifecycle—from raw data to explainable model insights. It combines classical machine learning and ensemble methods to build a robust, high-performance, and interpretable fraud detection solution.

The pipeline emphasizes real-world constraints such as class imbalance, feature engineering, and model explainability for business decision-making.

---

## Project Structure

```text
fraud-detection-model/
│
├── data/
│   ├── raw/                     # Original datasets
│   └── processed/              # Cleaned & feature-engineered data
│
├── notebooks/                  # EDA, modeling, SHAP analysis
├── src/                        # Modular ML pipeline code
├── models/                     # Saved Logistic Regression & XGBoost models
├── tests/                      # Unit tests for pipeline components
├── logs/                       # Training & pipeline logs
├── scripts/                    # Utility scripts
│
├── .github/workflows/         # CI pipeline
├── requirements.txt
└── README.md
```

## Work Completed

### Task 1: Data Analysis and Preprocessing

The dataset was cleaned, explored, and transformed into a model-ready format.

- Handled missing values, duplicates, and data type inconsistencies  
- Performed exploratory data analysis (EDA) on fraud patterns and class imbalance  
- Integrated geolocation data using IP-to-country mapping  
- Engineered behavioral features (transaction velocity, time-based patterns, user activity signals)  
- Applied encoding and feature scaling for model readiness  
- Addressed class imbalance using training-only resampling strategies  

---

### Task 2: Model Building and Evaluation

Multiple models were trained and evaluated under imbalanced classification settings.

- Built Logistic Regression as a baseline interpretable model  
- Trained XGBoost as the main ensemble model with hyperparameter tuning  
- Used ROC-AUC, AUC-PR, F1-score, and confusion matrix for evaluation  
- Applied stratified 5-fold cross-validation for robust performance estimation  
- Selected best model based on AUC-PR and recall-performance tradeoffs  

---

### Task 3: Model Explainability (SHAP)

Model interpretability was introduced to understand fraud prediction behavior.

- Extracted built-in feature importance from ensemble models  
- Generated SHAP summary plots for global feature impact analysis  
- Built SHAP force plots for individual predictions:
  - True Positive (correct fraud detection)  
  - False Positive (false alarm)  
  - False Negative (missed fraud)  
- Compared SHAP explanations with model-based feature importance  
- Identified key drivers influencing fraud predictions  
- Translated insights into actionable business recommendations  

---

### Key Results

- Best Model: XGBoost Classifier  
- Strong imbalance handling using `scale_pos_weight`  
- High ROC-AUC and AUC-PR across both datasets  
- Interpretable fraud detection pipeline using SHAP  

---

### Outcome

This project delivers a production-ready fraud detection system that accurate and explainable, enabling stakeholders to understand model decisions and apply insights to real-world fraud prevention strategies.
