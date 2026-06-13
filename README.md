# Task 1: Data Analysis and Preprocessing

## Overview

This project focuses on preparing fraud detection datasets for machine learning by performing data cleaning, exploratory data analysis (EDA), feature engineering, data transformation, geolocation enrichment, and class imbalance handling.

The goal is to produce clean, feature-rich datasets that are ready for model training and evaluation.

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
│       └── creditcard_cleaned.csv
│
├── logs/
│   └── pipeline.log
│
├── notebooks/
│   └── task1_eda.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── geolocation.py
│   ├── feature_engineering.py
│   ├── data_transformation.py
│   ├── imbalance_handling.py
│   └── pipeline.py
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_geolocation.py
│   ├── test_feature_engineering.py
│   ├── test_data_transformation.py
│   └── test_imbalance_handling.py
│
├── scripts/
├── .gitignore
├── requirements.txt
└── README.md
```

## Work Completed

### Data Cleaning

- Removed duplicate records.
- Handled missing values through appropriate validation and cleaning procedures.
- Corrected data types for timestamps, IP addresses, and target variables.
- Generated cleaned datasets for both Fraud_Data and CreditCard datasets.

### Exploratory Data Analysis (EDA)

- Analyzed distributions of key numerical and categorical features.
- Examined relationships between features and fraud labels.
- Quantified class imbalance in both datasets.
- Investigated fraud patterns across countries after geolocation enrichment.

### Geolocation Integration

- Converted IP addresses into integer format.
- Mapped transactions to countries using range-based IP lookups.
- Identified geographic fraud trends and country-level fraud statistics.

### Feature Engineering

Created additional fraud-detection features including:

- `hour_of_day`
- `day_of_week`
- `time_since_signup`
- `user_txn_count`
- `user_txn_velocity`
- `is_same_day`

These features capture temporal behavior, account activity, and transaction velocity patterns commonly associated with fraudulent activity.

### Data Transformation

- Applied one-hot encoding to categorical variables.
- Scaled numerical features using StandardScaler.
- Generated model-ready feature matrices for training and testing.

### Class Imbalance Handling

- Applied resampling techniques on the training set only.
- Documented class distributions before and after resampling.
- Preserved the original test distribution to ensure realistic model evaluation.

## Deliverables

- Cleaned datasets
- Geolocation-enriched datasets
- Feature-engineered datasets
- Train/Test splits
- Balanced training datasets
- EDA findings and visualizations
- Feature engineering documentation
- Resampling justification

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Pytest
- GitHub Actions
