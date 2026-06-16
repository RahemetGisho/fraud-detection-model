import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_transformation import transform_fraud_data, transform_creditcard

class TestDataTransformation(unittest.TestCase):

    def setUp(self):
        """Construct deterministic training and test datasets for isolation tests."""
        # Simulated raw, post-feature engineered Fraud training records
        self.mock_fraud_train = pd.DataFrame({
            "user_id": [1, 2, 3],
            "device_id": ["A", "B", "C"],
            "signup_time": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "purchase_time": ["2026-01-02", "2026-01-03", "2026-01-04"],
            "ip_address": [12345, 67890, 11223],
            "country": ["USA", "Canada", "UK"],
            "source": ["SEO", "Ads", "SEO"],
            "browser": ["Chrome", "Safari", "Chrome"],
            "sex": ["M", "F", "M"],
            "purchase_value": [10.0, 20.0, 30.0],
            "age": [30, 40, 50],
            "hour_of_day": [12, 14, 16],
            "day_of_week": [1, 2, 3],
            "time_since_signup": [86400, 86400, 86400],
            "user_txn_count": [1, 2, 1],
            "user_txn_velocity": [1.0, 2.0, 1.0],
            "account_age_days": [1.0, 1.0, 1.0],
            "transactions_per_hour": [0.04, 0.08, 0.04],
            "avg_purchase_value": [10.0, 20.0, 30.0],
            "purchase_deviation": [1.0, 1.0, 1.0],
            "time_since_prev_txn": [999999, 999999, 999999],
            "country_fraud_risk": [0.05, 0.02, 0.01],
            "platform_30min_velocity": [1, 1, 1],
            "cohort_purchase_deviation": [1.0, 1.0, 1.0],
            "class": [0, 1, 0]  # Target column
        })

        # Test set scenario containing an Out-of-Vocabulary (OOV) browser category ("Firefox")
        # and completely missing one category option entirely ("F" under sex)
        self.mock_fraud_test = pd.DataFrame({
            "user_id": [4],
            "device_id": ["D"],
            "signup_time": ["2026-01-04"],
            "purchase_time": ["2026-01-05"],
            "ip_address": [44556],
            "country": ["Germany"],
            "source": ["Ads"],
            "browser": ["Firefox"],  # New Category (OOV)
            "sex": ["M"],            # Missing 'F' representation entirely
            "purchase_value": [15.0],
            "age": [35],
            "hour_of_day": [13],
            "day_of_week": [2],
            "time_since_signup": [86400],
            "user_txn_count": [1],
            "user_txn_velocity": [1.0],
            "account_age_days": [1.0],
            "transactions_per_hour": [0.04],
            "avg_purchase_value": [15.0],
            "purchase_deviation": [1.0],
            "time_since_prev_txn": [999999],
            "country_fraud_risk": [0.03],
            "platform_30min_velocity": [1],
            "cohort_purchase_deviation": [1.0],
            "class": [1]
        })

        # Simplified baseline parameters for CreditCard evaluation
        self.mock_cc_train = pd.DataFrame({
            "Time": [0.0, 3600.0, 7200.0],
            "V1": [-1.35, 1.19, -0.96],
            "Amount": [149.62, 4.99, 378.66],
            "Class": [0, 0, 1]
        })

    def test_fraud_train_transformation_fit(self):
        """Verify the training pipeline extracts labels, fits weights, and encodes columns."""
        X_train, y_train, scaler, train_cols = transform_fraud_data(
            self.mock_fraud_train, fit=True
        )

        # 1. Evaluate label isolation
        self.assertEqual(len(y_train), 3)
        self.assertNotIn("class", X_train.columns)
        self.assertTrue((y_train.values == np.array([0, 1, 0])).all())

        # 2. Check column drop structural compliance
        dropped_cols = ["user_id", "device_id", "signup_time", "purchase_time", "ip_address", "country"]
        for col in dropped_cols:
            self.assertNotIn(col, X_train.columns)

        # 3. Check scaling performance (StandardScaler sets mean to ~0)
        self.assertIsInstance(scaler, StandardScaler)
        self.assertAlmostEqual(X_train["purchase_value"].mean(), 0.0, places=7)

        # 4. Verify categorical dummy extraction transformed boolean variables to ints
        self.assertIn("source_SEO", X_train.columns)
        self.assertIn("browser_Chrome", X_train.columns)
        self.assertIn("sex_M", X_train.columns)
        self.assertEqual(X_train["source_SEO"].dtype, np.int64)

        # 5. Confirm structural tracking outputs match
        self.assertEqual(X_train.shape[1], len(train_cols))

    def test_fraud_test_transformation_alignment(self):
        """Confirm inference transforms respect training schema bounds and isolate OOV variants."""
        # Step 1: Run standard training transformation pass to collect configuration context
        X_train, _, scaler, train_cols = transform_fraud_data(self.mock_fraud_train, fit=True)

        # Step 2: Transform test payload utilizing training references
        X_test, y_test, _, _ = transform_fraud_data(
            self.mock_fraud_test, scaler=scaler, fit=False, train_columns=train_cols
        )

        # 1. Dimensions must match perfectly to accommodate strict feature matrix interfaces
        self.assertEqual(list(X_test.columns), train_cols)

        # 2. Assert OOV Browser variants ("Firefox") were safely discarded during reindexing alignment
        self.assertNotIn("browser_Firefox", X_test.columns)

        # 3. Assert missing training components ("sex_F") fill with 0 instead of causing a KeyError
        self.assertIn("sex_F", X_test.columns)
        self.assertEqual(X_test["sex_F"].iloc[0], 0)

        # 4. Assert numerical metrics are scaled appropriately using training properties
        # Expected value formula: (Value - Train_Mean) / Train_SD
        # Train values for purchase_value: [10, 20, 30] -> Mean = 20, Sample SD = 8.1649658
        train_mean = 20.0
        train_std = np.std([10.0, 20.0, 30.0])  # population standard deviation used by sklearn
        expected_scaled_val = (15.0 - train_mean) / train_std
        self.assertAlmostEqual(X_test["purchase_value"].iloc[0], expected_scaled_val, places=5)

    def test_fraud_missing_arguments_exceptions(self):
        """Ensure missing scalers or validation tracking keys trigger ValueErrors during validation."""
        # Attempting to transform a test set without providing a scaler must fail
        with self.assertRaises(ValueError):
            transform_fraud_data(self.mock_fraud_test, scaler=None, fit=False, train_columns=["dummy"])

        # Attempting to transform a test set without training column states must fail
        scaler = StandardScaler()
        with self.assertRaises(ValueError):
            transform_fraud_data(self.mock_fraud_test, scaler=scaler, fit=False, train_columns=None)

    def test_creditcard_fit_and_transform(self):
        """Validate the CreditCard pipeline isolates labels and scales only specified variables."""
        # 1. Training Pass
        X_train, y_train, scaler, = transform_creditcard(self.mock_cc_train, fit=True)

        self.assertNotIn("Class", X_train.columns)
        self.assertEqual(len(y_train), 3)
        self.assertIsInstance(scaler, StandardScaler)
        
        # Unscaled attributes must retain their original input variance structure
        self.assertEqual(X_train["V1"].iloc[0], -1.35)
        # Scaled variables should align to a mean of ~0
        self.assertAlmostEqual(X_train["Amount"].mean(), 0.0, places=7)

        # 2. Testing Pass
        mock_cc_test = pd.DataFrame({
            "Time": [3600.0],
            "V1": [0.5],
            "Amount": [4.99],
            "Class": [0]
        })
        
        X_test, y_test, _ = transform_creditcard(mock_cc_test, scaler=scaler, fit=False)
        # Amount = 4.99 matches the middle row of the training data.
        # Its scaled amount value should match the training data row index 1 precisely.
        self.assertAlmostEqual(X_test["Amount"].iloc[0], X_train["Amount"].iloc[1], places=5)

    def test_creditcard_missing_scaler_exception(self):
        """Ensure creditcard test adjustments fail gracefully when configuration dependencies are missing."""
        with self.assertRaises(ValueError):
            transform_creditcard(self.mock_cc_train, scaler=None, fit=False)

if __name__ == "__main__":
    unittest.main()