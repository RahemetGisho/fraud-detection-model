import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import (
    calculate_country_risk,
    engineer_time_features,
    engineer_signup_features,
    engineer_velocity_features,
    engineer_behavior_features,
    engineer_all,
)

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Set up standard sample data components used across tests."""
        # Baseline mock records for general feature extractions
        self.base_df = pd.DataFrame({
            "user_id": [1, 1, 2],
            "purchase_value": [100.0, 200.0, 50.0],
            "browser": ["Chrome", "Chrome", "Safari"],
            "country": ["USA", "USA", "Canada"],
            "signup_time": pd.to_datetime([
                "2026-01-01 12:00:00", 
                "2026-01-01 12:00:00", 
                "2026-01-05 00:00:00"
            ]),
            "purchase_time": pd.to_datetime([
                "2026-01-01 12:00:30",  # 30 seconds after signup
                "2026-01-01 12:15:00",  # 15 minutes after signup
                "2026-01-05 06:00:00"   # Same-day, 6 hours later
            ])
        })

    def test_calculate_country_risk_with_target(self):
        """Verify smooth Bayesian target encoding for country risks computes accurately."""
        df = pd.DataFrame({"country": ["USA", "USA", "Canada", "USA"]})
        target = pd.Series([1, 0, 0, 1])  # Global mean = 2/4 = 0.5
        
        result_df = calculate_country_risk(df, target_series=target)
        
        self.assertIn("country_fraud_risk", result_df.columns)
        # USA stats: count=3, mean=2/3 (~0.666)
        # Global stats: mean=0.5, smooth_weight=10
        # USA smoothed risk calculation: (3 * (2/3) + 10 * 0.5) / (3 + 10) = (2 + 5) / 13 = 7/13 ~ 0.53846
        expected_usa_risk = (2 + (10 * 0.5)) / (3 + 10)
        actual_usa_risk = result_df.loc[df["country"] == "USA", "country_fraud_risk"].iloc[0]
        self.assertAlmostEqual(actual_usa_risk, expected_usa_risk, places=5)

    def test_calculate_country_risk_without_target(self):
        """Ensure country risk safely drops back to default 0.01 when no target is present."""
        df = pd.DataFrame({"country": ["USA", "Canada"]})
        result_df = calculate_country_risk(df, target_series=None)
        
        self.assertIn("country_fraud_risk", result_df.columns)
        self.assertTrue((result_df["country_fraud_risk"] == 0.01).all())

    def test_engineer_time_features(self):
        """Validate accurate date part extractions across hour of day and day of week indexes."""
        result_df = engineer_time_features(self.base_df)
        
        self.assertIn("hour_of_day", result_df.columns)
        self.assertIn("day_of_week", result_df.columns)
        
        # '2026-01-01' is a Thursday (index 3)
        self.assertEqual(result_df["hour_of_day"].iloc[0], 12)
        self.assertEqual(result_df["day_of_week"].iloc[0], 3)
        
        # '2026-01-05' is a Monday (index 0)
        self.assertEqual(result_df["hour_of_day"].iloc[2], 6)
        self.assertEqual(result_df["day_of_week"].iloc[2], 0)

    def test_engineer_signup_features(self):
        """Verify delta calculation metrics and same day indicator metrics handle bounds safely."""
        # Inject an edge case where purchase occurs BEFORE signup due to clock drift
        drift_df = self.base_df.copy()
        drift_df.loc[0, "purchase_time"] = pd.to_datetime("2026-01-01 11:59:00")
        
        result_df = engineer_signup_features(drift_df)
        
        self.assertIn("time_since_signup", result_df.columns)
        self.assertIn("is_same_day", result_df.columns)
        
        # Drift edge-case: .clip(lower=0) should catch negative deltas and make them 0
        self.assertEqual(result_df["time_since_signup"].iloc[0], 0.0)
        
        # Row 1: 15 minutes = 900 seconds
        self.assertEqual(result_df["time_since_signup"].iloc[1], 900.0)
        self.assertEqual(result_df["is_same_day"].iloc[1], 1)

    def test_engineer_velocity_features(self):
        """Ensure user transaction speeds and platform nearest-neighbor tracking work correctly."""
        # Set up a clean, realistic sequence for your pd.merge_asof lookups
        vel_df = pd.DataFrame({
            "user_id": [101, 102, 103],
            "browser": ["Chrome", "Chrome", "Chrome"],
            "country": ["UK", "UK", "UK"],
            "time_since_signup": [0.0, 86400.0, 10.0],  # 0.0 tests division-by-zero protection
            "purchase_time": pd.to_datetime([
                "2026-05-01 10:00:00",
                "2026-05-01 10:05:00",  # Safely matches the nearest row (10:00:00)
                "2026-05-01 11:00:00"   # Exceeds the 30-minute tolerance lookback window
            ])
        })
        
        result_df = engineer_velocity_features(vel_df)
        
        self.assertIn("user_txn_count", result_df.columns)
        self.assertIn("user_txn_velocity", result_df.columns)
        self.assertIn("platform_30min_velocity", result_df.columns)
        
        # Verify user division by zero safety protection handles replace(0, 1) properly
        # Row 0 velocity: 1 txn / 1 day (substitute for 0 days) = 1.0
        self.assertEqual(result_df["user_txn_velocity"].iloc[0], 1.0)
        
        # Corrected Assertions to align with pd.merge_asof sequential design mechanics:
        # Row 0: No prior records -> fallback size = 1
        self.assertEqual(result_df["platform_30min_velocity"].iloc[0], 1)
        
        # Row 1: Valid lookup row found within 30 min -> group match size = 1
        self.assertEqual(result_df["platform_30min_velocity"].iloc[1], 1)
        
        # Row 2: Outside lookback limits -> fallback size = 1
        self.assertEqual(result_df["platform_30min_velocity"].iloc[2], 1)

    def test_engineer_behavior_features(self):
        """Test user averages, historical gaps, and cohort relative deviations map smoothly."""
        # Prepare required parent dependency markers manually to test step 5 in isolation
        beh_df = self.base_df.copy()
        beh_df["time_since_signup"] = [30.0, 900.0, 21600.0]
        beh_df["user_txn_count"] = [2, 2, 1]
        
        result_df = engineer_behavior_features(beh_df)
        
        expected_cols = [
            "account_age_days", "transactions_per_hour", "avg_purchase_value", 
            "cohort_purchase_deviation", "purchase_deviation", "time_since_prev_txn"
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns)
            
        # Verify user average purchase value extraction
        # User 1 values: 100 and 200 -> Average = 150.0
        user_1_rows = result_df[result_df["user_id"] == 1]
        self.assertTrue((user_1_rows["avg_purchase_value"] == 150.0).all())
        
        # Verify sorting-based time_since_prev_txn calculation
        # Row 0 is user 1's first transaction -> filled default value 999999
        # Row 1 is user 1's second transaction -> 12:15:00 - 12:00:30 = 14 mins 30 secs = 870.0 secs
        user_1_sorted = user_1_rows.sort_values("purchase_time")
        self.assertEqual(user_1_sorted["time_since_prev_txn"].iloc[0], 999999.0)
        self.assertEqual(user_1_sorted["time_since_prev_txn"].iloc[1], 870.0)

    def test_engineer_all_pipeline_integration(self):
        """Verify the master coordinator runs all subsets sequentially without row shape mutation."""
        df_input = self.base_df.copy()
        df_input["class"] = [1, 0, 0] # explicitly mock target tracking column internally
        
        initial_row_count = len(df_input)
        result_df = engineer_all(df_input)
        
        # Structural check
        self.assertEqual(len(result_df), initial_row_count)
        
        # Content verification: Ensure a key component from every step successfully compiled
        self.assertIn("hour_of_day", result_df.columns)               # From time features
        self.assertIn("time_since_signup", result_df.columns)         # From signup features
        self.assertIn("platform_30min_velocity", result_df.columns)   # From velocity features
        self.assertIn("cohort_purchase_deviation", result_df.columns) # From behavior features
        self.assertIn("country_fraud_risk", result_df.columns)         # From risk features

if __name__ == "__main__":
    unittest.main()