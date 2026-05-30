"""
Unit Tests — FQHC SHAP Analysis
================================
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestFQHCData(unittest.TestCase):

    def setUp(self):
        from fqhc_model import generate_fqhc_data
        self.df = generate_fqhc_data(n_visits=500, seed=42)

    def test_shape(self):
        self.assertEqual(len(self.df), 500)

    def test_required_columns(self):
        for col in ["age", "sex_M", "hba1c", "systolic_bp", "bmi",
                    "has_diabetes", "has_hypertension", "race_ethnicity"]:
            self.assertIn(col, self.df.columns)

    def test_age_range(self):
        self.assertTrue((self.df["age"] >= 18).all())
        self.assertTrue((self.df["age"] <= 89).all())

    def test_binary_flags(self):
        for col in ["sex_M", "has_diabetes", "has_hypertension", "tobacco_current",
                    "housing_instability", "food_insecurity"]:
            self.assertTrue(self.df[col].isin([0, 1]).all(), f"{col} not binary")

    def test_hba1c_range(self):
        self.assertTrue((self.df["hba1c"] >= 4.0).all())
        self.assertTrue((self.df["hba1c"] <= 16.0).all())

    def test_condition_prevalence(self):
        """Condition prevalences should be in plausible clinical ranges."""
        self.assertGreater(self.df["has_hypertension"].mean(), 0.20)
        self.assertGreater(self.df["has_diabetes"].mean(), 0.08)
        self.assertGreater(self.df["has_obesity"].mean(), 0.20)

    def test_sdoh_features(self):
        for col in ["housing_instability", "food_insecurity", "transportation_barrier"]:
            self.assertIn(col, self.df.columns)
            self.assertTrue(self.df[col].isin([0, 1]).all())


class TestSHAPAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import xgboost
            import shap
        except ImportError:
            return
        from fqhc_model import generate_fqhc_data, load_or_train_model
        from shap_analysis import SHAPAnalyzer
        from sklearn.model_selection import train_test_split

        df = generate_fqhc_data(n_visits=1000, seed=42)
        model, X_train, X_test, y_train, y_test, feature_cols, demo_test = \
            load_or_train_model(
                model_path="/tmp/test_fqhc_model.pkl",
                n_train=1000,
                target_condition="diabetes",
            )
        cls.model = model
        cls.X_train = X_train
        cls.X_test = X_test
        cls.y_test = y_test
        cls.feature_cols = feature_cols
        cls.demo_test = demo_test
        cls.analyzer = SHAPAnalyzer(model, X_train, X_test, feature_cols)
        cls.shap_values = cls.analyzer.compute_shap_values(max_samples=500)

    def _skip_if_no_deps(self):
        try:
            import xgboost, shap
        except ImportError:
            self.skipTest("xgboost or shap not installed")

    def test_shap_values_shape(self):
        self._skip_if_no_deps()
        self.assertEqual(
            self.shap_values.shape[1],
            len(self.feature_cols)
        )
        self.assertGreater(self.shap_values.shape[0], 0)

    def test_global_importance_sorted(self):
        self._skip_if_no_deps()
        imp = self.analyzer.get_global_importance()
        self.assertEqual(list(imp.columns), ["feature", "mean_abs_shap", "rank"])
        # Should be sorted descending
        vals = imp["mean_abs_shap"].values
        self.assertTrue(all(vals[i] >= vals[i+1] for i in range(len(vals)-1)))

    def test_global_importance_non_negative(self):
        self._skip_if_no_deps()
        imp = self.analyzer.get_global_importance()
        self.assertTrue((imp["mean_abs_shap"] >= 0).all())

    def test_patient_explanation_structure(self):
        self._skip_if_no_deps()
        exp = self.analyzer.get_patient_explanation(0)
        self.assertIn("top_contributors", exp)
        self.assertIn("base_value", exp)
        self.assertIn("shap_sum", exp)
        for contrib in exp["top_contributors"]:
            self.assertIn("feature", contrib)
            self.assertIn("shap_value", contrib)
            self.assertIn("direction", contrib)

    def test_direction_labels(self):
        self._skip_if_no_deps()
        exp = self.analyzer.get_patient_explanation(0)
        for c in exp["top_contributors"]:
            self.assertIn(c["direction"], ["increases_risk", "decreases_risk"])
            expected_dir = "increases_risk" if c["shap_value"] > 0 else "decreases_risk"
            self.assertEqual(c["direction"], expected_dir)

    def test_equity_analysis_returns_dataframe(self):
        self._skip_if_no_deps()
        eq_df = self.analyzer.equity_analysis(self.demo_test, group_col="race_ethnicity")
        self.assertIsInstance(eq_df, pd.DataFrame)
        self.assertIn("feature", eq_df.columns)
        self.assertIn("disparity_ratio", eq_df.columns)

    def test_equity_disparity_positive(self):
        self._skip_if_no_deps()
        eq_df = self.analyzer.equity_analysis(self.demo_test, group_col="race_ethnicity")
        self.assertTrue((eq_df["disparity_ratio"] >= 1.0).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
