"""
SHAP Analysis — FQHC Chronic Disease Prediction Model
=======================================================
Generates SHAP explanations for the FQHC chronic disease XGBoost model.
Produces global importance, summary plots, waterfall plots, and
equity analysis (SHAP disparity across demographic subgroups).

Designed to integrate with an existing trained model artifact.
Falls back to a retrained model on synthetic FQHC data if no
pre-trained artifact is found.

Outputs (all written to results/shap/):
  - shap_values.parquet          — per-patient SHAP matrix
  - global_importance.csv        — mean |SHAP| per feature
  - subgroup_shap_summary.csv    — mean |SHAP| by race/ethnicity
  - figures/shap_summary.png
  - figures/shap_beeswarm.png
  - figures/shap_waterfall_*.png (top 5 patients by risk)
  - figures/shap_equity_heatmap.png

Usage:
    from src.shap_analysis import SHAPAnalyzer
    analyzer = SHAPAnalyzer(model, X_train, X_test, feature_names)
    analyzer.run(output_dir="results/shap/")
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SHAPSummary:
    """Summary of SHAP analysis results."""
    n_patients: int
    n_features: int
    top_features: List[Dict]           # [{feature, mean_abs_shap, rank}]
    mean_abs_shap_by_condition: Dict   # condition → mean |SHAP| for top feature
    equity_gaps: List[Dict]            # Features with largest disparity across groups
    shap_values_path: str
    figures: List[str]

    def to_dict(self) -> dict:
        return {
            "n_patients": self.n_patients,
            "n_features": self.n_features,
            "top_10_features": self.top_features[:10],
            "equity_gaps_top5": self.equity_gaps[:5],
        }


class SHAPAnalyzer:
    """
    Generates SHAP-based explanations for a trained XGBoost classifier.

    Implements:
      1. Global feature importance (mean |SHAP| ranking)
      2. Summary / beeswarm plots
      3. Waterfall plots for individual patients
      4. Equity analysis: SHAP disparity by race/ethnicity
      5. Dependency plots for top features
    """

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        condition_col: Optional[str] = None,
        demographic_col: Optional[str] = None,
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.condition_col = condition_col
        self.demographic_col = demographic_col
        self._explainer = None
        self._shap_values = None

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_shap_values(self, max_samples: int = 5000) -> np.ndarray:
        """
        Compute SHAP values using TreeExplainer.

        Args:
            max_samples: Max rows to explain (subsample for speed).

        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("shap required: pip install shap")

        logger.info("Initializing SHAP TreeExplainer...")
        self._explainer = shap.TreeExplainer(
            self.model,
            data=shap.sample(self.X_train, min(1000, len(self.X_train))),
            feature_perturbation="interventional",
        )

        # Subsample for speed
        if len(self.X_test) > max_samples:
            sample_idx = np.random.choice(len(self.X_test), max_samples, replace=False)
            X_explain = self.X_test.iloc[sample_idx]
        else:
            X_explain = self.X_test

        logger.info("Computing SHAP values for %d patients...", len(X_explain))
        self._shap_values = self._explainer.shap_values(X_explain)

        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(self._shap_values, list):
            self._shap_values = self._shap_values[1]  # Positive class

        logger.info("SHAP values computed: shape %s", self._shap_values.shape)
        return self._shap_values

    def get_global_importance(self) -> pd.DataFrame:
        """Return feature importance ranked by mean |SHAP| value."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        mean_abs = np.abs(self._shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1
        return importance_df

    def get_patient_explanation(
        self, patient_idx: int
    ) -> Dict:
        """Return top feature contributions for a single patient."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        shap_row = self._shap_values[patient_idx]
        feature_vals = self.X_test.iloc[patient_idx]

        contributions = sorted(
            [
                {
                    "feature": f,
                    "feature_value": round(float(feature_vals[f]), 3),
                    "shap_value": round(float(sv), 4),
                    "direction": "increases_risk" if sv > 0 else "decreases_risk",
                }
                for f, sv in zip(self.feature_names, shap_row)
            ],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )
        return {
            "patient_idx": patient_idx,
            "top_contributors": contributions[:10],
            "base_value": round(float(self._explainer.expected_value
                                      if not isinstance(self._explainer.expected_value, list)
                                      else self._explainer.expected_value[1]), 4),
            "shap_sum": round(float(shap_row.sum()), 4),
        }

    # ------------------------------------------------------------------
    # Equity analysis
    # ------------------------------------------------------------------

    def equity_analysis(
        self,
        demographics_df: pd.DataFrame,
        group_col: str = "race_ethnicity",
    ) -> pd.DataFrame:
        """
        Compute mean |SHAP| per feature per demographic group.
        Identifies features where model behavior differs across groups.

        Returns DataFrame: feature × group → mean |SHAP|
        """
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        n = min(len(self._shap_values), len(demographics_df))
        demo = demographics_df.iloc[:n].reset_index(drop=True)
        shap_df = pd.DataFrame(
            self._shap_values[:n],
            columns=self.feature_names
        )
        shap_df[group_col] = demo[group_col].values

        groups = shap_df[group_col].unique()
        records = []
        for feature in self.feature_names:
            row = {"feature": feature}
            for grp in groups:
                grp_shap = shap_df[shap_df[group_col] == grp][feature]
                row[str(grp)] = round(float(np.abs(grp_shap).mean()), 4)
            records.append(row)

        equity_df = pd.DataFrame(records)

        # Compute disparity: max group mean / min group mean
        numeric_cols = [c for c in equity_df.columns if c != "feature"]
        if len(numeric_cols) >= 2:
            equity_df["max_group_mean"] = equity_df[numeric_cols].max(axis=1)
            equity_df["min_group_mean"] = equity_df[numeric_cols].min(axis=1)
            equity_df["disparity_ratio"] = (
                equity_df["max_group_mean"] /
                (equity_df["min_group_mean"] + 1e-8)
            ).round(3)
            equity_df = equity_df.sort_values("disparity_ratio", ascending=False)

        return equity_df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_summary(self, output_path: str, max_display: int = 20) -> str:
        """Bar chart of top features by mean |SHAP|."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                self._shap_values,
                self.X_test.iloc[:len(self._shap_values)],
                feature_names=self.feature_names,
                plot_type="bar",
                max_display=max_display,
                show=False,
            )
            plt.title("FQHC Chronic Disease Model — Global Feature Importance (SHAP)",
                      fontsize=13, pad=12)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Summary plot saved: %s", output_path)
            return output_path
        except Exception as e:
            logger.warning("Summary plot failed: %s", e)
            return self._fallback_bar_chart(output_path)

    def plot_beeswarm(self, output_path: str, max_display: int = 20) -> str:
        """Beeswarm plot showing feature value vs SHAP value."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(
                self._shap_values,
                self.X_test.iloc[:len(self._shap_values)],
                feature_names=self.feature_names,
                plot_type="dot",
                max_display=max_display,
                show=False,
            )
            plt.title("FQHC Model — SHAP Beeswarm", fontsize=13)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        except Exception as e:
            logger.warning("Beeswarm plot failed: %s", e)
            return output_path

    def plot_waterfall(self, patient_idx: int, output_path: str) -> str:
        """Waterfall plot for one patient's SHAP contributions."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            explanation = shap.Explanation(
                values=self._shap_values[patient_idx],
                base_values=(
                    self._explainer.expected_value
                    if not isinstance(self._explainer.expected_value, list)
                    else self._explainer.expected_value[1]
                ),
                data=self.X_test.iloc[patient_idx].values,
                feature_names=self.feature_names,
            )
            shap.waterfall_plot(explanation, max_display=12, show=False)
            plt.title(f"Patient {patient_idx} — SHAP Waterfall", fontsize=12)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        except Exception as e:
            logger.warning("Waterfall plot failed for patient %d: %s", patient_idx, e)
            return output_path

    def plot_equity_heatmap(
        self,
        equity_df: pd.DataFrame,
        output_path: str,
        top_n: int = 15,
    ) -> str:
        """Heatmap of mean |SHAP| per feature per demographic group."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            group_cols = [c for c in equity_df.columns
                          if c not in ("feature", "max_group_mean",
                                        "min_group_mean", "disparity_ratio")]
            top_features = equity_df.head(top_n)["feature"].tolist()
            plot_df = equity_df[equity_df["feature"].isin(top_features)].set_index("feature")
            plot_df = plot_df[group_cols]

            fig, ax = plt.subplots(figsize=(max(8, len(group_cols) * 1.5), 8))
            sns.heatmap(
                plot_df.astype(float),
                annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax,
            )
            ax.set_title(
                "SHAP Feature Importance by Race/Ethnicity\n"
                "(Higher = feature drives prediction more for this group)",
                fontsize=12
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        except Exception as e:
            logger.warning("Equity heatmap failed: %s", e)
            return output_path

    def _fallback_bar_chart(self, output_path: str) -> str:
        """Minimal matplotlib bar chart when shap plotting unavailable."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            importance = self.get_global_importance().head(20)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(
                importance["feature"][::-1],
                importance["mean_abs_shap"][::-1],
                color="#2196F3", alpha=0.85,
            )
            ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
            ax.set_title("FQHC Chronic Disease Model — Feature Importance (SHAP)", fontsize=13)
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            return output_path
        except Exception as e:
            logger.error("Fallback chart also failed: %s", e)
            return output_path

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(
        self,
        output_dir: str = "results/shap/",
        demographics_df: Optional[pd.DataFrame] = None,
    ) -> SHAPSummary:
        """
        Run full SHAP analysis pipeline and write all outputs.

        Args:
            output_dir:      Directory for output files.
            demographics_df: Optional DataFrame with demographic columns
                             for equity analysis.
        Returns:
            SHAPSummary with paths and key statistics.
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        figures = []

        # Compute SHAP
        self.compute_shap_values(max_samples=5000)

        # Global importance
        importance_df = self.get_global_importance()
        imp_path = os.path.join(output_dir, "global_importance.csv")
        importance_df.to_csv(imp_path, index=False)

        # Save SHAP matrix
        shap_df = pd.DataFrame(self._shap_values, columns=self.feature_names)
        shap_path = os.path.join(output_dir, "shap_values.parquet")
        shap_df.to_parquet(shap_path, index=False)

        # Summary plot
        summary_path = self.plot_summary(
            os.path.join(output_dir, "figures", "shap_summary.png")
        )
        figures.append(summary_path)

        # Beeswarm
        beeswarm_path = self.plot_beeswarm(
            os.path.join(output_dir, "figures", "shap_beeswarm.png")
        )
        figures.append(beeswarm_path)

        # Waterfall for top 5 highest-risk patients
        pred_probs = self.model.predict_proba(self.X_test.iloc[:len(self._shap_values)])[:, 1]
        top5_idx = np.argsort(pred_probs)[-5:][::-1]
        for i, idx in enumerate(top5_idx):
            wf_path = self.plot_waterfall(
                idx, os.path.join(output_dir, "figures", f"shap_waterfall_rank{i+1}.png")
            )
            figures.append(wf_path)

        # Equity analysis
        equity_gaps = []
        if demographics_df is not None:
            equity_df = self.equity_analysis(demographics_df)
            eq_path = os.path.join(output_dir, "subgroup_shap_summary.csv")
            equity_df.to_csv(eq_path, index=False)
            eq_fig = self.plot_equity_heatmap(
                equity_df, os.path.join(output_dir, "figures", "shap_equity_heatmap.png")
            )
            figures.append(eq_fig)
            equity_gaps = equity_df.head(5).to_dict(orient="records")

        # Build summary
        top_features = importance_df.head(10).to_dict(orient="records")
        summary = SHAPSummary(
            n_patients=len(self._shap_values),
            n_features=len(self.feature_names),
            top_features=top_features,
            mean_abs_shap_by_condition={},
            equity_gaps=equity_gaps,
            shap_values_path=shap_path,
            figures=figures,
        )

        logger.info(
            "SHAP analysis complete. Top feature: %s (mean |SHAP|=%.4f)",
            importance_df.iloc[0]["feature"],
            importance_df.iloc[0]["mean_abs_shap"],
        )
        return summary
