"""
FQHC Chronic Disease Prediction — SHAP Explainability Dashboard
================================================================
Interactive Streamlit dashboard for exploring model predictions and
SHAP explanations for the FQHC chronic disease XGBoost model.

Pages:
  1. Overview       — Model performance, dataset statistics
  2. Global SHAP    — Feature importance bar + beeswarm plots
  3. Patient Explorer — Individual patient prediction + waterfall
  4. Equity Analysis — SHAP disparity by race/ethnicity
  5. Population Risk — Risk distribution by condition + subgroup

Run:
    streamlit run src/dashboard.py
    streamlit run src/dashboard.py --server.port 8501
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

try:
    import streamlit as st
    st.set_page_config(
        page_title="FQHC Chronic Disease — SHAP Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except ImportError:
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #9ba3c7; margin-top: 0.2rem; }
    .risk-high   { color: #ef5350; font-weight: 700; }
    .risk-mod    { color: #ffa726; font-weight: 700; }
    .risk-low    { color: #66bb6a; font-weight: 700; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        border-bottom: 2px solid #4fc3f7;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
        color: #e0e8ff;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model and computing SHAP values...")
def load_model_and_shap(target_condition: str = "diabetes", n_train: int = 20_000):
    """Load model + compute SHAP in one cached call."""
    from fqhc_model import load_or_train_model
    from shap_analysis import SHAPAnalyzer

    model, X_train, X_test, y_train, y_test, feature_cols, demo_test = \
        load_or_train_model(
            model_path=f"models/fqhc_{target_condition}.pkl",
            n_train=n_train,
            target_condition=target_condition,
        )

    analyzer = SHAPAnalyzer(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_cols,
    )
    analyzer.compute_shap_values(max_samples=3000)

    pred_probs = model.predict_proba(X_test)[:, 1]
    importance_df = analyzer.get_global_importance()
    equity_df = analyzer.equity_analysis(demo_test, group_col="race_ethnicity")

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y_test, pred_probs)
    auc_pr = average_precision_score(y_test, pred_probs)

    return {
        "model": model,
        "analyzer": analyzer,
        "X_test": X_test,
        "y_test": y_test,
        "pred_probs": pred_probs,
        "feature_cols": feature_cols,
        "importance_df": importance_df,
        "equity_df": equity_df,
        "demo_test": demo_test,
        "auc": auc,
        "auc_pr": auc_pr,
        "prevalence": float(y_test.mean()),
        "n_test": len(y_test),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.shields.io/badge/FQHC-Chronic%20Disease-4fc3f7?style=for-the-badge",
             use_column_width=True)
    st.markdown("## Settings")

    target_condition = st.selectbox(
        "Target Condition",
        options=["diabetes", "hypertension", "copd", "asthma", "depression",
                 "ckd", "heart_failure", "obesity"],
        index=0,
    )

    n_train = st.select_slider(
        "Training Set Size",
        options=[5_000, 10_000, 20_000, 50_000],
        value=20_000,
        help="Larger = more accurate model; slower to train",
    )

    page = st.radio(
        "Navigate",
        options=["📊 Overview", "🔍 Global SHAP", "👤 Patient Explorer",
                 "⚖️ Equity Analysis", "📈 Population Risk"],
        index=0,
    )

    st.divider()
    st.caption("FQHC Chronic Disease SHAP Dashboard")
    st.caption("AUC-ROC target: 0.927–0.934")
    st.caption("Visits in project: 503K")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = load_model_and_shap(target_condition=target_condition, n_train=n_train)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "📊 Overview":
    st.title("🏥 FQHC Chronic Disease Prediction")
    st.markdown(f"### {target_condition.replace('_', ' ').title()} Risk Model — SHAP Explainability Dashboard")

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("AUC-ROC", f"{data['auc']:.3f}"),
        ("AUC-PR", f"{data['auc_pr']:.3f}"),
        ("Test Set", f"{data['n_test']:,}"),
        ("Prevalence", f"{data['prevalence']:.1%}"),
        ("Features", f"{len(data['feature_cols'])}"),
    ]
    for col, (label, value) in zip([col1,col2,col3,col4,col5], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header">Top 10 SHAP Features</div>',
                    unsafe_allow_html=True)
        top10 = data["importance_df"].head(10).copy()
        top10["mean_abs_shap"] = top10["mean_abs_shap"].round(4)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ["#4fc3f7"] * 10
        ax.barh(top10["feature"][::-1], top10["mean_abs_shap"][::-1], color=colors, alpha=0.85)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title(f"Top Features — {target_condition.title()}", fontsize=11)
        ax.tick_params(labelsize=9)
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#1e2130")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3250")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_right:
        st.markdown('<div class="section-header">Risk Distribution</div>',
                    unsafe_allow_html=True)
        probs = data["pred_probs"]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(probs[data["y_test"] == 0], bins=40, alpha=0.6,
                 label=f"No {target_condition}", color="#66bb6a", density=True)
        ax2.hist(probs[data["y_test"] == 1], bins=40, alpha=0.6,
                 label=f"Has {target_condition}", color="#ef5350", density=True)
        ax2.set_xlabel("Predicted Probability", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.set_title("Predicted Risk Distribution by True Label", fontsize=11)
        ax2.legend(fontsize=9)
        fig2.patch.set_facecolor("#0e1117")
        ax2.set_facecolor("#1e2130")
        ax2.tick_params(colors="white")
        for label_obj in ax2.get_xticklabels() + ax2.get_yticklabels():
            label_obj.set_color("white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#2d3250")
        ax2.xaxis.label.set_color("white")
        ax2.yaxis.label.set_color("white")
        ax2.title.set_color("white")
        ax2.legend(labelcolor="white", facecolor="#1e2130", edgecolor="#2d3250")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Feature table
    st.markdown('<div class="section-header">Full Feature Importance Table</div>',
                unsafe_allow_html=True)
    display_df = data["importance_df"].copy()
    display_df["mean_abs_shap"] = display_df["mean_abs_shap"].round(5)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=300,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "feature": st.column_config.TextColumn("Feature", width="large"),
            "mean_abs_shap": st.column_config.NumberColumn("Mean |SHAP|", format="%.5f"),
        }
    )

# ---------------------------------------------------------------------------
# Page: Global SHAP
# ---------------------------------------------------------------------------

elif page == "🔍 Global SHAP":
    st.title("🔍 Global SHAP Feature Importance")
    st.markdown("Mean absolute SHAP value across all test-set patients. "
                "Higher = more influential in model predictions.")

    n_features = st.slider("Features to display", 5, min(40, len(data["feature_cols"])), 20)

    importance = data["importance_df"].head(n_features).copy()

    fig, ax = plt.subplots(figsize=(10, max(5, n_features * 0.35)))
    bars = ax.barh(
        importance["feature"][::-1],
        importance["mean_abs_shap"][::-1],
        color="#4fc3f7", alpha=0.85, edgecolor="#2d3250"
    )
    # Color top 5 differently
    for i, bar in enumerate(reversed(bars)):
        if i < 5:
            bar.set_color("#e91e63")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11, color="white")
    ax.set_title(f"FQHC {target_condition.title()} Model — Global Feature Importance",
                 fontsize=13, color="white")
    ax.tick_params(labelsize=9, colors="white")
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1e2130")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3250")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.caption("🔴 Top 5 features highlighted. SHAP = SHapley Additive exPlanations (Lundberg & Lee, 2017)")

    with st.expander("SHAP Value Distribution (feature-level)"):
        analyzer = data["analyzer"]
        shap_vals = analyzer._shap_values
        top5_features = importance.head(5)["feature"].tolist()

        fig3, axes = plt.subplots(1, min(5, len(top5_features)),
                                   figsize=(min(14, len(top5_features)*2.8), 4))
        if len(top5_features) == 1:
            axes = [axes]
        for ax_i, feat in zip(axes, top5_features):
            feat_idx = data["feature_cols"].index(feat)
            ax_i.hist(shap_vals[:, feat_idx], bins=30, color="#4fc3f7", alpha=0.8)
            ax_i.set_title(feat[:20], fontsize=8, color="white")
            ax_i.tick_params(labelsize=7, colors="white")
            ax_i.set_facecolor("#1e2130")
            for spine in ax_i.spines.values():
                spine.set_edgecolor("#2d3250")
        fig3.patch.set_facecolor("#0e1117")
        plt.suptitle("SHAP Value Distribution — Top 5 Features", color="white", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

# ---------------------------------------------------------------------------
# Page: Patient Explorer
# ---------------------------------------------------------------------------

elif page == "👤 Patient Explorer":
    st.title("👤 Individual Patient SHAP Explorer")
    st.markdown("Explore why the model assigned a specific risk score to any patient in the test set.")

    probs = data["pred_probs"]
    n_test = len(probs)

    col1, col2 = st.columns([1, 2])
    with col1:
        selection_mode = st.radio("Select patient by", ["Index", "Risk percentile"])
        if selection_mode == "Index":
            patient_idx = st.number_input("Patient index", 0, n_test - 1, 0)
        else:
            pct = st.slider("Risk percentile", 1, 99, 95)
            patient_idx = int(np.argsort(probs)[int(n_test * pct / 100)])

        prob = probs[patient_idx]
        true_label = int(data["y_test"].iloc[patient_idx])
        tier = "🔴 HIGH" if prob >= 0.35 else "🟡 MODERATE" if prob >= 0.20 else "🟢 LOW"

        st.metric("Predicted Risk", f"{prob:.1%}")
        st.metric("Risk Tier", tier)
        st.metric("True Label", f"{'Positive ✓' if true_label else 'Negative ✗'}")

        # Patient feature values
        st.markdown("**Clinical Features**")
        patient_features = data["X_test"].iloc[patient_idx]
        feature_display = patient_features.round(2).to_frame("Value")
        st.dataframe(feature_display, height=250)

    with col2:
        # SHAP waterfall
        st.markdown('<div class="section-header">SHAP Waterfall — Top Risk Drivers</div>',
                    unsafe_allow_html=True)

        explanation = data["analyzer"].get_patient_explanation(patient_idx)
        contributors = explanation["top_contributors"]

        contrib_df = pd.DataFrame(contributors)
        contrib_df["color"] = contrib_df["shap_value"].apply(
            lambda x: "#ef5350" if x > 0 else "#66bb6a"
        )
        contrib_df = contrib_df.sort_values("shap_value", key=abs, ascending=True).tail(12)

        fig_wf, ax_wf = plt.subplots(figsize=(8, 5))
        colors = contrib_df["color"].tolist()
        ax_wf.barh(contrib_df["feature"], contrib_df["shap_value"], color=colors, alpha=0.85)
        ax_wf.axvline(0, color="white", linewidth=0.8, alpha=0.5)
        ax_wf.set_xlabel("SHAP Value (contribution to log-odds)", fontsize=10, color="white")
        ax_wf.set_title(f"Patient {patient_idx} — Risk Drivers\n(red=increases risk, green=decreases)",
                        fontsize=11, color="white")
        ax_wf.tick_params(labelsize=9, colors="white")
        fig_wf.patch.set_facecolor("#0e1117")
        ax_wf.set_facecolor("#1e2130")
        for spine in ax_wf.spines.values():
            spine.set_edgecolor("#2d3250")
        plt.tight_layout()
        st.pyplot(fig_wf, use_container_width=True)
        plt.close()

        st.markdown("**Top 10 Feature Contributions**")
        st.dataframe(
            pd.DataFrame(explanation["top_contributors"])
            [["feature", "feature_value", "shap_value", "direction"]]
            .round(4),
            use_container_width=True,
            height=250,
        )

# ---------------------------------------------------------------------------
# Page: Equity Analysis
# ---------------------------------------------------------------------------

elif page == "⚖️ Equity Analysis":
    st.title("⚖️ Equity Analysis — SHAP Disparity by Race/Ethnicity")
    st.markdown(
        "Mean |SHAP| per feature per demographic group. "
        "High disparity ratio indicates model behavior differs across groups — "
        "a signal worth investigating for potential bias."
    )

    equity_df = data["equity_df"].copy()
    numeric_cols = [c for c in equity_df.columns
                    if c not in ("feature", "max_group_mean", "min_group_mean", "disparity_ratio")]

    # Disparity table
    st.markdown('<div class="section-header">Features with Highest Disparity Across Groups</div>',
                unsafe_allow_html=True)
    display_eq = equity_df.head(15)[["feature", "disparity_ratio"] + numeric_cols].round(4)
    st.dataframe(display_eq, use_container_width=True, height=350)

    # Heatmap
    st.markdown('<div class="section-header">SHAP Importance Heatmap by Group (Top 15 Features)</div>',
                unsafe_allow_html=True)
    top15 = equity_df.head(15).set_index("feature")[numeric_cols].astype(float)

    try:
        import seaborn as sns
        fig_eq, ax_eq = plt.subplots(figsize=(max(8, len(numeric_cols)*1.5), 7))
        sns.heatmap(top15, annot=True, fmt=".3f", cmap="YlOrRd",
                    linewidths=0.5, ax=ax_eq, cbar_kws={"label": "Mean |SHAP|"})
        ax_eq.set_title(
            f"SHAP Feature Importance by Race/Ethnicity — {target_condition.title()} Model",
            fontsize=12, color="white"
        )
        ax_eq.tick_params(colors="white", labelsize=8)
        fig_eq.patch.set_facecolor("#0e1117")
        ax_eq.set_facecolor("#1e2130")
        plt.tight_layout()
        st.pyplot(fig_eq, use_container_width=True)
        plt.close()
    except ImportError:
        st.info("Install seaborn for heatmap: pip install seaborn")

    st.info(
        "**Interpretation:** Disparity ratio > 1.5 indicates a feature influences predictions "
        "50%+ more strongly for one group than another. This does not automatically indicate "
        "bias — clinical differences may be real — but warrants review."
    )

# ---------------------------------------------------------------------------
# Page: Population Risk
# ---------------------------------------------------------------------------

elif page == "📈 Population Risk":
    st.title("📈 Population Risk Analysis")
    st.markdown("Risk distribution and high-risk patient breakdown by subgroup.")

    probs = data["pred_probs"]
    y = data["y_test"]
    demo = data["demo_test"].reset_index(drop=True)

    # Risk tier breakdown
    tiers = pd.cut(probs, bins=[0, 0.20, 0.35, 1.0],
                   labels=["Low (<20%)", "Moderate (20-35%)", "High (>35%)"])
    tier_counts = tiers.value_counts().sort_index()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Risk Tier Distribution</div>',
                    unsafe_allow_html=True)
        fig_tier, ax_tier = plt.subplots(figsize=(5, 4))
        colors_tier = ["#66bb6a", "#ffa726", "#ef5350"]
        ax_tier.bar(tier_counts.index, tier_counts.values, color=colors_tier, alpha=0.85)
        ax_tier.set_ylabel("Patients", color="white")
        ax_tier.tick_params(colors="white", labelsize=8)
        fig_tier.patch.set_facecolor("#0e1117")
        ax_tier.set_facecolor("#1e2130")
        for spine in ax_tier.spines.values():
            spine.set_edgecolor("#2d3250")
        plt.tight_layout()
        st.pyplot(fig_tier, use_container_width=True)
        plt.close()

        # Tier table
        tier_df = pd.DataFrame({
            "Risk Tier": tier_counts.index,
            "N Patients": tier_counts.values,
            "% of Population": (tier_counts.values / len(probs) * 100).round(1),
        })
        st.dataframe(tier_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">High-Risk Rate by Race/Ethnicity</div>',
                    unsafe_allow_html=True)
        demo_probs = demo.copy()
        demo_probs["prob"] = probs[:len(demo_probs)]
        demo_probs["high_risk"] = (demo_probs["prob"] >= 0.35).astype(int)

        race_risk = demo_probs.groupby("race_ethnicity").agg(
            n=("high_risk", "count"),
            pct_high_risk=("high_risk", "mean"),
        ).reset_index()
        race_risk["pct_high_risk"] = (race_risk["pct_high_risk"] * 100).round(1)
        race_risk = race_risk.sort_values("pct_high_risk", ascending=False)

        fig_race, ax_race = plt.subplots(figsize=(5, 4))
        ax_race.barh(race_risk["race_ethnicity"], race_risk["pct_high_risk"],
                     color="#4fc3f7", alpha=0.85)
        ax_race.set_xlabel("% High Risk (>35%)", color="white")
        ax_race.tick_params(colors="white", labelsize=8)
        fig_race.patch.set_facecolor("#0e1117")
        ax_race.set_facecolor("#1e2130")
        for spine in ax_race.spines.values():
            spine.set_edgecolor("#2d3250")
        plt.tight_layout()
        st.pyplot(fig_race, use_container_width=True)
        plt.close()

        st.dataframe(race_risk, use_container_width=True, hide_index=True)

    # Age-risk scatter
    st.markdown('<div class="section-header">Age vs. Predicted Risk</div>',
                unsafe_allow_html=True)
    age_vals = data["X_test"]["age"].values[:len(probs)] if "age" in data["X_test"].columns else np.zeros(len(probs))
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 4))
    scatter = ax_scatter.scatter(
        age_vals, probs,
        c=probs, cmap="RdYlGn_r", alpha=0.3, s=8
    )
    ax_scatter.set_xlabel("Age", color="white")
    ax_scatter.set_ylabel("Predicted Risk", color="white")
    ax_scatter.set_title(f"Age vs. {target_condition.title()} Risk", color="white")
    ax_scatter.tick_params(colors="white")
    fig_scatter.patch.set_facecolor("#0e1117")
    ax_scatter.set_facecolor("#1e2130")
    for spine in ax_scatter.spines.values():
        spine.set_edgecolor("#2d3250")
    plt.colorbar(scatter, ax=ax_scatter, label="Predicted Risk").ax.yaxis.label.set_color("white")
    plt.tight_layout()
    st.pyplot(fig_scatter, use_container_width=True)
    plt.close()
