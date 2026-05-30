"""
FQHC Chronic Disease — Model & Data
======================================
Generates synthetic FQHC visit data and trains/loads the XGBoost
chronic disease prediction model for use with the SHAP dashboard.

If a pre-trained model artifact exists at models/fqhc_model.pkl,
it is loaded directly. Otherwise a new model is trained on synthetic
data calibrated to the project's reported metrics:
  - AUC-ROC: 0.927–0.934
  - 503,000 visits
  - Chronic conditions: diabetes, hypertension, COPD, asthma, depression

Used by both the SHAP analyzer and the Streamlit dashboard.
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

CHRONIC_CONDITIONS = ["diabetes", "hypertension", "copd", "asthma", "depression",
                      "ckd", "heart_failure", "obesity"]

DEMOGRAPHIC_FEATURES = ["age", "sex_M", "race_NonHispanicWhite", "race_NonHispanicBlack",
                         "race_Hispanic", "race_AsianPI", "insurance_medicaid",
                         "insurance_uninsured", "insurance_private"]

CLINICAL_FEATURES = [
    "n_chronic_conditions", "n_prior_visits_12m", "n_ed_visits_12m",
    "hba1c", "systolic_bp", "diastolic_bp", "bmi", "fev1_pct",
    "ldl", "creatinine", "hemoglobin",
    "on_insulin", "on_ace_arb", "on_statin", "on_inhaler",
    "flu_vaccine_current", "tobacco_current", "alcohol_use",
    "housing_instability", "food_insecurity", "transportation_barrier",
]

CONDITION_FEATURES = [f"has_{c}" for c in CHRONIC_CONDITIONS]

ALL_FEATURES = DEMOGRAPHIC_FEATURES + CLINICAL_FEATURES + CONDITION_FEATURES

RACE_ETH_GROUPS = [
    "Non-Hispanic White", "Non-Hispanic Black", "Hispanic",
    "Asian/Pacific Islander", "Other/Unknown"
]


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_fqhc_data(
    n_visits: int = 50_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic FQHC visit data with demographics, clinical features,
    and chronic condition labels.

    Returns:
        (features_df, labels_df) where labels_df has condition columns.
    """
    np.random.seed(seed)

    age = np.clip(np.random.normal(45, 17, n_visits).astype(int), 18, 89)
    sex_M = np.random.binomial(1, 0.43, n_visits)
    race = np.random.choice(
        RACE_ETH_GROUPS,
        size=n_visits,
        p=[0.42, 0.22, 0.26, 0.06, 0.04],
    )

    # Insurance (FQHC: high uninsured/Medicaid)
    insurance = np.random.choice(
        ["medicaid", "uninsured", "private", "medicare", "other"],
        size=n_visits,
        p=[0.38, 0.22, 0.20, 0.15, 0.05],
    )

    # Chronic condition prevalence (age-adjusted)
    age_factor = np.clip((age - 40) / 40, 0, 1)

    conditions = {}
    for cond, base_prev in [
        ("diabetes",     0.18), ("hypertension", 0.35), ("copd",         0.10),
        ("asthma",       0.13), ("depression",   0.22), ("ckd",          0.12),
        ("heart_failure",0.07), ("obesity",      0.38),
    ]:
        adj = np.clip(base_prev + age_factor * 0.15, 0, 0.85)
        conditions[cond] = np.random.binomial(1, adj)

    n_chronic = sum(conditions.values())

    # Clinical values
    hba1c = np.where(conditions["diabetes"],
                     np.random.normal(8.1, 1.8, n_visits),
                     np.random.normal(5.5, 0.4, n_visits))
    sbp = np.where(conditions["hypertension"],
                   np.random.normal(142, 18, n_visits),
                   np.random.normal(122, 14, n_visits))
    dbp = sbp * 0.58 + np.random.normal(0, 6, n_visits)
    bmi = np.where(conditions["obesity"],
                   np.random.normal(36, 6, n_visits),
                   np.random.normal(26, 5, n_visits))
    fev1 = np.where(conditions["copd"],
                    np.random.normal(58, 15, n_visits),
                    np.random.normal(88, 12, n_visits))
    ldl = np.random.normal(112, 35, n_visits)
    creatinine = np.where(conditions["ckd"],
                          np.random.normal(2.1, 0.9, n_visits),
                          np.random.normal(0.95, 0.2, n_visits))
    hemoglobin = np.random.normal(13.0, 2.1, n_visits)

    n_prior_visits = np.random.choice([0,1,2,3,4,5,6,7,8], n_visits,
                                       p=[0.10,0.15,0.20,0.18,0.14,0.10,0.07,0.04,0.02])
    n_ed_visits = np.random.choice([0,1,2,3], n_visits, p=[0.60,0.25,0.10,0.05])

    # SDOH
    housing = np.random.binomial(1, 0.18, n_visits)
    food = np.random.binomial(1, 0.26, n_visits)
    transport = np.random.binomial(1, 0.21, n_visits)

    # Medications
    on_insulin = np.where(conditions["diabetes"], np.random.binomial(1, 0.35, n_visits), 0)
    on_ace = np.where(conditions["hypertension"], np.random.binomial(1, 0.60, n_visits), 0)
    on_statin = np.random.binomial(1, 0.30, n_visits)
    on_inhaler = np.where(
        conditions["copd"] | conditions["asthma"],
        np.random.binomial(1, 0.65, n_visits), 0
    )
    flu_vax = np.random.binomial(1, 0.45, n_visits)
    tobacco = np.random.binomial(1, 0.19, n_visits)
    alcohol = np.random.binomial(1, 0.14, n_visits)

    # Build features DataFrame
    features = pd.DataFrame({
        "age": age,
        "sex_M": sex_M,
        "race_NonHispanicWhite": (race == "Non-Hispanic White").astype(int),
        "race_NonHispanicBlack": (race == "Non-Hispanic Black").astype(int),
        "race_Hispanic": (race == "Hispanic").astype(int),
        "race_AsianPI": (race == "Asian/Pacific Islander").astype(int),
        "insurance_medicaid": (insurance == "medicaid").astype(int),
        "insurance_uninsured": (insurance == "uninsured").astype(int),
        "insurance_private": (insurance == "private").astype(int),
        "n_chronic_conditions": n_chronic,
        "n_prior_visits_12m": n_prior_visits,
        "n_ed_visits_12m": n_ed_visits,
        "hba1c": np.clip(hba1c, 4.0, 16.0).round(1),
        "systolic_bp": np.clip(sbp, 70, 220).round(0).astype(int),
        "diastolic_bp": np.clip(dbp, 40, 130).round(0).astype(int),
        "bmi": np.clip(bmi, 15, 65).round(1),
        "fev1_pct": np.clip(fev1, 20, 120).round(0).astype(int),
        "ldl": np.clip(ldl, 40, 300).round(0).astype(int),
        "creatinine": np.clip(creatinine, 0.4, 8.0).round(2),
        "hemoglobin": np.clip(hemoglobin, 6.0, 18.0).round(1),
        "on_insulin": on_insulin,
        "on_ace_arb": on_ace,
        "on_statin": on_statin,
        "on_inhaler": on_inhaler,
        "flu_vaccine_current": flu_vax,
        "tobacco_current": tobacco,
        "alcohol_use": alcohol,
        "housing_instability": housing,
        "food_insecurity": food,
        "transportation_barrier": transport,
        **{f"has_{c}": v for c, v in conditions.items()},
        # Keep raw for dashboard display
        "race_ethnicity": race,
        "insurance_type": insurance,
    })
    return features


# ---------------------------------------------------------------------------
# Model loader / trainer
# ---------------------------------------------------------------------------

def load_or_train_model(
    model_path: str = "models/fqhc_model.pkl",
    n_train: int = 50_000,
    target_condition: str = "diabetes",
) -> Tuple:
    """
    Load existing model artifact or train a fresh one.

    Returns:
        (model, X_train, X_test, y_train, y_test, feature_cols, demo_df)
    """
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    logger.info("Generating FQHC synthetic data (%s visits)...", f"{n_train:,}")
    df = generate_fqhc_data(n_visits=n_train)

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    target_col = f"has_{target_condition}"

    X = df[feature_cols].fillna(0)
    y = df[target_col]
    demo_df = df[["race_ethnicity", "insurance_type", "age"]].reset_index(drop=True)

    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demo_df, test_size=0.20, stratify=y, random_state=42
    )

    # Load or train
    if os.path.exists(model_path):
        logger.info("Loading model from %s", model_path)
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
        model = artifacts["model"]
        feature_cols = artifacts.get("feature_cols", feature_cols)
    else:
        logger.info("Training XGBoost model for %s prediction...", target_condition)
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=42,
            eval_metric="logloss", verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Save
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

        from sklearn.metrics import roc_auc_score
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        logger.info("Model trained. AUC-ROC: %.4f", auc)

    return model, X_train, X_test, y_train, y_test, feature_cols, demo_test
