# Changelog / Review Notes

## 2026-07-01 — Dashboard wired to the real trained models

**Bug found:** `src/dashboard.py` called `load_or_train_model(model_path=f"models/fqhc_{target_condition}.pkl", ...)`,
but no file matching that naming pattern exists in `models/`. Because
`os.path.exists(model_path)` was always `False`, the dashboard silently fell
back to training a brand-new XGBoost model on freshly-generated synthetic
data (`generate_fqhc_data()`) on every load — never touching the 5 real
models described in the README.

**Investigation turned up a second, deeper issue:** even after fixing the
path, the dashboard's feature pipeline (`fqhc_model.py`) and the real models
in `models/*.pkl` were built independently and were incompatible:

| | Real models (`notebooks/03_predictive_models.ipynb`) | Old dashboard pipeline |
|---|---|---|
| Algorithm | RandomForest / GradientBoosting / LogisticRegression (joblib) | XGBoost (pickle dict) |
| Features | 10–14 real NAMCS fields (`AGE`, `sex_binary`, `num_diagnoses`, race/season dummies, etc.) | 38 fabricated clinical fields (`hba1c`, `creatinine`, `on_insulin`, ...) that don't exist in NAMCS |
| Data | Real 503,799-visit NAMCS extract | 100% synthetic, regenerated per run |

Confirmed by inspection: `models/hypertension_classifier.pkl.feature_names_in_`
lists exactly 10 columns, none of which overlap with the synthetic
generator's columns. Calling `.predict_proba()` on the real model with the
synthetic feature matrix would raise a feature-mismatch error, not produce a
slightly-off number — so the original path fix alone would not have worked.

**Fix applied:**
1. Reproduced `data/processed/hc2024_processed.pkl` from the raw NAMCS SAS
   file (`data/raw/HC2024_SAS.sas7bdat`) using the exact feature-engineering
   steps in `notebooks/01_feature_engineering.ipynb` (503,799 records,
   matching the README).
2. Verified the 5 real `models/*.pkl` artifacts against the exact
   train/test split from `notebooks/03_predictive_models.ipynb`
   (`test_size=0.25, random_state=42`) and confirmed the reproduced
   AUC-ROC/F1 match the README exactly:
   hypertension 0.927/0.616, diabetes 0.884/0.126, mental_health 0.853/0.444,
   high_complexity 0.835/0.726, multimorbidity 0.934/0.709.
3. Added `src/fqhc_model.py::load_real_model()`, which loads the real
   joblib models and reconstructs their exact feature schema from a trimmed,
   git-tracked feature extract (`data/processed/hc2024_model_features.parquet`,
   ~1.8MB — only the columns needed by the 5 models, not the full 377MB
   processed dataset).
4. Rewired `src/dashboard.py` to call `load_real_model()` instead of the
   synthetic `load_or_train_model()`, and changed the condition dropdown to
   the 5 real conditions (`hypertension`, `diabetes`, `mental_health`,
   `high_complexity`, `multimorbidity`) instead of 8 invented ones (3 of
   which — copd, asthma, ckd, heart_failure, obesity — never had a trained
   model backing them).
5. Updated `src/shap_analysis.py::SHAPAnalyzer.compute_shap_values()` to
   branch on model type: `shap.TreeExplainer` for the RandomForest/
   GradientBoosting models, `shap.LinearExplainer` for the LogisticRegression
   multimorbidity model (previously hardcoded to `TreeExplainer`, which
   would have raised on the linear model).
6. Fixed two unrelated bugs surfaced during verification:
   - Base-value/expected-value handling assumed `shap`'s older list-based
     return format; the installed `shap==0.52.0` returns a 3D array for
     binary classifiers instead. Normalized both paths.
   - `matplotlib.pyplot` was only imported inside the Overview page's `if`
     block, so every other page (Global SHAP, Patient Explorer, Equity
     Analysis, Population Risk) crashed with `NameError: name 'plt' is not
     defined`. Moved the import to module scope.

**Kept, not removed:** `generate_fqhc_data()` and `load_or_train_model()`
(the synthetic XGBoost pipeline) remain in `src/fqhc_model.py` solely
because `tests/test_shap.py` exercises the generic `SHAPAnalyzer` interface
against them. They are no longer reachable from the live dashboard.

**Verification:** all 5 conditions' Overview pages were loaded in a running
`streamlit run src/dashboard.py` instance and confirmed to display the
README's exact AUC-ROC figures. `python -m pytest tests/test_shap.py -v`
passes 14/14.
