# Portfolio Project Brief: FQHC Chronic Disease Analysis

## One-Sentence Summary
Built machine learning models to predict chronic disease prevalence and multimorbidity in Federally Qualified Health Centers using 503K+ national survey visits, identifying $11.7B in savings through evidence-based intervention programs.

---

## Resume Bullet (Primary)

**Analyzed 503K+ health center visits from CDC NAMCS (2024), built multi-label chronic disease classifiers (hypertension AUC=0.93, multimorbidity AUC=0.93), quantified racial disparities in diagnosis rates, and identified $11.7 billion national savings opportunity through diabetes intervention (ROI 3.8x), hypertension control (ROI 2.4x), and integrated mental health programs at 1,400 FQHCs**

---

## Resume Bullet (Alternative - Technical Focus)

**Developed ensemble machine learning models (Random Forest, Gradient Boosting, Logistic Regression) for chronic disease prediction using complex survey data (503K visits, 107 health centers), achieved 0.93 AUC-ROC for multimorbidity classification, and performed health equity analysis revealing 1.3-1.4x disparity ratios in chronic disease prevalence**

---

## Interview Talking Points

### "Walk me through this project"

**Setup (30 sec):**
"I analyzed Federally Qualified Health Centers because they serve 30 million underserved patients annually with limited resources. Using CDC's 2024 national survey data—503,000 visits from 107 health centers—I built predictive models to optimize chronic disease management and quantify intervention ROI."

**Technical Approach (60 sec):**
"First, I engineered features from ICD-10 diagnosis codes to identify 8 chronic conditions, creating flags for diabetes, hypertension, mental health disorders, and multimorbidity. Then I built five models: Random Forest classifiers for individual conditions, Gradient Boosting for visit complexity prediction, and Logistic Regression for multimorbidity. The hypertension and multimorbidity models both achieved 0.93 AUC-ROC—strong discrimination for clinical applications."

**Business Impact (30 sec):**
"The health equity analysis revealed significant disparities—Black patients 1.3x higher hypertension rates, Hispanic patients 1.4x higher diabetes. I then modeled three national intervention programs: diabetes early intervention with 3.8x ROI, hypertension control at 2.4x ROI, and integrated mental health. Total savings potential: $11.7 billion across all 1,400 FQHCs."

### "What was your biggest challenge?"

**Class Imbalance:**
"Diabetes appeared in only 10% of visits—severe class imbalance. The initial model had 0.88 AUC but only 0.13 F1 score. I tried SMOTE oversampling, but it actually hurt performance on the test set. The solution was accepting the precision-recall trade-off: the model correctly identifies high-risk patients (high precision) but misses some cases (lower recall). For a screening application, this is acceptable—we catch the highest-risk 7% with 58% precision, which still provides clinical value for targeted intervention."

### "How would you deploy this?"

**Three-Phase Approach:**
1. **Retrospective Validation (3 months):** Run models on historical EHR data from participating FQHCs, validate predictions against actual outcomes, adjust decision thresholds
2. **Pilot Implementation (6 months):** Deploy at 5-10 FQHCs with dashboard showing daily risk scores, train care coordinators on interpretation, measure impact on care coordination time
3. **National Rollout:** API integration with major FQHC EHR systems (OCHIN, NextGen), automated flagging for high-risk patients, continuous monitoring

**Key Success Metric:** 20% increase in chronic disease screening completion rates within 6 months.

---

## Technical Deep-Dive

### Why These Algorithms?

**Random Forest for Chronic Disease:**
- Handles mixed feature types (demographics + counts + categorical)
- Robust to class imbalance with class_weight parameter
- Built-in feature importance for clinical interpretability
- No assumptions about feature distributions

**Gradient Boosting for Complexity:**
- Sequential error correction ideal for noisy healthcare data
- Better calibrated probabilities than Random Forest
- Handles non-linear relationships (age × diagnosis interactions)

**Logistic Regression for Multimorbidity:**
- Linear relationships interpretable for clinicians
- Regularization prevents overfitting on 503K samples
- Coefficients directly translate to odds ratios for papers

### Data Quality Handling

**Missing Data Strategy:**
- Diagnosis codes: `-9` coded as missing (not NaN)
- Demographics: 28% missing race/ethnicity (9 health centers didn't collect)
- Solution: Created "Unknown" category, sensitivity analysis showed minimal impact on model performance

**Survey Weights:**
- Applied VISWT (visit weights) for national estimates
- Did NOT use weights in model training (would bias toward larger health centers)
- Used weights only for ROI calculations and prevalence estimates

---

## Portfolio Positioning

**Differentiators vs. Typical Projects:**
1. **Real CDC data** (not Kaggle or synthetic)
2. **Population health focus** (not individual prediction)
3. **Health equity analysis** (addresses disparities)
4. **ROI quantification** ($11.7B is concrete, defensible)
5. **Survey methodology** (complex sampling, weighted estimates)

**Target Roles:**
- Clinical Data Scientist (Health Systems, ACOs)
- Healthcare Analytics Consultant (Advisory firms)
- Population Health Analyst (Payers, FQHCs)
- Health Equity Researcher (Academic medical centers)

---

## Metrics Summary

| Metric | Value |
|--------|-------|
| **Data** | 503,799 visits, 107 health centers, 123.8M weighted |
| **Models** | 5 classifiers (RF, GBM, LR) |
| **Best AUC** | 0.934 (multimorbidity), 0.927 (hypertension) |
| **Best F1** | 0.726 (high complexity) |
| **Disparity Ratios** | 1.3-1.4x in chronic disease rates |
| **ROI** | $11.7B savings, 1.48x overall return |
| **Execution Time** | 12 hours (data → models → ROI) |