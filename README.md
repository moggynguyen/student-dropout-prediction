# Student Dropout Prediction — Privacy Analysis

**Course:** MMA 609 (Responsible AI), Winter 2026 — Project (Topic 28)
**RAI Lens:** Privacy
**Files:** `Nguyen_Topic28_FinalCode.ipynb` (analysis notebook), `dataset.csv` (data), `Project.pdf` (assignment brief)

## 1. Objective

The project asks a single question:

> *Does a student dropout prediction model actually need sensitive personal data — parental
> background, financial status, demographic information — to accurately identify at-risk
> students, or can it perform just as well using only academic performance data?*

The analysis is framed for a stakeholder who must decide whether to approve a dropout-risk model
for deployment (e.g. a Chief Risk/Privacy Officer or Data Scientist weighing predictive accuracy
against re-identification and discrimination risk). The goal is not just to report accuracy, but
to determine whether removing sensitive features is a **defensible privacy trade-off**.

## 2. Dataset

- **Source:** UCI *Predict Students' Dropout and Academic Success* dataset — 4,424 students at a
  Portuguese higher-education institution, 35 features captured at enrollment (demographics,
  socioeconomic background, financial situation, academic performance, macroeconomic context).
- **Target:** `Target` ∈ {Graduate, Dropout, Enrolled}. Students still `Enrolled` (no final
  outcome yet) are dropped, leaving a **binary classification problem**: Graduate (60.9%, n=2,209)
  vs. Dropout (39.1%, n=1,421) — a moderate 1.55:1 class imbalance.
- **No missing values**, all columns numeric (no categorical encoding needed).

### Feature sensitivity classification (three-tier framework)

Grounded in GDPR Article 9, FERPA, and NIST guidance, every feature is assigned a sensitivity tier:

| Tier | Count | Examples | Rationale |
|---|---|---|---|
| **Tier 1 — High** | 12 | Marital status, Gender, Age, Nationality, Displaced, International, Debtor, Mother's/Father's qualification & occupation, Educational special needs | Legally protected characteristics, financial hardship, family-background profiling |
| **Tier 2 — Moderate** | 6 | Scholarship holder, Tuition fees up to date, Application mode/order, Previous qualification, Course | Reveals financial need or academic-entry context |
| **Tier 3 — Low** | 16 | Curricular units (1st/2nd sem: enrolled, approved, grade, evaluations, credited), Daytime/evening attendance, Unemployment rate, Inflation rate, GDP | Academic activity/performance and external macro factors only |

## 3. Methodology

### 3.1 Two-run experimental design

Every model is trained **twice** on an identical 75/25 stratified train/test split (`random_state=2025`):

- **Run 1 — All Features:** all 34 features (full personal + academic + macro data).
- **Run 2 — Low Sensitivity Only:** the 16 Tier-3 features only (academic activity + macro
  indicators — no demographic, financial, or family-background data).

### 3.2 Models

Three classifiers, each tuned with `GridSearchCV` (5-fold CV, `scoring='roc_auc'`), inside a
`StandardScaler` pipeline:

| Model | Key hyperparameters searched |
|---|---|
| **Logistic Regression** | `C` ∈ {0.1, 1, 10}, penalty ∈ {l1, l2}, `class_weight='balanced'` |
| **Random Forest** | `n_estimators` ∈ {100, 200}, `max_depth` ∈ {None, 6, 10}, `min_samples_leaf` ∈ {1, 2, 4}, `class_weight='balanced'` |
| **XGBoost** | `n_estimators` ∈ {50, 100, 150}, `max_depth` ∈ {3, 5, 7}, `learning_rate` ∈ {0.01, 0.1, 0.2}, `scale_pos_weight` set to the class ratio |

### 3.3 Evaluation dimensions

1. **Performance comparison** — Accuracy, AUC, weighted F1, recall (Run 1 vs Run 2, all 3 models).
2. **Overfitting analysis** — train vs. test accuracy gap and learning curves.
3. **ROC curves** for all 6 model/run combinations.
4. **Feature importance** — coefficients (LR) / impurity importance (RF, XGB), color-coded by
   sensitivity tier.
5. **Privacy Drop Analysis** — direct AUC/recall/accuracy delta between Run 1 and Run 2 per model,
   with a **3% AUC-drop acceptability threshold**.
6. **Privacy-Utility Curve** — a 5-stage progressive feature-removal sweep (All → drop Tier 1 →
   drop Tier 1+2 → drop Tier 1+2+macro → core academic only) tracking AUC and Dropout recall at
   each stage, for all three models and in a dedicated deep-dive on Logistic Regression (the best
   performer).
7. **Confusion matrices & dropout miss-rate** — how many actual dropouts each run fails to flag.
8. **Cross-validation summary** — 5-fold CV AUC mean ± std for every model/run.
9. **Threshold analysis** — precision/recall/F1/accuracy of the best model (Logistic Regression)
   at decision thresholds 0.3–0.7, since missing an at-risk student (false negative) is more
   costly than a false alarm in this use case.
10. **Formal anonymization analysis (k-anonymity, ℓ-diversity, t-closeness)** — quantifies
    re-identification risk directly, independent of any ML model, following Sweeney (2002) and
    Machanavajjhala et al. (2007). Three quasi-identifier sets are tested: Tier 1+2 (18 features),
    Tier 1 only (12), and a small representative subset (4: Age, Gender, Nationality, Marital
    status).
11. **Prediction-level comparison** — sample-level and full-test-set agreement between the Run 1
    and Run 2 Logistic Regression models, and a plain-language "core finding" translation.

## 4. Key Results

### 4.1 Model performance — Run 1 (all features) vs Run 2 (low sensitivity only)

| Run | Model | Accuracy | AUC | F1 |
|---|---|---|---|---|
| Run 1 | Logistic Regression | 0.912 | **0.957** | 0.912 |
| Run 1 | Random Forest | 0.905 | 0.955 | 0.905 |
| Run 1 | XGBoost | 0.910 | 0.952 | 0.909 |
| Run 2 | Logistic Regression | 0.890 | **0.940** | 0.890 |
| Run 2 | Random Forest | 0.882 | 0.937 | 0.881 |
| Run 2 | XGBoost | 0.881 | 0.935 | 0.881 |

**Logistic Regression is the top performer in both runs** and is used for all deep-dive analyses
(threshold tuning, privacy-utility curve, prediction comparison).

Random Forest overfits noticeably (train accuracy 0.981 vs test 0.905 in Run 1, a 7.6-point gap);
Logistic Regression generalizes best (near-zero train/test gap in both runs).

### 4.2 Privacy Drop Analysis (removing all Tier 1 + Tier 2 features)

| Model | Run 1 AUC | Run 2 AUC | AUC Drop | Verdict |
|---|---|---|---|---|
| Logistic Regression | 0.9573 | 0.9399 | −0.0174 | Acceptable |
| Random Forest | 0.9546 | 0.9373 | −0.0173 | Acceptable |
| XGBoost | 0.9521 | 0.9348 | −0.0173 | Acceptable |

All three models lose **less than 2% AUC** when every sensitive/moderate feature is removed —
well inside the 3% acceptability threshold set for this analysis.

### 4.3 Privacy-Utility Curve (Logistic Regression, progressive removal)

| Stage | Features used | AUC | Dropout Recall |
|---|---|---|---|
| Stage 0 — All features (36) | 34 | 0.9573 | 0.8901 |
| Stage 1 — Drop Tier 1 | 22 | 0.9548 | 0.8930 |
| Stage 2 — Drop Tier 1+2 | 16 | 0.9384 | 0.8563 |
| Stage 3 — Drop Tier 1+2 + macro | 12 | 0.9374 | 0.8563 |
| Stage 4 — Core academic only | 6 | 0.9376 | 0.8592 |

Most of the drop happens between Stage 1 and Stage 2 (removing Tier 2 / moderate features);
after that, AUC is essentially flat even as features shrink from 16 down to 6 — i.e. the bulk of
predictive power lives in a small set of core academic indicators, not in demographic or
financial data.

### 4.4 Formal privacy analysis (k-anonymity / ℓ-diversity / t-closeness)

| Quasi-identifier set | Records retained (k=5) | 5-Anonymous? | 2-Diverse? | t-Close (t=0.2)? |
|---|---|---|---|---|
| Tier 1+2 (18 features) | 0 (0.0%) | No | N/A (fully suppressed) | N/A |
| Tier 1 only (12 features) | 548 (15.1%) | No | No | No (max=0.723) |
| Small subset (4 features) | 3,535 (97.4%) | Yes | No | No (max=0.421) |

**Even the smallest, most heavily generalized quasi-identifier set (4 features) fails
ℓ-diversity and t-closeness** — some equivalence classes are 100% Dropout or 100% Graduate,
meaning an adversary who knows a student's age band, gender, nationality, and marital status can
infer their dropout outcome with high confidence, without ever touching the model
(a background-knowledge attack, per Machanavajjhala et al. 2007). Anonymization cannot fix this;
the sensitive features have to not be collected in the first place.

### 4.5 The core finding in plain numbers (Logistic Regression, test set)

- Total at-risk (Dropout) students in test set: **355**
- Caught using **all data** (Run 1): 316 (89.0%)
- Caught using **no sensitive data** (Run 2): 301 (84.8%)
- **Extra students missed by removing sensitive data: 15 out of 355**
- Overall prediction agreement between the two models: **93.0%**

### 4.6 Threshold recommendation

Because missing an at-risk student is costlier than a false alarm, lowering the Logistic
Regression decision threshold from the default 0.5 to **0.4** meaningfully improves Dropout
recall in both runs (e.g. Run 1: 89.0% → 90.7% recall) at an acceptable precision cost.

## 5. Conclusion

Across every lens tested — model performance, formal anonymization, and prediction-level
agreement — the evidence converges on the same answer: **the model does not need sensitive
personal data to work well.**

- Dropping all Tier 1 (high-sensitivity) and Tier 2 (moderate-sensitivity) features costs under
  2% AUC and misses only 15 additional at-risk students out of 355 in the test set.
- Those same sensitive features cannot be safely anonymized: k-anonymity requires suppressing the
  entire dataset at 18 quasi-identifiers, and even the smallest tested subset fails ℓ-diversity
  and t-closeness, leaving students open to outcome-inference attacks.
- Therefore, **feature exclusion at the pipeline level** (training only on academic-activity and
  macroeconomic features, as in Run 2) is the approach that resolves both the privacy risk and
  the utility requirement simultaneously — anonymization alone cannot.

**Recommendation to the stakeholder:** deploy the dropout-risk model using only the low-sensitivity
(Tier 3) feature set, with Logistic Regression as the production model (best AUC and best
train/test generalization) and a decision threshold of ~0.4 to prioritize catching at-risk
students. Do not collect or retain Tier 1/Tier 2 fields for this use case — the marginal accuracy
gain does not justify the re-identification and discrimination exposure.

## 6. Notebook Structure

1. **Data Exploration** — load, class balance, correlation heatmap, feature distributions,
   correlation ranking by sensitivity tier.
2. **Building Models** — train/test split, preprocessing checks, Run 1 / Run 2 training for
   Logistic Regression, Random Forest, and XGBoost.
3. **Evaluation** — performance comparison, overfitting/learning curves, ROC curves, feature
   importance, privacy drop analysis, privacy-utility curve, confusion matrices, cross-validation
   summary, threshold analysis, and the formal k-anonymity/ℓ-diversity/t-closeness study.
4. **Prediction-Level Comparison** — sample predictions, full test-set agreement, risk-score
   distributions, and the plain-numbers summary of the privacy/utility trade-off.
