# Experiment Plan

This document records the experimental design used across this
project's ML pipeline. All experiments described here were executed;
outcomes are in `docs/experiments/results-analysis.md`.

## Datasets

| Dataset | Role | Split source |
|---|---|---|
| CICIDS2017 | Primary | Custom stratified 70/10/20 split, post-cleaning |
| NSL-KDD | Cross-dataset validation | Dataset's own official KDDTrain+/KDDTest+ split |
| UNSW-NB15 | Cross-dataset validation | Dataset's own official training-set/testing-set split |

## Tasks per dataset

- **Binary classification** — benign/normal vs. attack, all three datasets.
- **Multiclass classification** — attack-category classification, all
  three datasets, with a documented minimum-sample threshold (50
  training examples) applied to exclude statistically unreliable rare
  classes (CICIDS2017: 3 classes excluded; NSL-KDD: 11 classes
  excluded; UNSW-NB15: no exclusions needed — all 10 categories
  sufficiently sampled).

## Models evaluated

| Model | Included in | Rationale |
|---|---|---|
| Logistic Regression | All binary tasks | Linear baseline |
| Decision Tree | All binary + multiclass tasks | Non-linear baseline |
| Random Forest | All binary + multiclass tasks | Ensemble comparison |
| XGBoost | All binary + multiclass tasks | Gradient-boosted comparison |
| SVM | CICIDS2017 binary only (50K stratified subsample) | Excluded elsewhere due to poor training/inference scaling at dataset size — a disclosed scope limitation, not an oversight |

## Evaluation protocol

- Preprocessing (deduplication, invalid-row removal, scaling, categorical
  encoding) fit exclusively on training data; applied without refitting
  to validation/test data.
- Metrics: accuracy, precision (macro & weighted), recall (macro &
  weighted), F1 (macro & weighted), ROC-AUC (binary) / ROC-AUC OvR macro
  (multiclass), confusion matrix, and (binary only) false positive rate
  and false negative rate. Accuracy is never reported alone.
- Every experiment run is logged with full configuration (dataset,
  task, model, hyperparameters, random seed, timing) to
  `ml/experiments/results/` as JSON — no result is reported without a
  corresponding logged run.

## Follow-up investigations (triggered by initial results)

1. **Feature optimization** (CICIDS2017): All Features vs.
   Random-Forest-importance-selected features vs. PCA (95% variance),
   evaluated on performance, feature count, and training/inference time.
2. **Explainability**: SHAP TreeExplainer on each dataset's best model,
   cross-validated against permutation importance (CICIDS2017 only, as
   the primary corroboration test).
3. **Leakage investigation** (triggered by CICIDS2017's 99%+ accuracy):
   exact-duplicate removal verification, train/test performance gap
   analysis, single-feature (Destination Port) ablation, near-duplicate
   row overlap detection, cross-method importance verification.
4. **Cross-dataset generalization study**: independent training on
   NSL-KDD and UNSW-NB15, direct comparison against CICIDS2017 results.
5. **Hyperparameter tuning** (NSL-KDD XGBoost): RandomizedSearchCV, 30
   configurations x 5-fold stratified cross-validation (150 fits),
   training data only — to test whether the CICIDS2017-to-NSL-KDD
   performance gap was a fixable configuration issue.
6. **Class-imbalance sanity check**: DummyClassifier (majority-class
   baseline) comparison across all three datasets, using the
   mathematical property that majority-class-only prediction caps macro
   recall at exactly 50%.

## Random seed and reproducibility

All experiments use `random_seed: 42` (set in `ml/config.yaml`) for
train/test splitting, model initialization, and cross-validation folds,
ensuring results are reproducible given the same input data.