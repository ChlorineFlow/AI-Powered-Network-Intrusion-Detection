# ML Pipeline

Describes the actual data flow implemented under `ml/src/` and
`ml/scripts/`, in execution order.

Raw dataset files (ml/data/raw/<dataset>/)
│
▼
Loader + schema validator (ml/src/data/loaders/, ml/src/data/validators/)
│ — normalizes column names/encoding quirks per dataset
│ — validates required columns are present before proceeding
▼
Cleaning (ml/src/preprocessing/clean.py)
│ — exact-duplicate removal (CICIDS2017: 10.89% of rows)
│ — invalid-row removal (Infinity/NaN/negative-duration)
▼
Rare-class filtering (multiclass only, min_class_samples threshold)
│
▼
Train/validation/test split (leakage-safe: split BEFORE any fitting)
│ — CICIDS2017: custom stratified 70/10/20 split
│ — NSL-KDD, UNSW-NB15: dataset's own official fixed split
▼
Feature preparation (ml/src/preprocessing/prep.py)
│ — categorical encoding (OneHotEncoder) fit on TRAIN ONLY
│ — scaling (StandardScaler, where used) fit on TRAIN ONLY
▼
Model training (ml/src/models/traditional/, ml/scripts/train.py)
│ — Logistic Regression, Decision Tree, Random Forest,
│ XGBoost, (SVM: CICIDS2017 binary subsample only)
▼
Evaluation (ml/src/evaluation/metrics.py)
│ — full multi-metric computation, never accuracy alone
▼
Experiment logging (ml/src/utils/experiment_logger.py)
│ — every run's config + results → ml/experiments/results/.json
▼
Model serialization (joblib → ml/saved_models/)
│
▼
Explainability (ml/src/explainability/, ml/scripts/explain_.py)
│ — SHAP TreeExplainer, cross-checked with permutation importance
▼
FastAPI inference service (ml/src/inference/app.py)
— loads saved models at startup, serves /predict, /predict/batch,
/explain, /model-info, /metrics


## Key design decisions

- **Leakage prevention is structural, not incidental.** Every
  preprocessing step that could leak test information into training
  (scaling, encoding, feature selection, resampling) is implemented to
  fit exclusively on the training split, enforced by writing separate
  `fit`/`transform` calls rather than fitting on the full dataset before
  splitting.
- **Each dataset has its own loader and prep module** (`cicids2017_loader.py`
  / `nsl_kdd_prep.py` / `unsw_nb15_prep.py`) rather than a single generic
  pipeline, because the three datasets have incompatible schemas,
  encodings, and known dataset-specific quirks (e.g. CICIDS2017's
  Unicode replacement-character label corruption, NSL-KDD's headerless
  format, UNSW-NB15's inverted class balance).
- **Experiment logging is mandatory, not optional** — every training
  script calls `log_experiment()` regardless of whether the result is
  "good," so that negative or unexpected results (e.g. the initial
  unfiltered NSL-KDD multiclass F1 of ~0.44-0.52) are preserved in the
  record rather than silently discarded.