# ml/ — Machine Learning Component

Everything related to datasets, preprocessing, feature engineering, model
training/evaluation, explainability, and the FastAPI inference service lives
here. No frontend or backend application code belongs in this folder.

## Status

Scaffolding only. No models trained yet, no data included.

## Setup (once dependencies are added in a later phase)

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

- `data/` — raw and processed datasets (not committed; see `data/README.md`)
- `notebooks/` — EDA and experimentation notebooks
- `src/` — pipeline source code (data, preprocessing, features, models,
  training, evaluation, explainability, inference, utils)
- `experiments/` — experiment configs, logs, and results
- `saved_models/` — serialized trained models (not committed)
- `figures/` — generated plots for reports/research
- `tests/` — unit tests for the ML pipeline

## Development order

1. Dataset loaders & validators
2. EDA
3. Preprocessing pipeline (leakage-safe: split before fit)
4. Baseline models (Logistic Regression, Decision Tree)
5. Advanced models (Random Forest, XGBoost, SVM)
6. Feature selection / optimization
7. Evaluation (accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
   FPR/FNR, macro & weighted averages)
8. Explainability (SHAP, feature importance, permutation importance)
9. FastAPI inference service

Deep learning models are out of scope for the current phase — classical ML
only.
