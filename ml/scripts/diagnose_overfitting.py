"""
Diagnostic: compare train-set vs test-set performance for the saved
XGBoost binary model. A large gap (near-perfect train, much lower test)
would indicate overfitting/memorization. A SMALL gap (both near-perfect)
does NOT rule out the near-duplicate-flow leakage concern discussed in
docs/research/methodology.md — it just means the model isn't classically
overfitting in the train-vs-test-accuracy sense. That's a separate,
harder problem: near-identical flows split across train/test, and
feature-level artifacts like Destination Port encoding testbed setup
rather than true attack behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.config import load_config


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]

    model_path = saved_models_dir / "cicids2017_binary_xgboost.joblib"
    pipeline = joblib.load(model_path)

    train_df = pd.read_csv(processed_dir / "binary_train.csv")
    test_df = pd.read_csv(processed_dir / "binary_test.csv")

    feature_cols = [c for c in train_df.columns if c != "Label"]

    X_train = train_df[feature_cols]
    y_train = (train_df["Label"] != "BENIGN").astype(int)
    X_test = test_df[feature_cols]
    y_test = (test_df["Label"] != "BENIGN").astype(int)

    print("Evaluating on TRAINING set (same data the model was fit on)...")
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    print_metrics(train_metrics, title="XGBoost — TRAIN set performance")

    print("\nEvaluating on TEST set (held-out data)...")
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    print_metrics(test_metrics, title="XGBoost — TEST set performance")

    gap = train_metrics["f1_macro"] - test_metrics["f1_macro"]
    print(f"\n{'=' * 60}")
    print(f"F1 (macro) gap (train - test): {gap:.4f}")
    print(f"{'=' * 60}")
    if gap < 0.01:
        print("Gap is very small. This means the model is NOT classically "
              "overfitting (memorizing training rows and failing on new "
              "ones). However, this does NOT rule out the near-duplicate-flow "
              "and feature-artifact concerns — those would make train AND "
              "test look equally good, since similar patterns exist on "
              "both sides of the split. See the Destination Port ablation "
              "test for that concern specifically.")
    else:
        print("Meaningful gap detected — suggests some degree of overfitting "
              "to training-specific patterns.")


if __name__ == "__main__":
    main()