"""
SHAP explainability for NSL-KDD's XGBoost binary model. Mirrors
explain_binary.py's methodology for CICIDS2017.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.preprocessing.nsl_kdd_prep import prepare_features, make_binary_label
from src.explainability.shap_explain import compute_shap_values, top_global_features
from src.utils.config import load_config

SHAP_SAMPLE_SIZE = 2000


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    figures_dir = ml_root / config["paths"]["figures_dir"] / "explainability"
    figures_dir.mkdir(parents=True, exist_ok=True)
    random_seed = config["random_seed"]

    print("Loading model and data...")
    model = joblib.load(saved_models_dir / "nsl_kdd_binary_xgboost.joblib")

    train_df = NSLKDDLoader(raw_dir, "KDDTrain+.txt").load()
    test_df = NSLKDDLoader(raw_dir, "KDDTest+.txt").load()
    X_train, X_test, _ = prepare_features(train_df, test_df)
    y_test = make_binary_label(test_df)

    sample_n = min(SHAP_SAMPLE_SIZE, len(X_test))
    sample_idx = X_test.sample(n=sample_n, random_state=random_seed).index
    X_sample = X_test.loc[sample_idx].reset_index(drop=True)
    print(f"SHAP sample: {X_sample.shape[0]} rows, {X_sample.shape[1]} features")

    print("\nComputing SHAP values (TreeExplainer)...")
    shap_values = compute_shap_values(model, X_sample)

    print("\n" + "=" * 70)
    print("Top contributing features (global, mean |SHAP value|) — NSL-KDD")
    print("=" * 70)
    ranking = top_global_features(shap_values, list(X_sample.columns), top_n=15)
    print(ranking.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(ranking["feature"][::-1], ranking["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("NSL-KDD — Top contributing features (global, not causal)")
    plt.tight_layout()
    plt.savefig(figures_dir / "nsl_kdd_global_feature_importance.png", dpi=120)
    plt.close()
    print(f"\nSaved: {figures_dir / 'nsl_kdd_global_feature_importance.png'}")


if __name__ == "__main__":
    main()