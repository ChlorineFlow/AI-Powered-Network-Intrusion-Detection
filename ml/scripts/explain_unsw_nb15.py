"""
SHAP explainability for UNSW-NB15's best binary model (Random Forest —
the actual Step 6 winner for this dataset, unlike CICIDS2017/NSL-KDD
where XGBoost won).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.preprocessing.unsw_nb15_prep import prepare_features
from src.explainability.shap_explain import compute_shap_values, top_global_features
from src.utils.config import load_config

SHAP_SAMPLE_SIZE = 2000


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    figures_dir = ml_root / config["paths"]["figures_dir"] / "explainability"
    figures_dir.mkdir(parents=True, exist_ok=True)
    random_seed = config["random_seed"]

    print("Loading model and data...")
    model = joblib.load(saved_models_dir / "unsw_nb15_binary_random_forest.joblib")

    train_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_training-set.csv").load()
    test_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_testing-set.csv").load()
    X_train, X_test, _ = prepare_features(train_df, test_df)

    sample_n = min(SHAP_SAMPLE_SIZE, len(X_test))
    sample_idx = X_test.sample(n=sample_n, random_state=random_seed).index
    X_sample = X_test.loc[sample_idx].reset_index(drop=True)
    print(f"SHAP sample: {X_sample.shape[0]} rows, {X_sample.shape[1]} features")

    print("\nComputing SHAP values (TreeExplainer) — Random Forest, may take a "
          "moment longer than a single tree...")
    shap_values = compute_shap_values(model, X_sample)

    # For binary RandomForestClassifier, shap.TreeExplainer returns
    # values with a class dimension; use the positive-class (attack) slice.
    if shap_values.values.ndim == 3:
        shap_values.values = shap_values.values[:, :, 1]

    print("\n" + "=" * 70)
    print("Top contributing features (global, mean |SHAP value|) — UNSW-NB15")
    print("=" * 70)
    ranking = top_global_features(shap_values, list(X_sample.columns), top_n=15)
    print(ranking.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(ranking["feature"][::-1], ranking["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("UNSW-NB15 — Top contributing features (global, not causal)")
    plt.tight_layout()
    plt.savefig(figures_dir / "unsw_nb15_global_feature_importance.png", dpi=120)
    plt.close()
    print(f"\nSaved: {figures_dir / 'unsw_nb15_global_feature_importance.png'}")


if __name__ == "__main__":
    main()