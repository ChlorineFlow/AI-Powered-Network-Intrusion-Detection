"""
Run SHAP explainability on the full-feature XGBoost binary model
(the best-performing model from Steps 6-7). Computes:
  1. Global feature importance ranking (top contributing features overall)
  2. A per-flow explanation for a few individual predictions, including
     at least one true attack and one false negative, to answer
     "why did the model classify this traffic as malicious?"

SHAP is computed on a sample of the test set, not all 504K rows, since
SHAP computation cost scales with sample count even for the exact
TreeExplainer. This is a standard, disclosed practice for large datasets.
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

from src.explainability.shap_explain import (
    compute_shap_values,
    top_global_features,
    explain_single_prediction,
)
from src.utils.config import load_config

SHAP_SAMPLE_SIZE = 2000  # representative sample of the test set


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    figures_dir = ml_root / config["paths"]["figures_dir"] / "explainability"
    figures_dir.mkdir(parents=True, exist_ok=True)
    random_seed = config["random_seed"]

    # The Step 6 XGBoost model was saved as a Pipeline; unwrap the
    # underlying classifier since TreeExplainer needs the raw booster.
    model_path = saved_models_dir / "cicids2017_binary_xgboost.joblib"
    print(f"Loading model: {model_path}")
    pipeline = joblib.load(model_path)
    xgb_model = pipeline.named_steps["clf"]

    print("Loading test split...")
    test_df = pd.read_csv(processed_dir / "binary_test.csv")
    feature_cols = [c for c in test_df.columns if c != "Label"]
    y_test = (test_df["Label"] != "BENIGN").astype(int)

    # Stratified-ish sample: take a random sample, but ensure some
    # attack rows are present so we can show an attack explanation.
    rng = np.random.RandomState(random_seed)
    attack_idx = test_df[y_test == 1].sample(
        n=min(500, (y_test == 1).sum()), random_state=random_seed
    ).index
    benign_idx = test_df[y_test == 0].sample(
        n=SHAP_SAMPLE_SIZE - len(attack_idx), random_state=random_seed
    ).index
    sample_idx = attack_idx.union(benign_idx)

    X_sample = test_df.loc[sample_idx, feature_cols].reset_index(drop=True)
    y_sample = y_test.loc[sample_idx].reset_index(drop=True)
    print(f"SHAP sample: {X_sample.shape[0]} rows "
          f"({(y_sample == 1).sum()} attack, {(y_sample == 0).sum()} benign)")

    print("\nComputing SHAP values (TreeExplainer)...")
    shap_values = compute_shap_values(xgb_model, X_sample)

    # ---- 1. Global feature importance ----
    print("\n" + "=" * 70)
    print("Top contributing features (global, mean |SHAP value|)")
    print("=" * 70)
    global_ranking = top_global_features(shap_values, feature_cols, top_n=15)
    print(global_ranking.to_string(index=False))

    plt.figure(figsize=(8, 6))
    plt.barh(global_ranking["feature"][::-1], global_ranking["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value| (impact on model output)")
    plt.title("Top contributing features — global (not causal)")
    plt.tight_layout()
    plt.savefig(figures_dir / "global_feature_importance.png", dpi=120)
    plt.close()
    print(f"Saved: {figures_dir / 'global_feature_importance.png'}")

    # ---- 2. Individual prediction explanations ----
    predictions = xgb_model.predict(X_sample)

    # Find one true positive (correctly caught attack) and one false
    # negative (missed attack), if available, to illustrate both cases.
    true_positive_candidates = X_sample[(y_sample == 1) & (predictions == 1)].index
    false_negative_candidates = X_sample[(y_sample == 1) & (predictions == 0)].index

    if len(true_positive_candidates) > 0:
        idx = true_positive_candidates[0]
        print("\n" + "=" * 70)
        print(f"Example: correctly detected attack (sample index {idx})")
        print("=" * 70)
        print("Top contributing features for this prediction:")
        print(explain_single_prediction(shap_values, idx, feature_cols).to_string(index=False))

    if len(false_negative_candidates) > 0:
        idx = false_negative_candidates[0]
        print("\n" + "=" * 70)
        print(f"Example: MISSED attack — false negative (sample index {idx})")
        print("=" * 70)
        print("Top contributing features for this prediction:")
        print(explain_single_prediction(shap_values, idx, feature_cols).to_string(index=False))
    else:
        print("\nNo false negatives found in this sample — model caught all "
              "sampled attacks. This does not mean zero false negatives "
              "overall (see the full-test-set confusion matrix from Step 6).")

    print(f"\nDone. Figures saved under: {figures_dir}")


if __name__ == "__main__":
    main()