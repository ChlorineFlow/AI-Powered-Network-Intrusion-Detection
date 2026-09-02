"""
Cross-check SHAP's global feature ranking (Step 10) against permutation
importance — an independent method that measures the actual performance
drop when each feature is randomly shuffled. Strong agreement between
the two methods is evidence that the model's reliance on these features
is genuine and not a quirk of one particular importance technique.

Run on a subsample of the test set since permutation importance requires
re-scoring the model many times (n_repeats x n_features evaluations).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from sklearn.inspection import permutation_importance

from src.utils.config import load_config

SAMPLE_SIZE = 20_000
N_REPEATS = 5


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    random_seed = config["random_seed"]

    print("Loading model and test data...")
    pipeline = joblib.load(saved_models_dir / "cicids2017_binary_xgboost.joblib")
    test_df = pd.read_csv(processed_dir / "binary_test.csv")

    feature_cols = [c for c in test_df.columns if c != "Label"]
    y_test = (test_df["Label"] != "BENIGN").astype(int)

    sample_df = test_df.sample(n=SAMPLE_SIZE, random_state=random_seed)
    X_sample = sample_df[feature_cols]
    y_sample = (sample_df["Label"] != "BENIGN").astype(int)

    print(f"Computing permutation importance on {SAMPLE_SIZE:,} test rows "
          f"({N_REPEATS} repeats per feature — this may take a few minutes)...")

    result = permutation_importance(
        pipeline, X_sample, y_sample,
        n_repeats=N_REPEATS, random_state=random_seed,
        scoring="f1_macro", n_jobs=-1,
    )

    ranking = pd.DataFrame({
        "feature": feature_cols,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)

    print(f"\n{'=' * 70}")
    print("Top 15 features by PERMUTATION importance")
    print(f"{'=' * 70}")
    print(ranking.head(15).to_string(index=False))

    print(f"\n{'=' * 70}")
    print("COMPARE with Step 10 SHAP global ranking (top 6 were):")
    print("  Bwd Packet Length Std, Init_Win_bytes_backward, Destination Port,")
    print("  Init_Win_bytes_forward, Bwd Packet Length Mean, Average Packet Size")
    print(f"{'=' * 70}")

    top_shap = {"Bwd Packet Length Std", "Init_Win_bytes_backward", "Destination Port",
                "Init_Win_bytes_forward", "Bwd Packet Length Mean", "Average Packet Size"}
    top_perm = set(ranking.head(6)["feature"])
    agreement = top_shap & top_perm
    print(f"\nOverlap between SHAP top-6 and permutation-importance top-6: "
          f"{len(agreement)}/6 features match: {agreement}")


if __name__ == "__main__":
    main()