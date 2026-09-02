"""
Ablation test: retrain XGBoost WITHOUT Destination Port, to check
whether performance holds up. If performance barely changes, the model
has genuine behavioral signal elsewhere. If performance drops sharply,
it confirms heavy reliance on a feature that is arguably a testbed
artifact (fixed attack-to-port mappings) rather than transferable
attack behavior — directly relevant to RQ5 (cross-dataset generalization).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    train_df = pd.read_csv(processed_dir / "binary_train.csv")
    test_df = pd.read_csv(processed_dir / "binary_test.csv")

    feature_cols = [c for c in train_df.columns if c not in ("Label", "Destination Port")]
    print(f"Training WITHOUT 'Destination Port' — {len(feature_cols)} features "
          f"(vs 78 with it).")

    X_train = train_df[feature_cols]
    y_train = (train_df["Label"] != "BENIGN").astype(int)
    X_test = test_df[feature_cols]
    y_test = (test_df["Label"] != "BENIGN").astype(int)

    model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        eval_metric="logloss", random_state=random_seed, n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, title="XGBoost WITHOUT Destination Port — Test Set")

    log_experiment(
        results_dir=results_dir, dataset="cicids2017", task="binary_ablation",
        model_name="xgboost_no_destination_port", metrics=metrics,
        training_time_seconds=training_time, inference_time_seconds=0.0,
        random_seed=random_seed,
        extra={"note": "Ablation: Destination Port removed to test reliance "
                        "on testbed-specific port-to-attack mapping."},
    )

    print(f"\n{'=' * 60}")
    print("COMPARE with full-feature XGBoost result from Step 6:")
    print("  f1_macro (all 78 features):   0.9986")
    print(f"  f1_macro (without port):      {metrics['f1_macro']:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()