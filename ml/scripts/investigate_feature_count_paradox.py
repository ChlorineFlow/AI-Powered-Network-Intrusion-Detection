"""
Investigates why the 5-common-feature 'oracle expert' (F1=0.836) scored
higher than the 122-feature specialized model (F1=0.798) on NSL-KDD
binary classification — a genuinely counter-intuitive result worth
verifying rather than reporting uncritically.

Two candidate explanations, tested separately by holding hyperparameters
FIXED and varying only the feature set:
  A) Hyperparameter difference (5-feature model used max_depth=5,
     n_estimators=150; specialized model used max_depth=8, n_estimators=200)
  B) Genuine feature-set effect (the 117 additional one-hot encoded
     categorical columns may add sparse noise that hurts a deeper tree
     more than the dense 5-feature set does)

Four configurations are trained, holding hyperparameters constant
within each pair, to isolate the effect of feature count alone.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from xgboost import XGBClassifier

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.preprocessing.nsl_kdd_prep import prepare_features, make_binary_label
from src.preprocessing.common_features import extract_nsl_kdd_common
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config


def run(name, X_train, y_train, X_test, y_test, n_estimators, max_depth, random_seed):
    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
        eval_metric="logloss", random_state=random_seed, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"{name:<55} F1 (macro) = {metrics['f1_macro']:.4f}  "
          f"(n_estimators={n_estimators}, max_depth={max_depth})")
    return metrics["f1_macro"]


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    random_seed = config["random_seed"]

    train_df = NSLKDDLoader(raw_dir, "KDDTrain+.txt").load()
    test_df = NSLKDDLoader(raw_dir, "KDDTest+.txt").load()

    # Full 122-feature set (same as the original specialized model)
    X_train_full, X_test_full, _ = prepare_features(train_df, test_df)
    y_train = make_binary_label(train_df)
    y_test = make_binary_label(test_df)

    # 5 common features (same as the oracle expert)
    X_train_common = extract_nsl_kdd_common(train_df)
    X_test_common = extract_nsl_kdd_common(test_df)

    print("=" * 80)
    print("Controlled comparison: feature count held as the ONLY variable per pair")
    print("=" * 80)

    print("\n--- Pair 1: ORIGINAL hyperparameters (max_depth=8, n_estimators=200) ---")
    run("Full 122 features, original hyperparams", X_train_full, y_train, X_test_full, y_test,
        200, 8, random_seed)
    run("5 common features, original hyperparams", X_train_common, y_train, X_test_common, y_test,
        200, 8, random_seed)

    print("\n--- Pair 2: ORACLE-EXPERT hyperparameters (max_depth=5, n_estimators=150) ---")
    run("Full 122 features, oracle-expert hyperparams", X_train_full, y_train, X_test_full, y_test,
        150, 5, random_seed)
    run("5 common features, oracle-expert hyperparams", X_train_common, y_train, X_test_common, y_test,
        150, 5, random_seed)


if __name__ == "__main__":
    main()