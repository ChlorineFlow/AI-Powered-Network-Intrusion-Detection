"""
Exploratory experiment: train ONE binary model on a combined dataset
built from a small, honestly-limited common feature subset shared
across CICIDS2017, NSL-KDD, and UNSW-NB15, then evaluate it separately
on each dataset's own held-out test set (plus a combined test set).

This directly tests the hypothesis raised after finding "no single
model wins across all three datasets": can ONE unified model achieve
consistent, if lower, performance across all three, as a trade-off
against the specialized per-dataset models already trained in this
project? This is an exploratory experiment using a hand-picked common
feature subset (see common_features.py for its documented limitations),
not a claim of solving cross-dataset NIDS generalization — see
docs/research/methodology.md for full framing and related work
(Sarhan et al. 2021's NetFlow-standardized datasets, which notably do
NOT cover this exact three-dataset combination).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.preprocessing.common_features import (
    extract_cicids2017_common,
    extract_nsl_kdd_common,
    extract_unsw_nb15_common,
    COMMON_FEATURE_NAMES,
)
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print(f"Common feature set used ({len(COMMON_FEATURE_NAMES)} features): "
          f"{COMMON_FEATURE_NAMES}")
    print("NOTE: packet-count features excluded — NSL-KDD does not provide "
          "a per-connection packet count comparable to the other two "
          "datasets. See common_features.py for full rationale.\n")

    # ---- Load and extract common features per dataset ----
    print("Loading CICIDS2017...")
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    cic_train_raw = pd.read_csv(processed_dir / "binary_train.csv")
    cic_test_raw = pd.read_csv(processed_dir / "binary_test.csv")
    cic_X_train = extract_cicids2017_common(cic_train_raw)
    cic_y_train = (cic_train_raw["Label"] != "BENIGN").astype(int)
    cic_X_test = extract_cicids2017_common(cic_test_raw)
    cic_y_test = (cic_test_raw["Label"] != "BENIGN").astype(int)

    print("Loading NSL-KDD...")
    nsl_raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    nsl_train_raw = NSLKDDLoader(nsl_raw_dir, "KDDTrain+.txt").load()
    nsl_test_raw = NSLKDDLoader(nsl_raw_dir, "KDDTest+.txt").load()
    nsl_X_train = extract_nsl_kdd_common(nsl_train_raw)
    nsl_y_train = (nsl_train_raw["label"] != "normal").astype(int)
    nsl_X_test = extract_nsl_kdd_common(nsl_test_raw)
    nsl_y_test = (nsl_test_raw["label"] != "normal").astype(int)

    print("Loading UNSW-NB15...")
    unsw_raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    unsw_train_raw = UNSWNB15TrainTestLoader(unsw_raw_dir, "UNSW_NB15_training-set.csv").load()
    unsw_test_raw = UNSWNB15TrainTestLoader(unsw_raw_dir, "UNSW_NB15_testing-set.csv").load()
    unsw_X_train = extract_unsw_nb15_common(unsw_train_raw)
    unsw_y_train = unsw_train_raw["label"]
    unsw_X_test = extract_unsw_nb15_common(unsw_test_raw)
    unsw_y_test = unsw_test_raw["label"]

    # ---- Combine training data from all three ----
    X_train_combined = pd.concat([cic_X_train, nsl_X_train, unsw_X_train], ignore_index=True)
    y_train_combined = pd.concat([cic_y_train, nsl_y_train, unsw_y_train], ignore_index=True)
    print(f"\nCombined training set: {X_train_combined.shape[0]:,} rows "
          f"({len(cic_X_train):,} CICIDS2017 + {len(nsl_X_train):,} NSL-KDD + "
          f"{len(unsw_X_train):,} UNSW-NB15)")
    print(f"Combined training label distribution:\n{y_train_combined.value_counts()}")

    # ---- Train ONE unified model ----
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight="balanced",
            random_state=random_seed, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=random_seed, n_jobs=-1,
        ),
    }

    test_sets = {
        "CICIDS2017": (cic_X_test, cic_y_test),
        "NSL-KDD": (nsl_X_test, nsl_y_test),
        "UNSW-NB15": (unsw_X_test, unsw_y_test),
    }
    X_test_combined = pd.concat([cic_X_test, nsl_X_test, unsw_X_test], ignore_index=True)
    y_test_combined = pd.concat([cic_y_test, nsl_y_test, unsw_y_test], ignore_index=True)

    for model_name, model in models.items():
        print(f"\n{'#' * 70}\nTraining UNIFIED model: {model_name}\n{'#' * 70}")

        start = time.time()
        model.fit(X_train_combined, y_train_combined)
        training_time = time.time() - start
        print(f"Training time: {training_time:.2f}s")

        # Evaluate on each dataset's own test set separately
        per_dataset_results = {}
        for name, (X_test, y_test) in test_sets.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(y_test, y_pred, y_proba)
            per_dataset_results[name] = metrics
            print_metrics(metrics, title=f"Unified {model_name} evaluated on {name} test set")

        # Evaluate on the combined test set
        y_pred_combined = model.predict(X_test_combined)
        y_proba_combined = model.predict_proba(X_test_combined)[:, 1]
        combined_metrics = compute_metrics(y_test_combined, y_pred_combined, y_proba_combined)
        print_metrics(combined_metrics, title=f"Unified {model_name} evaluated on COMBINED test set")

        log_experiment(
            results_dir=results_dir, dataset="unified_common_features",
            task="binary", model_name=model_name,
            metrics=combined_metrics,
            training_time_seconds=training_time, inference_time_seconds=0.0,
            random_seed=random_seed,
            extra={
                "common_features": COMMON_FEATURE_NAMES,
                "per_dataset_metrics": {k: v for k, v in per_dataset_results.items()},
                "note": "Exploratory unified model trained on a hand-picked "
                        "common feature subset across all 3 datasets combined. "
                        "See docs/research/methodology.md for limitations.",
            },
        )

    print(f"\n{'=' * 70}\nDone. Compare per-dataset F1 (macro) above against each "
          f"dataset's SPECIALIZED model results in "
          f"docs/experiments/results-analysis.md.")


if __name__ == "__main__":
    main()