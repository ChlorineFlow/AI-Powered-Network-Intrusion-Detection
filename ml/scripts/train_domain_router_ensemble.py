"""
Domain-routing ensemble ("mixture of specialized experts"): rather than
one flat unified model (train_unified_common_features.py), this trains
THREE lightweight experts — one per dataset, each using only the 5
common features but trained exclusively on its own dataset's data —
gated by a learned router that predicts which dataset an incoming
flow's common features most resemble.

All three experts and the router operate in the SAME 5-feature common
space, so a flow can always be routed to any expert regardless of
which dataset it truly came from — unlike the full specialized models
(78/122/194 native columns respectively), which cannot accept each
other's inputs at all.

Three configurations are compared:
  1. Flat unified model (baseline, from train_unified_common_features.py)
  2. Oracle-routed experts (upper bound: route using the TRUE source
     dataset label, never available in real deployment — measures the
     ceiling of what routing could achieve)
  3. Learned-router-routed experts (realistic: route using the router's
     own prediction, which can be wrong)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
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

DATASET_NAMES = ["CICIDS2017", "NSL-KDD", "UNSW-NB15"]


def build_expert(random_seed):
    return XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=random_seed, n_jobs=-1,
    )


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading and extracting common features for all 3 datasets...")

    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    cic_train_raw = pd.read_csv(processed_dir / "binary_train.csv")
    cic_test_raw = pd.read_csv(processed_dir / "binary_test.csv")
    cic_X_train = extract_cicids2017_common(cic_train_raw)
    cic_y_train = (cic_train_raw["Label"] != "BENIGN").astype(int)
    cic_X_test = extract_cicids2017_common(cic_test_raw)
    cic_y_test = (cic_test_raw["Label"] != "BENIGN").astype(int)

    nsl_raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    nsl_train_raw = NSLKDDLoader(nsl_raw_dir, "KDDTrain+.txt").load()
    nsl_test_raw = NSLKDDLoader(nsl_raw_dir, "KDDTest+.txt").load()
    nsl_X_train = extract_nsl_kdd_common(nsl_train_raw)
    nsl_y_train = (nsl_train_raw["label"] != "normal").astype(int)
    nsl_X_test = extract_nsl_kdd_common(nsl_test_raw)
    nsl_y_test = (nsl_test_raw["label"] != "normal").astype(int)

    unsw_raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    unsw_train_raw = UNSWNB15TrainTestLoader(unsw_raw_dir, "UNSW_NB15_training-set.csv").load()
    unsw_test_raw = UNSWNB15TrainTestLoader(unsw_raw_dir, "UNSW_NB15_testing-set.csv").load()
    unsw_X_train = extract_unsw_nb15_common(unsw_train_raw)
    unsw_y_train = unsw_train_raw["label"]
    unsw_X_test = extract_unsw_nb15_common(unsw_test_raw)
    unsw_y_test = unsw_test_raw["label"]

    train_sets = [
        (cic_X_train, cic_y_train), (nsl_X_train, nsl_y_train), (unsw_X_train, unsw_y_train)
    ]
    test_sets = {
        "CICIDS2017": (cic_X_test, cic_y_test),
        "NSL-KDD": (nsl_X_test, nsl_y_test),
        "UNSW-NB15": (unsw_X_test, unsw_y_test),
    }

    # ---- Step 1: Train 3 per-dataset experts (common features only) ----
    print("\n" + "=" * 70)
    print("Training 3 per-dataset EXPERTS (common features only)")
    print("=" * 70)
    experts = {}
    for name, (X_train, y_train) in zip(DATASET_NAMES, train_sets):
        print(f"Training expert for {name} ({len(X_train):,} rows)...")
        expert = build_expert(random_seed)
        expert.fit(X_train, y_train)
        experts[name] = expert

    # ---- Step 2: Train the domain router ----
    print("\n" + "=" * 70)
    print("Training the domain ROUTER (predicts source dataset from common features)")
    print("=" * 70)
    X_router_train = pd.concat([X for X, _ in train_sets], ignore_index=True)
    y_router_train = pd.Series(
        [0] * len(cic_X_train) + [1] * len(nsl_X_train) + [2] * len(unsw_X_train)
    )
    router = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=random_seed, n_jobs=-1,
    )
    start = time.time()
    router.fit(X_router_train, y_router_train)
    router_train_time = time.time() - start
    print(f"Router training time: {router_train_time:.2f}s")

    # ---- Step 3: Evaluate router accuracy on each dataset's own test set ----
    print("\n" + "=" * 70)
    print("Router domain-identification accuracy (per dataset test set)")
    print("=" * 70)
    router_accuracy = {}
    for idx, (name, (X_test, _)) in enumerate(test_sets.items()):
        router_pred = router.predict(X_test)
        acc = (router_pred == idx).mean()
        router_accuracy[name] = acc
        print(f"{name}: router correctly identified source domain for "
              f"{acc:.2%} of rows")

    # ---- Step 4: Evaluate oracle-routed and learned-router-routed ensembles ----
    for name, (X_test, y_test) in test_sets.items():
        print(f"\n{'#' * 70}\nEvaluating on {name} test set\n{'#' * 70}")

        # Oracle: always route to the TRUE expert (upper bound)
        oracle_pred = experts[name].predict(X_test)
        oracle_proba = experts[name].predict_proba(X_test)[:, 1]
        oracle_metrics = compute_metrics(y_test, oracle_pred, oracle_proba)
        print_metrics(oracle_metrics, title=f"ORACLE-routed experts on {name}")

        # Learned router: route based on router's own prediction
        router_pred = router.predict(X_test)
        routed_pred = np.zeros(len(X_test), dtype=int)
        routed_proba = np.zeros(len(X_test))
        for idx, dataset_name in enumerate(DATASET_NAMES):
            mask = router_pred == idx
            if mask.sum() > 0:
                routed_pred[mask] = experts[dataset_name].predict(X_test[mask])
                routed_proba[mask] = experts[dataset_name].predict_proba(X_test[mask])[:, 1]
        routed_metrics = compute_metrics(y_test, routed_pred, routed_proba)
        print_metrics(routed_metrics, title=f"LEARNED-ROUTER-routed experts on {name}")

        log_experiment(
            results_dir=results_dir, dataset=f"domain_routing_{name.lower().replace('-', '_')}",
            task="binary", model_name="router_ensemble",
            metrics=routed_metrics,
            training_time_seconds=router_train_time, inference_time_seconds=0.0,
            random_seed=random_seed,
            extra={
                "router_domain_accuracy": router_accuracy[name],
                "oracle_routed_f1_macro": oracle_metrics["f1_macro"],
                "learned_router_f1_macro": routed_metrics["f1_macro"],
                "note": "Comparison: oracle (perfect routing, upper bound) vs "
                        "learned router (realistic). Both operate on the same "
                        "5 common features used by train_unified_common_features.py.",
            },
        )

    print(f"\n{'=' * 70}\nDone. Compare oracle vs learned-router vs flat-unified-model "
          f"F1 (macro) across all 3 datasets.")


if __name__ == "__main__":
    main()