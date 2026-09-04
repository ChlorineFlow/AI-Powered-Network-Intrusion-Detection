"""
Retrains and PERSISTS (via joblib) the models built in the domain-
routing and unified-model experiments, so they can be loaded by the
FastAPI service. The original experiment scripts only logged metrics —
this script trains the same configurations and saves the actual model
artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.preprocessing.common_features import (
    extract_cicids2017_common,
    extract_nsl_kdd_common,
    extract_unsw_nb15_common,
)
from src.utils.config import load_config

DATASET_NAMES = ["cicids2017", "nsl_kdd", "unsw_nb15"]


def build_expert(seed):
    return XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=seed, n_jobs=-1,
    )


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    random_seed = config["random_seed"]

    print("Loading and extracting common features...")
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    cic_train_raw = pd.read_csv(processed_dir / "binary_train.csv")
    cic_X_train = extract_cicids2017_common(cic_train_raw)
    cic_y_train = (cic_train_raw["Label"] != "BENIGN").astype(int)

    nsl_raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    nsl_train_raw = NSLKDDLoader(nsl_raw_dir, "KDDTrain+.txt").load()
    nsl_X_train = extract_nsl_kdd_common(nsl_train_raw)
    nsl_y_train = (nsl_train_raw["label"] != "normal").astype(int)

    unsw_raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    unsw_train_raw = UNSWNB15TrainTestLoader(unsw_raw_dir, "UNSW_NB15_training-set.csv").load()
    unsw_X_train = extract_unsw_nb15_common(unsw_train_raw)
    unsw_y_train = unsw_train_raw["label"]

    train_sets = {
        "cicids2017": (cic_X_train, cic_y_train),
        "nsl_kdd": (nsl_X_train, nsl_y_train),
        "unsw_nb15": (unsw_X_train, unsw_y_train),
    }

    # ---- Train and save the 3 experts ----
    experts = {}
    for name, (X_train, y_train) in train_sets.items():
        print(f"Training + saving expert: {name}")
        expert = build_expert(random_seed)
        expert.fit(X_train, y_train)
        joblib.dump(expert, saved_models_dir / f"expert_common_{name}.joblib")
        experts[name] = expert

    # ---- Train and save the router ----
    print("Training + saving domain router...")
    X_router_train = pd.concat([X for X, _ in train_sets.values()], ignore_index=True)
    y_router_train = pd.Series(
        [0] * len(cic_X_train) + [1] * len(nsl_X_train) + [2] * len(unsw_X_train)
    )
    router = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss", random_state=random_seed, n_jobs=-1,
    )
    router.fit(X_router_train, y_router_train)
    joblib.dump(router, saved_models_dir / "domain_router.joblib")

    # ---- Train and save the flat unified model ----
    print("Training + saving flat unified model...")
    unified = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=random_seed, n_jobs=-1,
    )
    unified.fit(X_router_train, pd.concat(
        [y for _, y in train_sets.values()], ignore_index=True
    ))
    joblib.dump(unified, saved_models_dir / "unified_common_features_xgboost.joblib")

    print(f"\nAll models saved to: {saved_models_dir}")
    print("Files created: expert_common_cicids2017.joblib, "
          "expert_common_nsl_kdd.joblib, expert_common_unsw_nb15.joblib, "
          "domain_router.joblib, unified_common_features_xgboost.joblib")


if __name__ == "__main__":
    main()