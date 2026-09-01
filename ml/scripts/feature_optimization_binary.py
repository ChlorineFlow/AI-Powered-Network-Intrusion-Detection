"""
Compare All Features vs Selected Features (Random Forest importance) vs
PCA, using XGBoost (the Step 6 winner) as the evaluation model.
Evaluates performance, training time, inference time, and feature count
— per project rule §19. Fitting of selectors/PCA happens on training
data only.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from xgboost import XGBClassifier

from src.features.selection import select_features_by_importance, fit_pca
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config


def load_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    path = processed_dir / f"binary_{split_name}.csv"
    return pd.read_csv(path)


def train_and_eval_xgb(name, X_train, y_train, X_test, y_test,
                        results_dir, random_seed, extra):
    print(f"\n{'#' * 70}\n{name}\n{'#' * 70}")
    print(f"Feature count: {X_train.shape[1]}")

    model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        eval_metric="logloss", random_state=random_seed, n_jobs=-1,
    )

    start_train = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_train

    start_infer = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_infer

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, title=name)
    print(f"Training time: {training_time:.2f}s | Inference time: {inference_time:.2f}s")

    extra["feature_count"] = X_train.shape[1]
    log_experiment(
        results_dir=results_dir,
        dataset="cicids2017",
        task="binary_feature_optimization",
        model_name=f"xgboost_{name.lower().replace(' ', '_')}",
        metrics=metrics,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        random_seed=random_seed,
        extra=extra,
    )


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading processed splits...")
    train_df = load_split(processed_dir, "train")
    test_df = load_split(processed_dir, "test")

    feature_cols = [c for c in train_df.columns if c != "Label"]
    X_train = train_df[feature_cols]
    y_train = (train_df["Label"] != "BENIGN").astype(int)
    X_test = test_df[feature_cols]
    y_test = (test_df["Label"] != "BENIGN").astype(int)

    # ---- 1. All features (baseline for this comparison) ----
    train_and_eval_xgb(
        "All Features", X_train, y_train, X_test, y_test,
        results_dir, random_seed, extra={"method": "all_features"},
    )

    # ---- 2. Selected features (Random Forest importance, median threshold) ----
    print(f"\n{'=' * 70}\nFitting feature selector on training data...\n{'=' * 70}")
    selected_cols, selector = select_features_by_importance(X_train, y_train, random_seed)
    print(f"Selected {len(selected_cols)} of {len(feature_cols)} features:")
    print(selected_cols)

    train_and_eval_xgb(
        "Selected Features", X_train[selected_cols], y_train,
        X_test[selected_cols], y_test,
        results_dir, random_seed,
        extra={"method": "selected_features", "selected_columns": selected_cols},
    )

    # ---- 3. PCA (95% variance) ----
    print(f"\n{'=' * 70}\nFitting PCA on training data...\n{'=' * 70}")
    scaler, pca = fit_pca(X_train, random_seed, n_components=0.95)
    X_train_pca = pca.transform(scaler.transform(X_train))
    X_test_pca = pca.transform(scaler.transform(X_test))
    print(f"PCA reduced {X_train.shape[1]} features to {X_train_pca.shape[1]} "
          f"components (95% variance retained).")

    train_and_eval_xgb(
        "PCA", pd.DataFrame(X_train_pca), y_train, pd.DataFrame(X_test_pca), y_test,
        results_dir, random_seed,
        extra={"method": "pca", "n_components": int(X_train_pca.shape[1])},
    )

    print(f"\n{'=' * 70}\nDone. Compare the three logged experiments under "
          f"{results_dir} (task=binary_feature_optimization) to answer RQ2.")


if __name__ == "__main__":
    main()