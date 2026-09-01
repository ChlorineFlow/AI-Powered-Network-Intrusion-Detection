"""
Train and evaluate advanced models (Random Forest, XGBoost) on the full
binary training set, and SVM on a stratified subsample.

SVM subsample justification: SVC training time scales poorly with
dataset size (roughly O(n^2) to O(n^3)). On 1.76M rows this is
impractical to run to completion. A stratified subsample preserves the
class balance while making training feasible, and this limitation is
reported explicitly rather than silently training on less data without
disclosure.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.traditional.advanced import ADVANCED_MODELS, SVM_MODEL_BUILDER
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config

SVM_SUBSAMPLE_SIZE = 50_000  # stratified sample size for SVM only


def load_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    path = processed_dir / f"binary_{split_name}.csv"
    return pd.read_csv(path)


def run_model(name, pipeline, X_train, y_train, X_test, y_test,
              results_dir, saved_models_dir, random_seed, extra=None):
    print(f"\n{'#' * 70}\nTraining: {name}\n{'#' * 70}")

    start_train = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_train
    print(f"Training time: {training_time:.2f}s")

    start_infer = time.time()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_infer
    print(f"Inference time on {len(X_test):,} rows: {inference_time:.2f}s")

    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, title=f"{name} — Binary Classification (Test Set)")

    log_experiment(
        results_dir=results_dir,
        dataset="cicids2017",
        task="binary",
        model_name=name,
        metrics=metrics,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        random_seed=random_seed,
        extra=extra,
    )

    model_path = saved_models_dir / f"cicids2017_binary_{name}.joblib"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to: {model_path}")


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
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

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ---- Random Forest + XGBoost on full training data ----
    for name, builder in ADVANCED_MODELS.items():
        pipeline = builder(random_seed)
        run_model(name, pipeline, X_train, y_train, X_test, y_test,
                   results_dir, saved_models_dir, random_seed)

    # ---- SVM on stratified subsample (documented limitation) ----
    print(f"\n{'!' * 70}")
    print(f"SVM: using a stratified subsample of {SVM_SUBSAMPLE_SIZE:,} rows "
          f"from the training set (full-dataset SVM training is "
          f"computationally impractical at 1.76M rows). This is a "
          f"documented scope limitation, not a silent shortcut.")
    print(f"{'!' * 70}")

    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train,
        train_size=SVM_SUBSAMPLE_SIZE,
        stratify=y_train,
        random_state=random_seed,
    )
    print(f"SVM training subsample: {X_train_sub.shape}, "
          f"label distribution:\n{y_train_sub.value_counts()}")

    svm_pipeline = SVM_MODEL_BUILDER(random_seed)
    run_model(
        "svm_subsampled", svm_pipeline, X_train_sub, y_train_sub, X_test, y_test,
        results_dir, saved_models_dir, random_seed,
        extra={
            "note": "Trained on stratified subsample due to SVM computational "
                     "cost on large datasets — see docs/research/methodology.md",
            "subsample_size": SVM_SUBSAMPLE_SIZE,
            "full_train_size": len(X_train),
        },
    )


if __name__ == "__main__":
    main()