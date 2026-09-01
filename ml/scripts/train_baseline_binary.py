"""
Train and evaluate baseline models (Logistic Regression, Decision Tree)
on the binary classification task (BENIGN vs ATTACK), using the
leakage-safe splits saved by preprocess_cicids2017.py.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src.models.traditional.baseline import BASELINE_MODELS
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config


def load_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    path = processed_dir / f"binary_{split_name}.csv"
    return pd.read_csv(path)


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
    print(f"Train label distribution:\n{y_train.value_counts()}")

    for model_name, builder in BASELINE_MODELS.items():
        print(f"\n{'#' * 70}\nTraining: {model_name}\n{'#' * 70}")

        pipeline = builder(random_seed)

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
        print_metrics(metrics, title=f"{model_name} — Binary Classification (Test Set)")

        log_experiment(
            results_dir=results_dir,
            dataset="cicids2017",
            task="binary",
            model_name=model_name,
            metrics=metrics,
            training_time_seconds=training_time,
            inference_time_seconds=inference_time,
            random_seed=random_seed,
        )

        model_path = saved_models_dir / f"cicids2017_binary_{model_name}.joblib"
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()