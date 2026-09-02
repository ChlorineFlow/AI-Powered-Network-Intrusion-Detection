"""
Train and evaluate baseline + advanced models on NSL-KDD binary
classification (normal vs attack), using NSL-KDD's own fixed
train/test split (KDDTrain+.txt / KDDTest+.txt) — this is the
standard, correct way to use this dataset, not our custom splitter.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.preprocessing.nsl_kdd_prep import prepare_features, make_binary_label
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config

MODELS = {
    "logistic_regression": lambda seed: LogisticRegression(
        max_iter=1000, random_state=seed, class_weight="balanced", n_jobs=-1
    ),
    "decision_tree": lambda seed: DecisionTreeClassifier(
        random_state=seed, class_weight="balanced", max_depth=20
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    ),
    "xgboost": lambda seed: XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        eval_metric="logloss", random_state=seed, n_jobs=-1,
    ),
}


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading NSL-KDD train/test files (dataset's own fixed split)...")
    train_df = NSLKDDLoader(raw_dir, filename="KDDTrain+.txt").load()
    test_df = NSLKDDLoader(raw_dir, filename="KDDTest+.txt").load()
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    print("\nEncoding categorical features (fit on train only)...")
    X_train, X_test, encoder = prepare_features(train_df, test_df)
    y_train = make_binary_label(train_df)
    y_test = make_binary_label(test_df)
    print(f"Feature count after encoding: {X_train.shape[1]}")
    print(f"Train label distribution:\n{y_train.value_counts()}")

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, saved_models_dir / "nsl_kdd_binary_encoder.joblib")

    for model_name, builder in MODELS.items():
        print(f"\n{'#' * 70}\nTraining: {model_name}\n{'#' * 70}")

        model = builder(random_seed)
        # Logistic Regression benefits from scaling; tree models don't need it
        if model_name == "logistic_regression":
            pipeline = [("scaler", StandardScaler()), ("clf", model)]
            from sklearn.pipeline import Pipeline
            model = Pipeline(pipeline)

        start_train = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_train
        print(f"Training time: {training_time:.2f}s")

        start_infer = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_infer
        print(f"Inference time on {len(X_test):,} rows: {inference_time:.2f}s")

        metrics = compute_metrics(y_test, y_pred, y_proba)
        print_metrics(metrics, title=f"{model_name} — NSL-KDD Binary (Test Set)")

        log_experiment(
            results_dir=results_dir, dataset="nsl_kdd", task="binary",
            model_name=model_name, metrics=metrics,
            training_time_seconds=training_time, inference_time_seconds=inference_time,
            random_seed=random_seed,
            extra={"note": "NSL-KDD's own fixed train/test split used "
                            "(test set includes attack types not in train)."},
        )

        model_path = saved_models_dir / f"nsl_kdd_binary_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()