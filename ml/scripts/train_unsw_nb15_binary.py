"""
Train and evaluate baseline + advanced models on UNSW-NB15 binary
classification, using the dataset's own official fixed train/test
split (UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv).

Note: this dataset's official split has attacks as the majority class
(68% attack in train) — the opposite imbalance direction from
CICIDS2017 and NSL-KDD, documented in methodology.md.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.preprocessing.unsw_nb15_prep import prepare_features
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config

MODELS = {
    "logistic_regression": lambda seed: Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed, class_weight="balanced")),
    ]),
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
    raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading UNSW-NB15 official train/test files...")
    train_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_training-set.csv").load()
    test_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_testing-set.csv").load()
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")
    print(f"Train label distribution:\n{train_df['label'].value_counts()} "
          f"(1=attack, 0=normal — note: attack is the majority class here, "
          f"unlike CICIDS2017/NSL-KDD)")

    print("\nEncoding categorical features (fit on train only)...")
    X_train, X_test, encoder = prepare_features(train_df, test_df)
    y_train = train_df["label"]
    y_test = test_df["label"]
    print(f"Feature count after encoding: {X_train.shape[1]}")

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, saved_models_dir / "unsw_nb15_binary_encoder.joblib")

    for model_name, builder in MODELS.items():
        print(f"\n{'#' * 70}\nTraining: {model_name}\n{'#' * 70}")
        model = builder(random_seed)

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
        print_metrics(metrics, title=f"{model_name} — UNSW-NB15 Binary (Test Set)")

        log_experiment(
            results_dir=results_dir, dataset="unsw_nb15", task="binary",
            model_name=model_name, metrics=metrics,
            training_time_seconds=training_time, inference_time_seconds=inference_time,
            random_seed=random_seed,
            extra={"note": "Official UNSW-NB15 train/test split; attack is the "
                            "majority class (68% in train)."},
        )

        model_path = saved_models_dir / f"unsw_nb15_binary_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()