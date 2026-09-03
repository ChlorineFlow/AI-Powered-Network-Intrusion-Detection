"""
Multiclass attack classification on NSL-KDD. Restricted to attack
classes present in BOTH train and test sets — NSL-KDD's test set
intentionally includes attack types absent from training (by design,
to test generalization to unknown attacks), and a supervised multiclass
model cannot be evaluated on classes it never saw during training. This
mirrors the rare-class exclusion methodology already documented for
CICIDS2017, but here the criterion is "present in both splits" rather
than a raw sample-count threshold.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.preprocessing.nsl_kdd_prep import prepare_features
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config

MODELS = {
    "decision_tree": lambda seed: DecisionTreeClassifier(
        random_state=seed, class_weight="balanced", max_depth=20
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=100, max_depth=20, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    ),
    "xgboost": lambda seed: XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        eval_metric="mlogloss", random_state=seed, n_jobs=-1,
    ),
}


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading NSL-KDD train/test files...")
    train_df = NSLKDDLoader(raw_dir, filename="KDDTrain+.txt").load()
    test_df = NSLKDDLoader(raw_dir, filename="KDDTest+.txt").load()

    min_samples = config["multiclass"]["min_class_samples"]

    train_counts = train_df["label"].value_counts()
    rare_classes = set(train_counts[train_counts < min_samples].index)

    train_labels = set(train_df["label"].unique())
    test_labels = set(test_df["label"].unique())
    common_labels = (train_labels & test_labels) - rare_classes
    train_only = train_labels - test_labels
    test_only = test_labels - train_labels

    print(f"\nRare classes excluded (< {min_samples} training samples): "
          f"{sorted(rare_classes)}")

    print(f"\nLabels in train only ({len(train_only)}): {sorted(train_only)}")
    print(f"Labels in test only ({len(test_only)}) — excluded, unseen during "
          f"training, cannot be validly predicted: {sorted(test_only)}")
    print(f"Common labels used for multiclass evaluation ({len(common_labels)}): "
          f"{sorted(common_labels)}")

    train_df = train_df[train_df["label"].isin(common_labels)].reset_index(drop=True)
    test_df = test_df[test_df["label"].isin(common_labels)].reset_index(drop=True)
    print(f"\nAfter filtering — Train: {train_df.shape} | Test: {test_df.shape}")

    X_train, X_test, encoder = prepare_features(train_df, test_df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["label"])
    y_test = label_encoder.transform(test_df["label"])
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, saved_models_dir / "nsl_kdd_multiclass_feature_encoder.joblib")
    joblib.dump(label_encoder, saved_models_dir / "nsl_kdd_multiclass_label_encoder.joblib")

    for model_name, builder in MODELS.items():
        print(f"\n{'#' * 70}\nTraining: {model_name}\n{'#' * 70}")
        model = builder(random_seed)

        start_train = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_train
        print(f"Training time: {training_time:.2f}s")

        start_infer = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        inference_time = time.time() - start_infer
        print(f"Inference time on {len(X_test):,} rows: {inference_time:.2f}s")

        metrics = compute_metrics(y_test, y_pred, y_proba)
        print_metrics(metrics, title=f"{model_name} — NSL-KDD Multiclass (Test Set)")

        log_experiment(
            results_dir=results_dir, dataset="nsl_kdd", task="multiclass",
            model_name=model_name, metrics=metrics,
            training_time_seconds=training_time, inference_time_seconds=inference_time,
            random_seed=random_seed,
            extra={
                "classes": list(label_encoder.classes_),
                "excluded_test_only_classes": sorted(test_only),
                "excluded_rare_classes": sorted(rare_classes),
                "min_class_samples": min_samples,
                "note": "Restricted to classes present in both train and test, "
                        "AND with >= min_class_samples training examples "
                        "(same threshold and rationale as CICIDS2017 — see "
                        "docs/research/methodology.md).",
            },
        )

        model_path = saved_models_dir / f"nsl_kdd_multiclass_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()