"""
Multiclass attack-category classification on UNSW-NB15 (10 categories:
Normal + 9 attack types). Unlike NSL-KDD, all attack_cat values in
UNSW-NB15's official split appear in both train and test — no
rare/unseen-class filtering needed here.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.preprocessing.unsw_nb15_prep import prepare_features
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
    raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading UNSW-NB15 train/test files...")
    train_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_training-set.csv").load()
    test_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_testing-set.csv").load()

    train_cats = set(train_df["attack_cat"].unique())
    test_cats = set(test_df["attack_cat"].unique())
    print(f"Train categories ({len(train_cats)}): {sorted(train_cats)}")
    print(f"Test categories ({len(test_cats)}): {sorted(test_cats)}")
    print(f"Mismatch: {train_cats.symmetric_difference(test_cats) or 'none'}")

    X_train, X_test, encoder = prepare_features(train_df, test_df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df["attack_cat"])
    y_test = label_encoder.transform(test_df["attack_cat"])
    print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    saved_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, saved_models_dir / "unsw_nb15_multiclass_feature_encoder.joblib")
    joblib.dump(label_encoder, saved_models_dir / "unsw_nb15_multiclass_label_encoder.joblib")

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
        print_metrics(metrics, title=f"{model_name} — UNSW-NB15 Multiclass (Test Set)")

        log_experiment(
            results_dir=results_dir, dataset="unsw_nb15", task="multiclass",
            model_name=model_name, metrics=metrics,
            training_time_seconds=training_time, inference_time_seconds=inference_time,
            random_seed=random_seed,
            extra={"classes": list(label_encoder.classes_)},
        )

        model_path = saved_models_dir / f"unsw_nb15_multiclass_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()