"""
Train and evaluate models on the multiclass attack-classification task,
using the multiclass_{train,val,test}.csv splits saved in Step 4 (which
already exclude the 3 rare classes per documented methodology).

Runs Decision Tree, Random Forest, and XGBoost — the three models that
scale reasonably to this dataset size. Logistic Regression and SVM are
skipped for multiclass: Logistic Regression's one-vs-rest approach adds
limited value over the tree models already proven superior in Step 6,
and SVM's per-class overhead in a multiclass setting would multiply an
already-costly training time (see Step 6 SVM inference-time finding).
This is a disclosed scope decision, not an oversight.
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

from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config


def load_split(processed_dir: Path, split_name: str) -> pd.DataFrame:
    path = processed_dir / f"multiclass_{split_name}.csv"
    return pd.read_csv(path)


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
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading processed multiclass splits...")
    train_df = load_split(processed_dir, "train")
    test_df = load_split(processed_dir, "test")

    feature_cols = [c for c in train_df.columns if c != "Label"]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]

    # Label-encode class names to integers. Fit ONLY on training labels
    # to avoid leaking test-set label information; since Step 4's
    # rare-class filtering already guarantees the same class set exists
    # in train/val/test, this is safe.
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df["Label"])
    y_test = encoder.transform(test_df["Label"])
    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")

    encoder_path = saved_models_dir / "cicids2017_multiclass_label_encoder.joblib"
    saved_models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, encoder_path)
    print(f"Saved label encoder to: {encoder_path}")

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train class distribution:\n{train_df['Label'].value_counts()}")

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
        print_metrics(metrics, title=f"{model_name} — Multiclass (Test Set)")

        log_experiment(
            results_dir=results_dir,
            dataset="cicids2017",
            task="multiclass",
            model_name=model_name,
            metrics=metrics,
            training_time_seconds=training_time,
            inference_time_seconds=inference_time,
            random_seed=random_seed,
            extra={"classes": list(encoder.classes_)},
        )

        model_path = saved_models_dir / f"cicids2017_multiclass_{model_name}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()