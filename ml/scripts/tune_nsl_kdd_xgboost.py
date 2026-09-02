"""
Proper hyperparameter tuning for XGBoost on NSL-KDD, using
RandomizedSearchCV with stratified k-fold cross-validation on the
TRAINING set only. The test set is touched exactly once, at the end,
to report the final honest result — this is what distinguishes
legitimate tuning from p-hacking against the test set.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.preprocessing.nsl_kdd_prep import prepare_features, make_binary_label
from src.evaluation.metrics import compute_metrics, print_metrics
from src.utils.experiment_logger import log_experiment
from src.utils.config import load_config


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]
    raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
    results_dir = ml_root / config["paths"]["experiments_dir"] / "results"
    random_seed = config["random_seed"]

    print("Loading NSL-KDD data...")
    train_df = NSLKDDLoader(raw_dir, filename="KDDTrain+.txt").load()
    test_df = NSLKDDLoader(raw_dir, filename="KDDTest+.txt").load()

    X_train, X_test, encoder = prepare_features(train_df, test_df)
    y_train = make_binary_label(train_df)
    y_test = make_binary_label(test_df)

    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 15),
        "learning_rate": uniform(0.01, 0.29),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 10),
    }

    base_model = XGBClassifier(
        eval_metric="logloss", random_state=random_seed, n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="f1_macro",
        cv=cv,
        random_state=random_seed,
        n_jobs=-1,
        verbose=2,
    )

    print(f"\nRunning RandomizedSearchCV: 30 configurations x 5-fold CV = "
          f"150 fits on TRAINING data only ({len(X_train):,} rows)...")
    start = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start
    print(f"\nTuning completed in {tuning_time:.1f}s")

    print(f"\nBest cross-validated F1 (macro) on training folds: "
          f"{search.best_score_:.4f}")
    print(f"Best hyperparameters: {search.best_params_}")

    best_model = search.best_estimator_

    print("\nEvaluating tuned model on TEST set (touched once, for final reporting)...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    print_metrics(metrics, title="Tuned XGBoost — NSL-KDD Binary (Test Set)")

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"Untuned XGBoost F1 (macro):  0.7978")
    print(f"Tuned XGBoost F1 (macro):    {metrics['f1_macro']:.4f}")

    log_experiment(
        results_dir=results_dir, dataset="nsl_kdd", task="binary_tuned",
        model_name="xgboost_tuned", metrics=metrics,
        training_time_seconds=tuning_time, inference_time_seconds=0.0,
        random_seed=random_seed,
        extra={
            "best_params": search.best_params_,
            "best_cv_f1_macro": search.best_score_,
            "tuning_method": "RandomizedSearchCV, 30 iter, 5-fold StratifiedKFold, "
                              "scoring=f1_macro, fit on training set only",
        },
    )

    model_path = saved_models_dir / "nsl_kdd_binary_xgboost_tuned.joblib"
    joblib.dump(best_model, model_path)
    print(f"\nSaved tuned model to: {model_path}")


if __name__ == "__main__":
    main()