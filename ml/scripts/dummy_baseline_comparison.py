"""
Compares each trained model's real test performance against a trivial
"always predict majority class" baseline (sklearn DummyClassifier),
for all three datasets. This directly tests whether high reported
accuracy is simply a byproduct of class imbalance, or genuine lift
beyond what a naive classifier achieves for free.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.data.loaders.nsl_kdd_loader import NSLKDDLoader
from src.data.loaders.unsw_nb15_loader import UNSWNB15TrainTestLoader
from src.preprocessing.nsl_kdd_prep import prepare_features as nsl_prepare, make_binary_label
from src.preprocessing.unsw_nb15_prep import prepare_features as unsw_prepare
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config


def run_dummy(X_train, y_train, X_test, y_test, dataset_name, real_accuracy, real_f1_macro):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    y_proba = dummy.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba)

    print(f"\n{'=' * 70}\n{dataset_name}\n{'=' * 70}")
    print(f"Dummy baseline (always predict majority class):")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"Real trained model (best):")
    print(f"  Accuracy:    {real_accuracy:.4f}")
    print(f"  F1 (macro):  {real_f1_macro:.4f}")
    print(f"Lift over dummy baseline:")
    print(f"  Accuracy lift:   +{(real_accuracy - metrics['accuracy']) * 100:.2f} points")
    print(f"  F1 (macro) lift: +{(real_f1_macro - metrics['f1_macro']) * 100:.2f} points")


def main():
    config = load_config()
    ml_root = Path(__file__).resolve().parents[1]

    # ---- CICIDS2017 ----
    processed_dir = ml_root / config["paths"]["processed_dir"] / "cicids2017"
    train_df = pd.read_csv(processed_dir / "binary_train.csv")
    test_df = pd.read_csv(processed_dir / "binary_test.csv")
    feature_cols = [c for c in train_df.columns if c != "Label"]
    X_train = train_df[feature_cols]
    y_train = (train_df["Label"] != "BENIGN").astype(int)
    X_test = test_df[feature_cols]
    y_test = (test_df["Label"] != "BENIGN").astype(int)
    run_dummy(X_train, y_train, X_test, y_test, "CICIDS2017 (binary)",
              real_accuracy=0.9992, real_f1_macro=0.9986)

    # ---- NSL-KDD ----
    raw_dir = ml_root / "data" / "raw" / "nsl_kdd"
    train_df = NSLKDDLoader(raw_dir, filename="KDDTrain+.txt").load()
    test_df = NSLKDDLoader(raw_dir, filename="KDDTest+.txt").load()
    X_train, X_test, _ = nsl_prepare(train_df, test_df)
    y_train = make_binary_label(train_df)
    y_test = make_binary_label(test_df)
    run_dummy(X_train, y_train, X_test, y_test, "NSL-KDD (binary)",
              real_accuracy=0.7981, real_f1_macro=0.7978)

    # ---- UNSW-NB15 ----
    raw_dir = ml_root / "data" / "raw" / "unsw_nb15"
    train_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_training-set.csv").load()
    test_df = UNSWNB15TrainTestLoader(raw_dir, "UNSW_NB15_testing-set.csv").load()
    X_train, X_test, _ = unsw_prepare(train_df, test_df)
    y_train = train_df["label"]
    y_test = test_df["label"]
    run_dummy(X_train, y_train, X_test, y_test, "UNSW-NB15 (binary)",
              real_accuracy=0.9062, real_f1_macro=0.9038)


if __name__ == "__main__":
    main()