"""
Unit tests for cleaning and splitting logic, using small synthetic
DataFrames — not real dataset samples.
"""

import numpy as np
import pandas as pd

from src.preprocessing.clean import remove_duplicates, remove_invalid_rows
from src.preprocessing.split import (
    filter_rare_multiclass,
    make_binary_label,
    leakage_safe_split,
)


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [1, 1, 2, 3]})
    result = remove_duplicates(df)
    assert len(result) == 3


def test_remove_invalid_rows_drops_inf_and_nan():
    df = pd.DataFrame({
        "Flow Duration": [10, 20, 30, 40],
        "Flow Bytes/s": [1.0, np.inf, np.nan, 5.0],
    })
    result = remove_invalid_rows(df)
    assert len(result) == 2  # rows 0 and 3 survive


def test_remove_invalid_rows_drops_negative_duration():
    df = pd.DataFrame({
        "Flow Duration": [-13, 20],
        "Flow Bytes/s": [1.0, 2.0],
    })
    result = remove_invalid_rows(df)
    assert len(result) == 1
    assert result["Flow Duration"].iloc[0] == 20


def test_filter_rare_multiclass():
    df = pd.DataFrame({
        "Label": ["BENIGN"] * 100 + ["Heartbleed"] * 5 + ["DDoS"] * 60
    })
    filtered, excluded = filter_rare_multiclass(df, "Label", min_class_samples=50)
    assert "Heartbleed" in excluded
    assert "Heartbleed" not in filtered["Label"].values
    assert "BENIGN" in filtered["Label"].values
    assert "DDoS" in filtered["Label"].values


def test_make_binary_label():
    df = pd.DataFrame({"Label": ["BENIGN", "DDoS", "BENIGN"]})
    binary = make_binary_label(df, "Label")
    assert binary.tolist() == [0, 1, 0]


def test_leakage_safe_split_no_overlap():
    df = pd.DataFrame({
        "feature": range(200),
        "Label": ["BENIGN"] * 150 + ["ATTACK"] * 50,
    })
    train, val, test = leakage_safe_split(
        df, label_col="Label", test_size=0.2, val_size=0.1, random_seed=42
    )
    train_idx = set(train["feature"])
    val_idx = set(val["feature"])
    test_idx = set(test["feature"])

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)
    assert len(train) + len(val) + len(test) == len(df)