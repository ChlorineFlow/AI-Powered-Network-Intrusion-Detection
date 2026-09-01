"""
Leakage-safe splitting, including the documented multiclass rare-class
filtering rule (see docs/research/methodology.md and ml/config.yaml
'multiclass.min_class_samples').
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def filter_rare_multiclass(
    df: pd.DataFrame, label_col: str, min_class_samples: int
) -> tuple[pd.DataFrame, list[str]]:
    """Remove rows belonging to classes with fewer than
    min_class_samples total occurrences. Returns the filtered DataFrame
    and the list of excluded class names, for logging/documentation."""
    counts = df[label_col].value_counts()
    excluded = counts[counts < min_class_samples].index.tolist()
    if excluded:
        print(f"[split] Excluding {len(excluded)} rare classes from "
              f"multiclass task (< {min_class_samples} samples): {excluded}")
    filtered = df[~df[label_col].isin(excluded)].reset_index(drop=True)
    return filtered, excluded


def make_binary_label(df: pd.DataFrame, label_col: str) -> pd.Series:
    """BENIGN -> 0, everything else -> 1."""
    return (df[label_col] != "BENIGN").astype(int)


def leakage_safe_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    random_seed: int,
):
    """Stratified split into train/val/test. val_size is expressed as a
    fraction of the ORIGINAL dataset (matching config.yaml semantics),
    so it is recalculated as a fraction of the remaining data after the
    first split.

    Returns (train_df, val_df, test_df).
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_seed,
    )

    remaining_val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=remaining_val_fraction,
        stratify=train_val_df[label_col],
        random_state=random_seed,
    )

    print(f"[split] Train: {len(train_df):,} | Val: {len(val_df):,} | "
          f"Test: {len(test_df):,}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )