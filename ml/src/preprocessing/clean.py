"""
Cleaning steps applied BEFORE the train/test split — safe to run on the
full dataset because none of these steps look at the label distribution
in a way that could leak test information into training.
"""

import numpy as np
import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows. Must run before splitting, so the same
    flow record can never appear in both train and test."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"[clean] Removed {removed:,} duplicate rows "
          f"({removed / before:.2%} of {before:,}).")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Replace Infinity with NaN, then drop rows with NaN in any numeric
    column, and drop rows with a negative Flow Duration (a known data
    quality artifact in the original CICIDS2017 computation)."""
    before = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    if "Flow Duration" in df.columns:
        df = df[df["Flow Duration"] >= 0]

    df = df.dropna(subset=list(numeric_cols)).reset_index(drop=True)

    removed = before - len(df)
    print(f"[clean] Removed {removed:,} invalid rows (NaN/Infinity/negative "
          f"duration) ({removed / before:.2%} of {before:,}).")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Full pre-split cleaning sequence, in order."""
    df = remove_duplicates(df)
    df = remove_invalid_rows(df)
    return df