"""
NSL-KDD-specific preprocessing. Unlike CICIDS2017, NSL-KDD:
  - has categorical columns (protocol_type, service, flag) needing encoding
  - has a 'difficulty' column that is NOT a feature (drop it)
  - already ships with fixed train/test files (no need to split ourselves,
    though we carve a validation set out of the training file)
  - has no duplicates/NaN/Infinity issues (verified clean in verify_nsl_kdd.py)
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode categorical columns, fit ONLY on training data.
    Returns (X_train, X_test, feature_names)."""
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [
        c for c in train_df.columns
        if c not in categorical_cols + ["label", "difficulty"]
    ]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[categorical_cols])

    def transform(df):
        cat_encoded = encoder.transform(df[categorical_cols])
        cat_encoded_df = pd.DataFrame(
            cat_encoded, columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index,
        )
        return pd.concat([df[numeric_cols], cat_encoded_df], axis=1)

    X_train = transform(train_df)
    X_test = transform(test_df)
    return X_train, X_test, encoder


def make_binary_label(df: pd.DataFrame) -> pd.Series:
    """NSL-KDD uses 'normal' for benign traffic (lowercase, unlike
    CICIDS2017's 'BENIGN')."""
    return (df["label"] != "normal").astype(int)


def carve_validation_split(X_train, y_train, val_fraction, random_seed):
    return train_test_split(
        X_train, y_train, test_size=val_fraction,
        stratify=y_train, random_state=random_seed,
    )