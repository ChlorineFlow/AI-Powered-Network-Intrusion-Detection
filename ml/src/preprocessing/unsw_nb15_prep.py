"""
UNSW-NB15-specific preprocessing. Categorical columns (proto, service,
state) are one-hot encoded, fit ONLY on training data. 'label' ships
pre-encoded as 0 (normal) / 1 (attack) — no relabeling needed, unlike
NSL-KDD's string labels.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL_COLS = ["proto", "service", "state"]
NON_FEATURE_COLS = ["label", "attack_cat"]


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    numeric_cols = [
        c for c in train_df.columns
        if c not in CATEGORICAL_COLS + NON_FEATURE_COLS
    ]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[CATEGORICAL_COLS])

    def transform(df):
        cat_encoded = encoder.transform(df[CATEGORICAL_COLS])
        cat_encoded_df = pd.DataFrame(
            cat_encoded, columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
            index=df.index,
        )
        return pd.concat([df[numeric_cols], cat_encoded_df], axis=1)

    X_train = transform(train_df)
    X_test = transform(test_df)
    return X_train, X_test, encoder