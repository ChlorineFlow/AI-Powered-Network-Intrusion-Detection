"""
Feature selection and dimensionality reduction methods, fit ONLY on
training data to avoid leakage (per project rule §33).
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def select_features_by_importance(
    X_train: pd.DataFrame, y_train: pd.Series, random_seed: int, max_features: int = None
) -> tuple[list[str], SelectFromModel]:
    """Fit a Random Forest on training data only, and select features
    whose importance is at or above the median — a standard, defensible
    threshold that avoids hand-picking a feature count."""
    estimator = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=random_seed, n_jobs=-1
    )
    selector = SelectFromModel(estimator, threshold="median", max_features=max_features)
    selector.fit(X_train, y_train)

    selected_cols = X_train.columns[selector.get_support()].tolist()
    return selected_cols, selector


def fit_pca(X_train: pd.DataFrame, random_seed: int, n_components: float = 0.95):
    """Fit StandardScaler + PCA on training data only. n_components=0.95
    keeps enough components to explain 95% of variance."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=n_components, random_state=random_seed)
    pca.fit(X_train_scaled)

    return scaler, pca