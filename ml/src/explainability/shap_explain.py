"""
SHAP-based explainability for tree-based models. TreeExplainer is used
since it is exact (not approximate) for tree ensembles like XGBoost.

Per project rule (§22), explanations are described as "top contributing
features," never as causal relationships the model has "discovered."
"""

import numpy as np
import pandas as pd
import shap


def compute_shap_values(model, X_sample: pd.DataFrame):
    """Returns a shap.Explanation object for the given samples."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values


def top_global_features(shap_values, feature_names, top_n: int = 15) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, across all samples —
    a global 'top contributing features' ranking, not a causal claim."""
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return ranking.head(top_n)


def explain_single_prediction(shap_values, sample_index: int, feature_names, top_n: int = 10) -> pd.DataFrame:
    """Top contributing features for one specific prediction, with the
    signed SHAP value (direction of influence toward/away from the
    predicted class)."""
    row_shap = shap_values.values[sample_index]
    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": row_shap,
    })
    df["abs_shap_value"] = df["shap_value"].abs()
    return df.sort_values("abs_shap_value", ascending=False).head(top_n).drop(columns="abs_shap_value")