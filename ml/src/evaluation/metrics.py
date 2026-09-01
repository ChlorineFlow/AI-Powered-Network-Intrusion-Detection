"""
Multi-metric evaluation, as required by project rules — accuracy alone
is never sufficient, especially given the severe class imbalance found
during EDA.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute the full metric set required by project rules (§21).
    y_proba (probability of the positive class) is required for ROC-AUC
    on binary tasks; pass None to skip it."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            # Can happen if a class is entirely absent from y_true in a split
            metrics["roc_auc"] = None
            metrics["roc_auc_error"] = str(e)

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # False Positive Rate / False Negative Rate — only well-defined for
    # binary classification (2x2 confusion matrix).
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else None
        metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else None

    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Results"):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"{key}:")
            for row in value:
                print(f"  {row}")
        else:
            print(f"{key}: {value}")