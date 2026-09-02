"""
Multi-metric evaluation, as required by project rules — accuracy alone
is never sufficient, especially given the severe class imbalance found
during EDA. Supports both binary and multiclass classification.
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

    For binary tasks, pass y_proba as the probability of the positive
    class (1D array). For multiclass tasks, pass y_proba as the full
    probability matrix (n_samples, n_classes) from predict_proba().
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    n_classes = len(np.unique(y_true))

    if y_proba is not None:
        try:
            if n_classes == 2:
                proba_pos = y_proba[:, 1] if np.ndim(y_proba) == 2 else y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, proba_pos)
            else:
                metrics["roc_auc_ovr_macro"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
        except ValueError as e:
            metrics["roc_auc_error"] = str(e)

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

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