"""
Baseline model definitions. Each is a full sklearn Pipeline (scaler +
classifier where needed) so that fitting the pipeline on training data
never leaks information — StandardScaler is fit only on train, as
enforced by calling .fit() only on the training split.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_logistic_regression(random_seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            class_weight="balanced",  # mitigates the 80/20 imbalance
            n_jobs=-1,
        )),
    ])


def build_decision_tree(random_seed: int) -> Pipeline:
    return Pipeline([
        ("clf", DecisionTreeClassifier(
            random_state=random_seed,
            class_weight="balanced",
            max_depth=20,  # prevents unbounded overfitting on 1.7M rows
        )),
    ])


BASELINE_MODELS = {
    "logistic_regression": build_logistic_regression,
    "decision_tree": build_decision_tree,
}