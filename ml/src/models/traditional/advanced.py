"""
Advanced model definitions.

SVM is intentionally NOT trained on the full 1.76M-row training set —
sklearn's SVC scales roughly quadratically-to-cubically with sample
count, making full-dataset training impractical (multi-hour to
multi-day). Per documented methodology, SVM is evaluated on a
stratified subsample instead; see train_advanced_binary.py.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def build_random_forest(random_seed: int) -> Pipeline:
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )),
    ])


def build_xgboost(random_seed: int) -> Pipeline:
    return Pipeline([
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
        )),
    ])


def build_svm(random_seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,  # needed for ROC-AUC; slows training further
            random_state=random_seed,
        )),
    ])


ADVANCED_MODELS = {
    "random_forest": build_random_forest,
    "xgboost": build_xgboost,
}

# Kept separate from ADVANCED_MODELS because it runs on a subsample,
# not the full training set — see train_advanced_binary.py.
SVM_MODEL_BUILDER = build_svm