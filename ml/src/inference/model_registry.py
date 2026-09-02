"""
Loads and caches trained models at service startup. XGBoost is the
default/recommended model (best F1, ROC-AUC, and training time across
all rigor testing — see docs/research/methodology.md); other models
are available for the dashboard's Model Performance comparison view.
"""

from pathlib import Path
import joblib

DEFAULT_MODEL = "xgboost"

AVAILABLE_BINARY_MODELS = [
    "xgboost", "random_forest", "decision_tree", "logistic_regression",
]
AVAILABLE_MULTICLASS_MODELS = ["xgboost", "random_forest", "decision_tree"]


class ModelRegistry:
    def __init__(self, saved_models_dir: Path):
        self.saved_models_dir = saved_models_dir
        self._binary_models = {}
        self._multiclass_models = {}
        self._multiclass_encoder = None

    def load_all(self):
        for name in AVAILABLE_BINARY_MODELS:
            path = self.saved_models_dir / f"cicids2017_binary_{name}.joblib"
            if path.exists():
                self._binary_models[name] = joblib.load(path)
                print(f"[model_registry] Loaded binary model: {name}")
            else:
                print(f"[model_registry] WARNING: {path} not found, skipping.")

        for name in AVAILABLE_MULTICLASS_MODELS:
            path = self.saved_models_dir / f"cicids2017_multiclass_{name}.joblib"
            if path.exists():
                self._multiclass_models[name] = joblib.load(path)
                print(f"[model_registry] Loaded multiclass model: {name}")
            else:
                print(f"[model_registry] WARNING: {path} not found, skipping.")

        encoder_path = self.saved_models_dir / "cicids2017_multiclass_label_encoder.joblib"
        if encoder_path.exists():
            self._multiclass_encoder = joblib.load(encoder_path)
            print("[model_registry] Loaded multiclass label encoder.")

    def get_binary_model(self, name: str = DEFAULT_MODEL):
        if name not in self._binary_models:
            raise ValueError(
                f"Binary model '{name}' not available. "
                f"Loaded models: {list(self._binary_models.keys())}"
            )
        return self._binary_models[name]

    def get_multiclass_model(self, name: str = DEFAULT_MODEL):
        if name not in self._multiclass_models:
            raise ValueError(
                f"Multiclass model '{name}' not available. "
                f"Loaded models: {list(self._multiclass_models.keys())}"
            )
        return self._multiclass_models[name]

    def get_multiclass_encoder(self):
        if self._multiclass_encoder is None:
            raise RuntimeError("Multiclass label encoder not loaded.")
        return self._multiclass_encoder

    def available_binary_models(self):
        return list(self._binary_models.keys())

    def available_multiclass_models(self):
        return list(self._multiclass_models.keys())