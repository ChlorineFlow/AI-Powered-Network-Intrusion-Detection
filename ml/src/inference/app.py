"""
FastAPI ML inference service. Endpoints per project spec (master
prompt §11): /health, /predict, /predict/batch, /explain, /model-info,
/metrics.

This service is stateless per request and does not call out to Node
or React — it is called BY the Node/Express backend, per the mandated
three-layer architecture (master prompt §10).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.inference.model_registry import ModelRegistry, DEFAULT_MODEL
from src.inference.schemas import (
    PredictRequest, PredictBatchRequest, PredictResponse,
    ExplainRequest, ExplainResponse,
)
from src.utils.config import load_config

config = load_config()
ml_root = Path(__file__).resolve().parents[2]
saved_models_dir = ml_root / config["paths"]["saved_models_dir"]
results_dir = ml_root / config["paths"]["experiments_dir"] / "results"

app = FastAPI(
    title="NIDS ML Inference Service",
    description="Binary and multiclass network intrusion detection inference, "
                 "serving models trained on CICIDS2017.",
    version="0.1.0",
)

# Only the Node/Express backend should call this service directly in
# production; CORS is permissive here for local development only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

registry = ModelRegistry(saved_models_dir)


@app.on_event("startup")
def startup():
    registry.load_all()


def features_to_dataframe(features: dict, expected_columns: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame from a features dict, in the exact
    column order the model expects. Raises HTTPException on missing
    columns rather than silently filling defaults."""
    missing = [c for c in expected_columns if c not in features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature(s): {missing[:10]}"
                   f"{'...' if len(missing) > 10 else ''}",
        )
    row = {col: features[col] for col in expected_columns}
    return pd.DataFrame([row])


def get_expected_columns(model) -> list[str]:
    """Extract the feature-name order the fitted model expects."""
    # For a Pipeline, the final step (classifier) usually exposes this;
    # falls back to the pipeline's own feature_names_in_ if present.
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)
        if hasattr(clf, "feature_names_in_"):
            return list(clf.feature_names_in_)
    raise HTTPException(
        status_code=500,
        detail="Could not determine expected feature columns for this model.",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "binary_models_loaded": registry.available_binary_models(),
        "multiclass_models_loaded": registry.available_multiclass_models(),
    }


@app.get("/model-info")
def model_info():
    return {
        "default_model": DEFAULT_MODEL,
        "binary_models_available": registry.available_binary_models(),
        "multiclass_models_available": registry.available_multiclass_models(),
        "trained_on": "CICIDS2017",
        "note": "See docs/research/methodology.md for cross-dataset "
                "generalization findings and known limitations.",
    }


@app.get("/metrics")
def metrics():
    """Returns the most recent logged experiment result for each model,
    from ml/experiments/results/. Real, traceable numbers only — never
    fabricated."""
    if not results_dir.exists():
        return {"results": []}

    all_results = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    # Keep only the most recent result per (dataset, task, model) combo
    latest = {}
    for r in all_results:
        key = (r["dataset"], r["task"], r["model"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r

    return {"results": list(latest.values())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model_name = request.model or DEFAULT_MODEL
    try:
        model = registry.get_binary_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    expected_cols = get_expected_columns(model)
    X = features_to_dataframe(request.features, expected_cols)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = float(proba[pred])

    return PredictResponse(
        prediction="ATTACK" if pred == 1 else "BENIGN",
        confidence=confidence,
        model_used=model_name,
    )


@app.post("/predict/batch")
def predict_batch(request: PredictBatchRequest):
    model_name = request.model or DEFAULT_MODEL
    try:
        model = registry.get_binary_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not request.rows:
        raise HTTPException(status_code=422, detail="No rows provided.")

    expected_cols = get_expected_columns(model)
    missing_in_first = [c for c in expected_cols if c not in request.rows[0]]
    if missing_in_first:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required feature(s) in input rows: "
                   f"{missing_in_first[:10]}",
        )

    X = pd.DataFrame(request.rows)[expected_cols]
    preds = model.predict(X)
    probas = model.predict_proba(X)

    results = [
        {
            "prediction": "ATTACK" if p == 1 else "BENIGN",
            "confidence": float(probas[i][p]),
        }
        for i, p in enumerate(preds)
    ]
    return {"model_used": model_name, "count": len(results), "results": results}


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    """Top contributing features for a single prediction — described as
    contributing factors, never as causal relationships (project rule §22)."""
    model_name = request.model or DEFAULT_MODEL
    try:
        model = registry.get_binary_model(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # SHAP TreeExplainer needs the raw classifier, not the sklearn Pipeline
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(clf, "get_booster") and type(clf).__name__ not in (
        "XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier"
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Explainability is only supported for tree-based models, "
                   f"not '{model_name}'.",
        )

    expected_cols = get_expected_columns(model)
    X = features_to_dataframe(request.features, expected_cols)

    pred = model.predict(X)[0]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X)

    row_shap = shap_values.values[0]
    contributions = sorted(
        zip(expected_cols, row_shap), key=lambda x: abs(x[1]), reverse=True
    )[: request.top_n]

    return ExplainResponse(
        prediction="ATTACK" if pred == 1 else "BENIGN",
        top_contributing_features=[
            {"feature": f, "shap_value": float(v)} for f, v in contributions
        ],
        model_used=model_name,
    )