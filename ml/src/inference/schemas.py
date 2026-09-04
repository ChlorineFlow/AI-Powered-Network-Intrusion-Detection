"""Pydantic request/response schemas for the FastAPI service."""

from pydantic import BaseModel, Field
from typing import Optional


class FlowFeatures(BaseModel):
    """A single network flow's feature values. Field names match the
    CICIDS2017 column names exactly (with spaces, as in the trained
    models' expected input) via the alias mechanism."""
    model_config = {"populate_by_name": True, "extra": "allow"}
    # 'extra: allow' permits passing the full 78-feature dict as JSON
    # without redeclaring every single column name here explicitly.


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Dict of the 78 CICIDS2017 flow features, keyed by exact column name.")
    model: Optional[str] = Field(default="xgboost", description="Which trained model to use.")


class PredictBatchRequest(BaseModel):
    rows: list[dict] = Field(..., description="List of flow-feature dicts.")
    model: Optional[str] = Field(default="xgboost")


class PredictResponse(BaseModel):
    prediction: str  # "BENIGN" or "ATTACK"
    confidence: float
    model_used: str


class ExplainRequest(BaseModel):
    features: dict
    model: Optional[str] = Field(default="xgboost")
    top_n: int = 10


class ExplainResponse(BaseModel):
    prediction: str
    top_contributing_features: list[dict]
    model_used: str

class CommonFeaturesRequest(BaseModel):
    duration_sec: float = Field(..., description="Flow/connection duration in seconds")
    src_bytes: float = Field(..., description="Bytes sent from source")
    dst_bytes: float = Field(..., description="Bytes sent from destination")


class RoutedPredictResponse(BaseModel):
    prediction: str
    confidence: float
    routed_to_expert: str
    router_confidence: float


class UnifiedPredictResponse(BaseModel):
    prediction: str
    confidence: float
    model_used: str = "unified_common_features"