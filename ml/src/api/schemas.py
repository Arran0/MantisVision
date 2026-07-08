from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    species: str
    category: str  # Healthy | Moderate | Low
    condition: str | None  # None | Dried | Decayed | Diseased
    disease_subtype: str | None  # e.g. IceIce | Epiphyte | Bacterial | Unknown; null unless condition == "Diseased"
    health_score: float  # 0-10
    dried_percentage: float  # 0-100, heuristic (see config.py's anchor justification)
    decayed_percentage: float  # 0-100, heuristic (see config.py's anchor justification)
    confidence: float  # raw softmax probability of the predicted category
    confidence_calibrated: float | None  # temperature-scaled; null if no calibration.json yet
    explanation_bullets: list[str]
    recommendation: str
    gradcam_png_base64: str


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    category_names: list[str]
    condition_names: list[str]
    disease_subtype_names: list[str]
