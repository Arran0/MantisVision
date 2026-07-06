from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    species: str
    health: str
    confidence: float
    explanation: str
    recommendation: str
    gradcam_png_base64: str


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]
