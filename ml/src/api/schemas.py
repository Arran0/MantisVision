from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    species: str
    is_seaweed: bool
    condition: str
    # Derived display level (Healthy/Moderate/Low). None for Background.
    health: str | None
    health_score: float | None  # 0-100
    confidence: float
    disease_subtype: str | None
    dried_pct: float | None
    decayed_pct: float | None
    explanation: str
    recommendation: str
    gradcam_png_base64: str


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    species: str
    conditions: list[str]
    disease_subtypes: list[str]


class ReloadRequest(BaseModel):
    # Direct-download URL of the promoted run's checkpoint (a GitHub Release
    # asset, Supabase Storage object, etc.) — the same kind of URL used for
    # MODEL_URL at cold start.
    model_url: str


class ReloadResponse(BaseModel):
    status: str
    model_loaded: bool
    species: str
    conditions: list[str]
    disease_subtypes: list[str]
