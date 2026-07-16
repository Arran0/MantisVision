from __future__ import annotations

from pydantic import BaseModel


class MeasurementResultResponse(BaseModel):
    type: str  # "classification" | "regression" | "segmentation"
    value: str | float | None
    confidence: float | None
    explanation: str | None
    recommendation: str | None
    coverage: dict[str, float] | None
    mask_png_base64: str | None


class PredictionResponse(BaseModel):
    # Legacy flat fields — kept so the current PWA (apps/web/src/app/api/
    # predict/route.ts) keeps working unchanged.
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
    # Generic, forward-looking report: one entry per schema measurement.
    measurements: dict[str, MeasurementResultResponse]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    # Species is a per-image predicted classification now, not a fixed
    # schema-level value — report the full list of species classes the
    # loaded schema knows about instead of a single "active" one.
    species_classes: list[str]
    measurements: list[str]  # measurement keys the loaded checkpoint's schema defines
    # Set when a checkpoint file exists on disk but failed to load (corrupt
    # download, incompatible torch version, ...) — model_loaded is false in
    # that case too, but this says why, instead of /health just 500ing.
    error: str | None = None


class ReloadRequest(BaseModel):
    # Direct-download URL of the promoted run's checkpoint (a GitHub Release
    # asset, Supabase Storage object, etc.) — the same kind of URL used for
    # MODEL_URL at cold start.
    model_url: str


class ReloadResponse(BaseModel):
    status: str
    model_loaded: bool
    species_classes: list[str]
    measurements: list[str]
