from __future__ import annotations

from pydantic import BaseModel


class MeasurementResultResponse(BaseModel):
    type: str  # "classification" | "regression" | "segmentation"
    label: str
    value: str | float | None
    confidence: float | None
    explanation: str | None
    recommendation: str | None
    unit: str | None
    min: float | None
    max: float | None
    coverage: dict[str, float] | None
    seg_colors: dict[str, str] | None
    mask_png_base64: str | None
    gradcam_png_base64: str | None


class PredictionResponse(BaseModel):
    # The full schema-driven report — one entry per measurement, in schema
    # order. No flat/legacy fields: a client renders whichever measurements
    # are present (value is None where applies_when gated it off).
    measurements: dict[str, MeasurementResultResponse]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    # Species is a per-image predicted classification now, not a fixed
    # schema-level value — report the full list of species classes the
    # loaded schema knows about instead of a single "active" one.
    species_classes: list[str]
    measurements: list[str]  # measurement keys the loaded checkpoint's schema defines


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
