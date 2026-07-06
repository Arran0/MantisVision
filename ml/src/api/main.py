"""Inference API for the Kappaphycus alvarezii health classifier (Milestone 6).

Designed for future expansion: the response shape already carries `species`
so that adding real species identification later is additive, not breaking.
Additional endpoints (disease, predator, damage %) can be added as sibling
routers without touching this one.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config  # noqa: E402
from src.api.schemas import HealthCheckResponse, PredictionResponse  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402

app = FastAPI(title="Mantis Vision Inference API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to the deployed web app origin before production
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        checkpoint_path = config.checkpoints_dir / "best_model.pt"
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No trained model found. Run `python -m src.train` first, "
                    f"expected checkpoint at {checkpoint_path}"
                ),
            )
        _predictor = Predictor(checkpoint_path)
    return _predictor


@app.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    checkpoint_path = config.checkpoints_dir / "best_model.pt"
    model_loaded = checkpoint_path.exists()
    classes = config.class_names if not model_loaded else get_predictor().class_names
    return HealthCheckResponse(status="ok", model_loaded=model_loaded, classes=classes)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    predictor = get_predictor()
    result = predictor.predict(image_bytes)

    return PredictionResponse(
        species=result.species,
        health=result.health,
        confidence=result.confidence,
        explanation=result.explanation,
        recommendation=result.recommendation,
        gradcam_png_base64=result.gradcam_base64_png,
    )
