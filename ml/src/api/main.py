"""Inference API for the Kappaphycus alvarezii health classifier (Milestone 6).

Designed for future expansion: the response shape already carries `species`
so that adding real species identification later is additive, not breaking.
Additional endpoints (disease, predator, damage %) can be added as sibling
routers without touching this one.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import config  # noqa: E402
from src.api.schemas import HealthCheckResponse, PredictionResponse  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402

logger = logging.getLogger("mantis_vision.api")
logging.basicConfig(level=logging.INFO)

# A stalled or unreachable MODEL_URL must fail loudly, not hang the ASGI
# server forever with no log output (which is exactly what a bare
# urlretrieve() with no timeout does).
CHECKPOINT_DOWNLOAD_TIMEOUT_S = 60

app = FastAPI(title="Mantis Vision Inference API", version="0.1.0")

# In production, set WEB_APP_ORIGIN to the deployed PWA's origin (e.g. your
# Vercel URL) to stop other sites from calling this API from a browser.
# Defaults to "*" for local development, where the caller is just localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("WEB_APP_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Predictor | None = None


def _download_checkpoint(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    logger.info("Downloading model checkpoint from %s ...", url)
    with urllib.request.urlopen(url, timeout=CHECKPOINT_DOWNLOAD_TIMEOUT_S) as response:
        with open(tmp_path, "wb") as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
    tmp_path.rename(path)
    logger.info("Checkpoint downloaded to %s (%d bytes).", path, path.stat().st_size)


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


@app.on_event("startup")
def _load_model_on_startup() -> None:
    # ml/checkpoints/ is gitignored, so a fresh deploy (e.g. on Render or a
    # Hugging Face Space) won't have best_model.pt unless we fetch it. Set
    # MODEL_URL to a direct-download link (e.g. a GitHub Release asset) to
    # have it pulled down once at boot.
    #
    # Failures here are caught rather than left to crash the process: a bad
    # MODEL_URL should leave the API up and reporting model_loaded: false
    # (with the reason in the logs), not take the whole container down.
    checkpoint_path = config.checkpoints_dir / "best_model.pt"
    model_url = os.environ.get("MODEL_URL")
    try:
        if not checkpoint_path.exists() and model_url:
            _download_checkpoint(checkpoint_path, model_url)
        elif not checkpoint_path.exists():
            logger.warning("MODEL_URL is not set and no checkpoint is present at %s.", checkpoint_path)
        if checkpoint_path.exists():
            get_predictor()
            logger.info("Model loaded successfully.")
    except Exception:
        logger.exception("Failed to download or load the model checkpoint.")


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
