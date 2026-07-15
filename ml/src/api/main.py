"""Inference API for the seaweed multi-head classifier.

`species` is a real predicted classification (see the "species" measurement
in config.DEFAULT_SCHEMA) — /predict's `species` field is the model's actual
per-image prediction, not a fixed constant. Additional endpoints (predator,
damage %, ...) can be added as sibling routers without touching this one.

Run:
    uvicorn src.api.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import SCHEMA, config  # noqa: E402
from src.api.schemas import (  # noqa: E402
    HealthCheckResponse,
    MeasurementResultResponse,
    PredictionResponse,
    ReloadRequest,
    ReloadResponse,
)
from src.inference.predictor import Predictor  # noqa: E402

logger = logging.getLogger("mantis_vision.api")
logging.basicConfig(level=logging.INFO)

# A stalled or unreachable MODEL_URL must fail loudly, not hang the ASGI
# server forever with no log output (which is exactly what a bare
# urlretrieve() with no timeout does).
CHECKPOINT_DOWNLOAD_TIMEOUT_S = 60

_predictor: Predictor | None = None

# Serialises /admin/reload so two concurrent promotions can't interleave their
# download/verify/swap steps and race each other into a torn state. The actual
# `_predictor = ...` reassignment is already atomic under the GIL; the lock is
# about the multi-step sequence around it, not that single store.
_reload_lock = threading.Lock()


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


@asynccontextmanager
async def lifespan(_: FastAPI):
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
    yield


app = FastAPI(title="Mantis Vision Inference API", version="0.1.0", lifespan=lifespan)

# In production, set WEB_APP_ORIGIN to the deployed PWA's origin (e.g. your
# Vercel URL) to stop other sites from calling this API from a browser.
# Defaults to "*" for local development, where the caller is just localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("WEB_APP_ORIGIN", "*")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    checkpoint_path = config.checkpoints_dir / "best_model.pt"
    model_loaded = checkpoint_path.exists()
    if model_loaded:
        schema = get_predictor().schema
    else:
        # No checkpoint yet — report the active schema's own measurements so
        # /health is still informative before first load.
        schema = SCHEMA
    species_measurement = schema.find("species")
    return HealthCheckResponse(
        status="ok",
        model_loaded=model_loaded,
        species_classes=species_measurement.class_names() if species_measurement else [],
        measurements=[m.key for m in schema.measurements],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    predictor = get_predictor()
    result = predictor.predict(image_bytes)

    return PredictionResponse(
        species=result.species,
        is_seaweed=result.is_seaweed,
        condition=result.condition,
        health=result.health,
        health_score=result.health_score,
        confidence=result.confidence,
        disease_subtype=result.disease_subtype,
        dried_pct=result.dried_pct,
        decayed_pct=result.decayed_pct,
        explanation=result.explanation,
        recommendation=result.recommendation,
        gradcam_png_base64=result.gradcam_base64_png,
        measurements={
            key: MeasurementResultResponse(
                type=m.type,
                value=m.value,
                confidence=m.confidence,
                explanation=m.explanation,
                recommendation=m.recommendation,
                coverage=m.coverage,
                mask_png_base64=m.mask_png_base64,
            )
            for key, m in result.measurements.items()
        },
    )


@app.post("/admin/reload", response_model=ReloadResponse)
def reload_model(
    body: ReloadRequest,
    authorization: str | None = Header(default=None),
) -> ReloadResponse:
    """Hot-swap the live model with a promoted checkpoint — no process restart.

    Called by the web app's promote route (apps/web/.../retrain/promote) when
    an admin promotes a completed run. The new checkpoint is downloaded to a
    *separate* temp file and loaded into a fresh Predictor first; only once
    that succeeds is the module-level `_predictor` swapped and the temp file
    moved over best_model.pt (so a later cold restart serves the same
    version). A download or load failure leaves the currently-serving model
    completely untouched and returns 502 — a bad checkpoint can never take
    down live /predict traffic.
    """
    expected = os.environ.get("RELOAD_TOKEN")
    if not expected:
        # Fail closed: if no shared secret is configured, reload is disabled
        # rather than open to anyone who can reach the endpoint.
        raise HTTPException(
            status_code=503,
            detail="Model reload is not enabled (RELOAD_TOKEN is not set on this host).",
        )
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing reload token.")

    global _predictor
    with _reload_lock:
        checkpoint_path = config.checkpoints_dir / "best_model.pt"
        # Distinct staging file — never write over the currently-serving
        # best_model.pt until the download is verified loadable.
        staging_path = checkpoint_path.parent / "best_model.reload.pt"
        try:
            _download_checkpoint(staging_path, body.model_url)
            new_predictor = Predictor(staging_path)
        except Exception as e:  # noqa: BLE001 - any failure must leave the old model serving
            logger.exception("Model reload failed; keeping the currently-serving model.")
            # Clean up the staging file and its download-intermediate so a
            # failed reload leaves no half-written artifacts behind.
            staging_path.unlink(missing_ok=True)
            staging_path.with_suffix(".tmp").unlink(missing_ok=True)
            raise HTTPException(status_code=502, detail=f"Model reload failed: {e}") from e

        # New model verified loadable — swap it in and persist over
        # best_model.pt so a future cold restart picks up the same version.
        _predictor = new_predictor
        os.replace(staging_path, checkpoint_path)
        logger.info("Model hot-swapped from %s.", body.model_url)

    species_measurement = new_predictor.schema.find("species")
    return ReloadResponse(
        status="ok",
        model_loaded=True,
        species_classes=species_measurement.class_names() if species_measurement else [],
        measurements=[m.key for m in new_predictor.schema.measurements],
    )
