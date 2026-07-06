"""Single entrypoint for turning an uploaded image into the full MVP output:
species, health, confidence, explanation, recommendation, and a Grad-CAM
heatmap. Used by the FastAPI inference service (src/api/main.py).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from config import config
from src.data.transforms import build_transforms
from src.gradcam import generate_gradcam
from src.inference.explanations import explanation_for, recommendation_for
from src.models.efficientnet import load_checkpoint
from src.utils.seed import get_device

# Phase 1 supports exactly one species; species identification becomes its
# own model in a later milestone (see docs/STEP_BY_STEP.md).
SPECIES_NAME = "Kappaphycus alvarezii"


@dataclass
class PredictionResult:
    species: str
    health: str
    confidence: float
    explanation: str
    recommendation: str
    gradcam_base64_png: str


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.device = get_device(config.device)
        checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
        self.model, self.class_names = load_checkpoint(checkpoint_path, self.device)
        self.transform = build_transforms(config, train=False)

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            class_index = int(probs.argmax().item())
            confidence = float(probs[class_index].item())

        label = self.class_names[class_index]

        overlay = generate_gradcam(self.model, image, class_index, self.device)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        gradcam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return PredictionResult(
            species=SPECIES_NAME,
            health=label,
            confidence=confidence,
            explanation=explanation_for(label),
            recommendation=recommendation_for(label),
            gradcam_base64_png=gradcam_b64,
        )
