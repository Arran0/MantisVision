"""Single entrypoint for turning an uploaded image into the full MVP output:
species, health, confidence, explanation, recommendation, and a Grad-CAM
heatmap. Used by the FastAPI inference service (src/api/main.py).
"""
from __future__ import annotations

import base64
import gc
import io
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from config import config
from src.data.transforms import build_transforms
from src.inference.explanations import explanation_for, recommendation_for
from src.models.efficientnet import load_checkpoint
from src.utils.seed import get_device

# Phase 1 supports exactly one species; species identification becomes its
# own model in a later milestone (see docs/STEP_BY_STEP.md).
SPECIES_NAME = "Kappaphycus alvarezii"

# The Grad-CAM backward pass roughly doubles peak memory and drags in OpenCV,
# which pushes a 512 MB free-tier host over its limit. It's therefore opt-in:
# set ENABLE_GRADCAM=true only where there's enough RAM (~1 GB+). When off,
# gradcam_png_base64 comes back empty and the web UI simply omits the heatmap.
ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "false").lower() in ("1", "true", "yes")


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
        # Cap intra-op threads: on a small shared host the extra worker threads
        # cost memory without meaningfully speeding up single-image inference.
        torch.set_num_threads(1)
        self.device = get_device(config.device)
        checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
        self.model, self.class_names = load_checkpoint(checkpoint_path, self.device)
        self.transform = build_transforms(config, train=False)
        gc.collect()

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)
            class_index = int(probs.argmax().item())
            confidence = float(probs[class_index].item())

        label = self.class_names[class_index]

        gradcam_b64 = ""
        if ENABLE_GRADCAM:
            # Imported lazily so the pytorch-grad-cam / OpenCV stack is only
            # loaded (and only costs memory) when explicitly enabled.
            from src.gradcam import generate_gradcam

            overlay = generate_gradcam(self.model, image, class_index, self.device)
            buffer = io.BytesIO()
            Image.fromarray(overlay).save(buffer, format="PNG")
            gradcam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        del input_tensor
        return PredictionResult(
            species=SPECIES_NAME,
            health=label,
            confidence=confidence,
            explanation=explanation_for(label),
            recommendation=recommendation_for(label),
            gradcam_base64_png=gradcam_b64,
        )
