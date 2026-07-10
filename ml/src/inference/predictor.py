"""Single entrypoint for turning an uploaded image into the full multi-head
output: species, condition, derived health level + 0-100 score, disease
subtype, dried/decayed extent, explanation, recommendation, and a Grad-CAM
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

from config import DISEASE_MODERATE_MIN, SPECIES, config
from src.data.labels import BACKGROUND
from src.data.transforms import build_transforms
from src.inference.explanations import explanation_for, recommendation_for
from src.models.efficientnet import load_checkpoint
from src.utils.seed import get_device

# The Grad-CAM backward pass roughly doubles peak memory and drags in OpenCV,
# which pushes a 512 MB free-tier host over its limit. Opt-in via ENABLE_GRADCAM.
ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "false").lower() in ("1", "true", "yes")


@dataclass
class PredictionResult:
    species: str
    is_seaweed: bool
    condition: str
    health: str | None  # derived display level: Healthy/Moderate/Low, or None for Background
    health_score: float | None  # 0-100, None for Background
    confidence: float  # condition-head confidence
    disease_subtype: str | None
    dried_pct: float | None
    decayed_pct: float | None
    explanation: str
    recommendation: str
    gradcam_base64_png: str


def _derive_level(condition: str, health_score: float) -> str | None:
    """Discrete display level from condition + regressed score. Disease is
    split by the score (severity is read back out here, not from a folder)."""
    if condition == "Healthy":
        return "Healthy"
    if condition in ("Decay", "Dried"):
        return "Low"
    if condition == "Disease":
        return "Moderate" if health_score >= DISEASE_MODERATE_MIN else "Low"
    return None


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        # Cap intra-op threads: on a small shared host the extra worker threads
        # cost memory without meaningfully speeding up single-image inference.
        torch.set_num_threads(1)
        self.device = get_device(config.device)
        checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
        self.model, self.condition_classes, self.subtype_classes, species = load_checkpoint(
            checkpoint_path, self.device
        )
        self.species_name = (species or {}).get("name") or SPECIES["name"]
        self.transform = build_transforms(config, train=False)
        gc.collect()

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_tensor)
            cond_probs = F.softmax(outputs["condition"], dim=1).squeeze(0)
            cond_index = int(cond_probs.argmax().item())
            confidence = float(cond_probs[cond_index].item())
            health_score = float(outputs["health_score"].squeeze(0).item())
            dried_pct = float(outputs["dried_extent"].squeeze(0).item())
            decayed_pct = float(outputs["decayed_extent"].squeeze(0).item())
            subtype_index = int(outputs["disease_subtype"].squeeze(0).argmax().item())

        condition = self.condition_classes[cond_index]

        # Background short-circuit: the whole point of the N+1 class is to
        # refuse to invent a health assessment for a non-seaweed image.
        if condition == BACKGROUND:
            gradcam_b64 = self._maybe_gradcam(image, cond_index)
            del input_tensor
            return PredictionResult(
                species=self.species_name,
                is_seaweed=False,
                condition=condition,
                health=None,
                health_score=None,
                confidence=confidence,
                disease_subtype=None,
                dried_pct=None,
                decayed_pct=None,
                explanation=explanation_for(condition),
                recommendation=recommendation_for(condition),
                gradcam_base64_png=gradcam_b64,
            )

        level = _derive_level(condition, health_score)
        subtype = self.subtype_classes[subtype_index] if condition == "Disease" else None

        gradcam_b64 = self._maybe_gradcam(image, cond_index)
        del input_tensor
        return PredictionResult(
            species=self.species_name,
            is_seaweed=True,
            condition=condition,
            health=level,
            health_score=round(health_score, 1),
            confidence=confidence,
            disease_subtype=subtype,
            dried_pct=round(dried_pct, 1),
            decayed_pct=round(decayed_pct, 1),
            explanation=explanation_for(condition, subtype, level),
            recommendation=recommendation_for(condition, subtype),
            gradcam_base64_png=gradcam_b64,
        )

    def _maybe_gradcam(self, image: Image.Image, class_index: int) -> str:
        if not ENABLE_GRADCAM:
            return ""
        # Imported lazily so the pytorch-grad-cam / OpenCV stack is only loaded
        # (and only costs memory) when a heatmap is actually requested.
        from src.gradcam import generate_gradcam

        overlay = generate_gradcam(self.model, image, class_index, self.device)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
