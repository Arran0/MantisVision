"""Single entrypoint for turning an uploaded image into the full MVP output:
species, category, condition, health score, confidence, explanation bullets,
recommendation, and a Grad-CAM heatmap. Used by the FastAPI inference service
(src/api/main.py).
"""
from __future__ import annotations

import base64
import gc
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from config import SPECIES_DISPLAY_NAME, config
from src.data.transforms import build_transforms
from src.inference.explanations import explanation_bullets_for, recommendation_for
from src.models.efficientnet import load_checkpoint
from src.utils.seed import get_device

# The Grad-CAM backward pass roughly doubles peak memory and drags in OpenCV,
# which pushes a 512 MB free-tier host over its limit. It's therefore opt-in:
# set ENABLE_GRADCAM=true only where there's enough RAM (~1 GB+). When off,
# gradcam_png_base64 comes back empty and the web UI simply omits the heatmap.
ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "false").lower() in ("1", "true", "yes")


@dataclass
class PredictionResult:
    species: str
    category: str
    condition: str | None
    health_score: float
    confidence: float
    confidence_calibrated: float | None
    explanation_bullets: list[str]
    recommendation: str
    gradcam_base64_png: str


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        # Cap intra-op threads: on a small shared host the extra worker threads
        # cost memory without meaningfully speeding up single-image inference.
        torch.set_num_threads(1)
        self.device = get_device(config.device)
        checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
        self.model, self.category_names, self.condition_names = load_checkpoint(checkpoint_path, self.device)
        self.healthy_idx = self.category_names.index("Healthy")
        self.transform = build_transforms(config, train=False)

        # Calibration is optional: predictions work fine without it, just with
        # confidence_calibrated left as None (an honest signal that no
        # calibration check has been run yet — see src/calibrate.py).
        self.temperature: float | None = None
        calibration_path = checkpoint_path.parent / "calibration.json"
        if calibration_path.exists():
            self.temperature = json.loads(calibration_path.read_text()).get("temperature")

        gc.collect()

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            category_logits, condition_logits, score = self.model(input_tensor)
            category_probs = F.softmax(category_logits, dim=1).squeeze(0)
            category_idx = int(category_probs.argmax().item())
            confidence = float(category_probs[category_idx].item())

            confidence_calibrated = None
            if self.temperature:
                calibrated_probs = F.softmax(category_logits / self.temperature, dim=1).squeeze(0)
                confidence_calibrated = float(calibrated_probs[category_idx].item())

            category = self.category_names[category_idx]

            # The condition head is never trained on Healthy samples (masked
            # loss, see src/train.py), so its output there is undefined —
            # force it to None rather than trust an unsupervised guess.
            condition = None
            if category != self.category_names[self.healthy_idx]:
                condition_idx = int(F.softmax(condition_logits, dim=1).squeeze(0).argmax().item())
                condition_name = self.condition_names[condition_idx]
                condition = condition_name if condition_name != "None" else None

            health_score = float(score.squeeze(0).item())

        gradcam_b64 = ""
        if ENABLE_GRADCAM:
            # Imported lazily so the pytorch-grad-cam / OpenCV stack is only
            # loaded (and only costs memory) when explicitly enabled.
            from src.gradcam import generate_gradcam

            overlay = generate_gradcam(self.model, image, category_idx, self.device)
            buffer = io.BytesIO()
            Image.fromarray(overlay).save(buffer, format="PNG")
            gradcam_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        del input_tensor
        return PredictionResult(
            species=SPECIES_DISPLAY_NAME,
            category=category,
            condition=condition,
            health_score=health_score,
            confidence=confidence,
            confidence_calibrated=confidence_calibrated,
            explanation_bullets=explanation_bullets_for(category, condition),
            recommendation=recommendation_for(category, condition),
            gradcam_base64_png=gradcam_b64,
        )
