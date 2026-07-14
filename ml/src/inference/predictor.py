"""Single entrypoint for turning an uploaded image into the full schema-driven
report: one result per measurement (classification value + confidence,
regression value, or segmentation coverage/mask), plus a legacy flat-field
view (species, condition, health, health_score, ...) populated from the
schema's primary classification and well-known measurement keys, so the
current PWA keeps working unchanged. Used by the FastAPI inference service
(src/api/main.py).

Preset explanation/recommendation copy comes from the checkpoint's own
schema now (ClassDef.explanation/recommendation/note), not a hardcoded
dict — promoting a new checkpoint therefore hot-swaps both the model weights
and this copy together.
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

from config import ClassDef, MeasurementDef, Schema, config
from src.data.transforms import build_transforms
from src.models.efficientnet import load_checkpoint
from src.utils.seed import get_device

# The Grad-CAM backward pass roughly doubles peak memory and drags in OpenCV,
# which pushes a 512 MB free-tier host over its limit. Opt-in via ENABLE_GRADCAM.
ENABLE_GRADCAM = os.environ.get("ENABLE_GRADCAM", "false").lower() in ("1", "true", "yes")

# Segmentation overlay PNGs are cheap to compute (a forward-pass argmax, no
# backward pass) but still add response payload size; opt-in like Grad-CAM.
ENABLE_SEGMENTATION_OVERLAY = os.environ.get("ENABLE_SEGMENTATION_OVERLAY", "false").lower() in ("1", "true", "yes")

# Bucketing a regressed health_score into a coarse Healthy/Moderate/Low
# display level is legacy, PWA-facing convenience tied to today's condition
# vocabulary (Healthy/Decay/Dried/Disease) — the schema has no generic notion
# of "which classes get severity-bucketed", so this stays keyed to those
# specific names rather than fully generalized. An admin-renamed or novel
# condition class simply gets no derived level (None), which is a graceful
# degrade, not a crash.
def _derive_level(condition: str, health_score: float, disease_moderate_min: float) -> str | None:
    if condition == "Healthy":
        return "Healthy"
    if condition in ("Decay", "Dried"):
        return "Low"
    if condition == "Disease":
        return "Moderate" if health_score >= disease_moderate_min else "Low"
    return None


@dataclass
class MeasurementResult:
    type: str  # "classification" | "regression" | "segmentation"
    value: str | float | None  # class name, numeric value, or None (segmentation / not applicable)
    confidence: float | None  # classification only
    explanation: str | None
    recommendation: str | None
    coverage: dict[str, float] | None  # segmentation only: {seg_class_name: pct_of_frame}
    mask_png_base64: str | None  # segmentation only, "" unless ENABLE_SEGMENTATION_OVERLAY


@dataclass
class PredictionResult:
    # Legacy flat fields, populated from the primary classification + the
    # well-known measurement keys (health_score/dried_extent/decayed_extent/
    # disease_subtype) when the active schema still has them under those
    # names — kept for the current PWA, which reads exactly this shape.
    species: str
    is_seaweed: bool
    condition: str
    health: str | None
    health_score: float | None
    confidence: float
    disease_subtype: str | None
    dried_pct: float | None
    decayed_pct: float | None
    explanation: str
    recommendation: str
    gradcam_base64_png: str
    # Generic, forward-looking report: every measurement in the schema.
    measurements: dict[str, MeasurementResult]


def _augmented_recommendation(m: MeasurementDef, class_def: ClassDef | None, schema: Schema, predicted: dict) -> str:
    """A classification's recommendation, plus any *child* measurement's
    predicted class note — the generic form of "Disease's recommendation
    plus the predicted subtype's note" (child = a measurement whose
    applies_when gates on this exact m.key/class_def.name)."""
    base = (class_def.recommendation if class_def and class_def.recommendation else None) or (
        "No recommendation available for this class yet."
    )
    notes: list[str] = []
    if class_def is not None:
        for child in schema.measurements:
            if child.type != "classification":
                continue
            cond = child.applies_when
            if not cond or cond.key != m.key or cond.equals != class_def.name:
                continue
            child_class_name = predicted.get(child.key)
            child_class_def = next((c for c in child.classes if c.name == child_class_name), None)
            if child_class_def and child_class_def.note:
                notes.append(child_class_def.note)
    return " ".join([base, *notes]) if notes else base


def _encode_seg_overlay_png(class_map: torch.Tensor, seg_classes: list) -> str:
    import numpy as np

    palette = np.array(
        [tuple(int(c.color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) for c in seg_classes],
        dtype=np.uint8,
    )
    rgb = palette[class_map.cpu().numpy()]
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None) -> None:
        # Cap intra-op threads: on a small shared host the extra worker threads
        # cost memory without meaningfully speeding up single-image inference.
        torch.set_num_threads(1)
        self.device = get_device(config.device)
        checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
        self.model, self.schema = load_checkpoint(checkpoint_path, self.device)
        self.species_name = (
            next((s.name for s in self.schema.species if s.slug == self.schema.active_species_slug), None)
            or (self.schema.species[0].name if self.schema.species else "Unknown species")
        )
        self.transform = build_transforms(config, train=False)
        gc.collect()

    def predict(self, image_bytes: bytes) -> PredictionResult:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_tensor)

        schema = self.schema
        primary = schema.primary_classification()

        # Pass 1: resolve every classification measurement's predicted class
        # first (independent of applies_when), so a later measurement can be
        # gated on an earlier one's prediction regardless of schema order.
        predicted_class: dict[str, str] = {}
        predicted_index: dict[str, int] = {}
        classification_confidence: dict[str, float] = {}
        for m in schema.measurements:
            if m.type != "classification":
                continue
            probs = F.softmax(outputs[m.key], dim=1).squeeze(0)
            index = int(probs.argmax().item())
            predicted_class[m.key] = m.classes[index].name
            predicted_index[m.key] = index
            classification_confidence[m.key] = float(probs[index].item())

        is_background = primary is not None and predicted_class.get(primary.key) == primary.background_class

        measurements: dict[str, MeasurementResult] = {}
        for m in schema.measurements:
            applies = schema.applies(m, predicted_class)

            if m.type == "classification":
                class_name = predicted_class[m.key]
                if not applies:
                    measurements[m.key] = MeasurementResult("classification", None, None, None, None, None, None)
                    continue
                class_def = next((c for c in m.classes if c.name == class_name), None)
                measurements[m.key] = MeasurementResult(
                    type="classification",
                    value=class_name,
                    confidence=classification_confidence[m.key],
                    explanation=(class_def.explanation if class_def else None),
                    recommendation=_augmented_recommendation(m, class_def, schema, predicted_class),
                    coverage=None,
                    mask_png_base64=None,
                )
            elif m.type == "regression":
                raw = float(outputs[m.key].squeeze(0).item())
                measurements[m.key] = MeasurementResult(
                    type="regression",
                    value=round(raw, 1) if applies else None,
                    confidence=None,
                    explanation=None,
                    recommendation=None,
                    coverage=None,
                    mask_png_base64=None,
                )
            elif m.type == "segmentation":
                if not applies:
                    measurements[m.key] = MeasurementResult("segmentation", None, None, None, None, None, None)
                    continue
                probs = F.softmax(outputs[m.key], dim=1).squeeze(0)
                class_map = probs.argmax(dim=0)
                total = class_map.numel()
                coverage = {
                    seg_class.name: round(float((class_map == i).sum().item()) / total * 100.0, 1)
                    for i, seg_class in enumerate(m.seg_classes)
                }
                mask_b64 = _encode_seg_overlay_png(class_map, m.seg_classes) if ENABLE_SEGMENTATION_OVERLAY else ""
                measurements[m.key] = MeasurementResult(
                    type="segmentation",
                    value=None,
                    confidence=None,
                    explanation=None,
                    recommendation=None,
                    coverage=coverage,
                    mask_png_base64=mask_b64,
                )

        # --- Legacy flat fields, from the primary classification + the
        # well-known measurement keys when the schema still has them. ---
        condition_name = predicted_class.get(primary.key, "Unknown") if primary else "Unknown"
        confidence = classification_confidence.get(primary.key, 0.0) if primary else 0.0
        primary_result = measurements.get(primary.key) if primary else None

        health_result = measurements.get("health_score")
        dried_result = measurements.get("dried_extent")
        decayed_result = measurements.get("decayed_extent")
        subtype_result = measurements.get("disease_subtype")

        health_score_value = health_result.value if health_result and isinstance(health_result.value, (int, float)) else None
        level = (
            _derive_level(condition_name, health_score_value, schema.disease_moderate_min)
            if health_score_value is not None
            else None
        )

        gradcam_b64 = ""
        if primary is not None:
            gradcam_b64 = self._maybe_gradcam(image, primary.key, predicted_index[primary.key])

        del input_tensor
        return PredictionResult(
            species=self.species_name,
            is_seaweed=not is_background,
            condition=condition_name,
            health=level,
            health_score=health_score_value,
            confidence=confidence,
            disease_subtype=(subtype_result.value if subtype_result and isinstance(subtype_result.value, str) else None),
            dried_pct=(dried_result.value if dried_result and isinstance(dried_result.value, (int, float)) else None),
            decayed_pct=(decayed_result.value if decayed_result and isinstance(decayed_result.value, (int, float)) else None),
            explanation=(primary_result.explanation if primary_result and primary_result.explanation else "No explanation available yet."),
            recommendation=(primary_result.recommendation if primary_result and primary_result.recommendation else "No recommendation available yet."),
            gradcam_base64_png=gradcam_b64,
            measurements=measurements,
        )

    def _maybe_gradcam(self, image: Image.Image, measurement_key: str, class_index: int) -> str:
        if not ENABLE_GRADCAM:
            return ""
        # Imported lazily so the pytorch-grad-cam / OpenCV stack is only loaded
        # (and only costs memory) when a heatmap is actually requested.
        from src.gradcam import generate_gradcam

        overlay = generate_gradcam(self.model, image, measurement_key, class_index, self.device)
        buffer = io.BytesIO()
        Image.fromarray(overlay).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
