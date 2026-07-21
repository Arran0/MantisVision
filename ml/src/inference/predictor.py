"""Single entrypoint for turning an uploaded image into the full schema-driven
report: one result per measurement (classification value + confidence,
regression value, or segmentation coverage/mask), plus a legacy flat-field
view (species, condition, health, health_score, ...) populated from the
schema's primary classification and well-known measurement keys, so the
current PWA keeps working unchanged. Used by the FastAPI inference service
(src/api/main.py).

Preset explanation/recommendation copy comes from the checkpoint's own
schema now (ClassDef.explanation/recommendation/note for a classification,
RangeDef.explanation/recommendation per band of a regression's predicted
value), not a hardcoded dict — promoting a new checkpoint therefore
hot-swaps both the model weights and this copy together.
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

# Bucketing the regressed health_score into a coarse Healthy/Moderate/Low
# display level for the PWA — purely score-based against the schema's two
# thresholds, uniformly for any non-background subject. No condition/class
# name is special-cased: an admin-renamed or brand-new condition gets the
# same treatment as any other, since the level is a property of the score,
# not of which class was predicted.
def _derive_level(health_score: float, moderate_min: float, healthy_min: float) -> str:
    if health_score >= healthy_min:
        return "Healthy"
    if health_score >= moderate_min:
        return "Moderate"
    return "Low"


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


def _augmented_recommendation(m: MeasurementDef, class_def: ClassDef | None, schema: Schema, predicted: dict) -> str | None:
    """A classification's recommendation, plus any *child* measurement's
    predicted class note — the generic form of "Disease's recommendation
    plus the predicted subtype's note" (child = a measurement whose
    applies_when gates on this exact m.key/class_def.name). Returns None
    (rather than a filler string) when this particular measurement has
    nothing to say, so an aggregate across every measurement isn't padded
    with placeholder noise from ones the admin hasn't written copy for yet —
    the "nothing to show at all" fallback is applied once, downstream."""
    base = class_def.recommendation if class_def and class_def.recommendation else None
    notes: list[str] = []
    if class_def is not None:
        for child in schema.measurements:
            if child.type != "classification":
                continue
            gates_on_this = any(cond.key == m.key and cond.equals == class_def.name for cond in child.applies_when)
            if not gates_on_this:
                continue
            child_class_name = predicted.get(child.key)
            child_class_def = next((c for c in child.classes if c.name == child_class_name), None)
            if child_class_def and child_class_def.note:
                notes.append(child_class_def.note)
    parts = ([base] if base else []) + notes
    return " ".join(parts) if parts else None


def _collect_copy(schema: Schema, measurements: dict[str, MeasurementResult]) -> tuple[str, str]:
    """Combine every applicable measurement's explanation/recommendation into
    the flat top-level fields the current PWA renders, instead of surfacing
    only one measurement's copy (e.g. health_status) and silently dropping
    the rest (disease, colour, any regression's per-range copy, ...).
    Measurements that don't apply (gated off by applies_when) or that have no
    admin-authored copy contribute nothing; schema order determines the
    order the sentences appear in."""
    explanations = [
        result.explanation
        for m in schema.measurements
        if (result := measurements.get(m.key)) is not None and result.explanation
    ]
    recommendations = [
        result.recommendation
        for m in schema.measurements
        if (result := measurements.get(m.key)) is not None and result.recommendation
    ]
    explanation = " ".join(explanations) if explanations else "No explanation available yet."
    recommendation = " ".join(recommendations) if recommendations else "No recommendation available yet."
    return explanation, recommendation


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
                range_def = m.range_for(raw) if applies else None
                measurements[m.key] = MeasurementResult(
                    type="regression",
                    value=round(raw, 1) if applies else None,
                    confidence=None,
                    explanation=(range_def.explanation if range_def else None),
                    recommendation=(range_def.recommendation if range_def else None),
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

        # --- Legacy flat fields, from the well-known measurement keys when the
        # schema has them, falling back to the older names so pre-restructure
        # checkpoints keep populating the same PWA shape. ---
        confidence = classification_confidence.get(primary.key, 0.0) if primary else 0.0

        def first_result(*keys: str):
            for key in keys:
                result = measurements.get(key)
                if result is not None:
                    return result
            return None

        # Health status is now a labeled class (Healthy/Moderate/Low); older
        # checkpoints instead regressed a health_score we bucket into a level.
        health_status_result = measurements.get("health_status")
        health_score_result = measurements.get("health_score")
        health_score_value = (
            health_score_result.value if health_score_result and isinstance(health_score_result.value, (int, float)) else None
        )
        if health_status_result is not None and isinstance(health_status_result.value, str):
            level = health_status_result.value
        elif health_score_value is not None:
            level = _derive_level(health_score_value, schema.health_moderate_min, schema.health_healthy_min)
        else:
            level = None

        # Species is a real predicted classification now (see the "species"
        # measurement in DEFAULT_SCHEMA), not a fixed schema-wide constant —
        # older checkpoints predate it entirely, hence the fallback.
        species_result = measurements.get("species")
        species_value = (
            species_result.value if species_result and isinstance(species_result.value, str) else "Unknown species"
        )

        dried_result = first_result("dried", "dried_extent")
        decayed_result = first_result("decayed", "decayed_extent")
        disease_result = first_result("disease", "disease_subtype")
        disease_value = disease_result.value if disease_result and isinstance(disease_result.value, str) else None
        # "NoDisease" is the explicit no-finding class — surface it as no subtype.
        if disease_value == "NoDisease":
            disease_value = None

        # For the flat `condition` field, prefer the health status label.
        condition_name = (
            health_status_result.value
            if (health_status_result is not None and isinstance(health_status_result.value, str))
            else (predicted_class.get(primary.key, "Unknown") if primary else "Unknown")
        )

        gradcam_b64 = ""
        if primary is not None:
            gradcam_b64 = self._maybe_gradcam(image, primary.key, predicted_index[primary.key])

        explanation, recommendation = _collect_copy(schema, measurements)

        del input_tensor
        return PredictionResult(
            species=species_value,
            is_seaweed=not is_background,
            condition=condition_name,
            health=level,
            health_score=health_score_value,
            confidence=confidence,
            disease_subtype=disease_value,
            dried_pct=(dried_result.value if dried_result and isinstance(dried_result.value, (int, float)) else None),
            decayed_pct=(decayed_result.value if decayed_result and isinstance(decayed_result.value, (int, float)) else None),
            explanation=explanation,
            recommendation=recommendation,
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
