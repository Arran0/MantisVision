"""Single entrypoint for turning an uploaded image into the full schema-driven
report: one result per measurement (classification value + confidence,
regression value, or segmentation coverage/mask). Fully generic — nothing is
picked out as "primary"/"species"/"health" by a hardcoded key name; a client
renders whichever measurements are present, in schema order, using each
result's own `label`/`type`/`unit`/etc. Used by the FastAPI inference service
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


@dataclass
class MeasurementResult:
    type: str  # "classification" | "regression" | "segmentation"
    label: str
    value: str | float | None  # class name, numeric value, or None (segmentation / not applicable)
    confidence: float | None  # classification only
    explanation: str | None
    recommendation: str | None
    unit: str | None  # regression only
    min: float | None  # regression only — bounds for client-side meter scaling
    max: float | None  # regression only
    coverage: dict[str, float] | None  # segmentation only: {seg_class_name: pct_of_frame}
    seg_colors: dict[str, str] | None  # segmentation only: {seg_class_name: hex color}
    mask_png_base64: str | None  # segmentation only, "" unless ENABLE_SEGMENTATION_OVERLAY
    gradcam_png_base64: str | None  # classification only, "" unless ENABLE_GRADCAM and this is the target head


@dataclass
class PredictionResult:
    # Every measurement in the active schema, keyed by measurement key and in
    # schema order — the full, schema-driven report. A measurement gated off
    # by applies_when (e.g. health_status when seaweed_presence == "No")
    # still appears here with value=None, rather than being omitted, so a
    # client can distinguish "not applicable" from "not in this schema".
    measurements: dict[str, MeasurementResult]


def _empty_result(type_: str, label: str) -> MeasurementResult:
    return MeasurementResult(
        type=type_,
        label=label,
        value=None,
        confidence=None,
        explanation=None,
        recommendation=None,
        unit=None,
        min=None,
        max=None,
        coverage=None,
        seg_colors=None,
        mask_png_base64=None,
        gradcam_png_base64=None,
    )


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

        # A Grad-CAM backward pass is expensive (see ENABLE_GRADCAM above), so
        # only ever compute one per request rather than one per classification
        # head. There's no "primary" concept in the schema to defer to, so we
        # just pick the first classification measurement in schema order —
        # deterministic, and not tied to any particular key name.
        gradcam_target_key = next((m.key for m in schema.measurements if m.type == "classification"), None)

        measurements: dict[str, MeasurementResult] = {}
        for m in schema.measurements:
            applies = schema.applies(m, predicted_class)

            if m.type == "classification":
                if not applies:
                    measurements[m.key] = _empty_result("classification", m.label)
                    continue
                class_name = predicted_class[m.key]
                class_def = next((c for c in m.classes if c.name == class_name), None)
                gradcam_b64 = (
                    self._maybe_gradcam(image, m.key, predicted_index[m.key]) if m.key == gradcam_target_key else None
                )
                measurements[m.key] = MeasurementResult(
                    type="classification",
                    label=m.label,
                    value=class_name,
                    confidence=classification_confidence[m.key],
                    explanation=(class_def.explanation if class_def else None),
                    recommendation=_augmented_recommendation(m, class_def, schema, predicted_class),
                    unit=None,
                    min=None,
                    max=None,
                    coverage=None,
                    seg_colors=None,
                    mask_png_base64=None,
                    gradcam_png_base64=gradcam_b64,
                )
            elif m.type == "regression":
                raw = float(outputs[m.key].squeeze(0).item())
                range_def = m.range_for(raw) if applies else None
                measurements[m.key] = MeasurementResult(
                    type="regression",
                    label=m.label,
                    value=round(raw, 1) if applies else None,
                    confidence=None,
                    explanation=(range_def.explanation if range_def else None),
                    recommendation=(range_def.recommendation if range_def else None),
                    unit=m.unit,
                    min=m.min,
                    max=m.max,
                    coverage=None,
                    seg_colors=None,
                    mask_png_base64=None,
                    gradcam_png_base64=None,
                )
            elif m.type == "segmentation":
                if not applies:
                    measurements[m.key] = _empty_result("segmentation", m.label)
                    continue
                probs = F.softmax(outputs[m.key], dim=1).squeeze(0)
                class_map = probs.argmax(dim=0)
                total = class_map.numel()
                coverage = {
                    seg_class.name: round(float((class_map == i).sum().item()) / total * 100.0, 1)
                    for i, seg_class in enumerate(m.seg_classes)
                }
                seg_colors = {seg_class.name: seg_class.color for seg_class in m.seg_classes}
                mask_b64 = _encode_seg_overlay_png(class_map, m.seg_classes) if ENABLE_SEGMENTATION_OVERLAY else ""
                measurements[m.key] = MeasurementResult(
                    type="segmentation",
                    label=m.label,
                    value=None,
                    confidence=None,
                    explanation=None,
                    recommendation=None,
                    unit=None,
                    min=None,
                    max=None,
                    coverage=coverage,
                    seg_colors=seg_colors,
                    mask_png_base64=mask_b64,
                    gradcam_png_base64=None,
                )

        del input_tensor
        return PredictionResult(measurements=measurements)

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
