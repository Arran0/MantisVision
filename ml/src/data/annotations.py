"""Column/CSV-style per-image annotations — replaces folder-name-encoded
labels (src/data/labels.py's parse_class_folder) as the training pipeline's
source of truth.

A dataset split directory (dataset/<species_slug>/{train,validation,test}/)
now holds:

    images/<filename>              the photo
    masks/<measurement_key>/<file> ground-truth segmentation masks, one
                                    subfolder per segmentation measurement;
                                    single-channel PNG, pixel value = index
                                    into that measurement's seg_classes
    annotations.jsonl               one JSON object per image:
        {"filename": "0001.jpg",
         "measurements": {"condition": "Healthy", "health_score": 82.4},
         "masks": {"biofouling": "0001.png"}}

`measurements` holds classification (class name string) and regression
(numeric) values; `masks` maps a segmentation measurement's key to its mask
filename under masks/<key>/. Both are keyed by the schema's measurement.key
and both are optional per image and per measurement — a measurement missing
from a row simply contributes nothing to that head's loss for that image
(see derive_targets below), which is how "no lab data yet" and applies_when
gating (e.g. disease_subtype only meaningful when condition == "Disease")
both fall out of the same masking mechanism.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import MeasurementDef, Schema


@dataclass
class AnnotationRow:
    filename: str
    measurements: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)


def load_manifest(path: Path) -> list[AnnotationRow]:
    rows: list[AnnotationRow] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            rows.append(
                AnnotationRow(
                    filename=doc["filename"],
                    measurements=doc.get("measurements", {}),
                    masks=doc.get("masks", {}),
                )
            )
    return rows


def measurement_applies(measurement: MeasurementDef, values: dict) -> bool:
    """Mirrors Schema.applies / the TS measurementApplies — kept as a free
    function here too since callers often only have a values dict, not a
    Schema instance, at hand."""
    cond = measurement.applies_when
    if cond is None:
        return True
    parent_value = values.get(cond.key)
    if parent_value is None:
        return False
    if cond.equals is not None:
        return parent_value == cond.equals
    if cond.not_equals is not None:
        return parent_value != cond.not_equals
    return True


def derive_targets(schema: Schema, measurements: dict) -> dict:
    """Turns one image's raw `measurements` dict into per-measurement
    training targets + masks for every classification/regression measurement
    in the schema (segmentation is handled separately by
    load_segmentation_target, since it needs file I/O). A value that's
    missing, or whose applies_when isn't satisfied by this image's other
    values, gets mask 0.0 (contributes nothing to that head's loss)."""
    targets: dict = {}
    for m in schema.measurements:
        if m.type == "segmentation":
            continue
        applies = measurement_applies(m, measurements)
        value = measurements.get(m.key) if applies else None

        if m.type == "classification":
            names = m.class_names()
            if applies and isinstance(value, str) and value in names:
                targets[f"{m.key}_id"] = names.index(value)
                targets[f"{m.key}_mask"] = 1.0
            else:
                targets[f"{m.key}_id"] = 0
                targets[f"{m.key}_mask"] = 0.0
        elif m.type == "regression":
            if applies and isinstance(value, (int, float)) and not isinstance(value, bool):
                targets[m.key] = float(value)
                targets[f"{m.key}_mask"] = 1.0
            else:
                targets[m.key] = 0.0
                targets[f"{m.key}_mask"] = 0.0
    return targets


def load_segmentation_target(
    split_dir: Path, measurement: MeasurementDef, row: AnnotationRow
) -> tuple[torch.Tensor, float]:
    """Loads `measurement`'s ground-truth mask for `row`, if present, as a
    (H, W) long tensor of seg_classes indices plus a 0/1 "do we have a mask
    for this image" flag. Missing/unreadable -> a zero mask with flag 0.0, so
    the loss simply skips this image for this measurement (same masking
    convention as classification/regression)."""
    filename = row.masks.get(measurement.key)
    if not filename:
        return torch.zeros((1, 1), dtype=torch.long), 0.0
    path = split_dir / "masks" / measurement.key / filename
    if not path.exists():
        return torch.zeros((1, 1), dtype=torch.long), 0.0
    image = Image.open(path).convert("L")
    array = np.array(image, dtype=np.int64)
    return torch.from_numpy(array), 1.0
