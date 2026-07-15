"""Unit tests for src/data/annotations.py — the column/CSV-style per-image
annotation layer that replaced folder-name-encoded labels."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from config import AppliesWhen, ClassDef, MeasurementDef, Schema
from src.data.annotations import AnnotationRow, derive_targets, load_manifest, load_segmentation_target, measurement_applies


def _tiny_schema() -> Schema:
    condition = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background",
        classes=[ClassDef(name="Background"), ClassDef(name="Healthy"), ClassDef(name="Disease")],
    )
    subtype = MeasurementDef(
        key="disease_subtype",
        label="Disease subtype",
        type="classification",
        loss_weight=0.5,
        applies_when=[AppliesWhen(key="condition", equals="Disease")],
        classes=[ClassDef(name="IceIce"), ClassDef(name="Unknown")],
    )
    health = MeasurementDef(
        key="health_score",
        label="Health score",
        type="regression",
        loss_weight=1.0,
        min=0.0,
        max=100.0,
        applies_when=[AppliesWhen(key="condition", not_equals="Background")],
    )
    return Schema(
        health_moderate_min=45.0,
        health_healthy_min=75.0,
        measurements=[condition, subtype, health],
    )


def test_derive_targets_classification_present():
    schema = _tiny_schema()
    targets = derive_targets(schema, {"condition": "Healthy"})
    assert targets["condition_id"] == 1  # index of "Healthy"
    assert targets["condition_mask"] == 1.0


def test_derive_targets_classification_missing_value_is_masked():
    schema = _tiny_schema()
    targets = derive_targets(schema, {})
    assert targets["condition_mask"] == 0.0


def test_derive_targets_unknown_class_name_is_masked():
    schema = _tiny_schema()
    targets = derive_targets(schema, {"condition": "NotARealClass"})
    assert targets["condition_mask"] == 0.0


def test_applies_when_gates_disease_subtype():
    schema = _tiny_schema()
    subtype = schema.find("disease_subtype")

    # Only applies (and is trained) when condition == "Disease".
    assert measurement_applies(subtype, {"condition": "Disease"}) is True
    assert measurement_applies(subtype, {"condition": "Healthy"}) is False
    assert measurement_applies(subtype, {}) is False

    targets_with_disease = derive_targets(schema, {"condition": "Disease", "disease_subtype": "IceIce"})
    assert targets_with_disease["disease_subtype_mask"] == 1.0
    assert targets_with_disease["disease_subtype_id"] == 0  # index of "IceIce"

    # Even if a subtype value is present, it's masked out unless condition == Disease.
    targets_without_disease = derive_targets(schema, {"condition": "Healthy", "disease_subtype": "IceIce"})
    assert targets_without_disease["disease_subtype_mask"] == 0.0


def test_applies_when_not_equals_gates_regression():
    schema = _tiny_schema()
    health = schema.find("health_score")

    assert measurement_applies(health, {"condition": "Healthy"}) is True
    assert measurement_applies(health, {"condition": "Disease"}) is True
    assert measurement_applies(health, {"condition": "Background"}) is False

    background_targets = derive_targets(schema, {"condition": "Background", "health_score": 90})
    assert background_targets["health_score_mask"] == 0.0

    healthy_targets = derive_targets(schema, {"condition": "Healthy", "health_score": 82.5})
    assert healthy_targets["health_score_mask"] == 1.0
    assert healthy_targets["health_score"] == 82.5


def test_regression_missing_value_is_masked():
    schema = _tiny_schema()
    targets = derive_targets(schema, {"condition": "Healthy"})
    assert targets["health_score_mask"] == 0.0
    assert targets["health_score"] == 0.0


def test_load_manifest_roundtrip(tmp_path):
    manifest_path = tmp_path / "annotations.jsonl"
    manifest_path.write_text(
        '{"filename": "a.jpg", "measurements": {"condition": "Healthy"}}\n'
        '{"filename": "b.jpg", "measurements": {"condition": "Background"}, "masks": {}}\n'
    )
    rows = load_manifest(manifest_path)
    assert len(rows) == 2
    assert rows[0].filename == "a.jpg"
    assert rows[0].measurements["condition"] == "Healthy"
    assert rows[1].measurements["condition"] == "Background"


def test_load_segmentation_target_missing_mask_returns_flag_zero(tmp_path):
    seg_measurement = MeasurementDef(key="biofouling", label="Biofouling", type="segmentation", loss_weight=1.0)
    row = AnnotationRow(filename="a.jpg", measurements={}, masks={})
    mask, flag = load_segmentation_target(tmp_path, seg_measurement, row)
    assert flag == 0.0
    assert mask.shape == (1, 1)


def test_load_segmentation_target_present_mask_returns_class_indices(tmp_path):
    seg_measurement = MeasurementDef(key="biofouling", label="Biofouling", type="segmentation", loss_weight=1.0)
    mask_dir = tmp_path / "masks" / "biofouling"
    mask_dir.mkdir(parents=True)
    array = np.zeros((8, 8), dtype=np.uint8)
    array[2:5, 2:5] = 1  # "algae" class
    Image.fromarray(array, mode="L").save(mask_dir / "a.png")

    row = AnnotationRow(filename="a.jpg", measurements={}, masks={"biofouling": "a.png"})
    mask, flag = load_segmentation_target(tmp_path, seg_measurement, row)
    assert flag == 1.0
    assert mask.shape == (8, 8)
    assert torch.equal(mask[2:5, 2:5], torch.ones((3, 3), dtype=torch.long))
    assert mask[0, 0].item() == 0
