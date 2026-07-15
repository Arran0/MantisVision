"""Unit tests for _verify_dataset's dataset-adequacy checks (src/data/dataset.py).

An empty split used to slip straight through into training and crash deep
inside train.py's run_epoch with an opaque ZeroDivisionError ("float division
by zero") — this happens in practice with only a handful of labeled images,
since scripts/split_dataset.py's fixed 70/15/15 ratio can round a small pool
down to zero for a given split. These tests lock in the fix: an empty split
now raises a clear, actionable ValueError before any training starts.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from config import ClassDef, Config, MeasurementDef, Schema
from src.data.annotations import AnnotationRow
from src.data.dataset import _verify_dataset, get_dataloaders

IMAGE_SIZE = 32


def _schema_with_background() -> Schema:
    condition = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background",
        classes=[ClassDef(name="Background"), ClassDef(name="Healthy")],
    )
    return Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[condition])


def _schema_without_background() -> Schema:
    # Mirrors the current DEFAULT_SCHEMA: no measurement declares a
    # background_class at all (see the "drop background_class requirement"
    # migration) — schema.primary_classification() returns None for this.
    presence = MeasurementDef(
        key="seaweed_presence",
        label="Seaweed presence",
        type="classification",
        loss_weight=1.0,
        classes=[ClassDef(name="Yes"), ClassDef(name="No")],
    )
    return Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[presence])


def _write_split(split_dir, rows_spec):
    (split_dir / "images").mkdir(parents=True)
    manifest_lines = []
    rng = np.random.default_rng(0)
    for filename, measurements in rows_spec:
        array = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(split_dir / "images" / filename)
        manifest_lines.append(json.dumps({"filename": filename, "measurements": measurements, "masks": {}}))
    (split_dir / "annotations.jsonl").write_text("\n".join(manifest_lines) + "\n" if manifest_lines else "")


def test_verify_dataset_raises_on_empty_split_even_without_background_class():
    """The bug: with no measurement declaring a background_class (today's
    default schema), primary_classification() is None and the old code
    returned early with no check at all, letting a totally empty split
    reach training. Now checked unconditionally, before the background-class
    check even applies."""
    with pytest.raises(ValueError, match="'train' split has no images"):
        _verify_dataset([], _schema_without_background(), "train")


def test_verify_dataset_raises_on_empty_split_with_background_class():
    with pytest.raises(ValueError, match="'validation' split has no images"):
        _verify_dataset([], _schema_with_background(), "validation")


def test_verify_dataset_still_enforces_background_sample_when_declared():
    schema = _schema_with_background()
    non_empty_no_background = [AnnotationRow(filename="a.jpg", measurements={"condition": "Healthy"}, masks={})]
    with pytest.raises(ValueError, match="No 'Background'"):
        _verify_dataset(non_empty_no_background, schema, "test")


def test_get_dataloaders_raises_clear_error_instead_of_zero_division_on_empty_train_split(tmp_path):
    """End-to-end regression test for the reported crash: a train split with
    zero rows (e.g. from a too-small labeled pool being split 70/15/15) must
    fail with the actionable ValueError from _verify_dataset, not reach
    run_epoch's `total_loss / total` and raise ZeroDivisionError."""
    schema = _schema_without_background()
    dataset_root = tmp_path / "dataset"

    _write_split(dataset_root / "train", [])  # the reported failure case
    _write_split(dataset_root / "validation", [("v1.jpg", {"seaweed_presence": "Yes"})])
    _write_split(dataset_root / "test", [("t1.jpg", {"seaweed_presence": "Yes"})])

    cfg = Config(dataset_root=dataset_root, image_size=IMAGE_SIZE, batch_size=1, num_workers=0, device="cpu")

    with pytest.raises(ValueError, match="'train' split has no images"):
        get_dataloaders(cfg, schema)
