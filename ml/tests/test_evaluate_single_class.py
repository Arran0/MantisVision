"""Regression test for the 'float' object is not iterable crash seen in
production retraining runs (GitHub Actions run 29800432023): a classification
measurement with only one class (e.g. a freshly admin-added "species" that
hasn't gotten its second class yet) makes sklearn's roc_auc_score fall
through to its binary code path and return a bare NaN scalar instead of a
per-class array, and the old code did `enumerate(auc_scores)` unconditionally.

Builds a tiny real checkpoint + dataset (mirrors test_train_smoke.py's
style) with one classification measurement that has a single class, then
runs the real evaluate() entrypoint end-to-end and asserts it completes
without raising.
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from config import ClassDef, Config, MeasurementDef, Schema
from src.evaluate import evaluate
from src.models.efficientnet import build_model, save_checkpoint

IMAGE_SIZE = 32


def _make_single_class_schema() -> Schema:
    species = MeasurementDef(
        key="species",
        label="Species",
        type="classification",
        loss_weight=1.0,
        classes=[ClassDef(name="Kappaphycus_alvarezii")],
    )
    return Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[species])


def _write_split(split_dir, filenames) -> None:
    (split_dir / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    manifest_lines = []
    for filename in filenames:
        array = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(split_dir / "images" / filename)
        manifest_lines.append(json.dumps({"filename": filename, "measurements": {"species": "Kappaphycus_alvarezii"}, "masks": {}}))
    (split_dir / "annotations.jsonl").write_text("\n".join(manifest_lines) + "\n")


def test_evaluate_does_not_crash_on_a_single_class_measurement(tmp_path):
    schema = _make_single_class_schema()
    dataset_root = tmp_path / "dataset"
    for split, filenames in (
        ("train", ["a.jpg", "b.jpg"]),
        ("validation", ["c.jpg"]),
        ("test", ["d.jpg", "e.jpg", "f.jpg"]),
    ):
        _write_split(dataset_root / split, filenames)

    cfg = Config(
        dataset_root=dataset_root,
        checkpoints_dir=tmp_path / "checkpoints",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
        image_size=IMAGE_SIZE,
        batch_size=4,
        num_workers=0,
        device="cpu",
    )
    cfg.checkpoints_dir.mkdir(parents=True)

    model = build_model(schema, freeze_backbone=False, pretrained=False)
    save_checkpoint(model, schema, cfg.checkpoints_dir / "best_model.pt")

    results = evaluate(cfg=cfg)

    assert results["species"]["per_class"]["Kappaphycus_alvarezii"]["roc_auc"] is None
