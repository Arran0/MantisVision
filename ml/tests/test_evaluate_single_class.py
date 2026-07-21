"""Regression tests for the 'float' object is not iterable crash seen in
production retraining runs (GitHub Actions runs 29800432023 and 29802935957).

Root cause: evaluate()'s per-class ROC AUC previously relied on a single
roc_auc_score(multi_class="ovr", average=None) call, whose return SHAPE
depends on the data. With a small/sparse labeled set a test split routinely
contains only one of a measurement's classes, and sklearn then returns a
bare NaN *scalar* (for a 2-declared-class measurement) instead of a
per-class array — and the old code iterated that scalar.

These build tiny real checkpoints + datasets (mirroring test_train_smoke.py's
style) and run the real evaluate() entrypoint end-to-end for the exact
degenerate shapes that occur when an admin adds a measurement and labels only
a handful of images:
  - a 1-class classification (species, seen in run 29800432023)
  - a 2-class classification with only ONE class present in the test split
    (is_decayed, the boolean measurement that still crashed in run 29802935957)
  - a 2-class classification with BOTH classes present (AUC is well-defined)
"""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from config import ClassDef, Config, MeasurementDef, Schema
from src.evaluate import evaluate
from src.models.efficientnet import build_model, save_checkpoint

IMAGE_SIZE = 32


def _schema(measurement: MeasurementDef) -> Schema:
    return Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[measurement])


def _write_split(split_dir, rows) -> None:
    """rows: list of (filename, measurements_dict)."""
    (split_dir / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    manifest_lines = []
    for filename, measurements in rows:
        array = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(split_dir / "images" / filename)
        manifest_lines.append(json.dumps({"filename": filename, "measurements": measurements, "masks": {}}))
    (split_dir / "annotations.jsonl").write_text("\n".join(manifest_lines) + "\n")


def _cfg(tmp_path, dataset_root) -> Config:
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
    return cfg


def _run(tmp_path, measurement: MeasurementDef, splits: dict) -> dict:
    schema = _schema(measurement)
    dataset_root = tmp_path / "dataset"
    for split_name, rows in splits.items():
        _write_split(dataset_root / split_name, rows)
    cfg = _cfg(tmp_path, dataset_root)
    model = build_model(schema, freeze_backbone=False, pretrained=False)
    save_checkpoint(model, schema, cfg.checkpoints_dir / "best_model.pt")
    return evaluate(cfg=cfg)


def test_evaluate_single_class_measurement(tmp_path):
    species = MeasurementDef(
        key="species", label="Species", type="classification", loss_weight=1.0,
        classes=[ClassDef(name="Kappaphycus_alvarezii")],
    )
    only = {"species": "Kappaphycus_alvarezii"}
    results = _run(
        tmp_path, species,
        {"train": [("a.jpg", only), ("b.jpg", only)], "validation": [("c.jpg", only)],
         "test": [("d.jpg", only), ("e.jpg", only), ("f.jpg", only)]},
    )
    assert results["species"]["per_class"]["Kappaphycus_alvarezii"]["roc_auc"] is None


def test_evaluate_two_class_measurement_with_only_one_class_present(tmp_path):
    # The is_decayed / is_dried boolean case from run 29802935957: 2 classes
    # declared but the test split has only 'not_decayed'. sklearn's single-call
    # OvR returns a bare NaN scalar here — the exact shape that crashed before.
    is_decayed = MeasurementDef(
        key="is_decayed", label="Is decayed", type="classification", loss_weight=1.0,
        classes=[ClassDef(name="decayed"), ClassDef(name="not_decayed")],
    )
    nd = {"is_decayed": "not_decayed"}
    results = _run(
        tmp_path, is_decayed,
        {"train": [("a.jpg", nd), ("b.jpg", nd)], "validation": [("c.jpg", nd)],
         "test": [("d.jpg", nd), ("e.jpg", nd), ("f.jpg", nd)]},
    )
    per_class = results["is_decayed"]["per_class"]
    # Undefined (only one class present) -> None, not a crash.
    assert per_class["decayed"]["roc_auc"] is None
    assert per_class["not_decayed"]["roc_auc"] is None


def test_evaluate_two_class_measurement_with_both_classes_present(tmp_path):
    # When both classes appear, per-class OvR AUC is well-defined and returns a
    # real float (an improvement over the old code, which raised ValueError for
    # a genuinely-binary split and reported no AUC at all).
    is_decayed = MeasurementDef(
        key="is_decayed", label="Is decayed", type="classification", loss_weight=1.0,
        classes=[ClassDef(name="decayed"), ClassDef(name="not_decayed")],
    )
    d, nd = {"is_decayed": "decayed"}, {"is_decayed": "not_decayed"}
    results = _run(
        tmp_path, is_decayed,
        {"train": [("a.jpg", d), ("b.jpg", nd)], "validation": [("c.jpg", d)],
         "test": [("d.jpg", d), ("e.jpg", nd), ("f.jpg", d), ("g.jpg", nd)]},
    )
    per_class = results["is_decayed"]["per_class"]
    for name in ("decayed", "not_decayed"):
        auc = per_class[name]["roc_auc"]
        assert auc is None or isinstance(auc, float)
