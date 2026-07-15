"""End-to-end smoke test: builds a tiny synthetic dataset (images + a
column-based annotations.jsonl manifest + one segmentation mask) covering all
three measurement types, then runs the real train() entrypoint against it on
CPU and checks it produces a working, schema-carrying checkpoint.

This is the integration-level counterpart to test_annotations.py (pure
target-derivation logic) and test_model_losses.py (pure model/loss shapes and
a hand-built gradient-flow check) — here the full AnnotatedDataset ->
DataLoader -> collate -> model -> loss -> checkpoint pipeline runs together,
the way a real (tiny) retrain would.
"""
from __future__ import annotations

import json
import re

import numpy as np
import torch
from PIL import Image

from config import AppliesWhen, ClassDef, Config, MeasurementDef, SegClassDef, Schema
from src.models.efficientnet import load_checkpoint
from src.train import train

IMAGE_SIZE = 32


def _make_schema() -> Schema:
    condition = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background",
        classes=[ClassDef(name="Background"), ClassDef(name="Healthy"), ClassDef(name="Disease")],
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
    biofouling = MeasurementDef(
        key="biofouling",
        label="Biofouling",
        type="segmentation",
        loss_weight=1.0,
        seg_classes=[SegClassDef(name="background", color="#000000"), SegClassDef(name="algae", color="#22c55e")],
    )
    return Schema(
        health_moderate_min=45.0,
        health_healthy_min=75.0,
        measurements=[condition, health, biofouling],
    )


def _write_split(split_dir, rows_spec):
    """rows_spec: list of (filename, condition, health_score_or_None, has_mask)."""
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "masks" / "biofouling").mkdir(parents=True)

    manifest_lines = []
    rng = np.random.default_rng(0)
    for filename, condition, health_score, has_mask in rows_spec:
        array = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(split_dir / "images" / filename)

        measurements = {"condition": condition}
        masks = {}
        if health_score is not None:
            measurements["health_score"] = health_score
        if has_mask:
            mask_name = filename.replace(".jpg", ".png")
            mask_array = rng.integers(0, 2, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
            Image.fromarray(mask_array, mode="L").save(split_dir / "masks" / "biofouling" / mask_name)
            masks["biofouling"] = mask_name

        manifest_lines.append(json.dumps({"filename": filename, "measurements": measurements, "masks": masks}))

    (split_dir / "annotations.jsonl").write_text("\n".join(manifest_lines) + "\n")


def _build_synthetic_dataset(tmp_path, schema: Schema) -> Config:
    dataset_root = tmp_path / "dataset"

    # Every split needs at least one Background sample (the model's negative
    # class) plus at least one non-background sample so health_score/
    # biofouling actually get supervised somewhere.
    common_rows = [
        ("bg1.jpg", "Background", None, False),
        ("healthy1.jpg", "Healthy", 88.0, True),
        ("disease1.jpg", "Disease", 40.0, False),
        ("healthy2.jpg", "Healthy", 91.0, False),
    ]
    _write_split(dataset_root / "train", common_rows)
    _write_split(dataset_root / "validation", common_rows)
    _write_split(dataset_root / "test", common_rows)

    return Config(
        dataset_root=dataset_root,
        checkpoints_dir=tmp_path / "checkpoints",
        logs_dir=tmp_path / "logs",
        image_size=IMAGE_SIZE,
        # One batch covering all 4 samples: with a frozen backbone (BatchNorm
        # running stats untouched) and only 4 images total, splitting into
        # multiple tiny batches makes epoch-to-epoch loss too noisy to show a
        # clean trend over a short smoke run.
        batch_size=4,
        num_workers=0,
        device="cpu",
        frozen_epochs=15,
        finetune_epochs=0,  # keep the smoke test fast; phase 1 alone exercises the full pipeline
        early_stopping_patience=100,  # don't let early stopping cut the run short
    )


def test_train_smoke_end_to_end(tmp_path):
    schema = _make_schema()
    cfg = _build_synthetic_dataset(tmp_path, schema)

    train(cfg=cfg, schema=schema)

    # A checkpoint was written, carrying the schema it was trained with.
    checkpoint_path = cfg.checkpoints_dir / "best_model.pt"
    assert checkpoint_path.exists()

    model, loaded_schema = load_checkpoint(checkpoint_path, torch.device("cpu"))
    assert [m.key for m in loaded_schema.measurements] == ["condition", "health_score", "biofouling"]

    model.eval()
    with torch.no_grad():
        outputs = model(torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert outputs["condition"].shape == (1, 3)
    assert outputs["biofouling"].shape == (1, 2, IMAGE_SIZE, IMAGE_SIZE)

    # Train-mode loss trends down across the frozen-phase epochs on this tiny
    # fixed dataset (same fixed 4 images every epoch, so the model can
    # memorize). get_logger's FileHandler writes everything regardless of
    # caplog/propagation quirks, so read the log file directly. Deliberately
    # checking train_loss rather than val_loss: with only one 4-image batch
    # per epoch and a frozen backbone, BatchNorm's running stats (updated
    # every train-mode forward pass regardless of requires_grad) drift on
    # such a small, unrepresentative sample, making eval-mode (val) loss
    # noisy/non-monotonic independent of whether learning is happening.
    # train_loss is computed in train() mode and isn't subject to that
    # confound, so it's the direct signal that gradients are flowing.
    log_text = (cfg.logs_dir / "train.log").read_text()
    train_losses = [float(m.group(1)) for m in re.finditer(r"\[frozen \d+/\d+\] train_loss=([\d.]+)", log_text)]
    assert len(train_losses) == 15
    early_mean = sum(train_losses[:3]) / 3
    late_mean = sum(train_losses[-3:]) / 3
    assert late_mean < early_mean
