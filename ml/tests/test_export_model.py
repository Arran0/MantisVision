"""Verifies scripts/export_model.py's ONNX export is schema-driven: the
exported graph's output names/shapes must match whatever measurements the
checkpoint's schema defines (not a hardcoded 5-head list), including a
segmentation measurement's 4D per-pixel output."""
from __future__ import annotations

import copy
import json

import numpy as np
import onnxruntime as ort
import torch

from config import DEFAULT_SCHEMA, MeasurementDef, SegClassDef, config
from src.models.efficientnet import build_model, save_checkpoint


def _export(tmp_path, schema, monkeypatch):
    model = build_model(schema, freeze_backbone=False, pretrained=False)
    save_checkpoint(model, schema, tmp_path / "best_model.pt")

    # export_model.py reads paths off the module-level `config` singleton
    # directly (matching every other script in this repo), so redirect it
    # for the duration of the test.
    monkeypatch.setattr(config, "checkpoints_dir", tmp_path)
    import scripts.export_model

    scripts.export_model.main()

    return tmp_path / "seaweed_multihead.onnx", tmp_path / "class_names.json"


def test_export_default_schema(tmp_path, monkeypatch):
    onnx_path, meta_path = _export(tmp_path, DEFAULT_SCHEMA, monkeypatch)
    assert onnx_path.exists()

    meta = json.loads(meta_path.read_text())
    assert [m["key"] for m in meta["measurements"]] == [m.key for m in DEFAULT_SCHEMA.measurements]

    sess = ort.InferenceSession(str(onnx_path))
    output_names = [o.name for o in sess.get_outputs()]
    assert output_names == [m.key for m in DEFAULT_SCHEMA.measurements]

    result = sess.run(None, {"image": np.random.randn(1, 3, 224, 224).astype("float32")})
    assert result[output_names.index("seaweed_presence")].shape == (1, 2)  # Yes/No
    assert result[output_names.index("health_status")].shape == (1, 3)  # Healthy/Moderate/Low
    assert result[output_names.index("dried")].shape == (1,)  # regression


def test_export_schema_with_segmentation_measurement(tmp_path, monkeypatch):
    schema = copy.deepcopy(DEFAULT_SCHEMA)
    schema.measurements.append(
        MeasurementDef(
            key="biofouling",
            label="Biofouling",
            type="segmentation",
            loss_weight=1.0,
            seg_classes=[SegClassDef(name="background", color="#000000"), SegClassDef(name="algae", color="#22c55e")],
        )
    )
    onnx_path, _ = _export(tmp_path, schema, monkeypatch)

    sess = ort.InferenceSession(str(onnx_path))
    output_names = [o.name for o in sess.get_outputs()]
    assert "biofouling" in output_names

    result = sess.run(None, {"image": np.random.randn(1, 3, 224, 224).astype("float32")})
    biofouling_out = result[output_names.index("biofouling")]
    assert biofouling_out.shape == (1, 2, 224, 224)  # (batch, num_seg_classes, H, W)
