"""Unit tests for the dynamic (schema-driven) model heads and losses."""
from __future__ import annotations

import torch

from config import AppliesWhen, ClassDef, Config, MeasurementDef, RangeDef, SegClassDef, Schema, schema_to_dict, schema_from_dict
from src.losses import build_criterions, compute_losses
from src.models.efficientnet import build_model, load_checkpoint, save_checkpoint


def _schema_with_all_three_types() -> Schema:
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


def _fake_batch(schema: Schema, batch_size: int, image_size: int):
    images = torch.rand(batch_size, 3, image_size, image_size)
    alternating_mask = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(batch_size)])
    targets = {
        "condition_id": torch.randint(0, 3, (batch_size,)),
        "condition_mask": torch.ones(batch_size),
        "health_score": torch.rand(batch_size) * 100,
        "health_score_mask": alternating_mask,
        "biofouling_seg": torch.randint(0, 2, (batch_size, image_size, image_size)),
        "biofouling_seg_mask": alternating_mask,
    }
    return images, targets


def test_build_model_produces_one_output_per_measurement():
    schema = _schema_with_all_three_types()
    model = build_model(schema, freeze_backbone=True, pretrained=False)
    model.eval()

    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        outputs = model(images)

    assert set(outputs.keys()) == {"condition", "health_score", "biofouling"}
    assert outputs["condition"].shape == (2, 3)  # 3 classes
    assert outputs["health_score"].shape == (2,)
    assert torch.all(outputs["health_score"] >= 0) and torch.all(outputs["health_score"] <= 100)
    assert outputs["biofouling"].shape == (2, 2, 64, 64)  # (B, num_seg_classes, H, W) matches input resolution


def test_compute_losses_shapes_and_masking():
    schema = _schema_with_all_three_types()
    cfg = Config(device="cpu")
    model = build_model(schema, freeze_backbone=True, pretrained=False)
    criterions = build_criterions(schema, cfg)

    images, targets = _fake_batch(schema, batch_size=2, image_size=32)
    outputs = model(images)
    total, parts = compute_losses(outputs, targets, schema, criterions, cfg)

    assert total.dim() == 0  # scalar
    assert set(parts.keys()) == {"condition", "health_score", "biofouling", "total"}
    assert all(v >= 0 for v in parts.values())


def test_compute_losses_zero_mask_measurement_contributes_zero():
    schema = _schema_with_all_three_types()
    cfg = Config(device="cpu")
    model = build_model(schema, freeze_backbone=True, pretrained=False)
    criterions = build_criterions(schema, cfg)

    images = torch.rand(2, 3, 32, 32)
    outputs = model(images)
    targets = {
        "condition_id": torch.tensor([0, 1]),
        "condition_mask": torch.ones(2),
        "health_score": torch.zeros(2),
        "health_score_mask": torch.zeros(2),  # nobody has a health_score value yet
        "biofouling_seg": torch.zeros(2, 32, 32, dtype=torch.long),
        "biofouling_seg_mask": torch.zeros(2),  # nobody has a mask yet
    }
    _, parts = compute_losses(outputs, targets, schema, criterions, cfg)
    assert parts["health_score"] == 0.0
    assert parts["biofouling"] == 0.0


def test_training_step_reduces_loss_over_a_few_iterations():
    """A tight sanity check that gradients actually flow: repeatedly stepping
    on the *same* fixed batch should drive the loss down."""
    schema = _schema_with_all_three_types()
    cfg = Config(device="cpu")
    model = build_model(schema, freeze_backbone=False, pretrained=False)
    criterions = build_criterions(schema, cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    images, targets = _fake_batch(schema, batch_size=4, image_size=32)

    losses = []
    for _ in range(8):
        optimizer.zero_grad()
        outputs = model(images)
        total, _ = compute_losses(outputs, targets, schema, criterions, cfg)
        total.backward()
        optimizer.step()
        losses.append(total.item())

    assert losses[-1] < losses[0]


def test_checkpoint_round_trip_preserves_schema(tmp_path):
    schema = _schema_with_all_three_types()
    model = build_model(schema, freeze_backbone=False, pretrained=False)
    path = tmp_path / "model.pt"
    save_checkpoint(model, schema, path)

    loaded_model, loaded_schema = load_checkpoint(path, torch.device("cpu"))
    assert [m.key for m in loaded_schema.measurements] == ["condition", "health_score", "biofouling"]
    assert loaded_schema.find("condition").classes[0].name == "Background"

    images = torch.rand(1, 3, 48, 48)
    with torch.no_grad():
        outputs = loaded_model(images)
    assert outputs["condition"].shape == (1, 3)


def test_load_checkpoint_synthesizes_legacy_schema_for_pre_schema_checkpoints(tmp_path):
    """A checkpoint saved by the old save_checkpoint (condition_classes/
    subtype_classes/species, no "schema" key) must still load — even now that
    DEFAULT_SCHEMA has moved on to a different set of heads."""
    from config import legacy_schema_from_checkpoint
    from src.models.efficientnet import build_model as _build

    legacy_meta = {
        "condition_classes": ["Background", "Healthy", "Disease", "Decay", "Dried"],
        "subtype_classes": ["IceIce", "Epiphyte", "Bacterial", "Bleaching", "Unknown"],
        "species": {"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"},
    }
    # An old checkpoint's weights were saved with the old heads, i.e. those the
    # legacy synthesizer reproduces — build the state_dict from that schema.
    legacy_schema = legacy_schema_from_checkpoint(legacy_meta)
    model = _build(legacy_schema, freeze_backbone=False, pretrained=False)
    legacy_payload = {"model_state_dict": model.state_dict(), **legacy_meta}
    path = tmp_path / "legacy.pt"
    torch.save(legacy_payload, path)

    loaded_model, loaded_schema = load_checkpoint(path, torch.device("cpu"))
    assert loaded_schema.find("condition").class_names() == ["Background", "Healthy", "Disease", "Decay", "Dried"]
    assert loaded_schema.find("condition").background_class == "Background"
    # Preset copy recovered from DEFAULT_SCHEMA since the class names match.
    assert "No seaweed" in loaded_schema.find("condition").classes[0].explanation

    images = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        outputs = loaded_model(images)
    assert outputs["condition"].shape == (1, 5)


def test_schema_to_dict_from_dict_round_trip():
    schema = _schema_with_all_three_types()
    doc = schema_to_dict(schema)
    rebuilt = schema_from_dict(doc)
    assert [m.key for m in rebuilt.measurements] == [m.key for m in schema.measurements]
    assert rebuilt.find("biofouling").seg_classes[1].name == "algae"


def test_regression_ranges_round_trip_through_dict():
    severity = MeasurementDef(
        key="disease_severity",
        label="Disease severity",
        type="regression",
        loss_weight=0.5,
        min=0.0,
        max=100.0,
        ranges=[
            RangeDef(min=0.0, max=30.0, explanation="Mild.", recommendation="Monitor."),
            RangeDef(min=30.0, max=100.0, explanation="Severe.", recommendation="Isolate immediately."),
        ],
    )
    schema = Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[severity])

    doc = schema_to_dict(schema)
    rebuilt = schema_from_dict(doc)

    rebuilt_severity = rebuilt.find("disease_severity")
    assert len(rebuilt_severity.ranges) == 2
    assert rebuilt_severity.ranges[0].explanation == "Mild."
    assert rebuilt_severity.ranges[1].recommendation == "Isolate immediately."


def test_range_for_picks_first_matching_band_and_none_outside_every_range():
    severity = MeasurementDef(
        key="disease_severity",
        label="Disease severity",
        type="regression",
        loss_weight=0.5,
        min=0.0,
        max=100.0,
        ranges=[
            RangeDef(min=0.0, max=30.0, explanation="Mild."),
            RangeDef(min=30.0, max=60.0, explanation="Moderate."),
            RangeDef(min=60.0, max=100.0, explanation="Severe."),
        ],
    )
    assert severity.range_for(15.0).explanation == "Mild."
    assert severity.range_for(30.0).explanation == "Mild."  # boundary: first match wins
    assert severity.range_for(45.0).explanation == "Moderate."
    assert severity.range_for(100.0).explanation == "Severe."

    unbanded = MeasurementDef(key="gel_strength", label="Gel strength", type="regression", loss_weight=0.5, min=0.0, max=2000.0)
    assert unbanded.range_for(500.0) is None
