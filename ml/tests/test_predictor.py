"""Unit tests for the schema-driven Predictor: builds a real checkpoint
(exercising the actual save/load_checkpoint schema round-trip) but swaps in a
fixed-logits stub model afterwards, so predictions are deterministic and the
post-processing/gating logic (applies_when suppression, legacy flat-field
mapping, augmented recommendations, segmentation coverage) can be tested
precisely without depending on an actually-trained model."""
from __future__ import annotations

import io

import torch
import torch.nn as nn
from PIL import Image

from config import AppliesWhen, ClassDef, MeasurementDef, SegClassDef, Schema
from src.inference.predictor import Predictor
from src.models.efficientnet import build_model, save_checkpoint


def _schema() -> Schema:
    condition = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background",
        classes=[
            ClassDef(name="Background", explanation="No subject.", recommendation="Point at a specimen."),
            ClassDef(name="Healthy", explanation="Looks great.", recommendation="Keep monitoring."),
            ClassDef(name="Disease", explanation="Lesions present.", recommendation="Isolate and confirm."),
        ],
    )
    disease_subtype = MeasurementDef(
        key="disease_subtype",
        label="Disease subtype",
        type="classification",
        loss_weight=0.5,
        applies_when=AppliesWhen(key="condition", equals="Disease"),
        classes=[ClassDef(name="IceIce", note="Raise water movement."), ClassDef(name="Unknown", note="Consult a specialist.")],
    )
    health_score = MeasurementDef(
        key="health_score",
        label="Health score",
        type="regression",
        loss_weight=1.0,
        min=0.0,
        max=100.0,
        applies_when=AppliesWhen(key="condition", not_equals="Background"),
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
        measurements=[condition, disease_subtype, health_score, biofouling],
    )


class _FixedLogitsModel(nn.Module):
    """Forces every measurement's forward output regardless of input, so
    predictions in tests are deterministic."""

    def __init__(self, schema: Schema, class_choice: dict, regression_values: dict, seg_class_choice: dict):
        super().__init__()
        self.schema = schema
        self.class_choice = class_choice
        self.regression_values = regression_values
        self.seg_class_choice = seg_class_choice

    def forward(self, x: torch.Tensor) -> dict:
        batch = x.shape[0]
        out = {}
        for m in self.schema.measurements:
            if m.type == "classification":
                names = m.class_names()
                idx = names.index(self.class_choice[m.key])
                logits = torch.full((batch, len(names)), -10.0)
                logits[:, idx] = 10.0
                out[m.key] = logits
            elif m.type == "regression":
                out[m.key] = torch.full((batch,), float(self.regression_values.get(m.key, 0.0)))
            elif m.type == "segmentation":
                n = len(m.seg_classes)
                idx = self.seg_class_choice.get(m.key, 0)
                logits = torch.full((batch, n, x.shape[-2], x.shape[-1]), -10.0)
                logits[:, idx, :, :] = 10.0
                out[m.key] = logits
        return out


def _make_predictor(tmp_path, schema: Schema, class_choice: dict, regression_values: dict, seg_class_choice: dict) -> Predictor:
    model = build_model(schema, freeze_backbone=False, pretrained=False)
    path = tmp_path / "model.pt"
    save_checkpoint(model, schema, path)

    predictor = Predictor(path)
    predictor.model = _FixedLogitsModel(schema, class_choice, regression_values, seg_class_choice)
    return predictor


def _fake_image_bytes(size=32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color=(120, 180, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_healthy_prediction_masks_disease_subtype_but_keeps_health_score(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Healthy", "disease_subtype": "IceIce"},
        regression_values={"health_score": 88.0},
        seg_class_choice={"biofouling": 0},
    )

    result = predictor.predict(_fake_image_bytes())

    assert result.is_seaweed is True
    assert result.condition == "Healthy"
    assert result.health == "Healthy"
    assert result.health_score == 88.0
    assert result.disease_subtype is None  # masked: applies_when condition==Disease not satisfied
    assert result.explanation == "Looks great."
    assert result.recommendation == "Keep monitoring."  # no child note appended (subtype doesn't apply)

    assert result.measurements["condition"].value == "Healthy"
    assert result.measurements["disease_subtype"].value is None
    assert result.measurements["health_score"].value == 88.0


def test_disease_prediction_surfaces_subtype_and_augments_recommendation(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Disease", "disease_subtype": "IceIce"},
        regression_values={"health_score": 60.0},  # >= health_moderate_min (45), < health_healthy_min (75) -> "Moderate"
        seg_class_choice={"biofouling": 0},
    )

    result = predictor.predict(_fake_image_bytes())

    assert result.condition == "Disease"
    assert result.disease_subtype == "IceIce"
    assert result.health == "Moderate"
    assert result.recommendation == "Isolate and confirm. Raise water movement."

    assert result.measurements["disease_subtype"].value == "IceIce"
    assert result.measurements["disease_subtype"].confidence is not None


def test_disease_prediction_low_health_score_derives_low_level(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Disease", "disease_subtype": "Unknown"},
        regression_values={"health_score": 20.0},  # below health_moderate_min -> "Low"
        seg_class_choice={"biofouling": 0},
    )

    result = predictor.predict(_fake_image_bytes())
    assert result.health == "Low"


def test_level_is_purely_score_based_not_special_cased_by_condition_name(tmp_path):
    """The level derivation used to hardcode "Disease" as the only condition
    that could ever show "Moderate", and "Healthy" as the only one that could
    show "Healthy" — regardless of the actual regressed score. It's now
    purely a function of health_score against the two thresholds, so a
    "Disease" prediction with a high enough score shows "Healthy" too."""
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Disease", "disease_subtype": "Unknown"},
        regression_values={"health_score": 90.0},  # >= health_healthy_min (75)
        seg_class_choice={"biofouling": 0},
    )

    result = predictor.predict(_fake_image_bytes())
    assert result.health == "Healthy"


def test_background_prediction_masks_everything_and_reports_not_seaweed(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Background", "disease_subtype": "IceIce"},
        regression_values={"health_score": 99.0},
        seg_class_choice={"biofouling": 1},
    )

    result = predictor.predict(_fake_image_bytes())

    assert result.is_seaweed is False
    assert result.condition == "Background"
    assert result.health is None
    assert result.health_score is None  # masked: applies_when condition!=Background not satisfied
    assert result.disease_subtype is None
    assert result.explanation == "No subject."
    assert result.recommendation == "Point at a specimen."

    assert result.measurements["health_score"].value is None
    assert result.measurements["disease_subtype"].value is None


def test_segmentation_measurement_reports_coverage(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Healthy", "disease_subtype": "IceIce"},
        regression_values={"health_score": 80.0},
        seg_class_choice={"biofouling": 1},  # force every pixel to "algae"
    )

    result = predictor.predict(_fake_image_bytes())
    seg_result = result.measurements["biofouling"]
    assert seg_result.type == "segmentation"
    assert seg_result.coverage["algae"] == 100.0
    assert seg_result.coverage["background"] == 0.0
    assert seg_result.mask_png_base64 == ""  # ENABLE_SEGMENTATION_OVERLAY unset by default


def test_predictor_species_falls_back_to_unknown_when_schema_has_no_species_measurement(tmp_path):
    """Species is a real predicted classification now (see the "species"
    measurement in DEFAULT_SCHEMA), not a schema-wide constant. A schema
    without one (like this generic test schema, or a pre-restructure
    checkpoint) falls back to "Unknown species" rather than erroring."""
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Healthy", "disease_subtype": "IceIce"},
        regression_values={"health_score": 80.0},
        seg_class_choice={"biofouling": 0},
    )
    result = predictor.predict(_fake_image_bytes())
    assert result.species == "Unknown species"


def test_predictor_species_comes_from_its_own_predicted_classification(tmp_path):
    """When the schema does declare a "species" measurement, `result.species`
    is that measurement's actual predicted class — not a fixed value, and not
    tied to any notion of a single "active" species."""
    presence = MeasurementDef(
        key="presence",
        label="Presence",
        type="classification",
        loss_weight=1.0,
        background_class="No",
        classes=[ClassDef(name="Yes"), ClassDef(name="No")],
    )
    species = MeasurementDef(
        key="species",
        label="Species",
        type="classification",
        loss_weight=1.0,
        classes=[ClassDef(name="Kappaphycus_alvarezii"), ClassDef(name="Eucheuma_denticulatum")],
    )
    schema = Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[presence, species])

    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"presence": "Yes", "species": "Eucheuma_denticulatum"},
        regression_values={},
        seg_class_choice={},
    )
    result = predictor.predict(_fake_image_bytes())
    assert result.species == "Eucheuma_denticulatum"
