"""Unit tests for the schema-driven Predictor: builds a real checkpoint
(exercising the actual save/load_checkpoint schema round-trip) but swaps in a
fixed-logits stub model afterwards, so predictions are deterministic and the
post-processing/gating logic (applies_when suppression, augmented
recommendations, segmentation coverage, Grad-CAM target selection) can be
tested precisely without depending on an actually-trained model."""
from __future__ import annotations

import io

import torch
import torch.nn as nn
from PIL import Image

from config import AppliesWhen, ClassDef, MeasurementDef, RangeDef, SegClassDef, Schema
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
        applies_when=[AppliesWhen(key="condition", equals="Disease")],
        classes=[ClassDef(name="IceIce", note="Raise water movement."), ClassDef(name="Unknown", note="Consult a specialist.")],
    )
    health_score = MeasurementDef(
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

    condition_result = result.measurements["condition"]
    assert condition_result.value == "Healthy"
    assert condition_result.label == "Condition"
    assert condition_result.explanation == "Looks great."
    assert condition_result.recommendation == "Keep monitoring."  # no child note appended (subtype doesn't apply)

    assert result.measurements["disease_subtype"].value is None  # masked: applies_when condition==Disease not satisfied
    assert result.measurements["health_score"].value == 88.0


def test_disease_prediction_surfaces_subtype_and_augments_recommendation(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Disease", "disease_subtype": "IceIce"},
        regression_values={"health_score": 60.0},
        seg_class_choice={"biofouling": 0},
    )

    result = predictor.predict(_fake_image_bytes())

    assert result.measurements["condition"].value == "Disease"
    assert result.measurements["condition"].recommendation == "Isolate and confirm. Raise water movement."

    assert result.measurements["disease_subtype"].value == "IceIce"
    assert result.measurements["disease_subtype"].confidence is not None


def test_background_prediction_masks_everything(tmp_path):
    schema = _schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Background", "disease_subtype": "IceIce"},
        regression_values={"health_score": 99.0},
        seg_class_choice={"biofouling": 1},
    )

    result = predictor.predict(_fake_image_bytes())

    condition_result = result.measurements["condition"]
    assert condition_result.value == "Background"
    assert condition_result.explanation == "No subject."
    assert condition_result.recommendation == "Point at a specimen."

    assert result.measurements["health_score"].value is None  # masked: applies_when condition!=Background not satisfied
    assert result.measurements["disease_subtype"].value is None


def test_segmentation_measurement_reports_coverage_and_colors(tmp_path):
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
    assert seg_result.seg_colors == {"background": "#000000", "algae": "#22c55e"}
    assert seg_result.mask_png_base64 == ""  # ENABLE_SEGMENTATION_OVERLAY unset by default


def test_regression_measurement_surfaces_explanation_from_matching_range(tmp_path):
    """A regression measurement with per-band ranges should populate its
    MeasurementResult's explanation/recommendation from whichever range the
    predicted value falls into, the regression analogue of a classification's
    per-class ClassDef.explanation/recommendation."""
    condition = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background",
        classes=[ClassDef(name="Background"), ClassDef(name="Disease")],
    )
    severity = MeasurementDef(
        key="disease_severity",
        label="Disease severity",
        type="regression",
        loss_weight=0.5,
        unit="score",
        min=0.0,
        max=100.0,
        applies_when=[AppliesWhen(key="condition", not_equals="Background")],
        ranges=[
            RangeDef(min=0.0, max=30.0, explanation="Mild severity.", recommendation="Keep monitoring."),
            RangeDef(min=30.0, max=100.0, explanation="Severe.", recommendation="Isolate immediately."),
        ],
    )
    schema = Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[condition, severity])

    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Disease"},
        regression_values={"disease_severity": 72.0},
        seg_class_choice={},
    )
    result = predictor.predict(_fake_image_bytes())

    severity_result = result.measurements["disease_severity"]
    assert severity_result.value == 72.0
    assert severity_result.explanation == "Severe."
    assert severity_result.recommendation == "Isolate immediately."
    assert severity_result.unit == "score"
    assert severity_result.min == 0.0
    assert severity_result.max == 100.0


def test_regression_measurement_masked_by_applies_when_has_no_range_explanation(tmp_path):
    condition = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background",
        classes=[ClassDef(name="Background"), ClassDef(name="Disease")],
    )
    severity = MeasurementDef(
        key="disease_severity",
        label="Disease severity",
        type="regression",
        loss_weight=0.5,
        min=0.0,
        max=100.0,
        applies_when=[AppliesWhen(key="condition", not_equals="Background")],
        ranges=[RangeDef(min=0.0, max=100.0, explanation="Severe.")],
    )
    schema = Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[condition, severity])

    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"condition": "Background"},
        regression_values={"disease_severity": 72.0},
        seg_class_choice={},
    )
    result = predictor.predict(_fake_image_bytes())

    severity_result = result.measurements["disease_severity"]
    assert severity_result.value is None
    assert severity_result.explanation is None


def test_predictor_species_comes_from_its_own_predicted_classification(tmp_path):
    """Species is a real predicted classification, same as any other
    measurement — not a fixed value, and not tied to any notion of a single
    "active" species."""
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
    assert result.measurements["species"].value == "Eucheuma_denticulatum"


def _current_style_schema() -> Schema:
    """Mirrors DEFAULT_SCHEMA's shape post "drop background_class
    requirement" migration: seaweed_presence is a plain Yes/No classification
    with no background_class declared, so schema.primary_classification()
    returns None for it — there's no "primary classification" concept in
    this schema at all."""
    seaweed_presence = MeasurementDef(
        key="seaweed_presence",
        label="Seaweed presence",
        type="classification",
        loss_weight=1.0,
        classes=[ClassDef(name="Yes"), ClassDef(name="No")],
    )
    health_status = MeasurementDef(
        key="health_status",
        label="Health status",
        type="classification",
        loss_weight=1.0,
        applies_when=[AppliesWhen(key="seaweed_presence", equals="Yes")],
        classes=[ClassDef(name="Healthy"), ClassDef(name="Moderate"), ClassDef(name="Low")],
    )
    return Schema(health_moderate_min=45.0, health_healthy_min=75.0, measurements=[seaweed_presence, health_status])


def test_confidence_reflects_softmax_regardless_of_background_class(tmp_path):
    """Regression guard: each classification measurement's confidence comes
    straight from its own softmax output — it never depends on whether some
    OTHER measurement in the schema declares a background_class. This schema
    mirrors DEFAULT_SCHEMA's current shape (no background_class anywhere),
    where a flat "primary classification" concept doesn't exist at all."""
    schema = _current_style_schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"seaweed_presence": "Yes", "health_status": "Healthy"},
        regression_values={},
        seg_class_choice={},
    )
    result = predictor.predict(_fake_image_bytes())

    assert result.measurements["seaweed_presence"].confidence > 0.9
    assert result.measurements["health_status"].confidence > 0.9


def test_measurement_gated_off_reports_null_value_not_omitted(tmp_path):
    """seaweed_presence == "No" gates health_status off via applies_when —
    it must still appear in the report (value=None, confidence=None), not be
    silently dropped, so a client can tell "not applicable" from "not
    defined in this schema"."""
    schema = _current_style_schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"seaweed_presence": "No", "health_status": "Healthy"},
        regression_values={},
        seg_class_choice={},
    )
    result = predictor.predict(_fake_image_bytes())

    assert result.measurements["seaweed_presence"].value == "No"
    assert result.measurements["health_status"].value is None
    assert result.measurements["health_status"].confidence is None


def test_gradcam_only_computed_for_first_classification_measurement(tmp_path):
    """Grad-CAM is expensive (a backward pass per head), so only ever run for
    one measurement per request — the first classification measurement in
    schema order, deterministically, with no "primary"/background_class
    concept needed. Every other classification's gradcam_png_base64 is None
    (never attempted), distinct from "" (attempted but ENABLE_GRADCAM off)."""
    schema = _current_style_schema()
    predictor = _make_predictor(
        tmp_path, schema,
        class_choice={"seaweed_presence": "Yes", "health_status": "Healthy"},
        regression_values={},
        seg_class_choice={},
    )
    result = predictor.predict(_fake_image_bytes())

    assert result.measurements["seaweed_presence"].gradcam_png_base64 == ""  # ENABLE_GRADCAM unset by default
    assert result.measurements["health_status"].gradcam_png_base64 is None
