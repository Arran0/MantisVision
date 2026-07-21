"""Central configuration for the seaweed multi-head classifier.

Single source of truth for paths and hyperparameters, plus the fallback
default for the measurement schema below — every script (data loading,
training, evaluation, Grad-CAM, inference API) agrees on the same values.

The model is schema-driven (see src/models/efficientnet.py's build_model):
it grows one head per measurement the active Schema defines — classification,
regression, or segmentation — rather than a fixed set. There are no class
folders anywhere in this pipeline; per-image ground truth is a column/CSV-
style manifest (see src/data/annotations.py). Species is one such
measurement too (see DEFAULT_SCHEMA below) — a normal classification, not a
schema-level "active species" concept, so the dataset directory is no longer
species-scoped (see Config.dataset_dir).

health_status (Healthy/Moderate/Low) is a labeled classification the admin
assigns per image, not a bucket derived from a score. health_moderate_min/
health_healthy_min are kept only so an older checkpoint that still regresses
a health_score can be bucketed into the same display level (see
src/inference/predictor.py's _derive_level).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# --- Default taxonomy -------------------------------------------------------
# These are only the *fallback* values baked into DEFAULT_SCHEMA below (and
# used to synthesize a schema for a checkpoint saved before schemas existed —
# see legacy_schema_from_checkpoint). The real, admin-editable source of
# truth is the measurement schema itself.
DEFAULT_CONDITION_CLASSES = ["Background", "Healthy", "Disease", "Decay", "Dried"]
DEFAULT_DISEASE_SUBTYPES = ["IceIce", "Epiphyte", "Bacterial", "Bleaching", "Unknown"]

# Display-level thresholds (see predictor._derive_level): health_score at or
# above HEALTHY_MIN -> "Healthy"; at or above MODERATE_MIN (but below
# healthy) -> "Moderate"; otherwise "Low".
DEFAULT_HEALTH_MODERATE_MIN: float = 45.0
DEFAULT_HEALTH_HEALTHY_MIN: float = 75.0

# --- Measurement schema ----------------------------------------------------
# The taxonomy above (species/conditions/subtypes/anchors) is the *fallback*.
# The real, admin-editable source of truth is a "measurement schema": a JSON
# document describing every per-image measurement the model predicts
# (classification, regression, or segmentation), stored in Supabase
# (table measurement_schema) and exported by the retrain workflow to
# ml/metadata/schema.json (see scripts/export_schema.py). This mirrors
# apps/web/src/lib/schema.ts field-for-field — keep the two in sync.
#
# Loading is schema.json-if-present else DEFAULT_SCHEMA, so a repo checkout
# with no exported schema.json behaves identically to before this existed.


@dataclass
class ClassDef:
    name: str
    explanation: str | None = None
    recommendation: str | None = None
    note: str | None = None


@dataclass
class SegClassDef:
    name: str
    color: str = "#888888"


@dataclass
class RangeDef:
    """A band of a regression measurement's predicted value, with its own
    preset explanation/recommendation copy. Mirrors RangeDef in
    apps/web/src/lib/schema.ts — keep both in sync."""

    min: float
    max: float
    explanation: str | None = None
    recommendation: str | None = None


@dataclass
class AppliesWhen:
    key: str
    equals: str | None = None
    not_equals: str | None = None


@dataclass
class MeasurementDef:
    key: str
    label: str
    type: str  # "classification" | "regression" | "segmentation"
    loss_weight: float = 1.0
    # A list of AND-combined conditions (every one must hold for the
    # measurement to apply) — empty/absent means "always applies".
    applies_when: list[AppliesWhen] = field(default_factory=list)
    background_class: str | None = None
    classes: list[ClassDef] = field(default_factory=list)
    unit: str | None = None
    min: float = 0.0
    max: float = 100.0
    ranges: list[RangeDef] = field(default_factory=list)
    seg_classes: list[SegClassDef] = field(default_factory=list)

    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]

    def range_for(self, value: float) -> RangeDef | None:
        """The first range `value` falls into, if any. Mirrors rangeForValue
        in apps/web/src/lib/schema.ts — keep both in sync."""
        return next((r for r in self.ranges if r.min <= value <= r.max), None)


@dataclass
class Schema:
    health_moderate_min: float
    health_healthy_min: float
    measurements: list[MeasurementDef]

    def find(self, key: str) -> MeasurementDef | None:
        return next((m for m in self.measurements if m.key == key), None)

    def primary_classification(self) -> MeasurementDef | None:
        """The measurement that flags "no subject in frame" (analogous to the
        old Background condition) — the first classification measurement that
        declares a background_class, if any."""
        return next((m for m in self.measurements if m.type == "classification" and m.background_class), None)

    def applies(self, measurement: MeasurementDef, values: dict) -> bool:
        """Whether `measurement` is active given the current values of other
        measurements (keyed by measurement key -> class name) — every
        condition in applies_when must hold (AND). Mirrors measurementApplies
        in apps/web/src/lib/schema.ts exactly — keep both in sync."""
        for cond in measurement.applies_when:
            parent_value = values.get(cond.key)
            if parent_value is None:
                return False
            if cond.equals is not None and parent_value != cond.equals:
                return False
            if cond.not_equals is not None and parent_value == cond.not_equals:
                return False
        return True


def schema_from_dict(doc: dict) -> Schema:
    def _class(d: dict) -> ClassDef:
        return ClassDef(name=d["name"], explanation=d.get("explanation"), recommendation=d.get("recommendation"), note=d.get("note"))

    def _seg_class(d: dict) -> SegClassDef:
        return SegClassDef(name=d["name"], color=d.get("color", "#888888"))

    def _range(d: dict) -> RangeDef:
        return RangeDef(
            min=float(d["min"]), max=float(d["max"]), explanation=d.get("explanation"), recommendation=d.get("recommendation")
        )

    def _one_applies_when(d: dict) -> AppliesWhen:
        return AppliesWhen(key=d["key"], equals=d.get("equals"), not_equals=d.get("not_equals"))

    def _applies_when(raw: object) -> list[AppliesWhen]:
        # applies_when used to be a single condition object; it's now a list
        # of AND-combined conditions. Accept both shapes so a schema doc
        # saved before this change still loads.
        if not raw:
            return []
        if isinstance(raw, list):
            return [_one_applies_when(c) for c in raw]
        if isinstance(raw, dict):
            return [_one_applies_when(raw)]
        return []

    def _measurement(d: dict) -> MeasurementDef:
        if not isinstance(d, dict):
            raise ValueError(f"Each entry in schema 'measurements' must be an object, got {type(d).__name__}.")
        classes_raw = d.get("classes", [])
        ranges_raw = d.get("ranges", [])
        seg_classes_raw = d.get("seg_classes", [])
        return MeasurementDef(
            key=d["key"],
            label=d.get("label", d["key"]),
            type=d["type"],
            loss_weight=float(d.get("loss_weight", 1.0)),
            applies_when=_applies_when(d.get("applies_when")),
            background_class=d.get("background_class"),
            classes=[_class(c) for c in (classes_raw if isinstance(classes_raw, list) else [])],
            unit=d.get("unit"),
            min=float(d.get("min", 0.0)),
            max=float(d.get("max", 100.0)),
            ranges=[_range(r) for r in (ranges_raw if isinstance(ranges_raw, list) else [])],
            seg_classes=[_seg_class(c) for c in (seg_classes_raw if isinstance(seg_classes_raw, list) else [])],
        )

    measurements_raw = doc.get("measurements")
    if not isinstance(measurements_raw, list):
        raise ValueError(
            f"Schema document's 'measurements' field must be a list, got {type(measurements_raw).__name__}."
        )

    return Schema(
        health_moderate_min=float(doc.get("health_moderate_min", DEFAULT_HEALTH_MODERATE_MIN)),
        health_healthy_min=float(doc.get("health_healthy_min", DEFAULT_HEALTH_HEALTHY_MIN)),
        measurements=[_measurement(m) for m in measurements_raw],
    )


def schema_to_dict(schema: Schema) -> dict:
    def _class(c: ClassDef) -> dict:
        d = {"name": c.name}
        if c.explanation is not None:
            d["explanation"] = c.explanation
        if c.recommendation is not None:
            d["recommendation"] = c.recommendation
        if c.note is not None:
            d["note"] = c.note
        return d

    def _measurement(m: MeasurementDef) -> dict:
        d = {"key": m.key, "label": m.label, "type": m.type, "loss_weight": m.loss_weight}
        if m.applies_when:
            aws = []
            for cond in m.applies_when:
                aw = {"key": cond.key}
                if cond.equals is not None:
                    aw["equals"] = cond.equals
                if cond.not_equals is not None:
                    aw["not_equals"] = cond.not_equals
                aws.append(aw)
            d["applies_when"] = aws
        if m.background_class is not None:
            d["background_class"] = m.background_class
        if m.type == "classification":
            d["classes"] = [_class(c) for c in m.classes]
        elif m.type == "regression":
            d["unit"] = m.unit
            d["min"] = m.min
            d["max"] = m.max
            if m.ranges:
                d["ranges"] = [
                    {
                        "min": r.min,
                        "max": r.max,
                        **({"explanation": r.explanation} if r.explanation is not None else {}),
                        **({"recommendation": r.recommendation} if r.recommendation is not None else {}),
                    }
                    for r in m.ranges
                ]
        elif m.type == "segmentation":
            d["seg_classes"] = [{"name": c.name, "color": c.color} for c in m.seg_classes]
        return d

    return {
        "health_moderate_min": schema.health_moderate_min,
        "health_healthy_min": schema.health_healthy_min,
        "measurements": [_measurement(m) for m in schema.measurements],
    }


# The default schema — kept in sync with apps/web/src/lib/schema.ts's
# DEFAULT_SCHEMA and the SQL seed in
# supabase/migrations/20260716000010_drop_background_class_requirement.sql. Just a
# starting point: every measurement here (including seaweed_presence) is
# freely editable/removable from the admin Structure editor.
_WHEN_SEAWEED_PRESENT = [{"key": "seaweed_presence", "equals": "Yes"}]


def _lab_regression(key: str, label: str, unit: str, max_value: float, applies_when: list | None = None) -> dict:
    return {
        "key": key,
        "label": label,
        "type": "regression",
        "loss_weight": 0.5,
        "unit": unit,
        "min": 0.0,
        "max": max_value,
        "applies_when": applies_when if applies_when is not None else _WHEN_SEAWEED_PRESENT,
    }


DEFAULT_SCHEMA: Schema = schema_from_dict(
    {
        "health_moderate_min": DEFAULT_HEALTH_MODERATE_MIN,
        "health_healthy_min": DEFAULT_HEALTH_HEALTHY_MIN,
        "measurements": [
            {
                "key": "seaweed_presence",
                "label": "Seaweed presence",
                "type": "classification",
                "loss_weight": 1.0,
                "classes": [
                    {
                        "name": "Yes",
                        "explanation": "A seaweed specimen was detected in this image.",
                        "recommendation": "Continue with the assessment below.",
                    },
                    {
                        "name": "No",
                        "explanation": "No seaweed specimen was detected in this image.",
                        "recommendation": "Point the camera at a seaweed specimen, filling the frame, and try again.",
                    },
                ],
            },
            # Species is just another classification, same as any other
            # measurement — but it has no preset class here. An admin adds it
            # themselves from the Structure editor once they know which
            # species they're tracking; predictor.py already falls back to
            # "Unknown species" for a schema with no "species" measurement.
            {
                "key": "health_status",
                "label": "Health status",
                "type": "classification",
                "loss_weight": 1.0,
                "applies_when": _WHEN_SEAWEED_PRESENT,
                "classes": [
                    {
                        "name": "Healthy",
                        "explanation": "Vivid, even coloration with intact branching and no whitening, lesions, or breakage.",
                        "recommendation": "Continue routine monitoring. No action needed.",
                    },
                    {
                        "name": "Moderate",
                        "explanation": "Some discoloration or minor structural loss, but the specimen is largely intact.",
                        "recommendation": "Increase monitoring frequency and check water quality (temperature, salinity).",
                    },
                    {
                        "name": "Low",
                        "explanation": "Extensive discoloration, tissue loss, or structural breakdown across the specimen.",
                        "recommendation": "Remove affected fragments to prevent spread and investigate the cause promptly.",
                    },
                ],
            },
            {
                "key": "disease",
                "label": "Disease",
                "type": "classification",
                "loss_weight": 0.5,
                "applies_when": _WHEN_SEAWEED_PRESENT,
                "classes": [
                    {"name": "NoDisease", "explanation": "No disease detected."},
                    {"name": "IceIce", "note": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity."},
                    {"name": "Epiphyte", "note": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow."},
                    {"name": "Bacterial", "note": "Possible bacterial infection: isolate and consult a specialist before any treatment."},
                    {"name": "Bleaching", "note": "Bleaching suspected: check for temperature/light stress and relocate if possible."},
                ],
            },
            {
                "key": "disease_severity",
                "label": "Disease severity",
                "type": "regression",
                "loss_weight": 0.5,
                "unit": "score",
                "min": 0.0,
                "max": 100.0,
                "applies_when": [{"key": "disease", "not_equals": "NoDisease"}],
            },
            _lab_regression("dried", "Dried", "%", 100.0),
            _lab_regression("decayed", "Decayed", "%", 100.0),
            {
                "key": "colour",
                "label": "Colour",
                "type": "classification",
                "loss_weight": 0.5,
                "applies_when": _WHEN_SEAWEED_PRESENT,
                "classes": [
                    {"name": "Green"}, {"name": "Red"}, {"name": "Brown"}, {"name": "Yellow"},
                    {"name": "Orange"}, {"name": "White"}, {"name": "Black"},
                ],
            },
            _lab_regression("carrageenan_yield", "Carrageenan Yield", "%", 100.0),
            _lab_regression("gel_strength", "Gel Strength", "g/cm²", 2000.0),
            _lab_regression("viscosity", "Viscosity", "cP", 1000.0),
            _lab_regression("daily_growth_rate", "Daily Growth Rate", "%/day", 100.0),
            _lab_regression("mineral_ca", "Mineral Content — Ca", "mg/kg", 100000.0),
            _lab_regression("mineral_mg", "Mineral Content — Mg", "mg/kg", 100000.0),
            _lab_regression("mineral_k", "Mineral Content — K", "mg/kg", 100000.0),
            _lab_regression("mineral_na", "Mineral Content — Na", "mg/kg", 100000.0),
            _lab_regression("caw", "Clean Anhydrous Weed (CAW)", "%", 100.0),
            _lab_regression("impurities", "Impurities", "%", 100.0),
            _lab_regression("sulfate_content", "Sulfate Content", "%", 100.0),
            _lab_regression("acid_insoluble_ash", "Acid-Insoluble Ash", "%", 100.0),
            _lab_regression("ash_content", "Ash Content", "%", 100.0),
        ],
    }
)


# Preset copy for the pre-schema condition/subtype taxonomy. Kept
# self-contained (rather than recovered from DEFAULT_SCHEMA) so that old
# checkpoints reproduce their original explanations even now that
# DEFAULT_SCHEMA no longer contains a "condition"/"disease_subtype" head.
_LEGACY_CONDITION_COPY: dict[str, tuple[str, str]] = {
    "Background": (
        "No seaweed specimen was detected in this image.",
        "Point the camera at a seaweed specimen, filling the frame, and try again.",
    ),
    "Healthy": (
        "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
        "Continue routine monitoring. No action needed.",
    ),
    "Disease": (
        "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
        "Isolate affected line segments and confirm the pathogen before treating.",
    ),
    "Decay": (
        "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
        "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
    ),
    "Dried": (
        "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
        "Remove and dispose of dried-out material. Inspect the surrounding line for early damage.",
    ),
}
_LEGACY_SUBTYPE_NOTES: dict[str, str] = {
    "IceIce": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity.",
    "Epiphyte": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow.",
    "Bacterial": "Possible bacterial infection: isolate and consult a specialist before any treatment.",
    "Bleaching": "Bleaching suspected: check for temperature/light stress and relocate if possible.",
    "Unknown": "Subtype unclear: photograph affected areas closely and consult a specialist.",
}


def legacy_schema_from_checkpoint(payload: dict) -> Schema:
    """Synthesizes a Schema for a checkpoint saved before schemas existed
    (payload has condition_classes/subtype_classes/species but no schema key),
    so old checkpoints keep loading. Preset copy comes from the self-contained
    legacy tables above for any class name that matches; unmatched classes
    (e.g. an admin-renamed taxonomy) get no preset copy. The payload's
    `species` dict (if any) isn't reconstructed into a measurement: those old
    checkpoints predate species being a predicted head at all (it was a fixed,
    untrained constant), so their state_dict has no matching "species" head to
    load weights into."""
    condition_classes: list[str] = payload.get("condition_classes") or list(DEFAULT_CONDITION_CLASSES)
    subtype_classes: list[str] = payload.get("subtype_classes") or list(DEFAULT_DISEASE_SUBTYPES)

    condition_measurement = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background" if "Background" in condition_classes else None,
        classes=[
            ClassDef(
                name=name,
                explanation=_LEGACY_CONDITION_COPY[name][0] if name in _LEGACY_CONDITION_COPY else None,
                recommendation=_LEGACY_CONDITION_COPY[name][1] if name in _LEGACY_CONDITION_COPY else None,
            )
            for name in condition_classes
        ],
    )
    subtype_measurement = MeasurementDef(
        key="disease_subtype",
        label="Disease subtype",
        type="classification",
        loss_weight=0.5,
        applies_when=[AppliesWhen(key="condition", equals="Disease")] if "Disease" in condition_classes else [],
        classes=[
            ClassDef(name=name, note=_LEGACY_SUBTYPE_NOTES.get(name))
            for name in subtype_classes
        ],
    )
    regression_measurements = [
        MeasurementDef(
            key=key,
            label=label,
            type="regression",
            loss_weight=weight,
            unit=unit,
            min=0.0,
            max=100.0,
            applies_when=[AppliesWhen(key="condition", not_equals="Background")],
        )
        for key, label, unit, weight in [
            ("health_score", "Health score", "score", 1.0),
            ("dried_extent", "Dried extent", "pct", 0.5),
            ("decayed_extent", "Decayed extent", "pct", 0.5),
        ]
    ]

    return Schema(
        health_moderate_min=DEFAULT_HEALTH_MODERATE_MIN,
        health_healthy_min=DEFAULT_HEALTH_HEALTHY_MIN,
        measurements=[condition_measurement, subtype_measurement, *regression_measurements],
    )


# Env var override for tooling/tests; production loads from the path the
# retrain workflow exports to (see scripts/export_schema.py).
SCHEMA_PATH_ENV = "MANTIS_SCHEMA_PATH"


def load_schema(path: Path | None = None) -> Schema:
    """Loads the measurement schema from `path` (or MANTIS_SCHEMA_PATH, or
    ml/metadata/schema.json) if it exists, else returns DEFAULT_SCHEMA. A
    fresh checkout with no exported schema.json behaves identically to
    before schemas existed."""
    if path is None:
        env_path = os.environ.get(SCHEMA_PATH_ENV)
        path = Path(env_path) if env_path else (ML_ROOT / "metadata" / "schema.json")
    if path.exists():
        with open(path) as f:
            return schema_from_dict(json.load(f))
    return DEFAULT_SCHEMA


# Loaded once at import time; pass this (or a test-specific Schema instance)
# explicitly to dataset/model/loss code rather than re-reading the global —
# mirrors how `config` itself is threaded through explicitly everywhere.
SCHEMA: Schema = load_schema()


@dataclass
class Config:
    seed: int = 42

    # dataset/{train,validation,test}/images|masks/*+annotations.jsonl — not
    # species-scoped; species is a normal per-image classification column
    # (see the "species" measurement in DEFAULT_SCHEMA), so one dataset holds
    # every species you collect.
    dataset_root: Path = ML_ROOT / "dataset"

    checkpoints_dir: Path = ML_ROOT / "checkpoints"
    logs_dir: Path = ML_ROOT / "logs"
    reports_dir: Path = ML_ROOT / "reports"
    metadata_dir: Path = ML_ROOT / "metadata"

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

    # ImageNet stats, required because EfficientNet-B0 is pretrained on ImageNet
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Training schedule: freeze backbone for `frozen_epochs`, then unfreeze and
    # fine-tune the whole network for `finetune_epochs`.
    frozen_epochs: int = 10
    finetune_epochs: int = 20
    frozen_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4

    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 1e-4

    # Label smoothing on the primary (background-carrying) classification
    # head — a cheap, first-line defense against label noise. See train.py
    # for the (documented) heavier option. Each measurement's own weight in
    # the multi-task loss is its loss_weight field in the schema, not here.
    condition_label_smoothing: float = 0.1

    device: str = "cuda"  # falls back to cpu automatically, see utils.seed

    @property
    def dataset_dir(self) -> Path:
        return self.dataset_root

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.dataset_dir / "validation"

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / "test"


config = Config()
