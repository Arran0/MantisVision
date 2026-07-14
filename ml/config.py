"""Central configuration for the seaweed multi-head classifier.

Single source of truth for paths and hyperparameters, plus the fallback
default for the measurement schema below — every script (data loading,
training, evaluation, Grad-CAM, inference API) agrees on the same values.

The model is schema-driven (see src/models/efficientnet.py's build_model):
it grows one head per measurement the active Schema defines — classification,
regression, or segmentation — rather than a fixed set. There are no class
folders anywhere in this pipeline; per-image ground truth is a column/CSV-
style manifest (see src/data/annotations.py).

Discrete health *level* (Healthy/Moderate/Low) is NOT a class — it's derived
at inference purely from the regressed health_score against the schema's two
thresholds (see src/inference/predictor.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# --- Default species and taxonomy ------------------------------------------
# These are only the *fallback* values baked into DEFAULT_SCHEMA below (and
# used to synthesize a schema for a checkpoint saved before schemas existed —
# see legacy_schema_from_checkpoint). The real, admin-editable source of
# truth is the measurement schema itself.
SPECIES = {"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"}
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
class SpeciesDef:
    name: str
    slug: str


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
    applies_when: AppliesWhen | None = None
    background_class: str | None = None
    classes: list[ClassDef] = field(default_factory=list)
    unit: str | None = None
    min: float = 0.0
    max: float = 100.0
    seg_classes: list[SegClassDef] = field(default_factory=list)

    def class_names(self) -> list[str]:
        return [c.name for c in self.classes]


@dataclass
class Schema:
    species: list[SpeciesDef]
    active_species_slug: str
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
        measurements (keyed by measurement key -> class name). Mirrors
        measurementApplies in apps/web/src/lib/schema.ts exactly — keep both
        in sync."""
        cond = measurement.applies_when
        if cond is None:
            return True
        parent_value = values.get(cond.key)
        if parent_value is None:
            return False
        if cond.equals is not None:
            return parent_value == cond.equals
        if cond.not_equals is not None:
            return parent_value != cond.not_equals
        return True


def schema_from_dict(doc: dict) -> Schema:
    def _class(d: dict) -> ClassDef:
        return ClassDef(name=d["name"], explanation=d.get("explanation"), recommendation=d.get("recommendation"), note=d.get("note"))

    def _seg_class(d: dict) -> SegClassDef:
        return SegClassDef(name=d["name"], color=d.get("color", "#888888"))

    def _applies_when(d: dict | None) -> AppliesWhen | None:
        if not d:
            return None
        return AppliesWhen(key=d["key"], equals=d.get("equals"), not_equals=d.get("not_equals"))

    def _measurement(d: dict) -> MeasurementDef:
        return MeasurementDef(
            key=d["key"],
            label=d.get("label", d["key"]),
            type=d["type"],
            loss_weight=float(d.get("loss_weight", 1.0)),
            applies_when=_applies_when(d.get("applies_when")),
            background_class=d.get("background_class"),
            classes=[_class(c) for c in d.get("classes", [])],
            unit=d.get("unit"),
            min=float(d.get("min", 0.0)),
            max=float(d.get("max", 100.0)),
            seg_classes=[_seg_class(c) for c in d.get("seg_classes", [])],
        )

    return Schema(
        species=[SpeciesDef(name=s["name"], slug=s["slug"]) for s in doc["species"]],
        active_species_slug=doc["active_species_slug"],
        health_moderate_min=float(doc.get("health_moderate_min", DEFAULT_HEALTH_MODERATE_MIN)),
        health_healthy_min=float(doc.get("health_healthy_min", DEFAULT_HEALTH_HEALTHY_MIN)),
        measurements=[_measurement(m) for m in doc["measurements"]],
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
        if m.applies_when is not None:
            aw = {"key": m.applies_when.key}
            if m.applies_when.equals is not None:
                aw["equals"] = m.applies_when.equals
            if m.applies_when.not_equals is not None:
                aw["not_equals"] = m.applies_when.not_equals
            d["applies_when"] = aw
        if m.background_class is not None:
            d["background_class"] = m.background_class
        if m.type == "classification":
            d["classes"] = [_class(c) for c in m.classes]
        elif m.type == "regression":
            d["unit"] = m.unit
            d["min"] = m.min
            d["max"] = m.max
        elif m.type == "segmentation":
            d["seg_classes"] = [{"name": c.name, "color": c.color} for c in m.seg_classes]
        return d

    return {
        "species": [{"name": s.name, "slug": s.slug} for s in schema.species],
        "active_species_slug": schema.active_species_slug,
        "health_moderate_min": schema.health_moderate_min,
        "health_healthy_min": schema.health_healthy_min,
        "measurements": [_measurement(m) for m in schema.measurements],
    }


# Reproduces today's fixed taxonomy as a Schema — kept in sync with
# apps/web/src/lib/schema.ts's DEFAULT_SCHEMA and the SQL seed in
# supabase/migrations/20260714000005_measurement_schema.sql.
DEFAULT_SCHEMA: Schema = schema_from_dict(
    {
        "species": [{"name": SPECIES["name"], "slug": SPECIES["slug"]}],
        "active_species_slug": SPECIES["slug"],
        "health_moderate_min": DEFAULT_HEALTH_MODERATE_MIN,
        "health_healthy_min": DEFAULT_HEALTH_HEALTHY_MIN,
        "measurements": [
            {
                "key": "condition",
                "label": "Condition",
                "type": "classification",
                "loss_weight": 1.0,
                "background_class": "Background",
                "classes": [
                    {
                        "name": "Background",
                        "explanation": "No seaweed specimen was detected in this image.",
                        "recommendation": "Point the camera at a seaweed specimen, filling the frame, and try again.",
                    },
                    {
                        "name": "Healthy",
                        "explanation": "Vivid, even coloration with intact branching and no whitening, lesions, or breakage detected.",
                        "recommendation": "Continue routine monitoring. No action needed.",
                    },
                    {
                        "name": "Disease",
                        "explanation": "Discrete lesions or spotting consistent with a disease outbreak, distinct from generalized decay.",
                        "recommendation": "Isolate affected line segments and confirm the pathogen before treating.",
                    },
                    {
                        "name": "Decay",
                        "explanation": "Tissue melting with dark, mushy patches and a breakdown of branch structure, consistent with decay.",
                        "recommendation": "Remove affected fragments to prevent spread. Check water quality (temperature, salinity).",
                    },
                    {
                        "name": "Dried",
                        "explanation": "Tissue is brittle, bleached, and fully desiccated, with no living tissue remaining.",
                        "recommendation": "Remove and dispose of dried-out material. Inspect the surrounding line for early damage.",
                    },
                ],
            },
            {
                "key": "disease_subtype",
                "label": "Disease subtype",
                "type": "classification",
                "loss_weight": 0.5,
                "applies_when": {"key": "condition", "equals": "Disease"},
                "classes": [
                    {"name": "IceIce", "note": "Symptoms resemble ice-ice: raise water movement and reduce stress from high temperature/low salinity."},
                    {"name": "Epiphyte", "note": "Epiphyte overgrowth suspected: clean affected fronds and increase spacing/water flow."},
                    {"name": "Bacterial", "note": "Possible bacterial infection: isolate and consult a specialist before any treatment."},
                    {"name": "Bleaching", "note": "Bleaching suspected: check for temperature/light stress and relocate if possible."},
                    {"name": "Unknown", "note": "Subtype unclear: photograph affected areas closely and consult a specialist."},
                ],
            },
            {
                "key": "health_score",
                "label": "Health score",
                "type": "regression",
                "loss_weight": 1.0,
                "unit": "score",
                "min": 0.0,
                "max": 100.0,
                "applies_when": {"key": "condition", "not_equals": "Background"},
            },
            {
                "key": "dried_extent",
                "label": "Dried extent",
                "type": "regression",
                "loss_weight": 0.5,
                "unit": "pct",
                "min": 0.0,
                "max": 100.0,
                "applies_when": {"key": "condition", "not_equals": "Background"},
            },
            {
                "key": "decayed_extent",
                "label": "Decayed extent",
                "type": "regression",
                "loss_weight": 0.5,
                "unit": "pct",
                "min": 0.0,
                "max": 100.0,
                "applies_when": {"key": "condition", "not_equals": "Background"},
            },
        ],
    }
)


def legacy_schema_from_checkpoint(payload: dict) -> Schema:
    """Synthesizes a Schema for a checkpoint saved before schemas existed
    (payload has condition_classes/subtype_classes/species but no schema key),
    so old checkpoints keep loading. Preset copy is recovered from
    DEFAULT_SCHEMA for any class name that still matches; unmatched classes
    (e.g. an admin-renamed taxonomy) get no preset copy."""
    condition_classes: list[str] = payload.get("condition_classes") or list(DEFAULT_CONDITION_CLASSES)
    subtype_classes: list[str] = payload.get("subtype_classes") or list(DEFAULT_DISEASE_SUBTYPES)
    species: dict = payload.get("species") or dict(SPECIES)

    default_condition = DEFAULT_SCHEMA.find("condition")
    condition_by_name = {c.name: c for c in (default_condition.classes if default_condition else [])}
    default_subtype = DEFAULT_SCHEMA.find("disease_subtype")
    subtype_by_name = {c.name: c for c in (default_subtype.classes if default_subtype else [])}

    condition_measurement = MeasurementDef(
        key="condition",
        label="Condition",
        type="classification",
        loss_weight=1.0,
        background_class="Background" if "Background" in condition_classes else None,
        classes=[
            ClassDef(
                name=name,
                explanation=condition_by_name[name].explanation if name in condition_by_name else None,
                recommendation=condition_by_name[name].recommendation if name in condition_by_name else None,
            )
            for name in condition_classes
        ],
    )
    subtype_measurement = MeasurementDef(
        key="disease_subtype",
        label="Disease subtype",
        type="classification",
        loss_weight=0.5,
        applies_when=AppliesWhen(key="condition", equals="Disease") if "Disease" in condition_classes else None,
        classes=[
            ClassDef(name=name, note=subtype_by_name[name].note if name in subtype_by_name else None)
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
            applies_when=AppliesWhen(key="condition", not_equals="Background"),
        )
        for key, label, unit, weight in [
            ("health_score", "Health score", "score", 1.0),
            ("dried_extent", "Dried extent", "pct", 0.5),
            ("decayed_extent", "Decayed extent", "pct", 0.5),
        ]
    ]

    slug = species.get("slug", SPECIES["slug"])
    return Schema(
        species=[SpeciesDef(name=species.get("name", SPECIES["name"]), slug=slug)],
        active_species_slug=slug,
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

    # dataset/<species_slug>/{train,validation,test}/images|masks/*+annotations.jsonl
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
    def species_slug(self) -> str:
        return SCHEMA.active_species_slug

    @property
    def dataset_dir(self) -> Path:
        return self.dataset_root / SCHEMA.active_species_slug

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
