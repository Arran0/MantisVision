"""Central configuration for the seaweed health classifier.

Single source of truth for paths, hyperparameters, and the label taxonomy so
every script (data loading, training, calibration, evaluation, Grad-CAM,
inference API) agrees on the same values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# Phase 1 supports exactly one species. This is a placeholder pair, not a
# trained species classifier — the dataset directory convention below
# (dataset/<species_slug>/...) and the API response already carry a species
# field so a real classifier can replace this constant later without another
# breaking contract change (see docs/STEP_BY_STEP.md's roadmap).
SPECIES_SLUG = "kappaphycus_alvarezii"
SPECIES_DISPLAY_NAME = "Kappaphycus alvarezii"

# Fixed raw dataset folder names — always the same 5, always mean the same
# thing (see CLASS_TARGET_MAP below). "Disease" is deliberately NOT in this
# list: unlike Decay/Dried (which the labeling guide describes as inherently
# severe, hence always category=Low), a diseased specimen can be a small
# patch on an otherwise-Moderate specimen or a widespread Low-severity case.
# Disease severity therefore has to come from the folder name itself — see
# DISEASE_SEVERITIES/DISEASE_SUBTYPE_NAMES and parse_disease_folder() below —
# rather than a single fixed anchor.
CLASS_NAMES = [
    "Healthy",
    "Moderate",
    "Low",
    "Decay",
    "Dried",
]

# The taxonomy the model actually predicts: a 3-way severity category plus an
# optional condition tag. Condition is only meaningful when category != Healthy.
CATEGORY_NAMES = ["Healthy", "Moderate", "Low"]
CONDITION_NAMES = ["None", "Dried", "Decayed", "Diseased"]

# Disease-specific sub-taxonomy. Severity is folder-encoded (see
# parse_disease_folder), so a diseased specimen isn't forced to Low the way
# Decay/Dried are. Subtype starts as the 4 types already named in
# docs/STEP_BY_STEP.md's roadmap (Ice-Ice, Epiphyte, Bacterial, plus a
# catch-all Unknown) — extend this list once more subtypes have labeled
# photos; it's the one place the taxonomy is defined.
DISEASE_SEVERITIES = ["Moderate", "Low"]
DISEASE_SUBTYPE_NAMES = ["Unknown", "IceIce", "Epiphyte", "Bacterial"]

SCORE_MIN = 0.0
SCORE_MAX = 10.0
PCT_MIN = 0.0
PCT_MAX = 100.0


@dataclass(frozen=True)
class ClassTarget:
    category: str
    condition: str | None
    score_anchor: float
    # "N/A" (not "Unknown") when condition != "Diseased" — distinguishes "this
    # class has no disease subtype at all" from "this is a diseased sample
    # whose specific pathogen wasn't identified" (disease_subtype head is
    # masked to condition=="Diseased" samples during training either way).
    disease_subtype: str = "N/A"
    dried_pct_anchor: float = 0.0
    decayed_pct_anchor: float = 0.0


# Maps each of the 5 FIXED raw dataset folders to (category, condition,
# health-score anchor, dried%/decayed% anchors). Disease folders are handled
# separately by parse_disease_folder() below, since their category depends on
# the folder name (severity-encoded), not a single fixed value.
#
# Score anchors are a documented HEURISTIC, not measured biological ground
# truth (no expert-scored 0-10 dataset exists). Derived from the severity
# language in docs/DATASET_LABELING_GUIDE.md:
#   - Healthy=9.0: best-case description available ("bright green, no
#     whitening, no broken branches"); left headroom to 10 for a
#     hypothetical "pristine" example.
#   - Moderate=6.0: guide explicitly says "still actively growing" -> kept in
#     the upper-middle of the band rather than the bottom.
#   - Low=3.0 (generic/no named symptom): "significant discoloration, reduced
#     branching" with no described total-loss/rot -> placed above Decay/Dried,
#     which describe more advanced damage.
#   - Decay=2.0: "tissue melting, brown patches, rot" -> active degradation,
#     tissue still physically present.
#   - Dried=0.5: guide explicitly says "no living tissue anywhere in frame"
#     -> effectively total loss, anchored lowest, just above 0 (0 reserved
#     for a hypothetical "no specimen at all" edge case).
#
# dried_pct/decayed_pct anchors are likewise a documented heuristic (no
# pixel-level/segmentation ground truth exists): Dried is anchored high on
# dried_pct (~95%) with a small decayed_pct overlap (drying and decay can
# look visually similar at the edges); Decay is the mirror image; Low/Moderate
# get small nonzero decayed_pct anchors reflecting "small tissue loss" /
# "reduced branching" without full decay; Healthy is 0/0.
CLASS_TARGET_MAP: dict[str, ClassTarget] = {
    "Healthy": ClassTarget("Healthy", None, 9.0, dried_pct_anchor=0.0, decayed_pct_anchor=0.0),
    "Moderate": ClassTarget("Moderate", None, 6.0, dried_pct_anchor=0.0, decayed_pct_anchor=5.0),
    "Low": ClassTarget("Low", None, 3.0, dried_pct_anchor=0.0, decayed_pct_anchor=10.0),
    "Decay": ClassTarget("Low", "Decayed", 2.0, dried_pct_anchor=5.0, decayed_pct_anchor=70.0),
    "Dried": ClassTarget("Low", "Dried", 0.5, dried_pct_anchor=95.0, decayed_pct_anchor=5.0),
}

# Score anchor per disease severity — deliberately NOT as low as Decay/Dried:
# a diseased specimen is still, by definition, only "diseased" (lesions /
# infection), not melting or totally dried out. Moderate-severity disease sits
# just below plain Moderate (6.0) to reflect the disease patch pulling the
# score down a bit without crossing into Low; Low-severity disease sits close
# to the old flat Disease anchor from the single-class scheme.
_DISEASE_SEVERITY_SCORE_ANCHOR: dict[str, float] = {"Moderate": 5.0, "Low": 1.5}


def parse_disease_folder(name: str) -> ClassTarget | None:
    """Parses a raw dataset folder name of the form 'Disease_<Severity>' or
    'Disease_<Severity>_<Subtype>' into a ClassTarget. Returns None if `name`
    doesn't start with "Disease_" at all (i.e. isn't a disease folder).
    Raises ValueError for a "Disease_..." folder that doesn't match the
    expected severity/subtype vocabulary — a loud failure on a typo'd folder
    name is much better than silently mis-training on it.

    Examples: "Disease_Moderate", "Disease_Moderate_IceIce", "Disease_Low_Bacterial".
    """
    if not name.startswith("Disease_"):
        return None

    parts = name.split("_")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Malformed disease folder name '{name}'. Expected 'Disease_<Severity>' "
            f"or 'Disease_<Severity>_<Subtype>' (Severity in {DISEASE_SEVERITIES}, "
            f"Subtype in {DISEASE_SUBTYPE_NAMES})."
        )

    severity = parts[1]
    subtype = parts[2] if len(parts) == 3 else "Unknown"

    if severity not in DISEASE_SEVERITIES:
        raise ValueError(f"Unknown disease severity '{severity}' in folder '{name}'. Expected one of {DISEASE_SEVERITIES}.")
    if subtype not in DISEASE_SUBTYPE_NAMES:
        raise ValueError(f"Unknown disease subtype '{subtype}' in folder '{name}'. Expected one of {DISEASE_SUBTYPE_NAMES}.")

    return ClassTarget(
        category=severity,
        condition="Diseased",
        score_anchor=_DISEASE_SEVERITY_SCORE_ANCHOR[severity],
        disease_subtype=subtype,
        dried_pct_anchor=0.0,
        decayed_pct_anchor=0.0,
    )


def resolve_class_target(raw_class_name: str) -> ClassTarget:
    """Resolves any raw dataset folder name — one of the 5 fixed classes, or
    a 'Disease_<Severity>[_<Subtype>]' folder — to its training target.
    Raises ValueError for anything unrecognized (typo protection).
    """
    if raw_class_name in CLASS_TARGET_MAP:
        return CLASS_TARGET_MAP[raw_class_name]
    parsed = parse_disease_folder(raw_class_name)
    if parsed is None:
        raise ValueError(
            f"Unrecognized raw dataset folder '{raw_class_name}'. Expected one of "
            f"{list(CLASS_TARGET_MAP)} or a 'Disease_<Severity>[_<Subtype>]' folder."
        )
    return parsed


@dataclass
class Config:
    seed: int = 42

    dataset_dir: Path = ML_ROOT / "dataset" / SPECIES_SLUG
    train_dir: Path = ML_ROOT / "dataset" / SPECIES_SLUG / "train"
    val_dir: Path = ML_ROOT / "dataset" / SPECIES_SLUG / "validation"
    test_dir: Path = ML_ROOT / "dataset" / SPECIES_SLUG / "test"

    checkpoints_dir: Path = ML_ROOT / "checkpoints"
    logs_dir: Path = ML_ROOT / "logs"
    reports_dir: Path = ML_ROOT / "reports"

    # The 5 FIXED raw class names. Disease_* folders are discovered on disk at
    # load time (see src/data/dataset.py) rather than listed here, since any
    # number of severity/subtype combinations may or may not exist yet.
    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    category_names: list[str] = field(default_factory=lambda: list(CATEGORY_NAMES))
    condition_names: list[str] = field(default_factory=lambda: list(CONDITION_NAMES))
    disease_subtype_names: list[str] = field(default_factory=lambda: list(DISEASE_SUBTYPE_NAMES))

    score_min: float = SCORE_MIN
    score_max: float = SCORE_MAX
    pct_min: float = PCT_MIN
    pct_max: float = PCT_MAX
    # +/- uniform noise added to anchor targets on the train split only, so
    # the heads can't trivially collapse to a handful of constant outputs and
    # are forced to read graded severity/extent from pixels.
    score_anchor_jitter: float = 0.5
    pct_anchor_jitter: float = 5.0

    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "category": 1.0,
            "condition": 1.0,
            "score": 1.0,
            "disease_subtype": 1.0,
            "extent": 1.0,
        }
    )

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

    device: str = "cuda"  # falls back to cpu automatically, see utils.device

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    @property
    def num_conditions(self) -> int:
        return len(self.condition_names)

    @property
    def num_disease_subtypes(self) -> int:
        return len(self.disease_subtype_names)


config = Config()
