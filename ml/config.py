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

# Raw dataset folder names. NOTE: torchvision's ImageFolder assigns label
# indices by *alphabetical* folder order, not this list's order, so the
# actual index<->label mapping used at train/inference time is
# `train_ds.classes` (see src/data/dataset.py), persisted into every
# checkpoint. This list just needs to contain the same set of names as the
# dataset folders.
CLASS_NAMES = [
    "Healthy",
    "Moderate",
    "Low",
    "Decay",
    "Dried",
    "Disease",
]

# The taxonomy the model actually predicts: a 3-way severity category plus an
# optional condition tag. Condition is only meaningful when category != Healthy.
CATEGORY_NAMES = ["Healthy", "Moderate", "Low"]
CONDITION_NAMES = ["None", "Dried", "Decayed", "Diseased"]

SCORE_MIN = 0.0
SCORE_MAX = 10.0


@dataclass(frozen=True)
class ClassTarget:
    category: str
    condition: str | None
    score_anchor: float


# Maps each raw dataset folder to (category, condition, health-score anchor).
#
# Confirmed decision: Decay/Dried/Disease all map to category="Low" (rather
# than being split across Moderate/Low) because the labeling guide's own
# definitions of all three describe severe, advanced symptoms (rot, zero
# living tissue, active lesions) consistent with the Low tier — no dataset
# currently distinguishes a "moderate-severity dried" photo from a
# "low-severity dried" one.
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
#     branching" with no described total-loss/rot/infection -> placed above
#     the three named sub-conditions, which describe more advanced damage.
#   - Decay=2.0: "tissue melting, brown patches, rot" -> active degradation,
#     tissue still physically present.
#   - Disease=1.5: "visible lesions, infection symptoms" -> ranked marginally
#     below Decay for contagion risk, NOT because the guide ranks it
#     explicitly. This Decay-vs-Disease ordering is the most arbitrary part
#     of the scheme and the first candidate for revision once real
#     expert-scored data exists.
#   - Dried=0.5: guide explicitly says "no living tissue anywhere in frame"
#     -> effectively total loss, anchored lowest, just above 0 (0 reserved
#     for a hypothetical "no specimen at all" edge case).
CLASS_TARGET_MAP: dict[str, ClassTarget] = {
    "Healthy": ClassTarget("Healthy", None, 9.0),
    "Moderate": ClassTarget("Moderate", None, 6.0),
    "Low": ClassTarget("Low", None, 3.0),
    "Decay": ClassTarget("Low", "Decayed", 2.0),
    "Disease": ClassTarget("Low", "Diseased", 1.5),
    "Dried": ClassTarget("Low", "Dried", 0.5),
}


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

    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))
    category_names: list[str] = field(default_factory=lambda: list(CATEGORY_NAMES))
    condition_names: list[str] = field(default_factory=lambda: list(CONDITION_NAMES))

    score_min: float = SCORE_MIN
    score_max: float = SCORE_MAX
    # +/- uniform noise added to anchor targets on the train split only, so
    # the score head can't trivially collapse to 6 constant outputs and is
    # forced to read graded severity from pixels.
    score_anchor_jitter: float = 0.5

    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"category": 1.0, "condition": 1.0, "score": 1.0}
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
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_categories(self) -> int:
        return len(self.category_names)

    @property
    def num_conditions(self) -> int:
        return len(self.condition_names)


config = Config()
