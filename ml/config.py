"""Central configuration for the seaweed multi-head classifier.

Single source of truth for the active species, the label taxonomy, the
heuristic anchors that turn discrete labels into regression targets, paths,
and hyperparameters — so every script (data loading, training, evaluation,
Grad-CAM, inference API) agrees on the same values.

The model is multi-head (see src/models/efficientnet.py):
  1. condition       - what is in frame (incl. a Background negative class)
  2. health_score    - 0-100 regression
  3. disease_subtype - only meaningful when condition == "Disease"
  4. dried_extent    - 0-100 regression (% of frame dried)
  5. decayed_extent  - 0-100 regression (% of frame decayed)

Discrete health *level* (Healthy/Moderate/Low) is NOT a class — it's derived
at inference from condition + the regressed health_score (see
src/inference/predictor.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# --- Active species -------------------------------------------------------
# The dataset lives under dataset/<slug>/ and the slug prefixes every class
# folder (e.g. Kappaphycus_alvarezii_Healthy). Swapping to another species
# later is a one-line change here; nothing else hardcodes the species. There
# is no species *classifier* yet, so exactly one species is active at a time.
SPECIES = {"name": "Kappaphycus alvarezii", "slug": "Kappaphycus_alvarezii"}

# --- Condition head (N + 1) ----------------------------------------------
# "Background" is the +1 negative class, trained on diverse non-seaweed
# images (empty scenes, rocks, other objects) to stop false positives.
# Order is FIXED and authoritative (persisted into every checkpoint) — unlike
# the old ImageFolder alphabetical ordering, the custom SeaweedDataset maps
# folders to these indices explicitly.
CONDITION_CLASSES = ["Background", "Healthy", "Disease", "Decay", "Dried"]

# --- Disease subtype head -------------------------------------------------
# Supervised only on Disease samples. "Unknown" absorbs anything not (yet)
# categorised into a specific subtype.
DISEASE_SUBTYPES = ["IceIce", "Epiphyte", "Bacterial", "Bleaching", "Unknown"]

# Severity tokens that prefix a Decay/Dried/Disease folder name (e.g.
# <slug>_Low_Decay, <slug>_Moderate_Disease_IceIce). Decay and Dried only ever
# use "Low" — there are no Moderate/Healthy-severity buckets for them, since
# by definition a decayed or dried specimen is already at the bottom of the
# health range. Disease uses both. Severity sets the training anchor for the
# health score; at inference Disease's severity is read back out of the
# regressed score (see DISEASE_MODERATE_MIN below) rather than trusted as-is.
SEVERITIES = ["Moderate", "Low"]

# The only severity Decay/Dried folders may use.
FIXED_SEVERITY_CONDITIONS = {"Decay": "Low", "Dried": "Low"}

# --- Heuristic anchors ----------------------------------------------------
# Discrete labels have no ground-truth number attached, so we anchor each
# condition (and, for Disease, each severity) to a default target the
# regression heads learn toward. An admin can override any of these per image
# (see the admin dataset form); the anchor is only the fallback.
#
# Health score, 0-100 (higher = healthier). Keyed by condition, with Disease
# further split by severity.
HEALTH_SCORE_ANCHORS: dict[str, float] = {
    "Healthy": 90.0,
    "Disease:Moderate": 60.0,
    "Disease:Low": 30.0,
    "Decay": 20.0,
    "Dried": 5.0,
}

# % of frame dried / decayed, 0-100. Conditions not listed anchor to 0.
DRIED_EXTENT_ANCHORS: dict[str, float] = {"Dried": 90.0, "Decay": 10.0}
DECAYED_EXTENT_ANCHORS: dict[str, float] = {"Decay": 80.0, "Disease": 20.0}

# At inference, a Disease prediction is called "Moderate" if the regressed
# health score is at or above this, else "Low".
DISEASE_MODERATE_MIN: float = 45.0


@dataclass
class Config:
    seed: int = 42

    # dataset/<species_slug>/{train,validation,test}/<class_folder>/
    dataset_root: Path = ML_ROOT / "dataset"

    checkpoints_dir: Path = ML_ROOT / "checkpoints"
    logs_dir: Path = ML_ROOT / "logs"
    reports_dir: Path = ML_ROOT / "reports"
    metadata_dir: Path = ML_ROOT / "metadata"

    condition_classes: list[str] = field(default_factory=lambda: list(CONDITION_CLASSES))
    disease_subtypes: list[str] = field(default_factory=lambda: list(DISEASE_SUBTYPES))

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

    # Relative weights of each head's term in the multi-task loss. Regression
    # terms are on a 0-1 scale (targets divided by 100), so they're weighted up
    # to stay comparable to the cross-entropy terms.
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "condition": 1.0,
            "health_score": 1.0,
            "disease_subtype": 0.5,
            "dried_extent": 0.5,
            "decayed_extent": 0.5,
        }
    )
    # Label smoothing on the condition head — a cheap, first-line defense
    # against label noise. See train.py for the (documented) heavier option.
    condition_label_smoothing: float = 0.1

    device: str = "cuda"  # falls back to cpu automatically, see utils.seed

    @property
    def species_slug(self) -> str:
        return SPECIES["slug"]

    @property
    def dataset_dir(self) -> Path:
        return self.dataset_root / SPECIES["slug"]

    @property
    def train_dir(self) -> Path:
        return self.dataset_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.dataset_dir / "validation"

    @property
    def test_dir(self) -> Path:
        return self.dataset_dir / "test"

    @property
    def num_conditions(self) -> int:
        return len(self.condition_classes)

    @property
    def num_subtypes(self) -> int:
        return len(self.disease_subtypes)


config = Config()
