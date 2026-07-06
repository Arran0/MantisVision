"""Central configuration for the Kappaphycus alvarezii health classifier.

Single source of truth for paths, hyperparameters, and class labels so every
script (data loading, training, evaluation, Grad-CAM, inference API) agrees
on the same values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent

# Display/reference list of the 7 health classes. NOTE: torchvision's
# ImageFolder assigns label indices by *alphabetical* folder order, not this
# list's order, so the actual index<->label mapping used at train/inference
# time is `train_ds.classes` (see src/data/dataset.py), persisted into every
# checkpoint. This list just needs to contain the same set of names as the
# dataset folders.
CLASS_NAMES = [
    "Healthy",
    "Moderate",
    "Low",
    "Decay",
    "Dead",
    "Predator",
    "Disease",
]


@dataclass
class Config:
    seed: int = 42

    dataset_dir: Path = ML_ROOT / "dataset"
    train_dir: Path = ML_ROOT / "dataset" / "train"
    val_dir: Path = ML_ROOT / "dataset" / "validation"
    test_dir: Path = ML_ROOT / "dataset" / "test"

    checkpoints_dir: Path = ML_ROOT / "checkpoints"
    logs_dir: Path = ML_ROOT / "logs"
    reports_dir: Path = ML_ROOT / "reports"

    class_names: list[str] = field(default_factory=lambda: list(CLASS_NAMES))

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


config = Config()
