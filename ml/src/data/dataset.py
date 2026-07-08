"""Dataloader construction for the multi-head health-classification dataset.

Expects the ImageFolder-compatible layout described in the dataset spec:

    dataset/<species_slug>/train/<ClassName>/*.jpg
    dataset/<species_slug>/validation/<ClassName>/*.jpg
    dataset/<species_slug>/test/<ClassName>/*.jpg

`<ClassName>` is either one of the 5 fixed raw classes (config.CLASS_NAMES)
or a `Disease_<Severity>[_<Subtype>]` folder (config.parse_disease_folder) —
any number of severity/subtype combinations may exist, not all of them.
torchvision sorts folders alphabetically internally; we verify the mapping
explicitly below instead of trusting sort order, so renaming/reordering
CLASS_NAMES can't silently swap two labels.

Each raw `<ClassName>` folder is remapped through `config.resolve_class_target`
into six training targets per image: category, condition, health-score
anchor, disease subtype, and dried%/decayed% anchors (see config.py for the
mapping and its justification).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from config import Config, resolve_class_target
from src.data.transforms import build_transforms


@dataclass
class MultiHeadDataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    category_names: list[str]
    condition_names: list[str]
    disease_subtype_names: list[str]


class MultiHeadFolderDataset(Dataset):
    """Wraps an ImageFolder, remapping its single raw-class label into a
    dict of training targets via `config.resolve_class_target`.
    """

    def __init__(
        self,
        image_folder: ImageFolder,
        category_names: list[str],
        condition_names: list[str],
        disease_subtype_names: list[str],
        train: bool,
        score_jitter: float,
        pct_jitter: float,
        score_min: float,
        score_max: float,
        pct_min: float,
        pct_max: float,
    ) -> None:
        self.image_folder = image_folder
        self.category_names = category_names
        self.condition_names = condition_names
        self.disease_subtype_names = disease_subtype_names
        self.train = train
        self.score_jitter = score_jitter
        self.pct_jitter = pct_jitter
        self.score_min = score_min
        self.score_max = score_max
        self.pct_min = pct_min
        self.pct_max = pct_max

    def __len__(self) -> int:
        return len(self.image_folder)

    def _jitter(self, value: float, jitter: float, lo: float, hi: float) -> float:
        if self.train and jitter:
            value = value + random.uniform(-jitter, jitter)
        return min(max(value, lo), hi)

    def __getitem__(self, idx: int):
        image, raw_idx = self.image_folder[idx]
        raw_class_name = self.image_folder.classes[raw_idx]
        entry = resolve_class_target(raw_class_name)

        category_idx = self.category_names.index(entry.category)
        condition_idx = self.condition_names.index(entry.condition or "None")
        disease_subtype_idx = self.disease_subtype_names.index(
            entry.disease_subtype if entry.disease_subtype != "N/A" else "Unknown"
        )

        score = self._jitter(entry.score_anchor, self.score_jitter, self.score_min, self.score_max)
        dried_pct = self._jitter(entry.dried_pct_anchor, self.pct_jitter, self.pct_min, self.pct_max)
        decayed_pct = self._jitter(entry.decayed_pct_anchor, self.pct_jitter, self.pct_min, self.pct_max)

        targets = {
            "category": category_idx,
            "condition": condition_idx,
            "score": torch.tensor(score, dtype=torch.float32),
            "disease_subtype": disease_subtype_idx,
            "extent": torch.tensor([dried_pct, decayed_pct], dtype=torch.float32),
        }
        return image, targets


def _verify_class_mapping(dataset: ImageFolder, fixed_class_names: list[str]) -> None:
    found = set(dataset.classes)
    missing_fixed = set(fixed_class_names) - found
    if missing_fixed:
        raise ValueError(
            f"Missing required class folders: {sorted(missing_fixed)}\n"
            f"Found: {sorted(found)}"
        )
    # Anything beyond the 5 fixed classes must be a valid Disease_* folder —
    # resolve_class_target raises ValueError on anything unrecognized, which
    # is exactly the loud failure we want on a typo'd folder name.
    for name in found - set(fixed_class_names):
        resolve_class_target(name)


def get_multihead_dataloaders(cfg: Config) -> MultiHeadDataloaders:
    train_folder = ImageFolder(cfg.train_dir, transform=build_transforms(cfg, train=True))
    val_folder = ImageFolder(cfg.val_dir, transform=build_transforms(cfg, train=False))
    test_folder = ImageFolder(cfg.test_dir, transform=build_transforms(cfg, train=False))

    _verify_class_mapping(train_folder, cfg.class_names)
    _verify_class_mapping(val_folder, cfg.class_names)
    _verify_class_mapping(test_folder, cfg.class_names)

    def wrap(image_folder: ImageFolder, train: bool) -> MultiHeadFolderDataset:
        return MultiHeadFolderDataset(
            image_folder,
            cfg.category_names,
            cfg.condition_names,
            cfg.disease_subtype_names,
            train=train,
            score_jitter=cfg.score_anchor_jitter if train else 0.0,
            pct_jitter=cfg.pct_anchor_jitter if train else 0.0,
            score_min=cfg.score_min,
            score_max=cfg.score_max,
            pct_min=cfg.pct_min,
            pct_max=cfg.pct_max,
        )

    train_ds = wrap(train_folder, train=True)
    val_ds = wrap(val_folder, train=False)
    test_ds = wrap(test_folder, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return MultiHeadDataloaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        category_names=cfg.category_names,
        condition_names=cfg.condition_names,
        disease_subtype_names=cfg.disease_subtype_names,
    )
