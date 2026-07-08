"""Dataloader construction for the multi-head health-classification dataset.

Expects the ImageFolder-compatible layout described in the dataset spec:

    dataset/<species_slug>/train/<ClassName>/*.jpg
    dataset/<species_slug>/validation/<ClassName>/*.jpg
    dataset/<species_slug>/test/<ClassName>/*.jpg

`<ClassName>` directories must exactly match `config.CLASS_NAMES` (torchvision
sorts them alphabetically internally; we verify the mapping explicitly below
instead of trusting sort order, so renaming/reordering CLASS_NAMES can't
silently swap two labels).

Each raw `<ClassName>` folder is remapped through `config.CLASS_TARGET_MAP`
into three training targets per image: category, condition, and a health-score
anchor (see config.py for the mapping and its justification).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from config import ClassTarget, Config
from src.data.transforms import build_transforms


@dataclass
class MultiHeadDataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    category_names: list[str]
    condition_names: list[str]


class MultiHeadFolderDataset(Dataset):
    """Wraps an ImageFolder, remapping its single raw-class label into
    (category_idx, condition_idx, score_target) via `target_map`.
    """

    def __init__(
        self,
        image_folder: ImageFolder,
        target_map: dict[str, ClassTarget],
        category_names: list[str],
        condition_names: list[str],
        train: bool,
        jitter: float,
        score_min: float,
        score_max: float,
    ) -> None:
        self.image_folder = image_folder
        self.target_map = target_map
        self.category_names = category_names
        self.condition_names = condition_names
        self.train = train
        self.jitter = jitter
        self.score_min = score_min
        self.score_max = score_max

    def __len__(self) -> int:
        return len(self.image_folder)

    def __getitem__(self, idx: int):
        image, raw_idx = self.image_folder[idx]
        raw_class_name = self.image_folder.classes[raw_idx]
        entry = self.target_map[raw_class_name]

        category_idx = self.category_names.index(entry.category)
        condition_idx = self.condition_names.index(entry.condition or "None")

        score = entry.score_anchor
        if self.train and self.jitter:
            score = min(max(score + random.uniform(-self.jitter, self.jitter), self.score_min), self.score_max)

        return image, category_idx, condition_idx, torch.tensor(score, dtype=torch.float32)


def _verify_class_mapping(dataset: ImageFolder, expected: list[str]) -> None:
    found = sorted(dataset.classes)
    missing = set(expected) - set(found)
    unexpected = set(found) - set(expected)
    if missing or unexpected:
        raise ValueError(
            "Dataset class folders do not match config.CLASS_NAMES.\n"
            f"Missing folders: {sorted(missing) or 'none'}\n"
            f"Unexpected folders: {sorted(unexpected) or 'none'}\n"
            f"Expected: {expected}"
        )


def _verify_target_map(class_names: list[str], target_map: dict[str, ClassTarget]) -> None:
    if set(target_map.keys()) != set(class_names):
        raise ValueError(
            "config.CLASS_TARGET_MAP keys do not match config.CLASS_NAMES.\n"
            f"CLASS_TARGET_MAP keys: {sorted(target_map.keys())}\n"
            f"CLASS_NAMES: {sorted(class_names)}"
        )


def get_multihead_dataloaders(cfg: Config, target_map: dict[str, ClassTarget]) -> MultiHeadDataloaders:
    _verify_target_map(cfg.class_names, target_map)

    train_folder = ImageFolder(cfg.train_dir, transform=build_transforms(cfg, train=True))
    val_folder = ImageFolder(cfg.val_dir, transform=build_transforms(cfg, train=False))
    test_folder = ImageFolder(cfg.test_dir, transform=build_transforms(cfg, train=False))

    _verify_class_mapping(train_folder, cfg.class_names)
    _verify_class_mapping(val_folder, cfg.class_names)
    _verify_class_mapping(test_folder, cfg.class_names)

    def wrap(image_folder: ImageFolder, train: bool, jitter: float) -> MultiHeadFolderDataset:
        return MultiHeadFolderDataset(
            image_folder,
            target_map,
            cfg.category_names,
            cfg.condition_names,
            train=train,
            jitter=jitter,
            score_min=cfg.score_min,
            score_max=cfg.score_max,
        )

    train_ds = wrap(train_folder, train=True, jitter=cfg.score_anchor_jitter)
    val_ds = wrap(val_folder, train=False, jitter=0.0)
    test_ds = wrap(test_folder, train=False, jitter=0.0)

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
    )
