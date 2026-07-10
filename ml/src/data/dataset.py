"""Dataloader construction for the multi-head seaweed model.

Expects flat, ImageFolder-compatible class folders whose names encode
structured labels (see src/data/labels.py):

    dataset/<species_slug>/train/<class_folder>/*.jpg
    dataset/<species_slug>/validation/<class_folder>/*.jpg
    dataset/<species_slug>/test/<class_folder>/*.jpg

We keep torchvision's ImageFolder for robust, well-tested file discovery, then
wrap it so each sample yields the full set of multi-head targets rather than a
single class index. Folder -> targets goes entirely through labels.py, so the
naming convention lives in one place.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from config import Config
from src.data.labels import BACKGROUND, derive_targets, parse_class_folder
from src.data.transforms import build_transforms

# Keys produced per sample. Integer-typed targets (class indices) vs float
# targets (regression values + masks) are collated into different dtypes.
_INT_KEYS = ("condition_id", "subtype_id")
_FLOAT_KEYS = (
    "health_score",
    "dried_extent",
    "decayed_extent",
    "subtype_mask",
    "health_mask",
    "extent_mask",
)


@dataclass
class Dataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    condition_classes: list[str]
    subtype_classes: list[str]


class SeaweedDataset(Dataset):
    """Wraps ImageFolder; maps each sample's folder to multi-head targets."""

    def __init__(self, root, cfg: Config, train: bool) -> None:
        self.inner = ImageFolder(root, transform=build_transforms(cfg, train=train))
        self.cfg = cfg
        # Precompute targets per ImageFolder class index so we don't re-parse
        # a folder name on every __getitem__.
        self._targets_by_folder_idx: dict[int, dict] = {}
        for folder_name, folder_idx in self.inner.class_to_idx.items():
            parsed = parse_class_folder(folder_name, cfg.species_slug)
            self._targets_by_folder_idx[folder_idx] = derive_targets(parsed)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        image, folder_idx = self.inner[index]
        return image, self._targets_by_folder_idx[folder_idx]

    @property
    def folder_names(self) -> list[str]:
        return self.inner.classes


def _collate(batch):
    images = torch.stack([item[0] for item in batch])
    targets: dict[str, torch.Tensor] = {}
    for key in _INT_KEYS:
        targets[key] = torch.tensor([item[1][key] for item in batch], dtype=torch.long)
    for key in _FLOAT_KEYS:
        targets[key] = torch.tensor([item[1][key] for item in batch], dtype=torch.float32)
    return images, targets


def _verify_dataset(dataset: SeaweedDataset) -> None:
    # parse_class_folder already ran (would have raised on a bad name); here we
    # just insist the Background negative class is present, since the whole
    # point of the N+1 design is to have negatives to train against.
    if BACKGROUND not in dataset.folder_names:
        raise ValueError(
            f"No {BACKGROUND!r} folder found in {dataset.inner.root}. The model needs "
            "diverse non-seaweed images to avoid false positives — add a Background class."
        )


def get_dataloaders(cfg: Config) -> Dataloaders:
    train_ds = SeaweedDataset(cfg.train_dir, cfg, train=True)
    val_ds = SeaweedDataset(cfg.val_dir, cfg, train=False)
    test_ds = SeaweedDataset(cfg.test_dir, cfg, train=False)

    for ds in (train_ds, val_ds, test_ds):
        _verify_dataset(ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=_collate,
    )

    return Dataloaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        # Authoritative, fixed order from config (persisted into checkpoints) —
        # NOT ImageFolder's alphabetical folder order, which the multi-head
        # targets deliberately don't depend on.
        condition_classes=list(cfg.condition_classes),
        subtype_classes=list(cfg.disease_subtypes),
    )
