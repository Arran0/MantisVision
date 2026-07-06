"""Dataloader construction for the health-classification dataset.

Expects the ImageFolder-compatible layout described in the dataset spec:

    dataset/train/<ClassName>/*.jpg
    dataset/validation/<ClassName>/*.jpg
    dataset/test/<ClassName>/*.jpg

`<ClassName>` directories must exactly match `config.CLASS_NAMES` (torchvision
sorts them alphabetically internally; we verify the mapping explicitly below
instead of trusting sort order, so renaming/reordering CLASS_NAMES can't
silently swap two labels).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import Config
from src.data.transforms import build_transforms


@dataclass
class Dataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]


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


def get_dataloaders(cfg: Config) -> Dataloaders:
    train_ds = ImageFolder(cfg.train_dir, transform=build_transforms(cfg, train=True))
    val_ds = ImageFolder(cfg.val_dir, transform=build_transforms(cfg, train=False))
    test_ds = ImageFolder(cfg.test_dir, transform=build_transforms(cfg, train=False))

    _verify_class_mapping(train_ds, cfg.class_names)
    _verify_class_mapping(val_ds, cfg.class_names)
    _verify_class_mapping(test_ds, cfg.class_names)

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

    return Dataloaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        # ImageFolder assigns label indices by alphabetical folder order, which
        # does NOT match config.CLASS_NAMES' display order. Always decode
        # predictions using this list (it's what train.py saves into the
        # checkpoint), never cfg.class_names directly.
        class_names=train_ds.classes,
    )
