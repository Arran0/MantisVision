"""Dataloader construction for the schema-driven multi-head seaweed model.

Each split directory (dataset/<species_slug>/{train,validation,test}/) holds
flat images/ + masks/<key>/ + an annotations.jsonl manifest (see
src/data/annotations.py) — replacing the old ImageFolder class-folder
convention, since per-image column annotations (not folder names) are now the
source of truth. Which targets get built, and how many, is entirely driven by
the active Schema, so a newly admin-added measurement needs no dataset.py
change: its (target, mask) pair is produced generically by
annotations.derive_targets / load_segmentation_target.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config, Schema
from src.data.annotations import AnnotationRow, derive_targets, load_manifest, load_segmentation_target
from src.data.transforms import build_transforms


@dataclass
class Dataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    schema: Schema


class AnnotatedDataset(Dataset):
    """Wraps a split directory's annotations.jsonl manifest; each sample
    yields the image tensor plus the full set of per-measurement targets."""

    def __init__(self, root: Path, cfg: Config, schema: Schema, train: bool) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.schema = schema
        self.transform = build_transforms(cfg, train=train)
        self.rows: list[AnnotationRow] = load_manifest(self.root / "annotations.jsonl")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.root / "images" / row.filename).convert("RGB")
        image = self.transform(image)

        targets = derive_targets(self.schema, row.measurements)
        for m in self.schema.measurements:
            if m.type != "segmentation":
                continue
            mask, flag = load_segmentation_target(self.root, m, row)
            # Resize to the model's input resolution regardless of the mask's
            # source size (including the 1x1 "no mask" placeholder, which
            # nearest-neighbor-upsamples to a uniform all-zero mask).
            resized = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.cfg.image_size, self.cfg.image_size),
                mode="nearest",
            )
            targets[f"{m.key}_seg"] = resized.squeeze(0).squeeze(0).long()
            targets[f"{m.key}_seg_mask"] = flag

        return image, targets


def build_collate(schema: Schema):
    def _collate(batch):
        images = torch.stack([item[0] for item in batch])
        targets: dict[str, torch.Tensor] = {}
        for m in schema.measurements:
            if m.type == "classification":
                targets[f"{m.key}_id"] = torch.tensor([item[1][f"{m.key}_id"] for item in batch], dtype=torch.long)
                targets[f"{m.key}_mask"] = torch.tensor(
                    [item[1][f"{m.key}_mask"] for item in batch], dtype=torch.float32
                )
            elif m.type == "regression":
                targets[m.key] = torch.tensor([item[1][m.key] for item in batch], dtype=torch.float32)
                targets[f"{m.key}_mask"] = torch.tensor(
                    [item[1][f"{m.key}_mask"] for item in batch], dtype=torch.float32
                )
            elif m.type == "segmentation":
                targets[f"{m.key}_seg"] = torch.stack([item[1][f"{m.key}_seg"] for item in batch])
                targets[f"{m.key}_seg_mask"] = torch.tensor(
                    [item[1][f"{m.key}_seg_mask"] for item in batch], dtype=torch.float32
                )
        return images, targets

    return _collate


def _verify_dataset(rows: list[AnnotationRow], schema: Schema, split_name: str) -> None:
    # The model needs negatives to train against — the whole point of the N+1
    # design. Only enforced when the schema actually declares a background
    # class (a schema is free not to have one, though the web validator
    # requires at least one across the whole schema).
    primary = schema.primary_classification()
    if primary is None:
        return
    values = {row.measurements.get(primary.key) for row in rows}
    if primary.background_class not in values:
        raise ValueError(
            f"No {primary.background_class!r} (background) sample found in the {split_name!r} split "
            f"for measurement {primary.key!r}. The model needs diverse non-subject images to avoid "
            "false positives."
        )


def get_dataloaders(cfg: Config, schema: Schema) -> Dataloaders:
    train_ds = AnnotatedDataset(cfg.train_dir, cfg, schema, train=True)
    val_ds = AnnotatedDataset(cfg.val_dir, cfg, schema, train=False)
    test_ds = AnnotatedDataset(cfg.test_dir, cfg, schema, train=False)

    _verify_dataset(train_ds.rows, schema, "train")
    _verify_dataset(val_ds.rows, schema, "validation")
    _verify_dataset(test_ds.rows, schema, "test")

    collate = build_collate(schema)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return Dataloaders(train=train_loader, val=val_loader, test=test_loader, schema=schema)
