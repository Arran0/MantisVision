"""Image preprocessing + augmentation pipeline.

Pipeline order matches the spec: resize -> normalize -> augmentation (train
only). Augmentations are restricted to ones that don't change biological
meaning (no vertical flips, no color-inverting transforms, no aggressive
crops that could remove the diagnostic tissue in frame).
"""
from __future__ import annotations

import torch
from torchvision import transforms

from config import Config


class GaussianNoise:
    """Adds sensor-like noise so the model doesn't overfit to clean lab photos."""

    def __init__(self, std: float = 0.03) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def build_transforms(cfg: Config, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std)

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((int(cfg.image_size * 1.15), int(cfg.image_size * 1.15))),
            transforms.RandomCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            # Blur augmentation: field photos are often soft/out-of-focus, so
            # teach the model to tolerate it rather than shortcutting on
            # lab-sharp images. Combined with the noise + brightness jitter
            # below, this is the main defense (with label smoothing in
            # train.py) against real-world quality variation and label noise.
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            GaussianNoise(std=0.03),
            normalize,
        ]
    )
