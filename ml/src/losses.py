"""Schema-driven multi-task loss for the multi-head model, shared by
train.py and evaluate.py.

Every measurement in the active Schema contributes one term, combined as a
weighted sum (weight = that measurement's loss_weight):

  classification   cross-entropy, masked to samples where the measurement
                    applies and has a labeled value (see
                    src/data/annotations.py); the schema's primary
                    classification (the one with a background_class) gets
                    label smoothing as a first-line defense against label
                    noise, same as the old fixed "condition" head did
  regression        Smooth-L1 on the value normalized to its measurement's
                    [min, max] range, masked the same way
  segmentation      masked per-image mean pixel cross-entropy + soft Dice,
                    masked to samples with a ground-truth mask

A term is skipped (contributes 0) when its batch mask selects no samples
(e.g. a batch with no Disease examples, or a measurement no image has a
mask/value for yet), so an empty mask never produces a NaN — this is also how
a newly admin-added measurement with zero labeled data trains safely: its
term is just always 0 until real values start arriving.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config, Schema


def build_criterions(schema: Schema, cfg: Config) -> dict:
    criterions: dict = {}
    for m in schema.measurements:
        if m.type == "classification":
            # Label smoothing only for the primary (background-carrying)
            # classification — mirrors the old design, where only "condition"
            # (not "disease_subtype") was smoothed.
            smoothing = cfg.condition_label_smoothing if m.background_class else 0.0
            criterions[m.key] = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="none")
        elif m.type == "segmentation":
            criterions[m.key] = nn.CrossEntropyLoss(reduction="none")
    return criterions


def _masked_mean(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return per_sample.new_tensor(0.0)
    return (per_sample * mask).sum() / mask.sum()


def _regression_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    span = (hi - lo) or 1.0
    per_sample = F.smooth_l1_loss((pred - lo) / span, (target - lo) / span, reduction="none")
    return _masked_mean(per_sample, mask)


def _dice_loss(logits: torch.Tensor, mask_ids: torch.Tensor, num_classes: int, sample_mask: torch.Tensor) -> torch.Tensor:
    if sample_mask.sum() == 0:
        return logits.new_tensor(0.0)
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    one_hot = F.one_hot(mask_ids, num_classes=num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    dims = (2, 3)
    intersection = (probs * one_hot).sum(dims)
    union = probs.sum(dims) + one_hot.sum(dims)
    dice_per_class = (2 * intersection + 1e-6) / (union + 1e-6)
    dice_per_image = 1 - dice_per_class.mean(dim=1)
    return _masked_mean(dice_per_image, sample_mask)


def compute_losses(outputs: dict, targets: dict, schema: Schema, criterions: dict, cfg: Config) -> tuple[torch.Tensor, dict]:
    """Returns (total_loss, per_measurement_scalar_dict). Per-measurement
    values are detached floats for logging."""
    total: torch.Tensor | None = None
    parts: dict[str, float] = {}

    for m in schema.measurements:
        if m.type == "classification":
            mask = targets[f"{m.key}_mask"]
            per_sample = criterions[m.key](outputs[m.key], targets[f"{m.key}_id"])
            loss = _masked_mean(per_sample, mask)
        elif m.type == "regression":
            mask = targets[f"{m.key}_mask"]
            loss = _regression_loss(outputs[m.key], targets[m.key], mask, m.min, m.max)
        elif m.type == "segmentation":
            sample_mask = targets[f"{m.key}_seg_mask"]
            mask_ids = targets[f"{m.key}_seg"]
            per_pixel = criterions[m.key](outputs[m.key], mask_ids)
            per_image_ce = per_pixel.mean(dim=(1, 2))
            ce_loss = _masked_mean(per_image_ce, sample_mask)
            dice_loss = _dice_loss(outputs[m.key], mask_ids, outputs[m.key].shape[1], sample_mask)
            loss = 0.5 * ce_loss + 0.5 * dice_loss
        else:
            continue

        weighted = m.loss_weight * loss
        total = weighted if total is None else total + weighted
        parts[m.key] = float(loss.detach())

    if total is None:
        raise ValueError("Schema has no measurements to compute a loss over.")
    parts["total"] = float(total.detach())
    return total, parts
