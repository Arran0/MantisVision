"""Multi-task loss for the multi-head model, shared by train.py and evaluate.py.

Each head contributes a term, combined as a weighted sum (weights from
config.loss_weights):

  condition        cross-entropy over ALL samples (incl. Background), with
                   label smoothing as a first-line defense against label noise
  disease_subtype  cross-entropy, masked to Disease samples only
  health_score     Smooth-L1, masked to exclude Background (nothing to score)
  dried/decayed    Smooth-L1, masked to exclude Background

Regression targets and outputs are both on a 0-100 scale but divided by 100
before the loss so every term sits on a comparable ~0-1 magnitude. A head's
term is skipped when its batch mask selects no samples (e.g. a batch with no
Disease examples), so an empty mask never produces a NaN.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


def build_criterions(cfg: Config) -> dict:
    return {
        "condition": nn.CrossEntropyLoss(label_smoothing=cfg.condition_label_smoothing),
        # reduction="none" so we can apply the per-sample subtype mask ourselves
        "disease_subtype": nn.CrossEntropyLoss(reduction="none"),
    }


def _masked_smooth_l1(pred_0_100: torch.Tensor, target_0_100: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred_0_100.new_tensor(0.0)
    per = F.smooth_l1_loss(pred_0_100 / 100.0, target_0_100 / 100.0, reduction="none")
    return (per * mask).sum() / mask.sum()


def compute_losses(outputs: dict, targets: dict, cfg: Config, criterions: dict) -> tuple[torch.Tensor, dict]:
    """Returns (total_loss, per_head_scalar_dict). Per-head values are detached
    floats for logging."""
    w = cfg.loss_weights

    condition_loss = criterions["condition"](outputs["condition"], targets["condition_id"])

    subtype_mask = targets["subtype_mask"]
    if subtype_mask.sum() == 0:
        subtype_loss = outputs["disease_subtype"].new_tensor(0.0)
    else:
        per = criterions["disease_subtype"](outputs["disease_subtype"], targets["subtype_id"])
        subtype_loss = (per * subtype_mask).sum() / subtype_mask.sum()

    health_loss = _masked_smooth_l1(outputs["health_score"], targets["health_score"], targets["health_mask"])
    dried_loss = _masked_smooth_l1(outputs["dried_extent"], targets["dried_extent"], targets["extent_mask"])
    decayed_loss = _masked_smooth_l1(
        outputs["decayed_extent"], targets["decayed_extent"], targets["extent_mask"]
    )

    total = (
        w["condition"] * condition_loss
        + w["disease_subtype"] * subtype_loss
        + w["health_score"] * health_loss
        + w["dried_extent"] * dried_loss
        + w["decayed_extent"] * decayed_loss
    )

    parts = {
        "condition": float(condition_loss.detach()),
        "disease_subtype": float(subtype_loss.detach()),
        "health_score": float(health_loss.detach()),
        "dried_extent": float(dried_loss.detach()),
        "decayed_extent": float(decayed_loss.detach()),
        "total": float(total.detach()),
    }
    return total, parts
