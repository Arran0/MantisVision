"""Train the multi-head EfficientNet-B0 health classifier.

Three targets per image (category, condition, health-score anchor — see
config.CLASS_TARGET_MAP) trained jointly:
  - category: 3-way cross-entropy (Healthy/Moderate/Low)
  - condition: 4-way cross-entropy, masked to non-Healthy samples only (the
    condition head is never trained on Healthy examples, since "condition" is
    undefined for a healthy specimen)
  - score: SmoothL1 regression against a jittered anchor value in [0, 10]

Two-phase schedule (unchanged from the single-head model):
  1. Frozen backbone, train only the three heads (fast, stabilizes the new
     heads before touching pretrained weights).
  2. Unfreeze the backbone and fine-tune end-to-end at a lower LR.

Usage:
    python -m src.train
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_TARGET_MAP, config  # noqa: E402
from src.data.dataset import get_multihead_dataloaders  # noqa: E402
from src.models.efficientnet import build_multihead_model, save_checkpoint, unfreeze_backbone  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402


def run_epoch(model, loader, device, category_healthy_idx: int, optimizer=None) -> dict:
    train = optimizer is not None
    model.train() if train else model.eval()

    totals = {
        "loss": 0.0,
        "category_loss": 0.0,
        "condition_loss": 0.0,
        "score_loss": 0.0,
        "category_correct": 0,
        "condition_correct": 0,
        "condition_count": 0,
        "score_abs_error": 0.0,
        "n": 0,
    }
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, category_labels, condition_labels, score_targets in tqdm(loader, leave=False):
            images = images.to(device)
            category_labels = category_labels.to(device)
            condition_labels = condition_labels.to(device)
            score_targets = score_targets.to(device)
            batch_size = images.size(0)

            if train:
                optimizer.zero_grad()

            category_logits, condition_logits, score_pred = model(images)

            category_loss = F.cross_entropy(category_logits, category_labels)

            mask = (category_labels != category_healthy_idx).float()
            condition_loss_raw = F.cross_entropy(condition_logits, condition_labels, reduction="none")
            mask_sum = mask.sum().clamp(min=1.0)
            condition_loss = (condition_loss_raw * mask).sum() / mask_sum

            score_loss = F.smooth_l1_loss(score_pred, score_targets)

            loss = (
                config.loss_weights["category"] * category_loss
                + config.loss_weights["condition"] * condition_loss
                + config.loss_weights["score"] * score_loss
            )

            if train:
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item() * batch_size
            totals["category_loss"] += category_loss.item() * batch_size
            totals["condition_loss"] += condition_loss.item() * batch_size
            totals["score_loss"] += score_loss.item() * batch_size
            totals["category_correct"] += (category_logits.argmax(1) == category_labels).sum().item()
            condition_correct_mask = (condition_logits.argmax(1) == condition_labels).float() * mask
            totals["condition_correct"] += condition_correct_mask.sum().item()
            totals["condition_count"] += mask.sum().item()
            totals["score_abs_error"] += (score_pred - score_targets).abs().sum().item()
            totals["n"] += batch_size

    n = max(totals["n"], 1)
    condition_n = max(totals["condition_count"], 1)
    return {
        "loss": totals["loss"] / n,
        "category_loss": totals["category_loss"] / n,
        "condition_loss": totals["condition_loss"] / n,
        "score_loss": totals["score_loss"] / n,
        "category_acc": totals["category_correct"] / n,
        "condition_acc": totals["condition_correct"] / condition_n,
        "score_mae": totals["score_abs_error"] / n,
    }


def _log_epoch(logger, phase: str, epoch: int, total_epochs: int, train_m: dict, val_m: dict, seconds: float) -> None:
    logger.info(
        "[%s %02d/%02d] "
        "loss=%.4f (cat=%.4f cond=%.4f score=%.4f) "
        "val_loss=%.4f (cat=%.4f cond=%.4f score=%.4f) "
        "cat_acc=%.4f val_cat_acc=%.4f cond_acc=%.4f val_cond_acc=%.4f "
        "score_mae=%.3f val_score_mae=%.3f (%.1fs)",
        phase, epoch, total_epochs,
        train_m["loss"], train_m["category_loss"], train_m["condition_loss"], train_m["score_loss"],
        val_m["loss"], val_m["category_loss"], val_m["condition_loss"], val_m["score_loss"],
        train_m["category_acc"], val_m["category_acc"], train_m["condition_acc"], val_m["condition_acc"],
        train_m["score_mae"], val_m["score_mae"], seconds,
    )


def train() -> None:
    set_seed(config.seed)
    device = get_device(config.device)
    logger = get_logger("train", config.logs_dir)
    logger.info("Using device: %s", device)

    data = get_multihead_dataloaders(config, CLASS_TARGET_MAP)
    logger.info("Categories: %s | Conditions: %s", data.category_names, data.condition_names)
    category_healthy_idx = data.category_names.index("Healthy")

    model = build_multihead_model(
        num_categories=len(data.category_names),
        num_conditions=len(data.condition_names),
        freeze_backbone=True,
    )
    model.to(device)

    config.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_path = config.checkpoints_dir / "best_model.pt"

    def maybe_early_stop(val_loss: float, epoch_label: str) -> bool:
        nonlocal best_val_loss, epochs_without_improvement
        if best_val_loss - val_loss > config.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                data.category_names,
                data.condition_names,
                config.score_min,
                config.score_max,
                best_path,
                extra={"val_loss": val_loss, "epoch": epoch_label, "loss_weights": config.loss_weights},
            )
            logger.info("New best model saved (val_loss=%.4f) -> %s", val_loss, best_path)
        else:
            epochs_without_improvement += 1
            logger.info(
                "No improvement (%d/%d)", epochs_without_improvement, config.early_stopping_patience
            )

        return epochs_without_improvement >= config.early_stopping_patience

    # --- Phase 1: frozen backbone ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.frozen_lr,
        weight_decay=config.weight_decay,
    )
    logger.info("Phase 1: training classifier heads (backbone frozen)")
    for epoch in range(1, config.frozen_epochs + 1):
        start = time.time()
        train_m = run_epoch(model, data.train, device, category_healthy_idx, optimizer=optimizer)
        val_m = run_epoch(model, data.val, device, category_healthy_idx, optimizer=None)
        _log_epoch(logger, "frozen", epoch, config.frozen_epochs, train_m, val_m, time.time() - start)
        if maybe_early_stop(val_m["loss"], f"frozen-{epoch}"):
            logger.info("Early stopping during frozen phase.")
            return

    # --- Phase 2: fine-tune the whole network ---
    unfreeze_backbone(model)
    optimizer = AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
    epochs_without_improvement = 0  # reset patience for the new phase
    logger.info("Phase 2: fine-tuning full network (backbone unfrozen)")
    for epoch in range(1, config.finetune_epochs + 1):
        start = time.time()
        train_m = run_epoch(model, data.train, device, category_healthy_idx, optimizer=optimizer)
        val_m = run_epoch(model, data.val, device, category_healthy_idx, optimizer=None)
        _log_epoch(logger, "finetune", epoch, config.finetune_epochs, train_m, val_m, time.time() - start)
        if maybe_early_stop(val_m["loss"], f"finetune-{epoch}"):
            logger.info("Early stopping during fine-tune phase.")
            return

    logger.info("Training complete. Best val_loss=%.4f at %s", best_val_loss, best_path)


if __name__ == "__main__":
    train()
