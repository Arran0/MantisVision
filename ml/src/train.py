"""Train the EfficientNet-B0 health classifier.

Two-phase schedule:
  1. Frozen backbone, train only the classifier head (fast, stabilizes the
     new head before touching pretrained weights).
  2. Unfreeze the backbone and fine-tune end-to-end at a lower LR.

Usage:
    python -m src.train
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.efficientnet import build_model, save_checkpoint, unfreeze_backbone  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train() -> None:
    set_seed(config.seed)
    device = get_device(config.device)
    logger = get_logger("train", config.logs_dir)
    logger.info("Using device: %s", device)

    data = get_dataloaders(config)
    logger.info("Classes (index order): %s", data.class_names)

    model = build_model(num_classes=len(data.class_names), freeze_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
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
                data.class_names,
                best_path,
                extra={"val_loss": val_loss, "epoch": epoch_label},
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
    logger.info("Phase 1: training classifier head (backbone frozen)")
    for epoch in range(1, config.frozen_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, data.train, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, data.val, criterion, optimizer, device, train=False)
        logger.info(
            "[frozen %02d/%02d] train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f (%.1fs)",
            epoch, config.frozen_epochs, train_loss, train_acc, val_loss, val_acc, time.time() - start,
        )
        if maybe_early_stop(val_loss, f"frozen-{epoch}"):
            logger.info("Early stopping during frozen phase.")
            return

    # --- Phase 2: fine-tune the whole network ---
    unfreeze_backbone(model)
    optimizer = AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=config.weight_decay)
    epochs_without_improvement = 0  # reset patience for the new phase
    logger.info("Phase 2: fine-tuning full network (backbone unfrozen)")
    for epoch in range(1, config.finetune_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, data.train, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, data.val, criterion, optimizer, device, train=False)
        logger.info(
            "[finetune %02d/%02d] train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f (%.1fs)",
            epoch, config.finetune_epochs, train_loss, train_acc, val_loss, val_acc, time.time() - start,
        )
        if maybe_early_stop(val_loss, f"finetune-{epoch}"):
            logger.info("Early stopping during fine-tune phase.")
            return

    logger.info("Training complete. Best val_loss=%.4f at %s", best_val_loss, best_path)


if __name__ == "__main__":
    train()
