"""Train the schema-driven multi-head EfficientNet-B0 seaweed model.

Two-phase schedule:
  1. Frozen backbone, train only the heads (fast, stabilizes the new heads
     before touching pretrained weights).
  2. Unfreeze the backbone and fine-tune end-to-end at a lower LR.

The loss is multi-task (see src/losses.py) and driven entirely by the active
Schema: one term per measurement (classification/regression/segmentation),
each masked to the samples where it applies and has a value. Adding a new
measurement to the schema needs no change here — it just becomes one more
term in the sum, at 0 loss until labeled data exists for it.

Label-noise robustness: label smoothing on the primary (background-carrying)
classification measurement plus heavy augmentation (blur/brightness/noise,
see src/data/transforms.py) is the first-line defense. For heavier noise,
swap that measurement's criterion in src/losses.py for a Generalized
Cross-Entropy / symmetric loss — the rest of the pipeline is unaffected.

Usage:
    python -m src.train
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SCHEMA as _default_schema  # noqa: E402
from config import Config, Schema, config as _default_config  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.losses import build_criterions, compute_losses  # noqa: E402
from src.models.efficientnet import build_model, save_checkpoint, unfreeze_backbone  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402


def _to_device(targets: dict, device) -> dict:
    return {key: value.to(device) for key, value in targets.items()}


def run_epoch(model, loader, schema: Schema, criterions, cfg: Config, optimizer, device, train: bool) -> tuple[float, float]:
    model.train() if train else model.eval()

    primary = schema.primary_classification()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device)
            targets = _to_device(targets, device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss, _ = compute_losses(outputs, targets, schema, criterions, cfg)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            # Track the primary classification's accuracy as the
            # human-readable progress signal (skipped if the schema declares
            # no such measurement).
            if primary is not None:
                correct += (outputs[primary.key].argmax(1) == targets[f"{primary.key}_id"]).sum().item()
            total += batch_size

    accuracy = correct / total if primary is not None else 0.0
    return total_loss / total, accuracy


def train(cfg: Config | None = None, schema: Schema | None = None) -> None:
    """cfg/schema default to the process-wide config.config / config.SCHEMA
    (what every real invocation uses); both are overridable so tests and
    tooling can point at a synthetic Config/Schema without touching global
    state."""
    cfg = cfg if cfg is not None else _default_config
    schema = schema if schema is not None else _default_schema

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = get_logger("train", cfg.logs_dir)
    logger.info("Using device: %s", device)
    logger.info("Measurements: %s", [m.key for m in schema.measurements])

    data = get_dataloaders(cfg, schema)

    model = build_model(schema, freeze_backbone=True)
    model.to(device)

    criterions = build_criterions(schema, cfg)
    cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_path = cfg.checkpoints_dir / "best_model.pt"

    def maybe_early_stop(val_loss: float, epoch_label: str) -> bool:
        nonlocal best_val_loss, epochs_without_improvement
        if best_val_loss - val_loss > cfg.early_stopping_min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, schema, best_path, extra={"val_loss": val_loss, "epoch": epoch_label})
            logger.info("New best model saved (val_loss=%.4f) -> %s", val_loss, best_path)
        else:
            epochs_without_improvement += 1
            logger.info("No improvement (%d/%d)", epochs_without_improvement, cfg.early_stopping_patience)
        return epochs_without_improvement >= cfg.early_stopping_patience

    # --- Phase 1: frozen backbone ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.frozen_lr,
        weight_decay=cfg.weight_decay,
    )
    logger.info("Phase 1: training heads (backbone frozen)")
    for epoch in range(1, cfg.frozen_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, data.train, schema, criterions, cfg, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, data.val, schema, criterions, cfg, optimizer, device, train=False)
        logger.info(
            "[frozen %02d/%02d] train_loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f (%.1fs)",
            epoch, cfg.frozen_epochs, train_loss, train_acc, val_loss, val_acc, time.time() - start,
        )
        if maybe_early_stop(val_loss, f"frozen-{epoch}"):
            logger.info("Early stopping during frozen phase.")
            return

    # --- Phase 2: fine-tune the whole network ---
    unfreeze_backbone(model)
    optimizer = AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    epochs_without_improvement = 0  # reset patience for the new phase
    logger.info("Phase 2: fine-tuning full network (backbone unfrozen)")
    for epoch in range(1, cfg.finetune_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, data.train, schema, criterions, cfg, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, data.val, schema, criterions, cfg, optimizer, device, train=False)
        logger.info(
            "[finetune %02d/%02d] train_loss=%.4f acc=%.4f val_loss=%.4f val_acc=%.4f (%.1fs)",
            epoch, cfg.finetune_epochs, train_loss, train_acc, val_loss, val_acc, time.time() - start,
        )
        if maybe_early_stop(val_loss, f"finetune-{epoch}"):
            logger.info("Early stopping during fine-tune phase.")
            return

    logger.info("Training complete. Best val_loss=%.4f at %s", best_val_loss, best_path)


if __name__ == "__main__":
    train()
