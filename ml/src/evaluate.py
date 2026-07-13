"""Evaluate the best multi-head checkpoint on the held-out test split.

Reports, per head:
  - condition: accuracy, precision/recall/F1 (macro + per-class), confusion
    matrix image, one-vs-rest ROC AUC (incl. the Background class)
  - disease_subtype: classification report on the Disease subset only
  - health_score / dried_extent / decayed_extent: mean absolute error (0-100)

Per spec: never judge the model on accuracy alone.

Usage:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.efficientnet import load_checkpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402


def _mae(pred: list[float], target: list[float], mask: list[float]) -> float | None:
    pairs = [(p, t) for p, t, m in zip(pred, target, mask) if m > 0.5]
    if not pairs:
        return None
    return float(np.mean([abs(p - t) for p, t in pairs]))


def evaluate(checkpoint_path: Path | None = None) -> dict:
    set_seed(config.seed)
    device = get_device(config.device)
    logger = get_logger("evaluate", config.logs_dir)

    checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
    model, condition_classes, subtype_classes, _species = load_checkpoint(checkpoint_path, device)
    logger.info("Loaded checkpoint %s (conditions=%s)", checkpoint_path, condition_classes)

    data = get_dataloaders(config)
    assert data.condition_classes == condition_classes, "Checkpoint condition order != dataset."

    cond_labels, cond_preds, cond_probs = [], [], []
    sub_labels, sub_preds = [], []
    reg = {name: {"pred": [], "target": [], "mask": []} for name in ("health_score", "dried_extent", "decayed_extent")}

    with torch.no_grad():
        for images, targets in data.test:
            images = images.to(device)
            outputs = model(images)

            probs = F.softmax(outputs["condition"], dim=1).cpu().numpy()
            cond_probs.extend(probs.tolist())
            cond_preds.extend(probs.argmax(axis=1).tolist())
            cond_labels.extend(targets["condition_id"].numpy().tolist())

            sub_mask = targets["subtype_mask"].numpy()
            sub_pred_batch = outputs["disease_subtype"].argmax(1).cpu().numpy()
            sub_target_batch = targets["subtype_id"].numpy()
            for m, p, t in zip(sub_mask, sub_pred_batch, sub_target_batch):
                if m > 0.5:
                    sub_preds.append(int(p))
                    sub_labels.append(int(t))

            for name in reg:
                reg[name]["pred"].extend(outputs[name].cpu().numpy().tolist())
                reg[name]["target"].extend(targets[name].numpy().tolist())
            reg["health_score"]["mask"].extend(targets["health_mask"].numpy().tolist())
            reg["dried_extent"]["mask"].extend(targets["extent_mask"].numpy().tolist())
            reg["decayed_extent"]["mask"].extend(targets["extent_mask"].numpy().tolist())

    cond_labels_arr = np.array(cond_labels)
    cond_preds_arr = np.array(cond_preds)
    cond_probs_arr = np.array(cond_probs)

    report = classification_report(
        cond_labels_arr, cond_preds_arr, target_names=condition_classes, output_dict=True, zero_division=0
    )
    logger.info(
        "\n%s",
        classification_report(cond_labels_arr, cond_preds_arr, target_names=condition_classes, zero_division=0),
    )

    cm = confusion_matrix(cond_labels_arr, cond_preds_arr, labels=list(range(len(condition_classes))))
    per_class_accuracy = {
        name: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0
        for i, name in enumerate(condition_classes)
    }

    roc_auc: dict[str, float] = {}
    try:
        auc_scores = roc_auc_score(
            cond_labels_arr, cond_probs_arr, multi_class="ovr", average=None, labels=list(range(len(condition_classes)))
        )
        roc_auc = {condition_classes[i]: float(score) for i, score in enumerate(auc_scores)}
    except ValueError as e:
        logger.warning("Could not compute ROC AUC (likely a class missing from the test split): %s", e)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=condition_classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45
    )
    plt.tight_layout()
    cm_path = config.reports_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    subtype_report = None
    if sub_labels:
        subtype_report = classification_report(
            np.array(sub_labels), np.array(sub_preds), labels=list(range(len(subtype_classes))),
            target_names=subtype_classes, output_dict=True, zero_division=0,
        )

    results = {
        "condition": {
            "accuracy": report["accuracy"],
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "per_class": {
                name: {
                    "precision": report[name]["precision"],
                    "recall": report[name]["recall"],
                    "f1_score": report[name]["f1-score"],
                    "support": report[name]["support"],
                    "accuracy": per_class_accuracy[name],
                    "roc_auc": roc_auc.get(name),
                }
                for name in condition_classes
            },
            "confusion_matrix_path": str(cm_path),
        },
        "disease_subtype": subtype_report,
        "regression_mae": {
            name: _mae(reg[name]["pred"], reg[name]["target"], reg[name]["mask"]) for name in reg
        },
    }

    results_path = config.reports_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved evaluation results -> %s", results_path)
    logger.info("Saved confusion matrix -> %s", cm_path)

    return results


if __name__ == "__main__":
    evaluate()
