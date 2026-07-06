"""Evaluate the best checkpoint on the held-out test split.

Produces accuracy, precision, recall, F1 (macro + per-class), a confusion
matrix image, and one-vs-rest ROC AUC per class. Per spec: never judge the
model on accuracy alone.

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


def evaluate(checkpoint_path: Path | None = None) -> dict:
    set_seed(config.seed)
    device = get_device(config.device)
    logger = get_logger("evaluate", config.logs_dir)

    checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
    model, class_names = load_checkpoint(checkpoint_path, device)
    logger.info("Loaded checkpoint %s (classes=%s)", checkpoint_path, class_names)

    data = get_dataloaders(config)
    assert data.class_names == class_names, "Checkpoint class order does not match dataset."

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in data.test:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    all_probs_arr = np.array(all_probs)

    report = classification_report(
        all_labels_arr, all_preds_arr, target_names=class_names, output_dict=True, zero_division=0
    )
    logger.info(
        "\n%s",
        classification_report(all_labels_arr, all_preds_arr, target_names=class_names, zero_division=0),
    )

    per_class_accuracy = {}
    cm = confusion_matrix(all_labels_arr, all_preds_arr)
    for i, class_name in enumerate(class_names):
        support = cm[i].sum()
        per_class_accuracy[class_name] = float(cm[i, i] / support) if support else 0.0

    roc_auc = {}
    try:
        auc_scores = roc_auc_score(
            all_labels_arr, all_probs_arr, multi_class="ovr", average=None, labels=list(range(len(class_names)))
        )
        roc_auc = {class_names[i]: float(score) for i, score in enumerate(auc_scores)}
    except ValueError as e:
        logger.warning("Could not compute ROC AUC (likely a class missing from the test split): %s", e)

    config.reports_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45
    )
    plt.tight_layout()
    cm_path = config.reports_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    results = {
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
            for name in class_names
        },
        "confusion_matrix_path": str(cm_path),
    }

    results_path = config.reports_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved evaluation results -> %s", results_path)
    logger.info("Saved confusion matrix -> %s", cm_path)

    return results


if __name__ == "__main__":
    evaluate()
