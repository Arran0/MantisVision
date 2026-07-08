"""Evaluate the best checkpoint on the held-out test split.

Reports metrics per head, per spec: never judge the model on accuracy alone.
  - category: accuracy/precision/recall/F1 (macro + per-class), confusion
    matrix, one-vs-rest ROC AUC — same shape as the old single-head report,
    just 3-way instead of 6-way.
  - condition: same metrics, computed only on test samples whose category is
    not Healthy (the condition head is never trained on Healthy samples —
    see src/train.py — so including them would trivially inflate accuracy
    with an "always None" shortcut).
  - disease_subtype: same metrics, computed only on samples whose condition
    is "Diseased" (undefined otherwise, masked at training time too).
  - score: MAE/RMSE/R^2 against the heuristic anchor targets, plus a
    scatter plot and a per-raw-class strip plot. IMPORTANT CAVEAT: these
    numbers measure agreement with config.resolve_class_target's anchor
    values, NOT real biological ground truth — no expert-scored 0-10 dataset
    exists yet (see config.py's anchor justification comments).
  - extent (dried%/decayed%): same caveat — MAE/RMSE against heuristic
    anchors, not pixel-level/segmentation ground truth.
  - calibration: raw (uncalibrated) ECE is always reported. If
    `checkpoints/calibration.json` exists (produced by `python -m
    src.calibrate`), the calibrated ECE is reported alongside it.

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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from src.calibration import compute_ece  # noqa: E402
from src.data.dataset import get_multihead_dataloaders  # noqa: E402
from src.models.efficientnet import load_checkpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402


def _head_report(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray, names: list[str]) -> dict:
    report = classification_report(labels, preds, target_names=names, output_dict=True, zero_division=0, labels=list(range(len(names))))
    cm = confusion_matrix(labels, preds, labels=list(range(len(names))))
    per_class_accuracy = {}
    for i, name in enumerate(names):
        support = cm[i].sum()
        per_class_accuracy[name] = float(cm[i, i] / support) if support else 0.0

    roc_auc = {}
    try:
        auc_scores = roc_auc_score(labels, probs, multi_class="ovr", average=None, labels=list(range(len(names))))
        roc_auc = {names[i]: float(score) for i, score in enumerate(auc_scores)}
    except ValueError as e:
        roc_auc = {"error": str(e)}

    return {
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
            for name in names
        },
        "confusion_matrix": cm.tolist(),
    }


def _save_confusion_matrix(cm: list, names: list[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=names).plot(ax=ax, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(checkpoint_path: Path | None = None) -> dict:
    set_seed(config.seed)
    device = get_device(config.device)
    logger = get_logger("evaluate", config.logs_dir)

    checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
    model, category_names, condition_names, disease_subtype_names = load_checkpoint(checkpoint_path, device)
    logger.info(
        "Loaded checkpoint %s (categories=%s, conditions=%s, disease_subtypes=%s)",
        checkpoint_path, category_names, condition_names, disease_subtype_names,
    )

    data = get_multihead_dataloaders(config)
    assert data.category_names == category_names, "Checkpoint category order does not match dataset."
    assert data.condition_names == condition_names, "Checkpoint condition order does not match dataset."
    assert data.disease_subtype_names == disease_subtype_names, "Checkpoint disease-subtype order does not match dataset."
    healthy_idx = category_names.index("Healthy")
    diseased_idx = condition_names.index("Diseased")

    cat_labels, cat_preds, cat_probs = [], [], []
    cond_labels, cond_preds, cond_probs = [], [], []
    subtype_labels, subtype_preds, subtype_probs = [], [], []
    score_targets_all, score_preds_all, raw_class_all = [], [], []
    extent_targets_all, extent_preds_all = [], []

    # raw_class per sample is recovered from the underlying ImageFolder so the
    # score scatter/strip plots can be grouped by the original raw folders.
    raw_classes = data.test.dataset.image_folder.classes

    with torch.no_grad():
        idx = 0
        for images, targets in data.test:
            images = images.to(device)
            category_batch = targets["category"]
            condition_batch = targets["condition"]
            score_batch = targets["score"]
            subtype_batch = targets["disease_subtype"]
            extent_batch = targets["extent"]

            category_logits, condition_logits, score_pred, subtype_logits, extent_pred = model(images)
            cat_p = F.softmax(category_logits, dim=1).cpu().numpy()
            cond_p = F.softmax(condition_logits, dim=1).cpu().numpy()
            subtype_p = F.softmax(subtype_logits, dim=1).cpu().numpy()

            cat_labels.extend(category_batch.numpy().tolist())
            cat_preds.extend(cat_p.argmax(axis=1).tolist())
            cat_probs.extend(cat_p.tolist())

            cond_mask = (category_batch != healthy_idx).numpy()
            if cond_mask.any():
                cond_labels.extend(condition_batch.numpy()[cond_mask].tolist())
                cond_preds.extend(cond_p.argmax(axis=1)[cond_mask].tolist())
                cond_probs.extend(cond_p[cond_mask].tolist())

            subtype_mask = (condition_batch == diseased_idx).numpy()
            if subtype_mask.any():
                subtype_labels.extend(subtype_batch.numpy()[subtype_mask].tolist())
                subtype_preds.extend(subtype_p.argmax(axis=1)[subtype_mask].tolist())
                subtype_probs.extend(subtype_p[subtype_mask].tolist())

            score_targets_all.extend(score_batch.numpy().tolist())
            score_preds_all.extend(score_pred.cpu().numpy().tolist())
            extent_targets_all.extend(extent_batch.numpy().tolist())
            extent_preds_all.extend(extent_pred.cpu().numpy().tolist())

            batch_size = images.size(0)
            raw_class_all.extend(
                raw_classes[data.test.dataset.image_folder.samples[i][1]] for i in range(idx, idx + batch_size)
            )
            idx += batch_size

    category_result = _head_report(np.array(cat_labels), np.array(cat_preds), np.array(cat_probs), category_names)
    condition_result = (
        _head_report(np.array(cond_labels), np.array(cond_preds), np.array(cond_probs), condition_names)
        if cond_labels
        else {"note": "No non-Healthy samples in test split."}
    )
    subtype_result = (
        _head_report(np.array(subtype_labels), np.array(subtype_preds), np.array(subtype_probs), disease_subtype_names)
        if subtype_labels
        else {"note": "No Diseased-condition samples in test split."}
    )

    score_targets_arr = np.array(score_targets_all)
    score_preds_arr = np.array(score_preds_all)
    score_result = {
        "mae": float(mean_absolute_error(score_targets_arr, score_preds_arr)),
        "rmse": float(mean_squared_error(score_targets_arr, score_preds_arr) ** 0.5),
        "r2": float(r2_score(score_targets_arr, score_preds_arr)) if len(score_targets_arr) > 1 else None,
        "caveat": (
            "These metrics measure agreement with config.resolve_class_target's heuristic "
            "anchor values, NOT real biological ground truth -- no expert-scored 0-10 "
            "dataset exists yet."
        ),
    }

    extent_targets_arr = np.array(extent_targets_all)  # [N, 2] = (dried_pct, decayed_pct)
    extent_preds_arr = np.array(extent_preds_all)
    extent_result = {
        "dried_pct": {
            "mae": float(mean_absolute_error(extent_targets_arr[:, 0], extent_preds_arr[:, 0])),
            "rmse": float(mean_squared_error(extent_targets_arr[:, 0], extent_preds_arr[:, 0]) ** 0.5),
        },
        "decayed_pct": {
            "mae": float(mean_absolute_error(extent_targets_arr[:, 1], extent_preds_arr[:, 1])),
            "rmse": float(mean_squared_error(extent_targets_arr[:, 1], extent_preds_arr[:, 1]) ** 0.5),
        },
        "caveat": (
            "These metrics measure agreement with heuristic per-class dried%/decayed% anchors, "
            "NOT pixel-level/segmentation ground truth -- no such dataset exists yet."
        ),
    }

    # Raw ECE (uncalibrated, T=1) always reported.
    cat_confidences = np.array(cat_probs).max(axis=1)
    cat_correct = (np.array(cat_preds) == np.array(cat_labels)).astype(float)
    ece_raw = compute_ece(cat_confidences, cat_correct)
    calibration_result = {"ece_raw": ece_raw}

    calibration_path = config.checkpoints_dir / "calibration.json"
    if calibration_path.exists():
        calibration_data = json.loads(calibration_path.read_text())
        calibration_result["ece_calibrated"] = calibration_data.get("ece_after")
        calibration_result["temperature"] = calibration_data.get("temperature")
    else:
        calibration_result["note"] = "Run `python -m src.calibrate` to fit temperature scaling and report calibrated ECE."

    logger.info("Category accuracy: %.4f | ECE (raw): %.4f", category_result["accuracy"], ece_raw)
    if "accuracy" in condition_result:
        logger.info("Condition accuracy (non-Healthy only): %.4f", condition_result["accuracy"])
    if "accuracy" in subtype_result:
        logger.info("Disease subtype accuracy (Diseased only): %.4f", subtype_result["accuracy"])
    logger.info("Score MAE: %.3f RMSE: %.3f", score_result["mae"], score_result["rmse"])
    logger.info(
        "Extent MAE -- dried: %.2f decayed: %.2f",
        extent_result["dried_pct"]["mae"], extent_result["decayed_pct"]["mae"],
    )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    _save_confusion_matrix(category_result["confusion_matrix"], category_names, config.reports_dir / "confusion_matrix_category.png", "Category confusion matrix")
    if "confusion_matrix" in condition_result:
        _save_confusion_matrix(condition_result["confusion_matrix"], condition_names, config.reports_dir / "confusion_matrix_condition.png", "Condition confusion matrix (non-Healthy only)")
    if "confusion_matrix" in subtype_result:
        _save_confusion_matrix(subtype_result["confusion_matrix"], disease_subtype_names, config.reports_dir / "confusion_matrix_disease_subtype.png", "Disease subtype confusion matrix (Diseased only)")

    # Score scatter: predicted vs. target.
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(score_targets_arr, score_preds_arr, alpha=0.5, s=15)
    ax.plot([config.score_min, config.score_max], [config.score_min, config.score_max], linestyle="--", color="gray")
    ax.set_xlabel("Anchor target score")
    ax.set_ylabel("Predicted score")
    ax.set_title("Health score: predicted vs. anchor target")
    fig.tight_layout()
    fig.savefig(config.reports_dir / "score_scatter.png", dpi=150)
    plt.close(fig)

    # Extent scatter: predicted vs. target, one subplot per dried/decayed.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, (ax, label) in enumerate(zip(axes, ["Dried %", "Decayed %"])):
        ax.scatter(extent_targets_arr[:, i], extent_preds_arr[:, i], alpha=0.5, s=15)
        ax.plot([config.pct_min, config.pct_max], [config.pct_min, config.pct_max], linestyle="--", color="gray")
        ax.set_xlabel(f"Anchor target {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(config.reports_dir / "extent_scatter.png", dpi=150)
    plt.close(fig)

    # Score strip plot grouped by original raw class (discovered dynamically —
    # Disease_* folders vary in number/name), to visually confirm the model
    # produces a graded spread within each folder rather than constant
    # clusters (verifies the anchor-jitter technique worked).
    raw_class_order = sorted(set(raw_class_all))
    fig, ax = plt.subplots(figsize=(max(8, len(raw_class_order) * 1.1), 5))
    for i, raw_name in enumerate(raw_class_order):
        ys = [p for p, c in zip(score_preds_all, raw_class_all) if c == raw_name]
        xs = np.random.normal(i, 0.05, size=len(ys))
        ax.scatter(xs, ys, alpha=0.5, s=15)
    ax.set_xticks(range(len(raw_class_order)))
    ax.set_xticklabels(raw_class_order, rotation=45, ha="right")
    ax.set_ylabel("Predicted score")
    ax.set_title("Predicted score spread by raw dataset folder")
    fig.tight_layout()
    fig.savefig(config.reports_dir / "score_by_raw_class.png", dpi=150)
    plt.close(fig)

    results = {
        "category": category_result,
        "condition": condition_result,
        "disease_subtype": subtype_result,
        "score": score_result,
        "extent": extent_result,
        "calibration": calibration_result,
    }

    results_path = config.reports_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved evaluation results -> %s", results_path)

    return results


if __name__ == "__main__":
    evaluate()
