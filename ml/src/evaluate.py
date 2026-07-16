"""Evaluate the best multi-head checkpoint on the held-out test split.

Reports, per measurement:
  - classification: accuracy, precision/recall/F1 (macro + per-class),
    confusion matrix image (the schema's primary classification only), and
    one-vs-rest ROC AUC
  - regression: mean absolute error, on the samples where it applies
  - segmentation: mean IoU per mask class, on the samples with a ground-truth
    mask

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
from config import Config, Schema, config as _default_config  # noqa: E402
from src.data.dataset import get_dataloaders  # noqa: E402
from src.models.efficientnet import load_checkpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402


def _mae(pred: list[float], target: list[float], mask: list[float]) -> float | None:
    pairs = [(p, t) for p, t, m in zip(pred, target, mask) if m > 0.5]
    if not pairs:
        return None
    return float(np.mean([abs(p - t) for p, t in pairs]))


def _mean_iou(pred_masks: list[np.ndarray], target_masks: list[np.ndarray], num_classes: int) -> dict[str, float] | None:
    if not pred_masks:
        return None
    ious_per_class: dict[int, list[float]] = {c: [] for c in range(num_classes)}
    for pred, target in zip(pred_masks, target_masks):
        for c in range(num_classes):
            pred_c = pred == c
            target_c = target == c
            union = np.logical_or(pred_c, target_c).sum()
            if union == 0:
                continue
            intersection = np.logical_and(pred_c, target_c).sum()
            ious_per_class[c].append(float(intersection) / float(union))
    return {str(c): (float(np.mean(v)) if v else None) for c, v in ious_per_class.items()}


def evaluate(checkpoint_path: Path | None = None, cfg: Config | None = None) -> dict:
    cfg = cfg if cfg is not None else _default_config
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    logger = get_logger("evaluate", cfg.logs_dir)

    checkpoint_path = checkpoint_path or (cfg.checkpoints_dir / "best_model.pt")
    model, schema = load_checkpoint(checkpoint_path, device)
    logger.info("Loaded checkpoint %s (measurements=%s)", checkpoint_path, [m.key for m in schema.measurements])

    data = get_dataloaders(cfg, schema)

    results: dict = {}

    primary = schema.primary_classification()

    with torch.no_grad():
        per_measurement_state: dict[str, dict] = {
            m.key: (
                {"labels": [], "preds": [], "probs": []}
                if m.type == "classification"
                else {"pred": [], "target": [], "mask": []}
                if m.type == "regression"
                else {"pred_masks": [], "target_masks": []}
            )
            for m in schema.measurements
        }

        for images, targets in data.test:
            images = images.to(device)
            outputs = model(images)

            for m in schema.measurements:
                state = per_measurement_state[m.key]
                if m.type == "classification":
                    mask = targets[f"{m.key}_mask"].numpy()
                    probs = F.softmax(outputs[m.key], dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    ids = targets[f"{m.key}_id"].numpy()
                    for keep, p, t, prob in zip(mask > 0.5, preds, ids, probs):
                        if keep:
                            state["preds"].append(int(p))
                            state["labels"].append(int(t))
                            state["probs"].append(prob.tolist())
                elif m.type == "regression":
                    state["pred"].extend(outputs[m.key].cpu().numpy().tolist())
                    state["target"].extend(targets[m.key].numpy().tolist())
                    state["mask"].extend(targets[f"{m.key}_mask"].numpy().tolist())
                elif m.type == "segmentation":
                    seg_mask = targets[f"{m.key}_seg_mask"].numpy()
                    pred_classes = outputs[m.key].argmax(dim=1).cpu().numpy()
                    target_classes = targets[f"{m.key}_seg"].numpy()
                    for keep, p, t in zip(seg_mask > 0.5, pred_classes, target_classes):
                        if keep:
                            state["pred_masks"].append(p)
                            state["target_masks"].append(t)

    for m in schema.measurements:
        state = per_measurement_state[m.key]

        if m.type == "classification":
            class_names = m.class_names()
            if not state["labels"]:
                results[m.key] = None
                continue
            labels_arr = np.array(state["labels"])
            preds_arr = np.array(state["preds"])
            probs_arr = np.array(state["probs"])

            report = classification_report(
                labels_arr, preds_arr, labels=list(range(len(class_names))),
                target_names=class_names, output_dict=True, zero_division=0,
            )
            logger.info(
                "[%s]\n%s",
                m.key,
                classification_report(labels_arr, preds_arr, labels=list(range(len(class_names))), target_names=class_names, zero_division=0),
            )

            cm = confusion_matrix(labels_arr, preds_arr, labels=list(range(len(class_names))))
            per_class_accuracy = {
                name: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0 for i, name in enumerate(class_names)
            }

            roc_auc: dict[str, float | None] = {}
            try:
                auc_scores = roc_auc_score(
                    labels_arr, probs_arr, multi_class="ovr", average=None, labels=list(range(len(class_names)))
                )
                # A class with only one label present in this test split (0 or
                # all support) makes its one-vs-rest AUC undefined — sklearn
                # returns NaN for that entry (with an UndefinedMetricWarning)
                # rather than raising, unlike the all-classes-missing case
                # below. NaN must not reach json.dumps: Python serializes it
                # as the bare token `NaN`, which PostgREST rejects as invalid
                # JSON (opaque HTTP 400), so map it to None/null here.
                roc_auc = {
                    class_names[i]: (float(score) if not np.isnan(score) else None)
                    for i, score in enumerate(auc_scores)
                }
            except ValueError as e:
                logger.warning("[%s] Could not compute ROC AUC (likely a class missing from the test split): %s", m.key, e)

            cm_path = None
            if m is primary:
                # Only the primary classification gets a confusion-matrix
                # image — the others (e.g. disease_subtype) are reported as
                # plain classification_report dicts.
                cfg.reports_dir.mkdir(parents=True, exist_ok=True)
                fig, ax = plt.subplots(figsize=(8, 7))
                ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(
                    ax=ax, cmap="Blues", xticks_rotation=45
                )
                plt.tight_layout()
                cm_path = cfg.reports_dir / "confusion_matrix.png"
                fig.savefig(cm_path, dpi=150)
                plt.close(fig)

            results[m.key] = {
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
                "confusion_matrix_path": str(cm_path) if cm_path else None,
            }

        elif m.type == "regression":
            results[m.key] = {"mae": _mae(state["pred"], state["target"], state["mask"])}

        elif m.type == "segmentation":
            results[m.key] = {"mean_iou_per_class": _mean_iou(state["pred_masks"], state["target_masks"], len(m.seg_classes))}

    results_path = cfg.reports_dir / "evaluation_results.json"
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved evaluation results -> %s", results_path)

    return results


if __name__ == "__main__":
    evaluate()
