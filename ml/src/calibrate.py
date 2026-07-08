"""Fit and check confidence calibration for the category head.

Fits a temperature scalar on the validation split, then reports Expected
Calibration Error (ECE) on the held-out test split both before (T=1, raw
softmax) and after calibration, saving reliability diagrams for both and a
`calibration.json` the inference predictor picks up automatically.

This is the repeatable check for "is the reported confidence trustworthy" —
re-run this after every retrain.

Usage:
    python -m src.calibrate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLASS_TARGET_MAP, config  # noqa: E402
from src.calibration import compute_ece, fit_temperature, plot_reliability_diagram  # noqa: E402
from src.data.dataset import get_multihead_dataloaders  # noqa: E402
from src.models.efficientnet import load_checkpoint  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.seed import get_device, set_seed  # noqa: E402

N_BINS = 15


def _collect_category_logits(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, category_labels, _condition_labels, _score_targets in loader:
            images = images.to(device)
            category_logits, _, _ = model(images)
            all_logits.append(category_logits.cpu())
            all_labels.append(category_labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def calibrate(checkpoint_path: Path | None = None) -> dict:
    set_seed(config.seed)
    device = get_device(config.device)
    logger = get_logger("calibrate", config.logs_dir)

    checkpoint_path = checkpoint_path or (config.checkpoints_dir / "best_model.pt")
    model, category_names, condition_names = load_checkpoint(checkpoint_path, device)
    logger.info("Loaded checkpoint %s (categories=%s)", checkpoint_path, category_names)

    data = get_multihead_dataloaders(config, CLASS_TARGET_MAP)
    assert data.category_names == category_names, "Checkpoint category order does not match dataset."

    val_logits, val_labels = _collect_category_logits(model, data.val, device)
    test_logits, test_labels = _collect_category_logits(model, data.test, device)

    temperature = fit_temperature(val_logits, val_labels)
    logger.info("Fitted temperature T=%.4f on the validation split", temperature)

    def ece_for(logits: torch.Tensor, labels: torch.Tensor, temp: float) -> tuple[float, np.ndarray, np.ndarray]:
        probs = F.softmax(logits / temp, dim=1)
        confidences, preds = probs.max(dim=1)
        correct = (preds == labels).numpy().astype(float)
        confidences = confidences.numpy()
        return compute_ece(confidences, correct, n_bins=N_BINS), confidences, correct

    ece_before, conf_before, correct_before = ece_for(test_logits, test_labels, 1.0)
    ece_after, conf_after, correct_after = ece_for(test_logits, test_labels, temperature)

    logger.info("Test-split ECE before calibration (T=1.0): %.4f", ece_before)
    logger.info("Test-split ECE after calibration (T=%.4f): %.4f", temperature, ece_after)

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    before_path = config.reports_dir / "reliability_diagram_before.png"
    after_path = config.reports_dir / "reliability_diagram_after.png"
    plot_reliability_diagram(conf_before, correct_before, N_BINS, before_path, "Reliability — before calibration (T=1.0)")
    plot_reliability_diagram(conf_after, correct_after, N_BINS, after_path, f"Reliability — after calibration (T={temperature:.3f})")

    result = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "fit_split": "validation",
        "eval_split": "test",
        "n_bins": N_BINS,
        "reliability_diagram_before": str(before_path),
        "reliability_diagram_after": str(after_path),
    }

    calibration_path = config.checkpoints_dir / "calibration.json"
    calibration_path.write_text(json.dumps(result, indent=2))
    logger.info("Saved calibration -> %s", calibration_path)

    return result


if __name__ == "__main__":
    calibrate()
