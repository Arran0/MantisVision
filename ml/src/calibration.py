"""Confidence calibration utilities: Expected Calibration Error (ECE),
temperature scaling, and reliability diagrams.

This is the concrete answer to "is the reported confidence real or is it just
making things up." The raw softmax value the predictor reports
(`probs[argmax]`) is genuinely computed from the model's logits — it isn't
fabricated — but a model trained with plain cross-entropy is well known to be
*overconfident*: its softmax numbers don't necessarily match its real-world
accuracy. Calibration is what checks (and fixes) that gap.

  - `compute_ece` measures the gap between stated confidence and actual
    accuracy, bucketed into bins.
  - `fit_temperature` finds a single scalar T that rescales logits
    (`logits / T`) to minimize validation NLL — this is temperature scaling
    (Guo et al., 2017), the standard, simplest post-hoc calibration method.
  - `plot_reliability_diagram` renders the same information ECE summarizes,
    as a bar chart against the y=x perfectly-calibrated reference line.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: sum over bins of |accuracy - confidence|,
    weighted by the fraction of samples in each bin.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correct[in_bin].mean()
        ece += (count / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.01,
    min_temperature: float = 0.5,
    max_temperature: float = 10.0,
) -> float:
    """Fits a single scalar temperature T minimizing NLL of softmax(logits/T)
    on the given (validation) logits/labels via LBFGS. Fit on validation, not
    test, to avoid leaking calibration fitting into the held-out evaluation.

    T is bounded to [min_temperature, max_temperature] via a sigmoid
    reparameterization. Without this, unconstrained optimization on a small
    validation split can degenerately drive T -> 0 (which *sharpens* the
    softmax to near-one-hot and makes calibration worse, not better, since
    minimizing NLL is not the same objective as minimizing ECE) rather than
    the expected T > 1 correction for an overconfident network.
    """
    logits = logits.detach()
    labels = labels.detach()

    # Initialize so the starting temperature is T=1.0 (i.e. calibration starts
    # from "no change"), not the sigmoid midpoint.
    start_fraction = (1.0 - min_temperature) / (max_temperature - min_temperature)
    start_fraction = min(max(start_fraction, 1e-4), 1 - 1e-4)
    init_raw = torch.logit(torch.tensor(start_fraction))
    raw_param = init_raw.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([raw_param], lr=lr, max_iter=max_iter)

    def temperature_from_raw() -> torch.Tensor:
        return min_temperature + (max_temperature - min_temperature) * torch.sigmoid(raw_param)

    def closure():
        optimizer.zero_grad()
        temperature = temperature_from_raw()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature_from_raw().item())


def bin_stats(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> list[dict]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    stats = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if lo > 0 else (confidences >= lo) & (confidences <= hi)
        count = int(in_bin.sum())
        stats.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "count": count,
                "confidence": float(confidences[in_bin].mean()) if count else None,
                "accuracy": float(correct[in_bin].mean()) if count else None,
            }
        )
    return stats


def plot_reliability_diagram(confidences: np.ndarray, correct: np.ndarray, n_bins: int, path, title: str) -> None:
    import matplotlib.pyplot as plt

    stats = bin_stats(confidences, correct, n_bins)
    centers = [(s["lo"] + s["hi"]) / 2 for s in stats]
    accuracies = [s["accuracy"] or 0.0 for s in stats]
    width = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.bar(centers, accuracies, width=width * 0.9, edgecolor="black", alpha=0.7, label="Model")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
