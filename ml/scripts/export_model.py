"""Export the trained checkpoint to ONNX (Milestone 7 / future TFLite/CoreML
conversion from there). ONNX is the portable intermediate format that both
onnxruntime-web (browser/PWA) and mobile export toolchains can consume.

Usage:
    python scripts/export_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from src.models.efficientnet import load_checkpoint  # noqa: E402
from src.utils.seed import get_device  # noqa: E402


def main() -> None:
    device = get_device(config.device)
    checkpoint_path = config.checkpoints_dir / "best_model.pt"
    model, category_names, condition_names = load_checkpoint(checkpoint_path, device)

    dummy_input = torch.randn(1, 3, config.image_size, config.image_size, device=device)
    out_path = config.checkpoints_dir / "health_classifier.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        input_names=["image"],
        output_names=["category_logits", "condition_logits", "health_score"],
        dynamic_axes={
            "image": {0: "batch"},
            "category_logits": {0: "batch"},
            "condition_logits": {0: "batch"},
            "health_score": {0: "batch"},
        },
        opset_version=17,
    )

    labels_path = config.checkpoints_dir / "labels.json"
    labels_path.write_text(json.dumps({"category_names": category_names, "condition_names": condition_names}, indent=2))

    print(f"Exported ONNX model -> {out_path}")
    print(f"Saved category/condition names -> {labels_path}")


if __name__ == "__main__":
    main()
