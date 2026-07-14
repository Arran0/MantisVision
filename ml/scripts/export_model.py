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
from config import config, schema_to_dict  # noqa: E402
from src.models.efficientnet import load_checkpoint  # noqa: E402
from src.utils.seed import get_device  # noqa: E402


class _TupleOutputWrapper(torch.nn.Module):
    """The multi-head model's forward returns a dict; ONNX needs ordered
    tensor outputs, so we export through a wrapper that returns a fixed
    tuple in schema order."""

    def __init__(self, model: torch.nn.Module, output_names: list[str]) -> None:
        super().__init__()
        self.model = model
        self.output_names = output_names

    def forward(self, x):
        out = self.model(x)
        return tuple(out[name] for name in self.output_names)


def main() -> None:
    device = get_device(config.device)
    checkpoint_path = config.checkpoints_dir / "best_model.pt"
    model, schema = load_checkpoint(checkpoint_path, device)
    output_names = [m.key for m in schema.measurements]

    dummy_input = torch.randn(1, 3, config.image_size, config.image_size, device=device)
    out_path = config.checkpoints_dir / "seaweed_multihead.onnx"

    # A freshly constructed wrapper module starts in training=True regardless
    # of the inner model's own mode (already eval() from load_checkpoint), and
    # torch.onnx.export checks the top-level module it's given — so .eval()
    # the wrapper explicitly, or dropout/BatchNorm run in training mode during
    # export.
    wrapper = _TupleOutputWrapper(model, output_names).eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes={"image": {0: "batch"}, **{name: {0: "batch"} for name in output_names}},
        opset_version=18,
    )

    # The full schema — not just class name lists — is what a downstream
    # consumer (browser/mobile inference code) needs to interpret each
    # output tensor (which measurement it is, its type, its classes/range).
    labels_path = config.checkpoints_dir / "class_names.json"
    labels_path.write_text(json.dumps(schema_to_dict(schema), indent=2))

    print(f"Exported ONNX model -> {out_path}")
    print(f"Saved schema metadata -> {labels_path}")


if __name__ == "__main__":
    main()
