"""Grad-CAM explainability for the health classifier.

Highlights the image regions that most influenced the category prediction so
a farmer or reviewer can sanity-check *why* the model said "Low" instead of
just trusting the label. Used both as a standalone CLI and inside the
inference API (src/api/main.py).

Usage:
    python -m src.gradcam path/to/image.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config  # noqa: E402
from src.data.transforms import build_transforms  # noqa: E402
from src.models.efficientnet import MultiHeadEfficientNet, last_conv_layer, load_checkpoint  # noqa: E402
from src.utils.seed import get_device  # noqa: E402


class _CategoryHeadOnly(nn.Module):
    """pytorch_grad_cam expects a model returning a single logits tensor;
    the multi-head model returns (category, condition, score). This wraps it
    to expose only the category head, which is the one the heatmap explains.
    """

    def __init__(self, model: MultiHeadEfficientNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        category_logits, _condition_logits, _score = self.model(x)
        return category_logits


def generate_gradcam(
    model: MultiHeadEfficientNet,
    image: Image.Image,
    class_index: int,
    device: torch.device,
) -> np.ndarray:
    """Returns an RGB uint8 heatmap-overlaid image (H, W, 3), values 0-255."""
    # Imported here (not at module load) so the API can import this module
    # without pulling in the pytorch-grad-cam / OpenCV stack unless a heatmap
    # is actually requested — see ENABLE_GRADCAM in the predictor.
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    transform = build_transforms(config, train=False)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    resized = image.convert("RGB").resize((config.image_size, config.image_size))
    rgb_float = np.array(resized).astype(np.float32) / 255.0

    category_only_model = _CategoryHeadOnly(model)
    target_layers = [last_conv_layer(model)]
    cam = GradCAM(model=category_only_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_index)])[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return overlay


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.gradcam <image_path>")
        sys.exit(1)

    device = get_device(config.device)
    model, category_names, _condition_names = load_checkpoint(config.checkpoints_dir / "best_model.pt", device)

    image = Image.open(sys.argv[1])
    transform = build_transforms(config, train=False)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        category_logits, _condition_logits, _score = model(input_tensor)
        predicted_idx = int(category_logits.argmax(dim=1).item())

    print(f"Predicted category: {category_names[predicted_idx]}")

    overlay = generate_gradcam(model, image, predicted_idx, device)
    out_path = Path(sys.argv[1]).with_suffix(".gradcam.png")
    Image.fromarray(overlay).save(out_path)
    print(f"Saved heatmap -> {out_path}")


if __name__ == "__main__":
    main()
