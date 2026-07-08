"""EfficientNet-B0 transfer learning model for seaweed health classification.

Recommended baseline per spec: EfficientNet-B0 (best accuracy/size tradeoff
for a first model). Swap `build_model`'s backbone to try EfficientNetV2-S,
ConvNeXt-Tiny, or MobileNetV3 later without touching the training loop.
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_model(
    num_classes: int, freeze_backbone: bool = True, pretrained: bool = True
) -> nn.Module:
    # Inference loads its own trained weights straight after, so skip fetching
    # and materialising the ImageNet weights there (saves memory + a download).
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Call between the frozen warm-up phase and the fine-tuning phase."""
    for param in model.features.parameters():
        param.requires_grad = True


def last_conv_layer(model: nn.Module) -> nn.Module:
    """The final conv layer, used as the Grad-CAM target layer."""
    return model.features[-1]


def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    path,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device) -> tuple[nn.Module, list[str]]:
    payload = torch.load(path, map_location=device)
    class_names = payload["class_names"]
    model = build_model(
        num_classes=len(class_names), freeze_backbone=False, pretrained=False
    )
    model.load_state_dict(payload["model_state_dict"])
    # Release the checkpoint's state dict promptly — on a 512 MB host every
    # megabyte of headroom matters.
    del payload
    gc.collect()
    model.to(device)
    model.eval()
    return model, class_names
