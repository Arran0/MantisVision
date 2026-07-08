"""EfficientNet-B0 transfer-learning model for seaweed health classification.

Three heads share one backbone:
  - category_head: 3-way (Healthy/Moderate/Low)
  - condition_head: 4-way (None/Dried/Decayed/Diseased), only meaningful when
    category != Healthy (see the masked loss in src/train.py)
  - score_head: scalar health score in [0, 10], via sigmoid(x) * 10

Recommended baseline per spec: EfficientNet-B0 (best accuracy/size tradeoff
for a first model). Swap `build_multihead_model`'s backbone to try
EfficientNetV2-S, ConvNeXt-Tiny, or MobileNetV3 later without touching the
training loop.
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class MultiHeadEfficientNet(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_conditions: int,
        freeze_backbone: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        # Inference loads its own trained weights straight after, so skip
        # fetching and materialising the ImageNet weights there (saves memory
        # + a download).
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)

        # Kept as `.features`/`.avgpool` so unfreeze_backbone/last_conv_layer
        # (and Grad-CAM's target-layer lookup) don't need to change.
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        in_features = backbone.classifier[1].in_features

        def make_head(out_dim: int) -> nn.Sequential:
            # inplace=False: the pooled feature tensor below is shared by all
            # three heads, so an in-place dropout in one head would corrupt
            # the tensor the other heads' backward pass still needs.
            return nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(in_features, out_dim))

        self.category_head = make_head(num_categories)
        self.condition_head = make_head(num_conditions)
        self.score_head = make_head(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.flatten(self.avgpool(self.features(x)), 1)
        category_logits = self.category_head(features)
        condition_logits = self.condition_head(features)
        score = torch.sigmoid(self.score_head(features).squeeze(-1)) * 10.0
        return category_logits, condition_logits, score


def build_multihead_model(
    num_categories: int,
    num_conditions: int,
    freeze_backbone: bool = True,
    pretrained: bool = True,
) -> MultiHeadEfficientNet:
    return MultiHeadEfficientNet(
        num_categories=num_categories,
        num_conditions=num_conditions,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained,
    )


def unfreeze_backbone(model: MultiHeadEfficientNet) -> None:
    """Call between the frozen warm-up phase and the fine-tuning phase."""
    for param in model.features.parameters():
        param.requires_grad = True


def last_conv_layer(model: MultiHeadEfficientNet) -> nn.Module:
    """The final conv layer, used as the Grad-CAM target layer."""
    return model.features[-1]


def save_checkpoint(
    model: MultiHeadEfficientNet,
    category_names: list[str],
    condition_names: list[str],
    score_min: float,
    score_max: float,
    path,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "category_names": category_names,
        "condition_names": condition_names,
        "score_min": score_min,
        "score_max": score_max,
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device) -> tuple[MultiHeadEfficientNet, list[str], list[str]]:
    payload = torch.load(path, map_location=device)
    category_names = payload["category_names"]
    condition_names = payload["condition_names"]
    model = build_multihead_model(
        num_categories=len(category_names),
        num_conditions=len(condition_names),
        freeze_backbone=False,
        pretrained=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    # Release the checkpoint's state dict promptly — on a 512 MB host every
    # megabyte of headroom matters.
    del payload
    gc.collect()
    model.to(device)
    model.eval()
    return model, category_names, condition_names
