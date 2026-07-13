"""Multi-head EfficientNet-B0 for seaweed analysis.

A single shared EfficientNet-B0 backbone feeds five heads off the pooled
1280-d feature:

    condition        -> num_conditions-way logits (incl. Background)
    health_score     -> scalar 0-100 (sigmoid x 100)
    disease_subtype  -> num_subtypes-way logits (supervised on Disease only)
    dried_extent     -> scalar 0-100
    decayed_extent   -> scalar 0-100

Swap the backbone (EfficientNetV2-S, ConvNeXt-Tiny, ...) here without
touching the heads or the training loop.

IMPORTANT — the dropout in every head is `inplace=False`. The pooled feature
tensor is shared across all five heads; an in-place dropout would mutate that
shared tensor and corrupt sibling heads' forward/backward passes (this was a
real gradient-corruption bug in the earlier single-head-derived code, where
EfficientNet's stock classifier uses `Dropout(inplace=True)`).
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

FEATURE_DIM = 1280  # EfficientNet-B0 pooled feature width

# Names of the two regression heads whose raw output is squashed to 0-100.
_SCORE_HEADS = ("health_score", "dried_extent", "decayed_extent")


def _head(out_features: int) -> nn.Sequential:
    # inplace=False is mandatory here — see module docstring.
    return nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(FEATURE_DIM, out_features))


class MultiHeadSeaweedModel(nn.Module):
    def __init__(self, num_conditions: int, num_subtypes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.heads = nn.ModuleDict(
            {
                "condition": _head(num_conditions),
                "health_score": _head(1),
                "disease_subtype": _head(num_subtypes),
                "dried_extent": _head(1),
                "decayed_extent": _head(1),
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        pooled = torch.flatten(x, 1)  # shared across heads; must not be mutated in place

        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            logits = head(pooled)
            if name in _SCORE_HEADS:
                # Squash to 0-100 so the regression targets (also 0-100) and
                # the head output live on the same scale.
                out[name] = torch.sigmoid(logits).squeeze(-1) * 100.0
            else:
                out[name] = logits
        return out


class ConditionLogitsWrapper(nn.Module):
    """Exposes only the condition-head logits, so pytorch-grad-cam's
    ClassifierOutputTarget (which expects a plain logits tensor) can drive
    Grad-CAM on the multi-head model."""

    def __init__(self, model: MultiHeadSeaweedModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["condition"]


def build_model(
    num_conditions: int, num_subtypes: int, freeze_backbone: bool = True, pretrained: bool = True
) -> MultiHeadSeaweedModel:
    model = MultiHeadSeaweedModel(num_conditions, num_subtypes, pretrained=pretrained)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    return model


def unfreeze_backbone(model: MultiHeadSeaweedModel) -> None:
    """Call between the frozen warm-up phase and the fine-tuning phase."""
    for param in model.features.parameters():
        param.requires_grad = True


def last_conv_layer(model: MultiHeadSeaweedModel) -> nn.Module:
    """Final conv layer, used as the Grad-CAM target layer."""
    return model.features[-1]


def save_checkpoint(
    model: nn.Module,
    condition_classes: list[str],
    subtype_classes: list[str],
    species: dict,
    path,
    extra: dict | None = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "condition_classes": condition_classes,
        "subtype_classes": subtype_classes,
        "species": species,
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device) -> tuple[MultiHeadSeaweedModel, list[str], list[str], dict]:
    payload = torch.load(path, map_location=device)
    condition_classes = payload["condition_classes"]
    subtype_classes = payload["subtype_classes"]
    species = payload.get("species", {})
    model = build_model(
        num_conditions=len(condition_classes),
        num_subtypes=len(subtype_classes),
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
    return model, condition_classes, subtype_classes, species
