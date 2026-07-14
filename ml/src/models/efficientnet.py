"""Schema-driven multi-head EfficientNet-B0 for seaweed analysis.

A single shared EfficientNet-B0 backbone feeds one head per measurement in
the active Schema (see config.py):

    classification   -> num_classes-way logits
    regression       -> a scalar squashed (sigmoid) into [min, max]
    segmentation     -> a lightweight decoder off the pre-pool feature map,
                         upsampled to the input resolution, giving
                         num_seg_classes-way per-pixel logits

Adding a whole new measurement (e.g. "moisture", "biofouling") therefore
needs no change here — build_model reads the schema and grows a head for it
automatically. Heads with no ground-truth data yet simply never receive
gradient (see src/losses.py's masking), so they stay at their random init
until labeled data arrives.

IMPORTANT — the dropout in every classification/regression head is
`inplace=False`. The pooled feature tensor is shared across all heads; an
in-place dropout would mutate that shared tensor and corrupt sibling heads'
forward/backward passes (this was a real gradient-corruption bug in the
earlier single-head-derived code, where EfficientNet's stock classifier uses
`Dropout(inplace=True)`).
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from config import Schema, legacy_schema_from_checkpoint, schema_from_dict, schema_to_dict

FEATURE_DIM = 1280  # EfficientNet-B0 pooled feature width


def _head(out_features: int) -> nn.Sequential:
    # inplace=False is mandatory here — see module docstring.
    return nn.Sequential(nn.Dropout(p=0.3, inplace=False), nn.Linear(FEATURE_DIM, out_features))


class _SegmentationHead(nn.Module):
    """A minimal decoder: one conv block off the backbone's pre-pool feature
    map, then bilinear-upsampled to the input's actual resolution at forward
    time (so the model doesn't bake in a fixed image size)."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        logits = self.conv(feature_map)
        return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


class MultiHeadSeaweedModel(nn.Module):
    def __init__(self, schema: Schema, pretrained: bool = True) -> None:
        super().__init__()
        self.schema = schema
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        heads: dict[str, nn.Module] = {}
        seg_heads: dict[str, nn.Module] = {}
        self._regression_ranges: dict[str, tuple[float, float]] = {}
        for m in schema.measurements:
            if m.type == "classification":
                heads[m.key] = _head(max(len(m.classes), 1))
            elif m.type == "regression":
                heads[m.key] = _head(1)
                self._regression_ranges[m.key] = (m.min, m.max)
            elif m.type == "segmentation":
                seg_heads[m.key] = _SegmentationHead(FEATURE_DIM, max(len(m.seg_classes), 1))
        self.heads = nn.ModuleDict(heads)
        self.seg_heads = nn.ModuleDict(seg_heads)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.features(x)
        pooled = torch.flatten(self.avgpool(feature_map), 1)  # shared across heads; must not be mutated in place

        out: dict[str, torch.Tensor] = {}
        for m in self.schema.measurements:
            if m.type == "classification":
                out[m.key] = self.heads[m.key](pooled)
            elif m.type == "regression":
                logits = self.heads[m.key](pooled)
                lo, hi = self._regression_ranges[m.key]
                out[m.key] = torch.sigmoid(logits).squeeze(-1) * (hi - lo) + lo
            elif m.type == "segmentation":
                out[m.key] = self.seg_heads[m.key](feature_map, out_hw=x.shape[-2:])
        return out


class ClassificationLogitsWrapper(nn.Module):
    """Exposes one classification measurement's logits, so pytorch-grad-cam's
    ClassifierOutputTarget (which expects a plain logits tensor) can drive
    Grad-CAM on the multi-head model."""

    def __init__(self, model: MultiHeadSeaweedModel, measurement_key: str) -> None:
        super().__init__()
        self.model = model
        self.measurement_key = measurement_key

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[self.measurement_key]


def build_model(schema: Schema, freeze_backbone: bool = True, pretrained: bool = True) -> MultiHeadSeaweedModel:
    model = MultiHeadSeaweedModel(schema, pretrained=pretrained)
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


def save_checkpoint(model: nn.Module, schema: Schema, path, extra: dict | None = None) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "schema": schema_to_dict(schema),
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device) -> tuple[MultiHeadSeaweedModel, Schema]:
    payload = torch.load(path, map_location=device)
    if "schema" in payload:
        schema = schema_from_dict(payload["schema"])
    else:
        # Pre-schema checkpoint (condition_classes/subtype_classes/species
        # keys, no "schema" key) — synthesize an equivalent Schema so old
        # checkpoints keep loading.
        schema = legacy_schema_from_checkpoint(payload)
    model = build_model(schema, freeze_backbone=False, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    # Release the checkpoint's state dict promptly — on a 512 MB host every
    # megabyte of headroom matters.
    del payload
    gc.collect()
    model.to(device)
    model.eval()
    return model, schema
