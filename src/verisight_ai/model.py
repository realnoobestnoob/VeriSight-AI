"""Model architectures, staged freezing and the Grad-CAM target layer."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

SUPPORTED = ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0")


def build_model(architecture: str, pretrained: bool, num_classes: int = 2) -> nn.Module:
    if architecture not in SUPPORTED:
        raise ValueError(f"Unsupported architecture '{architecture}'. Choose from {SUPPORTED}.")
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    last = model.classifier[-1]
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


def pretrained_weight_id(architecture: str, pretrained: bool) -> str:
    if not pretrained:
        return "none (random init)"
    return {
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights.DEFAULT",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights.DEFAULT",
        "efficientnet_b0": "EfficientNet_B0_Weights.DEFAULT",
    }[architecture]


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: only the classifier head trains."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_blocks(model: nn.Module, n_blocks: int) -> list[str]:
    """Stage 2: unfreeze the last n blocks of the feature extractor. Returns their names."""
    blocks = list(model.features.children())
    chosen = blocks[-n_blocks:]
    names = []
    for idx, block in enumerate(blocks):
        if block in chosen:
            names.append(f"features.{idx}")
            for p in block.parameters():
                p.requires_grad = True
    return names


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    return model.features[-1]


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: cuda requested but no CUDA GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
