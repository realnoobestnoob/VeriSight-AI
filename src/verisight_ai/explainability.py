"""Grad-CAM for the synthetic class. Highlights show model influence, NOT proof of manipulation."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .config import SYNTHETIC_INDEX
from .model import gradcam_target_layer

GRADCAM_CAVEAT = (
    "Highlighted regions show where the model's synthetic-class score was most influenced. "
    "They are not evidence that those regions were edited or generated."
)


def compute_gradcam(model: torch.nn.Module, x: torch.Tensor, target_class: int = SYNTHETIC_INDEX):
    """Return a [0,1] heatmap (H, W) for a single preprocessed image tensor (1,3,H,W)."""
    model.eval()
    layer = gradcam_target_layer(model)
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            model.zero_grad(set_to_none=True)
            logits[0, target_class].backward()
    finally:
        handle.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=x.shape[-2:], mode="bilinear", align_corners=False
    )
    cam = cam[0, 0].numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)


def _colormap(v: np.ndarray) -> np.ndarray:
    """Simple blue->green->red colormap without matplotlib. v in [0,1] -> uint8 RGB."""
    r = np.clip(4 * v - 2, 0, 1)
    g = np.clip(2 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(2 - 4 * v, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = image.convert("RGB")
    heat = Image.fromarray(_colormap(cam)).resize(base.size, Image.BILINEAR)
    return Image.blend(base, heat, alpha)
