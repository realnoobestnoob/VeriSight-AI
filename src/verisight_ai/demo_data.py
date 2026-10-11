"""Tiny SYNTHETIC smoke-test dataset. Models trained on it prove nothing.

'REAL' = smooth noisy gradients, 'FAKE' = gradients with a regular high-frequency stripe.
Its only purpose is to let the full pipeline and CI run in seconds without real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def make_demo_dataset(
    root: str | Path, n_per_class: int = 60, size: int = 32, seed: int = 0
) -> Path:
    root = Path(root)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_per_class), ("test", max(n_per_class // 3, 10))):
        for cname in ("REAL", "FAKE"):
            d = root / split / cname
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                base = rng.uniform(40, 200, size=3)
                yy, xx = np.mgrid[0:size, 0:size] / size
                img = base[None, None, :] + 30 * (xx + yy)[..., None] * rng.uniform(0.5, 1.5, 3)
                img += rng.normal(0, 6, img.shape)
                if cname == "FAKE":
                    img[::2, :, :] += 25  # regular stripe pattern
                Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(
                    d / f"{cname}_{i:04d}.png"
                )
    return root
