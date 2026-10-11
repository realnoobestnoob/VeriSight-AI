from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image


def make_image(size=(48, 48), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8))


def image_bytes(fmt="PNG", size=(48, 48), seed=0) -> bytes:
    buf = io.BytesIO()
    make_image(size, seed).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return image_bytes("PNG")


@pytest.fixture
def bundle_dir(tmp_path):
    """A tiny, deterministic, randomly-initialised model bundle (a test double, not a real model)."""
    torch = pytest.importorskip("torch")
    from verisight_ai.inference import sha256_file
    from verisight_ai.model import build_model

    torch.manual_seed(0)
    model = build_model("mobilenet_v3_small", pretrained=False)
    d = tmp_path / "bundle"
    d.mkdir()
    torch.save(model.state_dict(), d / "model.pt")
    meta = {
        "model_release_id": "test-release-1",
        "architecture": "mobilenet_v3_small",
        "preprocessing": {
            "image_size": 32,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "calibration": {"method": "temperature_scaling", "temperature": 1.5, "applied": True},
        "decision": {"threshold": 0.5, "abstain_margin": 0.1},
        "mlflow_run_id": "run123",
        "git_commit": "abc123",
        "artifact_sha256": sha256_file(d / "model.pt"),
        "metrics_summary": {"internal_test": {"roc_auc": 0.95, "f1": 0.9, "n": 10}},
        "dataset": {"dataset_name": "TEST"},
    }
    (d / "metadata.json").write_text(json.dumps(meta))
    (d / "model_card.md").write_text("# test")
    return d
