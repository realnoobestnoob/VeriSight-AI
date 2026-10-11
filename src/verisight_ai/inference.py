"""Shared prediction path used by the app, the release gate and tests."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image

from .calibration import decide
from .config import CLASS_NAMES, SYNTHETIC_INDEX
from .explainability import compute_gradcam, overlay_heatmap
from .model import build_model
from .preprocessing import build_transform
from .schemas import ClassifierResult

log = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """No usable model bundle (missing, corrupt, or checksum mismatch)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_bundle_dir(model_dir: str | None, artifacts_dir: str = "model_artifacts") -> Path:
    """VERISIGHT_MODEL_DIR wins; otherwise follow model_artifacts/current.json."""
    if model_dir:
        return Path(model_dir)
    pointer = Path(artifacts_dir) / "current.json"
    if not pointer.is_file():
        raise ModelUnavailableError(
            "No promoted model release found (model_artifacts/current.json is missing)."
        )
    info = json.loads(pointer.read_text(encoding="utf-8"))
    return Path(artifacts_dir) / "releases" / info["release_id"]


class Predictor:
    def __init__(self, bundle_dir: str | Path, device: str = "cpu"):
        self.dir = Path(bundle_dir)
        meta_path = self.dir / "metadata.json"
        weights_path = self.dir / "model.pt"
        if not meta_path.is_file() or not weights_path.is_file():
            raise ModelUnavailableError(f"Model bundle incomplete at {self.dir}.")
        self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = self.metadata.get("artifact_sha256")
        if expected and sha256_file(weights_path) != expected:
            raise ModelUnavailableError("Model checksum mismatch; refusing to load.")
        self.device = torch.device(device)
        try:
            self.model = build_model(self.metadata["architecture"], pretrained=False)
            self.model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError("Model weights could not be loaded.") from exc
        self.model.to(self.device).eval()
        pre = self.metadata["preprocessing"]
        self.image_size = int(pre["image_size"])
        self.transform = build_transform(self.image_size, False, pre["mean"], pre["std"])
        cal = self.metadata.get("calibration", {})
        self.temperature = float(cal.get("temperature", 1.0)) if cal.get("applied") else 1.0
        self.calibrated = bool(cal.get("applied"))
        dec = self.metadata["decision"]
        self.threshold = float(dec["threshold"])
        self.margin = float(dec["abstain_margin"])
        self.release_id = self.metadata["model_release_id"]
        self.run_id = self.metadata.get("mlflow_run_id")

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).float()

    def predict(self, image: Image.Image) -> ClassifierResult:
        t0 = time.perf_counter()
        x = self._tensor(image)
        logits = self._logits(x)
        raw = float(torch.softmax(logits, dim=1)[0, SYNTHETIC_INDEX])
        calibrated = None
        if self.calibrated:
            calibrated = float(torch.softmax(logits / self.temperature, dim=1)[0, SYNTHETIC_INDEX])
        score = calibrated if calibrated is not None else raw
        status = decide(score, self.threshold, self.margin)
        latency = (time.perf_counter() - t0) * 1000
        return ClassifierResult(
            predicted_class=CLASS_NAMES[SYNTHETIC_INDEX if score >= self.threshold else 0],
            raw_score=raw,
            calibrated_score=calibrated,
            score_used=score,
            score_label="calibrated" if calibrated is not None else "raw",
            threshold=self.threshold,
            abstain_margin=self.margin,
            decision_status=status,
            model_release_id=self.release_id,
            training_run_id=self.run_id,
            latency_ms=latency,
        )

    def explain(self, image: Image.Image) -> Image.Image:
        x = self._tensor(image)
        cam = compute_gradcam(self.model, x)
        return overlay_heatmap(image.resize((self.image_size, self.image_size)), cam)
