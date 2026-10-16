"""Metrics, plots, robustness suite and CPU latency benchmark."""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .calibration import softmax_p_synthetic
from .config import CLASS_NAMES, SYNTHETIC_INDEX, PipelineConfig
from .train import collect_logits, make_loader

log = logging.getLogger(__name__)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE for the synthetic class: equal-width bins on P(FAKE), weighted |acc - conf|."""
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(p)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(ece)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = (labels == SYNTHETIC_INDEX).astype(int)
    pred = (scores >= threshold).astype(int)
    out = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "brier": float(brier_score_loss(y, scores)),
        "ece_10bin": expected_calibration_error(scores, y),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": CLASS_NAMES,
    }
    if len(set(y.tolist())) == 2:
        out["roc_auc"] = float(roc_auc_score(y, scores))
        out["pr_auc"] = float(average_precision_score(y, scores))
    else:
        out["roc_auc"] = out["pr_auc"] = None
    return out


def save_plots(labels, scores, metrics: dict, out_dir: Path, prefix: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    y = (labels == SYNTHETIC_INDEX).astype(int)
    paths = []

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = np.array(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix}: confusion matrix")
    p = out_dir / f"{prefix}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    if len(set(y.tolist())) == 2:
        for kind in ("roc", "pr"):
            fig, ax = plt.subplots(figsize=(4, 4))
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, scores)
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1], "--", color="grey")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
            else:
                pr, rc, _ = precision_recall_curve(y, scores)
                ax.plot(rc, pr)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
            ax.set_title(f"{prefix}: {kind.upper()}")
            p = out_dir / f"{prefix}_{kind}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=120)
            plt.close(fig)
            paths.append(p)

        edges = np.linspace(0, 1, 11)
        mids, acc = [], []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (scores >= lo) & (scores <= hi if hi == 1 else scores < hi)
            if m.any():
                mids.append(scores[m].mean())
                acc.append(y[m].mean())
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.plot(mids, acc, marker="o")
        ax.set_xlabel("Mean score")
        ax.set_ylabel("Fraction FAKE")
        ax.set_title(f"{prefix}: reliability")
        p = out_dir / f"{prefix}_reliability.png"
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


# ---- Robustness suite: exact, recorded transformation protocol --------------------------------
def _resize_roundtrip(img: Image.Image) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, w // 2), max(1, h // 2)), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


class _JpegReencode:
    """Picklable JPEG re-encode transform.

    A nested closure (the previous implementation) is not picklable under
    Python 3.14's default 'forkserver' multiprocessing start method, which
    causes DataLoader worker spawning to fail with a PicklingError.  A
    top-level class instance is always picklable.
    """

    __slots__ = ("quality",)

    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def _jpeg(quality: int) -> _JpegReencode:
    return _JpegReencode(quality)


def _center_crop_80(img: Image.Image) -> Image.Image:
    w, h = img.size
    cw, ch = max(1, int(w * 0.8)), max(1, int(h * 0.8))
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)


def _screenshot_like(img: Image.Image) -> Image.Image:
    w, h = img.size
    up = img.resize((w * 2, h * 2), Image.BICUBIC)
    return _jpeg(80)(up).resize((w, h), Image.BILINEAR)


ROBUSTNESS_PROTOCOL = {
    "resize_half_roundtrip": "downscale to 50% (bilinear) then upscale to original (bilinear)",
    "jpeg_q75": "JPEG re-encode at quality 75 (Pillow)",
    "jpeg_q50": "JPEG re-encode at quality 50 (Pillow)",
    "center_crop_80": "centre crop to 80% width/height then resize back (bilinear)",
    "screenshot_like": "upscale 2x (bicubic), JPEG q80, downscale to original (bilinear)",
}
_TRANSFORMS = {
    "resize_half_roundtrip": _resize_roundtrip,
    "jpeg_q75": _jpeg(75),
    "jpeg_q50": _jpeg(50),
    "center_crop_80": _center_crop_80,
    "screenshot_like": _screenshot_like,
}


def evaluate_split(
    model,
    df: pd.DataFrame,
    root: Path,
    cfg: PipelineConfig,
    device,
    temperature: float,
    threshold: float,
    pil_transform=None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    loader = make_loader(df, root, cfg.model.image_size, False, cfg, pil_transform, shuffle=False)
    logits, labels = collect_logits(model, loader, device)
    scores = softmax_p_synthetic(logits, temperature)
    return compute_metrics(labels, scores, threshold), labels, scores


def robustness_suite(
    model,
    test_df: pd.DataFrame,
    root: Path,
    cfg: PipelineConfig,
    device,
    temperature: float,
    threshold: float,
) -> dict:
    n = min(cfg.evaluation.robustness_max_images, len(test_df))
    sub = test_df.sample(n=n, random_state=cfg.seed) if n < len(test_df) else test_df
    results = {"n_images": int(len(sub)), "protocol": ROBUSTNESS_PROTOCOL, "results": {}}
    base, _, _ = evaluate_split(model, sub, root, cfg, device, temperature, threshold)
    results["results"]["unaltered_subset"] = base
    for name, fn in _TRANSFORMS.items():
        m, _, _ = evaluate_split(model, sub, root, cfg, device, temperature, threshold, fn)
        results["results"][name] = m
    return results


def benchmark_cpu_latency(model, image_size: int, warmup: int, runs: int) -> dict:
    """Single-image CPU forward latency (model only, excludes decode). Host-specific."""
    cpu_model = model.to("cpu").eval()
    x = torch.randn(1, 3, image_size, image_size)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            cpu_model(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            cpu_model(x)
            times.append((time.perf_counter() - t0) * 1000)
    out = {
        "runs": runs,
        "mean_ms": float(np.mean(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "torch_threads": torch.get_num_threads(),
    }
    try:
        import psutil

        out["process_rss_mb"] = float(psutil.Process().memory_info().rss / 1024 / 1024)
    except ImportError:
        out["process_rss_mb"] = None
    return out