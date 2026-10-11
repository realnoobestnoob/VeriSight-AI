"""Training: frozen-backbone head stage, then optional fine-tuning. Explicit MLflow logging."""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import CLASS_NAMES, PipelineConfig
from .model import (
    build_model,
    count_trainable,
    freeze_backbone,
    trainable_parameters,
    unfreeze_last_blocks,
)
from .preprocessing import build_transform

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, root: Path, transform, pil_transform=None):
        self.paths = [Path(root) / p for p in manifest["path"]]
        self.labels = manifest["label"].astype(int).tolist()
        self.transform = transform
        self.pil_transform = pil_transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.pil_transform is not None:
            img = self.pil_transform(img)
        return self.transform(img), self.labels[i]


def make_loader(df, root, image_size, train, cfg: PipelineConfig, pil_transform=None, shuffle=None):
    ds = ManifestDataset(df, root, build_transform(image_size, train), pil_transform)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    return DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=cfg.train.num_workers,
        generator=gen,
    )


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _epoch(model, loader, device, criterion, optimizer=None, backbone_eval=False):
    train = optimizer is not None
    model.train(train)
    if train and backbone_eval:
        model.features.eval()  # frozen backbone: keep BatchNorm statistics fixed
    total, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.item()) * len(y)
            n += len(y)
            correct += int((out.argmax(1) == y).sum().item())
    return total / max(n, 1), correct / max(n, 1)


@dataclass
class TrainResult:
    state_dict: dict
    history: list[dict] = field(default_factory=list)
    stage_summary: dict = field(default_factory=dict)
    best_val_loss: float = float("inf")


def _run_stage(
    name, model, cfg, lr, epochs, train_loader, val_loader, device, criterion, tracker, step0, best
) -> tuple[list[dict], int]:
    history: list[dict] = []
    if epochs <= 0:
        return history, step0
    opt = torch.optim.AdamW(trainable_parameters(model), lr=lr, weight_decay=cfg.train.weight_decay)
    stale, step = 0, step0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _epoch(
            model, train_loader, device, criterion, opt, backbone_eval=(name == "head")
        )
        va_loss, va_acc = _epoch(model, val_loader, device, criterion)
        step += 1
        row = {
            "stage": name,
            "epoch": epoch,
            "step": step,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "seconds": time.time() - t0,
        }
        history.append(row)
        tracker.log_metrics(
            {"train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc},
            step=step,
        )
        log.info(
            "[%s] epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
            name,
            epoch,
            epochs,
            tr_loss,
            va_loss,
            va_acc,
        )
        if va_loss < best["loss"] - 1e-6:
            best.update(loss=va_loss, state=copy.deepcopy(model.state_dict()), stage=name)
            stale = 0
        else:
            stale += 1
            if stale >= cfg.train.early_stopping_patience:
                log.info("[%s] early stopping after epoch %d", name, epoch)
                break
    return history, step


def run_training(
    cfg: PipelineConfig, manifest: pd.DataFrame, root: Path, tracker, device
) -> TrainResult:
    seed_everything(cfg.seed)
    size = cfg.model.image_size
    train_df = manifest[manifest.split == "train"]
    val_df = manifest[manifest.split == "val"]
    train_loader = make_loader(train_df, root, size, True, cfg)
    val_loader = make_loader(val_df, root, size, False, cfg, shuffle=False)

    model = build_model(cfg.model.architecture, cfg.model.pretrained).to(device)
    weight = None
    if cfg.train.class_weighting:
        counts = np.bincount(train_df["label"].astype(int), minlength=len(CLASS_NAMES))
        w = counts.sum() / (len(counts) * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    best = {"loss": float("inf"), "state": None, "stage": None}
    history: list[dict] = []

    freeze_backbone(model)
    tracker.log_params({"stage1_trainable_params": count_trainable(model)})
    h1, step = _run_stage(
        "head",
        model,
        cfg,
        cfg.train.head.lr,
        cfg.train.head.epochs,
        train_loader,
        val_loader,
        device,
        criterion,
        tracker,
        0,
        best,
    )
    history += h1
    head_best = best["loss"]

    ft = cfg.train.finetune
    unfrozen: list[str] = []
    if ft.enabled and ft.epochs > 0 and best["state"] is not None:
        model.load_state_dict(best["state"])  # start fine-tuning from the best head-stage weights
        unfrozen = unfreeze_last_blocks(model, ft.unfreeze_last_blocks)
        tracker.log_params(
            {
                "stage2_trainable_params": count_trainable(model),
                "stage2_unfrozen_blocks": ",".join(unfrozen),
            }
        )
        h2, step = _run_stage(
            "finetune",
            model,
            cfg,
            ft.lr,
            ft.epochs,
            train_loader,
            val_loader,
            device,
            criterion,
            tracker,
            step,
            best,
        )
        history += h2

    if best["state"] is None:
        raise RuntimeError("Training produced no checkpoint (all epochs set to 0?).")
    finetune_helped = best["stage"] == "finetune"
    summary = {
        "selected_stage": best["stage"],
        "head_best_val_loss": head_best if head_best != float("inf") else None,
        "final_best_val_loss": best["loss"],
        "finetune_helped_on_validation": finetune_helped,
        "unfrozen_blocks": unfrozen,
    }
    tracker.log_params(
        {"selected_stage": best["stage"], "finetune_helped_on_validation": finetune_helped}
    )
    return TrainResult(best["state"], history, summary, best["loss"])
