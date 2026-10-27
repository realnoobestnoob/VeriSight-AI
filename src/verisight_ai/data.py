"""Dataset validation, split manifests and leakage checks.

Supported layouts:
  1. <root>/train/{REAL,FAKE}/, <root>/val/{REAL,FAKE}/, <root>/test/{REAL,FAKE}/
     Official splits used as-is (Defactify layout).
  2. <root>/train/{REAL,FAKE}/, <root>/test/{REAL,FAKE}/  (no val/ folder)
     Val is carved from train with a seeded stratified split.
  3. <root>/{REAL,FAKE}/  (flat layout, no split folders)
     Seeded stratified train/val/test split generated from scratch.
Data is never downloaded automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split  # train extra

from .config import CLASS_MAPPING, CLASS_NAMES, DataConfig

log = logging.getLogger(__name__)
MANIFEST_COLUMNS = ["path", "label", "class_name", "split", "sha256"]


class DataValidationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def average_hash(path: Path, size: int = 8) -> int:
    with Image.open(path) as im:
        g = im.convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    bits = (a > a.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _class_dir(parent: Path, name: str) -> Path | None:
    for child in parent.iterdir() if parent.is_dir() else []:
        if child.is_dir() and child.name.upper() == name:
            return child
    return None


def _list_images(folder: Path, extensions: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensions}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _scan_split_folder(parent: Path, extensions: list[str]) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for cname in CLASS_NAMES:
        cdir = _class_dir(parent, cname)
        if cdir is None:
            raise DataValidationError(
                f"Missing class folder '{cname}' under {parent}. Expected folders: {CLASS_NAMES}."
            )
        out += [(p, cname) for p in _list_images(cdir, extensions)]
    return out


@dataclass
class DataReport:
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    corrupt_files: list[str] = field(default_factory=list)
    cross_split_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _subsample(items: list[tuple[Path, str]], max_per_class: int | None, seed: int):
    if not max_per_class:
        return items
    rng = np.random.default_rng(seed)
    out = []
    for cname in CLASS_NAMES:
        subset = [it for it in items if it[1] == cname]
        if len(subset) > max_per_class:
            idx = rng.choice(len(subset), size=max_per_class, replace=False)
            subset = [subset[i] for i in sorted(idx)]
        out += subset
    return out


def _drop_corrupt(items: list[tuple[Path, str]], report: DataReport):
    ok = []
    for path, cname in items:
        try:
            with Image.open(path) as im:
                im.verify()
            ok.append((path, cname))
        except Exception:  # noqa: BLE001 - any decode failure marks the file corrupt
            report.corrupt_files.append(str(path))
    return ok


def build_manifest(cfg: DataConfig, seed: int) -> tuple[pd.DataFrame, DataReport]:
    """Validate the dataset and build a reproducible train/val/test manifest."""
    root = Path(cfg.root)
    if not root.is_dir():
        raise DataValidationError(
            f"Dataset folder not found: {root}. Run scripts/download_defactify.py first, or "
            "run `python -m verisight_ai.cli demo-data` to create a tiny synthetic smoke dataset."
        )
    report = DataReport()
    train_dir = _class_dir_any(root, "train")
    val_dir = _class_dir_any(root, "val")
    test_dir = _class_dir_any(root, "test")
    frames = []
    if train_dir is not None and test_dir is not None:
        train_items = _subsample(
            _scan_split_folder(train_dir, cfg.extensions), cfg.max_per_class, seed
        )
        test_items = _subsample(
            _scan_split_folder(test_dir, cfg.extensions), cfg.max_per_class, seed
        )
        train_items = _drop_corrupt(train_items, report)
        test_items = _drop_corrupt(test_items, report)

        if val_dir is not None:
            # Layout 1: official train/val/test folders present — use them as-is.
            val_items = _subsample(
                _scan_split_folder(val_dir, cfg.extensions), cfg.max_per_class, seed
            )
            val_items = _drop_corrupt(val_items, report)
            val_labels = [CLASS_MAPPING[c] for _, c in val_items]
            if len(set(val_labels)) < 2:
                raise DataValidationError("Val data must contain both REAL and FAKE images.")
            log.info(
                "Official val/ folder found — using as-is (%d images).", len(val_items)
            )
            frames += [
                _frame(train_items, np.arange(len(train_items)), "train", root),
                _frame(val_items, np.arange(len(val_items)), "val", root),
            ]
        else:
            # Layout 2: no val/ folder — carve val from train.
            labels = [CLASS_MAPPING[c] for _, c in train_items]
            if len(set(labels)) < 2:
                raise DataValidationError("Training data must contain both REAL and FAKE images.")
            idx_train, idx_val = train_test_split(
                np.arange(len(train_items)),
                test_size=cfg.val_fraction,
                stratify=labels,
                random_state=seed,
            )
            log.info(
                "No val/ folder — carved from train/ (%d train, %d val).",
                len(idx_train),
                len(idx_val),
            )
            frames += [
                _frame(train_items, idx_train, "train", root),
                _frame(train_items, idx_val, "val", root),
            ]

        frames.append(_frame(test_items, np.arange(len(test_items)), "test", root))
    else:
        items = _subsample(_scan_split_folder(root, cfg.extensions), cfg.max_per_class, seed)
        items = _drop_corrupt(items, report)
        labels = [CLASS_MAPPING[c] for _, c in items]
        idx = np.arange(len(items))
        idx_rest, idx_test = train_test_split(
            idx, test_size=cfg.test_fraction, stratify=labels, random_state=seed
        )
        rest_labels = [labels[i] for i in idx_rest]
        rel_val = cfg.val_fraction / (1 - cfg.test_fraction)
        idx_train, idx_val = train_test_split(
            idx_rest, test_size=rel_val, stratify=rest_labels, random_state=seed
        )
        frames += [
            _frame(items, idx_train, "train", root),
            _frame(items, idx_val, "val", root),
            _frame(items, idx_test, "test", root),
        ]
        report.warnings.append(
            "No official train/test folders found; a random split was generated. Related images "
            "from one source may span splits."
        )
    manifest = pd.concat(frames, ignore_index=True)
    manifest = _remove_leakage(manifest, root, cfg, report)
    _validate_manifest(manifest, report)
    manifest = manifest.sort_values(["split", "path"]).reset_index(drop=True)
    return manifest[MANIFEST_COLUMNS], report


def _class_dir_any(root: Path, name: str) -> Path | None:
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == name:
            return child
    return None


def _frame(items, indices, split: str, root: Path) -> pd.DataFrame:
    rows = []
    for i in indices:
        path, cname = items[int(i)]
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "label": CLASS_MAPPING[cname],
                "class_name": cname,
                "split": split,
                "sha256": file_sha256(path),
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _remove_leakage(manifest: pd.DataFrame, root: Path, cfg: DataConfig, report: DataReport):
    """Exact duplicates across splits are removed from train/val so test stays untouched."""
    test_hashes = set(manifest.loc[manifest.split == "test", "sha256"])
    train_hashes = set(manifest.loc[manifest.split == "train", "sha256"])
    leak_test = manifest.split.isin(["train", "val"]) & manifest.sha256.isin(test_hashes)
    leak_val = (manifest.split == "val") & manifest.sha256.isin(train_hashes)
    drop = leak_test | leak_val
    report.cross_split_duplicates_removed = int(drop.sum())
    manifest = manifest[~drop].copy()
    # Within-split duplicates are kept but reported.
    dup_within = int(manifest.duplicated(subset=["split", "sha256"]).sum())
    if dup_within:
        report.warnings.append(f"{dup_within} exact duplicate file(s) exist within a single split.")
    if cfg.near_duplicate_max_hamming is not None:
        manifest = _remove_near_duplicates(manifest, root, cfg.near_duplicate_max_hamming, report)
    return manifest


def _remove_near_duplicates(manifest, root: Path, max_hamming: int, report: DataReport):
    hashes = {row.path: average_hash(root / row.path) for row in manifest.itertuples()}
    test_vals = [hashes[p] for p in manifest.loc[manifest.split == "test", "path"]]
    drop_paths = []
    for row in manifest[manifest.split != "test"].itertuples():
        h = hashes[row.path]
        if any(bin(h ^ t).count("1") <= max_hamming for t in test_vals):
            drop_paths.append(row.path)
    report.near_duplicates_removed = len(drop_paths)
    return manifest[~manifest.path.isin(drop_paths)].copy()


def _validate_manifest(manifest: pd.DataFrame, report: DataReport) -> None:
    for split in ("train", "val", "test"):
        sub = manifest[manifest.split == split]
        counts = {c: int((sub.class_name == c).sum()) for c in CLASS_NAMES}
        report.counts[split] = counts
        if sum(counts.values()) == 0:
            raise DataValidationError(f"The '{split}' split is empty after validation.")
        if min(counts.values()) == 0:
            raise DataValidationError(f"The '{split}' split is missing a class: {counts}.")
        ratio = max(counts.values()) / min(counts.values())
        if ratio > 1.5:
            report.warnings.append(f"Class imbalance in '{split}' split: {counts}.")


def manifest_hash(manifest: pd.DataFrame) -> str:
    payload = manifest.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_manifest(
    manifest: pd.DataFrame, report: DataReport, cfg: DataConfig, out_dir: Path
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    meta = {
        "dataset_name": cfg.dataset_name,
        "dataset_version": cfg.dataset_version,
        "dataset_source": cfg.dataset_source,
        "manifest_sha256": manifest_hash(manifest),
        "class_mapping": CLASS_MAPPING,
        "report": report.to_dict(),
        "known_limitations": [
            "Results on this dataset do not establish detection of deepfakes or real-world fakes.",
            "Check dataset licence/terms before redistributing any images.",
        ],
    }
    (out_dir / "manifest_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_external_manifest(root: Path, extensions: list[str]) -> pd.DataFrame:
    """External evaluation set with <root>/{REAL,FAKE}; all rows are split='external'."""
    items = _scan_split_folder(root, extensions)
    if not items:
        raise DataValidationError(f"No images found under {root}.")
    return _frame(items, np.arange(len(items)), "external", root)