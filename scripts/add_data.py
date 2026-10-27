"""
Extend the current Defactify dataset by adding new REAL and/or FAKE images
to the existing data/{train,val,test}/{REAL,FAKE}/ folders.

This script scans a source folder for images in a {REAL,FAKE}/ layout and copies them
into the existing dataset, deduplicating by SHA-256 hash to prevent adding duplicates
of images already in the dataset.

Usage:
    # Add new images to train/
    python scripts/add_data.py \
        --source /path/to/new/images \
        --split train

    # Add to test/ (with dedup check against entire dataset)
    python scripts/add_data.py \
        --source /path/to/new/test/images \
        --split test

The source folder must have this layout:
    /path/to/new/images/
        REAL/
            image1.jpg
            image2.png
            ...
        FAKE/
            gen1.jpg
            gen2.jpg
            ...

New images are copied (not moved) into data/defactify/{split}/{CLASS}/
with auto-incrementing filenames (e.g. REAL_009999.jpg if 9999 already exists).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ALLOWED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_hashes(root: Path, class_name: str) -> set[str]:
    """Collect SHA-256 hashes of all existing images for a class across all splits."""
    hashes = set()
    for split in ("train", "val", "test"):
        folder = root / split / class_name
        if not folder.is_dir():
            continue
        for img in folder.glob("*"):
            if img.is_file() and img.suffix.lower() in ALLOWED_FORMATS:
                hashes.add(sha256_file(img))
    return hashes


def add_images(source: Path, dest_root: Path, split: str, existing_hashes: dict[str, set[str]]):
    """Copy images from source to dest_root/{split}/{CLASS}/, skipping duplicates."""
    added = {"REAL": 0, "FAKE": 0}
    skipped = {"REAL": 0, "FAKE": 0}

    for class_name in ("REAL", "FAKE"):
        src_folder = source / class_name
        if not src_folder.is_dir():
            log.warning(f"Source folder missing: {src_folder}")
            continue

        dest_folder = dest_root / split / class_name
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Find the highest existing index
        existing = [f.name for f in dest_folder.glob(f"{class_name}_*.jpg")]
        max_idx = max([int(f.split("_")[1].split(".")[0]) for f in existing] or [0])

        images = sorted(
            [p for p in src_folder.glob("*") if p.is_file() and p.suffix.lower() in ALLOWED_FORMATS]
        )

        for src_img in images:
            h = sha256_file(src_img)
            if h in existing_hashes[class_name]:
                log.debug(f"Duplicate (exists in dataset): {src_img.name}")
                skipped[class_name] += 1
                continue

            # Copy with auto-incrementing filename
            max_idx += 1
            dest_img = dest_folder / f"{class_name}_{max_idx:06d}.jpg"
            try:
                from PIL import Image

                pil = Image.open(src_img).convert("RGB")
                pil.save(dest_img, format="JPEG", quality=95)
                log.info(f"Added: {class_name}/{dest_img.name}")
                added[class_name] += 1
            except Exception as e:
                log.error(f"Failed to convert {src_img.name}: {e}")

    return added, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--source", required=True, help="Source folder with REAL/ and/or FAKE/ subfolders"
    )
    p.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split to add images to",
    )
    p.add_argument(
        "--data-root",
        default="data/defactify",
        help="Root of the dataset folder (default: data/defactify)",
    )
    args = p.parse_args()

    source = Path(args.source).resolve()
    data_root = Path(args.data_root).resolve()

    if not source.is_dir():
        p.error(f"Source folder not found: {source}")
    if not data_root.is_dir():
        p.error(f"Data root not found: {data_root}. Run the Defactify download first.")

    log.info(f"Scanning existing hashes in {data_root}...")
    existing = {"REAL": scan_hashes(data_root, "REAL"), "FAKE": scan_hashes(data_root, "FAKE")}
    log.info(f"Found {len(existing['REAL'])} existing REAL images, {len(existing['FAKE'])} FAKE")

    log.info(f"Adding images from {source} to {args.split}/...")
    added, skipped = add_images(source, data_root, args.split, existing)

    log.info(f"✓ Added: REAL={added['REAL']}, FAKE={added['FAKE']}")
    if skipped["REAL"] + skipped["FAKE"] > 0:
        log.info(f"⊘ Skipped (duplicates): REAL={skipped['REAL']}, FAKE={skipped['FAKE']}")

    log.info(
        "Dataset extended. Next: run `python -m verisight_ai.cli pipeline --config configs/default.yaml`"
    )


if __name__ == "__main__":
    main()
