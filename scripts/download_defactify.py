"""
Download Rajarshi-Roy-research/Defactify_Image_Dataset from Hugging Face and convert it to
the REAL/FAKE folder layout expected by the VeriSight pipeline.

Usage:
    python scripts/download_defactify.py --out data/defactify
    python scripts/download_defactify.py --out data/defactify --split train --split test  # skip val
    python scripts/download_defactify.py --out data/defactify --max-per-split 5000        # quick test

This script requires:  pip install datasets pillow tqdm
(datasets and pillow are in requirements/dev.lock.txt)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

DATASET_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
LABEL_A_MAP = {0: "REAL", 1: "FAKE"}
HF_SPLIT_MAP = {
    # HF split name → local folder name
    "train": "train",
    "validation": "val",
    "test": "test",
}


def download(out: Path, splits: list[str], max_per_split: int | None) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install the datasets library first:  pip install datasets")

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    for hf_split in splits:
        local_name = HF_SPLIT_MAP.get(hf_split, hf_split)
        print(f"\n→ Downloading split '{hf_split}' → {out / local_name}/")
        ds = load_dataset(DATASET_ID, split=hf_split, streaming=True)

        counters: dict[str, int] = {"REAL": 0, "FAKE": 0}
        for cls in ("REAL", "FAKE"):
            (out / local_name / cls).mkdir(parents=True, exist_ok=True)

        total = max_per_split or None
        rows = iter(ds)
        if tqdm is not None:
            rows = tqdm(rows, desc=hf_split, total=total, unit="img")

        done = 0
        for row in rows:
            label_a = row["Label_A"]
            cls = LABEL_A_MAP.get(label_a)
            if cls is None:
                continue  # unexpected label; skip silently

            img = row["Image"]
            # img may be a PIL Image or a dict with 'bytes' / 'path'
            if hasattr(img, "save"):
                pil = img
            elif isinstance(img, dict) and img.get("bytes"):
                from PIL import Image as PILImage

                pil = PILImage.open(io.BytesIO(img["bytes"])).convert("RGB")
            else:
                continue  # unrecognised image format

            idx = counters[cls]
            counters[cls] += 1
            fname = out / local_name / cls / f"{cls}_{idx:06d}.jpg"
            pil.convert("RGB").save(fname, format="JPEG", quality=95)
            done += 1
            if max_per_split and done >= max_per_split:
                break

        total_written = sum(counters.values())
        print(f"   Written: REAL={counters['REAL']:,}  FAKE={counters['FAKE']:,}  "
              f"total={total_written:,}")
        if counters["REAL"] == 0 or counters["FAKE"] == 0:
            print("   WARNING: one class has 0 images — check the Label_A mapping.")

    print(f"\nDone. Data is in: {out.resolve()}")
    print("Next: set data.root in your config to", out.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", default="data/defactify", help="Output root folder")
    p.add_argument("--split", action="append", dest="splits",
                   help="HF split to download (train / validation / test). "
                        "Repeat to download multiple. Default: all three.")
    p.add_argument("--max-per-split", type=int, default=None,
                   help="Stop after this many images per split (for quick tests)")
    args = p.parse_args()
    splits = args.splits or list(HF_SPLIT_MAP.keys())
    unknown = [s for s in splits if s not in HF_SPLIT_MAP]
    if unknown:
        p.error(f"Unknown split(s): {unknown}. Valid: {list(HF_SPLIT_MAP)}")
    download(Path(args.out), splits, args.max_per_split)


if __name__ == "__main__":
    main()
