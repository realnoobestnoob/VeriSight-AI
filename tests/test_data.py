import shutil
from pathlib import Path

import pytest

from verisight_ai.config import DataConfig
from verisight_ai.data import (
    DataValidationError,
    build_manifest,
    manifest_hash,
    save_manifest,
)
from verisight_ai.demo_data import make_demo_dataset


@pytest.fixture
def demo_root(tmp_path):
    return make_demo_dataset(tmp_path / "demo", n_per_class=40, size=16)


def test_manifest_splits_and_no_overlap(demo_root):
    m, rep = build_manifest(DataConfig(root=demo_root), seed=1)
    assert set(m.split) == {"train", "val", "test"}
    sets = {s: set(m[m.split == s].sha256) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"])
    assert rep.counts["train"]["REAL"] > 0 and rep.counts["test"]["FAKE"] > 0


def test_manifest_is_reproducible(demo_root):
    a, _ = build_manifest(DataConfig(root=demo_root), seed=7)
    b, _ = build_manifest(DataConfig(root=demo_root), seed=7)
    assert manifest_hash(a) == manifest_hash(b)


def test_cross_split_duplicate_removed_from_train(demo_root):
    src = next((demo_root / "test" / "REAL").iterdir())
    shutil.copy(src, demo_root / "train" / "REAL" / "leak.png")
    m, rep = build_manifest(DataConfig(root=demo_root), seed=1)
    assert rep.cross_split_duplicates_removed >= 1
    assert "train/REAL/leak.png" not in set(m.path)


def test_corrupt_file_reported(demo_root):
    (demo_root / "train" / "FAKE" / "bad.png").write_bytes(b"not an image")
    m, rep = build_manifest(DataConfig(root=demo_root), seed=1)
    assert any("bad.png" in p for p in rep.corrupt_files)
    assert "train/FAKE/bad.png" not in set(m.path)


def test_missing_class_folder(tmp_path):
    (tmp_path / "train" / "REAL").mkdir(parents=True)
    (tmp_path / "test" / "REAL").mkdir(parents=True)
    with pytest.raises(DataValidationError, match="Missing class folder"):
        build_manifest(DataConfig(root=tmp_path), seed=1)


def test_missing_root_has_actionable_message(tmp_path):
    with pytest.raises(DataValidationError, match="download_defactify"):
        build_manifest(DataConfig(root=tmp_path / "nope"), seed=1)


def test_save_manifest_writes_hash(demo_root, tmp_path):
    cfg = DataConfig(root=demo_root)
    m, rep = build_manifest(cfg, seed=1)
    meta = save_manifest(m, rep, cfg, Path(tmp_path / "out"))
    assert meta["manifest_sha256"] == manifest_hash(m)
    assert (tmp_path / "out" / "manifest.csv").is_file()
