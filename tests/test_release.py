import pytest

pytest.importorskip("torch")
import json
import shutil


from verisight_ai.config import ReleaseConfig
from verisight_ai.release import promote, rollback, run_release_gate


def test_gate_passes_for_valid_bundle(bundle_dir):
    gate = run_release_gate(bundle_dir, ReleaseConfig(min_test_roc_auc=0.9, min_test_f1=0.8))
    assert gate["passed"], gate


def test_gate_fails_on_low_metric(bundle_dir):
    gate = run_release_gate(bundle_dir, ReleaseConfig(min_test_roc_auc=0.99, min_test_f1=None))
    assert not gate["passed"]
    assert any(c["check"] == "min_test_roc_auc" and not c["passed"] for c in gate["checks"])


def test_gate_fails_without_traceability(bundle_dir):
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    meta["mlflow_run_id"] = None
    (bundle_dir / "metadata.json").write_text(json.dumps(meta))
    assert not run_release_gate(bundle_dir, ReleaseConfig())["passed"]


def test_promote_then_rollback(bundle_dir, tmp_path):
    art = tmp_path / "artifacts"
    r1 = promote(bundle_dir, ReleaseConfig(), art)
    assert r1["release_id"] == "test-release-1" and r1["previous_release_id"] is None
    second = tmp_path / "second"
    shutil.copytree(bundle_dir, second)
    meta = json.loads((second / "metadata.json").read_text())
    meta["model_release_id"] = "test-release-2"
    (second / "metadata.json").write_text(json.dumps(meta))
    r2 = promote(second, ReleaseConfig(), art)
    assert r2["previous_release_id"] == "test-release-1"
    back = rollback(art)
    assert back["release_id"] == "test-release-1"
    assert len((art / "release_history.jsonl").read_text().splitlines()) == 3


def test_promote_blocked_when_gate_fails(bundle_dir, tmp_path):
    with pytest.raises(RuntimeError, match="gate failed"):
        promote(bundle_dir, ReleaseConfig(min_test_roc_auc=0.999), tmp_path / "a")
    assert not (tmp_path / "a" / "current.json").exists()


def test_rollback_without_previous(bundle_dir, tmp_path):
    promote(bundle_dir, ReleaseConfig(), tmp_path / "a")
    with pytest.raises(RuntimeError, match="previous"):
        rollback(tmp_path / "a")
