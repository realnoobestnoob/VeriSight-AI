"""End-to-end smoke on the tiny SYNTHETIC dataset. Verifies plumbing only, not model quality."""

import json
from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from verisight_ai.cli import main  # noqa: E402
from verisight_ai.config import ReleaseConfig  # noqa: E402
from verisight_ai.inference import Predictor  # noqa: E402
from verisight_ai.release import run_release_gate  # noqa: E402

pytestmark = pytest.mark.slow


def test_demo_pipeline_end_to_end(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"output_dir: {tmp_path / 'out'}\nmlflow:\n  enabled: false\n")
    assert main(["pipeline", "--config", str(cfg), "--demo"]) == 0
    summary = json.loads(capsys.readouterr().out)
    run_dir, bundle = Path(summary["run_dir"]), Path(summary["candidate"])
    for f in (
        "evaluation_report.json",
        "training_history.json",
        "best_checkpoint.pt",
        "data/manifest.csv",
    ):
        assert (run_dir / f).is_file(), f
    for f in ("model.pt", "metadata.json", "model_card.md"):
        assert (bundle / f).is_file(), f
    report = json.loads((run_dir / "evaluation_report.json").read_text())
    assert "internal_test" in report["metrics"] and "jpeg_q75" in report["robustness"]["results"]
    from PIL import Image

    assert Predictor(bundle).predict(Image.new("RGB", (40, 40))).model_release_id
    # With MLflow disabled the traceability check must fail, so the gate refuses promotion.
    gate = run_release_gate(bundle, ReleaseConfig(min_test_roc_auc=None, min_test_f1=None))
    assert not gate["passed"]
    assert any(
        c["check"] == "traceability_mlflow_run_id" and not c["passed"] for c in gate["checks"]
    )
