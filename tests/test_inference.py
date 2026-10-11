import pytest

pytest.importorskip("torch")
import json

from conftest import make_image

from verisight_ai.inference import ModelUnavailableError, Predictor, resolve_bundle_dir
from verisight_ai.schemas import DecisionStatus


def test_predict_returns_valid_result(bundle_dir):
    res = Predictor(bundle_dir).predict(make_image())
    assert 0.0 <= res.raw_score <= 1.0 and 0.0 <= res.score_used <= 1.0
    assert res.calibrated_score is not None and res.score_label == "calibrated"
    assert res.model_release_id == "test-release-1" and res.training_run_id == "run123"
    assert isinstance(res.decision_status, DecisionStatus) and res.latency_ms > 0


def test_uncalibrated_bundle_reports_raw(bundle_dir):
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    meta["calibration"] = {"applied": False}
    (bundle_dir / "metadata.json").write_text(json.dumps(meta))
    res = Predictor(bundle_dir).predict(make_image())
    assert res.calibrated_score is None and res.score_label == "raw"


def test_checksum_mismatch_refuses_to_load(bundle_dir):
    with (bundle_dir / "model.pt").open("ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(ModelUnavailableError, match="checksum"):
        Predictor(bundle_dir)


def test_missing_bundle():
    with pytest.raises(ModelUnavailableError):
        Predictor("does/not/exist")


def test_no_promoted_release(tmp_path):
    with pytest.raises(ModelUnavailableError, match="No promoted"):
        resolve_bundle_dir(None, str(tmp_path))


def test_explain_returns_image_of_model_size(bundle_dir):
    p = Predictor(bundle_dir)
    overlay = p.explain(make_image((40, 60)))
    assert overlay.size == (32, 32)
