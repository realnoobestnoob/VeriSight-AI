"""Candidate packaging, release gate, controlled promotion and rollback."""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import CLASS_MAPPING, ReleaseConfig
from .inference import ModelUnavailableError, Predictor, sha256_file
from .preprocessing import IMAGENET_MEAN, IMAGENET_STD
from .tracking import git_commit, package_versions

log = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """# Model card: {release_id}

**Intended use:** triage aid that estimates whether an image resembles the AI-generated class
seen in training ({dataset_name}). It is not an authenticity oracle or a fraud detector.

**Architecture:** {architecture} ({weights}), input {image_size}x{image_size}.
**Dataset:** {dataset_name} / {dataset_version}; manifest sha256 `{manifest_sha256}`.
**Score type:** {score_type}. Decision threshold {threshold:.3f} with abstain margin {margin:.2f}.
**MLflow run:** {run_id}. **Git commit:** {commit}.

## Measured results (from the run that produced this bundle)
{metrics_md}

## Out of scope / not demonstrated
- Detecting every kind of deepfake, face swap, or manipulated media.
- Proving that a document, offer or claim is genuine or fraudulent.
- Concluding an image is authentic because the score is low.
- Real-world performance: results on {dataset_name} reflect a narrow benchmark domain.

## Known limitations
{limitations}
"""


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def make_release_id(arch: str, manifest_sha: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{arch}-{stamp}-{manifest_sha[:8]}"


def package_candidate(
    *,
    out_dir: Path,
    state_dict: dict,
    cfg,
    manifest_meta: dict,
    calibration: dict,
    decision: dict,
    metrics: dict,
    run_id: str | None,
    pretrained_id: str,
) -> Path:
    manifest_sha = manifest_meta["manifest_sha256"]
    release_id = make_release_id(cfg.model.architecture, manifest_sha)
    bundle = out_dir / "candidate" / release_id
    bundle.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, bundle / "model.pt")
    commit = git_commit()
    meta = {
        "model_release_id": release_id,
        "architecture": cfg.model.architecture,
        "pretrained_weights": pretrained_id,
        "class_mapping": CLASS_MAPPING,
        "preprocessing": {
            "image_size": cfg.model.image_size,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "color": "RGB",
            "resize": "bilinear",
        },
        "calibration": calibration,
        "decision": decision,
        "dataset": {
            k: manifest_meta[k]
            for k in ("dataset_name", "dataset_version", "dataset_source", "manifest_sha256")
        },
        "mlflow_run_id": run_id,
        "git_commit": commit,
        "pipeline_version": cfg.pipeline_version,
        "package_versions": package_versions(),
        "artifact_sha256": sha256_file(bundle / "model.pt"),
        "metrics_summary": metrics,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "promoted": False,
    }
    (bundle / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    rows = []
    for name, m in metrics.items():
        if isinstance(m, dict) and "f1" in m:
            rows.append(
                f"- **{name}** (n={m['n']}): precision {_fmt(m['precision'])}, recall "
                f"{_fmt(m['recall'])}, F1 {_fmt(m['f1'])}, ROC-AUC {_fmt(m.get('roc_auc'))}, "
                f"PR-AUC {_fmt(m.get('pr_auc'))}, Brier {_fmt(m['brier'])}"
            )
    limitations = "\n".join(f"- {x}" for x in manifest_meta.get("known_limitations", []))
    (bundle / "model_card.md").write_text(
        MODEL_CARD_TEMPLATE.format(
            release_id=release_id,
            dataset_name=cfg.data.dataset_name,
            dataset_version=cfg.data.dataset_version,
            architecture=cfg.model.architecture,
            weights=pretrained_id,
            image_size=cfg.model.image_size,
            manifest_sha256=manifest_sha,
            score_type="calibrated probability (temperature scaling, fitted on validation)"
            if calibration.get("applied")
            else "raw softmax score (NOT calibrated)",
            threshold=decision["threshold"],
            margin=decision["abstain_margin"],
            run_id=run_id or "not tracked",
            commit=commit or "unknown",
            metrics_md="\n".join(rows) or "No metrics recorded.",
            limitations=limitations,
        ),
        encoding="utf-8",
    )
    return bundle


def run_release_gate(bundle: Path, rcfg: ReleaseConfig) -> dict:
    """Pre-defined acceptance checks. Returns {passed: bool, checks: [...]}; never promotes."""
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    bundle = Path(bundle)
    meta_path = bundle / "metadata.json"
    if not meta_path.is_file():
        return {
            "passed": False,
            "checks": [{"check": "metadata_present", "passed": False, "detail": str(meta_path)}],
        }
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    add("model_card_present", (bundle / "model_card.md").is_file())
    add("checksum_matches", sha256_file(bundle / "model.pt") == meta.get("artifact_sha256"))
    for key in ("mlflow_run_id", "git_commit"):
        add(f"traceability_{key}", bool(meta.get(key)), "missing" if not meta.get(key) else "")
    try:
        predictor = Predictor(bundle)
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        res = predictor.predict(Image.fromarray(arr))
        ok = bool(np.isfinite(res.raw_score) and 0 <= res.raw_score <= 1 and res.model_release_id)
        add("smoke_inference", ok, f"score={res.raw_score:.4f}")
    except ModelUnavailableError as exc:
        add("smoke_inference", False, str(exc))
    m = meta.get("metrics_summary", {})
    test = m.get("internal_test", {})
    if rcfg.min_test_roc_auc is not None:
        v = test.get("roc_auc")
        add(
            "min_test_roc_auc",
            v is not None and v >= rcfg.min_test_roc_auc,
            f"{v} (required >= {rcfg.min_test_roc_auc})",
        )
    if rcfg.min_test_f1 is not None:
        v = test.get("f1")
        add(
            "min_test_f1",
            v is not None and v >= rcfg.min_test_f1,
            f"{v} (required >= {rcfg.min_test_f1})",
        )
    if rcfg.max_cpu_latency_p95_ms is not None:
        v = m.get("cpu_latency", {}).get("p95_ms")
        add(
            "max_cpu_latency_p95_ms",
            v is not None and v <= rcfg.max_cpu_latency_p95_ms,
            f"{v} (required <= {rcfg.max_cpu_latency_p95_ms})",
        )
    if rcfg.require_external_eval:
        add("external_eval_present", "external_test" in m)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def promote(
    bundle: Path, rcfg: ReleaseConfig, artifacts_dir: str | Path = "model_artifacts"
) -> dict:
    """Controlled promotion: re-runs the gate, copies the bundle, records the previous release."""
    gate = run_release_gate(bundle, rcfg)
    if not gate["passed"]:
        failed = [c["check"] for c in gate["checks"] if not c["passed"]]
        raise RuntimeError(f"Release gate failed; not promoting. Failed checks: {failed}")
    art = Path(artifacts_dir)
    meta = json.loads((Path(bundle) / "metadata.json").read_text(encoding="utf-8"))
    rid = meta["model_release_id"]
    dest = art / "releases" / rid
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(bundle, dest)
    meta["promoted"] = True
    (dest / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # promoted flag changes metadata.json only; model.pt checksum is unchanged
    pointer = art / "current.json"
    previous = None
    if pointer.is_file():
        previous = json.loads(pointer.read_text(encoding="utf-8")).get("release_id")
    info = {
        "release_id": rid,
        "previous_release_id": previous,
        "promoted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    pointer.write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (art / "release_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "promote", **info}) + "\n")
    return info


def rollback(artifacts_dir: str | Path = "model_artifacts") -> dict:
    art = Path(artifacts_dir)
    pointer = art / "current.json"
    if not pointer.is_file():
        raise RuntimeError("No current release to roll back from.")
    cur = json.loads(pointer.read_text(encoding="utf-8"))
    prev = cur.get("previous_release_id")
    if not prev or not (art / "releases" / prev / "metadata.json").is_file():
        raise RuntimeError("No previous release available to roll back to.")
    info = {
        "release_id": prev,
        "previous_release_id": cur["release_id"],
        "promoted_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    pointer.write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (art / "release_history.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "rollback", **info}) + "\n")
    return info
