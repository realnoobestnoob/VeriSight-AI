"""Single orchestrator: validate -> data -> train -> calibrate/threshold -> evaluate -> package.

Order note: calibration and threshold selection use validation data only and run BEFORE the
final test evaluation, so the test set is evaluated exactly once with the frozen policy.
Each pipeline invocation writes to a fresh run directory, so earlier test evidence is never
overwritten.

Resume: if a run fails mid-pipeline, re-run with resume_dir pointing at the failed run's
directory (e.g. training_outputs/run-20261013-154652).  checkpoint.json tracks the last
successfully completed stage; completed stages are skipped and their outputs are loaded from
disk.  The pipeline picks up exactly where it left off.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from .calibration import choose_threshold, fit_temperature, softmax_p_synthetic
from .config import CLASS_MAPPING, PipelineConfig
from .data import (
    DataValidationError,
    build_manifest,
    load_external_manifest,
    save_manifest,
)
from .evaluate import (
    ROBUSTNESS_PROTOCOL,
    benchmark_cpu_latency,
    evaluate_split,
    robustness_suite,
    save_plots,
)
from .model import build_model, pretrained_weight_id, select_device
from .release import package_candidate
from .timing import (
    PipelineTimer,
    estimate_pipeline_seconds,
    print_pipeline_header,
)
from .tracking import git_commit, make_tracker, package_versions
from .train import collect_logits, make_loader, run_training

log = logging.getLogger(__name__)


def validate_environment(cfg: PipelineConfig) -> torch.device:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    probe = out / ".write_test"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(f"Output directory {out} is not writable: {exc}") from exc
    device = select_device(cfg.train.device)
    log.info(
        "Device: %s | torch %s | python %s",
        device,
        torch.__version__,
        platform.python_version(),
    )
    return device


def _write_json(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

_CKPT_FILE = "checkpoint.json"


def _write_checkpoint(run_dir: Path, stage: int, run_ts: str) -> None:
    """Atomically record the last successfully completed stage.

    Uses a tmp-file + rename so a crash mid-write never leaves a partial file.
    """
    tmp = run_dir / (_CKPT_FILE + ".tmp")
    tmp.write_text(json.dumps({"last_stage": stage, "run_ts": run_ts}, indent=2), encoding="utf-8")
    tmp.replace(run_dir / _CKPT_FILE)


def _read_checkpoint(run_dir: Path) -> dict:
    p = run_dir / _CKPT_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {"last_stage": 0}


# ── Per-stage artifact loaders (used on resume) ────────────────────────────────


def _resume_stage1(run_dir: Path):
    """Reload manifest DataFrame and manifest_meta dict from stage-1 disk output."""
    import pandas as pd

    data_dir = run_dir / "data"
    manifest = pd.read_csv(data_dir / "manifest.csv")
    manifest_meta = json.loads((data_dir / "manifest_meta.json").read_text(encoding="utf-8"))
    return manifest, manifest_meta


def _resume_stage2(run_dir: Path, cfg: PipelineConfig, device):
    """Reload model (on device) and raw state_dict from stage-2 disk output."""
    state_dict = torch.load(run_dir / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    model = build_model(cfg.model.architecture, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, state_dict


def _resume_stage3(run_dir: Path) -> tuple[dict, float, dict]:
    """Reload calibration and decision from stage-3 disk output."""
    data = json.loads((run_dir / "calibration_decision.json").read_text(encoding="utf-8"))
    calibration = data["calibration"]
    decision = data["decision"]
    temperature = calibration["temperature"] if calibration.get("applied") else 1.0
    return calibration, temperature, decision


def _resume_stage4(run_dir: Path) -> dict:
    """Reload metrics dict from stage-4 disk output."""
    report = json.loads((run_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    return report["metrics"]


# ── Pipeline orchestrator ──────────────────────────────────────────────────────


def run_pipeline(cfg: PipelineConfig, resume_dir: Path | None = None) -> dict:
    device = validate_environment(cfg)
    device_str = device.type  # "cpu" | "cuda" | "mps"

    # ── Resolve run directory ──────────────────────────────────────────────────
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.is_dir():
            raise RuntimeError(f"Resume directory not found: {run_dir}")
        cp = _read_checkpoint(run_dir)
        last_done = cp.get("last_stage", 0)
        run_ts = cp.get("run_ts", run_dir.name.removeprefix("run-"))
        log.info("Resuming %s — last completed stage: %d/5", run_dir.name, last_done)
        print(
            f"\nResuming {run_dir.name}"
            f"  —  stages 1–{last_done} already done,"
            f" continuing from stage {last_done + 1} ...",
            file=sys.stderr,
        )
        if last_done >= 5:
            log.info("Run already complete; returning existing summary.")
            summary_path = run_dir / "run_summary.json"
            if summary_path.is_file():
                return json.loads(summary_path.read_text(encoding="utf-8"))
            raise RuntimeError(
                f"checkpoint says stage 5 done but run_summary.json is missing in {run_dir}"
            )
    else:
        run_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg.output_dir) / f"run-{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=False)
        last_done = 0

    # ── Stage 1: data validation ───────────────────────────────────────────────
    import time as _time

    _t0 = _time.perf_counter()  # always record; only used for retroactive timing on fresh runs

    if last_done < 1:
        log.info("[1/6] Data validation and split manifests")
        print("\n[1/6] Data validation ...", file=sys.stderr)
        manifest, report = build_manifest(cfg.data, cfg.seed)
        manifest_meta = save_manifest(manifest, report, cfg.data, run_dir / "data")
        for w in report.warnings:
            log.warning(w)
    else:
        print("\n[1/6] Data validation — skipped (resuming)", file=sys.stderr)
        manifest, manifest_meta = _resume_stage1(run_dir)

    root = Path(cfg.data.root)

    # Image counts are needed for estimates/timer regardless of resume state.
    n_train = int((manifest.split == "train").sum())
    n_val = int((manifest.split == "val").sum())
    n_test = int((manifest.split == "test").sum())

    estimates = estimate_pipeline_seconds(
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=cfg.train.batch_size,
        head_epochs=cfg.train.head.epochs,
        finetune_epochs=cfg.train.finetune.epochs if cfg.train.finetune.enabled else 0,
        latency_runs=cfg.evaluation.latency_runs,
        robustness_images=cfg.evaluation.robustness_max_images,
        device_str=device_str,
    )
    if last_done == 0:
        # Fresh run: print the full header with time estimates.
        print_pipeline_header(n_train, n_val, n_test, device_str, cfg.model.architecture, estimates)

    timer = PipelineTimer(estimates)

    if last_done < 1:
        # Retroactively record stage 1 elapsed time (build_manifest ran above).
        _stage1_elapsed = _time.perf_counter() - _t0
        from .timing import StageResult, _fmt

        timer._results.append(StageResult("data_validation", _stage1_elapsed))
        elapsed_total = _time.perf_counter() - timer.start
        remaining = max(0, estimates["total"] - elapsed_total)
        print(
            f"  [stage 1/6] done in {_fmt(_stage1_elapsed)}"
            f"  |  elapsed {_fmt(elapsed_total)}"
            f"  |  est. remaining ~{_fmt(remaining)}",
            file=sys.stderr,
        )
        _write_checkpoint(run_dir, 1, run_ts)

    tracker = make_tracker(cfg.mlflow, run_name=f"{cfg.model.architecture}-{run_ts}")
    with tracker:
        tracker.set_tags(
            {
                "run_purpose": "candidate_training",
                "git_commit": git_commit() or "unknown",
                "pipeline_version": cfg.pipeline_version,
                "device": str(device),
                "dataset": f"{cfg.data.dataset_name}:{cfg.data.dataset_version}",
            }
        )
        tracker.log_params(
            {
                "architecture": cfg.model.architecture,
                "pretrained_weights": pretrained_weight_id(
                    cfg.model.architecture, cfg.model.pretrained
                ),
                "image_size": cfg.model.image_size,
                "seed": cfg.seed,
                "batch_size": cfg.train.batch_size,
                "weight_decay": cfg.train.weight_decay,
                "head_lr": cfg.train.head.lr,
                "head_epochs": cfg.train.head.epochs,
                "finetune_enabled": cfg.train.finetune.enabled,
                "finetune_lr": cfg.train.finetune.lr,
                "finetune_epochs": cfg.train.finetune.epochs,
                "unfreeze_last_blocks": cfg.train.finetune.unfreeze_last_blocks,
                "early_stopping_patience": cfg.train.early_stopping_patience,
                "optimizer": "AdamW",
                "class_mapping": CLASS_MAPPING,
                "manifest_sha256": manifest_meta["manifest_sha256"],
                "augmentation_train": "RandomResizedCrop(0.8-1.0),HFlip,ColorJitter(0.1)",
                "augmentation_eval": "none (resize + normalise)",
                "package_versions": package_versions(),
            }
        )
        tracker.log_dict(cfg.model_dump(mode="json"), "pipeline_config.json")
        tracker.log_artifact(run_dir / "data" / "manifest.csv", "data")
        tracker.log_artifact(run_dir / "data" / "manifest_meta.json", "data")

        # ── Stage 2: head training ─────────────────────────────────────────────
        if last_done < 2:
            log.info("[2/6] Head training")
            print("\n[2/6] Head training ...", file=sys.stderr)
            timer.begin_stage("head_training")
            result = run_training(cfg, manifest, root, tracker, device)
            timer.end_stage(
                "head_training",
                "head training",
                2,
                6,
                note=f"best val_loss={result.best_val_loss:.4f}"
                if result.best_val_loss < float("inf")
                else "",
            )
            _write_json(
                run_dir / "training_history.json",
                {"history": result.history, "summary": result.stage_summary},
            )
            tracker.log_artifact(run_dir / "training_history.json")

            model = build_model(cfg.model.architecture, pretrained=False)
            model.load_state_dict(result.state_dict)
            model.to(device).eval()
            torch.save(
                {k: v.cpu() for k, v in result.state_dict.items()},
                run_dir / "best_checkpoint.pt",
            )
            tracker.log_artifact(run_dir / "best_checkpoint.pt")
            state_dict = result.state_dict  # kept for package_candidate in stage 5
            _write_checkpoint(run_dir, 2, run_ts)
        else:
            print("\n[2/6] Training — skipped (resuming)", file=sys.stderr)
            model, state_dict = _resume_stage2(run_dir, cfg, device)

        # ── Stage 3: calibration and decision policy ───────────────────────────
        #    run_training already handles both head + finetune internally; this
        #    stage fits temperature scaling and the decision threshold on val only.
        if last_done < 3:
            print("\n[3/6] Calibration and decision policy ...", file=sys.stderr)
            timer.begin_stage("calibration")
            val_df = manifest[manifest.split == "val"]
            val_loader = make_loader(val_df, root, cfg.model.image_size, False, cfg, shuffle=False)
            val_logits, val_labels = collect_logits(model, val_loader, device)
            if cfg.calibration.enabled:
                calibration = fit_temperature(val_logits, val_labels)
            else:
                calibration = {"method": "none", "temperature": 1.0, "applied": False}
            temperature = calibration["temperature"] if calibration["applied"] else 1.0
            val_scores = softmax_p_synthetic(val_logits, temperature)
            thr = choose_threshold(val_scores, val_labels)
            decision = {
                "threshold": thr["threshold"],
                "abstain_margin": cfg.decision.abstain_margin,
                "criterion": thr["criterion"],
            }
            tracker.log_params(
                {
                    "calibration_applied": calibration["applied"],
                    "temperature": round(temperature, 4),
                    "decision_threshold": decision["threshold"],
                    "abstain_margin": decision["abstain_margin"],
                }
            )
            cal_note = (
                f"T={temperature:.3f} applied"
                if calibration.get("applied")
                else "not applied (no NLL improvement)"
            )
            timer.end_stage("calibration", "calibration", 3, 6, note=cal_note)
            # Persist calibration + decision so they can be reloaded on resume.
            # (They are also written inside evaluation_report.json, but that file
            # is only written at the end of stage 4 — too late to use for stage-4
            # resume without re-running calibration.)
            _write_json(
                run_dir / "calibration_decision.json",
                {"calibration": calibration, "decision": decision},
            )
            _write_checkpoint(run_dir, 3, run_ts)
        else:
            print("\n[3/6] Calibration — skipped (resuming)", file=sys.stderr)
            calibration, temperature, decision = _resume_stage3(run_dir)

        # ── Stage 4: evaluation ────────────────────────────────────────────────
        if last_done < 4:
            log.info("[4/6] Evaluation")
            print("\n[4/6] Evaluation ...", file=sys.stderr)
            timer.begin_stage("evaluation")
            # val_df may not be defined if we resumed past stage 3; re-slice cheaply.
            val_df = manifest[manifest.split == "val"]
            val_metrics, _, _ = evaluate_split(
                model, val_df, root, cfg, device, temperature, decision["threshold"]
            )
            metrics: dict = {"validation": val_metrics}

            test_df = manifest[manifest.split == "test"]
            test_metrics, t_labels, t_scores = evaluate_split(
                model, test_df, root, cfg, device, temperature, decision["threshold"]
            )
            metrics["internal_test"] = test_metrics
            plots = save_plots(t_labels, t_scores, test_metrics, run_dir / "plots", "internal_test")

            if cfg.data.external_root:
                try:
                    ext_df = load_external_manifest(
                        Path(cfg.data.external_root), cfg.data.extensions
                    )
                    ext_metrics, e_labels, e_scores = evaluate_split(
                        model,
                        ext_df,
                        Path(cfg.data.external_root),
                        cfg,
                        device,
                        temperature,
                        decision["threshold"],
                    )
                    ext_metrics["dataset_name"] = cfg.data.external_name or "external"
                    metrics["external_test"] = ext_metrics
                    plots += save_plots(
                        e_labels, e_scores, ext_metrics, run_dir / "plots", "external_test"
                    )
                except DataValidationError as exc:
                    log.warning("External evaluation skipped: %s", exc)
            else:
                log.warning("No external dataset configured; domain shift has NOT been evaluated.")

            robustness = robustness_suite(
                model, test_df, root, cfg, device, temperature, decision["threshold"]
            )
            metrics["robustness_summary"] = {
                k: {"f1": v["f1"], "roc_auc": v["roc_auc"]}
                for k, v in robustness["results"].items()
            }
            latency = benchmark_cpu_latency(
                model,
                cfg.model.image_size,
                cfg.evaluation.latency_warmup,
                cfg.evaluation.latency_runs,
            )
            metrics["cpu_latency"] = latency

            report_path = _write_json(
                run_dir / "evaluation_report.json",
                {
                    "metrics": metrics,
                    "robustness": robustness,
                    "calibration": calibration,
                    "decision": decision,
                    "robustness_protocol": ROBUSTNESS_PROTOCOL,
                    "note": "Test metrics were computed once, after calibration/threshold were frozen.",
                },
            )

            for split in ("validation", "internal_test", "external_test"):
                if split in metrics:
                    m = metrics[split]
                    tracker.log_metrics(
                        {
                            f"{split}_{k}": m[k]
                            for k in (
                                "precision",
                                "recall",
                                "f1",
                                "roc_auc",
                                "pr_auc",
                                "brier",
                                "ece_10bin",
                            )
                            if m.get(k) is not None
                        }
                    )
            for name, v in metrics["robustness_summary"].items():
                if v["f1"] is not None:
                    tracker.log_metric(f"robustness_{name}_f1", v["f1"])
            tracker.log_metrics(
                {
                    "cpu_latency_mean_ms": latency["mean_ms"],
                    "cpu_latency_p95_ms": latency["p95_ms"],
                }
            )
            tracker.log_artifact(report_path)
            for p_plot in plots:
                tracker.log_artifact(p_plot, "plots")

            eval_note = (
                f"test F1={test_metrics.get('f1', 0):.3f} "
                f"AUC={test_metrics.get('roc_auc') or 0:.3f} "
                f"lat_p95={latency['p95_ms']:.0f}ms"
            )
            timer.end_stage("evaluation", "evaluation", 4, 6, note=eval_note)
            _write_checkpoint(run_dir, 4, run_ts)
        else:
            print("\n[4/6] Evaluation — skipped (resuming)", file=sys.stderr)
            metrics = _resume_stage4(run_dir)

        # ── Stage 5: package candidate ─────────────────────────────────────────
        log.info("[5/6] Package candidate")
        print("\n[5/6] Packaging candidate ...", file=sys.stderr)
        timer.begin_stage("packaging")
        bundle = package_candidate(
            out_dir=run_dir,
            state_dict=state_dict,
            cfg=cfg,
            manifest_meta=manifest_meta,
            calibration=calibration,
            decision=decision,
            metrics=metrics,
            run_id=tracker.run_id,
            pretrained_id=pretrained_weight_id(cfg.model.architecture, cfg.model.pretrained),
        )
        for f in ("metadata.json", "model_card.md"):
            tracker.log_artifact(bundle / f, "candidate")
        tracker.log_artifact(bundle / "model.pt", "candidate")
        timer.end_stage("packaging", "packaging", 5, 6)
        _write_checkpoint(run_dir, 5, run_ts)

        # ── Summary ────────────────────────────────────────────────────────────
        timer.print_summary(str(bundle))

    summary = {
        "run_dir": str(run_dir),
        "candidate": str(bundle),
        "mlflow_run_id": tracker.run_id,
        "internal_test": {k: metrics["internal_test"].get(k) for k in ("f1", "roc_auc")},
        "next_step": (f"python -m verisight_ai.cli release --candidate {bundle} --config <cfg>"),
    }
    _write_json(run_dir / "run_summary.json", summary)
    return summary


_ = np  # numpy kept for downstream type consumers
