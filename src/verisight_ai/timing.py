"""Pipeline stage timing and pre-run time estimates.

Estimates are rough order-of-magnitude guides, not guarantees. They scale with image count,
epoch counts and device type. Label them clearly as estimates when surfacing to the user.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

_DIV = "─" * 60

# Seconds per image-batch-pass on each device type (measured on typical hardware).
# CPU: Intel i7 10th-gen (4-core). GPU: T4 Colab.
_SECS_PER_BATCH = {"cpu": 0.30, "cuda": 0.025, "mps": 0.07}


def _fmt(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def estimate_pipeline_seconds(
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    head_epochs: int,
    finetune_epochs: int,
    latency_runs: int,
    robustness_images: int,
    device_str: str,
) -> dict[str, float]:
    """Return per-stage and total estimates in seconds."""
    spb = _SECS_PER_BATCH.get(device_str, _SECS_PER_BATCH["cpu"])
    n_total = n_train + n_val + n_test

    data_val = n_total * 0.0012  # hash + probe
    head = (n_train / batch_size) * head_epochs * spb + (
        n_val / batch_size
    ) * head_epochs * spb * 0.6
    finetune = (n_train / batch_size) * finetune_epochs * spb + (
        n_val / batch_size
    ) * finetune_epochs * spb * 0.6
    calib = (n_val / batch_size) * spb * 0.5 + 3
    eval_ = (
        (n_test / batch_size) * spb * 7
        + (min(robustness_images, n_test) / batch_size) * spb * 5
        + latency_runs * spb * 0.3
    )
    package = 8

    stages = {
        "data_validation": data_val,
        "head_training": head,
        "fine_tuning": finetune,
        "calibration": calib,
        "evaluation": eval_,
        "packaging": package,
    }
    stages["total"] = sum(stages.values())
    return stages


@dataclass
class StageResult:
    name: str
    elapsed: float
    note: str = ""


@dataclass
class PipelineTimer:
    estimates: dict[str, float]
    start: float = field(default_factory=time.perf_counter)
    _results: list[StageResult] = field(default_factory=list)
    _stage_start: float = field(default_factory=time.perf_counter)

    def begin_stage(self, label: str) -> None:
        self._stage_start = time.perf_counter()

    def end_stage(
        self, stage_key: str, display_label: str, stage_num: int, total_stages: int, note: str = ""
    ) -> None:
        elapsed = time.perf_counter() - self._stage_start
        self._results.append(StageResult(stage_key, elapsed, note))
        elapsed_total = time.perf_counter() - self.start
        remaining_est = max(0, self.estimates["total"] - elapsed_total)
        suffix = f"  [{note}]" if note else ""
        print(
            f"  [stage {stage_num}/{total_stages}] done in {_fmt(elapsed)}"
            f"{suffix}"
            f"  |  elapsed {_fmt(elapsed_total)}"
            f"  |  est. remaining ~{_fmt(remaining_est)}",
            file=sys.stderr,
        )

    def print_summary(self, candidate: str) -> None:
        total_elapsed = time.perf_counter() - self.start
        print(file=sys.stderr)
        print(_DIV, file=sys.stderr)
        print(f"  Pipeline complete in {_fmt(total_elapsed)}", file=sys.stderr)
        print(f"  Candidate: {candidate}", file=sys.stderr)
        print(file=sys.stderr)
        print(f"  {'Stage':<20}  {'Actual':>10}  {'Estimate':>10}", file=sys.stderr)
        print(f"  {'─' * 20}  {'─' * 10}  {'─' * 10}", file=sys.stderr)
        for r in self._results:
            est = self.estimates.get(r.name, 0)
            print(f"  {r.name:<20}  {_fmt(r.elapsed):>10}  ~{_fmt(est):>9}", file=sys.stderr)
        print(_DIV, file=sys.stderr)


def print_pipeline_header(
    n_train: int,
    n_val: int,
    n_test: int,
    device_str: str,
    architecture: str,
    estimates: dict[str, float],
) -> None:
    n_total = n_train + n_val + n_test
    print(file=sys.stderr)
    print(_DIV, file=sys.stderr)
    print("  VeriSight AI — training pipeline", file=sys.stderr)
    print(
        f"  Dataset : {n_total:,} images  (train {n_train:,} / val {n_val:,} / test {n_test:,})",
        file=sys.stderr,
    )
    print(f"  Model   : {architecture}  |  Device: {device_str.upper()}", file=sys.stderr)
    print(
        f"  Est. total  : ~{_fmt(estimates['total'])}  (rough guide; varies by hardware)",
        file=sys.stderr,
    )
    print(f"  {'Stage':<20}  {'Estimate':>10}", file=sys.stderr)
    print(f"  {'─' * 20}  {'─' * 10}", file=sys.stderr)
    stage_labels = {
        "data_validation": "1. data validation",
        "head_training": "2. head training",
        "fine_tuning": "3. fine-tuning",
        "calibration": "4. calibration",
        "evaluation": "5. evaluation",
        "packaging": "6. packaging",
    }
    for key, label in stage_labels.items():
        print(f"  {label:<20}  ~{_fmt(estimates[key]):>9}", file=sys.stderr)
    print(_DIV, file=sys.stderr)
    print(file=sys.stderr)
