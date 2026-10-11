"""Thin MLflow wrapper. All logging is explicit (no autologging) and can be disabled for tests."""

from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

from .config import MlflowConfig

log = logging.getLogger(__name__)


def git_commit() -> str | None:
    """Git HEAD, or VERISIGHT_GIT_COMMIT when not in a checkout (e.g. Colab without .git)."""
    import os

    env = os.environ.get("VERISIGHT_GIT_COMMIT")
    if env:
        return env.strip()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - not a git checkout / git missing
        return None


def package_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pillow",
        "scikit-learn",
        "mlflow",
        "streamlit",
        "pydantic",
        "sqlalchemy",
    ]
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            continue
    return out


class NullTracker:
    """No-op tracker used when MLflow is disabled or unavailable."""

    run_id: str | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_params(self, params: dict[str, Any]) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None: ...
    def log_dict(self, d: dict, name: str) -> None: ...


class MlflowTracker(NullTracker):
    def __init__(self, cfg: MlflowConfig, run_name: str):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)
        self._run_name = run_name
        self.run_id = None

    def __enter__(self):
        run = self.mlflow.start_run(run_name=self._run_name)
        self.run_id = run.info.run_id
        return self

    def __exit__(self, exc_type, exc, tb):
        self.mlflow.end_run(status="FAILED" if exc_type else "FINISHED")
        return False

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: ("" if v is None else str(v))[:500] for k, v in params.items()}
        self.mlflow.log_params(flat)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.mlflow.set_tags({k: str(v)[:5000] for k, v in tags.items()})

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.mlflow.log_metric(key, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            if v is not None:
                self.mlflow.log_metric(k, float(v), step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        self.mlflow.log_artifact(str(path), artifact_path)

    def log_dict(self, d: dict, name: str) -> None:
        self.mlflow.log_dict(d, name)


def make_tracker(cfg: MlflowConfig, run_name: str) -> NullTracker:
    if not cfg.enabled:
        return NullTracker()
    try:
        return MlflowTracker(cfg, run_name)
    except ImportError:
        log.warning("mlflow is not installed; experiment tracking is disabled for this run.")
        return NullTracker()
