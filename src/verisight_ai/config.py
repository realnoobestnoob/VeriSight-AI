"""Typed configuration: pipeline settings (YAML) and runtime app settings (environment)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CLASS_NAMES = ["REAL", "FAKE"]  # index 0 = authentic photo class, index 1 = AI-generated class
CLASS_MAPPING = {name: i for i, name in enumerate(CLASS_NAMES)}
SYNTHETIC_INDEX = CLASS_MAPPING["FAKE"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    root: Path = Path("data/defactify")
    dataset_name: str = "Defactify"
    dataset_version: str = "unspecified"
    dataset_source: str = ""
    val_fraction: float = Field(0.15, gt=0.0, lt=0.5)
    test_fraction: float = Field(0.15, gt=0.0, lt=0.5)
    max_per_class: int | None = Field(None, gt=0)
    near_duplicate_max_hamming: int | None = Field(None, ge=0, le=16)
    external_root: Path | None = None
    external_name: str | None = None
    extensions: list[str] = [".jpg", ".jpeg", ".png", ".webp"]


class ModelConfig(_Strict):
    architecture: Literal["mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"] = (
        "mobilenet_v3_small"
    )
    pretrained: bool = True
    image_size: int = Field(128, ge=32, le=512)


class StageConfig(_Strict):
    epochs: int = Field(3, ge=0, le=200)
    lr: float = Field(1e-3, gt=0)


class FinetuneConfig(StageConfig):
    enabled: bool = True
    unfreeze_last_blocks: int = Field(4, ge=1, le=20)


class TrainConfig(_Strict):
    batch_size: int = Field(64, ge=1, le=1024)
    num_workers: int = Field(0, ge=0, le=16)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    weight_decay: float = Field(1e-4, ge=0)
    head: StageConfig = StageConfig(epochs=3, lr=1e-3)
    finetune: FinetuneConfig = FinetuneConfig(epochs=5, lr=1e-4)
    early_stopping_patience: int = Field(2, ge=1, le=50)
    class_weighting: bool = False


class CalibrationConfig(_Strict):
    enabled: bool = True


class DecisionConfig(_Strict):
    abstain_margin: float = Field(0.10, ge=0.0, lt=0.5)


class EvaluationConfig(_Strict):
    robustness_max_images: int = Field(1000, ge=1)
    latency_warmup: int = Field(10, ge=0)
    latency_runs: int = Field(100, ge=1)


class MlflowConfig(_Strict):
    enabled: bool = True
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "VeriSight AI"


class ReleaseConfig(_Strict):
    min_test_roc_auc: float | None = 0.90
    min_test_f1: float | None = 0.80
    max_cpu_latency_p95_ms: float | None = None
    require_external_eval: bool = False


class PipelineConfig(_Strict):
    pipeline_version: str = "0.1.0"
    seed: int = 42
    output_dir: Path = Path("training_outputs")
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    decision: DecisionConfig = DecisionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    mlflow: MlflowConfig = MlflowConfig()
    release: ReleaseConfig = ReleaseConfig()

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("seed must be >= 0")
        return v


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config. Raises ValueError with an actionable message."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    try:
        return PipelineConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError -> friendly message
        raise ValueError(f"Invalid config {p}:\n{exc}") from exc


class AppSettings(BaseModel):
    """Runtime settings for the web app, read from environment variables."""

    max_upload_mb: float = 8.0
    max_pixels: int = 25_000_000
    max_text_chars: int = 2000
    model_dir: str | None = None
    artifacts_dir: str = "model_artifacts"
    database_url: str | None = None
    llm_provider: Literal["none", "openai_compatible", "anthropic"] = "none"
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout_s: float = 20.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> AppSettings:
        e = os.environ if env is None else env

        def get(name: str) -> str | None:
            v = e.get(f"VERISIGHT_{name}")
            return v.strip() if v and v.strip() else None

        data: dict[str, object] = {}
        mapping = {
            "max_upload_mb": ("MAX_UPLOAD_MB", float),
            "max_pixels": ("MAX_PIXELS", int),
            "max_text_chars": ("MAX_TEXT_CHARS", int),
            "model_dir": ("MODEL_DIR", str),
            "artifacts_dir": ("ARTIFACTS_DIR", str),
            "database_url": ("DATABASE_URL", str),
            "llm_provider": ("LLM_PROVIDER", str),
            "llm_api_key": ("LLM_API_KEY", str),
            "llm_model": ("LLM_MODEL", str),
            "llm_base_url": ("LLM_BASE_URL", str),
            "llm_timeout_s": ("LLM_TIMEOUT_S", float),
            "log_level": ("LOG_LEVEL", str),
        }
        for field, (name, cast) in mapping.items():
            raw = get(name)
            if raw is not None:
                try:
                    data[field] = cast(raw)
                except ValueError as exc:
                    raise ValueError(f"VERISIGHT_{name} has an invalid value") from exc
        return cls.model_validate(data)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
