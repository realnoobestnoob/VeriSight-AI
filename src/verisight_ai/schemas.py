"""Typed result and LLM-output schemas shared by the app, tests and the database layer."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DecisionStatus(str, Enum):
    SYNTHETIC_LIKE = "synthetic_like"
    NO_STRONG_SIGNAL = "no_strong_signal"
    INCONCLUSIVE = "inconclusive"


DECISION_LABELS = {
    DecisionStatus.SYNTHETIC_LIKE: "Synthetic-like signals detected",
    DecisionStatus.NO_STRONG_SIGNAL: "No strong synthetic signal detected",
    DecisionStatus.INCONCLUSIVE: "Inconclusive",
}

InputType = Literal["upload", "camera_capture", "screenshot_upload", "video_frame_sample"]
ScanStatus = Literal["succeeded", "rejected_input", "failed"]
ErrorCategory = Literal[
    "invalid_image",
    "model_unavailable",
    "database_unavailable",
    "llm_unavailable",
    "internal_error",
]


class ClassifierResult(BaseModel):
    """Output of the image classifier only. Never mixed with scam-context evidence."""

    predicted_class: Literal["REAL", "FAKE"]
    raw_score: float = Field(description="Softmax P(FAKE) from the raw logits.")
    calibrated_score: float | None = Field(
        None, description="Only set if calibration was fitted, accepted and evaluated."
    )
    score_used: float
    score_label: Literal["raw", "calibrated"]
    threshold: float
    abstain_margin: float
    decision_status: DecisionStatus
    model_release_id: str
    training_run_id: str | None = None
    latency_ms: float

    @property
    def decision_label(self) -> str:
        return DECISION_LABELS[self.decision_status]


class ProvenanceResult(BaseModel):
    """Provenance (C2PA / Content Credentials) is a planned optional module."""

    status: Literal["verified", "not_found", "invalid", "not_checked"] = "not_checked"
    detail: str = "Provenance checking is not implemented in this release."


class ScamIndicator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    indicator: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=300)
    source: Literal["user_context", "image_ocr", "other_provided_evidence"]
    severity: Literal["low", "medium", "high"]


class EvidenceReport(BaseModel):
    """Evidence Interpreter output (LLM or deterministic fallback)."""

    model_config = ConfigDict(extra="forbid")
    summary: str = Field(min_length=1, max_length=1500)
    synthetic_media_assessment: Literal["synthetic-like", "no_strong_signal", "inconclusive"]
    scam_indicators: list[ScamIndicator] = Field(default_factory=list, max_length=12)
    known_facts: list[str] = Field(default_factory=list, max_length=12)
    unknowns: list[str] = Field(default_factory=list, max_length=12)
    recommended_checks: list[str] = Field(default_factory=list, max_length=12)


class InterpretedReport(BaseModel):
    report: EvidenceReport
    llm_used: bool
    llm_error_category: ErrorCategory | None = None
