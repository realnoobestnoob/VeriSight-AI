import json

import requests

from verisight_ai.config import AppSettings
from verisight_ai.llm_assistant import deterministic_indicators, interpret
from verisight_ai.schemas import ClassifierResult, DecisionStatus, EvidenceReport, ProvenanceResult


def _result(status=DecisionStatus.NO_STRONG_SIGNAL, score=0.2):
    return ClassifierResult(
        predicted_class="REAL",
        raw_score=score,
        calibrated_score=None,
        score_used=score,
        score_label="raw",
        threshold=0.5,
        abstain_margin=0.1,
        decision_status=status,
        model_release_id="r1",
        training_run_id=None,
        latency_ms=5.0,
    )


class FakeProvider:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, 0

    def complete(self, system, user, timeout_s):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.reply


GOOD = {
    "summary": "ok",
    "synthetic_media_assessment": "synthetic-like",  # tries to overrule
    "scam_indicators": [
        {
            "indicator": "OTP request",
            "evidence": "send me your OTP",
            "source": "user_context",
            "severity": "high",
        },
        {
            "indicator": "Invented",
            "evidence": "wire to account 999",
            "source": "user_context",
            "severity": "high",
        },
    ],
    "known_facts": ["a"],
    "unknowns": ["b"],
    "recommended_checks": ["c"],
}
S = AppSettings()
TEXT = "URGENT: please send me your OTP now"


def test_fallback_finds_indicators_with_evidence():
    inds = deterministic_indicators(TEXT)
    kinds = " ".join(i.indicator for i in inds).lower()
    assert "one-time code" in kinds and "urgency" in kinds
    assert all(i.evidence for i in inds)


def test_works_without_llm():
    out = interpret(_result(), ProvenanceResult(), TEXT, S, provider=None, allow_llm=False)
    assert not out.llm_used and out.report.scam_indicators
    assert out.report.synthetic_media_assessment == "no_strong_signal"


def test_llm_not_called_without_opt_in():
    p = FakeProvider(json.dumps(GOOD))
    interpret(_result(), ProvenanceResult(), TEXT, S, provider=p, allow_llm=False)
    assert p.calls == 0


def test_llm_cannot_overrule_classifier_and_hallucinated_evidence_dropped():
    out = interpret(
        _result(),
        ProvenanceResult(),
        TEXT,
        S,
        provider=FakeProvider("```json\n" + json.dumps(GOOD) + "\n```"),
        allow_llm=True,
    )
    assert out.llm_used
    assert out.report.synthetic_media_assessment == "no_strong_signal"
    assert [i.indicator for i in out.report.scam_indicators] == ["OTP request"]


def test_invalid_json_falls_back():
    out = interpret(
        _result(),
        ProvenanceResult(),
        TEXT,
        S,
        provider=FakeProvider("sorry, I cannot"),
        allow_llm=True,
    )
    assert not out.llm_used and out.llm_error_category == "llm_unavailable"
    assert isinstance(out.report, EvidenceReport)


def test_schema_violation_falls_back():
    bad = dict(GOOD, extra_field="x")
    out = interpret(
        _result(),
        ProvenanceResult(),
        TEXT,
        S,
        provider=FakeProvider(json.dumps(bad)),
        allow_llm=True,
    )
    assert not out.llm_used


def test_timeout_falls_back():
    out = interpret(
        _result(),
        ProvenanceResult(),
        TEXT,
        S,
        provider=FakeProvider(exc=requests.Timeout()),
        allow_llm=True,
    )
    assert not out.llm_used and out.report.recommended_checks


def test_text_is_truncated_to_limit():
    s = AppSettings(max_text_chars=20)
    out = interpret(_result(), ProvenanceResult(), "x" * 500 + " otp", s, allow_llm=False)
    assert not out.report.scam_indicators


def test_low_score_not_described_as_proof():
    out = interpret(_result(), ProvenanceResult(), None, S, allow_llm=False)
    assert any("does not prove" in u for u in out.report.unknowns)
