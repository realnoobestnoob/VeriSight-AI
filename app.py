"""VeriSight AI - Streamlit entry point. Canonical launch: `streamlit run app.py`."""

from __future__ import annotations

import hashlib
import logging

import plotly.express as px
import streamlit as st

from verisight_ai.analytics import (
    count_by,
    operational_summary,
    scans_per_day,
    verified_model_performance,
)
from verisight_ai.config import AppSettings
from verisight_ai.database import make_repository
from verisight_ai.explainability import GRADCAM_CAVEAT
from verisight_ai.inference import ModelUnavailableError, Predictor, resolve_bundle_dir
from verisight_ai.llm_assistant import interpret, make_provider
from verisight_ai.logging_setup import setup_logging
from verisight_ai.preprocessing import InvalidImageError, decode_image_bytes
from verisight_ai.schemas import DecisionStatus, ProvenanceResult

setup_logging()
log = logging.getLogger("verisight.app")
st.set_page_config(page_title="VeriSight AI", page_icon="🔎", layout="wide")

DISCLAIMER = (
    "VeriSight AI is a **triage aid**. It is not a guarantee of authenticity or fraud, and no "
    "single result here should be used for legal, identity, security or financial decisions."
)


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(model_dir: str | None, artifacts_dir: str):
    return Predictor(resolve_bundle_dir(model_dir, artifacts_dir))


@st.cache_resource
def get_repo(url: str | None):
    return make_repository(url)


settings = AppSettings.from_env()
repo = get_repo(settings.database_url)
provider = make_provider(settings)
st.session_state.setdefault("history", [])

st.title("🔎 VeriSight AI")
st.caption("Evidence-aware media triage and scam-context assistant")
st.info(DISCLAIMER)

tab_check, tab_dash, tab_about = st.tabs(["Check media", "Dashboard", "About & limits"])


def _record(input_type, file_type, file_hash, status, error=None, result=None, llm_used=False):
    scan_id = repo.record_scan(
        input_type=input_type,
        file_type=file_type,
        file_hash=file_hash,
        predicted_class=result.predicted_class if result else None,
        raw_score=result.raw_score if result else None,
        calibrated_score=result.calibrated_score if result else None,
        decision_status=result.decision_status.value if result else None,
        model_release_id=result.model_release_id if result else None,
        training_run_id=result.training_run_id if result else None,
        inference_latency_ms=result.latency_ms if result else None,
        status=status,
        error_category=error,
        llm_used=llm_used,
        provenance_status="not_checked",
    )
    log.info(
        "scan status=%s error=%s release=%s latency_ms=%s",
        status,
        error,
        result.model_release_id if result else None,
        f"{result.latency_ms:.0f}" if result else None,
    )
    return scan_id


with tab_check:
    st.subheader("1. Provide an image")
    mode = st.radio(
        "Input method",
        ["Upload an image", "Upload a saved screenshot", "Use camera"],
        horizontal=True,
    )
    data, input_type = None, "upload"
    if mode == "Use camera":
        shot = st.camera_input("Take a photo (this captures your camera, not your screen)")
        if shot is not None:
            data, input_type = shot.getvalue(), "camera_capture"
    else:
        up = st.file_uploader("JPEG, PNG or WebP", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            data = up.getvalue()
            input_type = "screenshot_upload" if mode == "Upload a saved screenshot" else "upload"

    st.subheader("2. Optional: add the message or context")
    user_text = st.text_area(
        "Message that came with the image, or where you received it",
        height=110,
        max_chars=settings.max_text_chars,
        help="Used for warning-sign review. Stays on this server unless you opt in below.",
    )
    allow_llm = False
    if provider is not None:
        allow_llm = st.checkbox(
            "Send my text and the classifier summary to the configured AI provider for a "
            "plain-language explanation (the image itself is never sent)",
            value=False,
        )
    else:
        st.caption("No external AI provider is configured; a built-in rule-based review is used.")

    if st.button("Analyse", type="primary", disabled=data is None):
        try:
            decoded = decode_image_bytes(data, settings.max_upload_bytes, settings.max_pixels)
        except InvalidImageError as exc:
            _record(input_type, None, None, "rejected_input", "invalid_image")
            st.error(str(exc))
            decoded = None
        if decoded is not None:
            try:
                predictor = get_predictor(settings.model_dir, settings.artifacts_dir)
            except ModelUnavailableError as exc:
                _record(input_type, decoded.file_type, None, "failed", "model_unavailable")
                st.error("The model is not available right now. " + str(exc))
                predictor = None
            if predictor is not None:
                try:
                    result = predictor.predict(decoded.image)
                    overlay = predictor.explain(decoded.image)
                except Exception:  # noqa: BLE001 - never show stack traces to users
                    log.exception("inference failed")
                    _record(input_type, decoded.file_type, None, "failed", "internal_error")
                    st.error("Something went wrong while analysing this image.")
                    result = None
                if result is not None:
                    provenance = ProvenanceResult()
                    interp = interpret(result, provenance, user_text, settings, provider, allow_llm)
                    fhash = hashlib.sha256(data).hexdigest()
                    scan_id = _record(
                        input_type,
                        decoded.file_type,
                        fhash,
                        "succeeded",
                        interp.llm_error_category,
                        result,
                        interp.llm_used,
                    )
                    st.session_state["last"] = {
                        "result": result,
                        "overlay": overlay,
                        "original": decoded.image,
                        "interp": interp,
                        "provenance": provenance,
                        "scan_id": scan_id,
                    }
                    st.session_state["history"].append(
                        {
                            "decision": result.decision_label,
                            "score": round(result.score_used, 3),
                            "score_type": result.score_label,
                            "model": result.model_release_id,
                        }
                    )

    last = st.session_state.get("last")
    if last:
        res, rep = last["result"], last["interp"].report
        st.divider()
        st.subheader("Results")
        icon = {
            DecisionStatus.SYNTHETIC_LIKE: "🟠",
            DecisionStatus.NO_STRONG_SIGNAL: "🟢",
            DecisionStatus.INCONCLUSIVE: "⚪",
        }[res.decision_status]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Synthetic-image assessment")
            st.markdown(f"### {icon} {res.decision_label}")
            kind = (
                "calibrated probability"
                if res.score_label == "calibrated"
                else "raw model score (not a probability)"
            )
            st.write(
                f"Score: **{res.score_used:.2f}**, a {kind}. Decision threshold {res.threshold:.2f}, "
                f"inconclusive band ±{res.abstain_margin:.2f}."
            )
            st.caption(
                "This only reflects resemblance to the AI-generated class seen in training. "
                "A low score does not prove the image is authentic."
            )
        with c2:
            st.markdown("#### Visual explanation (Grad-CAM)")
            a, b = st.columns(2)
            a.image(last["original"], caption="Original", width="stretch")
            b.image(last["overlay"], caption="Model influence", width="stretch")
            st.caption(GRADCAM_CAVEAT)

        st.markdown("#### Scam-context review")
        if last["interp"].llm_used:
            st.caption(
                "Explanation written by an external AI provider from the evidence shown here."
            )
        else:
            st.caption("Rule-based review of the text you supplied (no external AI used).")
        st.write(rep.summary)
        if rep.scam_indicators:
            for ind in rep.scam_indicators:
                st.markdown(
                    f"- **{ind.severity.upper()}**: {ind.indicator}. Evidence: “{ind.evidence}”"
                )
        else:
            st.write(
                "No warning signs were identified in the supplied text. That does not mean it is safe."
            )

        st.markdown("#### Provenance check")
        st.write(
            f"**{last['provenance'].status.replace('_', ' ').capitalize()}**. "
            f"{last['provenance'].detail} Missing provenance data is inconclusive."
        )

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### What remains unknown")
            for u in rep.unknowns:
                st.markdown(f"- {u}")
        with c4:
            st.markdown("#### Recommended next checks")
            for c in rep.recommended_checks:
                st.markdown(f"- {c}")

        summary_txt = "\n".join(
            [
                "VeriSight AI summary",
                f"Synthetic-image assessment: {res.decision_label}",
                f"Score ({res.score_label}): {res.score_used:.2f}",
                "",
                rep.summary,
                "",
                "Unknowns:",
                *[f"- {u}" for u in rep.unknowns],
                "",
                "Recommended checks:",
                *[f"- {c}" for c in rep.recommended_checks],
                "",
                DISCLAIMER.replace("**", ""),
            ]
        )
        st.download_button(
            "Download summary (text)", summary_txt, file_name="verisight_summary.txt"
        )

        if repo.enabled and last.get("scan_id"):
            st.markdown("**Was this helpful?** (feedback is opinion, not verified truth)")
            f1, f2, f3 = st.columns(3)
            for col, label, key in (
                (f1, "👍 Helpful", "helpful"),
                (f2, "👎 Not helpful", "not_helpful"),
                (f3, "🤔 Unsure", "unsure"),
            ):
                if col.button(label, key=f"fb_{key}"):
                    st.success("Thanks for the feedback.") if repo.record_feedback(
                        last["scan_id"], key
                    ) else st.warning("Feedback could not be saved.")

    if st.session_state["history"]:
        with st.expander("This session's scans (visible only to you)"):
            st.dataframe(st.session_state["history"], width="stretch")

with tab_dash:
    st.subheader("Operational usage")
    st.caption("App activity only. These are NOT real-world scam or threat rates.")
    if not repo.enabled:
        st.write("Scan logging is not configured, so there is no usage data to show.")
    else:
        df = repo.scan_frame()
        summ = operational_summary(df)
        if summ["total"] == 0:
            st.write("No scans recorded yet.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scans", summ["total"])
            m2.metric("Failure rate", f"{summ['failure_rate']:.1%}")
            m3.metric(
                "Latency p50 (ms)",
                f"{summ['latency_p50_ms']:.0f}" if summ["latency_p50_ms"] else "n/a",
            )
            m4.metric(
                "Latency p95 (ms)",
                f"{summ['latency_p95_ms']:.0f}" if summ["latency_p95_ms"] else "n/a",
            )
            st.plotly_chart(
                px.bar(scans_per_day(df), x="day", y="scans", title="Scans per day"),
                width="stretch",
            )
            ca, cb = st.columns(2)
            ca.plotly_chart(
                px.bar(count_by(df, "input_type"), x="input_type", y="count", title="Input types"),
                width="stretch",
            )
            cb.plotly_chart(
                px.bar(
                    count_by(df, "model_release_id"),
                    x="model_release_id",
                    y="count",
                    title="Model releases used",
                ),
                width="stretch",
            )
            errs = count_by(df[df.error_category.notna()], "error_category")
            if len(errs):
                st.plotly_chart(
                    px.bar(errs, x="error_category", y="count", title="Error categories"),
                    width="stretch",
                )
        st.subheader("User feedback")
        st.caption("Opinions from users, not verified labels.")
        fb = repo.feedback_frame()
        st.dataframe(fb, width="stretch") if len(fb) else st.write("No feedback yet.")
    st.subheader("Verified model performance")
    st.caption("From the labelled evaluation of the active model release, not from live traffic.")
    perf = verified_model_performance(settings.artifacts_dir)
    if perf is None:
        st.write("No promoted model release found.")
    else:
        st.write(
            f"Release `{perf['model_release_id']}` on {perf['dataset'].get('dataset_name', '?')}."
        )
        st.json(perf["metrics"], expanded=False)

with tab_about:
    st.markdown("""
**What this does:** estimates whether an image resembles the AI-generated class from its training
data, optionally reviews message text for common scam warning signs, and suggests independent checks.

**What it does not do:** detect every deepfake, validate documents or offers, prove authenticity,
or decide that something is fraud. Evidence channels are shown separately on purpose.

**Privacy:** uploaded images are processed in memory and are not saved. Scan metadata (time, input
type, model version, score, latency, status) may be logged without filenames or message text. Text
is only sent to an external AI provider if one is configured *and* you tick the opt-in box; the image
is never sent.
""")
